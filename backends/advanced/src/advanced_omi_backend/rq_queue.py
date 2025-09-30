"""
Redis Queue (RQ) configuration and job functions for Friend-Lite backend.

This module provides RQ-based job processing for transcription and memory tasks.
Uses Redis for job persistence and automatic recovery on server restart.
"""

import os
import logging
from typing import Dict, Any, Optional

import redis
from rq import Queue, Worker
from rq.job import Job

from advanced_omi_backend.models.job import JobPriority

logger = logging.getLogger(__name__)

# Global flag to track if Beanie is initialized in this process
_beanie_initialized = False

# Redis connection configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = redis.from_url(REDIS_URL)

# Queue definitions
TRANSCRIPTION_QUEUE = "transcription"
MEMORY_QUEUE = "memory"
DEFAULT_QUEUE = "default"

# Job retention configuration
JOB_RESULT_TTL = int(os.getenv("RQ_RESULT_TTL", 3600))  # 1 hour default

# Create queues with custom result TTL
transcription_queue = Queue(TRANSCRIPTION_QUEUE, connection=redis_conn, default_timeout=300)
memory_queue = Queue(MEMORY_QUEUE, connection=redis_conn, default_timeout=300)
default_queue = Queue(DEFAULT_QUEUE, connection=redis_conn, default_timeout=300)


def get_queue(queue_name: str = DEFAULT_QUEUE) -> Queue:
    """Get an RQ queue by name."""
    queues = {
        TRANSCRIPTION_QUEUE: transcription_queue,
        MEMORY_QUEUE: memory_queue,
        DEFAULT_QUEUE: default_queue,
    }
    return queues.get(queue_name, default_queue)


async def _ensure_beanie_initialized():
    """Ensure Beanie is initialized in the current process (for RQ workers)."""
    global _beanie_initialized

    if _beanie_initialized:
        return

    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        from beanie import init_beanie
        from advanced_omi_backend.models.conversation import Conversation
        from advanced_omi_backend.models.audio_session import AudioSession
        from advanced_omi_backend.models.user import User

        # Get MongoDB URI from environment
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

        # Create MongoDB client
        client = AsyncIOMotorClient(mongodb_uri)
        database = client.get_default_database("friend-lite")

        # Initialize Beanie
        await init_beanie(
            database=database,
            document_models=[User, Conversation, AudioSession],
        )

        _beanie_initialized = True
        logger.info("✅ Beanie initialized in RQ worker process")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Beanie in RQ worker: {e}")
        raise


# Job functions
def process_audio_job(
    client_id: str,
    user_id: str,
    user_email: str,
    audio_data: bytes,
    audio_rate: int,
    audio_width: int,
    audio_channels: int,
    audio_uuid: Optional[str] = None,
    timestamp: Optional[int] = None
) -> Dict[str, Any]:
    """
    RQ job function for audio file writing and database entry creation.

    This function is executed by RQ workers and can survive server restarts.
    """
    import asyncio
    import time
    import uuid
    from pathlib import Path
    from wyoming.audio import AudioChunk
    from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
    from advanced_omi_backend.database import get_collections

    try:
        logger.info(f"🔄 RQ: Starting audio processing for client {client_id}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def process():
                # Get repository
                collections = get_collections()
                from advanced_omi_backend.database import AudioChunksRepository
                repository = AudioChunksRepository(collections["chunks_col"])

                # Get chunk directory from env
                import os
                chunk_dir = Path(os.getenv("CHUNK_DIR", "/data/audio_chunks"))

                # Create audio UUID if not provided
                final_audio_uuid = audio_uuid or uuid.uuid4().hex
                final_timestamp = timestamp or int(time.time())

                # Create filename and file sink
                wav_filename = f"{final_timestamp}_{client_id}_{final_audio_uuid}.wav"
                file_path = chunk_dir / wav_filename

                # Create file sink
                sink = LocalFileSink(
                    file_path=str(file_path),
                    sample_rate=int(audio_rate),
                    channels=int(audio_channels),
                    sample_width=int(audio_width)
                )

                # Open sink and write audio
                await sink.open()
                audio_chunk = AudioChunk(
                    rate=audio_rate,
                    width=audio_width,
                    channels=audio_channels,
                    audio=audio_data
                )
                await sink.write(audio_chunk)
                await sink.close()

                # Create database entry
                await repository.create_chunk(
                    audio_uuid=final_audio_uuid,
                    audio_path=wav_filename,
                    client_id=client_id,
                    timestamp=final_timestamp,
                    user_id=user_id,
                    user_email=user_email,
                )

                logger.info(f"✅ RQ: Completed audio processing for client {client_id}, file: {wav_filename}")

                return {
                    "success": True,
                    "audio_uuid": final_audio_uuid,
                    "wav_filename": wav_filename,
                    "client_id": client_id
                }

            result = loop.run_until_complete(process())
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Audio processing failed for client {client_id}: {e}")
        raise


def process_transcript_job(
    conversation_id: str,
    audio_uuid: str,
    audio_path: str,
    version_id: str,
    user_id: str,
    trigger: str = "reprocess"
) -> Dict[str, Any]:
    """
    RQ job function for transcript processing.

    This function handles both new transcription and reprocessing.
    The 'trigger' parameter indicates the source: 'new', 'reprocess', 'retry', etc.

    This function is executed by RQ workers and can survive server restarts.
    """
    import asyncio
    from advanced_omi_backend.controllers.conversation_controller import _do_transcript_processing

    try:
        logger.info(f"🔄 RQ: Starting transcript processing for conversation {conversation_id} (trigger: {trigger})")

        # Run the async function in a new event loop
        # RQ workers run in separate processes, so we need our own event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Initialize Beanie in this worker process
            loop.run_until_complete(_ensure_beanie_initialized())

            result = loop.run_until_complete(
                _do_transcript_processing(
                    conversation_id=conversation_id,
                    audio_uuid=audio_uuid,
                    audio_path=audio_path,
                    version_id=version_id,
                    user_id=user_id,
                    trigger=trigger
                )
            )

            logger.info(f"✅ RQ: Completed transcript processing for conversation {conversation_id}")
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Transcript processing failed for conversation {conversation_id}: {e}")
        raise


def process_final_transcription_job(
    conversation_id: str,
    audio_uuid: str,
    audio_path: str,
    user_id: str,
    client_id: str,
    user_email: str
) -> Dict[str, Any]:
    """
    RQ job function for final high-quality transcription after streaming completes.

    This adds a new transcript version to an EXISTING conversation and triggers memory processing.

    Args:
        conversation_id: Existing conversation ID (created by streaming transcription)
        audio_uuid: Audio UUID
        audio_path: Path to audio file
        user_id: User ID
        client_id: Client ID
        user_email: User email

    Returns:
        Dict with processing results
    """
    import asyncio
    import uuid
    from datetime import UTC, datetime
    from pathlib import Path
    import soundfile as sf

    try:
        logger.info(f"🔄 RQ: Starting final transcription for conversation {conversation_id}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def process():
                # Initialize Beanie in this worker process
                await _ensure_beanie_initialized()

                # Import here to avoid circular dependencies
                from advanced_omi_backend.transcription_providers import get_transcription_provider
                from advanced_omi_backend.models.conversation import Conversation
                from advanced_omi_backend.config import CHUNK_DIR

                # Get conversation (should already exist from streaming transcription)
                conversation = await Conversation.find_one(Conversation.conversation_id == conversation_id)
                if not conversation:
                    raise RuntimeError(f"Conversation {conversation_id} not found - streaming transcription should have created it")

                # Read audio file
                audio_file_path = Path(CHUNK_DIR) / audio_path
                if not audio_file_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

                logger.info(f"📖 Reading audio file: {audio_file_path}")
                audio_data, sample_rate = sf.read(str(audio_file_path), dtype='int16')
                audio_bytes = audio_data.tobytes()

                # Get transcription provider
                provider = get_transcription_provider()
                if not provider:
                    raise RuntimeError("No transcription provider available")

                # Transcribe with high quality settings
                logger.info(f"🎤 Transcribing audio with {provider.name} (final quality)")
                transcript_result = await provider.transcribe(audio_bytes, sample_rate, diarize=True)

                # Normalize transcript
                transcript_text = ""
                segments = []

                if hasattr(transcript_result, "text"):
                    transcript_text = transcript_result.text
                    # TODO: Extract segments from transcript_result
                elif isinstance(transcript_result, dict):
                    transcript_text = transcript_result.get("text", "")
                    segments = transcript_result.get("segments", [])
                elif isinstance(transcript_result, str):
                    transcript_text = transcript_result

                # Add new transcript version to conversation
                version_id = str(uuid.uuid4())
                logger.info(f"📝 Adding final transcript version {version_id} to conversation {conversation_id}")

                # Convert segments to SpeakerSegment models
                segment_models = [Conversation.SpeakerSegment(**seg) for seg in segments] if segments else []

                conversation.add_transcript_version(
                    version_id=version_id,
                    transcript=transcript_text,
                    segments=segment_models,
                    provider=provider.name,
                    model=getattr(provider, "model_name", provider.name),
                    metadata={"trigger": "final_transcription", "audio_uuid": audio_uuid}
                )
                await conversation.save()

                logger.info(f"✅ Added final transcript version {version_id} to conversation {conversation_id}")

                # Enqueue memory processing on the final transcript
                logger.info(f"📤 Enqueuing memory processing for conversation {conversation_id}")
                enqueue_memory_processing(
                    client_id=client_id,
                    user_id=user_id,
                    user_email=user_email,
                    conversation_id=conversation_id
                )

                return {
                    "status": "success",
                    "conversation_id": conversation_id,
                    "version_id": version_id,
                    "transcript_length": len(transcript_text)
                }

            result = loop.run_until_complete(process())
            logger.info(f"✅ RQ: Completed final transcription for conversation {conversation_id}")
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Final transcription failed for conversation {conversation_id}: {e}", exc_info=True)
        raise


def process_initial_transcription_job(
    audio_uuid: str,
    audio_path: str,
    client_id: str,
    user_id: str,
    user_email: str
) -> Dict[str, Any]:
    """
    RQ job function for initial transcription of WebSocket audio.

    This handles the complete transcription flow:
    1. Load and transcribe the audio file
    2. Detect speech in transcript
    3. Create conversation if speech detected
    4. Add transcript as initial version
    5. Queue memory processing

    This function is executed by RQ workers and can survive server restarts.
    """
    import asyncio
    import uuid
    from datetime import UTC, datetime
    from pathlib import Path

    try:
        logger.info(f"🔄 RQ: Starting initial transcription for audio {audio_uuid}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def process():
                # Initialize Beanie in this worker process
                await _ensure_beanie_initialized()

                # Import here to avoid circular dependencies
                from advanced_omi_backend.transcription_providers import get_transcription_provider
                from advanced_omi_backend.repositories.audio_chunks_repository import AudioChunksRepository
                from advanced_omi_backend.models.conversation import Conversation
                from advanced_omi_backend.config import CHUNK_DIR
                import soundfile as sf

                # Get audio file path
                audio_file_path = Path(CHUNK_DIR) / audio_path

                if not audio_file_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

                # Read audio file
                logger.info(f"📖 Reading audio file: {audio_file_path}")
                audio_data, sample_rate = sf.read(str(audio_file_path), dtype='int16')
                audio_bytes = audio_data.tobytes()

                # Get transcription provider
                provider = get_transcription_provider()
                if not provider:
                    raise RuntimeError("No transcription provider available")

                # Transcribe audio
                logger.info(f"🎤 Transcribing audio with {provider.name}")
                transcript_result = await provider.transcribe(audio_bytes, sample_rate, diarize=True)

                # Normalize transcript
                transcript_text = ""
                segments = []
                words = []

                if hasattr(transcript_result, "text"):
                    transcript_text = transcript_result.text
                    words = getattr(transcript_result, "words", [])
                elif isinstance(transcript_result, dict):
                    transcript_text = transcript_result.get("text", "")
                    words = transcript_result.get("words", [])
                elif isinstance(transcript_result, str):
                    transcript_text = transcript_result

                # Analyze speech
                has_speech = bool(transcript_text and transcript_text.strip() and len(transcript_text.strip()) > 10)

                # Update audio_chunks with speech detection
                repo = AudioChunksRepository()
                await repo.update_speech_detection(
                    audio_uuid,
                    has_speech=has_speech,
                    reason=f"Transcribed {len(transcript_text)} chars" if has_speech else "Transcript too short"
                )

                # If no speech, mark as EMPTY and return
                if not has_speech:
                    logger.info(f"⏭️ No speech detected in {audio_uuid}, skipping conversation creation")
                    await repo.update_transcription_status(audio_uuid, "EMPTY", provider=provider.name)
                    return {"status": "no_speech", "audio_uuid": audio_uuid}

                # Create conversation
                logger.info(f"✅ Speech detected, creating conversation for {audio_uuid}")
                conversation_id = str(uuid.uuid4())
                version_id = str(uuid.uuid4())

                # Get audio session info
                audio_session = await repo.get_chunk(audio_uuid)
                if not audio_session:
                    raise RuntimeError(f"Audio session not found for {audio_uuid}")

                # Create conversation document
                conversation = Conversation(
                    conversation_id=conversation_id,
                    audio_uuid=audio_uuid,
                    user_id=audio_session["user_id"],
                    client_id=client_id,
                    title="Processing...",
                    summary="Processing...",
                    transcript_versions=[
                        Conversation.TranscriptVersion(
                            version_id=version_id,
                            transcript=transcript_text,
                            segments=segments,
                            provider=provider.name,
                            model=getattr(provider, "model_name", provider.name),
                            created_at=datetime.now(UTC),
                            processing_time_seconds=0.0,
                            metadata={}
                        )
                    ],
                    active_transcript_version=version_id,
                    memory_versions=[],
                    active_memory_version=None,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                    session_start=datetime.fromtimestamp(audio_session.get("timestamp", 0) / 1000, tz=UTC),
                    session_end=datetime.now(UTC),
                    duration_seconds=0.0,
                    speech_start_time=0.0,
                    speech_end_time=0.0,
                    speaker_names={},
                    action_items=[]
                )
                await conversation.insert()

                # Mark audio_chunks as having conversation
                await repo.mark_conversation_created(audio_uuid, conversation_id)

                logger.info(f"✅ Created conversation {conversation_id} for audio {audio_uuid}")

                # Enqueue memory processing
                logger.info(f"📤 Enqueuing memory processing for conversation {conversation_id}")
                enqueue_memory_processing(
                    client_id=client_id,
                    user_id=user_id,
                    user_email=user_email,
                    conversation_id=conversation_id
                )

                return {
                    "status": "success",
                    "audio_uuid": audio_uuid,
                    "conversation_id": conversation_id,
                    "has_speech": True
                }

            result = loop.run_until_complete(process())
            logger.info(f"✅ RQ: Completed initial transcription for audio {audio_uuid}")
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Initial transcription failed for audio {audio_uuid}: {e}", exc_info=True)
        raise


def process_memory_job(
    client_id: str,
    user_id: str,
    user_email: str,
    conversation_id: str
) -> Dict[str, Any]:
    """
    RQ job function for memory extraction and processing.

    This function is executed by RQ workers and can survive server restarts.
    """
    import asyncio
    import time
    from datetime import UTC, datetime
    from advanced_omi_backend.models.conversation import Conversation
    from advanced_omi_backend.memory import get_memory_service
    from advanced_omi_backend.users import get_user_by_id

    try:
        logger.info(f"🔄 RQ: Starting memory processing for conversation {conversation_id}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def process():
                # Initialize Beanie in this worker process
                await _ensure_beanie_initialized()

                start_time = time.time()

                # Get conversation data
                conversation_model = await Conversation.find_one(Conversation.conversation_id == conversation_id)
                if not conversation_model:
                    logger.warning(f"No conversation found for {conversation_id}")
                    return {"success": False, "error": "Conversation not found"}

                # Extract conversation text from transcript segments
                full_conversation = ""
                transcript = conversation_model.transcript
                if transcript:
                    dialogue_lines = []
                    for segment in transcript:
                        text = segment.get("text", "").strip()
                        if text:
                            speaker = segment.get("speaker", "Unknown")
                            dialogue_lines.append(f"{speaker}: {text}")
                    full_conversation = "\n".join(dialogue_lines)

                if len(full_conversation) < 10:
                    logger.warning(f"Conversation too short for memory processing: {conversation_id}")
                    return {"success": False, "error": "Conversation too short"}

                # Check primary speakers filter
                user = await get_user_by_id(user_id)
                if user and user.primary_speakers:
                    transcript_speakers = set()
                    for segment in conversation_model.transcript:
                        if 'identified_as' in segment and segment['identified_as'] and segment['identified_as'] != 'Unknown':
                            transcript_speakers.add(segment['identified_as'].strip().lower())

                    primary_speaker_names = {ps['name'].strip().lower() for ps in user.primary_speakers}

                    if transcript_speakers and not transcript_speakers.intersection(primary_speaker_names):
                        logger.info(f"Skipping memory - no primary speakers found in conversation {conversation_id}")
                        return {"success": True, "skipped": True, "reason": "No primary speakers"}

                # Process memory
                memory_service = get_memory_service()
                memory_result = await memory_service.add_memory(
                    full_conversation,
                    client_id,
                    conversation_id,
                    user_id,
                    user_email,
                    allow_update=True,
                )

                if memory_result:
                    success, created_memory_ids = memory_result

                    if success and created_memory_ids:
                        # Add memory references to conversation
                        conversation_model = await Conversation.find_one(Conversation.conversation_id == conversation_id)
                        if conversation_model:
                            memory_refs = [
                                {"memory_id": mid, "created_at": datetime.now(UTC).isoformat(), "status": "created"}
                                for mid in created_memory_ids
                            ]
                            conversation_model.memories.extend(memory_refs)
                            await conversation_model.save()

                        processing_time = time.time() - start_time
                        logger.info(f"✅ RQ: Completed memory processing for conversation {conversation_id} - created {len(created_memory_ids)} memories in {processing_time:.2f}s")

                        return {
                            "success": True,
                            "memories_created": len(created_memory_ids),
                            "processing_time": processing_time
                        }
                    else:
                        return {"success": True, "memories_created": 0, "skipped": True}
                else:
                    return {"success": False, "error": "Memory service returned False"}

            result = loop.run_until_complete(process())
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Memory processing failed for conversation {conversation_id}: {e}")
        raise


def process_cropping_job(
    client_id: str,
    user_id: str,
    audio_uuid: str,
    original_path: str,
    speech_segments: list,
    output_path: str
) -> Dict[str, Any]:
    """
    RQ job function for audio cropping.

    This function is executed by RQ workers and can survive server restarts.
    """
    import asyncio
    from advanced_omi_backend.audio_utils import _process_audio_cropping_with_relative_timestamps
    from advanced_omi_backend.database import get_collections, AudioChunksRepository

    try:
        logger.info(f"🔄 RQ: Starting audio cropping for audio {audio_uuid}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def process():
                # Get repository
                collections = get_collections()
                repository = AudioChunksRepository(collections["chunks_col"])

                # Convert list of lists to list of tuples
                segments_tuples = [tuple(seg) for seg in speech_segments]

                # Process cropping
                await _process_audio_cropping_with_relative_timestamps(
                    original_path,
                    segments_tuples,
                    output_path,
                    audio_uuid,
                    repository
                )

                logger.info(f"✅ RQ: Completed audio cropping for audio {audio_uuid}")

                return {
                    "success": True,
                    "audio_uuid": audio_uuid,
                    "output_path": output_path,
                    "segments": len(speech_segments)
                }

            result = loop.run_until_complete(process())
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Audio cropping failed for audio {audio_uuid}: {e}")
        raise


def enqueue_audio_processing(
    client_id: str,
    user_id: str,
    user_email: str,
    audio_data: bytes,
    audio_rate: int,
    audio_width: int,
    audio_channels: int,
    audio_uuid: Optional[str] = None,
    timestamp: Optional[int] = None,
    priority: JobPriority = JobPriority.NORMAL
) -> Job:
    """
    Enqueue an audio processing job (file writing + DB entry).

    Returns RQ Job object for tracking.
    """
    timeout_mapping = {
        JobPriority.URGENT: 120,  # 2 minutes
        JobPriority.HIGH: 90,     # 1.5 minutes
        JobPriority.NORMAL: 60,   # 1 minute
        JobPriority.LOW: 30       # 30 seconds
    }

    job = default_queue.enqueue(
        process_audio_job,
        client_id,
        user_id,
        user_email,
        audio_data,
        audio_rate,
        audio_width,
        audio_channels,
        audio_uuid,
        timestamp,
        job_timeout=timeout_mapping.get(priority, 60),
        result_ttl=JOB_RESULT_TTL,
        job_id=f"audio_{client_id}_{audio_uuid or 'new'}",
        description=f"Process audio for client {client_id}"
    )

    logger.info(f"📥 RQ: Enqueued audio job {job.id} for client {client_id}")
    return job


def enqueue_transcript_processing(
    conversation_id: str,
    audio_uuid: str,
    audio_path: str,
    version_id: str,
    user_id: str,
    priority: JobPriority = JobPriority.NORMAL,
    trigger: str = "reprocess"
) -> Job:
    """
    Enqueue a transcript processing job.

    Args:
        trigger: Source of the job - 'new', 'reprocess', 'retry', etc.

    Returns RQ Job object for tracking.
    """
    # Map our priority enum to RQ job timeout in seconds (higher priority = longer timeout)
    timeout_mapping = {
        JobPriority.URGENT: 600,  # 10 minutes
        JobPriority.HIGH: 480,    # 8 minutes
        JobPriority.NORMAL: 300,  # 5 minutes
        JobPriority.LOW: 180      # 3 minutes
    }

    job = transcription_queue.enqueue(
        process_transcript_job,
        conversation_id,
        audio_uuid,
        audio_path,
        version_id,
        user_id,
        trigger,
        job_timeout=timeout_mapping.get(priority, 300),
        result_ttl=JOB_RESULT_TTL,  # Keep completed jobs for 1 hour
        job_id=f"transcript_{conversation_id[:8]}_{trigger}",
        description=f"Process transcript for conversation {conversation_id[:8]} ({trigger})"
    )

    logger.info(f"📥 RQ: Enqueued transcript job {job.id} for conversation {conversation_id} (trigger: {trigger})")
    return job


def enqueue_final_transcription(
    conversation_id: str,
    audio_uuid: str,
    audio_path: str,
    user_id: str,
    client_id: str,
    user_email: str,
    priority: JobPriority = JobPriority.NORMAL
) -> Job:
    """
    Enqueue final high-quality transcription job after streaming transcription completes.

    This job adds a new high-quality transcript version to an EXISTING conversation
    (created by streaming transcription) and triggers memory processing.

    Args:
        conversation_id: Existing conversation ID (created by streaming transcription)
        audio_uuid: Audio UUID
        audio_path: Path to saved audio file
        user_id: User ID
        client_id: Client ID
        user_email: User email
        priority: Job priority

    Returns:
        RQ Job object for tracking.
    """
    timeout_mapping = {
        JobPriority.URGENT: 600,  # 10 minutes
        JobPriority.HIGH: 480,    # 8 minutes
        JobPriority.NORMAL: 300,  # 5 minutes
        JobPriority.LOW: 180      # 3 minutes
    }

    job = transcription_queue.enqueue(
        process_final_transcription_job,
        conversation_id,
        audio_uuid,
        audio_path,
        user_id,
        client_id,
        user_email,
        job_timeout=timeout_mapping.get(priority, 300),
        result_ttl=JOB_RESULT_TTL,
        job_id=f"final_transcript_{conversation_id[:12]}",
        description=f"Final transcription for conversation {conversation_id[:12]}"
    )

    logger.info(f"📥 RQ: Enqueued final transcription job {job.id} for conversation {conversation_id}")
    return job


def enqueue_initial_transcription(
    audio_uuid: str,
    audio_path: str,
    client_id: str,
    user_id: str,
    user_email: str,
    priority: JobPriority = JobPriority.NORMAL
) -> Job:
    """
    Enqueue initial transcription job for WebSocket audio streaming.

    This job:
    1. Transcribes the saved audio file
    2. Creates conversation if speech is detected
    3. Adds transcript as initial version
    4. Queues memory processing if conversation created

    Returns RQ Job object for tracking.
    """
    timeout_mapping = {
        JobPriority.URGENT: 600,  # 10 minutes
        JobPriority.HIGH: 480,    # 8 minutes
        JobPriority.NORMAL: 300,  # 5 minutes
        JobPriority.LOW: 180      # 3 minutes
    }

    job = transcription_queue.enqueue(
        process_initial_transcription_job,
        audio_uuid,
        audio_path,
        client_id,
        user_id,
        user_email,
        job_timeout=timeout_mapping.get(priority, 300),
        result_ttl=JOB_RESULT_TTL,
        job_id=f"initial_transcript_{audio_uuid[:12]}",
        description=f"Initial transcription for audio {audio_uuid[:12]}"
    )

    logger.info(f"📥 RQ: Enqueued initial transcription job {job.id} for audio {audio_uuid}")
    return job


def enqueue_memory_processing(
    client_id: str,
    user_id: str,
    user_email: str,
    conversation_id: str,
    priority: JobPriority = JobPriority.NORMAL
) -> Job:
    """
    Enqueue a memory processing job.

    Returns RQ Job object for tracking.
    """
    timeout_mapping = {
        JobPriority.URGENT: 3600,  # 60 minutes
        JobPriority.HIGH: 2400,    # 40 minutes
        JobPriority.NORMAL: 1800,  # 30 minutes
        JobPriority.LOW: 900       # 15 minutes
    }

    job = memory_queue.enqueue(
        process_memory_job,
        client_id,
        user_id,
        user_email,
        conversation_id,
        job_timeout=timeout_mapping.get(priority, 1800),
        result_ttl=JOB_RESULT_TTL,
        job_id=f"memory_{conversation_id[:8]}",
        description=f"Process memory for conversation {conversation_id[:8]}"
    )

    logger.info(f"📥 RQ: Enqueued memory job {job.id} for conversation {conversation_id}")
    return job


def enqueue_cropping(
    client_id: str,
    user_id: str,
    audio_uuid: str,
    original_path: str,
    speech_segments: list,
    output_path: str,
    priority: JobPriority = JobPriority.NORMAL
) -> Job:
    """
    Enqueue an audio cropping job.

    Returns RQ Job object for tracking.
    """
    timeout_mapping = {
        JobPriority.URGENT: 300,  # 5 minutes
        JobPriority.HIGH: 240,    # 4 minutes
        JobPriority.NORMAL: 180,  # 3 minutes
        JobPriority.LOW: 120      # 2 minutes
    }

    job = default_queue.enqueue(
        process_cropping_job,
        client_id,
        user_id,
        audio_uuid,
        original_path,
        speech_segments,
        output_path,
        job_timeout=timeout_mapping.get(priority, 180),
        result_ttl=JOB_RESULT_TTL,
        job_id=f"cropping_{audio_uuid[:8]}",
        description=f"Crop audio for {audio_uuid[:8]}"
    )

    logger.info(f"📥 RQ: Enqueued cropping job {job.id} for audio {audio_uuid}")
    return job


def get_job_stats() -> Dict[str, int]:
    """Get job statistics across all queues."""
    stats = {
        "total_jobs": 0,
        "queued_jobs": 0,
        "processing_jobs": 0,
        "completed_jobs": 0,
        "failed_jobs": 0,
        "cancelled_jobs": 0,
        "retrying_jobs": 0
    }

    try:
        for queue in [transcription_queue, memory_queue, default_queue]:
            # Queued jobs
            queued = len(queue.jobs)
            stats["queued_jobs"] += queued

            # Started jobs (currently processing)
            started = len(queue.started_job_registry)
            stats["processing_jobs"] += started

            # Finished jobs
            finished = len(queue.finished_job_registry)
            stats["completed_jobs"] += finished

            # Failed jobs
            failed = len(queue.failed_job_registry)
            stats["failed_jobs"] += failed

            # Deferred jobs (retrying)
            deferred = len(queue.deferred_job_registry)
            stats["retrying_jobs"] += deferred

        stats["total_jobs"] = sum(stats.values()) - stats["total_jobs"]  # Subtract initial 0

    except Exception as e:
        logger.error(f"Error getting job stats: {e}")

    return stats


def get_jobs(limit: int = 20, offset: int = 0, queue_name: str = None) -> Dict[str, Any]:
    """Get jobs with pagination."""
    try:
        queues_to_check = [transcription_queue, memory_queue, default_queue]
        if queue_name:
            queue = get_queue(queue_name)
            queues_to_check = [queue] if queue else []

        all_jobs = []

        for queue in queues_to_check:
            # Get jobs from different registries
            registries = [
                (queue.jobs, "queued"),
                (queue.started_job_registry.get_job_ids(), "processing"),
                (queue.finished_job_registry.get_job_ids(), "completed"),
                (queue.failed_job_registry.get_job_ids(), "failed"),
                (queue.deferred_job_registry.get_job_ids(), "retrying")
            ]

            for job_source, status in registries:
                if hasattr(job_source, '__iter__'):
                    job_ids = list(job_source)
                else:
                    job_ids = job_source

                for job_id in job_ids[:limit]:  # Limit per registry for performance
                    try:
                        if hasattr(job_source, '__iter__') and hasattr(job_id, 'id'):
                            # This is a Job object
                            job = job_id
                        else:
                            # This is a job ID string
                            job = Job.fetch(job_id, connection=redis_conn)

                        # Determine job type from function name or job ID
                        job_type = "unknown"
                        if "transcript" in job.id:
                            job_type = "reprocess_transcript"
                        elif "memory" in job.id:
                            job_type = "reprocess_memory"
                        elif "process_transcript_job" in (job.func_name or ""):
                            job_type = "reprocess_transcript"

                        job_data = {
                            "job_id": job.id,
                            "job_type": job_type,
                            "user_id": job.kwargs.get("user_id", "") if job.kwargs else "",
                            "status": status,
                            "priority": "normal",  # Default priority, could be enhanced later
                            "queue_name": queue.name,
                            "created_at": job.created_at.isoformat() if job.created_at else None,
                            "started_at": job.started_at.isoformat() if job.started_at else None,
                            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
                            "completed_at": job.ended_at.isoformat() if job.ended_at and status == "completed" else None,
                            "description": job.description or "",
                            "func_name": job.func_name if hasattr(job, 'func_name') else "",
                            "args": job.args[:2] if job.args else [],  # First 2 args for preview
                            "kwargs": dict(list(job.kwargs.items())[:3]) if job.kwargs else {},  # First 3 kwargs
                            # Don't include result in listing - use get_job endpoint for details
                            "data": {"description": job.description or ""},  # Job data for UI
                            "error_message": str(job.exc_info) if job.exc_info else None,
                            "retry_count": 0,  # RQ doesn't track retries this way
                            "max_retries": 3,  # Default max retries
                            "progress_percent": 100 if status == "completed" else 0,
                            "progress_message": f"Job {status}"
                        }
                        all_jobs.append(job_data)

                    except Exception as e:
                        logger.warning(f"Error fetching job {job_id}: {e}")
                        continue

        # Sort by created_at descending
        all_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Apply pagination
        total = len(all_jobs)
        paginated_jobs = all_jobs[offset:offset + limit]

        return {
            "jobs": paginated_jobs,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total
            }
        }

    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        return {
            "jobs": [],
            "pagination": {"total": 0, "limit": limit, "offset": offset, "has_more": False}
        }


def get_queue_health() -> Dict[str, Any]:
    """Get queue system health status."""
    try:
        # Check Redis connection
        redis_conn.ping()

        # Check if workers are running
        workers = Worker.all(connection=redis_conn)
        active_workers = [w for w in workers if w.state == 'busy' or w.state == 'idle']

        return {
            "status": "healthy" if active_workers else "no_workers",
            "redis_connected": True,
            "active_workers": len(active_workers),
            "total_workers": len(workers),
            "queues": {
                "transcription": len(transcription_queue.jobs),
                "memory": len(memory_queue.jobs),
                "default": len(default_queue.jobs)
            },
            "message": f"RQ healthy with {len(active_workers)} active workers" if active_workers else "RQ connected but no workers running"
        }

    except Exception as e:
        logger.error(f"RQ health check failed: {e}")
        return {
            "status": "unhealthy",
            "redis_connected": False,
            "active_workers": 0,
            "total_workers": 0,
            "message": f"RQ health check failed: {str(e)}"
        }