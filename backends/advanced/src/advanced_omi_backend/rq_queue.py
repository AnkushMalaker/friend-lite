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
        from advanced_omi_backend.models.audio_file import AudioFile
        from advanced_omi_backend.models.user import User

        # Get MongoDB URI from environment
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

        # Create MongoDB client
        client = AsyncIOMotorClient(mongodb_uri)
        database = client.get_default_database("friend-lite")

        # Initialize Beanie
        await init_beanie(
            database=database,
            document_models=[User, Conversation, AudioFile],
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
                from advanced_omi_backend.config import CHUNK_DIR
                repository = AudioChunksRepository(collections["chunks_col"])

                # Use CHUNK_DIR from config
                chunk_dir = CHUNK_DIR

                # Ensure directory exists
                chunk_dir.mkdir(parents=True, exist_ok=True)

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

                # Enqueue transcript processing for this audio file
                # First ensure Beanie is initialized for this worker process
                await _ensure_beanie_initialized()

                # Create a conversation entry
                from advanced_omi_backend.models.conversation import create_conversation
                import uuid as uuid_lib

                conversation_id = str(uuid_lib.uuid4())
                conversation = create_conversation(
                    conversation_id=conversation_id,
                    audio_uuid=final_audio_uuid,
                    user_id=user_id,
                    client_id=client_id
                )
                # Set placeholder title/summary
                conversation.title = "Processing..."
                conversation.summary = "Transcript processing in progress"
                await conversation.insert()

                logger.info(f"📝 RQ: Created conversation {conversation_id} for audio {final_audio_uuid}")

                # Now enqueue transcript processing (runs outside async context)
                version_id = str(uuid_lib.uuid4())

                return {
                    "success": True,
                    "audio_uuid": final_audio_uuid,
                    "conversation_id": conversation_id,
                    "wav_filename": wav_filename,
                    "client_id": client_id,
                    "version_id": version_id,
                    "file_path": str(file_path)
                }

            result = loop.run_until_complete(process())

            # Enqueue transcript processing job (outside async context)
            if result.get("success") and result.get("conversation_id"):
                transcript_job = enqueue_transcript_processing(
                    conversation_id=result["conversation_id"],
                    audio_uuid=result["audio_uuid"],
                    audio_path=result["file_path"],
                    version_id=result["version_id"],
                    user_id=user_id,
                    priority=JobPriority.NORMAL,
                    trigger="upload"
                )
                result["transcript_job_id"] = transcript_job.id
                logger.info(f"📥 RQ: Enqueued transcript job {transcript_job.id} for conversation {result['conversation_id']}")

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
    import time
    from pathlib import Path

    try:
        logger.info(f"🔄 RQ: Starting transcript processing for conversation {conversation_id} (trigger: {trigger})")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def process():
                # Initialize Beanie in this worker process
                await _ensure_beanie_initialized()

                # Import here to avoid circular dependencies
                from advanced_omi_backend.services.transcription import get_transcription_provider
                from advanced_omi_backend.models.conversation import Conversation

                start_time = time.time()

                # Get the transcription provider
                provider = get_transcription_provider(mode="batch")
                if not provider:
                    raise ValueError("No transcription provider available")

                provider_name = provider.name
                logger.info(f"Using transcription provider: {provider_name}")

                # Read the audio file
                audio_file_path = Path(audio_path)
                if not audio_file_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")

                # Load audio data
                with open(audio_file_path, 'rb') as f:
                    audio_data = f.read()

                # Transcribe the audio (assume 16kHz sample rate)
                transcription_result = await provider.transcribe(
                    audio_data=audio_data,
                    sample_rate=16000,
                    diarize=True
                )

                # Extract results
                transcript_text = transcription_result.get("text", "")
                segments = transcription_result.get("segments", [])
                words = transcription_result.get("words", [])

                # Calculate processing time
                processing_time = time.time() - start_time

                # Get the conversation using Beanie
                conversation = await Conversation.find_one(Conversation.conversation_id == conversation_id)
                if not conversation:
                    logger.error(f"Conversation {conversation_id} not found")
                    return {"success": False, "error": "Conversation not found"}

                # Convert segments to SpeakerSegment objects
                speaker_segments = [
                    Conversation.SpeakerSegment(
                        start=seg.get("start", 0),
                        end=seg.get("end", 0),
                        text=seg.get("text", ""),
                        speaker=seg.get("speaker", "unknown"),
                        confidence=seg.get("confidence")
                    )
                    for seg in segments
                ]

                # Add new transcript version
                provider_normalized = provider_name.lower() if provider_name else "unknown"

                conversation.add_transcript_version(
                    version_id=version_id,
                    transcript=transcript_text,
                    segments=speaker_segments,
                    provider=Conversation.TranscriptProvider(provider_normalized),
                    model=getattr(provider, 'model', 'unknown'),
                    processing_time_seconds=processing_time,
                    metadata={
                        "trigger": trigger,
                        "audio_file_size": len(audio_data),
                        "segment_count": len(segments),
                        "word_count": len(words)
                    },
                    set_as_active=True
                )

                # Generate title and summary from transcript
                if transcript_text and len(transcript_text.strip()) > 0:
                    first_sentence = transcript_text.split('.')[0].strip()
                    conversation.title = first_sentence[:50] + "..." if len(first_sentence) > 50 else first_sentence
                    conversation.summary = transcript_text[:150] + "..." if len(transcript_text) > 150 else transcript_text
                else:
                    conversation.title = "Empty Conversation"
                    conversation.summary = "No speech detected"

                # Save the updated conversation
                await conversation.save()

                logger.info(f"✅ Transcript processing completed for {conversation_id} in {processing_time:.2f}s")

                return {
                    "success": True,
                    "conversation_id": conversation_id,
                    "version_id": version_id,
                    "transcript": transcript_text,
                    "segments": [seg.model_dump() for seg in speaker_segments],
                    "provider": provider_name,
                    "processing_time_seconds": processing_time,
                    "trigger": trigger,
                    "metadata": {
                        "trigger": trigger,
                        "audio_file_size": len(audio_data),
                        "segment_count": len(speaker_segments),
                        "word_count": len(words)
                    }
                }

            result = loop.run_until_complete(process())
            logger.info(f"✅ RQ: Completed transcript processing for conversation {conversation_id}")
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Transcript processing failed for conversation {conversation_id}: {e}")
        raise


def listen_for_speech_job(
    audio_uuid: str,
    audio_path: str,
    client_id: str,
    user_id: str,
    user_email: str
) -> Dict[str, Any]:
    """
    RQ job function for initial audio transcription and speech detection.

    This job:
    1. Transcribes the audio file
    2. Detects if speech is present
    3. Creates a conversation ONLY if speech is detected
    4. Enqueues memory processing if conversation created

    Used by: Audio file uploads and WebSocket audio streams

    Args:
        audio_uuid: Audio UUID
        audio_path: Path to audio file
        client_id: Client ID
        user_id: User ID
        user_email: User email

    Returns:
        Dict with processing results
    """
    import asyncio
    import uuid
    import soundfile as sf
    from datetime import UTC, datetime
    from pathlib import Path

    try:
        logger.info(f"🔄 RQ: Listening for speech in audio {audio_uuid}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def process():
                # Initialize Beanie in this worker process
                await _ensure_beanie_initialized()

                # Import here to avoid circular dependencies
                from advanced_omi_backend.services.transcription import get_transcription_provider
                from advanced_omi_backend.models.conversation import Conversation
                from advanced_omi_backend.models.audio_file import AudioFile
                from advanced_omi_backend.config import CHUNK_DIR

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

                # Transcribe audio
                logger.info(f"🎤 Transcribing audio with {provider.name}")
                transcript_result = await provider.transcribe(audio_bytes, sample_rate, diarize=True)

                # Normalize transcript
                transcript_text = ""
                segments = []

                if hasattr(transcript_result, "text"):
                    transcript_text = transcript_result.text
                    segments = getattr(transcript_result, "segments", [])
                elif isinstance(transcript_result, dict):
                    transcript_text = transcript_result.get("text", "")
                    segments = transcript_result.get("segments", [])
                elif isinstance(transcript_result, str):
                    transcript_text = transcript_result

                version_id = str(uuid.uuid4())

                # Analyze speech
                has_speech = bool(transcript_text and transcript_text.strip() and len(transcript_text.strip()) > 10)

                logger.info(
                    f"📊 Speech analysis for {audio_uuid}: "
                    f"text_length={len(transcript_text)}, "
                    f"segments={len(segments)}, "
                    f"has_speech={has_speech}, "
                    f"preview={transcript_text[:100] if transcript_text else 'EMPTY'}"
                )

                # Update AudioFile with speech detection
                audio_file = await AudioFile.find_one(AudioFile.audio_uuid == audio_uuid)
                if audio_file:
                    audio_file.has_speech = has_speech
                    audio_file.speech_analysis = {
                        "transcript_length": len(transcript_text),
                        "segment_count": len(segments),
                        "reason": f"Transcribed {len(transcript_text)} chars" if has_speech else "Transcript too short"
                    }
                    await audio_file.save()

                # If no speech, return early
                if not has_speech:
                    logger.info(f"⏭️ No speech detected in {audio_uuid}, skipping conversation creation")
                    return {"status": "no_speech", "audio_uuid": audio_uuid}

                # Create conversation
                logger.info(f"✅ Speech detected, creating conversation for {audio_uuid}")
                new_conversation_id = str(uuid.uuid4())

                # Get timestamp from AudioFile
                timestamp_value = audio_file.timestamp if audio_file else 0
                if timestamp_value == 0:
                    logger.warning(f"Audio file {audio_uuid} has no timestamp, using current time")
                    session_start_time = datetime.now(UTC)
                else:
                    session_start_time = datetime.fromtimestamp(timestamp_value / 1000, tz=UTC)

                # Generate title and summary from transcript
                title = transcript_text[:50] + "..." if len(transcript_text) > 50 else transcript_text or "Conversation"
                summary = transcript_text[:150] + "..." if len(transcript_text) > 150 else transcript_text or "No transcript available"

                # Create conversation document
                conversation = Conversation(
                    conversation_id=new_conversation_id,
                    audio_uuid=audio_uuid,
                    user_id=user_id,
                    client_id=client_id,
                    title=title,
                    summary=summary,
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
                    session_start=session_start_time,
                    session_end=datetime.now(UTC),
                    duration_seconds=0.0,
                    speech_start_time=0.0,
                    speech_end_time=0.0,
                    speaker_names={},
                    action_items=[]
                )

                # Update legacy fields
                conversation._update_legacy_transcript_fields()
                await conversation.insert()

                # Link conversation to AudioFile
                if audio_file:
                    audio_file.conversation_id = new_conversation_id
                    await audio_file.save()

                logger.info(f"✅ Created conversation {new_conversation_id} for audio {audio_uuid}")

                # Enqueue memory processing
                logger.info(f"📤 Enqueuing memory processing for conversation {new_conversation_id}")
                enqueue_memory_processing(
                    client_id=client_id,
                    user_id=user_id,
                    user_email=user_email,
                    conversation_id=new_conversation_id
                )

                return {
                    "status": "success",
                    "audio_uuid": audio_uuid,
                    "conversation_id": new_conversation_id,
                    "has_speech": True
                }

            result = loop.run_until_complete(process())
            logger.info(f"✅ RQ: Completed speech detection for audio {audio_uuid}")
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Speech detection failed for audio {audio_uuid}: {e}", exc_info=True)
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
                segments = conversation_model.segments
                if segments:
                    dialogue_lines = []
                    for segment in segments:
                        # Handle both dict and object segments
                        if isinstance(segment, dict):
                            text = segment.get("text", "").strip()
                            speaker = segment.get("speaker", "Unknown")
                        else:
                            text = getattr(segment, "text", "").strip()
                            speaker = getattr(segment, "speaker", "Unknown")

                        if text:
                            dialogue_lines.append(f"{speaker}: {text}")
                    full_conversation = "\n".join(dialogue_lines)
                elif conversation_model.transcript and isinstance(conversation_model.transcript, str):
                    # Fallback: if segments are empty but transcript text exists
                    full_conversation = conversation_model.transcript

                if len(full_conversation) < 10:
                    logger.warning(f"Conversation too short for memory processing: {conversation_id}")
                    return {"success": False, "error": "Conversation too short"}

                # Check primary speakers filter
                user = await get_user_by_id(user_id)
                if user and user.primary_speakers:
                    transcript_speakers = set()
                    for segment in conversation_model.segments:
                        # Handle both dict and object segments
                        if isinstance(segment, dict):
                            identified_as = segment.get('identified_as')
                        else:
                            identified_as = getattr(segment, 'identified_as', None)

                        if identified_as and identified_as != 'Unknown':
                            transcript_speakers.add(identified_as.strip().lower())

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

    # Use clearer job type names
    job_type = "re-transcribe" if trigger == "reprocess" else trigger

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
        job_id=f"{job_type}_{conversation_id[:8]}",
        description=f"{job_type.capitalize()} conversation {conversation_id[:8]}"
    )

    logger.info(f"📥 RQ: Enqueued transcript job {job.id} for conversation {conversation_id} (trigger: {trigger})")
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
    Enqueue job to listen for speech and create conversation if detected.

    This job transcribes audio, detects speech, and creates a conversation
    ONLY if speech is detected. Used for initial audio uploads and streams.

    Args:
        audio_uuid: Audio UUID
        audio_path: Path to saved audio file
        client_id: Client ID
        user_id: User ID
        user_email: User email
        priority: Job priority

    Returns:
        RQ Job object for tracking
    """
    timeout_mapping = {
        JobPriority.URGENT: 600,  # 10 minutes
        JobPriority.HIGH: 480,    # 8 minutes
        JobPriority.NORMAL: 300,  # 5 minutes
        JobPriority.LOW: 180      # 3 minutes
    }

    job = transcription_queue.enqueue(
        listen_for_speech_job,
        audio_uuid,
        audio_path,
        client_id,
        user_id,
        user_email,
        job_timeout=timeout_mapping.get(priority, 300),
        result_ttl=JOB_RESULT_TTL,
        job_id=f"listen-for-speech_{audio_uuid[:12]}",
        description=f"Listen for speech in {audio_uuid[:12]}"
    )

    logger.info(f"📥 RQ: Enqueued listen-for-speech job {job.id} for audio {audio_uuid}")
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
                            job_type = "transcribe"
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