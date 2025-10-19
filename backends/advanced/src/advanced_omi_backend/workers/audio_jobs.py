"""
Audio-related RQ job functions.

This module contains jobs related to audio file processing and cropping.
"""

import asyncio
import os
import logging
import time
from typing import Dict, Any, Optional

from advanced_omi_backend.models.job import JobPriority, async_job

from advanced_omi_backend.controllers.queue_controller import (
    default_queue,
    _ensure_beanie_initialized,
    JOB_RESULT_TTL,
)

logger = logging.getLogger(__name__)


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

            # Enqueue transcript processing job chain (outside async context)
            if result.get("success") and result.get("conversation_id"):
                from .transcription_jobs import transcribe_full_audio_job, recognise_speakers_job
                from .memory_jobs import process_memory_job
                from advanced_omi_backend.controllers.queue_controller import transcription_queue, memory_queue, default_queue, JOB_RESULT_TTL

                conversation_id = result["conversation_id"]

                # Job 1: Transcribe audio to text
                transcript_job = transcription_queue.enqueue(
                    transcribe_full_audio_job,
                    conversation_id,
                    result["audio_uuid"],
                    result["file_path"],
                    result["version_id"],
                    user_id,
                    "upload",
                    job_timeout=600,
                    result_ttl=JOB_RESULT_TTL,
                    job_id=f"upload_{conversation_id[:8]}",
                    description=f"Transcribe audio for {conversation_id[:8]}",
                    meta={'audio_uuid': result["audio_uuid"], 'conversation_id': conversation_id}
                )
                logger.info(f"📥 RQ: Enqueued transcription job {transcript_job.id}")

                # Job 2: Recognize speakers (depends on transcription)
                speaker_job = transcription_queue.enqueue(
                    recognise_speakers_job,
                    conversation_id,
                    result["version_id"],
                    result["file_path"],
                    user_id,
                    "",  # transcript_text - will be read from DB
                    [],  # words - will be read from DB
                    depends_on=transcript_job,
                    job_timeout=600,
                    result_ttl=JOB_RESULT_TTL,
                    job_id=f"speaker_{conversation_id[:8]}",
                    description=f"Recognize speakers for {conversation_id[:8]}",
                    meta={'audio_uuid': result["audio_uuid"], 'conversation_id': conversation_id}
                )
                logger.info(f"📥 RQ: Enqueued speaker recognition job {speaker_job.id} (depends on {transcript_job.id})")

                # Job 3: Audio cropping (depends on speaker recognition)
                cropping_job = default_queue.enqueue(
                    process_cropping_job,
                    conversation_id,
                    result["file_path"],
                    user_id,
                    depends_on=speaker_job,
                    job_timeout=300,
                    result_ttl=JOB_RESULT_TTL,
                    job_id=f"crop_{conversation_id[:8]}",
                    description=f"Crop audio for {conversation_id[:8]}",
                    meta={'audio_uuid': result["audio_uuid"], 'conversation_id': conversation_id}
                )
                logger.info(f"📥 RQ: Enqueued audio cropping job {cropping_job.id} (depends on {speaker_job.id})")

                # Job 4: Extract memories (depends on cropping)
                memory_job = memory_queue.enqueue(
                    process_memory_job,
                    None,  # client_id - will be read from conversation in DB
                    user_id,
                    "",  # user_email - will be read from user in DB
                    conversation_id,
                    depends_on=cropping_job,
                    job_timeout=1800,
                    result_ttl=JOB_RESULT_TTL,
                    job_id=f"memory_{conversation_id[:8]}",
                    description=f"Extract memories for {conversation_id[:8]}",
                    meta={'audio_uuid': result["audio_uuid"], 'conversation_id': conversation_id}
                )
                logger.info(f"📥 RQ: Enqueued memory job {memory_job.id} (depends on {cropping_job.id})")

                result["transcript_job_id"] = transcript_job.id
                result["speaker_job_id"] = speaker_job.id
                result["cropping_job_id"] = cropping_job.id
                result["memory_job_id"] = memory_job.id

            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Audio processing failed for client {client_id}: {e}")
        raise


@async_job(redis=True, beanie=True)
async def process_cropping_job(
    conversation_id: str,
    audio_path: str,
    redis_client=None
) -> Dict[str, Any]:
    """
    RQ job function for audio cropping - removes silent segments from audio.

    This job:
    1. Reads transcript segments from conversation
    2. Extracts speech timestamps
    3. Creates cropped audio file with only speech segments
    4. Updates audio_chunks collection with cropped file path

    Args:
        conversation_id: Conversation ID
        audio_path: Path to original audio file
        redis_client: Redis client (injected by decorator)

    Returns:
        Dict with processing results
    """
    from pathlib import Path
    from advanced_omi_backend.utils.audio_utils import _process_audio_cropping_with_relative_timestamps
    from advanced_omi_backend.database import get_collections, AudioChunksRepository
    from advanced_omi_backend.models.conversation import Conversation
    from advanced_omi_backend.config import CHUNK_DIR

    try:
        logger.info(f"🔄 RQ: Starting audio cropping for conversation {conversation_id}")

        # Get conversation to access segments
        conversation = await Conversation.find_one(Conversation.conversation_id == conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Extract speech segments from transcript
        segments = conversation.segments
        if not segments or len(segments) == 0:
            logger.warning(f"⚠️ No segments found for conversation {conversation_id}, skipping cropping")
            return {
                "success": False,
                "conversation_id": conversation_id,
                "reason": "no_segments"
            }

        # Convert segments to (start, end) tuples
        speech_segments = [(seg.start, seg.end) for seg in segments]
        logger.info(f"Found {len(speech_segments)} speech segments for cropping")

        # Generate output path for cropped audio
        audio_uuid = conversation.audio_uuid
        original_path = Path(audio_path)
        cropped_filename = f"cropped_{original_path.name}"
        output_path = CHUNK_DIR / cropped_filename

        # Get repository for database updates
        collections = get_collections()
        repository = AudioChunksRepository(collections["chunks_col"])

        # Process cropping
        success = await _process_audio_cropping_with_relative_timestamps(
            str(original_path),
            speech_segments,
            str(output_path),
            audio_uuid,
            repository
        )

        if not success:
            logger.error(f"❌ RQ: Audio cropping failed for conversation {conversation_id}")
            return {
                "success": False,
                "conversation_id": conversation_id,
                "reason": "cropping_failed"
            }

        # Calculate cropped duration
        cropped_duration_seconds = sum(end - start for start, end in speech_segments)

        # Update conversation with cropped audio path
        conversation.cropped_audio_path = cropped_filename
        await conversation.save()
        logger.info(f"💾 Updated conversation {conversation_id[:12]} with cropped_audio_path: {cropped_filename}")

        logger.info(f"✅ RQ: Completed audio cropping for conversation {conversation_id} ({cropped_duration_seconds:.1f}s)")

        # Update job metadata with cropped duration
        from rq import get_current_job
        current_job = get_current_job()
        if current_job:
            if not current_job.meta:
                current_job.meta = {}
            current_job.meta['cropped_duration_seconds'] = round(cropped_duration_seconds, 1)
            current_job.meta['segments_cropped'] = len(speech_segments)
            current_job.save_meta()

        return {
            "success": True,
            "conversation_id": conversation_id,
            "audio_uuid": audio_uuid,
            "original_path": str(original_path),
            "cropped_path": str(output_path),
            "cropped_filename": cropped_filename,
            "segments_count": len(speech_segments),
            "cropped_duration_seconds": cropped_duration_seconds
        }

    except Exception as e:
        logger.error(f"❌ RQ: Audio cropping failed for conversation {conversation_id}: {e}")
        raise


@async_job(redis=True, beanie=True)
async def audio_streaming_persistence_job(
    session_id: str,
    user_id: str,
    client_id: str,
    redis_client=None
) -> Dict[str, Any]:
    """
    Long-running RQ job that progressively writes audio chunks to disk as they arrive.

    Opens a WAV file immediately and appends chunks in real-time, making the file
    available for playback in the UI before the session completes.

    Runs in parallel with transcription processing to reduce memory pressure.

    Args:
        session_id: Stream session ID
        user_id: User ID
        client_id: Client ID
        redis_client: Redis client (injected by decorator)

    Returns:
        Dict with audio_file_path, chunk_count, total_bytes, duration_seconds

    Note: user_email is fetched from the database when needed.
    """
    logger.info(f"🎵 Starting audio persistence for session {session_id}")

    # Setup audio persistence consumer group (separate from transcription consumer)
    audio_stream_name = f"audio:stream:{client_id}"
    audio_group_name = "audio_persistence"
    audio_consumer_name = f"persistence-{session_id[:8]}"

    try:
        await redis_client.xgroup_create(
            audio_stream_name,
            audio_group_name,
            "0",
            mkstream=True
        )
        logger.info(f"📦 Created audio persistence consumer group for {audio_stream_name}")
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.warning(f"Failed to create audio consumer group: {e}")
        logger.debug(f"Audio consumer group already exists for {audio_stream_name}")

    # Job control
    session_key = f"audio:session:{session_id}"
    max_runtime = 3540  # 59 minutes
    start_time = time.time()

    from advanced_omi_backend.config import CHUNK_DIR
    from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
    from wyoming.audio import AudioChunk

    # Ensure directory exists
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    # File rotation state
    current_conversation_id = None
    file_sink = None
    file_path = None
    wav_filename = None
    conversation_chunk_count = 0
    conversation_start_time = None

    # Audio collection stats (across all conversations in this session)
    total_chunk_count = 0
    total_bytes = 0
    end_signal_received = False
    consecutive_empty_reads = 0
    max_empty_reads = 3  # Exit after 3 consecutive empty reads (deterministic check)
    conversation_count = 0

    while True:
        # Check timeout
        if time.time() - start_time > max_runtime:
            logger.warning(f"⏱️ Timeout reached for audio persistence {session_id}")
            # Close current file if open
            if file_sink:
                await file_sink.close()
                logger.info(f"✅ Closed file on timeout: {wav_filename}")
            break

        # Check if session is finalizing (user stopped recording or WebSocket disconnected)
        session_status = await redis_client.hget(session_key, "status")
        if session_status and session_status.decode() in ["finalizing", "complete"]:
            logger.info(f"🛑 Session finalizing detected, writing final chunks...")
            # Give a brief moment for any in-flight chunks to arrive
            await asyncio.sleep(0.5)
            # Do one final read to write remaining chunks to current file
            if file_sink:
                try:
                    final_messages = await redis_client.xreadgroup(
                        audio_group_name,
                        audio_consumer_name,
                        {audio_stream_name: ">"},
                        count=50,
                        block=500
                    )
                    if final_messages:
                        for stream_name, msgs in final_messages:
                            for message_id, fields in msgs:
                                audio_data = fields.get(b"audio_data", b"")
                                chunk_id = fields.get(b"chunk_id", b"").decode()
                                if chunk_id != "END" and len(audio_data) > 0:
                                    chunk = AudioChunk(
                                        rate=16000,
                                        width=2,
                                        channels=1,
                                        audio=audio_data
                                    )
                                    await file_sink.write(chunk)
                                    conversation_chunk_count += 1
                                    total_chunk_count += 1
                                    total_bytes += len(audio_data)
                                await redis_client.xack(audio_stream_name, audio_group_name, message_id)
                        logger.info(f"📦 Final read wrote {len(final_messages[0][1]) if final_messages else 0} more chunks")
                except Exception as e:
                    logger.debug(f"Final audio read error (non-fatal): {e}")

                # Close final file
                await file_sink.close()
                logger.info(f"✅ Closed final file: {wav_filename} ({conversation_chunk_count} chunks)")
            break

        # Check for conversation change (file rotation signal)
        conversation_key = f"conversation:current:{session_id}"
        new_conversation_id = await redis_client.get(conversation_key)

        if new_conversation_id:
            new_conversation_id = new_conversation_id.decode()

            # Conversation changed - rotate to new file
            if new_conversation_id != current_conversation_id:
                # Close previous file if exists
                if file_sink:
                    await file_sink.close()
                    duration = (time.time() - conversation_start_time) if conversation_start_time else 0
                    logger.info(
                        f"✅ Closed conversation {current_conversation_id[:12]} file: {wav_filename} "
                        f"({conversation_chunk_count} chunks, {duration:.1f}s)"
                    )

                # Open new file for new conversation
                current_conversation_id = new_conversation_id
                conversation_count += 1
                conversation_chunk_count = 0
                conversation_start_time = time.time()

                timestamp = int(time.time() * 1000)
                wav_filename = f"{timestamp}_{client_id}_{current_conversation_id}.wav"
                file_path = CHUNK_DIR / wav_filename

                file_sink = LocalFileSink(
                    file_path=str(file_path),
                    sample_rate=16000,
                    channels=1,
                    sample_width=2
                )
                await file_sink.open()
                logger.info(
                    f"📁 Opened new file for conversation #{conversation_count} ({current_conversation_id[:12]}): {file_path}"
                )

                # Store file path in Redis (keyed by conversation_id, not session_id)
                audio_file_key = f"audio:file:{current_conversation_id}"
                await redis_client.set(audio_file_key, str(file_path), ex=3600)
                logger.info(f"💾 Stored audio file path in Redis: {audio_file_key}")
        else:
            # Key deleted - conversation ended, close current file
            if file_sink and current_conversation_id:
                await file_sink.close()
                duration = (time.time() - conversation_start_time) if conversation_start_time else 0
                logger.info(
                    f"✅ Closed conversation {current_conversation_id[:12]} file after conversation ended: {wav_filename} "
                    f"({conversation_chunk_count} chunks, {duration:.1f}s)"
                )
                file_sink = None  # Clear sink to prevent writing to closed file
                current_conversation_id = None

        # If no file open yet, wait for conversation to be created
        if not file_sink:
            await asyncio.sleep(0.5)
            continue

        # Read audio chunks from stream (non-blocking)
        try:
            audio_messages = await redis_client.xreadgroup(
                audio_group_name,
                audio_consumer_name,
                {audio_stream_name: ">"},
                count=20,  # Read up to 20 chunks at a time for efficiency
                block=500  # 500ms timeout
            )

            if audio_messages:
                # Reset empty read counter - we got messages
                consecutive_empty_reads = 0

                for stream_name, msgs in audio_messages:
                    for message_id, fields in msgs:
                        # Extract audio data
                        audio_data = fields.get(b"audio_data", b"")
                        chunk_id = fields.get(b"chunk_id", b"").decode()

                        # Check for END signal
                        if chunk_id == "END":
                            logger.info(f"📡 Received END signal in audio persistence")
                            end_signal_received = True
                        elif len(audio_data) > 0:
                            # Write chunk immediately to file
                            chunk = AudioChunk(
                                rate=16000,
                                width=2,
                                channels=1,
                                audio=audio_data
                            )
                            await file_sink.write(chunk)
                            conversation_chunk_count += 1
                            total_chunk_count += 1
                            total_bytes += len(audio_data)

                            # Log every 40 chunks to avoid spam
                            if total_chunk_count % 40 == 0:
                                logger.info(
                                    f"📦 Session {session_id[:12]}: {total_chunk_count} total chunks "
                                    f"(conversation {current_conversation_id[:12]}: {conversation_chunk_count} chunks)"
                                )

                        # ACK the message
                        await redis_client.xack(audio_stream_name, audio_group_name, message_id)
            else:
                # No new messages - stream might be empty
                if end_signal_received:
                    consecutive_empty_reads += 1
                    logger.info(f"📭 No new messages ({consecutive_empty_reads}/{max_empty_reads} empty reads after END signal)")

                    if consecutive_empty_reads >= max_empty_reads:
                        logger.info(f"✅ Stream empty after END signal - stopping audio collection")
                        break

        except Exception as audio_error:
            # Stream might not exist yet or other transient errors
            logger.debug(f"Audio stream read error (non-fatal): {audio_error}")

        await asyncio.sleep(0.1)  # Check every 100ms for responsiveness

    # Job complete - calculate final stats
    runtime_seconds = time.time() - start_time

    # Calculate duration (16kHz, 16-bit mono = 32000 bytes/second)
    if total_bytes > 0:
        duration = total_bytes / (16000 * 2 * 1)  # sample_rate * sample_width * channels
    else:
        logger.warning(f"⚠️ No audio chunks written for session {session_id}")
        duration = 0.0

    logger.info(
        f"🎵 Audio persistence job complete for session {session_id}: "
        f"{conversation_count} conversations, {total_chunk_count} total chunks, "
        f"{total_bytes / 1024 / 1024:.2f} MB, {runtime_seconds:.1f}s runtime"
    )

    # Clean up Redis tracking keys
    audio_job_key = f"audio_persistence:session:{session_id}"
    await redis_client.delete(audio_job_key)
    conversation_key = f"conversation:current:{session_id}"
    await redis_client.delete(conversation_key)
    logger.info(f"🧹 Cleaned up tracking keys for session {session_id}")

    return {
        "session_id": session_id,
        "conversation_count": conversation_count,
        "last_audio_file_path": str(file_path) if file_path else None,
        "total_chunk_count": total_chunk_count,
        "total_bytes": total_bytes,
        "duration_seconds": duration,
        "runtime_seconds": runtime_seconds
    }


# Enqueue wrapper functions

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
):
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
        description=f"Process audio for client {client_id}",
        meta={'audio_uuid': audio_uuid} if audio_uuid else {}
    )

    logger.info(f"📥 RQ: Enqueued audio job {job.id} for client {client_id}")
    return job


def enqueue_cropping(
    conversation_id: str,
    audio_path: str,
    user_id: str,
    priority: JobPriority = JobPriority.NORMAL
):
    """
    Enqueue an audio cropping job.

    Args:
        conversation_id: Conversation ID
        audio_path: Path to audio file
        user_id: User ID
        priority: Job priority level

    Returns:
        RQ Job object for tracking.
    """
    timeout_mapping = {
        JobPriority.URGENT: 300,  # 5 minutes
        JobPriority.HIGH: 240,    # 4 minutes
        JobPriority.NORMAL: 180,  # 3 minutes
        JobPriority.LOW: 120      # 2 minutes
    }

    job = default_queue.enqueue(
        process_cropping_job,
        conversation_id,
        audio_path,
        user_id,
        job_timeout=timeout_mapping.get(priority, 180),
        result_ttl=JOB_RESULT_TTL,
        job_id=f"crop_{conversation_id[:12]}",
        description=f"Crop audio for conversation {conversation_id[:12]}",
        meta={'conversation_id': conversation_id}
    )

    logger.info(f"📥 RQ: Enqueued cropping job {job.id} for conversation {conversation_id}")
    return job
