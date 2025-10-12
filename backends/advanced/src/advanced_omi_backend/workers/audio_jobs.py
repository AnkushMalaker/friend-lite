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
                from advanced_omi_backend.controllers.queue_controller import transcription_queue, memory_queue, JOB_RESULT_TTL

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
                    meta={'audio_uuid': result["audio_uuid"]}
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
                    meta={'audio_uuid': result["audio_uuid"]}
                )
                logger.info(f"📥 RQ: Enqueued speaker recognition job {speaker_job.id} (depends on {transcript_job.id})")

                # Job 3: Extract memories (depends on speaker recognition)
                memory_job = memory_queue.enqueue(
                    process_memory_job,
                    None,  # client_id - will be read from conversation in DB
                    user_id,
                    "",  # user_email - will be read from user in DB
                    conversation_id,
                    depends_on=speaker_job,
                    job_timeout=1800,
                    result_ttl=JOB_RESULT_TTL,
                    job_id=f"memory_{conversation_id[:8]}",
                    description=f"Extract memories for {conversation_id[:8]}",
                    meta={'audio_uuid': result["audio_uuid"]}
                )
                logger.info(f"📥 RQ: Enqueued memory job {memory_job.id} (depends on {speaker_job.id})")

                result["transcript_job_id"] = transcript_job.id
                result["speaker_job_id"] = speaker_job.id
                result["memory_job_id"] = memory_job.id

            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Audio processing failed for client {client_id}: {e}")
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


@async_job(redis=True, beanie=True)
async def audio_streaming_persistence_job(
    session_id: str,
    user_id: str,
    user_email: str,
    client_id: str,
    redis_client=None
) -> Dict[str, Any]:
    """
    Long-running RQ job that collects audio chunks from Redis stream and writes to disk progressively.

    Runs in parallel with transcription processing to reduce memory pressure on WebSocket.

    Args:
        session_id: Stream session ID
        user_id: User ID
        user_email: User email
        client_id: Client ID
        redis_client: Redis client (injected by decorator)

    Returns:
        Dict with audio_file_path, chunk_count, total_bytes, duration_seconds
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

    # Audio collection
    audio_chunks = []
    chunk_count = 0
    total_bytes = 0
    end_signal_received = False
    consecutive_empty_reads = 0
    max_empty_reads = 3  # Exit after 3 consecutive empty reads (deterministic check)

    while True:
        # Check timeout
        if time.time() - start_time > max_runtime:
            logger.warning(f"⏱️ Timeout reached for audio persistence {session_id}")
            break

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
                            audio_chunks.append(audio_data)
                            chunk_count += 1
                            total_bytes += len(audio_data)

                            # Log every 40 chunks to avoid spam
                            if chunk_count % 40 == 0:
                                logger.info(f"📦 Collected {chunk_count} audio chunks ({total_bytes / 1024 / 1024:.2f} MB)")

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

    # Write complete audio file
    if audio_chunks:
        from advanced_omi_backend.audio_utils import write_audio_file

        complete_audio = b''.join(audio_chunks)
        timestamp = int(time.time() * 1000)

        logger.info(f"💾 Writing {len(audio_chunks)} chunks ({total_bytes / 1024 / 1024:.2f} MB) to disk")

        wav_filename, file_path, duration = await write_audio_file(
            raw_audio_data=complete_audio,
            audio_uuid=session_id,
            client_id=client_id,
            user_id=user_id,
            user_email=user_email,
            timestamp=timestamp,
            validate=False
        )
        logger.info(f"✅ Wrote audio file: {wav_filename} ({duration:.1f}s, {chunk_count} chunks)")

        # Store file path in Redis for finalize job to find
        audio_file_key = f"audio:file:{session_id}"
        await redis_client.set(audio_file_key, file_path, ex=3600)
        logger.info(f"💾 Stored audio file path in Redis: {audio_file_key}")
    else:
        logger.warning(f"⚠️ No audio chunks collected for session {session_id}")
        file_path = None
        duration = 0.0

    # Clean up Redis tracking key
    audio_job_key = f"audio_persistence:session:{session_id}"
    await redis_client.delete(audio_job_key)
    logger.info(f"🧹 Cleaned up tracking key {audio_job_key}")

    return {
        "session_id": session_id,
        "audio_file_path": file_path,
        "chunk_count": chunk_count,
        "total_bytes": total_bytes,
        "duration_seconds": duration,
        "runtime_seconds": time.time() - start_time
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
    client_id: str,
    user_id: str,
    audio_uuid: str,
    original_path: str,
    speech_segments: list,
    output_path: str,
    priority: JobPriority = JobPriority.NORMAL
):
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
        description=f"Crop audio for {audio_uuid[:8]}",
        meta={'audio_uuid': audio_uuid}
    )

    logger.info(f"📥 RQ: Enqueued cropping job {job.id} for audio {audio_uuid}")
    return job
