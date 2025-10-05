"""
Audio-related RQ job functions.

This module contains jobs related to audio file processing and cropping.
"""

import os
import logging
from typing import Dict, Any, Optional

from advanced_omi_backend.models.job import JobPriority

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

            # Enqueue transcript processing job (outside async context)
            if result.get("success") and result.get("conversation_id"):
                from .transcription_jobs import enqueue_transcript_processing
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
        description=f"Process audio for client {client_id}"
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
        description=f"Crop audio for {audio_uuid[:8]}"
    )

    logger.info(f"📥 RQ: Enqueued cropping job {job.id} for audio {audio_uuid}")
    return job
