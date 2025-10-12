"""
Conversation-related RQ job functions.

This module contains jobs related to conversation management and updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any

from advanced_omi_backend.models.job import async_job
from advanced_omi_backend.controllers.queue_controller import (
    transcription_queue,
    REDIS_URL,
)

logger = logging.getLogger(__name__)


@async_job(redis=True, beanie=True)
async def open_conversation_job(
    session_id: str,
    user_id: str,
    user_email: str,
    client_id: str,
    speech_detected_at: float,
    redis_client=None
) -> Dict[str, Any]:
    """
    Long-running RQ job that creates and continuously updates conversation with transcription results.

    Creates conversation when speech is detected, then monitors and updates until session ends.

    Args:
        session_id: Stream session ID
        user_id: User ID
        user_email: User email
        client_id: Client ID
        speech_detected_at: Timestamp when speech was first detected
        redis_client: Redis client (injected by decorator)

    Returns:
        Dict with conversation_id, final_result_count, runtime_seconds
    """
    from advanced_omi_backend.services.audio_stream import TranscriptionResultsAggregator
    from advanced_omi_backend.models.conversation import Conversation

    import uuid
    from advanced_omi_backend.models.conversation import create_conversation

    logger.info(f"📝 Creating and opening conversation for session {session_id} (speech detected at {speech_detected_at})")

    # Create minimal streaming conversation
    conversation_id = str(uuid.uuid4())
    conversation = create_conversation(
        conversation_id=conversation_id,
        audio_uuid=session_id,
        user_id=user_id,
        client_id=client_id,
        title="Recording...",
        summary="Transcribing audio..."
    )

    # Save to database
    await conversation.insert()
    logger.info(f"✅ Created streaming conversation {conversation_id} for session {session_id}")

    # Store conversation_id in Redis for finalize job to find
    conversation_key = f"conversation:session:{session_id}"
    await redis_client.set(conversation_key, conversation_id, ex=3600)
    logger.info(f"💾 Stored conversation ID in Redis: {conversation_key}")

    # Use redis_client parameter
    aggregator = TranscriptionResultsAggregator(redis_client)

    # Job control
    session_key = f"audio:session:{session_id}"
    max_runtime = 3540  # 59 minutes (graceful exit before RQ timeout at 60 min)
    start_time = time.time()

    last_result_count = 0
    finalize_received = False

    while True:
        # Check if session is finalizing (set by producer when recording stops)
        if not finalize_received:
            status = await redis_client.hget(session_key, "status")
            if status and status.decode() in ["finalizing", "complete"]:
                finalize_received = True
                logger.info(f"🛑 Session finalizing, waiting for audio persistence job to complete...")
                break  # Exit immediately when finalize signal received

        # Check timeout
        if time.time() - start_time > max_runtime:
            logger.warning(f"⏱️ Timeout reached for {conversation_id}")
            break

        # Get combined results from aggregator
        combined = await aggregator.get_combined_results(session_id)
        current_count = combined["chunk_count"]

        # Update conversation if new results arrived
        if current_count > last_result_count:
            # Update conversation in MongoDB
            conversation = await Conversation.find_one(
                Conversation.conversation_id == conversation_id
            )

            if conversation:
                conversation.transcript = combined["text"]
                conversation.segments = combined["segments"]
                await conversation.save()

                logger.info(
                    f"📊 Updated conversation {conversation_id}: "
                    f"{current_count} results, {len(combined['text'])} chars, {len(combined['segments'])} segments"
                )
            else:
                logger.warning(f"⚠️ Conversation {conversation_id} not found")

            last_result_count = current_count

        await asyncio.sleep(1)  # Check every second for responsiveness

    logger.info(f"✅ Conversation {conversation_id} updates complete, waiting for audio file to be ready...")

    # Wait for audio_streaming_persistence_job to complete and write the file path
    # Poll for the audio file key - this is deterministic, not a timeout-based grace period
    audio_file_key = f"audio:file:{session_id}"
    file_path_bytes = None
    max_wait_audio = 30  # Maximum 30 seconds to wait for audio file
    wait_start = time.time()

    while time.time() - wait_start < max_wait_audio:
        file_path_bytes = await redis_client.get(audio_file_key)
        if file_path_bytes:
            wait_duration = time.time() - wait_start
            logger.info(f"✅ Audio file ready after {wait_duration:.1f}s")
            break

        # Check if still within reasonable time
        elapsed = time.time() - wait_start
        if elapsed % 5 == 0:  # Log every 5 seconds
            logger.info(f"⏳ Waiting for audio file... ({elapsed:.0f}s elapsed)")

        await asyncio.sleep(0.5)  # Check every 500ms

    if not file_path_bytes:
        logger.error(f"❌ Audio file path not found in Redis after {max_wait_audio}s")
        logger.warning(f"⚠️ Audio persistence job may have failed or is still running - cannot enqueue batch transcription")
    else:
        file_path = file_path_bytes.decode()
        logger.info(f"📁 Retrieved audio file path: {file_path}")

        # Enqueue complete batch processing job chain
        from advanced_omi_backend.controllers.queue_controller import start_batch_processing_jobs

        job_ids = start_batch_processing_jobs(
            conversation_id=conversation_id,
            audio_uuid=session_id,
            user_id=user_id,
            user_email=user_email,
            audio_file_path=file_path
        )

        logger.info(
            f"📥 RQ: Enqueued batch processing chain: "
            f"{job_ids['transcription']} → {job_ids['speaker_recognition']} → {job_ids['memory']}"
        )

        # Wait a moment to ensure jobs are registered in RQ
        await asyncio.sleep(0.5)

    # DON'T mark session as complete yet - dependent jobs are still processing
    # Session remains in "finalizing" status until process_memory_job completes
    logger.info(f"⏳ Session {session_id} remains in 'finalizing' status while batch jobs process")

    # Clean up Redis streams to prevent memory leaks
    try:
        # Delete the audio input stream
        audio_stream_key = f"audio:stream:{client_id}"
        await redis_client.delete(audio_stream_key)
        logger.info(f"🧹 Deleted audio stream: {audio_stream_key}")

        # Delete the transcription results stream
        results_stream_key = f"transcription:results:{session_id}"
        await redis_client.delete(results_stream_key)
        logger.info(f"🧹 Deleted results stream: {results_stream_key}")

        # Set TTL on session key (expire after 1 hour)
        await redis_client.expire(session_key, 3600)
        logger.info(f"⏰ Set TTL on session key: {session_key}")
    except Exception as cleanup_error:
        logger.warning(f"⚠️ Error during stream cleanup: {cleanup_error}")

    # Clean up Redis tracking key so new speech detection jobs can start
    open_job_key = f"open_conversation:session:{session_id}"
    await redis_client.delete(open_job_key)
    logger.info(f"🧹 Cleaned up tracking key {open_job_key}")

    return {
        "conversation_id": conversation_id,
        "final_result_count": last_result_count,
        "runtime_seconds": time.time() - start_time
    }
