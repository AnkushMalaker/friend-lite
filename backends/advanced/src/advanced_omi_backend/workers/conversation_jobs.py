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
    grace_period_start = None
    grace_period_seconds = 15  # Wait 15s after last result before stopping

    while True:
        # Check if session is finalizing (set by producer when recording stops)
        if not finalize_received:
            status = await redis_client.hget(session_key, "status")
            if status and status.decode() in ["finalizing", "complete"]:
                finalize_received = True
                logger.info(f"🛑 Session finalizing, entering grace period")

        # Check timeout
        if time.time() - start_time > max_runtime:
            logger.warning(f"⏱️ Timeout reached for {conversation_id}")
            break

        # Get combined results from aggregator
        combined = await aggregator.get_combined_results(session_id)
        current_count = combined["chunk_count"]

        # Update conversation if new results arrived
        if current_count > last_result_count:
            # Reset grace period when new results arrive
            if finalize_received:
                grace_period_start = time.time()
                logger.info(f"🔄 New results ({current_count - last_result_count} added), resetting grace period")

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
        else:
            # No new results
            if finalize_received and grace_period_start is None:
                # First time with no new results after finalize
                grace_period_start = time.time()
                logger.info(f"⏳ Starting grace period, waiting for final transcription results...")

        # Check grace period timeout
        if finalize_received and grace_period_start:
            grace_elapsed = time.time() - grace_period_start
            if grace_elapsed >= grace_period_seconds:
                logger.info(f"✅ Grace period complete ({grace_elapsed:.1f}s), no new results. Stopping conversation updates.")
                break

        await asyncio.sleep(1)  # Check every second for responsiveness

    logger.info(f"✅ Conversation {conversation_id} closed after updates")

    # Clean up Redis tracking key so new speech detection jobs can start
    open_job_key = f"open_conversation:session:{session_id}"
    await redis_client.delete(open_job_key)
    logger.info(f"🧹 Cleaned up tracking key {open_job_key}")

    return {
        "conversation_id": conversation_id,
        "final_result_count": last_result_count,
        "runtime_seconds": time.time() - start_time
    }
