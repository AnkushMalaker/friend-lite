"""
Conversation-related RQ job functions.

This module contains jobs related to conversation management and updates.
"""

import logging
from typing import Dict, Any

from advanced_omi_backend.controllers.queue_controller import (
    transcription_queue,
    _ensure_beanie_initialized,
    REDIS_URL,
)

logger = logging.getLogger(__name__)


def open_conversation_job(
    conversation_id: str,
    session_id: str,
    user_id: str,
    redis_url: str = None
) -> Dict[str, Any]:
    """
    Long-running RQ job that continuously updates conversation with new transcription results.
    Runs until signaled to stop by finalize job.

    This is part of the V2 architecture using RQ jobs as orchestrators.

    Args:
        conversation_id: Conversation ID to update
        session_id: Stream session ID
        user_id: User ID
        redis_url: Redis connection URL

    Returns:
        Dict with conversation_id, final_result_count, runtime_seconds
    """
    import asyncio
    import time
    from datetime import datetime, timezone

    try:
        logger.info(f"📝 RQ: Opening conversation {conversation_id} for updates")

        # Get redis_url from environment if not provided
        if redis_url is None:
            redis_url = REDIS_URL

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def process():
                # Initialize Beanie
                await _ensure_beanie_initialized()

                # Import here to avoid circular dependencies
                from advanced_omi_backend.services.audio_stream import TranscriptionResultsAggregator
                from advanced_omi_backend.models.conversation import Conversation
                import redis.asyncio as redis_async

                # Connect to Redis
                redis_client = redis_async.from_url(redis_url)
                aggregator = TranscriptionResultsAggregator(redis_client)

                # Job control
                stop_signal_key = f"job:control:{conversation_id}:stop_open_conversation"
                max_runtime = 3600  # 1 hour max
                start_time = time.time()

                last_result_count = 0
                last_update_time = time.time()

                finalize_received = False
                finalize_timestamp = None
                grace_period_start = None
                grace_period_seconds = 15  # Wait 15s after last result before stopping

                while True:
                    # Check for stop signal from finalize job
                    if not finalize_received:
                        stop_signal = await redis_client.get(stop_signal_key)
                        if stop_signal:
                            finalize_timestamp = float(stop_signal.decode() if isinstance(stop_signal, bytes) else stop_signal)
                            finalize_received = True
                            logger.info(f"🛑 Finalize signal received at {finalize_timestamp}, entering grace period")

                    # Check timeout
                    if time.time() - start_time > max_runtime:
                        logger.warning(f"⏱️ Timeout reached for {conversation_id}")
                        break

                    # Get latest results
                    results = await aggregator.get_session_results(session_id)
                    current_count = len(results)

                    # Update conversation if new results arrived
                    if current_count > last_result_count:
                        # Reset grace period when new results arrive
                        if finalize_received:
                            grace_period_start = time.time()
                            logger.info(f"🔄 New results ({current_count - last_result_count} added), resetting grace period")
                        # Combine transcript
                        full_transcript = " ".join([r.get("text", "") for r in results if r.get("text")])

                        # Extract and combine segments
                        all_segments = []
                        for result in results:
                            if "segments" in result and result["segments"]:
                                for seg in result["segments"]:
                                    all_segments.append(Conversation.SpeakerSegment(
                                        start=seg.get("start", 0.0),
                                        end=seg.get("end", 0.0),
                                        text=seg.get("text", ""),
                                        speaker=seg.get("speaker", "Speaker 0"),
                                        confidence=seg.get("confidence")
                                    ))
                        all_segments.sort(key=lambda s: s.start)

                        # Update conversation in MongoDB
                        conversation = await Conversation.find_one(
                            Conversation.conversation_id == conversation_id
                        )

                        if conversation:
                            conversation.transcript = full_transcript
                            conversation.segments = all_segments
                            await conversation.save()

                            logger.info(
                                f"📊 Updated conversation {conversation_id}: "
                                f"{current_count} results, {len(full_transcript)} chars, {len(all_segments)} segments"
                            )
                        else:
                            logger.warning(f"⚠️ Conversation {conversation_id} not found")

                        last_result_count = current_count
                        last_update_time = time.time()
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

            result = loop.run_until_complete(process())
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Open conversation failed for {conversation_id}: {e}", exc_info=True)
        raise


async def create_conversation_from_streaming_results(
    session_id: str,
    user_id: str,
    user_email: str,
    client_id: str,
    transcript: str,
    results: list
) -> str:
    """
    Create a conversation from streaming transcription results.

    Args:
        session_id: Stream session ID
        user_id: User ID
        user_email: User email
        client_id: Client ID
        transcript: Combined transcript text
        results: List of transcription results from Redis

    Returns:
        Conversation ID
    """
    import uuid
    from datetime import datetime, timezone
    from advanced_omi_backend.models.conversation import Conversation

    # Ensure Beanie is initialized
    await _ensure_beanie_initialized()

    # Create conversation ID
    conversation_id = str(uuid.uuid4())

    # Extract and combine segments from all Redis results
    all_segments = []
    for result in results:
        if "segments" in result and result["segments"]:
            # Each result may have multiple segments
            for seg in result["segments"]:
                all_segments.append(Conversation.SpeakerSegment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", ""),
                    speaker=seg.get("speaker", "Speaker 0"),
                    confidence=seg.get("confidence")
                ))

    # Sort segments by start time
    all_segments.sort(key=lambda s: s.start)

    # Create transcript version
    transcript_version = Conversation.TranscriptVersion(
        version_id=str(uuid.uuid4()),
        transcript=transcript,
        segments=all_segments,
        provider=Conversation.TranscriptProvider.DEEPGRAM,  # From streaming worker
        model="nova-3",
        created_at=datetime.now(timezone.utc),
        processing_time_seconds=sum([r.get("processing_time", 0) for r in results]),
        metadata={
            "session_id": session_id,
            "chunk_count": len(results),
            "streaming_mode": True,
            "segment_count": len(all_segments)
        }
    )

    # Generate title and summary
    title = "Conversation"
    summary = "No content"

    try:
        from advanced_omi_backend.llm_client import async_generate

        # Generate title
        if transcript and len(transcript.strip()) >= 10:
            title_prompt = f"""Generate a concise, descriptive title (3-6 words) for this conversation transcript:

"{transcript[:500]}"

Rules:
- Maximum 6 words
- Capture the main topic or theme
- No quotes or special characters
- Examples: "Planning Weekend Trip", "Work Project Discussion", "Medical Appointment"

Title:"""
            title = await async_generate(title_prompt, temperature=0.3)
            title = title.strip().strip('"').strip("'") or "Conversation"
            logger.info(f"📝 Generated title: {title}")

        # Generate summary
        if transcript and len(transcript.strip()) >= 10:
            summary_prompt = f"""Generate a brief, informative summary (1-2 sentences, max 120 characters) for this conversation:

"{transcript[:1000]}"

Rules:
- Maximum 120 characters
- 1-2 complete sentences
- Capture key topics and outcomes
- Use present tense
- Be specific and informative

Summary:"""
            summary = await async_generate(summary_prompt, temperature=0.3)
            summary = summary.strip().strip('"').strip("'") or "No content"
            logger.info(f"📝 Generated summary: {summary}")

    except Exception as gen_error:
        logger.warning(f"⚠️ Failed to generate title/summary: {gen_error}")
        # Use fallbacks
        if transcript and len(transcript) > 0:
            words = transcript.split()[:6]
            title = " ".join(words)
            title = title[:40] + "..." if len(title) > 40 else title
            summary = transcript[:120] + "..." if len(transcript) > 120 else transcript

    # Create conversation document
    conversation = Conversation(
        conversation_id=conversation_id,
        audio_uuid=session_id,  # Use session_id as audio_uuid for now
        user_id=user_id,
        client_id=client_id,
        created_at=datetime.now(timezone.utc),
        title=title,
        summary=summary,
        transcript_versions=[transcript_version],
        active_transcript_version=transcript_version.version_id,
        memory_versions=[],
        active_memory_version=None,
        # Legacy fields (auto-populated from active versions)
        transcript=transcript,
        segments=all_segments,
        memories=[],
        memory_count=0
    )

    # Save conversation
    await conversation.insert()
    logger.info(f"✅ Created conversation {conversation_id} from streaming session {session_id}")

    # Enqueue memory processing
    from .memory_jobs import enqueue_memory_processing
    enqueue_memory_processing(
        conversation_id=conversation_id,
        user_id=user_id,
        user_email=user_email,
        client_id=client_id
    )

    return conversation_id
