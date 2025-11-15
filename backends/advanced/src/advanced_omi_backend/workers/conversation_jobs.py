"""
Conversation-related RQ job functions.

This module contains jobs related to conversation management and updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any

from advanced_omi_backend.models.job import async_job

logger = logging.getLogger(__name__)


@async_job(redis=True, beanie=True)
async def open_conversation_job(
    session_id: str,
    user_id: str,
    client_id: str,
    speech_detected_at: float,
    speech_job_id: str = None,
    *,
    redis_client=None
) -> Dict[str, Any]:
    """
    Long-running RQ job that creates and continuously updates conversation with transcription results.

    Creates conversation when speech is detected, then monitors and updates until session ends.

    Args:
        session_id: Stream session ID
        user_id: User ID
        client_id: Client ID
        speech_detected_at: Timestamp when speech was first detected
        speech_job_id: Optional speech detection job ID to update with conversation_id
        redis_client: Redis client (injected by decorator)

    Returns:
        Dict with conversation_id, final_result_count, runtime_seconds

    Note: user_email is fetched from the database when needed.
    """
    from advanced_omi_backend.services.audio_stream import TranscriptionResultsAggregator
    from advanced_omi_backend.models.conversation import Conversation, create_conversation
    from rq import get_current_job

    logger.info(f"📝 Creating and opening conversation for session {session_id} (speech detected at {speech_detected_at})")

    # Get current job for meta storage
    current_job = get_current_job()

    # Create minimal streaming conversation (conversation_id auto-generated)
    conversation = create_conversation(
        audio_uuid=session_id,
        user_id=user_id,
        client_id=client_id,
        title="Recording...",
        summary="Transcribing audio..."
    )

    # Save to database
    await conversation.insert()
    conversation_id = conversation.conversation_id  # Get the auto-generated ID
    logger.info(f"✅ Created streaming conversation {conversation_id} for session {session_id}")

    # Update speech detection job metadata with conversation_id
    if speech_job_id:
        try:
            from rq.job import Job
            from advanced_omi_backend.controllers.queue_controller import redis_conn

            speech_job = Job.fetch(speech_job_id, connection=redis_conn)
            if speech_job and speech_job.meta:
                # Only update if conversation_id not already set (first conversation wins)
                if not speech_job.meta.get('conversation_id'):
                    speech_job.meta['conversation_id'] = conversation_id
                    # Remove session_level flag - now linked to conversation
                    speech_job.meta.pop('session_level', None)
                    speech_job.save_meta()
                    logger.info(f"🔗 Updated speech job {speech_job_id[:12]} with conversation_id")
                else:
                    logger.info(f"⏭️ Speech job {speech_job_id[:12]} already linked to conversation {speech_job.meta.get('conversation_id')[:12]}")

                # Also update the speaker check job if referenced in speech job metadata
                # Only update if it doesn't already have a conversation_id (first conversation wins)
                speaker_check_job_id = speech_job.meta.get('speaker_check_job_id')
                if speaker_check_job_id:
                    try:
                        speaker_check_job = Job.fetch(speaker_check_job_id, connection=redis_conn)
                        if speaker_check_job and speaker_check_job.meta:
                            # Only update if conversation_id not already set
                            if not speaker_check_job.meta.get('conversation_id'):
                                speaker_check_job.meta['conversation_id'] = conversation_id
                                speaker_check_job.save_meta()
                                logger.info(f"🔗 Updated speaker check job {speaker_check_job_id} with conversation_id")
                            else:
                                logger.info(f"⏭️ Speaker check job {speaker_check_job_id} already linked to conversation {speaker_check_job.meta.get('conversation_id')[:12]}")
                    except Exception as speaker_err:
                        logger.warning(f"⚠️ Failed to update speaker check job metadata: {speaker_err}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update speech job metadata: {e}")

    # Store conversation_id in Redis for finalize job to find
    conversation_key = f"conversation:session:{session_id}"
    await redis_client.set(conversation_key, conversation_id, ex=3600)
    logger.info(f"💾 Stored conversation ID in Redis: {conversation_key}")

    # Signal audio persistence job to rotate to this conversation's file
    current_conversation_key = f"conversation:current:{session_id}"
    await redis_client.set(current_conversation_key, conversation_id, ex=3600)
    logger.info(f"🔄 Signaled audio persistence to rotate file for conversation {conversation_id[:12]}")

    # Use redis_client parameter
    aggregator = TranscriptionResultsAggregator(redis_client)

    # Job control
    session_key = f"audio:session:{session_id}"
    max_runtime = 3540  # 59 minutes (graceful exit before RQ timeout at 60 min)
    start_time = time.time()

    last_result_count = 0
    finalize_received = False

    # Inactivity timeout configuration
    import os
    inactivity_timeout_seconds = float(os.getenv("SPEECH_INACTIVITY_THRESHOLD_SECONDS", "60"))
    inactivity_timeout_minutes = inactivity_timeout_seconds / 60
    last_meaningful_speech_time = time.time()  # Initialize with conversation start
    timeout_triggered = False  # Track if closure was due to timeout
    last_inactivity_log_time = time.time()  # Track when we last logged inactivity
    last_word_count = 0  # Track word count to detect actual new speech

    logger.info(f"📊 Conversation timeout configured: {inactivity_timeout_minutes} minutes ({inactivity_timeout_seconds}s)")

    while True:
        # Check if session is finalizing (set by producer when recording stops)
        if not finalize_received:
            status = await redis_client.hget(session_key, "status")
            if status and status.decode() in ["finalizing", "complete"]:
                finalize_received = True
                logger.info(f"🛑 Session finalizing, waiting for audio persistence job to complete...")
                break  # Exit immediately when finalize signal received

        # Check max runtime timeout
        if time.time() - start_time > max_runtime:
            logger.warning(f"⏱️ Max runtime reached for {conversation_id}")
            break

        # Get combined results from aggregator
        combined = await aggregator.get_combined_results(session_id)
        current_count = combined["chunk_count"]

        # Analyze speech content using detailed analysis
        from advanced_omi_backend.utils.conversation_utils import analyze_speech

        transcript_data = {
            "text": combined["text"],
            "words": combined.get("words", [])
        }
        speech_analysis = analyze_speech(transcript_data)

        # Extract speaker information from segments
        speakers = []
        segments = combined.get("segments", [])
        if segments:
            for seg in segments:
                speaker = seg.get("speaker", "Unknown")
                if speaker and speaker != "Unknown" and speaker not in speakers:
                    speakers.append(speaker)

        # Check if NEW speech arrived (word count increased)
        # Track word count instead of chunk count to avoid resetting on noise/silence chunks
        current_word_count = speech_analysis.get("word_count", 0)
        if current_word_count > last_word_count:
            last_meaningful_speech_time = time.time()
            last_word_count = current_word_count
            # Store timestamp in Redis for visibility/debugging
            await redis_client.set(
                f"conversation:last_speech:{conversation_id}",
                last_meaningful_speech_time,
                ex=3600  # 1 hour TTL
            )
            logger.debug(f"🗣️ New speech detected (word count: {current_word_count}), updated last_speech timestamp")

        # Update job meta with current state
        if current_job:
            if not current_job.meta:
                current_job.meta = {}

            from datetime import datetime

            # Set created_at only once (first time we update metadata)
            if 'created_at' not in current_job.meta:
                current_job.meta['created_at'] = datetime.now().isoformat()

            current_job.meta.update({
                "conversation_id": conversation_id,
                "audio_uuid": session_id,  # Link to session for job grouping
                "client_id": client_id,  # Ensure client_id is always present
                "transcript": combined["text"][:500] + "..." if len(combined["text"]) > 500 else combined["text"],  # First 500 chars
                "transcript_length": len(combined["text"]),
                "speakers": speakers,
                "word_count": speech_analysis.get("word_count", 0),
                "duration_seconds": speech_analysis.get("duration", 0),
                "has_speech": speech_analysis.get("has_speech", False),
                "last_update": datetime.now().isoformat(),
                "inactivity_seconds": time.time() - last_meaningful_speech_time,
                "chunks_processed": current_count
            })
            current_job.save_meta()

        # Check inactivity timeout and log every 10 seconds
        inactivity_duration = time.time() - last_meaningful_speech_time
        current_time = time.time()

        # Log inactivity every 10 seconds
        if current_time - last_inactivity_log_time >= 10:
            logger.info(f"⏱️ Time since last speech: {inactivity_duration:.1f}s (timeout: {inactivity_timeout_seconds:.0f}s)")
            last_inactivity_log_time = current_time

        if inactivity_duration > inactivity_timeout_seconds:
            logger.info(
                f"🕐 Conversation {conversation_id} inactive for "
                f"{inactivity_duration/60:.1f} minutes (threshold: {inactivity_timeout_minutes} min), "
                f"auto-closing conversation (session remains active for next conversation)..."
            )
            # DON'T set session to finalizing - just close this conversation
            # Session remains "active" so new conversations can be created
            # Only user manual stop or WebSocket disconnect should finalize the session
            timeout_triggered = True
            finalize_received = True
            break

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
    # Audio persistence now writes files per-conversation, so key uses conversation_id
    audio_file_key = f"audio:file:{conversation_id}"
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
            logger.info(f"⏳ Waiting for audio file (conversation {conversation_id[:12]})... ({elapsed:.0f}s elapsed)")

        await asyncio.sleep(0.5)  # Check every 500ms

    if not file_path_bytes:
        logger.error(f"❌ Audio file path not found in Redis after {max_wait_audio}s (key: {audio_file_key})")
        logger.warning(f"⚠️ Audio persistence job may not have rotated file yet - cannot enqueue batch transcription")
    else:
        file_path = file_path_bytes.decode()
        logger.info(f"📁 Retrieved audio file path: {file_path}")

        # Update conversation with audio file path
        conversation = await Conversation.find_one(
            Conversation.conversation_id == conversation_id
        )
        if conversation:
            # Store just the filename (relative to CHUNK_DIR)
            from pathlib import Path
            audio_filename = Path(file_path).name
            conversation.audio_path = audio_filename
            await conversation.save()
            logger.info(f"💾 Updated conversation {conversation_id[:12]} with audio_path: {audio_filename}")
        else:
            logger.warning(f"⚠️ Conversation {conversation_id} not found for audio_path update")

        # Enqueue post-conversation processing pipeline
        from advanced_omi_backend.controllers.queue_controller import start_post_conversation_jobs

        job_ids = start_post_conversation_jobs(
            conversation_id=conversation_id,
            audio_uuid=session_id,
            audio_file_path=file_path,
            user_id=user_id,
            post_transcription=True  # Run batch transcription for streaming audio
        )

        logger.info(
            f"📥 Pipeline: transcribe({job_ids['transcription']}) → "
            f"speaker({job_ids['speaker_recognition']}) → "
            f"[memory({job_ids['memory']}) + title({job_ids['title_summary']})]"
        )

        # Wait a moment to ensure jobs are registered in RQ
        await asyncio.sleep(0.5)

    # Clean up Redis streams to prevent memory leaks
    try:
        # NOTE: Do NOT delete audio:stream:{client_id} here!
        # The audio stream is per-client (WebSocket connection), not per-conversation.
        # It's still actively receiving audio and will be reused by the next conversation.
        # Only delete it on WebSocket disconnect (handled in websocket_controller.py)

        # Delete the transcription results stream (per-session/conversation)
        results_stream_key = f"transcription:results:{session_id}"
        await redis_client.delete(results_stream_key)
        logger.info(f"🧹 Deleted results stream: {results_stream_key}")

        # Set TTL on session key (expire after 1 hour)
        await redis_client.expire(session_key, 3600)
        logger.info(f"⏰ Set TTL on session key: {session_key}")
    except Exception as cleanup_error:
        logger.warning(f"⚠️ Error during stream cleanup: {cleanup_error}")

    # Clean up Redis tracking keys so speech detection job knows conversation is complete
    open_job_key = f"open_conversation:session:{session_id}"
    await redis_client.delete(open_job_key)
    logger.info(f"🧹 Cleaned up tracking key {open_job_key}")

    # Delete the conversation:current signal so audio persistence knows conversation ended
    current_conversation_key = f"conversation:current:{session_id}"
    await redis_client.delete(current_conversation_key)
    logger.info(f"🧹 Deleted conversation:current signal for session {session_id[:12]}")

    # Increment conversation count for this session
    conversation_count_key = f"session:conversation_count:{session_id}"
    conversation_count = await redis_client.incr(conversation_count_key)
    await redis_client.expire(conversation_count_key, 3600)  # 1 hour TTL
    logger.info(f"📊 Conversation count for session {session_id}: {conversation_count}")

    # Check if session is still active (user still recording) and restart listening jobs
    session_status = await redis_client.hget(session_key, "status")
    if session_status:
        status_str = session_status.decode() if isinstance(session_status, bytes) else session_status

        if status_str == "active":
            # Session still active - enqueue new speech detection for next conversation
            logger.info(f"🔄 Enqueueing new speech detection (conversation #{conversation_count + 1})")

            from advanced_omi_backend.controllers.queue_controller import transcription_queue, redis_conn, JOB_RESULT_TTL
            from advanced_omi_backend.workers.transcription_jobs import stream_speech_detection_job

            # Enqueue speech detection job for next conversation (audio persistence keeps running)
            speech_job = transcription_queue.enqueue(
                stream_speech_detection_job,
                session_id,
                user_id,
                client_id,
                job_timeout=3600,
                result_ttl=JOB_RESULT_TTL,
                job_id=f"speech-detect_{session_id[:12]}_{conversation_count}",
                description=f"Listening for speech (conversation #{conversation_count + 1})",
                meta={'audio_uuid': session_id, 'client_id': client_id, 'session_level': True}
            )

            # Store job ID for cleanup (keyed by client_id for WebSocket cleanup)
            try:
                redis_conn.set(f"speech_detection_job:{client_id}", speech_job.id, ex=3600)
                logger.info(f"📌 Stored speech detection job ID for client {client_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to store job ID for {client_id}: {e}")

            logger.info(f"✅ Enqueued speech detection job {speech_job.id}")
        else:
            logger.info(f"Session {session_id} status={status_str}, not restarting (user stopped recording)")
    else:
        logger.info(f"Session {session_id} not found, not restarting (session ended)")

    return {
        "conversation_id": conversation_id,
        "conversation_count": conversation_count,
        "final_result_count": last_result_count,
        "runtime_seconds": time.time() - start_time,
        "timeout_triggered": timeout_triggered
    }


@async_job(redis=True, beanie=True)
async def generate_title_summary_job(
    conversation_id: str,
    *,
    redis_client=None
) -> Dict[str, Any]:
    """
    Generate title and summary for a conversation using LLM.

    This job runs independently of transcription and memory jobs to ensure
    conversations always get meaningful titles and summaries, even if other
    processing steps fail.

    Uses the utility functions from conversation_utils for consistent title/summary generation.

    Args:
        conversation_id: Conversation ID
        redis_client: Redis client (injected by decorator)

    Returns:
        Dict with generated title and summary
    """
    from advanced_omi_backend.models.conversation import Conversation
    from advanced_omi_backend.utils.conversation_utils import (
        generate_title_with_speakers,
        generate_summary_with_speakers
    )

    logger.info(f"📝 Starting title/summary generation for conversation {conversation_id}")

    start_time = time.time()

    # Get the conversation
    conversation = await Conversation.find_one(Conversation.conversation_id == conversation_id)
    if not conversation:
        logger.error(f"Conversation {conversation_id} not found")
        return {"success": False, "error": "Conversation not found"}

    # Get segments from active transcript version
    segments = conversation.segments or []

    if not segments or len(segments) == 0:
        logger.warning(f"⚠️ No segments available for conversation {conversation_id}")
        return {
            "success": False,
            "error": "No segments available",
            "conversation_id": conversation_id
        }

    # Generate title and summary using speaker-aware utilities
    try:
        logger.info(f"🤖 Generating title/summary using LLM for conversation {conversation_id}")

        # Convert segments to dict format expected by utils
        segment_dicts = [
            {
                "speaker": seg.speaker,
                "text": seg.text,
                "start": seg.start,
                "end": seg.end
            }
            for seg in segments
        ]

        # Generate title and summary with speaker awareness
        title = await generate_title_with_speakers(segment_dicts)
        summary = await generate_summary_with_speakers(segment_dicts)

        conversation.title = title
        conversation.summary = summary

        logger.info(f"✅ Generated title: '{conversation.title}', summary: '{conversation.summary}'")

    except Exception as gen_error:
        logger.error(f"❌ Title/summary generation failed: {gen_error}")
        return {
            "success": False,
            "error": str(gen_error),
            "conversation_id": conversation_id,
            "processing_time_seconds": time.time() - start_time
        }

    # Save the updated conversation
    await conversation.save()

    processing_time = time.time() - start_time

    # Update job metadata
    from rq import get_current_job
    current_job = get_current_job()
    if current_job:
        if not current_job.meta:
            current_job.meta = {}
        current_job.meta.update({
            "conversation_id": conversation_id,
            "title": conversation.title,
            "summary": conversation.summary,
            "segment_count": len(segments),
            "processing_time": processing_time
        })
        current_job.save_meta()

    logger.info(f"✅ Title/summary generation completed for {conversation_id} in {processing_time:.2f}s")

    return {
        "success": True,
        "conversation_id": conversation_id,
        "title": conversation.title,
        "summary": conversation.summary,
        "processing_time_seconds": processing_time
    }
