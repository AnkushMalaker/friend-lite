"""
Transcription-related RQ job functions.

This module contains all jobs related to speech-to-text transcription processing.
"""

import asyncio
import os
import logging
import time
from typing import Dict, Any

from advanced_omi_backend.models.job import JobPriority, BaseRQJob, async_job

from advanced_omi_backend.controllers.queue_controller import (
    transcription_queue,
    redis_conn,
    _ensure_beanie_initialized,
    JOB_RESULT_TTL,
    REDIS_URL,
)

logger = logging.getLogger(__name__)


@async_job(redis=True, beanie=True)
async def process_transcript_job(
    conversation_id: str,
    audio_uuid: str,
    audio_path: str,
    version_id: str,
    user_id: str,
    trigger: str = "reprocess",
    redis_client=None
) -> Dict[str, Any]:
    """
    RQ job function for transcript processing.

    This function handles both new transcription and reprocessing.
    The 'trigger' parameter indicates the source: 'new', 'reprocess', 'retry', etc.

    Args:
        conversation_id: Conversation ID
        audio_uuid: Audio UUID (unused but kept for compatibility)
        audio_path: Path to audio file
        version_id: Version ID for new transcript
        user_id: User ID (unused but kept for compatibility)
        trigger: Trigger source
        redis_client: Redis client (injected by decorator)

    Returns:
        Dict with processing results
    """
    from pathlib import Path
    from advanced_omi_backend.services.transcription import get_transcription_provider
    from advanced_omi_backend.models.conversation import Conversation

    logger.info(f"🔄 RQ: Starting transcript processing for conversation {conversation_id} (trigger: {trigger})")

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

    logger.info(f"📊 Transcription complete: {len(transcript_text)} chars, {len(segments)} segments, {len(words)} words")

    # Speaker Recognition Integration
    speaker_identified_segments = []
    try:
        from advanced_omi_backend.speaker_recognition_client import SpeakerRecognitionClient

        speaker_client = SpeakerRecognitionClient()
        if speaker_client.enabled:
            logger.info(f"🎤 Speaker recognition enabled, identifying speakers...")

            # Prepare transcript data with word-level timings
            transcript_data = {
                "text": transcript_text,
                "words": words
            }

            # Call speaker recognition service to match and identify speakers
            speaker_result = await speaker_client.diarize_identify_match(
                audio_path=str(audio_file_path),
                transcript_data=transcript_data,
                user_id=user_id
            )

            if speaker_result and "segments" in speaker_result:
                speaker_identified_segments = speaker_result["segments"]
                logger.info(f"🎤 Speaker recognition returned {len(speaker_identified_segments)} identified segments")

                # Replace original segments with identified segments
                if speaker_identified_segments:
                    segments = speaker_identified_segments
                    logger.info(f"🎤 Using identified segments from speaker recognition service")
            else:
                logger.info(f"🎤 Speaker recognition returned no segments, using original transcription segments")
        else:
            logger.info(f"🎤 Speaker recognition disabled, using original speaker labels from transcription")

    except Exception as speaker_error:
        logger.warning(f"⚠️ Speaker recognition failed: {speaker_error}")
        logger.warning(f"Continuing with original transcription speaker labels")
        import traceback
        logger.debug(traceback.format_exc())

    # Calculate processing time
    processing_time = time.time() - start_time

    # Get the conversation using Beanie
    conversation = await Conversation.find_one(Conversation.conversation_id == conversation_id)
    if not conversation:
        logger.error(f"Conversation {conversation_id} not found")
        return {"success": False, "error": "Conversation not found"}

    # Convert segments to SpeakerSegment objects
    speaker_segments = []
    for seg in segments:
        # Use identified_as if available (from speaker recognition), otherwise use speaker label
        speaker_name = seg.get("identified_as") or seg.get("speaker", "Unknown")

        speaker_segments.append(
            Conversation.SpeakerSegment(
                start=seg.get("start", 0),
                end=seg.get("end", 0),
                text=seg.get("text", ""),
                speaker=speaker_name,
                confidence=seg.get("confidence")
            )
        )

    logger.info(f"📊 Created {len(speaker_segments)} speaker segments")

    # Add new transcript version
    provider_normalized = provider_name.lower() if provider_name else "unknown"

    # Prepare metadata with speaker recognition info
    metadata = {
        "trigger": trigger,
        "audio_file_size": len(audio_data),
        "segment_count": len(segments),
        "word_count": len(words)
    }

    # Add speaker recognition metadata if available
    if speaker_identified_segments:
        # Extract unique identified speakers
        identified_speakers = set()
        for seg in speaker_identified_segments:
            identified_as = seg.get("identified_as", "Unknown")
            if identified_as != "Unknown":
                identified_speakers.add(identified_as)

        metadata["speaker_recognition"] = {
            "enabled": True,
            "identified_speakers": list(identified_speakers),
            "speaker_count": len(identified_speakers),
            "total_segments": len(speaker_identified_segments)
        }
    else:
        metadata["speaker_recognition"] = {
            "enabled": False,
            "reason": "disabled or failed"
        }

    conversation.add_transcript_version(
        version_id=version_id,
        transcript=transcript_text,
        segments=speaker_segments,
        provider=Conversation.TranscriptProvider(provider_normalized),
        model=getattr(provider, 'model', 'unknown'),
        processing_time_seconds=processing_time,
        metadata=metadata,
        set_as_active=True
    )

    # Generate title and summary from transcript using LLM
    if transcript_text and len(transcript_text.strip()) > 0:
        try:
            from advanced_omi_backend.llm_client import async_generate

            # Prepare prompt for LLM
            prompt = f"""Based on this conversation transcript, generate a concise title and summary.

Transcript:
{transcript_text[:2000]}

Respond in this exact format:
Title: <concise title under 50 characters>
Summary: <brief summary under 150 characters>"""

            logger.info(f"🤖 Generating title/summary using LLM for conversation {conversation_id}")
            llm_response = await async_generate(prompt, temperature=0.7)

            # Parse LLM response
            lines = llm_response.strip().split('\n')
            title = None
            summary = None

            for line in lines:
                if line.startswith('Title:'):
                    title = line.replace('Title:', '').strip()
                elif line.startswith('Summary:'):
                    summary = line.replace('Summary:', '').strip()

            # Use LLM-generated title/summary if valid, otherwise fallback
            if title and len(title) > 0:
                conversation.title = title[:50] + "..." if len(title) > 50 else title
            else:
                # Fallback to first sentence if LLM didn't provide title
                first_sentence = transcript_text.split('.')[0].strip()
                conversation.title = first_sentence[:50] + "..." if len(first_sentence) > 50 else first_sentence

            if summary and len(summary) > 0:
                conversation.summary = summary[:150] + "..." if len(summary) > 150 else summary
            else:
                # Fallback to truncated transcript if LLM didn't provide summary
                conversation.summary = transcript_text[:150] + "..." if len(transcript_text) > 150 else transcript_text

            logger.info(f"✅ Generated title: '{conversation.title}', summary: '{conversation.summary}'")

        except Exception as llm_error:
            logger.warning(f"⚠️ LLM title/summary generation failed: {llm_error}")
            # Fallback to simple truncation
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


@async_job(redis=True, beanie=True)
async def stream_speech_detection_job(
    session_id: str,
    user_id: str,
    user_email: str,
    client_id: str,
    redis_client=None
) -> Dict[str, Any]:
    """
    Job that monitors transcription stream for speech (STREAMING MODE ONLY).

    Decorated with @async_job to handle setup/teardown automatically.

        Job lifecycle:
        1. Monitors transcription stream for speech
        2. When speech detected:
           - Checks if conversation already open (prevents duplicates)
           - If no open conversation: creates conversation + starts open_conversation_job
           - Exits (job completes)
        3. New stream_speech_detection_job can be started when conversation closes

        This architecture alternates between "listening for speech" and "actively recording conversation".

        This is part of the V2 architecture using RQ jobs as orchestrators.

        For batch/upload mode, conversations are created upfront and process_transcript_job is used.

        Args:
            session_id: Stream session ID
            user_id: User ID
            user_email: User email
            client_id: Client ID

        Returns:
            Dict with session_id, conversation_id, open_conversation_job_id, runtime_seconds
        """
    from advanced_omi_backend.services.audio_stream import TranscriptionResultsAggregator
    from .conversation_jobs import open_conversation_job

    logger.info(f"🔍 RQ: Starting stream speech detection for session {session_id}")

    # Use redis_client from decorator
    aggregator = TranscriptionResultsAggregator(redis_client)

    # Job control
    session_key = f"audio:session:{session_id}"
    max_runtime = 3540  # 59 minutes (graceful exit before RQ timeout at 60 min)
    start_time = time.time()

    conversation_id = None
    open_conversation_job_id = None

    while True:
        # Check if session has ended (status = "finalizing" or "complete")
        session_status = await redis_client.hget(session_key, "status")
        if session_status:
            status_str = session_status.decode() if isinstance(session_status, bytes) else session_status
            if status_str in ["finalizing", "complete"]:
                logger.info(f"🛑 Session {status_str}, stopping speech detection")
                break

        # Check timeout
        if time.time() - start_time > max_runtime:
            logger.warning(f"⏱️ Timeout reached for {session_id}")
            break

        # Get combined transcription results (aggregator does the combining)
        combined = await aggregator.get_combined_results(session_id)

        if not combined["text"]:
            await asyncio.sleep(2)  # Check every 2 seconds
            continue

        # Analyze for speech using centralized detection from utils
        from advanced_omi_backend.utils.conversation_utils import analyze_speech
        transcript_data = {
            "text": combined["text"],
            "words": combined["words"]
        }
        speech_analysis = analyze_speech(transcript_data)
        has_speech = speech_analysis["has_speech"]

        print(f"🔍 SPEECH ANALYSIS: session={session_id}, has_speech={has_speech}, conv_id={conversation_id}, words={speech_analysis.get('word_count', 0)}")
        logger.info(
            f"🔍 Speech analysis for {session_id}: has_speech={has_speech}, "
            f"conversation_id={conversation_id}, word_count={speech_analysis.get('word_count', 0)}"
        )

        if has_speech and not conversation_id:
            print(f"💬 SPEECH DETECTED! Checking if enrolled speakers present...")
            logger.info(f"💬 Speech detected in {session_id}!")

            # Check if we should filter by enrolled speakers (two-stage filter: text first, then speaker)
            record_only_enrolled = os.getenv("RECORD_ONLY_ENROLLED_SPEAKERS", "false").lower() == "true"

            if record_only_enrolled:
                logger.info(f"🎤 Checking if enrolled speakers are present...")

                from advanced_omi_backend.speaker_recognition_client import SpeakerRecognitionClient

                # Get raw transcription results (with chunk IDs)
                raw_results = await aggregator.get_session_results(session_id)

                # Check if enrolled speaker is speaking
                speaker_client = SpeakerRecognitionClient()
                enrolled_speaker_present = await speaker_client.check_if_enrolled_speaker_present(
                    redis_client=redis_client,
                    client_id=client_id,
                    session_id=session_id,
                    user_id=user_id,
                    transcription_results=raw_results
                )

                if not enrolled_speaker_present:
                    logger.info(f"⏭️ Meaningful speech detected but not from enrolled speakers, continuing to listen...")
                    await asyncio.sleep(2)
                    continue

                logger.info(f"✅ Enrolled speaker detected! Proceeding to create conversation...")

            # Check if conversation job already running for this session
            open_job_key = f"open_conversation:session:{session_id}"
            existing_job = await redis_client.get(open_job_key)

            if existing_job:
                # Already have an open conversation job running
                open_conversation_job_id = existing_job.decode()
                logger.info(f"✅ Conversation job already running: {open_conversation_job_id}")
            else:
                # No conversation job running - enqueue one
                speech_detected_at = time.time()
                logger.info(f"📝 Enqueueing open_conversation_job (speech detected at {speech_detected_at})")

                # Start open_conversation_job to create and monitor conversation
                open_job = transcription_queue.enqueue(
                    open_conversation_job,
                    session_id,
                    user_id,
                    user_email,
                    client_id,
                    speech_detected_at,
                    job_timeout=3600,
                    result_ttl=600,
                    job_id=f"open-conv_{session_id[:12]}",
                    description=f"Open conversation for session {session_id[:12]}"
                )
                open_conversation_job_id = open_job.id

                # Store job tracking (TTL handles cleanup automatically)
                await redis_client.set(
                    open_job_key,
                    open_job.id,
                    ex=3600  # Expire after 1 hour
                )

                logger.info(f"✅ Enqueued conversation job {open_job.id}")

            # Exit this job now that conversation job is running
            logger.info(f"🏁 Exiting speech detection job - conversation job is now managing session")
            break
        else:
            if not has_speech:
                logger.debug(f"⏭️ No speech detected yet (words: {speech_analysis.get('word_count', 0)})")
            else:
                logger.debug(f"ℹ️ Speech detected but conversation already exists: {conversation_id}")

        await asyncio.sleep(2)  # Check every 2 seconds

    logger.info(f"✅ Stream speech detection complete for {session_id}")

    return {
        "session_id": session_id,
        "open_conversation_job_id": open_conversation_job_id,
        "runtime_seconds": time.time() - start_time
    }


@async_job(redis=True, beanie=True)
async def finalize_streaming_transcription_job(
    session_id: str,
    user_id: str,
    user_email: str,
    client_id: str,
    redis_client=None
) -> Dict[str, Any]:
    """
    RQ job function for finalizing streaming transcription.

    This job:
    1. Coordinates with open_conversation_job if it exists (V2 architecture)
    2. Waits for final chunks to be processed
    3. Aggregates results from Redis Streams
    4. Updates existing conversation OR creates new one if speech detected
    5. Reads audio file from audio_streaming_persistence_job

    Args:
        session_id: Stream session ID
        user_id: User ID
        user_email: User email
        client_id: Client ID
        redis_client: Redis client (injected by decorator)

    Returns:
        Dict with processing results
    """
    import re
    import uuid
    from advanced_omi_backend.models.conversation import Conversation, create_conversation
    from advanced_omi_backend.services.audio_stream import TranscriptionResultsAggregator

    logger.info(f"🔄 RQ: Finalizing streaming transcription for session {session_id}")

    # Use redis_client parameter
    aggregator = TranscriptionResultsAggregator(redis_client)

    # Look up conversation_id from Redis (created by open_conversation_job)
    conversation_id = None
    conversation_key = f"conversation:session:{session_id}"
    stored_conversation_id = await redis_client.get(conversation_key)
    if stored_conversation_id:
        conversation_id = stored_conversation_id.decode()
        logger.info(f"📋 Found conversation {conversation_id} for session {session_id}")

    # Look up open_conversation_job_id from Redis
    open_conversation_job_id = None
    open_job_key = f"open_conversation:session:{session_id}"
    stored_job_id = await redis_client.get(open_job_key)
    if stored_job_id:
        open_conversation_job_id = stored_job_id.decode()
        logger.info(f"📋 Found open conversation job {open_conversation_job_id}")

    # V2 Architecture: Wait for open_conversation_job to complete if it exists
    # Note: Session status is already set to "finalizing" by producer
    # The conversation job will detect this and enter grace period automatically
    if open_conversation_job_id:
        logger.info(f"📋 Waiting for open_conversation_job {open_conversation_job_id} to complete...")

        # Wait for open_conversation_job to finish by checking if tracking key still exists
        # The job deletes this key when it completes
        max_wait_seconds = 30  # Job should finish within grace period (~15s) + buffer
        wait_interval = 1  # Check every second for responsiveness
        elapsed = 0

        while elapsed < max_wait_seconds:
            job_still_running = await redis_client.exists(open_job_key)

            if not job_still_running:
                logger.info(f"✅ open_conversation_job completed after {elapsed}s")
                break

            await asyncio.sleep(wait_interval)
            elapsed += wait_interval
        else:
            logger.warning(f"⚠️ open_conversation_job did not complete within {max_wait_seconds}s, proceeding anyway")

        # Get combined results from aggregator (does the combining for us)
        combined = await aggregator.get_combined_results(session_id)
        full_transcript = combined["text"]
        logger.info(f"📝 Combined transcript: {len(full_transcript)} characters")

        # Convert segments to SpeakerSegment objects
        all_segments = []
        for seg in combined["segments"]:
            all_segments.append(Conversation.SpeakerSegment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", ""),
                speaker=seg.get("speaker", "Speaker 0"),
                confidence=seg.get("confidence")
            ))
        logger.info(f"📊 Extracted {len(all_segments)} segments")

        # Get audio file path from Redis (written by audio_streaming_persistence_job)
        audio_file_key = f"audio:file:{session_id}"
        file_path_bytes = await redis_client.get(audio_file_key)

        if not file_path_bytes:
            logger.error(f"❌ Audio file path not found in Redis for session {session_id}")
            logger.warning(f"⚠️ Audio persistence job may have failed - check job status")
            return {"success": False, "error": "Audio file not found"}

        file_path = file_path_bytes.decode()
        logger.info(f"📁 Retrieved audio file path from Redis: {file_path}")

        # Extract filename from path for return value
        from pathlib import Path
        wav_filename = Path(file_path).name

        # Speaker Recognition Integration
        speaker_identified_segments = []
        try:
            from advanced_omi_backend.speaker_recognition_client import SpeakerRecognitionClient

            speaker_client = SpeakerRecognitionClient()
            if speaker_client.enabled:
                logger.info(f"🎤 Speaker recognition enabled, identifying speakers...")

                # Prepare transcript data with word-level timings (from combined results)
                transcript_data = {
                    "text": full_transcript,
                    "words": combined["words"]
                }

                # Call speaker recognition service to match and identify speakers
                speaker_result = await speaker_client.diarize_identify_match(
                    audio_path=file_path,
                    transcript_data=transcript_data,
                    user_id=user_id
                )

                if speaker_result and "segments" in speaker_result:
                    speaker_identified_segments = speaker_result["segments"]
                    logger.info(f"🎤 Speaker recognition returned {len(speaker_identified_segments)} identified segments")
                    logger.info(f"🎤 Original streaming segments: {len(all_segments)}")

                    # Create time-based speaker mapping
                    # Map time ranges to identified speakers for updating original segments
                    def get_speaker_at_time(timestamp: float, speaker_segments: list) -> str:
                        """Get the identified speaker active at a given timestamp."""
                        for seg in speaker_segments:
                            seg_start = seg.get("start", 0.0)
                            seg_end = seg.get("end", 0.0)
                            # Check if timestamp falls within this segment
                            if seg_start <= timestamp <= seg_end:
                                return seg.get("identified_as") or seg.get("speaker", "Unknown")
                        return None

                    # Update each streaming segment's speaker based on its timestamp
                    updated_count = 0
                    for seg in all_segments:
                        # Use the middle of the segment for speaker lookup
                        seg_mid = (seg.start + seg.end) / 2.0
                        identified_speaker = get_speaker_at_time(seg_mid, speaker_identified_segments)

                        if identified_speaker and identified_speaker != "Unknown":
                            original_speaker = seg.speaker
                            seg.speaker = identified_speaker
                            updated_count += 1
                            logger.debug(f"🎤   Segment [{seg.start:.1f}-{seg.end:.1f}] '{original_speaker}' -> '{identified_speaker}'")

                    # Ensure segments remain sorted by start time
                    all_segments.sort(key=lambda s: s.start)
                    logger.info(f"🎤 Updated {updated_count}/{len(all_segments)} streaming segments with speaker identifications")
                else:
                    logger.info(f"🎤 Speaker recognition returned no segments, keeping original transcription segments")
            else:
                logger.info(f"🎤 Speaker recognition disabled, using original speaker labels from transcription")

        except Exception as speaker_error:
            logger.warning(f"⚠️ Speaker recognition failed: {speaker_error}")
            logger.warning(f"Continuing with original transcription speaker labels")
            import traceback
            logger.debug(traceback.format_exc())

    # Mark session as complete in Redis
    session_key = f"audio:session:{session_id}"
    await redis_client.hset(session_key, mapping={
        "status": "complete",
        "completed_at": str(time.time())
    })
    logger.info(f"✅ Marked session {session_id} as complete")

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

        if conversation_id:
            # Update existing conversation
            conversation = await Conversation.find_one(Conversation.conversation_id == conversation_id)
            if conversation:
                conversation.transcript = full_transcript
                conversation.segments = all_segments
                await conversation.save()
                logger.info(f"✅ Updated conversation {conversation_id} with complete transcript and {len(all_segments)} segments")

                # Enqueue batch transcription of complete audio file for final high-quality transcript
                final_version_id = str(uuid.uuid4())
                logger.info(f"📝 Enqueueing batch transcription of complete audio file for final transcript version {final_version_id}")

                enqueue_transcript_processing(
                    conversation_id=conversation_id,
                    audio_uuid=session_id,
                    audio_path=file_path,
                    version_id=final_version_id,
                    user_id=user_id,
                    priority=JobPriority.HIGH,
                    trigger="streaming_final"
                )
                logger.info(f"✅ Batch transcription job enqueued for conversation {conversation_id}")

                return {
                    "success": True,
                    "conversation_id": conversation_id,
                    "session_id": session_id,
                    "action": "updated",
                    "transcript_length": len(full_transcript),
                    "segment_count": len(all_segments),
                    "audio_file": wav_filename,
                    "batch_transcription_version": final_version_id
                }
            else:
                logger.error(f"❌ Conversation {conversation_id} not found")
                return {"success": False, "error": "Conversation not found"}

        else:
            # No conversation created yet - check for speech
            transcript_clean = full_transcript.strip()
            words_only = re.sub(r'[^\w\s]', '', transcript_clean)
            words = words_only.split()
            word_count = len(words)
            avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
            short_words = [w for w in words if len(w) <= 2]
            short_word_ratio = len(short_words) / word_count if word_count > 0 else 0

            has_speech = bool(
                transcript_clean and
                len(transcript_clean) > 50 and
                word_count >= 8 and
                avg_word_length >= 3.0 and
                short_word_ratio < 0.5
            )

            logger.info(
                f"🔍 Speech detection: {has_speech} "
                f"(length: {len(transcript_clean)}, words: {word_count}, "
                f"avg_len: {avg_word_length:.1f}, short_ratio: {short_word_ratio:.1%})"
            )

            if has_speech:
                logger.info(f"💬 Speech detected - creating conversation")

                # Create conversation with transcript
                new_conversation_id = str(uuid.uuid4())

                # Generate title and summary from transcript using LLM
                try:
                    from advanced_omi_backend.llm_client import async_generate

                    # Prepare prompt for LLM
                    prompt = f"""Based on this conversation transcript, generate a concise title and summary.

Transcript:
{full_transcript[:2000]}

Respond in this exact format:
Title: <concise title under 50 characters>
Summary: <brief summary under 150 characters>"""

                    logger.info(f"🤖 Generating title/summary using LLM for new conversation")
                    llm_response = await async_generate(prompt, temperature=0.7)

                    # Parse LLM response
                    lines = llm_response.strip().split('\n')
                    title = None
                    summary = None

                    for line in lines:
                        if line.startswith('Title:'):
                            title = line.replace('Title:', '').strip()
                        elif line.startswith('Summary:'):
                            summary = line.replace('Summary:', '').strip()

                    # Use LLM-generated title/summary if valid, otherwise fallback
                    if not title or len(title) == 0:
                        first_sentence = full_transcript.split('.')[0].strip() if full_transcript else "Conversation"
                        title = first_sentence[:50] + "..." if len(first_sentence) > 50 else first_sentence

                    if not summary or len(summary) == 0:
                        summary = full_transcript[:120] + "..." if len(full_transcript) > 120 else (full_transcript or "No content")

                    # Truncate if needed
                    title = title[:50] + "..." if len(title) > 50 else title
                    summary = summary[:150] + "..." if len(summary) > 150 else summary

                    logger.info(f"✅ Generated title: '{title}', summary: '{summary}'")

                except Exception as llm_error:
                    logger.warning(f"⚠️ LLM title/summary generation failed: {llm_error}")
                    # Fallback to simple truncation
                    first_sentence = full_transcript.split('.')[0].strip() if full_transcript else "Conversation"
                    title = first_sentence[:50] + "..." if len(first_sentence) > 50 else first_sentence
                    summary = full_transcript[:120] + "..." if len(full_transcript) > 120 else (full_transcript or "No content")

                conversation = create_conversation(
                    conversation_id=new_conversation_id,
                    audio_uuid=session_id,
                    user_id=user_id,
                    client_id=client_id,
                    title=title,
                    summary=summary,
                    transcript=full_transcript,
                    segments=all_segments
                )

                await conversation.insert()
                logger.info(f"✅ Created conversation {new_conversation_id} with transcript ({len(full_transcript)} chars)")

                # Enqueue batch transcription of complete audio file for final high-quality transcript
                final_version_id = str(uuid.uuid4())
                logger.info(f"📝 Enqueueing batch transcription of complete audio file for final transcript version {final_version_id}")

                enqueue_transcript_processing(
                    conversation_id=new_conversation_id,
                    audio_uuid=session_id,
                    audio_path=file_path,
                    version_id=final_version_id,
                    user_id=user_id,
                    priority=JobPriority.HIGH,
                    trigger="streaming_final"
                )
                logger.info(f"✅ Batch transcription job enqueued for conversation {new_conversation_id}")

                return {
                    "success": True,
                    "conversation_id": new_conversation_id,
                    "session_id": session_id,
                    "action": "created",
                    "transcript_length": len(full_transcript),
                    "segment_count": len(all_segments),
                    "audio_file": wav_filename,
                    "batch_transcription_version": final_version_id
                }
            else:
                logger.info(f"⏭️ No speech detected in streaming session")

                return {
                    "success": True,
                    "conversation_id": None,
                    "session_id": session_id,
                    "action": "skipped",
                    "reason": "no_speech",
                    "transcript_length": len(full_transcript),
                    "audio_file": wav_filename
                }


# Enqueue wrapper functions

def enqueue_transcript_processing(
    conversation_id: str,
    audio_uuid: str,
    audio_path: str,
    version_id: str,
    user_id: str,
    priority: JobPriority = JobPriority.NORMAL,
    trigger: str = "reprocess"
):
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


def enqueue_streaming_finalization(
    session_id: str,
    user_id: str,
    user_email: str,
    client_id: str,
    priority: JobPriority = JobPriority.HIGH
):
    """
    Enqueue a streaming transcription finalization job.

    This job handles the final processing when a streaming session ends.
    Conversation ID, job ID, and audio file path are looked up from Redis inside the job.

    Args:
        session_id: Stream session ID
        user_id: User ID
        user_email: User email
        client_id: Client ID
        priority: Job priority (default HIGH for responsive UI)

    Returns:
        RQ Job object for tracking
    """
    timeout_mapping = {
        JobPriority.URGENT: 180,  # 3 minutes
        JobPriority.HIGH: 120,    # 2 minutes
        JobPriority.NORMAL: 90,   # 1.5 minutes
        JobPriority.LOW: 60       # 1 minute
    }

    job = transcription_queue.enqueue(
        finalize_streaming_transcription_job,
        session_id,
        user_id,
        user_email,
        client_id,
        job_timeout=timeout_mapping.get(priority, 120),
        result_ttl=JOB_RESULT_TTL,
        job_id=f"finalize-stream_{session_id[:12]}",
        description=f"Finalize streaming session {session_id[:12]}"
    )

    logger.info(f"📥 RQ: Enqueued streaming finalization job {job.id} for session {session_id}")
    return job


# RQ Callback functions for speech detection job
def on_speech_detection_success(job, connection, result, *args, **kwargs):
    """
    RQ callback when speech detection job succeeds.

    Args:
        job: The RQ Job instance
        connection: Redis connection
        result: Job result dictionary
        *args, **kwargs: Additional arguments
    """
    session_id = job.meta.get('session_id', 'unknown')
    conversation_id = result.get('conversation_id')

    if conversation_id:
        logger.info(f"✅ Speech detection succeeded for session {session_id}: Created conversation {conversation_id}")
    else:
        logger.info(f"✅ Speech detection succeeded for session {session_id}: No conversation created (no speech or already exists)")


def on_speech_detection_failure(job, connection, type, value, traceback):
    """
    RQ callback when speech detection job fails.

    Args:
        job: The RQ Job instance
        connection: Redis connection
        type: Exception type
        value: Exception value
        traceback: Exception traceback
    """
    session_id = job.meta.get('session_id', 'unknown')
    logger.error(f"❌ Speech detection failed for session {session_id}: {type.__name__}: {value}")
    logger.debug(f"Speech detection failure traceback: {traceback}")
