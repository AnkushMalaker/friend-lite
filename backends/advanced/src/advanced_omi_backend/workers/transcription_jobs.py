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
    JOB_RESULT_TTL,
    REDIS_URL,
)

logger = logging.getLogger(__name__)


async def apply_speaker_recognition(
    audio_path: str,
    transcript_text: str,
    words: list,
    segments: list,
    user_id: str,
    conversation_id: str = None
) -> list:
    """
    Apply speaker recognition to segments using the speaker recognition service.

    This is a reusable helper function that can be called from any job.

    Args:
        audio_path: Path to the audio file
        transcript_text: Full transcript text
        words: Word-level timing data
        segments: List of Conversation.SpeakerSegment objects
        user_id: User ID
        conversation_id: Optional conversation ID for logging

    Returns:
        Updated list of segments with identified speakers
    """
    try:
        from advanced_omi_backend.speaker_recognition_client import SpeakerRecognitionClient

        speaker_client = SpeakerRecognitionClient()
        if not speaker_client.enabled:
            logger.info(f"🎤 Speaker recognition disabled, using original speaker labels")
            return segments

        logger.info(f"🎤 Speaker recognition enabled, identifying speakers{f' for {conversation_id}' if conversation_id else ''}...")

        # Prepare transcript data with word-level timings
        transcript_data = {
            "text": transcript_text,
            "words": words
        }

        # Call speaker recognition service to match and identify speakers
        speaker_result = await speaker_client.diarize_identify_match(
            audio_path=audio_path,
            transcript_data=transcript_data,
            user_id=user_id
        )

        if not speaker_result or "segments" not in speaker_result:
            logger.info(f"🎤 Speaker recognition returned no segments, keeping original transcription segments")
            return segments

        speaker_identified_segments = speaker_result["segments"]
        logger.info(f"🎤 Speaker recognition returned {len(speaker_identified_segments)} identified segments")
        logger.info(f"🎤 Original segments: {len(segments)}")

        # Create time-based speaker mapping
        def get_speaker_at_time(timestamp: float, speaker_segments: list) -> str:
            """Get the identified speaker active at a given timestamp."""
            for seg in speaker_segments:
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", 0.0)
                if seg_start <= timestamp <= seg_end:
                    return seg.get("identified_as") or seg.get("speaker", "Unknown")
            return None

        # Update each segment's speaker based on its timestamp
        updated_count = 0
        for seg in segments:
            seg_mid = (seg.start + seg.end) / 2.0
            identified_speaker = get_speaker_at_time(seg_mid, speaker_identified_segments)

            if identified_speaker and identified_speaker != "Unknown":
                original_speaker = seg.speaker
                seg.speaker = identified_speaker
                updated_count += 1
                logger.debug(f"🎤   Segment [{seg.start:.1f}-{seg.end:.1f}] '{original_speaker}' -> '{identified_speaker}'")

        # Ensure segments remain sorted by start time
        segments.sort(key=lambda s: s.start)
        logger.info(f"🎤 Updated {updated_count}/{len(segments)} segments with speaker identifications")

        return segments

    except Exception as speaker_error:
        logger.warning(f"⚠️ Speaker recognition failed: {speaker_error}")
        logger.warning(f"Continuing with original transcription speaker labels")
        import traceback
        logger.debug(traceback.format_exc())
        return segments


@async_job(redis=True, beanie=True)
async def transcribe_full_audio_job(
    conversation_id: str,
    audio_uuid: str,
    audio_path: str,
    version_id: str,
    trigger: str = "reprocess",
    *,
    redis_client=None
) -> Dict[str, Any]:
    """
    RQ job function for transcribing full audio to text (transcription only, no speaker recognition).

    This job:
    1. Transcribes audio to text with generic speaker labels (Speaker 0, Speaker 1, etc.)
    2. Generates title and summary
    3. Saves transcript version to conversation
    4. Returns results for downstream jobs (speaker recognition, memory)

    Speaker recognition is handled by a separate job (recognise_speakers_job).

    Args:
        conversation_id: Conversation ID
        audio_uuid: Audio UUID (unused but kept for compatibility)
        audio_path: Path to audio file
        version_id: Version ID for new transcript
        trigger: Trigger source
        redis_client: Redis client (injected by decorator)

    Returns:
        Dict with processing results including transcript data for next job
    """
    from pathlib import Path
    from advanced_omi_backend.services.transcription import get_transcription_provider
    from advanced_omi_backend.models.conversation import Conversation

    logger.info(f"🔄 RQ: Starting transcript processing for conversation {conversation_id} (trigger: {trigger})")

    start_time = time.time()

    # Get the conversation
    conversation = await Conversation.find_one(Conversation.conversation_id == conversation_id)
    if not conversation:
        raise ValueError(f"Conversation {conversation_id} not found")

    # Use the provided audio path
    actual_audio_path = audio_path
    logger.info(f"📁 Using audio for transcription: {audio_path}")

    # Get the transcription provider
    provider = get_transcription_provider(mode="batch")
    if not provider:
        raise ValueError("No transcription provider available")

    provider_name = provider.name
    logger.info(f"Using transcription provider: {provider_name}")

    # Read the audio file
    audio_file_path = Path(actual_audio_path)
    if not audio_file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {actual_audio_path}")

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

    # Calculate processing time (transcription only)
    processing_time = time.time() - start_time

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

    # Prepare metadata (transcription only - speaker recognition will add its own metadata)
    metadata = {
        "trigger": trigger,
        "audio_file_size": len(audio_data),
        "segment_count": len(segments),
        "word_count": len(words),
        "words": words,  # Store words for speaker recognition job to read
        "speaker_recognition": {
            "enabled": False,
            "reason": "handled_by_separate_job"
        }
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

    # Update job metadata with title and summary for UI display
    from rq import get_current_job
    current_job = get_current_job()
    if current_job:
        if not current_job.meta:
            current_job.meta = {}
        current_job.meta.update({
            "conversation_id": conversation_id,
            "title": conversation.title,
            "summary": conversation.summary,
            "transcript_length": len(transcript_text),
            "word_count": len(words),
            "processing_time": processing_time
        })
        current_job.save_meta()

    return {
        "success": True,
        "conversation_id": conversation_id,
        "version_id": version_id,
        "audio_path": str(audio_file_path),
        "transcript": transcript_text,
        "segments": [seg.model_dump() for seg in speaker_segments],
        "words": words,  # Needed by speaker recognition
        "provider": provider_name,
        "processing_time_seconds": processing_time,
        "trigger": trigger
    }


@async_job(redis=True, beanie=True)
async def stream_speech_detection_job(
    session_id: str,
    user_id: str,
    client_id: str,
    *,
    redis_client=None
) -> Dict[str, Any]:
    """
    Listen for meaningful speech, optionally check for enrolled speakers, then start conversation.

    Simple flow:
        1. Listen for meaningful speech
        2. If speaker filter enabled → check for enrolled speakers
        3. If criteria met → start open_conversation_job and EXIT
        4. Conversation will restart new speech detection when complete

    Args:
        session_id: Stream session ID
        user_id: User ID
        client_id: Client ID
        redis_client: Redis client (injected by decorator)

    Returns:
        Dict with session info and conversation_job_id or no_speech_detected

    Note: user_email is fetched from the database when needed.
    """
    from advanced_omi_backend.services.audio_stream import TranscriptionResultsAggregator
    from advanced_omi_backend.utils.conversation_utils import analyze_speech
    from .conversation_jobs import open_conversation_job
    from rq import get_current_job

    logger.info(f"🔍 Starting speech detection for session {session_id[:12]}")

    # Setup
    aggregator = TranscriptionResultsAggregator(redis_client)
    current_job = get_current_job()
    session_key = f"audio:session:{session_id}"
    start_time = time.time()
    max_runtime = 3540  # 59 minutes

    # Get conversation count
    conversation_count_key = f"session:conversation_count:{session_id}"
    conversation_count_bytes = await redis_client.get(conversation_count_key)
    conversation_count = int(conversation_count_bytes) if conversation_count_bytes else 0

    # Check if speaker filtering is enabled
    speaker_filter_enabled = os.getenv("RECORD_ONLY_ENROLLED_SPEAKERS", "false").lower() == "true"
    logger.info(f"📊 Conversation #{conversation_count + 1}, Speaker filter: {'enabled' if speaker_filter_enabled else 'disabled'}")

    # Update job metadata to show status
    if current_job:
        if not current_job.meta:
            current_job.meta = {}
        current_job.meta.update({
            "status": "listening_for_speech",
            "session_id": session_id,
            "audio_uuid": session_id,
            "client_id": client_id,
            "session_level": True  # Mark as session-level job
        })
        current_job.save_meta()

    # Main loop: Listen for speech
    while True:
        # Exit conditions
        session_status = await redis_client.hget(session_key, "status")
        if session_status and session_status.decode() in ["complete", "closed"]:
            logger.info(f"🛑 Session ended, exiting")
            break

        if time.time() - start_time > max_runtime:
            logger.warning(f"⏱️ Max runtime reached, exiting")
            break

        # Get transcription results
        combined = await aggregator.get_combined_results(session_id)
        if not combined["text"]:
            await asyncio.sleep(2)
            continue

        # Step 1: Check for meaningful speech
        transcript_data = {"text": combined["text"], "words": combined.get("words", [])}
        speech_analysis = analyze_speech(transcript_data)

        logger.info(
            f"🔍 {speech_analysis.get('word_count', 0)} words, "
            f"{speech_analysis.get('duration', 0):.1f}s, "
            f"has_speech: {speech_analysis.get('has_speech', False)}"
        )

        if not speech_analysis.get("has_speech", False):
            await asyncio.sleep(2)
            continue

        logger.info(f"💬 Meaningful speech detected!")

        # Add session event for speech detected
        from datetime import datetime
        await redis_client.hset(
            session_key,
            "last_event",
            f"speech_detected:{datetime.utcnow().isoformat()}"
        )
        await redis_client.hset(
            session_key,
            "speech_detected_at",
            datetime.utcnow().isoformat()
        )

        # Step 2: If speaker filter enabled, check for enrolled speakers
        identified_speakers = []
        speaker_check_job = None  # Initialize for later reference
        if speaker_filter_enabled:
            logger.info(f"🎤 Enqueuing speaker check job...")

            # Add session event for speaker check starting
            await redis_client.hset(
                session_key,
                "last_event",
                f"speaker_check_starting:{datetime.utcnow().isoformat()}"
            )
            await redis_client.hset(
                session_key,
                "speaker_check_status",
                "checking"
            )
            from .speaker_jobs import check_enrolled_speakers_job

            # Enqueue speaker check as a separate trackable job
            speaker_check_job = transcription_queue.enqueue(
                check_enrolled_speakers_job,
                session_id,
                user_id,
                client_id,
                job_timeout=300,  # 5 minutes for speaker recognition
                result_ttl=600,
                job_id=f"speaker-check_{session_id[:12]}_{conversation_count}",
                description=f"Speaker check for conversation #{conversation_count+1}",
                meta={'audio_uuid': session_id, 'client_id': client_id}
            )

            # Poll for result (with timeout)
            max_wait = 30  # 30 seconds max
            poll_interval = 0.5
            waited = 0
            enrolled_present = False

            while waited < max_wait:
                try:
                    speaker_check_job.refresh()
                except Exception as e:
                    from rq.exceptions import NoSuchJobError
                    if isinstance(e, NoSuchJobError):
                        logger.warning(f"⚠️ Speaker check job disappeared from Redis (likely completed quickly), assuming not enrolled")
                        break
                    else:
                        raise

                if speaker_check_job.is_finished:
                    result = speaker_check_job.result
                    enrolled_present = result.get("enrolled_present", False)
                    identified_speakers = result.get("identified_speakers", [])
                    logger.info(f"✅ Speaker check completed: enrolled={enrolled_present}")

                    # Update session event for speaker check complete
                    await redis_client.hset(
                        session_key,
                        "last_event",
                        f"speaker_check_complete:{datetime.utcnow().isoformat()}"
                    )
                    await redis_client.hset(
                        session_key,
                        "speaker_check_status",
                        "enrolled" if enrolled_present else "not_enrolled"
                    )
                    if identified_speakers:
                        await redis_client.hset(
                            session_key,
                            "identified_speakers",
                            ",".join(identified_speakers)
                        )
                    break
                elif speaker_check_job.is_failed:
                    logger.warning(f"⚠️ Speaker check job failed, assuming not enrolled")

                    # Update session event for speaker check failed
                    await redis_client.hset(
                        session_key,
                        "last_event",
                        f"speaker_check_failed:{datetime.utcnow().isoformat()}"
                    )
                    await redis_client.hset(
                        session_key,
                        "speaker_check_status",
                        "failed"
                    )
                    break
                await asyncio.sleep(poll_interval)
                waited += poll_interval
            else:
                # Timeout - assume not enrolled
                logger.warning(f"⏱️ Speaker check timed out after {max_wait}s, assuming not enrolled")
                enrolled_present = False

                # Update session event for speaker check timeout
                await redis_client.hset(
                    session_key,
                    "last_event",
                    f"speaker_check_timeout:{datetime.utcnow().isoformat()}"
                )
                await redis_client.hset(
                    session_key,
                    "speaker_check_status",
                    "timeout"
                )

            # Log speaker check result but proceed with conversation regardless
            if enrolled_present:
                logger.info(f"✅ Enrolled speaker(s) found: {', '.join(identified_speakers) if identified_speakers else 'Unknown'}")
            else:
                logger.info(f"ℹ️ No enrolled speakers found, but proceeding with conversation anyway")

        # Step 3: Start conversation and EXIT
        speech_detected_at = time.time()
        open_job_key = f"open_conversation:session:{session_id}"

        # Enqueue conversation job with speech detection job ID
        from datetime import datetime

        speech_job_id = current_job.id if current_job else None

        open_job = transcription_queue.enqueue(
            open_conversation_job,
            session_id,
            user_id,
            client_id,
            speech_detected_at,
            speech_job_id,  # Pass speech detection job ID
            job_timeout=3600,
            result_ttl=JOB_RESULT_TTL,  # Use configured TTL (24 hours) instead of 10 minutes
            job_id=f"open-conv_{session_id[:12]}_{conversation_count}",
            description=f"Conversation #{conversation_count+1} for {session_id[:12]}",
            meta={'audio_uuid': session_id, 'client_id': client_id}
        )

        # Track the job
        await redis_client.set(open_job_key, open_job.id, ex=3600)

        # Store metadata in speech detection job
        if current_job:
            if not current_job.meta:
                current_job.meta = {}

            # Remove session_level flag now that conversation is starting
            current_job.meta.pop('session_level', None)

            current_job.meta.update({
                "conversation_job_id": open_job.id,
                "speaker_check_job_id": speaker_check_job.id if speaker_check_job else None,
                "detected_speakers": identified_speakers,
                "speech_detected_at": datetime.fromtimestamp(speech_detected_at).isoformat(),
                "session_id": session_id,
                "audio_uuid": session_id,  # For job grouping
                "client_id": client_id  # For job grouping
            })
            current_job.save_meta()

        logger.info(f"✅ Started conversation job {open_job.id}, exiting speech detection")

        return {
            "session_id": session_id,
            "user_id": user_id,
            "client_id": client_id,
            "conversation_job_id": open_job.id,
            "speech_detected_at": datetime.fromtimestamp(speech_detected_at).isoformat(),
            "runtime_seconds": time.time() - start_time
        }

    # Session ended without speech
    logger.info(f"✅ Session ended without speech")
    return {
        "session_id": session_id,
        "user_id": user_id,
        "client_id": client_id,
        "no_speech_detected": True,
        "runtime_seconds": time.time() - start_time
    }


