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
    user_id: str,
    trigger: str = "reprocess",
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
        user_id: User ID
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

    # Calculate processing time (transcription only)
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

    return {
        "success": True,
        "conversation_id": conversation_id,
        "version_id": version_id,
        "audio_path": str(audio_file_path),
        "user_id": user_id,
        "transcript": transcript_text,
        "segments": [seg.model_dump() for seg in speaker_segments],
        "words": words,  # Needed by speaker recognition
        "provider": provider_name,
        "processing_time_seconds": processing_time,
        "trigger": trigger
    }


@async_job(redis=True, beanie=True)
async def recognise_speakers_job(
    conversation_id: str,
    version_id: str,
    audio_path: str,
    user_id: str,
    transcript_text: str,
    words: list,
    redis_client=None
) -> Dict[str, Any]:
    """
    RQ job function for identifying speakers in a transcribed conversation.

    This job runs after transcription and:
    1. Calls speaker recognition service to identify speakers
    2. Updates the transcript version with identified speaker labels
    3. Returns results for downstream jobs (memory)

    Args:
        conversation_id: Conversation ID
        version_id: Transcript version ID to update
        audio_path: Path to audio file
        user_id: User ID
        transcript_text: Transcript text from transcription job
        words: Word-level timing data from transcription job
        redis_client: Redis client (injected by decorator)

    Returns:
        Dict with processing results
    """
    from advanced_omi_backend.models.conversation import Conversation
    from advanced_omi_backend.speaker_recognition_client import SpeakerRecognitionClient

    logger.info(f"🎤 RQ: Starting speaker recognition for conversation {conversation_id}")

    start_time = time.time()

    # Get the conversation
    conversation = await Conversation.find_one(Conversation.conversation_id == conversation_id)
    if not conversation:
        logger.error(f"Conversation {conversation_id} not found")
        return {"success": False, "error": "Conversation not found"}

    # Find the transcript version to update
    transcript_version = None
    for version in conversation.transcript_versions:
        if version.version_id == version_id:
            transcript_version = version
            break

    if not transcript_version:
        logger.error(f"Transcript version {version_id} not found")
        return {"success": False, "error": "Transcript version not found"}

    # Check if speaker recognition is enabled
    speaker_client = SpeakerRecognitionClient()
    if not speaker_client.enabled:
        logger.info(f"🎤 Speaker recognition disabled, skipping")
        return {
            "success": True,
            "conversation_id": conversation_id,
            "version_id": version_id,
            "speaker_recognition_enabled": False,
            "processing_time_seconds": 0
        }

    # Call speaker recognition service
    try:
        logger.info(f"🎤 Calling speaker recognition service...")

        # Read transcript text and words from the transcript version
        # (Parameters may be empty if called via job dependency)
        actual_transcript_text = transcript_text or transcript_version.transcript or ""
        actual_words = words if words else []

        # If words not provided, we need to get them from metadata
        if not actual_words and transcript_version.metadata:
            actual_words = transcript_version.metadata.get("words", [])

        if not actual_transcript_text:
            logger.warning(f"🎤 No transcript text found in version {version_id}")
            return {
                "success": False,
                "conversation_id": conversation_id,
                "version_id": version_id,
                "error": "No transcript text available",
                "processing_time_seconds": 0
            }

        transcript_data = {
            "text": actual_transcript_text,
            "words": actual_words
        }

        speaker_result = await speaker_client.diarize_identify_match(
            audio_path=audio_path,
            transcript_data=transcript_data,
            user_id=user_id
        )

        if not speaker_result or "segments" not in speaker_result:
            logger.warning(f"🎤 Speaker recognition returned no segments")
            return {
                "success": True,
                "conversation_id": conversation_id,
                "version_id": version_id,
                "speaker_recognition_enabled": True,
                "identified_speakers": [],
                "processing_time_seconds": time.time() - start_time
            }

        speaker_segments = speaker_result["segments"]
        logger.info(f"🎤 Speaker recognition returned {len(speaker_segments)} segments")

        # Update the transcript version segments with identified speakers
        updated_segments = []
        for seg in speaker_segments:
            speaker_name = seg.get("identified_as") or seg.get("speaker", "Unknown")
            updated_segments.append(
                Conversation.SpeakerSegment(
                    start=seg.get("start", 0),
                    end=seg.get("end", 0),
                    text=seg.get("text", ""),
                    speaker=speaker_name,
                    confidence=seg.get("confidence")
                )
            )

        # Update the transcript version
        transcript_version.segments = updated_segments

        # Extract unique identified speakers for metadata
        identified_speakers = set()
        for seg in speaker_segments:
            identified_as = seg.get("identified_as", "Unknown")
            if identified_as != "Unknown":
                identified_speakers.add(identified_as)

        # Update metadata
        if not transcript_version.metadata:
            transcript_version.metadata = {}

        transcript_version.metadata["speaker_recognition"] = {
            "enabled": True,
            "identified_speakers": list(identified_speakers),
            "speaker_count": len(identified_speakers),
            "total_segments": len(speaker_segments),
            "processing_time_seconds": time.time() - start_time
        }

        # Update legacy fields if this is the active version
        if conversation.active_transcript_version == version_id:
            conversation.segments = updated_segments

        await conversation.save()

        processing_time = time.time() - start_time
        logger.info(f"✅ Speaker recognition completed for {conversation_id} in {processing_time:.2f}s")

        return {
            "success": True,
            "conversation_id": conversation_id,
            "version_id": version_id,
            "user_id": user_id,
            "speaker_recognition_enabled": True,
            "identified_speakers": list(identified_speakers),
            "segment_count": len(updated_segments),
            "processing_time_seconds": processing_time
        }

    except Exception as speaker_error:
        logger.error(f"❌ Speaker recognition failed: {speaker_error}")
        import traceback
        logger.debug(traceback.format_exc())

        return {
            "success": False,
            "conversation_id": conversation_id,
            "version_id": version_id,
            "error": str(speaker_error),
            "processing_time_seconds": time.time() - start_time
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

        For batch/upload mode, conversations are created upfront and transcribe_full_audio_job is used.

        Args:
            session_id: Stream session ID
            user_id: User ID
            user_email: User email
            client_id: Client ID

        Returns:
            Dict with session_id, conversation_id, open_conversation_job_id, detected_speakers, runtime_seconds
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
    detected_speakers = []  # Track enrolled speakers detected during speech detection

    while True:
        # Check if session has ended (status = "finalizing" or "complete")
        # session_status = await redis_client.hget(session_key, "status")
        # if session_status:
        #     status_str = session_status.decode() if isinstance(session_status, bytes) else session_status
        #     if status_str in ["finalizing", "complete"]:
        #         logger.info(f"🛑 Session {status_str}, stopping speech detection")
        #         break

        # # Check timeout
        # if time.time() - start_time > max_runtime:
        #     logger.warning(f"⏱️ Timeout reached for {session_id}")
        #     break

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

                # Check if enrolled speaker is speaking (also returns speaker recognition results)
                speaker_client = SpeakerRecognitionClient()
                enrolled_speaker_present, speaker_recognition_result = await speaker_client.check_if_enrolled_speaker_present(
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

                # Extract identified speakers from the result
                identified_speakers = []
                if speaker_recognition_result and "segments" in speaker_recognition_result:
                    for seg in speaker_recognition_result["segments"]:
                        identified_as = seg.get("identified_as")
                        # Filter out None and "Unknown" values
                        if identified_as and identified_as != "Unknown" and identified_as not in identified_speakers:
                            identified_speakers.append(identified_as)

                    num_segments = len(speaker_recognition_result["segments"])

                    if identified_speakers:
                        speakers_str = ", ".join(identified_speakers)
                        logger.info(f"✅ Enrolled speaker(s) detected: {speakers_str}")
                        logger.info(f"🎤 Speaker recognition returned {num_segments} segments with {len(identified_speakers)} enrolled speaker(s)")
                        print(f"✅ ENROLLED SPEAKERS DETECTED: {speakers_str} ({num_segments} segments)")
                        detected_speakers = identified_speakers  # Store for return value
                    else:
                        logger.info(f"✅ Enrolled speaker detected! (no identified_as field in segments)")
                        logger.info(f"🎤 Speaker recognition returned {num_segments} segments during enrollment check")
                else:
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
        "detected_speakers": detected_speakers,
        "runtime_seconds": time.time() - start_time
    }


