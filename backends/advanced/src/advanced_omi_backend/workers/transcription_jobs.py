"""
Transcription-related RQ job functions.

This module contains all jobs related to speech-to-text transcription processing.
"""

import os
import logging
from typing import Dict, Any, Optional

from advanced_omi_backend.models.job import JobPriority

from advanced_omi_backend.controllers.queue_controller import (
    transcription_queue,
    redis_conn,
    _ensure_beanie_initialized,
    JOB_RESULT_TTL,
    REDIS_URL,
)

logger = logging.getLogger(__name__)


def listen_for_speech_job(
    audio_uuid: str,
    audio_path: str,
    client_id: str,
    user_id: str,
    user_email: str
) -> Dict[str, Any]:
    """
    RQ job function for initial audio transcription and speech detection.

    This job:
    1. Transcribes the audio file
    2. Detects if speech is present
    3. Creates a conversation ONLY if speech is detected
    4. Enqueues memory processing if conversation created

    Used by: Audio file uploads and WebSocket audio streams

    Args:
        audio_uuid: Audio UUID
        audio_path: Path to audio file
        client_id: Client ID
        user_id: User ID
        user_email: User email

    Returns:
        Dict with processing results
    """
    import asyncio
    import uuid
    import soundfile as sf
    from datetime import UTC, datetime
    from pathlib import Path

    try:
        logger.info(f"🔄 RQ: Listening for speech in audio {audio_uuid}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def process():
                # Initialize Beanie in this worker process
                await _ensure_beanie_initialized()

                # Import here to avoid circular dependencies
                from advanced_omi_backend.services.transcription import get_transcription_provider
                from advanced_omi_backend.models.conversation import Conversation
                from advanced_omi_backend.models.audio_file import AudioFile
                from advanced_omi_backend.config import CHUNK_DIR

                # Read audio file
                audio_file_path = Path(CHUNK_DIR) / audio_path
                if not audio_file_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

                logger.info(f"📖 Reading audio file: {audio_file_path}")
                audio_data, sample_rate = sf.read(str(audio_file_path), dtype='int16')
                audio_bytes = audio_data.tobytes()

                # Get transcription provider
                provider = get_transcription_provider()
                if not provider:
                    raise RuntimeError("No transcription provider available")

                # Transcribe audio
                logger.info(f"🎤 Transcribing audio with {provider.name}")
                transcript_result = await provider.transcribe(audio_bytes, sample_rate, diarize=True)

                # Normalize transcript
                transcript_text = ""
                segments = []
                words = []

                if hasattr(transcript_result, "text"):
                    transcript_text = transcript_result.text
                    segments = getattr(transcript_result, "segments", [])
                    words = getattr(transcript_result, "words", [])
                elif isinstance(transcript_result, dict):
                    transcript_text = transcript_result.get("text", "")
                    segments = transcript_result.get("segments", [])
                    words = transcript_result.get("words", [])
                elif isinstance(transcript_result, str):
                    transcript_text = transcript_result

                version_id = str(uuid.uuid4())

                # Analyze speech using centralized detection from utils
                from advanced_omi_backend.utils.conversation_utils import analyze_speech
                transcript_data = {
                    "text": transcript_text,
                    "words": words
                }
                speech_analysis = analyze_speech(transcript_data)

                logger.info(
                    f"📊 Speech analysis for {audio_uuid}: "
                    f"word_count={speech_analysis['word_count']}, "
                    f"duration={speech_analysis.get('duration', 0):.1f}s, "
                    f"segments={len(segments)}, "
                    f"has_speech={speech_analysis['has_speech']}, "
                    f"reason={speech_analysis['reason']}, "
                    f"preview={transcript_text[:100] if transcript_text else 'EMPTY'}"
                )

                # Update AudioFile with speech detection
                audio_file = await AudioFile.find_one(AudioFile.audio_uuid == audio_uuid)
                if audio_file:
                    audio_file.has_speech = speech_analysis["has_speech"]
                    audio_file.speech_analysis = {
                        "word_count": speech_analysis["word_count"],
                        "duration": speech_analysis.get("duration", 0.0),
                        "segment_count": len(segments),
                        "reason": speech_analysis["reason"],
                        "fallback": speech_analysis.get("fallback", False)
                    }
                    await audio_file.save()

                # If no speech, return early
                if not speech_analysis["has_speech"]:
                    logger.info(f"⏭️ No speech detected in {audio_uuid}, skipping conversation creation")
                    return {"status": "no_speech", "audio_uuid": audio_uuid}

                # Create conversation
                logger.info(f"✅ Speech detected, creating conversation for {audio_uuid}")
                new_conversation_id = str(uuid.uuid4())

                # Get timestamp from AudioFile
                timestamp_value = audio_file.timestamp if audio_file else 0
                if timestamp_value == 0:
                    logger.warning(f"Audio file {audio_uuid} has no timestamp, using current time")
                    session_start_time = datetime.now(UTC)
                else:
                    session_start_time = datetime.fromtimestamp(timestamp_value / 1000, tz=UTC)

                # Generate title and summary from transcript
                title = transcript_text[:50] + "..." if len(transcript_text) > 50 else transcript_text or "Conversation"
                summary = transcript_text[:150] + "..." if len(transcript_text) > 150 else transcript_text or "No transcript available"

                # Create conversation document
                conversation = Conversation(
                    conversation_id=new_conversation_id,
                    audio_uuid=audio_uuid,
                    user_id=user_id,
                    client_id=client_id,
                    title=title,
                    summary=summary,
                    transcript_versions=[
                        Conversation.TranscriptVersion(
                            version_id=version_id,
                            transcript=transcript_text,
                            segments=segments,
                            provider=provider.name,
                            model=getattr(provider, "model_name", provider.name),
                            created_at=datetime.now(UTC),
                            processing_time_seconds=0.0,
                            metadata={}
                        )
                    ],
                    active_transcript_version=version_id,
                    memory_versions=[],
                    active_memory_version=None,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                    session_start=session_start_time,
                    session_end=datetime.now(UTC),
                    duration_seconds=0.0,
                    speech_start_time=0.0,
                    speech_end_time=0.0,
                    speaker_names={},
                    action_items=[]
                )

                # Update legacy fields
                conversation._update_legacy_transcript_fields()
                await conversation.insert()

                # Link conversation to AudioFile
                if audio_file:
                    audio_file.conversation_id = new_conversation_id
                    await audio_file.save()

                logger.info(f"✅ Created conversation {new_conversation_id} for audio {audio_uuid}")

                # Enqueue memory processing
                from .memory_jobs import enqueue_memory_processing
                logger.info(f"📤 Enqueuing memory processing for conversation {new_conversation_id}")
                enqueue_memory_processing(
                    client_id=client_id,
                    user_id=user_id,
                    user_email=user_email,
                    conversation_id=new_conversation_id
                )

                return {
                    "status": "success",
                    "audio_uuid": audio_uuid,
                    "conversation_id": new_conversation_id,
                    "has_speech": True
                }

            result = loop.run_until_complete(process())
            logger.info(f"✅ RQ: Completed speech detection for audio {audio_uuid}")
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Speech detection failed for audio {audio_uuid}: {e}", exc_info=True)
        raise


def process_transcript_job(
    conversation_id: str,
    audio_uuid: str,
    audio_path: str,
    version_id: str,
    user_id: str,
    trigger: str = "reprocess"
) -> Dict[str, Any]:
    """
    RQ job function for transcript processing.

    This function handles both new transcription and reprocessing.
    The 'trigger' parameter indicates the source: 'new', 'reprocess', 'retry', etc.

    This function is executed by RQ workers and can survive server restarts.
    """
    import asyncio
    import time
    from pathlib import Path

    try:
        logger.info(f"🔄 RQ: Starting transcript processing for conversation {conversation_id} (trigger: {trigger})")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def process():
                # Initialize Beanie in this worker process
                await _ensure_beanie_initialized()

                # Import here to avoid circular dependencies
                from advanced_omi_backend.services.transcription import get_transcription_provider
                from advanced_omi_backend.models.conversation import Conversation

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

                # Calculate processing time
                processing_time = time.time() - start_time

                # Get the conversation using Beanie
                conversation = await Conversation.find_one(Conversation.conversation_id == conversation_id)
                if not conversation:
                    logger.error(f"Conversation {conversation_id} not found")
                    return {"success": False, "error": "Conversation not found"}

                # Convert segments to SpeakerSegment objects
                speaker_segments = [
                    Conversation.SpeakerSegment(
                        start=seg.get("start", 0),
                        end=seg.get("end", 0),
                        text=seg.get("text", ""),
                        speaker=seg.get("speaker", "unknown"),
                        confidence=seg.get("confidence")
                    )
                    for seg in segments
                ]

                # Add new transcript version
                provider_normalized = provider_name.lower() if provider_name else "unknown"

                conversation.add_transcript_version(
                    version_id=version_id,
                    transcript=transcript_text,
                    segments=speaker_segments,
                    provider=Conversation.TranscriptProvider(provider_normalized),
                    model=getattr(provider, 'model', 'unknown'),
                    processing_time_seconds=processing_time,
                    metadata={
                        "trigger": trigger,
                        "audio_file_size": len(audio_data),
                        "segment_count": len(segments),
                        "word_count": len(words)
                    },
                    set_as_active=True
                )

                # Generate title and summary from transcript
                if transcript_text and len(transcript_text.strip()) > 0:
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

            result = loop.run_until_complete(process())
            logger.info(f"✅ RQ: Completed transcript processing for conversation {conversation_id}")
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Transcript processing failed for conversation {conversation_id}: {e}")
        raise


def stream_speech_detection_job(
    session_id: str,
    user_id: str,
    user_email: str,
    client_id: str,
    redis_url: str = None
) -> Dict[str, Any]:
    """
    RQ job that monitors transcription stream for speech (STREAMING MODE ONLY).

    Job lifecycle:
    1. Monitors transcription stream for speech
    2. When speech detected:
       - Checks if conversation already open (prevents duplicates)
       - If no open conversation: creates conversation + starts open_conversation_job
       - Exits (job completes)
    3. New stream_speech_detection_job can be started when conversation closes

    This architecture alternates between "listening for speech" and "actively recording conversation".

    This is part of the V2 architecture using RQ jobs as orchestrators.

    For batch/upload mode, use listen_for_speech_job instead.

    Args:
        session_id: Stream session ID
        user_id: User ID
        user_email: User email
        client_id: Client ID
        redis_url: Redis connection URL

    Returns:
        Dict with session_id, conversation_id, open_conversation_job_id, runtime_seconds
    """
    import asyncio
    import time

    try:
        logger.info(f"🔍 RQ: Starting stream speech detection for session {session_id}")

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
                from .conversation_jobs import create_conversation_from_streaming_results, open_conversation_job
                import redis.asyncio as redis_async

                # Connect to Redis
                redis_client = redis_async.from_url(redis_url)
                aggregator = TranscriptionResultsAggregator(redis_client)

                # Job control
                session_key = f"audio:session:{session_id}"
                max_runtime = 3600  # 1 hour max
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

                    # Get latest transcription results
                    results = await aggregator.get_session_results(session_id)

                    if not results:
                        await asyncio.sleep(2)  # Check every 2 seconds
                        continue

                    # Combine transcript and word-level data
                    full_transcript = " ".join([r.get("text", "") for r in results if r.get("text")])
                    all_words = []
                    for r in results:
                        if "words" in r and r["words"]:
                            all_words.extend(r["words"])

                    # Analyze for speech using centralized detection from utils
                    from advanced_omi_backend.utils.conversation_utils import analyze_speech
                    transcript_data = {
                        "text": full_transcript,
                        "words": all_words
                    }
                    speech_analysis = analyze_speech(transcript_data)
                    has_speech = speech_analysis["has_speech"]

                    print(f"🔍 SPEECH ANALYSIS: session={session_id}, has_speech={has_speech}, conv_id={conversation_id}, words={speech_analysis.get('word_count', 0)}")
                    logger.info(
                        f"🔍 Speech analysis for {session_id}: has_speech={has_speech}, "
                        f"conversation_id={conversation_id}, word_count={speech_analysis.get('word_count', 0)}"
                    )

                    if has_speech and not conversation_id:
                        print(f"💬 SPEECH DETECTED! Creating conversation for {session_id}")
                        logger.info(f"💬 Speech detected in {session_id}! Creating conversation...")

                        # Check if there's already an open_conversation_job running for this session
                        open_job_key = f"open_conversation:session:{session_id}"
                        stored_job_data = await redis_client.get(open_job_key)
                        logger.info(f"🔑 Checking Redis key {open_job_key}: {stored_job_data}")

                        if stored_job_data:
                            # Parse stored job data (format: "job_id:conversation_id")
                            job_data_str = stored_job_data.decode() if isinstance(stored_job_data, bytes) else stored_job_data
                            try:
                                stored_job_id, stored_conversation_id = job_data_str.split(":", 1)

                                # Check if the job is actually running
                                from rq.job import Job
                                import redis as redis_sync
                                sync_redis_conn = redis_sync.from_url(redis_url)
                                job = Job.fetch(stored_job_id, connection=sync_redis_conn)

                                if job.is_started and not job.is_finished and not job.is_failed:
                                    # Job is still running, reuse existing conversation
                                    conversation_id = stored_conversation_id
                                    open_conversation_job_id = stored_job_id
                                    logger.info(f"✅ Open conversation job {stored_job_id} already running for conversation {conversation_id}")
                                else:
                                    # Job is not running, clean up and create new conversation
                                    logger.warning(f"⚠️ Open conversation job {stored_job_id} is not running (status: {job.get_status()})")
                                    await redis_client.delete(open_job_key)
                                    stored_job_data = None  # Fall through to create new conversation
                            except Exception as job_check_error:
                                logger.warning(f"⚠️ Error checking job status: {job_check_error}")
                                await redis_client.delete(open_job_key)
                                stored_job_data = None  # Fall through to create new conversation

                        if not stored_job_data:
                            # No open conversation job running - create minimal conversation
                            # The open_conversation_job will populate it with ALL results from the beginning
                            logger.info(f"📝 Creating minimal conversation, open_conversation_job will populate transcript...")
                            conversation_id = await create_conversation_from_streaming_results(
                                session_id=session_id,
                                user_id=user_id,
                                user_email=user_email,
                                client_id=client_id,
                                transcript="",  # Empty, will be populated by open_conversation_job
                                results=[]  # Empty, open_conversation_job will fetch all from Redis
                            )
                            logger.info(f"✅ Conversation created: {conversation_id}")

                            # Start Open Conversation Job to monitor updates
                            logger.info(f"🚀 Enqueueing open_conversation_job for {conversation_id}")
                            open_job = transcription_queue.enqueue(
                                open_conversation_job,
                                conversation_id,
                                session_id,
                                user_id,
                                redis_url,
                                job_timeout=3600,
                                result_ttl=600,
                                job_id=f"open-conv_{conversation_id[:12]}",
                                description=f"Open conversation {conversation_id[:12]} for updates"
                            )
                            open_conversation_job_id = open_job.id
                            logger.info(f"✅ Open conversation job enqueued: {open_job.id}")

                            # Store job info in Redis for future speech detection checks
                            await redis_client.set(
                                open_job_key,
                                f"{open_job.id}:{conversation_id}",
                                ex=3600  # Expire after 1 hour
                            )
                            logger.info(f"💾 Stored job tracking in Redis: {open_job_key}")

                            logger.info(f"✅ Created conversation {conversation_id} with job {open_job.id}")

                            # Exit this job now that conversation is open
                            # A new stream_speech_detection_job will be started when this conversation closes
                            logger.info(f"🏁 Exiting speech detection job - conversation {conversation_id} is now open")
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
                    "conversation_id": conversation_id,
                    "open_conversation_job_id": open_conversation_job_id,
                    "runtime_seconds": time.time() - start_time
                }

            result = loop.run_until_complete(process())
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Stream speech detection failed for session {session_id}: {e}", exc_info=True)
        raise


def finalize_streaming_transcription_job(
    session_id: str,
    user_id: str,
    user_email: str,
    client_id: str,
    conversation_id: Optional[str],
    audio_chunks: list,
    redis_url: str,
    open_conversation_job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    RQ job function for finalizing streaming transcription.

    This job:
    1. Coordinates with open_conversation_job if it exists (V2 architecture)
    2. Waits for final chunks to be processed
    3. Aggregates results from Redis Streams
    4. Updates existing conversation OR creates new one if speech detected
    5. Writes final audio file

    Args:
        session_id: Stream session ID
        user_id: User ID
        user_email: User email
        client_id: Client ID
        conversation_id: Existing conversation ID (if created during recording) or None
        audio_chunks: List of raw audio chunks as bytes
        redis_url: Redis connection URL
        open_conversation_job_id: Open conversation job ID (V2 architecture) or None

    Returns:
        Dict with processing results
    """
    import asyncio
    import time
    import re
    import uuid
    from datetime import datetime, timezone

    try:
        logger.info(f"🔄 RQ: Finalizing streaming transcription for session {session_id}")

        # Validate and fix redis_url
        if not redis_url or not redis_url.startswith(("redis://", "rediss://", "unix://")):
            logger.warning(f"⚠️ Invalid redis_url provided: {redis_url}, using REDIS_URL constant instead")
            redis_url = REDIS_URL
            logger.info(f"✅ Using REDIS_URL: {redis_url}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def process():
                # Initialize Beanie
                await _ensure_beanie_initialized()

                # Import here to avoid circular dependencies
                from advanced_omi_backend.models.conversation import Conversation
                from advanced_omi_backend.audio_utils import write_audio_file
                from advanced_omi_backend.services.audio_stream import TranscriptionResultsAggregator
                from .conversation_jobs import create_conversation_from_streaming_results
                import redis.asyncio as redis_async

                # Connect to Redis
                redis_client = redis_async.from_url(redis_url)
                aggregator = TranscriptionResultsAggregator(redis_client)

                # V2 Architecture: Coordinate with open_conversation_job if it exists
                if conversation_id and open_conversation_job_id:
                    finalize_timestamp = time.time()

                    # Set stop signal with timestamp
                    await redis_client.set(
                        f"job:control:{conversation_id}:stop_open_conversation",
                        str(finalize_timestamp)
                    )
                    logger.info(f"🛑 Sent stop signal to open conversation job {open_conversation_job_id}")

                    # Wait for open conversation job to finish
                    logger.info(f"⏳ Waiting for open conversation job {open_conversation_job_id} to finish...")
                    max_wait_job = 60  # 1 minute
                    start_wait_job = time.time()

                    try:
                        # Import synchronously (Job.fetch is synchronous)
                        from rq.job import Job
                        import redis as redis_sync

                        # Need synchronous Redis connection for Job.fetch
                        sync_redis_conn = redis_sync.from_url(redis_url)
                        open_job = Job.fetch(open_conversation_job_id, connection=sync_redis_conn)

                        while not open_job.is_finished:
                            if time.time() - start_wait_job > max_wait_job:
                                logger.warning(f"⏱️ Timeout waiting for job {open_conversation_job_id}")
                                break
                            await asyncio.sleep(1)
                            # Refresh job status
                            open_job.refresh()

                        logger.info(f"✅ Open conversation job finished")
                    except Exception as job_wait_error:
                        logger.warning(f"⚠️ Error waiting for open conversation job: {job_wait_error}")

                # Wait for consumer to finish processing last chunks
                # Give time for workers to process final chunks (20-chunk buffer = 5 seconds)
                logger.info(f"⏳ Waiting for consumer to finish transcribing last chunks...")
                max_wait_seconds = 30  # Reduced from 60 with smaller buffer size
                wait_interval = 3  # Check every 3 seconds
                elapsed = 0
                stable_count = 0  # Count how many times results haven't changed
                required_stable_checks = 2  # Need 2 consecutive stable checks

                while elapsed < max_wait_seconds:
                    # Check if there are transcription results
                    results = await aggregator.get_session_results(session_id)
                    result_count = len(results)

                    if result_count > 0:
                        logger.info(f"📊 Got {result_count} transcription results so far...")

                    # Wait and check again
                    await asyncio.sleep(wait_interval)
                    elapsed += wait_interval

                    # Get results again to see if more arrived
                    new_results = await aggregator.get_session_results(session_id)
                    new_count = len(new_results)

                    if new_count == result_count:
                        # No new results
                        stable_count += 1
                        logger.info(f"⏸️  No new results (stable check {stable_count}/{required_stable_checks})")

                        if stable_count >= required_stable_checks:
                            # Results have been stable for multiple checks, we're done
                            logger.info(f"✅ Results stable for {stable_count} checks ({elapsed}s elapsed), processing complete")
                            results = new_results
                            break
                    else:
                        # New results arrived, reset stable counter
                        stable_count = 0
                        logger.info(f"📈 New results arrived: {result_count} → {new_count}")

                    results = new_results

                logger.info(f"📊 Final result count: {len(results)} transcription results")

                # Combine transcript text
                full_transcript = " ".join([r.get("text", "") for r in results if r.get("text")])
                logger.info(f"📝 Combined transcript: {len(full_transcript)} characters")

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
                logger.info(f"📊 Extracted {len(all_segments)} segments")

                # Write complete audio file
                complete_audio = b''.join(audio_chunks)
                timestamp = int(time.time() * 1000)

                wav_filename, file_path, duration = await write_audio_file(
                    raw_audio_data=complete_audio,
                    audio_uuid=session_id,
                    client_id=client_id,
                    user_id=user_id,
                    user_email=user_email,
                    timestamp=timestamp,
                    validate=False
                )
                logger.info(f"📁 Wrote audio file: {wav_filename} ({duration:.1f}s)")

                # Mark session as complete in Redis
                session_key = f"audio:session:{session_id}"
                await redis_client.hset(session_key, mapping={
                    "status": "complete",
                    "completed_at": str(time.time())
                })
                logger.info(f"✅ Marked session {session_id} as complete")

                if conversation_id:
                    # Update existing conversation
                    conversation = await Conversation.find_one(Conversation.conversation_id == conversation_id)
                    if conversation:
                        conversation.transcript = full_transcript
                        conversation.segments = all_segments
                        await conversation.save()
                        logger.info(f"✅ Updated conversation {conversation_id} with complete transcript and {len(all_segments)} segments")

                        # Enqueue batch transcription of complete audio file for final high-quality transcript
                        import uuid
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

                        # Create conversation
                        new_conversation_id = await create_conversation_from_streaming_results(
                            session_id=session_id,
                            user_id=user_id,
                            user_email=user_email,
                            client_id=client_id,
                            transcript=full_transcript,
                            results=results
                        )

                        logger.info(f"✅ Created conversation {new_conversation_id}")

                        # Enqueue batch transcription of complete audio file for final high-quality transcript
                        import uuid
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

            result = loop.run_until_complete(process())
            logger.info(f"✅ RQ: Completed streaming transcription finalization for session {session_id}")
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Streaming transcription finalization failed for session {session_id}: {e}", exc_info=True)
        raise


# Enqueue wrapper functions

def enqueue_initial_transcription(
    audio_uuid: str,
    audio_path: str,
    client_id: str,
    user_id: str,
    user_email: str,
    priority: JobPriority = JobPriority.NORMAL
):
    """
    Enqueue job to listen for speech and create conversation if detected.

    This job transcribes audio, detects speech, and creates a conversation
    ONLY if speech is detected. Used for initial audio uploads and streams.

    Args:
        audio_uuid: Audio UUID
        audio_path: Path to saved audio file
        client_id: Client ID
        user_id: User ID
        user_email: User email
        priority: Job priority

    Returns:
        RQ Job object for tracking
    """
    timeout_mapping = {
        JobPriority.URGENT: 600,  # 10 minutes
        JobPriority.HIGH: 480,    # 8 minutes
        JobPriority.NORMAL: 300,  # 5 minutes
        JobPriority.LOW: 180      # 3 minutes
    }

    job = transcription_queue.enqueue(
        listen_for_speech_job,
        audio_uuid,
        audio_path,
        client_id,
        user_id,
        user_email,
        job_timeout=timeout_mapping.get(priority, 300),
        result_ttl=JOB_RESULT_TTL,
        job_id=f"listen-for-speech_{audio_uuid[:12]}",
        description=f"Listen for speech in {audio_uuid[:12]}"
    )

    logger.info(f"📥 RQ: Enqueued listen-for-speech job {job.id} for audio {audio_uuid}")
    return job


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
    conversation_id: Optional[str],
    audio_chunks: list,
    priority: JobPriority = JobPriority.HIGH,
    open_conversation_job_id: Optional[str] = None
):
    """
    Enqueue a streaming transcription finalization job.

    This job handles the final processing when a streaming session ends.

    Args:
        session_id: Stream session ID
        user_id: User ID
        user_email: User email
        client_id: Client ID
        conversation_id: Existing conversation ID or None
        audio_chunks: List of raw audio chunk bytes
        priority: Job priority (default HIGH for responsive UI)
        open_conversation_job_id: Open conversation job ID (V2 architecture) or None

    Returns:
        RQ Job object for tracking
    """
    timeout_mapping = {
        JobPriority.URGENT: 180,  # 3 minutes
        JobPriority.HIGH: 120,    # 2 minutes
        JobPriority.NORMAL: 90,   # 1.5 minutes
        JobPriority.LOW: 60       # 1 minute
    }

    # Use Redis URL from queue controller
    job_redis_url = REDIS_URL

    job = transcription_queue.enqueue(
        finalize_streaming_transcription_job,
        session_id,
        user_id,
        user_email,
        client_id,
        conversation_id,
        audio_chunks,
        job_redis_url,
        open_conversation_job_id,
        job_timeout=timeout_mapping.get(priority, 120),
        result_ttl=JOB_RESULT_TTL,
        job_id=f"finalize-stream_{session_id[:12]}",
        description=f"Finalize streaming session {session_id[:12]}"
    )

    logger.info(f"📥 RQ: Enqueued streaming finalization job {job.id} for session {session_id}")
    return job
