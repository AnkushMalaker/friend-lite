"""
System controller for handling system-related business logic.
"""

import asyncio
import io
import json
import logging
import os
import shutil
import time
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, File, Query, UploadFile
from fastapi.responses import JSONResponse
from wyoming.audio import AudioChunk

from advanced_omi_backend.client_manager import generate_client_id
from advanced_omi_backend.config import load_diarization_settings_from_file, save_diarization_settings_to_file
from advanced_omi_backend.database import chunks_col
from advanced_omi_backend.job_tracker import FileStatus, JobStatus, get_job_tracker
from advanced_omi_backend.processors import AudioProcessingItem, get_processor_manager
from advanced_omi_backend.task_manager import get_task_manager
from advanced_omi_backend.users import User

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")


async def get_current_metrics():
    """Get current system metrics."""
    try:
        # Get memory provider configuration
        memory_provider = os.getenv("MEMORY_PROVIDER", "friend_lite").lower()
        
        # Get basic system metrics
        metrics = {
            "timestamp": int(time.time()),
            "memory_provider": memory_provider,
            "memory_provider_supports_threshold": memory_provider == "friend_lite",
        }

        return metrics

    except Exception as e:
        audio_logger.error(f"Error fetching metrics: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to fetch metrics: {str(e)}"}
        )


async def get_auth_config():
    """Get authentication configuration for frontend."""
    return {
        "auth_method": "email",
        "registration_enabled": False,  # Only admin can create users
        "features": {
            "email_login": True,
            "user_id_login": False,  # Deprecated
            "registration": False,
        },
    }


async def get_all_processing_tasks():
    """Get all active processing tasks."""
    try:
        processor_manager = get_processor_manager()
        return processor_manager.get_all_processing_status()
    except Exception as e:
        logger.error(f"Error getting processing tasks: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get processing tasks: {str(e)}"}
        )


async def get_processing_task_status(client_id: str):
    """Get processing task status for a specific client."""
    try:
        processor_manager = get_processor_manager()
        processing_status = processor_manager.get_processing_status(client_id)

        # Check if transcription is marked as started but not completed, and verify with database
        stages = processing_status.get("stages", {})
        transcription_stage = stages.get("transcription", {})

        """This is a hack to update it the DB INCASE a process failed
        if transcription_stage.get("status") == "started" and not transcription_stage.get("completed", False):
            # Check if transcription is actually complete by checking the database
            try:
                chunk = await chunks_col.find_one({"client_id": client_id})
                if chunk and chunk.get("transcript") and len(chunk.get("transcript", [])) > 0:
                    # Transcription is complete! Update the processor state
                    processor_manager.track_processing_stage(
                        client_id,
                        "transcription",
                        "completed",
                        {"audio_uuid": chunk.get("audio_uuid"), "segments": len(chunk.get("transcript", []))}
                    )
                    logger.info(f"Detected transcription completion for client {client_id} ({len(chunk.get('transcript', []))} segments)")
                    # Get updated status
                    processing_status = processor_manager.get_processing_status(client_id)
            except Exception as e:
                logger.debug(f"Error checking transcription completion: {e}")
        """
        return processing_status
    except Exception as e:
        logger.error(f"Error getting processing task status for {client_id}: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get processing task status: {str(e)}"}
        )


async def get_processor_status():
    """Get processor queue status and health."""
    try:
        processor_manager = get_processor_manager()

        # Get queue sizes
        status = {
            "queues": {
                "audio_queue": processor_manager.audio_queue.qsize(),
                "transcription_queue": processor_manager.transcription_queue.qsize(),
                "memory_queue": processor_manager.memory_queue.qsize(),
                "cropping_queue": processor_manager.cropping_queue.qsize(),
            },
            "processors": {
                "audio_processor": "running",
                "transcription_processor": "running",
                "memory_processor": "running",
                "cropping_processor": "running",
            },
            "active_clients": len(processor_manager.active_file_sinks),
            "active_audio_uuids": len(processor_manager.active_audio_uuids),
            "processing_tasks": len(processor_manager.processing_tasks),
            "timestamp": int(time.time()),
        }

        # Get task manager status if available
        try:
            task_manager = get_task_manager()
            if task_manager:
                task_status = task_manager.get_health_status()
                status["task_manager"] = task_status
        except Exception as e:
            status["task_manager"] = {"error": str(e)}

        return status

    except Exception as e:
        logger.error(f"Error getting processor status: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get processor status: {str(e)}"}
        )


async def process_audio_files(
    user: User, files: list[UploadFile], device_name: str, auto_generate_client: bool
):
    """Process uploaded audio files through the transcription pipeline."""
    # Need to import here because we import the routes into main, causing circular imports
    from advanced_omi_backend.main import cleanup_client_state, create_client_state

    # Process files through complete transcription pipeline like WebSocket clients
    try:
        if not files:
            return JSONResponse(status_code=400, content={"error": "No files provided"})

        processed_files = []
        processed_conversations = []

        for file_index, file in enumerate(files):
            client_id = None
            client_state = None

            try:
                # Validate file type (only WAV for now)
                if not file.filename or not file.filename.lower().endswith(".wav"):
                    processed_files.append(
                        {
                            "filename": file.filename or "unknown",
                            "status": "error",
                            "error": "Only WAV files are currently supported",
                        }
                    )
                    continue

                # Generate unique client ID for each file to create separate conversations
                file_device_name = f"{device_name}-{file_index + 1:03d}"
                client_id = generate_client_id(user, file_device_name)

                # Create separate client state for this file
                client_state = await create_client_state(client_id, user, file_device_name)

                audio_logger.info(
                    f"📁 Processing file {file_index + 1}/{len(files)}: {file.filename} with client_id: {client_id}"
                )

                processor_manager = get_processor_manager()

                # Read file content
                content = await file.read()

                # Process WAV file
                with wave.open(io.BytesIO(content), "rb") as wav_file:
                    # Get audio parameters
                    sample_rate = wav_file.getframerate()
                    sample_width = wav_file.getsampwidth()
                    channels = wav_file.getnchannels()

                    # Read all audio data
                    audio_data = wav_file.readframes(wav_file.getnframes())

                    # Convert to mono if stereo
                    if channels == 2:
                        # Convert stereo to mono by averaging channels
                        if sample_width == 2:
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        else:
                            audio_array = np.frombuffer(audio_data, dtype=np.int32)

                        # Reshape to separate channels and average
                        audio_array = audio_array.reshape(-1, 2)
                        audio_data = (
                            np.mean(audio_array, axis=1).astype(audio_array.dtype).tobytes()
                        )
                        channels = 1

                    # Ensure sample rate is 16kHz (resample if needed)
                    if sample_rate != 16000:
                        audio_logger.warning(
                            f"File {file.filename} has sample rate {sample_rate}Hz, expected 16kHz. Processing anyway."
                        )

                    # Process audio in larger chunks for faster file processing
                    # Use larger chunks (32KB) for optimal performance
                    chunk_size = 32 * 1024  # 32KB chunks
                    base_timestamp = int(time.time())

                    for i in range(0, len(audio_data), chunk_size):
                        chunk_data = audio_data[i : i + chunk_size]

                        # Calculate relative timestamp for this chunk
                        chunk_offset_bytes = i
                        chunk_offset_seconds = chunk_offset_bytes / (
                            sample_rate * sample_width * channels
                        )
                        chunk_timestamp = base_timestamp + int(chunk_offset_seconds)

                        # Create AudioChunk
                        chunk = AudioChunk(
                            audio=chunk_data,
                            rate=sample_rate,
                            width=sample_width,
                            channels=channels,
                            timestamp=chunk_timestamp,
                        )

                        # Add to application-level processing queue

                        audio_item = AudioProcessingItem(
                            client_id=client_id,
                            user_id=user.user_id,
                            user_email=user.email,
                            audio_chunk=chunk,
                            timestamp=chunk.timestamp,
                        )
                        await processor_manager.queue_audio(audio_item)

                        # Yield control occasionally to prevent blocking the event loop
                        if i % (chunk_size * 10) == 0:  # Every 10 chunks (~320KB)
                            await asyncio.sleep(0)

                    processed_files.append(
                        {
                            "filename": file.filename,
                            "sample_rate": sample_rate,
                            "channels": channels,
                            "duration_seconds": len(audio_data)
                            / (sample_rate * sample_width * channels),
                            "size_bytes": len(audio_data),
                            "client_id": client_id,
                            "status": "processed",
                        }
                    )

                    audio_logger.info(
                        f"✅ Processed audio file: {file.filename} ({len(audio_data)} bytes)"
                    )

                # Wait briefly for transcription manager to be created by background processor
                audio_logger.info(
                    f"⏳ Waiting for transcription manager to be created for client {client_id}"
                )
                await asyncio.sleep(2.0)  # Give transcription processor time to create manager

                # Close client audio to trigger transcription completion (flush_final_transcript)
                audio_logger.info(
                    f"📞 About to call close_client_audio for upload client {client_id}"
                )
                processor_manager = get_processor_manager()
                audio_logger.info(f"📞 Got processor manager, calling close_client_audio now...")
                await processor_manager.close_client_audio(client_id)
                audio_logger.info(
                    f"🔚 Successfully called close_client_audio for upload client {client_id}"
                )

                # Wait for this file's transcription processing to complete
                audio_logger.info(f"📁 Waiting for transcription to process file: {file.filename}")

                # Wait for chunks to be processed by the audio saver
                await asyncio.sleep(1.0)

                # Wait for file processing to complete using task tracking
                # Increase timeout based on file duration (3x duration + 60s buffer)
                audio_duration = len(audio_data) / (sample_rate * sample_width * channels)
                max_wait_time = max(
                    120, int(audio_duration * 3) + 60
                )  # At least 2 minutes, or 3x duration + 60s
                wait_interval = 2.0  # Reduced from 0.5s to 2s to reduce polling spam
                elapsed_time = 0

                audio_logger.info(
                    f"📁 Audio duration: {audio_duration:.1f}s, max wait time: {max_wait_time}s"
                )

                # Use concrete task tracking instead of database polling
                while elapsed_time < max_wait_time:
                    try:
                        # Check processing status using task tracking
                        processing_status = processor_manager.get_processing_status(client_id)

                        # Check if transcription stage is complete
                        stages = processing_status.get("stages", {})
                        transcription_stage = stages.get("transcription", {})

                        # If transcription is marked as started but not completed, check database
                        if transcription_stage.get(
                            "status"
                        ) == "started" and not transcription_stage.get("completed", False):
                            # Check if transcription is actually complete by checking the database
                            try:
                                chunk = await chunks_col.find_one({"client_id": client_id})
                                if (
                                    chunk
                                    and chunk.get("transcript")
                                    and len(chunk.get("transcript", [])) > 0
                                ):
                                    # Transcription is complete! Update the processor state
                                    processor_manager.track_processing_stage(
                                        client_id,
                                        "transcription",
                                        "completed",
                                        {
                                            "audio_uuid": chunk.get("audio_uuid"),
                                            "segments": len(chunk.get("transcript", [])),
                                        },
                                    )
                                    audio_logger.info(
                                        f"📁 Transcription completed for file: {file.filename} ({len(chunk.get('transcript', []))} segments)"
                                    )
                                    break
                            except Exception as e:
                                audio_logger.debug(f"Error checking transcription completion: {e}")

                        if transcription_stage.get("completed", False):
                            audio_logger.info(
                                f"📁 Transcription completed for file: {file.filename}"
                            )
                            break

                        # Check for errors
                        if transcription_stage.get("error"):
                            audio_logger.warning(
                                f"📁 Transcription error for file: {file.filename}: {transcription_stage.get('error')}"
                            )
                            break

                    except Exception as e:
                        audio_logger.debug(f"Error checking processing status: {e}")

                    await asyncio.sleep(wait_interval)
                    elapsed_time += wait_interval

                if elapsed_time >= max_wait_time:
                    audio_logger.warning(f"📁 Transcription timed out for file: {file.filename}")

                # Signal end of conversation - trigger memory processing
                await client_state.close_current_conversation()

                # Give cleanup time to complete
                await asyncio.sleep(0.5)

                # Track conversation created
                conversation_info = {
                    "client_id": client_id,
                    "filename": file.filename,
                    "status": "completed" if elapsed_time < max_wait_time else "timed_out",
                }
                processed_conversations.append(conversation_info)

            except Exception as e:
                audio_logger.error(f"Error processing file {file.filename}: {e}")
                processed_files.append(
                    {"filename": file.filename or "unknown", "status": "error", "error": str(e)}
                )
            finally:
                # Always clean up client state to prevent accumulation
                if client_id and client_state:
                    try:
                        await cleanup_client_state(client_id)
                        audio_logger.info(f"🧹 Cleaned up client state for {client_id}")
                    except Exception as cleanup_error:
                        audio_logger.error(
                            f"❌ Error cleaning up client state for {client_id}: {cleanup_error}"
                        )

        return {
            "message": f"Processed {len(files)} files",
            "files": processed_files,
            "conversations": processed_conversations,
            "successful": len([f for f in processed_files if f.get("status") != "error"]),
            "failed": len([f for f in processed_files if f.get("status") == "error"]),
        }

    except Exception as e:
        audio_logger.error(f"Error in process_audio_files: {e}")
        return JSONResponse(status_code=500, content={"error": f"File processing failed: {str(e)}"})


def get_audio_duration(file_content: bytes) -> float:
    """Get duration of WAV file in seconds using wave library."""
    try:
        with wave.open(io.BytesIO(file_content), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / sample_rate
            return duration
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return 0.0


async def process_audio_files_async(
    background_tasks: BackgroundTasks, user: User, files: list[UploadFile], device_name: str
):
    """Start async processing of uploaded audio files. Returns job ID immediately."""
    try:
        if not files:
            return JSONResponse(status_code=400, content={"error": "No files provided"})

        # Read all file contents immediately to avoid file handle issues
        file_data = []
        for file in files:
            try:
                content = await file.read()
                file_data.append((file.filename, content))
                audio_logger.info(f"📥 Read file: {file.filename} ({len(content)} bytes)")
            except Exception as e:
                audio_logger.error(f"❌ Failed to read file {file.filename}: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to read file {file.filename}: {str(e)}"},
                )

        # Create job
        job_tracker = get_job_tracker()
        filenames = [filename for filename, _ in file_data]
        job_id = await job_tracker.create_job(user.user_id, device_name, filenames)

        # Start background processing with file contents
        background_tasks.add_task(process_files_with_content, job_id, file_data, user, device_name)

        audio_logger.info(f"🚀 Started async processing job {job_id} with {len(files)} files")

        return {
            "job_id": job_id,
            "message": f"Started processing {len(files)} files",
            "status_url": f"/api/process-audio-files/jobs/{job_id}",
            "total_files": len(files),
        }

    except Exception as e:
        audio_logger.error(f"Error starting async file processing: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to start processing: {str(e)}"}
        )


async def get_processing_job_status(job_id: str):
    """Get status of an async file processing job."""
    try:
        job_tracker = get_job_tracker()
        job = await job_tracker.get_job(job_id)

        if not job:
            return JSONResponse(status_code=404, content={"error": "Job not found"})

        return job.to_dict()

    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get job status: {str(e)}"}
        )


async def list_processing_jobs():
    """List all active processing jobs."""
    try:
        job_tracker = get_job_tracker()
        active_jobs = await job_tracker.get_active_jobs()

        return {"active_jobs": len(active_jobs), "jobs": [job.to_dict() for job in active_jobs]}

    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to list jobs: {str(e)}"})


async def process_files_with_content(
    job_id: str, file_data: list[tuple[str, bytes]], user: User, device_name: str
):
    """Background task to process uploaded files using pre-read content."""
    # Import here to avoid circular imports
    from advanced_omi_backend.main import cleanup_client_state, create_client_state

    audio_logger.info(
        f"🚀 process_files_with_content called for job {job_id} with {len(file_data)} files"
    )
    job_tracker = get_job_tracker()

    try:
        # Update job status to processing
        await job_tracker.update_job_status(job_id, JobStatus.PROCESSING)

        for file_index, (filename, content) in enumerate(file_data):
            client_id = None
            client_state = None

            try:
                audio_logger.info(
                    f"🔧 [Job {job_id}] Processing file {file_index + 1}/{len(file_data)}: {filename}, content type: {type(content)}, size: {len(content)}"
                )
                # Set current file
                await job_tracker.set_current_file(job_id, filename)
                await job_tracker.update_file_status(job_id, filename, FileStatus.PROCESSING)

                audio_logger.info(
                    f"🚀 [Job {job_id}] Processing file {file_index + 1}/{len(file_data)}: {filename}"
                )

                # Check duration and skip if too long
                audio_logger.info(
                    f"🔍 [Job {job_id}] About to check duration for {filename}, content size: {len(content)} bytes"
                )
                try:
                    duration = get_audio_duration(content)
                    audio_logger.info(
                        f"🔍 [Job {job_id}] Duration check successful: {duration:.2f}s for {filename}"
                    )
                except Exception as duration_error:
                    audio_logger.error(
                        f"❌ [Job {job_id}] Duration check failed for {filename}: {duration_error}"
                    )
                    raise
                if duration > 1200:  # 20 minutes
                    error_msg = f"File duration ({duration/60:.1f} minutes) exceeds 20-minute limit"
                    audio_logger.error(f"🔴 {error_msg}")
                    await job_tracker.update_file_status(
                        job_id, filename, FileStatus.SKIPPED, error_message=error_msg
                    )
                    continue

                # Validate file type
                if not filename or not filename.lower().endswith(".wav"):
                    error_msg = "Only WAV files are currently supported"
                    await job_tracker.update_file_status(
                        job_id, filename, FileStatus.FAILED, error_message=error_msg
                    )
                    continue

                # Generate unique client ID for each file
                file_device_name = f"{device_name}-{file_index + 1:03d}"
                client_id = generate_client_id(user, file_device_name)

                # Update job tracker with client ID
                await job_tracker.update_file_status(
                    job_id, filename, FileStatus.PROCESSING, client_id=client_id
                )

                # Create client state
                client_state = await create_client_state(client_id, user, file_device_name)

                # Process WAV file
                with wave.open(io.BytesIO(content), "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    sample_width = wav_file.getsampwidth()
                    channels = wav_file.getnchannels()
                    audio_data = wav_file.readframes(wav_file.getnframes())

                    # Convert to mono if stereo
                    if channels == 2:
                        if sample_width == 2:
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        else:
                            audio_array = np.frombuffer(audio_data, dtype=np.int32)
                        audio_array = audio_array.reshape(-1, 2)
                        audio_data = (
                            np.mean(audio_array, axis=1).astype(audio_array.dtype).tobytes()
                        )
                        channels = 1

                    # Process audio in chunks
                    processor_manager = get_processor_manager()
                    chunk_size = 32 * 1024
                    base_timestamp = int(time.time())

                    for i in range(0, len(audio_data), chunk_size):
                        chunk_data = audio_data[i : i + chunk_size]
                        chunk_offset_bytes = i
                        chunk_offset_seconds = chunk_offset_bytes / (
                            sample_rate * sample_width * channels
                        )
                        chunk_timestamp = base_timestamp + int(chunk_offset_seconds)

                        chunk = AudioChunk(
                            audio=chunk_data,
                            rate=sample_rate,
                            width=sample_width,
                            channels=channels,
                            timestamp=chunk_timestamp,
                        )

                        audio_item = AudioProcessingItem(
                            client_id=client_id,
                            user_id=user.user_id,
                            user_email=user.email,
                            audio_chunk=chunk,
                            timestamp=chunk.timestamp,
                        )
                        await processor_manager.queue_audio(audio_item)

                        if i % (chunk_size * 10) == 0:  # Yield control occasionally
                            await asyncio.sleep(0)

                # Wait briefly for transcription manager to be created
                await asyncio.sleep(2.0)

                # Close client audio to trigger transcription completion
                await processor_manager.close_client_audio(client_id)

                # Wait for processing to complete with dynamic timeout
                max_wait_time = max(120, int(duration * 2) + 60)  # 2x duration + 60s buffer
                wait_interval = 2.0
                elapsed_time = 0

                audio_logger.info(
                    f"⏳ [Job {job_id}] Waiting for transcription (max {max_wait_time}s)"
                )

                # Track whether memory processing has been triggered to avoid duplicate calls
                memory_triggered = False

                while elapsed_time < max_wait_time:
                    try:
                        # Check database for completion status
                        chunk = await chunks_col.find_one({"client_id": client_id})
                        if chunk:
                            transcription_status = chunk.get("transcription_status", "PENDING")
                            memory_status = chunk.get("memory_processing_status", "PENDING")

                            # Update job tracker with current status
                            await job_tracker.update_file_status(
                                job_id,
                                filename,
                                FileStatus.PROCESSING,
                                audio_uuid=chunk.get("audio_uuid"),
                                transcription_status=transcription_status,
                                memory_status=memory_status,
                            )

                            # First check if transcription is complete to trigger memory processing
                            if transcription_status in ["COMPLETED", "EMPTY", "FAILED"]:
                                # Trigger memory processing if not already done
                                if memory_status == "PENDING" and not memory_triggered:
                                    audio_logger.info(
                                        f"🚀 [Job {job_id}] Transcription complete, triggering memory processing: {filename}"
                                    )
                                    await client_state.close_current_conversation()
                                    memory_triggered = True
                                    # Continue to next iteration to check memory status
                                    continue

                                # Check if memory processing is also complete
                                if memory_status in ["COMPLETED", "FAILED", "SKIPPED"]:
                                    audio_logger.info(
                                        f"✅ [Job {job_id}] File processing completed: {filename}"
                                    )
                                    await job_tracker.update_file_status(
                                        job_id, filename, FileStatus.COMPLETED
                                    )
                                    break

                    except Exception as e:
                        audio_logger.debug(f"Error checking processing status: {e}")

                    await asyncio.sleep(wait_interval)
                    elapsed_time += wait_interval

                if elapsed_time >= max_wait_time:
                    error_msg = f"Processing timed out after {max_wait_time}s"
                    audio_logger.warning(f"⏰ [Job {job_id}] {error_msg}: {filename}")
                    await job_tracker.update_file_status(
                        job_id, filename, FileStatus.FAILED, error_message=error_msg
                    )

                # Signal end of conversation - trigger memory processing
                await client_state.close_current_conversation()
                await asyncio.sleep(0.5)

            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                audio_logger.error(f"❌ [Job {job_id}] {error_msg}")
                await job_tracker.update_file_status(
                    job_id, filename, FileStatus.FAILED, error_message=error_msg
                )
            finally:
                # Always clean up client state to prevent accumulation
                if client_id and client_state:
                    try:
                        await cleanup_client_state(client_id)
                        audio_logger.info(
                            f"🧹 [Job {job_id}] Cleaned up client state for {client_id}"
                        )
                    except Exception as cleanup_error:
                        audio_logger.error(
                            f"❌ [Job {job_id}] Error cleaning up client state for {client_id}: {cleanup_error}"
                        )

        # Mark job as completed
        await job_tracker.update_job_status(job_id, JobStatus.COMPLETED)
        audio_logger.info(f"🎉 [Job {job_id}] All files processed")

    except Exception as e:
        error_msg = f"Job processing failed: {str(e)}"
        audio_logger.error(f"💥 [Job {job_id}] {error_msg}")
        await job_tracker.update_job_status(job_id, JobStatus.FAILED, error_msg)


# Configuration functions moved to config.py to avoid circular imports


async def get_diarization_settings():
    """Get current diarization settings."""
    try:
        # Reload from file to get latest settings
        settings = load_diarization_settings_from_file()
        return {
            "settings": settings,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting diarization settings: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get settings: {str(e)}"}
        )


async def save_diarization_settings(settings: dict):
    """Save diarization settings."""
    try:
        # Validate settings
        valid_keys = {
            "diarization_source", "similarity_threshold", "min_duration", "collar", 
            "min_duration_off", "min_speakers", "max_speakers"
        }
        
        for key, value in settings.items():
            if key not in valid_keys:
                return JSONResponse(
                    status_code=400, content={"error": f"Invalid setting key: {key}"}
                )
            
            # Type validation
            if key in ["min_speakers", "max_speakers"]:
                if not isinstance(value, int) or value < 1 or value > 20:
                    return JSONResponse(
                        status_code=400, content={"error": f"Invalid value for {key}: must be integer 1-20"}
                    )
            elif key == "diarization_source":
                if not isinstance(value, str) or value not in ["pyannote", "deepgram"]:
                    return JSONResponse(
                        status_code=400, content={"error": f"Invalid value for {key}: must be 'pyannote' or 'deepgram'"}
                    )
            else:
                if not isinstance(value, (int, float)) or value < 0:
                    return JSONResponse(
                        status_code=400, content={"error": f"Invalid value for {key}: must be positive number"}
                    )
        
        # Get current settings and merge with new values
        current_settings = load_diarization_settings_from_file()
        current_settings.update(settings)
        
        # Save to file
        if save_diarization_settings_to_file(current_settings):
            logger.info(f"Updated and saved diarization settings: {settings}")
            
            return {
                "message": "Diarization settings saved successfully",
                "settings": current_settings,
                "status": "success"
            }
        else:
            # Even if file save fails, we've updated the in-memory settings
            logger.warning("Settings updated in memory but file save failed")
            return {
                "message": "Settings updated (file save failed)",
                "settings": current_settings,
                "status": "partial"
            }
        
    except Exception as e:
        logger.error(f"Error saving diarization settings: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to save settings: {str(e)}"}
        )


async def get_speaker_configuration(user: User):
    """Get current user's primary speakers configuration."""
    try:
        return {
            "primary_speakers": user.primary_speakers,
            "user_id": user.user_id,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting speaker configuration for user {user.user_id}: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get speaker configuration: {str(e)}"}
        )


async def update_speaker_configuration(user: User, primary_speakers: list[dict]):
    """Update current user's primary speakers configuration."""
    try:
        # Validate speaker data format
        for speaker in primary_speakers:
            if not isinstance(speaker, dict):
                return JSONResponse(
                    status_code=400, content={"error": "Each speaker must be a dictionary"}
                )
            
            required_fields = ["speaker_id", "name", "user_id"]
            for field in required_fields:
                if field not in speaker:
                    return JSONResponse(
                        status_code=400, content={"error": f"Missing required field: {field}"}
                    )
        
        # Enforce server-side user_id and add timestamp to each speaker
        for speaker in primary_speakers:
            speaker["user_id"] = user.user_id  # Override client-supplied user_id
            speaker["selected_at"] = datetime.now(UTC).isoformat()
        
        # Update user model
        user.primary_speakers = primary_speakers
        await user.save()
        
        logger.info(f"Updated primary speakers configuration for user {user.user_id}: {len(primary_speakers)} speakers")
        
        return {
            "message": "Primary speakers configuration updated successfully",
            "primary_speakers": primary_speakers,
            "count": len(primary_speakers),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error updating speaker configuration for user {user.user_id}: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to update speaker configuration: {str(e)}"}
        )


async def get_enrolled_speakers(user: User):
    """Get enrolled speakers from speaker recognition service."""
    try:
        from advanced_omi_backend.speaker_recognition_client import SpeakerRecognitionClient
        
        # Initialize speaker recognition client
        speaker_client = SpeakerRecognitionClient()
        
        if not speaker_client.enabled:
            return {
                "speakers": [],
                "service_available": False,
                "message": "Speaker recognition service is not configured or disabled",
                "status": "success"
            }
        
        # Get enrolled speakers - using hardcoded user_id=1 for now (as noted in speaker_recognition_client.py)
        speakers = await speaker_client.get_enrolled_speakers(user_id="1")
        
        return {
            "speakers": speakers.get("speakers", []) if speakers else [],
            "service_available": True,
            "message": "Successfully retrieved enrolled speakers",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting enrolled speakers for user {user.user_id}: {e}")
        return {
            "speakers": [],
            "service_available": False,
            "message": f"Failed to retrieve speakers: {str(e)}",
            "status": "error"
        }


async def get_speaker_service_status():
    """Check speaker recognition service health status."""
    try:
        from advanced_omi_backend.speaker_recognition_client import SpeakerRecognitionClient
        
        # Initialize speaker recognition client
        speaker_client = SpeakerRecognitionClient()
        
        if not speaker_client.enabled:
            return {
                "service_available": False,
                "healthy": False,
                "message": "Speaker recognition service is not configured or disabled",
                "status": "disabled"
            }
        
        # Perform health check
        health_result = await speaker_client.health_check()
        
        if health_result:
            return {
                "service_available": True,
                "healthy": True,
                "message": "Speaker recognition service is healthy",
                "service_url": speaker_client.service_url,
                "status": "healthy"
            }
        else:
            return {
                "service_available": False,
                "healthy": False,
                "message": "Speaker recognition service is not responding",
                "service_url": speaker_client.service_url,
                "status": "unhealthy"
            }
        
    except Exception as e:
        logger.error(f"Error checking speaker service status: {e}")
        return {
            "service_available": False,
            "healthy": False,
            "message": f"Health check failed: {str(e)}",
            "status": "error"
        }


# Memory Configuration Management Functions

async def get_memory_config_raw():
    """Get current memory configuration YAML as plain text."""
    try:
        from advanced_omi_backend.memory_config_loader import get_config_loader
        
        config_loader = get_config_loader()
        config_path = config_loader.config_path
        
        if not os.path.exists(config_path):
            return JSONResponse(
                status_code=404, content={"error": f"Memory config file not found: {config_path}"}
            )
        
        with open(config_path, 'r') as file:
            config_yaml = file.read()
        
        return {
            "config_yaml": config_yaml,
            "config_path": config_path,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error reading memory config: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to read memory config: {str(e)}"}
        )


async def update_memory_config_raw(config_yaml: str):
    """Update memory configuration YAML and hot reload."""
    try:
        import yaml
        from advanced_omi_backend.memory_config_loader import get_config_loader
        
        # First validate YAML syntax
        try:
            yaml.safe_load(config_yaml)
        except yaml.YAMLError as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid YAML syntax: {str(e)}"}
            )
        
        config_loader = get_config_loader()
        config_path = config_loader.config_path
        
        # Create backup
        backup_path = f"{config_path}.bak"
        if os.path.exists(config_path):
            shutil.copy2(config_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        # Write new configuration
        with open(config_path, 'w') as file:
            file.write(config_yaml)
        
        # Hot reload configuration
        reload_success = config_loader.reload_config()
        
        if reload_success:
            logger.info("Memory configuration updated and reloaded successfully")
            return {
                "message": "Memory configuration updated and reloaded successfully",
                "config_path": config_path,
                "backup_created": os.path.exists(backup_path),
                "status": "success"
            }
        else:
            return JSONResponse(
                status_code=500, content={"error": "Configuration saved but reload failed"}
            )
        
    except Exception as e:
        logger.error(f"Error updating memory config: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to update memory config: {str(e)}"}
        )


async def validate_memory_config(config_yaml: str):
    """Validate memory configuration YAML syntax."""
    try:
        import yaml
        from advanced_omi_backend.memory_config_loader import MemoryConfigLoader
        
        # Parse YAML
        try:
            parsed_config = yaml.safe_load(config_yaml)
            if not parsed_config:
                return JSONResponse(
                    status_code=400, content={"error": "Configuration file is empty"}
                )
        except yaml.YAMLError as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid YAML syntax: {str(e)}"}
            )
        
        # Create a temporary config loader to validate structure
        try:
            # Create a temporary file for validation
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
                tmp_file.write(config_yaml)
                tmp_path = tmp_file.name
            
            # Try to load with MemoryConfigLoader to validate structure
            temp_loader = MemoryConfigLoader(tmp_path)
            temp_loader.validate_config()
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return {
                "message": "Configuration is valid",
                "status": "success"
            }
            
        except ValueError as e:
            return JSONResponse(
                status_code=400, content={"error": f"Configuration validation failed: {str(e)}"}
            )
        
    except Exception as e:
        logger.error(f"Error validating memory config: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to validate memory config: {str(e)}"}
        )


async def reload_memory_config():
    """Reload memory configuration from file."""
    try:
        from advanced_omi_backend.memory_config_loader import get_config_loader
        
        config_loader = get_config_loader()
        reload_success = config_loader.reload_config()
        
        if reload_success:
            logger.info("Memory configuration reloaded successfully")
            return {
                "message": "Memory configuration reloaded successfully",
                "config_path": config_loader.config_path,
                "status": "success"
            }
        else:
            return JSONResponse(
                status_code=500, content={"error": "Failed to reload memory configuration"}
            )
        
    except Exception as e:
        logger.error(f"Error reloading memory config: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to reload memory config: {str(e)}"}
        )


async def delete_all_user_memories(user: User):
    """Delete all memories for the current user."""
    try:
        from advanced_omi_backend.memory import get_memory_service
        
        memory_service = get_memory_service()
        
        # Delete all memories for the user
        deleted_count = await memory_service.delete_all_user_memories(user.user_id)
        
        logger.info(f"Deleted {deleted_count} memories for user {user.user_id}")
        
        return {
            "message": f"Successfully deleted {deleted_count} memories",
            "deleted_count": deleted_count,
            "user_id": user.user_id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error deleting all memories for user {user.user_id}: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to delete memories: {str(e)}"}
        )
