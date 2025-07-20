"""
System and utility routes for Friend-Lite API.

Handles metrics, auth config, file processing, and other system utilities.
"""

import asyncio
import io
import logging
import time
import wave

import numpy as np
from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import JSONResponse
from wyoming.audio import AudioChunk

from advanced_omi_backend.auth import current_active_user, current_superuser
from advanced_omi_backend.client_manager import generate_client_id
from advanced_omi_backend.database import chunks_col
from advanced_omi_backend.debug_system_tracker import get_debug_tracker
from advanced_omi_backend.processors import AudioProcessingItem, get_processor_manager
from advanced_omi_backend.task_manager import get_task_manager
from advanced_omi_backend.users import User

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")

router = APIRouter(tags=["system"])


@router.get("/metrics")
async def get_current_metrics(current_user: User = Depends(current_superuser)):
    """Get current system metrics. Admin only."""
    try:
        debug_tracker = get_debug_tracker()

        # Get basic system metrics
        metrics = {
            "timestamp": int(time.time()),
            "debug_tracker_available": debug_tracker is not None,
        }

        if debug_tracker:
            # Add debug tracker metrics if available
            recent_transactions = debug_tracker.get_recent_transactions(limit=10)
            metrics.update(
                {
                    "recent_transactions_count": len(recent_transactions),
                    "recent_transactions": recent_transactions,
                }
            )

        return metrics

    except Exception as e:
        audio_logger.error(f"Error fetching metrics: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to fetch metrics: {str(e)}"}
        )


@router.get("/auth/config")
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


@router.get("/processor/tasks")
async def get_all_processing_tasks(current_user: User = Depends(current_superuser)):
    """Get all active processing tasks. Admin only."""
    try:
        processor_manager = get_processor_manager()
        return processor_manager.get_all_processing_status()
    except Exception as e:
        logger.error(f"Error getting processing tasks: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get processing tasks: {str(e)}"}
        )


@router.get("/processor/tasks/{client_id}")
async def get_processing_task_status(
    client_id: str, 
    current_user: User = Depends(current_superuser)
):
    """Get processing task status for a specific client. Admin only."""
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
            status_code=500,
            content={"error": f"Failed to get processing task status: {str(e)}"}
        )


@router.get("/processor/status")
async def get_processor_status(current_user: User = Depends(current_superuser)):
    """Get processor queue status and health. Admin only."""
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
            status_code=500, 
            content={"error": f"Failed to get processor status: {str(e)}"}
        )


@router.post("/process-audio-files")
async def process_audio_files(
    current_user: User = Depends(current_superuser),
    files: list[UploadFile] = File(...),
    device_name: str = Query(default="upload"),
    auto_generate_client: bool = Query(default=True),
):
    """Process uploaded audio files through the transcription pipeline. Admin only."""
    # Need to import here because we import the routes into main, causing circular imports
    from advanced_omi_backend.main import cleanup_client_state, create_client_state

    # Process files through complete transcription pipeline like WebSocket clients
    try:
        if not files:
            return JSONResponse(status_code=400, content={"error": "No files provided"})

        processed_files = []
        processed_conversations = []

        for file_index, file in enumerate(files):
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
                client_id = generate_client_id(current_user, file_device_name)

                # Create separate client state for this file
                client_state = await create_client_state(client_id, current_user, file_device_name)

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
                            user_id=current_user.user_id,
                            audio_chunk=chunk,
                            timestamp=chunk.timestamp
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
                audio_logger.info(f"⏳ Waiting for transcription manager to be created for client {client_id}")
                await asyncio.sleep(2.0)  # Give transcription processor time to create manager
                
                # Close client audio to trigger transcription completion (flush_final_transcript)
                audio_logger.info(f"📞 About to call close_client_audio for upload client {client_id}")
                processor_manager = get_processor_manager()
                audio_logger.info(f"📞 Got processor manager, calling close_client_audio now...")
                await processor_manager.close_client_audio(client_id)
                audio_logger.info(f"🔚 Successfully called close_client_audio for upload client {client_id}")

                # Wait for this file's transcription processing to complete
                audio_logger.info(f"📁 Waiting for transcription to process file: {file.filename}")

                # Wait for chunks to be processed by the audio saver
                await asyncio.sleep(1.0)

                # Wait for file processing to complete using task tracking
                # Increase timeout based on file duration (3x duration + 60s buffer)
                audio_duration = len(audio_data) / (sample_rate * sample_width * channels)
                max_wait_time = max(120, int(audio_duration * 3) + 60)  # At least 2 minutes, or 3x duration + 60s
                wait_interval = 2.0  # Reduced from 0.5s to 2s to reduce polling spam
                elapsed_time = 0
                
                audio_logger.info(f"📁 Audio duration: {audio_duration:.1f}s, max wait time: {max_wait_time}s")

                # Use concrete task tracking instead of database polling
                while elapsed_time < max_wait_time:
                    try:
                        # Check processing status using task tracking
                        processing_status = processor_manager.get_processing_status(client_id)
                        
                        # Check if transcription stage is complete
                        stages = processing_status.get("stages", {})
                        transcription_stage = stages.get("transcription", {})
                        
                        # If transcription is marked as started but not completed, check database
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
                                    audio_logger.info(f"📁 Transcription completed for file: {file.filename} ({len(chunk.get('transcript', []))} segments)")
                                    break
                            except Exception as e:
                                audio_logger.debug(f"Error checking transcription completion: {e}")
                        
                        if transcription_stage.get("completed", False):
                            audio_logger.info(f"📁 Transcription completed for file: {file.filename}")
                            break
                        
                        # Check for errors
                        if transcription_stage.get("error"):
                            audio_logger.warning(f"📁 Transcription error for file: {file.filename}: {transcription_stage.get('error')}")
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

                # Clean up client state to prevent accumulation of active clients
                await cleanup_client_state(client_id)
                audio_logger.info(
                    f"📁 Completed processing file {file_index + 1}/{len(files)}: {file.filename} - client cleaned up"
                )

            except Exception as e:
                audio_logger.error(f"Error processing file {file.filename}: {e}")
                # Clean up client state even on error to prevent accumulation
                if "client_state" in locals():
                    await cleanup_client_state(client_id)
                processed_files.append(
                    {"filename": file.filename or "unknown", "status": "error", "error": str(e)}
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
