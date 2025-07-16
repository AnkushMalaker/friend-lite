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

from advanced_omi_backend.auth import current_superuser
from advanced_omi_backend.debug_system_tracker import get_debug_tracker
from advanced_omi_backend.main import cleanup_client_state, create_client_state
from advanced_omi_backend.users import User, generate_client_id

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
        return JSONResponse(status_code=500, content={"error": "Failed to fetch metrics"})


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


@router.post("/process-audio-files")
async def process_audio_files(
    current_user: User = Depends(current_superuser),
    files: list[UploadFile] = File(...),
    device_name: str = Query(default="upload"),
    auto_generate_client: bool = Query(default=True),
):
    """Process uploaded audio files through the transcription pipeline. Admin only."""
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

                        # Add to processing queue - this starts the transcription pipeline
                        await client_state.chunk_queue.put(chunk)

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

                # Wait for this file's transcription processing to complete
                audio_logger.info(f"📁 Waiting for transcription to process file: {file.filename}")

                # Wait for chunks to be processed by the audio saver
                await asyncio.sleep(1.0)

                # Wait for transcription queue to be processed for this file
                max_wait_time = 60  # 1 minute per file
                wait_interval = 0.5
                elapsed_time = 0

                while elapsed_time < max_wait_time:
                    if (
                        client_state.transcription_queue.empty()
                        and client_state.chunk_queue.empty()
                    ):
                        audio_logger.info(f"📁 Transcription completed for file: {file.filename}")
                        break

                    await asyncio.sleep(wait_interval)
                    elapsed_time += wait_interval

                if elapsed_time >= max_wait_time:
                    audio_logger.warning(f"📁 Transcription timed out for file: {file.filename}")

                # Close this conversation by sending None to chunk queue
                await client_state.chunk_queue.put(None)

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
