"""
Audio file upload and processing controller.

Handles audio file uploads and enqueues them for processing via RQ jobs.
"""

import io
import logging
import time
import uuid
import wave
from pathlib import Path

import numpy as np
from fastapi import UploadFile
from fastapi.responses import JSONResponse

from advanced_omi_backend.config import CHUNK_DIR
from advanced_omi_backend.models.user import User
from advanced_omi_backend.rq_queue import enqueue_audio_processing

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")


def generate_client_id(user: User, device_name: str) -> str:
    """Generate client ID for uploaded files."""
    user_id_suffix = str(user.id)[-6:]
    return f"{user_id_suffix}-{device_name}"


async def upload_and_process_audio_files(
    user: User,
    files: list[UploadFile],
    device_name: str = "upload",
    auto_generate_client: bool = True,
) -> dict:
    """
    Upload audio files and enqueue them for RQ processing.

    Unlike WebSocket streaming, file uploads are processed as complete files
    through RQ jobs for better scalability and resource management.
    """
    try:
        if not files:
            return JSONResponse(status_code=400, content={"error": "No files provided"})

        processed_files = []
        enqueued_jobs = []
        client_id = generate_client_id(user, device_name)

        for file_index, file in enumerate(files):
            try:
                # Validate file type (only WAV for now)
                if not file.filename or not file.filename.lower().endswith(".wav"):
                    processed_files.append({
                        "filename": file.filename or "unknown",
                        "status": "error",
                        "error": "Only WAV files are currently supported",
                    })
                    continue

                audio_logger.info(
                    f"📁 Uploading file {file_index + 1}/{len(files)}: {file.filename}"
                )

                # Read file content
                content = await file.read()

                # Validate WAV file and get parameters
                try:
                    with wave.open(io.BytesIO(content), "rb") as wav_file:
                        sample_rate = wav_file.getframerate()
                        sample_width = wav_file.getsampwidth()
                        channels = wav_file.getnchannels()
                        frame_count = wav_file.getnframes()
                        duration = frame_count / sample_rate

                        # Read audio data
                        audio_data = wav_file.readframes(frame_count)

                except Exception as e:
                    processed_files.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": f"Invalid WAV file: {str(e)}",
                    })
                    continue

                # Convert to mono if stereo
                if channels == 2:
                    if sample_width == 2:
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    else:
                        audio_array = np.frombuffer(audio_data, dtype=np.int32)

                    # Reshape to separate channels and average
                    audio_array = audio_array.reshape(-1, 2)
                    audio_data = np.mean(audio_array, axis=1).astype(audio_array.dtype).tobytes()
                    channels = 1

                # Check sample rate
                if sample_rate != 16000:
                    processed_files.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": f"Sample rate must be 16kHz, got {sample_rate}Hz",
                    })
                    continue

                # Save audio file to disk
                audio_uuid = str(uuid.uuid4())
                audio_filename = f"{audio_uuid}.wav"
                audio_path = CHUNK_DIR / audio_filename

                # Write processed audio to file
                with wave.open(str(audio_path), "wb") as wav_out:
                    wav_out.setnchannels(channels)
                    wav_out.setsampwidth(sample_width)
                    wav_out.setframerate(sample_rate)
                    wav_out.writeframes(audio_data)

                audio_logger.info(
                    f"💾 Saved audio file: {audio_filename} "
                    f"({duration:.1f}s, {sample_rate}Hz, {channels}ch)"
                )

                # Enqueue RQ job for processing
                job = enqueue_audio_processing(
                    client_id=client_id,
                    user_id=user.user_id,
                    user_email=user.email,
                    audio_uuid=audio_uuid,
                    audio_file_path=str(audio_path),
                    timestamp=int(time.time() * 1000),
                )

                processed_files.append({
                    "filename": file.filename,
                    "status": "enqueued",
                    "audio_uuid": audio_uuid,
                    "job_id": job.id,
                    "duration_seconds": round(duration, 2),
                })

                enqueued_jobs.append({
                    "job_id": job.id,
                    "audio_uuid": audio_uuid,
                    "filename": file.filename,
                })

                audio_logger.info(
                    f"✅ Enqueued RQ job {job.id} for {file.filename} (audio_uuid: {audio_uuid})"
                )

            except Exception as e:
                audio_logger.error(f"Error processing file {file.filename}: {e}")
                processed_files.append({
                    "filename": file.filename or "unknown",
                    "status": "error",
                    "error": str(e),
                })

        return {
            "message": f"Uploaded and enqueued {len(enqueued_jobs)} file(s) for processing",
            "client_id": client_id,
            "files": processed_files,
            "jobs": enqueued_jobs,
            "summary": {
                "total": len(files),
                "enqueued": len(enqueued_jobs),
                "failed": len([f for f in processed_files if f.get("status") == "error"]),
            },
        }

    except Exception as e:
        audio_logger.error(f"Error in upload_and_process_audio_files: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"File upload failed: {str(e)}"}
        )
