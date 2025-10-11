"""
Audio file upload and processing controller.

Handles audio file uploads and processes them directly.
Simplified to write files immediately and enqueue transcription.
"""

import logging
import time
import uuid
from datetime import UTC, datetime

from fastapi import UploadFile
from fastapi.responses import JSONResponse

from advanced_omi_backend.audio_utils import AudioValidationError, write_audio_file
from advanced_omi_backend.models.job import JobPriority
from advanced_omi_backend.models.user import User
from advanced_omi_backend.models.conversation import Conversation
from advanced_omi_backend.workers.transcription_jobs import enqueue_transcript_processing
from advanced_omi_backend.workers.memory_jobs import enqueue_memory_processing

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
    Upload audio files and process them directly.

    Simplified flow:
    1. Validate and read WAV file
    2. Write audio file and create AudioSession immediately
    3. Enqueue transcription job (same as WebSocket path)
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

                # Generate audio UUID and timestamp
                audio_uuid = str(uuid.uuid4())
                timestamp = int(time.time() * 1000)

                # Validate, write audio file and create AudioSession (all in one)
                try:
                    wav_filename, file_path, duration = await write_audio_file(
                        raw_audio_data=content,
                        audio_uuid=audio_uuid,
                        client_id=client_id,
                        user_id=user.user_id,
                        user_email=user.email,
                        timestamp=timestamp,
                        validate=True  # Validate WAV format, convert stereo→mono
                    )
                except AudioValidationError as e:
                    processed_files.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": str(e),
                    })
                    continue

                audio_logger.info(
                    f"📊 {file.filename}: {duration:.1f}s → {wav_filename}"
                )

                # Create conversation immediately for uploaded files
                conversation_id = str(uuid.uuid4())
                version_id = str(uuid.uuid4())

                # Generate title from filename
                title = file.filename.rsplit('.', 1)[0][:50] if file.filename else "Uploaded Audio"

                conversation = Conversation(
                    conversation_id=conversation_id,
                    audio_uuid=audio_uuid,
                    user_id=user.user_id,
                    client_id=client_id,
                    title=title,
                    summary="Processing uploaded audio file...",
                    transcript_versions=[],
                    active_transcript_version=None,
                    memory_versions=[],
                    active_memory_version=None,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                    session_start=datetime.fromtimestamp(timestamp / 1000, tz=UTC),
                    session_end=datetime.fromtimestamp(timestamp / 1000, tz=UTC),
                    duration_seconds=duration,
                    speech_start_time=0.0,
                    speech_end_time=duration,
                    speaker_names={},
                    action_items=[]
                )
                await conversation.insert()

                audio_logger.info(f"📝 Created conversation {conversation_id} for uploaded file")

                # Enqueue transcript processing job
                transcript_job = enqueue_transcript_processing(
                    conversation_id=conversation_id,
                    audio_uuid=audio_uuid,
                    audio_path=file_path,
                    version_id=version_id,
                    user_id=user.user_id,
                    priority=JobPriority.HIGH,
                    trigger="file_upload"
                )

                # Enqueue memory processing job (will run after transcript completes)
                memory_job = enqueue_memory_processing(
                    client_id=client_id,
                    user_id=user.user_id,
                    user_email=user.email,
                    conversation_id=conversation_id,
                    priority=JobPriority.NORMAL
                )

                processed_files.append({
                    "filename": file.filename,
                    "status": "processing",
                    "audio_uuid": audio_uuid,
                    "conversation_id": conversation_id,
                    "transcript_job_id": transcript_job.id,
                    "memory_job_id": memory_job.id,
                    "duration_seconds": round(duration, 2),
                })

                enqueued_jobs.append({
                    "transcript_job_id": transcript_job.id,
                    "memory_job_id": memory_job.id,
                    "conversation_id": conversation_id,
                    "audio_uuid": audio_uuid,
                    "filename": file.filename,
                })

                audio_logger.info(
                    f"✅ Processed {file.filename} → conversation {conversation_id}, "
                    f"transcript job {transcript_job.id}, memory job {memory_job.id}"
                )

            except Exception as e:
                audio_logger.error(f"Error processing file {file.filename}: {e}")
                processed_files.append({
                    "filename": file.filename or "unknown",
                    "status": "error",
                    "error": str(e),
                })

        return {
            "message": f"Uploaded and processing {len(enqueued_jobs)} file(s)",
            "client_id": client_id,
            "files": processed_files,
            "jobs": enqueued_jobs,
            "summary": {
                "total": len(files),
                "processing": len(enqueued_jobs),
                "failed": len([f for f in processed_files if f.get("status") == "error"]),
            },
        }

    except Exception as e:
        audio_logger.error(f"Error in upload_and_process_audio_files: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"File upload failed: {str(e)}"}
        )
