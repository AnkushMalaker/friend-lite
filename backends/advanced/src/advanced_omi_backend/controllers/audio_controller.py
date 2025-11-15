"""
Audio file upload and processing controller.

Handles audio file uploads and processes them directly.
Simplified to write files immediately and enqueue transcription.

Also includes audio cropping operations that work with the Conversation model.
"""

import logging
import time
import uuid
from pathlib import Path

from fastapi import UploadFile
from fastapi.responses import JSONResponse

from advanced_omi_backend.utils.audio_utils import (
    AudioValidationError,
    write_audio_file,
    _process_audio_cropping_with_relative_timestamps,
)
from advanced_omi_backend.models.job import JobPriority
from advanced_omi_backend.models.user import User
from advanced_omi_backend.models.conversation import create_conversation
from advanced_omi_backend.models.conversation import Conversation

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

                # Create conversation immediately for uploaded files (conversation_id auto-generated)
                version_id = str(uuid.uuid4())

                # Generate title from filename
                title = file.filename.rsplit('.', 1)[0][:50] if file.filename else "Uploaded Audio"

                conversation = create_conversation(
                    audio_uuid=audio_uuid,
                    user_id=user.user_id,
                    client_id=client_id,
                    title=title,
                    summary="Processing uploaded audio file..."
                )
                # Set audio_path so the frontend can play the audio
                conversation.audio_path = wav_filename
                await conversation.insert()
                conversation_id = conversation.conversation_id  # Get the auto-generated ID

                audio_logger.info(f"📝 Created conversation {conversation_id} for uploaded file")

                # Enqueue post-conversation processing job chain
                from advanced_omi_backend.controllers.queue_controller import start_post_conversation_jobs

                job_ids = start_post_conversation_jobs(
                    conversation_id=conversation_id,
                    audio_uuid=audio_uuid,
                    audio_file_path=file_path,
                    user_id=user.user_id,
                    post_transcription=True  # Run batch transcription for uploads
                )

                processed_files.append({
                    "filename": file.filename,
                    "status": "processing",
                    "audio_uuid": audio_uuid,
                    "conversation_id": conversation_id,
                    "transcript_job_id": job_ids['transcription'],
                    "speaker_job_id": job_ids['speaker_recognition'],
                    "memory_job_id": job_ids['memory'],
                    "duration_seconds": round(duration, 2),
                })

                audio_logger.info(
                    f"✅ Processed {file.filename} → conversation {conversation_id}, "
                    f"jobs: {job_ids['transcription']} → {job_ids['speaker_recognition']} → {job_ids['memory']}"
                )

            except (OSError, IOError) as e:
                # File I/O errors during audio processing
                audio_logger.exception(f"File I/O error processing {file.filename}")
                processed_files.append({
                    "filename": file.filename or "unknown",
                    "status": "error",
                    "error": str(e),
                })
            except Exception as e:
                # Unexpected errors during file processing
                audio_logger.exception(f"Unexpected error processing file {file.filename}")
                processed_files.append({
                    "filename": file.filename or "unknown",
                    "status": "error",
                    "error": str(e),
                })

        successful_files = [f for f in processed_files if f.get("status") == "processing"]
        failed_files = [f for f in processed_files if f.get("status") == "error"]

        return {
            "message": f"Uploaded and processing {len(successful_files)} file(s)",
            "client_id": client_id,
            "files": processed_files,
            "summary": {
                "total": len(files),
                "processing": len(successful_files),
                "failed": len(failed_files),
            },
        }

    except (OSError, IOError) as e:
        # File system errors during upload handling
        audio_logger.exception("File I/O error in upload_and_process_audio_files")
        return JSONResponse(
            status_code=500, content={"error": f"File upload failed: {str(e)}"}
        )
    except Exception as e:
        # Unexpected errors in upload handler
        audio_logger.exception("Unexpected error in upload_and_process_audio_files")
        return JSONResponse(
            status_code=500, content={"error": f"File upload failed: {str(e)}"}
        )


async def get_cropped_audio_info(audio_uuid: str, user: User):
    """
    Get audio cropping metadata from the conversation.

    This is an audio service operation that retrieves cropping-related metadata
    such as speech segments, cropped audio path, and cropping timestamps.

    Used for: Checking cropping status and retrieving audio processing details.
    Works with: Conversation model.
    """
    try:
        # Find the conversation
        conversation = await Conversation.find_one(Conversation.audio_uuid == audio_uuid)
        if not conversation:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser:
            if conversation.user_id != str(user.user_id):
                return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        return {
            "audio_uuid": audio_uuid,
            "cropped_audio_path": conversation.cropped_audio_path,
            "speech_segments": conversation.speech_segments if hasattr(conversation, 'speech_segments') else [],
            "cropped_duration": conversation.cropped_duration if hasattr(conversation, 'cropped_duration') else None,
            "cropped_at": conversation.cropped_at if hasattr(conversation, 'cropped_at') else None,
            "original_audio_path": conversation.audio_path,
        }

    except Exception as e:
        # Database or unexpected errors when fetching audio metadata
        audio_logger.exception("Error fetching cropped audio info")
        return JSONResponse(status_code=500, content={"error": "Error fetching cropped audio info"})


async def reprocess_audio_cropping(audio_uuid: str, user: User):
    """
    Re-process audio cropping operation for an audio file.

    This is an audio service operation that re-runs the audio cropping process
    to extract only speech segments from the full audio file.

    Used for: Re-processing audio when cropping failed or needs updating.
    Works with: Conversation model and audio_utils cropping functions.
    """
    try:
        # Find the conversation
        conversation = await Conversation.find_one(Conversation.audio_uuid == audio_uuid)
        if not conversation:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser:
            if conversation.user_id != str(user.user_id):
                return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        audio_path = conversation.audio_path
        if not audio_path:
            return JSONResponse(
                status_code=400, content={"error": "No audio file found for this conversation"}
            )

        # Check if file exists - try multiple possible locations
        possible_paths = [
            Path("/app/audio_chunks") / audio_path,
            Path(audio_path),  # fallback to relative path
        ]

        full_audio_path = None
        for path in possible_paths:
            if path.exists():
                full_audio_path = path
                break

        if not full_audio_path:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Audio file not found on disk",
                    "details": f"Conversation exists but audio file '{audio_path}' is missing from expected locations",
                    "searched_paths": [str(p) for p in possible_paths]
                }
            )

        # Get speech segments from the conversation
        speech_segments = conversation.speech_segments if hasattr(conversation, 'speech_segments') else []
        if not speech_segments:
            return JSONResponse(
                status_code=400,
                content={"error": "No speech segments found for this conversation"}
            )

        # Generate output path for cropped audio
        cropped_filename = f"cropped_{audio_uuid}.wav"
        output_path = Path("/app/audio_chunks") / cropped_filename

        # Reprocess the audio cropping (simplified without repository)
        try:
            from advanced_omi_backend.utils.audio_utils import extract_speech_segments

            success = await extract_speech_segments(
                str(full_audio_path),
                speech_segments,
                str(output_path)
            )

            if success:
                # Update conversation with cropped audio path
                conversation.cropped_audio_path = cropped_filename
                await conversation.save()

                audio_logger.info(f"Successfully reprocessed audio cropping for {audio_uuid}")
                return JSONResponse(
                    content={"message": f"Audio cropping reprocessed for {audio_uuid}"}
                )
            else:
                audio_logger.error(f"Failed to reprocess audio cropping for {audio_uuid}")
                return JSONResponse(
                    status_code=500, content={"error": "Failed to reprocess audio cropping"}
                )

        except (OSError, IOError) as processing_error:
            # File I/O errors during audio cropping
            audio_logger.exception("File I/O error during audio cropping reprocessing")
            return JSONResponse(
                status_code=500,
                content={"error": f"Audio processing failed: {str(processing_error)}"},
            )
        except Exception as processing_error:
            # Unexpected errors during cropping operation
            audio_logger.exception("Unexpected error during audio cropping reprocessing")
            return JSONResponse(
                status_code=500,
                content={"error": f"Audio processing failed: {str(processing_error)}"},
            )

    except Exception as e:
        # Database or unexpected errors in reprocessing handler
        audio_logger.exception("Error reprocessing audio cropping")
        return JSONResponse(status_code=500, content={"error": "Error reprocessing audio cropping"})
