"""
Audio file upload and serving routes.

Handles audio file uploads, processing job management, and audio file serving.
"""

from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from advanced_omi_backend.auth import current_superuser, current_active_user
from advanced_omi_backend.controllers import audio_controller
from advanced_omi_backend.models.user import User
from advanced_omi_backend.app_config import get_audio_chunk_dir

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/upload")
async def upload_audio_files(
    current_user: User = Depends(current_superuser),
    files: list[UploadFile] = File(...),
    device_name: str = Query(default="upload", description="Device name for uploaded files"),
    auto_generate_client: bool = Query(default=True, description="Auto-generate client ID"),
):
    """
    Upload and process audio files. Admin only.

    Audio files are saved to disk and enqueued for processing via RQ jobs.
    This allows for scalable processing of large files without blocking the API.

    Returns:
        - List of uploaded files with their processing job IDs
        - Summary of enqueued vs failed uploads
    """
    return await audio_controller.upload_and_process_audio_files(
        current_user, files, device_name, auto_generate_client
    )


@router.get("/{filename}")
async def serve_audio_file(
    filename: str,
    current_user: User = Depends(current_active_user),
):
    """
    Serve audio files for playback.

    This endpoint allows authenticated users to access audio files stored on disk.
    Users can only access their own audio files (ownership verified via conversation lookup).

    Args:
        filename: Name of the audio file (e.g., "audio_uuid.wav" or "cropped_audio_uuid.wav")
        current_user: Authenticated user

    Returns:
        FileResponse with the audio file

    Raises:
        404: If file not found
        403: If user doesn't own the conversation associated with this audio
    """
    # Get audio chunk directory
    audio_dir = get_audio_chunk_dir()
    file_path = audio_dir / filename

    # Check if file exists
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Security check: Verify user owns the conversation associated with this audio
    # Extract audio_uuid from filename (handle both "uuid.wav" and "cropped_uuid.wav")
    from advanced_omi_backend.models.conversation import Conversation

    # Remove "cropped_" prefix if present
    audio_uuid = filename.replace("cropped_", "").replace(".wav", "")

    # Find conversation by audio_uuid
    conversation = await Conversation.find_one(Conversation.audio_uuid == audio_uuid)

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found for this audio file")

    # Check ownership (admins can access all files)
    if not current_user.is_superuser and conversation.user_id != str(current_user.user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    # Serve the file
    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename
    )
