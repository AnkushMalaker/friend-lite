"""
Audio file upload routes.

Handles audio file uploads and processing job management.
"""

from fastapi import APIRouter, Depends, File, Query, UploadFile

from advanced_omi_backend.auth import current_superuser
from advanced_omi_backend.controllers import audio_controller
from advanced_omi_backend.models.user import User

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
