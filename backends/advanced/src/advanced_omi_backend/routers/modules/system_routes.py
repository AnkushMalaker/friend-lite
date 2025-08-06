"""
System and utility routes for Friend-Lite API.

Handles metrics, auth config, file processing, and other system utilities.
"""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile

from advanced_omi_backend.auth import current_superuser
from advanced_omi_backend.controllers import system_controller
from advanced_omi_backend.users import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


@router.get("/metrics")
async def get_current_metrics(current_user: User = Depends(current_superuser)):
    """Get current system metrics. Admin only."""
    return await system_controller.get_current_metrics()


@router.get("/auth/config")
async def get_auth_config():
    """Get authentication configuration for frontend."""
    return await system_controller.get_auth_config()


@router.get("/processor/tasks")
async def get_all_processing_tasks(current_user: User = Depends(current_superuser)):
    """Get all active processing tasks. Admin only."""
    return await system_controller.get_all_processing_tasks()


@router.get("/processor/tasks/{client_id}")
async def get_processing_task_status(
    client_id: str, current_user: User = Depends(current_superuser)
):
    """Get processing task status for a specific client. Admin only."""
    return await system_controller.get_processing_task_status(client_id)


@router.get("/processor/status")
async def get_processor_status(current_user: User = Depends(current_superuser)):
    """Get processor queue status and health. Admin only."""
    return await system_controller.get_processor_status()


@router.post("/process-audio-files")
async def process_audio_files(
    current_user: User = Depends(current_superuser),
    files: list[UploadFile] = File(...),
    device_name: str = Query(default="upload"),
    auto_generate_client: bool = Query(default=True),
):
    """Process uploaded audio files through the transcription pipeline. Admin only."""
    return await system_controller.process_audio_files(
        current_user, files, device_name, auto_generate_client
    )


@router.post("/process-audio-files-async")
async def process_audio_files_async(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(current_superuser),
    files: list[UploadFile] = File(...),
    device_name: str = Query(default="upload"),
):
    """Start async processing of uploaded audio files. Returns job ID immediately. Admin only."""
    return await system_controller.process_audio_files_async(
        background_tasks, current_user, files, device_name
    )


@router.get("/process-audio-files/jobs/{job_id}")
async def get_processing_job_status(job_id: str, current_user: User = Depends(current_superuser)):
    """Get status of an async file processing job. Admin only."""
    return await system_controller.get_processing_job_status(job_id)


@router.get("/process-audio-files/jobs")
async def list_processing_jobs(current_user: User = Depends(current_superuser)):
    """List all active processing jobs. Admin only."""
    return await system_controller.list_processing_jobs()
