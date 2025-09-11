"""
System and utility routes for Friend-Lite API.

Handles metrics, auth config, file processing, and other system utilities.
"""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile

from advanced_omi_backend.auth import current_active_user, current_superuser
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


@router.get("/diarization-settings")
async def get_diarization_settings(current_user: User = Depends(current_superuser)):
    """Get current diarization settings. Admin only."""
    return await system_controller.get_diarization_settings()


@router.post("/diarization-settings")
async def save_diarization_settings(
    settings: dict,
    current_user: User = Depends(current_superuser)
):
    """Save diarization settings. Admin only."""
    return await system_controller.save_diarization_settings(settings)


@router.get("/speaker-configuration")
async def get_speaker_configuration(current_user: User = Depends(current_active_user)):
    """Get current user's primary speakers configuration."""
    return await system_controller.get_speaker_configuration(current_user)


@router.post("/speaker-configuration")
async def update_speaker_configuration(
    primary_speakers: list[dict],
    current_user: User = Depends(current_active_user)
):
    """Update current user's primary speakers configuration."""
    return await system_controller.update_speaker_configuration(current_user, primary_speakers)


@router.get("/enrolled-speakers")
async def get_enrolled_speakers(current_user: User = Depends(current_active_user)):
    """Get enrolled speakers from speaker recognition service."""
    return await system_controller.get_enrolled_speakers(current_user)


@router.get("/speaker-service-status")
async def get_speaker_service_status(current_user: User = Depends(current_superuser)):
    """Check speaker recognition service health status. Admin only."""
    return await system_controller.get_speaker_service_status()


# Memory Configuration Management Endpoints

@router.get("/admin/memory/config/raw")
async def get_memory_config_raw(current_user: User = Depends(current_superuser)):
    """Get current memory configuration YAML as plain text. Admin only."""
    return await system_controller.get_memory_config_raw()


@router.post("/admin/memory/config/raw")
async def update_memory_config_raw(
    config_yaml: str,
    current_user: User = Depends(current_superuser)
):
    """Update memory configuration YAML and hot reload. Admin only."""
    return await system_controller.update_memory_config_raw(config_yaml)


@router.post("/admin/memory/config/validate")
async def validate_memory_config(
    config_yaml: str,
    current_user: User = Depends(current_superuser)
):
    """Validate memory configuration YAML syntax. Admin only."""
    return await system_controller.validate_memory_config(config_yaml)


@router.post("/admin/memory/config/reload")
async def reload_memory_config(current_user: User = Depends(current_superuser)):
    """Reload memory configuration from file. Admin only."""
    return await system_controller.reload_memory_config()


@router.delete("/admin/memory/delete-all")
async def delete_all_user_memories(current_user: User = Depends(current_active_user)):
    """Delete all memories for the current user."""
    return await system_controller.delete_all_user_memories(current_user)
