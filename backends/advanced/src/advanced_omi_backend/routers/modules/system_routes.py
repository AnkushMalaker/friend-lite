"""
System and utility routes for Friend-Lite API.

Handles metrics, auth config, file processing, and other system utilities.
"""

import logging

from advanced_omi_backend.auth import current_active_user, current_superuser
from advanced_omi_backend.controllers import system_controller
from advanced_omi_backend.users import User
from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile

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


@router.get("/processor/status")
async def get_processor_status(current_user: User = Depends(current_superuser)):
    """Get processor queue status and health. Admin only."""
    return await system_controller.get_processor_status()


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


@router.get("/jobs/all")
async def list_all_jobs(current_user: User = Depends(current_superuser)):
    """List all jobs from MongoDB (including completed/failed). Admin only."""
    return await system_controller.list_all_jobs()


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


@router.get("/processor/overview")
async def get_processor_overview_route(current_user: User = Depends(current_superuser)):
    """Get comprehensive processor overview with pipeline stats. Admin only."""
    return await system_controller.get_processor_overview()

@router.get("/processor/history")
async def get_processor_history_route(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    current_user: User = Depends(current_superuser)
):
    """Get paginated processing history. Admin only."""
    return await system_controller.get_processor_history(page, per_page)

@router.get("/processor/clients/{client_id}")
async def get_client_processing_detail_route(
    client_id: str,
    current_user: User = Depends(current_superuser)
):
    """Get detailed processing information for specific client. Admin only."""
    return await system_controller.get_client_processing_detail(client_id)


@router.get("/processor/bottlenecks")
async def get_pipeline_bottlenecks_route(current_user: User = Depends(current_superuser)):
    """Get pipeline bottleneck analysis with recommendations. Admin only."""
    return await system_controller.get_pipeline_bottlenecks()


@router.get("/processor/pipeline-health")
async def get_pipeline_health_route(current_user: User = Depends(current_superuser)):
    """Get comprehensive pipeline health metrics. Admin only."""
    return await system_controller.get_pipeline_health()


@router.get("/processor/queue-metrics")
async def get_queue_metrics_route(current_user: User = Depends(current_superuser)):
    """Get real-time queue metrics and performance data. Admin only."""
    return await system_controller.get_queue_metrics()


@router.get("/processor/sessions/{audio_uuid}")
async def get_session_pipeline_route(
    audio_uuid: str,
    current_user: User = Depends(current_superuser)
):
    """Get detailed pipeline timeline for a specific audio session. Admin only."""
    return await system_controller.get_session_pipeline(audio_uuid)


