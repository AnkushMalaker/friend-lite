"""
System and utility routes for Friend-Lite API.

Handles metrics, auth config, and other system utilities.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request

from advanced_omi_backend.auth import current_active_user, current_superuser
from advanced_omi_backend.controllers import system_controller
from advanced_omi_backend.models.user import User

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


@router.get("/streaming/status")
async def get_streaming_status(request: Request, current_user: User = Depends(current_superuser)):
    """Get status of active streaming sessions and Redis Streams health. Admin only."""
    return await system_controller.get_streaming_status(request)


@router.post("/streaming/cleanup")
async def cleanup_stuck_stream_workers(request: Request, current_user: User = Depends(current_superuser)):
    """Clean up stuck Redis Stream workers and pending messages. Admin only."""
    return await system_controller.cleanup_stuck_stream_workers(request)


@router.post("/streaming/cleanup-sessions")
async def cleanup_old_sessions(request: Request, max_age_seconds: int = 3600, current_user: User = Depends(current_superuser)):
    """Clean up old session tracking metadata. Admin only."""
    return await system_controller.cleanup_old_sessions(request, max_age_seconds)
