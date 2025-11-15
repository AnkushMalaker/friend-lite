"""
System controller for handling system-related business logic.
"""

import logging
import os
import shutil
import time
from datetime import UTC, datetime

from advanced_omi_backend.config import (
    load_diarization_settings_from_file,
    save_diarization_settings_to_file,
)
from advanced_omi_backend.models.user import User
from advanced_omi_backend.task_manager import get_task_manager
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")


async def get_current_metrics():
    """Get current system metrics."""
    try:
        # Get memory provider configuration
        memory_provider = os.getenv("MEMORY_PROVIDER", "friend_lite").lower()
        
        # Get basic system metrics
        metrics = {
            "timestamp": int(time.time()),
            "memory_provider": memory_provider,
            "memory_provider_supports_threshold": memory_provider == "friend_lite",
        }

        return metrics

    except Exception as e:
        audio_logger.error(f"Error fetching metrics: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to fetch metrics: {str(e)}"}
        )


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


async def get_all_processing_tasks():
    """Get all active processing tasks.

    NOTE: This function is deprecated - old processor architecture has been removed.
    Kept for backward compatibility but always returns empty list.
    """
    logger.warning("get_all_processing_tasks called - deprecated function")
    return []


async def get_processing_task_status(client_id: str):
    """Get processing task status for a specific client.

    NOTE: This function is deprecated - old processor architecture has been removed.
    Kept for backward compatibility but always returns None.
    """
    logger.warning(f"get_processing_task_status called for {client_id} - deprecated function")
    return None


async def get_processor_status():
    """Get RQ worker and queue status."""
    try:
        # Get RQ queue health (new architecture)
        from advanced_omi_backend.controllers.queue_controller import get_queue_health
        queue_health = get_queue_health()

        status = {
            "architecture": "rq_workers",  # New RQ-based architecture
            "timestamp": int(time.time()),
            "workers": {
                "total": queue_health.get("total_workers", 0),
                "active": queue_health.get("active_workers", 0),
                "idle": queue_health.get("idle_workers", 0),
                "details": queue_health.get("workers", [])
            },
            "queues": {
                "transcription": queue_health.get("queues", {}).get("transcription", {}),
                "memory": queue_health.get("queues", {}).get("memory", {}),
                "default": queue_health.get("queues", {}).get("default", {})
            }
        }

        # Get task manager status if available
        try:
            task_manager = get_task_manager()
            if task_manager:
                task_status = task_manager.get_health_status()
                status["task_manager"] = task_status
        except Exception as e:
            status["task_manager"] = {"error": str(e)}

        return status

    except Exception as e:
        logger.error(f"Error getting processor status: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get processor status: {str(e)}"}
        )


# Audio file processing functions moved to audio_controller.py


# Configuration functions moved to config.py to avoid circular imports


async def get_diarization_settings():
    """Get current diarization settings."""
    try:
        # Reload from file to get latest settings
        settings = load_diarization_settings_from_file()
        return {
            "settings": settings,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting diarization settings: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get settings: {str(e)}"}
        )


async def save_diarization_settings(settings: dict):
    """Save diarization settings."""
    try:
        # Validate settings
        valid_keys = {
            "diarization_source", "similarity_threshold", "min_duration", "collar", 
            "min_duration_off", "min_speakers", "max_speakers"
        }
        
        for key, value in settings.items():
            if key not in valid_keys:
                return JSONResponse(
                    status_code=400, content={"error": f"Invalid setting key: {key}"}
                )
            
            # Type validation
            if key in ["min_speakers", "max_speakers"]:
                if not isinstance(value, int) or value < 1 or value > 20:
                    return JSONResponse(
                        status_code=400, content={"error": f"Invalid value for {key}: must be integer 1-20"}
                    )
            elif key == "diarization_source":
                if not isinstance(value, str) or value not in ["pyannote", "deepgram"]:
                    return JSONResponse(
                        status_code=400, content={"error": f"Invalid value for {key}: must be 'pyannote' or 'deepgram'"}
                    )
            else:
                if not isinstance(value, (int, float)) or value < 0:
                    return JSONResponse(
                        status_code=400, content={"error": f"Invalid value for {key}: must be positive number"}
                    )
        
        # Get current settings and merge with new values
        current_settings = load_diarization_settings_from_file()
        current_settings.update(settings)
        
        # Save to file
        if save_diarization_settings_to_file(current_settings):
            logger.info(f"Updated and saved diarization settings: {settings}")
            
            return {
                "message": "Diarization settings saved successfully",
                "settings": current_settings,
                "status": "success"
            }
        else:
            # Even if file save fails, we've updated the in-memory settings
            logger.warning("Settings updated in memory but file save failed")
            return {
                "message": "Settings updated (file save failed)",
                "settings": current_settings,
                "status": "partial"
            }
        
    except Exception as e:
        logger.error(f"Error saving diarization settings: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to save settings: {str(e)}"}
        )


async def get_speaker_configuration(user: User):
    """Get current user's primary speakers configuration."""
    try:
        return {
            "primary_speakers": user.primary_speakers,
            "user_id": user.user_id,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting speaker configuration for user {user.user_id}: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get speaker configuration: {str(e)}"}
        )


async def update_speaker_configuration(user: User, primary_speakers: list[dict]):
    """Update current user's primary speakers configuration."""
    try:
        # Validate speaker data format
        for speaker in primary_speakers:
            if not isinstance(speaker, dict):
                return JSONResponse(
                    status_code=400, content={"error": "Each speaker must be a dictionary"}
                )
            
            required_fields = ["speaker_id", "name", "user_id"]
            for field in required_fields:
                if field not in speaker:
                    return JSONResponse(
                        status_code=400, content={"error": f"Missing required field: {field}"}
                    )
        
        # Enforce server-side user_id and add timestamp to each speaker
        for speaker in primary_speakers:
            speaker["user_id"] = user.user_id  # Override client-supplied user_id
            speaker["selected_at"] = datetime.now(UTC).isoformat()
        
        # Update user model
        user.primary_speakers = primary_speakers
        await user.save()
        
        logger.info(f"Updated primary speakers configuration for user {user.user_id}: {len(primary_speakers)} speakers")
        
        return {
            "message": "Primary speakers configuration updated successfully",
            "primary_speakers": primary_speakers,
            "count": len(primary_speakers),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error updating speaker configuration for user {user.user_id}: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to update speaker configuration: {str(e)}"}
        )


async def get_enrolled_speakers(user: User):
    """Get enrolled speakers from speaker recognition service."""
    try:
        from advanced_omi_backend.speaker_recognition_client import (
            SpeakerRecognitionClient,
        )

        # Initialize speaker recognition client
        speaker_client = SpeakerRecognitionClient()
        
        if not speaker_client.enabled:
            return {
                "speakers": [],
                "service_available": False,
                "message": "Speaker recognition service is not configured or disabled",
                "status": "success"
            }
        
        # Get enrolled speakers - using hardcoded user_id=1 for now (as noted in speaker_recognition_client.py)
        speakers = await speaker_client.get_enrolled_speakers(user_id="1")
        
        return {
            "speakers": speakers.get("speakers", []) if speakers else [],
            "service_available": True,
            "message": "Successfully retrieved enrolled speakers",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting enrolled speakers for user {user.user_id}: {e}")
        return {
            "speakers": [],
            "service_available": False,
            "message": f"Failed to retrieve speakers: {str(e)}",
            "status": "error"
        }


async def get_speaker_service_status():
    """Check speaker recognition service health status."""
    try:
        from advanced_omi_backend.speaker_recognition_client import (
            SpeakerRecognitionClient,
        )

        # Initialize speaker recognition client
        speaker_client = SpeakerRecognitionClient()
        
        if not speaker_client.enabled:
            return {
                "service_available": False,
                "healthy": False,
                "message": "Speaker recognition service is not configured or disabled",
                "status": "disabled"
            }
        
        # Perform health check
        health_result = await speaker_client.health_check()
        
        if health_result:
            return {
                "service_available": True,
                "healthy": True,
                "message": "Speaker recognition service is healthy",
                "service_url": speaker_client.service_url,
                "status": "healthy"
            }
        else:
            return {
                "service_available": False,
                "healthy": False,
                "message": "Speaker recognition service is not responding",
                "service_url": speaker_client.service_url,
                "status": "unhealthy"
            }
        
    except Exception as e:
        logger.error(f"Error checking speaker service status: {e}")
        return {
            "service_available": False,
            "healthy": False,
            "message": f"Health check failed: {str(e)}",
            "status": "error"
        }


# Memory Configuration Management Functions

async def get_memory_config_raw():
    """Get current memory configuration YAML as plain text."""
    try:
        from advanced_omi_backend.memory_config_loader import get_config_loader
        
        config_loader = get_config_loader()
        config_path = config_loader.config_path
        
        if not os.path.exists(config_path):
            return JSONResponse(
                status_code=404, content={"error": f"Memory config file not found: {config_path}"}
            )
        
        with open(config_path, 'r') as file:
            config_yaml = file.read()
        
        return {
            "config_yaml": config_yaml,
            "config_path": config_path,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error reading memory config: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to read memory config: {str(e)}"}
        )


async def update_memory_config_raw(config_yaml: str):
    """Update memory configuration YAML and hot reload."""
    try:
        import yaml
        from advanced_omi_backend.memory_config_loader import get_config_loader

        # First validate YAML syntax
        try:
            yaml.safe_load(config_yaml)
        except yaml.YAMLError as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid YAML syntax: {str(e)}"}
            )
        
        config_loader = get_config_loader()
        config_path = config_loader.config_path
        
        # Create backup
        backup_path = f"{config_path}.bak"
        if os.path.exists(config_path):
            shutil.copy2(config_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        # Write new configuration
        with open(config_path, 'w') as file:
            file.write(config_yaml)
        
        # Hot reload configuration
        reload_success = config_loader.reload_config()
        
        if reload_success:
            logger.info("Memory configuration updated and reloaded successfully")
            return {
                "message": "Memory configuration updated and reloaded successfully",
                "config_path": config_path,
                "backup_created": os.path.exists(backup_path),
                "status": "success"
            }
        else:
            return JSONResponse(
                status_code=500, content={"error": "Configuration saved but reload failed"}
            )
        
    except Exception as e:
        logger.error(f"Error updating memory config: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to update memory config: {str(e)}"}
        )


async def validate_memory_config(config_yaml: str):
    """Validate memory configuration YAML syntax."""
    try:
        import yaml
        from advanced_omi_backend.memory_config_loader import MemoryConfigLoader

        # Parse YAML
        try:
            parsed_config = yaml.safe_load(config_yaml)
            if not parsed_config:
                return JSONResponse(
                    status_code=400, content={"error": "Configuration file is empty"}
                )
        except yaml.YAMLError as e:
            return JSONResponse(
                status_code=400, content={"error": f"Invalid YAML syntax: {str(e)}"}
            )
        
        # Create a temporary config loader to validate structure
        try:
            # Create a temporary file for validation
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
                tmp_file.write(config_yaml)
                tmp_path = tmp_file.name
            
            # Try to load with MemoryConfigLoader to validate structure
            temp_loader = MemoryConfigLoader(tmp_path)
            temp_loader.validate_config()
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return {
                "message": "Configuration is valid",
                "status": "success"
            }
            
        except ValueError as e:
            return JSONResponse(
                status_code=400, content={"error": f"Configuration validation failed: {str(e)}"}
            )
        
    except Exception as e:
        logger.error(f"Error validating memory config: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to validate memory config: {str(e)}"}
        )


async def reload_memory_config():
    """Reload memory configuration from file."""
    try:
        from advanced_omi_backend.memory_config_loader import get_config_loader
        
        config_loader = get_config_loader()
        reload_success = config_loader.reload_config()
        
        if reload_success:
            logger.info("Memory configuration reloaded successfully")
            return {
                "message": "Memory configuration reloaded successfully",
                "config_path": config_loader.config_path,
                "status": "success"
            }
        else:
            return JSONResponse(
                status_code=500, content={"error": "Failed to reload memory configuration"}
            )
        
    except Exception as e:
        logger.error(f"Error reloading memory config: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to reload memory config: {str(e)}"}
        )


async def delete_all_user_memories(user: User):
    """Delete all memories for the current user."""
    try:
        from advanced_omi_backend.memory import get_memory_service
        
        memory_service = get_memory_service()
        
        # Delete all memories for the user
        deleted_count = await memory_service.delete_all_user_memories(user.user_id)
        
        logger.info(f"Deleted {deleted_count} memories for user {user.user_id}")
        
        return {
            "message": f"Successfully deleted {deleted_count} memories",
            "deleted_count": deleted_count,
            "user_id": user.user_id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error deleting all memories for user {user.user_id}: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to delete memories: {str(e)}"}
        )


