"""
System controller for handling system-related business logic.
"""

import io
import logging
import os
import shutil
import time
import wave
from datetime import UTC, datetime

from advanced_omi_backend.client_manager import get_client_manager
from advanced_omi_backend.config import (
    load_diarization_settings_from_file,
    save_diarization_settings_to_file,
)
from advanced_omi_backend.job_tracker import get_job_tracker
from advanced_omi_backend.processors import get_processor_manager
from advanced_omi_backend.task_manager import get_task_manager
from advanced_omi_backend.users import User
from fastapi import BackgroundTasks, UploadFile
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


# Legacy controller methods removed - unified pipeline uses job-based tracking


async def get_processor_status():
    """Get processor queue status and health."""
    try:
        processor_manager = get_processor_manager()

        # Get queue sizes
        status = {
            "queues": {
                "audio_queue": processor_manager.audio_queue.qsize(),
                "transcription_queue": processor_manager.transcription_queue.qsize(),
                "memory_queue": processor_manager.memory_queue.qsize(),
                "cropping_queue": processor_manager.cropping_queue.qsize(),
            },
            "processors": {
                "audio_processor": "running",
                "transcription_processor": "running",
                "memory_processor": "running",
                "cropping_processor": "running",
            },
            "active_clients": len(processor_manager.active_file_sinks),
            "active_audio_uuids": len(processor_manager.active_audio_uuids),
            "processing_tasks": len(processor_manager.processing_tasks),
            "timestamp": int(time.time()),
        }

        # Get pipeline tracker status with enhanced metrics
        try:
            pipeline_tracker = get_task_manager()  # Uses backward compatibility alias
            if pipeline_tracker:
                pipeline_status = pipeline_tracker.get_health_status()
                status["pipeline_tracker"] = pipeline_status

                # Add pipeline-specific metrics
                status["pipeline_health"] = {
                    stage: {
                        "queue_depth": metrics.current_depth,
                        "avg_queue_time_ms": metrics.avg_queue_time_ms,
                        "avg_processing_time_ms": metrics.avg_processing_time_ms,
                        "total_processed": metrics.total_completed,
                        "total_failed": metrics.total_failed,
                        "status": "healthy" if metrics.avg_queue_time_ms < 5000 else "degraded"
                    }
                    for stage, metrics in pipeline_tracker.queue_metrics.items()
                }

                # Add bottleneck analysis
                bottleneck_analysis = pipeline_tracker.get_bottleneck_analysis()
                status["bottlenecks"] = bottleneck_analysis["bottlenecks"]
                status["overall_pipeline_health"] = bottleneck_analysis["overall_health"]

        except Exception as e:
            status["pipeline_tracker"] = {"error": str(e)}
            status["pipeline_health"] = {"error": str(e)}

        return status

    except Exception as e:
        logger.error(f"Error getting processor status: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get processor status: {str(e)}"}
        )


def get_audio_duration(file_content: bytes) -> float:
    """Get duration of WAV file in seconds using wave library."""
    try:
        with wave.open(io.BytesIO(file_content), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / sample_rate
            return duration
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
        return 0.0


async def process_audio_files_async(
    background_tasks: BackgroundTasks, user: User, files: list[UploadFile], device_name: str
):
    """Start async processing of uploaded audio files. Returns job ID immediately."""
    try:
        if not files:
            return JSONResponse(status_code=400, content={"error": "No files provided"})

        # Read all file contents immediately to avoid file handle issues
        file_data = []
        for file in files:
            try:
                content = await file.read()
                file_data.append((file.filename, content))
                audio_logger.info(f"📥 Read file: {file.filename} ({len(content)} bytes)")
            except Exception as e:
                audio_logger.error(f"❌ Failed to read file {file.filename}: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to read file {file.filename}: {str(e)}"},
                )

        # Use unified processing pipeline
        from advanced_omi_backend.unified_file_upload import (
            process_files_unified_background,
        )

        job_tracker = get_job_tracker()
        filenames = [filename for filename, _ in file_data]
        batch_job_id = await job_tracker.create_job(user.user_id, device_name, filenames)

        # Start background processing using unified pipeline
        background_tasks.add_task(
            process_files_unified_background,
            batch_job_id,
            file_data,
            user,
            device_name
        )

        audio_logger.info(f"🚀 Started unified async processing: batch_job_id={batch_job_id}, files={len(files)}")

        return {
            "job_id": batch_job_id,
            "message": f"Started processing {len(files)} files using unified pipeline",
            "status_url": f"/api/process-audio-files/jobs/{batch_job_id}",
            "total_files": len(files),
            "pipeline_type": "unified"
        }

    except Exception as e:
        audio_logger.error(f"Error starting async file processing: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to start processing: {str(e)}"}
        )


async def get_processing_job_status(job_id: str):
    """Get status of an async file processing job."""
    try:
        job_tracker = get_job_tracker()
        job = await job_tracker.get_job(job_id)

        if not job:
            return JSONResponse(status_code=404, content={"error": "Job not found"})

        return job.to_dict()

    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get job status: {str(e)}"}
        )


async def list_processing_jobs():
    """List all active processing jobs."""
    try:
        job_tracker = get_job_tracker()
        active_jobs = await job_tracker.get_active_jobs()

        return {"active_jobs": len(active_jobs), "jobs": [job.to_dict() for job in active_jobs]}

    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to list jobs: {str(e)}"})


async def list_all_jobs():
    """List all jobs from MongoDB (including completed/failed)."""
    try:
        job_tracker = get_job_tracker()

        # Get all jobs from MongoDB
        if job_tracker.jobs_col is None:
            return {"error": "Jobs collection not available", "jobs": []}

        # Find all jobs, sorted by creation time (most recent first)
        cursor = job_tracker.jobs_col.find({}).sort("created_at", -1).limit(100)

        jobs = []
        async for doc in cursor:
            try:
                from advanced_omi_backend.job_tracker import ProcessingJob
                job = ProcessingJob.from_mongo_dict(doc)
                jobs.append(job.to_dict())
            except Exception as e:
                logger.error(f"Failed to deserialize job {doc.get('job_id')}: {e}")

        return {
            "total_jobs": len(jobs),
            "jobs": jobs
        }

    except Exception as e:
        logger.error(f"Error listing all jobs: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to list all jobs: {str(e)}"})


# Legacy function removed - now using unified pipeline via process_files_unified_background()


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


async def get_processor_overview():
    """Get comprehensive processor overview with job-tracker-based pipeline stats."""
    try:
        processor_manager = get_processor_manager()
        task_manager = get_task_manager()
        job_tracker = get_job_tracker()
        client_manager = get_client_manager()

        # Get pipeline metrics from job tracker
        job_metrics = await job_tracker.get_pipeline_metrics()

        # Get actual queue sizes and active processing status
        # active_tasks should show if something is ACTUALLY being processed, not just if worker is alive
        # Use queue size > 0 as indicator that stage is actively processing
        pipeline_stats = {
            "audio": {
                "queue_size": processor_manager.audio_queue.qsize(),
                "active_tasks": 1 if processor_manager.audio_queue.qsize() > 0 else 0,
                "avg_processing_time_ms": job_metrics.get("stage_metrics", {}).get("audio", {}).get("avg_processing_lag_seconds", 0) * 1000,
                "success_rate": _calculate_success_rate(job_metrics.get("stage_metrics", {}).get("audio", {})),
                "throughput_per_minute": job_metrics.get("stage_metrics", {}).get("audio", {}).get("total_processed", 0) / 60
            },
            "transcription": {
                "queue_size": processor_manager.transcription_queue.qsize(),
                "active_tasks": 1 if processor_manager.transcription_queue.qsize() > 0 else 0,
                "avg_processing_time_ms": job_metrics.get("stage_metrics", {}).get("transcription", {}).get("avg_processing_lag_seconds", 0) * 1000,
                "success_rate": _calculate_success_rate(job_metrics.get("stage_metrics", {}).get("transcription", {})),
                "throughput_per_minute": job_metrics.get("stage_metrics", {}).get("transcription", {}).get("total_processed", 0) / 60
            },
            "memory": {
                "queue_size": processor_manager.memory_queue.qsize(),
                "active_tasks": 1 if processor_manager.memory_queue.qsize() > 0 else 0,
                "avg_processing_time_ms": job_metrics.get("stage_metrics", {}).get("memory", {}).get("avg_processing_lag_seconds", 0) * 1000,
                "success_rate": _calculate_success_rate(job_metrics.get("stage_metrics", {}).get("memory", {})),
                "throughput_per_minute": job_metrics.get("stage_metrics", {}).get("memory", {}).get("total_processed", 0) / 60
            },
            "cropping": {
                "queue_size": processor_manager.cropping_queue.qsize(),
                "active_tasks": 1 if processor_manager.cropping_queue.qsize() > 0 else 0,
                "avg_processing_time_ms": job_metrics.get("stage_metrics", {}).get("cropping", {}).get("avg_processing_lag_seconds", 0) * 1000,
                "success_rate": _calculate_success_rate(job_metrics.get("stage_metrics", {}).get("cropping", {})),
                "throughput_per_minute": job_metrics.get("stage_metrics", {}).get("cropping", {}).get("total_processed", 0) / 60
            }
        }

        # Get system health metrics
        task_health = task_manager.get_health_status()
        queue_health = processor_manager.get_queue_health_status()

        # Get recent activity
        recent_activity = processor_manager.get_processing_history(limit=10)

        # Calculate uptime from process start (approximation using task manager start time)
        process_start_time = task_health.get("start_time", time.time())
        uptime_seconds = time.time() - process_start_time
        uptime_hours = uptime_seconds / 3600

        overview = {
            "pipeline_stats": pipeline_stats,
            "job_tracker_metrics": job_metrics,  # Include raw job tracker data
            "system_health": {
                "total_active_clients": len(client_manager._active_clients),
                "total_processing_tasks": job_metrics.get("active_pipeline_jobs", 0) + job_metrics.get("active_batch_jobs", 0),
                "task_manager_healthy": task_health.get("healthy", False),
                "error_rate": task_health.get("recent_errors", 0) / max(task_health.get("completed_tasks", 1), 1),
                "uptime_hours": uptime_hours
            },
            "queue_health": queue_health,
            "recent_activity": recent_activity[:5]  # Last 5 activities
        }

        return overview
    except Exception as e:
        logger.error(f"Error getting processor overview: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get processor overview: {str(e)}"}
        )


def _calculate_success_rate(stage_data: dict) -> float:
    """Helper to calculate success rate from stage metrics."""
    if not stage_data:
        return 1.0
    total_processed = stage_data.get("total_processed", 0)
    total_failed = stage_data.get("total_failed", 0)
    total = total_processed + total_failed
    if total == 0:
        return 1.0
    return total_processed / total

async def get_processor_history(page: int = 1, per_page: int = 50):
    """Get paginated processing history."""
    try:
        processor_manager = get_processor_manager()

        # Calculate offset
        offset = (page - 1) * per_page

        # Get full history and paginate
        full_history = processor_manager.get_processing_history(limit=1000)  # Get more for pagination
        total_items = len(full_history)

        # Paginate
        paginated_history = full_history[offset:offset + per_page]

        return {
            "history": paginated_history,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total_items,
                "total_pages": (total_items + per_page - 1) // per_page
            }
        }
    except Exception as e:
        logger.error(f"Error getting processor history: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get processor history: {str(e)}"}
        )

async def get_client_processing_detail(client_id: str):
    """Get detailed processing information for specific client."""
    try:
        from advanced_omi_backend.client_manager import get_client_manager

        processor_manager = get_processor_manager()
        client_manager = get_client_manager()

        # Get task manager tasks for this client
        task_manager = get_task_manager()
        client_tasks = task_manager.get_tasks_for_client(client_id)

        # Try to get client info, but don't fail if client is inactive
        client = client_manager.get_client(client_id)

        # If no client and no task data, return 404
        if not client and not client_tasks:
            return JSONResponse(
                status_code=404, content={"error": f"No data found for client {client_id}"}
            )

        detail = {
            "client_id": client_id,
            "client_info": {
                "user_id": getattr(client, "user_id", "unknown") if client else "unknown",
                "user_email": getattr(client, "user_email", "unknown") if client else "unknown",
                "current_audio_uuid": getattr(client, "current_audio_uuid", None) if client else None,
                "conversation_start_time": getattr(client, "conversation_start_time", None) if client else None,
                "sample_rate": getattr(client, "sample_rate", None) if client else None,
                "status": "active" if client else "inactive"
            },
            "active_tasks": [
                {
                    "task_id": f"{task.name}_{id(task.task)}",
                    "task_name": task.name,
                    "task_type": task.metadata.get("type", "unknown"),
                    "created_at": datetime.fromtimestamp(task.created_at, UTC).isoformat(),
                    "completed_at": datetime.fromtimestamp(task.completed_at, UTC).isoformat() if task.completed_at else None,
                    "error": task.error,
                    "cancelled": task.cancelled
                }
                for task in client_tasks
            ]
        }

        return detail
    except Exception as e:
        logger.error(f"Error getting client processing detail for {client_id}: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get client detail: {str(e)}"}
        )


# New Pipeline-Specific Endpoints

async def get_pipeline_bottlenecks():
    """Get pipeline bottleneck analysis with recommendations."""
    try:
        from advanced_omi_backend.task_manager import get_task_manager

        pipeline_tracker = get_task_manager()
        bottleneck_analysis = pipeline_tracker.get_bottleneck_analysis()

        return {
            "analysis_timestamp": int(time.time()),
            "bottlenecks": [
                {
                    **bottleneck,
                    "recommendation": _get_bottleneck_recommendation(bottleneck)
                }
                for bottleneck in bottleneck_analysis["bottlenecks"]
            ],
            "slowest_stage": bottleneck_analysis.get("slowest_stage"),
            "slowest_stage_total_time_ms": bottleneck_analysis.get("slowest_stage_total_time_ms", 0),
            "overall_pipeline_health": bottleneck_analysis["overall_health"],
            "healthy_stages": [
                stage for stage, metrics in pipeline_tracker.queue_metrics.items()
                if metrics.avg_queue_time_ms < 5000 and metrics.avg_processing_time_ms < 10000
            ]
        }
    except Exception as e:
        logger.error(f"Error getting pipeline bottlenecks: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get bottlenecks: {str(e)}"}
        )


async def get_pipeline_health():
    """Get comprehensive pipeline health metrics."""
    try:
        from advanced_omi_backend.task_manager import get_task_manager

        pipeline_tracker = get_task_manager()
        processor_manager = get_processor_manager()

        # Calculate end-to-end metrics
        active_sessions = len(pipeline_tracker.audio_sessions)
        completed_today = sum(
            metrics.total_completed for metrics in pipeline_tracker.queue_metrics.values()
        )

        # Calculate average end-to-end time (estimated)
        stage_times = [
            metrics.avg_processing_time_ms + metrics.avg_queue_time_ms
            for metrics in pipeline_tracker.queue_metrics.values()
            if metrics.avg_processing_time_ms > 0
        ]
        avg_end_to_end_time = sum(stage_times) if stage_times else 0

        return {
            "overall_status": pipeline_tracker.get_bottleneck_analysis()["overall_health"],
            "active_sessions": active_sessions,
            "completed_today": completed_today,
            "average_end_to_end_time_ms": avg_end_to_end_time,
            "stage_performance": {
                stage: {
                    "avg_time_ms": metrics.avg_processing_time_ms + metrics.avg_queue_time_ms,
                    "success_rate": (
                        metrics.total_completed / (metrics.total_completed + metrics.total_failed)
                        if (metrics.total_completed + metrics.total_failed) > 0 else 1.0
                    ) * 100,
                    "status": _get_stage_health_status(metrics)
                }
                for stage, metrics in pipeline_tracker.queue_metrics.items()
            },
            "trends": {
                "throughput_trend": "stable",  # Could be calculated from historical data
                "latency_trend": "stable",     # Could be calculated from historical data
                "error_rate_trend": "stable"  # Could be calculated from historical data
            }
        }
    except Exception as e:
        logger.error(f"Error getting pipeline health: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get pipeline health: {str(e)}"}
        )


async def get_queue_metrics():
    """Get real-time queue metrics and performance data."""
    try:
        from advanced_omi_backend.task_manager import get_task_manager

        pipeline_tracker = get_task_manager()
        processor_manager = get_processor_manager()

        return {
            "timestamp": int(time.time()),
            "queues": {
                stage: {
                    "current_depth": metrics.current_depth,
                    "total_enqueued": metrics.total_enqueued,
                    "total_dequeued": metrics.total_dequeued,
                    "total_completed": metrics.total_completed,
                    "total_failed": metrics.total_failed,
                    "avg_queue_time_ms": metrics.avg_queue_time_ms,
                    "avg_processing_time_ms": metrics.avg_processing_time_ms,
                    "health_status": _get_stage_health_status(metrics),
                    "last_updated": int(metrics.last_updated)
                }
                for stage, metrics in pipeline_tracker.queue_metrics.items()
            }
        }
    except Exception as e:
        logger.error(f"Error getting queue metrics: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get queue metrics: {str(e)}"}
        )


async def get_session_pipeline(audio_uuid: str):
    """Get detailed pipeline timeline for a specific audio session."""
    try:
        from advanced_omi_backend.task_manager import get_task_manager

        pipeline_tracker = get_task_manager()
        events = pipeline_tracker.get_pipeline_events(audio_uuid)

        if not events:
            return JSONResponse(
                status_code=404, content={"error": f"No pipeline events found for audio UUID: {audio_uuid}"}
            )

        # Calculate stage status
        stages = {}
        for event in events:
            stage = event.stage
            if stage not in stages:
                stages[stage] = {"status": "pending", "events": []}

            stages[stage]["events"].append({
                "event_type": event.event_type,
                "timestamp": int(event.timestamp),
                "queue_size": event.queue_size,
                "processing_time_ms": event.processing_time_ms,
                "metadata": event.metadata
            })

            if event.event_type == "complete":
                stages[stage]["status"] = "completed"
                stages[stage]["processing_time_ms"] = event.processing_time_ms
            elif event.event_type == "failed":
                stages[stage]["status"] = "failed"
                stages[stage]["error"] = event.metadata.get("error", "Unknown error")
            elif event.event_type == "dequeue" and stages[stage]["status"] == "pending":
                stages[stage]["status"] = "in_progress"

        return {
            "audio_uuid": audio_uuid,
            "conversation_id": events[0].conversation_id if events else None,
            "status": "completed" if all(s.get("status") == "completed" for s in stages.values()) else "processing",
            "created_at": int(events[0].timestamp) if events else None,
            "stages": stages,
            "timeline": [
                {
                    "timestamp": int(event.timestamp),
                    "event": event.event_type,
                    "stage": event.stage,
                    "queue_size": event.queue_size,
                    "processing_time_ms": event.processing_time_ms
                }
                for event in events
            ]
        }
    except Exception as e:
        logger.error(f"Error getting session pipeline for {audio_uuid}: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get session pipeline: {str(e)}"}
        )


# Helper functions for pipeline analysis

def _get_bottleneck_recommendation(bottleneck: dict) -> str:
    """Generate recommendations for pipeline bottlenecks."""
    stage = bottleneck.get("stage", "")
    bottleneck_type = bottleneck.get("type", "")

    if bottleneck_type == "queue_lag":
        if stage == "memory":
            return "Consider scaling LLM processing or increasing memory timeout"
        elif stage == "transcription":
            return "Consider additional transcription workers or check Deepgram quota"
        elif stage == "audio":
            return "Check audio processing performance and file I/O"
        elif stage == "cropping":
            return "Audio cropping backlog - consider parallel processing"
        else:
            return f"Queue lag detected in {stage} - consider scaling resources"
    elif bottleneck_type == "processing_lag":
        if stage == "memory":
            return "Memory extraction taking too long - check LLM performance"
        elif stage == "transcription":
            return "Transcription processing slow - check provider performance"
        else:
            return f"Processing lag in {stage} - optimize or scale processing"
    else:
        return f"Performance issue detected in {stage} stage"


def _get_stage_health_status(metrics) -> str:
    """Determine health status for a pipeline stage."""
    if metrics.avg_queue_time_ms > 15000:  # 15+ second queue time
        return "critical"
    elif metrics.avg_queue_time_ms > 5000:  # 5+ second queue time
        return "degraded"
    elif metrics.avg_processing_time_ms > 30000:  # 30+ second processing
        return "degraded"
    elif metrics.total_failed > 0 and metrics.total_completed == 0:
        return "failing"
    else:
        return "healthy"



