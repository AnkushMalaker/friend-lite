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
# TODO: Remove old processor architecture
# from advanced_omi_backend.processors import get_processor_manager
def get_processor_manager():
    """Stub - processors being removed."""
    return None
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
    """Get all active processing tasks."""
    try:
        processor_manager = get_processor_manager()
        return processor_manager.get_all_processing_status()
    except Exception as e:
        logger.error(f"Error getting processing tasks: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get processing tasks: {str(e)}"}
        )


async def get_processing_task_status(client_id: str):
    """Get processing task status for a specific client."""
    try:
        processor_manager = get_processor_manager()
        processing_status = processor_manager.get_processing_status(client_id)

        # Check if transcription is marked as started but not completed, and verify with database
        stages = processing_status.get("stages", {})
        transcription_stage = stages.get("transcription", {})

        """This is a hack to update it the DB INCASE a process failed
        if transcription_stage.get("status") == "started" and not transcription_stage.get("completed", False):
            # Check if transcription is actually complete by checking the database
            try:
                chunk = await chunks_col.find_one({"client_id": client_id})
                if chunk and chunk.get("transcript") and len(chunk.get("transcript", [])) > 0:
                    # Transcription is complete! Update the processor state
                    processor_manager.track_processing_stage(
                        client_id,
                        "transcription",
                        "completed",
                        {"audio_uuid": chunk.get("audio_uuid"), "segments": len(chunk.get("transcript", []))}
                    )
                    logger.info(f"Detected transcription completion for client {client_id} ({len(chunk.get('transcript', []))} segments)")
                    # Get updated status
                    processing_status = processor_manager.get_processing_status(client_id)
            except Exception as e:
                logger.debug(f"Error checking transcription completion: {e}")
        """
        return processing_status
    except Exception as e:
        logger.error(f"Error getting processing task status for {client_id}: {e}")
        return JSONResponse(
            status_code=500, content={"error": f"Failed to get processing task status: {str(e)}"}
        )


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


async def get_streaming_status(request):
    """Get status of active streaming sessions and Redis Streams health."""
    import time
    from advanced_omi_backend.controllers.queue_controller import (
        transcription_queue,
        memory_queue,
        default_queue,
        all_jobs_complete_for_session
    )

    try:
        # Get Redis client from request.app.state (initialized during startup)
        redis_client = request.app.state.redis_audio_stream

        if not redis_client:
            return JSONResponse(
                status_code=503,
                content={"error": "Redis client for audio streaming not initialized"}
            )

        # Get all sessions (both active and completed)
        session_keys = await redis_client.keys("audio:session:*")
        active_sessions = []
        completed_sessions_from_redis = []

        for key in session_keys:
            session_data = await redis_client.hgetall(key)
            if not session_data:
                continue

            session_id = key.decode().split(":")[-1]
            started_at = float(session_data.get(b"started_at", b"0"))
            last_chunk_at = float(session_data.get(b"last_chunk_at", b"0"))
            status = session_data.get(b"status", b"").decode()

            session_obj = {
                "session_id": session_id,
                "user_id": session_data.get(b"user_id", b"").decode(),
                "client_id": session_data.get(b"client_id", b"").decode(),
                "provider": session_data.get(b"provider", b"").decode(),
                "mode": session_data.get(b"mode", b"").decode(),
                "status": status,
                "chunks_published": int(session_data.get(b"chunks_published", b"0")),
                "started_at": started_at,
                "last_chunk_at": last_chunk_at,
                "age_seconds": time.time() - started_at,
                "idle_seconds": time.time() - last_chunk_at
            }

            # Separate active and completed sessions
            # Check if all jobs are complete (including failed jobs)
            all_jobs_done = all_jobs_complete_for_session(session_id)

            # Session is completed if:
            # 1. Redis status says complete/finalized AND all jobs done, OR
            # 2. All jobs are done (even if status isn't complete yet)
            # This ensures sessions with failed jobs move to completed
            if status in ["complete", "completed", "finalized"] or all_jobs_done:
                if all_jobs_done:
                    # All jobs complete - this is truly a completed session
                    # Update Redis status if it wasn't already marked complete
                    if status not in ["complete", "completed", "finalized"]:
                        await redis_client.hset(key, "status", "complete")
                        logger.info(f"✅ Marked session {session_id} as complete (all jobs terminal)")

                    completed_sessions_from_redis.append({
                        "session_id": session_id,
                        "client_id": session_data.get(b"client_id", b"").decode(),
                        "conversation_id": session_data.get(b"conversation_id", b"").decode() if b"conversation_id" in session_data else None,
                        "has_conversation": bool(session_data.get(b"conversation_id", b"")),
                        "action": session_data.get(b"action", b"complete").decode(),
                        "reason": session_data.get(b"reason", b"").decode() if b"reason" in session_data else "",
                        "completed_at": last_chunk_at,
                        "audio_file": session_data.get(b"audio_file", b"").decode() if b"audio_file" in session_data else ""
                    })
                else:
                    # Status says complete but jobs still processing - keep in active
                    active_sessions.append(session_obj)
            else:
                # This is an active session
                active_sessions.append(session_obj)

        # Get stream health for all streams (per-client streams)
        # Categorize as active or completed based on consumer activity
        active_streams = {}
        completed_streams = {}

        # Discover all audio streams
        stream_keys = await redis_client.keys("audio:stream:*")
        current_time = time.time()

        for stream_key in stream_keys:
            stream_name = stream_key.decode() if isinstance(stream_key, bytes) else stream_key
            try:
                # Check if stream exists
                stream_info = await redis_client.execute_command('XINFO', 'STREAM', stream_name)

                # Parse stream info (returns flat list of key-value pairs)
                info_dict = {}
                for i in range(0, len(stream_info), 2):
                    key = stream_info[i].decode() if isinstance(stream_info[i], bytes) else str(stream_info[i])
                    value = stream_info[i+1]

                    # Skip complex binary structures like first-entry and last-entry
                    # which contain message data that can't be JSON serialized
                    if key in ["first-entry", "last-entry"]:
                        # Just extract the message ID (first element)
                        if isinstance(value, list) and len(value) > 0:
                            msg_id = value[0]
                            if isinstance(msg_id, bytes):
                                msg_id = msg_id.decode()
                            value = msg_id
                        else:
                            value = None
                    elif isinstance(value, bytes):
                        try:
                            value = value.decode()
                        except UnicodeDecodeError:
                            # Binary data that can't be decoded, skip it
                            value = "<binary>"

                    info_dict[key] = value

                # Calculate stream age from last entry
                stream_age_seconds = 0
                last_entry_id = info_dict.get("last-entry")
                if last_entry_id:
                    try:
                        # Redis Stream IDs format: "milliseconds-sequence"
                        last_timestamp_ms = int(last_entry_id.split('-')[0])
                        last_timestamp_s = last_timestamp_ms / 1000
                        stream_age_seconds = current_time - last_timestamp_s
                    except (ValueError, IndexError, AttributeError):
                        stream_age_seconds = 0

                # Get consumer groups
                groups = await redis_client.execute_command('XINFO', 'GROUPS', stream_name)

                stream_data = {
                    "stream_length": info_dict.get("length", 0),
                    "first_entry_id": info_dict.get("first-entry"),
                    "last_entry_id": last_entry_id,
                    "stream_age_seconds": stream_age_seconds,
                    "consumer_groups": [],
                    "total_pending": 0
                }

                # Track if stream has any active consumers
                has_active_consumer = False
                min_consumer_idle_ms = float('inf')

                # Parse consumer groups
                for group in groups:
                    group_dict = {}
                    for i in range(0, len(group), 2):
                        key = group[i].decode() if isinstance(group[i], bytes) else str(group[i])
                        value = group[i+1]
                        if isinstance(value, bytes):
                            try:
                                value = value.decode()
                            except UnicodeDecodeError:
                                value = "<binary>"
                        group_dict[key] = value

                    group_name = group_dict.get("name", "unknown")
                    if isinstance(group_name, bytes):
                        group_name = group_name.decode()

                    # Get consumers for this group
                    consumers = await redis_client.execute_command('XINFO', 'CONSUMERS', stream_name, group_name)
                    consumer_list = []
                    consumer_pending_total = 0

                    for consumer in consumers:
                        consumer_dict = {}
                        for i in range(0, len(consumer), 2):
                            key = consumer[i].decode() if isinstance(consumer[i], bytes) else str(consumer[i])
                            value = consumer[i+1]
                            if isinstance(value, bytes):
                                try:
                                    value = value.decode()
                                except UnicodeDecodeError:
                                    value = "<binary>"
                            consumer_dict[key] = value

                        consumer_name = consumer_dict.get("name", "unknown")
                        if isinstance(consumer_name, bytes):
                            consumer_name = consumer_name.decode()

                        consumer_pending = int(consumer_dict.get("pending", 0))
                        consumer_idle_ms = int(consumer_dict.get("idle", 0))
                        consumer_pending_total += consumer_pending

                        # Track minimum idle time
                        min_consumer_idle_ms = min(min_consumer_idle_ms, consumer_idle_ms)

                        # Consumer is active if idle < 5 minutes (300000ms)
                        if consumer_idle_ms < 300000:
                            has_active_consumer = True

                        consumer_list.append({
                            "name": consumer_name,
                            "pending": consumer_pending,
                            "idle_ms": consumer_idle_ms
                        })

                    # Get group-level pending count (may be 0 even if consumers have pending)
                    try:
                        pending = await redis_client.xpending(stream_name, group_name)
                        group_pending_count = int(pending[0]) if pending else 0
                    except Exception:
                        group_pending_count = 0

                    # Use the maximum of group-level pending or sum of consumer pending
                    # (Sometimes group pending is 0 but consumers still have pending messages)
                    effective_pending = max(group_pending_count, consumer_pending_total)

                    stream_data["consumer_groups"].append({
                        "name": str(group_name),
                        "consumers": consumer_list,
                        "pending": int(effective_pending)
                    })

                    stream_data["total_pending"] += int(effective_pending)

                # Determine if stream is active or completed
                # Active: has active consumers OR pending messages OR recent activity (< 5 min)
                # Completed: no active consumers and idle > 5 minutes but < 1 hour
                is_active = (
                    has_active_consumer or
                    stream_data["total_pending"] > 0 or
                    stream_age_seconds < 300  # Less than 5 minutes old
                )

                if is_active:
                    active_streams[stream_name] = stream_data
                else:
                    # Mark as completed (will be cleaned up when > 1 hour old)
                    stream_data["idle_seconds"] = stream_age_seconds
                    completed_streams[stream_name] = stream_data

            except Exception as e:
                # Stream doesn't exist or error getting info
                logger.debug(f"Error processing stream {stream_name}: {e}")
                continue

        # Get RQ queue stats - include all registries
        rq_stats = {
            "transcription_queue": {
                "queued": transcription_queue.count,
                "processing": len(transcription_queue.started_job_registry),
                "completed": len(transcription_queue.finished_job_registry),
                "failed": len(transcription_queue.failed_job_registry),
                "cancelled": len(transcription_queue.canceled_job_registry),
                "deferred": len(transcription_queue.deferred_job_registry)
            },
            "memory_queue": {
                "queued": memory_queue.count,
                "processing": len(memory_queue.started_job_registry),
                "completed": len(memory_queue.finished_job_registry),
                "failed": len(memory_queue.failed_job_registry),
                "cancelled": len(memory_queue.canceled_job_registry),
                "deferred": len(memory_queue.deferred_job_registry)
            },
            "default_queue": {
                "queued": default_queue.count,
                "processing": len(default_queue.started_job_registry),
                "completed": len(default_queue.finished_job_registry),
                "failed": len(default_queue.failed_job_registry),
                "cancelled": len(default_queue.canceled_job_registry),
                "deferred": len(default_queue.deferred_job_registry)
            }
        }

        return {
            "active_sessions": active_sessions,
            "completed_sessions": completed_sessions_from_redis,
            "active_streams": active_streams,
            "completed_streams": completed_streams,
            "stream_health": active_streams,  # Backward compatibility - use active_streams
            "rq_queues": rq_stats,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error getting streaming status: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get streaming status: {str(e)}"}
        )


async def cleanup_stuck_stream_workers(request):
    """Clean up stuck Redis Stream consumers and pending messages from all active streams."""
    import time

    try:
        # Get Redis client from request.app.state (initialized during startup)
        redis_client = request.app.state.redis_audio_stream

        if not redis_client:
            return JSONResponse(
                status_code=503,
                content={"error": "Redis client for audio streaming not initialized"}
            )

        cleanup_results = {}
        total_cleaned = 0
        total_deleted_consumers = 0
        total_deleted_streams = 0
        current_time = time.time()

        # Discover all audio streams (per-client streams)
        stream_keys = await redis_client.keys("audio:stream:*")

        for stream_key in stream_keys:
            stream_name = stream_key.decode() if isinstance(stream_key, bytes) else stream_key

            try:
                # First check stream age - delete old streams (>1 hour) immediately
                stream_info = await redis_client.execute_command('XINFO', 'STREAM', stream_name)

                # Parse stream info
                info_dict = {}
                for i in range(0, len(stream_info), 2):
                    key_name = stream_info[i].decode() if isinstance(stream_info[i], bytes) else str(stream_info[i])
                    info_dict[key_name] = stream_info[i+1]

                stream_length = int(info_dict.get("length", 0))
                last_entry = info_dict.get("last-entry")

                # Check if stream is old
                should_delete_stream = False
                stream_age = 0

                if stream_length == 0:
                    should_delete_stream = True
                    stream_age = 0
                elif last_entry and isinstance(last_entry, list) and len(last_entry) > 0:
                    try:
                        last_id = last_entry[0]
                        if isinstance(last_id, bytes):
                            last_id = last_id.decode()
                        last_timestamp_ms = int(last_id.split('-')[0])
                        last_timestamp_s = last_timestamp_ms / 1000
                        stream_age = current_time - last_timestamp_s

                        # Delete streams older than 1 hour (3600 seconds)
                        if stream_age > 3600:
                            should_delete_stream = True
                    except (ValueError, IndexError):
                        pass

                if should_delete_stream:
                    await redis_client.delete(stream_name)
                    total_deleted_streams += 1
                    cleanup_results[stream_name] = {
                        "message": f"Deleted old stream (age: {stream_age:.0f}s, length: {stream_length})",
                        "cleaned": 0,
                        "deleted_consumers": 0,
                        "deleted_stream": True,
                        "stream_age": stream_age
                    }
                    continue

                # Get consumer groups
                groups = await redis_client.execute_command('XINFO', 'GROUPS', stream_name)

                if not groups:
                    cleanup_results[stream_name] = {"message": "No consumer groups found", "cleaned": 0, "deleted_stream": False}
                    continue

                # Parse first group
                group_dict = {}
                group = groups[0]
                for i in range(0, len(group), 2):
                    key = group[i].decode() if isinstance(group[i], bytes) else str(group[i])
                    value = group[i+1]
                    if isinstance(value, bytes):
                        try:
                            value = value.decode()
                        except UnicodeDecodeError:
                            value = str(value)
                    group_dict[key] = value

                group_name = group_dict.get("name", "unknown")
                if isinstance(group_name, bytes):
                    group_name = group_name.decode()

                pending_count = int(group_dict.get("pending", 0))

                # Get consumers for this group to check per-consumer pending
                consumers = await redis_client.execute_command('XINFO', 'CONSUMERS', stream_name, group_name)

                cleaned_count = 0
                total_consumer_pending = 0

                # Clean up pending messages for each consumer AND delete dead consumers
                deleted_consumers = 0
                for consumer in consumers:
                    consumer_dict = {}
                    for i in range(0, len(consumer), 2):
                        key = consumer[i].decode() if isinstance(consumer[i], bytes) else str(consumer[i])
                        value = consumer[i+1]
                        if isinstance(value, bytes):
                            try:
                                value = value.decode()
                            except UnicodeDecodeError:
                                value = str(value)
                        consumer_dict[key] = value

                    consumer_name = consumer_dict.get("name", "unknown")
                    if isinstance(consumer_name, bytes):
                        consumer_name = consumer_name.decode()

                    consumer_pending = int(consumer_dict.get("pending", 0))
                    consumer_idle_ms = int(consumer_dict.get("idle", 0))
                    total_consumer_pending += consumer_pending

                    # Check if consumer is dead (idle > 5 minutes = 300000ms)
                    is_dead = consumer_idle_ms > 300000

                    if consumer_pending > 0:
                        logger.info(f"Found {consumer_pending} pending messages for consumer {consumer_name} (idle: {consumer_idle_ms}ms)")

                        # Get pending messages for this specific consumer
                        try:
                            pending_messages = await redis_client.execute_command(
                                'XPENDING', stream_name, group_name, '-', '+', str(consumer_pending), consumer_name
                            )

                            # XPENDING returns flat list: [msg_id, consumer, idle_ms, delivery_count, msg_id, ...]
                            # Parse in groups of 4
                            for i in range(0, len(pending_messages), 4):
                                if i < len(pending_messages):
                                    msg_id = pending_messages[i]
                                    if isinstance(msg_id, bytes):
                                        msg_id = msg_id.decode()

                                    # Claim the message to a cleanup worker
                                    try:
                                        await redis_client.execute_command(
                                            'XCLAIM', stream_name, group_name, 'cleanup-worker', '0', msg_id
                                        )

                                        # Acknowledge it immediately
                                        await redis_client.xack(stream_name, group_name, msg_id)
                                        cleaned_count += 1
                                    except Exception as claim_error:
                                        logger.warning(f"Failed to claim/ack message {msg_id}: {claim_error}")

                        except Exception as consumer_error:
                            logger.error(f"Error processing consumer {consumer_name}: {consumer_error}")

                    # Delete dead consumers (idle > 5 minutes with no pending messages)
                    if is_dead and consumer_pending == 0:
                        try:
                            await redis_client.execute_command(
                                'XGROUP', 'DELCONSUMER', stream_name, group_name, consumer_name
                            )
                            deleted_consumers += 1
                            logger.info(f"🧹 Deleted dead consumer {consumer_name} (idle: {consumer_idle_ms}ms)")
                        except Exception as delete_error:
                            logger.warning(f"Failed to delete consumer {consumer_name}: {delete_error}")

                if total_consumer_pending == 0 and deleted_consumers == 0:
                    cleanup_results[stream_name] = {"message": "No pending messages or dead consumers", "cleaned": 0, "deleted_consumers": 0, "deleted_stream": False}
                    continue

                total_cleaned += cleaned_count
                total_deleted_consumers += deleted_consumers
                cleanup_results[stream_name] = {
                    "message": f"Cleaned {cleaned_count} pending messages, deleted {deleted_consumers} dead consumers",
                    "cleaned": cleaned_count,
                    "deleted_consumers": deleted_consumers,
                    "deleted_stream": False,
                    "original_pending": pending_count
                }

            except Exception as e:
                cleanup_results[stream_name] = {
                    "error": str(e),
                    "cleaned": 0
                }

        return {
            "success": True,
            "total_cleaned": total_cleaned,
            "total_deleted_consumers": total_deleted_consumers,
            "total_deleted_streams": total_deleted_streams,
            "streams": cleanup_results,  # New key for per-stream results
            "providers": cleanup_results,  # Keep for backward compatibility with frontend
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error cleaning up stuck workers: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": f"Failed to cleanup stuck workers: {str(e)}"}
        )


async def cleanup_old_sessions(request, max_age_seconds: int = 3600):
    """Clean up old session tracking metadata and old audio streams from Redis."""
    import time

    try:
        # Get Redis client from request.app.state (initialized during startup)
        redis_client = request.app.state.redis_audio_stream

        if not redis_client:
            return JSONResponse(
                status_code=503,
                content={"error": "Redis client for audio streaming not initialized"}
            )

        # Get all session keys
        session_keys = await redis_client.keys("audio:session:*")
        cleaned_sessions = 0
        old_sessions = []

        current_time = time.time()

        for key in session_keys:
            session_data = await redis_client.hgetall(key)
            if not session_data:
                continue

            session_id = key.decode().split(":")[-1]
            started_at = float(session_data.get(b"started_at", b"0"))
            status = session_data.get(b"status", b"").decode()

            age_seconds = current_time - started_at

            # Clean up sessions older than max_age or stuck in "finalizing"
            should_clean = (
                age_seconds > max_age_seconds or
                (status == "finalizing" and age_seconds > 300)  # Finalizing for more than 5 minutes
            )

            if should_clean:
                old_sessions.append({
                    "session_id": session_id,
                    "age_seconds": age_seconds,
                    "status": status
                })
                await redis_client.delete(key)
                cleaned_sessions += 1

        # Also clean up old audio streams (per-client streams that are inactive)
        stream_keys = await redis_client.keys("audio:stream:*")
        cleaned_streams = 0
        old_streams = []

        for stream_key in stream_keys:
            stream_name = stream_key.decode() if isinstance(stream_key, bytes) else stream_key

            try:
                # Check stream info to get last activity
                stream_info = await redis_client.execute_command('XINFO', 'STREAM', stream_name)

                # Parse stream info
                info_dict = {}
                for i in range(0, len(stream_info), 2):
                    key_name = stream_info[i].decode() if isinstance(stream_info[i], bytes) else str(stream_info[i])
                    info_dict[key_name] = stream_info[i+1]

                stream_length = int(info_dict.get("length", 0))
                last_entry = info_dict.get("last-entry")

                # Check stream age via last entry ID (Redis Stream IDs are timestamps)
                should_delete = False
                age_seconds = 0

                if stream_length == 0:
                    # Empty stream - safe to delete
                    should_delete = True
                    reason = "empty"
                elif last_entry and isinstance(last_entry, list) and len(last_entry) > 0:
                    # Extract timestamp from last entry ID
                    last_id = last_entry[0]
                    if isinstance(last_id, bytes):
                        last_id = last_id.decode()

                    # Redis Stream IDs format: "milliseconds-sequence"
                    try:
                        last_timestamp_ms = int(last_id.split('-')[0])
                        last_timestamp_s = last_timestamp_ms / 1000
                        age_seconds = current_time - last_timestamp_s

                        # Delete streams older than max_age regardless of size
                        if age_seconds > max_age_seconds:
                            should_delete = True
                            reason = "old"
                    except (ValueError, IndexError):
                        # If we can't parse timestamp, check if first entry is old
                        first_entry = info_dict.get("first-entry")
                        if first_entry and isinstance(first_entry, list) and len(first_entry) > 0:
                            try:
                                first_id = first_entry[0]
                                if isinstance(first_id, bytes):
                                    first_id = first_id.decode()
                                first_timestamp_ms = int(first_id.split('-')[0])
                                first_timestamp_s = first_timestamp_ms / 1000
                                age_seconds = current_time - first_timestamp_s

                                if age_seconds > max_age_seconds:
                                    should_delete = True
                                    reason = "old_unparseable"
                            except (ValueError, IndexError):
                                pass

                if should_delete:
                    await redis_client.delete(stream_name)
                    cleaned_streams += 1
                    old_streams.append({
                        "stream_name": stream_name,
                        "reason": reason,
                        "age_seconds": age_seconds,
                        "length": stream_length
                    })

            except Exception as e:
                logger.debug(f"Error checking stream {stream_name}: {e}")
                continue

        return {
            "success": True,
            "cleaned_sessions": cleaned_sessions,
            "cleaned_streams": cleaned_streams,
            "cleaned_session_details": old_sessions,
            "cleaned_stream_details": old_streams,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error cleaning up old sessions: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": f"Failed to cleanup old sessions: {str(e)}"}
        )
