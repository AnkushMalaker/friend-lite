"""
Health check routes for Friend-Lite backend.

This module provides health check endpoints for monitoring the application's status.
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any

import aiohttp
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient

from advanced_omi_backend.controllers.queue_controller import redis_conn
from advanced_omi_backend.client_manager import get_client_manager
from advanced_omi_backend.llm_client import async_health_check
from advanced_omi_backend.memory import get_memory_service
from advanced_omi_backend.services.transcription import get_transcription_provider

# Create router
router = APIRouter(tags=["health"])

# Logging setup
logger = logging.getLogger(__name__)
application_logger = logging.getLogger("audio_processing")

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
mongo_client = AsyncIOMotorClient(MONGODB_URI)

# Memory service
memory_service = get_memory_service()

# Transcription provider
transcription_provider = get_transcription_provider()

# Qdrant Configuration
QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", "qdrant")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")


@router.get("/auth/health")
async def auth_health_check():
    """Pre-flight health check for authentication service connectivity."""
    try:
        # Test database connectivity
        await mongo_client.admin.command("ping")
        
        # Test memory service if available
        if memory_service:
            try:
                await asyncio.wait_for(memory_service.test_connection(), timeout=2.0)
                memory_status = "ok"
            except Exception as e:
                logger.warning(f"Memory service health check failed: {e}")
                memory_status = "degraded"
        else:
            memory_status = "unavailable"
        
        return {
            "status": "ok",
            "database": "ok", 
            "memory_service": memory_status,
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"Auth health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "detail": "Service connectivity check failed",
                "error_type": "connection_failure",
                "timestamp": int(time.time())
            }
        )


@router.get("/health")
async def health_check():
    """Comprehensive health check for all services."""
    health_status = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "services": {},
        "config": {
            "mongodb_uri": MONGODB_URI,
            "qdrant_url": f"http://{QDRANT_BASE_URL}:{QDRANT_PORT}",
            "transcription_service": (
                f"Speech to Text ({transcription_provider.name})"
                if transcription_provider
                else "Speech to Text (Not Configured)"
            ),
            "asr_uri": (
                f"{transcription_provider.mode.upper()} ({transcription_provider.name})"
                if transcription_provider
                else "Not configured"
            ),
            "transcription_provider": os.getenv("TRANSCRIPTION_PROVIDER", "auto-detect"),
            "provider_type": (
                transcription_provider.mode if transcription_provider else "none"
            ),
            "chunk_dir": str(os.getenv("CHUNK_DIR", "./audio_chunks")),
            "active_clients": get_client_manager().get_client_count(),
            "new_conversation_timeout_minutes": float(os.getenv("NEW_CONVERSATION_TIMEOUT_MINUTES", "1.5")),
            "audio_cropping_enabled": os.getenv("AUDIO_CROPPING_ENABLED", "true").lower() == "true",
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_model": os.getenv("OPENAI_MODEL"),
            "llm_base_url": os.getenv("OPENAI_BASE_URL"),
        },
    }

    overall_healthy = True
    critical_services_healthy = True
    
    # Get configuration once at the start
    memory_provider = os.getenv("MEMORY_PROVIDER", "friend_lite")
    speaker_service_url = os.getenv("SPEAKER_SERVICE_URL")
    openmemory_mcp_url = os.getenv("OPENMEMORY_MCP_URL")

    # Check MongoDB (critical service)
    try:
        await asyncio.wait_for(mongo_client.admin.command("ping"), timeout=5.0)
        health_status["services"]["mongodb"] = {
            "status": "✅ Connected",
            "healthy": True,
            "critical": True,
        }
    except asyncio.TimeoutError:
        health_status["services"]["mongodb"] = {
            "status": "❌ Connection Timeout (5s)",
            "healthy": False,
            "critical": True,
        }
        overall_healthy = False
        critical_services_healthy = False
    except Exception as e:
        health_status["services"]["mongodb"] = {
            "status": f"❌ Connection Failed: {str(e)}",
            "healthy": False,
            "critical": True,
        }
        overall_healthy = False
        critical_services_healthy = False

    # Check Redis and RQ Workers (critical for queue processing)
    try:
        from rq import Worker

        # Test Redis connection
        await asyncio.wait_for(asyncio.to_thread(redis_conn.ping), timeout=5.0)

        # Count active workers
        workers = Worker.all(connection=redis_conn)
        worker_count = len(workers)
        active_workers = len([w for w in workers if w.state == 'busy'])
        idle_workers = worker_count - active_workers

        health_status["services"]["redis"] = {
            "status": "✅ Connected",
            "healthy": True,
            "critical": True,
            "worker_count": worker_count,
            "active_workers": active_workers,
            "idle_workers": idle_workers
        }
    except asyncio.TimeoutError:
        health_status["services"]["redis"] = {
            "status": "❌ Connection Timeout (5s)",
            "healthy": False,
            "critical": True,
            "worker_count": 0
        }
        overall_healthy = False
        critical_services_healthy = False
    except Exception as e:
        health_status["services"]["redis"] = {
            "status": f"❌ Connection Failed: {str(e)}",
            "healthy": False,
            "critical": True,
            "worker_count": 0
        }
        overall_healthy = False
        critical_services_healthy = False

    # Check LLM service (non-critical service - may not be running)
    try:
        llm_health = await asyncio.wait_for(async_health_check(), timeout=8.0)
        
        # Determine overall health for audioai service based on LLM and embedder status
        is_llm_healthy = "✅" in llm_health.get("status", "")
        
        # Determine embedder health based on provider
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if llm_provider == "ollama":
            is_embedder_healthy = "✅" in llm_health.get("embedder_status", "") or llm_health.get("embedder_status") == "⚠️ Embedder Model Not Configured"
        else:
            # For OpenAI and other providers, embedder status is not applicable, so consider it healthy
            is_embedder_healthy = True
        
        audioai_overall_healthy = is_llm_healthy and is_embedder_healthy

        health_status["services"]["audioai"] = {
            "status": llm_health.get("status", "❌ Unknown"),
            "healthy": audioai_overall_healthy,
            "base_url": llm_health.get("base_url", ""),
            "model": llm_health.get("default_model", ""),
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "embedder_model": llm_health.get("embedder_model", ""),
            "embedder_status": llm_health.get("embedder_status", ""),
            "critical": False,
        }
    except asyncio.TimeoutError:
        health_status["services"]["audioai"] = {
            "status": "⚠️ Connection Timeout (8s) - Service may not be running",
            "healthy": False,
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "critical": False,
            "embedder_model": os.getenv("OLLAMA_EMBEDDER_MODEL"),
            "embedder_status": "❌ Not Checked (Timeout)"
        }
        overall_healthy = False
    except Exception as e:
        health_status["services"]["audioai"] = {
            "status": f"⚠️ Connection Failed: {str(e)} - Service may not be running",
            "healthy": False,
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "critical": False,
            "embedder_model": os.getenv("OLLAMA_EMBEDDER_MODEL"),
            "embedder_status": "❌ Not Checked (Connection Failed)"
        }
        overall_healthy = False

    # Check memory service (provider-dependent)
    if memory_provider == "friend_lite":
        try:
            # Test Friend-Lite memory service connection with timeout
            test_success = await asyncio.wait_for(memory_service.test_connection(), timeout=8.0)
            if test_success:
                health_status["services"]["memory_service"] = {
                    "status": "✅ Friend-Lite Memory Connected",
                    "healthy": True,
                    "provider": "friend_lite",
                    "critical": False,
                }
            else:
                health_status["services"]["memory_service"] = {
                    "status": "⚠️ Friend-Lite Memory Test Failed",
                    "healthy": False,
                    "provider": "friend_lite",
                    "critical": False,
                }
                overall_healthy = False
        except asyncio.TimeoutError:
            health_status["services"]["memory_service"] = {
                "status": "⚠️ Friend-Lite Memory Timeout (8s) - Check Qdrant",
                "healthy": False,
                "provider": "friend_lite",
                "critical": False,
            }
            overall_healthy = False
        except Exception as e:
            health_status["services"]["memory_service"] = {
                "status": f"⚠️ Friend-Lite Memory Failed: {str(e)}",
                "healthy": False,
                "provider": "friend_lite",
                "critical": False,
            }
            overall_healthy = False
    elif memory_provider == "openmemory_mcp":
        # OpenMemory MCP check is handled separately above
        health_status["services"]["memory_service"] = {
            "status": "✅ Using OpenMemory MCP",
            "healthy": True,
            "provider": "openmemory_mcp",
            "critical": False,
        }
    else:
        health_status["services"]["memory_service"] = {
            "status": f"❌ Unknown memory provider: {memory_provider}",
            "healthy": False,
            "provider": memory_provider,
            "critical": False,
        }
        overall_healthy = False

    # Check Speech to Text service based on configured provider
    if transcription_provider:
        provider_name = transcription_provider.name
        provider_type = transcription_provider.mode

        # Generic provider health check - let each provider handle its own connection logic
        try:
            # Test provider connection
            await transcription_provider.connect("health-check")
            await transcription_provider.disconnect()

            health_status["services"]["speech_to_text"] = {
                "status": "✅ Provider Available",
                "healthy": True,
                "type": provider_type.title(),
                "provider": provider_name,
                "critical": False,
            }
        except Exception as e:
            health_status["services"]["speech_to_text"] = {
                "status": f"⚠️ Provider Error: {str(e)}",
                "healthy": False,
                "type": provider_type.title(),
                "provider": provider_name,
                "critical": False,
            }
            # Don't mark overall health as unhealthy for transcription issues
            # since the service may be external or optional
    else:
        # No transcription service configured
        health_status["services"]["speech_to_text"] = {
            "status": "❌ No transcription service configured",
            "healthy": False,
            "type": "None",
            "provider": "None",
            "critical": False,
        }
        overall_healthy = False

    # Check Speaker Recognition service (non-critical - optional feature)
    if speaker_service_url:
        try:
            # Make a health check request to the speaker service
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{speaker_service_url}/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        health_status["services"]["speaker_recognition"] = {
                            "status": "✅ Connected",
                            "healthy": True,
                            "url": speaker_service_url,
                            "critical": False,
                        }
                    else:
                        health_status["services"]["speaker_recognition"] = {
                            "status": f"⚠️ Unhealthy: HTTP {response.status}",
                            "healthy": False,
                            "url": speaker_service_url,
                            "critical": False,
                        }
                        overall_healthy = False
        except asyncio.TimeoutError:
            health_status["services"]["speaker_recognition"] = {
                "status": "⚠️ Connection Timeout (5s)",
                "healthy": False,
                "url": speaker_service_url,
                "critical": False,
            }
            overall_healthy = False
        except Exception as e:
            health_status["services"]["speaker_recognition"] = {
                "status": f"⚠️ Connection Failed: {str(e)}",
                "healthy": False,
                "url": speaker_service_url,
                "critical": False,
            }
            overall_healthy = False

    # Check OpenMemory MCP service (if configured)
    if memory_provider == "openmemory_mcp" and openmemory_mcp_url:
        try:
            # Make a health check request to the OpenMemory MCP service
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{openmemory_mcp_url}/api/v1/apps/", timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        health_status["services"]["openmemory_mcp"] = {
                            "status": "✅ Connected",
                            "healthy": True,
                            "url": openmemory_mcp_url,
                            "provider": "openmemory_mcp",
                            "critical": False,
                        }
                    else:
                        health_status["services"]["openmemory_mcp"] = {
                            "status": f"⚠️ Unhealthy: HTTP {response.status}",
                            "healthy": False,
                            "url": openmemory_mcp_url,
                            "provider": "openmemory_mcp",
                            "critical": False,
                        }
                        overall_healthy = False
        except asyncio.TimeoutError:
            health_status["services"]["openmemory_mcp"] = {
                "status": "⚠️ Connection Timeout (5s)",
                "healthy": False,
                "url": openmemory_mcp_url,
                "provider": "openmemory_mcp",
                "critical": False,
            }
            overall_healthy = False
        except Exception as e:
            health_status["services"]["openmemory_mcp"] = {
                "status": f"⚠️ Connection Failed: {str(e)}",
                "healthy": False,
                "url": openmemory_mcp_url,
                "provider": "openmemory_mcp",
                "critical": False,
            }
            overall_healthy = False

    # Set overall status
    health_status["overall_healthy"] = overall_healthy
    health_status["critical_services_healthy"] = critical_services_healthy

    if not critical_services_healthy:
        health_status["status"] = "critical"
    elif not overall_healthy:
        health_status["status"] = "degraded"
    else:
        health_status["status"] = "healthy"

    # Add helpful messages
    if not overall_healthy:
        messages = []
        if not critical_services_healthy:
            messages.append(
                "Critical services (MongoDB) are unavailable - core functionality will not work"
            )

        unhealthy_optional = [
            name
            for name, service in health_status["services"].items()
            if not service["healthy"] and not service.get("critical", True)
        ]
        if unhealthy_optional:
            messages.append(f"Optional services unavailable: {', '.join(unhealthy_optional)}")

        health_status["message"] = "; ".join(messages)

    return JSONResponse(content=health_status, status_code=200)


@router.get("/readiness")
async def readiness_check():
    """Simple readiness check for container orchestration."""
    # Use debug level for health check to reduce log spam
    logger.debug("Readiness check requested")
    
    # Only check critical services for readiness
    try:
        # Quick MongoDB ping to ensure we can serve requests
        await asyncio.wait_for(mongo_client.admin.command("ping"), timeout=2.0)
        return JSONResponse(content={"status": "ready", "timestamp": int(time.time())}, status_code=200)
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            content={"status": "not_ready", "error": str(e), "timestamp": int(time.time())}, 
            status_code=503
        )