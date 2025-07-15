#!/usr/bin/env python3
"""Unified Omi-audio service

* Accepts Opus packets over a WebSocket (`/ws`) or PCM over a WebSocket (`/ws_pcm`).
* Uses a central queue to decouple audio ingestion from processing.
* A saver consumer buffers PCM and writes 30-second WAV chunks to `./audio_chunks/`.
* A transcription consumer sends each chunk to a Wyoming ASR service.
* The transcript is stored in **mem0** and MongoDB.

"""

import asyncio
import concurrent.futures
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

from utils.logging import audio_logger

from routers import audio_chunks_router
from routers import user_router
from routers import memory_router
from routers import action_items_router
from utils.transcribe import TranscriptionManager
from wyoming.audio import AudioChunk

import ollama
import openai
from dotenv import load_dotenv
from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
from omi.decoder import OmiOpusDecoder
from wyoming.client import AsyncTcpClient

# from debug_utils import memory_debug
from memory import get_memory_service, shutdown_memory_service
# init_memory_config, 
from metrics import (
    get_metrics_collector,
    start_metrics_collection,
    stop_metrics_collection,
)
from action_items_service import ActionItemsService
from memory import get_memory_client
from conversation_manager import active_clients, create_client_state, cleanup_client_state

###############################################################################
# SETUP
###############################################################################

# Load environment variables first
load_dotenv()

# Mem0 telemetry configuration is now handled in the memory module



RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

###############################################################################
# CONFIGURATION
###############################################################################

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client.get_default_database("friend-lite")
users_col = db["users"]
speakers_col = db["speakers"]  # New collection for speaker management
action_items_col = db["action_items"]  # New collection for action items

# Audio Configuration
OMI_SAMPLE_RATE = 16_000  # Hz
OMI_CHANNELS = 1
OMI_SAMPLE_WIDTH = 2  # bytes (16‑bit)
SEGMENT_SECONDS = 60  # length of each stored chunk
TARGET_SAMPLES = OMI_SAMPLE_RATE * SEGMENT_SECONDS

# Conversation timeout configuration
NEW_CONVERSATION_TIMEOUT_MINUTES = float(
    os.getenv("NEW_CONVERSATION_TIMEOUT_MINUTES", "1.5")
)

# Audio cropping configuration
AUDIO_CROPPING_ENABLED = os.getenv("AUDIO_CROPPING_ENABLED", "true").lower() == "true"
MIN_SPEECH_SEGMENT_DURATION = float(
    os.getenv("MIN_SPEECH_SEGMENT_DURATION", "1.0")
)  # seconds
CROPPING_CONTEXT_PADDING = float(
    os.getenv("CROPPING_CONTEXT_PADDING", "0.1")
)  # seconds of padding around speech

# Directory where WAV chunks are written
CHUNK_DIR = Path("./audio_chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# Ollama & Qdrant Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", "qdrant")

# Memory configuration is now handled in the memory module
# Initialize it with our Ollama and Qdrant URLs
# init_memory_config(
#     ollama_base_url=OLLAMA_BASE_URL,
#     qdrant_base_url=QDRANT_BASE_URL,
# )

# Speaker service configuration

# Thread pool executors
_DEC_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=os.cpu_count() or 4,
    thread_name_prefix="opus_io",
)

# Initialize memory service, speaker service, and ollama client
memory_service = get_memory_service()
if os.getenv("LLM_PROVIDER") == "ollama":
    llm_client = ollama.Client(host=OLLAMA_BASE_URL)
elif os.getenv("LLM_PROVIDER") == "openai":
    llm_client = openai.Client(api_key=os.getenv("LLM_API_KEY"))

action_items_service = ActionItemsService(action_items_col, llm_client)




###############################################################################
# CORE APPLICATION LOGIC
###############################################################################


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    audio_logger.info("Starting application...")

    # Start metrics collection
    
    if os.getenv("METRICS_COLLECTION_ENABLE"):
        await start_metrics_collection(CHUNK_DIR)
        audio_logger.info("Metrics collection started")

    audio_logger.info(
        "Application ready - clients will have individual processing pipelines."
    )

    try:
        yield
    finally:
        # Shutdown
        audio_logger.info("Shutting down application...")

        # Clean up all active clients
        for client_id in list(active_clients.keys()):
            await cleanup_client_state(client_id)

        # Stop metrics collection and save final report
        await stop_metrics_collection(CHUNK_DIR)
        audio_logger.info("Metrics collection stopped")

        # Shutdown memory service and speaker service
        shutdown_memory_service()
        audio_logger.info("Memory and speaker services shut down.")

        audio_logger.info("Shutdown complete.")


# FastAPI Application
app = FastAPI(lifespan=lifespan)
app.mount("/audio", StaticFiles(directory=CHUNK_DIR), name="audio")
app.include_router(audio_chunks_router.router)
app.include_router(user_router.router)
app.include_router(memory_router.router)
app.include_router(action_items_router.router)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, user_id: Optional[str] = Query(None), audio_chunk_utils: audio_chunks_router.AudioChunkUtils = Depends(audio_chunks_router.get_audio_chunk_utils)):
    """Accepts WebSocket connections, decodes Opus audio, and processes per-client."""
    await ws.accept()

    # Use user_id if provided, otherwise generate a random client_id
    client_id = user_id if user_id else f"client_{str(uuid.uuid4())}"
    audio_logger.info(f"🔌 WebSocket connection accepted - Client: {client_id}, User ID: {user_id}")
    
    decoder = OmiOpusDecoder()
    _decode_packet = partial(decoder.decode_packet, strip_header=False)

    # Create client state and start processing
    client_state = await create_client_state(client_id, audio_chunk_utils, {
        "CHUNK_DIR": CHUNK_DIR,
        "OMI_SAMPLE_RATE": OMI_SAMPLE_RATE,
        "OMI_CHANNELS": OMI_CHANNELS,
        "OMI_SAMPLE_WIDTH": OMI_SAMPLE_WIDTH,
        "NEW_CONVERSATION_TIMEOUT_MINUTES": NEW_CONVERSATION_TIMEOUT_MINUTES,
        "AUDIO_CROPPING_ENABLED": AUDIO_CROPPING_ENABLED,
        "MIN_SPEECH_SEGMENT_DURATION": MIN_SPEECH_SEGMENT_DURATION,
        "CROPPING_CONTEXT_PADDING": CROPPING_CONTEXT_PADDING,
        "_DEC_IO_EXECUTOR": _DEC_IO_EXECUTOR,
        "action_items_service": action_items_service,
    }, TranscriptionManager)

    try:
        packet_count = 0
        total_bytes = 0
        while True:
            packet = await ws.receive_bytes()
            packet_count += 1
            total_bytes += len(packet)
            
            start_time = time.time()
            loop = asyncio.get_running_loop()
            pcm_data = await loop.run_in_executor(
                _DEC_IO_EXECUTOR, _decode_packet, packet
            )
            decode_time = time.time() - start_time
            
            if pcm_data:
                audio_logger.debug(f"🎵 Decoded packet #{packet_count}: {len(packet)} bytes -> {len(pcm_data)} PCM bytes (took {decode_time:.3f}s)")
                chunk = AudioChunk(
                    audio=pcm_data,
                    rate=OMI_SAMPLE_RATE,
                    width=OMI_SAMPLE_WIDTH,
                    channels=OMI_CHANNELS,
                    timestamp=int(time.time()),
                )
                await client_state.chunk_queue.put(chunk)
                
                # Log every 1000th packet to avoid spam
                if packet_count % 1000 == 0:
                    audio_logger.info(f"📊 Processed {packet_count} packets ({total_bytes} bytes total) for client {client_id}")

                # Track audio chunk received in metrics
                get_metrics_collector().record_audio_chunk_received(client_id)
                get_metrics_collector().record_client_activity(client_id)

    except WebSocketDisconnect:
        audio_logger.info(f"🔌 WebSocket disconnected - Client: {client_id}, Packets: {packet_count}, Total bytes: {total_bytes}")
    except Exception as e:
        audio_logger.error(f"❌ WebSocket error for client {client_id}: {e}", exc_info=True)
    finally:
        # Clean up client state
        await cleanup_client_state(client_id)


@app.websocket("/ws_pcm")
async def ws_endpoint_pcm(ws: WebSocket, user_id: Optional[str] = Query(None), audio_chunk_utils: audio_chunks_router.AudioChunkUtils = Depends(audio_chunks_router.get_audio_chunk_utils)):
    """Accepts WebSocket connections, processes PCM audio per-client."""
    await ws.accept()

    # Use user_id if provided, otherwise generate a random client_id
    client_id = user_id if user_id else f"client_{uuid.uuid4().hex[:8]}"
    audio_logger.info(f"🔌 PCM WebSocket connection accepted - Client: {client_id}, User ID: {user_id}")
    
    # Create client state and start processing
    client_state = await create_client_state(client_id, audio_chunk_utils, {
        "CHUNK_DIR": CHUNK_DIR,
        "OMI_SAMPLE_RATE": OMI_SAMPLE_RATE,
        "OMI_CHANNELS": OMI_CHANNELS,
        "OMI_SAMPLE_WIDTH": OMI_SAMPLE_WIDTH,
        "NEW_CONVERSATION_TIMEOUT_MINUTES": NEW_CONVERSATION_TIMEOUT_MINUTES,
        "AUDIO_CROPPING_ENABLED": AUDIO_CROPPING_ENABLED,
        "MIN_SPEECH_SEGMENT_DURATION": MIN_SPEECH_SEGMENT_DURATION,
        "CROPPING_CONTEXT_PADDING": CROPPING_CONTEXT_PADDING,
        "_DEC_IO_EXECUTOR": _DEC_IO_EXECUTOR,
        "action_items_service": action_items_service,
    }, TranscriptionManager)

    try:
        packet_count = 0
        total_bytes = 0
        while True:
            packet = await ws.receive_bytes()
            packet_count += 1
            total_bytes += len(packet)
            
            if packet:
                audio_logger.debug(f"🎵 Received PCM packet #{packet_count}: {len(packet)} bytes")
                chunk = AudioChunk(
                    audio=packet,
                    rate=16000,
                    width=2,
                    channels=1,
                    timestamp=int(time.time()),
                )
                await client_state.chunk_queue.put(chunk)
                
                # Log every 1000th packet to avoid spam
                if packet_count % 1000 == 0:
                    audio_logger.info(f"📊 Processed {packet_count} PCM packets ({total_bytes} bytes total) for client {client_id}")
                        

                # Track audio chunk received in metrics
                get_metrics_collector().record_audio_chunk_received(client_id)
                get_metrics_collector().record_client_activity(client_id)
    except WebSocketDisconnect:
        audio_logger.info(f"🔌 PCM WebSocket disconnected - Client: {client_id}, Packets: {packet_count}, Total bytes: {total_bytes}")
    except Exception as e:
        audio_logger.error(f"❌ PCM WebSocket error for client {client_id}: {e}", exc_info=True)
    finally:
        # Clean up client state
        await cleanup_client_state(client_id)











# class SpeakerEnrollmentRequest(BaseModel):
#     speaker_id: str
#     speaker_name: str
#     audio_file_path: str
#     start_time: Optional[float] = None
#     end_time: Optional[float] = None


# class SpeakerIdentificationRequest(BaseModel):
#     audio_file_path: str
#     start_time: Optional[float] = None
#     end_time: Optional[float] = None


# class ActionItemUpdateRequest(BaseModel):
#     status: str  # "open", "in_progress", "completed", "cancelled"


# class ActionItemCreateRequest(BaseModel):
#     description: str
#     assignee: Optional[str] = "unassigned"
#     due_date: Optional[str] = "not_specified"
#     priority: Optional[str] = "medium"
#     context: Optional[str] = ""


@app.get("/health")
async def health_check():
    """Comprehensive health check for all services."""
    health_status = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "services": {},
        "config": {
            "mongodb_uri": MONGODB_URI,
            "llm_url": OLLAMA_BASE_URL if os.getenv("LLM_PROVIDER") == "ollama" else LLM_BASE_URL,
            "qdrant_url": f"http://QDRANT_BASE_URL:6333",
            "asr_uri": os.getenv("OFFLINE_ASR_TCP_URI", "tcp://192.168.0.110:8765/"),
            "chunk_dir": str(CHUNK_DIR),
            "active_clients": len(active_clients),
            "new_conversation_timeout_minutes": NEW_CONVERSATION_TIMEOUT_MINUTES,
            "action_items_enabled": True,
            "audio_cropping_enabled": AUDIO_CROPPING_ENABLED,
        },
    }

    overall_healthy = True
    critical_services_healthy = True

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

    # Check LLM Service (Ollama or OpenAI)
    llm_provider = os.getenv("LLM_PROVIDER")
    health_status["services"]["llm"] = {
        "status": "Unknown",
        "healthy": False,
        "critical": False,
    }

    if llm_provider == "ollama":
        try:
            loop = asyncio.get_running_loop()
            models = await asyncio.wait_for(
                loop.run_in_executor(None, llm_client.list), timeout=8.0
            )
            model_count = len(models.get("models", []))
            health_status["services"]["llm"] = {
                "status": "✅ Connected (Ollama)",
                "healthy": True,
                "models": model_count,
                "critical": False,
            }
        except asyncio.TimeoutError:
            health_status["services"]["llm"] = {
                "status": "⚠️ Connection Timeout (8s) - Ollama service may not be running",
                "healthy": False,
                "critical": False,
            }
            overall_healthy = False
        except Exception as e:
            health_status["services"]["llm"] = {
                "status": f"⚠️ Connection Failed (Ollama): {str(e)}",
                "healthy": False,
                "critical": False,
            }
            overall_healthy = False
    elif llm_provider == "openai":
        try:
            loop = asyncio.get_running_loop()
            # For OpenAI, a simple list models call or a dummy completion call can serve as a health check
            # Using models.list() as it's a common way to check API connectivity
            models = await asyncio.wait_for(
                loop.run_in_executor(None, llm_client.models.list), timeout=8.0
            )
            model_count = len(models.data) # OpenAI returns models in .data
            health_status["services"]["llm"] = {
                "status": "✅ Connected (OpenAI)",
                "healthy": True,
                "models": model_count,
                "critical": False,
            }
        except asyncio.TimeoutError:
            health_status["services"]["llm"] = {
                "status": "⚠️ Connection Timeout (8s) - OpenAI API may be unreachable",
                "healthy": False,
                "critical": False,
            }
            overall_healthy = False
        except Exception as e:
            health_status["services"]["llm"] = {
                "status": f"⚠️ Connection Failed (OpenAI): {str(e)}",
                "healthy": False,
                "critical": False,
            }
            overall_healthy = False
    else:
        health_status["services"]["llm"] = {
            "status": "⚠️ LLM_PROVIDER not specified or invalid",
            "healthy": False,
            "critical": False,
        }
        overall_healthy = False

    # Check mem0 (depends on Ollama and Qdrant)
    try:
        # Test memory service connection with timeout
        test_success = memory_service.test_connection()
        if test_success:
            health_status["services"]["mem0"] = {
                "status": "✅ Connected",
                "healthy": True,
                "critical": False,
            }
        else:
            health_status["services"]["mem0"] = {
                "status": "⚠️ Connection Test Failed",
                "healthy": False,
                "critical": False,
            }
            overall_healthy = False
    except asyncio.TimeoutError:
        health_status["services"]["mem0"] = {
            "status": "⚠️ Connection Test Timeout (10s) - Depends on Ollama/Qdrant",
            "healthy": False,
            "critical": False,
        }
        overall_healthy = False
    except Exception as e:
        health_status["services"]["mem0"] = {
            "status": f"⚠️ Connection Test Failed: {str(e)} - Check Ollama/Qdrant services",
            "healthy": False,
            "critical": False,
        }
        overall_healthy = False

    # Check ASR service (non-critical - may be external)
    try:
        test_client = AsyncTcpClient.from_uri(os.getenv("OFFLINE_ASR_TCP_URI", "tcp://192.168.0.110:8765/"))
        await asyncio.wait_for(test_client.connect(), timeout=5.0)
        await test_client.disconnect()
        health_status["services"]["asr"] = {
            "status": "✅ Connected",
            "healthy": True,
            "uri": os.getenv("OFFLINE_ASR_TCP_URI", "tcp://192.168.0.110:8765/"),
            "critical": False,
        }
    except asyncio.TimeoutError:
        health_status["services"]["asr"] = {
            "status": f"⚠️ Connection Timeout (5s) - Check external ASR service",
            "healthy": False,
            "uri": OFFLINE_ASR_TCP_URI,
            "critical": False,
        }
        overall_healthy = False
    except Exception as e:
        health_status["services"]["asr"] = {
            "status": f"⚠️ Connection Failed: {str(e)} - Check external ASR service",
            "healthy": False,
            "uri": OFFLINE_ASR_TCP_URI,
            "critical": False,
        }
        overall_healthy = False

    # Track health check results in metrics
    # The metrics_collector is now passed to ClientState and used there.
    # No need to get it here.
    pass

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
            messages.append(
                f"Optional services unavailable: {', '.join(unhealthy_optional)}"
            )

        health_status["message"] = "; ".join(messages)

    return JSONResponse(content=health_status, status_code=200)


@app.get("/readiness")
async def readiness_check():
    """Simple readiness check for container orchestration."""
    return JSONResponse(
        content={"status": "ready", "timestamp": int(time.time())}, status_code=200
    )


@app.post("/api/close_conversation")
async def close_current_conversation(client_id: str):
    """Close the current conversation for a specific client."""
    if client_id not in active_clients:
        return JSONResponse(
            content={"error": f"Client '{client_id}' not found or not connected"},
            status_code=404,
        )

    client_state = active_clients[client_id]
    if not client_state.connected:
        return JSONResponse(
            content={"error": f"Client '{client_id}' is not connected"}, status_code=400
        )

    try:
        # Close the current conversation
        await client_state._close_current_conversation()

        # Reset conversation state but keep client connected
        client_state.current_audio_uuid = None
        client_state.conversation_start_time = time.time()
        client_state.last_transcript_time = None

        logger.info(f"Manually closed conversation for client {client_id}")

        return JSONResponse(
            content={
                "message": f"Successfully closed current conversation for client '{client_id}'",
                "client_id": client_id,
                "timestamp": int(time.time()),
            }
        )

    except Exception as e:
        logger.error(f"Error closing conversation for client {client_id}: {e}")
        return JSONResponse(
            content={"error": f"Failed to close conversation: {str(e)}"},
            status_code=500,
        )


@app.get("/api/active_clients")
async def get_active_clients():
    """Get list of currently active/connected clients."""
    client_info = {}

    for client_id, client_state in active_clients.items():
        client_info[client_id] = {
            "connected": client_state.connected,
            "current_audio_uuid": client_state.current_audio_uuid,
            "conversation_start_time": client_state.conversation_start_time,
            "last_transcript_time": client_state.last_transcript_time,
            "has_active_conversation": client_state.current_audio_uuid is not None,
        }

    return JSONResponse(
        content={"active_clients_count": len(active_clients), "clients": client_info}
    )


@app.get("/api/debug/speech_segments")
async def debug_speech_segments():
    """Debug endpoint to check current speech segments for all active clients."""
    debug_info = {
        "active_clients": len(active_clients),
        "audio_cropping_enabled": AUDIO_CROPPING_ENABLED,
        "min_speech_duration": MIN_SPEECH_SEGMENT_DURATION,
        "cropping_padding": CROPPING_CONTEXT_PADDING,
        "clients": {},
    }

    for client_id, client_state in active_clients.items():
        debug_info["clients"][client_id] = {
            "current_audio_uuid": client_state.current_audio_uuid,
            "speech_segments": {
                uuid: segments
                for uuid, segments in client_state.speech_segments.items()
            },
            "current_speech_start": dict(client_state.current_speech_start),
            "connected": client_state.connected,
            "last_transcript_time": client_state.last_transcript_time,
        }

    return JSONResponse(content=debug_info)


@app.get("/api/debug/memory-processing")
async def debug_memory_processing(
    user_id: Optional[str] = None,
    limit: int = 50,
    since_timestamp: Optional[int] = None,
):
    """Get debug information about memory processing operations."""
    try:
        # debug_entries = memory_debug.get_debug_entries(
        #     user_id=user_id, limit=limit, since_timestamp=since_timestamp
        # )

        pass
        # return JSONResponse(
        #     content={
        #         "debug_entries": debug_entries,
        #         "total_entries": len(debug_entries),
        #         "user_filter": user_id,
        #         "limit": limit,
        #         "since_timestamp": since_timestamp,
        #     }
        # )

    except Exception as e:
        audio_logger.error(f"Error getting memory processing debug info: {e}")
        return JSONResponse(
            status_code=500, content={"error": "Failed to get debug information"}
        )


@app.get("/api/debug/memory-processing/stats")
async def debug_memory_processing_stats(user_id: Optional[str] = None):
    """Get statistics about memory processing operations."""
    try:
        # stats = memory_debug.get_debug_stats(user_id=user_id)

        pass
        # return JSONResponse(content={"user_id": user_id, "statistics": stats})

    except Exception as e:
        audio_logger.error(f"Error getting memory processing stats: {e}")
        return JSONResponse(
            status_code=500, content={"error": "Failed to get debug statistics"}
        )


@app.get("/api/metrics")
async def get_current_metrics():
    """Get current metrics summary for monitoring dashboard."""
    try:
        metrics_summary = self.metrics_collector.get_current_metrics_summary()
        return metrics_summary
    except Exception as e:
        audio_logger.error(f"Error getting current metrics: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


###############################################################################
# ENTRYPOINT
###############################################################################

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    audio_logger.info("Starting Omi unified service at ws://%s:%s/ws", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=False)
