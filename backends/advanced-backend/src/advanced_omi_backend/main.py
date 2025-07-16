#!/usr/bin/env python3
"""Unified Omi-audio service

* Accepts Opus packets over a WebSocket (`/ws`) or PCM over a WebSocket (`/ws_pcm`).
* Uses a central queue to decouple audio ingestion from processing.
* A saver consumer buffers PCM and writes 30-second WAV chunks to `./data/audio_chunks/`.
* A transcription consumer sends each chunk to a Wyoming ASR service.
* The transcript is stored in **mem0** and MongoDB.

"""
import logging

logging.basicConfig(level=logging.INFO)

import asyncio
import concurrent.futures
import os
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Optional

# Import Beanie for user management
from beanie import init_beanie
from dotenv import load_dotenv
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
from omi.decoder import OmiOpusDecoder
from wyoming.audio import AudioChunk
from wyoming.client import AsyncTcpClient

# Import authentication components
from advanced_omi_backend.auth import (
    bearer_backend,
    cookie_backend,
    create_admin_user_if_needed,
    fastapi_users,
    websocket_auth,
)
from advanced_omi_backend.client import ClientState
from advanced_omi_backend.database import AudioChunksCollectionHelper
from advanced_omi_backend.debug_system_tracker import (
    get_debug_tracker,
    init_debug_tracker,
    shutdown_debug_tracker,
)
from advanced_omi_backend.memory import (
    get_memory_service,
    init_memory_config,
    shutdown_memory_service,
)
from advanced_omi_backend.transcription_providers import get_transcription_provider
from advanced_omi_backend.users import User, generate_client_id, register_client_to_user

###############################################################################
# SETUP
###############################################################################

# Load environment variables first
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advanced-backend")
audio_logger = logging.getLogger("audio_processing")

# Conditional Deepgram import
try:
    from deepgram import DeepgramClient, FileSource, PrerecordedOptions  # type: ignore

    DEEPGRAM_AVAILABLE = True
    logger.info("✅ Deepgram SDK available")
except ImportError:
    DEEPGRAM_AVAILABLE = False
    logger.warning("Deepgram SDK not available. Install with: pip install deepgram-sdk")
audio_cropper_logger = logging.getLogger("audio_cropper")


###############################################################################
# CONFIGURATION
###############################################################################

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client.get_default_database("friend-lite")
chunks_col = db["audio_chunks"]
users_col = db["users"]
speakers_col = db["speakers"]

# Audio Configuration
OMI_SAMPLE_RATE = 16_000  # Hz
OMI_CHANNELS = 1
OMI_SAMPLE_WIDTH = 2  # bytes (16‑bit)
SEGMENT_SECONDS = 60  # length of each stored chunk
TARGET_SAMPLES = OMI_SAMPLE_RATE * SEGMENT_SECONDS

# Conversation timeout configuration
NEW_CONVERSATION_TIMEOUT_MINUTES = float(os.getenv("NEW_CONVERSATION_TIMEOUT_MINUTES", "1.5"))

# Audio cropping configuration
AUDIO_CROPPING_ENABLED = os.getenv("AUDIO_CROPPING_ENABLED", "true").lower() == "true"
MIN_SPEECH_SEGMENT_DURATION = float(os.getenv("MIN_SPEECH_SEGMENT_DURATION", "1.0"))  # seconds
CROPPING_CONTEXT_PADDING = float(
    os.getenv("CROPPING_CONTEXT_PADDING", "0.1")
)  # seconds of padding around speech

# Directory where WAV chunks are written
CHUNK_DIR = Path("./audio_chunks")  # This will be mounted to ./data/audio_chunks by Docker
CHUNK_DIR.mkdir(parents=True, exist_ok=True)


# Transcription Configuration
TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OFFLINE_ASR_TCP_URI = os.getenv("OFFLINE_ASR_TCP_URI", "tcp://localhost:8765")

# Determine which transcription service to use
USE_ONLINE_TRANSCRIPTION = bool(TRANSCRIPTION_PROVIDER and (DEEPGRAM_API_KEY or MISTRAL_API_KEY))
USE_OFFLINE_ASR = not USE_ONLINE_TRANSCRIPTION and bool(OFFLINE_ASR_TCP_URI)


transcription_provider = get_transcription_provider(TRANSCRIPTION_PROVIDER)
if transcription_provider:
    logger.info(f"✅ Using {transcription_provider.name} transcription provider")
else:
    logger.info("⚠️ No online transcription provider configured")

# Ollama & Qdrant Configuration
QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", "qdrant")

# Memory configuration is now handled in the memory module
# Initialize it with our Ollama and Qdrant URLs
init_memory_config(
    qdrant_base_url=QDRANT_BASE_URL,
)

# Speaker service configuration

# Thread pool executors
_DEC_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=os.cpu_count() or 4,
    thread_name_prefix="opus_io",
)

# Initialize memory service
memory_service = get_memory_service()

###############################################################################
# UTILITY FUNCTIONS & HELPER CLASSES
###############################################################################


# Initialize repository and global state
ac_db_collection_helper = AudioChunksCollectionHelper(chunks_col)
active_clients: dict[str, ClientState] = {}

# Client-to-user mapping for reliable permission checking
client_to_user_mapping: dict[str, str] = {}  # client_id -> user_id

# Initialize client manager with active_clients reference
from advanced_omi_backend.client_manager import init_client_manager

init_client_manager(active_clients)

# Initialize client utilities with the mapping dictionaries
from advanced_omi_backend.client_manager import (
    init_client_user_mapping,
    register_client_user_mapping,
    track_client_user_relationship,
    unregister_client_user_mapping,
)

# Client ownership tracking for database records
# Since we're in development, we'll track all client-user relationships in memory
# This will be populated when clients connect and persisted in database records
all_client_user_mappings: dict[str, str] = (
    {}
)  # client_id -> user_id (includes disconnected clients)

# Initialize client user mapping with both dictionaries
init_client_user_mapping(client_to_user_mapping, all_client_user_mappings)


def get_user_clients(user_id: str) -> list[str]:
    """Get all currently active client IDs that belong to a specific user."""
    return [
        client_id
        for client_id, mapped_user_id in client_to_user_mapping.items()
        if mapped_user_id == user_id
    ]


async def create_client_state(
    client_id: str, user: User, device_name: Optional[str] = None
) -> ClientState:
    """Create and register a new client state."""
    client_state = ClientState(
        client_id, ac_db_collection_helper, CHUNK_DIR, user.user_id, user.email
    )
    active_clients[client_id] = client_state

    # Register client-user mapping (for active clients)
    register_client_user_mapping(client_id, user.user_id)

    # Also track in persistent mapping (for database queries)
    track_client_user_relationship(client_id, user.user_id)

    # Register client in user model (persistent)
    await register_client_to_user(user, client_id, device_name)

    await client_state.start_processing()

    return client_state


async def cleanup_client_state(client_id: str):
    """Clean up and remove client state."""
    if client_id in active_clients:
        client_state = active_clients[client_id]
        await client_state.disconnect()
        del active_clients[client_id]

    # Unregister client-user mapping
    unregister_client_user_mapping(client_id)


###############################################################################
# CORE APPLICATION LOGIC
###############################################################################


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    audio_logger.info("Starting application...")

    # Initialize Beanie for user management
    try:
        await init_beanie(
            database=mongo_client.get_default_database("friend-lite"),
            document_models=[User],
        )
        audio_logger.info("Beanie initialized for user management")
    except Exception as e:
        audio_logger.error(f"Failed to initialize Beanie: {e}")
        raise

    # Create admin user if needed
    try:
        await create_admin_user_if_needed()
    except Exception as e:
        audio_logger.error(f"Failed to create admin user: {e}")
        # Don't raise here as this is not critical for startup

    # Start metrics collection
    # Initialize debug tracker
    init_debug_tracker()
    audio_logger.info("Metrics collection started")

    # Pre-initialize memory service to avoid blocking during first use
    try:
        audio_logger.info("Pre-initializing memory service...")
        await asyncio.wait_for(
            memory_service.initialize(), timeout=120
        )  # 2 minute timeout for startup
        audio_logger.info("Memory service pre-initialized successfully")
    except asyncio.TimeoutError:
        audio_logger.warning(
            "Memory service pre-initialization timed out - will initialize on first use"
        )
    except Exception as e:
        audio_logger.warning(
            f"Memory service pre-initialization failed: {e} - will initialize on first use"
        )

    # SystemTracker is used for monitoring and debugging
    audio_logger.info("Using SystemTracker for monitoring and debugging")

    audio_logger.info("Application ready - clients will have individual processing pipelines.")

    try:
        yield
    finally:
        # Shutdown
        audio_logger.info("Shutting down application...")

        # Clean up all active clients
        for client_id in list(active_clients.keys()):
            await cleanup_client_state(client_id)

        # Stop metrics collection and save final report
        # Shutdown debug tracker
        shutdown_debug_tracker()
        audio_logger.info("Metrics collection stopped")

        # Shutdown memory service and speaker service
        shutdown_memory_service()
        audio_logger.info("Memory and speaker services shut down.")

        audio_logger.info("Shutdown complete.")


# FastAPI Application
app = FastAPI(lifespan=lifespan)
app.mount("/audio", StaticFiles(directory=CHUNK_DIR), name="audio")

# Add authentication routers
app.include_router(
    fastapi_users.get_auth_router(cookie_backend),
    prefix="/auth/cookie",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_auth_router(bearer_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)


# API endpoints
from advanced_omi_backend.routers.api_router import router as api_router

app.include_router(api_router)


@app.websocket("/ws_omi")
async def ws_endpoint(
    ws: WebSocket,
    token: Optional[str] = Query(None),
    device_name: Optional[str] = Query(None),
):
    """Accepts WebSocket connections, decodes Opus audio, and processes per-client."""
    # TODO: Accept parameters or some type of "audio config" message from the client to setup
    # the proper file sink.

    # Authenticate user before accepting WebSocket connection
    user = await websocket_auth(ws, token)
    if not user:
        await ws.close(code=1008, reason="Authentication required")
        return

    await ws.accept()

    # Generate proper client_id using user and device_name
    client_id = generate_client_id(user, device_name)
    audio_logger.info(
        f"🔌 WebSocket connection accepted - User: {user.user_id} ({user.email}), Client: {client_id}"
    )

    decoder = OmiOpusDecoder()
    _decode_packet = partial(decoder.decode_packet, strip_header=False)

    # Create client state and start processing
    client_state = await create_client_state(client_id, user, device_name)

    # Track WebSocket connection
    # tracker = get_debug_tracker()
    # tracker.track_websocket_connected(user.user_id, client_id)

    try:
        packet_count = 0
        total_bytes = 0
        while True:
            packet = await ws.receive_bytes()
            packet_count += 1
            total_bytes += len(packet)

            start_time = time.time()
            loop = asyncio.get_running_loop()
            pcm_data = await loop.run_in_executor(_DEC_IO_EXECUTOR, _decode_packet, packet)
            decode_time = time.time() - start_time

            if pcm_data:
                audio_logger.debug(
                    f"🎵 Decoded packet #{packet_count}: {len(packet)} bytes -> {len(pcm_data)} PCM bytes (took {decode_time:.3f}s)"
                )
                chunk = AudioChunk(
                    audio=pcm_data,
                    rate=OMI_SAMPLE_RATE,
                    width=OMI_SAMPLE_WIDTH,
                    channels=OMI_CHANNELS,
                    timestamp=int(time.time()),
                )
                await client_state.chunk_queue.put(chunk)

                # # Track audio chunk with debug tracker
                # if packet_count == 1:  # Create transaction on first audio chunk
                #     client_state.transaction_id = tracker.create_transaction(
                #         user.user_id, client_id
                #     )
                # if hasattr(client_state, "transaction_id") and client_state.transaction_id:
                #     tracker.track_audio_chunk(client_state.transaction_id, len(pcm_data))

                # Log every 1000th packet to avoid spam
                if packet_count % 1000 == 0:
                    audio_logger.info(
                        f"📊 Processed {packet_count} packets ({total_bytes} bytes total) for client {client_id}"
                    )

    except WebSocketDisconnect:
        audio_logger.info(
            f"🔌 WebSocket disconnected - Client: {client_id}, Packets: {packet_count}, Total bytes: {total_bytes}"
        )
    except Exception as e:
        audio_logger.error(f"❌ WebSocket error for client {client_id}: {e}", exc_info=True)
    finally:
        # # Track WebSocket disconnection
        # tracker = get_debug_tracker()
        # tracker.track_websocket_disconnected(client_id)

        # Clean up client state
        await cleanup_client_state(client_id)


@app.websocket("/ws_pcm")
async def ws_endpoint_pcm(
    ws: WebSocket, token: Optional[str] = Query(None), device_name: Optional[str] = Query(None)
):
    """Accepts WebSocket connections, processes PCM audio per-client."""
    # Authenticate user before accepting WebSocket connection
    user = await websocket_auth(ws, token)
    if not user:
        await ws.close(code=1008, reason="Authentication required")
        return

    await ws.accept()

    # Generate proper client_id using user and device_name
    client_id = generate_client_id(user, device_name)
    audio_logger.info(
        f"🔌 PCM WebSocket connection accepted - User: {user.user_id} ({user.email}), Client: {client_id}"
    )

    # Create client state and start processing
    client_state = await create_client_state(client_id, user, device_name)

    # Track WebSocket connection
    tracker = get_debug_tracker()
    tracker.track_websocket_connected(user.user_id, client_id)

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

                # Track audio chunk with debug tracker
                if packet_count == 1:  # Create transaction on first audio chunk
                    client_state.transaction_id = tracker.create_transaction(
                        user.user_id, client_id
                    )
                if hasattr(client_state, "transaction_id") and client_state.transaction_id:
                    tracker.track_audio_chunk(client_state.transaction_id, len(packet))

                # Log every 1000th packet to avoid spam
                if packet_count % 1000 == 0:
                    audio_logger.info(
                        f"📊 Processed {packet_count} PCM packets ({total_bytes} bytes total) for client {client_id}"
                    )
    except WebSocketDisconnect:
        audio_logger.info(
            f"🔌 PCM WebSocket disconnected - Client: {client_id}, Packets: {packet_count}, Total bytes: {total_bytes}"
        )
    except Exception as e:
        audio_logger.error(f"❌ PCM WebSocket error for client {client_id}: {e}", exc_info=True)
    finally:
        # Track WebSocket disconnection
        tracker = get_debug_tracker()
        tracker.track_websocket_disconnected(client_id)

        # Clean up client state
        await cleanup_client_state(client_id)


@app.get("/health")
async def health_check():
    """Comprehensive health check for all services."""
    health_status = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "services": {},
        "config": {
            "mongodb_uri": MONGODB_URI,
            "qdrant_url": f"http://{QDRANT_BASE_URL}:6333",
            "transcription_service": (
                f"Speech to Text ({transcription_provider.name})"
                if transcription_provider
                else (
                    "Speech to Text (Offline ASR)"
                    if USE_OFFLINE_ASR
                    else "Speech to Text (Not Configured)"
                )
            ),
            "asr_uri": (
                f"REST API ({transcription_provider.name})"
                if transcription_provider
                else OFFLINE_ASR_TCP_URI if USE_OFFLINE_ASR else "Not configured"
            ),
            "transcription_provider": TRANSCRIPTION_PROVIDER or "offline",
            "online_transcription_enabled": USE_ONLINE_TRANSCRIPTION,
            "offline_asr_enabled": USE_OFFLINE_ASR,
            "chunk_dir": str(CHUNK_DIR),
            "active_clients": len(active_clients),
            "new_conversation_timeout_minutes": NEW_CONVERSATION_TIMEOUT_MINUTES,
            "audio_cropping_enabled": AUDIO_CROPPING_ENABLED,
            "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
            "llm_model": os.getenv("OPENAI_MODEL", "gpt-4o"),
            "llm_base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
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

    # Check LLM service (non-critical service - may not be running)
    try:
        from advanced_omi_backend.llm_client import async_health_check

        llm_health = await asyncio.wait_for(async_health_check(), timeout=8.0)
        health_status["services"]["audioai"] = {
            "status": llm_health.get("status", "❌ Unknown"),
            "healthy": "✅" in llm_health.get("status", ""),
            "base_url": llm_health.get("base_url", ""),
            "model": llm_health.get("default_model", ""),
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "critical": False,
        }
    except asyncio.TimeoutError:
        health_status["services"]["audioai"] = {
            "status": "⚠️ Connection Timeout (8s) - Service may not be running",
            "healthy": False,
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "critical": False,
        }
        overall_healthy = False
    except Exception as e:
        health_status["services"]["audioai"] = {
            "status": f"⚠️ Connection Failed: {str(e)} - Service may not be running",
            "healthy": False,
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "critical": False,
        }
        overall_healthy = False

    # Check mem0 (depends on Ollama and Qdrant)
    try:
        # Test memory service connection with timeout
        test_success = await memory_service.test_connection()
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
            "status": "⚠️ Connection Test Timeout (60s) - Depends on Ollama/Qdrant",
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

    # Check Speech to Text service based on configuration
    if USE_ONLINE_TRANSCRIPTION and transcription_provider:
        # Check online transcription provider
        try:
            # For online providers, we just check that API keys are configured
            provider_name = transcription_provider.name
            if provider_name == "Deepgram" and DEEPGRAM_API_KEY:
                health_status["services"]["speech_to_text"] = {
                    "status": "✅ API Key Configured",
                    "healthy": True,
                    "type": "REST API",
                    "provider": "Deepgram",
                    "critical": False,
                }
            elif provider_name == "Mistral" and MISTRAL_API_KEY:
                health_status["services"]["speech_to_text"] = {
                    "status": "✅ API Key Configured",
                    "healthy": True,
                    "type": "REST API",
                    "provider": "Mistral",
                    "critical": False,
                }
            else:
                health_status["services"]["speech_to_text"] = {
                    "status": f"❌ API Key Missing for {provider_name}",
                    "healthy": False,
                    "type": "REST API",
                    "provider": provider_name,
                    "critical": False,
                }
                overall_healthy = False
        except Exception as e:
            health_status["services"]["speech_to_text"] = {
                "status": f"⚠️ Provider Error: {str(e)}",
                "healthy": False,
                "type": "REST API",
                "provider": TRANSCRIPTION_PROVIDER,
                "critical": False,
            }
            overall_healthy = False
    elif USE_OFFLINE_ASR:
        # Check offline ASR service (non-critical - may be external)
        try:
            test_client = AsyncTcpClient.from_uri(OFFLINE_ASR_TCP_URI)
            await asyncio.wait_for(test_client.connect(), timeout=5.0)
            await test_client.disconnect()
            health_status["services"]["speech_to_text"] = {
                "status": "✅ Connected",
                "healthy": True,
                "type": "Offline TCP",
                "provider": "Wyoming ASR",
                "uri": OFFLINE_ASR_TCP_URI,
                "critical": False,
            }
        except asyncio.TimeoutError:
            health_status["services"]["speech_to_text"] = {
                "status": f"⚠️ Connection Timeout (5s) - Check external ASR service",
                "healthy": False,
                "type": "Offline TCP",
                "provider": "Wyoming ASR",
                "uri": OFFLINE_ASR_TCP_URI,
                "critical": False,
            }
            overall_healthy = False
        except Exception as e:
            health_status["services"]["speech_to_text"] = {
                "status": f"⚠️ Connection Failed: {str(e)} - Check external ASR service",
                "healthy": False,
                "type": "Offline TCP",
                "provider": "Wyoming ASR",
                "uri": OFFLINE_ASR_TCP_URI,
                "critical": False,
            }
            overall_healthy = False
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

    # Track health check results in debug tracker
    try:
        tracker = get_debug_tracker()
        # Can add health check tracking to debug tracker if needed
        pass
    except Exception as e:
        audio_logger.error(f"Failed to record health check metrics: {e}")

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


@app.get("/readiness")
async def readiness_check():
    """Simple readiness check for container orchestration."""
    return JSONResponse(content={"status": "ready", "timestamp": int(time.time())}, status_code=200)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    audio_logger.info("Starting Omi unified service at ws://%s:%s/ws", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=False)
