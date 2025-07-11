"""
Friend Lite Advanced Backend - Main Application
A clean, modular main application file that orchestrates all components.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api_routes import router as api_router
from deps import current_user_ws

# Authentication and user management
from auth import (
    bearer_backend,
    cookie_backend,
    create_admin_user_if_needed,
    fastapi_users,
)

# Application modules
from services import get_config, initialize_all_services
from websocket_handler import (
    handle_pcm_websocket_connection,
    handle_websocket_connection,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("🚀 Starting Friend Lite Advanced Backend")
    
    try:
        # Initialize all services
        services = await initialize_all_services()
        logger.info("✅ All services initialized successfully")
        
        # Create admin user if needed
        await create_admin_user_if_needed()
        logger.info("👤 Admin user setup completed")
        
        # Store services in app state
        app.state.services = services
        
        # Register service routers
        register_service_routers(app)
        logger.info("📡 Service routers registered successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise
    finally:
        logger.info("🛑 Shutting down Friend Lite Advanced Backend")

# Create FastAPI application
app = FastAPI(
    title="Friend Lite Advanced Backend",
    description="Advanced backend for Friend Lite with real-time audio processing",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get configuration
config = get_config()

# Mount static files
app.mount("/audio", StaticFiles(directory=config["CHUNK_DIR"]), name="audio")

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

# Add memory debug router
try:
    from memory_debug_api import debug_router
    app.include_router(debug_router)
except ImportError:
    logger.warning("Memory debug router not available")

# Add main API router
app.include_router(api_router)

# Register service routers
def register_service_routers(app: FastAPI):
    """Register all service routers with the FastAPI app."""
    try:
        services = app.state.services
        transcript_service_manager = services.get("transcript_service_manager")
        
        if transcript_service_manager:
            routers = transcript_service_manager.get_all_routers()
            for service_name, router in routers:
                app.include_router(router, tags=[f"{service_name}-service"])
                logger.info(f"📡 Registered API router for {service_name} service")
        else:
            logger.warning("Transcript service manager not available for router registration")
    except Exception as e:
        logger.error(f"Failed to register service routers: {e}")

# WebSocket endpoints
@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(..., description="Unique client ID"),
    user = Depends(current_user_ws),          # ← injects authenticated user
):
    """WebSocket endpoint for Opus audio streaming."""
    await websocket.accept()                        # only reached when auth passed
    try:
        await handle_websocket_connection(websocket, client_id, user)
    except WebSocketDisconnect:
        logger.info("WS disconnected: %s (%s)", client_id, user.id)

@app.websocket("/ws_pcm")
async def websocket_pcm_endpoint(
    websocket: WebSocket,
    client_id: str = Query(...),
    user = Depends(current_user_ws),
):
    """WebSocket endpoint for PCM audio streaming with JSON messages."""
    await websocket.accept()
    try:
        await handle_pcm_websocket_connection(websocket, client_id, user)
    except WebSocketDisconnect:
        logger.info("PCM WS disconnected: %s (%s)", client_id, user.id)



# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic application information."""
    return {
        "name": "Friend Lite Advanced Backend",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn

    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"🌟 Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )