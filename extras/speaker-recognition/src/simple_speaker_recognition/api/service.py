"""FastAPI service for speaker recognition and diarization - fully refactored."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast, Optional, Union

import torch
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field
from pydantic_settings import BaseSettings

from simple_speaker_recognition.api.core.utils import get_data_directory
from simple_speaker_recognition.core.audio_backend import AudioBackend
from simple_speaker_recognition.core.unified_speaker_db import UnifiedSpeakerDB
from simple_speaker_recognition.database import init_db

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("speaker_service")


class Settings(BaseSettings):
    """Service configuration settings."""
    similarity_threshold: float = Field(default=0.15, description="Cosine similarity threshold for speaker identification (0.1-0.3 typical for ECAPA-TDNN)")
    data_dir: Path = Field(default_factory=get_data_directory, description="Directory for storing speaker data")
    enrollment_audio_dir: Path = Field(default_factory=lambda: get_data_directory() / "enrollment_audio", description="Directory for storing enrollment audio files")
    max_file_seconds: int = Field(default=180, description="Maximum file duration in seconds")
    deepgram_api_key: Optional[str] = Field(default=None, description="Deepgram API key for wrapper service")
    deepgram_base_url: str = Field(default="https://api.deepgram.com", description="Deepgram API base URL")
    hf_token: Optional[str] = Field(default=None, description="Hugging Face token for Pyannote models")

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables


# Get HF_TOKEN from environment and create settings
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is required. Please set it before running the service.")

hf_token = cast(str, hf_token)
auth = Settings()  # Load other settings from env vars or .env file

# Override Deepgram API key from environment if available
if os.getenv("DEEPGRAM_API_KEY"):
    auth.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

# Set HF token in auth settings for consistency
auth.hf_token = hf_token

# Global variables for storing initialized resources
audio_backend: AudioBackend
speaker_db: UnifiedSpeakerDB
# Device selection with environment override
log.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    log.info(f"CUDA device count: {torch.cuda.device_count()}")
    log.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    log.info(f"CUDA version: {torch.version.cuda}")

compute_mode = os.getenv("COMPUTE_MODE", "cpu").lower()
if compute_mode == "gpu" and torch.cuda.is_available():
    device = torch.device("cuda")
    log.info("Using GPU mode via COMPUTE_MODE=gpu environment variable")
elif compute_mode == "gpu" and not torch.cuda.is_available():
    device = torch.device("cpu")
    log.warning("COMPUTE_MODE=gpu requested but CUDA not available, falling back to CPU")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler for startup and shutdown."""
    global audio_backend, speaker_db
    
    # Startup: Initialize database and load models
    log.info("=== Speaker Recognition Service Starting ===")
    log.info("Version: 2025-08-05-refactored")
    log.info("This version uses modular router architecture")
    log.info("Initializing database...")
    init_db()
    
    log.info("Loading models...")
    assert hf_token is not None
    audio_backend = AudioBackend(hf_token, device)
    speaker_db = UnifiedSpeakerDB(
        emb_dim=audio_backend.embedder.dimension,
        base_dir=auth.data_dir,
        similarity_thr=auth.similarity_threshold,
    )
    log.info("Models ready ✔ – device=%s", device)
    
    # Ensure enrollment audio directory exists
    auth.enrollment_audio_dir.mkdir(parents=True, exist_ok=True)
    log.info("Enrollment audio directory ready: %s", auth.enrollment_audio_dir)
    
    # Ensure admin user exists
    admin_user_id = speaker_db.ensure_admin_user()
    log.info("Admin user ready ✔ – user_id=%s", admin_user_id)
    
    # Yield control to the application
    yield
    
    # Shutdown: Clean up resources if needed
    log.info("Shutting down speaker recognition service")


app = FastAPI(title="Simple Speaker Recognition Service", version="0.2.0", lifespan=lifespan)

react_ui_host = os.getenv("REACT_UI_HOST", "localhost") + ":" + os.getenv("REACT_UI_PORT", "5173")
# Add CORS middleware for direct WebSocket connections from HTTPS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[react_ui_host, "https://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include all routers
from .routers import (
    users_router,
    speakers_router,
    enrollment_router,
    identification_router,
    deepgram_router,
    websocket_router
)

# Include routers with appropriate tags and prefixes
app.include_router(users_router, tags=["users"])
app.include_router(speakers_router, tags=["speakers"])  
app.include_router(enrollment_router, tags=["enrollment"])
app.include_router(identification_router, tags=["identification"])
app.include_router(deepgram_router, tags=["deepgram"])
app.include_router(websocket_router, tags=["websocket"])


async def get_db() -> UnifiedSpeakerDB:
    """Get speaker database dependency."""
    return speaker_db


@app.get("/health")
async def health(db: UnifiedSpeakerDB = Depends(get_db)):
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.2.0-refactored",
        "device": str(device),
        "speakers": db.get_speaker_count(),
        "architecture": "modular-routers"
    }


def main():
    """Main entry point for the service."""
    host = os.getenv("SPEAKER_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("SPEAKER_SERVICE_PORT", "8085"))
    
    log.info(f"Starting Refactored Speaker Service on {host}:{port}")
    uvicorn.run("simple_speaker_recognition.api.service:app", host=host, port=port, reload=bool(os.getenv("DEV", False)))


if __name__ == "__main__":
    main()