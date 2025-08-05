"""FastAPI service for speaker recognition and diarization - fully refactored."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import torch
import uvicorn
from fastapi import Depends, FastAPI
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
    deepgram_api_key: str | None = Field(default=None, description="Deepgram API key for wrapper service")
    deepgram_base_url: str = Field(default="https://api.deepgram.com", description="Deepgram API base URL")

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

# Global variables for storing initialized resources
audio_backend: AudioBackend
speaker_db: UnifiedSpeakerDB
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
    
    # Yield control to the application
    yield
    
    # Shutdown: Clean up resources if needed
    log.info("Shutting down speaker recognition service")


app = FastAPI(title="Simple Speaker Recognition Service", version="0.2.0", lifespan=lifespan)

# Import and include all routers
from .routers import (
    users_router,
    speakers_router,
    enrollment_router,
    identification_router,
    deepgram_router
)

# Include routers with appropriate tags and prefixes
app.include_router(users_router, tags=["users"])
app.include_router(speakers_router, tags=["speakers"])  
app.include_router(enrollment_router, tags=["enrollment"])
app.include_router(identification_router, tags=["identification"])
app.include_router(deepgram_router, tags=["deepgram"])


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