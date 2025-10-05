"""
Application configuration for Friend-Lite backend.

Centralizes all application-level configuration including database connections,
service configurations, and environment variables that were previously in main.py.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

from advanced_omi_backend.constants import OMI_CHANNELS, OMI_SAMPLE_RATE, OMI_SAMPLE_WIDTH
from advanced_omi_backend.services.transcription import get_transcription_provider

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AppConfig:
    """Centralized application configuration."""

    def __init__(self):
        # MongoDB Configuration
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
        self.mongo_client = AsyncIOMotorClient(self.mongodb_uri)
        self.db = self.mongo_client.get_default_database("friend-lite")
        self.chunks_col = self.db["audio_chunks"]
        self.users_col = self.db["users"]
        self.speakers_col = self.db["speakers"]

        # Audio Configuration
        self.segment_seconds = 60  # length of each stored chunk
        self.target_samples = OMI_SAMPLE_RATE * self.segment_seconds
        self.audio_chunk_dir = Path("./audio_chunks")
        self.audio_chunk_dir.mkdir(parents=True, exist_ok=True)

        # Conversation timeout configuration
        self.new_conversation_timeout_minutes = float(
            os.getenv("NEW_CONVERSATION_TIMEOUT_MINUTES", "1.5")
        )

        # Audio cropping configuration
        self.audio_cropping_enabled = os.getenv("AUDIO_CROPPING_ENABLED", "true").lower() == "true"
        self.min_speech_segment_duration = float(os.getenv("MIN_SPEECH_SEGMENT_DURATION", "1.0"))
        self.cropping_context_padding = float(os.getenv("CROPPING_CONTEXT_PADDING", "0.1"))

        # Transcription Configuration
        self.transcription_provider_name = os.getenv("TRANSCRIPTION_PROVIDER")
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")

        # Get configured transcription provider
        self.transcription_provider = get_transcription_provider(self.transcription_provider_name)
        if self.transcription_provider:
            logger.info(
                f"✅ Using {self.transcription_provider.name} transcription provider ({self.transcription_provider.mode})"
            )
        else:
            logger.warning("⚠️ No transcription provider configured - speech-to-text will not be available")

        # External Services Configuration
        self.qdrant_base_url = os.getenv("QDRANT_BASE_URL", "qdrant")
        self.qdrant_port = os.getenv("QDRANT_PORT", "6333")
        self.memory_provider = os.getenv("MEMORY_PROVIDER", "friend_lite").lower()

        # Redis Configuration
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # CORS Configuration
        default_origins = "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3002"
        self.cors_origins = os.getenv("CORS_ORIGINS", default_origins)
        self.allowed_origins = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

        # Tailscale support
        self.tailscale_regex = r"http://100\.\d{1,3}\.\d{1,3}\.\d{1,3}:3000"

        # Thread pool configuration
        self.max_workers = os.cpu_count() or 4

        # Memory service configuration
        self.memory_service_supports_threshold = self.memory_provider == "friend_lite"


# Global configuration instance
app_config = AppConfig()


def get_app_config() -> AppConfig:
    """Get the global application configuration instance."""
    return app_config


def get_audio_chunk_dir() -> Path:
    """Get the audio chunk directory."""
    return app_config.audio_chunk_dir


def get_mongo_collections():
    """Get MongoDB collections."""
    return {
        'chunks': app_config.chunks_col,
        'users': app_config.users_col,
        'speakers': app_config.speakers_col,
    }


def get_redis_config():
    """Get Redis configuration."""
    return {
        'url': app_config.redis_url,
        'encoding': "utf-8",
        'decode_responses': False
    }