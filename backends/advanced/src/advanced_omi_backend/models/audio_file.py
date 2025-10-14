"""
AudioFile models for Friend-Lite backend.

This module contains the Beanie Document model for audio_chunks collection,
which stores ALL audio files (both with and without speech). This is the
storage layer - all audio gets stored here with its metadata.

Note: Named AudioFile (not AudioChunk) to avoid confusion with wyoming.audio.AudioChunk
which is the in-memory streaming audio data structure.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from beanie import Document, Indexed


class AudioFile(Document):
    """
    Audio file model representing persisted audio files in MongoDB.

    The audio_chunks collection stores ALL raw audio files (both with and without speech).
    This is just for audio file storage and metadata. If speech is detected, a
    Conversation document is created which contains transcripts and memories.

    This is different from wyoming.audio.AudioChunk which is for streaming audio data.
    """

    # Core identifiers
    audio_uuid: Indexed(str, unique=True) = Field(description="Unique audio identifier")
    audio_path: str = Field(description="Path to raw audio file")
    client_id: Indexed(str) = Field(description="Client device identifier")
    timestamp: Indexed(int) = Field(description="Unix timestamp in milliseconds")

    # User information
    user_id: Indexed(str) = Field(description="User who owns this audio")
    user_email: Optional[str] = Field(None, description="User email")

    # Audio processing
    cropped_audio_path: Optional[str] = Field(None, description="Path to cropped audio (speech only)")

    # Speech-driven conversation linking
    conversation_id: Optional[str] = Field(
        None,
        description="Link to Conversation if speech was detected"
    )
    has_speech: bool = Field(default=False, description="Whether speech was detected")
    speech_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Speech detection results"
    )

    class Settings:
        name = "audio_chunks"
        indexes = [
            "audio_uuid",
            "client_id",
            "user_id",
            "timestamp"
        ]