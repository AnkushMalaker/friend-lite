"""
AudioChunk models for Friend-Lite backend.

This module contains the Beanie Document model for audio_chunks collection,
which stores ALL audio sessions (both with and without speech). This is the
storage layer - all audio gets stored here with its metadata.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from beanie import Document, Indexed


class AudioChunk(Document):
    """
    Audio chunk model representing all audio sessions.

    The audio_chunks collection stores ALL audio sessions - with or without speech.
    It contains transcript and memory versions for backward compatibility with
    existing data, though new speech-detected sessions will also create a
    Conversation document.
    """
    

    # Core identifiers
    audio_uuid: Indexed(str, unique=True) = Field(description="Unique audio identifier")
    audio_path: str = Field(description="Path to audio file")
    client_id: Indexed(str) = Field(description="Client device identifier")
    timestamp: Indexed(int) = Field(description="Unix timestamp in milliseconds")

    # User information
    user_id: Indexed(str) = Field(description="User who owns this audio")


    # Legacy compatibility fields
    transcript: List[Dict[str, Any]] = Field(default_factory=list, description="Legacy transcript field")
    speakers_identified: List[str] = Field(default_factory=list, description="Legacy speakers field")
    memories: List[Dict[str, Any]] = Field(default_factory=list, description="Legacy memories field")
    transcription_status: str = Field(default="PENDING", description="Legacy transcription status")
    memory_processing_status: str = Field(default="PENDING", description="Legacy memory status")
    raw_transcript_data: Dict[str, Any] = Field(default_factory=dict, description="Raw transcript data")

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