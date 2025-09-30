"""
Audio processing data types for unified pipeline.

Provides common data structures for both WebSocket and file upload processing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import uuid

from .job_tracker import AudioSource


@dataclass
class AudioProcessingItem:
    """Common data structure for all audio processing (WebSocket and file upload)."""

    # Identifiers
    audio_uuid: str
    user_id: str
    user_email: str

    # Audio source information
    audio_source: AudioSource  # WEBSOCKET or FILE_UPLOAD
    client_id: Optional[str] = None        # For websocket processing
    device_name: Optional[str] = None      # For file upload processing

    # Audio data (one of these will be set)
    audio_chunks: Optional[List[bytes]] = None     # For websocket (buffered chunks)
    audio_file_path: Optional[str] = None          # For file upload

    # Audio format information
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 2 bytes = 16-bit

    # Processing metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None

    @classmethod
    def from_websocket(
        cls,
        audio_chunks: List[bytes],
        client_id: str,
        user_id: str,
        user_email: str,
        sample_rate: int = 16000
    ) -> "AudioProcessingItem":
        """Create from WebSocket audio chunks."""
        return cls(
            audio_uuid=str(uuid.uuid4()),
            user_id=user_id,
            user_email=user_email,
            audio_source=AudioSource.WEBSOCKET,
            client_id=client_id,
            audio_chunks=audio_chunks,
            sample_rate=sample_rate
        )

    @classmethod
    def from_file_upload(
        cls,
        audio_file_path: str,
        client_id: str,
        device_name: str,
        user_id: str,
        user_email: str
    ) -> "AudioProcessingItem":
        """Create from uploaded file."""
        file_path = Path(audio_file_path)
        return cls(
            audio_uuid=str(uuid.uuid4()),
            user_id=user_id,
            user_email=user_email,
            audio_source=AudioSource.FILE_UPLOAD,
            client_id=client_id,
            device_name=device_name,
            audio_file_path=audio_file_path,
            file_size_bytes=file_path.stat().st_size if file_path.exists() else None
        )

    def get_identifier(self) -> str:
        """Get the appropriate identifier for this processing item."""
        if self.audio_source == AudioSource.WEBSOCKET:
            return self.client_id
        else:
            return self.device_name or "file_upload"


@dataclass
class TranscriptionItem:
    """Data for transcription processing."""
    audio_uuid: str
    audio_file_path: str
    client_id: str
    user_id: str
    user_email: str
    job_id: Optional[str] = None
    audio_chunk: Optional[any] = None  # For legacy transcription flow


@dataclass
class MemoryProcessingItem:
    """Data for memory processing."""
    conversation_id: str
    user_id: str
    user_email: str
    client_id: str  # Required for memory service
    transcript_version_id: Optional[str] = None  # Use None for active version
    job_id: Optional[str] = None


@dataclass
class CroppingItem:
    """Data for audio cropping/optimization."""
    audio_uuid: str
    audio_file_path: str
    segments: List
    job_id: Optional[str] = None