"""
Data models for Friend-Lite backend.

This module contains Pydantic models that define the structure and validation
for all data entities in the Friend-Lite system.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class ProcessingStatus(str, Enum):
    """Status enum for processing operations."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EMPTY = "EMPTY"


class TranscriptSegment(BaseModel):
    """Individual transcript segment with speaker and timing."""
    text: str
    speaker: str
    start: float
    end: float
    speaker_id: Optional[str] = None
    confidence: Optional[float] = None


class TranscriptVersion(BaseModel):
    """A versioned transcript processing result."""
    version_id: str
    segments: List[TranscriptSegment] = []
    status: ProcessingStatus = ProcessingStatus.PENDING
    provider: Optional[str] = None
    model: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None
    processing_run_id: Optional[str] = None
    raw_data: Dict[str, Any] = {}
    speakers_identified: List[str] = []
    error_message: Optional[str] = None


class MemoryEntry(BaseModel):
    """Individual memory/fact extracted from conversation."""
    text: str
    category: Optional[str] = None
    confidence: Optional[float] = None
    created_at: datetime
    metadata: Dict[str, Any] = {}


class MemoryVersion(BaseModel):
    """A versioned memory extraction result."""
    version_id: str
    memories: List[MemoryEntry] = []
    memory_count: int = 0
    status: ProcessingStatus = ProcessingStatus.PENDING
    provider: Optional[str] = None
    model: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None
    processing_run_id: Optional[str] = None
    transcript_version_id: Optional[str] = None
    error_message: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Auto-calculate memory_count from memories list
        self.memory_count = len(self.memories)


class ActionItem(BaseModel):
    """Action item extracted from conversation."""
    text: str
    category: Optional[str] = None
    priority: Optional[str] = None
    due_date: Optional[datetime] = None
    completed: bool = False
    created_at: datetime


class VersionInfo(BaseModel):
    """Version metadata for UI display."""
    transcript_count: int = 0
    memory_count: int = 0
    active_transcript_version: Optional[str] = None
    active_memory_version: Optional[str] = None


class Conversation(BaseModel):
    """
    Main conversation model representing a user-facing conversation.

    This model implements the speech-driven architecture where conversations
    are only created when speech is detected, and supports versioned
    transcript and memory processing.
    """
    # Core identifiers
    conversation_id: str
    audio_uuid: str  # Link to audio_chunks collection
    user_id: str
    client_id: str

    # Basic conversation metadata
    title: str = "Conversation"
    summary: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    session_start: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Speaker information
    speaker_names: Dict[str, str] = {}

    # Versioned transcript system
    transcript_versions: List[TranscriptVersion] = []
    active_transcript_version: Optional[str] = None

    # Versioned memory system
    memory_versions: List[MemoryVersion] = []
    active_memory_version: Optional[str] = None

    # Action items and tasks
    action_items: List[ActionItem] = []

    # Processing status (for backward compatibility)
    memory_processing_status: ProcessingStatus = ProcessingStatus.PENDING

    # Primary fields (auto-populated from active versions)
    transcript: List[TranscriptSegment] = []
    speakers_identified: List[str] = []
    memories: List[MemoryEntry] = []

    # Audio file paths (from audio_chunks)
    audio_path: Optional[str] = None
    cropped_audio_path: Optional[str] = None

    def get_version_info(self) -> VersionInfo:
        """Get version metadata for UI."""
        return VersionInfo(
            transcript_count=len(self.transcript_versions),
            memory_count=len(self.memory_versions),
            active_transcript_version=self.active_transcript_version,
            active_memory_version=self.active_memory_version
        )

    def get_active_transcript_version(self) -> Optional[TranscriptVersion]:
        """Get the currently active transcript version."""
        if not self.active_transcript_version:
            return None
        for version in self.transcript_versions:
            if version.version_id == self.active_transcript_version:
                return version
        return None

    def get_active_memory_version(self) -> Optional[MemoryVersion]:
        """Get the currently active memory version."""
        if not self.active_memory_version:
            return None
        for version in self.memory_versions:
            if version.version_id == self.active_memory_version:
                return version
        return None

    def populate_primary_fields(self) -> None:
        """Populate primary fields from active versions."""
        # Clear existing primary fields
        self.transcript = []
        self.speakers_identified = []
        self.memories = []

        # Populate from active transcript version
        active_transcript = self.get_active_transcript_version()
        if active_transcript:
            self.transcript = active_transcript.segments
            self.speakers_identified = active_transcript.speakers_identified

        # Populate from active memory version
        active_memory = self.get_active_memory_version()
        if active_memory:
            self.memories = active_memory.memories
            self.memory_processing_status = active_memory.status

    def activate_transcript_version(self, version_id: str) -> bool:
        """Activate a specific transcript version."""
        # Verify version exists
        for version in self.transcript_versions:
            if version.version_id == version_id:
                self.active_transcript_version = version_id
                self.populate_primary_fields()
                return True
        return False

    def activate_memory_version(self, version_id: str) -> bool:
        """Activate a specific memory version."""
        # Verify version exists
        for version in self.memory_versions:
            if version.version_id == version_id:
                self.active_memory_version = version_id
                self.populate_primary_fields()
                return True
        return False

    class Config:
        """Pydantic configuration."""
        # Allow datetime objects
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
        # Use enum values in JSON
        use_enum_values = True


class ConversationListItem(BaseModel):
    """Lightweight conversation model for list views."""
    conversation_id: str
    audio_uuid: str
    client_id: str
    title: str = "Conversation"
    summary: Optional[str] = None
    timestamp: float  # Unix timestamp for compatibility
    created_at: Optional[str] = None  # ISO string
    duration_seconds: float = 0.0
    has_memory: bool = False
    memory_processing_status: ProcessingStatus = ProcessingStatus.PENDING

    # Version information
    version_info: VersionInfo = VersionInfo()

    # Primary fields for display
    transcript: List[TranscriptSegment] = []
    speakers_identified: List[str] = []
    memories: List[MemoryEntry] = []
    action_items: List[ActionItem] = []

    # Audio file paths
    audio_path: Optional[str] = None
    cropped_audio_path: Optional[str] = None
    debug_audio_url: Optional[str] = None


class ProcessingRun(BaseModel):
    """Processing run tracking model."""
    run_id: str
    conversation_id: str
    audio_uuid: str
    run_type: str  # 'transcript' or 'memory'
    user_id: str
    trigger: str  # 'manual_reprocess', 'initial_processing', etc.
    status: ProcessingStatus = ProcessingStatus.PENDING
    config_hash: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_version_id: Optional[str] = None


# API Response Models
class ConversationResponse(BaseModel):
    """API response for single conversation."""
    conversation: Conversation


class ConversationsResponse(BaseModel):
    """API response for conversations list."""
    conversations: Dict[str, List[ConversationListItem]]


class VersionHistoryResponse(BaseModel):
    """API response for version history."""
    conversation_id: str
    active_transcript_version: Optional[str]
    active_memory_version: Optional[str]
    transcript_versions: List[TranscriptVersion]
    memory_versions: List[MemoryVersion]