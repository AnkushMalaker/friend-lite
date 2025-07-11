"""
Data models and schemas for the Friend Lite application.
Contains essential Pydantic models for API requests/responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# API Request/Response models
class SpeakerAssignment(BaseModel):
    """Model for assigning speakers to conversations."""
    speaker: str = Field(..., description="Speaker identifier")

class TranscriptUpdate(BaseModel):
    """Model for updating transcript segments."""
    text: str = Field(..., description="Updated transcript text")

class CloseConversationRequest(BaseModel):
    """Model for closing a conversation."""
    client_id: str = Field(..., description="Client identifier")

# Client state models
class ClientStateInfo(BaseModel):
    """Information about client state."""
    client_id: str
    connected: bool
    current_audio_uuid: Optional[str] = None
    sample_count: int = 0
    conversation_transcripts: int = 0

# Health and status models
class HealthStatus(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    timestamp: float = Field(..., description="Check timestamp")
    services: Optional[Dict[str, str]] = None
    environment: Optional[Dict[str, Any]] = None

# Error models
class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = None
    code: Optional[str] = None
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())