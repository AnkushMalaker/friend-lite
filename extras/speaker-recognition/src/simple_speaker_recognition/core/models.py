"""Pydantic models for speaker recognition API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class EnrollRequest(BaseModel):
    """Request model for speaker enrollment."""
    speaker_id: str
    speaker_name: str
    start: Optional[float] = None
    end: Optional[float] = None


class BatchEnrollRequest(BaseModel):
    """Request model for batch speaker enrollment."""
    speaker_id: str
    speaker_name: str


class IdentifyRequest(BaseModel):
    """Request model for speaker identification."""
    start: Optional[float] = None
    end: Optional[float] = None


class VerifyRequest(BaseModel):
    """Request model for speaker verification."""
    speaker_id: str
    start: Optional[float] = None
    end: Optional[float] = None


class DiarizeRequest(BaseModel):
    """Request model for speaker diarization."""
    min_duration: Optional[float] = Field(default=None, description="Minimum duration for speaker segments (seconds)")
    max_speakers: Optional[int] = Field(default=None, description="Maximum number of speakers to detect")


class DiarizeAndIdentifyRequest(BaseModel):
    """Request model for combined diarization and identification."""
    min_duration: Optional[float] = Field(default=0.5, description="Minimum duration for speaker segments (seconds)")
    similarity_threshold: Optional[float] = Field(default=None, description="Override default similarity threshold for identification")
    identify_only_enrolled: bool = Field(default=False, description="Only return segments for enrolled speakers")


class InferenceRequest(BaseModel):
    """Request model for speaker inference on diarized segments."""
    segments: List[dict] = Field(..., description="Diarized transcript segments with speaker, start, end, text")


class InferenceSegment(BaseModel):
    """Model for enhanced transcript segment with speaker identification."""
    speaker: int = Field(..., description="Original speaker ID from diarization")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds") 
    text: str = Field(..., description="Transcript text")
    identified_speaker: Optional[str] = Field(None, description="Identified speaker name")
    confidence: Optional[float] = Field(None, description="Identification confidence score")