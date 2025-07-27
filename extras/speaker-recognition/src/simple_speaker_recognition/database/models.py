"""SQLAlchemy models for speaker recognition system."""

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from . import Base

class User(Base):
    """User model for multi-user support."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    speakers = relationship("Speaker", back_populates="user", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"

class Speaker(Base):
    """Speaker profile model."""
    __tablename__ = "speakers"
    
    id = Column(String(100), primary_key=True)  # User-defined speaker ID
    name = Column(String(200), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    embedding_version = Column(Integer, default=1)
    embedding_config = Column(Text)  # JSON: {"method": "mean", "params": {...}}
    embedding_data = Column(Text)  # JSON: serialized embedding vector
    audio_segments_metadata = Column(Text)  # JSON: references to audio segments
    notes = Column(Text)  # Optional notes about the speaker
    
    # Relationships
    user = relationship("User", back_populates="speakers")
    enrollment_sessions = relationship("EnrollmentSession", back_populates="speaker", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="speaker")
    audio_segments = relationship("SpeakerAudioSegment", back_populates="speaker", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Speaker(id='{self.id}', name='{self.name}', user_id={self.user_id})>"

class EnrollmentSession(Base):
    """Enrollment session tracking model."""
    __tablename__ = "enrollment_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    speaker_id = Column(String(100), ForeignKey("speakers.id"), nullable=False)
    audio_file_path = Column(String(500), nullable=False)
    duration_seconds = Column(Float)  # Total audio duration
    speech_duration_seconds = Column(Float)  # Actual speech duration
    quality_score = Column(Float)  # Overall quality score (0.0 to 1.0)
    snr_db = Column(Float)  # Signal-to-noise ratio in dB
    created_at = Column(DateTime, default=datetime.utcnow)
    enrollment_method = Column(String(50))  # 'live_recording' or 'file_upload'
    
    # Relationships
    speaker = relationship("Speaker", back_populates="enrollment_sessions")
    
    def __repr__(self):
        return f"<EnrollmentSession(id={self.id}, speaker_id='{self.speaker_id}', quality={self.quality_score})>"

class SpeakerAudioSegment(Base):
    """Individual audio segments for speakers."""
    __tablename__ = "speaker_audio_segments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    speaker_id = Column(String(100), ForeignKey("speakers.id"), nullable=False)
    audio_file_path = Column(String(500), nullable=False)  # Path to segment file
    original_file_path = Column(String(500))  # Original source file
    start_time = Column(Float, nullable=False)  # Start time in original file
    end_time = Column(Float, nullable=False)    # End time in original file
    duration_seconds = Column(Float, nullable=False)
    quality_score = Column(Float)  # Segment quality (0.0 to 1.0)
    embedding = Column(Text)  # JSON: Individual segment embedding
    transcription = Column(Text)  # Transcribed text
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    speaker = relationship("Speaker", back_populates="audio_segments")
    
    def __repr__(self):
        return f"<SpeakerAudioSegment(id={self.id}, speaker_id='{self.speaker_id}', duration={self.duration_seconds})>"

class Annotation(Base):
    """Audio segment annotation model."""
    __tablename__ = "annotations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    audio_file_path = Column(String(500), nullable=False)
    audio_file_hash = Column(String(32))  # MD5 hash of audio file for consistent identification
    audio_file_name = Column(String(255))  # Original filename for display
    start_time = Column(Float, nullable=False)  # Start time in seconds
    end_time = Column(Float, nullable=False)    # End time in seconds
    speaker_id = Column(String(100), ForeignKey("speakers.id"), nullable=True)  # Can be NULL for unknown speakers
    speaker_label = Column(String(100))  # For unknown speakers: "unknown_1", "unknown_2", etc.
    deepgram_speaker_label = Column(String(50))  # Original Deepgram label: "speaker_0", "speaker_1", etc.
    label = Column(String(20), nullable=False)  # 'CORRECT', 'INCORRECT', 'UNCERTAIN'
    confidence = Column(Float)  # Model confidence in the annotation (0.0 to 1.0)
    transcription = Column(Text)  # Text content of the segment
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)  # Optional annotation notes
    
    # Relationships
    speaker = relationship("Speaker", back_populates="annotations")
    user = relationship("User", back_populates="annotations")
    
    def __repr__(self):
        return f"<Annotation(id={self.id}, speaker_id='{self.speaker_id}', label='{self.label}')>"

class ProcessingJob(Base):
    """Background processing job tracking."""
    __tablename__ = "processing_jobs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_type = Column(String(50), nullable=False)  # 'enrollment', 'export', 'annotation', etc.
    status = Column(String(20), nullable=False, default='pending')  # 'pending', 'running', 'completed', 'failed'
    input_data = Column(Text)  # JSON string of input parameters
    output_data = Column(Text)  # JSON string of results
    progress = Column(Float, default=0.0)  # Progress percentage (0.0 to 100.0)
    error_message = Column(Text)  # Error details if status is 'failed'
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    def __repr__(self):
        return f"<ProcessingJob(id={self.id}, type='{self.job_type}', status='{self.status}')>"

class ExportHistory(Base):
    """Export operation history."""
    __tablename__ = "export_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    export_type = Column(String(50), nullable=False)  # 'single_speaker', 'bulk', 'annotations'
    format_type = Column(String(20), nullable=False)  # 'concatenated', 'segments', 'metadata'
    file_path = Column(String(500))  # Path to generated export file
    file_size_bytes = Column(Integer)
    speaker_ids = Column(Text)  # JSON array of exported speaker IDs
    created_at = Column(DateTime, default=datetime.utcnow)
    downloaded_at = Column(DateTime)  # When the file was downloaded
    cleaned_up = Column(Boolean, default=False)  # Whether temp files were cleaned up
    
    # Relationships
    user = relationship("User")
    
    def __repr__(self):
        return f"<ExportHistory(id={self.id}, type='{self.export_type}', user_id={self.user_id})>"