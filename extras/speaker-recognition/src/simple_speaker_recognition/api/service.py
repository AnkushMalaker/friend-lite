"""FastAPI service for speaker recognition and diarization."""

import json
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import aiohttp
import numpy as np
import torch
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import Field
from pydantic_settings import BaseSettings
from simple_speaker_recognition.core.audio_backend import AudioBackend
from simple_speaker_recognition.core.models import (
    BatchEnrollRequest,
    DeepgramTranscriptionRequest,
    DiarizeAndIdentifyRequest,
    DiarizeRequest,
    EnhancedTranscriptionResponse,
    IdentifyRequest,
    IdentifyResponse,
    SpeakerStatus,
    UserRequest,
    UserResponse,
    VerifyRequest,
)
from simple_speaker_recognition.core.unified_speaker_db import UnifiedSpeakerDB
from simple_speaker_recognition.database import get_db_session, init_db
from simple_speaker_recognition.database.models import Speaker
from simple_speaker_recognition.utils.audio_processing import get_audio_info


def get_data_directory() -> Path:
    """Get the appropriate data directory.
    
    Uses Docker path if available (/app/data), otherwise falls back to local development path.
    This pattern matches the database module's approach.
    """
    docker_path = Path("/app/data")
    if docker_path.exists():
        return docker_path
    else:
        # For local development, use 'data' directory relative to project root
        return Path("data")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("speaker_service")


def safe_format_confidence(confidence: Any, context: str = "") -> str:
    """Safely format confidence values with comprehensive validation and logging.
    
    Args:
        confidence: The confidence value to format (can be any type)
        context: Context string for logging (e.g., "speaker_identification", "audio_processing")
    
    Returns:
        str: Safely formatted confidence string (e.g., "0.856" or "invalid")
    """
    try:
        # Log the raw input for debugging
        log.debug(f"safe_format_confidence called with: type={type(confidence)}, value={confidence}, context='{context}'")
        
        # Handle None values
        if confidence is None:
            log.debug(f"Confidence is None in context '{context}', returning '0.000'")
            return "0.000"
        
        # Convert to Python float, handling various input types
        if isinstance(confidence, (int, float)):
            conf_val = float(confidence)
        elif isinstance(confidence, np.number):
            # Handle numpy scalars (including float32, float64, etc.)
            conf_val = float(confidence.item())
            log.debug(f"Converted numpy {type(confidence)} to Python float: {conf_val}")
        elif hasattr(confidence, '__float__'):
            # Handle objects that can be converted to float
            conf_val = float(confidence)
            log.debug(f"Converted {type(confidence)} to float via __float__: {conf_val}")
        else:
            log.warning(f"Cannot convert confidence type {type(confidence)} to float in context '{context}': {confidence}")
            return "invalid"
        
        # Check for special float values
        if np.isnan(conf_val):
            log.warning(f"Confidence is NaN in context '{context}', returning '0.000'")
            return "0.000"
        elif np.isinf(conf_val):
            log.warning(f"Confidence is infinite ({conf_val}) in context '{context}', returning '1.000' or '0.000'")
            return "1.000" if conf_val > 0 else "0.000"
        elif conf_val < 0:
            log.warning(f"Confidence is negative ({conf_val}) in context '{context}', clamping to 0.000")
            return "0.000"
        elif conf_val > 1:
            log.warning(f"Confidence is > 1 ({conf_val}) in context '{context}', clamping to 1.000")
            return "1.000"
        
        # Format the valid confidence value
        formatted = f"{conf_val:.3f}"
        log.debug(f"Successfully formatted confidence {conf_val} -> '{formatted}' in context '{context}'")
        return formatted
        
    except Exception as e:
        log.error(f"Exception in safe_format_confidence for {confidence} (type: {type(confidence)}) in context '{context}': {e}")
        return "error"


class Settings(BaseSettings):
    """Service configuration settings."""
    similarity_threshold: float = Field(default=0.15, description="Cosine similarity threshold for speaker identification (0.1-0.3 typical for ECAPA-TDNN)")
    data_dir: Path = Field(default_factory=get_data_directory, description="Directory for storing speaker data")
    enrollment_audio_dir: Path = Field(default_factory=lambda: get_data_directory() / "enrollment_audio", description="Directory for storing enrollment audio files")
    max_file_seconds: int = Field(default=180, description="Maximum file duration in seconds")
    deepgram_api_key: Optional[str] = Field(default=None, description="Deepgram API key for wrapper service")
    deepgram_base_url: str = Field(default="https://api.deepgram.com", description="Deepgram API base URL")

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"


def secure_temp_file(suffix: str = ".wav") -> tempfile._TemporaryFileWrapper:
    """Create a secure temporary file."""
    return tempfile.NamedTemporaryFile(delete=False, suffix=suffix)


def extract_user_id_from_speaker_id(speaker_id: str) -> int:
    """Extract user_id from speaker_id format: user_{user_id}_..."""
    if not speaker_id.startswith("user_"):
        raise HTTPException(400, f"Invalid speaker_id format. Expected 'user_{{user_id}}_...', got: {speaker_id}")
    
    try:
        parts = speaker_id.split("_")
        if len(parts) < 2:
            raise ValueError("Not enough parts")
        user_id = int(parts[1])
        return user_id
    except (ValueError, IndexError):
        raise HTTPException(400, f"Invalid speaker_id format. Cannot extract user_id from: {speaker_id}")


def save_enrollment_audio(user_id: int, speaker_id: str, audio_data: bytes, filename: str, enrollment_type: str = "upload") -> Path:
    """Save enrollment audio file to disk.
    
    Args:
        user_id: User ID
        speaker_id: Speaker ID
        audio_data: Audio file data
        filename: Original filename
        enrollment_type: Type of enrollment (upload, recording, append)
    
    Returns:
        Path to saved audio file
    """
    # Create directory structure: data/enrollment_audio/{user_id}/{speaker_id}/
    speaker_audio_dir = auth.enrollment_audio_dir / str(user_id) / speaker_id
    speaker_audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = Path(filename).stem.replace(" ", "_").replace("/", "_")
    extension = Path(filename).suffix or ".wav"
    unique_filename = f"{timestamp}_{enrollment_type}_{safe_filename}{extension}"
    
    # Save audio file
    audio_path = speaker_audio_dir / unique_filename
    with open(audio_path, "wb") as f:
        f.write(audio_data)
    
    log.info(f"Saved enrollment audio: {audio_path}")
    return audio_path


def save_enrollment_manifest(user_id: int, speaker_id: str, audio_files: List[dict]) -> Path:
    """Save or update enrollment manifest file.
    
    Args:
        user_id: User ID
        speaker_id: Speaker ID
        audio_files: List of audio file information
    
    Returns:
        Path to manifest file
    """
    manifest_dir = auth.enrollment_audio_dir / str(user_id) / speaker_id
    manifest_path = manifest_dir / "enrollment_manifest.json"
    
    # Load existing manifest if it exists
    existing_files = []
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)
                existing_files = manifest_data.get("audio_files", [])
        except Exception as e:
            log.warning(f"Failed to load existing manifest: {e}")
    
    # Combine existing and new files
    all_files = existing_files + audio_files
    
    # Create manifest data
    manifest_data = {
        "speaker_id": speaker_id,
        "user_id": user_id,
        "total_files": len(all_files),
        "last_updated": datetime.now().isoformat(),
        "audio_files": all_files
    }
    
    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    
    log.info(f"Updated enrollment manifest: {manifest_path}")
    return manifest_path


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
    log.info("Version: 2025-08-03-confidence-fix")
    log.info("This version includes enhanced confidence value formatting with type checking")
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


app = FastAPI(title="Simple Speaker Recognition Service", version="0.1.0", lifespan=lifespan)


async def get_db() -> UnifiedSpeakerDB:
    """Get speaker database dependency."""
    return speaker_db


@app.get("/health")
async def health(db: UnifiedSpeakerDB = Depends(get_db)):
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": str(device),
        "speakers": db.get_speaker_count(),
    }


@app.get("/users")
async def list_users():
    """List all users."""
    from simple_speaker_recognition.database import get_db_session
    from simple_speaker_recognition.database.queries import UserQueries
    
    db = get_db_session()
    try:
        users = UserQueries.get_all_users(db)
        return [
            UserResponse(
                id=user.id,
                username=user.username,
                created_at=user.created_at.isoformat()
            ) for user in users
        ]
    finally:
        db.close()


@app.post("/users")
async def create_user(request: UserRequest):
    """Create or get existing user."""
    from simple_speaker_recognition.database import get_db_session
    from simple_speaker_recognition.database.queries import UserQueries
    
    db = get_db_session()
    try:
        user = UserQueries.get_or_create_user(db, request.username)
        return UserResponse(
            id=user.id,
            username=user.username,
            created_at=user.created_at.isoformat()
        )
    finally:
        db.close()


@app.post("/enroll/upload")
async def enroll_upload(
    file: UploadFile = File(..., description="WAV/FLAC <3 min"),
    speaker_id: str = Form(..., description="Unique speaker identifier"),
    speaker_name: str = Form(..., description="Speaker display name"),
    start: Optional[float] = Form(None, description="Start time in seconds"),
    end: Optional[float] = Form(None, description="End time in seconds"),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """Enroll a speaker from uploaded audio file."""
    # Extract user_id from speaker_id
    user_id = extract_user_id_from_speaker_id(speaker_id)
    log.info(f"Enrolling speaker: {speaker_name} (ID: {speaker_id}, User: {user_id})")
    
    # Read file content
    file_content = await file.read()
    
    # Persist temporary file for processing
    with secure_temp_file() as tmp:
        tmp.write(file_content)
        tmp_path = Path(tmp.name)
    try:
        log.info(f"Loading audio file: {tmp_path}")
        wav = audio_backend.load_wave(tmp_path, start, end)
        log.info(f"Audio loaded, shape: {wav.shape}")
        
        # Get audio info
        audio_info = get_audio_info(str(tmp_path))
        duration = audio_info["duration_seconds"]
        
        log.info("Computing speaker embedding...")
        emb = await audio_backend.async_embed(wav)
        log.info(f"Embedding computed, shape: {emb.shape}")
        
        # Save audio file to enrollment directory
        saved_path = save_enrollment_audio(
            user_id=user_id,
            speaker_id=speaker_id,
            audio_data=file_content,
            filename=file.filename or "upload.wav",
            enrollment_type="upload"
        )
        
        # Create manifest entry
        audio_file_info = {
            "filename": saved_path.name,
            "path": str(saved_path.relative_to(auth.enrollment_audio_dir)),
            "duration_seconds": duration,
            "start_time": start,
            "end_time": end,
            "upload_time": datetime.now().isoformat(),
            "original_filename": file.filename
        }
        
        # Save manifest
        save_enrollment_manifest(user_id, speaker_id, [audio_file_info])
        
        log.info(f"Adding speaker to database...")
        updated = await db.add_speaker(speaker_id, speaker_name, emb[0], user_id, sample_count=1, total_duration=duration)
        
        if updated:
            log.info(f"Successfully updated existing speaker: {speaker_id}")
        else:
            log.info(f"Successfully enrolled new speaker: {speaker_id}")
        
        return {
            "updated": updated, 
            "speaker_id": speaker_id,
            "audio_saved": True,
            "audio_path": str(saved_path.relative_to(auth.enrollment_audio_dir))
        }
    except Exception as e:
        log.error(f"Error during enrollment: {e}")
        raise HTTPException(500, f"Enrollment failed: {str(e)}") from e
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/enroll/batch")
async def enroll_batch(
    files: List[UploadFile] = File(..., description="Multiple audio files for same speaker"),
    speaker_id: str = Form(..., description="Unique speaker identifier"),
    speaker_name: str = Form(..., description="Speaker display name"),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """Enroll a speaker using multiple audio segments, computing average embedding."""
    # Extract user_id from speaker_id
    user_id = extract_user_id_from_speaker_id(speaker_id)
    log.info(f"Batch enrolling speaker: {speaker_name} (ID: {speaker_id}, User: {user_id}) with {len(files)} files")
    
    embeddings = []
    temp_paths = []
    total_duration = 0.0
    saved_audio_files = []
    
    try:
        # Process each audio file
        for i, file in enumerate(files):
            log.info(f"Processing file {i+1}/{len(files)}: {file.filename}")
            
            # Read file content
            file_content = await file.read()
            
            # Save to temporary file for processing
            with secure_temp_file() as tmp:
                tmp.write(file_content)
                tmp_path = Path(tmp.name)
                temp_paths.append(tmp_path)
            
            # Load and embed
            try:
                # Get accurate duration from original file
                audio_info = get_audio_info(str(tmp_path))
                duration = audio_info["duration_seconds"]
                total_duration += duration
                
                wav = audio_backend.load_wave(tmp_path)
                emb = await audio_backend.async_embed(wav)
                embeddings.append(emb[0])
                
                # Save audio file to enrollment directory
                saved_path = save_enrollment_audio(
                    user_id=user_id,
                    speaker_id=speaker_id,
                    audio_data=file_content,
                    filename=file.filename or f"batch_{i+1}.wav",
                    enrollment_type="batch"
                )
                
                # Track saved audio file info
                audio_file_info = {
                    "filename": saved_path.name,
                    "path": str(saved_path.relative_to(auth.enrollment_audio_dir)),
                    "duration_seconds": duration,
                    "upload_time": datetime.now().isoformat(),
                    "original_filename": file.filename,
                    "batch_index": i + 1
                }
                saved_audio_files.append(audio_file_info)
                
                log.info(f"Successfully embedded and saved file {i+1}, duration: {duration:.2f}s")
            except Exception as e:
                log.warning(f"Failed to process file {i+1}: {e}")
                continue
        
        if not embeddings:
            raise HTTPException(400, "No valid audio files could be processed")
        
        # Save manifest with all audio files
        save_enrollment_manifest(user_id, speaker_id, saved_audio_files)
        
        # Compute average embedding
        log.info(f"Computing average embedding from {len(embeddings)} segments")
        embeddings_array = np.array(embeddings)
        average_embedding = np.mean(embeddings_array, axis=0)
        
        # Normalize the average embedding
        average_embedding = average_embedding / np.linalg.norm(average_embedding)
        
        log.info(f"Average embedding computed, shape: {average_embedding.shape}")
        
        # Add to database with proper counts
        updated = await db.add_speaker(
            speaker_id, 
            speaker_name, 
            average_embedding, 
            user_id,
            sample_count=len(embeddings),
            total_duration=total_duration
        )
        
        if updated:
            log.info(f"Successfully updated existing speaker: {speaker_id}")
        else:
            log.info(f"Successfully enrolled new speaker: {speaker_id}")
        
        return {
            "updated": updated,
            "speaker_id": speaker_id,
            "num_segments": len(embeddings),
            "num_files": len(files),
            "total_duration": round(total_duration, 2),
            "audio_saved": True,
            "saved_files": len(saved_audio_files)
        }
        
    except Exception as e:
        log.error(f"Error during batch enrollment: {e}")
        raise HTTPException(500, f"Batch enrollment failed: {str(e)}") from e
    finally:
        # Clean up temporary files
        for tmp_path in temp_paths:
            try:
                tmp_path.unlink(missing_ok=True)
            except:
                pass


@app.post("/enroll/append")
async def enroll_append(
    files: List[UploadFile] = File(..., description="Multiple audio files to append to existing speaker"),
    speaker_id: str = Form(..., description="Existing speaker identifier"),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """Append audio segments to an existing speaker, computing weighted average embedding."""
    # Extract user_id from speaker_id
    user_id = extract_user_id_from_speaker_id(speaker_id)
    log.info(f"Appending to speaker: {speaker_id} (User: {user_id}) with {len(files)} files")
    
    # First, verify the speaker exists
    
    db_session = get_db_session()
    try:
        existing_speaker = db_session.query(Speaker).filter(
            Speaker.id == speaker_id,
            Speaker.user_id == user_id
        ).first()
        
        if not existing_speaker:
            raise HTTPException(404, f"Speaker {speaker_id} not found for user {user_id}")
        
        if not existing_speaker.embedding_data:
            raise HTTPException(400, f"Speaker {speaker_id} has no existing embedding")
        
        # Get existing embedding and counts
        existing_embedding = np.array(json.loads(existing_speaker.embedding_data), dtype=np.float32)
        existing_count = existing_speaker.audio_sample_count or 1
        existing_duration = existing_speaker.total_audio_duration or 0.0
        
        log.info(f"Existing speaker has {existing_count} samples, {existing_duration:.2f}s duration")
        
    finally:
        db_session.close()
    
    embeddings = []
    temp_paths = []
    new_total_duration = 0.0
    saved_audio_files = []
    
    try:
        # Process each new audio file
        for i, file in enumerate(files):
            log.info(f"Processing new file {i+1}/{len(files)}: {file.filename}")
            
            # Read file content
            file_content = await file.read()
            
            # Save to temporary file for processing
            with secure_temp_file() as tmp:
                tmp.write(file_content)
                tmp_path = Path(tmp.name)
                temp_paths.append(tmp_path)
            
            # Load and embed
            try:
                # Get accurate duration from original file
                audio_info = get_audio_info(str(tmp_path))
                duration = audio_info["duration_seconds"]
                new_total_duration += duration
                
                wav = audio_backend.load_wave(tmp_path)
                emb = await audio_backend.async_embed(wav)
                embeddings.append(emb[0])
                
                # Save audio file to enrollment directory
                saved_path = save_enrollment_audio(
                    user_id=user_id,
                    speaker_id=speaker_id,
                    audio_data=file_content,
                    filename=file.filename or f"append_{i+1}.wav",
                    enrollment_type="append"
                )
                
                # Track saved audio file info
                audio_file_info = {
                    "filename": saved_path.name,
                    "path": str(saved_path.relative_to(auth.enrollment_audio_dir)),
                    "duration_seconds": duration,
                    "upload_time": datetime.now().isoformat(),
                    "original_filename": file.filename,
                    "append_index": i + 1,
                    "append_operation": True
                }
                saved_audio_files.append(audio_file_info)
                
                log.info(f"Successfully embedded and saved new file {i+1}, duration: {duration:.2f}s")
            except Exception as e:
                log.warning(f"Failed to process file {i+1}: {e}")
                continue
        
        if not embeddings:
            raise HTTPException(400, "No valid audio files could be processed")
        
        # Update manifest with appended files
        save_enrollment_manifest(user_id, speaker_id, saved_audio_files)
        
        # Compute average of new embeddings
        log.info(f"Computing average embedding from {len(embeddings)} new segments")
        new_embeddings_array = np.array(embeddings)
        new_average_embedding = np.mean(new_embeddings_array, axis=0)
        
        # Compute weighted average: (old_embedding * old_count + new_embedding * new_count) / (old_count + new_count)
        new_count = len(embeddings)
        total_count = existing_count + new_count
        
        weighted_embedding = (existing_embedding * existing_count + new_average_embedding * new_count) / total_count
        
        # Normalize the weighted embedding
        weighted_embedding = weighted_embedding / np.linalg.norm(weighted_embedding)
        
        log.info(f"Weighted average embedding computed from {existing_count} + {new_count} = {total_count} samples")
        
        # Update speaker in database
        updated = await db.add_speaker(
            speaker_id, 
            existing_speaker.name, 
            weighted_embedding, 
            user_id,
            sample_count=total_count,
            total_duration=existing_duration + new_total_duration
        )
        
        log.info(f"Successfully appended to speaker: {speaker_id}")
        
        return {
            "updated": True,
            "speaker_id": speaker_id,
            "previous_samples": existing_count,
            "new_samples": new_count,
            "total_samples": total_count,
            "previous_duration": round(existing_duration, 2),
            "new_duration": round(new_total_duration, 2),
            "total_duration": round(existing_duration + new_total_duration, 2),
            "audio_saved": True,
            "saved_files": len(saved_audio_files)
        }
        
    except Exception as e:
        log.error(f"Error during append enrollment: {e}")
        raise HTTPException(500, f"Append enrollment failed: {str(e)}") from e
    finally:
        # Clean up temporary files
        for tmp_path in temp_paths:
            try:
                tmp_path.unlink(missing_ok=True)
            except:
                pass


@app.post("/diarize-and-identify")
async def diarize_and_identify(
    file: UploadFile = File(..., description="Audio file for diarization and speaker identification"),
    req: DiarizeAndIdentifyRequest = Depends(),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """
    Perform speaker diarization and identify enrolled speakers in one step.
    
    This endpoint:
    1. Runs pyannote diarization to segment audio by speaker
    2. For each segment, extracts embeddings and identifies enrolled speakers
    3. Returns segments with both diarization labels and identified speaker names
    """
    log.info("Processing diarize-and-identify request")
    
    with secure_temp_file() as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    
    try:
        # Step 1: Perform diarization
        log.info(f"Step 1: Performing speaker diarization on {tmp_path}")
        segments = await audio_backend.async_diarize(tmp_path)
        
        # Apply minimum duration filter if specified
        if req.min_duration is not None:
            original_count = len(segments)
            segments = [s for s in segments if s["duration"] >= req.min_duration]
            if len(segments) < original_count:
                log.info(f"Filtered out {original_count - len(segments)} segments shorter than {req.min_duration}s")
        
        # Step 2: Identify speakers for each segment
        log.info(f"Step 2: Identifying speakers for {len(segments)} segments")
        enhanced_segments = []
        identified_speakers = set()
        unknown_speakers = set()
        
        # Use custom threshold if provided, otherwise use default
        threshold = req.similarity_threshold if req.similarity_threshold is not None else db.similarity_thr
        
        for i, segment in enumerate(segments):
            try:
                speaker_label = segment["speaker"]
                start_time = segment["start"]
                end_time = segment["end"]
                
                # Skip very short segments (less than min_duration)
                if end_time - start_time < req.min_duration:
                    log.debug(f"Skipping segment {i+1}: too short ({end_time - start_time:.2f}s)")
                    continue
                
                # Load audio segment
                wav = audio_backend.load_wave(tmp_path, start_time, end_time)
                
                # Generate embedding
                emb = await audio_backend.async_embed(wav)
                
                # Identify speaker with custom threshold
                found = False
                speaker_info = None
                confidence = 0.0
                
                # Try to identify speaker (UnifiedSpeakerDB handles speaker existence check internally)
                # Temporarily override threshold for this identification
                original_threshold = db.similarity_thr
                db.similarity_thr = threshold
                try:
                    found, speaker_info, confidence = await db.identify(emb, user_id=req.user_id)
                    
                    # Validate confidence value at source
                    if confidence is not None:
                        if not isinstance(confidence, (int, float, np.number)):
                            log.warning(f"Invalid confidence type from speaker_db.identify: {type(confidence)}, value: {confidence}")
                            confidence = 0.0
                        elif np.isnan(confidence) or np.isinf(confidence):
                            log.warning(f"Invalid confidence value from speaker_db.identify: {confidence} (NaN/Inf)")
                            confidence = 0.0
                finally:
                    db.similarity_thr = original_threshold
                
                # Build enhanced segment
                enhanced_segment = {
                    "speaker": speaker_label,
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "duration": round(end_time - start_time, 3),
                    "identified_as": speaker_info["name"] if found and speaker_info else None,
                    "identified_id": speaker_info["id"] if found and speaker_info else None,
                    "confidence": round(float(confidence), 3) if confidence else 0.0,
                    "status": "identified" if found else "unknown"
                }
                
                # Track identified vs unknown speakers
                if found and speaker_info:
                    identified_speakers.add(speaker_info["name"])
                    # Use safe confidence formatting utility
                    confidence_str = safe_format_confidence(confidence, "diarization_segment")
                    log.debug(f"Segment {i+1}: Identified as {speaker_info['name']} (confidence: {confidence_str})")
                else:
                    unknown_speakers.add(speaker_label)
                    log.debug(f"Segment {i+1}: Unknown speaker {speaker_label}")
                
                # Only add segment if it's identified or we're not filtering
                if not req.identify_only_enrolled or found:
                    enhanced_segments.append(enhanced_segment)
                    
            except Exception as e:
                log.warning(f"Error processing segment {i+1}: {str(e)}")
                # Add segment with error status unless filtering
                if not req.identify_only_enrolled:
                    enhanced_segments.append({
                        "speaker": segment["speaker"],
                        "start": segment["start"],
                        "end": segment["end"],
                        "duration": segment["duration"],
                        "identified_as": None,
                        "identified_id": None,
                        "confidence": 0.0,
                        "status": "error",
                        "error": str(e)
                    })
        
        # Calculate summary statistics
        total_duration = max(s["end"] for s in segments) if segments else 0
        
        log.info(f"Diarization and identification complete - {len(identified_speakers)} identified, {len(unknown_speakers)} unknown")
        
        return {
            "segments": enhanced_segments,
            "summary": {
                "total_duration": round(total_duration, 2),
                "num_segments": len(enhanced_segments),
                "num_diarized_speakers": len(set(s["speaker"] for s in segments)),
                "identified_speakers": sorted(list(identified_speakers)),
                "unknown_speakers": sorted(list(unknown_speakers)),
                "similarity_threshold": threshold,
                "filtered": req.identify_only_enrolled
            }
        }
        
    except Exception as e:
        log.error(f"Error during diarize-and-identify: {e}")
        raise HTTPException(500, f"Diarize and identify failed: {str(e)}") from e
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/identify", response_model=IdentifyResponse)
async def identify(
    file: UploadFile = File(..., description="Audio file for speaker identification"),
    req: DiarizeAndIdentifyRequest = Depends(),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """
    Identify the speaker in an audio file.
    
    This endpoint is optimized for real-time processing:
    1. Assumes the audio contains speech from a single speaker
    2. Extracts embedding from the entire audio chunk
    3. Identifies the enrolled speaker
    4. Returns a single identification result
    
    Designed for use with utterance boundaries in real-time transcription.
    """
    log.info("Processing identify request")
    
    with secure_temp_file() as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
        
        # Debug: Copy WAV file to debug directory
        try:
            debug_dir = Path("/app/debug")
            if not debug_dir.exists():
                log.error(f"Debug directory does not exist, creating: {debug_dir}")
            
            # Create filename with timestamp and original filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            original_name = getattr(file, 'filename', 'utterance.wav') or 'utterance.wav'
            debug_filename = f"{timestamp}_{original_name}"
            debug_path = debug_dir / debug_filename
            
            # Copy the temp file to debug location
            shutil.copy2(tmp_path, debug_path)
            
            log.info(f"🐛 [DEBUG] WAV file dumped to: {debug_path}")
        except Exception as e:
            log.warning(f"Failed to dump debug WAV file: {e}")
    
    try:
        # Get audio info for duration
        audio_info = get_audio_info(str(tmp_path))
        duration = audio_info.get('duration', 0.0)
        
        log.info(f"Processing audio: {duration:.2f}s duration")
        
        # Load the entire audio file (no segmentation needed)
        wav = audio_backend.load_wave(tmp_path)
        
        # Generate embedding for the entire utterance
        emb = await audio_backend.async_embed(wav)
        
        # Use custom threshold if provided, otherwise use default
        threshold = req.similarity_threshold if req.similarity_threshold is not None else db.similarity_thr
        
        # Identify speaker with custom threshold
        found = False
        speaker_info = None
        confidence = 0.0
        
        # Temporarily override threshold for this identification
        original_threshold = db.similarity_thr
        db.similarity_thr = threshold
        try:
            found, speaker_info, confidence = await db.identify(emb, user_id=req.user_id)
            
            # Validate confidence value
            if confidence is not None:
                if not isinstance(confidence, (int, float, np.number)):
                    log.warning(f"Invalid confidence type from speaker_db.identify: {type(confidence)}, value: {confidence}")
                    confidence = 0.0
                elif np.isnan(confidence) or np.isinf(confidence):
                    log.warning(f"Invalid confidence value from speaker_db.identify: {confidence} (NaN/Inf)")
                    confidence = 0.0
        finally:
            db.similarity_thr = original_threshold
        
        # Build response
        if found and speaker_info:
            confidence_str = safe_format_confidence(confidence, "speaker_identification")
            log.info(f"Speaker identified as {speaker_info['name']} (confidence: {confidence_str})")
            
            return IdentifyResponse(
                found=True,
                speaker_id=speaker_info["id"],
                speaker_name=speaker_info["name"],
                confidence=round(float(confidence), 3),
                status=SpeakerStatus.IDENTIFIED,
                similarity_threshold=threshold,
                duration=round(duration, 3)
            )
        else:
            log.info(f"Speaker not identified (confidence: {confidence:.3f} < threshold: {threshold:.3f})")
            
            return IdentifyResponse(
                found=False,
                speaker_id=None,
                speaker_name=None,
                confidence=round(float(confidence), 3),
                status=SpeakerStatus.UNKNOWN,
                similarity_threshold=threshold,
                duration=round(duration, 3)
            )
        
    except Exception as e:
        log.error(f"Error during speaker identification: {e}")
        raise HTTPException(500, f"Speaker identification failed: {str(e)}") from e
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/speakers")
async def list_speakers(user_id: Optional[int] = None, db: UnifiedSpeakerDB = Depends(get_db)):
    """List all enrolled speakers, optionally filtered by user."""
    from simple_speaker_recognition.database import get_db_session
    from simple_speaker_recognition.database.models import Speaker
    
    db_session = get_db_session()
    try:
        if user_id is not None:
            # Filter by user
            query_speakers = db_session.query(Speaker).filter(Speaker.user_id == user_id).all()
        else:
            # Return all speakers
            query_speakers = db_session.query(Speaker).all()
        
        speakers = [
            {
                "id": speaker.id,
                "name": speaker.name,
                "user_id": speaker.user_id,
                "created_at": speaker.created_at.isoformat() if speaker.created_at else None,
                "updated_at": speaker.updated_at.isoformat() if speaker.updated_at else None,
                "audio_sample_count": speaker.audio_sample_count or 0,
                "total_audio_duration": speaker.total_audio_duration or 0.0
            }
            for speaker in query_speakers
        ]
    finally:
        db_session.close()
    
    return {"speakers": speakers}


@app.get("/speakers/analysis")
async def get_speakers_analysis(
    user_id: Optional[int] = Query(default=None, description="User ID to filter speakers"),
    method: str = Query(default="umap", description="Dimensionality reduction method: umap, tsne, pca"),
    cluster_method: str = Query(default="dbscan", description="Clustering method: dbscan, kmeans"),
    similarity_threshold: float = Query(default=0.8, description="Threshold for finding similar speakers"),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """Get comprehensive analysis of speaker embeddings including clustering and visualization data."""
    from simple_speaker_recognition.database import get_db_session
    from simple_speaker_recognition.database.models import Speaker
    from simple_speaker_recognition.utils.analysis import create_speaker_analysis
    
    log.info(f"Generating speaker analysis for user_id={user_id}, method={method}, cluster_method={cluster_method}")
    
    db_session = get_db_session()
    try:
        # Get speakers, optionally filtered by user
        if user_id is not None:
            query_speakers = db_session.query(Speaker).filter(Speaker.user_id == user_id).all()
        else:
            query_speakers = db_session.query(Speaker).all()
        
        if not query_speakers:
            return {
                "status": "success",
                "message": "No speakers found",
                "visualization": {
                    "speakers": [],
                    "embeddings_2d": [],
                    "embeddings_3d": [],
                    "cluster_labels": [],
                    "colors": []
                },
                "clustering": {"n_clusters": 0, "method": cluster_method},
                "similar_speakers": [],
                "quality_metrics": {"n_speakers": 0},
                "parameters": {
                    "reduction_method": method,
                    "cluster_method": cluster_method,
                    "similarity_threshold": similarity_threshold
                }
            }
        
        # Extract embeddings for analysis
        embeddings_dict = {}
        for speaker in query_speakers:
            if speaker.embedding_data:
                try:
                    embedding = np.array(json.loads(speaker.embedding_data), dtype=np.float32)
                    embeddings_dict[speaker.id] = embedding
                except (json.JSONDecodeError, ValueError) as e:
                    log.warning(f"Invalid embedding data for speaker {speaker.id}: {e}")
                    continue
        
        if not embeddings_dict:
            return {
                "status": "success",
                "message": "No valid embeddings found",
                "visualization": {
                    "speakers": [],
                    "embeddings_2d": [],
                    "embeddings_3d": [],
                    "cluster_labels": [],
                    "colors": []
                },
                "clustering": {"n_clusters": 0, "method": cluster_method},
                "similar_speakers": [],
                "quality_metrics": {"n_speakers": 0},
                "parameters": {
                    "reduction_method": method,
                    "cluster_method": cluster_method,
                    "similarity_threshold": similarity_threshold
                }
            }
        
        # Perform analysis
        analysis_result = create_speaker_analysis(
            embeddings_dict=embeddings_dict,
            method=method,
            cluster_method=cluster_method,
            similarity_threshold=similarity_threshold
        )
        
        return analysis_result
        
    except Exception as e:
        log.error(f"Error in speaker analysis: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        db_session.close()


@app.get("/speakers/{speaker_id}/audio")
async def get_speaker_audio_files(speaker_id: str, db: UnifiedSpeakerDB = Depends(get_db)):
    """Get list of saved audio files for a speaker."""
    # Extract user_id from speaker_id for authorization
    user_id = extract_user_id_from_speaker_id(speaker_id)
    
    # Check if speaker exists
    from simple_speaker_recognition.database import get_db_session
    from simple_speaker_recognition.database.models import Speaker
    
    db_session = get_db_session()
    try:
        speaker = db_session.query(Speaker).filter(
            Speaker.id == speaker_id,
            Speaker.user_id == user_id
        ).first()
        
        if not speaker:
            raise HTTPException(404, f"Speaker {speaker_id} not found for user {user_id}")
            
    finally:
        db_session.close()
    
    # Check if manifest file exists
    manifest_path = auth.enrollment_audio_dir / str(user_id) / speaker_id / "enrollment_manifest.json"
    
    if not manifest_path.exists():
        return {
            "speaker_id": speaker_id,
            "audio_files": [],
            "total_files": 0,
            "message": "No audio files found for this speaker"
        }
    
    try:
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        
        return {
            "speaker_id": speaker_id,
            "user_id": user_id,
            "audio_files": manifest_data.get("audio_files", []),
            "total_files": manifest_data.get("total_files", 0),
            "last_updated": manifest_data.get("last_updated")
        }
        
    except Exception as e:
        log.error(f"Error reading manifest for speaker {speaker_id}: {e}")
        raise HTTPException(500, f"Failed to read audio files manifest: {str(e)}")


@app.get("/speakers/{speaker_id}/audio/{filename}")
async def download_speaker_audio_file(speaker_id: str, filename: str, db: UnifiedSpeakerDB = Depends(get_db)):
    """Download a specific audio file for a speaker."""
    # Extract user_id from speaker_id for authorization
    user_id = extract_user_id_from_speaker_id(speaker_id)
    
    # Check if speaker exists
    from simple_speaker_recognition.database import get_db_session
    from simple_speaker_recognition.database.models import Speaker
    
    db_session = get_db_session()
    try:
        speaker = db_session.query(Speaker).filter(
            Speaker.id == speaker_id,
            Speaker.user_id == user_id
        ).first()
        
        if not speaker:
            raise HTTPException(404, f"Speaker {speaker_id} not found for user {user_id}")
            
    finally:
        db_session.close()
    
    # Construct file path (security: only allow files within speaker's directory)
    speaker_audio_dir = auth.enrollment_audio_dir / str(user_id) / speaker_id
    file_path = speaker_audio_dir / filename
    
    # Security check: ensure file is within the expected directory
    try:
        file_path = file_path.resolve()
        speaker_audio_dir = speaker_audio_dir.resolve()
        if not str(file_path).startswith(str(speaker_audio_dir)):
            raise HTTPException(403, "Access to file outside speaker directory is forbidden")
    except Exception:
        raise HTTPException(400, "Invalid file path")
    
    if not file_path.exists():
        raise HTTPException(404, f"Audio file {filename} not found for speaker {speaker_id}")
    
    if not file_path.is_file():
        raise HTTPException(400, f"Path {filename} is not a file")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="audio/wav"
    )


# ============================================================================
# Deepgram API Wrapper Endpoints
# ============================================================================

def sanitize_params_for_deepgram(params: Dict[str, Any]) -> Dict[str, str]:
    """Convert parameters to string format expected by Deepgram API."""
    sanitized = {}
    for key, value in params.items():
        if value is None:
            continue
        elif isinstance(value, bool):
            # Convert boolean to lowercase string
            sanitized[key] = str(value).lower()
        elif isinstance(value, (int, float)):
            sanitized[key] = str(value)
        elif isinstance(value, str):
            sanitized[key] = value
        elif isinstance(value, list):
            # Handle list parameters (like keywords, search terms)
            sanitized[key] = ','.join(str(item) for item in value)
        else:
            # Convert other types to string
            sanitized[key] = str(value)
    return sanitized


async def forward_to_deepgram(
    audio_data: bytes,
    content_type: str,
    params: Dict[str, Any],
    deepgram_api_key: str
) -> Dict[str, Any]:
    """Forward audio to Deepgram API and return response."""
    url = f"{auth.deepgram_base_url}/v1/listen"
    
    headers = {
        "Authorization": f"Token {deepgram_api_key}",
        "Content-Type": content_type
    }
    
    # Sanitize parameters for Deepgram API
    sanitized_params = sanitize_params_for_deepgram(params)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers=headers,
            data=audio_data,
            params=sanitized_params
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                log.error(f"Deepgram API error: {response.status} - {error_text}")
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Deepgram API error: {error_text}"
                )
            
            result = await response.json()
            log.info("Successfully received Deepgram response")
            return result


async def extract_and_identify_speakers(
    audio_data: bytes,
    deepgram_response: Dict[str, Any],
    user_id: Optional[int],
    confidence_threshold: float = 0.15
) -> Dict[str, Any]:
    """Extract speaker segments and identify speakers."""
    enhanced_response = deepgram_response.copy()
    
    if not user_id:
        log.warning("No user_id provided, skipping speaker identification")
        return enhanced_response
    
    try:
        # Parse Deepgram response to extract segments
        from simple_speaker_recognition.utils.deepgram_parser import DeepgramParser
        parser = DeepgramParser()
        
        # Create temporary file for audio processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Get actual audio duration for boundary validation
            audio_info = get_audio_info(str(tmp_path))
            audio_duration = audio_info["duration_seconds"]
            log.info(f"Audio file duration: {audio_duration:.6f}s")
            
            # Extract words from Deepgram response
            results = deepgram_response.get("results", {})
            channels = results.get("channels", [])
            
            if not channels:
                log.warning("No channels found in Deepgram response")
                return enhanced_response
            
            channel = channels[0]
            alternatives = channel.get("alternatives", [])
            
            if not alternatives:
                log.warning("No alternatives found in Deepgram response")
                return enhanced_response
            
            words = alternatives[0].get("words", [])
            
            # Group consecutive words by speaker to create segments
            speaker_segments = []
            enhanced_words = []
            current_segment = None
            
            for word_info in words:
                speaker = word_info.get("speaker", 0)
                start_time = word_info.get("start", 0)
                end_time = word_info.get("end", 0)
                
                # Create enhanced word with default values
                enhanced_word = word_info.copy()
                enhanced_word.update({
                    "identified_speaker_id": None,
                    "identified_speaker_name": None,
                    "speaker_identification_confidence": None,
                    "speaker_status": SpeakerStatus.UNKNOWN.value
                })
                
                # Check if we need to start a new segment (speaker changed)
                if current_segment is None or current_segment["speaker"] != speaker:
                    # Save previous segment
                    if current_segment is not None:
                        speaker_segments.append(current_segment)
                    
                    # Start new segment
                    current_segment = {
                        "speaker": speaker,
                        "start": start_time,
                        "end": end_time,
                        "words": [word_info],
                        "word_indices": [len(enhanced_words)],  # Track word indices for later update
                        "identified": False
                    }
                else:
                    # Continue current segment
                    current_segment["end"] = end_time
                    current_segment["words"].append(word_info)
                    current_segment["word_indices"].append(len(enhanced_words))
                
                enhanced_words.append(enhanced_word)
            
            # Don't forget the last segment
            if current_segment is not None:
                speaker_segments.append(current_segment)
            
            # Process each speaker segment for identification
            
            for segment_idx, segment_info in enumerate(speaker_segments):
                try:
                    # Extract audio segment
                    start_time = segment_info["start"]
                    end_time = segment_info["end"]
                    
                    # Validate segment boundaries against actual audio duration
                    if start_time >= audio_duration:
                        log.warning(f"Segment {segment_idx} start time {start_time:.6f}s exceeds audio duration {audio_duration:.6f}s, skipping")
                        # Apply error status to all words in this segment
                        for word_idx in segment_info["word_indices"]:
                            enhanced_words[word_idx].update({
                                "identified_speaker_id": None,
                                "identified_speaker_name": None,
                                "speaker_identification_confidence": 0.0,
                                "speaker_status": SpeakerStatus.ERROR.value
                            })
                        continue
                        
                    if end_time > audio_duration:
                        log.warning(f"Segment {segment_idx} end time {end_time:.6f}s exceeds audio duration {audio_duration:.6f}s, clipping to {audio_duration:.6f}s")
                        end_time = audio_duration
                        
                    if end_time <= start_time:
                        log.warning(f"Segment {segment_idx} has invalid duration (start: {start_time:.6f}s, end: {end_time:.6f}s), skipping")
                        # Apply error status to all words in this segment
                        for word_idx in segment_info["word_indices"]:
                            enhanced_words[word_idx].update({
                                "identified_speaker_id": None,
                                "identified_speaker_name": None,
                                "speaker_identification_confidence": 0.0,
                                "speaker_status": SpeakerStatus.ERROR.value
                            })
                        continue
                    
                    # Load and extract segment
                    wav = audio_backend.load_wave(tmp_path, start_time, end_time)
                    
                    # Get embedding
                    emb = await audio_backend.async_embed(wav)
                    
                    # Identify speaker
                    found, speaker_info, confidence = await speaker_db.identify(emb, user_id=user_id)
                    
                    # Validate confidence value at source
                    if confidence is not None:
                        if not isinstance(confidence, (int, float, np.number)):
                            log.warning(f"Invalid confidence type from speaker_db.identify: {type(confidence)}, value: {confidence}")
                            confidence = 0.0
                        elif np.isnan(confidence) or np.isinf(confidence):
                            log.warning(f"Invalid confidence value from speaker_db.identify: {confidence} (NaN/Inf)")
                            confidence = 0.0
                    
                    # Store identification result for this segment
                    segment_result = None
                    
                    if found and confidence >= confidence_threshold:
                        segment_result = {
                            "speaker_id": speaker_info["id"],
                            "speaker_name": speaker_info["name"],
                            "confidence": confidence,
                            "status": SpeakerStatus.IDENTIFIED.value
                        }
                        # Use safe confidence formatting utility
                        confidence_str = safe_format_confidence(confidence, "deepgram_speaker_identification")
                        log.info(f"Identified segment {segment_idx} (speaker {segment_info['speaker']}) as {speaker_info['name']} (confidence: {confidence_str})")
                    else:
                        segment_result = {
                            "speaker_id": None,
                            "speaker_name": None,
                            "confidence": confidence if confidence is not None else 0.0,
                            "status": SpeakerStatus.UNKNOWN.value
                        }
                        # Use safe confidence formatting utility
                        confidence_str = safe_format_confidence(confidence, "deepgram_speaker_unknown")
                        log.info(f"Segment {segment_idx} (speaker {segment_info['speaker']}) not identified (confidence: {confidence_str})")
                    
                    # Apply identification to all words in this segment
                    for word_idx in segment_info["word_indices"]:
                        enhanced_words[word_idx].update({
                            "identified_speaker_id": segment_result["speaker_id"],
                            "identified_speaker_name": segment_result["speaker_name"],
                            "speaker_identification_confidence": segment_result["confidence"],
                            "speaker_status": segment_result["status"]
                        })
                    
                    # Store result for summary
                    segment_info["identification"] = segment_result
                        
                except Exception as e:
                    log.warning(f"Error identifying segment {segment_idx}: {e}")
                    # Apply error status to all words in this segment
                    for word_idx in segment_info["word_indices"]:
                        enhanced_words[word_idx].update({
                            "identified_speaker_id": None,
                            "identified_speaker_name": None,
                            "speaker_identification_confidence": 0.0,
                            "speaker_status": SpeakerStatus.ERROR.value
                        })
            
            # Update the response with enhanced words
            enhanced_response["results"]["channels"][0]["alternatives"][0]["words"] = enhanced_words
            
            # Collect unique identified speakers
            identified_speakers = {}
            for segment in speaker_segments:
                if "identification" in segment:
                    result = segment["identification"]
                    if result["status"] == SpeakerStatus.IDENTIFIED.value:
                        # Use the original Deepgram speaker ID as key
                        speaker_key = str(segment["speaker"])
                        # Only store the first occurrence of each identified speaker
                        if speaker_key not in identified_speakers:
                            identified_speakers[speaker_key] = result
            
            # Add speaker enhancement metadata
            enhanced_response["speaker_enhancement"] = {
                "enabled": True,
                "user_id": user_id,
                "confidence_threshold": confidence_threshold,
                "identified_speakers": identified_speakers,
                "total_segments": len(speaker_segments),
                "identified_segments": len([s for s in speaker_segments if s.get("identification", {}).get("status") == SpeakerStatus.IDENTIFIED.value]),
                "total_speakers": len(set(s["speaker"] for s in speaker_segments)),
                "identified_count": len(identified_speakers)
            }
            
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)
    
    except Exception as e:
        log.error(f"Error during speaker identification: {e}")
        # Add error info to response but don't fail the request
        enhanced_response["speaker_enhancement"] = {
            "enabled": True,
            "error": str(e),
            "status": "failed"
        }
    
    return enhanced_response


@app.post("/v1/listen")
async def deepgram_compatible_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    # Deepgram-compatible query parameters
    model: str = Query(default="nova-3", description="Model to use for transcription"),
    language: str = Query(default="multi", description="Language code"),
    version: str = Query(default="latest", description="Model version"),
    punctuate: bool = Query(default=True, description="Add punctuation"),
    profanity_filter: bool = Query(default=False, description="Filter profanity"),
    diarize: bool = Query(default=True, description="Enable speaker diarization"),
    diarize_version: str = Query(default="latest", description="Diarization model version"),
    multichannel: bool = Query(default=False, description="Process multiple channels"),
    alternatives: int = Query(default=1, description="Number of alternative transcripts"),
    numerals: bool = Query(default=True, description="Convert numbers to numerals"),
    smart_format: bool = Query(default=True, description="Enable smart formatting"),
    paragraphs: bool = Query(default=True, description="Organize into paragraphs"),
    utterances: bool = Query(default=True, description="Organize into utterances"),
    detect_language: bool = Query(default=False, description="Detect language automatically"),
    summarize: bool = Query(default=False, description="Generate summary"),
    sentiment: bool = Query(default=False, description="Analyze sentiment"),
    # Speaker enhancement parameters
    enhance_speakers: bool = Query(default=True, description="Enable speaker identification enhancement"),
    user_id: Optional[int] = Query(default=None, description="User ID for speaker identification"),
    speaker_confidence_threshold: float = Query(default=0.15, description="Minimum confidence for speaker identification"),
    # Authentication
    authorization: Optional[str] = Header(default=None, description="Authorization header"),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """Deepgram-compatible transcription endpoint with speaker enhancement."""
    
    # Extract API key from authorization header
    api_key = None
    if authorization:
        if authorization.startswith("Token "):
            api_key = authorization[6:]
        elif authorization.startswith("Bearer "):
            api_key = authorization[7:]
        else:
            api_key = authorization
    
    # Use provided API key or fall back to service default
    deepgram_key = api_key or auth.deepgram_api_key
    if not deepgram_key:
        raise HTTPException(
            status_code=401,
            detail="Deepgram API key required. Provide via Authorization header or DEEPGRAM_API_KEY environment variable."
        )
    
    log.info(f"Processing Deepgram-compatible transcription request: {file.filename}")
    
    try:
        # Read audio file
        audio_data = await file.read()
        content_type = file.content_type or "audio/wav"
        
        # Prepare parameters for Deepgram API
        deepgram_params = {
            "model": model,
            "language": language,
            "version": version,
            "punctuate": punctuate,
            "profanity_filter": profanity_filter,
            "diarize": diarize,
            "diarize_version": diarize_version,
            "multichannel": multichannel,
            "alternatives": alternatives,
            "numerals": numerals,
            "smart_format": smart_format,
            "paragraphs": paragraphs,
            "utterances": utterances,
            "detect_language": detect_language,
            "summarize": summarize,
            "sentiment": sentiment,
        }
        
        # Remove None values and custom parameters not for Deepgram
        deepgram_params = {k: v for k, v in deepgram_params.items() if v is not None}
        
        # Remove our custom parameters that shouldn't go to Deepgram
        custom_params = ['enhance_speakers', 'user_id', 'speaker_confidence_threshold']
        for param in custom_params:
            deepgram_params.pop(param, None)
        
        log.info(f"Forwarding request to Deepgram API with params: {deepgram_params}")
        
        # Forward to Deepgram
        deepgram_response = await forward_to_deepgram(
            audio_data=audio_data,
            content_type=content_type,
            params=deepgram_params,
            deepgram_api_key=deepgram_key
        )
        
        # Enhance with speaker identification if enabled
        if enhance_speakers and diarize:
            log.info("Enhancing transcription with speaker identification")
            enhanced_response = await extract_and_identify_speakers(
                audio_data=audio_data,
                deepgram_response=deepgram_response,
                user_id=user_id,
                confidence_threshold=speaker_confidence_threshold
            )
            return enhanced_response
        else:
            log.info("Returning original Deepgram response (speaker enhancement disabled)")
            return deepgram_response
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error in Deepgram-compatible transcription: {e}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")


async def combine_transcription_and_diarization(
    deepgram_response: Dict[str, Any],
    diarization_segments: List[Dict[str, Any]],
    user_id: Optional[int],
    confidence_threshold: float = 0.15
) -> Dict[str, Any]:
    """Combine Deepgram transcription with internal diarization segments."""
    enhanced_response = deepgram_response.copy()
    
    try:
        # Debug: Log diarization segments for boundary analysis
        log.info(f"Debug - Received {len(diarization_segments)} diarization segments:")
        for i, seg in enumerate(diarization_segments):
            log.info(f"  Segment {i}: speaker={seg.get('speaker')}, start={seg.get('start'):.3f}s, end={seg.get('end'):.3f}s, status={seg.get('status')}, id={seg.get('identified_as')}")
        
        # Extract words from Deepgram response
        results = deepgram_response.get("results", {})
        channels = results.get("channels", [])
        
        if not channels:
            log.warning("No channels found in Deepgram response")
            return enhanced_response
        
        channel = channels[0]
        alternatives = channel.get("alternatives", [])
        
        if not alternatives:
            log.warning("No alternatives found in Deepgram response")
            return enhanced_response
        
        words = alternatives[0].get("words", [])
        
        # Create enhanced words by matching timestamps with diarization segments
        enhanced_words = []
        
        for word_info in words:
            word_start = word_info.get("start", 0)
            word_end = word_info.get("end", 0)
            word_middle = (word_start + word_end) / 2
            word_text = word_info.get("punctuated_word", word_info.get("word", ""))
            
            # Find which diarization segment this word belongs to
            matching_segment = None
            best_overlap = 0
            best_overlap_ratio = 0
            
            # First try: find segment with best overlap ratio
            # This is more robust than just checking if word middle is within segment
            for segment in diarization_segments:
                # Calculate overlap between word and segment
                overlap_start = max(word_start, segment["start"])
                overlap_end = min(word_end, segment["end"])
                overlap_duration = max(0, overlap_end - overlap_start)
                word_duration = word_end - word_start
                
                # Calculate overlap ratio (what percentage of the word overlaps)
                overlap_ratio = overlap_duration / word_duration if word_duration > 0 else 0
                
                # Prefer segments with higher overlap ratio
                if overlap_ratio > best_overlap_ratio or (overlap_ratio == best_overlap_ratio and overlap_duration > best_overlap):
                    best_overlap = overlap_duration
                    best_overlap_ratio = overlap_ratio
                    matching_segment = segment
            
            # Second try: if no good overlap, find nearest segment
            # But only use this as fallback if overlap ratio is very low
            if not matching_segment or best_overlap_ratio < 0.1:
                min_distance = float('inf')
                nearest_segment = None
                
                for segment in diarization_segments:
                    # Calculate distance from word middle to segment
                    if word_middle < segment["start"]:
                        distance = segment["start"] - word_middle
                    elif word_middle > segment["end"]:
                        distance = word_middle - segment["end"]
                    else:
                        distance = 0  # Word is within segment
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_segment = segment
                
                # Only use nearest segment if it's reasonably close (within 1 second)
                # and we don't have a decent overlap already
                if nearest_segment and min_distance < 1.0 and best_overlap_ratio < 0.5:
                    matching_segment = nearest_segment
            
            # Debug logging for problematic boundary words
            if word_text.lower() in ["her", "her.", "so"] or 11.0 <= word_middle <= 13.0:
                assigned_speaker = matching_segment.get("identified_as", "unknown") if matching_segment else "none"
                log.info(f"Debug boundary word: '{word_text}' ({word_start:.3f}s-{word_end:.3f}s) → {assigned_speaker} (overlap_ratio: {best_overlap_ratio:.3f})")
            
            # Create enhanced word with speaker information
            enhanced_word = word_info.copy()
            
            if matching_segment:
                # Map internal diarization labels to Deepgram-style speaker IDs
                speaker_id = matching_segment["speaker"]
                
                enhanced_word.update({
                    "speaker": speaker_id,
                    "speaker_confidence": min(matching_segment["confidence"], 1.0),
                    "identified_speaker_id": matching_segment.get("identified_id"),
                    "identified_speaker_name": matching_segment.get("identified_as"),
                    "speaker_identification_confidence": matching_segment["confidence"],
                    "speaker_status": matching_segment["status"]
                })
            else:
                # Fallback for words not in any segment (should be rare now)
                enhanced_word.update({
                    "speaker": 0,
                    "speaker_confidence": 0.0,
                    "identified_speaker_id": None,
                    "identified_speaker_name": None,
                    "speaker_identification_confidence": 0.0,
                    "speaker_status": "unknown"
                })
            
            enhanced_words.append(enhanced_word)
        
        # Update the response with enhanced words
        enhanced_response["results"]["channels"][0]["alternatives"][0]["words"] = enhanced_words
        
        # Group consecutive words with same speaker into segments for cleaner output
        grouped_segments = []
        if enhanced_words:
            current_segment = None
            
            for word in enhanced_words:
                speaker = word.get("speaker")
                speaker_name = word.get("identified_speaker_name")
                speaker_id = word.get("identified_speaker_id")
                confidence = word.get("speaker_identification_confidence", 0.0)
                status = word.get("speaker_status", "unknown")
                
                # Check if we need to start a new segment (speaker changed)
                if (current_segment is None or 
                    current_segment["speaker"] != speaker or 
                    current_segment.get("identified_speaker_id") != speaker_id):
                    
                    # Save previous segment if it exists
                    if current_segment is not None:
                        grouped_segments.append(current_segment)
                    
                    # Start new segment
                    current_segment = {
                        "speaker": speaker,
                        "start": word.get("start", 0),
                        "end": word.get("end", 0),
                        "text": word.get("punctuated_word", word.get("word", "")),
                        "identified_speaker_id": speaker_id,
                        "identified_speaker_name": speaker_name,
                        "confidence": confidence,
                        "speaker_identification_confidence": confidence,
                        "speaker_status": status
                    }
                else:
                    # Continue current segment
                    current_segment["end"] = word.get("end", current_segment["end"])
                    current_segment["text"] += " " + word.get("punctuated_word", word.get("word", ""))
                    # Update confidence to average if we have multiple words
                    if confidence > 0:
                        current_segment["confidence"] = max(current_segment["confidence"], confidence)
                        current_segment["speaker_identification_confidence"] = current_segment["confidence"]
            
            # Add final segment
            if current_segment is not None:
                grouped_segments.append(current_segment)
        
        # Add grouped segments to response for easier consumption
        enhanced_response["speaker_segments"] = grouped_segments
        
        # Create speaker enhancement metadata
        identified_speakers = {}
        for segment in diarization_segments:
            if segment["status"] == "identified" and segment.get("identified_id"):
                speaker_key = str(segment["speaker"])
                if speaker_key not in identified_speakers:
                    identified_speakers[speaker_key] = {
                        "speaker_id": segment["identified_id"],
                        "speaker_name": segment["identified_as"],
                        "confidence": segment["confidence"],
                        "status": "identified"
                    }
        
        # Add speaker enhancement metadata
        enhanced_response["speaker_enhancement"] = {
            "enabled": True,
            "method": "hybrid_deepgram_internal",
            "user_id": user_id,
            "confidence_threshold": confidence_threshold,
            "identified_speakers": identified_speakers,
            "total_segments": len(diarization_segments),
            "identified_segments": len([s for s in diarization_segments if s["status"] == "identified"]),
            "total_speakers": len(set(s["speaker"] for s in diarization_segments)),
            "identified_count": len(identified_speakers)
        }
        
        return enhanced_response
        
    except Exception as e:
        log.error(f"Error combining transcription and diarization: {e}")
        # Return original response with error info
        enhanced_response["speaker_enhancement"] = {
            "enabled": True,
            "method": "hybrid_deepgram_internal",
            "error": str(e),
            "status": "failed"
        }
        return enhanced_response


@app.post("/v1/transcribe-and-diarize")
async def hybrid_transcribe_and_diarize(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe and diarize"),
    # Deepgram-compatible query parameters for transcription
    model: str = Query(default="nova-3", description="Model to use for transcription"),
    language: str = Query(default="multi", description="Language code"),
    version: str = Query(default="latest", description="Model version"),
    punctuate: bool = Query(default=True, description="Add punctuation"),
    profanity_filter: bool = Query(default=False, description="Filter profanity"),
    multichannel: bool = Query(default=False, description="Process multiple channels"),
    alternatives: int = Query(default=1, description="Number of alternative transcripts"),
    numerals: bool = Query(default=True, description="Convert numbers to numerals"),
    smart_format: bool = Query(default=True, description="Enable smart formatting"),
    paragraphs: bool = Query(default=True, description="Organize into paragraphs"),
    utterances: bool = Query(default=True, description="Organize into utterances"),
    detect_language: bool = Query(default=False, description="Detect language automatically"),
    summarize: bool = Query(default=False, description="Generate summary"),
    sentiment: bool = Query(default=False, description="Analyze sentiment"),
    # Speaker diarization parameters (for internal service)
    user_id: Optional[int] = Query(default=None, description="User ID for speaker identification"),
    speaker_confidence_threshold: float = Query(default=0.15, description="Minimum confidence for speaker identification"),
    similarity_threshold: float = Query(default=0.15, description="Cosine similarity threshold for speaker identification"),
    min_duration: float = Query(default=1.0, description="Minimum segment duration in seconds"),
    # Authentication
    authorization: Optional[str] = Header(default=None, description="Authorization header"),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """Hybrid endpoint: Deepgram transcription + internal speaker diarization and identification."""
    
    # Extract API key from authorization header
    api_key = None
    if authorization:
        if authorization.startswith("Token "):
            api_key = authorization[6:]
        elif authorization.startswith("Bearer "):
            api_key = authorization[7:]
        else:
            api_key = authorization
    
    # Use provided API key or fall back to service default
    deepgram_key = api_key or auth.deepgram_api_key
    if not deepgram_key:
        raise HTTPException(
            status_code=401,
            detail="Deepgram API key required. Provide via Authorization header or DEEPGRAM_API_KEY environment variable."
        )
    
    log.info(f"Processing hybrid transcribe-and-diarize request: {file.filename}")
    
    try:
        # Read audio file
        audio_data = await file.read()
        content_type = file.content_type or "audio/wav"
        
        # Step 1: Get transcription from Deepgram (NO diarization)
        deepgram_params = {
            "model": model,
            "language": language,
            "version": version,
            "punctuate": punctuate,
            "profanity_filter": profanity_filter,
            "diarize": False,  # Disable Deepgram diarization
            "multichannel": multichannel,
            "alternatives": alternatives,
            "numerals": numerals,
            "smart_format": smart_format,
            "paragraphs": paragraphs,
            "utterances": utterances,
            "detect_language": detect_language,
            "summarize": summarize,
            "sentiment": sentiment,
        }
        
        # Remove None values
        deepgram_params = {k: v for k, v in deepgram_params.items() if v is not None}
        
        log.info(f"Getting transcription from Deepgram API with params: {deepgram_params}")
        
        # Forward to Deepgram for transcription only
        deepgram_response = await forward_to_deepgram(
            audio_data=audio_data,
            content_type=content_type,
            params=deepgram_params,
            deepgram_api_key=deepgram_key
        )
        
        # Step 2: Get speaker diarization from internal service
        log.info("Performing internal speaker diarization and identification")
        
        # Create temporary file for internal diarization
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Step 2a: Perform diarization
            log.info(f"Performing speaker diarization on {tmp_path}")
            segments = await audio_backend.async_diarize(tmp_path)
            
            # Apply minimum duration filter
            original_count = len(segments)
            segments = [s for s in segments if s["duration"] >= min_duration]
            if len(segments) < original_count:
                log.info(f"Filtered out {original_count - len(segments)} segments shorter than {min_duration}s")
            
            # Step 2b: Identify speakers for each segment
            log.info(f"Identifying speakers for {len(segments)} segments")
            enhanced_segments = []
            
            for i, segment in enumerate(segments):
                try:
                    speaker_label = segment["speaker"]
                    start_time = segment["start"]
                    end_time = segment["end"]
                    
                    # Load audio segment
                    wav = audio_backend.load_wave(tmp_path, start_time, end_time)
                    
                    # Generate embedding
                    emb = await audio_backend.async_embed(wav)
                    
                    # Identify speaker
                    found, speaker_info, confidence = await speaker_db.identify(emb, user_id=user_id)
                    
                    # Validate confidence value
                    if confidence is not None:
                        if not isinstance(confidence, (int, float, np.number)):
                            log.warning(f"Invalid confidence type: {type(confidence)}, value: {confidence}")
                            confidence = 0.0
                        elif np.isnan(confidence) or np.isinf(confidence):
                            log.warning(f"Invalid confidence value: {confidence} (NaN/Inf)")
                            confidence = 0.0
                    
                    # Build enhanced segment
                    enhanced_segment = {
                        "speaker": speaker_label,
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "duration": round(end_time - start_time, 3),
                        "identified_as": speaker_info["name"] if found and speaker_info else None,
                        "identified_id": speaker_info["id"] if found and speaker_info else None,
                        "confidence": round(float(confidence), 3) if confidence else 0.0,
                        "status": "identified" if found else "unknown"
                    }
                    
                    enhanced_segments.append(enhanced_segment)
                    
                    if found and speaker_info:
                        confidence_str = safe_format_confidence(confidence, "hybrid_segment")
                        log.debug(f"Segment {i+1}: Identified as {speaker_info['name']} (confidence: {confidence_str})")
                    
                except Exception as e:
                    log.warning(f"Error processing segment {i+1}: {e}")
                    continue
            
            # Step 3: Combine Deepgram transcription with internal diarization
            enhanced_response = await combine_transcription_and_diarization(
                deepgram_response=deepgram_response,
                diarization_segments=enhanced_segments,
                user_id=user_id,
                confidence_threshold=speaker_confidence_threshold
            )
            
            return enhanced_response
            
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error in hybrid transcribe-and-diarize: {e}")
        raise HTTPException(500, f"Hybrid processing failed: {str(e)}")


@app.post("/speakers/reset")
async def reset_speakers(user_id: Optional[int] = None, db: UnifiedSpeakerDB = Depends(get_db)):
    """Reset all speakers (optionally for a specific user)."""
    if user_id is not None:
        await db.reset_user(user_id)
    else:
        # Reset all speakers (admin function)
        from simple_speaker_recognition.database import get_db_session
        from simple_speaker_recognition.database.models import Speaker
        
        db_session = get_db_session()
        try:
            db_session.query(Speaker).delete()
            db_session.commit()
        finally:
            db_session.close()
    
    return {"reset": True}


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(
    speaker_id: str, 
    delete_audio: bool = False,
    db: UnifiedSpeakerDB = Depends(get_db)
):
    """Delete a speaker and optionally delete associated audio files."""
    try:
        # Extract user_id from speaker_id for authorization
        user_id = extract_user_id_from_speaker_id(speaker_id)
        
        # Delete audio files if requested
        audio_deleted = False
        if delete_audio:
            speaker_audio_dir = auth.enrollment_audio_dir / str(user_id) / speaker_id
            if speaker_audio_dir.exists():
                try:
                    shutil.rmtree(speaker_audio_dir)
                    audio_deleted = True
                    log.info(f"Deleted audio directory for speaker {speaker_id}: {speaker_audio_dir}")
                except Exception as e:
                    log.warning(f"Failed to delete audio directory for speaker {speaker_id}: {e}")
        
        # Delete speaker from database
        await db.delete_speaker(speaker_id, user_id)
        
        result = {"deleted": True}
        if delete_audio:
            result["audio_deleted"] = audio_deleted
            
        return result
        
    except KeyError:
        raise HTTPException(404, "speaker not found") from None


def main():
    """Main entry point for the service."""
    # uvicorn import is at the top of the file
    
    host = os.getenv("SPEAKER_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("SPEAKER_SERVICE_PORT", "8085"))
    
    log.info(f"Starting Speaker Service on {host}:{port}")
    uvicorn.run("simple_speaker_recognition.api.service:app", host=host, port=port, reload=bool(os.getenv("DEV", False)))


@app.get("/v1/health")
async def deepgram_health():
    """Health check endpoint for Deepgram wrapper compatibility."""
    return {
        "status": "ok",
        "service": "Deepgram Speaker Enhancement Wrapper",
        "deepgram_configured": bool(auth.deepgram_api_key),
        "speaker_recognition_available": True,
        "device": str(device),
        "enrolled_speakers": speaker_db.get_speaker_count()
    }


@app.get("/deepgram/config")
async def get_deepgram_config():
    """Get Deepgram configuration for frontend.
    
    Returns the Deepgram API key if configured, allowing the frontend
    to make direct WebSocket connections to Deepgram for streaming.
    """
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    
    if not deepgram_api_key:
        raise HTTPException(
            status_code=404,
            detail="Deepgram API key not configured on server"
        )
    
    return {
        "api_key": deepgram_api_key,
        "status": "configured"
    }


@app.get("/speakers/export")
async def export_speakers(
    user_id: Optional[int] = Query(None, description="Export speakers for specific user (admin can export all)"),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """Export all speakers with embeddings for backup.
    
    Returns a JSON file containing all speaker data including embeddings,
    suitable for backup and restore operations.
    """
    db_session = get_db_session()
    try:
        # Query speakers based on user_id
        query = db_session.query(Speaker)
        if user_id:
            query = query.filter(Speaker.user_id == user_id)
        
        speakers = query.all()
        
        # Build export data
        export_data = {
            "version": "1.0",
            "export_date": datetime.utcnow().isoformat() + "Z",
            "user_id": user_id,
            "total_speakers": len(speakers),
            "speakers": []
        }
        
        for speaker in speakers:
            speaker_data = {
                "id": speaker.id,
                "name": speaker.name,
                "user_id": speaker.user_id,
                "created_at": speaker.created_at.isoformat() + "Z" if speaker.created_at else None,
                "updated_at": speaker.updated_at.isoformat() + "Z" if speaker.updated_at else None,
                "audio_sample_count": speaker.audio_sample_count,
                "total_audio_duration": speaker.total_audio_duration,
                "embedding_version": speaker.embedding_version,
                "notes": speaker.notes
            }
            
            # Include embedding data if available
            if speaker.embedding_data:
                try:
                    embedding = json.loads(speaker.embedding_data)
                    speaker_data["embedding_data"] = embedding
                except json.JSONDecodeError:
                    log.warning(f"Invalid embedding data for speaker {speaker.id}")
                    speaker_data["embedding_data"] = None
            else:
                speaker_data["embedding_data"] = None
            
            export_data["speakers"].append(speaker_data)
        
        # Return as downloadable JSON file
        return export_data
        
    except Exception as e:
        log.error(f"Error exporting speakers: {e}")
        raise HTTPException(status_code=500, detail="Failed to export speakers")
    finally:
        db_session.close()


@app.post("/speakers/import")
async def import_speakers(
    file: UploadFile = File(..., description="JSON file containing speaker data"),
    merge_strategy: str = Form("skip", description="How to handle conflicts: 'skip' or 'replace'"),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """Import speakers from a JSON backup file.
    
    Supports two merge strategies:
    - skip: Skip speakers that already exist
    - replace: Replace existing speakers with imported data
    """
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="File must be a JSON file")
    
    # Read and parse JSON file
    try:
        contents = await file.read()
        import_data = json.loads(contents)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    
    # Validate format
    if "version" not in import_data or "speakers" not in import_data:
        raise HTTPException(status_code=400, detail="Invalid import file format")
    
    if import_data["version"] != "1.0":
        raise HTTPException(status_code=400, detail=f"Unsupported version: {import_data['version']}")
    
    # Import statistics
    imported_count = 0
    skipped_count = 0
    replaced_count = 0
    errors = []
    
    db_session = get_db_session()
    try:
        for speaker_data in import_data["speakers"]:
            try:
                # Check if speaker exists
                existing_speaker = db_session.query(Speaker).filter(
                    Speaker.id == speaker_data["id"]
                ).first()
                
                if existing_speaker:
                    if merge_strategy == "skip":
                        skipped_count += 1
                        continue
                    elif merge_strategy == "replace":
                        # Delete existing speaker
                        db_session.delete(existing_speaker)
                        replaced_count += 1
                
                # Create new speaker
                speaker = Speaker(
                    id=speaker_data["id"],
                    name=speaker_data["name"],
                    user_id=speaker_data["user_id"],
                    audio_sample_count=speaker_data.get("audio_sample_count", 0),
                    total_audio_duration=speaker_data.get("total_audio_duration", 0.0),
                    embedding_version=speaker_data.get("embedding_version", 1),
                    notes=speaker_data.get("notes")
                )
                
                # Set embedding data if available
                if speaker_data.get("embedding_data"):
                    speaker.embedding_data = json.dumps(speaker_data["embedding_data"])
                
                db_session.add(speaker)
                imported_count += 1
                
            except Exception as e:
                errors.append(f"Failed to import speaker {speaker_data.get('id', 'unknown')}: {str(e)}")
                log.error(f"Error importing speaker: {e}")
        
        # Commit all changes
        db_session.commit()
        
        # Rebuild FAISS index after import
        db._rebuild_faiss_mapping()
        db._save_faiss_index()
        
    except Exception as e:
        db_session.rollback()
        log.error(f"Error during import: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")
    finally:
        db_session.close()
    
    # Return import summary
    return {
        "success": True,
        "imported": imported_count,
        "skipped": skipped_count,
        "replaced": replaced_count,
        "errors": errors,
        "total_processed": len(import_data["speakers"])
    }


if __name__ == "__main__":
    main()