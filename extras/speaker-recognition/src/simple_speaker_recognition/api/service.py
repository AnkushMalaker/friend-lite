"""FastAPI service for speaker recognition and diarization."""

import asyncio
import logging
import os
import tempfile
import shutil
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, cast
from datetime import datetime

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import Field
from pydantic_settings import BaseSettings

from simple_speaker_recognition.core.audio_backend import AudioBackend
from simple_speaker_recognition.core.models import (
    BatchEnrollRequest,
    DiarizeAndIdentifyRequest,
    DiarizeRequest,
    IdentifyRequest,
    UserRequest,
    UserResponse,
    VerifyRequest,
)
from simple_speaker_recognition.core.unified_speaker_db import UnifiedSpeakerDB
from simple_speaker_recognition.database import init_db
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


class Settings(BaseSettings):
    """Service configuration settings."""
    similarity_threshold: float = Field(default=0.15, description="Cosine similarity threshold for speaker identification (0.1-0.3 typical for ECAPA-TDNN)")
    data_dir: Path = Field(default_factory=get_data_directory, description="Directory for storing speaker data")
    enrollment_audio_dir: Path = Field(default_factory=lambda: get_data_directory() / "enrollment_audio", description="Directory for storing enrollment audio files")
    max_file_seconds: int = Field(default=180, description="Maximum file duration in seconds")

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

# Global variables for storing initialized resources
audio_backend: AudioBackend
speaker_db: UnifiedSpeakerDB
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler for startup and shutdown."""
    global audio_backend, speaker_db
    
    # Startup: Initialize database and load models
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
    from simple_speaker_recognition.database import get_db_session
    from simple_speaker_recognition.database.models import Speaker
    import json
    
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
                    log.debug(f"Segment {i+1}: Identified as {speaker_info['name']} (confidence: {confidence:.3f})")
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


if __name__ == "__main__":
    main()