"""FastAPI service for speaker recognition and diarization."""

import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, cast

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import Field
from pydantic_settings import BaseSettings

from simple_speaker_recognition.core.audio_backend import AudioBackend
from simple_speaker_recognition.core.models import (
    BatchEnrollRequest,
    DiarizeAndIdentifyRequest,
    DiarizeRequest,
    IdentifyRequest,
    VerifyRequest,
)
from simple_speaker_recognition.core.speaker_db import SpeakerDB

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("speaker_service")


class Settings(BaseSettings):
    """Service configuration settings."""
    similarity_threshold: float = Field(default=0.15, description="Cosine similarity threshold for speaker identification (0.1-0.3 typical for ECAPA-TDNN)")
    data_dir: Path = Field(default=Path("data"), description="Directory for storing speaker data")
    max_file_seconds: int = Field(default=180, description="Maximum file duration in seconds")

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"


def secure_temp_file(suffix: str = ".wav") -> tempfile._TemporaryFileWrapper:
    """Create a secure temporary file."""
    return tempfile.NamedTemporaryFile(delete=False, suffix=suffix)


# Get HF_TOKEN from environment and create settings
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is required. Please set it before running the service.")

hf_token = cast(str, hf_token)
auth = Settings()  # Load other settings from env vars or .env file

# Global variables for storing initialized resources
audio_backend: AudioBackend
speaker_db: SpeakerDB
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler for startup and shutdown."""
    global audio_backend, speaker_db
    
    # Startup: Load models and initialize resources
    log.info("Loading models ..")
    assert hf_token is not None
    audio_backend = AudioBackend(hf_token, device)
    speaker_db = SpeakerDB(
        emb_dim=audio_backend.embedder.dimension,
        base_dir=auth.data_dir,
        similarity_thr=auth.similarity_threshold,
    )
    log.info("Models ready ✔ – device=%s", device)
    
    # Yield control to the application
    yield
    
    # Shutdown: Clean up resources if needed
    log.info("Shutting down speaker recognition service")


app = FastAPI(title="Simple Speaker Recognition Service", version="0.1.0", lifespan=lifespan)


async def get_db() -> SpeakerDB:
    """Get speaker database dependency."""
    return speaker_db


@app.get("/health")
async def health(db: SpeakerDB = Depends(get_db)):
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": str(device),
        "speakers": len(db.speakers),
    }


@app.post("/enroll/upload")
async def enroll_upload(
    file: UploadFile = File(..., description="WAV/FLAC <3 min"),
    speaker_id: str = Form(..., description="Unique speaker identifier"),
    speaker_name: str = Form(..., description="Speaker display name"),
    start: Optional[float] = Form(None, description="Start time in seconds"),
    end: Optional[float] = Form(None, description="End time in seconds"),
    db: SpeakerDB = Depends(get_db),
):
    """Enroll a speaker from uploaded audio file."""
    log.info(f"Enrolling speaker: {speaker_name} (ID: {speaker_id})")
    
    # Persist temporary file
    with secure_temp_file() as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        log.info(f"Loading audio file: {tmp_path}")
        wav = audio_backend.load_wave(tmp_path, start, end)
        log.info(f"Audio loaded, shape: {wav.shape}")
        
        log.info("Computing speaker embedding...")
        emb = await audio_backend.async_embed(wav)
        log.info(f"Embedding computed, shape: {emb.shape}")
        
        log.info(f"Adding speaker to database...")
        updated = await db.add_speaker(speaker_id, speaker_name, emb[0])
        
        if updated:
            log.info(f"Successfully updated existing speaker: {speaker_id}")
        else:
            log.info(f"Successfully enrolled new speaker: {speaker_id}")
        
        return {"updated": updated, "speaker_id": speaker_id}
    except Exception as e:
        log.error(f"Error during enrollment: {e}")
        raise HTTPException(500, f"Enrollment failed: {str(e)}") from e
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/enroll/batch")
async def enroll_batch(
    files: List[UploadFile] = File(..., description="Multiple audio files for same speaker"),
    req: BatchEnrollRequest = Depends(),
    db: SpeakerDB = Depends(get_db),
):
    """Enroll a speaker using multiple audio segments, computing average embedding."""
    log.info(f"Batch enrolling speaker: {req.speaker_name} (ID: {req.speaker_id}) with {len(files)} files")
    
    embeddings = []
    temp_paths = []
    
    try:
        # Process each audio file
        for i, file in enumerate(files):
            log.info(f"Processing file {i+1}/{len(files)}: {file.filename}")
            
            # Save to temporary file
            with secure_temp_file() as tmp:
                tmp.write(await file.read())
                tmp_path = Path(tmp.name)
                temp_paths.append(tmp_path)
            
            # Load and embed
            try:
                wav = audio_backend.load_wave(tmp_path)
                emb = await audio_backend.async_embed(wav)
                embeddings.append(emb[0])
                log.info(f"Successfully embedded file {i+1}")
            except Exception as e:
                log.warning(f"Failed to process file {i+1}: {e}")
                continue
        
        if not embeddings:
            raise HTTPException(400, "No valid audio files could be processed")
        
        # Compute average embedding
        log.info(f"Computing average embedding from {len(embeddings)} segments")
        embeddings_array = np.array(embeddings)
        average_embedding = np.mean(embeddings_array, axis=0)
        
        # Normalize the average embedding
        average_embedding = average_embedding / np.linalg.norm(average_embedding)
        
        log.info(f"Average embedding computed, shape: {average_embedding.shape}")
        
        # Add to database
        updated = await db.add_speaker(req.speaker_id, req.speaker_name, average_embedding)
        
        if updated:
            log.info(f"Successfully updated existing speaker: {req.speaker_id}")
        else:
            log.info(f"Successfully enrolled new speaker: {req.speaker_id}")
        
        return {
            "updated": updated,
            "speaker_id": req.speaker_id,
            "num_segments": len(embeddings),
            "num_files": len(files)
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


@app.post("/diarize-and-identify")
async def diarize_and_identify(
    file: UploadFile = File(..., description="Audio file for diarization and speaker identification"),
    req: DiarizeAndIdentifyRequest = Depends(),
    db: SpeakerDB = Depends(get_db),
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
                
                if db.speakers:  # Only try to identify if we have enrolled speakers
                    # Temporarily override threshold for this identification
                    original_threshold = db.similarity_thr
                    db.similarity_thr = threshold
                    try:
                        found, speaker_info, confidence = await db.identify(emb)
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
async def list_speakers(db: SpeakerDB = Depends(get_db)):
    """List all enrolled speakers."""
    return {"speakers": db.speakers}


@app.post("/speakers/reset")
async def reset_speakers(db: SpeakerDB = Depends(get_db)):
    """Reset all speakers."""
    await db.reset()
    return {"reset": True}


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str, db: SpeakerDB = Depends(get_db)):
    """Delete a speaker."""
    try:
        await db.delete_speaker(speaker_id)
        return {"deleted": True}
    except KeyError:
        raise HTTPException(404, "speaker not found") from None


def main():
    """Main entry point for the service."""
    import uvicorn
    
    host = os.getenv("SPEAKER_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("SPEAKER_SERVICE_PORT", "8085"))
    
    log.info(f"Starting Speaker Service on {host}:{port}")
    uvicorn.run("simple_speaker_recognition.api.service:app", host=host, port=port, reload=bool(os.getenv("DEV", False)))


if __name__ == "__main__":
    main()