#!/usr/bin/env python3
"""
Speaker Recognition Service

A standalone FastAPI service for speaker diarization, enrollment, and identification.
This service handles the heavy GPU computations separately from the main backend.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import faiss
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pyannote.audio import Audio, Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("speaker_service")

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token for gated models
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

# Initialize models
try:
    diar = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    ).to(device)
    log.info("Diarization pipeline loaded successfully")
except Exception as e:
    log.warning(f"Failed to load diarization pipeline: {e}")
    diar = None

try:
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=device
    )
    log.info("Embedding model loaded successfully")
except Exception as e:
    log.warning(f"Failed to load embedding model: {e}")
    embedding_model = None

try:
    audio_loader = Audio(sample_rate=16000, mono="downmix")
    log.info("Audio loader initialized successfully")
except Exception as e:
    log.warning(f"Failed to initialize audio loader: {e}")
    audio_loader = None

# Speaker database
EMB_DIM = embedding_model.dimension if embedding_model else 512
index = faiss.IndexFlatIP(EMB_DIM)
enrolled_speakers: List[Dict] = []

log.info(f"Speaker recognition service initialized with embedding dimension: {EMB_DIM}")

# Pydantic models
class SpeakerEnrollmentRequest(BaseModel):
    speaker_id: str
    speaker_name: str
    audio_file_path: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class SpeakerIdentificationRequest(BaseModel):
    audio_file_path: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class DiarizationRequest(BaseModel):
    audio_file_path: str

class DiarizationResponse(BaseModel):
    segments: List[Dict]
    speakers_identified: List[str]
    speaker_embeddings: Dict[str, List[float]]

# FastAPI app
app = FastAPI(
    title="Speaker Recognition Service",
    description="Speaker diarization, enrollment, and identification service",
    version="0.1.0"
)

# Helper functions
def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2 normalize embedding for cosine similarity."""
    return embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)

def enroll_speaker(speaker_id: str, speaker_name: str, audio_file: str, 
                  start_time: Optional[float] = None, end_time: Optional[float] = None) -> bool:
    """Enroll a new speaker from an audio file."""
    if not audio_loader or not embedding_model:
        log.error("Audio loader or embedding model not available")
        return False
        
    try:
        # Load audio
        if start_time is not None and end_time is not None:
            segment = Segment(start_time, end_time)
            waveform, _ = audio_loader.crop(audio_file, segment)
        else:
            waveform, _ = audio_loader(audio_file)
        
        # Extract embedding
        waveform = waveform.unsqueeze(0)  # Add batch dimension
        embedding = embedding_model(waveform)
        embedding = normalize_embedding(embedding)
        
        # Check if speaker already exists
        for existing_speaker in enrolled_speakers:
            if existing_speaker["id"] == speaker_id:
                log.info(f"Updating existing speaker: {speaker_id}")
                existing_speaker["name"] = speaker_name
                existing_speaker["embedding"] = embedding[0]
                
                # Rebuild FAISS index
                global index
                index = faiss.IndexFlatIP(EMB_DIM)
                if enrolled_speakers:
                    embeddings = np.vstack([spk["embedding"] for spk in enrolled_speakers])
                    index.add(embeddings.astype(np.float32))
                return True
        
        # Add new speaker
        enrolled_speakers.append({
            "id": speaker_id,
            "name": speaker_name,
            "embedding": embedding[0]
        })
        
        # Add to FAISS index
        index.add(embedding.astype(np.float32))
        
        log.info(f"Successfully enrolled speaker: {speaker_id} ({speaker_name})")
        return True
        
    except Exception as e:
        log.error(f"Failed to enroll speaker {speaker_id}: {e}")
        return False

def identify_speaker(embedding: np.ndarray) -> Optional[str]:
    """Identify a speaker from their embedding."""
    if len(enrolled_speakers) == 0:
        return None
    
    # Ensure embedding is 2D and normalized
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    embedding = normalize_embedding(embedding)
    
    # Search in FAISS index
    similarities, indices = index.search(embedding.astype(np.float32), 1)
    
    if similarities[0, 0] > SIMILARITY_THRESHOLD:
        speaker_idx = indices[0, 0]
        if speaker_idx < len(enrolled_speakers):
            return enrolled_speakers[speaker_idx]["id"]
    
    return None

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": {
            "diarization": diar is not None,
            "embedding": embedding_model is not None,
            "audio_loader": audio_loader is not None
        },
        "enrolled_speakers": len(enrolled_speakers)
    }

@app.post("/enroll")
async def enroll_speaker_endpoint(request: SpeakerEnrollmentRequest):
    """Enroll a new speaker."""
    if not request.audio_file_path:
        raise HTTPException(status_code=400, detail="audio_file_path is required")
    
    success = enroll_speaker(
        request.speaker_id,
        request.speaker_name,
        request.audio_file_path,
        request.start_time,
        request.end_time
    )
    
    if success:
        return {"success": True, "message": f"Speaker {request.speaker_id} enrolled successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to enroll speaker")

@app.post("/enroll/upload")
async def enroll_speaker_upload(
    speaker_id: str,
    speaker_name: str,
    audio_file: UploadFile = File(...),
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
):
    """Enroll a speaker from uploaded audio file."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        content = await audio_file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        success = enroll_speaker(speaker_id, speaker_name, tmp_file_path, start_time, end_time)
        if success:
            return {"success": True, "message": f"Speaker {speaker_id} enrolled successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to enroll speaker")
    finally:
        # Clean up temporary file
        Path(tmp_file_path).unlink(missing_ok=True)

@app.post("/identify")
async def identify_speaker_endpoint(request: SpeakerIdentificationRequest):
    """Identify a speaker from audio."""
    if not audio_loader or not embedding_model:
        raise HTTPException(status_code=503, detail="Audio processing models not available")
    
    try:
        # Load audio
        if request.start_time is not None and request.end_time is not None:
            segment = Segment(request.start_time, request.end_time)
            waveform, _ = audio_loader.crop(request.audio_file_path, segment)
        else:
            waveform, _ = audio_loader(request.audio_file_path)
        
        # Extract embedding
        waveform = waveform.unsqueeze(0)
        embedding = embedding_model(waveform)
        embedding = normalize_embedding(embedding)
        
        # Identify speaker
        identified_speaker = identify_speaker(embedding[0])
        
        if identified_speaker:
            # Get speaker info
            speaker_info = next((s for s in enrolled_speakers if s["id"] == identified_speaker), None)
            return {
                "identified": True,
                "speaker_id": identified_speaker,
                "speaker_info": speaker_info
            }
        else:
            return {"identified": False}
            
    except Exception as e:
        log.error(f"Error identifying speaker: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to identify speaker: {str(e)}")

@app.post("/diarize")
async def diarize_audio(request: DiarizationRequest) -> DiarizationResponse:
    """Perform speaker diarization on audio file."""
    if not diar or not audio_loader or not embedding_model:
        raise HTTPException(status_code=503, detail="Diarization models not available")
    
    try:
        # Run diarization
        def run_diarization(file_path: str):
            return diar(file_path)
        
        loop = asyncio.get_running_loop()
        diar_result = await loop.run_in_executor(None, run_diarization, request.audio_file_path)

        # Collect segments
        segments = []
        for turn, _, speaker in diar_result.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": float(turn.start),
                "end": float(turn.end),
                "verified_speaker": None
            })

        # Extract embeddings and identify speakers
        speaker_embeddings = {}
        speakers_identified = set()
        
        for spk in diar_result.labels():
            try:
                # Get longest segment for this speaker
                speaker_timeline = diar_result.label_timeline(spk)
                longest_segment = max(speaker_timeline, key=lambda x: x.duration)
                
                # Extract embedding
                waveform, _ = audio_loader.crop(request.audio_file_path, longest_segment)
                waveform = waveform.unsqueeze(0)
                embedding = embedding_model(waveform)
                embedding = normalize_embedding(embedding)
                speaker_embeddings[spk] = embedding[0]
                
                # Identify speaker
                identified_speaker = identify_speaker(embedding[0])
                if not identified_speaker:
                    identified_speaker = f"unknown_speaker_{len(enrolled_speakers) + hash(str(embedding[0][:4])) % 1000}"
                
                speakers_identified.add(identified_speaker)
                
                # Update segments
                for seg in segments:
                    if seg["speaker"] == spk:
                        seg["verified_speaker"] = identified_speaker
                        
            except Exception as e:
                log.error(f"Failed to process speaker {spk}: {e}")
                for seg in segments:
                    if seg["speaker"] == spk:
                        seg["verified_speaker"] = f"speaker_{spk}"
                speakers_identified.add(f"speaker_{spk}")

        return DiarizationResponse(
            segments=segments,
            speakers_identified=list(speakers_identified),
            speaker_embeddings={spk: emb.tolist() for spk, emb in speaker_embeddings.items()}
        )
        
    except Exception as e:
        log.error(f"Error during diarization: {e}")
        raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}")

@app.get("/speakers")
async def list_speakers():
    """List all enrolled speakers."""
    return {
        "speakers": [{"id": spk["id"], "name": spk["name"]} for spk in enrolled_speakers],
        "count": len(enrolled_speakers)
    }

@app.delete("/speakers/{speaker_id}")
async def remove_speaker(speaker_id: str):
    """Remove an enrolled speaker."""
    global enrolled_speakers, index
    
    original_count = len(enrolled_speakers)
    enrolled_speakers = [spk for spk in enrolled_speakers if spk["id"] != speaker_id]
    removed = len(enrolled_speakers) < original_count
    
    if removed:
        # Rebuild FAISS index
        index = faiss.IndexFlatIP(EMB_DIM)
        if enrolled_speakers:
            embeddings = np.vstack([spk["embedding"] for spk in enrolled_speakers])
            index.add(embeddings.astype(np.float32))
        
        log.info(f"Removed speaker: {speaker_id}")
        return {"success": True, "message": f"Speaker {speaker_id} removed"}
    else:
        raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info") 