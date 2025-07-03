"""
Speaker Diarization Service

This module provides speaker diarization functionality using pyannote.audio
and speaker identification capabilities.
"""

# Standard library imports
import os
import logging
import json
import traceback
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Union

# Third-party imports
import torch
import numpy as np
import faiss
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pyannote.audio import Audio, Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment

# Configuration
os.environ["HF_TOKEN"] = ""
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))

# Logging setup
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("speaker_service")

# Device configuration
device = torch.device("cuda")


# Data Models
@dataclass
class DiarizedSegment:
    """Represents a single diarized audio segment with speaker information."""
    speaker: str
    start: float
    end: float
    verified_speaker: Optional[str] = None


@dataclass
class DiarizationRequest:
    """Request model for diarization operations."""
    audio_file_path: str


@dataclass
class DiarizationResponse:
    """Response model containing diarization results."""
    segments: List[DiarizedSegment]
    speaker_identified: List[str]
    speaker_embeddings: Dict[str, List[float]]


@dataclass
class SpeakerEnrollmentRequest:
    speaker_id: str
    speaker_name: str
    audio_file_path: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


# Global model instances
diar = None
audio_loader = None
embedding_model = None
enrolled_speakers: List[Dict] = []

# FAISS index for speaker embeddings
EMB_DIM = 512  # Default dimension, will be updated when model loads
index = faiss.IndexFlatIP(EMB_DIM)

# Enrolled speakers path
ENROLLED_SPEAKERS_PATH = "enrolled_speakers.json"


def initialize_models():
    """Initialize all required models and components."""
    global diar, audio_loader, embedding_model, EMB_DIM, index
    
    # Initialize diarization pipeline
    try:
        diar = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
        ).to(device)
        log.info("Diarization pipeline loaded successfully")
    except Exception as e:
        log.warning(f"Failed to load diarization pipeline: {e}")
        diar = None

    # Initialize speaker embedding model
    try:
        embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=device
        )
        EMB_DIM = embedding_model.dimension
        # Recreate FAISS index with correct dimension
        index = faiss.IndexFlatIP(EMB_DIM)
        log.info("Speaker embedding model loaded successfully")
    except Exception as e:
        log.warning(f"Failed to load speaker embedding model: {e}")
        embedding_model = None

    # Initialize audio loader
    try:
        audio_loader = Audio(sample_rate=16000, mono="downmix")
        log.info("Audio loader loaded successfully")
    except Exception as e:
        log.warning(f"Failed to load audio loader: {e}")
        audio_loader = None

    log.info(f"Speaker Recognition service initialized with embedding dimension: {EMB_DIM}")


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize speaker embedding vector (NumPy version)."""
    return embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)


def save_enrolled_speakers(filepath: str = ENROLLED_SPEAKERS_PATH):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Only save id, name, and embedding as list
            json.dump([
                {"id": s["id"], "name": s["name"], "embedding": s["embedding"].tolist()}
                for s in enrolled_speakers
            ], f, indent=2)
        log.info(f"Enrolled speakers saved to {filepath}")
    except Exception as e:
        log.error(f"Failed to save enrolled speakers: {e}")


def load_enrolled_speakers(filepath: str = ENROLLED_SPEAKERS_PATH):
    global enrolled_speakers, index
    try:
        if not os.path.exists(filepath):
            log.info(f"No enrolled speakers file found at {filepath}")
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            enrolled_speakers.clear()
            for entry in data:
                enrolled_speakers.append({
                    "id": entry["id"],
                    "name": entry["name"],
                    "embedding": np.array(entry["embedding"], dtype=np.float32)
                })
            # Rebuild FAISS index
            index = faiss.IndexFlatIP(EMB_DIM)
            if enrolled_speakers:
                embeddings = np.vstack([spk["embedding"] for spk in enrolled_speakers])
                index.add(embeddings.astype(np.float32))
        log.info(f"Loaded {len(enrolled_speakers)} enrolled speakers from {filepath}")
    except Exception as e:
        log.error(f"Failed to load enrolled speakers: {e}")


def enroll_speaker(speaker_id: str, speaker_name: str, audio_file: str, start_time: Optional[float] = None, end_time: Optional[float] = None, save: bool = True) -> bool:
    """Enroll a speaker from an audio file (optionally a segment)."""
    global index, enrolled_speakers
    if not audio_loader or not embedding_model:
        log.error("Audio loader or embedding model not available")
        return False

    try:
        if start_time is not None and end_time is not None:
            segment = Segment(start_time, end_time)
            waveform, _ = audio_loader.crop(audio_file, segment)
        else:
            waveform, _ = audio_loader(audio_file)

        waveform = waveform.unsqueeze(0)
        embedding = embedding_model(waveform)
        # Ensure embedding is a numpy array
        if hasattr(embedding, 'detach'):
            embedding = embedding.detach().cpu().numpy()
        embedding = normalize_embedding(embedding)

        # Update existing speaker if present
        for existing_speaker in enrolled_speakers:
            if existing_speaker["id"] == speaker_id:
                log.info(f"Updating existing speaker: {speaker_id}")
                existing_speaker["name"] = speaker_name
                existing_speaker["embedding"] = embedding[0]
                # Rebuild FAISS index
                index = faiss.IndexFlatIP(EMB_DIM)
                if enrolled_speakers:
                    embeddings = np.vstack([spk["embedding"] for spk in enrolled_speakers])
                    index.add(embeddings.astype(np.float32))
                if save:
                    save_enrolled_speakers()
                return True

        # Add new speaker
        enrolled_speakers.append({
            "id": speaker_id,
            "name": speaker_name,
            "embedding": embedding[0]
        })
        index.add(embedding.astype(np.float32))
        log.info(f"Successfully enrolled speaker: {speaker_id} ({speaker_name})")
        if save:
            save_enrolled_speakers()
        return True

    except Exception as e:
        log.error(f"Failed to enroll speaker {speaker_id}: {e}")
        traceback.print_exc()
        return False


def identify_speaker(embedding: np.ndarray) -> Optional[str]:
    """Identify speaker from embedding using enrolled speakers and FAISS."""
    if len(enrolled_speakers) == 0:
        log.info("No enrolled speakers available for identification")
        return None
    
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    embedding = normalize_embedding(embedding)
    
    log.info(f"Identifying speaker against {len(enrolled_speakers)} enrolled speakers")
    similarities, indices = index.search(embedding.astype(np.float32), 1)
    similarity_score = similarities[0, 0]
    
    log.info(f"Best similarity score: {similarity_score:.4f} (threshold: 0.15)")
    
    # Also show enrolled speaker names for debugging
    enrolled_names = [spk["id"] for spk in enrolled_speakers]
    log.info(f"Available enrolled speakers: {enrolled_names}")
    
    if similarity_score > 0.15:  # Lowered threshold for testing
        speaker_idx = indices[0, 0]
        if speaker_idx < len(enrolled_speakers):
            identified_speaker = enrolled_speakers[speaker_idx]["id"]
            log.info(f"Speaker identified as: {identified_speaker}")
            return identified_speaker
    
    log.info("Speaker not identified (below threshold)")
    return None


def diarize_audio(request: DiarizationRequest) -> DiarizationResponse:
    """
    Perform speaker diarization on audio file.
    
    Args:
        request: DiarizationRequest containing audio file path
        
    Returns:
        DiarizationResponse with segments, identified speakers, and embeddings
        
    Raises:
        Exception: If models are not available or diarization fails
    """
    if not diar or not audio_loader or not embedding_model:
        raise Exception("Diarization models not available")
    
    try:
        # Perform diarization
        diar_result = diar(request.audio_file_path)

        # Extract segments
        segments = []
        for turn, _, speaker in diar_result.itertracks(yield_label=True):
            segments.append(DiarizedSegment(
                speaker=speaker,
                start=float(turn.start),
                end=float(turn.end),
                verified_speaker=None
            ))
        
        # Process each speaker
        speaker_embeddings = {}
        speaker_identified = set()

        for spk in diar_result.labels():
            try:
                # Get the longest segment for each speaker
                speaker_timeline = diar_result.label_timeline(spk)
                longest_segment = max(speaker_timeline, key=lambda x: x.duration)

                # Extract audio for the segment
                waveform, _ = audio_loader.crop(request.audio_file_path, longest_segment)
                waveform = waveform.unsqueeze(0)

                # Generate embedding
                embedding = embedding_model(waveform)
                # Ensure embedding is a numpy array
                if hasattr(embedding, 'detach'):
                    embedding = embedding.detach().cpu().numpy()
                embedding = normalize_embedding(embedding)
                speaker_embeddings[spk] = embedding[0]

                # Identify speaker
                log.info(f"Processing speaker {spk} - attempting identification")
                identified_speaker = identify_speaker(embedding)

                if not identified_speaker:
                    identified_speaker = f"Unknown_speaker_{len(enrolled_speakers) + hash(str(embedding[:4])) % 1000}"
                    log.info(f"Speaker {spk} not identified, using fallback name: {identified_speaker}")
                else:
                    log.info(f"Speaker {spk} successfully identified as: {identified_speaker}")

                speaker_identified.add(identified_speaker)

                # Update segments with verified speaker
                for seg in segments:
                    if seg.speaker == spk:
                        seg.verified_speaker = identified_speaker

            except Exception as e:
                log.error(f"Failed to process speaker '{spk}': {e}")
                # Fallback naming for failed speakers
                for seg in segments:
                    if seg.speaker == spk:
                        seg.verified_speaker = f"speaker_{spk}"
                speaker_identified.add(f"speaker_{spk}")

        return DiarizationResponse(
            segments=segments,
            speaker_identified=list(speaker_identified),
            speaker_embeddings={spk: emb.tolist() for spk, emb in speaker_embeddings.items()}
        )

    except Exception as e:
        log.error(f"Error performing speaker diarization: {e}")
        raise Exception(f"Failed to perform speaker diarization: {str(e)}")


def convert_to_json_format(diarization_response: DiarizationResponse) -> dict:
    """
    Convert DiarizationResponse to JSON format compatible with the GUI.
    
    Args:
        diarization_response: The response from diarize_audio
        
    Returns:
        Dictionary in the format expected by the GUI
    """
    segments = []
    for segment in diarization_response.segments:
        segments.append({
            'speaker': segment.speaker,
            'start': segment.start,
            'end': segment.end,
            'verified_speaker': segment.verified_speaker
        })
    
    return {
        'segments': segments,
        'speaker_identified': diarization_response.speaker_identified,
        'speaker_embeddings': diarization_response.speaker_embeddings
    }


def enroll_speaker_endpoint(request: SpeakerEnrollmentRequest):
    if not request.audio_file_path:
        raise HTTPException(status_code=400, detail="Audio file path is required")
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


# Initialize models when module is imported
initialize_models()
load_enrolled_speakers() 