"""
Speaker Diarization Service

This module provides speaker diarization functionality using pyannote.audio
and speaker identification capabilities.
"""

# Standard library imports
import os
import logging
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


# Global model instances
diar = None
audio_loader = None
embedding_model = None
enrolled_speakers: List[Dict] = []

# FAISS index for speaker embeddings
EMB_DIM = 512  # Default dimension, will be updated when model loads
index = faiss.IndexFlatIP(EMB_DIM)


def initialize_models():
    """Initialize all required models and components."""
    global diar, audio_loader, embedding_model, EMB_DIM
    
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


def normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Normalize speaker embedding vector.
    
    Args:
        embedding: Raw embedding tensor
        
    Returns:
        Normalized embedding tensor
    """
    return embedding / torch.norm(embedding, dim=1, keepdim=True)


def identify_speaker(embedding: torch.Tensor) -> Optional[str]:
    """Identify speaker from embedding.
    
    Args:
        embedding: Normalized speaker embedding
        
    Returns:
        Speaker identifier or None if not recognized
    """
    # TODO: Implement actual speaker identification logic
    # This would contain your actual speaker identification logic
    # For now, return None to use the fallback naming
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
                embedding = normalize_embedding(embedding)
                speaker_embeddings[spk] = embedding[0]

                # Identify speaker
                identified_speaker = identify_speaker(embedding[0])

                if not identified_speaker:
                    identified_speaker = f"Unknown_speaker_{len(enrolled_speakers) + hash(str(embedding[0][:4])) % 1000}"

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


# Initialize models when module is imported
initialize_models() 