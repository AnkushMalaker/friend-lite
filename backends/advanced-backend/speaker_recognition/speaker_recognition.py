# backends/advanced-backend/speaker_recognition.py
import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

# Check for required dependencies with better error messages
import faiss
import numpy as np
import torch
from scipy.spatial.distance import cdist

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

log = logging.getLogger("speaker")

# ------------------------------------------------------------------ #
# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")   # put your gated HuggingFace token here
SIMILARITY_THRESHOLD = 0.85  # Cosine similarity threshold (higher = more strict)

# Load pipelines once at process start
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Diarization pipeline - v3.1 is still SOTA + GPU friendly
try:
    diar = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    ).to(device)
except Exception as e:
    log.warning(f"Failed to load diarization pipeline: {e}")
    diar = None

# Embedding model for speaker verification using SpeechBrain
# This is more reliable than the raw pyannote/embedding model
try:
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=device
    )
except Exception as e:
    log.warning(f"Failed to load embedding model: {e}")
    embedding_model = None

# Audio loading utility 
try:
    audio_loader = Audio(sample_rate=16000, mono="downmix")
except Exception as e:
    log.warning(f"Failed to initialize audio loader: {e}")
    audio_loader = None

# Speaker database - in-memory for now (could be moved to MongoDB later)
EMB_DIM = embedding_model.dimension if embedding_model else 512  # Default dimension
index = faiss.IndexFlatIP(EMB_DIM)  # Inner product index for cosine similarity
enrolled_speakers: List[Dict] = []  # List of {id, name, embedding} dicts

log.info(f"Speaker recognition initialized with embedding dimension: {EMB_DIM}")

# ------------------------------------------------------------------ #
# Speaker Management Functions
# ------------------------------------------------------------------ #

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2 normalize embedding for cosine similarity."""
    return embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)

def enroll_speaker(speaker_id: str, speaker_name: str, audio_file: str, 
                  start_time: Optional[float] = None, end_time: Optional[float] = None) -> bool:
    """
    Enroll a new speaker from an audio file.
    
    Args:
        speaker_id: Unique identifier for the speaker
        speaker_name: Human-readable name for the speaker
        audio_file: Path to audio file
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
    
    Returns:
        bool: True if enrollment successful, False otherwise
    """
    if not audio_loader or not embedding_model:
        log.error("Audio loader or embedding model not available for speaker enrollment")
        return False
        
    try:
        # Load audio
        if start_time is not None and end_time is not None:
            segment = Segment(start_time, end_time)
            waveform, _ = audio_loader.crop(audio_file, segment)
        else:
            waveform, _ = audio_loader(audio_file)
        
        # Extract embedding
        # PretrainedSpeakerEmbedding expects (batch, channels, samples)
        waveform = waveform.unsqueeze(0)  # Add batch dimension
        embedding = embedding_model(waveform)  # Shape: (1, embedding_dim)
        embedding = normalize_embedding(embedding)
        
        # Check if speaker already exists
        for existing_speaker in enrolled_speakers:
            if existing_speaker["id"] == speaker_id:
                log.warning(f"Speaker {speaker_id} already enrolled. Updating embedding.")
                # Update existing speaker
                existing_speaker["name"] = speaker_name
                existing_speaker["embedding"] = embedding[0]
                # Update FAISS index (we'd need to rebuild it, so just add new one)
                index.add(embedding.astype(np.float32))
                return True
        
        # Add new speaker
        enrolled_speakers.append({
            "id": speaker_id,
            "name": speaker_name,
            "embedding": embedding[0]  # Remove batch dimension
        })
        
        # Add to FAISS index
        index.add(embedding.astype(np.float32))
        
        log.info(f"Successfully enrolled speaker: {speaker_id} ({speaker_name})")
        return True
        
    except Exception as e:
        log.error(f"Failed to enroll speaker {speaker_id}: {e}")
        return False

def identify_speaker(embedding: np.ndarray) -> Optional[str]:
    """
    Identify a speaker from their embedding.
    
    Args:
        embedding: Speaker embedding to identify
    
    Returns:
        str: Speaker ID if identified, None if no match
    """
    if len(enrolled_speakers) == 0:
        return None
    
    embedding = normalize_embedding(embedding.reshape(1, -1))
    
    # Search in FAISS index
    similarities, indices = index.search(embedding.astype(np.float32), 1)
    
    if similarities[0, 0] > SIMILARITY_THRESHOLD:
        speaker_idx = indices[0, 0]
        if speaker_idx < len(enrolled_speakers):
            return enrolled_speakers[speaker_idx]["id"]
    
    return None

def list_enrolled_speakers() -> List[Dict]:
    """Get list of all enrolled speakers."""
    return [{"id": spk["id"], "name": spk["name"]} for spk in enrolled_speakers]

def remove_speaker(speaker_id: str) -> bool:
    """
    Remove an enrolled speaker.
    Note: This is a simple implementation. In production, you'd want to rebuild the FAISS index.
    """
    global enrolled_speakers
    original_count = len(enrolled_speakers)
    enrolled_speakers = [spk for spk in enrolled_speakers if spk["id"] != speaker_id]
    removed = len(enrolled_speakers) < original_count
    
    if removed:
        # Rebuild FAISS index
        global index
        index = faiss.IndexFlatIP(EMB_DIM)
        if enrolled_speakers:
            embeddings = np.vstack([spk["embedding"] for spk in enrolled_speakers])
            index.add(embeddings.astype(np.float32))
        log.info(f"Removed speaker: {speaker_id}")
    
    return removed

# ------------------------------------------------------------------ #
# Audio Processing Functions  
# ------------------------------------------------------------------ #

async def process_file(wav_path: Path, audio_uuid: str, mongo_chunks):
    """
    Process audio file for speaker diarization and identification.
    
    Args:
        wav_path: Path to WAV file
        audio_uuid: Unique identifier for this audio
        mongo_chunks: MongoDB collection for storing results
    """
    log.info("Processing audio file: %s", wav_path.name)
    
    if not diar or not audio_loader or not embedding_model:
        log.warning("Speaker recognition models not available, skipping processing for %s", audio_uuid)
        return
    
    try:
        # Define a wrapper function for the diarization pipeline
        def run_diarization(file_path: str):
            return diar(file_path)
        
        # Run diarization in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        diar_result = await loop.run_in_executor(None, run_diarization, str(wav_path))

        # Collect segments per speaker
        segments = []
        for turn, _, speaker in diar_result.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,  # Temporary speaker label from diarization
                "start": float(turn.start),
                "end": float(turn.end),
                "verified_speaker": None  # Will be filled with identified speaker
            })

        # Extract embeddings for each diarized speaker and identify them
        speaker_embeddings = {}
        for spk in diar_result.labels():
            try:
                # Get all segments for this speaker
                speaker_timeline = diar_result.label_timeline(spk)
                
                # Extract embedding from the longest segment (most reliable)
                longest_segment = max(speaker_timeline, key=lambda x: x.duration)
                
                # Load audio for this segment
                waveform, _ = audio_loader.crop(str(wav_path), longest_segment)
                waveform = waveform.unsqueeze(0)  # Add batch dimension
                
                # Extract embedding
                embedding = embedding_model(waveform)
                embedding = normalize_embedding(embedding)
                speaker_embeddings[spk] = embedding[0]  # Remove batch dimension
                
                # Try to identify this speaker
                identified_speaker = identify_speaker(embedding[0])
                
                if identified_speaker:
                    log.info(f"Identified speaker {spk} as {identified_speaker}")
                else:
                    # Create anonymous speaker ID
                    identified_speaker = f"unknown_speaker_{len(enrolled_speakers) + hash(str(embedding[0][:4])) % 1000}"
                    log.info(f"Unknown speaker {spk}, assigned ID: {identified_speaker}")
                
                # Update segments with identified speaker
                for seg in segments:
                    if seg["speaker"] == spk:
                        seg["verified_speaker"] = identified_speaker
                        
            except Exception as e:
                log.error(f"Failed to process speaker {spk}: {e}")
                # Fall back to temporary speaker ID
                for seg in segments:
                    if seg["speaker"] == spk:
                        seg["verified_speaker"] = f"speaker_{spk}"

        # Store results in MongoDB
        transcript_segments = []
        speakers_identified = set()
        
        for seg in segments:
            # Create transcript segment with speaker info
            transcript_segment = {
                "speaker": seg["verified_speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": "",  # Text will be filled by transcription system
                "diarization_confidence": 1.0  # Could add confidence scores
            }
            transcript_segments.append(transcript_segment)
            speakers_identified.add(seg["verified_speaker"])

        # Update MongoDB
        await mongo_chunks.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    "diarization_segments": segments,  # Raw diarization output
                    "speaker_embeddings": {spk: emb.tolist() for spk, emb in speaker_embeddings.items()}
                },
                "$addToSet": {"speakers_identified": {"$each": list(speakers_identified)}}
            },
        )
        
        log.info("Speaker diarization completed for %s with %d speakers", 
                audio_uuid, len(speakers_identified))
        
    except Exception as e:
        log.error(f"Error processing audio file {wav_path}: {e}")
        raise

# ------------------------------------------------------------------ #
# Exports
# ------------------------------------------------------------------ #

__all__ = [
    "process_file", 
    "enroll_speaker", 
    "identify_speaker", 
    "list_enrolled_speakers", 
    "remove_speaker",
    "diar", 
    "embedding_model",
    "SIMILARITY_THRESHOLD"
]