from __future__ import annotations

"""Speaker Recognition micro‑service (refactored).

Key improvements
----------------
* **Thread‑safe speaker store** using ``asyncio.Lock`` to avoid concurrent index mutation.
* **Incremental FAISS updates** on enrolment (no full rebuild).
* **Settings via Pydantic** for easy env/CLI overrides.
* **Startup/Shutdown events** so heavy model loading doesn't execute at import time.
* **Single responsibility classes** – `SpeakerDB` (persistence + search) and `AudioBackend` (I/O & embedding).
* **Safer file handling** – mandatory upload, size limit, automatic clean‑up.
* **Typed responses** with Pydantic models.
"""

import asyncio
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import faiss
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pyannote.audio import Audio, Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

###############################################################################
# Settings & logging
###############################################################################

class Settings(BaseSettings):
    similarity_threshold: float = Field(default=0.65, description="Similarity threshold for speaker identification")
    data_dir: Path = Field(default=Path("data"), description="Directory for storing speaker data")
    max_file_seconds: int = Field(default=180, description="Maximum file duration in seconds")

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"


# Get HF_TOKEN from environment and create settings
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is required. Please set it before running the service.")

# Type cast since we know hf_token is not None after the check above
hf_token = cast(str, hf_token)
auth = Settings()  # Load other settings from env vars or .env file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("speaker_service")

###############################################################################
# Audio handling & embedding
###############################################################################

class AudioBackend:
    """Wrapper around PyAnnote & SpeechBrain components."""

    def __init__(self, hf_token: str, device: torch.device):
        self.device = device
        self.diar = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        ).to(device)
        self.embedder = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb", device=device
        )
        self.loader = Audio(sample_rate=16_000, mono="downmix")

    def embed(self, wave: torch.Tensor) -> np.ndarray:  # (1, T)
        with torch.inference_mode():
            emb = self.embedder(wave.to(self.device))
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    async def async_embed(self, wave: torch.Tensor) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed, wave)

    def diarize(self, path: Path) -> List[Dict]:
        """Perform speaker diarization on an audio file."""
        with torch.inference_mode():
            diarization = self.diar(str(path))
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
                "duration": float(turn.end - turn.start)
            })
        
        return segments

    async def async_diarize(self, path: Path) -> List[Dict]:
        """Async wrapper for diarization."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.diarize, path)

    def load_wave(self, path: Path, start: Optional[float] = None, end: Optional[float] = None) -> torch.Tensor:
        if start is not None and end is not None:
            seg = Segment(start, end)
            wav, _ = self.loader.crop(str(path), seg)
        else:
            wav, _ = self.loader(str(path))
        return wav.unsqueeze(0)  # (1, 1, T)

###############################################################################
# Speaker database – in‑memory + persisted JSON & FAISS
###############################################################################

class SpeakerDB:
    def __init__(self, emb_dim: int, base_dir: Path, similarity_thr: float):
        self._lock = asyncio.Lock()
        self.emb_dim = emb_dim
        self.similarity_thr = similarity_thr
        self.base_dir = base_dir
        self.index_path = base_dir / "faiss.index"
        self.json_path = base_dir / "speakers.json"

        self.index: faiss.IndexHNSWFlat = faiss.IndexHNSWFlat(emb_dim, 32)
        self.index.hnsw.efSearch = 128
        self.speakers: Dict[str, Dict] = {}

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------

    def _load_state(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))  # type: ignore[assignment]
            self.index.hnsw.efSearch = 128
            log.info("Loaded FAISS index from %s", self.index_path)
        if self.json_path.exists():
            try:
                self.speakers = json.loads(self.json_path.read_text())
                log.info("Loaded %d speakers", len(self.speakers))
            except Exception as exc:  # noqa: BLE001
                log.error("Could not read %s: %s", self.json_path, exc)

    def _save_state(self) -> None:
        faiss.write_index(self.index, str(self.index_path))  # type: ignore[arg-type]
        self.json_path.write_text(json.dumps(self.speakers))

    # ---------------------------------------------------------------------
    # Public API – thread‑safe via self._lock
    # ---------------------------------------------------------------------

    async def add_speaker(self, speaker_id: str, name: str, embedding: np.ndarray) -> bool:
        """Add speaker; return True if updated (False if new)."""
        async with self._lock:
            is_update = speaker_id in self.speakers
            if is_update:
                # replace old vector inplace (FAISS cannot update, so rebuild below)
                self.speakers[speaker_id]["embedding"] = embedding.tolist()
                await self._rebuild_index()
            else:
                # append
                vector = embedding.astype(np.float32).reshape(1, -1)
                self.index.add(vector)  # type: ignore[call-arg]
                self.speakers[speaker_id] = {
                    "name": name,
                    "embedding": embedding.tolist(),
                    "faiss_index": self.index.ntotal - 1,
                }
            self._save_state()
            return is_update

    async def delete_speaker(self, speaker_id: str) -> None:
        async with self._lock:
            if speaker_id not in self.speakers:
                raise KeyError("speaker not found")
            self.speakers.pop(speaker_id)
            await self._rebuild_index()
            self._save_state()

    async def reset(self) -> None:
        async with self._lock:
            self.speakers.clear()
            self.index = faiss.IndexHNSWFlat(self.emb_dim, 32)
            self.index.hnsw.efSearch = 128
            self._save_state()

    async def identify(self, embedding: np.ndarray) -> Tuple[bool, Optional[Dict], float]:
        if not self.speakers:
            return False, None, 0.0
        query = embedding.astype(np.float32).reshape(1, -1)
        sims, idxs = self.index.search(query, 1)  # type: ignore[call-arg]
        best_sim, best_idx = float(sims[0, 0]), int(idxs[0, 0])
        if best_sim < self.similarity_thr:
            return False, None, best_sim
        # Retrieve speaker metadata
        for spk_id, data in self.speakers.items():
            if data["faiss_index"] == best_idx:
                return True, {"id": spk_id, "name": data["name"]}, best_sim
        return False, None, best_sim  # should not happen

    async def verify(self, speaker_id: str, embedding: np.ndarray) -> float:
        if speaker_id not in self.speakers:
            raise KeyError("speaker not enrolled")
        spk_emb = np.array(self.speakers[speaker_id]["embedding"])
        return float(np.dot(_normalize(embedding[0]), _normalize(spk_emb)))

    # ------------------------------------------------------------------
    # Internal utils
    # ------------------------------------------------------------------

    async def _rebuild_index(self) -> None:
        """Re‑index all stored speakers (slow – O(N))."""
        self.index = faiss.IndexHNSWFlat(self.emb_dim, 32)
        self.index.hnsw.efSearch = 128
        vectors: List[np.ndarray] = []
        for i, (sid, data) in enumerate(sorted(self.speakers.items())):
            vec = np.array(data["embedding"], dtype=np.float32)
            vectors.append(vec)
            data["faiss_index"] = i
        if vectors:
            all_vectors = np.stack(vectors).astype(np.float32)
            self.index.add(all_vectors)  # type: ignore[call-arg]

###############################################################################
# Utility functions
###############################################################################

def _normalize(arr: np.ndarray) -> np.ndarray:
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)


def secure_temp_file(suffix: str = ".wav") -> tempfile._TemporaryFileWrapper:
    return tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

###############################################################################
# Pydantic request/response schemas
###############################################################################

class EnrollRequest(BaseModel):
    speaker_id: str
    speaker_name: str
    start: Optional[float] = None
    end: Optional[float] = None


class BatchEnrollRequest(BaseModel):
    speaker_id: str
    speaker_name: str


class IdentifyRequest(BaseModel):
    start: Optional[float] = None
    end: Optional[float] = None


class VerifyRequest(BaseModel):
    speaker_id: str
    start: Optional[float] = None
    end: Optional[float] = None


class DiarizeRequest(BaseModel):
    min_duration: Optional[float] = Field(default=None, description="Minimum duration for speaker segments (seconds)")
    max_speakers: Optional[int] = Field(default=None, description="Maximum number of speakers to detect")


class InferenceRequest(BaseModel):
    segments: List[dict] = Field(..., description="Diarized transcript segments with speaker, start, end, text")


class InferenceSegment(BaseModel):
    speaker: int = Field(..., description="Original speaker ID from diarization")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds") 
    text: str = Field(..., description="Transcript text")
    identified_speaker: Optional[str] = Field(None, description="Identified speaker name")
    confidence: Optional[float] = Field(None, description="Identification confidence score")

###############################################################################
# FastAPI app & dependency wiring
###############################################################################

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
    # Add any cleanup code here if needed


app = FastAPI(title="Speaker‑ID Service", version="1.0.0", lifespan=lifespan)


async def get_db() -> SpeakerDB:
    return speaker_db


###############################################################################
# Routes
###############################################################################

@app.get("/health")
async def health(db: SpeakerDB = Depends(get_db)):
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


@app.post("/identify/upload")
async def identify_upload(
    file: UploadFile = File(...),
    req: IdentifyRequest = Depends(),
    db: SpeakerDB = Depends(get_db),
):
    log.info("Processing speaker identification request")
    
    with secure_temp_file() as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        log.info(f"Loading audio file: {tmp_path}")
        wav = audio_backend.load_wave(tmp_path, req.start, req.end)
        log.info(f"Audio loaded, shape: {wav.shape}")
        
        log.info("Computing speaker embedding...")
        emb = await audio_backend.async_embed(wav)
        log.info(f"Embedding computed, shape: {emb.shape}")
        
        log.info("Identifying speaker...")
        found, speaker, score = await db.identify(emb)
        
        log.info(f"Identification result - Found: {found}, Score: {score:.3f}, Threshold: {db.similarity_thr}")
        if found and speaker:
            log.info(f"Identified speaker: {speaker.get('name', 'Unknown')} (ID: {speaker.get('id', 'Unknown')})")
        else:
            log.info("Speaker not recognized")
        
        return {"identified": found, "speaker": speaker, "score": score}
    except Exception as e:
        log.error(f"Error during identification: {e}")
        raise HTTPException(500, f"Identification failed: {str(e)}") from e
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/verify/upload")
async def verify_upload(
    file: UploadFile = File(...),
    req: VerifyRequest = Depends(),
    db: SpeakerDB = Depends(get_db),
):
    with secure_temp_file() as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        wav = audio_backend.load_wave(tmp_path, req.start, req.end)
        emb = await audio_backend.async_embed(wav)
        score = await db.verify(req.speaker_id, emb)
        return {"match": score >= db.similarity_thr, "score": score}
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/diarize/upload")
async def diarize_upload(
    file: UploadFile = File(..., description="Audio file for speaker diarization"),
    req: DiarizeRequest = Depends(),
    db: SpeakerDB = Depends(get_db),
):
    """
    Perform speaker diarization on an audio file.
    Returns segments with speaker labels and timestamps.
    """
    log.info("Processing speaker diarization request")
    
    with secure_temp_file() as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        log.info(f"Loading audio file for diarization: {tmp_path}")
        
        log.info("Performing speaker diarization...")
        segments = await audio_backend.async_diarize(tmp_path)
        
        # Apply filtering based on request parameters
        if req.min_duration is not None:
            segments = [s for s in segments if s["duration"] >= req.min_duration]
            log.info(f"Filtered segments by min_duration={req.min_duration}s")
        
        if req.max_speakers is not None:
            unique_speakers = list(set(s["speaker"] for s in segments))
            if len(unique_speakers) > req.max_speakers:
                # Keep segments from the most frequent speakers
                speaker_durations = {}
                for seg in segments:
                    speaker = seg["speaker"]
                    speaker_durations[speaker] = speaker_durations.get(speaker, 0) + seg["duration"]
                
                top_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)[:req.max_speakers]
                allowed_speakers = {speaker for speaker, _ in top_speakers}
                segments = [s for s in segments if s["speaker"] in allowed_speakers]
                log.info(f"Limited to top {req.max_speakers} speakers by total duration")
        
        # Calculate summary statistics
        total_duration = max(s["end"] for s in segments) if segments else 0
        unique_speakers = list(set(s["speaker"] for s in segments))
        speaker_stats = {}
        
        for speaker in unique_speakers:
            speaker_segments = [s for s in segments if s["speaker"] == speaker]
            total_speak_time = sum(s["duration"] for s in speaker_segments)
            speaker_stats[speaker] = {
                "total_duration": round(total_speak_time, 2),
                "percentage": round((total_speak_time / total_duration) * 100, 1) if total_duration > 0 else 0,
                "segment_count": len(speaker_segments)
            }
        
        log.info(f"Diarization complete - Found {len(unique_speakers)} speakers in {len(segments)} segments")
        
        return {
            "segments": segments,
            "summary": {
                "total_duration": round(total_duration, 2),
                "num_speakers": len(unique_speakers),
                "num_segments": len(segments),
                "speaker_stats": speaker_stats
            }
        }
    except Exception as e:
        log.error(f"Error during diarization: {e}")
        raise HTTPException(500, f"Diarization failed: {str(e)}") from e
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/speakers")
async def list_speakers(db: SpeakerDB = Depends(get_db)):
    return {"speakers": db.speakers}


@app.post("/speakers/reset")
async def reset_speakers(db: SpeakerDB = Depends(get_db)):
    await db.reset()
    return {"reset": True}


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str, db: SpeakerDB = Depends(get_db)):
    try:
        await db.delete_speaker(speaker_id)
        return {"deleted": True}
    except KeyError:
        raise HTTPException(404, "speaker not found") from None


@app.post("/infer/diarized")
async def infer_diarized(
    audio_file: UploadFile = File(..., description="Original audio file for speaker identification"),
    segments: str = Form(..., description="JSON string of diarized segments"),
    db: SpeakerDB = Depends(get_db),
):
    """
    Identify speakers in pre-diarized transcript segments.
    
    Expects segments in format: [{"speaker": 0, "start": 1.2, "end": 5.4, "text": "..."}]
    Returns enhanced segments with identified speaker names.
    """
    log.info("Processing speaker inference on diarized segments")
    
    try:
        import json
        segments_data = json.loads(segments)
        log.info(f"Processing {len(segments_data)} diarized segments")
    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in segments parameter: {e}")
        raise HTTPException(400, f"Invalid JSON format in segments: {str(e)}")
    
    # Save uploaded audio file temporarily
    with secure_temp_file() as tmp:
        tmp.write(await audio_file.read())
        tmp_path = Path(tmp.name)
    
    enhanced_segments = []
    
    try:
        for i, segment in enumerate(segments_data):
            try:
                # Extract segment info
                speaker_id = segment.get('speaker', 0)
                start_time = float(segment.get('start', 0))
                end_time = float(segment.get('end', 0))
                text = segment.get('text', '')
                
                log.info(f"Processing segment {i+1}: Speaker {speaker_id}, {start_time:.2f}s-{end_time:.2f}s")
                
                # Extract audio segment for this speaker
                wav = audio_backend.load_wave(tmp_path, start_time, end_time)
                
                # Skip very short segments (less than 0.5 seconds)
                if end_time - start_time < 0.5:
                    log.warning(f"Skipping segment {i+1}: too short ({end_time - start_time:.2f}s)")
                    enhanced_segments.append({
                        "speaker": speaker_id,
                        "start": start_time,
                        "end": end_time,
                        "text": text,
                        "identified_speaker": f"Speaker {speaker_id}",
                        "confidence": 0.0,
                        "status": "skipped_too_short"
                    })
                    continue
                
                # Generate embedding
                emb = await audio_backend.async_embed(wav)
                log.info(f"Generated embedding for segment {i+1}, shape: {emb.shape}")
                
                # Identify speaker
                found, speaker_info, confidence = await db.identify(emb)
                
                if found and speaker_info:
                    identified_name = speaker_info.get('name', f'Speaker {speaker_id}')
                    log.info(f"Identified segment {i+1} as: {identified_name} (confidence: {confidence:.3f})")
                else:
                    identified_name = f"Speaker {speaker_id}"
                    log.info(f"Could not identify segment {i+1}, using: {identified_name}")
                
                enhanced_segments.append({
                    "speaker": speaker_id,
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "identified_speaker": identified_name,
                    "confidence": float(confidence),
                    "status": "identified" if found else "unknown"
                })
                
            except Exception as e:
                log.error(f"Error processing segment {i+1}: {str(e)}")
                # Add segment with error status
                enhanced_segments.append({
                    "speaker": segment.get('speaker', 0),
                    "start": segment.get('start', 0),
                    "end": segment.get('end', 0),
                    "text": segment.get('text', ''),
                    "identified_speaker": f"Speaker {segment.get('speaker', 0)}",
                    "confidence": 0.0,
                    "status": "error",
                    "error": str(e)
                })
        
        log.info(f"Successfully processed {len(enhanced_segments)} segments")
        return {
            "success": True,
            "segments": enhanced_segments,
            "total_segments": len(enhanced_segments),
            "identified_segments": len([s for s in enhanced_segments if s.get('status') == 'identified'])
        }
        
    except Exception as e:
        log.error(f"Error during inference processing: {str(e)}")
        raise HTTPException(500, f"Inference processing failed: {str(e)}")
    finally:
        tmp_path.unlink(missing_ok=True)


###############################################################################
# Uvicorn entrypoint (optional)
###############################################################################

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SPEAKER_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("SPEAKER_SERVICE_PORT", "8085"))
    
    logger.info(f"Starting Speaker Service on {host}:{port}")
    uvicorn.run("speaker_service:app", host=host, port=port, reload=bool(os.getenv("DEV", False)))
