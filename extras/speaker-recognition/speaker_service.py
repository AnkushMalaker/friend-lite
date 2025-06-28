#!/usr/bin/env python3
import asyncio
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional

import faiss  # type: ignore
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from pyannote.audio import Audio, Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment

# -----------------------------------------------------------------------------
# Configuration ----------------------------------------------------------------
# -----------------------------------------------------------------------------

class Settings:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
        self.index_path = Path("data/faiss.index")
        self.speakers_json = Path("data/speakers.json")

auth = Settings()

# -----------------------------------------------------------------------------
# Logging ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("speaker_service")

# -----------------------------------------------------------------------------
# Device & models --------------------------------------------------------------
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", device)

diar = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=auth.hf_token).to(device)
log.info("pyannote diarization pipeline loaded ✔")
    
embedder = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=device)
EMB_DIM = embedder.dimension
log.info("SpeechBrain ECAPA embedder loaded ✔ (dim=%d)", EMB_DIM)

audio_loader = Audio(sample_rate=16000, mono="downmix")

# -----------------------------------------------------------------------------
# FAISS index & persistence ----------------------------------------------------
# -----------------------------------------------------------------------------

# Build HNSW index (approximate, but 10–50× faster than FlatIP once N > 10 k)
index: faiss.IndexHNSWFlat = faiss.IndexHNSWFlat(EMB_DIM, 32)  # efConstruction=32 by default
index.hnsw.efSearch = 128

speakers: Dict[str, Dict] = {}  # speaker_id -> {name, embedding, faiss_index}

if auth.index_path.exists():
    index = faiss.read_index(str(auth.index_path))  # type: ignore[arg-type]
    log.info("Loaded FAISS index from disk (%s)", auth.index_path)
if auth.speakers_json.exists():
    speakers = json.loads(auth.speakers_json.read_text())
    log.info("Loaded %d enrolled speakers", len(speakers))

# -----------------------------------------------------------------------------
# Helper utils -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def _normalize(emb: np.ndarray) -> np.ndarray:
    return emb / np.linalg.norm(emb, axis=-1, keepdims=True)


@torch.inference_mode()
def _embed(waveform: torch.Tensor) -> np.ndarray:  # (1, T)
    emb = embedder(waveform.to(device))
    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    return _normalize(emb)


def _save_state() -> None:
    faiss.write_index(index, str(auth.index_path))  # type: ignore[arg-type]
    auth.speakers_json.write_text(json.dumps(speakers))


# -----------------------------------------------------------------------------
# Pydantic request/response Schemas -------------------------------------------
# -----------------------------------------------------------------------------

class EnrollBody(BaseModel):
    speaker_id: str
    speaker_name: str
    audio_path: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None


class IdentifyBody(BaseModel):
    audio_path: str
    start: Optional[float] = None
    end: Optional[float] = None


class VerifyBody(BaseModel):
    speaker_id: str
    audio_path: str
    start: Optional[float] = None
    end: Optional[float] = None


class DiarizeBody(BaseModel):
    audio_path: str


# -----------------------------------------------------------------------------
# FastAPI app ------------------------------------------------------------------
# -----------------------------------------------------------------------------

app = FastAPI(title="Speaker-ID Service", version="0.3.0")

# -----------------------------------------------------------------------------
# Core functions ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def _load_wave(path: str, start: Optional[float], end: Optional[float]):
    if start is not None and end is not None:
        seg = Segment(start, end)
        wav, _ = audio_loader.crop(path, seg)
    else:
        wav, _ = audio_loader(path)
    return wav.unsqueeze(0)  # (1, 1, T)


def _add_speaker(speaker_id: str, name: str, embedding: np.ndarray) -> tuple[bool, str]:
    """Add or update speaker. Returns (is_update, message)"""
    is_update = speaker_id in speakers
    faiss_idx = index.ntotal  # Current index before adding
    
    if is_update:
        # Update existing speaker
        old_faiss_idx = speakers[speaker_id]["faiss_index"]
        speakers[speaker_id] = {
            "name": name, 
            "embedding": embedding.tolist(),
            "faiss_index": faiss_idx
        }
        # Add new embedding to index (we'll rebuild index to remove old one later)
        index.add(embedding.reshape(1, -1).astype(np.float32))
        
        # Rebuild index without the old embedding
        _rebuild_faiss_index()
        return True, f"Updated existing speaker '{speaker_id}'"
    else:
        # Add new speaker
        speakers[speaker_id] = {
            "name": name, 
            "embedding": embedding.tolist(),
            "faiss_index": faiss_idx
        }
        index.add(embedding.reshape(1, -1).astype(np.float32))
        return False, f"Enrolled new speaker '{speaker_id}'"


def _rebuild_faiss_index():
    """Rebuild FAISS index from current speakers dict"""
    global index
    index = faiss.IndexHNSWFlat(EMB_DIM, 32)
    index.hnsw.efSearch = 128
    
    # Sort speakers by their current faiss_index to maintain order
    sorted_speakers = sorted(speakers.items(), key=lambda x: x[1]["faiss_index"])
    
    if sorted_speakers:
        embeddings = []
        for i, (speaker_id, speaker_data) in enumerate(sorted_speakers):
            embeddings.append(np.array(speaker_data["embedding"]))
            # Update faiss_index to new position
            speakers[speaker_id]["faiss_index"] = i
        
        all_embs = np.stack(embeddings).astype(np.float32)
        index.add(all_embs)


# -----------------------------------------------------------------------------
# Endpoints --------------------------------------------------------------------
# -----------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
        "speakers": len(speakers),
    }


@app.post("/enroll")
async def enroll(body: EnrollBody):
    if not body.audio_path:
        raise HTTPException(400, detail="audio_path required")

    wav = _load_wave(body.audio_path, body.start, body.end)
    loop = asyncio.get_running_loop()
    emb = await loop.run_in_executor(None, _embed, wav)  # (1,1088) or (1,192)

    is_update, message = _add_speaker(body.speaker_id, body.speaker_name, emb[0])
    _save_state()
    return {
        "success": True, 
        "speaker_id": body.speaker_id, 
        "updated": is_update,
        "message": message
    }


@app.post("/enroll/upload")
async def enroll_upload(speaker_id: str, speaker_name: str, file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        body = EnrollBody(speaker_id=speaker_id, speaker_name=speaker_name, audio_path=tmp_path)
        return await enroll(body)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/identify/upload")
async def identify_upload(file: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        body = IdentifyBody(audio_path=tmp_path)
        return await identify(body)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/identify")
async def identify(body: IdentifyBody):
    # Check if any speakers are enrolled
    if not speakers:
        return {"identified": False, "score": 0.0, "message": "No speakers enrolled"}
    
    wav = _load_wave(body.audio_path, body.start, body.end)
    emb = await asyncio.get_running_loop().run_in_executor(None, _embed, wav)

    sims, idxs = index.search(emb.astype(np.float32), 1)  # type: ignore
    best_sim, best_idx = float(sims[0, 0]), int(idxs[0, 0])
    
    # Find speaker by faiss index
    speaker_found = None
    for speaker_id, speaker_data in speakers.items():
        if speaker_data["faiss_index"] == best_idx:
            speaker_found = {"id": speaker_id, "name": speaker_data["name"]}
            break
    
    if not speaker_found:
        log.error(f"No speaker found for faiss index {best_idx}")
        return {"identified": False, "score": best_sim, "message": "Index out of sync - please restart service"}
    
    if best_sim < auth.similarity_threshold:
        return {"identified": False, "score": best_sim}
    
    return {"identified": True, "speaker": speaker_found, "score": best_sim}


@app.post("/verify")
async def verify(body: VerifyBody):
    wav = _load_wave(body.audio_path, body.start, body.end)
    emb = await asyncio.get_running_loop().run_in_executor(None, _embed, wav)

    # find speaker 
    if body.speaker_id not in speakers:
        raise HTTPException(404, f"speaker_id {body.speaker_id} not enrolled")

    spk_emb = np.array(speakers[body.speaker_id]["embedding"])
    score = float(np.dot(_normalize(emb[0]), _normalize(spk_emb)))
    return {"match": score >= auth.similarity_threshold, "score": score}


@app.post("/diarize")
async def diarize(body: DiarizeBody):
    if diar is None:
        raise HTTPException(503, "Diarization model not loaded")

    diar_out = await asyncio.get_running_loop().run_in_executor(None, diar, body.audio_path)

    segments = [
        {"speaker": label, "start": float(seg.start), "end": float(seg.end)}
        for seg, _, label in diar_out.itertracks(yield_label=True)
    ]
    return {"segments": segments}


@app.get("/speakers")
async def list_speakers():
    return {"speakers": speakers}


@app.post("/speakers/reset")
async def reset_speakers():
    global speakers, index
    speakers = {}
    index = faiss.IndexHNSWFlat(EMB_DIM, 32)
    index.hnsw.efSearch = 128
    _save_state()
    return {"reset": True, "message": "All speakers cleared"}


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    global speakers, index

    if speaker_id not in speakers:
        raise HTTPException(404, "speaker not found")

    # Remove speaker
    speakers.pop(speaker_id)
    
    # Rebuild index without that speaker
    _rebuild_faiss_index()
    _save_state()
    return {"deleted": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("speaker_service:app", host="0.0.0.0", port=8001, reload=False) 