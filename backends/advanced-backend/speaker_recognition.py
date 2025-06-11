# backends/advanced-backend/speaker_recognition.py
import asyncio
import logging
import os
from pathlib import Path

import faiss  # verification store
import numpy as np
import torch
from pyannote.audio import Model, Pipeline
from pyannote.core import Segment

log = logging.getLogger("speaker")

# ------------------------------------------------------------------ #
# Load pipelines once at process start
HF_TOKEN = os.getenv("HF_TOKEN")   # put your gated HuggingFace token here

# Diarization – v3.1 is still SOTA + GPU friendly
diar = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN,
).to("cuda" if torch.cuda.is_available() else "cpu")

# Embedding model used by that pipeline internally (same token)
emb_model = Model.from_pretrained(
    "pyannote/embedding",    # small 512-d WeSpeaker
    use_auth_token=HF_TOKEN,
    map_location="cuda" if torch.cuda.is_available() else "cpu",
)

EMB_DIM = emb_model.dimension
index = faiss.IndexFlatIP(EMB_DIM)         # cosine similarity after L2-norm

# persistent metadata
enrolled_ids: list[str] = []               # parallel to index vectors

# ------------------------------------------------------------------ #
async def process_file(wav_path: Path, audio_uuid: str, mongo_chunks):
    """Runs diarization, returns list[dict] "speaker / start / end"."""
    log.info("Diarizing %s", wav_path.name)
    
    # Run diarization in executor to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    diar_result = await loop.run_in_executor(None, diar, str(wav_path))

    # Collect segments per speaker
    segments = []
    for turn, _, speaker in diar_result.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": float(turn.start),
            "end": float(turn.end),
        })

    # ---------------------------------------------------------------- verify
    # one embedding per speaker (average of frames)
    emb_per_spk = {}
    for spk in diar_result.labels():
        emb = emb_model.crop(
            {"audio": str(wav_path)}, diar_result.support(speaker=spk)
        )
        emb = emb / np.linalg.norm(emb)   # L2 normalise
        emb_per_spk[spk] = emb

        # compare to store
        if len(enrolled_ids):
            D, I = index.search(emb[np.newaxis, :].astype("float32"), 1)
            if D[0, 0] > 0.8:             # similarity threshold
                verified_id = enrolled_ids[I[0, 0]]
            else:
                verified_id = f"spk_{len(enrolled_ids)}"
                index.add(emb[np.newaxis, :].astype("float32"))
                enrolled_ids.append(verified_id)
        else:
            verified_id = "spk_0"
            index.add(emb[np.newaxis, :].astype("float32"))
            enrolled_ids.append(verified_id)

        # map temporary speaker label to verified id
        for seg in segments:
            if seg["speaker"] == spk:
                seg["speaker"] = verified_id

    # ---------------------------------------------------------------- store
    await mongo_chunks.update_one(
        {"audio_uuid": audio_uuid},
        {
            "$set": {"transcript": segments},
            "$addToSet": {"speakers_identified": {"$each": list(set(s['speaker'] for s in segments))}}
        },
    )
    log.info("Speaker tags stored for %s", audio_uuid) 