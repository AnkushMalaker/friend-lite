"""Speaker database with FAISS indexing and persistence."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

log = logging.getLogger(__name__)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to unit length."""
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)


class SpeakerDB:
    """Thread-safe speaker database with FAISS indexing and JSON persistence."""

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
        """Load FAISS index and speaker metadata from disk."""
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
        """Save FAISS index and speaker metadata to disk."""
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
        """Delete a speaker from the database."""
        async with self._lock:
            if speaker_id not in self.speakers:
                raise KeyError("speaker not found")
            self.speakers.pop(speaker_id)
            await self._rebuild_index()
            self._save_state()

    async def reset(self) -> None:
        """Clear all speakers from the database."""
        async with self._lock:
            self.speakers.clear()
            self.index = faiss.IndexHNSWFlat(self.emb_dim, 32)
            self.index.hnsw.efSearch = 128
            self._save_state()

    async def identify(self, embedding: np.ndarray) -> Tuple[bool, Optional[Dict], float]:
        """Identify speaker from embedding using FAISS search."""
        if not self.speakers or self.index.ntotal == 0:
            return False, None, 0.0
        
        # Normalize query embedding to unit length for cosine similarity
        query_emb = _normalize(embedding.astype(np.float32))
        
        # Use FAISS to find nearest neighbors
        k = min(10, self.index.ntotal)  # Search top-k candidates
        similarities, indices = self.index.search(query_emb.reshape(1, -1), k)
        
        best_similarity = -1.0
        best_speaker = None
        
        # Check each candidate from FAISS search
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            # Find speaker by FAISS index
            speaker_found = None
            for spk_id, data in self.speakers.items():
                if data.get("faiss_index") == idx:
                    speaker_found = (spk_id, data)
                    break
            
            if speaker_found is None:
                continue
                
            spk_id, data = speaker_found
            cosine_similarity = float(similarity)
            
            log.debug(f"Speaker {spk_id}: similarity = {cosine_similarity:.4f}")
            
            if cosine_similarity > best_similarity:
                best_similarity = cosine_similarity
                best_speaker = {"id": spk_id, "name": data["name"]}
        
        # Check if best similarity meets threshold
        if best_similarity >= self.similarity_thr:
            log.info(f"Identified speaker: {best_speaker['name']} (similarity: {best_similarity:.4f}, threshold: {self.similarity_thr})")
            return True, best_speaker, best_similarity
        else:
            log.info(f"No speaker identified (best similarity: {best_similarity:.4f}, threshold: {self.similarity_thr})")
            return False, None, best_similarity

    async def verify(self, speaker_id: str, embedding: np.ndarray) -> float:
        """Verify speaker identity against stored embedding."""
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