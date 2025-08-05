"""Unified speaker database combining SQLite metadata with FAISS performance."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import faiss
import numpy as np

from simple_speaker_recognition.database import get_db_session
from simple_speaker_recognition.database.models import Speaker, User
from simple_speaker_recognition.database.queries import UserQueries

log = logging.getLogger(__name__)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize array to unit length."""
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)


class UnifiedSpeakerDB:
    """Unified speaker database combining SQLite metadata with FAISS performance."""

    def __init__(self, emb_dim: int, base_dir: Path, similarity_thr: float):
        self._lock = asyncio.Lock()
        self.emb_dim = emb_dim
        self.similarity_thr = similarity_thr
        self.base_dir = base_dir
        self.index_path = base_dir / "faiss.index"

        # FAISS index for fast similarity search
        self.index: faiss.IndexHNSWFlat = faiss.IndexHNSWFlat(emb_dim, 32)
        self.index.hnsw.efSearch = 128
        
        # Mapping from FAISS index position to (user_id, speaker_id)
        self.faiss_to_speaker: Dict[int, Tuple[int, str]] = {}

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self) -> None:
        """Load FAISS index and rebuild mapping from SQLite."""
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.index.hnsw.efSearch = 128
                log.info("Loaded FAISS index from %s", self.index_path)
            except Exception as e:
                log.warning("Could not load FAISS index, creating new: %s", e)
                self.index = faiss.IndexHNSWFlat(self.emb_dim, 32)
                self.index.hnsw.efSearch = 128
        
        # Rebuild mapping from SQLite
        self._rebuild_faiss_mapping()

    def _rebuild_faiss_mapping(self) -> None:
        """Rebuild FAISS index from SQLite data."""
        db = get_db_session()
        try:
            speakers = db.query(Speaker).all()
            self.faiss_to_speaker.clear()
            
            if not speakers:
                log.info("No speakers found in database")
                return
            
            # Recreate FAISS index
            self.index = faiss.IndexHNSWFlat(self.emb_dim, 32)
            self.index.hnsw.efSearch = 128
            
            vectors = []
            for i, speaker in enumerate(speakers):
                embedding_data = cast(Optional[str], speaker.embedding_data)
                if embedding_data:
                    try:
                        embedding = np.array(json.loads(embedding_data), dtype=np.float32)
                        vectors.append(embedding)
                        self.faiss_to_speaker[i] = (cast(int, speaker.user_id), cast(str, speaker.id))
                    except (json.JSONDecodeError, ValueError) as e:
                        log.warning("Invalid embedding data for speaker %s: %s", speaker.id, e)
            
            if vectors:
                all_vectors = np.stack(vectors).astype(np.float32)
                self.index.add(all_vectors)
                log.info("Rebuilt FAISS index with %d speakers", len(vectors))
            
        except Exception as e:
            log.error("Error rebuilding FAISS mapping: %s", e)
        finally:
            db.close()

    def _save_faiss_index(self) -> None:
        """Save FAISS index to disk."""
        try:
            faiss.write_index(self.index, str(self.index_path))
        except Exception as e:
            log.error("Error saving FAISS index: %s", e)

    async def add_speaker(self, speaker_id: str, name: str, embedding: np.ndarray, user_id: int, 
                         sample_count: int = 1, total_duration: float = 0.0) -> bool:
        """Add speaker with user association; return True if updated (False if new)."""
        async with self._lock:
            db = get_db_session()
            try:
                # Check if speaker exists
                existing_speaker = db.query(Speaker).filter(
                    Speaker.id == speaker_id,
                    Speaker.user_id == user_id
                ).first()
                
                is_update = existing_speaker is not None
                
                # Prepare embedding data
                embedding_json = json.dumps(embedding.tolist())
                
                if is_update:
                    # Update existing speaker (replace enrollment)
                    existing_speaker.name = name  # type: ignore[assignment]
                    existing_speaker.embedding_data = embedding_json  # type: ignore[assignment]
                    existing_speaker.audio_sample_count = sample_count  # type: ignore[assignment]
                    existing_speaker.total_audio_duration = total_duration  # type: ignore[assignment]
                    log.info("Updated existing speaker: %s (user: %d) with %d samples", speaker_id, user_id, sample_count)
                else:
                    # Create new speaker
                    new_speaker = Speaker(
                        id=speaker_id,
                        name=name,
                        user_id=user_id,
                        embedding_data=embedding_json,
                        audio_sample_count=sample_count,
                        total_audio_duration=total_duration
                    )
                    db.add(new_speaker)
                    log.info("Added new speaker: %s (user: %d) with %d samples", speaker_id, user_id, sample_count)
                
                db.commit()
                
                # Rebuild FAISS index to include changes
                self._rebuild_faiss_mapping()
                self._save_faiss_index()
                
                return is_update
                
            except Exception as e:
                db.rollback()
                log.error("Error adding speaker %s: %s", speaker_id, e)
                raise
            finally:
                db.close()

    async def delete_speaker(self, speaker_id: str, user_id: int) -> None:
        """Delete a speaker from the database."""
        async with self._lock:
            db = get_db_session()
            try:
                speaker = db.query(Speaker).filter(
                    Speaker.id == speaker_id,
                    Speaker.user_id == user_id
                ).first()
                
                if not speaker:
                    raise KeyError(f"Speaker {speaker_id} not found for user {user_id}")
                
                db.delete(speaker)
                db.commit()
                
                # Rebuild FAISS index
                self._rebuild_faiss_mapping()
                self._save_faiss_index()
                
                log.info("Deleted speaker: %s (user: %d)", speaker_id, user_id)
                
            except Exception as e:
                db.rollback()
                log.error("Error deleting speaker %s: %s", speaker_id, e)
                raise
            finally:
                db.close()

    async def reset_user(self, user_id: int) -> None:
        """Clear all speakers for a specific user."""
        async with self._lock:
            db = get_db_session()
            try:
                db.query(Speaker).filter(Speaker.user_id == user_id).delete()
                db.commit()
                
                # Rebuild FAISS index
                self._rebuild_faiss_mapping()
                self._save_faiss_index()
                
                log.info("Reset all speakers for user: %d", user_id)
                
            except Exception as e:
                db.rollback()
                log.error("Error resetting speakers for user %d: %s", user_id, e)
                raise
            finally:
                db.close()

    async def identify(self, embedding: np.ndarray, user_id: Optional[int] = None) -> Tuple[bool, Optional[Dict], float]:
        """Identify speaker from embedding using cosine similarity."""
        if self.index.ntotal == 0:
            return False, None, 0.0
        
        # Normalize query embedding
        query_emb = _normalize(embedding.astype(np.float32))
        
        best_similarity = -1.0
        best_speaker = None
        
        db = get_db_session()
        try:
            # Get all speakers (optionally filtered by user)
            query = db.query(Speaker)
            if user_id is not None:
                query = query.filter(Speaker.user_id == user_id)
            
            speakers = query.all()
            
            for speaker in speakers:
                embedding_data = cast(Optional[str], speaker.embedding_data)
                if embedding_data:
                    try:
                        stored_emb = np.array(json.loads(embedding_data), dtype=np.float32)
                        stored_emb = _normalize(stored_emb)
                        
                        # Compute cosine similarity
                        similarity = float(np.dot(query_emb.flatten(), stored_emb.flatten()))
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_speaker = {
                                "id": speaker.id,
                                "name": speaker.name,
                                "user_id": speaker.user_id
                            }
                    except (json.JSONDecodeError, ValueError) as e:
                        log.warning("Invalid embedding for speaker %s: %s", speaker.id, e)
            
            # Check if best similarity meets threshold
            if best_similarity >= self.similarity_thr and best_speaker is not None:
                log.info("Identified speaker: %s (similarity: %.4f)", 
                        best_speaker['name'], best_similarity)
                return True, best_speaker, best_similarity
            else:
                log.info("No speaker identified (best similarity: %.4f)", best_similarity)
                return False, None, best_similarity
                
        except Exception as e:
            log.error("Error during identification: %s", e)
            return False, None, 0.0
        finally:
            db.close()

    async def verify(self, speaker_id: str, embedding: np.ndarray, user_id: int) -> float:
        """Verify speaker identity against stored embedding."""
        db = get_db_session()
        try:
            speaker = db.query(Speaker).filter(
                Speaker.id == speaker_id,
                Speaker.user_id == user_id
            ).first()
            
            embedding_data = cast(Optional[str], speaker.embedding_data) if speaker else None
            if not speaker or not embedding_data:
                raise KeyError(f"Speaker {speaker_id} not enrolled for user {user_id}")
            
            stored_emb = np.array(json.loads(embedding_data), dtype=np.float32)
            return float(np.dot(_normalize(embedding.flatten()), _normalize(stored_emb)))
            
        except Exception as e:
            log.error("Error during verification: %s", e)
            raise
        finally:
            db.close()

    def get_speakers_for_user(self, user_id: int) -> List[Dict]:
        """Get all speakers for a specific user."""
        db = get_db_session()
        try:
            speakers = db.query(Speaker).filter(Speaker.user_id == user_id).all()
            return [
                {
                    "id": cast(str, speaker.id),
                    "name": cast(str, speaker.name),
                    "user_id": cast(int, speaker.user_id),
                    "created_at": speaker.created_at,
                    "updated_at": speaker.updated_at,
                    "audio_sample_count": cast(Optional[int], speaker.audio_sample_count) or 0,
                    "total_audio_duration": cast(Optional[float], speaker.total_audio_duration) or 0.0
                }
                for speaker in speakers
            ]
        finally:
            db.close()

    def get_speakers_with_embeddings(self, user_id: int) -> Dict[str, Dict]:
        """Get all speakers with their embeddings for a specific user."""
        db = get_db_session()
        try:
            speakers = db.query(Speaker).filter(Speaker.user_id == user_id).all()
            result = {}
            for speaker in speakers:
                embedding_data = cast(Optional[str], speaker.embedding_data)
                if embedding_data:
                    try:
                        embedding = json.loads(embedding_data)
                        result[cast(str, speaker.id)] = {
                            "name": cast(str, speaker.name),
                            "embedding": embedding
                        }
                    except (json.JSONDecodeError, ValueError) as e:
                        log.warning("Invalid embedding for speaker %s: %s", speaker.id, e)
            return result
        finally:
            db.close()

    def get_speaker_count(self) -> int:
        """Get total number of enrolled speakers."""
        db = get_db_session()
        try:
            return db.query(Speaker).count()
        finally:
            db.close()

    def ensure_admin_user(self) -> int:
        """Ensure admin user exists and return user ID."""
        db = get_db_session()
        try:
            admin_user = UserQueries.get_or_create_user(db, "admin")
            return cast(int, admin_user.id)
        finally:
            db.close()