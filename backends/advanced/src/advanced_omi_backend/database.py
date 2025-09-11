"""
Database configuration and utilities for the Friend-Lite backend.

This module provides centralized database access to avoid duplication
across main.py and router modules.
"""

import logging
import os
from datetime import UTC, datetime

from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client.get_default_database("friend-lite")

# Collection references
chunks_col = db["audio_chunks"]
users_col = db["users"]
speakers_col = db["speakers"]


def get_database():
    """Get the MongoDB database instance."""
    return db


def get_collections():
    """Get commonly used collection references."""
    return {
        "chunks_col": chunks_col,
        "users_col": users_col,
        "speakers_col": speakers_col,
    }


class AudioChunksRepository:
    """Async helpers for the audio_chunks collection."""

    def __init__(self, collection):
        self.col = collection

    async def create_chunk(
        self,
        *,
        audio_uuid,
        audio_path,
        client_id,
        timestamp,
        user_id=None,
        user_email=None,
        transcript=None,
        speakers_identified=None,
        memories=None,
        transcription_status="PENDING",
        memory_processing_status="PENDING",
    ):
        doc = {
            "audio_uuid": audio_uuid,
            "audio_path": audio_path,
            "client_id": client_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "user_email": user_email,
            "transcript": transcript or [],  # List of conversation segments
            "speakers_identified": speakers_identified or [],  # List of identified speakers
            "memories": memories or [],  # List of memory references created from this audio
            "transcription_status": transcription_status,  # PENDING, COMPLETED, FAILED, EMPTY
            "memory_processing_status": memory_processing_status,  # PENDING, COMPLETED, FAILED, SKIPPED
            "raw_transcript_data": {},  # Raw response from transcription provider
        }
        await self.col.insert_one(doc)

    async def add_transcript_segment(self, audio_uuid, transcript_segment):
        """Add a single transcript segment to the conversation."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid}, {"$push": {"transcript": transcript_segment}}
        )

    async def add_speaker(self, audio_uuid, speaker_id):
        """Add a speaker to the speakers_identified list if not already present."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$addToSet": {"speakers_identified": speaker_id}},
        )

    async def store_raw_transcript_data(self, audio_uuid, raw_data, provider):
        """Store raw transcript data from transcription provider."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    "raw_transcript_data": {
                        "provider": provider,
                        "data": raw_data,
                        "stored_at": datetime.now(UTC).isoformat(),
                    }
                }
            },
        )

    async def get_chunk(self, audio_uuid):
        """Get a chunk by audio_uuid."""
        return await self.col.find_one({"audio_uuid": audio_uuid})

    async def add_memory_reference(self, audio_uuid: str, memory_id: str, status: str = "created"):
        """Add memory reference to audio chunk."""
        memory_ref = {
            "memory_id": memory_id,
            "created_at": datetime.now(UTC).isoformat(),
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        result = await self.col.update_one(
            {"audio_uuid": audio_uuid}, {"$push": {"memories": memory_ref}}
        )
        if result.modified_count > 0:
            logger.info(f"Added memory reference {memory_id} to audio {audio_uuid}")
        return result.modified_count > 0

    async def update_memory_status(self, audio_uuid: str, memory_id: str, status: str):
        """Update memory status in audio chunk."""
        result = await self.col.update_one(
            {"audio_uuid": audio_uuid, "memories.memory_id": memory_id},
            {
                "$set": {
                    "memories.$.status": status,
                    "memories.$.updated_at": datetime.now(UTC).isoformat(),
                }
            },
        )
        if result.modified_count > 0:
            logger.info(f"Updated memory {memory_id} status to {status} for audio {audio_uuid}")
        return result.modified_count > 0

    async def remove_memory_reference(self, audio_uuid: str, memory_id: str):
        """Remove memory reference from audio chunk."""
        result = await self.col.update_one(
            {"audio_uuid": audio_uuid}, {"$pull": {"memories": {"memory_id": memory_id}}}
        )
        if result.modified_count > 0:
            logger.info(f"Removed memory reference {memory_id} from audio {audio_uuid}")
        return result.modified_count > 0

    async def get_chunk_by_audio_uuid(self, audio_uuid: str):
        """Get a chunk document by audio_uuid."""
        return await self.col.find_one({"audio_uuid": audio_uuid})

    async def get_transcript_segments(self, audio_uuid: str):
        """Get transcript segments for a specific audio UUID."""
        document = await self.col.find_one({"audio_uuid": audio_uuid}, {"transcript": 1})
        if document and "transcript" in document:
            return document["transcript"]
        return []

    async def update_transcript(self, audio_uuid, full_transcript):
        """Update the entire transcript list (for compatibility)."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid}, {"$set": {"transcript": full_transcript}}
        )

    async def update_segment_timing(self, audio_uuid, segment_index, start_time, end_time):
        """Update timing information for a specific transcript segment."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    f"transcript.{segment_index}.start": start_time,
                    f"transcript.{segment_index}.end": end_time,
                }
            },
        )

    async def update_segment_speaker(self, audio_uuid, segment_index, speaker_id):
        """Update the speaker for a specific transcript segment."""
        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$set": {f"transcript.{segment_index}.speaker": speaker_id}},
        )
        if result.modified_count > 0:
            logger.info(f"Updated segment {segment_index} speaker to {speaker_id} for {audio_uuid}")
        return result.modified_count > 0

    async def update_cropped_audio(
        self,
        audio_uuid: str,
        cropped_path: str,
        speech_segments: list[tuple[float, float]],
    ):
        """Update the chunk with cropped audio information."""
        cropped_duration = sum(end - start for start, end in speech_segments)

        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    "cropped_audio_path": cropped_path,
                    "speech_segments": [
                        {"start": start, "end": end} for start, end in speech_segments
                    ],
                    "cropped_duration": cropped_duration,
                    "cropped_at": datetime.now(UTC),
                }
            },
        )
        if result.modified_count > 0:
            logger.info(f"Updated cropped audio info for {audio_uuid}: {cropped_path}")
        return result.modified_count > 0

    async def update_transcription_status(
        self, audio_uuid: str, status: str, provider: str = None, error_message: str = None
    ):
        """Update transcription status and completion timestamp."""
        update_doc = {
            "transcription_status": status,
            "transcription_updated_at": datetime.now(UTC).isoformat(),
        }
        if provider:
            update_doc["transcription_provider"] = provider
        if status == "COMPLETED":
            update_doc["transcription_completed_at"] = datetime.now(UTC).isoformat()
        if error_message:
            update_doc["transcription_error"] = error_message

        result = await self.col.update_one({"audio_uuid": audio_uuid}, {"$set": update_doc})
        if result.modified_count > 0:
            logger.info(f"Updated transcription status to {status} for {audio_uuid}")
        return result.modified_count > 0

    async def update_memory_processing_status(
        self, audio_uuid: str, status: str, error_message: str = None
    ):
        """Update memory processing status and completion timestamp."""
        update_doc = {
            "memory_processing_status": status,
            "memory_processing_updated_at": datetime.now(UTC).isoformat(),
        }
        if status == "COMPLETED":
            update_doc["memory_processing_completed_at"] = datetime.now(UTC).isoformat()
        if error_message:
            update_doc["memory_processing_error"] = error_message

        result = await self.col.update_one({"audio_uuid": audio_uuid}, {"$set": update_doc})
        if result.modified_count > 0:
            logger.info(f"Updated memory processing status to {status} for {audio_uuid}")
        return result.modified_count > 0
