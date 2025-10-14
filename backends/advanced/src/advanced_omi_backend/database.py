"""
Database configuration and utilities for the Friend-Lite backend.

This module provides centralized database access to avoid duplication
across main.py and router modules.
"""

import logging
import os
import time
from datetime import UTC, datetime
from typing import Optional
import uuid

from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
mongo_client = AsyncIOMotorClient(
    MONGODB_URI,
    maxPoolSize=50,  # Increased pool size for concurrent operations
    minPoolSize=10,  # Keep minimum connections ready
    maxIdleTimeMS=45000,  # Keep idle connections for 45 seconds
    serverSelectionTimeoutMS=5000,  # Fail fast if server unavailable
    socketTimeoutMS=20000,  # 20 second timeout for operations
)
db = mongo_client.get_default_database("friend-lite")

# Collection references (for non-Beanie collections)
users_col = db["users"]
chunks_col = db["audio_chunks"]  # Still used by AudioChunksRepository

# Note: conversations collection managed by Beanie
# Note: processing_runs replaced by RQ job tracking
# Beanie initialization happens in main.py during application startup


def get_database():
    """Get the MongoDB database instance."""
    return db


def get_collections():
    """Get commonly used collection references."""
    return {
        "users_col": users_col,
        "chunks_col": chunks_col,
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
        # Create initial transcript version if provided
        transcript_versions = []
        active_transcript_version = None

        if transcript:
            version_id = str(uuid.uuid4())
            transcript_versions.append({
                "version_id": version_id,
                "segments": transcript,
                "status": transcription_status,
                "provider": None,
                "created_at": datetime.now(UTC).isoformat(),
                "processing_run_id": None,
                "raw_data": {},
                "speakers_identified": speakers_identified or []
            })
            active_transcript_version = version_id

        # Create initial memory version if provided
        memory_versions = []
        active_memory_version = None

        if memories:
            version_id = str(uuid.uuid4())
            memory_versions.append({
                "version_id": version_id,
                "memories": memories,
                "status": memory_processing_status,
                "created_at": datetime.now(UTC).isoformat(),
                "processing_run_id": None,
                "transcript_version_id": active_transcript_version
            })
            active_memory_version = version_id

        doc = {
            "audio_uuid": audio_uuid,
            "audio_path": audio_path,
            "client_id": client_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "user_email": user_email,

            # Versioned transcript data
            "transcript_versions": transcript_versions,
            "active_transcript_version": active_transcript_version,

            # Versioned memory data
            "memory_versions": memory_versions,
            "active_memory_version": active_memory_version,

            # Compatibility fields (computed from active versions)
            "transcript": transcript or [],
            "speakers_identified": speakers_identified or [],
            "memories": memories or [],
            "transcription_status": transcription_status,
            "memory_processing_status": memory_processing_status,
            "raw_transcript_data": {}
        }
        await self.col.insert_one(doc)

    async def add_transcript_segment(self, audio_uuid, transcript_segment):
        """Add a single transcript segment to the conversation.

        Interface compatibility method - adds to active transcript version.
        Creates first transcript version if none exists.
        """
        chunk = await self.get_chunk(audio_uuid)
        if not chunk:
            return False

        active_version = chunk.get("active_transcript_version")
        if not active_version:
            # Create initial version if none exists
            version_id = str(uuid.uuid4())
            version_data = {
                "version_id": version_id,
                "segments": [transcript_segment],
                "status": "PENDING",
                "provider": None,
                "created_at": datetime.now(UTC).isoformat(),
                "processing_run_id": None,
                "raw_data": {},
                "speakers_identified": []
            }

            result = await self.col.update_one(
                {"audio_uuid": audio_uuid},
                {
                    "$push": {"transcript_versions": version_data},
                    "$set": {
                        "active_transcript_version": version_id,
                        # Update compatibility field too
                        "transcript": [transcript_segment]
                    }
                }
            )
        else:
            # Add to existing active version
            result = await self.col.update_one(
                {"audio_uuid": audio_uuid},
                {
                    "$push": {
                        f"transcript_versions.$[version].segments": transcript_segment,
                        # Update compatibility field too
                        "transcript": transcript_segment
                    }
                },
                array_filters=[{"version.version_id": active_version}]
            )

        return result.modified_count > 0

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
        """Get transcript segments for a specific audio UUID from active version."""
        document = await self.col.find_one(
            {"audio_uuid": audio_uuid},
            {"transcript_versions": 1, "active_transcript_version": 1, "transcript": 1}
        )

        if not document:
            return []

        # Try to get from active version first (new versioned approach)
        active_version_id = document.get("active_transcript_version")
        if active_version_id and "transcript_versions" in document:
            for version in document["transcript_versions"]:
                if version.get("version_id") == active_version_id:
                    return version.get("segments", [])

        # Fallback to legacy transcript field for backward compatibility
        if "transcript" in document:
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


    async def update_memory_processing_status(
        self, audio_uuid: str, status: str, error_message: str = None
    ):
        """Update memory processing status and completion timestamp.

        Interface compatibility method - updates active memory version.
        """
        chunk = await self.get_chunk(audio_uuid)
        if not chunk:
            return False

        active_version = chunk.get("active_memory_version")
        if not active_version:
            # Create initial memory version if none exists
            version_id = str(uuid.uuid4())
            version_data = {
                "version_id": version_id,
                "memories": [],
                "status": status,
                "created_at": datetime.now(UTC).isoformat(),
                "processing_run_id": None,
                "transcript_version_id": chunk.get("active_transcript_version")
            }
            if error_message:
                version_data["error_message"] = error_message

            result = await self.col.update_one(
                {"audio_uuid": audio_uuid},
                {
                    "$push": {"memory_versions": version_data},
                    "$set": {
                        "active_memory_version": version_id,
                        "memory_processing_status": status,
                        "memory_processing_updated_at": datetime.now(UTC).isoformat(),
                    }
                }
            )
        else:
            # Update existing active version
            update_doc = {
                f"memory_versions.$[version].status": status,
                f"memory_versions.$[version].updated_at": datetime.now(UTC),
                "memory_processing_status": status,
                "memory_processing_updated_at": datetime.now(UTC).isoformat(),
            }
            if status == "COMPLETED":
                update_doc["memory_processing_completed_at"] = datetime.now(UTC).isoformat()
            if error_message:
                update_doc[f"memory_versions.$[version].error_message"] = error_message
                update_doc["memory_processing_error"] = error_message

            result = await self.col.update_one(
                {"audio_uuid": audio_uuid},
                {"$set": update_doc},
                array_filters=[{"version.version_id": active_version}]
            )

        if result.modified_count > 0:
            logger.info(f"Updated memory processing status to {status} for {audio_uuid}")
        return result.modified_count > 0

    async def update_transcription_status(
        self, audio_uuid: str, status: str, error_message: Optional[str] = None, provider: Optional[str] = None
    ):
        """Update transcription processing status and completion timestamp.

        Interface compatibility method - updates active transcript version.
        """
        chunk = await self.get_chunk(audio_uuid)
        if not chunk:
            return False

        active_version = chunk.get("active_transcript_version")
        if not active_version:
            # Create initial transcript version if none exists
            version_id = str(uuid.uuid4())
            version_data = {
                "version_id": version_id,
                "transcript": "",
                "segments": [],
                "status": status,
                "provider": provider,
                "created_at": datetime.now(UTC).isoformat(),
                "processing_run_id": None,
                "raw_data": {},
                "speakers_identified": []
            }
            if error_message:
                version_data["error_message"] = error_message

            update_doc = {
                "active_transcript_version": version_id,
                "transcription_status": status,
                "transcription_updated_at": datetime.now(UTC).isoformat(),
            }
            if status == "COMPLETED":
                update_doc["transcription_completed_at"] = datetime.now(UTC).isoformat()

            result = await self.col.update_one(
                {"audio_uuid": audio_uuid},
                {
                    "$push": {"transcript_versions": version_data},
                    "$set": update_doc
                }
            )
        else:
            # Update existing active version
            update_doc = {
                "transcript_versions.$[version].status": status,
                "transcript_versions.$[version].updated_at": datetime.now(UTC).isoformat(),
                "transcription_status": status,
                "transcription_updated_at": datetime.now(UTC).isoformat(),
            }
            if status == "COMPLETED":
                update_doc["transcription_completed_at"] = datetime.now(UTC).isoformat()
            if error_message:
                update_doc["transcript_versions.$[version].error_message"] = error_message
                update_doc["transcription_error"] = error_message
            if provider:
                update_doc["transcript_versions.$[version].provider"] = provider
                update_doc["transcript_provider"] = provider

            result = await self.col.update_one(
                {"audio_uuid": audio_uuid},
                {"$set": update_doc},
                array_filters=[{"version.version_id": active_version}]
            )

        if result.modified_count > 0:
            logger.info(f"Updated transcription status to {status} for {audio_uuid}")
        return result.modified_count > 0

    # ========================================
    # SPEECH-DRIVEN CONVERSATIONS METHODS
    # ========================================

    async def add_audio_file_path(self, audio_uuid: str, file_path: str):
        """Add new audio file path to existing session."""
        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$push": {"audio_file_paths": file_path},
                "$set": {"updated_at": datetime.now(UTC).isoformat()}
            }
        )
        if result.modified_count > 0:
            logger.info(f"Added audio file path {file_path} to session {audio_uuid}")
        return result.modified_count > 0

    async def update_speech_detection(self, audio_uuid: str, **speech_data):
        """Update speech detection results."""
        update_doc = {
            "updated_at": datetime.now(UTC).isoformat()
        }

        # Add speech detection fields
        for key, value in speech_data.items():
            if key in ["has_speech", "conversation_created", "conversation_id",
                      "speech_start_time", "speech_end_time", "status"]:
                update_doc[key] = value

        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$set": update_doc}
        )
        if result.modified_count > 0:
            logger.info(f"Updated speech detection for {audio_uuid}: {speech_data}")
        return result.modified_count > 0

    async def mark_conversation_created(self, audio_uuid: str, conversation_id: str):
        """Mark that conversation was created for this audio."""
        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$set": {
                "conversation_created": True,
                "conversation_id": conversation_id,
                "has_speech": True,
                "status": "speech_detected",
                "updated_at": datetime.now(UTC).isoformat()
            }}
        )
        if result.modified_count > 0:
            logger.info(f"Marked conversation created for {audio_uuid} with ID {conversation_id}")
        return result.modified_count > 0

    async def get_sessions_with_speech(self, user_id: str, limit: int = 100):
        """Get audio sessions that have detected speech."""
        cursor = self.col.find({
            "user_id": user_id,
            "has_speech": True,
            "conversation_created": True
        }).sort("timestamp", -1).limit(limit)

        return await cursor.to_list(length=None)

    async def update_transcription_status(
        self, audio_uuid: str, status: str, error_message: str = None, provider: str = None
    ):
        """Update transcription status and completion timestamp.
        
        Args:
            audio_uuid: UUID of the audio chunk
            status: New status ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'EMPTY')
            error_message: Optional error message if status is 'FAILED'
            provider: Optional provider name for successful transcriptions
        """
        update_doc = {
            "transcription_status": status,
            "updated_at": datetime.now(UTC).isoformat()
        }
        
        if status == "COMPLETED":
            update_doc["transcription_completed_at"] = datetime.now(UTC).isoformat()
            if provider:
                update_doc["transcription_provider"] = provider
        elif status == "FAILED" and error_message:
            update_doc["transcription_error"] = error_message
        elif status == "EMPTY":
            update_doc["transcription_completed_at"] = datetime.now(UTC).isoformat()
            if provider:
                update_doc["transcription_provider"] = provider
            
        result = await self.col.update_one(
            {"audio_uuid": audio_uuid}, {"$set": update_doc}
        )
        return result.modified_count > 0


# ConversationsRepository removed - use Beanie Conversation model directly
# ProcessingRunsRepository removed - use RQ job tracking instead
