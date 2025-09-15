"""
Database configuration and utilities for the Friend-Lite backend.

This module provides centralized database access to avoid duplication
across main.py and router modules.
"""

import logging
import os
from datetime import UTC, datetime
from typing import Optional
import uuid

from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client.get_default_database("friend-lite")

# Collection references
chunks_col = db["audio_chunks"]
processing_runs_col = db["processing_runs"]
users_col = db["users"]
speakers_col = db["speakers"]


def get_database():
    """Get the MongoDB database instance."""
    return db


def get_collections():
    """Get commonly used collection references."""
    return {
        "chunks_col": chunks_col,
        "processing_runs_col": processing_runs_col,
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
                "created_at": datetime.now(UTC),
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
                "created_at": datetime.now(UTC),
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
                "created_at": datetime.now(UTC),
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
                "created_at": datetime.now(UTC),
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
        self, audio_uuid: str, status: str, provider: str = None, error_message: str = None
    ):
        """Update transcription status and completion timestamp.

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
                "segments": [],
                "status": status,
                "provider": provider,
                "created_at": datetime.now(UTC),
                "processing_run_id": None,
                "raw_data": {},
                "speakers_identified": []
            }
            if error_message:
                version_data["error_message"] = error_message

            result = await self.col.update_one(
                {"audio_uuid": audio_uuid},
                {
                    "$push": {"transcript_versions": version_data},
                    "$set": {
                        "active_transcript_version": version_id,
                        "transcription_status": status,
                        "transcription_updated_at": datetime.now(UTC).isoformat(),
                    }
                }
            )
        else:
            # Update existing active version
            update_doc = {
                f"transcript_versions.$[version].status": status,
                f"transcript_versions.$[version].updated_at": datetime.now(UTC),
                "transcription_status": status,
                "transcription_updated_at": datetime.now(UTC).isoformat(),
            }
            if provider:
                update_doc[f"transcript_versions.$[version].provider"] = provider
                update_doc["transcription_provider"] = provider
            if status == "COMPLETED":
                update_doc["transcription_completed_at"] = datetime.now(UTC).isoformat()
            if error_message:
                update_doc[f"transcript_versions.$[version].error_message"] = error_message
                update_doc["transcription_error"] = error_message

            result = await self.col.update_one(
                {"audio_uuid": audio_uuid},
                {"$set": update_doc},
                array_filters=[{"version.version_id": active_version}]
            )

        if result.modified_count > 0:
            logger.info(f"Updated transcription status to {status} for {audio_uuid}")
        return result.modified_count > 0

    # Additional interface compatibility methods

    async def update_transcript(self, audio_uuid, full_transcript):
        """Update the entire transcript list (for compatibility).

        Interface compatibility method - updates active transcript version.
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
                "segments": full_transcript,
                "status": "PENDING",
                "provider": None,
                "created_at": datetime.now(UTC),
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
                        "transcript": full_transcript
                    }
                }
            )
        else:
            # Update existing active version
            result = await self.col.update_one(
                {"audio_uuid": audio_uuid, "transcript_versions.version_id": active_version},
                {
                    "$set": {
                        "transcript_versions.$.segments": full_transcript,
                        "transcript": full_transcript
                    }
                }
            )

        return result.modified_count > 0

    async def store_raw_transcript_data(self, audio_uuid, raw_data, provider):
        """Store raw transcript data from transcription provider.

        Interface compatibility method - stores in active transcript version.
        """
        chunk = await self.get_chunk(audio_uuid)
        if not chunk:
            return False

        active_version = chunk.get("active_transcript_version")
        if not active_version:
            return False

        raw_data_with_timestamp = {
            "provider": provider,
            "data": raw_data,
            "stored_at": datetime.now(UTC).isoformat(),
        }

        result = await self.col.update_one(
            {"audio_uuid": audio_uuid, "transcript_versions.version_id": active_version},
            {
                "$set": {
                    "transcript_versions.$.raw_data": raw_data_with_timestamp,
                    "transcript_versions.$.provider": provider,
                    "raw_transcript_data": raw_data_with_timestamp
                }
            }
        )

        return result.modified_count > 0

    # New versioned methods for reprocessing functionality
    async def create_transcript_version(
        self,
        audio_uuid: str,
        segments: list = None,
        processing_run_id: str = None,
        provider: str = None,
        raw_data: dict = None
    ) -> Optional[str]:
        """Create a new transcript version."""
        version_id = str(uuid.uuid4())
        version_data = {
            "version_id": version_id,
            "segments": segments or [],
            "status": "PENDING",
            "provider": provider,
            "created_at": datetime.now(UTC),
            "processing_run_id": processing_run_id,
            "raw_data": raw_data or {},
            "speakers_identified": []
        }

        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$push": {"transcript_versions": version_data}}
        )

        if result.modified_count > 0:
            logger.info(f"Created new transcript version {version_id} for {audio_uuid}")
            return version_id
        return None

    async def create_memory_version(
        self,
        audio_uuid: str,
        transcript_version_id: str,
        memories: list = None,
        processing_run_id: str = None
    ) -> Optional[str]:
        """Create a new memory version."""
        version_id = str(uuid.uuid4())
        version_data = {
            "version_id": version_id,
            "memories": memories or [],
            "status": "PENDING",
            "created_at": datetime.now(UTC),
            "processing_run_id": processing_run_id,
            "transcript_version_id": transcript_version_id
        }

        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$push": {"memory_versions": version_data}}
        )

        if result.modified_count > 0:
            logger.info(f"Created new memory version {version_id} for {audio_uuid}")
            return version_id
        return None

    async def activate_transcript_version(self, audio_uuid: str, version_id: str) -> bool:
        """Activate a specific transcript version."""
        # First verify the version exists
        chunk = await self.col.find_one(
            {"audio_uuid": audio_uuid, "transcript_versions.version_id": version_id}
        )
        if not chunk:
            return False

        # Find the version and update compatibility fields
        version_data = None
        for version in chunk.get("transcript_versions", []):
            if version["version_id"] == version_id:
                version_data = version
                break

        if not version_data:
            return False

        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    "active_transcript_version": version_id,
                    "transcript": version_data["segments"],
                    "speakers_identified": version_data["speakers_identified"],
                    "transcription_status": version_data["status"],
                    "raw_transcript_data": version_data.get("raw_data", {})
                }
            }
        )

        if result.modified_count > 0:
            logger.info(f"Activated transcript version {version_id} for {audio_uuid}")
        return result.modified_count > 0

    async def activate_memory_version(self, audio_uuid: str, version_id: str) -> bool:
        """Activate a specific memory version."""
        # First verify the version exists
        chunk = await self.col.find_one(
            {"audio_uuid": audio_uuid, "memory_versions.version_id": version_id}
        )
        if not chunk:
            return False

        # Find the version and update compatibility fields
        version_data = None
        for version in chunk.get("memory_versions", []):
            if version["version_id"] == version_id:
                version_data = version
                break

        if not version_data:
            return False

        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    "active_memory_version": version_id,
                    "memories": version_data["memories"],
                    "memory_processing_status": version_data["status"]
                }
            }
        )

        if result.modified_count > 0:
            logger.info(f"Activated memory version {version_id} for {audio_uuid}")
        return result.modified_count > 0

    async def get_version_history(self, audio_uuid: str) -> dict:
        """Get all version history for a conversation."""
        chunk = await self.col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return {}

        return {
            "audio_uuid": audio_uuid,
            "active_transcript_version": chunk.get("active_transcript_version"),
            "active_memory_version": chunk.get("active_memory_version"),
            "transcript_versions": chunk.get("transcript_versions", []),
            "memory_versions": chunk.get("memory_versions", [])
        }


class ProcessingRunsRepository:
    """Repository for processing run tracking."""

    def __init__(self, collection):
        self.col = collection

    async def create_run(
        self,
        *,
        audio_uuid: str,
        run_type: str,  # 'transcript' or 'memory'
        user_id: str,
        trigger: str,  # 'manual_reprocess', 'initial_processing', etc.
        config_hash: str = None
    ) -> str:
        """Create a new processing run."""
        run_id = str(uuid.uuid4())
        doc = {
            "run_id": run_id,
            "audio_uuid": audio_uuid,
            "run_type": run_type,
            "user_id": user_id,
            "trigger": trigger,
            "config_hash": config_hash,
            "status": "PENDING",
            "started_at": datetime.now(UTC),
            "completed_at": None,
            "error_message": None,
            "result_version_id": None
        }
        await self.col.insert_one(doc)
        logger.info(f"Created processing run {run_id} for {audio_uuid}")
        return run_id

    async def update_run_status(
        self,
        run_id: str,
        status: str,
        error_message: str = None,
        result_version_id: str = None
    ) -> bool:
        """Update processing run status."""
        update_doc = {
            "status": status,
            "updated_at": datetime.now(UTC)
        }
        if status in ["COMPLETED", "FAILED"]:
            update_doc["completed_at"] = datetime.now(UTC)
        if error_message:
            update_doc["error_message"] = error_message
        if result_version_id:
            update_doc["result_version_id"] = result_version_id

        result = await self.col.update_one(
            {"run_id": run_id},
            {"$set": update_doc}
        )

        if result.modified_count > 0:
            logger.info(f"Updated processing run {run_id} status to {status}")
        return result.modified_count > 0

    async def get_run(self, run_id: str):
        """Get a processing run by ID."""
        return await self.col.find_one({"run_id": run_id})

    async def get_runs_for_audio(self, audio_uuid: str):
        """Get all processing runs for an audio UUID."""
        cursor = self.col.find({"audio_uuid": audio_uuid}).sort("started_at", -1)
        return await cursor.to_list(length=None)
