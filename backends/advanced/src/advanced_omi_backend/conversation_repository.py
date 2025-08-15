"""Conversation Repository following proper repository pattern.

This module provides a clean abstraction layer for conversation data access,
separating domain logic from database implementation details. It replaces the
mixed concerns in AudioChunksCollectionHelper with proper separation.
"""

import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from advanced_omi_backend.transcript_coordinator import get_transcript_coordinator

logger = logging.getLogger(__name__)


class ConversationRepository:
    """Repository for conversation data with proper domain abstraction.

    This replaces AudioChunksCollectionHelper with clean domain-focused methods
    and proper async coordination integration.
    """

    def __init__(self, chunks_collection):
        self.col = chunks_collection
        self.coordinator = get_transcript_coordinator()
        logger.info("ConversationRepository initialized")

    # ==================== Conversation Lifecycle ====================

    async def create_conversation(
        self,
        audio_uuid: str,
        client_id: str,
        user_id: str,
        user_email: str,
        timestamp: float,
        session_id: Optional[str] = None,
    ) -> bool:
        """Create a new conversation record."""
        try:
            doc = {
                "audio_uuid": audio_uuid,
                "client_id": client_id,
                "user_id": user_id,
                "user_email": user_email,
                "timestamp": timestamp,
                "created_at": datetime.now(UTC),
                "session_id": session_id,
                "transcript": [],
                "speakers_identified": [],
                "memories": [],
                "memory_processing_status": "pending",
                "transcription_status": "pending",
            }

            await self.col.insert_one(doc)
            logger.info(f"Created conversation {audio_uuid} for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create conversation {audio_uuid}: {e}", exc_info=True)
            return False

    async def get_conversation(self, audio_uuid: str) -> Optional[Dict[str, Any]]:
        """Get a complete conversation by audio_uuid."""
        return await self.col.find_one({"audio_uuid": audio_uuid})

    async def conversation_exists(self, audio_uuid: str) -> bool:
        """Check if a conversation exists."""
        result = await self.col.find_one({"audio_uuid": audio_uuid}, {"_id": 1})
        return result is not None

    # ==================== Transcript Management ====================

    async def add_transcript_segment(self, audio_uuid: str, segment: Dict[str, Any]) -> bool:
        """Add a single transcript segment and signal completion."""
        try:
            result = await self.col.update_one(
                {"audio_uuid": audio_uuid}, {"$push": {"transcript": segment}}
            )

            if result.modified_count > 0:
                # Signal transcript coordinator that new content is available
                # This enables immediate processing by waiting memory processors
                self.coordinator.signal_transcript_ready(audio_uuid)
                logger.debug(f"Added transcript segment to {audio_uuid} and signaled completion")
                return True
            else:
                logger.warning(f"No conversation found to add transcript segment: {audio_uuid}")
                return False
        except Exception as e:
            logger.error(f"Failed to add transcript segment to {audio_uuid}: {e}", exc_info=True)
            return False

    async def get_transcript_segments(self, audio_uuid: str) -> List[Dict[str, Any]]:
        """Get all transcript segments for a conversation."""
        try:
            document = await self.col.find_one({"audio_uuid": audio_uuid}, {"transcript": 1})
            if document and "transcript" in document:
                return document["transcript"]
            return []
        except Exception as e:
            logger.error(f"Failed to get transcript segments for {audio_uuid}: {e}", exc_info=True)
            return []

    async def get_full_conversation_text(self, audio_uuid: str) -> Optional[str]:
        """Get the complete conversation as a single text string.

        This method is optimized for memory processing and provides a clean
        interface for getting conversation content. Falls back to raw transcript
        data if segments are not available.
        """
        try:
            segments = await self.get_transcript_segments(audio_uuid)
            if segments:
                # Build conversation text from segments with speaker information (preferred method)
                dialogue_lines = []
                for segment in segments:
                    text = segment.get("text", "").strip()
                    if text:
                        # Check for speaker information in multiple possible fields
                        speaker = None
                        
                        # Look for speaker in different possible fields from speaker recognition
                        if "speaker_parts" in segment and segment["speaker_parts"]:
                            # Speaker recognition format with identified speaker names
                            speaker = segment["speaker_parts"][0].get("speaker", "Unknown")
                        elif "speaker" in segment:
                            # Generic speaker label (e.g., "Speaker 0" or actual name)
                            speaker = segment["speaker"]
                        
                        # Format as dialogue line
                        if speaker and speaker != "Unknown" and speaker != "N/A":
                            dialogue_lines.append(f"{speaker}: {text}")
                        else:
                            # No speaker info available, just use the text
                            dialogue_lines.append(text)

                if dialogue_lines:
                    full_text = "\n\n".join(dialogue_lines).strip()
                    logger.debug(
                        f"Retrieved dialogue conversation text from segments for {audio_uuid}: {len(full_text)} chars, {len(dialogue_lines)} dialogue lines"
                    )
                    return full_text if full_text else None

            # Fallback: Check raw transcript data if no segments or empty segments
            document = await self.col.find_one({"audio_uuid": audio_uuid}, {"raw_transcript_data": 1})
            if document and document.get("raw_transcript_data"):
                raw_data = document["raw_transcript_data"]
                raw_text = raw_data.get("data", {}).get("text", "")
                if raw_text and raw_text.strip():
                    logger.debug(
                        f"Retrieved full conversation text from raw transcript for {audio_uuid}: {len(raw_text)} chars"
                    )
                    return raw_text.strip()

            return None

        except Exception as e:
            logger.error(
                f"Failed to get full conversation text for {audio_uuid}: {e}", exc_info=True
            )
            return None

    async def update_transcription_status(self, audio_uuid: str, status: str) -> bool:
        """Update the transcription status of a conversation."""
        try:
            result = await self.col.update_one(
                {"audio_uuid": audio_uuid},
                {
                    "$set": {
                        "transcription_status": status,
                        "transcription_updated_at": datetime.now(UTC),
                    }
                },
            )

            if result.modified_count > 0:
                logger.info(f"Updated transcription status for {audio_uuid} to {status}")
                return True
            else:
                logger.warning(
                    f"No conversation found to update transcription status: {audio_uuid}"
                )
                return False
        except Exception as e:
            logger.error(
                f"Failed to update transcription status for {audio_uuid}: {e}", exc_info=True
            )
            return False

    # ==================== Memory Management ====================

    async def add_memory_reference(
        self, audio_uuid: str, memory_id: str, status: str = "created"
    ) -> bool:
        """Add a memory reference to a conversation."""
        try:
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
                logger.info(f"Added memory reference {memory_id} to conversation {audio_uuid}")
                return True
            else:
                logger.warning(f"No conversation found to add memory reference: {audio_uuid}")
                return False
        except Exception as e:
            logger.error(f"Failed to add memory reference to {audio_uuid}: {e}", exc_info=True)
            return False

    async def update_memory_processing_status(self, audio_uuid: str, status: str) -> bool:
        """Update the memory processing status of a conversation."""
        try:
            result = await self.col.update_one(
                {"audio_uuid": audio_uuid},
                {
                    "$set": {
                        "memory_processing_status": status,
                        "memory_processing_updated_at": datetime.now(UTC),
                    }
                },
            )

            if result.modified_count > 0:
                logger.info(f"Updated memory processing status for {audio_uuid} to {status}")
                return True
            else:
                logger.warning(
                    f"No conversation found to update memory processing status: {audio_uuid}"
                )
                return False
        except Exception as e:
            logger.error(
                f"Failed to update memory processing status for {audio_uuid}: {e}", exc_info=True
            )
            return False

    # ==================== Speaker Management ====================

    async def add_speaker(self, audio_uuid: str, speaker_id: str) -> bool:
        """Add a speaker to the conversation's identified speakers list."""
        try:
            result = await self.col.update_one(
                {"audio_uuid": audio_uuid},
                {"$addToSet": {"speakers_identified": speaker_id}},
            )

            if result.modified_count > 0:
                logger.info(f"Added speaker {speaker_id} to conversation {audio_uuid}")
                return True
            else:
                # Could be that speaker was already in the set, which is fine
                logger.debug(f"Speaker {speaker_id} already present in conversation {audio_uuid}")
                return True
        except Exception as e:
            logger.error(f"Failed to add speaker to {audio_uuid}: {e}", exc_info=True)
            return False

    async def update_segment_speaker(
        self, audio_uuid: str, segment_index: int, speaker_id: str
    ) -> bool:
        """Update the speaker for a specific transcript segment."""
        try:
            result = await self.col.update_one(
                {"audio_uuid": audio_uuid},
                {"$set": {f"transcript.{segment_index}.speaker": speaker_id}},
            )

            if result.modified_count > 0:
                logger.info(
                    f"Updated segment {segment_index} speaker to {speaker_id} for {audio_uuid}"
                )
                return True
            else:
                logger.warning(f"No segment found to update speaker: {audio_uuid}[{segment_index}]")
                return False
        except Exception as e:
            logger.error(f"Failed to update segment speaker for {audio_uuid}: {e}", exc_info=True)
            return False

    # ==================== Query Methods ====================

    async def get_conversations_for_user(
        self, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get conversations for a specific user."""
        try:
            cursor = self.col.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
            return await cursor.to_list()
        except Exception as e:
            logger.error(f"Failed to get conversations for user {user_id}: {e}", exc_info=True)
            return []

    async def get_conversations_with_memories(
        self, user_ids: Optional[List[str]] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get conversations that have memory references."""
        try:
            query = {"memories": {"$exists": True, "$not": {"$size": 0}}}
            if user_ids:
                query["user_id"] = {"$in": user_ids}

            cursor = self.col.find(query).sort("timestamp", -1).limit(limit)
            return await cursor.to_list()
        except Exception as e:
            logger.error(f"Failed to get conversations with memories: {e}", exc_info=True)
            return []


# Global instance management
_conversation_repository: Optional[ConversationRepository] = None


def get_conversation_repository(chunks_collection=None) -> ConversationRepository:
    """Get the global ConversationRepository instance."""
    global _conversation_repository
    if _conversation_repository is None:
        if chunks_collection is None:
            # Import here to avoid circular imports
            from advanced_omi_backend.database import chunks_col

            chunks_collection = chunks_col
        _conversation_repository = ConversationRepository(chunks_collection)
    return _conversation_repository
