"""Conversation Manager for handling conversation lifecycle and processing coordination.

This module separates conversation management concerns from ClientState to follow
the Single Responsibility Principle. It handles conversation closure, memory processing
queuing, and audio cropping coordination.
"""

import logging
import os
from typing import Optional

from advanced_omi_backend.processors import (
    AudioCroppingItem,
    MemoryProcessingItem,
    get_processor_manager,
)
from advanced_omi_backend.transcript_coordinator import get_transcript_coordinator

# Configuration
AUDIO_CROPPING_ENABLED = os.getenv("AUDIO_CROPPING_ENABLED", "false").lower() == "true"

audio_logger = logging.getLogger("audio")


class ConversationManager:
    """Manages conversation lifecycle and processing coordination.

    This class handles the responsibilities previously mixed into ClientState,
    providing a clean separation of concerns for conversation management.
    """

    def __init__(self):
        self.coordinator = get_transcript_coordinator()
        audio_logger.info("ConversationManager initialized")

    async def close_conversation(
        self,
        client_id: str,
        audio_uuid: str,
        user_id: str,
        user_email: Optional[str],
        conversation_start_time: float,
        speech_segments: dict,
        chunk_dir,  # Can be Path or str
    ) -> bool:
        """Close a conversation and coordinate all necessary processing.

        Args:
            client_id: Client identifier
            audio_uuid: Unique audio conversation identifier
            user_id: User identifier
            user_email: User email
            db_helper: Database helper instance
            conversation_start_time: When conversation started
            speech_segments: Speech segments for cropping
            chunk_dir: Directory for audio chunks

        Returns:
            True if conversation was closed successfully
        """
        audio_logger.info(f"🔒 Closing conversation {audio_uuid} for client {client_id}")

        try:
            # Get processor manager
            processor_manager = get_processor_manager()

            # Step 1: Close audio file in processor
            await processor_manager.close_client_audio(client_id)

            # Step 2: Queue memory processing if we have required data
            await self._queue_memory_processing(
                client_id=client_id,
                audio_uuid=audio_uuid,
                user_id=user_id,
                user_email=user_email,
            )

            # Step 3: Queue audio cropping if enabled and we have segments
            await self._queue_audio_cropping(
                client_id=client_id,
                audio_uuid=audio_uuid,
                user_id=user_id,
                conversation_start_time=conversation_start_time,
                speech_segments=speech_segments,
                chunk_dir=chunk_dir,
            )

            audio_logger.info(f"✅ Successfully closed conversation {audio_uuid}")
            return True

        except Exception as e:
            audio_logger.error(f"❌ Error closing conversation {audio_uuid}: {e}", exc_info=True)
            return False

    async def _queue_memory_processing(
        self,
        client_id: str,
        audio_uuid: str,
        user_id: str,
        user_email: Optional[str],
    ):
        """Queue memory processing for the conversation.

        Uses event coordination to ensure transcript is ready before processing.
        """
        audio_logger.info(f"💭 Memory processing check for client {client_id}:")
        audio_logger.info(f"    - audio_uuid: {audio_uuid}")
        audio_logger.info(f"    - user_id: {user_id}")
        audio_logger.info(f"    - user_email: {user_email}")

        if not all([audio_uuid, user_id, user_email]):
            audio_logger.warning(f"💭 Memory processing skipped - missing required data:")
            audio_logger.warning(f"    - audio_uuid: {bool(audio_uuid)}")
            audio_logger.warning(f"    - user_id: {bool(user_id)}")
            audio_logger.warning(f"    - user_email: {bool(user_email)}")
            return

        audio_logger.info(f"💭 Queuing memory processing for conversation {audio_uuid}")

        # Queue memory processing - the processor will handle event coordination
        processor_manager = get_processor_manager()
        await processor_manager.queue_memory(
            MemoryProcessingItem(
                client_id=client_id,
                user_id=user_id,
                user_email=user_email,
                audio_uuid=audio_uuid,
            )
        )

    async def _queue_audio_cropping(
        self,
        client_id: str,
        audio_uuid: str,
        user_id: str,
        conversation_start_time: float,
        speech_segments: dict,
        chunk_dir: str,
    ):
        """Queue audio cropping if enabled and speech segments are available."""
        if not AUDIO_CROPPING_ENABLED:
            audio_logger.debug(f"Audio cropping disabled for {audio_uuid}")
            return

        if audio_uuid not in speech_segments:
            audio_logger.debug(f"No speech segments found for {audio_uuid}")
            return

        segments = speech_segments[audio_uuid]
        if not segments:
            audio_logger.debug(f"Empty speech segments for {audio_uuid}")
            return

        # Build audio file paths following processor naming convention
        timestamp = int(conversation_start_time)
        wav_filename = f"{timestamp}_{client_id}_{audio_uuid}.wav"
        original_path = f"{str(chunk_dir)}/{wav_filename}"
        cropped_path = str(original_path).replace(".wav", "_cropped.wav")

        audio_logger.info(
            f"✂️ Queuing audio cropping for {audio_uuid} " f"with {len(segments)} speech segments"
        )

        processor_manager = get_processor_manager()
        await processor_manager.queue_cropping(
            AudioCroppingItem(
                client_id=client_id,
                user_id=user_id,
                audio_uuid=audio_uuid,
                original_path=original_path,
                speech_segments=segments,
                output_path=cropped_path,
            )
        )


# Global singleton instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get the global ConversationManager instance."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
