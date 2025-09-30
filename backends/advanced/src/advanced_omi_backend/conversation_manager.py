"""Conversation Manager for handling conversation lifecycle and processing coordination.

This module separates conversation management concerns from ClientState to follow
the Single Responsibility Principle. It handles conversation closure, memory processing
queuing, and audio cropping coordination.
"""

import logging
from typing import Optional

from advanced_omi_backend.processors import (
    get_processor_manager,
)
from advanced_omi_backend.transcript_coordinator import get_transcript_coordinator

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

            # Step 1: Close audio file in processor (only if transcription not already completed)
            # Check if transcription is already completed to avoid double-flushing
            processing_status = processor_manager.get_processing_status(client_id)
            transcription_completed = processing_status.get("stages", {}).get("transcription", {}).get("completed", False)
            
            if not transcription_completed:
                audio_logger.info(f"🔄 Transcription not completed, calling close_client_audio for {client_id}")
                await processor_manager.close_client_audio(client_id)
            else:
                audio_logger.info(f"✅ Transcription already completed, skipping close_client_audio for {client_id}")

            # Step 2: Enqueue final high-quality transcription via RQ
            # This will add a new transcript version and trigger memory processing
            from advanced_omi_backend.database import AudioChunksRepository

            repo = AudioChunksRepository()
            audio_session = await repo.get_chunk(audio_uuid)

            if audio_session and audio_session.get("conversation_id"):
                # Only enqueue if conversation was created (speech detected)
                from advanced_omi_backend.rq_queue import enqueue_final_transcription

                conversation_id = audio_session["conversation_id"]
                audio_logger.info(f"📤 Enqueuing final transcription job for conversation {conversation_id}")

                job = enqueue_final_transcription(
                    conversation_id=conversation_id,
                    audio_uuid=audio_uuid,
                    audio_path=audio_session["audio_file_path"],
                    user_id=user_id,
                    client_id=client_id,
                    user_email=user_email or ""
                )
                audio_logger.info(f"✅ Enqueued final transcription job {job.id} for conversation {conversation_id}")
            else:
                audio_logger.info(f"⏭️ No conversation created for {audio_uuid} (no speech detected), skipping final transcription")

            # Step 3: Audio cropping is now handled at processor level after transcription
            # This ensures cropping happens with diarization segments when available
            # See transcription.py _queue_diarization_based_cropping() method

            audio_logger.info(f"✅ Successfully closed conversation {audio_uuid}")
            return True

        except Exception as e:
            audio_logger.error(f"❌ Error closing conversation {audio_uuid}: {e}", exc_info=True)
            return False



# Global singleton instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get the global ConversationManager instance."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
