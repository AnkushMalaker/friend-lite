"""Conversation Manager for handling conversation lifecycle and processing coordination.

This module separates conversation management concerns from ClientState to follow
the Single Responsibility Principle. It handles conversation closure, memory processing
queuing, and audio cropping coordination.
"""

import logging
from typing import Optional

audio_logger = logging.getLogger("audio")


class ConversationManager:
    """Manages conversation lifecycle and processing coordination.

    This class handles the responsibilities previously mixed into ClientState,
    providing a clean separation of concerns for conversation management.

    V2 Architecture: Uses RQ jobs for all transcription and memory processing.
    """

    def __init__(self):
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
            # V2 Architecture: All processing handled by RQ jobs
            # Step 1: Enqueue final high-quality transcription via RQ
            # This will add a new transcript version and trigger memory processing
            from advanced_omi_backend.database import AudioChunksRepository

            repo = AudioChunksRepository()
            audio_session = await repo.get_chunk(audio_uuid)

            if audio_session and audio_session.get("conversation_id"):
                # Only enqueue if conversation was created (speech detected)
                import uuid
                from advanced_omi_backend.workers.transcription_jobs import transcribe_full_audio_job
                from advanced_omi_backend.controllers.queue_controller import transcription_queue, JOB_RESULT_TTL

                conversation_id = audio_session["conversation_id"]
                version_id = str(uuid.uuid4())  # Generate new version ID for final transcription
                audio_logger.info(f"📤 Enqueuing final transcription job for conversation {conversation_id}")

                job = transcription_queue.enqueue(
                    transcribe_full_audio_job,
                    conversation_id,
                    audio_uuid,
                    audio_session["audio_file_path"],
                    version_id,
                    user_id,
                    job_timeout=300,
                    result_ttl=JOB_RESULT_TTL,
                    job_id=f"transcript-reprocess_{conversation_id[:12]}",
                    description=f"Final transcription for conversation {conversation_id[:12]} (conversation close)"
                )
                audio_logger.info(f"✅ Enqueued final transcription job {job.id} for conversation {conversation_id}")
            else:
                audio_logger.info(f"⏭️ No conversation created for {audio_uuid} (no speech detected), skipping final transcription")

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
