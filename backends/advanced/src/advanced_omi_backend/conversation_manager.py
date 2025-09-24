"""Conversation Manager for handling conversation lifecycle and processing coordination.

This module separates conversation management concerns from ClientState to follow
the Single Responsibility Principle. It handles conversation closure, memory processing
queuing, and audio cropping coordination.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Dict, List, Optional

from advanced_omi_backend.database import ConversationsRepository, conversations_col
from advanced_omi_backend.llm_client import async_generate
from advanced_omi_backend.processors import get_processor_manager

audio_logger = logging.getLogger("audio")


class ConversationManager:
    """Manages conversation lifecycle and processing coordination.

    This class handles the responsibilities previously mixed into ClientState,
    providing a clean separation of concerns for conversation management.
    """

    def __init__(self):
        audio_logger.info("ConversationManager initialized")

    async def create_conversation(self, audio_uuid: str, transcript_data: dict, speech_analysis: dict, chunk_repo):
        """Create conversation entry for detected speech."""
        try:
            # Get audio session info from audio_chunks
            audio_session = await chunk_repo.get_chunk(audio_uuid)
            if not audio_session:
                audio_logger.error(f"No audio session found for {audio_uuid}")
                return None

            # Create conversation data (title and summary will be generated after speaker recognition)
            conversation_id = str(uuid.uuid4())
            conversation_data = {
                "conversation_id": conversation_id,
                "audio_uuid": audio_uuid,
                "user_id": audio_session["user_id"],
                "client_id": audio_session["client_id"],
                "title": "Processing...",  # Placeholder - will be updated after speaker recognition
                "summary": "Processing...",  # Placeholder - will be updated after speaker recognition

                # Versioned system (source of truth)
                "transcript_versions": [],
                "active_transcript_version": None,
                "memory_versions": [],
                "active_memory_version": None,

                # Legacy compatibility fields (auto-populated on read)
                # Note: These will be auto-populated from active versions when retrieved

                "duration_seconds": speech_analysis.get("duration", 0.0),
                "speech_start_time": speech_analysis.get("speech_start", 0.0),
                "speech_end_time": speech_analysis.get("speech_end", 0.0),
                "speaker_names": {},
                "action_items": [],
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
                "session_start": datetime.fromtimestamp(audio_session.get("timestamp", 0) / 1000, tz=UTC),
                "session_end": datetime.now(UTC),
            }

            # Create conversation in conversations collection
            conversations_repo = ConversationsRepository(conversations_col)
            await conversations_repo.create_conversation(conversation_data)

            # Mark audio_chunks as having speech and link to conversation
            await chunk_repo.mark_conversation_created(audio_uuid, conversation_id)

            audio_logger.info(f"✅ Created conversation {conversation_id} for audio {audio_uuid} (speech detected)")
            return conversation_id

        except Exception as e:
            audio_logger.error(f"Failed to create conversation for {audio_uuid}: {e}", exc_info=True)
            return None

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

            # Step 2: Memory processing is now handled by transcription completion
            # This eliminates race conditions and event coordination issues
            audio_logger.info(f"💭 Memory processing will be triggered by transcription completion for {audio_uuid}")

            # Step 3: Audio cropping is now handled at processor level after transcription
            # This ensures cropping happens with diarization segments when available
            # See transcription.py _queue_diarization_based_cropping() method

            audio_logger.info(f"✅ Successfully closed conversation {audio_uuid}")
            return True

        except Exception as e:
            audio_logger.error(f"❌ Error closing conversation {audio_uuid}: {e}", exc_info=True)
            return False

    async def generate_title(
        self,
        *,
        speaker_segments: Optional[List[Dict]] = None,
        text: Optional[str] = None
    ) -> str:
        """Generate conversation title with speaker-aware formatting when available.

        Args:
            speaker_segments: List of segments with speaker info (preferred)
            text: Raw conversation text (fallback)

        Returns:
            Generated title (max 40 characters)
        """
        # Validation
        if not speaker_segments and not text:
            return "Conversation"

        # Format conversation text (unified approach)
        if speaker_segments:
            conversation_text = self._format_segments_with_speakers(speaker_segments[:10])
            context = "this conversation with speakers"
            include_speakers_instruction = "- Include speaker names when relevant"
        else:
            conversation_text = text[:500] if text else ""
            context = "this conversation transcript"
            include_speakers_instruction = "- Focus on main topic"

        if not conversation_text.strip():
            return "Conversation"

        try:
            # Unified prompt (consistent constraints)
            prompt = f"Generate a concise, descriptive title (max 40 characters) for {context}:"\
                + f"{conversation_text}"\
                + "Rules:\n"\
                + "- Maximum 40 characters\n"\
                + f"{include_speakers_instruction}\n"\
                + "- Capture the main topic\n"\
                + "- Be specific and informative\n"\
                + "Title:"

            title = await async_generate(prompt, temperature=0.3)
            return self._clean_and_truncate_title(title)

        except Exception as e:
            audio_logger.warning(f"Failed to generate LLM title: {e}")
            # Fallback to simple title generation
            words = conversation_text.split()[:6]
            title = " ".join(words)
            return title[:40] + "..." if len(title) > 40 else title or "Conversation"

    async def generate_summary(
        self,
        *,
        speaker_segments: Optional[List[Dict]] = None,
        text: Optional[str] = None
    ) -> str:
        """Generate conversation summary with speaker-aware formatting when available.

        Args:
            speaker_segments: List of segments with speaker info (preferred)
            text: Raw conversation text (fallback)

        Returns:
            Generated summary (max 120 characters)
        """
        # Validation
        if not speaker_segments and not text:
            return "No content"

        # Format conversation text (unified approach)
        if speaker_segments:
            conversation_text = self._format_segments_with_speakers(speaker_segments)
            context = "this conversation with speakers"
            include_speakers_instruction = "- Include speaker names when relevant (e.g., \"John discusses X with Sarah\")"
        else:
            conversation_text = text[:1000] if text else ""
            context = "this conversation transcript"
            include_speakers_instruction = "- Focus on key topics and outcomes"

        if not conversation_text.strip():
            return "No content"

        try:
            # Unified prompt (consistent constraints)
            prompt = f"Generate a brief, informative summary (1-2 sentences, max 120 characters) for {context}:"\
                + f"\n\n\"{conversation_text}\"\n\n"\
                + "Rules:\n"\
                + "- Maximum 120 characters\n"\
                + "- 1-2 complete sentences\n"\
                + f"{include_speakers_instruction}\n"\
                + "- Capture key topics and outcomes\n"\
                + "- Use present tense\n"\
                + "- Be specific and informative\n\n"\
                + "Summary:"

            summary = await async_generate(prompt, temperature=0.3)
            return self._clean_and_truncate_summary(summary)

        except Exception as e:
            audio_logger.warning(f"Failed to generate LLM summary: {e}")
            # Fallback to simple summary generation
            return conversation_text[:120] + "..." if len(conversation_text) > 120 else conversation_text or "No content"

    def _format_segments_with_speakers(self, segments: List[Dict]) -> str:
        """Helper to format segments with speaker names."""
        conversation_text = ""
        for segment in segments:
            speaker = segment.get("speaker", "")
            text = segment.get("text", "").strip()
            if text:
                if speaker:
                    conversation_text += f"{speaker}: {text}\n"
                else:
                    conversation_text += f"{text}\n"
        return conversation_text

    def _clean_and_truncate_title(self, title: str) -> str:
        """Helper to clean and truncate title."""
        title = title.strip().strip('"').strip("'")
        return title[:40] + "..." if len(title) > 40 else title or "Conversation"

    def _clean_and_truncate_summary(self, summary: str) -> str:
        """Helper to clean and truncate summary."""
        summary = summary.strip().strip('"').strip("'")
        return summary[:120] + "..." if len(summary) > 120 else summary or "No content"

    async def create_conversation_with_processing(
        self,
        audio_uuid: str,
        transcript_data: dict,
        speech_analysis: dict,
        speaker_segments: List[Dict],
        chunk_repo
    ) -> Optional[str]:
        """High-level method to create conversation with complete processing.

        This method handles:
        1. Basic conversation creation
        2. Title and summary generation
        3. Transcript version creation and activation
        4. Conversation updates with speaker info

        Args:
            audio_uuid: Audio UUID for the conversation
            transcript_data: Transcript data from transcription provider
            speech_analysis: Speech detection analysis results
            speaker_segments: Processed segments with speaker information
            chunk_repo: AudioChunksRepository instance

        Returns:
            conversation_id if successful, None if failed
        """
        try:
            # Step 1: Create basic conversation
            conversation_id = await self.create_conversation(
                audio_uuid, transcript_data, speech_analysis, chunk_repo
            )
            if not conversation_id:
                audio_logger.error(f"Failed to create basic conversation for {audio_uuid}")
                return None

            # Step 2: Create and activate initial transcript version
            conversations_repo = ConversationsRepository(conversations_col)
            conversation = await conversations_repo.get_conversation(conversation_id)

            if conversation and not conversation.get("active_transcript_version"):
                # Create initial transcript version
                version_id = await conversations_repo.create_transcript_version(
                    conversation_id=conversation_id,
                    segments=speaker_segments,
                    provider="speech_detection",
                    raw_data={}
                )
                if version_id:
                    # Activate this version
                    await conversations_repo.activate_transcript_version(conversation_id, version_id)
                    audio_logger.info(f"✅ Created and activated initial transcript version {version_id} for conversation {conversation_id}")

            # Step 3: Generate title and summary with speaker awareness
            title = await self.generate_title(speaker_segments=speaker_segments)
            summary = await self.generate_summary(speaker_segments=speaker_segments)

            # Step 4: Extract speaker information
            speaker_names = {}
            speakers_found = set()
            for segment in speaker_segments:
                speaker_name = segment.get("identified_as") or segment.get("speaker")
                if speaker_name:
                    speakers_found.add(speaker_name)
                    # Map speaker_id to name if available
                    speaker_id = segment.get("speaker_id", "")
                    if speaker_id:
                        speaker_names[speaker_id] = speaker_name

            # Step 5: Update conversation with final content
            update_data = {
                "title": title,
                "summary": summary,
                "speaker_names": speaker_names,
                "updated_at": datetime.now(UTC)
            }
            await conversations_repo.update_conversation(conversation_id, update_data)

            audio_logger.info(f"✅ Completed conversation processing for {conversation_id} with {len(speaker_segments)} segments, {len(speakers_found)} speakers")
            return conversation_id

        except Exception as e:
            audio_logger.error(f"Failed to create conversation with processing for {audio_uuid}: {e}", exc_info=True)
            return None


# Global singleton instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get the global ConversationManager instance."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
