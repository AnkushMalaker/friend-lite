"""Simplified ClientState that only manages state, no processing.

This module provides a lightweight ClientState class that tracks conversation
state, timestamps, and speech segments. All processing is handled at the
application level by the ProcessorManager.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from advanced_omi_backend.conversation_manager import get_conversation_manager
from advanced_omi_backend.database import AudioChunksRepository
from advanced_omi_backend.task_manager import get_task_manager
from wyoming.audio import AudioChunk

# Get loggers
audio_logger = logging.getLogger("audio_processing")

# Configuration constants
NEW_CONVERSATION_TIMEOUT_MINUTES = float(os.getenv("NEW_CONVERSATION_TIMEOUT_MINUTES", "1.5"))


class ClientState:
    """Manages conversation state for a single client connection."""

    def __init__(
        self,
        client_id: str,
        ac_db_collection_helper: AudioChunksRepository,
        chunk_dir: Path,
        user_id: str,
        user_email: Optional[str] = None,
    ):
        self.client_id = client_id
        self.connected = True
        self.db_helper = ac_db_collection_helper
        self.chunk_dir = chunk_dir

        # Store user data for memory processing
        self.user_id = user_id
        self.user_email = user_email

        # Conversation state tracking
        self.last_transcript_time: Optional[float] = None
        self.conversation_start_time: float = time.time()
        self.current_audio_uuid: Optional[str] = None

        # Speech segment tracking for audio cropping
        self.speech_segments: Dict[str, List[Tuple[float, float]]] = {}
        self.current_speech_start: Dict[str, Optional[float]] = {}

        # NOTE: Removed in-memory transcript storage for single source of truth
        # Transcripts are stored only in MongoDB via TranscriptionManager

        # Track if conversation has been closed
        self.conversation_closed: bool = False

        # Audio configuration - sample rate for this client's audio stream
        self.sample_rate: Optional[int] = None

        # Debug tracking
        self.transaction_id: Optional[str] = None

        audio_logger.info(f"Created client state for {client_id}")

    def update_audio_received(self, chunk: AudioChunk):
        """Update state when audio is received."""
        # Check if we should start a new conversation
        if self.should_start_new_conversation():
            asyncio.create_task(self.start_new_conversation())

    def set_current_audio_uuid(self, audio_uuid: str):
        """Set the current audio UUID when processor creates a new file."""
        self.current_audio_uuid = audio_uuid
        self.conversation_closed = False  # Reset for new audio file

    def record_speech_start(self, audio_uuid: str, timestamp: float):
        """Record the start of a speech segment."""
        self.current_speech_start[audio_uuid] = timestamp
        audio_logger.info(f"Recorded speech start for {audio_uuid}: {timestamp}")

    def record_speech_end(self, audio_uuid: str, timestamp: float):
        """Record the end of a speech segment."""
        if (
            audio_uuid in self.current_speech_start
            and self.current_speech_start[audio_uuid] is not None
        ):
            start_time = self.current_speech_start[audio_uuid]
            if start_time is not None:
                if audio_uuid not in self.speech_segments:
                    self.speech_segments[audio_uuid] = []
                self.speech_segments[audio_uuid].append((start_time, timestamp))
                self.current_speech_start[audio_uuid] = None
                duration = timestamp - start_time
                audio_logger.info(
                    f"Recorded speech segment for {audio_uuid}: {start_time:.3f} -> {timestamp:.3f} "
                    f"(duration: {duration:.3f}s)"
                )
        else:
            audio_logger.warning(f"Speech end recorded for {audio_uuid} but no start time found")

    def update_transcript_received(self):
        """Update timestamp when transcript is received (for timeout detection)."""
        self.last_transcript_time = time.time()

    def should_start_new_conversation(self) -> bool:
        """Check if we should start a new conversation based on timeout."""
        if self.last_transcript_time is None:
            return False

        current_time = time.time()
        time_since_last_transcript = current_time - self.last_transcript_time
        timeout_seconds = NEW_CONVERSATION_TIMEOUT_MINUTES * 60

        return time_since_last_transcript > timeout_seconds

    async def close_current_conversation(self):
        """Close the current conversation and queue necessary processing."""
        # Prevent double closure
        if self.conversation_closed:
            audio_logger.debug(
                f"🔒 Conversation already closed for client {self.client_id}, skipping"
            )
            return

        self.conversation_closed = True

        if not self.current_audio_uuid:
            audio_logger.info(f"🔒 No active conversation to close for client {self.client_id}")
            return

        # Debug logging for memory processing investigation
        audio_logger.info(f"🔍 ClientState close_current_conversation debug for {self.client_id}:")
        audio_logger.info(f"    - current_audio_uuid: {self.current_audio_uuid}")
        audio_logger.info(f"    - user_id: {self.user_id}")
        audio_logger.info(f"    - user_email: {self.user_email}")
        audio_logger.info(f"    - client_id: {self.client_id}")

        # Use ConversationManager for clean separation of concerns
        conversation_manager = get_conversation_manager()
        success = await conversation_manager.close_conversation(
            client_id=self.client_id,
            audio_uuid=self.current_audio_uuid,
            user_id=self.user_id,
            user_email=self.user_email,
            conversation_start_time=self.conversation_start_time,
            speech_segments=self.speech_segments,
            chunk_dir=self.chunk_dir,
        )

        if success:
            # Clean up speech segments for this conversation
            if self.current_audio_uuid in self.speech_segments:
                del self.speech_segments[self.current_audio_uuid]
            if self.current_audio_uuid in self.current_speech_start:
                del self.current_speech_start[self.current_audio_uuid]
        else:
            audio_logger.warning(f"⚠️ Conversation closure had issues for {self.current_audio_uuid}")

    async def start_new_conversation(self):
        """Start a new conversation by closing current and resetting state."""
        await self.close_current_conversation()

        # Reset conversation state
        self.current_audio_uuid = None
        self.conversation_start_time = time.time()
        self.last_transcript_time = None
        self.conversation_closed = False

        audio_logger.info(
            f"Client {self.client_id}: Started new conversation due to "
            f"{NEW_CONVERSATION_TIMEOUT_MINUTES}min timeout"
        )

    async def disconnect(self):
        """Clean disconnect of client state."""
        if not self.connected:
            return

        self.connected = False
        audio_logger.info(f"Disconnecting client {self.client_id}")

        # Close current conversation
        await self.close_current_conversation()

        # Cancel any tasks for this client
        task_manager = get_task_manager()
        await task_manager.cancel_tasks_for_client(self.client_id)

        # Clean up state
        self.speech_segments.clear()
        self.current_speech_start.clear()

        audio_logger.info(f"Client {self.client_id} disconnected and cleaned up")
