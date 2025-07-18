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

from wyoming.audio import AudioChunk

from advanced_omi_backend.database import AudioChunksCollectionHelper
from advanced_omi_backend.processors import (
    AudioCroppingItem,
    MemoryProcessingItem,
    get_processor_manager,
)
from advanced_omi_backend.task_manager import get_task_manager

# Get loggers
audio_logger = logging.getLogger("audio_processing")

# Configuration constants
NEW_CONVERSATION_TIMEOUT_MINUTES = float(os.getenv("NEW_CONVERSATION_TIMEOUT_MINUTES", "1.5"))
AUDIO_CROPPING_ENABLED = os.getenv("AUDIO_CROPPING_ENABLED", "true").lower() == "true"


class ClientState:
    """Manages conversation state for a single client connection."""

    def __init__(
        self,
        client_id: str,
        ac_db_collection_helper: AudioChunksCollectionHelper,
        chunk_dir: Path,
        user_id: Optional[str] = None,
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
        
        # Conversation transcript collection
        self.conversation_transcripts: List[str] = []
        
        # Track if conversation has been closed
        self.conversation_closed: bool = False
        
        # Debug tracking
        self.transaction_id: Optional[str] = None
        
        audio_logger.info(f"Created simplified client state for {client_id}")

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

    def add_transcript(self, transcript_text: str):
        """Add a transcript to the conversation."""
        self.conversation_transcripts.append(transcript_text)
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
        
        audio_logger.info(f"🔒 Closing conversation {self.current_audio_uuid}")
        
        # Get processor manager
        processor_manager = get_processor_manager()
        
        # Close audio file in processor
        await processor_manager.close_client_audio(self.client_id)
        
        # Process memory if we have transcripts
        full_conversation = ""
        transcript_source = ""
        
        if self.conversation_transcripts and self.current_audio_uuid:
            full_conversation = " ".join(self.conversation_transcripts).strip()
            transcript_source = f"memory ({len(self.conversation_transcripts)} segments)"
        elif self.current_audio_uuid and self.db_helper:
            # Fallback: get transcripts from database
            try:
                audio_logger.info(
                    f"💭 Conversation transcripts list empty, checking database for {self.current_audio_uuid}"
                )
                db_transcripts = await self.db_helper.get_transcript_segments(self.current_audio_uuid)
                if db_transcripts:
                    transcript_texts = [segment.get("text", "") for segment in db_transcripts]
                    full_conversation = " ".join(transcript_texts).strip()
                    transcript_source = f"database ({len(db_transcripts)} segments)"
                    audio_logger.info(
                        f"💭 Retrieved {len(db_transcripts)} transcript segments from database"
                    )
            except Exception as e:
                audio_logger.error(
                    f"💭 Error retrieving transcripts from database for {self.current_audio_uuid}: {e}"
                )
        
        # Queue memory processing if we have content
        if full_conversation and self.current_audio_uuid and self.user_id and self.user_email:
            if len(full_conversation) >= 1:  # Process even very short conversations
                audio_logger.info(
                    f"💭 Queuing memory processing for conversation {self.current_audio_uuid} "
                    f"from {transcript_source} (length: {len(full_conversation)} chars)"
                )
                
                await processor_manager.queue_memory(
                    MemoryProcessingItem(
                        client_id=self.client_id,
                        user_id=self.user_id,
                        user_email=self.user_email,
                        audio_uuid=self.current_audio_uuid,
                        full_conversation=full_conversation,
                        db_helper=self.db_helper
                    )
                )
        
        # Queue audio cropping if enabled and we have speech segments
        if AUDIO_CROPPING_ENABLED and self.current_audio_uuid in self.speech_segments:
            speech_segments = self.speech_segments[self.current_audio_uuid]
            if speech_segments:
                # Get the audio file path from processor
                # This assumes the processor follows the same naming convention
                timestamp = int(self.conversation_start_time)
                wav_filename = f"{timestamp}_{self.client_id}_{self.current_audio_uuid}.wav"
                original_path = f"{self.chunk_dir}/{wav_filename}"
                cropped_path = str(original_path).replace(".wav", "_cropped.wav")
                
                audio_logger.info(
                    f"✂️ Queuing audio cropping for {self.current_audio_uuid} "
                    f"with {len(speech_segments)} speech segments"
                )
                
                await processor_manager.queue_cropping(
                    AudioCroppingItem(
                        client_id=self.client_id,
                        user_id=self.user_id,
                        audio_uuid=self.current_audio_uuid,
                        original_path=original_path,
                        speech_segments=speech_segments,
                        output_path=cropped_path
                    )
                )
            
            # Clean up segments for this conversation
            del self.speech_segments[self.current_audio_uuid]
            if self.current_audio_uuid in self.current_speech_start:
                del self.current_speech_start[self.current_audio_uuid]

    async def start_new_conversation(self):
        """Start a new conversation by closing current and resetting state."""
        await self.close_current_conversation()
        
        # Reset conversation state
        self.current_audio_uuid = None
        self.conversation_start_time = time.time()
        self.last_transcript_time = None
        self.conversation_transcripts.clear()
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
        self.conversation_transcripts.clear()
        
        audio_logger.info(f"Client {self.client_id} disconnected and cleaned up")