"""Application-level processors for audio, transcription, memory, and cropping.

This module implements global processing queues and processors that handle
all processing tasks at the application level, decoupled from individual
client connections.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from wyoming.audio import AudioChunk

from advanced_omi_backend.audio_cropping_utils import (
    _process_audio_cropping_with_relative_timestamps,
)
from advanced_omi_backend.users import get_user_by_id
from advanced_omi_backend.client_manager import get_client_manager
from advanced_omi_backend.database import AudioChunksRepository
from advanced_omi_backend.memory import get_memory_service

# Lazy import to avoid config loading issues
# from advanced_omi_backend.memory import get_memory_service
from advanced_omi_backend.task_manager import get_task_manager
from advanced_omi_backend.transcription import TranscriptionManager

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")

# Audio configuration constants
OMI_SAMPLE_RATE = 16_000
OMI_CHANNELS = 1
OMI_SAMPLE_WIDTH = 2
SEGMENT_SECONDS = 60
TARGET_SAMPLES = OMI_SAMPLE_RATE * SEGMENT_SECONDS


@dataclass
class AudioProcessingItem:
    """Item for audio processing queue."""

    client_id: str
    user_id: str
    user_email: str
    audio_chunk: AudioChunk
    audio_uuid: Optional[str] = None
    timestamp: Optional[int] = None


@dataclass
class TranscriptionItem:
    """Item for transcription processing queue."""

    client_id: str
    user_id: str
    audio_uuid: str
    audio_chunk: AudioChunk


@dataclass
class MemoryProcessingItem:
    """Item for memory processing queue."""

    client_id: str
    user_id: str
    user_email: str
    audio_uuid: str


@dataclass
class AudioCroppingItem:
    """Item for audio cropping queue."""

    client_id: str
    user_id: str
    audio_uuid: str
    original_path: str
    speech_segments: list[tuple[float, float]]
    output_path: str


class ProcessorManager:
    """Manages all application-level processors and queues."""

    def __init__(self, chunk_dir: Path, audio_chunks_repository: AudioChunksRepository):
        self.chunk_dir = chunk_dir
        self.repository = audio_chunks_repository

        # Global processing queues
        self.audio_queue: asyncio.Queue[Optional[AudioProcessingItem]] = asyncio.Queue()
        self.transcription_queue: asyncio.Queue[Optional[TranscriptionItem]] = asyncio.Queue()
        self.memory_queue: asyncio.Queue[Optional[MemoryProcessingItem]] = asyncio.Queue()
        self.cropping_queue: asyncio.Queue[Optional[AudioCroppingItem]] = asyncio.Queue()

        # Processor tasks
        self.audio_processor_task: Optional[asyncio.Task] = None
        self.transcription_processor_task: Optional[asyncio.Task] = None
        self.memory_processor_task: Optional[asyncio.Task] = None
        self.cropping_processor_task: Optional[asyncio.Task] = None

        # Services - lazy import
        self.memory_service = None
        self.task_manager = get_task_manager()
        self.client_manager = get_client_manager()

        # Track active file sinks per client
        self.active_file_sinks: dict[str, LocalFileSink] = {}
        self.active_audio_uuids: dict[str, str] = {}

        # Transcription managers pool
        self.transcription_managers: dict[str, TranscriptionManager] = {}

        # Shutdown flag
        self.shutdown_flag = False

        # Task tracking for specific processing jobs
        self.processing_tasks: dict[str, dict[str, str]] = {}  # client_id -> {stage: task_id}

        # Direct state tracking for synchronous operations
        self.processing_state: dict[str, dict[str, Any]] = {}  # client_id -> {stage: state_info}
        
        # Track clients currently being closed to prevent duplicate close operations
        self.closing_clients: set[str] = set()

    async def start(self):
        """Start all processors."""
        logger.info("Starting application-level processors...")

        # Create processor tasks
        self.audio_processor_task = asyncio.create_task(
            self._audio_processor(), name="audio_processor"
        )
        self.transcription_processor_task = asyncio.create_task(
            self._transcription_processor(), name="transcription_processor"
        )
        self.memory_processor_task = asyncio.create_task(
            self._memory_processor(), name="memory_processor"
        )
        self.cropping_processor_task = asyncio.create_task(
            self._cropping_processor(), name="cropping_processor"
        )

        # Track processor tasks in task manager
        self.task_manager.track_task(
            self.audio_processor_task, "audio_processor", {"type": "processor"}
        )
        self.task_manager.track_task(
            self.transcription_processor_task, "transcription_processor", {"type": "processor"}
        )
        self.task_manager.track_task(
            self.memory_processor_task, "memory_processor", {"type": "processor"}
        )
        self.task_manager.track_task(
            self.cropping_processor_task, "cropping_processor", {"type": "processor"}
        )

        logger.info("All processors started successfully")

    async def _should_process_memory(self, user_id: str, audio_uuid: str) -> tuple[bool, str]:
        """
        Determine if memory processing should proceed based on primary speakers configuration.
        
        Implements graceful degradation:
        - No primary speakers configured → Process all (True)
        - Speaker service unavailable → Process all (True) 
        - No speakers identified → Process all (True)
        - Primary speakers found → Process (True)
        - Only non-primary speakers → Skip (False)
        
        Args:
            user_id: User ID to check primary speakers configuration
            audio_uuid: Audio UUID to check transcript speakers
            
        Returns:
            Tuple of (should_process: bool, reason: str)
        """
        try:
            # Get user's primary speaker configuration
            user = await get_user_by_id(user_id)
            if not user or not user.primary_speakers:
                return True, "No primary speakers configured - processing all conversations"
            
            audio_logger.info(f"🔍 Checking primary speakers filter for {audio_uuid} - user has {len(user.primary_speakers)} primary speakers configured")
            
            # Get transcript with speaker identification
            chunk = await self.repository.get_chunk_by_audio_uuid(audio_uuid)
            if not chunk or not chunk.get('transcript'):
                return True, "No transcript data available - processing conversation"
            
            # Extract speakers from transcript segments (normalized for comparison)
            transcript_speakers = set()
            transcript_speaker_originals = {}  # Keep original names for logging
            total_segments = 0
            identified_segments = 0
            
            for segment in chunk['transcript']:
                total_segments += 1
                if 'identified_as' in segment and segment['identified_as'] and segment['identified_as'] != 'Unknown':
                    original_name = segment['identified_as']
                    normalized_name = original_name.strip().lower()
                    transcript_speakers.add(normalized_name)
                    transcript_speaker_originals[normalized_name] = original_name
                    identified_segments += 1
            
            if not transcript_speakers:
                return True, f"No speakers identified in transcript ({identified_segments}/{total_segments} segments) - processing conversation"
            
            # Check if any primary speakers are present (normalized comparison)
            primary_speaker_names = {ps['name'].strip().lower() for ps in user.primary_speakers}
            primary_speaker_originals = {ps['name'].strip().lower(): ps['name'] for ps in user.primary_speakers}
            found_primary_speakers_normalized = transcript_speakers.intersection(primary_speaker_names)
            
            if found_primary_speakers_normalized:
                # Convert back to original names for display
                found_primary_originals = [primary_speaker_originals[name] for name in found_primary_speakers_normalized]
                audio_logger.info(f"✅ Primary speakers found in conversation: {found_primary_originals} - processing memory")
                return True, f"Primary speakers detected: {', '.join(found_primary_originals)}"
            else:
                # Show original names in logs
                transcript_originals = [transcript_speaker_originals[name] for name in transcript_speakers]
                primary_originals = [primary_speaker_originals[name] for name in primary_speaker_names]
                audio_logger.info(f"❌ No primary speakers found - transcript speakers: {transcript_originals}, primary speakers: {primary_originals} - skipping memory processing")
                return False, f"Only non-primary speakers found: {', '.join(transcript_originals)}"
                
        except Exception as e:
            # On any error, default to processing (fail-safe)
            audio_logger.warning(f"Error checking primary speakers filter for {audio_uuid}: {e} - defaulting to process conversation")
            return True, f"Error in speaker filtering: {str(e)} - processing conversation as fallback"

    async def shutdown(self):
        """Shutdown all processors gracefully."""
        logger.info("Shutting down processors...")
        self.shutdown_flag = True

        # Signal all queues to stop
        await self.audio_queue.put(None)
        await self.transcription_queue.put(None)
        await self.memory_queue.put(None)
        await self.cropping_queue.put(None)

        # Wait for processors to complete with timeout
        tasks = [
            ("audio_processor", self.audio_processor_task, 30.0),
            ("transcription_processor", self.transcription_processor_task, 60.0),
            ("memory_processor", self.memory_processor_task, 300.0),  # 5 minutes for LLM
            ("cropping_processor", self.cropping_processor_task, 60.0),
        ]

        for name, task, timeout in tasks:
            if task:
                try:
                    await asyncio.wait_for(task, timeout=timeout)
                    logger.info(f"{name} shut down gracefully")
                except asyncio.TimeoutError:
                    logger.warning(f"{name} did not shut down within {timeout}s, cancelling")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        logger.info(f"{name} cancelled successfully")

        # Clean up transcription managers
        for manager in self.transcription_managers.values():
            try:
                await manager.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting transcription manager: {e}")

        # Close any remaining file sinks
        for sink in self.active_file_sinks.values():
            try:
                await sink.close()
            except Exception as e:
                logger.error(f"Error closing file sink: {e}")

        logger.info("All processors shut down")

    def _new_local_file_sink(
        self, file_path: str, sample_rate: Optional[int] = None
    ) -> LocalFileSink:
        """Create a properly configured LocalFileSink with dynamic sample rate."""
        effective_sample_rate = sample_rate or OMI_SAMPLE_RATE
        return LocalFileSink(
            file_path=file_path,
            sample_rate=int(effective_sample_rate),
            channels=int(OMI_CHANNELS),
            sample_width=int(OMI_SAMPLE_WIDTH),
        )

    async def queue_audio(self, item: AudioProcessingItem):
        """Queue audio for processing."""
        audio_logger.info(
            f"📥 queue_audio called for client {item.client_id}, audio chunk: {len(item.audio_chunk.audio)} bytes"
        )
        await self.audio_queue.put(item)
        queue_size = self.audio_queue.qsize()
        audio_logger.info(
            f"✅ Successfully queued audio for client {item.client_id}, queue size: {queue_size}"
        )

    async def queue_transcription(self, item: TranscriptionItem):
        """Queue audio for transcription."""
        audio_logger.debug(
            f"📥 queue_transcription called for client {item.client_id}, audio_uuid: {item.audio_uuid}"
        )
        await self.transcription_queue.put(item)
        audio_logger.debug(
            f"📤 Successfully put item in transcription_queue for client {item.client_id}, queue size: {self.transcription_queue.qsize()}"
        )

    async def queue_memory(self, item: MemoryProcessingItem):
        """Queue conversation for memory processing."""
        audio_logger.info(
            f"📥 queue_memory called for client {item.client_id}, audio_uuid: {item.audio_uuid}"
        )
        audio_logger.info(f"📥 Memory queue size before: {self.memory_queue.qsize()}")
        await self.memory_queue.put(item)
        audio_logger.info(f"📥 Memory queue size after: {self.memory_queue.qsize()}")
        audio_logger.info(f"✅ Successfully queued memory processing item for {item.audio_uuid}")

    async def queue_cropping(self, item: AudioCroppingItem):
        """Queue audio for cropping."""
        await self.cropping_queue.put(item)

    def track_processing_task(
        self, client_id: str, stage: str, task_id: str, metadata: dict[str, Any] | None = None
    ):
        """Track a processing task for a specific client and stage."""
        if client_id not in self.processing_tasks:
            self.processing_tasks[client_id] = {}
        self.processing_tasks[client_id][stage] = task_id
        logger.info(f"Tracking task {task_id} for client {client_id} stage {stage}")

    def track_processing_stage(
        self, client_id: str, stage: str, status: str, metadata: dict[str, Any] | None = None
    ):
        """Track processing stage completion directly for synchronous operations."""
        if client_id not in self.processing_state:
            self.processing_state[client_id] = {}

        self.processing_state[client_id][stage] = {
            "status": status,  # "started", "completed", "failed"
            "completed": status == "completed",
            "error": None if status != "failed" else metadata.get("error") if metadata else None,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        logger.info(f"Tracking stage {stage} as {status} for client {client_id}")

    def get_processing_status(self, client_id: str) -> dict[str, Any]:
        """Get processing status for a specific client using both direct state and task tracking."""
        logger.debug(f"Getting processing status for client {client_id}")
        logger.debug(
            f"Available client_ids in processing_tasks: {list(self.processing_tasks.keys())}"
        )
        logger.debug(
            f"Available client_ids in processing_state: {list(self.processing_state.keys())}"
        )

        stages = {}

        # First, get task tracking (for asynchronous operations like memory/cropping)
        if client_id in self.processing_tasks:
            client_tasks = self.processing_tasks[client_id]
            for stage, task_id in client_tasks.items():
                logger.info(f"Looking up task {task_id} for stage {stage}")
                task_info = self.task_manager.get_task_info(task_id)
                logger.info(f"Task info for {task_id}: {task_info}")
                if task_info:
                    stages[stage] = {
                        "task_id": task_id,
                        "completed": task_info.completed_at is not None,
                        "error": task_info.error,
                        "created_at": task_info.created_at,
                        "completed_at": task_info.completed_at,
                        "cancelled": task_info.cancelled,
                    }
                else:
                    stages[stage] = {
                        "task_id": task_id,
                        "completed": False,
                        "error": "Task not found",
                        "created_at": None,
                        "completed_at": None,
                        "cancelled": False,
                    }

        # Then, get direct state tracking (for synchronous operations like audio, transcription)
        # Direct state takes PRECEDENCE over task tracking for the same stage
        if client_id in self.processing_state:
            client_state = self.processing_state[client_id]
            for stage, state_info in client_state.items():
                stages[stage] = {
                    "completed": state_info["completed"],
                    "error": state_info["error"],
                    "status": state_info["status"],
                    "metadata": state_info["metadata"],
                    "timestamp": state_info["timestamp"],
                }
                logger.debug(f"Direct state - {stage}: {state_info['status']} (takes precedence)")

        # If no stages found, return no_tasks
        if not stages:
            return {"status": "no_tasks", "stages": {}}

        # Check if all stages are complete
        all_complete = all(stage_info["completed"] for stage_info in stages.values())

        return {
            "status": "complete" if all_complete else "processing",
            "stages": stages,
            "client_id": client_id,
        }

    def cleanup_processing_tasks(self, client_id: str):
        """Clean up processing task tracking for a client."""
        if client_id in self.processing_tasks:
            del self.processing_tasks[client_id]
            logger.debug(f"Cleaned up processing tasks for client {client_id}")

        if client_id in self.processing_state:
            del self.processing_state[client_id]
            logger.debug(f"Cleaned up processing state for client {client_id}")

    def get_all_processing_status(self) -> dict[str, Any]:
        """Get processing status for all clients."""
        # Get all client IDs from both tracking types
        all_client_ids = set(self.processing_tasks.keys()) | set(self.processing_state.keys())
        return {client_id: self.get_processing_status(client_id) for client_id in all_client_ids}

    async def mark_transcription_failed(self, client_id: str, error: str):
        """Mark transcription as failed and clean up transcription manager.

        This method handles transcription failures without closing audio files,
        allowing long recordings to continue even if intermediate transcriptions fail.

        Args:
            client_id: The client ID whose transcription failed
            error: The error message describing the failure
        """
        # Mark as failed in state tracking
        self.track_processing_stage(client_id, "transcription", "failed", {"error": error})

        # Remove transcription manager to allow fresh retry
        if client_id in self.transcription_managers:
            try:
                manager = self.transcription_managers.pop(client_id)
                await manager.disconnect()
                audio_logger.info(f"🧹 Removed failed transcription manager for {client_id}")
            except Exception as cleanup_error:
                audio_logger.error(
                    f"❌ Error cleaning up transcription manager for {client_id}: {cleanup_error}"
                )

        # Do NOT close audio files - client may still be streaming
        # Audio will be closed when client disconnects or sends audio-stop
        audio_logger.warning(
            f"❌ Transcription failed for {client_id}: {error}, keeping audio session open"
        )

    async def close_client_audio(self, client_id: str):
        """Close audio file for a client when conversation ends."""
        audio_logger.info(f"🔚 close_client_audio called for client {client_id}")
        
        # Check if already closing to prevent duplicate operations
        if client_id in self.closing_clients:
            audio_logger.info(f"⏭️ Client {client_id} already being closed, skipping duplicate close")
            return
            
        # Mark as being closed
        self.closing_clients.add(client_id)

        # First, flush ASR to complete any pending transcription
        if client_id in self.transcription_managers:
            try:
                manager = self.transcription_managers[client_id]
                audio_logger.info(
                    f"🔄 Found transcription manager - flushing ASR for client {client_id}"
                )
                audio_logger.info(
                    f"📊 Transcription manager state - has manager: {manager is not None}, type: {type(manager).__name__}"
                )

                # Get audio duration for flush timeout calculation
                audio_duration = None
                if client_id in self.active_audio_uuids:
                    audio_uuid = self.active_audio_uuids[client_id]
                    audio_logger.info(f"📌 Active audio UUID for flush: {audio_uuid}")
                    # Try to estimate duration from file sink if available
                    if client_id in self.active_file_sinks:
                        try:
                            sink = self.active_file_sinks[client_id]
                            # Estimate duration based on samples written (if accessible)
                            # For now, use None and let flush_final_transcript handle timeout
                            audio_logger.info(f"📁 File sink exists for client {client_id}")
                        except Exception as e:
                            audio_logger.warning(f"⚠️ Error accessing file sink: {e}")

                flush_start_time = time.time()
                audio_logger.info(
                    f"📤 Calling flush_final_transcript for client {client_id} (manager: {manager})"
                )
                try:
                    await manager.process_collected_audio(audio_duration)
                    flush_duration = time.time() - flush_start_time
                    audio_logger.info(
                        f"✅ ASR flush completed for client {client_id} in {flush_duration:.2f}s"
                    )
                    # Mark transcription as completed after successful flush
                    self.track_processing_stage(
                        client_id, "transcription", "completed", {"flushed": True}
                    )
                except Exception as flush_error:
                    audio_logger.error(
                        f"❌ Error during flush_final_transcript: {flush_error}", exc_info=True
                    )
                    # Mark transcription as failed on flush error
                    self.track_processing_stage(
                        client_id, "transcription", "failed", {"error": str(flush_error)}
                    )
                    raise

                # Verify that transcription was marked as completed after flush
                current_status = self.get_processing_status(client_id)
                transcription_stage = current_status.get("stages", {}).get("transcription", {})
                audio_logger.info(
                    f"🔍 Post-flush transcription status: {transcription_stage.get('status', 'unknown')} (completed: {transcription_stage.get('completed', False)})"
                )
            except Exception as e:
                audio_logger.error(
                    f"❌ Error flushing ASR for client {client_id}: {e}", exc_info=True
                )
        else:
            audio_logger.warning(
                f"⚠️ No transcription manager found for client {client_id} - cannot flush transcription"
            )

        # Then close the audio file
        if client_id in self.active_file_sinks:
            try:
                sink = self.active_file_sinks[client_id]
                await sink.close()
                del self.active_file_sinks[client_id]

                if client_id in self.active_audio_uuids:
                    del self.active_audio_uuids[client_id]

                audio_logger.info(f"Closed audio file for client {client_id}")
            except Exception as e:
                audio_logger.error(f"Error closing audio file for client {client_id}: {e}")
        
        # Remove from closing set now that we're done
        self.closing_clients.discard(client_id)
        audio_logger.info(f"✅ Completed close_client_audio for client {client_id}")

    async def ensure_transcription_manager(self, client_id: str):
        """Ensure a transcription manager exists for the given client.
        
        This can be called early (e.g., on audio-start) to create the manager
        before audio chunks arrive.
        """
        if client_id not in self.transcription_managers:
            audio_logger.info(
                f"🔌 Creating transcription manager for client {client_id} (early creation)"
            )
            manager = TranscriptionManager(
                chunk_repo=self.repository, processor_manager=self
            )
            try:
                await manager.connect(client_id)
                self.transcription_managers[client_id] = manager
                audio_logger.info(
                    f"✅ Successfully created transcription manager for {client_id}"
                )
            except Exception as e:
                audio_logger.error(
                    f"❌ Failed to create transcription manager for {client_id}: {e}"
                )
                raise
        else:
            audio_logger.debug(
                f"♻️ Transcription manager already exists for client {client_id}"
            )

    async def _audio_processor(self):
        """Process audio chunks and save to files."""
        audio_logger.info("Audio processor started")

        try:
            while not self.shutdown_flag:
                try:
                    # Get item with timeout to allow periodic health checks
                    queue_size = self.audio_queue.qsize()
                    if queue_size > 0:
                        audio_logger.info(
                            f"🔄 Audio processor waiting for items, queue size: {queue_size}"
                        )
                    item = await asyncio.wait_for(self.audio_queue.get(), timeout=30.0)
                    
                    audio_logger.info(
                        f"📦 Audio processor dequeued item for client {item.client_id if item else 'None'}"
                    )

                    if item is None:  # Shutdown signal
                        audio_logger.info("🛑 Audio processor received shutdown signal")
                        self.audio_queue.task_done()
                        break

                    try:
                        # Get or create file sink for this client
                        if item.client_id not in self.active_file_sinks:
                            audio_logger.info(
                                f"🆕 Creating new audio file sink for client {item.client_id}"
                            )
                            # Get client state to access/store sample rate
                            client_state = self.client_manager.get_client(item.client_id)
                            audio_logger.info(
                                f"👤 Client state lookup for {item.client_id}: {client_state is not None}"
                            )

                            # Store sample rate from first audio chunk
                            if client_state and client_state.sample_rate is None:
                                client_state.sample_rate = item.audio_chunk.rate
                                audio_logger.info(
                                    f"📊 Set sample rate to {client_state.sample_rate}Hz for client {item.client_id}"
                                )

                            # Get sample rate for file sink (use client state or fallback to chunk rate)
                            file_sample_rate = None
                            if client_state and client_state.sample_rate:
                                file_sample_rate = client_state.sample_rate
                            else:
                                file_sample_rate = item.audio_chunk.rate
                                audio_logger.warning(
                                    f"Using chunk sample rate {file_sample_rate}Hz for {item.client_id} (no client state)"
                                )

                            # Create new file
                            audio_uuid = uuid.uuid4().hex
                            timestamp = item.timestamp or int(time.time())
                            wav_filename = f"{timestamp}_{item.client_id}_{audio_uuid}.wav"

                            sink = self._new_local_file_sink(
                                f"{self.chunk_dir}/{wav_filename}", file_sample_rate
                            )
                            await sink.open()

                            self.active_file_sinks[item.client_id] = sink
                            self.active_audio_uuids[item.client_id] = audio_uuid

                            # Create database entry
                            await self.repository.create_chunk(
                                audio_uuid=audio_uuid,
                                audio_path=wav_filename,
                                client_id=item.client_id,
                                timestamp=timestamp,
                                user_id=item.user_id,
                                user_email=item.user_email,
                            )

                            # Notify client state about new audio UUID
                            if client_state:
                                client_state.set_current_audio_uuid(audio_uuid)

                            # Track audio processing completion directly (synchronous operation)
                            self.track_processing_stage(
                                item.client_id,
                                "audio",
                                "completed",
                                {
                                    "audio_uuid": audio_uuid,
                                    "wav_filename": wav_filename,
                                    "file_created": True,
                                },
                            )

                            audio_logger.info(
                                f"Created new audio file for client {item.client_id}: {wav_filename}"
                            )

                        # Write audio chunk
                        sink = self.active_file_sinks[item.client_id]
                        await sink.write(item.audio_chunk)

                        # Queue for transcription
                        audio_uuid = self.active_audio_uuids[item.client_id]
                        audio_logger.debug(
                            f"🔄 About to queue transcription for client {item.client_id}, audio_uuid: {audio_uuid}"
                        )
                        await self.queue_transcription(
                            TranscriptionItem(
                                client_id=item.client_id,
                                user_id=item.user_id,
                                audio_uuid=audio_uuid,
                                audio_chunk=item.audio_chunk,
                            )
                        )
                        audio_logger.debug(
                            f"✅ Successfully queued transcription for client {item.client_id}, audio_uuid: {audio_uuid}"
                        )

                    except Exception as e:
                        audio_logger.error(
                            f"Error processing audio for client {item.client_id}: {e}",
                            exc_info=True,
                        )
                    finally:
                        self.audio_queue.task_done()
                        audio_logger.info(
                            f"✅ Completed processing audio item for client {item.client_id if item else 'None'}"
                        )

                except asyncio.TimeoutError:
                    # Periodic health check
                    active_clients = len(self.active_file_sinks)
                    queue_size = self.audio_queue.qsize()
                    if queue_size > 0 or active_clients > 0:
                        audio_logger.info(
                            f"⏰ Audio processor timeout (periodic health check): {active_clients} active files, "
                            f"{queue_size} items in queue"
                        )

        except Exception as e:
            audio_logger.error(f"Fatal error in audio processor: {e}", exc_info=True)
        finally:
            audio_logger.info("Audio processor stopped")

    async def _transcription_processor(self):
        """Process transcription requests."""
        audio_logger.info("Transcription processor started")

        try:
            while not self.shutdown_flag:
                try:
                    item = await asyncio.wait_for(self.transcription_queue.get(), timeout=30.0)

                    if item is None:  # Shutdown signal
                        self.transcription_queue.task_done()
                        break

                    try:
                        # Get or create transcription manager for client
                        if item.client_id not in self.transcription_managers:
                            audio_logger.info(
                                f"🔌 Creating new transcription manager for client {item.client_id}"
                            )
                            manager = TranscriptionManager(
                                chunk_repo=self.repository, processor_manager=self
                            )
                            try:
                                await manager.connect(item.client_id)
                                self.transcription_managers[item.client_id] = manager
                                audio_logger.info(
                                    f"✅ Successfully created transcription manager for {item.client_id}"
                                )
                            except Exception as e:
                                audio_logger.error(
                                    f"❌ Failed to create transcription manager for {item.client_id}: {e}"
                                )
                                # Mark transcription as failed when manager creation fails
                                self.track_processing_stage(
                                    item.client_id, "transcription", "failed", {"error": str(e)}
                                )
                                self.transcription_queue.task_done()
                                continue
                        else:
                            audio_logger.debug(
                                f"♻️ Reusing existing transcription manager for client {item.client_id}"
                            )

                        manager = self.transcription_managers[item.client_id]

                        # Process transcription chunk
                        audio_logger.debug(
                            f"🎵 Processing transcribe_chunk for client {item.client_id}, audio_uuid: {item.audio_uuid}"
                        )

                        try:
                            # Add timeout for transcription processing (5 minutes)
                            async with asyncio.timeout(300):  # 5 minute timeout
                                await manager.transcribe_chunk(
                                    item.audio_uuid, item.audio_chunk, item.client_id
                                )
                            audio_logger.debug(
                                f"✅ Completed transcribe_chunk for client {item.client_id}"
                            )
                        except asyncio.TimeoutError:
                            audio_logger.error(
                                f"❌ Transcription timeout for client {item.client_id} after 5 minutes"
                            )
                            # Mark transcription as failed on timeout
                            self.track_processing_stage(
                                item.client_id,
                                "transcription",
                                "failed",
                                {"error": "Transcription timeout (5 minutes)"},
                            )
                        except Exception as e:
                            audio_logger.error(
                                f"❌ Error in transcribe_chunk for client {item.client_id}: {e}",
                                exc_info=True,
                            )
                            # Mark transcription as failed when chunk processing fails
                            self.track_processing_stage(
                                item.client_id, "transcription", "failed", {"error": str(e)}
                            )

                        # Track transcription as started using direct state tracking - ONLY ONCE per audio session
                        # Check if we haven't already marked this transcription as started for this audio UUID
                        current_transcription_status = self.processing_state.get(
                            item.client_id, {}
                        ).get("transcription", {})
                        current_audio_uuid = current_transcription_status.get("metadata", {}).get(
                            "audio_uuid"
                        )

                        # Only mark as started if this is a new audio UUID or no transcription status exists
                        if current_audio_uuid != item.audio_uuid:
                            audio_logger.info(
                                f"🎯 Starting transcription tracking for new audio UUID: {item.audio_uuid}"
                            )
                            self.track_processing_stage(
                                item.client_id,
                                "transcription",
                                "started",
                                {"audio_uuid": item.audio_uuid, "chunk_processing": True},
                            )
                        else:
                            audio_logger.debug(
                                f"⏩ Skipping transcription status update - already tracking audio UUID: {item.audio_uuid}"
                            )

                    except Exception as e:
                        audio_logger.error(
                            f"Error processing transcription for client {item.client_id}: {e}",
                            exc_info=True,
                        )
                    finally:
                        self.transcription_queue.task_done()

                except asyncio.TimeoutError:
                    # Periodic health check only (NO cleanup based on client active status)
                    queue_size = self.transcription_queue.qsize()
                    active_managers = len(self.transcription_managers)
                    audio_logger.debug(
                        f"Transcription processor health: {active_managers} managers, "
                        f"{queue_size} items in queue"
                    )

        except Exception as e:
            audio_logger.error(f"Fatal error in transcription processor: {e}", exc_info=True)
        finally:
            audio_logger.info("Transcription processor stopped")

    async def _memory_processor(self):
        """Process memory/LLM requests."""
        audio_logger.info("Memory processor started")

        try:
            while not self.shutdown_flag:
                try:
                    item = await asyncio.wait_for(self.memory_queue.get(), timeout=30.0)

                    if item is None:  # Shutdown signal
                        self.memory_queue.task_done()
                        break

                    try:
                        # Create background task for memory processing
                        task = asyncio.create_task(self._process_memory_item(item))

                        # Track task with 5 minute timeout
                        task_name = f"memory_{item.client_id}_{item.audio_uuid}"
                        actual_task_id = self.task_manager.track_task(
                            task,
                            task_name,
                            {
                                "client_id": item.client_id,
                                "audio_uuid": item.audio_uuid,
                                "type": "memory",
                                "timeout": 3600,  # 60 minutes
                            },
                        )

                        # Register task with client for tracking (use the actual task_id from TaskManager)
                        self.track_processing_task(
                            item.client_id,
                            "memory",
                            actual_task_id,
                            {"audio_uuid": item.audio_uuid},
                        )

                    except Exception as e:
                        audio_logger.error(
                            f"Error queuing memory processing for {item.audio_uuid}: {e}",
                            exc_info=True,
                        )
                    finally:
                        self.memory_queue.task_done()

                except asyncio.TimeoutError:
                    # Periodic health check
                    queue_size = self.memory_queue.qsize()
                    audio_logger.debug(f"Memory processor health: {queue_size} items in queue")

        except Exception as e:
            audio_logger.error(f"Fatal error in memory processor: {e}", exc_info=True)
        finally:
            audio_logger.info("Memory processor stopped")

    async def _process_memory_item(self, item: MemoryProcessingItem):
        """Process a single memory item."""
        start_time = time.time()
        audio_logger.info(f"🚀 MEMORY PROCESSING STARTED for {item.audio_uuid} at {start_time}")

        # Track memory processing start
        self.track_processing_stage(
            item.client_id,
            "memory",
            "started",
            {"audio_uuid": item.audio_uuid, "started_at": start_time},
        )

        try:
            # Use ConversationRepository for clean data access with event coordination
            from advanced_omi_backend.conversation_repository import (
                get_conversation_repository,
            )

            conversation_repo = get_conversation_repository()

            # Memory processing is now data-driven - transcript should be available
            # since this is queued AFTER transcription completion
            full_conversation = await conversation_repo.get_full_conversation_text(item.audio_uuid)

            if not full_conversation:
                audio_logger.warning(
                    f"No valid conversation text found for {item.audio_uuid}, skipping memory processing"
                )
                return None
            if len(full_conversation) < 10:  # Minimum length check
                audio_logger.warning(
                    f"Conversation too short for memory processing ({len(full_conversation)} chars): {item.audio_uuid}"
                )
                return None

            # Debug tracking removed for cleaner architecture

            # Check if memory processing should proceed based on primary speakers configuration
            should_process, filter_reason = await self._should_process_memory(item.user_id, item.audio_uuid)
            audio_logger.info(f"🎯 Speaker filter decision for {item.audio_uuid}: {filter_reason}")
            
            if not should_process:
                # Update memory processing status to skipped
                await self.repository.update_memory_processing_status(
                    item.audio_uuid, "SKIPPED", error_message=filter_reason
                )
                
                # Track completion
                self.track_processing_stage(
                    item.client_id,
                    "memory",
                    "completed",
                    {
                        "audio_uuid": item.audio_uuid,
                        "status": "skipped",
                        "reason": filter_reason,
                        "completed_at": time.time(),
                    },
                )
                audio_logger.info(f"⏭️ Skipped memory processing for {item.audio_uuid}: {filter_reason}")
                return None

            # Lazy import memory service
            if self.memory_service is None:
                audio_logger.info(f"🔧 Initializing memory service for {item.audio_uuid}...")
                self.memory_service = get_memory_service()
                audio_logger.info(f"✅ Memory service initialized for {item.audio_uuid}")

            # Process memory with timeout
            audio_logger.info(f"🔥 About to call add_memory() for {item.audio_uuid}...")
            memory_result = await asyncio.wait_for(
                self.memory_service.add_memory(
                    full_conversation,
                    item.client_id,
                    item.audio_uuid,
                    item.user_id,
                    item.user_email,
                    allow_update=True,
                    db_helper=None,  # Using ConversationRepository now
                ),
                timeout=3600,  # 60 minutes
            )

            if memory_result:
                # Check if this was a successful result with actual memories created
                success, created_memory_ids = memory_result
                logger.info(f"Memory result: {memory_result}")

                if success and created_memory_ids:
                    # Memories were actually created
                    audio_logger.info(
                        f"✅ Successfully processed memory for {item.audio_uuid} - created {len(created_memory_ids)} memories"
                    )

                    # Add memory references to MongoDB conversation document
                    try:
                        for memory_id in created_memory_ids:
                            await conversation_repo.add_memory_reference(
                                item.audio_uuid, memory_id, "created"
                            )
                        audio_logger.info(
                            f"📝 Added {len(created_memory_ids)} memory references to MongoDB for {item.audio_uuid}"
                        )
                    except Exception as e:
                        audio_logger.warning(f"Failed to add memory references to MongoDB: {e}")

                    # Update database memory processing status to completed
                    try:
                        await conversation_repo.update_memory_processing_status(
                            item.audio_uuid, "COMPLETED"
                        )
                        audio_logger.info(
                            f"📝 Updated memory processing status to COMPLETED for {item.audio_uuid}"
                        )
                    except Exception as e:
                        audio_logger.warning(f"Failed to update memory status: {e}")

                    # Track memory processing completion
                    self.track_processing_stage(
                        item.client_id,
                        "memory",
                        "completed",
                        {
                            "audio_uuid": item.audio_uuid,
                            "memories_created": len(created_memory_ids),
                            "processing_time": time.time() - start_time,
                        },
                    )
                elif success and not created_memory_ids:
                    # Successful processing but no memories created (likely empty transcript)
                    audio_logger.info(
                        f"✅ Memory processing completed for {item.audio_uuid} but no memories created (likely empty transcript)"
                    )

                    # Update database memory processing status to skipped
                    try:
                        await conversation_repo.update_memory_processing_status(
                            item.audio_uuid, "SKIPPED"
                        )
                        audio_logger.info(
                            f"📝 Updated memory processing status to SKIPPED for {item.audio_uuid} (no memories created - empty transcript)"
                        )
                    except Exception as e:
                        audio_logger.warning(f"Failed to update memory status: {e}")

                    # Track memory processing completion (even though no memories created)
                    self.track_processing_stage(
                        item.client_id,
                        "memory",
                        "completed",
                        {
                            "audio_uuid": item.audio_uuid,
                            "memories_created": 0,
                            "processing_time": time.time() - start_time,
                            "status": "skipped",
                        },
                    )
                else:
                    # This shouldn't happen, but handle it gracefully
                    audio_logger.warning(
                        f"⚠️ Unexpected memory result for {item.audio_uuid}: success={success}, ids={created_memory_ids}"
                    )

                    # Update database memory processing status to failed
                    try:
                        await conversation_repo.update_memory_processing_status(
                            item.audio_uuid, "FAILED"
                        )
                        audio_logger.warning(
                            f"📝 Updated memory processing status to FAILED for {item.audio_uuid} (unexpected result)"
                        )
                    except Exception as e:
                        audio_logger.warning(f"Failed to update memory status: {e}")
                        audio_logger.warning(
                            f"📝 Updated memory processing status to FAILED for {item.audio_uuid}"
                        )

                    # Track memory processing failure
                    self.track_processing_stage(
                        item.client_id,
                        "memory",
                        "failed",
                        {
                            "audio_uuid": item.audio_uuid,
                            "error": f"Unexpected result: success={success}, ids={created_memory_ids}",
                            "processing_time": time.time() - start_time,
                        },
                    )

            else:
                audio_logger.warning(f"⚠️ Memory service returned False for {item.audio_uuid}")

                # Update database memory processing status to failed
                try:
                    await conversation_repo.update_memory_processing_status(
                        item.audio_uuid, "FAILED"
                    )
                    audio_logger.warning(
                        f"📝 Updated memory processing status to FAILED for {item.audio_uuid} (memory service returned False)"
                    )
                except Exception as e:
                    audio_logger.warning(f"Failed to update memory status: {e}")
                    audio_logger.warning(
                        f"📝 Updated memory processing status to FAILED for {item.audio_uuid}"
                    )

                # Track memory processing failure
                self.track_processing_stage(
                    item.client_id,
                    "memory",
                    "failed",
                    {
                        "audio_uuid": item.audio_uuid,
                        "error": "Memory service returned False",
                        "processing_time": time.time() - start_time,
                    },
                )

        except asyncio.TimeoutError:
            audio_logger.error(f"Memory processing timed out for {item.audio_uuid}")

            # Update database memory processing status to failed
            try:
                await conversation_repo.update_memory_processing_status(item.audio_uuid, "FAILED")
                audio_logger.error(
                    f"📝 Updated memory processing status to FAILED for {item.audio_uuid} (timeout: 5 minutes)"
                )
            except Exception as e:
                audio_logger.warning(f"Failed to update memory status: {e}")

            # Track memory processing timeout failure
            self.track_processing_stage(
                item.client_id,
                "memory",
                "failed",
                {
                    "audio_uuid": item.audio_uuid,
                    "error": "Processing timeout (5 minutes)",
                    "processing_time": time.time() - start_time,
                },
            )

        except Exception as e:
            audio_logger.error(f"Error processing memory for {item.audio_uuid}: {e}")

            # Update database memory processing status to failed
            try:
                await conversation_repo.update_memory_processing_status(item.audio_uuid, "FAILED")
                audio_logger.error(
                    f"📝 Updated memory processing status to FAILED for {item.audio_uuid} (exception: {str(e)})"
                )
            except Exception as repo_e:
                audio_logger.warning(f"Failed to update memory status: {repo_e}")

            # Track memory processing exception failure
            self.track_processing_stage(
                item.client_id,
                "memory",
                "failed",
                {
                    "audio_uuid": item.audio_uuid,
                    "error": f"Exception: {str(e)}",
                    "processing_time": time.time() - start_time,
                },
            )

        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        audio_logger.info(
            f"🏁 MEMORY PROCESSING COMPLETED for {item.audio_uuid} in {processing_time_ms:.1f}ms (end time: {end_time})"
        )

    async def _cropping_processor(self):
        """Process audio cropping requests."""
        audio_logger.info("Audio cropping processor started")

        try:
            while not self.shutdown_flag:
                try:
                    item = await asyncio.wait_for(self.cropping_queue.get(), timeout=30.0)

                    if item is None:  # Shutdown signal
                        self.cropping_queue.task_done()
                        break

                    try:
                        # Create background task for cropping
                        task = asyncio.create_task(
                            _process_audio_cropping_with_relative_timestamps(
                                item.original_path,
                                item.speech_segments,
                                item.output_path,
                                item.audio_uuid,
                            )
                        )

                        # Track task
                        task_name = f"cropping_{item.client_id}_{item.audio_uuid}"
                        actual_task_id = self.task_manager.track_task(
                            task,
                            task_name,
                            {
                                "client_id": item.client_id,
                                "audio_uuid": item.audio_uuid,
                                "type": "cropping",
                                "segments": len(item.speech_segments),
                            },
                        )

                        # Register task with client for tracking (use the actual task_id from TaskManager)
                        self.track_processing_task(
                            item.client_id,
                            "cropping",
                            actual_task_id,
                            {"audio_uuid": item.audio_uuid, "segments": len(item.speech_segments)},
                        )

                        audio_logger.info(
                            f"✂️ Queued audio cropping for {item.audio_uuid} "
                            f"with {len(item.speech_segments)} segments"
                        )

                    except Exception as e:
                        audio_logger.error(
                            f"Error queuing audio cropping for {item.audio_uuid}: {e}",
                            exc_info=True,
                        )
                    finally:
                        self.cropping_queue.task_done()

                except asyncio.TimeoutError:
                    # Periodic health check
                    queue_size = self.cropping_queue.qsize()
                    audio_logger.debug(f"Cropping processor health: {queue_size} items in queue")

        except Exception as e:
            audio_logger.error(f"Fatal error in cropping processor: {e}", exc_info=True)
        finally:
            audio_logger.info("Audio cropping processor stopped")


# Global processor manager instance
_processor_manager: Optional[ProcessorManager] = None


def init_processor_manager(chunk_dir: Path, db_helper: AudioChunksRepository):
    """Initialize the global processor manager."""
    global _processor_manager
    _processor_manager = ProcessorManager(chunk_dir, db_helper)
    return _processor_manager


def get_processor_manager() -> ProcessorManager:
    """Get the global processor manager instance."""
    if _processor_manager is None:
        raise RuntimeError("ProcessorManager not initialized. Call init_processor_manager first.")
    return _processor_manager
