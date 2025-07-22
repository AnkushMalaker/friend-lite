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
from typing import Any, Dict, List, Optional, Tuple

from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from wyoming.audio import AudioChunk

from advanced_omi_backend.audio_cropping_utils import (
    _process_audio_cropping_with_relative_timestamps,
)
from advanced_omi_backend.client_manager import get_client_manager
from advanced_omi_backend.database import AudioChunksCollectionHelper
from advanced_omi_backend.debug_system_tracker import PipelineStage, get_debug_tracker

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
    db_helper: Optional[AudioChunksCollectionHelper] = None


@dataclass
class AudioCroppingItem:
    """Item for audio cropping queue."""

    client_id: str
    user_id: str
    audio_uuid: str
    original_path: str
    speech_segments: List[Tuple[float, float]]
    output_path: str


class ProcessorManager:
    """Manages all application-level processors and queues."""

    def __init__(self, chunk_dir: Path, db_helper: AudioChunksCollectionHelper):
        self.chunk_dir = chunk_dir
        self.db_helper = db_helper

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
        self.active_file_sinks: Dict[str, LocalFileSink] = {}
        self.active_audio_uuids: Dict[str, str] = {}

        # Transcription managers pool
        self.transcription_managers: Dict[str, TranscriptionManager] = {}

        # Shutdown flag
        self.shutdown_flag = False

        # Task tracking for specific processing jobs
        self.processing_tasks: Dict[str, Dict[str, str]] = {}  # client_id -> {stage: task_id}

        # Direct state tracking for synchronous operations
        self.processing_state: Dict[str, Dict[str, Any]] = {}  # client_id -> {stage: state_info}

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

    def _new_local_file_sink(self, file_path: str) -> LocalFileSink:
        """Create a properly configured LocalFileSink."""
        return LocalFileSink(
            file_path=file_path,
            sample_rate=int(OMI_SAMPLE_RATE),
            channels=int(OMI_CHANNELS),
            sample_width=int(OMI_SAMPLE_WIDTH),
        )

    async def queue_audio(self, item: AudioProcessingItem):
        """Queue audio for processing."""
        await self.audio_queue.put(item)

    async def queue_transcription(self, item: TranscriptionItem):
        """Queue audio for transcription."""
        audio_logger.info(
            f"📥 queue_transcription called for client {item.client_id}, audio_uuid: {item.audio_uuid}"
        )
        await self.transcription_queue.put(item)
        audio_logger.info(
            f"📤 Successfully put item in transcription_queue for client {item.client_id}, queue size: {self.transcription_queue.qsize()}"
        )

    async def queue_memory(self, item: MemoryProcessingItem):
        """Queue conversation for memory processing."""
        await self.memory_queue.put(item)

    async def queue_cropping(self, item: AudioCroppingItem):
        """Queue audio for cropping."""
        await self.cropping_queue.put(item)

    def track_processing_task(
        self, client_id: str, stage: str, task_id: str, metadata: Dict[str, Any] = None
    ):
        """Track a processing task for a specific client and stage."""
        if client_id not in self.processing_tasks:
            self.processing_tasks[client_id] = {}
        self.processing_tasks[client_id][stage] = task_id
        logger.info(f"Tracking task {task_id} for client {client_id} stage {stage}")

    def track_processing_stage(
        self, client_id: str, stage: str, status: str, metadata: Dict[str, Any] = None
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

    def get_processing_status(self, client_id: str) -> Dict[str, Any]:
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

    def get_all_processing_status(self) -> Dict[str, Any]:
        """Get processing status for all clients."""
        # Get all client IDs from both tracking types
        all_client_ids = set(self.processing_tasks.keys()) | set(self.processing_state.keys())
        return {client_id: self.get_processing_status(client_id) for client_id in all_client_ids}

    async def close_client_audio(self, client_id: str):
        """Close audio file for a client when conversation ends."""
        audio_logger.info(f"🔚 close_client_audio called for client {client_id}")

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
                    await manager.flush_final_transcript(audio_duration)
                    flush_duration = time.time() - flush_start_time
                    audio_logger.info(
                        f"✅ ASR flush completed for client {client_id} in {flush_duration:.2f}s"
                    )
                except Exception as flush_error:
                    audio_logger.error(
                        f"❌ Error during flush_final_transcript: {flush_error}", exc_info=True
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

    async def _audio_processor(self):
        """Process audio chunks and save to files."""
        audio_logger.info("Audio processor started")

        try:
            while not self.shutdown_flag:
                try:
                    # Get item with timeout to allow periodic health checks
                    item = await asyncio.wait_for(self.audio_queue.get(), timeout=30.0)

                    if item is None:  # Shutdown signal
                        self.audio_queue.task_done()
                        break

                    try:
                        # Get or create file sink for this client
                        if item.client_id not in self.active_file_sinks:
                            # Create new file
                            audio_uuid = uuid.uuid4().hex
                            timestamp = item.timestamp or int(time.time())
                            wav_filename = f"{timestamp}_{item.client_id}_{audio_uuid}.wav"

                            sink = self._new_local_file_sink(f"{self.chunk_dir}/{wav_filename}")
                            await sink.open()

                            self.active_file_sinks[item.client_id] = sink
                            self.active_audio_uuids[item.client_id] = audio_uuid

                            # Create database entry
                            await self.db_helper.create_chunk(
                                audio_uuid=audio_uuid,
                                audio_path=wav_filename,
                                client_id=item.client_id,
                                timestamp=timestamp,
                            )

                            # Notify client state about new audio UUID
                            client_state = self.client_manager.get_client(item.client_id)
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
                        audio_logger.info(
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
                        audio_logger.info(
                            f"✅ Successfully queued transcription for client {item.client_id}, audio_uuid: {audio_uuid}"
                        )

                    except Exception as e:
                        audio_logger.error(
                            f"Error processing audio for client {item.client_id}: {e}",
                            exc_info=True,
                        )
                    finally:
                        self.audio_queue.task_done()

                except asyncio.TimeoutError:
                    # Periodic health check
                    active_clients = len(self.active_file_sinks)
                    queue_size = self.audio_queue.qsize()
                    audio_logger.debug(
                        f"Audio processor health: {active_clients} active files, "
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
                                chunk_repo=self.db_helper, processor_manager=self
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
                            await manager.transcribe_chunk(
                                item.audio_uuid, item.audio_chunk, item.client_id
                            )
                            audio_logger.debug(
                                f"✅ Completed transcribe_chunk for client {item.client_id}"
                            )
                        except Exception as e:
                            audio_logger.error(
                                f"❌ Error in transcribe_chunk for client {item.client_id}: {e}",
                                exc_info=True,
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
                        self.task_manager.track_task(
                            task,
                            task_name,
                            {
                                "client_id": item.client_id,
                                "audio_uuid": item.audio_uuid,
                                "type": "memory",
                                "timeout": 300.0,  # 5 minutes
                            },
                        )

                        # Register task with client for tracking (use the actual task_id from TaskManager)
                        actual_task_id = f"{task_name}_{id(task)}"
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

    async def _get_transcript_with_coordination(self, db_helper, audio_uuid: str) -> Optional[list]:
        """Get transcript from database using event coordination (no polling/retry).

        This method uses the TranscriptCoordinator to wait for transcript completion
        events rather than polling the database with retries.
        """
        from advanced_omi_backend.transcript_coordinator import (
            get_transcript_coordinator,
        )

        # First, check if transcript already exists
        try:
            transcript_segments = await db_helper.get_transcript_segments(audio_uuid)
            if transcript_segments:
                audio_logger.info(f"Transcript already available for {audio_uuid}")
                return transcript_segments
        except Exception as e:
            audio_logger.warning(f"Error checking existing transcript for {audio_uuid}: {e}")

        # If not available, wait for transcript completion event
        coordinator = get_transcript_coordinator()
        audio_logger.info(f"Waiting for transcript completion event for {audio_uuid}")

        transcript_ready = await coordinator.wait_for_transcript_completion(
            audio_uuid, timeout=30.0
        )
        if not transcript_ready:
            audio_logger.warning(f"Transcript completion event timeout for {audio_uuid}")
            return None

        # Now get the transcript from database (should be available)
        try:
            transcript_segments = await db_helper.get_transcript_segments(audio_uuid)
            if transcript_segments:
                audio_logger.info(f"Retrieved transcript for {audio_uuid} after event coordination")
                return transcript_segments
            else:
                audio_logger.error(
                    f"Transcript event received but no transcript found in DB for {audio_uuid}"
                )
                return None
        except Exception as e:
            audio_logger.error(f"Error retrieving transcript after event for {audio_uuid}: {e}")
            return None

    async def _process_memory_item(self, item: MemoryProcessingItem):
        """Process a single memory item."""
        start_time = time.time()

        tracker = get_debug_tracker()
        transaction_id = tracker.create_transaction(item.user_id, item.client_id, item.audio_uuid)

        try:
            # Get transcript from database with event coordination (no polling/retry)
            if not item.db_helper:
                raise ValueError(
                    f"No db_helper provided for memory processing of {item.audio_uuid}"
                )

            transcript_segments = await self._get_transcript_with_coordination(
                item.db_helper, item.audio_uuid
            )
            if not transcript_segments:
                audio_logger.warning(
                    f"No transcripts found for {item.audio_uuid} via event coordination, skipping memory processing"
                )
                return None

            # Build full conversation from segments
            transcript_texts = [
                segment.get("text", "")
                for segment in transcript_segments
                if segment.get("text", "").strip()
            ]
            if not transcript_texts:
                audio_logger.warning(
                    f"No valid transcript text found for {item.audio_uuid}, skipping memory processing"
                )
                return None

            full_conversation = " ".join(transcript_texts).strip()
            if len(full_conversation) < 10:  # Minimum length check
                audio_logger.warning(
                    f"Conversation too short for memory processing ({len(full_conversation)} chars): {item.audio_uuid}"
                )
                return None

            tracker.track_event(
                transaction_id,
                PipelineStage.MEMORY_STARTED,
                metadata={"conversation_length": len(full_conversation)},
            )

            # Lazy import memory service
            if self.memory_service is None:
                from advanced_omi_backend.memory import get_memory_service

                self.memory_service = get_memory_service()

            # Process memory with timeout
            memory_result = await asyncio.wait_for(
                self.memory_service.add_memory(
                    full_conversation,
                    item.client_id,
                    item.audio_uuid,
                    item.user_id,
                    item.user_email,
                    db_helper=item.db_helper,
                ),
                timeout=300.0,  # 5 minutes
            )

            if memory_result:
                # Check if this was a successful result with actual memories created
                success, created_memory_ids = memory_result

                if success and created_memory_ids:
                    # Memories were actually created
                    audio_logger.info(
                        f"✅ Successfully processed memory for {item.audio_uuid} - created {len(created_memory_ids)} memories"
                    )

                    # Update database memory processing status to completed
                    if item.db_helper:
                        await item.db_helper.update_memory_processing_status(
                            item.audio_uuid, "COMPLETED"
                        )
                        audio_logger.info(
                            f"📝 Updated memory processing status to COMPLETED for {item.audio_uuid}"
                        )
                elif success and not created_memory_ids:
                    # Successful processing but no memories created (likely empty transcript)
                    audio_logger.info(
                        f"✅ Memory processing completed for {item.audio_uuid} but no memories created (likely empty transcript)"
                    )

                    # Update database memory processing status to skipped
                    if item.db_helper:
                        await item.db_helper.update_memory_processing_status(
                            item.audio_uuid,
                            "SKIPPED",
                            error_message="No memories created (empty transcript)",
                        )
                        audio_logger.info(
                            f"📝 Updated memory processing status to SKIPPED for {item.audio_uuid}"
                        )
                else:
                    # This shouldn't happen, but handle it gracefully
                    audio_logger.warning(
                        f"⚠️ Unexpected memory result for {item.audio_uuid}: success={success}, ids={created_memory_ids}"
                    )

                    # Update database memory processing status to failed
                    if item.db_helper:
                        await item.db_helper.update_memory_processing_status(
                            item.audio_uuid,
                            "FAILED",
                            error_message="Unexpected memory processing result",
                        )
                        audio_logger.warning(
                            f"📝 Updated memory processing status to FAILED for {item.audio_uuid}"
                        )

                tracker.track_event(
                    transaction_id,
                    PipelineStage.MEMORY_COMPLETED,
                    metadata={"processing_time": time.time() - start_time},
                )
            else:
                audio_logger.warning(f"⚠️ Memory service returned False for {item.audio_uuid}")

                # Update database memory processing status to failed
                if item.db_helper:
                    await item.db_helper.update_memory_processing_status(
                        item.audio_uuid, "FAILED", error_message="Memory service returned False"
                    )
                    audio_logger.warning(
                        f"📝 Updated memory processing status to FAILED for {item.audio_uuid}"
                    )

                tracker.track_event(
                    transaction_id,
                    PipelineStage.MEMORY_COMPLETED,
                    success=False,
                    error_message="Memory service returned False",
                    metadata={"processing_time": time.time() - start_time},
                )

        except asyncio.TimeoutError:
            audio_logger.error(f"Memory processing timed out for {item.audio_uuid}")

            # Update database memory processing status to failed
            if item.db_helper:
                await item.db_helper.update_memory_processing_status(
                    item.audio_uuid, "FAILED", error_message="Processing timeout (5 minutes)"
                )
                audio_logger.error(
                    f"📝 Updated memory processing status to FAILED (timeout) for {item.audio_uuid}"
                )

            tracker.track_event(
                transaction_id,
                PipelineStage.MEMORY_COMPLETED,
                success=False,
                error_message="Processing timeout (5 minutes)",
                metadata={"processing_time": time.time() - start_time},
            )
        except Exception as e:
            audio_logger.error(f"Error processing memory for {item.audio_uuid}: {e}")

            # Update database memory processing status to failed
            if item.db_helper:
                await item.db_helper.update_memory_processing_status(
                    item.audio_uuid, "FAILED", error_message=f"Exception: {str(e)}"
                )
                audio_logger.error(
                    f"📝 Updated memory processing status to FAILED (exception) for {item.audio_uuid}"
                )

            tracker.track_event(
                transaction_id,
                PipelineStage.MEMORY_COMPLETED,
                success=False,
                error_message=f"Exception: {str(e)}",
                metadata={"processing_time": time.time() - start_time},
            )

        processing_time_ms = (time.time() - start_time) * 1000
        audio_logger.info(
            f"🔄 Completed memory processing for {item.audio_uuid} in {processing_time_ms:.1f}ms"
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
                        self.task_manager.track_task(
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
                        actual_task_id = f"{task_name}_{id(task)}"
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


def init_processor_manager(chunk_dir: Path, db_helper: AudioChunksCollectionHelper):
    """Initialize the global processor manager."""
    global _processor_manager
    _processor_manager = ProcessorManager(chunk_dir, db_helper)
    return _processor_manager


def get_processor_manager() -> ProcessorManager:
    """Get the global processor manager instance."""
    if _processor_manager is None:
        raise RuntimeError("ProcessorManager not initialized. Call init_processor_manager first.")
    return _processor_manager
