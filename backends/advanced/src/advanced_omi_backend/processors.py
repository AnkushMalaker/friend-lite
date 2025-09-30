"""Application-level processors for audio, transcription, memory, and cropping.

This module implements global processing queues and processors that handle
all processing tasks at the application level, decoupled from individual
client connections.
"""

import asyncio
import logging
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Tuple

# Import TranscriptionManager for type hints
from typing import TYPE_CHECKING, Any, Optional

from advanced_omi_backend.audio_processing_types import (
    AudioProcessingItem,
    TranscriptionItem,
    MemoryProcessingItem,
    CroppingItem,
)
from advanced_omi_backend.audio_utils import (
    _process_audio_cropping_with_relative_timestamps,
)
from advanced_omi_backend.client_manager import get_client_manager
from advanced_omi_backend.database import (
    AudioChunksRepository,
    ConversationsRepository,
    conversations_col,
)
from advanced_omi_backend.job_tracker import (
    AudioSource,
    StageEvent,
    get_job_tracker,
)
from advanced_omi_backend.memory import get_memory_service
from advanced_omi_backend.task_manager import get_pipeline_tracker
from advanced_omi_backend.users import get_user_by_id
from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from wyoming.audio import AudioChunk

# Lazy import to avoid config loading issues
# from advanced_omi_backend.memory import get_memory_service

if TYPE_CHECKING:
    from advanced_omi_backend.transcription import TranscriptionManager

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")

# Audio configuration constants
OMI_SAMPLE_RATE = 16_000
OMI_CHANNELS = 1
OMI_SAMPLE_WIDTH = 2
SEGMENT_SECONDS = 60
TARGET_SAMPLES = OMI_SAMPLE_RATE * SEGMENT_SECONDS

if TYPE_CHECKING:
    from advanced_omi_backend.transcription import TranscriptionManager


# Legacy data classes removed - now using unified types from audio_processing_types.py


class ProcessorManager:
    """Manages all application-level processors and queues."""

    def __init__(self, chunk_dir: Path, audio_chunks_repository: AudioChunksRepository):
        self.chunk_dir = chunk_dir
        self.repository = audio_chunks_repository

        # Unified pipeline queues with job tracking (job_id, item) tuples
        self.audio_queue: asyncio.Queue[Optional[Tuple[str, AudioProcessingItem]]] = asyncio.Queue()
        self.transcription_queue: asyncio.Queue[Optional[Tuple[str, TranscriptionItem]]] = asyncio.Queue()
        self.memory_queue: asyncio.Queue[Optional[Tuple[str, MemoryProcessingItem]]] = asyncio.Queue()
        self.cropping_queue: asyncio.Queue[Optional[Tuple[str, CroppingItem]]] = asyncio.Queue()

        # Processor tasks
        self.audio_processor_task: Optional[asyncio.Task] = None
        self.transcription_processor_task: Optional[asyncio.Task] = None
        self.memory_processor_task: Optional[asyncio.Task] = None
        self.cropping_processor_task: Optional[asyncio.Task] = None

        # Services - lazy import
        self.memory_service = None
        self.pipeline_tracker = get_pipeline_tracker()
        self.client_manager = get_client_manager()
        self.job_tracker = get_job_tracker()  # Add job tracker instance

        # Track active file sinks per client
        self.active_file_sinks: dict[str, LocalFileSink] = {}
        self.active_audio_uuids: dict[str, str] = {}

        # Transcription managers pool
        self.transcription_managers: dict[str, "TranscriptionManager"] = {}

        # Shutdown flag
        self.shutdown_flag = False

        # Task tracking for specific processing jobs
        self.processing_tasks: dict[str, dict[str, str]] = {}  # client_id -> {stage: task_id}

        # Track clients currently being closed to prevent duplicate close operations
        self.closing_clients: set[str] = set()

        # Track pipeline job completion
        self.completed_pipeline_jobs: set[str] = set()

    async def _update_memory_status(self, conversation_id: str, status: str):
        """Update memory processing status for conversation."""
        try:
            conversations_repo = ConversationsRepository(conversations_col)
            await conversations_repo.update_memory_processing_status(conversation_id, status)

            audio_logger.info(f"📝 Updated memory status to {status} for conversation {conversation_id}")
        except Exception as e:
            audio_logger.error(f"Failed to update memory status to {status} for conversation {conversation_id}: {e}")

    async def submit_audio_for_processing(self, processing_item: AudioProcessingItem) -> str:
        """Submit audio processing item and return job_id for tracking.

        This is the unified entry point for both WebSocket and file upload processing.
        """
        # Create pipeline job
        job_id = await self.job_tracker.create_pipeline_job(
            audio_source=processing_item.audio_source,
            user_id=processing_item.user_id,
            identifier=processing_item.get_identifier(),
            audio_uuid=processing_item.audio_uuid
        )

        # Track enqueue event
        await self.job_tracker.track_stage_event(job_id, "audio", StageEvent.ENQUEUE)

        # Queue for processing
        await self.audio_queue.put((job_id, processing_item))

        logger.info(f"Submitted audio processing job {job_id} from {processing_item.audio_source.value}")
        return job_id

    async def complete_pipeline_job_if_ready(self, job_id: str):
        """Complete pipeline job if all stages are done or mark as failed if any stage failed."""
        try:
            audio_logger.info(f"⏱️  [PIPELINE] Checking if job {job_id} is ready for completion")

            # Check if job is already completed
            if job_id in self.completed_pipeline_jobs:
                audio_logger.info(f"⏱️  [PIPELINE] Job {job_id} already marked as completed")
                return

            # Get job status from job tracker
            job = await self.job_tracker.get_job(job_id)
            if not job:
                audio_logger.warning(f"⏱️  [PIPELINE] Job {job_id} not found in tracker")
                return

            # Check if all required stages are complete
            required_stages = ["audio", "transcription"]

            # Get pipeline stages as list, convert to dict by stage name
            stages_list = job.to_dict().get("pipeline_stages", [])
            stage_metrics = {stage["stage"]: stage for stage in stages_list}

            if "memory" in stage_metrics:
                required_stages.append("memory")

            audio_logger.info(f"⏱️  [PIPELINE] Required stages for job {job_id}: {required_stages}")

            # Check for stage failures or completion
            all_complete = True
            has_failure = False
            failed_stage = None
            stage_status = {}

            for stage in required_stages:
                stage_info = stage_metrics.get(stage, {})
                status_value = stage_info.get("status", "pending")

                # A stage is complete if status is "completed" or if it has a complete_time
                is_completed = (status_value == "completed" or stage_info.get("complete_time") is not None)
                stage_status[stage] = "completed" if is_completed else "incomplete"

                # Check if stage failed
                if status_value == "failed" or stage_info.get("error_message"):
                    has_failure = True
                    failed_stage = stage
                    stage_status[stage] = "failed"
                    break

                # Check if stage completed
                if not is_completed:
                    all_complete = False

            audio_logger.info(f"⏱️  [PIPELINE] Stage status for job {job_id}: {stage_status}")

            # Fail fast: if any stage failed, mark job as failed immediately
            if has_failure:
                await self.job_tracker.update_job_status(job_id, JobStatus.FAILED)
                self.completed_pipeline_jobs.add(job_id)
                audio_logger.error(f"❌ [PIPELINE] Failed job {job_id} (stage '{failed_stage}' failed)")
                return

            if all_complete:
                # Complete the pipeline job
                await self.job_tracker.complete_pipeline_job(job_id)
                self.completed_pipeline_jobs.add(job_id)

                audio_logger.info(f"✅ [PIPELINE] Completed job {job_id} (all stages finished)")
            else:
                audio_logger.info(f"⏱️  [PIPELINE] Job {job_id} not ready - waiting for: {[s for s, status in stage_status.items() if status == 'incomplete']}")

        except Exception as e:
            audio_logger.error(f"❌ [PIPELINE] Error checking job completion {job_id}: {e}", exc_info=True)

    async def start(self):
        """Start all processors."""
        # Create processor tasks
        self.audio_processor_task = asyncio.create_task(
            self._audio_processor_unified(), name="audio_processor"
        )
        self.transcription_processor_task = asyncio.create_task(
            self._transcription_processor_unified(), name="transcription_processor"
        )
        self.memory_processor_task = asyncio.create_task(
            self._memory_processor_unified(), name="memory_processor"
        )
        self.cropping_processor_task = asyncio.create_task(
            self._cropping_processor_unified(), name="cropping_processor"
        )

        # Track processor tasks in pipeline tracker
        self.pipeline_tracker.track_task(
            self.audio_processor_task, "audio_processor", {"type": "processor"}
        )
        self.pipeline_tracker.track_task(
            self.transcription_processor_task, "transcription_processor", {"type": "processor"}
        )
        self.pipeline_tracker.track_task(
            self.memory_processor_task, "memory_processor", {"type": "processor"}
        )
        self.pipeline_tracker.track_task(
            self.cropping_processor_task, "cropping_processor", {"type": "processor"}
        )

    async def _should_process_memory(self, user_id: str, conversation_id: str) -> tuple[bool, str]:
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
            conversation_id: Conversation ID to check transcript speakers

        Returns:
            Tuple of (should_process: bool, reason: str)
        """
        try:
            # Get user's primary speaker configuration
            user = await get_user_by_id(user_id)
            if not user or not user.primary_speakers:
                return True, "No primary speakers configured - processing all conversations"

            audio_logger.info(f"🔍 Checking primary speakers filter for conversation {conversation_id} - user has {len(user.primary_speakers)} primary speakers configured")

            # Get conversation data from conversations collection
            conversations_repo = ConversationsRepository(conversations_col)
            conversation = await conversations_repo.get_conversation(conversation_id)
            if not conversation or not conversation.get('transcript'):
                return True, "No transcript data available - processing conversation"
            
            # Extract speakers from transcript segments (normalized for comparison)
            transcript_speakers = set()
            transcript_speaker_originals = {}  # Keep original names for logging
            total_segments = 0
            identified_segments = 0
            
            for segment in conversation['transcript']:
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
            audio_logger.warning(f"Error checking primary speakers filter for {conversation_id}: {e} - defaulting to process conversation")
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

    # Compatibility methods for reprocessing - redirect to unified queues
    async def queue_transcription(self, item: TranscriptionItem):
        """Queue transcription item directly (for reprocessing scenarios)."""
        await self.transcription_queue.put((None, item))  # No job_id for direct queuing

    async def queue_memory(self, item: MemoryProcessingItem):
        """Queue memory item directly (for reprocessing scenarios)."""
        await self.memory_queue.put((None, item))  # No job_id for direct queuing

    async def queue_cropping(self, item: CroppingItem):
        """Queue cropping item directly (for reprocessing scenarios)."""
        await self.cropping_queue.put((None, item))  # No job_id for direct queuing

    def track_processing_task(
        self, client_id: str, stage: str, task_id: str, metadata: dict[str, Any] | None = None
    ):
        """Track a processing task for a specific client and stage."""
        if client_id not in self.processing_tasks:
            self.processing_tasks[client_id] = {}
        self.processing_tasks[client_id][stage] = task_id
        logger.info(f"Tracking task {task_id} for client {client_id} stage {stage}")

    # Legacy client-based tracking methods removed - unified pipeline uses job-based tracking

    def cleanup_processing_tasks(self, client_id: str):
        """Clean up processing task tracking for a client."""
        if client_id in self.processing_tasks:
            del self.processing_tasks[client_id]
            logger.debug(f"Cleaned up processing tasks for client {client_id}")

    # Legacy _is_stale method removed - unified pipeline uses job-based tracking

    # Legacy cleanup and stats methods removed - unified pipeline uses job-based tracking

    # Legacy get_pipeline_statistics removed - use job tracker metrics instead

    def get_processing_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent processing history from task manager."""
        history = []

        try:
            # Get completed tasks from pipeline tracker (get the last N items)
            completed_tasks = self.pipeline_tracker.completed_tasks[-limit:] if self.pipeline_tracker.completed_tasks else []

            for task_info in completed_tasks:
                task_type = task_info.metadata.get("type", "unknown")
                if task_type in ["memory", "cropping", "transcription_chunk"]:
                    history.append({
                        "client_id": task_info.metadata.get("client_id", "unknown"),
                        "conversation_id": task_info.metadata.get("conversation_id"),
                        "task_type": task_type,
                        "started_at": datetime.fromtimestamp(task_info.created_at, UTC).isoformat(),
                        "completed_at": datetime.fromtimestamp(task_info.completed_at, UTC).isoformat() if task_info.completed_at else None,
                        "duration_ms": (task_info.completed_at - task_info.created_at) * 1000 if task_info.completed_at else None,
                        "status": "completed" if task_info.completed_at and not task_info.error else "failed",
                        "error": task_info.error
                    })

            return sorted(history, key=lambda x: x["started_at"], reverse=True)
        except Exception as e:
            logger.error(f"Error getting processing history: {e}")
            return []

    def get_queue_health_status(self) -> dict[str, str]:
        """Determine queue health based on depth and processing rates."""
        health_status = {}

        queue_sizes = {
            "audio": self.audio_queue.qsize(),
            "transcription": self.transcription_queue.qsize(),
            "memory": self.memory_queue.qsize(),
            "cropping": self.cropping_queue.qsize()
        }

        for queue_name, size in queue_sizes.items():
            if size == 0:
                health_status[queue_name] = "idle"
            elif size < 5:
                health_status[queue_name] = "healthy"
            elif size < 20:
                health_status[queue_name] = "busy"
            else:
                health_status[queue_name] = "overloaded"

        return health_status

    async def mark_transcription_failed(self, client_id: str, error: str):
        """Mark transcription as failed and clean up transcription manager.

        This method handles transcription failures without closing audio files,
        allowing long recordings to continue even if intermediate transcriptions fail.

        Args:
            client_id: The client ID whose transcription failed
            error: The error message describing the failure
        """
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

                flush_start_time = time.time()
                audio_logger.info(
                    f"📤 Calling flush_final_transcript for client {client_id} (manager: {manager})"
                )
                try:
                    await manager.process_collected_audio()
                    flush_duration = time.time() - flush_start_time
                    audio_logger.info(
                        f"✅ ASR flush completed for client {client_id} in {flush_duration:.2f}s"
                    )
                except Exception as flush_error:
                    audio_logger.error(
                        f"❌ Error during flush_final_transcript: {flush_error}", exc_info=True
                    )
                    raise
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
        from advanced_omi_backend.transcription import TranscriptionManager
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


    # TODO: Replace with unified implementation
        audio_logger.info("Memory processor started")

        try:
            while not self.shutdown_flag:
                try:
                    queue_size_before = self.memory_queue.qsize()
                    item = await asyncio.wait_for(self.memory_queue.get(), timeout=30.0)
                    queue_size_after = self.memory_queue.qsize()

                    if item is None:  # Shutdown signal
                        self.memory_queue.task_done()
                        break

                    # Track pipeline dequeue event - find audio_uuid from conversation_id
                    try:
                        conversations_repo = ConversationsRepository(conversations_col)
                        conversation = await conversations_repo.get_conversation(item.conversation_id)
                        audio_uuid = conversation.get("audio_uuid") if conversation else None
                        if audio_uuid:
                            self.pipeline_tracker.track_dequeue(
                                "memory",
                                audio_uuid,
                                queue_size_after,
                                {
                                    "conversation_id": item.conversation_id,
                                    "client_id": item.client_id,
                                    "queue_size_before": queue_size_before
                                }
                            )
                    except Exception as e:
                        audio_logger.warning(f"Failed to track memory dequeue for conversation {item.conversation_id}: {e}")

                    try:
                        # Create background task for memory processing
                        task = asyncio.create_task(self._process_memory_item(item))

                        # Track task with 5 minute timeout
                        task_name = f"memory_{item.client_id}_{item.conversation_id}"
                        actual_task_id = self.pipeline_tracker.track_task(
                            task,
                            task_name,
                            {
                                "client_id": item.client_id,
                                "conversation_id": item.conversation_id,
                                "type": "memory",
                                "timeout": 3600,  # 60 minutes
                            },
                        )

                        # Register task with client for tracking (use the actual task_id from TaskManager)
                        self.track_processing_task(
                            item.client_id,
                            "memory",
                            actual_task_id,
                            {"conversation_id": item.conversation_id},
                        )

                    except Exception as e:
                        audio_logger.error(
                            f"Error queuing memory processing for {item.conversation_id}: {e}",
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
        """Process a single memory item (speech-driven conversations architecture)."""
        start_time = time.time()
        audio_logger.info(f"⏱️  [MEMORY] Starting memory processing for conversation {item.conversation_id}")

        try:
            # Get conversation data directly from conversations collection (speech-driven architecture)
            fetch_start = time.time()
            conversations_repo = ConversationsRepository(conversations_col)
            conversation = await conversations_repo.get_conversation(item.conversation_id)
            fetch_time = time.time() - fetch_start
            audio_logger.info(f"⏱️  [MEMORY] Fetched conversation in {fetch_time:.2f}s")

            if not conversation:
                audio_logger.error(
                    f"❌ [MEMORY] No conversation found for {item.conversation_id}, elapsed: {time.time() - start_time:.2f}s"
                )
                raise ValueError(f"No conversation found for {item.conversation_id}")

            # Extract conversation text from transcript segments
            transcript_start = time.time()
            full_conversation = ""
            transcript = conversation.get("transcript", [])
            if transcript:
                dialogue_lines = []
                for segment in transcript:
                    text = segment.get("text", "").strip()
                    if text:
                        speaker = segment.get("speaker", "Unknown")
                        dialogue_lines.append(f"{speaker}: {text}")
                full_conversation = "\n".join(dialogue_lines)
            transcript_time = time.time() - transcript_start
            audio_logger.info(f"⏱️  [MEMORY] Extracted transcript ({len(full_conversation)} chars) in {transcript_time:.2f}s")

            if not transcript:
                audio_logger.error(
                    f"❌ [MEMORY] No transcript found in conversation {item.conversation_id}, elapsed: {time.time() - start_time:.2f}s"
                )
                raise ValueError(f"No transcript found for {item.conversation_id}")

            if len(full_conversation) < 10:  # Minimum length check
                audio_logger.warning(
                    f"⏭️  [MEMORY] Conversation too short ({len(full_conversation)} chars), skipping. Elapsed: {time.time() - start_time:.2f}s"
                )
                return None

            # Check if memory processing should proceed based on primary speakers configuration
            filter_start = time.time()
            should_process, filter_reason = await self._should_process_memory(item.user_id, item.conversation_id)
            filter_time = time.time() - filter_start
            audio_logger.info(f"⏱️  [MEMORY] Speaker filter check in {filter_time:.2f}s: {filter_reason}")

            if not should_process:
                await self._update_memory_status(item.conversation_id, "skipped")
                audio_logger.info(f"⏭️  [MEMORY] Skipped (filter). Total elapsed: {time.time() - start_time:.2f}s")
                return None

            # Lazy import memory service
            if self.memory_service is None:
                init_start = time.time()
                audio_logger.info(f"⏱️  [MEMORY] Initializing memory service...")
                self.memory_service = get_memory_service()
                init_time = time.time() - init_start
                audio_logger.info(f"⏱️  [MEMORY] Memory service initialized in {init_time:.2f}s")

            # Process memory with timeout
            memory_start = time.time()
            audio_logger.info(f"⏱️  [MEMORY] Calling memory_service.add_memory()...")
            memory_result = await asyncio.wait_for(
                self.memory_service.add_memory(
                    full_conversation,
                    item.client_id,
                    item.conversation_id,  # Use conversation_id instead of audio_uuid
                    item.user_id,
                    item.user_email,
                    allow_update=True,
                ),
                timeout=3600,  # 60 minutes
            )
            memory_time = time.time() - memory_start
            audio_logger.info(f"⏱️  [MEMORY] Memory service completed in {memory_time:.2f}s: {memory_result}")

            if memory_result:
                # Check if this was a successful result with actual memories created
                success, created_memory_ids = memory_result
                logger.info(f"Memory result: {memory_result}")

                if success:
                    db_start = time.time()
                    # Add memory references to conversations collection (speech-driven architecture)
                    try:
                        conversations_repo = ConversationsRepository(conversations_col)

                        # Add memory references to conversation
                        memory_refs = [{"memory_id": mid, "created_at": datetime.now(UTC).isoformat(), "status": "created"} for mid in created_memory_ids]
                        await conversations_repo.add_memories(item.conversation_id, memory_refs)

                        # Update memory processing status
                        await conversations_repo.update_memory_processing_status(item.conversation_id, "completed")

                        db_time = time.time() - db_start
                        total_time = time.time() - start_time
                        audio_logger.info(
                            f"✅ [MEMORY] Success! Created {len(created_memory_ids)} memories (DB update: {db_time:.2f}s). Total: {total_time:.2f}s"
                        )
                    except Exception as e:
                        audio_logger.error(f"❌ [MEMORY] Failed to add memory references: {e}")
                        raise

                elif success and not created_memory_ids:
                    # Successful processing but no memories created (likely empty transcript)
                    await self._update_memory_status(item.conversation_id, "skipped")
                    audio_logger.info(
                        f"⏭️  [MEMORY] No memories created (empty transcript). Total elapsed: {time.time() - start_time:.2f}s"
                    )
                else:
                    # This shouldn't happen, but handle it gracefully
                    error_msg = f"Unexpected result: success={success}, ids={created_memory_ids}"
                    audio_logger.error(f"❌ [MEMORY] {error_msg}. Elapsed: {time.time() - start_time:.2f}s")
                    await self._update_memory_status(item.conversation_id, "failed")
                    raise ValueError(error_msg)

            else:
                error_msg = "Memory service returned False"
                audio_logger.error(f"❌ [MEMORY] {error_msg}. Elapsed: {time.time() - start_time:.2f}s")
                await self._update_memory_status(item.conversation_id, "failed")
                raise ValueError(error_msg)

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            audio_logger.error(f"❌ [MEMORY] Timeout after {elapsed:.2f}s")
            await self._update_memory_status(item.conversation_id, "failed")
            raise

        except Exception as e:
            elapsed = time.time() - start_time
            audio_logger.error(f"❌ [MEMORY] Exception after {elapsed:.2f}s: {e}", exc_info=True)
            await self._update_memory_status(item.conversation_id, "failed")
            raise

        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        audio_logger.info(
            f"🏁 MEMORY PROCESSING COMPLETED for conversation {item.conversation_id} in {processing_time_ms:.1f}ms (end time: {end_time})"
        )

    async def _cropping_processor(self):
        """Process audio cropping requests."""
        audio_logger.info("Audio cropping processor started")

        try:
            while not self.shutdown_flag:
                try:
                    queue_size_before = self.cropping_queue.qsize()
                    item = await asyncio.wait_for(self.cropping_queue.get(), timeout=30.0)
                    queue_size_after = self.cropping_queue.qsize()

                    if item is None:  # Shutdown signal
                        self.cropping_queue.task_done()
                        break

                    # Track pipeline dequeue event
                    self.pipeline_tracker.track_dequeue(
                        "cropping",
                        item.audio_uuid,
                        queue_size_after,
                        {
                            "client_id": item.client_id,
                            "queue_size_before": queue_size_before
                        }
                    )

                    try:
                        # Create background task for cropping
                        task = asyncio.create_task(
                            _process_audio_cropping_with_relative_timestamps(
                                item.original_path,
                                item.speech_segments,
                                item.output_path,
                                item.audio_uuid,
                                self.repository,
                            )
                        )

                        # Track task
                        task_name = f"cropping_{item.client_id}_{item.audio_uuid}"
                        actual_task_id = self.pipeline_tracker.track_task(
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

    # Unified processor methods with job tracking

    async def _audio_processor_unified(self):
        """Process unified audio items with job tracking."""
        audio_logger.info("Unified audio processor started")

        try:
            while not self.shutdown_flag:
                try:
                    # Get item with timeout
                    item = await asyncio.wait_for(self.audio_queue.get(), timeout=30.0)

                    if item is None:  # Shutdown signal
                        audio_logger.info("🛑 Audio processor received shutdown signal")
                        self.audio_queue.task_done()
                        break

                    job_id, processing_item = item

                    try:
                        audio_start = time.time()
                        audio_logger.info(f"⏱️  [AUDIO] Starting audio processing for job {job_id}")

                        # Track dequeue
                        await self.job_tracker.track_stage_event(job_id, "audio", StageEvent.DEQUEUE)

                        # Process based on source
                        if processing_item.audio_source == AudioSource.WEBSOCKET:
                            audio_file_path = await self._process_websocket_audio(processing_item)
                        else:
                            audio_file_path = await self._process_file_upload_audio(processing_item)

                        audio_time = time.time() - audio_start
                        audio_logger.info(f"⏱️  [AUDIO] Audio processing completed in {audio_time:.2f}s")

                        # Track completion
                        await self.job_tracker.track_stage_event(job_id, "audio", StageEvent.COMPLETE)

                        # Create transcription item and queue
                        # client_id must exist (set during AudioProcessingItem creation)
                        if not processing_item.client_id:
                            raise ValueError(f"Missing client_id in processing_item for job {job_id}")

                        transcription_item = TranscriptionItem(
                            audio_file_path=audio_file_path,
                            audio_uuid=processing_item.audio_uuid,
                            client_id=processing_item.client_id,
                            user_id=processing_item.user_id,
                            user_email=processing_item.user_email
                        )

                        # Mark audio stage complete
                        await self.job_tracker.track_stage_event(job_id, "audio", StageEvent.COMPLETE)

                        # Track transcription enqueue
                        await self.job_tracker.track_stage_event(job_id, "transcription", StageEvent.ENQUEUE)
                        await self.transcription_queue.put((job_id, transcription_item))

                        audio_logger.info(f"✅ [AUDIO] Complete for job {job_id}. Total: {audio_time:.2f}s")

                        # Check if pipeline job can be completed
                        await self.complete_pipeline_job_if_ready(job_id)

                    except Exception as e:
                        elapsed = time.time() - audio_start if 'audio_start' in locals() else 0
                        audio_logger.error(f"❌ [AUDIO] Error after {elapsed:.2f}s for job {job_id}: {e}", exc_info=True)
                        # Track failure
                        await self.job_tracker.track_stage_event(job_id, "audio", StageEvent.ERROR)
                        # Check if job should be marked as failed
                        await self.complete_pipeline_job_if_ready(job_id)
                    finally:
                        self.audio_queue.task_done()

                except asyncio.TimeoutError:
                    # Periodic health check
                    queue_size = self.audio_queue.qsize()
                    if queue_size > 0:
                        audio_logger.debug(f"Unified audio processor health: {queue_size} items in queue")

        except Exception as e:
            audio_logger.error(f"Fatal error in unified audio processor: {e}", exc_info=True)
        finally:
            audio_logger.info("Unified audio processor stopped")

    async def _process_websocket_audio(self, item: AudioProcessingItem) -> str:
        """Process WebSocket audio chunks into WAV file."""
        import tempfile

        # Create temporary WAV file
        timestamp = int(time.time())
        wav_filename = f"{timestamp}_{item.device_name}_{item.audio_uuid}.wav"
        wav_file_path = str(self.chunk_dir / wav_filename)

        # Create file sink and write audio chunks
        sink = self._new_local_file_sink(wav_file_path, item.sample_rate)
        await sink.open()

        try:
            # Write all audio chunks to file
            for chunk_data in item.audio_chunks:
                from wyoming.audio import AudioChunk
                chunk = AudioChunk(
                    audio=chunk_data,
                    rate=item.sample_rate,
                    width=item.sample_width,
                    channels=item.channels
                )
                await sink.write(chunk)

            # Create database entry
            await self.repository.create_chunk(
                audio_uuid=item.audio_uuid,
                audio_path=wav_filename,
                client_id=f"{item.user_id[-8:]}-{item.device_name}",  # Generate client_id
                timestamp=timestamp,
                user_id=item.user_id,
                user_email=item.user_email,
            )

            audio_logger.info(f"📁 Created WebSocket audio file: {wav_filename} ({len(item.audio_chunks)} chunks)")
            return wav_file_path

        finally:
            await sink.close()

    async def _process_file_upload_audio(self, item: AudioProcessingItem) -> str:
        """Process uploaded audio file."""
        # For file uploads, audio_file_path should already be set
        if not item.audio_file_path:
            raise ValueError("File upload audio item missing audio_file_path")

        # Verify file exists
        from pathlib import Path
        if not Path(item.audio_file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {item.audio_file_path}")

        # Create audio_chunks database entry (unified with websocket flow)
        import time
        timestamp = int(time.time())
        audio_filename = Path(item.audio_file_path).name

        await self.repository.create_chunk(
            audio_uuid=item.audio_uuid,
            audio_path=audio_filename,
            client_id=item.client_id,
            timestamp=timestamp,
            user_id=item.user_id,
            user_email=item.user_email,
        )

        audio_logger.info(f"📁 Stored audio session for file upload: {item.audio_uuid}")
        return item.audio_file_path

    async def _transcription_processor_unified(self):
        """Process unified transcription items with job tracking."""
        audio_logger.info("Unified transcription processor started")

        try:
            while not self.shutdown_flag:
                try:
                    # Get item with timeout
                    item = await asyncio.wait_for(self.transcription_queue.get(), timeout=30.0)

                    if item is None:  # Shutdown signal
                        audio_logger.info("🛑 Unified transcription processor received shutdown signal")
                        self.transcription_queue.task_done()
                        break

                    job_id, transcription_item = item

                    try:
                        trans_start = time.time()
                        audio_logger.info(f"⏱️  [TRANSCRIPTION] Starting transcription for job {job_id}")

                        # Track dequeue
                        if job_id:
                            await self.job_tracker.track_stage_event(job_id, "transcription", StageEvent.DEQUEUE)

                        # Process transcription
                        conversation_id = await self._process_unified_transcription(transcription_item)

                        trans_time = time.time() - trans_start
                        audio_logger.info(f"⏱️  [TRANSCRIPTION] Transcription completed in {trans_time:.2f}s")

                        # Mark transcription stage complete
                        if job_id:
                            await self.job_tracker.track_stage_event(job_id, "transcription", StageEvent.COMPLETE)

                        # If conversation was created (speech detected), queue for memory processing
                        if conversation_id:
                            memory_item = MemoryProcessingItem(
                                conversation_id=conversation_id,
                                user_id=transcription_item.user_id,
                                user_email=transcription_item.user_email,
                                client_id=transcription_item.client_id,
                                transcript_version_id=None  # Use active version
                            )

                            # Track memory enqueue
                            if job_id:
                                await self.job_tracker.track_stage_event(job_id, "memory", StageEvent.ENQUEUE)
                            await self.memory_queue.put((job_id, memory_item))

                            audio_logger.info(f"✅ [TRANSCRIPTION] Complete for job {job_id}. Conversation: {conversation_id}. Total: {trans_time:.2f}s")
                        else:
                            audio_logger.info(f"⏭️  [TRANSCRIPTION] Complete for job {job_id}. No speech detected. Total: {trans_time:.2f}s")

                        # Check if pipeline job can be completed
                        if job_id:
                            await self.complete_pipeline_job_if_ready(job_id)

                    except Exception as e:
                        audio_logger.error(f"Error in unified transcription processing for job {job_id}: {e}", exc_info=True)
                        # Track failure
                        if job_id:
                            await self.job_tracker.track_stage_event(job_id, "transcription", StageEvent.ERROR)
                            # Check if job should be marked as failed
                            await self.complete_pipeline_job_if_ready(job_id)
                    finally:
                        self.transcription_queue.task_done()

                except asyncio.TimeoutError:
                    # Periodic health check
                    queue_size = self.transcription_queue.qsize()
                    if queue_size > 0:
                        audio_logger.debug(f"Unified transcription processor health: {queue_size} items in queue")

        except Exception as e:
            audio_logger.error(f"Fatal error in unified transcription processor: {e}", exc_info=True)
        finally:
            audio_logger.info("Unified transcription processor stopped")

    async def _process_unified_transcription(self, item: TranscriptionItem) -> Optional[str]:
        """Process transcription using existing transcription infrastructure."""
        try:
            # Use the existing transcription infrastructure
            # This involves creating a client session and processing the complete audio file

            # Use client_id directly from TranscriptionItem (already set during AudioProcessingItem creation)
            client_id = item.client_id

            # Read the audio file and convert to AudioChunk format
            from pathlib import Path
            import wave

            wav_path = Path(item.audio_file_path)
            if not wav_path.exists():
                raise FileNotFoundError(f"Audio file not found: {item.audio_file_path}")

            # Read WAV file and create audio chunks
            with wave.open(str(wav_path), 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()

            from wyoming.audio import AudioChunk

            # Create a large audio chunk from the entire file
            audio_chunk = AudioChunk(
                audio=frames,
                rate=sample_rate,
                width=sample_width,
                channels=channels
            )

            # Initialize transcription manager for this unified processing
            from advanced_omi_backend.transcription import TranscriptionManager

            # Create a temporary transcription manager
            # Disable internal memory queuing since unified pipeline handles it
            temp_manager = TranscriptionManager(
                chunk_repo=self.repository,
                processor_manager=self,
                skip_memory_queuing=True  # Unified pipeline handles memory queuing
            )

            try:
                # Connect the manager
                await temp_manager.connect(client_id)

                # Process the audio chunk (this will handle speech detection and conversation creation)
                await temp_manager.transcribe_chunk(
                    item.audio_uuid,
                    audio_chunk,
                    client_id
                )

                # Process collected audio to finalize transcription
                conversation_id = await temp_manager.process_collected_audio()

                audio_logger.info(f"Unified transcription processed audio file {item.audio_file_path}, conversation: {conversation_id}")
                return conversation_id

            finally:
                # Clean up the temporary manager
                try:
                    await temp_manager.disconnect()
                except Exception as cleanup_error:
                    audio_logger.warning(f"Error cleaning up temp transcription manager: {cleanup_error}")

        except Exception as e:
            audio_logger.error(f"Error in unified transcription processing: {e}", exc_info=True)
            raise

    async def _memory_processor_unified(self):
        """Process unified memory items with job tracking."""
        audio_logger.info("Unified memory processor started")

        try:
            while not self.shutdown_flag:
                try:
                    # Get item with timeout
                    item = await asyncio.wait_for(self.memory_queue.get(), timeout=30.0)

                    if item is None:  # Shutdown signal
                        audio_logger.info("🛑 Unified memory processor received shutdown signal")
                        self.memory_queue.task_done()
                        break

                    job_id, memory_item = item

                    try:
                        # Track dequeue
                        if job_id:
                            await self.job_tracker.track_stage_event(job_id, "memory", StageEvent.DEQUEUE)

                        # Use existing memory processing logic
                        await self._process_memory_item(memory_item)

                        # Track completion
                        if job_id:
                            await self.job_tracker.track_stage_event(job_id, "memory", StageEvent.COMPLETE)
                            audio_logger.info(f"✅ Unified memory processing completed for job {job_id}")

                            # Check if pipeline job can be completed
                            await self.complete_pipeline_job_if_ready(job_id)
                        else:
                            audio_logger.info(f"✅ Memory processing completed (no job tracking)")

                    except Exception as e:
                        audio_logger.error(f"Error in unified memory processing for job {job_id}: {e}", exc_info=True)
                        # Track failure
                        if job_id:
                            await self.job_tracker.track_stage_event(job_id, "memory", StageEvent.ERROR)
                    finally:
                        self.memory_queue.task_done()

                except asyncio.TimeoutError:
                    # Periodic health check
                    queue_size = self.memory_queue.qsize()
                    if queue_size > 0:
                        audio_logger.debug(f"Unified memory processor health: {queue_size} items in queue")

        except Exception as e:
            audio_logger.error(f"Fatal error in unified memory processor: {e}", exc_info=True)
        finally:
            audio_logger.info("Unified memory processor stopped")

    async def _cropping_processor_unified(self):
        """Process unified cropping items with job tracking."""
        audio_logger.info("Unified cropping processor started")

        try:
            while not self.shutdown_flag:
                try:
                    # Get item with timeout
                    item = await asyncio.wait_for(self.cropping_queue.get(), timeout=30.0)

                    if item is None:  # Shutdown signal
                        audio_logger.info("🛑 Unified cropping processor received shutdown signal")
                        self.cropping_queue.task_done()
                        break

                    job_id, cropping_item = item

                    try:
                        # Track dequeue
                        await self.job_tracker.track_stage_event(job_id, "cropping", StageEvent.DEQUEUE)

                        # Process cropping using existing cropping logic
                        await _process_audio_cropping_with_relative_timestamps(
                            cropping_item.original_audio_path,
                            cropping_item.speech_segments,
                            cropping_item.output_audio_path,
                            cropping_item.audio_uuid,
                            self.repository,
                        )

                        # Track completion
                        await self.job_tracker.track_stage_event(job_id, "cropping", StageEvent.COMPLETE)

                        audio_logger.info(f"✅ Unified cropping processing completed for job {job_id}")

                        # Check if pipeline job can be completed
                        await self.complete_pipeline_job_if_ready(job_id)

                    except Exception as e:
                        audio_logger.error(f"Error in unified cropping processing for job {job_id}: {e}", exc_info=True)
                        # Track failure
                        await self.job_tracker.track_stage_event(job_id, "cropping", StageEvent.ERROR)
                    finally:
                        self.cropping_queue.task_done()

                except asyncio.TimeoutError:
                    # Periodic health check
                    queue_size = self.cropping_queue.qsize()
                    if queue_size > 0:
                        audio_logger.debug(f"Unified cropping processor health: {queue_size} items in queue")

        except Exception as e:
            audio_logger.error(f"Fatal error in unified cropping processor: {e}", exc_info=True)
        finally:
            audio_logger.info("Unified cropping processor stopped")

    # Client cleanup methods (moved from PipelineTracker)

    async def cleanup_client_tasks(self, client_id: str, timeout: float = 30.0) -> None:
        """Clean up client-specific resources and processing state."""
        logger.info(f"🧹 Starting client cleanup for {client_id}")

        try:
            # 1. Close active file sinks
            if client_id in self.active_file_sinks:
                try:
                    await self.active_file_sinks[client_id].close()
                    del self.active_file_sinks[client_id]
                    logger.debug(f"✅ Closed file sink for {client_id}")
                except Exception as e:
                    logger.error(f"❌ Error closing file sink for {client_id}: {e}")

            # 2. Close transcription managers
            if client_id in self.transcription_managers:
                try:
                    await self.transcription_managers[client_id].disconnect()
                    del self.transcription_managers[client_id]
                    logger.debug(f"✅ Closed transcription manager for {client_id}")
                except Exception as e:
                    logger.error(f"❌ Error closing transcription manager for {client_id}: {e}")

            # 3. Clean up processing tasks
            if client_id in self.processing_tasks:
                del self.processing_tasks[client_id]
                logger.debug(f"✅ Cleaned up processing tasks for {client_id}")

            # 4. Note: We don't cancel pipeline processing tasks (memory, cropping)
            # as these should continue independently after client disconnect
            logger.info(f"✅ Client cleanup completed for {client_id}")

        except Exception as e:
            logger.error(f"❌ Error during client cleanup for {client_id}: {e}", exc_info=True)

    def cleanup_processing_tasks(self, client_id: str) -> None:
        """Clean up processing task tracking for a client (non-async version)."""
        if client_id in self.processing_tasks:
            del self.processing_tasks[client_id]
            logger.debug(f"✅ Cleaned up processing task tracking for {client_id}")


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
