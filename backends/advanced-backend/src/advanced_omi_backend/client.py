import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from wyoming.audio import AudioChunk

from advanced_omi_backend.audio_cropping_utils import (
    _process_audio_cropping_with_relative_timestamps,
)
from advanced_omi_backend.debug_system_tracker import PipelineStage, get_debug_tracker
from advanced_omi_backend.memory import get_memory_service
from advanced_omi_backend.transcription import TranscriptionManager
from advanced_omi_backend.users import get_user_by_client_id

# Get loggers
audio_logger = logging.getLogger("audio_processing")

# Configuration constants
NEW_CONVERSATION_TIMEOUT_MINUTES = float(os.getenv("NEW_CONVERSATION_TIMEOUT_MINUTES", "1.5"))
AUDIO_CROPPING_ENABLED = os.getenv("AUDIO_CROPPING_ENABLED", "true").lower() == "true"
MIN_SPEECH_SEGMENT_DURATION = float(os.getenv("MIN_SPEECH_SEGMENT_DURATION", "1.0"))
CROPPING_CONTEXT_PADDING = float(os.getenv("CROPPING_CONTEXT_PADDING", "0.1"))

# Audio configuration constants
OMI_SAMPLE_RATE = 16_000
OMI_CHANNELS = 1
OMI_SAMPLE_WIDTH = 2
SEGMENT_SECONDS = 60
TARGET_SAMPLES = OMI_SAMPLE_RATE * SEGMENT_SECONDS

# Get services
memory_service = get_memory_service()


class ClientState:
    """Manages all state for a single client connection."""

    def __init__(
        self,
        client_id: str,
        ac_db_collection_helper,
        chunk_dir: Path,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
    ):
        self.client_id = client_id
        self.connected = True
        self.db_helper = ac_db_collection_helper
        self.chunk_dir = chunk_dir
        # Store minimal user data needed for memory processing (avoids tight coupling to User model)
        self.user_id = user_id
        self.user_email = user_email

        # Per-client queues
        self.chunk_queue = asyncio.Queue[Optional[AudioChunk]]()
        self.transcription_queue = asyncio.Queue[Tuple[Optional[str], Optional[AudioChunk]]]()
        self.memory_queue = asyncio.Queue[
            Tuple[Optional[str], Optional[str], Optional[str]]
        ]()  # (transcript, client_id, audio_uuid)

        # Per-client file sink
        self.file_sink: Optional[LocalFileSink] = None
        self.current_audio_uuid: Optional[str] = None

        # Per-client transcription manager
        self.transcription_manager: Optional[TranscriptionManager] = None

        # Conversation timeout tracking
        self.last_transcript_time: Optional[float] = None
        self.conversation_start_time: float = time.time()

        # Prevent double conversation closure
        self.conversation_closed: bool = False

        # Speech segment tracking for audio cropping
        self.speech_segments: dict[str, list[tuple[float, float]]] = (
            {}
        )  # audio_uuid -> [(start, end), ...]
        self.current_speech_start: dict[str, Optional[float]] = {}  # audio_uuid -> start_time

        # Conversation transcript collection for end-of-conversation memory processing
        self.conversation_transcripts: list[str] = (
            []
        )  # Collect all transcripts for this conversation

        # Tasks for this client
        self.saver_task: Optional[asyncio.Task] = None
        self.transcription_task: Optional[asyncio.Task] = None
        self.memory_task: Optional[asyncio.Task] = None
        self.background_memory_task: Optional[asyncio.Task] = None

        # Debug tracking
        self.transaction_id: Optional[str] = None

    def _new_local_file_sink(self, file_path):
        """Create a properly configured LocalFileSink with all wave parameters set."""
        # TODO: Use client.sample_rate etc here
        sink = LocalFileSink(
            file_path=file_path,
            sample_rate=int(OMI_SAMPLE_RATE),
            channels=int(OMI_CHANNELS),
            sample_width=int(OMI_SAMPLE_WIDTH),
        )
        return sink

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
            if start_time is not None:  # Type guard
                if audio_uuid not in self.speech_segments:
                    self.speech_segments[audio_uuid] = []
                self.speech_segments[audio_uuid].append((start_time, timestamp))
                self.current_speech_start[audio_uuid] = None
                duration = timestamp - start_time
                audio_logger.info(
                    f"Recorded speech segment for {audio_uuid}: {start_time:.3f} -> {timestamp:.3f} (duration: {duration:.3f}s)"
                )
        else:
            audio_logger.warning(f"Speech end recorded for {audio_uuid} but no start time found")

    async def start_processing(self):
        """Start the processing tasks for this client."""
        self.saver_task = asyncio.create_task(self._audio_saver())
        self.transcription_task = asyncio.create_task(self._transcription_processor())
        self.memory_task = asyncio.create_task(self._memory_processor())
        audio_logger.info(f"Started processing tasks for client {self.client_id}")

    async def disconnect(self):
        """Clean disconnect of client state."""
        if not self.connected:
            return

        self.connected = False
        audio_logger.info(f"Disconnecting client {self.client_id}")

        # Close current conversation with all processing before signaling shutdown
        await self._close_current_conversation()

        # Signal processors to stop
        await self.chunk_queue.put(None)
        await self.transcription_queue.put((None, None))
        await self.memory_queue.put((None, None, None))

        # Wait for tasks to complete gracefully, with cancellation fallback
        # Use longer timeouts for transcription tasks that may be waiting on Deepgram API
        transcription_timeout = (
            60.0  # 1 minute for transcription (Deepgram can take time for large files)
        )
        saver_timeout = (
            60.0  # 1 minute for saver (handles conversation closure and memory processing)
        )
        default_timeout = 15.0  # 15 seconds for other tasks (increased from 3s)

        tasks_to_cleanup = []
        if self.saver_task:
            tasks_to_cleanup.append(("saver", self.saver_task, saver_timeout))
        if self.transcription_task:
            tasks_to_cleanup.append(
                ("transcription", self.transcription_task, transcription_timeout)
            )
        if self.memory_task:
            tasks_to_cleanup.append(("memory", self.memory_task, default_timeout))

        # Background memory task gets much longer timeout since it could be doing Ollama processing
        if self.background_memory_task:
            tasks_to_cleanup.append(
                ("background_memory", self.background_memory_task, 300.0)
            )  # 5 minutes

        for task_name, task, timeout in tasks_to_cleanup:
            try:
                # Try to wait for graceful completion with task-specific timeout
                await asyncio.wait_for(task, timeout=timeout)
                audio_logger.debug(
                    f"Task {task_name} completed gracefully for client {self.client_id}"
                )
            except asyncio.TimeoutError:
                audio_logger.warning(
                    f"Task {task_name} did not complete gracefully after {timeout}s, cancelling for client {self.client_id}"
                )
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    audio_logger.debug(
                        f"Task {task_name} cancelled successfully for client {self.client_id}"
                    )
            except Exception as e:
                audio_logger.error(
                    f"Error waiting for task {task_name} to complete for client {self.client_id}: {e}"
                )
                task.cancel()

        # Clean up transcription manager
        if self.transcription_manager:
            await self.transcription_manager.disconnect()
            self.transcription_manager = None

        # Clean up any remaining speech segment tracking
        self.speech_segments.clear()
        self.current_speech_start.clear()
        self.conversation_transcripts.clear()  # Clear conversation transcripts

        audio_logger.info(f"Client {self.client_id} disconnected and cleaned up")

    def _should_start_new_conversation(self) -> bool:
        """Check if we should start a new conversation based on timeout."""
        if self.last_transcript_time is None:
            return False  # No transcript yet, keep current conversation

        current_time = time.time()
        time_since_last_transcript = current_time - self.last_transcript_time
        timeout_seconds = NEW_CONVERSATION_TIMEOUT_MINUTES * 60

        return time_since_last_transcript > timeout_seconds

    async def _close_current_conversation(self):
        """Close the current conversation with proper cleanup including audio cropping and speaker processing."""
        # Prevent double closure
        if self.conversation_closed:
            audio_logger.debug(
                f"🔒 Conversation already closed for client {self.client_id}, skipping"
            )
            return

        self.conversation_closed = True

        if self.file_sink:
            # Store current audio info before closing
            current_uuid = self.current_audio_uuid
            current_path = self.file_sink.file_path

            audio_logger.info(f"🔒 Closing conversation {current_uuid}, file: {current_path}")

            # Flush any remaining transcript from ASR before waiting for queue
            if self.transcription_manager:
                try:
                    # Calculate audio duration for proportional timeout
                    audio_duration = time.time() - self.conversation_start_time
                    audio_logger.info(
                        f"🏁 Flushing final transcript for {current_uuid} (duration: {audio_duration:.1f}s)"
                    )
                    await self.transcription_manager.flush_final_transcript(audio_duration)
                except Exception as e:
                    audio_logger.error(f"Error flushing final transcript for {current_uuid}: {e}")

            # Wait for transcription queue to finish with timeout to prevent hanging
            try:
                await asyncio.wait_for(
                    self.transcription_queue.join(), timeout=60.0
                )  # Increased timeout for final transcript
                audio_logger.info("Transcription queue processing completed")
            except asyncio.TimeoutError:
                audio_logger.warning(
                    f"Transcription queue join timed out after 15 seconds for {current_uuid}"
                )

            # Allow more time for batch transcript processing to complete and be stored in database
            # This helps prevent race conditions where transcription completes after conversation close
            await asyncio.sleep(2.0)

            # Process memory at end of conversation if we have transcripts
            # First check conversation_transcripts list, then fall back to database
            full_conversation = ""
            transcript_source = ""

            if self.conversation_transcripts and current_uuid:
                full_conversation = " ".join(self.conversation_transcripts).strip()
                transcript_source = f"memory ({len(self.conversation_transcripts)} segments)"
            elif current_uuid and self.db_helper:
                # Fallback: get transcripts from database if conversation_transcripts is empty
                # This handles the race condition where transcript processing completes after conversation close
                try:
                    audio_logger.info(
                        f"💭 Conversation transcripts list empty, checking database for {current_uuid}"
                    )
                    db_transcripts = await self.db_helper.get_transcript_segments(current_uuid)
                    if db_transcripts:
                        transcript_texts = [segment.get("text", "") for segment in db_transcripts]
                        full_conversation = " ".join(transcript_texts).strip()
                        transcript_source = f"database ({len(db_transcripts)} segments)"
                        audio_logger.info(
                            f"💭 Retrieved {len(db_transcripts)} transcript segments from database"
                        )
                except Exception as e:
                    audio_logger.error(
                        f"💭 Error retrieving transcripts from database for {current_uuid}: {e}"
                    )

            if full_conversation and current_uuid:
                # MODIFIED: Process all transcripts for memory storage, regardless of length
                # Additional safety check - ensure we have some content
                if len(full_conversation) < 1:
                    audio_logger.info(
                        f"💭 Skipping memory processing for conversation {current_uuid} - completely empty"
                    )
                else:
                    # Process even very short conversations to ensure all transcripts are stored
                    audio_logger.info(
                        f"💭 Queuing memory processing for conversation {current_uuid} from {transcript_source} (length: {len(full_conversation)} chars)"
                    )
                    audio_logger.info(
                        f"💭 Full conversation text: {full_conversation[:200]}..."
                    )  # Log first 200 chars

                    # Use stored user information instead of database lookup
                    # This prevents lookup failures after client cleanup
                    if self.user_id and self.user_email:
                        # Process memory in background to avoid blocking conversation close
                        self.background_memory_task = asyncio.create_task(
                            self._process_memory_background(
                                full_conversation, current_uuid, self.user_id, self.user_email
                            )
                        )
                    else:
                        audio_logger.error(
                            f"💭 Cannot process memory for {current_uuid}: no user information stored for client {self.client_id}"
                        )

                audio_logger.info(f"💭 Memory processing queued in background for {current_uuid}")
            else:
                audio_logger.info(
                    f"ℹ️ No transcripts to process for memory in conversation {current_uuid}"
                )

            if self.file_sink:
                await self.file_sink.close()
            else:
                audio_logger.warning(f"File sink was None during close for client {self.client_id}")

            # Track successful audio chunk save in metrics
            try:
                # Removed old metrics call - using SystemTracker instead
                file_path = Path(current_path)
                if file_path.exists():
                    # Estimate duration (60 seconds per chunk is TARGET_SAMPLES)
                    duration_seconds = SEGMENT_SECONDS

                    # Calculate voice activity if we have speech segments
                    voice_activity_seconds = 0
                    if current_uuid and current_uuid in self.speech_segments:
                        for start, end in self.speech_segments[current_uuid]:
                            voice_activity_seconds += end - start

                    audio_logger.debug(
                        f"📊 Recorded audio chunk metrics: {duration_seconds}s total, {voice_activity_seconds}s voice activity"
                    )
                else:
                    audio_logger.warning(f"📊 Audio file not found after save: {current_path}")
            except Exception as e:
                audio_logger.error(f"📊 Error recording audio metrics: {e}")

            self.file_sink = None

            # Process audio cropping if we have speech segments
            if current_uuid and current_path:
                if current_uuid in self.speech_segments:
                    speech_segments = self.speech_segments[current_uuid]
                    audio_logger.info(
                        f"🎯 Found {len(speech_segments)} speech segments for {current_uuid}: {speech_segments}"
                    )
                    audio_logger.info(f"🎯 Audio file path: {current_path}")
                    if speech_segments:  # Only crop if we have speech segments
                        cropped_path = str(current_path).replace(".wav", "_cropped.wav")

                        # Process in background - won't block
                        asyncio.create_task(
                            self._process_audio_cropping(
                                f"{self.chunk_dir}/{current_path}",
                                speech_segments,
                                f"{self.chunk_dir}/{cropped_path}",
                                current_uuid,
                            )
                        )
                        audio_logger.info(
                            f"✂️ Queued audio cropping for {current_path} with {len(speech_segments)} speech segments"
                        )
                    else:
                        audio_logger.info(
                            f"⚠️ Empty speech segments list found for {current_path}, skipping cropping"
                        )

                    # Clean up segments for this conversation
                    del self.speech_segments[current_uuid]
                    if current_uuid in self.current_speech_start:
                        del self.current_speech_start[current_uuid]
                else:
                    audio_logger.info(
                        f"⚠️ No speech segments found for {current_path} (uuid: {current_uuid}), skipping cropping"
                    )

        else:
            audio_logger.info(f"🔒 No active file sink to close for client {self.client_id}")

    async def start_new_conversation(self):
        """Start a new conversation by closing current conversation and resetting state."""
        await self._close_current_conversation()

        # Reset conversation state
        self.current_audio_uuid = None
        self.conversation_start_time = time.time()
        self.last_transcript_time = None
        self.conversation_transcripts.clear()  # Clear collected transcripts for new conversation
        self.conversation_closed = False  # Reset closure flag for new conversation

        audio_logger.info(
            f"Client {self.client_id}: Started new conversation due to {NEW_CONVERSATION_TIMEOUT_MINUTES}min timeout"
        )

    async def _process_audio_cropping(
        self,
        original_path: str,
        speech_segments: list[tuple[float, float]],
        output_path: str,
        audio_uuid: str,
    ):
        """Background task for audio cropping using ffmpeg."""
        await _process_audio_cropping_with_relative_timestamps(
            original_path, speech_segments, output_path, audio_uuid
        )

    async def _process_memory_background(
        self, full_conversation: str, audio_uuid: str, user_id: str, user_email: str
    ):
        """Background task for memory processing to avoid blocking conversation close."""
        start_time = time.time()

        # User information is now passed directly to avoid database lookup issues after cleanup

        tracker = get_debug_tracker()
        transaction_id = tracker.create_transaction(user_id, self.client_id, audio_uuid)
        tracker.track_event(
            transaction_id,
            PipelineStage.MEMORY_STARTED,
            metadata={"conversation_length": len(full_conversation)},
        )

        try:
            # Track memory storage request
            # Removed old metrics call - using SystemTracker instead

            # Add general memory with fallback handling
            memory_result = await memory_service.add_memory(
                full_conversation,
                self.client_id,
                audio_uuid,
                user_id,
                user_email,
                db_helper=self.db_helper,
            )

            if memory_result:
                audio_logger.info(f"✅ Successfully added conversation memory for {audio_uuid}")
                tracker.track_event(
                    transaction_id,
                    PipelineStage.MEMORY_COMPLETED,
                    metadata={"processing_time": time.time() - start_time},
                )
            else:
                audio_logger.warning(
                    f"⚠️ Memory service returned False for {audio_uuid} - may have timed out"
                )
                tracker.track_event(
                    transaction_id,
                    PipelineStage.MEMORY_COMPLETED,
                    success=False,
                    error_message="Memory service returned False",
                    metadata={"processing_time": time.time() - start_time},
                )

        except Exception as e:
            audio_logger.error(f"❌ Error processing memory for {audio_uuid}: {e}")
            tracker.track_event(
                transaction_id,
                PipelineStage.MEMORY_COMPLETED,
                success=False,
                error_message=f"Exception during memory processing: {str(e)}",
                metadata={"processing_time": time.time() - start_time},
            )

        # Log processing summary
        processing_time_ms = (time.time() - start_time) * 1000
        audio_logger.info(
            f"🔄 Completed background memory processing for {audio_uuid} in {processing_time_ms:.1f}ms"
        )

    async def _audio_saver(self):
        """Per-client audio saver consumer."""
        try:
            while self.connected:
                audio_chunk = await self.chunk_queue.get()

                if audio_chunk is None:  # Disconnect signal
                    self.chunk_queue.task_done()
                    break

                try:
                    # Check if we should start a new conversation due to timeout
                    if self._should_start_new_conversation():
                        await self.start_new_conversation()

                    if self.file_sink is None:
                        # Create new file sink for this client
                        self.current_audio_uuid = uuid.uuid4().hex
                        timestamp = audio_chunk.timestamp or int(time.time())
                        wav_filename = f"{timestamp}_{self.client_id}_{self.current_audio_uuid}.wav"
                        audio_logger.info(
                            f"Creating file sink with: rate={int(OMI_SAMPLE_RATE)}, channels={int(OMI_CHANNELS)}, width={int(OMI_SAMPLE_WIDTH)}"
                        )
                        self.file_sink = self._new_local_file_sink(
                            f"{self.chunk_dir}/{wav_filename}"
                        )
                        await self.file_sink.open()

                        # Reset conversation closure flag when starting new audio
                        self.conversation_closed = False

                        await self.db_helper.create_chunk(
                            audio_uuid=self.current_audio_uuid,
                            audio_path=wav_filename,
                            client_id=self.client_id,
                            timestamp=timestamp,
                        )

                    await self.file_sink.write(audio_chunk)

                    # Queue for transcription
                    await self.transcription_queue.put((self.current_audio_uuid, audio_chunk))

                except Exception as e:
                    audio_logger.error(
                        f"Error processing audio chunk for client {self.client_id}: {e}"
                    )
                finally:
                    # Always mark task as done
                    self.chunk_queue.task_done()

        except Exception as e:
            audio_logger.error(
                f"Error in audio saver for client {self.client_id}: {e}", exc_info=True
            )
        finally:
            # Close current conversation with all processing when audio saver ends
            await self._close_current_conversation()

    async def _transcription_processor(self):
        """Per-client transcription processor."""
        try:
            while self.connected:
                try:
                    audio_uuid, chunk = await self.transcription_queue.get()

                    if audio_uuid is None or chunk is None:  # Disconnect signal
                        self.transcription_queue.task_done()
                        break

                    try:

                        # Get or create transcription manager
                        if self.transcription_manager is None:
                            self.transcription_manager = TranscriptionManager(
                                chunk_repo=self.db_helper
                            )
                            try:
                                await self.transcription_manager.connect(self.client_id)
                            except Exception as e:
                                audio_logger.error(
                                    f"Failed to create transcription manager for client {self.client_id}: {e}"
                                )
                                self.transcription_queue.task_done()
                                continue

                        # Process transcription
                        try:
                            await self.transcription_manager.transcribe_chunk(
                                audio_uuid, chunk, self.client_id
                            )
                            # Track transcription success
                            pass
                        except Exception as e:
                            audio_logger.error(
                                f"Error transcribing for client {self.client_id}: {e}"
                            )
                            # Track transcription failure
                            pass
                            # Recreate transcription manager on error
                            if self.transcription_manager:
                                await self.transcription_manager.disconnect()
                                self.transcription_manager = None

                    except Exception as e:
                        audio_logger.error(
                            f"Error processing transcription item for client {self.client_id}: {e}"
                        )
                    finally:
                        # Always mark task as done
                        self.transcription_queue.task_done()

                except asyncio.CancelledError:
                    # Handle cancellation gracefully
                    audio_logger.debug(
                        f"Transcription processor cancelled for client {self.client_id}"
                    )
                    break
                except Exception as e:
                    audio_logger.error(
                        f"Error in transcription processor loop for client {self.client_id}: {e}",
                        exc_info=True,
                    )

        except asyncio.CancelledError:
            audio_logger.debug(f"Transcription processor cancelled for client {self.client_id}")
        except Exception as e:
            audio_logger.error(
                f"Error in transcription processor for client {self.client_id}: {e}",
                exc_info=True,
            )
        finally:
            audio_logger.debug(f"Transcription processor stopped for client {self.client_id}")

    async def _memory_processor(self):
        """Per-client memory processor - currently unused as memory processing happens at conversation end."""
        try:
            while self.connected:
                try:
                    transcript, client_id, audio_uuid = await self.memory_queue.get()

                    if (
                        transcript is None or client_id is None or audio_uuid is None
                    ):  # Disconnect signal
                        self.memory_queue.task_done()
                        break

                    try:
                        # Memory processing now happens at conversation end, so this is effectively a no-op
                        # Keeping the processor running to avoid breaking the queue system
                        audio_logger.debug(
                            f"Memory processor received item but processing is now done at conversation end"
                        )
                    except Exception as e:
                        audio_logger.error(
                            f"Error processing memory item for client {self.client_id}: {e}"
                        )
                    finally:
                        # Always mark task as done
                        self.memory_queue.task_done()

                except asyncio.CancelledError:
                    # Handle cancellation gracefully
                    audio_logger.debug(f"Memory processor cancelled for client {self.client_id}")
                    break
                except Exception as e:
                    audio_logger.error(
                        f"Error in memory processor loop for client {self.client_id}: {e}",
                        exc_info=True,
                    )

        except asyncio.CancelledError:
            audio_logger.debug(f"Memory processor cancelled for client {self.client_id}")
        except Exception as e:
            audio_logger.error(
                f"Error in memory processor for client {self.client_id}: {e}",
                exc_info=True,
            )
        finally:
            audio_logger.debug(f"Memory processor stopped for client {self.client_id}")
