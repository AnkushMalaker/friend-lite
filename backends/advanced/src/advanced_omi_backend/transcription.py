import asyncio
import logging
import os
import time
from typing import Optional

from wyoming.audio import AudioChunk

from advanced_omi_backend.client_manager import get_client_manager
from advanced_omi_backend.conversation_repository import get_conversation_repository
from advanced_omi_backend.speaker_recognition_client import SpeakerRecognitionClient
from advanced_omi_backend.transcript_coordinator import get_transcript_coordinator
from advanced_omi_backend.transcription_providers import (
    BaseTranscriptionProvider,
    get_transcription_provider,
)
from advanced_omi_backend.config import load_diarization_settings_from_file

# ASR Configuration
TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER")  # Optional: 'deepgram' or 'parakeet'

logger = logging.getLogger(__name__)


class TranscriptionManager:
    """Manages transcription using any configured transcription provider."""

    def __init__(self, chunk_repo=None, processor_manager=None):
        self.provider: Optional[BaseTranscriptionProvider] = get_transcription_provider(
            TRANSCRIPTION_PROVIDER
        )
        self._current_audio_uuid = None
        self._audio_buffer = []  # Buffer for collecting audio chunks
        self._audio_start_time = None  # Track when audio collection started
        self._max_collection_time = 600.0  # 10 minutes timeout - allow longer conversations
        self._current_transaction_id = None  # Track current debug transaction
        self.chunk_repo = chunk_repo  # Database repository for chunks
        self.client_manager = get_client_manager()  # Cached client manager instance
        self.processor_manager = (
            processor_manager  # Reference to processor manager for completion tracking
        )
        self._client_id = None

        # Collection state tracking
        self._collecting = False
        self._collection_task = None

        # Optional speaker recognition
        self.speaker_client = SpeakerRecognitionClient()
        if self.speaker_client.enabled:
            logger.info("Speaker recognition integration enabled")

    def _get_current_client(self):
        """Get the current client state using ClientManager."""
        if not self._client_id:
            return None
        return self.client_manager.get_client(self._client_id)

    # REMOVED: Memory processing is now handled exclusively by conversation closure
    # to prevent duplicate processing. The _queue_memory_processing method has been
    # removed as part of the fix for double memory generation issue.

    async def connect(self, client_id: str | None = None):
        """Initialize transcription service for the client."""
        self._client_id = client_id

        if not self.provider:
            raise Exception("No transcription provider configured")

        try:
            await self.provider.connect(client_id)
            logger.info(
                f"{self.provider.name} transcription initialized for client {self._client_id}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to {self.provider.name} transcription service: {e}")
            raise

    async def process_collected_audio(self, audio_duration_seconds: Optional[float] = None):
        """Unified processing for all transcription providers."""
        logger.info(f"🚀 process_collected_audio called for client {self._client_id}")
        logger.info(
            f"📊 Current state - buffer size: {len(self._audio_buffer) if self._audio_buffer else 0}, collecting: {self._collecting}"
        )

        if not self.provider:
            logger.error("No transcription provider available")
            return

        # Cancel collection timeout task first to prevent interference
        if self._collection_task and not self._collection_task.done():
            logger.info(f"🛑 Cancelling collection timeout task before processing")
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                logger.info(f"✅ Collection task cancelled successfully")
            except Exception as e:
                logger.error(f"❌ Error cancelling collection task: {e}")

        # Get transcript from provider
        transcript_result = await self._get_transcript(audio_duration_seconds)

        # Process the result uniformly
        await self._process_transcript_result(transcript_result)

    async def _get_transcript(self, audio_duration_seconds: Optional[float] = None):
        """Get transcript from any provider using unified interface."""
        if not self.provider:
            logger.error("No transcription provider available")
            return None

        try:
            # For all providers, combine collected audio and call transcribe
            combined_audio = b"".join(chunk.audio for chunk in self._audio_buffer if chunk.audio)
            sample_rate = self._get_sample_rate()

            if not combined_audio or not sample_rate:
                logger.warning("No audio data or sample rate available for transcription")
                return None

            # Check if we should request diarization based on configuration
            config = load_diarization_settings_from_file()
            diarization_source = config.get("diarization_source", "pyannote")
            
            # Request diarization if using Deepgram as diarization source
            should_diarize = (diarization_source == "deepgram" and 
                            self.provider.name in ["Deepgram", "Deepgram-Streaming"])
            
            if should_diarize:
                logger.info(f"Requesting diarization from {self.provider.name} (diarization_source=deepgram)")
            
            return await self.provider.transcribe(combined_audio, sample_rate, diarize=should_diarize)

        except Exception as e:
            logger.error(f"Error getting transcript from {self.provider.name}: {e}")
            return None
        finally:
            # Clear the buffer for all provider types
            self._audio_buffer.clear()
            self._audio_start_time = None
            self._collecting = False

    def _get_sample_rate(self):
        """Get sample rate from client state or audio buffer."""
        current_client = self._get_current_client()
        if current_client and current_client.sample_rate:
            return current_client.sample_rate
        elif self._audio_buffer:
            return self._audio_buffer[0].rate
        return None

    async def _process_transcript_result(self, transcript_result):
        """Process transcript result uniformly for all providers."""
        if not transcript_result or not self._current_audio_uuid:
            logger.info(f"⚠️ No transcript result to process for {self._current_audio_uuid}")
            # Even with no transcript, signal completion to unblock memory processing
            if self._current_audio_uuid:
                coordinator = get_transcript_coordinator()
                coordinator.signal_transcript_ready(self._current_audio_uuid)
                logger.info(
                    f"⚠️ Signaled transcript completion (no data) for {self._current_audio_uuid}"
                )
            return

        start_time = time.time()

        try:
            # Store raw transcript data
            provider_name = self.provider.name if self.provider else "unknown"
            if self.chunk_repo:
                await self.chunk_repo.store_raw_transcript_data(
                    self._current_audio_uuid, transcript_result, provider_name
                )

            # Normalize transcript result
            normalized_result = self._normalize_transcript_result(transcript_result)
            if not normalized_result.get("text"):
                logger.warning(
                    f"No text in normalized transcript result for {self._current_audio_uuid}"
                )
                # Signal completion even with empty text to unblock memory processing
                coordinator = get_transcript_coordinator()
                coordinator.signal_transcript_ready(self._current_audio_uuid)
                logger.warning(
                    f"⚠️ Signaled transcript completion (empty text) for {self._current_audio_uuid}"
                )
                return

            # Get speaker diarization with word matching (if available)
            final_segments = []
            # Prepare transcript data for speaker service (define early so it's available for fallback)
            transcript_data = {
                "words": normalized_result.get("words", []),
                "text": normalized_result.get("text", ""),
            }
            if self.speaker_client.enabled and self._current_audio_uuid and self.chunk_repo:
                try:
                    # Get audio file path from database
                    chunk_data = await self.chunk_repo.get_chunk(self._current_audio_uuid)
                    if chunk_data and "audio_path" in chunk_data:
                        audio_path = chunk_data["audio_path"]
                        full_audio_path = f"/app/audio_chunks/{audio_path}"

                        logger.info(
                            f"🎤 Getting speaker diarization with word matching for: {full_audio_path}"
                        )

                        # Get user_id from client state
                        current_client = self._get_current_client()
                        user_id = current_client.user_id if current_client else None

                        # Call new speaker service endpoint
                        speaker_result = await self.speaker_client.diarize_identify_match(
                            full_audio_path, transcript_data, user_id=user_id
                        )

                        if speaker_result and speaker_result.get("segments"):
                            final_segments = speaker_result["segments"]
                            logger.info(
                                f"🎤 Speaker service returned {len(final_segments)} segments with matched text"
                            )
                        else:
                            logger.info("🎤 Speaker service returned no segments")
                    else:
                        logger.warning("No audio path found for speaker diarization")

                except Exception as e:
                    logger.warning(f"Speaker diarization with matching failed: {e}")

            # Only store segments if we got them from speaker service
            if not final_segments:
                logger.info(
                    f"📝 No diarization available - creating single segment from raw transcript for {self._current_audio_uuid}"
                )
                # Create a single segment from the raw transcript when speaker recognition is disabled
                if transcript_data and transcript_data.get("text"):
                    final_segments = [{
                        "text": transcript_data["text"],
                        "start": 0.0,
                        "end": 0.0,  # Duration unknown without audio analysis
                        "speaker": "",
                        "speaker_id": "",
                        "confidence": 1.0,
                    }]

            # Store final segments with required fields
            if self.chunk_repo and final_segments:
                for segment in final_segments:
                    # Add required fields for database storage
                    segment_to_store = {
                        "text": segment.get("text", ""),
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                        "speaker": segment.get("identified_as") or segment.get("speaker", ""),
                        "speaker_id": segment.get("speaker_id", ""),
                        "confidence": segment.get("confidence", 0.0),
                        "chunk_sequence": 0,
                        "absolute_timestamp": time.time() + segment.get("start", 0.0),
                    }
                    await self.chunk_repo.add_transcript_segment(
                        self._current_audio_uuid, segment_to_store
                    )

                # Add speakers if we have them
                speakers_found = set()
                for segment in final_segments:
                    # Use identified name if available, fallback to generic label
                    speaker_name = segment.get("identified_as") or segment.get("speaker")
                    if speaker_name:
                        speakers_found.add(speaker_name)

                for speaker in speakers_found:
                    await self.chunk_repo.add_speaker(self._current_audio_uuid, speaker)

            # Update client state
            current_client = self._get_current_client()
            if current_client:
                current_client.update_transcript_received()

            # Signal transcript coordinator
            coordinator = get_transcript_coordinator()
            coordinator.signal_transcript_ready(self._current_audio_uuid)

            # Queue memory processing now that transcription is complete
            await self._queue_memory_processing()

            # Queue audio cropping if we have diarization segments and cropping is enabled
            if final_segments and os.getenv("AUDIO_CROPPING_ENABLED", "false").lower() == "true":
                await self._queue_diarization_based_cropping(final_segments)

            # Update database transcription status
            if self.chunk_repo:
                status = "EMPTY" if not normalized_result.get("text").strip() else "COMPLETED"
                await self.chunk_repo.update_transcription_status(
                    self._current_audio_uuid, status, provider=provider_name
                )

            # Mark transcription as completed
            if self.processor_manager and self._client_id:
                self.processor_manager.track_processing_stage(
                    self._client_id,
                    "transcription",
                    "completed",
                    {
                        "audio_uuid": self._current_audio_uuid,
                        "segments": len(final_segments),
                        "provider": provider_name,
                    },
                )

        except Exception as e:
            logger.error(f"Error processing transcript result: {e}")
            # Update database transcription status to failed
            if self.chunk_repo and self._current_audio_uuid:
                await self.chunk_repo.update_transcription_status(
                    self._current_audio_uuid, "FAILED", error_message=str(e)
                )
        finally:
            # Log total processing time
            total_duration = time.time() - start_time
            logger.info(
                f"⏱️ Total transcript processing time: {total_duration:.2f}s for client {self._client_id}"
            )

    def _normalize_transcript_result(self, transcript_result):
        """Normalize transcript result to consistent format."""
        if isinstance(transcript_result, str):
            # Handle string response (legacy offline ASR)
            return {"text": transcript_result, "words": [], "segments": []}
        elif isinstance(transcript_result, dict):
            # Handle dict response (modern providers)
            return {
                "text": transcript_result.get("text", ""),
                "words": transcript_result.get("words", []),
                "segments": transcript_result.get("segments", []),
            }
        else:
            # Invalid format
            return {"text": "", "words": [], "segments": []}

    # REMOVED: All segment creation methods have been removed.
    # Segments are now only created by the speaker service endpoint /v1/diarize-identify-match
    # which handles diarization, speaker identification, and word-to-speaker matching.
    # This keeps all the segment creation logic in one place (speaker service).

    async def _queue_memory_processing(self):
        """Queue memory processing now that transcription is complete.

        This is data-driven - no event coordination needed since we know transcript is ready.
        """
        try:
            # Get user info from persistent conversation data instead of ephemeral client state
            conversation_repo = get_conversation_repository()
            conversation = await conversation_repo.get_conversation(self._current_audio_uuid)
            if not conversation:
                logger.warning(
                    f"No conversation found for memory processing {self._current_audio_uuid}"
                )
                return

            # Check if we have required data
            if not all(
                [self._current_audio_uuid, conversation.get("user_id"), conversation.get("user_email")]
            ):
                logger.warning(
                    f"Memory processing skipped - missing required data for {self._current_audio_uuid}"
                )
                logger.warning(f"    - audio_uuid: {bool(self._current_audio_uuid)}")
                logger.warning(
                    f"    - user_id: {bool(conversation.get('user_id'))}"
                )
                logger.warning(
                    f"    - user_email: {bool(conversation.get('user_email'))}"
                )
                return

            logger.info(
                f"💭 Queuing memory processing for completed transcription {self._current_audio_uuid}"
            )

            # Import here to avoid circular imports
            from advanced_omi_backend.processors import (
                MemoryProcessingItem,
                get_processor_manager,
            )

            # Queue memory processing - no event coordination needed
            processor_manager = get_processor_manager()
            await processor_manager.queue_memory(
                MemoryProcessingItem(
                    client_id=self._client_id,
                    user_id=conversation["user_id"],
                    user_email=conversation["user_email"],
                    audio_uuid=self._current_audio_uuid,
                )
            )

        except Exception as e:
            logger.error(f"Error queuing memory processing for {self._current_audio_uuid}: {e}")

    async def _queue_diarization_based_cropping(self, segments):
        """Queue audio cropping based on diarization segments."""
        try:
            # Import here to avoid circular imports
            from advanced_omi_backend.processors import (
                AudioCroppingItem,
                get_processor_manager,
            )

            # Get current client for user info
            current_client = self._get_current_client()
            if not current_client:
                logger.warning(f"No client state available for cropping {self._current_audio_uuid}")
                return

            # Get audio file path from database
            if not self.chunk_repo:
                logger.warning(
                    f"No chunk repository available for cropping {self._current_audio_uuid}"
                )
                return

            chunk_data = await self.chunk_repo.get_chunk(self._current_audio_uuid)
            if not chunk_data or "audio_path" not in chunk_data:
                logger.warning(f"No audio path found for cropping {self._current_audio_uuid}")
                return

            # Build file paths
            audio_filename = chunk_data["audio_path"]
            original_path = f"/app/audio_chunks/{audio_filename}"
            cropped_path = original_path.replace(".wav", "_cropped.wav")

            # Convert segments to cropping format (start, end tuples)
            cropping_segments = []
            for seg in segments:
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                if end > start:  # Only include valid segments
                    cropping_segments.append((start, end))

            if not cropping_segments:
                logger.debug(
                    f"No valid cropping segments from diarization for {self._current_audio_uuid}"
                )
                return

            logger.info(
                f"✂️ Queuing diarization-based cropping for {self._current_audio_uuid} "
                f"with {len(cropping_segments)} segments"
            )

            # Queue cropping with processor manager
            processor_manager = get_processor_manager()
            await processor_manager.queue_cropping(
                AudioCroppingItem(
                    client_id=self._client_id,
                    user_id=current_client.user_id,
                    audio_uuid=self._current_audio_uuid,
                    original_path=original_path,
                    speech_segments=cropping_segments,
                    output_path=cropped_path,
                )
            )

        except Exception as e:
            logger.error(
                f"Error queuing diarization-based cropping for {self._current_audio_uuid}: {e}"
            )

    async def disconnect(self):
        """Cleanly disconnect from transcription service."""
        logger.info(
            f"🔌 disconnect called for client {self._client_id} - provider: {self.provider.name if self.provider else 'None'}"
        )

        if not self.provider:
            logger.warning("No provider to disconnect")
            return

        # Cancel collection timeout task first to prevent interference
        if self._collection_task and not self._collection_task.done():
            logger.info(f"🛑 Cancelling collection timeout task for client {self._client_id}")
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                logger.info(f"✅ Collection task cancelled successfully")
            except Exception as e:
                logger.error(f"❌ Error cancelling collection task: {e}")

        # Process any remaining audio before disconnect
        if self._collecting or self._audio_buffer:
            logger.info(
                f"📊 Processing remaining audio on disconnect - buffer size: {len(self._audio_buffer)}"
            )
            await self.process_collected_audio()

        # Disconnect the provider
        try:
            await self.provider.disconnect()
            logger.info(
                f"{self.provider.name} transcription disconnected for client {self._client_id}"
            )
        except Exception as e:
            logger.error(f"Error disconnecting from {self.provider.name}: {e}")

    async def _collection_timeout_handler(self):
        """Handle collection timeout - process audio after timeout period."""
        logger.info(
            f"⏰ Collection timeout handler started for client {self._client_id} ({self._max_collection_time}s)"
        )
        try:
            await asyncio.sleep(self._max_collection_time)
            if self._collecting and self._audio_buffer:
                logger.info(
                    f"⏰ Collection timeout reached for client {self._client_id}, processing audio (buffer: {len(self._audio_buffer)} chunks)"
                )
                await self.process_collected_audio()
            else:
                logger.info(
                    f"⏰ Collection timeout reached but no audio to process (collecting: {self._collecting}, buffer: {len(self._audio_buffer) if self._audio_buffer else 0})"
                )
        except asyncio.CancelledError:
            logger.info(f"⏰ Collection timeout cancelled for client {self._client_id}")
        except Exception as e:
            logger.error(f"❌ Error in collection timeout handler: {e}", exc_info=True)

    async def transcribe_chunk(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Process audio chunk using the configured transcription provider."""
        if not self.provider:
            logger.error("No transcription provider available")
            return

        # Clean mode-based dispatch - no exception handling for control flow
        if self.provider.mode == "streaming":
            # Streaming providers process chunks immediately
            try:
                await self.provider.process_streaming_chunk(audio_uuid, chunk, client_id)
            except Exception as e:
                logger.error(f"Error in streaming processing for {audio_uuid}: {e}")
                await self._reconnect()
        else:
            # Batch providers collect chunks for later processing
            await self._collect_audio_chunk(audio_uuid, chunk, client_id)

    async def _collect_audio_chunk(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Collect audio chunk for batch processing."""
        logger.debug(
            f"📥 _collect_audio_chunk called for client {client_id}, audio_uuid: {audio_uuid}"
        )
        try:
            # Update current audio UUID
            if self._current_audio_uuid != audio_uuid:
                self._current_audio_uuid = audio_uuid
                logger.info(
                    f"🆕 New audio_uuid for {self.provider.name if self.provider else 'online'} batch: {audio_uuid}"
                )

                # Reset collection state for new audio session
                self._audio_buffer.clear()
                self._audio_start_time = time.time()
                self._collecting = True

                # Start collection timeout task
                if self._collection_task and not self._collection_task.done():
                    self._collection_task.cancel()
                self._collection_task = asyncio.create_task(self._collection_timeout_handler())

            # Add chunk to buffer if we have audio data
            if chunk.audio and len(chunk.audio) > 0:
                # Get sample rate from client state (set by audio processor)
                current_client = self._get_current_client()
                if current_client and current_client.sample_rate:
                    # Use sample rate from client state
                    expected_rate = current_client.sample_rate
                    if chunk.rate != expected_rate:
                        logger.warning(
                            f"⚠️ Sample rate mismatch for {client_id}: expected {expected_rate}Hz, got {chunk.rate}Hz"
                        )
                else:
                    # Fallback: no client state available, just log chunk rate
                    logger.info(
                        f"📊 Processing chunk with sample rate {chunk.rate}Hz for client {client_id} (no client state)"
                    )

                self._audio_buffer.append(chunk)
                logger.debug(
                    f"📦 Collected {len(chunk.audio)} bytes for {audio_uuid} (total chunks: {len(self._audio_buffer)})"
                )
            else:
                logger.warning(f"⚠️ Empty audio chunk received for {audio_uuid}")

        except Exception as e:
            logger.error(f"Error collecting audio chunk for {audio_uuid}: {e}")

    async def _reconnect(self):
        """Attempt to reconnect to transcription service."""
        if not self.provider:
            logger.warning("No provider to reconnect")
            return

        logger.info("Attempting to reconnect to transcription service...")

        try:
            await self.provider.disconnect()
            await asyncio.sleep(2)  # Brief delay before reconnecting
            await self.provider.connect(self._client_id)
            logger.info(f"Successfully reconnected to {self.provider.name}")
        except Exception as e:
            logger.error(f"Reconnection to {self.provider.name} failed: {e}")
