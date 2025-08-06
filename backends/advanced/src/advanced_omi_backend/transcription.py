import asyncio
import logging
import os
import time
from typing import Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.vad import VoiceStarted, VoiceStopped

from advanced_omi_backend.client_manager import get_client_manager
from advanced_omi_backend.speaker_recognition_client import SpeakerRecognitionClient
from advanced_omi_backend.transcript_coordinator import get_transcript_coordinator
from advanced_omi_backend.transcription_providers import (
    OnlineTranscriptionProvider,
    get_transcription_provider,
)

# ASR Configuration
OFFLINE_ASR_TCP_URI = os.getenv("OFFLINE_ASR_TCP_URI", "tcp://192.168.0.110:8765/")
TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER")  # Optional: 'deepgram' or 'mistral'

logger = logging.getLogger(__name__)


class TranscriptionManager:
    """Manages transcription using either Deepgram batch API or offline ASR service."""

    # TODO: Accept callbacks list
    def __init__(self, chunk_repo=None, processor_manager=None):
        self.client = None
        self._current_audio_uuid = None
        self.online_provider: Optional[OnlineTranscriptionProvider] = get_transcription_provider(
            TRANSCRIPTION_PROVIDER
        )
        self.use_online_transcription = self.online_provider is not None
        self._audio_buffer = []  # Buffer for collecting audio chunks
        self._audio_start_time = None  # Track when audio collection started
        self._max_collection_time = 600.0  # 10 minutes timeout - allow longer conversations
        self._current_transaction_id = None  # Track current debug transaction
        self.chunk_repo = chunk_repo  # Database repository for chunks
        self.client_manager = get_client_manager()  # Cached client manager instance
        self.processor_manager = (
            processor_manager  # Reference to processor manager for completion tracking
        )

        # Event-driven ASR event handling for offline ASR
        self._event_queue = asyncio.Queue()
        self._event_reader_task = None
        self._stop_event = asyncio.Event()
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

        if self.use_online_transcription:
            # For online batch processing, we just need to ensure we have a provider
            if not self.online_provider:
                raise Exception("No online transcription provider configured")
            logger.info(
                f"{self.online_provider.name} batch transcription initialized for client {self._client_id}"
            )
            return

        try:
            self.client = AsyncTcpClient.from_uri(OFFLINE_ASR_TCP_URI)
            await self.client.connect()
            logger.info(f"Connected to offline ASR service at {OFFLINE_ASR_TCP_URI}")

            # Start the background event reader task for offline ASR
            self._stop_event.clear()
            self._event_reader_task = asyncio.create_task(self._read_events_continuously())
        except Exception as e:
            logger.error(f"Failed to connect to offline ASR service: {e}")
            self.client = None
            raise

    async def process_collected_audio(self, audio_duration_seconds: Optional[float] = None):
        """Unified processing for all transcription providers."""
        logger.info(f"🚀 process_collected_audio called for client {self._client_id}")
        logger.info(
            f"📊 Current state - buffer size: {len(self._audio_buffer) if self._audio_buffer else 0}, collecting: {self._collecting}"
        )

        # Cancel collection timeout task first to prevent interference (online only)
        if (
            self.use_online_transcription
            and self._collection_task
            and not self._collection_task.done()
        ):
            logger.info(f"🛑 Cancelling collection timeout task before processing")
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                logger.info(f"✅ Collection task cancelled successfully")
            except Exception as e:
                logger.error(f"❌ Error cancelling collection task: {e}")

        # Get transcript from appropriate provider
        transcript_result = await self._get_transcript(audio_duration_seconds)

        # Process the result uniformly
        await self._process_transcript_result(transcript_result)

    async def _get_transcript(self, audio_duration_seconds: Optional[float] = None):
        """Get transcript from online or offline provider."""
        if self.use_online_transcription:
            return await self._get_online_transcript()
        else:
            return await self._get_offline_transcript(audio_duration_seconds)

    async def _get_online_transcript(self):
        """Get transcript from online provider (Deepgram, etc.)."""
        if not self._audio_buffer:
            logger.info(f"⚠️ No audio data collected for client {self._client_id}")
            return None

        try:
            # Combine all audio chunks into a single buffer
            combined_audio = b"".join(chunk.audio for chunk in self._audio_buffer if chunk.audio)
            if not combined_audio:
                logger.warning(f"No valid audio data found for client {self._client_id}")
                return None

            # Get sample rate from client state
            current_client = self._get_current_client()
            sample_rate = None
            if current_client and current_client.sample_rate:
                sample_rate = current_client.sample_rate
            elif self._audio_buffer:
                sample_rate = self._audio_buffer[0].rate
            else:
                logger.error("❌ No sample rate available - cannot transcribe")
                return None

            # Call transcription provider
            transcript_result = await self.online_provider.transcribe(combined_audio, sample_rate)
            return transcript_result

        except Exception as e:
            logger.error(f"Error getting online transcript: {e}")
            return None
        finally:
            # Clear the buffer
            self._audio_buffer.clear()
            self._audio_start_time = None
            self._collecting = False

    async def _get_offline_transcript(self, audio_duration_seconds: Optional[float] = None):
        """Get transcript from offline ASR."""
        if not self.client or not self._current_audio_uuid:
            return None

        try:
            # Send AudioStop to signal end of audio stream
            audio_stop = AudioStop(timestamp=int(time.time()))
            await self.client.write_event(audio_stop.event())

            # Calculate timeout
            if audio_duration_seconds:
                proportional_timeout = audio_duration_seconds / 6.0
                max_wait = max(3.0, min(60.0, proportional_timeout))
            else:
                max_wait = 5.0

            start_time = time.time()
            collected_text = []

            # Wait for events from the background queue
            while (time.time() - start_time) < max_wait:
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                    if Transcript.is_type(event.type):
                        transcript_obj = Transcript.from_event(event)
                        transcript_text = transcript_obj.text.strip()
                        if transcript_text:
                            collected_text.append(transcript_text)
                except asyncio.TimeoutError:
                    break

            # Return text if we got any
            if collected_text:
                return {"text": " ".join(collected_text), "words": [], "segments": []}
            return None

        except Exception as e:
            logger.error(f"Error getting offline transcript: {e}")
            return None

    async def _process_transcript_result(self, transcript_result):
        """Process transcript result uniformly for all providers."""
        if not transcript_result or not self._current_audio_uuid:
            logger.info(f"⚠️ No transcript result to process for {self._current_audio_uuid}")
            # Even with no transcript, signal completion to unblock memory processing
            if self._current_audio_uuid:
                coordinator = get_transcript_coordinator()
                coordinator.signal_transcript_ready(self._current_audio_uuid)
                logger.info(f"⚠️ Signaled transcript completion (no data) for {self._current_audio_uuid}")
            return

        start_time = time.time()

        try:
            # Store raw transcript data
            provider_name = (
                self.online_provider.name if self.use_online_transcription else "offline_asr"
            )
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
                logger.warning(f"⚠️ Signaled transcript completion (empty text) for {self._current_audio_uuid}")
                return

            # Get speaker diarization with word matching (if available)
            final_segments = []
            if self.speaker_client.enabled and self._current_audio_uuid and self.chunk_repo:
                try:
                    # Get audio file path from database
                    chunk_data = await self.chunk_repo.get_chunk(self._current_audio_uuid)
                    if chunk_data and "audio_path" in chunk_data:
                        audio_path = chunk_data["audio_path"]
                        full_audio_path = f"/app/audio_chunks/{audio_path}"

                        logger.info(f"🎤 Getting speaker diarization with word matching for: {full_audio_path}")

                        # Prepare transcript data for speaker service
                        transcript_data = {
                            "words": normalized_result.get("words", []),
                            "text": normalized_result.get("text", "")
                        }

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
                    f"📝 No diarization available - storing raw transcript without segments for {self._current_audio_uuid}"
                )

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
                    await self.chunk_repo.add_transcript_segment(self._current_audio_uuid, segment_to_store)

                # Add speakers if we have them
                speakers_found = set()
                for segment in final_segments:
                    if segment.get("speaker"):
                        speakers_found.add(segment["speaker"])

                for speaker in speakers_found:
                    await self.chunk_repo.add_speaker(self._current_audio_uuid, speaker)

            # Update client state
            current_client = self._get_current_client()
            if current_client:
                current_client.update_transcript_received()

            # Signal transcript coordinator
            coordinator = get_transcript_coordinator()
            coordinator.signal_transcript_ready(self._current_audio_uuid)

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

    async def _queue_diarization_based_cropping(self, segments):
        """Queue audio cropping based on diarization segments."""
        try:
            # Import here to avoid circular imports
            from advanced_omi_backend.processors import AudioCroppingItem, get_processor_manager
            
            # Get current client for user info
            current_client = self._get_current_client()
            if not current_client:
                logger.warning(f"No client state available for cropping {self._current_audio_uuid}")
                return
                
            # Get audio file path from database
            if not self.chunk_repo:
                logger.warning(f"No chunk repository available for cropping {self._current_audio_uuid}")
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
                logger.debug(f"No valid cropping segments from diarization for {self._current_audio_uuid}")
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
            logger.error(f"Error queuing diarization-based cropping for {self._current_audio_uuid}: {e}")

    async def disconnect(self):
        """Cleanly disconnect from ASR service."""
        logger.info(
            f"🔌 disconnect called for client {self._client_id} - online: {self.use_online_transcription}"
        )

        if self.use_online_transcription:
            # Cancel collection task first to prevent interference
            if self._collection_task and not self._collection_task.done():
                logger.info(f"🛑 Cancelling collection timeout task for client {self._client_id}")
                self._collection_task.cancel()
                try:
                    await self._collection_task
                except asyncio.CancelledError:
                    logger.info(f"✅ Collection task cancelled successfully")
                except Exception as e:
                    logger.error(f"❌ Error cancelling collection task: {e}")

            # For batch processing, process any remaining audio
            if self._collecting or self._audio_buffer:
                logger.info(
                    f"📊 Processing remaining audio on disconnect - buffer size: {len(self._audio_buffer)}"
                )
                await self.process_collected_audio()

            logger.info(
                f"{self.online_provider.name if self.online_provider else 'Online'} batch transcription disconnected for client {self._client_id}"
            )
            return

        # Stop the background event reader task
        if self._event_reader_task:
            self._stop_event.set()
            try:
                await asyncio.wait_for(self._event_reader_task, timeout=2.0)
                logger.debug("Event reader task completed gracefully")
            except asyncio.TimeoutError:
                logger.warning("Event reader task did not stop gracefully, cancelling")
                self._event_reader_task.cancel()
                try:
                    await self._event_reader_task
                except asyncio.CancelledError:
                    logger.debug("Event reader task cancelled successfully")
            except Exception as e:
                logger.error(f"Error stopping event reader task: {e}")
                self._event_reader_task.cancel()
            finally:
                self._event_reader_task = None

        if self.client:
            try:
                await self.client.disconnect()
                logger.info("Disconnected from offline ASR service")
            except Exception as e:
                logger.error(f"Error disconnecting from offline ASR service: {e}")
            finally:
                self.client = None

    async def _read_events_continuously(self):
        """Background task that continuously reads events from ASR and puts them in queue."""
        logger.info("Started background ASR event reader task")
        try:
            while not self._stop_event.is_set() and self.client:
                try:
                    # Read events without timeout - this maximizes streaming bandwidth
                    event = await self.client.read_event()
                    if event is None:
                        break

                    # Put event in queue for processing
                    await self._event_queue.put(event)

                except Exception as e:
                    if not self._stop_event.is_set():
                        logger.error(f"Error reading ASR event: {e}")
                        # Brief pause before retry to avoid tight error loop
                        await asyncio.sleep(0.1)
                    break
        except asyncio.CancelledError:
            logger.info("Background ASR event reader task cancelled")
        finally:
            logger.info("Background ASR event reader task stopped")

    async def _process_events_from_queue(self, audio_uuid: str, client_id: str):
        """Process any available events from the queue (non-blocking)."""
        try:
            while True:
                try:
                    # Get events from queue without blocking
                    event = self._event_queue.get_nowait()
                    await self._process_asr_event(event, audio_uuid, client_id)
                except asyncio.QueueEmpty:
                    # No more events available, return
                    break
        except Exception as e:
            logger.error(f"Error processing events from queue: {e}")

    async def _process_asr_event(self, event, audio_uuid: str, client_id: str):
        """Process a single ASR event."""
        logger.info(f"🎤 Received ASR event type: {event.type} for {audio_uuid}")

        if Transcript.is_type(event.type):
            transcript_obj = Transcript.from_event(event)
            transcript_text = transcript_obj.text.strip()

            # Handle both Transcript and StreamingTranscript types
            # Check the 'final' attribute from the event data, not the reconstructed object
            is_final = event.data.get("final", True)  # Default to True for standard Transcript

            # Only process final transcripts, ignore partial ones
            if not is_final:
                logger.info(f"Ignoring partial transcript for {audio_uuid}: {transcript_text}")
                return

            if transcript_text:
                logger.info(f"Transcript for {audio_uuid}: {transcript_text} (final: {is_final})")

                # Track successful transcription
                # Note: Transaction tracking requires user_id which isn't available here
                # Individual transcription success tracked in main processing pipeline

                # For offline ASR, we no longer process individual segments here
                # They get collected and processed through the unified flow
                logger.info(
                    f"📝 Received offline transcript segment for {audio_uuid}: {transcript_text}"
                )
                # The text will be collected by _get_offline_transcript() method

                # Update transcript time for conversation timeout tracking
                current_client = self.client_manager.get_client(client_id)
                if current_client:
                    current_client.update_transcript_received()
                    logger.info(
                        f"Updated transcript timestamp for conversation: '{transcript_text}'"
                    )

                # Signal transcript coordinator that transcript is ready
                coordinator = get_transcript_coordinator()
                coordinator.signal_transcript_ready(audio_uuid)

        elif VoiceStarted.is_type(event.type):
            logger.info(f"VoiceStarted event received for {audio_uuid}")
            current_time = time.time()
            current_client = self.client_manager.get_client(client_id)
            if current_client:
                current_client.record_speech_start(audio_uuid, current_time)
                logger.info(f"🎤 Voice started for {audio_uuid} at {current_time}")
            else:
                logger.warning(
                    f"Client {client_id} not found in active_clients for VoiceStarted event"
                )

        elif VoiceStopped.is_type(event.type):
            logger.info(f"VoiceStopped event received for {audio_uuid}")
            current_time = time.time()
            current_client = self.client_manager.get_client(client_id)
            if current_client:
                current_client.record_speech_end(audio_uuid, current_time)
                logger.info(f"🔇 Voice stopped for {audio_uuid} at {current_time}")
            else:
                logger.warning(
                    f"Client {client_id} not found in active_clients for VoiceStopped event"
                )

    async def _collection_timeout_handler(self):
        """Handle collection timeout - process audio after 1.5 minutes."""
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

    # Note: The Deepgram-specific implementation has been moved to transcription_providers.py
    # This allows for a cleaner provider-based architecture supporting multiple ASR services

    async def transcribe_chunk(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Collect audio chunk for batch processing or transcribe using offline ASR."""
        if self.use_online_transcription:
            await self._collect_audio_chunk(audio_uuid, chunk, client_id)
        else:
            await self._transcribe_chunk_offline(audio_uuid, chunk, client_id)

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
                    f"🆕 New audio_uuid for {self.online_provider.name if self.online_provider else 'online'} batch: {audio_uuid}"
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

    async def _transcribe_chunk_offline(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Transcribe using offline ASR service."""
        if not self.client:
            logger.error(f"No ASR connection available for {audio_uuid}")
            # Track transcription failure handled by main pipeline
            return

        # Track transcription request
        start_time = time.time()
        # Note: Transcription requests tracked by main pipeline

        try:
            if self._current_audio_uuid != audio_uuid:
                self._current_audio_uuid = audio_uuid
                logger.info(f"New audio_uuid: {audio_uuid}")
                transcribe = Transcribe()
                await self.client.write_event(transcribe.event())
                audio_start = AudioStart(
                    rate=chunk.rate,
                    width=chunk.width,
                    channels=chunk.channels,
                    timestamp=chunk.timestamp,
                )
                await self.client.write_event(audio_start.event())

            # Send the audio chunk
            logger.debug(f"🎵 Sending {len(chunk.audio)} bytes audio chunk to ASR for {audio_uuid}")
            await self.client.write_event(chunk.event())

            # Process any available events from the background queue (non-blocking)
            await self._process_events_from_queue(audio_uuid, client_id)

        except Exception as e:
            logger.error(f"Error in offline transcribe_chunk for {audio_uuid}: {e}")
            # Track transcription failure handled by main pipeline
            # Attempt to reconnect on error
            await self._reconnect()

    async def _reconnect(self):
        """Attempt to reconnect to ASR service."""
        if self.use_online_transcription:
            # For batch processing, no reconnection needed
            logger.info(
                f"{self.online_provider.name if self.online_provider else 'Online'} batch processing - no reconnection required"
            )
            return

        logger.info("Attempting to reconnect to ASR service...")

        await self.disconnect()
        await asyncio.sleep(2)  # Brief delay before reconnecting
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
