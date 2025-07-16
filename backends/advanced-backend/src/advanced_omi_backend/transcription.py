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
from advanced_omi_backend.debug_system_tracker import PipelineStage, get_debug_tracker
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
    def __init__(self, chunk_repo=None):
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

        # Event-driven ASR event handling for offline ASR
        self._event_queue = asyncio.Queue()
        self._event_reader_task = None
        self._stop_event = asyncio.Event()
        self._client_id = None

        # Collection state tracking
        self._collecting = False
        self._collection_task = None

    def _get_current_client(self):
        """Get the current client state using ClientManager."""
        if not self._client_id:
            return None
        return self.client_manager.get_client(self._client_id)

    def _get_or_create_transaction(self, user_id: str, client_id: str, audio_uuid: str) -> str:
        """Get or create a debug transaction for tracking transcription progress."""
        if not self._current_transaction_id:
            debug_tracker = get_debug_tracker()
            self._current_transaction_id = debug_tracker.create_transaction(
                user_id=user_id, client_id=client_id, conversation_id=audio_uuid
            )
        return self._current_transaction_id

    def _track_transcription_event(
        self, stage: PipelineStage, success: bool = True, error_message: str = None, **metadata
    ):
        """Track a transcription event using the debug tracker."""
        if self._current_transaction_id:
            debug_tracker = get_debug_tracker()
            debug_tracker.track_event(
                self._current_transaction_id, stage, success, error_message, **metadata
            )

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

    async def flush_final_transcript(self, audio_duration_seconds: Optional[float] = None):
        """Process collected audio and generate final transcript."""
        if self.use_online_transcription:
            await self._process_collected_audio()
        else:
            await self._flush_offline_asr(audio_duration_seconds)

    async def _process_collected_audio(self):
        """Process all collected audio chunks using Deepgram file upload API."""
        if not self._audio_buffer:
            logger.info(f"No audio data collected for client {self._client_id}")
            return

        try:
            logger.info(
                f"Processing {len(self._audio_buffer)} audio chunks for client {self._client_id}"
            )

            # Combine all audio chunks into a single buffer
            combined_audio = b"".join(chunk.audio for chunk in self._audio_buffer if chunk.audio)

            if not combined_audio:
                logger.warning(f"No valid audio data found for client {self._client_id}")
                return

            # Send to online provider for transcription
            if self.online_provider is None:
                logger.error("Online provider is None, this shouldn't happen")
                return
            transcript_text = await self.online_provider.transcribe(combined_audio)

            if transcript_text and self._current_audio_uuid:
                logger.info(
                    f"{self.online_provider.name} batch transcript for {self._current_audio_uuid}: {transcript_text}"
                )

                # Create transcript segment
                transcript_segment = {
                    "speaker": f"speaker_{self._client_id}",
                    "text": transcript_text,
                    "start": 0.0,
                    "end": 0.0,
                }

                # Store in database
                if self.chunk_repo:
                    await self.chunk_repo.add_transcript_segment(
                        self._current_audio_uuid, transcript_segment
                    )
                    await self.chunk_repo.add_speaker(
                        self._current_audio_uuid, f"speaker_{self._client_id}"
                    )

                # Update client state
                current_client = self._get_current_client()
                if current_client:
                    current_client.last_transcript_time = time.time()
                    current_client.conversation_transcripts.append(transcript_text)

                logger.info(
                    f"Added {self.online_provider.name} batch transcript for {self._current_audio_uuid} to DB"
                )

        except Exception as e:
            logger.error(f"Error processing collected audio: {e}")
        finally:
            # Clear the buffer
            self._audio_buffer.clear()
            self._audio_start_time = None
            self._collecting = False

    async def _flush_offline_asr(self, audio_duration_seconds: Optional[float] = None):
        """Flush final transcript from offline ASR by sending AudioStop."""
        if self.client and self._current_audio_uuid:
            try:
                logger.info(
                    f"🏁 Flushing final transcript from offline ASR for audio {self._current_audio_uuid}"
                )
                # Send AudioStop to signal end of audio stream
                audio_stop = AudioStop(timestamp=int(time.time()))
                await self.client.write_event(audio_stop.event())

                # Calculate proportional timeout: 5 seconds per 30 seconds of audio
                # Ratio: 5/30 = 1/6 ≈ 0.167
                if audio_duration_seconds:
                    proportional_timeout = audio_duration_seconds / 6.0
                    # Set reasonable bounds: minimum 3 seconds, maximum 60 seconds
                    max_wait = max(3.0, min(60.0, proportional_timeout))
                    logger.info(
                        f"🏁 Calculated timeout: {max_wait:.1f}s for {audio_duration_seconds:.1f}s of audio"
                    )
                else:
                    max_wait = 5.0  # Default fallback
                    logger.info("🏁 Using default timeout: 5.0s (no audio duration provided)")

                start_time = time.time()

                # Wait for events from the background queue instead of direct reading
                # This avoids conflicts with the background event reader task
                while (time.time() - start_time) < max_wait:
                    try:
                        # Try to get event from queue with a short timeout
                        event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)

                        logger.info(f"🏁 Final flush - received event type: {event.type}")
                        if Transcript.is_type(event.type):
                            transcript_obj = Transcript.from_event(event)
                            transcript_text = transcript_obj.text.strip()
                            if transcript_text:
                                logger.info(f"🏁 Final transcript: {transcript_text}")

                                # Process final transcript the same way
                                transcript_segment = {
                                    "speaker": f"speaker_{self._client_id}",
                                    "text": transcript_text,
                                    "start": 0.0,
                                    "end": 0.0,
                                }

                                if self.chunk_repo:
                                    await self.chunk_repo.add_transcript_segment(
                                        self._current_audio_uuid, transcript_segment
                                    )

                                # Update client state
                                current_client = self._get_current_client()
                                if current_client:
                                    current_client.conversation_transcripts.append(transcript_text)
                                    logger.info(f"🏁 Added final transcript to conversation")

                    except asyncio.TimeoutError:
                        # No more events available
                        break

                logger.info(f"🏁 Finished flushing ASR for {self._current_audio_uuid}")
            except Exception as e:
                logger.error(f"Error flushing offline ASR transcript: {e}")

    async def disconnect(self):
        """Cleanly disconnect from ASR service."""
        if self.use_online_transcription:
            # For batch processing, just process any remaining audio
            if self._collecting or self._audio_buffer:
                await self._process_collected_audio()

            # Cancel collection task if running
            if self._collection_task and not self._collection_task.done():
                self._collection_task.cancel()
                try:
                    await self._collection_task
                except asyncio.CancelledError:
                    pass

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

                # Create transcript segment with new format
                transcript_segment = {
                    "speaker": f"speaker_{client_id}",
                    "text": transcript_text,
                    "start": 0.0,
                    "end": 0.0,
                }

                # Store transcript segment in DB immediately
                if self.chunk_repo:
                    await self.chunk_repo.add_transcript_segment(audio_uuid, transcript_segment)
                    await self.chunk_repo.add_speaker(audio_uuid, f"speaker_{client_id}")
                    logger.info(f"📝 Added transcript segment for {audio_uuid} to DB.")

                # Update transcript time for conversation timeout tracking
                current_client = self.client_manager.get_client(client_id)
                if current_client:
                    current_client.last_transcript_time = time.time()
                    # Collect transcript for end-of-conversation memory processing
                    current_client.conversation_transcripts.append(transcript_text)
                    logger.info(f"Added transcript to conversation collection: '{transcript_text}'")

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
        try:
            await asyncio.sleep(self._max_collection_time)
            if self._collecting and self._audio_buffer:
                logger.info(
                    f"Collection timeout reached for client {self._client_id}, processing audio"
                )
                await self._process_collected_audio()
        except asyncio.CancelledError:
            logger.debug(f"Collection timeout cancelled for client {self._client_id}")
        except Exception as e:
            logger.error(f"Error in collection timeout handler: {e}")

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
        try:
            # Update current audio UUID
            if self._current_audio_uuid != audio_uuid:
                self._current_audio_uuid = audio_uuid
                logger.info(
                    f"New audio_uuid for {self.online_provider.name if self.online_provider else 'online'} batch: {audio_uuid}"
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
                self._audio_buffer.append(chunk)
                logger.debug(
                    f"Collected {len(chunk.audio)} bytes for {audio_uuid} (total chunks: {len(self._audio_buffer)})"
                )
            else:
                logger.warning(f"Empty audio chunk received for {audio_uuid}")

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
