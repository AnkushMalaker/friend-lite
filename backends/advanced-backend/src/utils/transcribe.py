import asyncio
import logging
import os
import time
from functools import partial
from typing import Optional, Tuple

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart
from wyoming.client import AsyncTcpClient
from wyoming.vad import VoiceStarted, VoiceStopped

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Conditional Deepgram import
try:
    from deepgram import DeepgramClient, FileSource, PrerecordedOptions  # type: ignore

    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False

logger = logging.getLogger("advanced-backend")
audio_logger = logging.getLogger("audio_processing")

# Configuration values (copied from main.py)
OFFLINE_ASR_TCP_URI = os.getenv("OFFLINE_ASR_TCP_URI", "tcp://192.168.0.110:8765/")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

USE_DEEPGRAM = bool(DEEPGRAM_API_KEY and DEEPGRAM_AVAILABLE)
if DEEPGRAM_API_KEY and not DEEPGRAM_AVAILABLE:
    audio_logger.error(
        "DEEPGRAM_API_KEY provided but Deepgram SDK not available. Falling back to offline ASR."
    )
audio_logger.info(
    f"Transcription strategy: {'Deepgram' if USE_DEEPGRAM else 'Offline ASR'}"
)

deepgram_client = None
if USE_DEEPGRAM:
    audio_logger.warning(
        "Deepgram transcription requested but not yet implemented. Falling back to offline ASR."
    )
    USE_DEEPGRAM = False

from conversation_manager import record_speech_start, record_speech_end

class TranscriptionManager:
    """Manages transcription using either Deepgram or offline ASR service."""

    def __init__(self, action_item_callback=None, audio_chunk_utils=None, metrics_collector=None, active_clients=None):
        self.client = None
        self._current_audio_uuid = None
        self._streaming = False
        self.use_deepgram = USE_DEEPGRAM
        self.deepgram_client = deepgram_client
        self._audio_buffer = []  # Buffer for Deepgram batch processing
        self.audio_chunk_utils = audio_chunk_utils
        self.action_item_callback = action_item_callback  # Callback to queue action items
        self.metrics_collector = metrics_collector
        self.active_clients = active_clients

    async def connect(self):
        """Establish connection to ASR service (only for offline ASR)."""
        
        if self.use_deepgram:
            try:
                deepgram_options = DeepgramClientOptions(options={"keepalive": "true", "termination_exception_connect": "true"})
                self.client = DeepgramClient(os.getenv('DEEPGRAM_API_KEY'), deepgram_options)
                connection = await self.client.listen.websocket.v()
            except Exception as e:
                audio_logger.error(f"Failed to connect to Deepgram: {e}")
                self.client = None
                raise
            audio_logger.info("Using Deepgram transcription - no connection needed")
            

        try:
            self.client = AsyncTcpClient.from_uri(OFFLINE_ASR_TCP_URI)
            await self.client.connect()
            audio_logger.info(
                f"Connected to offline ASR service at {OFFLINE_ASR_TCP_URI}"
            )
        except Exception as e:
            audio_logger.error(f"Failed to connect to offline ASR service: {e}")
            self.client = None
            raise

    async def disconnect(self):
        """Cleanly disconnect from ASR service."""
        if self.use_deepgram:
            audio_logger.info("Using Deepgram - no disconnection needed")
            return

        if self.client:
            try:
                await self.client.disconnect()
                audio_logger.info("Disconnected from offline ASR service")
            except Exception as e:
                audio_logger.error(f"Error disconnecting from offline ASR service: {e}")
            finally:
                self.client = None

    async def transcribe_chunk(
        self, audio_uuid: str, chunk: AudioChunk, client_id: str
    ):
        """Transcribe a single chunk using either Deepgram or offline ASR."""
        if self.use_deepgram:
            await self._transcribe_chunk_deepgram(audio_uuid, chunk, client_id)
        else:
            await self._transcribe_chunk_offline(audio_uuid, chunk, client_id)

    async def _transcribe_chunk_deepgram(
        self, audio_uuid: str, chunk: AudioChunk, client_id: str
    ):
        """Transcribe using Deepgram API."""
        raise NotImplementedError(
            "Deepgram transcription is not yet implemented. Please use offline ASR by not setting DEEPGRAM_API_KEY."
        )

    async def _process_deepgram_buffer(self, audio_uuid: str, client_id: str):
        """Process buffered audio with Deepgram."""
        raise NotImplementedError("Deepgram transcription is not yet implemented.")

    async def _transcribe_chunk_offline(
        self, audio_uuid: str, chunk: AudioChunk, client_id: str
    ):
        """Transcribe using offline ASR service."""
        if not self.client:
            audio_logger.error(f"No ASR connection available for {audio_uuid}")
            # Track transcription failure
            self.metrics_collector.record_transcription_result(False)
            return

        # Track transcription request
        start_time = time.time()
        self.metrics_collector.record_transcription_request()

        try:
            if self._current_audio_uuid != audio_uuid:
                self._current_audio_uuid = audio_uuid
                audio_logger.info(f"New audio_uuid: {audio_uuid}")
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
            await self.client.write_event(chunk.event())

            # Read and process any available events (non-blocking)
            try:
                while True:
                    event = await asyncio.wait_for(
                        self.client.read_event(), timeout=0.001
                    )  # this is a quick poll, feels like a better solution can exist
                    if event is None:
                        break

                    if Transcript.is_type(event.type):
                        transcript_obj = Transcript.from_event(event)
                        transcript_text = transcript_obj.text.strip()

                        # Handle both Transcript and StreamingTranscript types
                        # Check the 'final' attribute from the event data, not the reconstructed object
                        is_final = event.data.get(
                            "final", True
                        )  # Default to True for standard Transcript

                        # Only process final transcripts, ignore partial ones
                        if not is_final:
                            audio_logger.info(
                                f"Ignoring partial transcript for {audio_uuid}: {transcript_text}"
                            )
                            continue

                        if transcript_text:
                            audio_logger.info(
                                f"Transcript for {audio_uuid}: {transcript_text} (final: {is_final})"
                            )

                            # Track successful transcription with latency
                            latency_ms = (time.time() - start_time) * 1000
                            self.metrics_collector.record_transcription_result(
                                True, latency_ms
                            )

                            # Create transcript segment with new format
                            transcript_segment = {
                                "speaker": f"speaker_{client_id}",
                                "text": transcript_text,
                                "start": 0.0,
                                "end": 0.0,
                            }

                            # Store transcript segment in DB immediately

                            await self.audio_chunk_utils.chunk_repo.add_transcript_segment(audio_uuid, transcript_segment)

                            # Queue for action item processing using callback (async, non-blocking)
                            if self.action_item_callback:
                                await self.action_item_callback(transcript_text, client_id, audio_uuid)

                            await self.audio_chunk_utils.chunk_repo.add_speaker(audio_uuid, f"speaker_{client_id}")
                            audio_logger.info(f"Added transcript segment for {audio_uuid} to DB.")
                            
                            # Update transcript time for conversation timeout tracking
                            if client_id in self.active_clients:
                                self.active_clients[client_id].last_transcript_time = (
                                    time.time()
                                )
                                # Collect transcript for end-of-conversation memory processing
                                self.active_clients[
                                    client_id
                                ].conversation_transcripts.append(transcript_text)
                                audio_logger.info(
                                    f"Added transcript to conversation collection: {GREEN}'{transcript_text}'{RESET}"
                                )

                    elif VoiceStarted.is_type(event.type):
                        audio_logger.info(
                            f"VoiceStarted event received for {audio_uuid}"
                        )
                        current_time = time.time()
                        if client_id in self.active_clients:
                            record_speech_start(
                                self.active_clients[client_id], audio_uuid, current_time
                            )
                            audio_logger.info(
                                f"🎤 Voice started for {audio_uuid} at {current_time}"
                            )

                    elif VoiceStopped.is_type(event.type):
                        audio_logger.info(
                            f"VoiceStopped event received for {audio_uuid}"
                        )
                        current_time = time.time()
                        if client_id in self.active_clients:
                            record_speech_end(
                                self.active_clients[client_id], audio_uuid, current_time
                            )
                            audio_logger.info(
                                f"🔇 Voice stopped for {audio_uuid} at {current_time}"
                            )

            except asyncio.TimeoutError:
                # No events available right now, that's fine
                pass

        except Exception as e:
            audio_logger.error(
                f"Error in offline transcribe_chunk for {audio_uuid}: {e}"
            )
            # Track transcription failure
            self.metrics_collector.record_transcription_result(False)
            # Attempt to reconnect on error
            await self._reconnect()

    async def _reconnect(self):
        """Attempt to reconnect to ASR service."""
        audio_logger.info("Attempting to reconnect to ASR service...")

        # Track reconnection attempt
        self.metrics_collector.record_service_reconnection("asr-service")

        await self.disconnect()
        await asyncio.sleep(2)  # Brief delay before reconnecting
        try:
            await self.connect()
        except Exception as e:
            audio_logger.error(f"Reconnection failed: {e}")