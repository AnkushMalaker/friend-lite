#!/usr/bin/env python3
"""Unified Omi-audio service

* Accepts Opus packets over a WebSocket (`/ws`) or PCM over a WebSocket (`/ws_pcm`).
* Uses a central queue to decouple audio ingestion from processing.
* A saver consumer buffers PCM and writes 30-second WAV chunks to `./audio_chunks/`.
* A transcription consumer sends each chunk to a Wyoming ASR service.
* The transcript is stored in **mem0** and MongoDB.

"""
import logging
logging.basicConfig(level=logging.INFO)

import asyncio
import concurrent.futures
import json
import os
import time
import uuid
import wave
import io
import numpy as np
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Any, List
from bson import ObjectId

# Import Beanie for user management
from beanie import init_beanie
import ollama
import websockets
from dotenv import load_dotenv
from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from fastapi import Depends, FastAPI, Query, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
from omi.decoder import OmiOpusDecoder
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.vad import VoiceStarted, VoiceStopped

from action_items_service import ActionItemsService

# Import authentication components
from auth import (
    bearer_backend,
    cookie_backend,
    create_admin_user_if_needed,
    current_active_user,
    current_superuser,
    fastapi_users,
    get_user_manager,
    websocket_auth,
    ADMIN_EMAIL
)

from memory import get_memory_service, init_memory_config, shutdown_memory_service
from metrics import (
    get_metrics_collector,
    start_metrics_collection,
    stop_metrics_collection,
)
from users import User, UserCreate, UserRead, get_user_db, generate_client_id, get_user_by_client_id, register_client_to_user

# Import failure recovery system
from failure_recovery import (
    init_failure_recovery_system,
    shutdown_failure_recovery_system,
    perform_startup_recovery,
    get_failure_recovery_router,
    get_queue_tracker,
    get_persistent_queue,
    QueueType,
    QueueStatus,
    QueueItem,
    MessagePriority
)

###############################################################################
# SETUP
###############################################################################

# Load environment variables first
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advanced-backend")
audio_logger = logging.getLogger("audio_processing")

# Conditional Deepgram import
try:
    from deepgram import DeepgramClient, FileSource, PrerecordedOptions  # type: ignore

    DEEPGRAM_AVAILABLE = True
    logger.info("✅ Deepgram SDK available")
except ImportError:
    DEEPGRAM_AVAILABLE = False
    logger.warning("Deepgram SDK not available. Install with: pip install deepgram-sdk")
audio_cropper_logger = logging.getLogger("audio_cropper")


###############################################################################
# CONFIGURATION
###############################################################################

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client.get_default_database("friend-lite")
chunks_col = db["audio_chunks"]
users_col = db["users"]
speakers_col = db["speakers"]  # New collection for speaker management
action_items_col = db["action_items"]  # New collection for action items

# Audio Configuration
OMI_SAMPLE_RATE = 16_000  # Hz
OMI_CHANNELS = 1
OMI_SAMPLE_WIDTH = 2  # bytes (16‑bit)
SEGMENT_SECONDS = 60  # length of each stored chunk
TARGET_SAMPLES = OMI_SAMPLE_RATE * SEGMENT_SECONDS

# Conversation timeout configuration
NEW_CONVERSATION_TIMEOUT_MINUTES = float(os.getenv("NEW_CONVERSATION_TIMEOUT_MINUTES", "1.5"))

# Audio cropping configuration
AUDIO_CROPPING_ENABLED = os.getenv("AUDIO_CROPPING_ENABLED", "true").lower() == "true"
MIN_SPEECH_SEGMENT_DURATION = float(os.getenv("MIN_SPEECH_SEGMENT_DURATION", "1.0"))  # seconds
CROPPING_CONTEXT_PADDING = float(
    os.getenv("CROPPING_CONTEXT_PADDING", "0.1")
)  # seconds of padding around speech

# Directory where WAV chunks are written
CHUNK_DIR = Path("./audio_chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# ASR Configuration
OFFLINE_ASR_TCP_URI = os.getenv("OFFLINE_ASR_TCP_URI", "tcp://192.168.0.110:8765/")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Determine transcription strategy based on environment variables
# For WebSocket implementation, we don't need the Deepgram SDK
USE_DEEPGRAM = bool(DEEPGRAM_API_KEY)
if DEEPGRAM_API_KEY and not DEEPGRAM_AVAILABLE:
    audio_logger.info(
        "DEEPGRAM_API_KEY provided. Using WebSocket implementation (Deepgram SDK not required)."
    )

audio_logger.info(
    f"Transcription strategy: {'Deepgram WebSocket' if USE_DEEPGRAM else 'Offline ASR'}"
)

# Deepgram client placeholder (not needed for WebSocket implementation)
deepgram_client = None

# Ollama & Qdrant Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", "qdrant")

# Memory configuration is now handled in the memory module
# Initialize it with our Ollama and Qdrant URLs
init_memory_config(
    ollama_base_url=OLLAMA_BASE_URL,
    qdrant_base_url=QDRANT_BASE_URL,
)

# Speaker service configuration

# Thread pool executors
_DEC_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=os.cpu_count() or 4,
    thread_name_prefix="opus_io",
)

# Initialize memory service, speaker service, and ollama client
memory_service = get_memory_service()
ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

action_items_service = ActionItemsService(action_items_col, ollama_client)

###############################################################################
# AUDIO PROCESSING FUNCTIONS
###############################################################################


async def _process_audio_cropping_with_relative_timestamps(
    original_path: str,
    speech_segments: list[tuple[float, float]],
    output_path: str,
    audio_uuid: str,
) -> bool:
    """
    Process audio cropping with automatic relative timestamp conversion.
    This function handles both live processing and reprocessing scenarios.
    """
    try:
        # Convert absolute timestamps to relative timestamps
        # Extract file start time from filename: timestamp_client_uuid.wav
        filename = original_path.split("/")[-1]
        audio_logger.info(f"🕐 Parsing filename: {filename}")
        filename_parts = filename.split("_")
        if len(filename_parts) < 3:
            audio_logger.error(f"Invalid filename format: {filename}. Expected format: timestamp_client_id_audio_uuid.wav")
            return False
        
        try:
            file_start_timestamp = float(filename_parts[0])
        except ValueError as e:
            audio_logger.error(f"Cannot parse timestamp from filename {filename}: {e}")
            return False

        # Convert speech segments to relative timestamps
        relative_segments = []
        for start_abs, end_abs in speech_segments:
            # Validate input timestamps
            if start_abs >= end_abs:
                audio_logger.warning(f"⚠️ Invalid speech segment: start={start_abs} >= end={end_abs}, skipping")
                continue
                
            start_rel = start_abs - file_start_timestamp
            end_rel = end_abs - file_start_timestamp

            # Ensure relative timestamps are positive (sanity check)
            if start_rel < 0:
                audio_logger.warning(f"⚠️ Negative start timestamp: {start_rel} (absolute: {start_abs}, file_start: {file_start_timestamp}), clamping to 0.0")
                start_rel = 0.0
            if end_rel < 0:
                audio_logger.warning(f"⚠️ Negative end timestamp: {end_rel} (absolute: {end_abs}, file_start: {file_start_timestamp}), skipping segment")
                continue

            relative_segments.append((start_rel, end_rel))

        audio_logger.info(
            f"🕐 Converting timestamps for {audio_uuid}: file_start={file_start_timestamp}"
        )
        audio_logger.info(f"🕐 Absolute segments: {speech_segments}")
        audio_logger.info(f"🕐 Relative segments: {relative_segments}")

        # Validate that we have valid relative segments after conversion
        if not relative_segments:
            audio_logger.warning(f"No valid relative segments after timestamp conversion for {audio_uuid}")
            return False

        success = await _crop_audio_with_ffmpeg(original_path, relative_segments, output_path)
        if success:
            # Update database with cropped file info (keep original absolute timestamps for reference)
            cropped_filename = output_path.split("/")[-1]
            await chunk_repo.update_cropped_audio(audio_uuid, cropped_filename, speech_segments)
            audio_logger.info(f"Successfully processed cropped audio: {cropped_filename}")
            return True
        else:
            audio_logger.error(f"Failed to crop audio for {audio_uuid}")
            return False
    except Exception as e:
        audio_logger.error(f"Error in audio cropping task for {audio_uuid}: {e}", exc_info=True)
        return False


async def _crop_audio_with_ffmpeg(
    original_path: str, speech_segments: list[tuple[float, float]], output_path: str
) -> bool:
    """Use ffmpeg to crop audio - runs as async subprocess, no GIL issues"""
    audio_cropper_logger.info(
        f"Cropping audio {original_path} with {len(speech_segments)} speech segments"
    )

    if not AUDIO_CROPPING_ENABLED:
        audio_cropper_logger.info(f"Audio cropping disabled, skipping {original_path}")
        return False

    if not speech_segments:
        audio_cropper_logger.warning(f"No speech segments to crop for {original_path}")
        return False

    # Check if the original file exists
    if not os.path.exists(original_path):
        audio_cropper_logger.error(f"Original audio file does not exist: {original_path}")
        return False

    # Filter out segments that are too short
    filtered_segments = []
    for start, end in speech_segments:
        duration = end - start
        if duration >= MIN_SPEECH_SEGMENT_DURATION:
            # Add padding around speech segments
            padded_start = max(0, start - CROPPING_CONTEXT_PADDING)
            padded_end = end + CROPPING_CONTEXT_PADDING
            filtered_segments.append((padded_start, padded_end))
        else:
            audio_cropper_logger.debug(
                f"Skipping short segment: {start}-{end} ({duration:.2f}s < {MIN_SPEECH_SEGMENT_DURATION}s)"
            )

    if not filtered_segments:
        audio_cropper_logger.warning(
            f"No segments meet minimum duration ({MIN_SPEECH_SEGMENT_DURATION}s) for {original_path}"
        )
        return False

    audio_cropper_logger.info(
        f"Cropping audio {original_path} with {len(filtered_segments)} speech segments (filtered from {len(speech_segments)})"
    )

    try:
        # Build ffmpeg filter for concatenating speech segments
        filter_parts = []
        for i, (start, end) in enumerate(filtered_segments):
            duration = end - start
            filter_parts.append(
                f"[0:a]atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS[seg{i}]"
            )

        # Concatenate all segments
        inputs = "".join(f"[seg{i}]" for i in range(len(filtered_segments)))
        concat_filter = f"{inputs}concat=n={len(filtered_segments)}:v=0:a=1[out]"

        full_filter = ";".join(filter_parts + [concat_filter])

        # Run ffmpeg as async subprocess
        cmd = [
            "ffmpeg",
            "-y",  # -y = overwrite output
            "-i",
            original_path,
            "-filter_complex",
            full_filter,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",  # Keep same format as original
            output_path,
        ]

        audio_cropper_logger.info(f"Running ffmpeg command: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        if stdout:
            audio_cropper_logger.debug(f"FFMPEG stdout: {stdout.decode()}")

        if process.returncode == 0:
            # Calculate cropped duration
            cropped_duration = sum(end - start for start, end in filtered_segments)
            audio_cropper_logger.info(
                f"Successfully cropped {original_path} -> {output_path} ({cropped_duration:.1f}s from {len(filtered_segments)} segments)"
            )
            return True
        else:
            error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
            audio_logger.error(f"ffmpeg failed for {original_path}: {error_msg}")
            return False

    except Exception as e:
        audio_logger.error(f"Error running ffmpeg on {original_path}: {e}", exc_info=True)
        return False


###############################################################################
# UTILITY FUNCTIONS & HELPER CLASSES
###############################################################################


def _new_local_file_sink(file_path):
    """Create a properly configured LocalFileSink with all wave parameters set."""
    sink = LocalFileSink(
        file_path=file_path,
        sample_rate=int(OMI_SAMPLE_RATE),
        channels=int(OMI_CHANNELS),
        sample_width=int(OMI_SAMPLE_WIDTH),
    )
    return sink



class ChunkRepo:
    """Async helpers for the audio_chunks collection."""

    def __init__(self, collection):
        self.col = collection

    async def create_chunk(
        self,
        *,
        audio_uuid,
        audio_path,
        client_id,
        timestamp,
        transcript=None,
        speakers_identified=None,
    ):
        doc = {
            "audio_uuid": audio_uuid,
            "audio_path": audio_path,
            "client_id": client_id,
            "timestamp": timestamp,
            "transcript": transcript or [],  # List of conversation segments
            "speakers_identified": speakers_identified or [],  # List of identified speakers
        }
        await self.col.insert_one(doc)

    async def add_transcript_segment(self, audio_uuid, transcript_segment):
        """Add a single transcript segment to the conversation."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid}, {"$push": {"transcript": transcript_segment}}
        )

    async def add_speaker(self, audio_uuid, speaker_id):
        """Add a speaker to the speakers_identified list if not already present."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$addToSet": {"speakers_identified": speaker_id}},
        )

    async def update_transcript(self, audio_uuid, full_transcript):
        """Update the entire transcript list (for compatibility)."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid}, {"$set": {"transcript": full_transcript}}
        )

    async def update_segment_timing(self, audio_uuid, segment_index, start_time, end_time):
        """Update timing information for a specific transcript segment."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    f"transcript.{segment_index}.start": start_time,
                    f"transcript.{segment_index}.end": end_time,
                }
            },
        )

    async def update_segment_speaker(self, audio_uuid, segment_index, speaker_id):
        """Update the speaker for a specific transcript segment."""
        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$set": {f"transcript.{segment_index}.speaker": speaker_id}},
        )
        if result.modified_count > 0:
            audio_logger.info(
                f"Updated segment {segment_index} speaker to {speaker_id} for {audio_uuid}"
            )
        return result.modified_count > 0

    async def update_cropped_audio(
        self,
        audio_uuid: str,
        cropped_path: str,
        speech_segments: list[tuple[float, float]],
    ):
        """Update the chunk with cropped audio information."""
        cropped_duration = sum(end - start for start, end in speech_segments)

        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    "cropped_audio_path": cropped_path,
                    "speech_segments": [
                        {"start": start, "end": end} for start, end in speech_segments
                    ],
                    "cropped_duration": cropped_duration,
                    "cropped_at": time.time(),
                }
            },
        )
        if result.modified_count > 0:
            audio_logger.info(f"Updated cropped audio info for {audio_uuid}: {cropped_path}")
        return result.modified_count > 0


class TranscriptionManager:
    """Manages transcription using either Deepgram or offline ASR service."""

    def __init__(self, action_item_callback=None):
        self.client = None
        self._current_audio_uuid = None
        self.use_deepgram = USE_DEEPGRAM
        self.deepgram_client = deepgram_client
        self._audio_buffer = []  # Buffer for Deepgram batch processing
        self.action_item_callback = action_item_callback  # Callback to queue action items

    async def connect(self, client_id: str | None = None):
        """Establish connection to ASR service."""
        self._client_id = client_id

        if self.use_deepgram:
            await self._connect_deepgram()
            return

        try:
            self.client = AsyncTcpClient.from_uri(OFFLINE_ASR_TCP_URI)
            await self.client.connect()
            audio_logger.info(f"Connected to offline ASR service at {OFFLINE_ASR_TCP_URI}")
        except Exception as e:
            audio_logger.error(f"Failed to connect to offline ASR service: {e}")
            self.client = None
            raise

    async def flush_final_transcript(self):
        """Flush any remaining transcript from ASR by sending AudioStop."""
        if self.use_deepgram:
            await self._flush_deepgram()
        else:
            await self._flush_offline_asr()

    async def _flush_deepgram(self):
        """Flush final transcript from Deepgram by closing the stream."""
        if self.deepgram_connected and self.deepgram_ws:
            try:
                # Send close frame to signal end of audio stream
                # Deepgram will send final results when the connection closes
                audio_logger.info(f"Flushing final transcript from Deepgram for client {self._client_id}")
                await self.deepgram_ws.send(b'{"type": "CloseStream"}')
                # Give Deepgram a moment to process final audio and send results
                await asyncio.sleep(1.0)
            except Exception as e:
                audio_logger.error(f"Error flushing Deepgram transcript: {e}")

    async def _flush_offline_asr(self):
        """Flush final transcript from offline ASR by sending AudioStop."""
        if self.client and self._current_audio_uuid:
            try:
                audio_logger.info(f"🏁 Flushing final transcript from offline ASR for audio {self._current_audio_uuid}")
                # Send AudioStop to signal end of audio stream
                audio_stop = AudioStop(timestamp=int(time.time()))
                await self.client.write_event(audio_stop.event())
                
                # Wait longer for final transcripts and process any remaining events
                max_wait = 5.0  # Wait up to 5 seconds for final transcripts
                start_time = time.time()
                
                while (time.time() - start_time) < max_wait:
                    try:
                        event = await asyncio.wait_for(self.client.read_event(), timeout=0.5)
                        if event is None:
                            break
                            
                        audio_logger.info(f"🏁 Final flush - received event type: {event.type}")
                        if Transcript.is_type(event.type):
                            transcript_obj = Transcript.from_event(event)
                            transcript_text = transcript_obj.text.strip()
                            if transcript_text:
                                audio_logger.info(f"🏁 Final transcript: {transcript_text}")
                                
                                # Process final transcript the same way
                                transcript_segment = {
                                    "speaker": f"speaker_{self._client_id}",
                                    "text": transcript_text,
                                    "start": 0.0,
                                    "end": 0.0,
                                }
                                
                                await chunk_repo.add_transcript_segment(self._current_audio_uuid, transcript_segment)
                                
                                # Update client state
                                global active_clients
                                if self._client_id in active_clients:
                                    active_clients[self._client_id].conversation_transcripts.append(transcript_text)
                                    audio_logger.info(f"🏁 Added final transcript to conversation")
                                
                    except asyncio.TimeoutError:
                        # No more events available
                        break
                        
                audio_logger.info(f"🏁 Finished flushing ASR for {self._current_audio_uuid}")
            except Exception as e:
                audio_logger.error(f"Error flushing offline ASR transcript: {e}")

    async def disconnect(self):
        """Cleanly disconnect from ASR service."""
        if self.use_deepgram:
            await self._disconnect_deepgram()
            return

        if self.client:
            try:
                await self.client.disconnect()
                audio_logger.info("Disconnected from offline ASR service")
            except Exception as e:
                audio_logger.error(f"Error disconnecting from offline ASR service: {e}")
            finally:
                self.client = None

    async def _connect_deepgram(self):
        """Establish WebSocket connection to Deepgram."""
        if not DEEPGRAM_API_KEY:
            raise Exception("DEEPGRAM_API_KEY is required for Deepgram transcription")

        try:
            # Deepgram WebSocket URL with configuration parameters
            params = {
                "sample_rate": "16000",
                "encoding": "linear16",  # PCM audio
                "channels": "1",
                "model": "nova-2",
                "language": "en-US",
                "smart_format": "true",
                "interim_results": "false",
                "punctuate": "true",
                "diarize": "true",
            }

            # Build URL with parameters
            param_string = "&".join([f"{k}={v}" for k, v in params.items()])
            ws_url = f"wss://api.deepgram.com/v1/listen?{param_string}"

            # Headers for authentication
            headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

            # Connect to Deepgram WebSocket
            self.deepgram_ws = await websockets.connect(ws_url, extra_headers=headers)

            self.deepgram_connected = True
            audio_logger.info(f"Connected to Deepgram WebSocket for client {self._client_id}")

            # Start listening for responses
            asyncio.create_task(self._listen_for_deepgram_responses())

        except Exception as e:
            audio_logger.error(f"Failed to connect to Deepgram WebSocket: {e}")
            self.deepgram_connected = False
            raise

    async def _disconnect_deepgram(self):
        """Disconnect from Deepgram WebSocket."""
        self.deepgram_connected = False
        if self.deepgram_ws:
            try:
                await self.deepgram_ws.close()
                audio_logger.info(
                    f"Disconnected from Deepgram WebSocket for client {self._client_id}"
                )
            except Exception as e:
                audio_logger.error(f"Error disconnecting from Deepgram WebSocket: {e}")
            finally:
                self.deepgram_ws = None

    async def _listen_for_deepgram_responses(self):
        """Listen for responses from Deepgram WebSocket."""
        if not self.deepgram_ws:
            return

        try:
            async for message in self.deepgram_ws:
                if not self.deepgram_connected:
                    break

                try:
                    data = json.loads(message)
                    await self._handle_deepgram_response(data)
                except json.JSONDecodeError as e:
                    audio_logger.error(f"Failed to parse Deepgram response: {e}")
                except Exception as e:
                    audio_logger.error(f"Error handling Deepgram response: {e}")

        except websockets.exceptions.ConnectionClosed:
            audio_logger.info("Deepgram WebSocket connection closed")
            self.deepgram_connected = False
        except Exception as e:
            audio_logger.error(f"Error in Deepgram response listener: {e}")
            self.deepgram_connected = False

    async def _handle_deepgram_response(self, data):
        """Handle transcript response from Deepgram."""
        try:
            # Check if we have a transcript
            if data.get("channel", {}).get("alternatives", []):
                alternative = data["channel"]["alternatives"][0]
                transcript_text = alternative.get("transcript", "").strip()

                # Only process if we have actual text
                if transcript_text:
                    audio_logger.info(
                        f"Deepgram transcript for {self._current_audio_uuid}: {transcript_text}"
                    )

                    # Track successful transcription
                    metrics_collector = get_metrics_collector()
                    metrics_collector.record_transcription_result(True)

                    # Check for speaker information
                    speaker_id = f"speaker_{self._client_id}"
                    words = alternative.get("words", [])
                    if words and words[0].get("speaker") is not None:
                        speaker_id = f"speaker_{words[0]['speaker']}"

                    # Create transcript segment
                    transcript_segment = {
                        "speaker": speaker_id,
                        "text": transcript_text,
                        "start": 0.0,  # Deepgram provides timestamps but we'll use 0 for now
                        "end": 0.0,
                    }

                    # Store in database if we have a current audio UUID
                    if self._current_audio_uuid and self._client_id:
                        # We'll need to access these globals - they're defined later in the module
                        # Use globals() to access them safely
                        global chunk_repo, active_clients

                        await chunk_repo.add_transcript_segment(
                            self._current_audio_uuid, transcript_segment
                        )
                        await chunk_repo.add_speaker(self._current_audio_uuid, speaker_id)

                        # Update client state
                        if self._client_id in active_clients:
                            active_clients[self._client_id].last_transcript_time = time.time()
                            active_clients[self._client_id].conversation_transcripts.append(
                                transcript_text
                            )

                        audio_logger.info(
                            f"Added Deepgram transcript segment for {self._current_audio_uuid} to DB."
                        )

        except Exception as e:
            audio_logger.error(f"Error handling Deepgram transcript: {e}")

    async def transcribe_chunk(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Transcribe a single chunk using either Deepgram or offline ASR."""
        if self.use_deepgram:
            await self._transcribe_chunk_deepgram(audio_uuid, chunk, client_id)
        else:
            await self._transcribe_chunk_offline(audio_uuid, chunk, client_id)

    async def _transcribe_chunk_deepgram(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Transcribe using Deepgram WebSocket."""
        if not self.deepgram_connected or not self.deepgram_ws:
            audio_logger.error(f"Deepgram WebSocket not connected for {audio_uuid}")
            # Track transcription failure
            metrics_collector = get_metrics_collector()
            metrics_collector.record_transcription_result(False)
            return

        # Track transcription request
        start_time = time.time()
        metrics_collector = get_metrics_collector()
        metrics_collector.record_transcription_request()

        try:
            # Update current audio UUID for response handling
            if self._current_audio_uuid != audio_uuid:
                self._current_audio_uuid = audio_uuid
                audio_logger.info(f"New audio_uuid for Deepgram: {audio_uuid}")

            # Send audio chunk to Deepgram WebSocket as binary data
            if chunk.audio and len(chunk.audio) > 0:
                await self.deepgram_ws.send(chunk.audio)
                audio_logger.debug(f"Sent {len(chunk.audio)} bytes to Deepgram for {audio_uuid}")
            else:
                audio_logger.warning(f"Empty audio chunk received for {audio_uuid}")

        except websockets.exceptions.ConnectionClosed:
            audio_logger.error(
                f"Deepgram WebSocket connection closed unexpectedly for {audio_uuid}"
            )
            self.deepgram_connected = False
            # Track transcription failure
            metrics_collector.record_transcription_result(False)
            # Attempt to reconnect
            await self._reconnect_deepgram()
        except Exception as e:
            audio_logger.error(f"Error sending audio to Deepgram for {audio_uuid}: {e}")
            # Track transcription failure
            metrics_collector.record_transcription_result(False)
            # Attempt to reconnect on error
            await self._reconnect_deepgram()

    async def _transcribe_chunk_offline(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Transcribe using offline ASR service."""
        if not self.client:
            audio_logger.error(f"No ASR connection available for {audio_uuid}")
            # Track transcription failure
            metrics_collector = get_metrics_collector()
            metrics_collector.record_transcription_result(False)
            return

        # Track transcription request
        start_time = time.time()
        metrics_collector = get_metrics_collector()
        metrics_collector.record_transcription_request()

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
            audio_logger.info(f"🎵 Sending {len(chunk.audio)} bytes audio chunk to ASR for {audio_uuid}")
            await self.client.write_event(chunk.event())

            # Read and process any available events (non-blocking)
            try:
                while True:
                    event = await asyncio.wait_for(
                        self.client.read_event(), timeout=0.1
                    )  # Increased timeout for better ASR response handling
                    if event is None:
                        break

                    audio_logger.info(f"🎤 Received ASR event type: {event.type} for {audio_uuid}")
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
                            metrics_collector.record_transcription_result(True, latency_ms)

                            # Create transcript segment with new format
                            transcript_segment = {
                                "speaker": f"speaker_{client_id}",
                                "text": transcript_text,
                                "start": 0.0,
                                "end": 0.0,
                            }

                            # Store transcript segment in DB immediately
                            await chunk_repo.add_transcript_segment(audio_uuid, transcript_segment)

                            # Update client state with transcript for memory processing
                            global active_clients
                            if client_id in active_clients:
                                active_clients[client_id].last_transcript_time = time.time()
                                active_clients[client_id].conversation_transcripts.append(transcript_text)
                                audio_logger.info(f"✅ Added transcript to conversation: '{transcript_text}' (total: {len(active_clients[client_id].conversation_transcripts)})")
                            else:
                                audio_logger.warning(f"⚠️ Client {client_id} not found in active_clients for transcript update")

                            # Queue for action item processing using callback (async, non-blocking)
                            if self.action_item_callback:
                                await self.action_item_callback(
                                    transcript_text, client_id, audio_uuid
                                )

                            await chunk_repo.add_speaker(audio_uuid, f"speaker_{client_id}")
                            audio_logger.info(f"📝 Added transcript segment for {audio_uuid} to DB.")

                            # Update transcript time for conversation timeout tracking
                            if client_id in active_clients:
                                active_clients[client_id].last_transcript_time = time.time()
                                # Collect transcript for end-of-conversation memory processing
                                active_clients[client_id].conversation_transcripts.append(
                                    transcript_text
                                )
                                audio_logger.info(
                                    f"Added transcript to conversation collection: '{transcript_text}'"
                                )

                    elif VoiceStarted.is_type(event.type):
                        audio_logger.info(f"VoiceStarted event received for {audio_uuid}")
                        current_time = time.time()
                        if client_id in active_clients:
                            active_clients[client_id].record_speech_start(audio_uuid, current_time)
                            audio_logger.info(
                                f"🎤 Voice started for {audio_uuid} at {current_time}"
                            )
                        else:
                            audio_logger.warning(f"Client {client_id} not found in active_clients for VoiceStarted event")

                    elif VoiceStopped.is_type(event.type):
                        audio_logger.info(f"VoiceStopped event received for {audio_uuid}")
                        current_time = time.time()
                        if client_id in active_clients:
                            active_clients[client_id].record_speech_end(audio_uuid, current_time)
                            audio_logger.info(
                                f"🔇 Voice stopped for {audio_uuid} at {current_time}"
                            )
                        else:
                            audio_logger.warning(f"Client {client_id} not found in active_clients for VoiceStopped event")

            except asyncio.TimeoutError:
                # No events available right now, that's fine
                pass

        except Exception as e:
            audio_logger.error(f"Error in offline transcribe_chunk for {audio_uuid}: {e}")
            # Track transcription failure
            metrics_collector.record_transcription_result(False)
            # Attempt to reconnect on error
            await self._reconnect()

    async def _reconnect_deepgram(self):
        """Attempt to reconnect to Deepgram WebSocket."""
        audio_logger.info("Attempting to reconnect to Deepgram WebSocket...")

        # Track reconnection attempt
        metrics_collector = get_metrics_collector()
        metrics_collector.record_service_reconnection("deepgram-websocket")

        await self._disconnect_deepgram()
        await asyncio.sleep(2)  # Brief delay before reconnecting
        try:
            await self._connect_deepgram()
        except Exception as e:
            audio_logger.error(f"Deepgram reconnection failed: {e}")

    async def _reconnect(self):
        """Attempt to reconnect to ASR service."""
        if self.use_deepgram:
            await self._reconnect_deepgram()
            return

        audio_logger.info("Attempting to reconnect to ASR service...")

        # Track reconnection attempt
        metrics_collector = get_metrics_collector()
        metrics_collector.record_service_reconnection("asr-service")

        await self.disconnect()
        await asyncio.sleep(2)  # Brief delay before reconnecting
        try:
            await self.connect()
        except Exception as e:
            audio_logger.error(f"Reconnection failed: {e}")


class ClientState:
    """Manages all state for a single client connection."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.connected = True

        # Per-client queues
        self.chunk_queue = asyncio.Queue[Optional[AudioChunk]]()
        self.transcription_queue = asyncio.Queue[Tuple[Optional[str], Optional[AudioChunk]]]()
        self.memory_queue = asyncio.Queue[
            Tuple[Optional[str], Optional[str], Optional[str]]
        ]()  # (transcript, client_id, audio_uuid)
        self.action_item_queue = asyncio.Queue[
            Tuple[Optional[str], Optional[str], Optional[str]]
        ]()  # (transcript_text, client_id, audio_uuid)

        # Per-client file sink
        self.file_sink: Optional[LocalFileSink] = None
        self.current_audio_uuid: Optional[str] = None

        # Per-client transcription manager
        self.transcription_manager: Optional[TranscriptionManager] = None

        # Conversation timeout tracking
        self.last_transcript_time: Optional[float] = None
        self.conversation_start_time: float = time.time()

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
        self.action_item_task: Optional[asyncio.Task] = None

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
        self.action_item_task = asyncio.create_task(self._action_item_processor())
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
        await self.action_item_queue.put((None, None, None))

        # Wait for tasks to complete
        if self.saver_task:
            await self.saver_task
        if self.transcription_task:
            await self.transcription_task
        if self.memory_task:
            await self.memory_task
        if self.action_item_task:
            await self.action_item_task

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
        if self.file_sink:
            # Store current audio info before closing
            current_uuid = self.current_audio_uuid
            current_path = self.file_sink.file_path

            audio_logger.info(f"🔒 Closing conversation {current_uuid}, file: {current_path}")

            # Flush any remaining transcript from ASR before waiting for queue
            if self.transcription_manager:
                try:
                    audio_logger.info(f"🏁 Flushing final transcript for {current_uuid}")
                    await self.transcription_manager.flush_final_transcript()
                except Exception as e:
                    audio_logger.error(f"Error flushing final transcript for {current_uuid}: {e}")

            # Wait for transcription queue to finish with timeout to prevent hanging
            try:
                await asyncio.wait_for(self.transcription_queue.join(), timeout=60.0)  # Increased timeout for final transcript
                audio_logger.info("Transcription queue processing completed")
            except asyncio.TimeoutError:
                audio_logger.warning(f"Transcription queue join timed out after 15 seconds for {current_uuid}")
            
            # Small delay to allow final processing to complete
            await asyncio.sleep(0.5)

            # Process memory at end of conversation if we have transcripts
            if self.conversation_transcripts and current_uuid:
                full_conversation = " ".join(self.conversation_transcripts)
                audio_logger.info(
                    f"💭 Queuing memory processing for conversation {current_uuid} with {len(self.conversation_transcripts)} transcript segments"
                )
                audio_logger.info(f"💭 Individual transcripts: {self.conversation_transcripts}")
                audio_logger.info(
                    f"💭 Full conversation text: {full_conversation[:200]}..."
                )  # Log first 200 chars

                # Process memory in background to avoid blocking conversation close
                asyncio.create_task(self._process_memory_background(
                    full_conversation, current_uuid
                ))
                
                audio_logger.info(f"💭 Memory processing queued in background for {current_uuid}")
            else:
                audio_logger.info(
                    f"ℹ️ No transcripts to process for memory in conversation {current_uuid}"
                )

            await self.file_sink.close()

            # Track successful audio chunk save in metrics
            try:
                metrics_collector = get_metrics_collector()
                file_path = Path(current_path)
                if file_path.exists():
                    # Estimate duration (60 seconds per chunk is TARGET_SAMPLES)
                    duration_seconds = SEGMENT_SECONDS

                    # Calculate voice activity if we have speech segments
                    voice_activity_seconds = 0
                    if current_uuid and current_uuid in self.speech_segments:
                        for start, end in self.speech_segments[current_uuid]:
                            voice_activity_seconds += end - start

                    metrics_collector.record_audio_chunk_saved(
                        duration_seconds, voice_activity_seconds
                    )
                    audio_logger.debug(
                        f"📊 Recorded audio chunk metrics: {duration_seconds}s total, {voice_activity_seconds}s voice activity"
                    )
                else:
                    metrics_collector.record_audio_chunk_failed()
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
                                f"{CHUNK_DIR}/{current_path}",
                                speech_segments,
                                f"{CHUNK_DIR}/{cropped_path}",
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

    async def _process_memory_background(self, full_conversation: str, audio_uuid: str):
        """Background task for memory processing to avoid blocking conversation close."""
        start_time = time.time()
        
        try:
            # Track memory storage request
            metrics_collector = get_metrics_collector()
            metrics_collector.record_memory_storage_request()

            # Add general memory with fallback handling
            # First resolve client_id to user information
            user = await get_user_by_client_id(self.client_id)
            if user:
                memory_result = await memory_service.add_memory(
                    full_conversation, self.client_id, audio_uuid, user.user_id, user.email
                )
            else:
                audio_logger.error(f"Could not resolve client_id {self.client_id} to user for memory storage")
                memory_result = False
            if memory_result:
                audio_logger.info(
                    f"✅ Successfully added conversation memory for {audio_uuid}"
                )
                metrics_collector.record_memory_storage_result(True)
            else:
                audio_logger.warning(
                    f"⚠️ Memory service returned False for {audio_uuid} - may have timed out"
                )
                metrics_collector.record_memory_storage_result(False)

        except Exception as e:
            audio_logger.error(
                f"❌ Error processing memory for {audio_uuid}: {e}"
            )
            metrics_collector = get_metrics_collector()
            metrics_collector.record_memory_storage_result(False)

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
                        self.file_sink = _new_local_file_sink(f"{CHUNK_DIR}/{wav_filename}")
                        await self.file_sink.open()

                        await chunk_repo.create_chunk(
                            audio_uuid=self.current_audio_uuid,
                            audio_path=wav_filename,
                            client_id=self.client_id,
                            timestamp=timestamp,
                        )

                    await self.file_sink.write(audio_chunk)

                    # Queue for transcription
                    await self.transcription_queue.put((self.current_audio_uuid, audio_chunk))

                except Exception as e:
                    audio_logger.error(f"Error processing audio chunk for client {self.client_id}: {e}")
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
                audio_uuid, chunk = await self.transcription_queue.get()

                if audio_uuid is None or chunk is None:  # Disconnect signal
                    self.transcription_queue.task_done()
                    break

                try:
                    # Get or create transcription manager
                    if self.transcription_manager is None:
                        # Create callback function to queue action items
                        async def action_item_callback(transcript_text, client_id, audio_uuid):
                            await self.action_item_queue.put((transcript_text, client_id, audio_uuid))

                        self.transcription_manager = TranscriptionManager(
                            action_item_callback=action_item_callback
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
                    except Exception as e:
                        audio_logger.error(f"Error transcribing for client {self.client_id}: {e}")
                        # Recreate transcription manager on error
                        if self.transcription_manager:
                            await self.transcription_manager.disconnect()
                            self.transcription_manager = None

                except Exception as e:
                    audio_logger.error(f"Error processing transcription item for client {self.client_id}: {e}")
                finally:
                    # Always mark task as done
                    self.transcription_queue.task_done()

        except Exception as e:
            audio_logger.error(
                f"Error in transcription processor for client {self.client_id}: {e}",
                exc_info=True,
            )

    async def _memory_processor(self):
        """Per-client memory processor - currently unused as memory processing happens at conversation end."""
        try:
            while self.connected:
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
                    audio_logger.error(f"Error processing memory item for client {self.client_id}: {e}")
                finally:
                    # Always mark task as done
                    self.memory_queue.task_done()

        except Exception as e:
            audio_logger.error(
                f"Error in memory processor for client {self.client_id}: {e}",
                exc_info=True,
            )

    async def _action_item_processor(self):
        """
        Processes transcript segments from the per-client action item queue.
        
        This processor handles queue management and delegates the actual 
        action item processing to the ActionItemsService.
        """
        try:
            while self.connected:
                transcript_text, client_id, audio_uuid = await self.action_item_queue.get()

                if (
                    transcript_text is None or client_id is None or audio_uuid is None
                ):  # Disconnect signal
                    self.action_item_queue.task_done()
                    break

                try:
                    # Resolve client_id to user information
                    user = await get_user_by_client_id(client_id)
                    if user:
                        # Delegate action item processing to the service
                        action_item_count = await action_items_service.process_transcript_for_action_items(
                            transcript_text, client_id, audio_uuid, user.user_id, user.email
                        )
                    else:
                        audio_logger.error(f"Could not resolve client_id {client_id} to user for action item processing")
                        action_item_count = 0
                    
                    if action_item_count > 0:
                        audio_logger.info(
                            f"🎯 Action item processor completed: {action_item_count} items processed for {audio_uuid}"
                        )
                    else:
                        audio_logger.debug(
                            f"ℹ️ Action item processor completed: no items found for {audio_uuid}"
                        )
                        
                except Exception as e:
                    audio_logger.error(f"Error processing action item for client {self.client_id}: {e}")
                finally:
                    # Always mark task as done
                    self.action_item_queue.task_done()

        except Exception as e:
            audio_logger.error(
                f"Error in action item processor for client {self.client_id}: {e}",
                exc_info=True,
            )


# Initialize repository and global state
chunk_repo = ChunkRepo(chunks_col)
active_clients: dict[str, ClientState] = {}

# Client-to-user mapping for reliable permission checking
client_to_user_mapping: dict[str, str] = {}  # client_id -> user_id


def register_client_user_mapping(client_id: str, user_id: str):
    """Register that a client belongs to a specific user."""
    client_to_user_mapping[client_id] = user_id
    audio_logger.debug(f"Registered client {client_id} for user {user_id}")


def unregister_client_user_mapping(client_id: str):
    """Unregister a client from user mapping."""
    if client_id in client_to_user_mapping:
        user_id = client_to_user_mapping.pop(client_id)
        audio_logger.debug(f"Unregistered client {client_id} from user {user_id}")


def get_user_clients(user_id: str) -> list[str]:
    """Get all currently active client IDs that belong to a specific user."""
    return [client_id for client_id, mapped_user_id in client_to_user_mapping.items() 
            if mapped_user_id == user_id]


# Client ownership tracking for database records
# Since we're in development, we'll track all client-user relationships in memory
# This will be populated when clients connect and persisted in database records
all_client_user_mappings: dict[str, str] = {}  # client_id -> user_id (includes disconnected clients)


def track_client_user_relationship(client_id: str, user_id: str):
    """Track that a client belongs to a user (persists after disconnection for database queries)."""
    all_client_user_mappings[client_id] = user_id


def client_belongs_to_user(client_id: str, user_id: str) -> bool:
    """Check if a client belongs to a specific user."""
    return all_client_user_mappings.get(client_id) == user_id


def get_user_clients_all(user_id: str) -> list[str]:
    """Get all client IDs (active and inactive) that belong to a specific user."""
    return [client_id for client_id, mapped_user_id in all_client_user_mappings.items() 
            if mapped_user_id == user_id]


async def create_client_state(client_id: str, user: User, device_name: Optional[str] = None) -> ClientState:
    """Create and register a new client state."""
    client_state = ClientState(client_id)
    active_clients[client_id] = client_state
    
    # Register client-user mapping (for active clients)
    register_client_user_mapping(client_id, user.user_id)
    
    # Also track in persistent mapping (for database queries)
    track_client_user_relationship(client_id, user.user_id)
    
    # Register client in user model (persistent)
    await register_client_to_user(user, client_id, device_name)
    
    await client_state.start_processing()

    # Track client connection in metrics
    metrics_collector = get_metrics_collector()
    metrics_collector.record_client_connection(client_id)

    return client_state


async def cleanup_client_state(client_id: str):
    """Clean up and remove client state."""
    if client_id in active_clients:
        client_state = active_clients[client_id]
        await client_state.disconnect()
        del active_clients[client_id]

    # Unregister client-user mapping
    unregister_client_user_mapping(client_id)

    # Track client disconnection in metrics
    metrics_collector = get_metrics_collector()
    metrics_collector.record_client_disconnection(client_id)


###############################################################################
# CORE APPLICATION LOGIC
###############################################################################


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    audio_logger.info("Starting application...")

    # Initialize Beanie for user management
    try:
        await init_beanie(
            database=mongo_client.get_default_database("friend-lite"),
            document_models=[User],
        )
        audio_logger.info("Beanie initialized for user management")
    except Exception as e:
        audio_logger.error(f"Failed to initialize Beanie: {e}")
        raise

    # Create admin user if needed
    try:
        await create_admin_user_if_needed()
    except Exception as e:
        audio_logger.error(f"Failed to create admin user: {e}")
        # Don't raise here as this is not critical for startup

    # Start metrics collection
    await start_metrics_collection()
    audio_logger.info("Metrics collection started")

    # Pre-initialize memory service to avoid blocking during first use
    try:
        audio_logger.info("Pre-initializing memory service...")
        await asyncio.wait_for(memory_service.initialize(), timeout=120)  # 2 minute timeout for startup
        audio_logger.info("Memory service pre-initialized successfully")
    except asyncio.TimeoutError:
        audio_logger.warning("Memory service pre-initialization timed out - will initialize on first use")
    except Exception as e:
        audio_logger.warning(f"Memory service pre-initialization failed: {e} - will initialize on first use")

    # Initialize failure recovery system
    try:
        audio_logger.info("Initializing failure recovery system...")
        await init_failure_recovery_system(
            queue_tracker_db="data/queue_tracker.db",
            persistent_queue_db="data/persistent_queues.db",
            start_monitoring=True,
            start_recovery=True,
            recovery_interval=30
        )
        audio_logger.info("Failure recovery system initialized successfully")
        
        # Perform startup recovery for any items that were processing when service stopped
        await perform_startup_recovery()
        audio_logger.info("Startup recovery completed")
        
    except Exception as e:
        audio_logger.error(f"Failed to initialize failure recovery system: {e}")
        # Don't raise here as this is not critical for basic operation

    audio_logger.info("Application ready - clients will have individual processing pipelines.")

    try:
        yield
    finally:
        # Shutdown
        audio_logger.info("Shutting down application...")

        # Clean up all active clients
        for client_id in list(active_clients.keys()):
            await cleanup_client_state(client_id)

        # Stop metrics collection and save final report
        await stop_metrics_collection()
        audio_logger.info("Metrics collection stopped")

        # Shutdown memory service and speaker service
        shutdown_memory_service()
        audio_logger.info("Memory and speaker services shut down.")

        # Shutdown failure recovery system
        try:
            await shutdown_failure_recovery_system()
            audio_logger.info("Failure recovery system shut down.")
        except Exception as e:
            audio_logger.error(f"Error shutting down failure recovery system: {e}")

        audio_logger.info("Shutdown complete.")


# FastAPI Application
app = FastAPI(lifespan=lifespan)
app.mount("/audio", StaticFiles(directory=CHUNK_DIR), name="audio")

# Add authentication routers
app.include_router(
    fastapi_users.get_auth_router(cookie_backend),
    prefix="/auth/cookie",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_auth_router(bearer_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

# Add memory debug router
from memory_debug_api import debug_router
app.include_router(debug_router)

# Add failure recovery router
failure_recovery_router = get_failure_recovery_router()
app.include_router(failure_recovery_router)


@app.websocket("/ws")
async def ws_endpoint(
    ws: WebSocket,
    token: Optional[str] = Query(None),
    device_name: Optional[str] = Query(None),
):
    """Accepts WebSocket connections, decodes Opus audio, and processes per-client."""
    # Authenticate user before accepting WebSocket connection
    user = await websocket_auth(ws, token)
    if not user:
        await ws.close(code=1008, reason="Authentication required")
        return

    await ws.accept()

    # Generate proper client_id using user and device_name
    client_id = generate_client_id(user, device_name)
    audio_logger.info(f"🔌 WebSocket connection accepted - User: {user.user_id} ({user.email}), Client: {client_id}")

    decoder = OmiOpusDecoder()
    _decode_packet = partial(decoder.decode_packet, strip_header=False)

    # Create client state and start processing
    client_state = await create_client_state(client_id, user, device_name)

    try:
        packet_count = 0
        total_bytes = 0
        while True:
            packet = await ws.receive_bytes()
            packet_count += 1
            total_bytes += len(packet)

            start_time = time.time()
            loop = asyncio.get_running_loop()
            pcm_data = await loop.run_in_executor(_DEC_IO_EXECUTOR, _decode_packet, packet)
            decode_time = time.time() - start_time

            if pcm_data:
                audio_logger.debug(
                    f"🎵 Decoded packet #{packet_count}: {len(packet)} bytes -> {len(pcm_data)} PCM bytes (took {decode_time:.3f}s)"
                )
                chunk = AudioChunk(
                    audio=pcm_data,
                    rate=OMI_SAMPLE_RATE,
                    width=OMI_SAMPLE_WIDTH,
                    channels=OMI_CHANNELS,
                    timestamp=int(time.time()),
                )
                await client_state.chunk_queue.put(chunk)

                # Log every 1000th packet to avoid spam
                if packet_count % 1000 == 0:
                    audio_logger.info(
                        f"📊 Processed {packet_count} packets ({total_bytes} bytes total) for client {client_id}"
                    )

                # Track audio chunk received in metrics
                metrics_collector = get_metrics_collector()
                metrics_collector.record_audio_chunk_received(client_id)
                metrics_collector.record_client_activity(client_id)

    except WebSocketDisconnect:
        audio_logger.info(
            f"🔌 WebSocket disconnected - Client: {client_id}, Packets: {packet_count}, Total bytes: {total_bytes}"
        )
    except Exception as e:
        audio_logger.error(f"❌ WebSocket error for client {client_id}: {e}", exc_info=True)
    finally:
        # Clean up client state
        await cleanup_client_state(client_id)


@app.websocket("/ws_pcm")
async def ws_endpoint_pcm(
    ws: WebSocket, 
    token: Optional[str] = Query(None),
    device_name: Optional[str] = Query(None)
):
    """Accepts WebSocket connections, processes PCM audio per-client."""
    # Authenticate user before accepting WebSocket connection
    user = await websocket_auth(ws, token)
    if not user:
        await ws.close(code=1008, reason="Authentication required")
        return

    await ws.accept()

    # Generate proper client_id using user and device_name
    client_id = generate_client_id(user, device_name)
    audio_logger.info(
        f"🔌 PCM WebSocket connection accepted - User: {user.user_id} ({user.email}), Client: {client_id}"
    )

    # Create client state and start processing
    client_state = await create_client_state(client_id, user, device_name)

    try:
        packet_count = 0
        total_bytes = 0
        while True:
            packet = await ws.receive_bytes()
            packet_count += 1
            total_bytes += len(packet)

            if packet:
                audio_logger.debug(f"🎵 Received PCM packet #{packet_count}: {len(packet)} bytes")
                chunk = AudioChunk(
                    audio=packet,
                    rate=16000,
                    width=2,
                    channels=1,
                    timestamp=int(time.time()),
                )
                await client_state.chunk_queue.put(chunk)

                # Log every 1000th packet to avoid spam
                if packet_count % 1000 == 0:
                    audio_logger.info(
                        f"📊 Processed {packet_count} PCM packets ({total_bytes} bytes total) for client {client_id}"
                    )

                # Track audio chunk received in metrics
                metrics_collector = get_metrics_collector()
                metrics_collector.record_audio_chunk_received(client_id)
                metrics_collector.record_client_activity(client_id)
    except WebSocketDisconnect:
        audio_logger.info(
            f"🔌 PCM WebSocket disconnected - Client: {client_id}, Packets: {packet_count}, Total bytes: {total_bytes}"
        )
    except Exception as e:
        audio_logger.error(f"❌ PCM WebSocket error for client {client_id}: {e}", exc_info=True)
    finally:
        # Clean up client state
        await cleanup_client_state(client_id)


@app.get("/api/conversations")
async def get_conversations(current_user: User = Depends(current_active_user)):
    """Get conversations. Admins see all conversations, users see only their own."""
    try:
        # Build query based on user permissions
        if not current_user.is_superuser:
            # Regular users can only see their own conversations
            user_client_ids = get_user_clients_all(current_user.user_id)
            if not user_client_ids:
                # User has no clients, return empty result
                return {"conversations": {}}
            query = {"client_id": {"$in": user_client_ids}}
        else:
            query = {}
        
        # Get audio chunks and group by client_id
        cursor = chunks_col.find(query).sort("timestamp", -1)
        conversations = {}

        async for chunk in cursor:
            client_id = chunk.get("client_id", "unknown")
            if client_id not in conversations:
                conversations[client_id] = []

            conversations[client_id].append(
                {
                    "audio_uuid": chunk["audio_uuid"],
                    "audio_path": chunk["audio_path"],
                    "cropped_audio_path": chunk.get("cropped_audio_path"),
                    "timestamp": chunk["timestamp"],
                    "transcript": chunk.get("transcript", []),
                    "speakers_identified": chunk.get("speakers_identified", []),
                    "speech_segments": chunk.get("speech_segments", []),
                    "cropped_duration": chunk.get("cropped_duration"),
                }
            )

        return {"conversations": conversations}
    except Exception as e:
        audio_logger.error(f"Error getting conversations: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/conversations/{audio_uuid}/cropped")
async def get_cropped_audio_info(audio_uuid: str, current_user: User = Depends(current_active_user)):
    """Get cropped audio information for a specific conversation. Users can only access their own conversations."""
    try:
        # Find the conversation first
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})
        
        # Check ownership for non-admin users
        if not current_user.is_superuser:
            if not client_belongs_to_user(chunk["client_id"], current_user.user_id):
                return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        return {
            "audio_uuid": audio_uuid,
            "original_audio_path": chunk["audio_path"],
            "cropped_audio_path": chunk.get("cropped_audio_path"),
            "speech_segments": chunk.get("speech_segments", []),
            "cropped_duration": chunk.get("cropped_duration"),
            "cropped_at": chunk.get("cropped_at"),
            "has_cropped_version": bool(chunk.get("cropped_audio_path")),
        }
    except Exception as e:
        audio_logger.error(f"Error getting cropped audio info: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/conversations/{audio_uuid}/reprocess")
async def reprocess_audio_cropping(audio_uuid: str, current_user: User = Depends(current_active_user)):
    """Trigger reprocessing of audio cropping for a specific conversation. Users can only reprocess their own conversations."""
    try:
        # Find the conversation first
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})
        
        # Check ownership for non-admin users
        if not current_user.is_superuser:
            if not client_belongs_to_user(chunk["client_id"], current_user.user_id):
                return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        original_path = f"{CHUNK_DIR}/{chunk['audio_path']}"
        if not Path(original_path).exists():
            return JSONResponse(status_code=404, content={"error": "Original audio file not found"})

        # Check if we have speech segments
        speech_segments = chunk.get("speech_segments", [])
        if not speech_segments:
            return JSONResponse(
                status_code=400,
                content={"error": "No speech segments available for cropping"},
            )

        # Convert speech segments from dict format to tuple format
        speech_segments_tuples = [(seg["start"], seg["end"]) for seg in speech_segments]

        cropped_filename = chunk["audio_path"].replace(".wav", "_cropped.wav")
        cropped_path = f"{CHUNK_DIR}/{cropped_filename}"

        # Process in background using shared logic
        async def reprocess_task():
            audio_logger.info(f"🔄 Starting reprocess for {audio_uuid}")
            await _process_audio_cropping_with_relative_timestamps(
                original_path, speech_segments_tuples, cropped_path, audio_uuid
            )

        asyncio.create_task(reprocess_task())

        return {"message": "Reprocessing started", "audio_uuid": audio_uuid}
    except Exception as e:
        audio_logger.error(f"Error reprocessing audio: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/users", response_model=List[UserRead])
async def get_users(current_user: User = Depends(current_superuser)):
    """Retrieves all users from the database. Admin-only endpoint."""
    try:
        # Use Beanie to query users properly - this handles datetime serialization automatically
        users = await User.find_all().to_list()
        return users
    except Exception as e:
        audio_logger.error(f"Error fetching users: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error fetching users"})


@app.post("/api/create_user")
async def create_user_admin(
    user_data: UserCreate, 
    current_user: User = Depends(current_superuser)
):
    """Creates a new user in the database. Admin-only endpoint."""
    try:
        # Get user manager for proper user creation
        user_db_gen = get_user_db()
        user_db = await user_db_gen.__anext__()
        user_manager_gen = get_user_manager(user_db)
        user_manager = await user_manager_gen.__anext__()
        
        # Check if user already exists
        existing_user = await user_db.get_by_email(user_data.email)
        if existing_user:
            return JSONResponse(
                status_code=409, 
                content={"message": f"User with email {user_data.email} already exists"}
            )

        # Create new user using fastapi-users manager
        new_user = await user_manager.create(user_data)
        
        return JSONResponse(
            status_code=201,
            content={
                "message": f"User {user_data.email} created successfully",
                "user": {
                    "id": str(new_user.id),
                    "email": new_user.email,
                    "display_name": new_user.display_name,
                    "is_active": new_user.is_active,
                    "is_superuser": new_user.is_superuser,
                    "is_verified": new_user.is_verified,
                },
            },
        )
    except Exception as e:
        audio_logger.error(f"Error creating user: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error creating user"})


@app.delete("/api/delete_user")
async def delete_user(
    user_id: str,
    delete_conversations: bool = False,
    delete_memories: bool = False,
    current_user: User = Depends(current_superuser),
):
    """Deletes a user from the database with optional data cleanup."""
    try:
        # Validate user_id format
        if not user_id or user_id == "Unknown":
            return JSONResponse(
                status_code=400, 
                content={"message": "Invalid user ID provided. Cannot delete user with ID 'Unknown'."}
            )
        
        # Validate ObjectId format
        try:
            object_id = ObjectId(user_id)
        except Exception:
            return JSONResponse(
                status_code=400, 
                content={"message": f"Invalid user ID format: '{user_id}'. Must be a valid MongoDB ObjectId."}
            )
        
        # Query the correct collection that fastapi-users actually uses
        fastapi_users_col = db["fastapi_users"]
        
        # Check if user exists
        existing_user = await fastapi_users_col.find_one({"_id": object_id})
        if not existing_user:
            return JSONResponse(status_code=404, content={"message": f"User {user_id} not found"})
        
        # Prevent deletion of administrator user
        user_email = existing_user.get("email", "")
        is_superuser = existing_user.get("is_superuser", False)
        
        if is_superuser or user_email == ADMIN_EMAIL:
            return JSONResponse(
                status_code=403, 
                content={"message": f"Cannot delete administrator user. Admin users are protected from deletion."}
            )

        deleted_data = {}

        # Delete user from fastapi_users collection
        user_result = await fastapi_users_col.delete_one({"_id": object_id})
        deleted_data["user_deleted"] = user_result.deleted_count > 0

        if delete_conversations:
            # Delete all conversations (audio chunks) for this user
            conversations_result = await chunks_col.delete_many({"client_id": user_id})
            deleted_data["conversations_deleted"] = conversations_result.deleted_count

        if delete_memories:
            # Delete all memories for this user using the memory service
            try:
                memory_count = await asyncio.get_running_loop().run_in_executor(
                    None, memory_service.delete_all_user_memories, user_id
                )
                deleted_data["memories_deleted"] = memory_count
            except Exception as mem_error:
                audio_logger.error(f"Error deleting memories for user {user_id}: {mem_error}")
                deleted_data["memories_deleted"] = 0
                deleted_data["memory_deletion_error"] = str(mem_error)

        # Build message based on what was deleted
        message = f"User {user_id} deleted successfully"
        deleted_items = []
        if delete_conversations and deleted_data.get("conversations_deleted", 0) > 0:
            deleted_items.append(f"{deleted_data['conversations_deleted']} conversations")
        if delete_memories and deleted_data.get("memories_deleted", 0) > 0:
            deleted_items.append(f"{deleted_data['memories_deleted']} memories")

        if deleted_items:
            message += f" along with {' and '.join(deleted_items)}"

        return JSONResponse(
            status_code=200, content={"message": message, "deleted_data": deleted_data}
        )
    except Exception as e:
        audio_logger.error(f"Error deleting user: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error deleting user"})


@app.get("/api/memories")
async def get_memories(current_user: User = Depends(current_active_user), user_id: Optional[str] = None, limit: int = 100):
    """Retrieves memories from the mem0 store. Admins can specify user_id, users see only their own."""
    try:
        # Determine which user's memories to retrieve
        if current_user.is_superuser and user_id:
            # Admin can request specific user's memories
            target_user_id = user_id
        else:
            # Regular users can only see their own memories
            target_user_id = current_user.user_id
        
        all_memories = await asyncio.get_running_loop().run_in_executor(
            None, memory_service.get_all_memories, target_user_id, limit
        )
        return JSONResponse(content=all_memories)
    except Exception as e:
        audio_logger.error(f"Error fetching memories: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error fetching memories"})


@app.get("/api/memories/search")
async def search_memories(query: str, current_user: User = Depends(current_active_user), user_id: Optional[str] = None, limit: int = 10):
    """Search memories using semantic similarity. Admins can specify user_id, users search only their own."""
    try:
        # Determine which user's memories to search
        if current_user.is_superuser and user_id:
            # Admin can search specific user's memories
            target_user_id = user_id
        else:
            # Regular users can only search their own memories
            target_user_id = current_user.user_id
        
        relevant_memories = await asyncio.get_running_loop().run_in_executor(
            None, memory_service.search_memories, query, target_user_id, limit
        )
        return JSONResponse(content=relevant_memories)
    except Exception as e:
        audio_logger.error(f"Error searching memories: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error searching memories"})


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str, current_user: User = Depends(current_active_user)):
    """Delete a specific memory by ID. Requires authentication."""
    try:
        await asyncio.get_running_loop().run_in_executor(
            None, memory_service.delete_memory, memory_id
        )
        return JSONResponse(content={"message": f"Memory {memory_id} deleted successfully"})
    except Exception as e:
        audio_logger.error(f"Error deleting memory {memory_id}: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error deleting memory"})


@app.get("/api/admin/memories/debug")
async def get_all_memories_debug(current_user: User = Depends(current_superuser), limit: int = 100):
    """Admin-only endpoint to get all memories across all users with debug information."""
    try:
        # Get all users from database
        all_users = await User.find().to_list()
        
        # Convert datetime objects to ISO strings for JSON serialization
        def convert_datetime_to_string(obj):
            if hasattr(obj, 'isoformat'):  # datetime objects
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime_to_string(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime_to_string(item) for item in obj]
            else:
                return obj
        
        admin_user_dict = {
            "id": current_user.user_id,
            "email": current_user.email,
            "is_superuser": current_user.is_superuser
        }
        
        debug_info = {
            "total_users": len(all_users),
            "admin_user": convert_datetime_to_string(admin_user_dict),
            "users_with_memories": []
        }
        
        total_memories = 0
        
        # Check memories for all database users
        for user in all_users:
            try:
                user_memories = await asyncio.get_running_loop().run_in_executor(
                    None, memory_service.get_all_memories, user.user_id, limit
                )
                
                # Include client information from user model
                registered_clients = user.registered_clients if hasattr(user, 'registered_clients') else []
                
                # Convert user object to dict and handle datetime serialization
                user_dict = {
                    "user_id": user.user_id,
                    "email": user.email,
                    "display_name": user.display_name,
                    "is_superuser": user.is_superuser,
                    "memory_count": len(user_memories),
                    "memories": user_memories,
                    "registered_clients": registered_clients,
                    "client_count": len(registered_clients)
                }
                
                user_info = convert_datetime_to_string(user_dict)
                
                debug_info["users_with_memories"].append(user_info)
                total_memories += len(user_memories)
                
            except Exception as e:
                audio_logger.error(f"Error fetching memories for user {user.user_id}: {e}")
                # Convert user object to dict and handle datetime serialization for error case
                error_user_dict = {
                    "user_id": user.user_id,
                    "email": user.email,
                    "display_name": user.display_name,
                    "is_superuser": user.is_superuser,
                    "memory_count": 0,
                    "memories": [],
                    "registered_clients": [],
                    "client_count": 0,
                    "error": str(e)
                }
                
                error_user_info = convert_datetime_to_string(error_user_dict)
                debug_info["users_with_memories"].append(error_user_info)
        
        debug_info["total_memories"] = total_memories
        
        return JSONResponse(content=debug_info)
    except Exception as e:
        audio_logger.error(f"Error fetching admin debug memories: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error fetching admin debug memories"})


@app.get("/api/admin/memories")
async def get_all_memories_admin(current_user: User = Depends(current_superuser), limit: int = 200):
    """Admin-only endpoint to get all memories across all users in a clean format."""
    try:
        # Get all users from database
        all_users = await User.find().to_list()
        
        all_memories = []
        
        # Collect memories for all users
        for user in all_users:
            try:
                user_memories = await asyncio.get_running_loop().run_in_executor(
                    None, memory_service.get_all_memories, user.user_id, limit
                )
                
                # Enrich each memory with user information
                for memory in user_memories:
                    memory_with_user = {
                        **memory,
                        "owner_user_id": user.user_id,
                        "owner_email": user.email,
                        "owner_display_name": user.display_name
                    }
                    all_memories.append(memory_with_user)
                    
            except Exception as e:
                audio_logger.error(f"Error fetching memories for user {user.user_id}: {e}")
                continue
        
        # Sort by creation date (newest first)
        all_memories.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return JSONResponse(content={
            "total_memories": len(all_memories),
            "total_users": len(all_users),
            "memories": all_memories[:limit]  # Respect limit
        })
    except Exception as e:
        audio_logger.error(f"Error fetching admin memories: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error fetching admin memories"})


@app.post("/api/conversations/{audio_uuid}/speakers")
async def add_speaker_to_conversation(audio_uuid: str, speaker_id: str, current_user: User = Depends(current_active_user)):
    """Add a speaker to the speakers_identified list for a conversation. Users can only modify their own conversations."""
    try:
        # Find the conversation first
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})
        
        # Check ownership for non-admin users
        if not current_user.is_superuser:
            if not client_belongs_to_user(chunk["client_id"], current_user.user_id):
                return JSONResponse(status_code=404, content={"error": "Conversation not found"})
        
        await chunk_repo.add_speaker(audio_uuid, speaker_id)
        return JSONResponse(
            content={"message": f"Speaker {speaker_id} added to conversation {audio_uuid}"}
        )
    except Exception as e:
        audio_logger.error(f"Error adding speaker: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error adding speaker"})


@app.put("/api/conversations/{audio_uuid}/transcript/{segment_index}")
async def update_transcript_segment(
    audio_uuid: str,
    segment_index: int,
    current_user: User = Depends(current_active_user),
    speaker_id: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
):
    """Update a specific transcript segment with speaker or timing information. Users can only modify their own conversations."""
    try:
        # Find the conversation first
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})
        
        # Check ownership for non-admin users
        if not current_user.is_superuser:
            if not client_belongs_to_user(chunk["client_id"], current_user.user_id):
                return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        update_doc = {}

        if speaker_id is not None:
            update_doc[f"transcript.{segment_index}.speaker"] = speaker_id
            # Also add to speakers_identified if not already present
            await chunk_repo.add_speaker(audio_uuid, speaker_id)

        if start_time is not None:
            update_doc[f"transcript.{segment_index}.start"] = start_time

        if end_time is not None:
            update_doc[f"transcript.{segment_index}.end"] = end_time

        if not update_doc:
            return JSONResponse(status_code=400, content={"error": "No update parameters provided"})

        result = await chunks_col.update_one({"audio_uuid": audio_uuid}, {"$set": update_doc})

        if result.modified_count == 0:
            return JSONResponse(status_code=400, content={"error": "No changes were made"})

        return JSONResponse(content={"message": "Transcript segment updated successfully"})

    except Exception as e:
        audio_logger.error(f"Error updating transcript segment: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})


# class SpeakerEnrollmentRequest(BaseModel):
#     speaker_id: str
#     speaker_name: str
#     audio_file_path: str
#     start_time: Optional[float] = None
#     end_time: Optional[float] = None


# class SpeakerIdentificationRequest(BaseModel):
#     audio_file_path: str
#     start_time: Optional[float] = None
#     end_time: Optional[float] = None


# class ActionItemUpdateRequest(BaseModel):
#     status: str  # "open", "in_progress", "completed", "cancelled"


# class ActionItemCreateRequest(BaseModel):
#     description: str
#     assignee: Optional[str] = "unassigned"
#     due_date: Optional[str] = "not_specified"
#     priority: Optional[str] = "medium"
#     context: Optional[str] = ""


@app.get("/health")
async def health_check():
    """Comprehensive health check for all services."""
    health_status = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "services": {},
        "config": {
            "mongodb_uri": MONGODB_URI,
            "ollama_url": OLLAMA_BASE_URL,
            "qdrant_url": f"http://{QDRANT_BASE_URL}:6333",
            "transcription_service": ("Deepgram WebSocket" if USE_DEEPGRAM else "Offline ASR"),
            "asr_uri": (OFFLINE_ASR_TCP_URI if not USE_DEEPGRAM else "wss://api.deepgram.com"),
            "deepgram_enabled": USE_DEEPGRAM,
            "chunk_dir": str(CHUNK_DIR),
            "active_clients": len(active_clients),
            "new_conversation_timeout_minutes": NEW_CONVERSATION_TIMEOUT_MINUTES,
            "action_items_enabled": True,
            "audio_cropping_enabled": AUDIO_CROPPING_ENABLED,
        },
    }

    overall_healthy = True
    critical_services_healthy = True

    # Check MongoDB (critical service)
    try:
        await asyncio.wait_for(mongo_client.admin.command("ping"), timeout=5.0)
        health_status["services"]["mongodb"] = {
            "status": "✅ Connected",
            "healthy": True,
            "critical": True,
        }
    except asyncio.TimeoutError:
        health_status["services"]["mongodb"] = {
            "status": "❌ Connection Timeout (5s)",
            "healthy": False,
            "critical": True,
        }
        overall_healthy = False
        critical_services_healthy = False
    except Exception as e:
        health_status["services"]["mongodb"] = {
            "status": f"❌ Connection Failed: {str(e)}",
            "healthy": False,
            "critical": True,
        }
        overall_healthy = False
        critical_services_healthy = False

    # Check Ollama (non-critical service - may not be running)
    try:
        # Run in executor to avoid blocking the main thread
        loop = asyncio.get_running_loop()
        models = await asyncio.wait_for(loop.run_in_executor(None, ollama_client.list), timeout=8.0)
        model_count = len(models.get("models", []))
        health_status["services"]["ollama"] = {
            "status": "✅ Connected",
            "healthy": True,
            "models": model_count,
            "critical": False,
        }
    except asyncio.TimeoutError:
        health_status["services"]["ollama"] = {
            "status": "⚠️ Connection Timeout (8s) - Service may not be running",
            "healthy": False,
            "critical": False,
        }
        overall_healthy = False
    except Exception as e:
        health_status["services"]["ollama"] = {
            "status": f"⚠️ Connection Failed: {str(e)} - Service may not be running",
            "healthy": False,
            "critical": False,
        }
        overall_healthy = False

    # Check mem0 (depends on Ollama and Qdrant)
    try:
        # Test memory service connection with timeout
        test_success = await memory_service.test_connection()
        if test_success:
            health_status["services"]["mem0"] = {
                "status": "✅ Connected",
                "healthy": True,
                "critical": False,
            }
        else:
            health_status["services"]["mem0"] = {
                "status": "⚠️ Connection Test Failed",
                "healthy": False,
                "critical": False,
            }
            overall_healthy = False
    except asyncio.TimeoutError:
        health_status["services"]["mem0"] = {
            "status": "⚠️ Connection Test Timeout (60s) - Depends on Ollama/Qdrant",
            "healthy": False,
            "critical": False,
        }
        overall_healthy = False
    except Exception as e:
        health_status["services"]["mem0"] = {
            "status": f"⚠️ Connection Test Failed: {str(e)} - Check Ollama/Qdrant services",
            "healthy": False,
            "critical": False,
        }
        overall_healthy = False

    # Check ASR service based on configuration
    if USE_DEEPGRAM:
        # Check Deepgram WebSocket connectivity
        if DEEPGRAM_API_KEY:
            health_status["services"]["deepgram"] = {
                "status": "✅ API Key Configured",
                "healthy": True,
                "type": "WebSocket",
                "critical": False,
            }
        else:
            health_status["services"]["deepgram"] = {
                "status": "❌ API Key Missing",
                "healthy": False,
                "type": "WebSocket",
                "critical": False,
            }
            overall_healthy = False
    else:
        # Check offline ASR service (non-critical - may be external)
        try:
            test_client = AsyncTcpClient.from_uri(OFFLINE_ASR_TCP_URI)
            await asyncio.wait_for(test_client.connect(), timeout=5.0)
            await test_client.disconnect()
            health_status["services"]["asr"] = {
                "status": "✅ Connected",
                "healthy": True,
                "uri": OFFLINE_ASR_TCP_URI,
                "critical": False,
            }
        except asyncio.TimeoutError:
            health_status["services"]["asr"] = {
                "status": f"⚠️ Connection Timeout (5s) - Check external ASR service",
                "healthy": False,
                "uri": OFFLINE_ASR_TCP_URI,
                "critical": False,
            }
            overall_healthy = False
        except Exception as e:
            health_status["services"]["asr"] = {
                "status": f"⚠️ Connection Failed: {str(e)} - Check external ASR service",
                "healthy": False,
                "uri": OFFLINE_ASR_TCP_URI,
                "critical": False,
            }
            overall_healthy = False

    # Track health check results in metrics
    try:
        metrics_collector = get_metrics_collector()
        for service_name, service_info in health_status["services"].items():
            success = service_info.get("healthy", False)
            failure_reason = None if success else service_info.get("status", "Unknown failure")
            metrics_collector.record_service_health_check(service_name, success, failure_reason)

        # Also track overall system health
        metrics_collector.record_service_health_check(
            "friend-backend", overall_healthy, "System health check"
        )
    except Exception as e:
        audio_logger.error(f"Failed to record health check metrics: {e}")

    # Set overall status
    health_status["overall_healthy"] = overall_healthy
    health_status["critical_services_healthy"] = critical_services_healthy

    if not critical_services_healthy:
        health_status["status"] = "critical"
    elif not overall_healthy:
        health_status["status"] = "degraded"
    else:
        health_status["status"] = "healthy"

    # Add helpful messages
    if not overall_healthy:
        messages = []
        if not critical_services_healthy:
            messages.append(
                "Critical services (MongoDB) are unavailable - core functionality will not work"
            )

        unhealthy_optional = [
            name
            for name, service in health_status["services"].items()
            if not service["healthy"] and not service.get("critical", True)
        ]
        if unhealthy_optional:
            messages.append(f"Optional services unavailable: {', '.join(unhealthy_optional)}")

        health_status["message"] = "; ".join(messages)

    return JSONResponse(content=health_status, status_code=200)


@app.get("/readiness")
async def readiness_check():
    """Simple readiness check for container orchestration."""
    return JSONResponse(content={"status": "ready", "timestamp": int(time.time())}, status_code=200)


@app.post("/api/close_conversation")
async def close_current_conversation(client_id: str, current_user: User = Depends(current_active_user)):
    """Close the current conversation for a specific client. Users can only close their own conversations."""
    # Validate client ownership
    if not current_user.is_superuser and not client_belongs_to_user(client_id, current_user.user_id):
        logger.warning(f"User {current_user.user_id} attempted to close conversation for client {client_id} without permission")
        return JSONResponse(
            content={
                "error": "Access forbidden. You can only close your own conversations.",
                "details": f"Client '{client_id}' does not belong to your account."
            },
            status_code=403,
        )
    
    if client_id not in active_clients:
        return JSONResponse(
            content={"error": f"Client '{client_id}' not found or not connected"},
            status_code=404,
        )

    client_state = active_clients[client_id]
    if not client_state.connected:
        return JSONResponse(
            content={"error": f"Client '{client_id}' is not connected"}, status_code=400
        )

    try:
        # Close the current conversation
        await client_state._close_current_conversation()

        # Reset conversation state but keep client connected
        client_state.current_audio_uuid = None
        client_state.conversation_start_time = time.time()
        client_state.last_transcript_time = None

        logger.info(f"Manually closed conversation for client {client_id} by user {current_user.id}")

        return JSONResponse(
            content={
                "message": f"Successfully closed current conversation for client '{client_id}'",
                "client_id": client_id,
                "timestamp": int(time.time()),
            }
        )

    except Exception as e:
        logger.error(f"Error closing conversation for client {client_id}: {e}")
        return JSONResponse(
            content={"error": f"Failed to close conversation: {str(e)}"},
            status_code=500,
        )


@app.get("/api/active_clients")
async def get_active_clients(current_user: User = Depends(current_active_user)):
    """Get list of currently active/connected clients. Admins see all, users see only their own."""
    client_info = {}

    for client_id, client_state in active_clients.items():
        # Filter clients based on user permissions
        if not current_user.is_superuser:
            # Regular users can only see clients that belong to them
            if not client_belongs_to_user(client_id, current_user.user_id):
                continue

        client_info[client_id] = {
            "connected": client_state.connected,
            "current_audio_uuid": client_state.current_audio_uuid,
            "conversation_start_time": client_state.conversation_start_time,
            "last_transcript_time": client_state.last_transcript_time,
            "has_active_conversation": client_state.current_audio_uuid is not None,
        }

    return JSONResponse(
        content={"active_clients_count": len(client_info), "clients": client_info}
    )


@app.get("/api/debug/speech_segments")
async def debug_speech_segments(current_user: User = Depends(current_active_user)):
    """Debug endpoint to check current speech segments. Admins see all clients, users see only their own."""
    filtered_clients = {}
    
    for client_id, client_state in active_clients.items():
        # Filter clients based on user permissions
        if not current_user.is_superuser:
            # Regular users can only see clients that belong to them
            if not client_belongs_to_user(client_id, current_user.user_id):
                continue
                
        filtered_clients[client_id] = {
            "current_audio_uuid": client_state.current_audio_uuid,
            "speech_segments": {
                uuid: segments for uuid, segments in client_state.speech_segments.items()
            },
            "current_speech_start": dict(client_state.current_speech_start),
            "connected": client_state.connected,
            "last_transcript_time": client_state.last_transcript_time,
        }

    debug_info = {
        "active_clients": len(filtered_clients),
        "audio_cropping_enabled": AUDIO_CROPPING_ENABLED,
        "min_speech_duration": MIN_SPEECH_SEGMENT_DURATION,
        "cropping_padding": CROPPING_CONTEXT_PADDING,
        "clients": filtered_clients,
    }

    return JSONResponse(content=debug_info)


@app.get("/api/debug/audio-cropping")
async def get_audio_cropping_debug(current_user: User = Depends(current_superuser)):
    """Get detailed debug information about the audio cropping system."""
    # Get speech segments for all active clients
    speech_segments_info = {}
    for client_id, client_state in active_clients.items():
        if client_state.connected:
            speech_segments_info[client_id] = {
                "current_audio_uuid": client_state.current_audio_uuid,
                "speech_segments": dict(client_state.speech_segments),
                "current_speech_start": dict(client_state.current_speech_start),
                "total_segments": sum(len(segments) for segments in client_state.speech_segments.values()),
            }

    # Get recent audio chunks with cropping status
    recent_chunks = []
    try:
        cursor = chunks_col.find().sort("timestamp", -1).limit(10)
        async for chunk in cursor:
            recent_chunks.append({
                "audio_uuid": chunk["audio_uuid"],
                "timestamp": chunk["timestamp"],
                "client_id": chunk["client_id"],
                "audio_path": chunk["audio_path"],
                "has_cropped_version": bool(chunk.get("cropped_audio_path")),
                "cropped_audio_path": chunk.get("cropped_audio_path"),
                "speech_segments_count": len(chunk.get("speech_segments", [])),
                "cropped_duration": chunk.get("cropped_duration"),
            })
    except Exception as e:
        audio_logger.error(f"Error getting recent chunks: {e}")
        recent_chunks = []

    return JSONResponse(
        content={
            "timestamp": time.time(),
            "audio_cropping_config": {
                "enabled": AUDIO_CROPPING_ENABLED,
                "min_speech_duration": MIN_SPEECH_SEGMENT_DURATION,
                "cropping_padding": CROPPING_CONTEXT_PADDING,
            },
            "asr_config": {
                "use_deepgram": USE_DEEPGRAM,
                "offline_asr_uri": OFFLINE_ASR_TCP_URI,
                "deepgram_available": DEEPGRAM_AVAILABLE,
            },
            "active_clients_speech_segments": speech_segments_info,
            "recent_audio_chunks": recent_chunks,
        }
    )


@app.get("/api/metrics")
async def get_current_metrics(current_user: User = Depends(current_superuser)):
    """Get current metrics summary for monitoring dashboard. Admin-only endpoint."""
    try:
        metrics_collector = get_metrics_collector()
        metrics_summary = metrics_collector.get_current_metrics_summary()
        return metrics_summary
    except Exception as e:
        audio_logger.error(f"Error getting current metrics: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/auth/config")
async def get_auth_config():
    """Get authentication configuration for UI."""
    return {
        "auth_methods": {
            "email_password": True,
            "registration": False,  # Public registration disabled
            "admin_user_creation": True,  # Only admins can create users
        },
        "endpoints": {
            "jwt_login": "/auth/jwt/login",
            "cookie_login": "/auth/cookie/login",
            "register": None,  # Public registration disabled
            "admin_create_user": "/api/create_user",  # Admin-only user creation
        },
        "admin_user": {
            "email": os.getenv("ADMIN_EMAIL",  'admin@example.com'),
        },
    }


###############################################################################
# ACTION ITEMS API ENDPOINTS
###############################################################################

from typing import List
from pydantic import BaseModel

class ActionItemCreate(BaseModel):
    description: str
    assignee: Optional[str] = "unassigned"
    due_date: Optional[str] = "not_specified"
    priority: Optional[str] = "medium"
    context: Optional[str] = ""

class ActionItemUpdate(BaseModel):
    description: Optional[str] = None
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    context: Optional[str] = None

@app.get("/api/action-items")
async def get_action_items(current_user: User = Depends(current_active_user), user_id: Optional[str] = None):
    """Get action items. Admins can specify user_id, users see only their own."""
    try:
        # Determine which user's action items to retrieve
        if current_user.is_superuser and user_id:
            target_user_id = user_id
        else:
            target_user_id = current_user.user_id
        
        # Query action items from database
        query = {"user_id": target_user_id}
        cursor = action_items_col.find(query).sort("created_at", -1)
        action_items = []
        
        async for item in cursor:
            # Convert ObjectId to string for JSON serialization
            item["_id"] = str(item["_id"])
            action_items.append(item)
        
        return {"action_items": action_items, "count": len(action_items)}
    except Exception as e:
        audio_logger.error(f"Error getting action items: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/action-items")
async def create_action_item(item: ActionItemCreate, current_user: User = Depends(current_active_user)):
    """Create a new action item."""
    try:
        action_item_doc = {
            "description": item.description,
            "assignee": item.assignee,
            "due_date": item.due_date,
            "priority": item.priority,
            "status": "open",
            "context": item.context,
            "user_id": current_user.user_id,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        
        result = await action_items_col.insert_one(action_item_doc)
        action_item_doc["_id"] = str(result.inserted_id)
        
        return {"message": "Action item created successfully", "action_item": action_item_doc}
    except Exception as e:
        audio_logger.error(f"Error creating action item: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/action-items/{item_id}")
async def get_action_item(item_id: str, current_user: User = Depends(current_active_user)):
    """Get a specific action item. Users can only access their own."""
    try:
        from bson import ObjectId
        
        # Build query with user restrictions
        query: dict[str, Any] = {"_id": ObjectId(item_id)}
        if not current_user.is_superuser:
            query["user_id"] = current_user.user_id
        
        item = await action_items_col.find_one(query)
        if not item:
            return JSONResponse(status_code=404, content={"error": "Action item not found"})
        
        item["_id"] = str(item["_id"])
        return {"action_item": item}
    except Exception as e:
        audio_logger.error(f"Error getting action item: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.put("/api/action-items/{item_id}")
async def update_action_item(item_id: str, updates: ActionItemUpdate, current_user: User = Depends(current_active_user)):
    """Update an action item. Users can only update their own."""
    try:
        from bson import ObjectId
        
        # Build query with user restrictions
        query: dict[str, Any] = {"_id": ObjectId(item_id)}
        if not current_user.is_superuser:
            query["user_id"] = current_user.user_id
        
        # Build update document
        update_doc = {"updated_at": time.time()}
        for field, value in updates.dict(exclude_unset=True).items():
            if value is not None:
                update_doc[field] = value
        
        result = await action_items_col.update_one(query, {"$set": update_doc})
        
        if result.matched_count == 0:
            return JSONResponse(status_code=404, content={"error": "Action item not found or access denied"})
        
        return {"message": "Action item updated successfully"}
    except Exception as e:
        audio_logger.error(f"Error updating action item: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/api/action-items/{item_id}")
async def delete_action_item(item_id: str, current_user: User = Depends(current_active_user)):
    """Delete an action item. Users can only delete their own."""
    try:
        from bson import ObjectId
        
        # Build query with user restrictions
        query: dict[str, Any] = {"_id": ObjectId(item_id)}
        if not current_user.is_superuser:
            query["user_id"] = current_user.user_id
        
        result = await action_items_col.delete_one(query)
        
        if result.deleted_count == 0:
            return JSONResponse(status_code=404, content={"error": "Action item not found or access denied"})
        
        return {"message": "Action item deleted successfully"}
    except Exception as e:
        audio_logger.error(f"Error deleting action item: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/action-items/stats")
async def get_action_items_stats(current_user: User = Depends(current_active_user), user_id: Optional[str] = None):
    """Get action items statistics. Admins can specify user_id, users see only their own stats."""
    try:
        # Determine which user's stats to retrieve
        if current_user.is_superuser and user_id:
            target_user_id = user_id
        else:
            target_user_id = current_user.user_id
        
        # Aggregate stats from action items collection
        pipeline = [
            {"$match": {"user_id": target_user_id}},
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        cursor = action_items_col.aggregate(pipeline)
        status_counts = {}
        total_count = 0
        
        async for doc in cursor:
            status = doc["_id"]
            count = doc["count"]
            status_counts[status] = count
            total_count += count
        
        stats = {
            "total": total_count,
            "by_status": status_counts,
            "open": status_counts.get("open", 0),
            "in_progress": status_counts.get("in_progress", 0),
            "completed": status_counts.get("completed", 0),
            "cancelled": status_counts.get("cancelled", 0),
        }
        
        return {"stats": stats}
    except Exception as e:
        audio_logger.error(f"Error getting action items stats: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/process-audio-files")
async def process_audio_files(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(current_active_user),
    device_name: Optional[str] = "file_upload"
):
    """Process uploaded audio files (.wav) and add them to the audio processing pipeline. Each file creates a separate conversation."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        processed_files = []
        processed_conversations = []
        
        for file_index, file in enumerate(files):
            # Check if file is a WAV file
            if not file.filename or not file.filename.lower().endswith('.wav'):
                audio_logger.warning(f"Skipping non-WAV file: {file.filename}")
                continue
                
            try:
                # Generate unique client ID for each file to create separate conversations
                file_device_name = f"{device_name}-{file_index + 1}"
                client_id = generate_client_id(current_user, file_device_name)
                
                # Create separate client state for this file
                client_state = await create_client_state(client_id, current_user, file_device_name)
                
                audio_logger.info(f"📁 Processing file {file_index + 1}/{len(files)}: {file.filename} with client_id: {client_id}")
                
                # Read file content
                content = await file.read()
                
                # Process WAV file
                with wave.open(io.BytesIO(content), 'rb') as wav_file:
                    # Get audio parameters
                    sample_rate = wav_file.getframerate()
                    sample_width = wav_file.getsampwidth()
                    channels = wav_file.getnchannels()
                    
                    # Read all audio data
                    audio_data = wav_file.readframes(wav_file.getnframes())
                    
                    # Convert to mono if stereo
                    if channels == 2:
                        # Convert stereo to mono by averaging channels
                        if sample_width == 2:
                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        else:
                            audio_array = np.frombuffer(audio_data, dtype=np.int32)
                        
                        # Reshape to separate channels and average
                        audio_array = audio_array.reshape(-1, 2)
                        audio_data = np.mean(audio_array, axis=1).astype(audio_array.dtype).tobytes()
                        channels = 1
                    
                    # Ensure sample rate is 16kHz (resample if needed)
                    if sample_rate != 16000:
                        audio_logger.warning(f"File {file.filename} has sample rate {sample_rate}Hz, expected 16kHz. Processing anyway.")
                    
                    # Process audio in larger chunks for faster file processing
                    # File uploads don't need to simulate real-time streaming delays
                    # Use larger chunks (32KB) for optimal performance
                    chunk_size = 32 * 1024  # 32KB chunks (was 1KB with 10ms delays)
                    base_timestamp = int(time.time())
                    
                    for i in range(0, len(audio_data), chunk_size):
                        chunk_data = audio_data[i:i + chunk_size]
                        
                        # Calculate relative timestamp for this chunk
                        chunk_offset_bytes = i
                        chunk_offset_seconds = chunk_offset_bytes / (sample_rate * sample_width * channels)
                        chunk_timestamp = base_timestamp + int(chunk_offset_seconds)
                        
                        # Create AudioChunk
                        chunk = AudioChunk(
                            audio=chunk_data,
                            rate=sample_rate,
                            width=sample_width,
                            channels=channels,
                            timestamp=chunk_timestamp,
                        )
                        
                        # Add to processing queue
                        await client_state.chunk_queue.put(chunk)
                        
                        # For file uploads, we don't need delays like real-time streams
                        # Just yield control occasionally to prevent blocking the event loop
                        if i % (chunk_size * 10) == 0:  # Every 10 chunks (~320KB)
                            await asyncio.sleep(0)
                    
                    # Track in metrics
                    metrics_collector = get_metrics_collector()
                    metrics_collector.record_audio_chunk_received(client_id)
                    metrics_collector.record_client_activity(client_id)
                    
                    processed_files.append({
                        "filename": file.filename,
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "duration_seconds": len(audio_data) / (sample_rate * sample_width * channels),
                        "size_bytes": len(audio_data),
                        "client_id": client_id
                    })
                    
                    audio_logger.info(f"✅ Processed audio file: {file.filename} ({len(audio_data)} bytes)")
                
                # Wait for this file's transcription processing to complete
                audio_logger.info(f"📁 Waiting for transcription to process file: {file.filename}")
                
                # Wait for chunks to be processed by the audio saver
                await asyncio.sleep(1.0)
                
                # Wait for transcription queue to be processed for this file
                max_wait_time = 60  # 1 minute per file
                wait_interval = 0.5
                elapsed_time = 0
                
                while elapsed_time < max_wait_time:
                    if (client_state.transcription_queue.empty() and 
                        client_state.chunk_queue.empty()):
                        audio_logger.info(f"📁 Transcription completed for file: {file.filename}")
                        break
                        
                    await asyncio.sleep(wait_interval)
                    elapsed_time += wait_interval
                
                if elapsed_time >= max_wait_time:
                    audio_logger.warning(f"📁 Transcription timed out for file: {file.filename}")
                
                # Close this conversation
                await client_state.chunk_queue.put(None)
                
                # Give cleanup time to complete
                await asyncio.sleep(0.5)
                
                # Track conversation created
                conversation_info = {
                    "client_id": client_id,
                    "filename": file.filename,
                    "status": "completed" if elapsed_time < max_wait_time else "timed_out"
                }
                processed_conversations.append(conversation_info)
                
                audio_logger.info(f"📁 Completed processing file {file_index + 1}/{len(files)}: {file.filename}")
                    
            except Exception as e:
                audio_logger.error(f"Error processing file {file.filename}: {e}")
                continue
        
        # All files have been processed individually
        audio_logger.info(f"📁 Completed processing all {len(processed_files)} files with {len(processed_conversations)} conversations created")
        
        return {
            "message": f"Successfully processed {len(processed_files)} audio files into {len(processed_conversations)} separate conversations",
            "processed_files": processed_files,
            "conversations": processed_conversations,
            "total_conversations_created": len(processed_conversations)
        }
        
    except Exception as e:
        audio_logger.error(f"Error in process_audio_files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio files: {str(e)}")


###############################################################################
# ENTRYPOINT
###############################################################################

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    audio_logger.info("Starting Omi unified service at ws://%s:%s/ws", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=False)
