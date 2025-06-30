#!/usr/bin/env python3
"""Unified Omi-audio service

* Accepts Opus packets over a WebSocket (`/ws`) or PCM over a WebSocket (`/ws_pcm`).
* Uses a central queue to decouple audio ingestion from processing.
* A saver consumer buffers PCM and writes 30-second WAV chunks to `./audio_chunks/`.
* A transcription consumer sends each chunk to a Wyoming ASR service.
* The transcript is stored in **mem0** and MongoDB.

"""

import asyncio
import concurrent.futures
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ollama  # Ollama python client
import openai
# import openrouter


from dotenv import load_dotenv
from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
from omi.decoder import OmiOpusDecoder  # OmiSDK
from pydantic import BaseModel
from typing import List, Dict, Any
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe
from wyoming.vad import VoiceStarted, VoiceStopped

import speaker_client as speaker_recognition
from memory import get_memory_service, init_memory_config, shutdown_memory_service

# Check if speaker service is available
SPEAKER_SERVICE_AVAILABLE = speaker_recognition.speaker_recognition is not None

###############################################################################
# SETUP
###############################################################################

# Load environment variables first
load_dotenv()

# Mem0 telemetry configuration is now handled in the memory module

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advanced-backend")
audio_logger = logging.getLogger("audio_processing")

# Conditional Deepgram import
try:
    from deepgram import DeepgramClient, FileSource, PrerecordedOptions
    DEEPGRAM_AVAILABLE = True
    logger.info("Deepgram avaialable")
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
CROPPING_CONTEXT_PADDING = float(os.getenv("CROPPING_CONTEXT_PADDING", "0.1"))  # seconds of padding around speech

# Directory where WAV chunks are written
CHUNK_DIR = Path("./audio_chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# ASR Configuration
OFFLINE_ASR_TCP_URI = os.getenv("OFFLINE_ASR_TCP_URI", "tcp://192.168.0.110:8765/")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Determine transcription strategy based on environment variables
USE_DEEPGRAM = bool(DEEPGRAM_API_KEY and DEEPGRAM_AVAILABLE)
if DEEPGRAM_API_KEY and not DEEPGRAM_AVAILABLE:
    audio_logger.error("DEEPGRAM_API_KEY provided but Deepgram SDK not available. Falling back to offline ASR.")
audio_logger.info(f"Transcription strategy: {'Deepgram' if USE_DEEPGRAM else 'Offline ASR'}")

# Deepgram client placeholder (not implemented)
deepgram_client = None
if USE_DEEPGRAM:
    audio_logger.warning("Deepgram transcription requested but not yet implemented. Falling back to offline ASR.")
    USE_DEEPGRAM = False

# Ollama & Qdrant Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://ollama:11434")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_CHOICE = os.getenv("LLM_CHOICE", "gpt-4o-mini")

# Memory configuration is now handled in the memory module
# Initialize it with our Ollama and Qdrant URLs
init_memory_config(
    ollama_base_url=OLLAMA_BASE_URL,
    qdrant_base_url=QDRANT_BASE_URL,
)

# Thread pool executors
_DEC_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=os.cpu_count() or 4,
    thread_name_prefix="opus_io",
)

# Initialize memory service and ollama client
memory_service = get_memory_service()
ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

# Initialize action items service
from action_items_service import ActionItemsService
action_items_service = ActionItemsService(action_items_col, ollama_client)

###############################################################################
# AUDIO PROCESSING FUNCTIONS
###############################################################################

async def _process_audio_cropping_with_relative_timestamps(
    original_path: str, 
    speech_segments: List[Tuple[float, float]], 
    output_path: str, 
    audio_uuid: str
) -> bool:
    """
    Process audio cropping with automatic relative timestamp conversion.
    This function handles both live processing and reprocessing scenarios.
    """
    try:
        # Convert absolute timestamps to relative timestamps
        # Extract file start time from filename: timestamp_client_uuid.wav
        filename = original_path.split('/')[-1]
        file_start_timestamp = float(filename.split('_')[0])
        
        # Convert speech segments to relative timestamps
        relative_segments = []
        for start_abs, end_abs in speech_segments:
            start_rel = start_abs - file_start_timestamp
            end_rel = end_abs - file_start_timestamp
            
            # Ensure relative timestamps are positive (sanity check)
            if start_rel < 0:
                audio_logger.warning(f"⚠️ Negative start timestamp: {start_rel}, clamping to 0.0")
                start_rel = 0.0
            if end_rel < 0:
                audio_logger.warning(f"⚠️ Negative end timestamp: {end_rel}, skipping segment")
                continue
                
            relative_segments.append((start_rel, end_rel))
        
        audio_logger.info(f"🕐 Converting timestamps for {audio_uuid}: file_start={file_start_timestamp}")
        audio_logger.info(f"🕐 Absolute segments: {speech_segments}")
        audio_logger.info(f"🕐 Relative segments: {relative_segments}")
        
        success = await _crop_audio_with_ffmpeg(original_path, relative_segments, output_path)
        if success:
            # Update database with cropped file info (keep original absolute timestamps for reference)
            cropped_filename = output_path.split('/')[-1]
            await chunk_repo.update_cropped_audio(audio_uuid, cropped_filename, speech_segments)
            audio_logger.info(f"Successfully processed cropped audio: {cropped_filename}")
            return True
        else:
            audio_logger.error(f"Failed to crop audio for {audio_uuid}")
            return False
    except Exception as e:
        audio_logger.error(f"Error in audio cropping task for {audio_uuid}: {e}")
        return False


async def _crop_audio_with_ffmpeg(original_path: str, speech_segments: List[Tuple[float, float]], output_path: str) -> bool:
    """Use ffmpeg to crop audio - runs as async subprocess, no GIL issues"""
    audio_cropper_logger.info(f"Cropping audio {original_path} with {len(speech_segments)} speech segments")
    
    if not AUDIO_CROPPING_ENABLED:
        audio_cropper_logger.info(f"Audio cropping disabled, skipping {original_path}")
        return False
    
    if not speech_segments:
        audio_cropper_logger.warning(f"No speech segments to crop for {original_path}")
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
            audio_cropper_logger.debug(f"Skipping short segment: {start}-{end} ({duration:.2f}s < {MIN_SPEECH_SEGMENT_DURATION}s)")
    
    if not filtered_segments:
        audio_cropper_logger.warning(f"No segments meet minimum duration ({MIN_SPEECH_SEGMENT_DURATION}s) for {original_path}")
        return False
        
    audio_cropper_logger.info(f"Cropping audio {original_path} with {len(filtered_segments)} speech segments (filtered from {len(speech_segments)})")
    
    try:
        # Build ffmpeg filter for concatenating speech segments
        filter_parts = []
        for i, (start, end) in enumerate(filtered_segments):
            duration = end - start
            filter_parts.append(f"[0:a]atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS[seg{i}]")
        
        # Concatenate all segments
        inputs = "".join(f"[seg{i}]" for i in range(len(filtered_segments)))
        concat_filter = f"{inputs}concat=n={len(filtered_segments)}:v=0:a=1[out]"
        
        full_filter = ";".join(filter_parts + [concat_filter])
        
        # Run ffmpeg as async subprocess
        cmd = [
            "ffmpeg", "-y",  # -y = overwrite output
            "-i", original_path,
            "-filter_complex", full_filter,
            "-map", "[out]",
            "-c:a", "pcm_s16le",  # Keep same format as original
            output_path
        ]
        
        audio_cropper_logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        if stdout:
            audio_cropper_logger.debug(f"FFMPEG stdout: {stdout.decode()}")
        
        if process.returncode == 0:
            # Calculate cropped duration
            cropped_duration = sum(end - start for start, end in filtered_segments)
            audio_cropper_logger.info(f"Successfully cropped {original_path} -> {output_path} ({cropped_duration:.1f}s from {len(filtered_segments)} segments)")
            return True
        else:
            error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
            audio_logger.error(f"ffmpeg failed for {original_path}: {error_msg}")
            return False
            
    except Exception as e:
        audio_logger.error(f"Error running ffmpeg on {original_path}: {e}")
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


async def _open_file_sink_properly(sink):
    """Open a file sink and ensure all wave parameters are set correctly."""
    await sink.open()
    # Ensure compression type is set immediately after opening
    if hasattr(sink, '_file_handle') and sink._file_handle:
        # Re-set parameters in the correct order to ensure they stick
        sink._file_handle.setnchannels(int(OMI_CHANNELS))
        sink._file_handle.setsampwidth(int(OMI_SAMPLE_WIDTH))
        sink._file_handle.setframerate(int(OMI_SAMPLE_RATE))
    return sink


# Memory processing is now handled by the memory service module

# Speaker recognition queue and worker
SPKR_QUEUE: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

async def speaker_worker():
    """Background worker for speaker diarization and verification."""
    
    if not SPEAKER_SERVICE_AVAILABLE:
        audio_logger.info("Speaker service not available - speaker worker will not process tasks")
        return
    
    while True:
        try:
            wav_path, audio_uuid = await SPKR_QUEUE.get()
                # Run speaker processing directly since it's already async
            assert speaker_recognition is not None
            await speaker_recognition.process_file(
                CHUNK_DIR / wav_path,
                audio_uuid,
                chunks_col,
            )
        except Exception as e:
            audio_logger.error("Speaker worker failed: %s", e)
        finally:
            SPKR_QUEUE.task_done()


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
            {"audio_uuid": audio_uuid}, 
            {"$push": {"transcript": transcript_segment}}
        )
    
    async def add_speaker(self, audio_uuid, speaker_id):
        """Add a speaker to the speakers_identified list if not already present."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$addToSet": {"speakers_identified": speaker_id}}
        )
    
    async def update_transcript(self, audio_uuid, full_transcript):
        """Update the entire transcript list (for compatibility)."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid}, 
            {"$set": {"transcript": full_transcript}}
        )
    
    async def update_segment_timing(self, audio_uuid, segment_index, start_time, end_time):
        """Update timing information for a specific transcript segment."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    f"transcript.{segment_index}.start": start_time,
                    f"transcript.{segment_index}.end": end_time
                }
            }
        )
    
    async def update_segment_speaker(self, audio_uuid, segment_index, speaker_id):
        """Update the speaker for a specific transcript segment."""
        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$set": {f"transcript.{segment_index}.speaker": speaker_id}}
        )
        if result.modified_count > 0:
            audio_logger.info(f"Updated segment {segment_index} speaker to {speaker_id} for {audio_uuid}")
        return result.modified_count > 0

    async def update_cropped_audio(self, audio_uuid: str, cropped_path: str, speech_segments: List[Tuple[float, float]]):
        """Update the chunk with cropped audio information."""
        cropped_duration = sum(end - start for start, end in speech_segments)
        
        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    "cropped_audio_path": cropped_path,
                    "speech_segments": [{"start": start, "end": end} for start, end in speech_segments],
                    "cropped_duration": cropped_duration,
                    "cropped_at": time.time()
                }
            }
        )
        if result.modified_count > 0:
            audio_logger.info(f"Updated cropped audio info for {audio_uuid}: {cropped_path}")
        return result.modified_count > 0


class TranscriptionManager:
    """Manages transcription using either Deepgram or offline ASR service."""

    def __init__(self):
        self.client = None
        self._current_audio_uuid = None
        self._streaming = False
        self.use_deepgram = USE_DEEPGRAM
        self.deepgram_client = deepgram_client
        self._audio_buffer = []  # Buffer for Deepgram batch processing

    async def connect(self):
        """Establish connection to ASR service (only for offline ASR)."""
        if self.use_deepgram:
            audio_logger.info("Using Deepgram transcription - no connection needed")
            return
            
        try:
            self.client = AsyncTcpClient.from_uri(OFFLINE_ASR_TCP_URI)
            await self.client.connect()
            audio_logger.info(f"Connected to offline ASR service at {OFFLINE_ASR_TCP_URI}")
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

    async def transcribe_chunk(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Transcribe a single chunk using either Deepgram or offline ASR."""
        if self.use_deepgram:
            await self._transcribe_chunk_deepgram(audio_uuid, chunk, client_id)
        else:
            await self._transcribe_chunk_offline(audio_uuid, chunk, client_id)

    async def _transcribe_chunk_deepgram(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Transcribe using Deepgram API."""
        raise NotImplementedError("Deepgram transcription is not yet implemented. Please use offline ASR by not setting DEEPGRAM_API_KEY.")

    async def _process_deepgram_buffer(self, audio_uuid: str, client_id: str):
        """Process buffered audio with Deepgram."""
        raise NotImplementedError("Deepgram transcription is not yet implemented.")

    async def _transcribe_chunk_offline(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Transcribe using offline ASR service."""
        if not self.client:
            audio_logger.error(f"No ASR connection available for {audio_uuid}")
            return

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
                    event = await asyncio.wait_for(self.client.read_event(), timeout=0.001) # this is a quick poll, feels like a better solution can exist
                    if event is None:
                        break
                        
                    if Transcript.is_type(event.type):
                        transcript_obj = Transcript.from_event(event)
                        transcript_text = transcript_obj.text.strip()
                        
                        # Handle both Transcript and StreamingTranscript types
                        # Check the 'final' attribute from the event data, not the reconstructed object
                        is_final = event.data.get('final', True)  # Default to True for standard Transcript
                        
                        # Only process final transcripts, ignore partial ones
                        if not is_final:
                            audio_logger.info(f"Ignoring partial transcript for {audio_uuid}: {transcript_text}")
                            continue
                        
                        if transcript_text:
                            audio_logger.info(f"Transcript for {audio_uuid}: {transcript_text} (final: {is_final})")
                            
                            # Create transcript segment with new format
                            transcript_segment = {
                                "speaker": f"speaker_{client_id}",
                                "text": transcript_text,
                                "start": 0.0,
                                "end": 0.0
                            }
                            
                            # Store transcript segment in DB immediately
                            await chunk_repo.add_transcript_segment(audio_uuid, transcript_segment)
                            await chunk_repo.add_speaker(audio_uuid, f"speaker_{client_id}")
                            audio_logger.info(f"Added transcript segment for {audio_uuid} to DB.")
                            
                            # Update transcript time for conversation timeout tracking
                            if client_id in active_clients:
                                active_clients[client_id].last_transcript_time = time.time()
                                # Collect transcript for end-of-conversation memory processing
                                active_clients[client_id].conversation_transcripts.append(transcript_text)
                                audio_logger.info(f"Added transcript to conversation collection: '{transcript_text}'")
                    
                    elif VoiceStarted.is_type(event.type):
                        audio_logger.info(f"VoiceStarted event received for {audio_uuid}")
                        current_time = time.time()
                        if client_id in active_clients:
                            active_clients[client_id].record_speech_start(audio_uuid, current_time)
                            audio_logger.info(f"🎤 Voice started for {audio_uuid} at {current_time}")
                    
                    elif VoiceStopped.is_type(event.type):
                        audio_logger.info(f"VoiceStopped event received for {audio_uuid}")
                        current_time = time.time()
                        if client_id in active_clients:
                            active_clients[client_id].record_speech_end(audio_uuid, current_time)
                            audio_logger.info(f"🔇 Voice stopped for {audio_uuid} at {current_time}")
                            
            except asyncio.TimeoutError:
                # No events available right now, that's fine
                pass
                
        except Exception as e:
            audio_logger.error(f"Error in offline transcribe_chunk for {audio_uuid}: {e}")
            # Attempt to reconnect on error
            await self._reconnect()

    async def _reconnect(self):
        """Attempt to reconnect to ASR service."""
        audio_logger.info("Attempting to reconnect to ASR service...")
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
        self.memory_queue = asyncio.Queue[Tuple[Optional[str], Optional[str], Optional[str]]]()  # (transcript, client_id, audio_uuid)
        
        # Per-client file sink
        self.file_sink: Optional[LocalFileSink] = None
        self.current_audio_uuid: Optional[str] = None
        
        # Per-client transcription manager
        self.transcription_manager: Optional[TranscriptionManager] = None
        
        # Conversation timeout tracking
        self.last_transcript_time: Optional[float] = None
        self.conversation_start_time: float = time.time()
        
        # Speech segment tracking for audio cropping
        self.speech_segments: Dict[str, List[Tuple[float, float]]] = {}  # audio_uuid -> [(start, end), ...]
        self.current_speech_start: Dict[str, Optional[float]] = {}  # audio_uuid -> start_time
        
        # Conversation transcript collection for end-of-conversation memory processing
        self.conversation_transcripts: List[str] = []  # Collect all transcripts for this conversation
        
        # Tasks for this client
        self.saver_task: Optional[asyncio.Task] = None
        self.transcription_task: Optional[asyncio.Task] = None
        self.memory_task: Optional[asyncio.Task] = None
    
    def record_speech_start(self, audio_uuid: str, timestamp: float):
        """Record the start of a speech segment."""
        self.current_speech_start[audio_uuid] = timestamp
        audio_logger.info(f"Recorded speech start for {audio_uuid}: {timestamp}")
        
    def record_speech_end(self, audio_uuid: str, timestamp: float):
        """Record the end of a speech segment."""
        if audio_uuid in self.current_speech_start and self.current_speech_start[audio_uuid] is not None:
            start_time = self.current_speech_start[audio_uuid]
            if start_time is not None:  # Type guard
                if audio_uuid not in self.speech_segments:
                    self.speech_segments[audio_uuid] = []
                self.speech_segments[audio_uuid].append((start_time, timestamp))
                self.current_speech_start[audio_uuid] = None
                duration = timestamp - start_time
                audio_logger.info(f"Recorded speech segment for {audio_uuid}: {start_time:.3f} -> {timestamp:.3f} (duration: {duration:.3f}s)")
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
        
        # Wait for tasks to complete
        if self.saver_task:
            await self.saver_task
        if self.transcription_task:
            await self.transcription_task
        if self.memory_task:
            await self.memory_task
            
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
            current_path = self.file_sink._file_path.name if hasattr(self.file_sink, '_file_path') else None
            
            audio_logger.info(f"🔒 Closing conversation {current_uuid}, file: {current_path}")
            
            # Process memory at end of conversation if we have transcripts
            if self.conversation_transcripts and current_uuid:
                full_conversation = " ".join(self.conversation_transcripts)
                audio_logger.info(f"💭 Processing memory for conversation {current_uuid} with {len(self.conversation_transcripts)} transcript segments")
                audio_logger.info(f"💭 Full conversation text: {full_conversation[:200]}...")  # Log first 200 chars
                 
                try:
                    # Add general memory
                    success = memory_service.add_memory(full_conversation, self.client_id, current_uuid)
                    if success:
                        audio_logger.info(f"✅ Successfully added conversation memory for {current_uuid}")
                    else:
                        audio_logger.error(f"❌ Failed to add conversation memory for {current_uuid}")
                    
                    # Extract and store action items from the full conversation using new MongoDB service
                    action_item_count = await action_items_service.extract_and_store_action_items(full_conversation, self.client_id, current_uuid)
                    if action_item_count > 0:
                        audio_logger.info(f"🎯 Extracted {action_item_count} action items from conversation {current_uuid}")
                    else:
                        audio_logger.info(f"ℹ️ No action items found in conversation {current_uuid}")
                        
                except Exception as e:
                    audio_logger.error(f"❌ Error processing memory and action items for {current_uuid}: {e}")
            else:
                audio_logger.info(f"ℹ️ No transcripts to process for memory in conversation {current_uuid}")
            
            await self.file_sink.close()
            self.file_sink = None
            
            # Process audio cropping if we have speech segments
            if current_uuid and current_path:
                if current_uuid in self.speech_segments:
                    speech_segments = self.speech_segments[current_uuid]
                    audio_logger.info(f"🎯 Found {len(speech_segments)} speech segments for {current_uuid}: {speech_segments}")
                    if speech_segments:  # Only crop if we have speech segments
                        cropped_path = current_path.replace('.wav', '_cropped.wav')
                        
                        # Process in background - won't block
                        asyncio.create_task(self._process_audio_cropping(
                            f"{CHUNK_DIR}/{current_path}",
                            speech_segments, 
                            f"{CHUNK_DIR}/{cropped_path}",
                            current_uuid
                        ))
                        audio_logger.info(f"✂️ Queued audio cropping for {current_path} with {len(speech_segments)} speech segments")
                    else:
                        audio_logger.info(f"⚠️ Empty speech segments list found for {current_path}, skipping cropping")
                    
                    # Clean up segments for this conversation
                    del self.speech_segments[current_uuid]
                    if current_uuid in self.current_speech_start:
                        del self.current_speech_start[current_uuid]
                else:
                    audio_logger.info(f"⚠️ No speech segments found for {current_path} (uuid: {current_uuid}), skipping cropping")
            
                # Queue for speaker processing if we have a completed file and speaker service is available
                if SPEAKER_SERVICE_AVAILABLE:
                    await SPKR_QUEUE.put((current_path, current_uuid))
                    audio_logger.info(f"🎭 Queued {current_path} for speaker processing")
                else:
                    audio_logger.debug(f"Speaker service not available - skipping speaker processing for {current_path}")
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
        
        audio_logger.info(f"Client {self.client_id}: Started new conversation due to {NEW_CONVERSATION_TIMEOUT_MINUTES}min timeout")
    
    async def _process_audio_cropping(self, original_path: str, speech_segments: List[Tuple[float, float]], output_path: str, audio_uuid: str):
        """Background task for audio cropping using ffmpeg."""
        await _process_audio_cropping_with_relative_timestamps(original_path, speech_segments, output_path, audio_uuid)
    
    async def _audio_saver(self):
        """Per-client audio saver consumer."""
        try:
            while self.connected:
                audio_chunk = await self.chunk_queue.get()
                
                if audio_chunk is None:  # Disconnect signal
                    break
                
                # Check if we should start a new conversation due to timeout
                if self._should_start_new_conversation():
                    await self.start_new_conversation()
                    
                if self.file_sink is None:
                    # Create new file sink for this client
                    self.current_audio_uuid = uuid.uuid4().hex
                    timestamp = audio_chunk.timestamp or int(time.time())
                    wav_filename = f"{timestamp}_{self.client_id}_{self.current_audio_uuid}.wav"
                    audio_logger.info(f"Creating file sink with: rate={int(OMI_SAMPLE_RATE)}, channels={int(OMI_CHANNELS)}, width={int(OMI_SAMPLE_WIDTH)}")
                    self.file_sink = _new_local_file_sink(f"{CHUNK_DIR}/{wav_filename}")
                    try:
                        await _open_file_sink_properly(self.file_sink)
                        audio_logger.info(f"File sink opened successfully for {wav_filename}")
                    except Exception as e:
                        audio_logger.error(f"Failed to open file sink: {e}")
                        raise
                    
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
            audio_logger.error(f"Error in audio saver for client {self.client_id}: {e}", exc_info=True)
        finally:
            # Close current conversation with all processing when audio saver ends
            await self._close_current_conversation()
    
    async def _transcription_processor(self):
        """Per-client transcription processor."""
        try:
            while self.connected:
                audio_uuid, chunk = await self.transcription_queue.get()
                
                if audio_uuid is None or chunk is None:  # Disconnect signal
                    break
                
                # Get or create transcription manager
                if self.transcription_manager is None:
                    self.transcription_manager = TranscriptionManager()
                    try:
                        await self.transcription_manager.connect()
                    except Exception as e:
                        audio_logger.error(f"Failed to create transcription manager for client {self.client_id}: {e}")
                        continue
                
                # Process transcription
                try:
                    await self.transcription_manager.transcribe_chunk(audio_uuid, chunk, self.client_id)
                except Exception as e:
                    audio_logger.error(f"Error transcribing for client {self.client_id}: {e}")
                    # Recreate transcription manager on error
                    if self.transcription_manager:
                        await self.transcription_manager.disconnect()
                        self.transcription_manager = None
                        
        except Exception as e:
            audio_logger.error(f"Error in transcription processor for client {self.client_id}: {e}", exc_info=True)
    
    async def _memory_processor(self):
        """Per-client memory processor - currently unused as memory processing happens at conversation end."""
        try:
            while self.connected:
                transcript, client_id, audio_uuid = await self.memory_queue.get()
                
                if transcript is None or client_id is None or audio_uuid is None:  # Disconnect signal
                    break
                
                # Memory processing now happens at conversation end, so this is effectively a no-op
                # Keeping the processor running to avoid breaking the queue system
                audio_logger.debug(f"Memory processor received item but processing is now done at conversation end")
                        
        except Exception as e:
            audio_logger.error(f"Error in memory processor for client {self.client_id}: {e}", exc_info=True)


# Initialize repository and global state
chunk_repo = ChunkRepo(chunks_col)
active_clients: dict[str, ClientState] = {}


async def create_client_state(client_id: str) -> ClientState:
    """Create and register a new client state."""
    client_state = ClientState(client_id)
    active_clients[client_id] = client_state
    await client_state.start_processing()
    return client_state


async def cleanup_client_state(client_id: str):
    """Clean up and remove client state."""
    if client_id in active_clients:
        client_state = active_clients[client_id]
        await client_state.disconnect()
        del active_clients[client_id]


###############################################################################
# CORE APPLICATION LOGIC
###############################################################################

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    audio_logger.info("Starting application...")
    
    if SPEAKER_SERVICE_AVAILABLE:
        asyncio.create_task(speaker_worker(), name="speaker-worker")
        audio_logger.info("Speaker recognition worker started")
    else:
        audio_logger.info("Speaker service not available - skipping speaker worker")
        
    audio_logger.info("Application ready - clients will have individual processing pipelines.")

    try:
        yield
    finally:
        # Shutdown
        audio_logger.info("Shutting down application...")

        # Clean up all active clients
        for client_id in list(active_clients.keys()):
            await cleanup_client_state(client_id)
        
        # Shutdown memory service
        shutdown_memory_service()
        audio_logger.info("Memory service shut down.")
        
        audio_logger.info("Shutdown complete.")


# FastAPI Application
app = FastAPI(lifespan=lifespan)
app.mount("/audio", StaticFiles(directory=CHUNK_DIR), name="audio")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, user_id: Optional[str] = Query(None)):
    """Accepts WebSocket connections, decodes Opus audio, and processes per-client."""
    audio_logger.info(f"WebSocket connection hit")
    await ws.accept()

    # Use user_id if provided, otherwise generate a random client_id
    client_id = user_id if user_id else f"client_{uuid.uuid4().hex[:8]}"
    audio_logger.info(f"Client {client_id}: WebSocket connection accepted (user_id: {user_id}).")
    decoder = OmiOpusDecoder()
    _decode_packet = partial(decoder.decode_packet, strip_header=False)
    
    # Create client state and start processing
    client_state = await create_client_state(client_id)

    try:
        while True:
            packet = await ws.receive_bytes()
            loop = asyncio.get_running_loop()
            pcm_data = await loop.run_in_executor(
                _DEC_IO_EXECUTOR, _decode_packet, packet
            )
            if pcm_data:
                audio_logger.debug(f"Received {len(pcm_data)} bytes of PCM data")
                chunk = AudioChunk(
                    audio=pcm_data,
                    rate=OMI_SAMPLE_RATE,
                    width=OMI_SAMPLE_WIDTH,
                    channels=OMI_CHANNELS,
                    timestamp=int(time.time()),
                )
                await client_state.chunk_queue.put(chunk)

    except WebSocketDisconnect:
        audio_logger.info(f"Client {client_id}: WebSocket disconnected.")
    except Exception as e:
        audio_logger.error(f"Client {client_id}: An error occurred: {e}", exc_info=True)
    finally:
        # Clean up client state
        await cleanup_client_state(client_id)


@app.websocket("/ws_pcm")
async def ws_endpoint_pcm(ws: WebSocket, user_id: Optional[str] = Query(None)):
    """Accepts WebSocket connections, processes PCM audio per-client."""
    await ws.accept()
    
    # Use user_id if provided, otherwise generate a random client_id
    client_id = user_id if user_id else f"client_{uuid.uuid4().hex[:8]}"
    audio_logger.info(f"Client {client_id}: WebSocket connection accepted (user_id: {user_id}).")
    
    # Create client state and start processing
    client_state = await create_client_state(client_id)

    try:
        while True:
            packet = await ws.receive_bytes()
            if packet:
                chunk = AudioChunk(
                    audio=packet,
                    rate=16000,
                    width=2,
                    channels=1,
                    timestamp=int(time.time()),
                )
                await client_state.chunk_queue.put(chunk)
    except WebSocketDisconnect:
        audio_logger.info(f"Client {client_id}: WebSocket disconnected.")
    except Exception as e:
        audio_logger.error(f"Client {client_id}: An error occurred: {e}", exc_info=True)
    finally:
        # Clean up client state
        await cleanup_client_state(client_id)


@app.get("/api/conversations")
async def get_conversations():
    """Get all conversations grouped by client_id."""
    try:
        # Get all audio chunks and group by client_id
        cursor = chunks_col.find({}).sort("timestamp", -1)
        conversations = {}
        
        async for chunk in cursor:
            client_id = chunk.get("client_id", "unknown")
            if client_id not in conversations:
                conversations[client_id] = []
            
            conversations[client_id].append({
                "audio_uuid": chunk["audio_uuid"],
                "audio_path": chunk["audio_path"],
                "cropped_audio_path": chunk.get("cropped_audio_path"),
                "timestamp": chunk["timestamp"],
                "transcript": chunk.get("transcript", []),
                "speakers_identified": chunk.get("speakers_identified", []),
                "speech_segments": chunk.get("speech_segments", []),
                "cropped_duration": chunk.get("cropped_duration")
            })
        
        return {"conversations": conversations}
    except Exception as e:
        audio_logger.error(f"Error getting conversations: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/conversations/{audio_uuid}/cropped")
async def get_cropped_audio_info(audio_uuid: str):
    """Get cropped audio information for a specific conversation."""
    try:
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})
        
        return {
            "audio_uuid": audio_uuid,
            "original_audio_path": chunk["audio_path"],
            "cropped_audio_path": chunk.get("cropped_audio_path"),
            "speech_segments": chunk.get("speech_segments", []),
            "cropped_duration": chunk.get("cropped_duration"),
            "cropped_at": chunk.get("cropped_at"),
            "has_cropped_version": bool(chunk.get("cropped_audio_path"))
        }
    except Exception as e:
        audio_logger.error(f"Error getting cropped audio info: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/conversations/{audio_uuid}/reprocess")  
async def reprocess_audio_cropping(audio_uuid: str):
    """Trigger reprocessing of audio cropping for a specific conversation."""
    try:
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})
        
        original_path = f"{CHUNK_DIR}/{chunk['audio_path']}"
        if not Path(original_path).exists():
            return JSONResponse(status_code=404, content={"error": "Original audio file not found"})
        
        # Check if we have speech segments
        speech_segments = chunk.get("speech_segments", [])
        if not speech_segments:
            return JSONResponse(status_code=400, content={"error": "No speech segments available for cropping"})
        
        # Convert speech segments from dict format to tuple format  
        speech_segments_tuples = [(seg["start"], seg["end"]) for seg in speech_segments]
        
        cropped_filename = chunk['audio_path'].replace('.wav', '_cropped.wav')
        cropped_path = f"{CHUNK_DIR}/{cropped_filename}"
        
        # Process in background using shared logic
        async def reprocess_task():
            audio_logger.info(f"🔄 Starting reprocess for {audio_uuid}")
            await _process_audio_cropping_with_relative_timestamps(original_path, speech_segments_tuples, cropped_path, audio_uuid)
        
        asyncio.create_task(reprocess_task())
        
        return {"message": "Reprocessing started", "audio_uuid": audio_uuid}
    except Exception as e:
        audio_logger.error(f"Error reprocessing audio: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/users")
async def get_users():
    """Retrieves all users from the database."""
    try:
        cursor = users_col.find()
        users = []
        for doc in await cursor.to_list(length=100):
            doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
            users.append(doc)
        return JSONResponse(content=users)
    except Exception as e:
        audio_logger.error(f"Error fetching users: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error fetching users"}
        )


@app.post("/api/create_user")
async def create_user(user_id: str):
    """Creates a new user in the database."""
    try:
        # Check if user already exists
        existing_user = await users_col.find_one({"user_id": user_id})
        if existing_user:
            return JSONResponse(
                status_code=409, 
                content={"message": f"User {user_id} already exists"}
            )
        
        # Create new user
        result = await users_col.insert_one({"user_id": user_id})
        return JSONResponse(
            status_code=201,
            content={"message": f"User {user_id} created successfully", "id": str(result.inserted_id)}
        )
    except Exception as e:
        audio_logger.error(f"Error creating user: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Error creating user"}
        )


@app.delete("/api/delete_user")
async def delete_user(user_id: str, delete_conversations: bool = False, delete_memories: bool = False):
    """Deletes a user from the database with optional data cleanup."""
    try:
        # Check if user exists
        existing_user = await users_col.find_one({"user_id": user_id})
        if not existing_user:
            return JSONResponse(
                status_code=404,
                content={"message": f"User {user_id} not found"}
            )
        
        deleted_data = {}
        
        # Delete user from users collection
        user_result = await users_col.delete_one({"user_id": user_id})
        deleted_data["user_deleted"] = user_result.deleted_count > 0
        
        if delete_conversations:
            # Delete all conversations (audio chunks) for this user
            conversations_result = await chunks_col.delete_many({"client_id": user_id})
            deleted_data["conversations_deleted"] = conversations_result.deleted_count
        
        if delete_memories:
            # Delete all memories for this user using the memory service
            try:
                memory_count = memory_service.delete_all_user_memories(user_id)
                deleted_data["memories_deleted"] = memory_count
            except Exception as mem_error:
                audio_logger.error(f"Error deleting memories for user {user_id}: {mem_error}")
                deleted_data["memories_deleted"] = 0
                deleted_data["memory_deletion_error"] = str(mem_error)
        
        # Build message based on what was deleted
        message = f"User {user_id} deleted successfully"
        deleted_items = []
        if delete_conversations and deleted_data.get('conversations_deleted', 0) > 0:
            deleted_items.append(f"{deleted_data['conversations_deleted']} conversations")
        if delete_memories and deleted_data.get('memories_deleted', 0) > 0:
            deleted_items.append(f"{deleted_data['memories_deleted']} memories")
        
        if deleted_items:
            message += f" along with {' and '.join(deleted_items)}"
        
        return JSONResponse(
            status_code=200,
            content={
                "message": message,
                "deleted_data": deleted_data
            }
        )
    except Exception as e:
        audio_logger.error(f"Error deleting user: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Error deleting user"}
        )


@app.get("/api/memories")
async def get_memories(user_id: str, limit: int = 100):
    """Retrieves memories from the mem0 store with optional filtering."""
    try:
        all_memories = memory_service.get_all_memories(user_id=user_id, limit=limit)
        return JSONResponse(content=all_memories)
    except Exception as e:
        audio_logger.error(f"Error fetching memories: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error fetching memories"}
        )


@app.get("/api/memories/search")
async def search_memories(user_id: str, query: str, limit: int = 10):
    """Search memories using semantic similarity for better retrieval."""
    try:
        relevant_memories = memory_service.search_memories(query=query, user_id=user_id, limit=limit)
        return JSONResponse(content=relevant_memories)
    except Exception as e:
        audio_logger.error(f"Error searching memories: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error searching memories"}
        )


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a specific memory by ID."""
    try:
        memory_service.delete_memory(memory_id=memory_id)
        return JSONResponse(
            content={"message": f"Memory {memory_id} deleted successfully"}
        )
    except Exception as e:
        audio_logger.error(f"Error deleting memory {memory_id}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error deleting memory"}
        )


@app.post("/api/conversations/{audio_uuid}/speakers")
async def add_speaker_to_conversation(audio_uuid: str, speaker_id: str):
    """Add a speaker to the speakers_identified list for a conversation."""
    try:
        await chunk_repo.add_speaker(audio_uuid, speaker_id)
        return JSONResponse(
            content={"message": f"Speaker {speaker_id} added to conversation {audio_uuid}"}
        )
    except Exception as e:
        audio_logger.error(f"Error adding speaker: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error adding speaker"}
        )


@app.put("/api/conversations/{audio_uuid}/transcript/{segment_index}")
async def update_transcript_segment(
    audio_uuid: str, 
    segment_index: int,
    speaker_id: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
):
    """Update a specific transcript segment with speaker or timing information."""
    try:
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
            return JSONResponse(
                status_code=400,
                content={"error": "No update parameters provided"}
            )
        
        result = await chunks_col.update_one(
            {"audio_uuid": audio_uuid},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            return JSONResponse(
                status_code=404,
                content={"error": "Conversation not found"}
            )
        
        return JSONResponse(content={"message": "Transcript segment updated successfully"})
        
    except Exception as e:
        audio_logger.error(f"Error updating transcript segment: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

class SpeakerEnrollmentRequest(BaseModel):
    speaker_id: str
    speaker_name: str
    audio_file_path: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class SpeakerIdentificationRequest(BaseModel):
    audio_file_path: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class ActionItemUpdateRequest(BaseModel):
    status: str  # "open", "in_progress", "completed", "cancelled"

class ActionItemCreateRequest(BaseModel):
    description: str
    assignee: Optional[str] = "unassigned"
    due_date: Optional[str] = "not_specified"
    priority: Optional[str] = "medium"
    context: Optional[str] = ""

@app.post("/api/speakers/enroll")
async def enroll_speaker(request: SpeakerEnrollmentRequest):
    """
    Enroll a new speaker from an audio file.
    
    Args:
        request: SpeakerEnrollmentRequest containing speaker_id, speaker_name, audio_file_path, start_time, end_time
    """
    if not SPEAKER_SERVICE_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Speaker service is not available. Please set SPEAKER_SERVICE_URL environment variable."}
        )
        
    try:
        # Full path to audio file
        full_audio_path = CHUNK_DIR / request.audio_file_path
        
        if not full_audio_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Audio file not found: {request.audio_file_path}"}
            )
        
        # Enroll speaker using speaker_recognition module
        success = speaker_recognition.enroll_speaker(
            speaker_id=request.speaker_id,
            speaker_name=request.speaker_name,
            audio_file=str(full_audio_path),
            start_time=request.start_time,
            end_time=request.end_time
        )
        
        if success:
            # Store speaker info in MongoDB
            speaker_doc = {
                "speaker_id": request.speaker_id,
                "speaker_name": request.speaker_name,
                "audio_file_path": request.audio_file_path,
                "start_time": request.start_time,
                "end_time": request.end_time,
                "enrolled_at": time.time()
            }
            
            await speakers_col.insert_one(speaker_doc)
            
            return JSONResponse(content={
                "message": f"Speaker {request.speaker_id} enrolled successfully",
                "speaker_id": request.speaker_id,
                "speaker_name": request.speaker_name
            })
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to enroll speaker"}
            )
            
    except Exception as e:
        audio_logger.error(f"Error enrolling speaker: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.get("/api/speakers")
async def list_speakers():
    """Get list of all enrolled speakers."""
    if not SPEAKER_SERVICE_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Speaker service is not available. Please set SPEAKER_SERVICE_URL environment variable."}
        )
        
    try:
        # Get speakers from speaker_recognition module
        enrolled_speakers = speaker_recognition.list_enrolled_speakers()
        
        # Get additional info from MongoDB
        mongo_speakers = []
        async for speaker in speakers_col.find({}, {"_id": 0}):
            mongo_speakers.append(speaker)
        
        # Combine information
        speakers_map = {s["speaker_id"]: s for s in mongo_speakers}
        
        result = []
        for speaker in enrolled_speakers:
            speaker_info = speakers_map.get(speaker["speaker_id"], {})
            result.append({
                "id": speaker["speaker_id"],
                "name": speaker["speaker_name"],
                "audio_file_path": speaker_info.get("audio_file_path"),
                "enrolled_at": speaker_info.get("enrolled_at")
            })
        
        return JSONResponse(content={"speakers": result})
        
    except Exception as e:
        audio_logger.error(f"Error listing speakers: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.delete("/api/speakers/{speaker_id}")
async def remove_speaker(speaker_id: str):
    """Remove an enrolled speaker."""
    if not SPEAKER_SERVICE_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Speaker service is not available. Please set SPEAKER_SERVICE_URL environment variable."}
        )
        
    try:
        # Remove from speaker_recognition module
        success = speaker_recognition.remove_speaker(speaker_id)
        
        if success:
            # Remove from MongoDB
            await speakers_col.delete_one({"speaker_id": speaker_id})
            
            return JSONResponse(content={
                "message": f"Speaker {speaker_id} removed successfully"
            })
        else:
            return JSONResponse(
                status_code=404,
                content={"error": f"Speaker {speaker_id} not found"}
            )
            
    except Exception as e:
        audio_logger.error(f"Error removing speaker: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.get("/api/speakers/{speaker_id}")
async def get_speaker_info(speaker_id: str):
    """Get detailed information about a specific speaker."""
    try:
        # Get from MongoDB
        speaker = await speakers_col.find_one({"speaker_id": speaker_id}, {"_id": 0})
        
        if not speaker:
            return JSONResponse(
                status_code=404,
                content={"error": f"Speaker {speaker_id} not found"}
            )
        
        return JSONResponse(content=speaker)
        
    except Exception as e:
        audio_logger.error(f"Error getting speaker info: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.post("/api/speakers/identify")
async def identify_speaker_from_file(request: SpeakerIdentificationRequest):
    """
    Identify a speaker from an audio file segment.
    
    Args:
        request: SpeakerIdentificationRequest containing audio_file_path, start_time, end_time
    """
    if not SPEAKER_SERVICE_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Speaker service is not available. Please set SPEAKER_SERVICE_URL environment variable."}
        )
        
    try:
        # Full path to audio file
        full_audio_path = CHUNK_DIR / request.audio_file_path
        
        if not full_audio_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Audio file not found: {request.audio_file_path}"}
            )
        
        # Use speaker_recognition module's audio loading and embedding extraction
        if not speaker_recognition.audio_loader or not speaker_recognition.embedding_model:
            return JSONResponse(
                status_code=503,
                content={"error": "Speaker recognition models not available"}
            )
        
        # Load audio
        if request.start_time is not None and request.end_time is not None:
            from pyannote.core import Segment
            segment = Segment(request.start_time, request.end_time)
            waveform, _ = speaker_recognition.audio_loader.crop(str(full_audio_path), segment)
        else:
            waveform, _ = speaker_recognition.audio_loader(str(full_audio_path))
        
        # Extract and normalize embedding
        waveform = waveform.unsqueeze(0)  # Add batch dimension
        embedding = speaker_recognition.embedding_model(waveform)
        embedding = speaker_recognition.normalize_embedding(embedding)
        
        # Identify speaker
        identified_speaker = speaker_recognition.identify_speaker(embedding[0])
        
        if identified_speaker:
            # Get speaker info
            speaker_info = await speakers_col.find_one({"speaker_id": identified_speaker}, {"_id": 0})
            
            return JSONResponse(content={
                "identified": True,
                "speaker_id": identified_speaker,
                "speaker_info": speaker_info
            })
        else:
            return JSONResponse(content={
                "identified": False,
                "message": "No matching speaker found"
            })
        
    except Exception as e:
        audio_logger.error(f"Error identifying speaker: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.get("/api/action-items")
async def get_action_items(user_id: str, status: Optional[str] = None, limit: int = 50):
    """Get action items for a user with optional status filtering."""
    try:
        action_items = await action_items_service.get_action_items(user_id=user_id, limit=limit, status_filter=status)
        return JSONResponse(content={
            "action_items": action_items,
            "count": len(action_items),
            "user_id": user_id,
            "status_filter": status
        })
    except Exception as e:
        audio_logger.error(f"Error fetching action items for user {user_id}: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": "Failed to fetch action items"}
        )


@app.get("/api/action-items/search")
async def search_action_items(user_id: str, query: str, limit: int = 20):
    """Search action items by text query."""
    try:
        action_items = await action_items_service.search_action_items(query=query, user_id=user_id, limit=limit)
        return JSONResponse(content={
            "action_items": action_items,
            "count": len(action_items),
            "query": query,
            "user_id": user_id
        })
    except Exception as e:
        audio_logger.error(f"Error searching action items for user {user_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to search action items"}
        )


@app.post("/api/action-items")
async def create_action_item(user_id: str, request: ActionItemCreateRequest):
    """Manually create a new action item."""
    try:
        # Create action item using new MongoDB service
        action_item = await action_items_service.create_action_item(
            user_id=user_id,
            description=request.description,
            assignee=request.assignee,
            due_date=request.due_date,
            priority=request.priority,
            context=request.context
        )
        
        if action_item:
            return JSONResponse(content={
                "message": "Action item created successfully",
                "action_item": action_item
            })
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to create action item"}
            )
            
    except Exception as e:
        audio_logger.error(f"Error creating action item for user {user_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to create action item"}
        )


@app.put("/api/action-items/{action_item_id}")
async def update_action_item_status(action_item_id: str, request: ActionItemUpdateRequest):
    """Update the status of an action item."""
    try:
        # Validate status
        valid_statuses = ["open", "in_progress", "completed", "cancelled"]
        if request.status not in valid_statuses:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid status. Must be one of: {', '.join(valid_statuses)}"}
            )
        
        success = await action_items_service.update_action_item_status(action_item_id, request.status)
        
        if success:
            return JSONResponse(content={
                "message": f"Action item status updated to {request.status}",
                "action_item_id": action_item_id,
                "new_status": request.status
            })
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Action item not found"}
            )
            
    except Exception as e:
        audio_logger.error(f"Error updating action item {action_item_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to update action item"}
        )


@app.delete("/api/action-items/{action_item_id}")
async def delete_action_item(action_item_id: str):
    """Delete an action item."""
    try:
        success = await action_items_service.delete_action_item(action_item_id)
        
        if success:
            return JSONResponse(content={
                "message": "Action item deleted successfully",
                "action_item_id": action_item_id
            })
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Action item not found"}
            )
            
    except Exception as e:
        audio_logger.error(f"Error deleting action item {action_item_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to delete action item"}
        )


@app.get("/api/action-items/stats")
async def get_action_item_stats(user_id: str):
    """Get action item statistics for a user."""
    try:
        # Get statistics using new MongoDB service
        stats = await action_items_service.get_action_item_stats(user_id=user_id)
        
        return JSONResponse(content={
            "user_id": user_id,
            "statistics": stats
        })
        
    except Exception as e:
        audio_logger.error(f"Error getting action item stats for user {user_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get action item statistics"}
        )


@app.post("/api/conversations/{audio_uuid}/extract-action-items")
async def extract_action_items_from_conversation(audio_uuid: str):
    """Manually trigger action item extraction for a specific conversation."""
    try:
        # Get the conversation from MongoDB
        conversation = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not conversation:
            return JSONResponse(
                status_code=404,
                content={"error": "Conversation not found"}
            )
        
        # Build full transcript from segments
        transcript_segments = conversation.get("transcript", [])
        if not transcript_segments:
            return JSONResponse(
                status_code=400,
                content={"error": "No transcript available for this conversation"}
            )
        
        # Combine all transcript segments into one text
        full_transcript = " ".join([
            segment.get("text", "") for segment in transcript_segments
            if isinstance(segment, dict) and segment.get("text")
        ])
        
        if not full_transcript.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Empty transcript for this conversation"}
            )
        
        client_id = conversation.get("client_id")
        if not client_id:
            return JSONResponse(
                status_code=400,
                content={"error": "No client_id found for this conversation"}
            )
        
        # Extract action items using new MongoDB service
        action_item_count = await action_items_service.extract_and_store_action_items(
            full_transcript, client_id, audio_uuid
        )
        
        return JSONResponse(content={
            "message": f"Extracted {action_item_count} action items from conversation",
            "audio_uuid": audio_uuid,
            "action_items_extracted": action_item_count,
            "transcript_length": len(full_transcript)
        })
        
    except Exception as e:
        audio_logger.error(f"Error extracting action items from conversation {audio_uuid}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to extract action items"}
        )


@app.get("/health")
async def health_check():
    """Comprehensive health check for all services."""
    health_status = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "services": {},
        "config": {
            "mongodb_uri": MONGODB_URI,
            "llm_url": LLM_BASE_URL if LLM_BASE_URL else "Not configured",
            "qdrant_url": f"http://{QDRANT_BASE_URL}:6333" if QDRANT_BASE_URL else "Not configured",
            "asr_uri": OFFLINE_ASR_TCP_URI,
            "chunk_dir": str(CHUNK_DIR),
            "active_clients": len(active_clients),
            "new_conversation_timeout_minutes": NEW_CONVERSATION_TIMEOUT_MINUTES,
            "action_items_enabled": True,
            "audio_cropping_enabled": AUDIO_CROPPING_ENABLED
        }
    }
    
    overall_healthy = True
    critical_services_healthy = True
    
    # Check MongoDB (critical service)
    try:
        await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=5.0)
        health_status["services"]["mongodb"] = {
            "status": "✅ Connected",
            "healthy": True,
            "critical": True
        }
    except asyncio.TimeoutError:
        health_status["services"]["mongodb"] = {
            "status": "❌ Connection Timeout (5s)",
            "healthy": False,
            "critical": True
        }
        overall_healthy = False
        critical_services_healthy = False
    except Exception as e:
        health_status["services"]["mongodb"] = {
            "status": f"❌ Connection Failed: {str(e)}",
            "healthy": False,
            "critical": True
        }
        overall_healthy = False
        critical_services_healthy = False
    
    # Check Ollama (non-critical service - may not be running)
    try:
        # Run in executor to avoid blocking the main thread
        loop = asyncio.get_running_loop()
        models = await asyncio.wait_for(
            loop.run_in_executor(None, ollama_client.list), 
            timeout=8.0
        )
        model_count = len(models.get('models', []))
        print(f"model_count: {model_count}")
        health_status["services"]["ollama"] = {
            "status": "✅ Connected",
            "healthy": True,
            "models": model_count,
            "critical": False
        }
    except asyncio.TimeoutError:
        health_status["services"]["ollama"] = {
            "status": "⚠️ Connection Timeout (8s) - Service may not be running",
            "healthy": False,
            "critical": False
        }
        overall_healthy = False
    except Exception as e:
        health_status["services"]["ollama"] = {
            "status": f"⚠️ Connection Failed: {str(e)} - Service may not be running",
            "healthy": False,
            "critical": False
        }
        overall_healthy = False
    
    # Check mem0 (depends on Ollama and Qdrant)
    try:
        # Test memory service connection with timeout
        test_success = memory_service.test_connection()
        if test_success:
            health_status["services"]["mem0"] = {
                "status": "✅ Connected",
                "healthy": True,
                "critical": False
            }
        else:
            health_status["services"]["mem0"] = {
                "status": "⚠️ Connection Test Failed",
                "healthy": False,
                "critical": False
            }
            overall_healthy = False
    except asyncio.TimeoutError:
        health_status["services"]["mem0"] = {
            "status": "⚠️ Connection Test Timeout (10s) - Depends on Ollama/Qdrant",
            "healthy": False,
            "critical": False
        }
        overall_healthy = False
    except Exception as e:
        health_status["services"]["mem0"] = {
            "status": f"⚠️ Connection Test Failed: {str(e)} - Check Ollama/Qdrant services",
            "healthy": False,
            "critical": False
        }
        overall_healthy = False
    
    # Check ASR service (non-critical - may be external)
    try:
        test_client = AsyncTcpClient.from_uri(OFFLINE_ASR_TCP_URI)
        await asyncio.wait_for(test_client.connect(), timeout=5.0)
        await test_client.disconnect()
        health_status["services"]["asr"] = {
            "status": "✅ Connected",
            "healthy": True,
            "uri": OFFLINE_ASR_TCP_URI,
            "critical": False
        }
    except asyncio.TimeoutError:
        health_status["services"]["asr"] = {
            "status": f"⚠️ Connection Timeout (5s) - Check external ASR service",
            "healthy": False,
            "uri": OFFLINE_ASR_TCP_URI,
            "critical": False
        }
        overall_healthy = False
    except Exception as e:
        health_status["services"]["asr"] = {
            "status": f"⚠️ Connection Failed: {str(e)} - Check external ASR service",
            "healthy": False,
            "uri": OFFLINE_ASR_TCP_URI,
            "critical": False
        }
        overall_healthy = False
    
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
            messages.append("Critical services (MongoDB) are unavailable - core functionality will not work")
        
        unhealthy_optional = [
            name for name, service in health_status["services"].items() 
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
async def close_current_conversation(client_id: str):
    """Close the current conversation for a specific client."""
    if client_id not in active_clients:
        return JSONResponse(
            content={"error": f"Client '{client_id}' not found or not connected"},
            status_code=404
        )
    
    client_state = active_clients[client_id]
    if not client_state.connected:
        return JSONResponse(
            content={"error": f"Client '{client_id}' is not connected"},
            status_code=400
        )
    
    try:
        # Close the current conversation
        await client_state._close_current_conversation()
        
        # Reset conversation state but keep client connected
        client_state.current_audio_uuid = None
        client_state.conversation_start_time = time.time()
        client_state.last_transcript_time = None
        
        logger.info(f"Manually closed conversation for client {client_id}")
        
        return JSONResponse(content={
            "message": f"Successfully closed current conversation for client '{client_id}'",
            "client_id": client_id,
            "timestamp": int(time.time())
        })
    
    except Exception as e:
        logger.error(f"Error closing conversation for client {client_id}: {e}")
        return JSONResponse(
            content={"error": f"Failed to close conversation: {str(e)}"},
            status_code=500
        )


@app.get("/api/active_clients")
async def get_active_clients():
    """Get list of currently active/connected clients."""
    client_info = {}
    
    for client_id, client_state in active_clients.items():
        client_info[client_id] = {
            "connected": client_state.connected,
            "current_audio_uuid": client_state.current_audio_uuid,
            "conversation_start_time": client_state.conversation_start_time,
            "last_transcript_time": client_state.last_transcript_time,
            "has_active_conversation": client_state.current_audio_uuid is not None
        }
    
    return JSONResponse(content={
        "active_clients_count": len(active_clients),
        "clients": client_info
    })




@app.get("/api/debug/speech_segments")
async def debug_speech_segments():
    """Debug endpoint to check current speech segments for all active clients."""
    debug_info = {
        "active_clients": len(active_clients),
        "audio_cropping_enabled": AUDIO_CROPPING_ENABLED,
        "min_speech_duration": MIN_SPEECH_SEGMENT_DURATION,
        "cropping_padding": CROPPING_CONTEXT_PADDING,
        "clients": {}
    }
    
    for client_id, client_state in active_clients.items():
        debug_info["clients"][client_id] = {
            "current_audio_uuid": client_state.current_audio_uuid,
            "speech_segments": {
                uuid: segments for uuid, segments in client_state.speech_segments.items()
            },
            "current_speech_start": dict(client_state.current_speech_start),
            "connected": client_state.connected,
            "last_transcript_time": client_state.last_transcript_time
        }
    
    return JSONResponse(content=debug_info)


###############################################################################
# ENTRYPOINT
###############################################################################

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    audio_logger.info("Starting Omi unified service at ws://%s:%s/ws", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=False)
