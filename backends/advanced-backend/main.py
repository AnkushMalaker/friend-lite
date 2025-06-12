#!/usr/bin/env python3
"""Unified Omi-audio service

* Accepts Opus packets over a WebSocket (`/ws`) or PCM over a WebSocket (`/ws_pcm`).
* Uses a central queue to decouple audio ingestion from processing.
* A saver consumer buffers PCM and writes 30-second WAV chunks to `./audio_chunks/`.
* A transcription consumer sends each chunk to a Wyoming ASR service.
* The transcript is stored in **mem0** and MongoDB.

"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import ollama  # Ollama python client
from dotenv import load_dotenv
from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from mem0 import Memory  # mem0 core
from motor.motor_asyncio import AsyncIOMotorClient
from omi.decoder import OmiOpusDecoder  # OmiSDK
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe

###############################################################################
# SETUP
###############################################################################

# Load environment variables first
load_dotenv()

# Configure Mem0 telemetry based on environment variable
# Set default to False for privacy unless explicitly enabled
if not os.getenv("MEM0_TELEMETRY"):
    os.environ["MEM0_TELEMETRY"] = "False"

# Logging setup
logging.basicConfig(level=logging.INFO)
audio_logger = logging.getLogger("audio_processing")

import speaker_recognition

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

# Audio Configuration
OMI_SAMPLE_RATE = 16_000  # Hz
OMI_CHANNELS = 1
OMI_SAMPLE_WIDTH = 2  # bytes (16‑bit)
SEGMENT_SECONDS = 60  # length of each stored chunk
TARGET_SAMPLES = OMI_SAMPLE_RATE * SEGMENT_SECONDS

# Conversation timeout configuration
NEW_CONVERSATION_TIMEOUT_MINUTES = int(os.getenv("NEW_CONVERSATION_TIMEOUT_MINUTES", "10"))

# Directory where WAV chunks are written
CHUNK_DIR = Path("./audio_chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# ASR Configuration
OFFLINE_ASR_TCP_URI = os.getenv("OFFLINE_ASR_TCP_URI", "tcp://192.168.0.110:8765/")

# Ollama & Qdrant Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", "qdrant")

# Mem0 organization configuration
MEM0_ORGANIZATION_ID = os.getenv("MEM0_ORGANIZATION_ID", "friend-lite-org")
MEM0_PROJECT_ID = os.getenv("MEM0_PROJECT_ID", "audio-conversations")
MEM0_APP_ID = os.getenv("MEM0_APP_ID", "omi-backend")

# Mem0 Configuration
MEM0_CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:latest",
            "ollama_base_url": OLLAMA_BASE_URL,
            "temperature": 0,
            "max_tokens": 2000,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "embedding_dims": 768,
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "omi_memories",
            "embedding_model_dims": 768,
            "host": QDRANT_BASE_URL,
            "port": 6333,
        },
    },
    "custom_prompt": "Extract meaningful preferences, facts, and experiences from the conversation. Focus on personal information, habits, and contextual details that would be useful for future interactions.",
}

# Thread pool executors
_DEC_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=os.cpu_count() or 4,
    thread_name_prefix="opus_io",
)

# Initialize mem0 and ollama client
memory = Memory.from_config(MEM0_CONFIG)
ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

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


# Global variable to hold the memory instance in each worker process
_process_memory = None

def _init_process_memory():
    """Initialize memory instance once per worker process."""
    global _process_memory
    if _process_memory is None:
        _process_memory = Memory.from_config(MEM0_CONFIG)
    return _process_memory


def _add_memory_to_store(transcript: str, client_id: str, audio_uuid: str) -> bool:
    """
    Function to add memory in a separate process.
    This function will be pickled and run in a process pool.
    Uses a persistent memory instance per process.
    """
    try:
        # Get or create the persistent memory instance for this process
        process_memory = _init_process_memory()
        process_memory.add(
            transcript,
            user_id=client_id,
            metadata={
                "source": "offline_streaming",
                "audio_uuid": audio_uuid,
                "timestamp": int(time.time()),
                "conversation_context": "audio_transcription",
                "device_type": "audio_recording",
                "organization_id": MEM0_ORGANIZATION_ID,
                "project_id": MEM0_PROJECT_ID,
                "app_id": MEM0_APP_ID,
            },
        )
        return True
    except Exception as e:
        # Log to stderr since we're in a separate process
        import sys
        print(f"Error in memory process for {audio_uuid}: {e}", file=sys.stderr)
        return False


# Process pool executor with initializer for heavy memory operations
_MEMORY_PROCESS_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
    max_workers=2,  # Keep this low to avoid overwhelming the system
    initializer=_init_process_memory,  # Initialize memory once per process
)

# Speaker recognition queue and worker
SPKR_QUEUE: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

async def speaker_worker():
    """Background worker for speaker diarization and verification."""
    while True:
        try:
            wav_path, audio_uuid = await SPKR_QUEUE.get()
                # Run speaker processing directly since it's already async
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
        """Update speaker information for a specific transcript segment."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$set": {f"transcript.{segment_index}.speaker": speaker_id}}
        )
        # Also add to speakers_identified if not already present
        await self.add_speaker(audio_uuid, speaker_id)


class TranscriptionManager:
    """Manages persistent connection to ASR service with ordered processing."""

    def __init__(self):
        self.client = None
        self._current_audio_uuid = None
        self._streaming = False

    async def connect(self):
        """Establish connection to ASR service."""
        try:
            self.client = AsyncTcpClient.from_uri(OFFLINE_ASR_TCP_URI)
            await self.client.connect()
            audio_logger.info(f"Connected to ASR service at {OFFLINE_ASR_TCP_URI}")
        except Exception as e:
            audio_logger.error(f"Failed to connect to ASR service: {e}")
            self.client = None
            raise

    async def disconnect(self):
        """Cleanly disconnect from ASR service."""
        if self.client:
            try:
                await self.client.disconnect()
                audio_logger.info("Disconnected from ASR service")
            except Exception as e:
                audio_logger.error(f"Error disconnecting from ASR service: {e}")
            finally:
                self.client = None

    async def transcribe_chunk(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Transcribe a single chunk with ordered processing."""
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
                                "speaker": f"speaker_{client_id}",  # Default speaker ID
                                "text": transcript_text,
                                "start": 0.0,  # Will need timing info from ASR for accurate values
                                "end": 0.0     # Will need timing info from ASR for accurate values
                            }
                            
                            # Store transcript segment in DB immediately (fast)
                            await chunk_repo.add_transcript_segment(audio_uuid, transcript_segment)
                            await chunk_repo.add_speaker(audio_uuid, f"speaker_{client_id}")
                            audio_logger.info(f"Added transcript segment for {audio_uuid} to DB.")
                            
                            # Update transcript time for conversation timeout tracking
                            if client_id in active_clients:
                                active_clients[client_id].last_transcript_time = time.time()
                            
                            # Queue memory processing (slow, non-blocking)
                            if client_id in active_clients:
                                await active_clients[client_id].memory_queue.put((transcript_text, client_id, audio_uuid))
                            else:
                                audio_logger.warning(f"Client {client_id} not found for memory processing")
            except asyncio.TimeoutError:
                # No events available right now, that's fine
                pass
                
        except Exception as e:
            audio_logger.error(f"Error in transcribe_chunk for {audio_uuid}: {e}")
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
        
        # Tasks for this client
        self.saver_task: Optional[asyncio.Task] = None
        self.transcription_task: Optional[asyncio.Task] = None
        self.memory_task: Optional[asyncio.Task] = None
    
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
            
        # Clean up file sink
        if self.file_sink:
            await self.file_sink.close()
            self.file_sink = None
            
        # Clean up transcription manager
        if self.transcription_manager:
            await self.transcription_manager.disconnect()
            self.transcription_manager = None
            
        audio_logger.info(f"Client {self.client_id} disconnected and cleaned up")
    
    def _should_start_new_conversation(self) -> bool:
        """Check if we should start a new conversation based on timeout."""
        if self.last_transcript_time is None:
            return False  # No transcript yet, keep current conversation
        
        current_time = time.time()
        time_since_last_transcript = current_time - self.last_transcript_time
        timeout_seconds = NEW_CONVERSATION_TIMEOUT_MINUTES * 60
        
        return time_since_last_transcript > timeout_seconds
    
    async def start_new_conversation(self):
        """Start a new conversation by closing current file sink and resetting state."""
        if self.file_sink:
            # Store current audio info before closing
            current_uuid = self.current_audio_uuid
            current_path = self.file_sink._file_path.name if hasattr(self.file_sink, '_file_path') else None
            
            await self.file_sink.close()
            self.file_sink = None
            
            # Queue for speaker processing if we have a completed file
            if current_uuid and current_path:
                await SPKR_QUEUE.put((current_path, current_uuid))
                audio_logger.info(f"Queued {current_path} for speaker processing")
            
            audio_logger.info(f"Client {self.client_id}: Started new conversation due to {NEW_CONVERSATION_TIMEOUT_MINUTES}min timeout")
        
        self.current_audio_uuid = None
        self.conversation_start_time = time.time()
        self.last_transcript_time = None
    
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
            if self.file_sink:
                # Store current audio info before closing
                current_uuid = self.current_audio_uuid
                current_path = self.file_sink._file_path.name if hasattr(self.file_sink, '_file_path') else None
                
                await self.file_sink.close()
                self.file_sink = None
                
                # Queue for speaker processing if we have a completed file
                if current_uuid and current_path:
                    await SPKR_QUEUE.put((current_path, current_uuid))
                    audio_logger.info(f"Queued {current_path} for speaker processing")
    
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
        """Per-client memory processor - handles memory.add operations in background."""
        try:
            while self.connected:
                transcript, client_id, audio_uuid = await self.memory_queue.get()
                
                if transcript is None or client_id is None or audio_uuid is None:  # Disconnect signal
                    break
                
                # Process memory in background (this is the slow operation)
                # Run in separate process to completely isolate heavy ML operations
                try:
                    loop = asyncio.get_running_loop()
                    success = await loop.run_in_executor(
                        _MEMORY_PROCESS_EXECUTOR,
                        _add_memory_to_store,
                        transcript,
                        client_id,
                        audio_uuid
                    )
                    if success:
                        audio_logger.info(f"Added transcript for {audio_uuid} to mem0 (client: {client_id}).")
                    else:
                        audio_logger.error(f"Failed to add memory for {audio_uuid}")
                except Exception as e:
                    audio_logger.error(f"Error adding memory for {audio_uuid}: {e}")
                        
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
    
    asyncio.create_task(speaker_worker(), name="speaker-worker")
    audio_logger.info("Speaker recognition worker started")
        
    audio_logger.info("Application ready - clients will have individual processing pipelines.")

    try:
        yield
    finally:
        # Shutdown
        audio_logger.info("Shutting down application...")

        # Clean up all active clients
        for client_id in list(active_clients.keys()):
            await cleanup_client_state(client_id)
        
        # Shutdown process pool executor
        _MEMORY_PROCESS_EXECUTOR.shutdown(wait=True)
        audio_logger.info("Memory process pool shut down.")
        
        audio_logger.info("Shutdown complete.")


# FastAPI Application
app = FastAPI(lifespan=lifespan)
app.mount("/audio", StaticFiles(directory=CHUNK_DIR), name="audio")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, user_id: Optional[str] = Query(None)):
    """Accepts WebSocket connections, decodes Opus audio, and processes per-client."""
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
    """Retrieves the last 20 transcribed audio chunks from MongoDB."""
    try:
        cursor = chunks_col.find(
            {"transcript": {"$exists": True, "$not": {"$size": 0}}}, 
            sort=[("timestamp", -1)], 
            limit=20
        )
        conversations = []
        for doc in await cursor.to_list(length=20):
            doc["_id"] = str(doc["_id"])
            conversations.append(doc)
        return JSONResponse(content=conversations)
    except Exception as e:
        audio_logger.error(f"Error fetching conversations: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error fetching conversations"}
        )


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
            # Delete all memories for this user from mem0
            try:
                # Get all memories for the user first to count them
                user_memories = memory.get_all(user_id=user_id)
                memory_count = len(user_memories) if user_memories else 0
                
                # Delete all memories for this user using the proper mem0 API
                if memory_count > 0:
                    memory.delete_all(user_id=user_id)
                    audio_logger.info(f"Deleted {memory_count} memories for user {user_id}")
                
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
        all_memories = memory.get_all(
            user_id=user_id,
            limit=limit,
        )
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
        relevant_memories = memory.search(
            query=query,
            user_id=user_id,
            limit=limit,
        )
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
        memory.delete(memory_id=memory_id)
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


@app.post("/api/speakers/enroll")
async def enroll_speaker(
    speaker_id: str,
    speaker_name: str,
    audio_file_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
):
    """
    Enroll a new speaker from an audio file.
    
    Args:
        speaker_id: Unique identifier for the speaker
        speaker_name: Human-readable name for the speaker  
        audio_file_path: Path to the audio file (relative to audio_chunks directory)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
    """
    try:
        # Full path to audio file
        full_audio_path = CHUNK_DIR / audio_file_path
        
        if not full_audio_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Audio file not found: {audio_file_path}"}
            )
        
        # Enroll speaker using speaker_recognition module
        success = speaker_recognition.enroll_speaker(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            audio_file=str(full_audio_path),
            start_time=start_time,
            end_time=end_time
        )
        
        if success:
            # Store speaker info in MongoDB
            speaker_doc = {
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "audio_file_path": audio_file_path,
                "start_time": start_time,
                "end_time": end_time,
                "enrolled_at": time.time()
            }
            
            await speakers_col.insert_one(speaker_doc)
            
            return JSONResponse(content={
                "message": f"Speaker {speaker_id} enrolled successfully",
                "speaker_id": speaker_id,
                "speaker_name": speaker_name
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
            speaker_info = speakers_map.get(speaker["id"], {})
            result.append({
                "id": speaker["id"],
                "name": speaker["name"],
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
async def identify_speaker_from_file(
    audio_file_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
):
    """
    Identify a speaker from an audio file segment.
    
    Args:
        audio_file_path: Path to the audio file (relative to audio_chunks directory)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
    """
    try:
        # Full path to audio file
        full_audio_path = CHUNK_DIR / audio_file_path
        
        if not full_audio_path.exists():
            return JSONResponse(
                status_code=404,
                content={"error": f"Audio file not found: {audio_file_path}"}
            )
        
        # Load audio and extract embedding using imports from pyannote
        from pyannote.core import Segment
        from pyannote.audio import Audio
        audio_loader = Audio(sample_rate=16000, mono="downmix")
        
        if start_time is not None and end_time is not None:
            segment = Segment(start_time, end_time)
            waveform, _ = audio_loader.crop(str(full_audio_path), segment)
        else:
            waveform, _ = audio_loader(str(full_audio_path))
        
        # Extract embedding
        waveform = waveform.unsqueeze(0)  # Add batch dimension
        embedding = speaker_recognition.embedding_model(waveform)
        
        # Normalize embedding
        import numpy as np
        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        
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
                "message": "Speaker not recognized"
            })
        
    except Exception as e:
        audio_logger.error(f"Error identifying speaker: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
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
            "ollama_url": OLLAMA_BASE_URL,
            "qdrant_url": f"http://{QDRANT_BASE_URL}:6333",
            "asr_uri": OFFLINE_ASR_TCP_URI,
            "chunk_dir": str(CHUNK_DIR),
            "active_clients": len(active_clients),
            "new_conversation_timeout_minutes": NEW_CONVERSATION_TIMEOUT_MINUTES
        }
    }
    
    overall_healthy = True
    
    # Check MongoDB
    try:
        await mongo_client.admin.command('ping')
        health_status["services"]["mongodb"] = {
            "status": "✅ Connected",
            "healthy": True
        }
    except Exception as e:
        health_status["services"]["mongodb"] = {
            "status": f"❌ Connection Failed: {str(e)}",
            "healthy": False
        }
        overall_healthy = False
    
    # Check Ollama
    try:
        models = ollama_client.list()
        model_count = len(models.get('models', []))
        health_status["services"]["ollama"] = {
            "status": "✅ Connected",
            "healthy": True,
            "models": model_count
        }
    except Exception as e:
        health_status["services"]["ollama"] = {
            "status": f"❌ Connection Failed: {str(e)}",
            "healthy": False
        }
        overall_healthy = False
    
    # Check mem0
    try:
        # Simple check - try to initialize memory (this should be fast)
        test_memory = Memory.from_config(MEM0_CONFIG)
        health_status["services"]["mem0"] = {
            "status": "✅ Connected",
            "healthy": True
        }
    except Exception as e:
        health_status["services"]["mem0"] = {
            "status": f"❌ Connection Failed: {str(e)}",
            "healthy": False
        }
        overall_healthy = False
    
    # Check ASR service
    try:
        test_client = AsyncTcpClient.from_uri(OFFLINE_ASR_TCP_URI)
        await test_client.connect()
        await test_client.disconnect()
        health_status["services"]["asr"] = {
            "status": "✅ Connected",
            "healthy": True,
            "uri": OFFLINE_ASR_TCP_URI
        }
    except Exception as e:
        health_status["services"]["asr"] = {
            "status": f"❌ Connection Failed: {str(e)}",
            "healthy": False,
            "uri": OFFLINE_ASR_TCP_URI
        }
        overall_healthy = False
    
    # Set overall status
    health_status["overall_healthy"] = overall_healthy
    if not overall_healthy:
        health_status["status"] = "unhealthy"
    
    return JSONResponse(content=health_status, status_code=200)


@app.get("/readiness")
async def readiness_check():
    """Simple readiness check for container orchestration."""
    return JSONResponse(content={"status": "ready", "timestamp": int(time.time())}, status_code=200)


###############################################################################
# ENTRYPOINT
###############################################################################

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    audio_logger.info("Starting Omi unified service at ws://%s:%s/ws", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=False)
