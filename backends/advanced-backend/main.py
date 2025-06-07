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
from pathlib import Path
from typing import Optional, Tuple

import ollama  # Ollama python client
from dotenv import load_dotenv
from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from mem0 import Memory  # mem0 core
from motor.motor_asyncio import AsyncIOMotorClient
from omi.decoder import OmiOpusDecoder  # OmiSDK
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import Event

logging.basicConfig(level=logging.INFO)

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
mongo_client = AsyncIOMotorClient(MONGODB_URI)  # async + pooled
db = mongo_client.get_default_database("friend-lite")  # "friend-lite"
chunks_col = db["audio_chunks"]  # collection handle
users_col = db["users"]  # collection handle


# Client State Management
# ----------------------
class ClientState:
    """Manages all state for a single client connection."""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.connected = True
        
        # Per-client queues
        self.chunk_queue = asyncio.Queue[Optional[AudioChunk]]()
        self.transcription_queue = asyncio.Queue[Tuple[Optional[str], Optional[AudioChunk]]]()
        
        # Per-client file sink
        self.file_sink: Optional[LocalFileSink] = None
        self.current_audio_uuid: Optional[str] = None
        
        # Per-client transcription manager
        self.transcription_manager: Optional[TranscriptionManager] = None
        
        # Tasks for this client
        self.saver_task: Optional[asyncio.Task] = None
        self.transcription_task: Optional[asyncio.Task] = None
    
    async def start_processing(self):
        """Start the processing tasks for this client."""
        self.saver_task = asyncio.create_task(self._audio_saver())
        self.transcription_task = asyncio.create_task(self._transcription_processor())
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
        
        # Wait for tasks to complete
        if self.saver_task:
            await self.saver_task
        if self.transcription_task:
            await self.transcription_task
            
        # Clean up file sink
        if self.file_sink:
            await self.file_sink.close()
            self.file_sink = None
            
        # Clean up transcription manager
        if self.transcription_manager:
            await self.transcription_manager.disconnect()
            self.transcription_manager = None
            
        audio_logger.info(f"Client {self.client_id} disconnected and cleaned up")
    
    async def _audio_saver(self):
        """Per-client audio saver consumer."""
        try:
            while self.connected:
                audio_chunk = await self.chunk_queue.get()
                
                if audio_chunk is None:  # Disconnect signal
                    break
                    
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
                await self.file_sink.close()
                self.file_sink = None
    
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


# Global client state registry
active_clients: dict[str, ClientState] = {}


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
            "transcription": transcript,
            "speakers_identified": speakers_identified,
        }
        await self.col.insert_one(doc)

    async def add_transcript(self, audio_uuid, transcript):
        await self.col.update_one(
            {"audio_uuid": audio_uuid}, {"$set": {"transcription": transcript}}
        )


chunk_repo = ChunkRepo(chunks_col)  # Instantiate ChunkRepo

load_dotenv()

###############################################################################
# Configuration
###############################################################################
OMI_SAMPLE_RATE = 16_000  # Hz
OMI_CHANNELS = 1
OMI_SAMPLE_WIDTH = 2  # bytes (16‑bit)
SEGMENT_SECONDS = 60  # length of each stored chunk
TARGET_SAMPLES = OMI_SAMPLE_RATE * SEGMENT_SECONDS

# Directory where WAV chunks are written
CHUNK_DIR = Path("./audio_chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# ---- Offline ASR Configuration --------------------------------------------- #
OFFLINE_ASR_TCP_URI = os.getenv("OFFLINE_ASR_TCP_URI", "tcp://192.168.0.110:8765/")

# ---- mem0 + Ollama --------------------------------------------------------- #
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", "qdrant")
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
}

# Thread pool executor for CPU-bound and blocking I/O operations
_DEC_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=os.cpu_count() or 4,
    thread_name_prefix="opus_io",
)

memory = Memory.from_config(MEM0_CONFIG)
ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

# Logging setup
logging.basicConfig(level=logging.INFO)
audio_logger = logging.getLogger("audio_processing")


###############################################################################
# Audio Saver Consumer
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
        # sink._file_handle.setcomptype('NONE', 'not compressed')
    return sink


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
# Transcription Consumer
###############################################################################


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
        else:
            audio_logger.info(f"Same audio_uuid: {audio_uuid}")
            await self.client.write_event(chunk.event())

                # if Transcript.is_type(event.type):
                #     transcript_obj = Transcript.from_event(event)
                #     transcript = transcript_obj.text.strip()
                #     if transcript:
                #         audio_logger.info(f"Transcript for {audio_uuid}: {transcript}")
                #         await chunk_repo.add_transcript(audio_uuid, transcript)
                #         memory.add(
                #             transcript,
                #             user_id=client_id,
                #             metadata={
                #                 "source": "offline_streaming",
                #                 "audio_uuid": audio_uuid,
                #             },
                #         )
                #         audio_logger.info(
                #             f"Added transcript for {audio_uuid} to DB and mem0."
                #         )


    async def _reconnect(self):
        """Attempt to reconnect to ASR service."""
        audio_logger.info("Attempting to reconnect to ASR service...")
        await self.disconnect()
        await asyncio.sleep(2)  # Brief delay before reconnecting
        try:
            await self.connect()
        except Exception as e:
            audio_logger.error(f"Reconnection failed: {e}")





@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    audio_logger.info("Starting application...")
    audio_logger.info("Application ready - clients will have individual processing pipelines.")

    try:
        yield
    finally:
        # Shutdown
        audio_logger.info("Shutting down application...")

        # Clean up all active clients
        for client_id in list(active_clients.keys()):
            await cleanup_client_state(client_id)
        
        audio_logger.info("Shutdown complete.")


###############################################################################
# FastAPI Application
###############################################################################

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
    
    # Create client state and start processing
    client_state = await create_client_state(client_id)

    try:
        while True:
            packet = await ws.receive_bytes()
            loop = asyncio.get_running_loop()
            pcm_data = await loop.run_in_executor(
                _DEC_IO_EXECUTOR, decoder.decode_packet, packet
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
    """
    Retrieves the last 20 transcribed audio chunks from MongoDB.
    """
    try:
        cursor = chunks_col.find(
            {"transcription": {"$ne": None}}, sort=[("timestamp", -1)], limit=20
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
    """
    Retrieves all users from the database.
    """
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
    """
    Creates a new user in the database.
    """
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
async def delete_user(user_id: str):
    """
    Deletes a user from the database.
    """
    try:
        result = await users_col.delete_one({"user_id": user_id})
        if result.deleted_count == 0:
            return JSONResponse(
                status_code=404,
                content={"message": f"User {user_id} not found"}
            )
        
        return JSONResponse(
            status_code=200,
            content={"message": f"User {user_id} deleted successfully"}
        )
    except Exception as e:
        audio_logger.error(f"Error deleting user: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Error deleting user"}
        )

@app.get("/api/memories")
async def get_memories(user_id: str):
    """
    Retrieves all memories from the mem0 store.
    """
    try:
        all_memories = memory.get_all(
            user_id=user_id,
        )
        return JSONResponse(content=all_memories)
    except Exception as e:
        audio_logger.error(f"Error fetching memories: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error fetching memories"}
        )


###############################################################################
# Entrypoint
###############################################################################
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    audio_logger.info("Starting Omi unified service at ws://%s:%s/ws", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=False)
