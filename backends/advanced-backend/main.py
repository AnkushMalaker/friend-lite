#!/usr/bin/env python3
"""Unified Omi-audio service

* Accepts Opus packets over a WebSocket (`/ws`).
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
import wave
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import ollama  # Ollama python client
import websockets.exceptions
from dotenv import load_dotenv
from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from mem0 import Memory  # mem0 core
from motor.motor_asyncio import AsyncIOMotorClient
from omi.decoder import OmiOpusDecoder  # OmiSDK
from pydub import AudioSegment
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.event import Event

MONGODB_URI = os.getenv("MONGODB_URI")
mongo_client = AsyncIOMotorClient(MONGODB_URI)  # async + pooled
db = mongo_client.get_default_database("friend-lite")  # "friend-lite"
chunks_col = db["audio_chunks"]  # collection handle

# Central Queues
# ----------------
# (client_id, pcm_data) or (client_id, None) for disconnect
central_chunk_queue = asyncio.Queue[Tuple[str, Optional[AudioChunk]]]()
# (audio_uuid, pcm_bytes, client_id)
transcription_input_queue = asyncio.Queue[Tuple[str, AudioChunk, str]]()


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
            "host": "qdrant",
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


def write_wav_sync(path: Path, pcm_bytes: bytes):
    """Synchronously writes PCM data to a WAV file."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(OMI_CHANNELS)
        wav_file.setsampwidth(OMI_SAMPLE_WIDTH)
        wav_file.setframerate(OMI_SAMPLE_RATE)
        wav_file.writeframes(pcm_bytes)


_new_local_file_sink = partial(
    LocalFileSink,
    sample_rate=OMI_SAMPLE_RATE,
    channels=OMI_CHANNELS,
    sample_width=OMI_SAMPLE_WIDTH,
)

def _audio_chunk_to_audio_segment(audio_chunk: AudioChunk) -> AudioSegment:
    return AudioSegment(
        data=audio_chunk.audio,
        frame_rate=OMI_SAMPLE_RATE,
        channels=OMI_CHANNELS,
        sample_width=OMI_SAMPLE_WIDTH,
    )

async def audio_saver_consumer():
    """Consumes PCM data from the central queue, buffers it, and saves it to WAV files."""
    _save_file_handle: Optional[LocalFileSink] = None
    while True:
        try:
            client_id, audio_chunk = await central_chunk_queue.get()

            if audio_chunk is None:  # Sentinel for client disconnection
                audio_logger.info(
                    f"Client {client_id} disconnected, flushing remaining buffer."
                )
                if _save_file_handle is not None:
                    await _save_file_handle.close()
                _save_file_handle = None
                central_chunk_queue.task_done()
                continue
                
            if _save_file_handle is None:
                audio_uuid = uuid.uuid4().hex
                timestamp = audio_chunk.timestamp or int(time.time())
                wav_filename = f"{timestamp}_{client_id}_{audio_uuid}.wav"
                _save_file_handle = _new_local_file_sink(f"{CHUNK_DIR}/{wav_filename}")

                await chunk_repo.create_chunk(
                    audio_uuid=audio_uuid,
                    audio_path=wav_filename,
                    client_id=client_id,
                    timestamp=timestamp,
                )

            await _save_file_handle.write(_audio_chunk_to_audio_segment(audio_chunk))
            
            central_chunk_queue.task_done()

            # Queue the chunk for transcription
            await transcription_input_queue.put((audio_uuid, audio_chunk, client_id))
        except Exception as e:
            audio_logger.error(f"Error in audio_saver_consumer: {e}", exc_info=True)
        finally:
            if _save_file_handle is not None:
                await _save_file_handle.close()
                _save_file_handle = None


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

    async def transcribe_chunk(self, audio_uuid: str, chunk_reference: AudioChunk, client_id: str):
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
                rate=chunk_reference.rate,
                width=chunk_reference.width,
                channels=chunk_reference.channels,
                timestamp=chunk_reference.timestamp,
            )
            await self.client.write_event(audio_start.event())
        else:
            audio_logger.info(f"Same audio_uuid: {audio_uuid}")
            await self.client.write_event(chunk_reference.event())

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


# Global transcription manager instance
transcription_manager = TranscriptionManager()


async def transcription_consumer():
    """Consumes audio chunks from a queue and sends them for transcription with persistent connection."""
    # Establish initial connection
    try:
        await transcription_manager.connect()
    except Exception as e:
        audio_logger.error(f"Failed to start transcription consumer: {e}")
        return

    try:
        audio_uuid = None
        while True:
            try:
                audio_uuid, chunk_reference, client_id = await transcription_input_queue.get()
                if audio_uuid != audio_uuid:
                    audio_logger.info(f"New audio_uuid: {audio_uuid}")
                    audio_uuid = audio_uuid
                else:
                    audio_logger.info(f"Same audio_uuid: {audio_uuid}")
                await transcription_manager.transcribe_chunk(
                    audio_uuid, chunk_reference, client_id
                )
                transcription_input_queue.task_done()
            except Exception as e:
                audio_logger.error(
                    f"Error in transcription_consumer: {e}", exc_info=True
                )
    finally:
        await transcription_manager.disconnect()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    audio_logger.info("Starting application...")

    # Start consumer tasks
    audio_saver_task = asyncio.create_task(audio_saver_consumer())
    audio_logger.info("Started audio saver consumer task.")

    transcription_task = asyncio.create_task(transcription_consumer())
    audio_logger.info("Started transcription consumer task.")

    try:
        yield
    finally:
        # Shutdown
        audio_logger.info("Shutting down application...")

        # Cancel consumer tasks
        audio_saver_task.cancel()
        transcription_task.cancel()

        # Wait for tasks to finish
        try:
            await asyncio.gather(
                audio_saver_task, transcription_task, return_exceptions=True
            )
        except Exception as e:
            audio_logger.error(f"Error during task shutdown: {e}")

        # Clean up transcription manager connection
        await transcription_manager.disconnect()
        audio_logger.info("Shutdown complete.")


###############################################################################
# FastAPI Application
###############################################################################

app = FastAPI(lifespan=lifespan)

app.mount("/audio", StaticFiles(directory=CHUNK_DIR), name="audio")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """Accepts WebSocket connections, decodes Opus audio, and puts it on a central queue."""
    await ws.accept()

    client_id = f"client_{uuid.uuid4().hex[:8]}"
    audio_logger.info(f"Client {client_id}: WebSocket connection accepted.")
    decoder = OmiOpusDecoder()

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
                central_chunk_queue.put_nowait((client_id, chunk))

    except WebSocketDisconnect:
        audio_logger.info(f"Client {client_id}: WebSocket disconnected.")
        await central_chunk_queue.put((client_id, None))  # Signal disconnect
    except Exception as e:
        audio_logger.error(f"Client {client_id}: An error occurred: {e}", exc_info=True)
        await central_chunk_queue.put((client_id, None))  # Signal disconnect


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


@app.get("/api/memories")
async def get_memories():
    """
    Retrieves all memories from the mem0 store.
    """
    try:
        all_memories = memory.get_all()
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
