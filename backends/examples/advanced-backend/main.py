#!/usr/bin/env python3
"""Unified Omi-audio service

* Accepts Opus packets over a WebSocket (`/ws`).
* Uses **OmiSDK** (`OmiOpusDecoder`) to convert them to 16-kHz/16-bit/mono PCM.
* Buffers PCM and writes 30-second WAV chunks to `./audio_chunks/`.
* Each 30-second chunk is passed to **`transcribe_audio()`** (user-supplied) to get a transcript.
* The transcript is stored in **mem0** (vector store backed by Qdrant, embeddings/LLM via Ollama).

"""
from __future__ import annotations

import logging
import os
import wave
from datetime import datetime
from pathlib import Path
from typing import List

import ollama  # Ollama python client
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from mem0 import Memory  # mem0 core
from omi.decoder import OmiOpusDecoder  # OmiSDK

load_dotenv()

###############################################################################
# Configuration
###############################################################################
SAMPLE_RATE = 16_000  # Hz
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes (16‑bit)
SEGMENT_SECONDS = 30  # length of each stored chunk
TARGET_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS

# Directory where WAV chunks are written
CHUNK_DIR = Path("./audio_chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# ---- Deepgram Configuration ------------------------------------------------ #
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", '')

# ---- mem0 + Ollama --------------------------------------------------------- #
MEM0_CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:latest",
            "ollama_base_url": "http://ollama:11434",
            "temperature": 0,
            "max_tokens": 2000,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "embedding_dims": 768,
            "ollama_base_url": "http://ollama:11434",
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "omi_memories",
            "embedding_model_dims": 768,
            "host": "qdrant",
            "port": 6333
        },
    },
}

memory = Memory.from_config(MEM0_CONFIG)
ollama_client = ollama.Client(host="http://ollama:11434")  # noqa: S110

###############################################################################
# STT function
###############################################################################

def transcribe_deepgram(pcm_bytes: bytes) -> str:
    """Transcribe PCM audio bytes using Deepgram."""
    if not DEEPGRAM_API_KEY:
        audio_logger.error("DEEPGRAM_API_KEY not set. Skipping transcription.")
        return "[transcription unavailable - DEEPGRAM_API_KEY not set]"

    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        source = {'buffer': pcm_bytes, 'mimetype': 'audio/wav'} # Deepgram expects WAV
        options = PrerecordedOptions(
            smart_format=True, model="nova-2", language="en-US"
        )
        response = deepgram.listen.prerecorded.v('1').transcribe_file(source, options)
        # Extract transcript from the first channel and first alternative
        if response.results and response.results.channels and \
           response.results.channels[0].alternatives:
            transcript = response.results.channels[0].alternatives[0].transcript
            if transcript:
                 return transcript
        audio_logger.warning("Deepgram transcription returned empty or unexpected response format.")
        return "[transcription unavailable - empty response from Deepgram]"
    except Exception as e:
        audio_logger.error(f"Deepgram transcription failed: {e}")
        return f"[transcription error: {e}]"


def transcribe_audio(pcm_bytes: bytes) -> str:
    """Turn raw PCM into text.

    This function now delegates to `transcribe_deepgram`.
    It's kept for potential future use with other STT engines or for compatibility.
    """
    return transcribe_deepgram(pcm_bytes)

###############################################################################
# FastAPI WebSocket server
###############################################################################
app = FastAPI(title="Omi Unified Audio Service")

audio_logger = logging.getLogger("audio_service")
audio_logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")




class ClientState:
    """Keep per-connection state."""

    def __init__(self, ws: WebSocket, client_id: int):
        self.ws = ws
        self.id = client_id
        self.pcm_frames: List[bytes] = []  # raw PCM chunks
        self.sample_count = 0  # samples accumulated in current segment
        self.segment_index = 0
        self.start_time = datetime.utcnow().isoformat()
        self.decoder = OmiOpusDecoder()

    # --------------------------------------------------------------------- #
    async def add_opus_packet(self, packet: bytes):
        """Decode a single Opus packet -> PCM, buffer it, handle rollover."""
        pcm = self.decoder.decode_packet(packet, strip_header=False)
        if pcm is None:
            audio_logger.error("Client %s: decode failed (packet dropped)", self.id)
            return
        audio_logger.debug("Client %s: decoded packet", self.id)

        self.pcm_frames.append(pcm)
        self.sample_count += len(pcm) // SAMPLE_WIDTH  # samples per frame

        if self.sample_count >= TARGET_SAMPLES:
            await self._flush_segment()

    # ------------------------------------------------------------------ #
    async def _flush_segment(self):
        if not self.pcm_frames:
            return

        pcm_bytes = b"".join(self.pcm_frames)
        duration = self.sample_count / SAMPLE_RATE

        # Write WAV to disk ------------------------------------------------ #
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        wav_path = CHUNK_DIR / f"client{self.id}_{timestamp}_{self.segment_index}.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)
        audio_logger.info("Client %s: wrote %s (%.1fs)", self.id, wav_path.name, duration)

        # Transcribe and store in mem0 ------------------------------------ #
        transcript = transcribe_deepgram(pcm_bytes)
        if transcript and not transcript.startswith("[transcription unavailable") and not transcript.startswith("[transcription error"):
            memory.add(transcript, user_id=f"client_{self.id}")
            audio_logger.info("Client %s: transcript added to memory – %s", self.id, transcript[:60])
        elif transcript:
            audio_logger.warning("Client %s: transcription issue – %s", self.id, transcript)

        # Reset buffers
        self.pcm_frames.clear()
        self.sample_count = 0
        self.segment_index += 1


# ------------------------------------------------------------------------- #
# WebSocket endpoint
# ------------------------------------------------------------------------- #

clients: List[ClientState] = []


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    # Accept handshake
    await ws.accept()
    client_id = len(clients) + 1
    state = ClientState(ws, client_id)
    clients.append(state)
    audio_logger.info("Client %s connected", client_id)

    try:
        while True:
            try:
                audio_logger.debug("Client %s: waiting for packet...", client_id)
                packet = await ws.receive_bytes()
                audio_logger.debug("Client %s: received packet (size: %d bytes)", client_id, len(packet))
            except WebSocketDisconnect:
                break  # graceful client close
            await state.add_opus_packet(packet)
    finally:
        clients.remove(state)
        # Flush remainder (<30s) before goodbye
        await state._flush_segment()
        audio_logger.info("Client %s disconnected", client_id)


###############################################################################
# Entrypoint
###############################################################################
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    audio_logger.info("Starting Omi unified service at ws://%s:%s/ws", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=False)
