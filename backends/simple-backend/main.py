#!/usr/bin/env python3
"""Simple WebSocket Audio Service

* Accepts Opus packets over a WebSocket (`/ws`).
* Uses OmiSDK (`OmiOpusDecoder`) to convert them to 16-kHz/16-bit/mono PCM.
* Buffers PCM and writes 30-second WAV chunks to `./audio_chunks/`.
"""
from __future__ import annotations

import logging
import os
import wave
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from omi.decoder import OmiOpusDecoder  # OmiSDK

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

###############################################################################
# FastAPI WebSocket server
###############################################################################
app = FastAPI(title="Simple Audio Service")

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
    audio_logger.info("Starting simple audio service at ws://%s:%s/ws", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=False)
