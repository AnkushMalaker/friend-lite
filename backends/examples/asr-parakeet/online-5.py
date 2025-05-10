#!/usr/bin/env python3
"""
Streaming ASR over WebSockets with Moonshine.

Run:
  # regular Torch backend
  pip install "useful-moonshine@git+https://github.com/usefulsensors/moonshine.git"
  export KERAS_BACKEND=torch              # or tensorflow / jax
  python moonshine_ws.py --model moonshine/tiny

  # ONNX backend (best for Raspberry Pi, Jetson, etc.)
  pip install "useful-moonshine-onnx@git+https://github.com/usefulsensors/moonshine.git#subdirectory=moonshine-onnx"
  python moonshine_ws.py --use-onnx --model moonshine/tiny
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# -----------------------------------------------------------------------------#
# Command-line arguments
# -----------------------------------------------------------------------------#
parser = argparse.ArgumentParser(description="WebSocket → Moonshine streamer")
parser.add_argument("--host", default="0.0.0.0", help="bind address")
parser.add_argument("--port", type=int, default=8000, help="bind port")
parser.add_argument("--model", default="moonshine/tiny",
                    help="Moonshine model name or local path")
parser.add_argument("--use-onnx", action="store_true",
                    help="Use ONNX runtime (requires useful-moonshine-onnx)")
parser.add_argument("--chunk-sec", type=float, default=1.0,
                    help="Decode every N seconds of audio")
args, _ = parser.parse_known_args()

# -----------------------------------------------------------------------------#
# Pick Moonshine implementation
# -----------------------------------------------------------------------------#
if args.use_onnx or os.getenv("MOONSHINE_USE_ONNX") == "1":
    import moonshine_onnx as moonshine
    moonshine_backend = "onnx"
else:
    # Make sure Keras knows which backend you want *before* importing Moonshine
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import moonshine
    moonshine_backend = os.environ["KERAS_BACKEND"]

logging.info("Loaded Moonshine backend: %s", moonshine_backend)
model_name = args.model
model = moonshine.load_model(model_name) if hasattr(moonshine, "load_model") else None

# -----------------------------------------------------------------------------#
# Audio & buffering helpers
# -----------------------------------------------------------------------------#
SAMPLE_RATE = 16_000           # Moonshine default
SAMPLE_WIDTH = 2               # 16-bit PCM
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH
MIN_CHUNK_BYTES = int(args.chunk_sec * BYTES_PER_SECOND)

def pcm_bytes_to_float32(pcm: bytes) -> np.ndarray:
    """INT16 PCM → float32 in range [-1, 1]."""
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return audio / 32768.0

# ‼️ You already have this
def opus_to_pcm(opus_bytes: bytes) -> bytes: ...   # noqa: E701, placeholder

class Session:
    """Keeps per-client audio and partial transcript."""
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.partial: Optional[str] = None

    async def feed(self, opus_bytes: bytes) -> Optional[str]:
        """Append audio, run ASR when chunk is big enough, return new text."""
        self.buffer.extend(opus_to_pcm(opus_bytes))
        if len(self.buffer) < MIN_CHUNK_BYTES:
            return None

        # Decode & clear buffer
        pcm = bytes(self.buffer)
        self.buffer.clear()
        audio_f32 = pcm_bytes_to_float32(pcm)

        # Moonshine API accepts ndarray (torch/tf/jax backends) or raw PCM for ONNX.
        text_list = moonshine.transcribe(audio_f32, model_name) \
                    if model is None else model.transcribe(audio_f32)
        text = text_list[0] if isinstance(text_list, (list, tuple)) else str(text_list)

        # Return incremental result
        return text.strip()

# -----------------------------------------------------------------------------#
# FastAPI WebSocket server
# -----------------------------------------------------------------------------#
app = FastAPI(title="Moonshine-WS")

@app.get("/")
def index():
    return HTMLResponse("""
    <html><body>
    <h2>Moonshine WebSocket demo</h2>
    <p>Connect a WebSocket client to <code>ws://HOST:PORT/ws</code> and send OPUS frames.</p>
    </body></html>""")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session()
    try:
        while True:
            opus_bytes = await ws.receive_bytes()
            transcript = await session.feed(opus_bytes)
            if transcript:
                await ws.send_text(transcript)
    except WebSocketDisconnect:
        logging.info("Client disconnected")
    except Exception as exc:
        logging.exception("Error in ASR loop: %s", exc)
        await ws.close(code=1011, reason=str(exc))

# -----------------------------------------------------------------------------#
# Entrypoint
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
