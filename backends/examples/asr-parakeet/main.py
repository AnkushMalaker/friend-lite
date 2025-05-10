"""WebSocket ASR server powered by NVIDIA Parakeet‑TDT‑0.6b‑v2 via NeMo.

Endpoint:  ws://<host>:8080/ws
Protocol :
  • Client streams **16‑kHz, 16‑bit PCM (little‑endian, mono)** audio frames as binary messages.
  • Send the text message "EOS" when done.
  • Server replies with one JSON message: {"text": "<transcript>"} then closes.

This iteration still buffers the full stream before inference; incremental streaming can be added in a future pass.
"""
import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Union

import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
import torch
import websockets

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
SAMPLE_RATE = 16_000
HOST = "0.0.0.0"
PORT = 8080

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Loading NeMo model %s on %s", MODEL_ID, device)

# Load pretrained Parakeet model
asr_model: nemo_asr.models.ASRModel = nemo_asr.models.ASRModel.from_pretrained(
    model_name=MODEL_ID, map_location=device
)
asr_model.eval()

async def transcribe_pcm(pcm_bytes: bytes) -> str:
    """Convert raw PCM to text using NeMo – writes to a temp WAV then calls transcribe."""
    # int16 LE → float32 in [-1, 1]
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE, subtype="PCM_16")
        tmp_path = Path(tmp.name)

    try:
        result = asr_model.transcribe([str(tmp_path)])  # returns list of SpeechRecognitionResult
        return result[0].text  # first sample, transcription string
    finally:
        tmp_path.unlink(missing_ok=True)

async def handler(ws):
    logger.info("Client connected: %s", ws.remote_address)
    buffer = bytearray()
    try:
        async for message in ws:
            if isinstance(message, (bytes, bytearray)):
                buffer.extend(message)
            elif isinstance(message, str) and message == "EOS":
                break
    except websockets.ConnectionClosed:
        logger.info("Connection closed prematurely")
        return

    logger.info("Received %d bytes – running ASR", len(buffer))
    text = await asyncio.to_thread(transcribe_pcm, bytes(buffer))
    await ws.send(json.dumps({"text": text}))
    await ws.close()
    logger.info("Transcript sent; connection closed.")

async def main() -> None:
    logger.info("WebSocket server starting on %s:%d", HOST, PORT)
    async with websockets.serve(handler, HOST, PORT, max_size=2**24, ping_interval=20):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down.")