#!/usr/bin/env python3
"""
WebSocket live-caption server using Moonshine ONNX + Silero VAD.

▸  Client  →  binary frames: 16-kHz mono float32 PCM
▸  Server  →  text frames   : {"text": "<caption>", "final": true|false}

Dependencies
------------
pip install websockets sounddevice numpy silero-vad moonshine-onnx
"""

import argparse
import asyncio
import json
import logging
import time
from collections import deque
from typing import Deque

import numpy as np
from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
from silero_vad import VADIterator, load_silero_vad
from websockets.asyncio.server import ServerConnection, serve

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
SAMPLING_RATE = 16_000
CHUNK_SAMPLES = 512                       # Silero requirement (32 ms @ 16 kHz)
CHUNK_BYTES   = CHUNK_SAMPLES * 2         # float32 → 4 B per sample

LOOKBACK_CHUNKS  = 5                      # prepend a little context
MAX_SPEECH_SECS  = 15
MIN_REFRESH_SECS = 0.20

# --------------------------------------------------------------------------- #
class Transcriber:
    """Thin wrapper around Moonshine ONNX for synchronous calls."""

    def __init__(self, model_name: str):
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.tokenizer = load_tokenizer()

        self.rate = SAMPLING_RATE
        self.__call__(np.zeros(self.rate, np.float32))  # warm-up

    def __call__(self, pcm: np.ndarray) -> str:
        tokens = self.model.generate(pcm[np.newaxis, :].astype(np.float32))
        return self.tokenizer.decode_batch(tokens)[0]


# --------------------------------------------------------------------------- #
async def handle_client(
    websocket: ServerConnection, model_name: str
) -> None:
    """
    One WebSocket = one live-caption session.

    Incoming: binary frames of arbitrary length (16-kHz float32 PCM).
    Outgoing: JSON text frames with keys:
              • text  – entire current line (incl. cached context)
              • final – True when VAD says the utterance ended
    """
    logger.info("Handling client")
    transcriber     = Transcriber(model_name)
    vad_model       = load_silero_vad(onnx=True)
    vad_iterator    = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=0.2,
    )

    caption_cache: Deque[str] = deque(maxlen=10)  # small history for look-back
    speech_buf     = np.empty(0, np.float32)      # rolling window
    byte_backlog   = bytearray()                  # leftover bytes < 512 samples
    recording      = False
    last_refresh_t = 0.0

    async for message in websocket:
        logger.debug(f"Received message: {len(message)}")
        if not isinstance(message, (bytes, bytearray)):
            continue  # ignore text frames from the client

        # ------------------------------------------------------------------ #
        # 1)  Accumulate bytes until we have ≥ 512 samples (=CHUNK_BYTES)
        # ------------------------------------------------------------------ #
        byte_backlog.extend(message)
        while len(byte_backlog) >= CHUNK_BYTES:
            chunk_bytes = byte_backlog[:CHUNK_BYTES]
            del byte_backlog[:CHUNK_BYTES]

            logger.debug(f"Chunk bytes: {len(chunk_bytes)}")
            # fast float32 unpack without copy
            chunk = np.frombuffer(chunk_bytes, dtype=np.int16)

            speech_buf = np.concatenate((speech_buf, chunk))
            if not recording:
                # keep only the look-back context while idle
                max_idle = LOOKBACK_CHUNKS * CHUNK_SAMPLES
                speech_buf = speech_buf[-max_idle:]

            # ---------------------------------------------------------------- #
            # 2)  Run VAD on this 32-ms chunk
            # ---------------------------------------------------------------- #
            vad_event = vad_iterator(chunk)
            if vad_event:
                logger.info(f"VAD event: {vad_event}")
                if "start" in vad_event and not recording:
                    recording      = True
                    last_refresh_t = time.time()

                if "end" in vad_event and recording:
                    recording = False
                    await send_caption(
                        websocket, speech_buf, transcriber, caption_cache, final=True
                    )
                    speech_buf = np.empty(0, np.float32)
                    soft_reset(vad_iterator)
                    continue

            # ---------------------------------------------------------------- #
            # 3)  If we are inside speech, push incremental refreshes
            # ---------------------------------------------------------------- #
            if recording:
                now = time.time()
                speech_len_sec = len(speech_buf) / SAMPLING_RATE

                if (
                    speech_len_sec > MAX_SPEECH_SECS
                    or now - last_refresh_t > MIN_REFRESH_SECS
                ):
                    await send_caption(
                        websocket,
                        speech_buf,
                        transcriber,
                        caption_cache,
                        final=False,
                    )
                    last_refresh_t = now

    # WebSocket closed – flush anything still buffered
    if recording and len(speech_buf):
        await send_caption(
            websocket, speech_buf, transcriber, caption_cache, final=True
        )


# --------------------------------------------------------------------------- #
async def send_caption(
    ws: ServerConnection,
    speech: np.ndarray,
    transcriber: Transcriber,
    cache: Deque[str],
    final: bool,
):
    """
    Produce transcription, prepend recent captions for context, and send.
    """
    loop   = asyncio.get_running_loop()
    text   = await loop.run_in_executor(None, transcriber, speech)
    merged = build_line_with_cache(text, cache)
    print(f"Sending caption: {merged}")

    if final:
        cache.append(text)

    await ws.send(json.dumps({"text": merged, "final": final}))


def build_line_with_cache(text: str, cache: Deque[str], width: int = 80) -> str:
    """
    Right-justify the current utterance and prepend as much cached
    context as fits within `width` characters – same logic as the demo.
    """
    for prev in reversed(cache):
        cat = f"{prev} {text}"
        if len(cat) > width:
            break
        text = cat
    if len(text) > width:
        text = text[-width:]
    return text.rjust(width)


def soft_reset(vad_it: VADIterator) -> None:
    """Reset only the iterator’s state, not the underlying model."""
    vad_it.triggered = False
    vad_it.temp_end  = 0
    vad_it.current_sample = 0


# --------------------------------------------------------------------------- #
async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="moonshine/base",
        choices=["moonshine/base", "moonshine/tiny"],
        help="Moonshine model to load",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Interface to bind the WebSocket server"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Port to bind the WebSocket server"
    )
    args = parser.parse_args()

    print(
        f"Starting Moonshine live-caption server on "
        f"ws://{args.host}:{args.port}   (model={args.model_name})"
    )
    async with serve(
        lambda ws: handle_client(ws, args.model_name),
        args.host,
        args.port,
        max_size=None,  # allow unlimited frame size
        ping_interval=10,
        ping_timeout=60
    ) as server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
