#!/usr/bin/env python3
"""
WebSocket live-caption server using Nemo Parakeet ASR + Silero VAD.

▸  Client  →  binary frames: 16-kHz mono float32 PCM
▸  Server  →  text frames   : {"text": "<caption>", "final": true|false}

Dependencies
------------
pip install websockets sounddevice numpy silero-vad nemo_toolkit[asr] soundfile
"""

import argparse
import asyncio
import json
import logging
import os
import tempfile
from collections import deque
from typing import Deque

import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
from silero_vad import VADIterator, load_silero_vad
from websockets.asyncio.server import ServerConnection, serve

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
SAMPLING_RATE = 16_000
CHUNK_SAMPLES = 512                       # Silero requirement (32 ms @ 16 kHz)
CHUNK_BYTES   = CHUNK_SAMPLES * 2         # int16 → 2 B per sample

MAX_SPEECH_SECS  = 30                     # Max duration of a speech segment

# --------------------------------------------------------------------------- #
class Transcriber:
    """Thin wrapper around Nemo ASR for synchronous calls."""

    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v2"):
        logger.info(f"Loading Nemo ASR model: {model_name}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        self.rate = SAMPLING_RATE
        # Warm-up call
        logger.info("Warming up Nemo ASR model...")
        try:
            self.__call__(np.zeros(self.rate // 10, np.float32))  # 0.1s silence
            logger.info("Nemo ASR model warmed up successfully.")
        except Exception as e:
            logger.error(f"Error during ASR model warm-up: {e}")


    def __call__(self, pcm: np.ndarray) -> str:
        tmpfile_name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile_name = tmpfile.name
            
            sf.write(tmpfile_name, pcm, self.rate)
            
            results = self.model.transcribe([tmpfile_name], batch_size=1)
            
            if results and isinstance(results, list) and len(results) > 0:
                # Check if the first result is an object with a .text attribute
                if hasattr(results[0], 'text') and results[0].text is not None:
                    return results[0].text
                # Check if the first result is directly a string (some models/configs)
                elif isinstance(results[0], str):
                    return results[0]
            return "" # Return empty string if transcription failed or no text
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""
        finally:
            if tmpfile_name and os.path.exists(tmpfile_name):
                os.remove(tmpfile_name)


transcriber = Transcriber() # Uses default model
# --------------------------------------------------------------------------- #
async def handle_client(
    websocket: ServerConnection
) -> None:
    """
    One WebSocket = one live-caption session.

    Incoming: binary frames of arbitrary length (16-kHz float32 PCM).
    Outgoing: JSON text frames with keys:
              • text  – entire current line (incl. cached context)
              • final – True when VAD says the utterance ended
    """
    logger.info("Handling client")
    vad_model       = load_silero_vad(onnx=True)
    vad_iterator    = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=0.4, # VAD sensitivity
    )

    caption_cache: Deque[str] = deque(maxlen=10)  # small history for look-back
    speech_buf     = np.empty(0, np.float32)      # rolling window
    byte_backlog   = bytearray()                  # leftover bytes < CHUNK_SAMPLES
    recording      = False
    
    async for message in websocket:
        logger.debug(f"Received message: {len(message)}")
        if not isinstance(message, (bytes, bytearray)):
            continue  # ignore text frames from the client

        # ------------------------------------------------------------------ #
        # 1)  Accumulate bytes until we have ≥ CHUNK_SAMPLES
        # ------------------------------------------------------------------ #
        byte_backlog.extend(message)
        while len(byte_backlog) >= CHUNK_BYTES:
            chunk_bytes = byte_backlog[:CHUNK_BYTES]
            del byte_backlog[:CHUNK_BYTES]

            logger.debug(f"Chunk bytes: {len(chunk_bytes)}")
            chunk = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            
            if recording: # NEW: Accumulate only if already recording
                speech_buf = np.concatenate((speech_buf, chunk))
            # If not recording, speech_buf remains as it was (likely empty or from a previous reset)
            # It will be initialized with the current chunk upon a VAD "start" event.

            logger.debug(f"Speech buffer: {len(speech_buf)}")
            # Limit total speech_buf to avoid excessive memory usage if VAD fails to end
            if len(speech_buf) > MAX_SPEECH_SECS * SAMPLING_RATE:
                logger.warning(f"Speech buffer exceeded {MAX_SPEECH_SECS}s, truncating.")
                speech_buf = speech_buf[-(MAX_SPEECH_SECS * SAMPLING_RATE):]


            # ---------------------------------------------------------------- #
            # 2)  Run VAD on this chunk
            # ---------------------------------------------------------------- #
            try:
                vad_event = vad_iterator(chunk)
                logger.debug(f"VAD event: {vad_event}")
            except Exception as e:
                logger.error(f"Error during VAD: {e}")

            if vad_event:
                logger.info(f"VAD event: {vad_event}")
                if "start" in vad_event and not recording:
                    recording = True
                    speech_buf = chunk # MODIFIED: Start fresh from this chunk

                if "end" in vad_event and recording:
                    recording = False
                    if len(speech_buf) > SAMPLING_RATE * 0.2: # Min audio length for transcription (e.g. 0.2s)
                        logger.info(f"VAD end detected. Transcribing {len(speech_buf)/SAMPLING_RATE:.2f}s of audio.")
                        await send_caption(
                            websocket, speech_buf, transcriber, caption_cache, final=True
                        )
                    else:
                        logger.info("VAD end detected, but speech too short. Discarding.")
                    speech_buf = np.empty(0, np.float32) # Clear buffer after processing
                    soft_reset(vad_iterator) # Reset VAD state
                    continue
            
            # ---------------------------------------------------------------- #
            # 3) If speech buffer is too long without VAD end, force process (optional)
            # ---------------------------------------------------------------- #
            if recording and len(speech_buf) >= MAX_SPEECH_SECS * SAMPLING_RATE:
                logger.warning(f"Max speech length {MAX_SPEECH_SECS}s reached without VAD end. Forcing transcription.")
                await send_caption(
                    websocket, speech_buf, transcriber, caption_cache, final=True # Treat as final
                )
                speech_buf = np.empty(0, np.float32)
                recording = False
                soft_reset(vad_iterator)


    # WebSocket closed – flush anything still buffered
    if recording and len(speech_buf) > SAMPLING_RATE * 0.2:
        logger.info("WebSocket closed. Transcribing remaining buffered audio.")
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
    # Run blocking ASR call in a separate thread
    text   = await loop.run_in_executor(None, transcriber, speech)
    
    if not text: # If transcription is empty or failed
        logger.warning("Transcription resulted in empty text. Not sending.")
        return

    merged = build_line_with_cache(text, cache)
    logger.info(f"Sending caption: {merged} (final: {final})")

    if final: # Cache only final transcriptions
        cache.append(text)

    await ws.send(json.dumps({"text": merged, "final": final}))


def build_line_with_cache(text: str, cache: Deque[str], width: int = 80) -> str:
    """
    Right-justify the current utterance and prepend as much cached
    context as fits within `width` characters.
    """
    # Only prepend from cache if the current text is not empty
    if text:
        for prev in reversed(cache):
            cat = f"{prev} {text}"
            if len(cat) > width:
                break
            text = cat
    # Ensure text does not exceed width, take the rightmost part if it does
    if len(text) > width:
        text = text[-width:]
    return text.rjust(width)


def soft_reset(vad_it: VADIterator) -> None:
    """Reset only the iterator's state, not the underlying model."""
    vad_it.triggered = False
    vad_it.temp_end  = 0
    vad_it.current_sample = 0
    logger.debug("VAD iterator soft reset.")


# --------------------------------------------------------------------------- #
async def main() -> None:
    parser = argparse.ArgumentParser(description="Nemo ASR WebSocket Server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Interface to bind the WebSocket server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Port to bind the WebSocket server (default: 8765)"
    )
    # Add Nemo model name as an argument if desired, otherwise Transcriber uses its default
    # parser.add_argument(
    #     "--nemo_model_name",
    #     default="nvidia/parakeet-tdt-0.6b-v2",
    #     help="Nemo ASR model name from HuggingFace or NGC (default: nvidia/parakeet-tdt-0.6b-v2)"
    # )
    args = parser.parse_args()

    print(
        f"Starting Nemo Parakeet ASR live-caption server on "
        f"ws://{args.host}:{args.port}"
    )
    # To pass model name from args: lambda ws: handle_client(ws, args.nemo_model_name)
    # And update handle_client and Transcriber to accept it.
    # For now, Transcriber uses its internal default.
    async with serve(
        handle_client, # Corrected: pass the function directly
        args.host,
        args.port,
        max_size=None,  # allow unlimited frame size
        ping_interval=10,
        ping_timeout=60
    ) as server:
        await server.serve_forever()


if __name__ == "__main__":
    # Set up logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__) # Re-initialize logger with new format for main scope
    
    # Consider adding torch.set_num_threads(1) for CPU inference if multi-client causes issues
    # import torch
    # torch.set_num_threads(1) # Example

    asyncio.run(main())
