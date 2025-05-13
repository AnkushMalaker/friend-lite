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
from typing import List, Optional
import concurrent.futures

import ollama  # Ollama python client
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from mem0 import Memory  # mem0 core
from omi.decoder import OmiOpusDecoder  # OmiSDK
from websockets.asyncio.client import connect
from websockets.protocol import State as WebsocketState # Corrected import for State
import asyncio
import json
import websockets.exceptions

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

# ---- Offline ASR Configuration --------------------------------------------- #
OFFLINE_ASR_WS_URI = os.getenv("OFFLINE_ASR_WS_URI", "ws://192.168.0.110:8765/")

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
            "port": 6333
        },
    },
}

# Thread pool executor for CPU-bound and blocking I/O operations
_DEC_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=os.cpu_count() or 4,
    thread_name_prefix="opus_io",
)

memory = Memory.from_config(MEM0_CONFIG)
ollama_client = ollama.Client(host=OLLAMA_BASE_URL)  # noqa: S110

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


###############################################################################
# Continuous Offline ASR Transcription Handler
###############################################################################
async def continuous_offline_transcription_handler(client_state: ClientState, parakeet_ws_uri: str):
    """Handles continuous transcription for a client using Offline ASR."""
    try:
        async with connect(parakeet_ws_uri) as parakeet_ws:
            audio_logger.info(f"Client {client_state.id}: Connected to Offline ASR at {parakeet_ws_uri}")

            send_task_running = True

            async def send_pcm_to_parakeet():
                nonlocal send_task_running
                try:
                    while True:
                        audio_logger.debug(f"Number of PCM frames in queue: {client_state.pcm_frames.qsize()}")
                        pcm_data = await client_state.pcm_frames.get()
                        if pcm_data is None:  # Sentinel to stop
                            audio_logger.info(f"Client {client_state.id}: PCM send task received stop signal from main WebSocket handler.")
                            client_state.pcm_frames.task_done()
                            break
                        
                        # Check WebSocket state before sending
                        if parakeet_ws.protocol.state != WebsocketState.OPEN:
                            audio_logger.warning(f"Client {client_state.id}: Offline WebSocket not OPEN (state: {parakeet_ws.protocol.state.name}). Re-queueing data and breaking send loop.")
                            await client_state.pcm_frames.put(pcm_data) 
                            break

                        await parakeet_ws.send(pcm_data)
                        # yield so receiver / other tasks may run
                        await asyncio.sleep(0)
                        # client_state.pcm_frames.task_done()
                except websockets.exceptions.ConnectionClosed:
                    audio_logger.warning(f"Client {client_state.id}: Connection to Offline ASR closed while sending PCM.")
                except Exception as e:
                    audio_logger.error(f"Client {client_state.id}: Error in send_pcm_to_parakeet: {type(e).__name__} {e}")
                finally:
                    send_task_running = False
                    # Ensure parakeet_ws is closed if sender initiated closure, to help receiver task terminate.
                    if pcm_data is None and parakeet_ws.protocol.state == WebsocketState.OPEN:
                         audio_logger.info(f"Client {client_state.id}: Sender closing Offline WS.")
                         await parakeet_ws.close()


            async def receive_transcripts_from_parakeet():
                try:
                    async for message_str in parakeet_ws:
                        try:
                            message_json = json.loads(message_str)
                            if message_json.get('final') and 'text' in message_json:
                                transcript = message_json['text']
                                if not transcript.strip(): 
                                    audio_logger.debug(f"Client {client_state.id}: Received empty final transcript.")
                                    continue

                                audio_logger.info(f"Client {client_state.id}: Transcript: {transcript[:60]}")
                                
                                memory.add(transcript, user_id=f"client_{client_state.id}", metadata={"source": "offline_streaming"})
                                audio_logger.info(f"Client {client_state.id}: Stored transcript in mem0.")

                                await client_state.ws.send_text(json.dumps({"transcript": transcript, "client_id": client_state.id}))
                            # Example for handling partial transcripts if needed in future
                            # elif 'text' in message_json and not message_json.get('final'):
                            #     partial_transcript = message_json['text']
                            #     if partial_transcript.strip():
                            #         await client_state.ws.send_text(json.dumps({"partial_transcript": partial_transcript, "client_id": client_state.id}))

                        except json.JSONDecodeError:
                            audio_logger.error(f"Client {client_state.id}: Malformed JSON from Offline: {message_str}")
                        except websockets.exceptions.ConnectionClosed:
                            audio_logger.warning(f"Client {client_state.id}: Original client WebSocket closed while sending transcript.")
                            break # Stop if original client connection is gone
                        except Exception as e:
                            audio_logger.error(f"Client {client_state.id}: Error processing Offline message: {type(e).__name__} {e}")
                except websockets.exceptions.ConnectionClosedOK:
                     audio_logger.info(f"Client {client_state.id}: Offline ASR connection closed gracefully (receiver).")
                except websockets.exceptions.ConnectionClosedError as e:
                    audio_logger.warning(f"Client {client_state.id}: Offline ASR connection closed with error (receiver): {e}")
                except Exception as e:
                    audio_logger.error(f"Client {client_state.id}: Error in receive_transcripts_from_parakeet: {type(e).__name__} {e}")


            sender_task = asyncio.create_task(send_pcm_to_parakeet())
            receiver_task = asyncio.create_task(receive_transcripts_from_parakeet())
            
            await asyncio.gather(sender_task, receiver_task)
            
            # for task in pending:
            #     task.cancel()
            #     try:
            #         await task 
            # except asyncio.CancelledError:
            #     pass 
            # except Exception: # Log other exceptions during cancellation
            #     audio_logger.error(f"Client {client_state.id}: Error during cancellation of a sub-task for Offline handler.")


            # if parakeet_ws.protocol.state == WebsocketState.OPEN:
            #     await parakeet_ws.close()

    except (ConnectionRefusedError, websockets.exceptions.InvalidURI, websockets.exceptions.WebSocketException) as e: # More specific connection errors
        audio_logger.error(f"Client {client_state.id}: Failed to connect or maintain connection with Offline ASR at {parakeet_ws_uri}: {type(e).__name__} {e}")
    except Exception as e:
        audio_logger.error(f"Client {client_state.id}: Overall Offline transcription handler failed: {type(e).__name__} {e}")
    finally:
        audio_logger.info(f"Client {client_state.id}: Offline transcription handler fully terminated.")


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
        self.pcm_frames: asyncio.Queue[Optional[bytes]] = asyncio.Queue()  # raw PCM chunks, Optional for None sentinel
        self.save_pcm_frames: List[bytes] = []  # raw PCM chunks for WAV saving
        self.sample_count = 0  # samples accumulated in current segment
        self.segment_index = 0
        self.start_time = datetime.utcnow().isoformat()
        self.decoder = OmiOpusDecoder()

    # ----- Private helpers -------------------------------------------------- #
    async def _decode_packet_async(self, packet: bytes) -> Optional[bytes]:
        """Off-load Opus decode to thread-pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _DEC_IO_EXECUTOR,
            # lambda to pass keyword arg
            lambda: self.decoder.decode_packet(packet, strip_header=False)
        )
    
    async def _write_wav_async(self, path: Path, pcm_bytes: bytes):
        """Blocking wave-write off-loaded."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _DEC_IO_EXECUTOR,
            self._write_wav_sync, path, pcm_bytes
        )
    
    @staticmethod
    def _write_wav_sync(path: Path, pcm_bytes: bytes):
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_bytes)

    # --------------------------------------------------------------------- #
    async def add_opus_packet(self, packet: bytes):
        """Decode a single Opus packet -> PCM, buffer it, handle rollover."""
        pcm = await self._decode_packet_async(packet)
        if pcm is None:
            audio_logger.error("Client %s: decode failed (packet dropped)", self.id)
            return
        audio_logger.debug("Client %s: decoded packet", self.id)

        await self.pcm_frames.put(pcm)
        self.save_pcm_frames.append(pcm)

        self.sample_count += len(pcm) // SAMPLE_WIDTH  # samples per frame
        
        # micro-yield so other tasks run even if packets flood in
        await asyncio.sleep(0)

        if self.sample_count >= TARGET_SAMPLES:
            await self._flush_segment()

    # ------------------------------------------------------------------ #
    async def _flush_segment(self):
        if not self.save_pcm_frames:
            return

        pcm_bytes = b"".join(self.save_pcm_frames)
        duration = self.sample_count / SAMPLE_RATE

        # Write WAV to disk, offloaded to thread pool
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        wav_path = CHUNK_DIR / f"client{self.id}_{timestamp}_{self.segment_index}.wav"
        await self._write_wav_async(wav_path, pcm_bytes)
        audio_logger.info("Client %s: wrote %s (%.1fs)", self.id, wav_path.name, duration)

        # Reset buffers for WAV saving
        self.save_pcm_frames.clear()
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

    # Start the continuous Offline transcription task for this client
    transcription_task = asyncio.create_task(
        continuous_offline_transcription_handler(state, OFFLINE_ASR_WS_URI)
    )

    try:
        while True:
            try:
                packet = await ws.receive_bytes()
            except WebSocketDisconnect:
                audio_logger.info(f"Client {client_id} WebSocket disconnected by client.")
                break  # Graceful client close
            except Exception as e: # Catch other errors on receive_bytes
                audio_logger.error(f"Client {client_id}: Error receiving packet: {type(e).__name__} {e}")
                break
            await state.add_opus_packet(packet)
            # yield to avoid starving write/recv tasks
            await asyncio.sleep(0)
    except Exception as e: 
        audio_logger.error(f"Client {client_id}: Unhandled error in main WebSocket loop: {type(e).__name__} {e}")
    finally:
        audio_logger.info(f"Client {client_id} disconnecting sequence initiated...")
        
        # Signal the transcription handler's PCM sending loop to stop
        # This needs to be done before attempting to remove the client or await the task
        await state.pcm_frames.put(None) 
        
        if state in clients: 
            clients.remove(state)
        
        # Wait for the transcription task to finish cleaning up
        if transcription_task and not transcription_task.done():
            audio_logger.info(f"Client {client_id}: Waiting for transcription task (up to 5s)...")
            try:
                await asyncio.wait_for(transcription_task, timeout=5.0)
                audio_logger.info(f"Client {client_id}: Transcription task completed.")
            except asyncio.TimeoutError:
                audio_logger.warning(f"Client {client_id}: Timeout waiting for transcription task. Cancelling.")
                transcription_task.cancel()
                try:
                    await transcription_task # Allow cancellation to be processed
                except asyncio.CancelledError:
                    audio_logger.info(f"Client {client_id}: Transcription task cancelled successfully.")
                except Exception as e_cancel:
                     audio_logger.error(f"Client {client_id}: Error during transcription task cancellation: {type(e_cancel).__name__} {e_cancel}")
            except Exception as e_shutdown:
                audio_logger.error(f"Client {client_id}: Error during transcription task graceful shutdown: {type(e_shutdown).__name__} {e_shutdown}")
                transcription_task.cancel() # Ensure cancellation on other errors

        # Flush any remaining audio segment to disk
        # This check is important to ensure there's something to flush.
        if state.sample_count > 0 or len(state.save_pcm_frames) > 0 : 
            audio_logger.info(f"Client {client_id}: Flushing final audio segment to disk.")
            await state._flush_segment()

        audio_logger.info("Client %s disconnected fully.", client_id)
    

###############################################################################
# Entrypoint
###############################################################################
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    audio_logger.info("Starting Omi unified service at ws://%s:%s/ws", host, port)
    uvicorn.run("main:app", host=host, port=port, reload=False)