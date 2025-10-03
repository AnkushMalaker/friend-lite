"""
Parakeet (NeMo) transcription provider implementations.

Provides both batch and streaming transcription using NeMo's Parakeet ASR models.
"""

import asyncio
import json
import logging
import tempfile
from typing import Dict, Optional

import httpx
import numpy as np
import websockets
from easy_audio_interfaces.audio_interfaces import AudioChunk
from easy_audio_interfaces.filesystem import LocalFileSink

from advanced_omi_backend.models.transcription import (
    BatchTranscriptionProvider,
    StreamingTranscriptionProvider,
)

logger = logging.getLogger(__name__)

class ParakeetProvider(BatchTranscriptionProvider):
    """Parakeet HTTP batch transcription provider."""

    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip('/')
        self.transcribe_url = f"{self.service_url}/transcribe"

    @property
    def name(self) -> str:
        return "parakeet"

    async def transcribe(self, audio_data: bytes, sample_rate: int, **kwargs) -> dict:
        """Transcribe audio using Parakeet HTTP service."""
        try:
            
            logger.info(f"Sending {len(audio_data)} bytes to Parakeet service at {self.transcribe_url}")
            
            # Convert PCM bytes to audio file for upload
            if sample_rate != 16000:
                logger.warning(f"Sample rate {sample_rate} != 16000, audio may not be optimal")
            
            # Assume 16-bit PCM
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / np.iinfo(np.int16).max  # Normalize to [-1, 1]
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # sf.write(tmp_file.name, audio_array, 16000)  # Force 16kHz
                async with LocalFileSink(tmp_file.name, sample_rate, 1) as sink:
                    await sink.write(AudioChunk(
                        rate=sample_rate,
                        width=2,
                        channels=1,
                        audio=audio_data,
                    ))

                tmp_filename = tmp_file.name
            
            try:
                # Upload file to Parakeet service
                async with httpx.AsyncClient(timeout=180.0) as client:
                    with open(tmp_filename, "rb") as f:
                        files = {"file": ("audio.wav", f, "audio/wav")}
                        response = await client.post(self.transcribe_url, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Parakeet transcription successful: {len(result.get('text', ''))} chars, {len(result.get('words', []))} words")
                    return result
                else:
                    error_msg = f"Parakeet service error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    
                    # For 5xx errors, raise exception to trigger retry/failure handling
                    if response.status_code >= 500:
                        raise RuntimeError(f"Parakeet service unavailable: HTTP {response.status_code}")
                    
                    # For 4xx errors, return empty result (client error, won't retry)
                    return {"text": "", "words": [], "segments": []}
                    
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_filename):
                    os.unlink(tmp_filename)
                    
        except Exception as e:
            logger.error(f"Error calling Parakeet service: {e}")
            raise e


class ParakeetStreamingProvider(StreamingTranscriptionProvider):
    """Parakeet WebSocket streaming transcription provider."""

    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip('/')
        self.ws_url = service_url.replace("http://", "ws://").replace("https://", "wss://") + "/stream"
        self._streams: Dict[str, Dict] = {}  # client_id -> stream data

    @property
    def name(self) -> str:
        return "parakeet"

    async def start_stream(self, client_id: str, sample_rate: int = 16000, diarize: bool = False):
        """Start a WebSocket connection for streaming transcription.
        
        Args:
            client_id: Unique client identifier
            sample_rate: Audio sample rate
            diarize: Whether to enable speaker diarization (ignored - Parakeet doesn't support diarization)
        """
        if diarize:
            logger.warning(f"Parakeet streaming provider does not support diarization, ignoring diarize=True for client {client_id}")
        try:
            logger.info(f"Starting Parakeet streaming for client {client_id}")
            
            # Connect to WebSocket
            websocket = await websockets.connect(self.ws_url)
            
            # Send transcribe event to start session
            session_config = {
                "vad_enabled": True,
                "vad_silence_ms": 1000,
                "time_interval_seconds": 30,
                "return_interim_results": True,
                "min_audio_seconds": 0.5
            }
            
            start_message = {
                "type": "transcribe",
                "session_id": client_id,
                "config": session_config
            }
            
            await websocket.send(json.dumps(start_message))
            
            # Wait for session_started confirmation
            response = await websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("type") != "session_started":
                raise RuntimeError(f"Failed to start session: {response_data}")
            
            # Store stream data
            self._streams[client_id] = {
                "websocket": websocket,
                "sample_rate": sample_rate,
                "session_id": client_id,
                "interim_results": [],
                "final_result": None
            }
            
            logger.info(f"Parakeet WebSocket connected for client {client_id}")
            
        except Exception as e:
            logger.error(f"Failed to start Parakeet streaming for {client_id}: {e}")
            raise

    async def process_audio_chunk(self, client_id: str, audio_chunk: bytes) -> Optional[dict]:
        """Send audio chunk to WebSocket and process responses."""
        if client_id not in self._streams:
            logger.error(f"No active stream for client {client_id}")
            return None
            
        try:
            stream_data = self._streams[client_id]
            websocket = stream_data["websocket"]
            sample_rate = stream_data["sample_rate"]
            
            # Send audio_chunk event
            chunk_message = {
                "type": "audio_chunk",
                "session_id": client_id,
                "rate": sample_rate,
                "width": 2,  # 16-bit
                "channels": 1
            }
            
            await websocket.send(json.dumps(chunk_message))
            await websocket.send(audio_chunk)
            
            # Check for responses (non-blocking)
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    result = json.loads(response)
                    
                    if result.get("type") == "interim_result":
                        # Store interim result but don't return it (handled by backend differently)
                        stream_data["interim_results"].append(result)
                        logger.debug(f"Received interim result: {result.get('text', '')[:50]}...")
                    elif result.get("type") == "final_result":
                        # This shouldn't happen during chunk processing, but store it
                        stream_data["final_result"] = result
                        logger.debug(f"Received final result during chunk processing: {result.get('text', '')[:50]}...")
                        
            except asyncio.TimeoutError:
                # No response available, continue
                pass
                
            return None  # Streaming, no final result yet
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for {client_id}: {e}")
            return None

    async def end_stream(self, client_id: str) -> dict:
        """Close WebSocket connection and return final transcription."""
        if client_id not in self._streams:
            logger.error(f"No active stream for client {client_id}")
            return {"text": "", "words": [], "segments": []}
            
        try:
            stream_data = self._streams[client_id]
            websocket = stream_data["websocket"]
            
            # Send finalize event
            finalize_message = {
                "type": "finalize",
                "session_id": client_id
            }
            await websocket.send(json.dumps(finalize_message))
            
            # Wait for final result
            try:
                end_time = asyncio.get_event_loop().time() + 5.0  # 5 second timeout
                while asyncio.get_event_loop().time() < end_time:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    result = json.loads(response)
                    
                    if result.get("type") == "final_result":
                        stream_data["final_result"] = result
                        break
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for final result from {client_id}")
                
            # Close WebSocket
            await websocket.close()
            
            # Prepare final result
            final_result = stream_data.get("final_result")
            if final_result:
                result_data = {
                    "text": final_result.get("text", ""),
                    "words": final_result.get("words", []),
                    "segments": final_result.get("segments", [])
                }
            else:
                # Fallback: aggregate interim results if no final result received
                interim_texts = [r.get("text", "") for r in stream_data["interim_results"]]
                all_words = []
                for r in stream_data["interim_results"]:
                    all_words.extend(r.get("words", []))
                    
                result_data = {
                    "text": " ".join(interim_texts),
                    "words": all_words,
                    "segments": []
                }
            
            logger.info(f"Parakeet streaming completed for {client_id}: {len(result_data.get('text', ''))} chars")
            
            # Clean up
            del self._streams[client_id]
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error ending stream for {client_id}: {e}")
            # Clean up on error
            if client_id in self._streams:
                try:
                    await self._streams[client_id]["websocket"].close()
                except:
                    pass
                del self._streams[client_id]
            return {"text": "", "words": [], "segments": []}

    async def transcribe(self, audio_data: bytes, sample_rate: int, **kwargs) -> dict:
        """For streaming provider, this method is not typically used."""
        logger.warning("transcribe() called on streaming provider - use streaming methods instead")
        return {"text": "", "words": [], "segments": []}

    async def disconnect(self):
        """Close all active WebSocket connections."""
        for client_id in list(self._streams.keys()):
            try:
                websocket = self._streams[client_id]["websocket"]
                await websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket for {client_id}: {e}")
            finally:
                del self._streams[client_id]
        
        logger.info("All Parakeet streaming connections closed")


