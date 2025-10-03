"""
Deepgram transcription provider implementations.

Provides both batch and streaming transcription using Deepgram's Nova-3 model.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Optional

import httpx
import websockets

from advanced_omi_backend.models.transcription import (
    BatchTranscriptionProvider,
    StreamingTranscriptionProvider,
)

logger = logging.getLogger(__name__)

class DeepgramProvider(BatchTranscriptionProvider):
    """Deepgram batch transcription provider using Nova-3 model."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.deepgram.com/v1/listen"

    @property
    def name(self) -> str:
        return "deepgram"

    async def transcribe(self, audio_data: bytes, sample_rate: int, diarize: bool = False) -> dict:
        """Transcribe audio using Deepgram's REST API.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
            diarize: Whether to enable speaker diarization
        """
        try:
            params = {
                "model": "nova-3",
                "language": "multi",
                "smart_format": "true",
                "punctuate": "true",
                "diarize": "true" if diarize else "false",
                "encoding": "linear16",
                "sample_rate": str(sample_rate),
                "channels": "1",
            }

            headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "audio/raw"}

            logger.info(f"Sending {len(audio_data)} bytes to Deepgram API")

            # Calculate dynamic timeout based on audio file size
            estimated_duration = len(audio_data) / (sample_rate * 2 * 1)  # 16-bit mono
            processing_timeout = max(
                120, int(estimated_duration * 3)
            )  # Min 2 minutes, 3x audio duration

            timeout_config = httpx.Timeout(
                connect=30.0,
                read=processing_timeout,
                write=max(
                    180.0, int(len(audio_data) / (sample_rate * 2))
                ),  # bytes per second for 16-bit PCM
                pool=10.0,
            )

            logger.info(
                f"Estimated audio duration: {estimated_duration:.1f}s, timeout: {processing_timeout}s"
            )

            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    self.url, params=params, headers=headers, content=audio_data
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"Deepgram response: {result}")

                    # Extract transcript from response
                    if result.get("results", {}).get("channels", []) and result["results"][
                        "channels"
                    ][0].get("alternatives", []):

                        alternative = result["results"]["channels"][0]["alternatives"][0]

                        # Extract segments from diarized utterances if available
                        segments = []
                        if "paragraphs" in alternative and alternative["paragraphs"].get("paragraphs"):
                            transcript = alternative["paragraphs"]["transcript"].strip()
                            logger.info(
                                f"Deepgram diarized transcription successful: {len(transcript)} characters"
                            )

                            # Extract speaker segments from utterances
                            for paragraph in alternative["paragraphs"]["paragraphs"]:
                                for sentence in paragraph.get("sentences", []):
                                    segments.append({
                                        "text": sentence.get("text", "").strip(),
                                        "speaker": f"Speaker {paragraph.get('speaker', 'unknown')}",
                                        "start": sentence.get("start", 0),
                                        "end": sentence.get("end", 0),
                                        "confidence": None  # Deepgram doesn't provide segment-level confidence
                                    })
                        else:
                            transcript = alternative.get("transcript", "").strip()
                            logger.info(
                                f"Deepgram basic transcription successful: {len(transcript)} characters"
                            )

                        if transcript:
                            # Extract speech timing information for logging
                            words = alternative.get("words", [])
                            if words:
                                first_word_start = words[0].get("start", 0)
                                last_word_end = words[-1].get("end", 0)
                                speech_duration = last_word_end - first_word_start

                                # Calculate audio duration from data size
                                audio_duration = len(audio_data) / (
                                    sample_rate * 2 * 1
                                )  # 16-bit mono
                                speech_percentage = (
                                    (speech_duration / audio_duration) * 100
                                    if audio_duration > 0
                                    else 0
                                )

                                logger.info(
                                    f"Deepgram speech analysis: {speech_duration:.1f}s speech detected in {audio_duration:.1f}s audio ({speech_percentage:.1f}%)"
                                )

                                # Check confidence levels
                                confidences = [
                                    w.get("confidence", 0) for w in words if "confidence" in w
                                ]
                                if confidences:
                                    avg_confidence = sum(confidences) / len(confidences)
                                    low_confidence_count = sum(1 for c in confidences if c < 0.5)
                                    logger.info(
                                        f"Deepgram confidence: avg={avg_confidence:.2f}, {low_confidence_count}/{len(words)} words <0.5 confidence"
                                    )

                                # Keep raw transcript and word data without formatting
                                logger.info(
                                    f"Keeping raw transcript with word-level data: {len(transcript)} characters, {len(segments)} segments"
                                )
                                return {
                                    "text": transcript,
                                    "words": words,
                                    "segments": segments,
                                }
                            else:
                                # No word-level data, return basic transcript
                                logger.info(
                                    "No word-level data available, returning basic transcript"
                                )
                                return {"text": transcript, "words": [], "segments": []}
                        else:
                            logger.warning("Deepgram returned empty transcript")
                            return {"text": "", "words": [], "segments": []}
                    else:
                        logger.warning("Deepgram response missing expected transcript structure")
                        return {"text": "", "words": [], "segments": []}
                else:
                    logger.error(f"Deepgram API error: {response.status_code} - {response.text}")
                    return {"text": "", "words": [], "segments": []}

        except httpx.TimeoutException as e:
            timeout_type = "unknown"
            if "connect" in str(e).lower():
                timeout_type = "connection"
            elif "read" in str(e).lower():
                timeout_type = "read"
            elif "write" in str(e).lower():
                timeout_type = "write (upload)"
            elif "pool" in str(e).lower():
                timeout_type = "connection pool"
            logger.error(
                f"HTTP {timeout_type} timeout during Deepgram API call for {len(audio_data)} bytes: {e}"
            )
            return {"text": "", "words": [], "segments": []}
        except Exception as e:
            logger.error(f"Error calling Deepgram API: {e}")
            return {"text": "", "words": [], "segments": []}


class DeepgramStreamingProvider(StreamingTranscriptionProvider):
    """Deepgram streaming transcription provider using WebSocket connection."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws_url = "wss://api.deepgram.com/v1/listen"
        self._streams: Dict[str, Dict] = {}  # client_id -> stream data

    @property
    def name(self) -> str:
        return "deepgram"

    async def start_stream(self, client_id: str, sample_rate: int = 16000, diarize: bool = False):
        """Start a WebSocket connection for streaming transcription.
        
        Args:
            client_id: Unique client identifier
            sample_rate: Audio sample rate
            diarize: Whether to enable speaker diarization
        """
        try:
            logger.info(f"Starting Deepgram streaming for client {client_id} (diarize={diarize})")
            
            # WebSocket connection parameters
            params = {
                "model": "nova-3",
                "language": "multi",
                "smart_format": "true",
                "punctuate": "true",
                "diarize": "true" if diarize else "false",
                "encoding": "linear16",
                "sample_rate": str(sample_rate),
                "channels": "1",
                "interim_results": "true",
                "endpointing": "300",  # 300ms silence for endpoint detection
            }

            # Build WebSocket URL with parameters
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            ws_url = f"{self.ws_url}?{query_string}"
            
            # Connect to WebSocket
            websocket = await websockets.connect(
                ws_url,
                extra_headers={"Authorization": f"Token {self.api_key}"}
            )
            
            # Store stream data
            self._streams[client_id] = {
                "websocket": websocket,
                "final_transcript": "",
                "words": [],
                "stream_id": str(uuid.uuid4())
            }
            
            logger.info(f"Deepgram WebSocket connected for client {client_id}")
            
        except Exception as e:
            logger.error(f"Failed to start Deepgram streaming for {client_id}: {e}")
            raise

    async def process_audio_chunk(self, client_id: str, audio_chunk: bytes) -> Optional[dict]:
        """Send audio chunk to WebSocket and process responses."""
        if client_id not in self._streams:
            logger.error(f"No active stream for client {client_id}")
            return None
            
        try:
            stream_data = self._streams[client_id]
            websocket = stream_data["websocket"]
            
            # Send audio chunk
            await websocket.send(audio_chunk)
            
            # Check for responses (non-blocking)
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    result = json.loads(response)
                    
                    if result.get("type") == "Results":
                        channel = result.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        
                        if alternatives:
                            alt = alternatives[0]
                            is_final = channel.get("is_final", False)
                            
                            if is_final:
                                # Accumulate final transcript and words
                                transcript = alt.get("transcript", "")
                                words = alt.get("words", [])
                                
                                if transcript.strip():
                                    stream_data["final_transcript"] += transcript + " "
                                    stream_data["words"].extend(words)
                                    
                                logger.debug(f"Final transcript chunk: {transcript}")
                                    
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
            
            # Send close message
            close_msg = json.dumps({"type": "CloseStream"})
            await websocket.send(close_msg)
            
            # Wait a bit for final responses
            try:
                end_time = asyncio.get_event_loop().time() + 2.0  # 2 second timeout
                while asyncio.get_event_loop().time() < end_time:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    result = json.loads(response)
                    
                    if result.get("type") == "Results":
                        channel = result.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        
                        if alternatives and channel.get("is_final", False):
                            alt = alternatives[0]
                            transcript = alt.get("transcript", "")
                            words = alt.get("words", [])
                            
                            if transcript.strip():
                                stream_data["final_transcript"] += transcript
                                stream_data["words"].extend(words)
                                
            except asyncio.TimeoutError:
                pass
                
            # Close WebSocket
            await websocket.close()
            
            # Prepare final result
            final_transcript = stream_data["final_transcript"].strip()
            final_words = stream_data["words"]
            
            logger.info(f"Deepgram streaming completed for {client_id}: {len(final_transcript)} chars, {len(final_words)} words")
            
            # Clean up
            del self._streams[client_id]
            
            return {
                "text": final_transcript,
                "words": final_words,
                "segments": []
            }
            
        except Exception as e:
            logger.error(f"Error ending stream for {client_id}: {e}")
            # Clean up on error
            if client_id in self._streams:
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
        
        logger.info("All Deepgram streaming connections closed")


