"""
Transcription provider abstraction for multiple ASR services (online and offline).

Provider Output Formats (2025):
--------------------------------
All providers return a standardized dictionary with the following structure:
{
    "text": str,        # Full transcript text
    "words": List[dict],  # Word-level data (if available)
    "segments": List[dict]  # Speaker segments (if available)
}

Word object format (when available):
{
    "word": str,        # The word text
    "start": float,     # Start time in seconds
    "end": float,       # End time in seconds
    "confidence": float,  # Confidence score (0-1)
    "speaker": int      # Speaker ID (optional)
}

Provider-specific behaviors:
- Deepgram: Returns rich word-level timestamps with confidence scores
- NeMo Parakeet: Returns word-level timestamps (streaming and batch modes)
"""

import abc
import asyncio
import json
import logging
import os
import tempfile
import uuid
from typing import Dict, Optional

import httpx
import numpy as np
import websockets
from easy_audio_interfaces.audio_interfaces import AudioChunk
from easy_audio_interfaces.filesystem import LocalFileSink

logger = logging.getLogger(__name__)

class BaseTranscriptionProvider(abc.ABC):
    """Abstract base class for all transcription providers."""

    @abc.abstractmethod
    async def transcribe(self, audio_data: bytes, sample_rate: int, **kwargs) -> dict:
        """
        Transcribe audio data to text with word-level timestamps.

        Args:
            audio_data: Raw audio bytes (PCM format)
            sample_rate: Audio sample rate (Hz)
            **kwargs: Additional parameters (e.g. diarize=True for speaker diarization)

        Returns:
            Dictionary containing:
            - text: Transcribed text string
            - words: List of word-level data with timestamps (required)
            - segments: List of speaker segments (empty for non-RTTM providers)
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the provider name for logging."""
        pass

    @property
    @abc.abstractmethod
    def mode(self) -> str:
        """Return 'streaming' or 'batch' for processing mode."""
        pass

    async def connect(self, client_id: Optional[str] = None):
        """Initialize/connect the provider. Default implementation does nothing."""
        pass

    async def disconnect(self):
        """Cleanup/disconnect the provider. Default implementation does nothing."""
        pass


class StreamingTranscriptionProvider(BaseTranscriptionProvider):
    """Base class for streaming transcription providers."""

    @property
    def mode(self) -> str:
        return "streaming"

    @abc.abstractmethod
    async def start_stream(self, client_id: str, sample_rate: int = 16000, diarize: bool = False):
        """Start a transcription stream for a client.
        
        Args:
            client_id: Unique client identifier
            sample_rate: Audio sample rate
            diarize: Whether to enable speaker diarization (provider-dependent)
        """
        pass

    @abc.abstractmethod
    async def process_audio_chunk(self, client_id: str, audio_chunk: bytes) -> Optional[dict]:
        """
        Process audio chunk and return partial/final transcription.
        
        Returns:
            None for partial results, dict with transcription for final results
        """
        pass

    @abc.abstractmethod
    async def end_stream(self, client_id: str) -> dict:
        """End stream and return final transcription with word-level timestamps."""
        pass


class BatchTranscriptionProvider(BaseTranscriptionProvider):
    """Base class for batch transcription providers."""

    @property
    def mode(self) -> str:
        return "batch"
    
    @abc.abstractmethod
    async def transcribe(self, audio_data: bytes, sample_rate: int, diarize: bool = False) -> dict:
        """Transcribe audio data.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
            diarize: Whether to enable speaker diarization (provider-dependent)
        """
        pass


class DeepgramProvider(BatchTranscriptionProvider):
    """Deepgram batch transcription provider using Nova-3 model."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.deepgram.com/v1/listen"

    @property
    def name(self) -> str:
        return "Deepgram"

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

                        # Use diarized transcript if available
                        if "paragraphs" in alternative and alternative["paragraphs"].get(
                            "transcript"
                        ):
                            transcript = alternative["paragraphs"]["transcript"].strip()
                            logger.info(
                                f"Deepgram diarized transcription successful: {len(transcript)} characters"
                            )
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
                                    f"Keeping raw transcript with word-level data: {len(transcript)} characters"
                                )
                                return {
                                    "text": transcript,
                                    "words": words,
                                    "segments": [],
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
        return "Deepgram-Streaming"

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


class ParakeetProvider(BatchTranscriptionProvider):
    """Parakeet HTTP batch transcription provider."""

    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip('/')
        self.transcribe_url = f"{self.service_url}/transcribe"

    @property
    def name(self) -> str:
        return "Parakeet"

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
            return {"text": "", "words": [], "segments": []}


class ParakeetStreamingProvider(StreamingTranscriptionProvider):
    """Parakeet WebSocket streaming transcription provider."""

    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip('/')
        self.ws_url = service_url.replace("http://", "ws://").replace("https://", "wss://") + "/stream"
        self._streams: Dict[str, Dict] = {}  # client_id -> stream data

    @property
    def name(self) -> str:
        return "Parakeet-Streaming"

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


def get_transcription_provider(
    provider_name: Optional[str] = None,
    mode: Optional[str] = None,
) -> Optional[BaseTranscriptionProvider]:
    """
    Factory function to get the appropriate transcription provider.

    Args:
        provider_name: Name of the provider ('deepgram', 'parakeet').
                      If None, will auto-select based on available configuration.
        mode: Processing mode ('streaming', 'batch'). If None, defaults to 'batch'.

    Returns:
        An instance of BaseTranscriptionProvider, or None if no provider is configured.

    Raises:
        RuntimeError: If a specific provider is requested but not properly configured.
    """
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    parakeet_url = os.getenv("PARAKEET_ASR_URL")
    
    if provider_name:
        provider_name = provider_name.lower()
    
    if mode is None:
        mode = "batch"
    mode = mode.lower()

    # Handle specific provider requests
    if provider_name == "deepgram":
        if not deepgram_key:
            raise RuntimeError(
                "Deepgram transcription provider requested but DEEPGRAM_API_KEY not configured"
            )
        logger.info(f"Using Deepgram transcription provider in {mode} mode")
        if mode == "streaming":
            return DeepgramStreamingProvider(deepgram_key)
        else:
            return DeepgramProvider(deepgram_key)

    elif provider_name == "parakeet":
        if not parakeet_url:
            raise RuntimeError(
                "Parakeet ASR provider requested but PARAKEET_ASR_URL not configured"
            )
        logger.info(f"Using Parakeet transcription provider in {mode} mode")
        return ParakeetProvider(parakeet_url)
    
    elif provider_name == "offline":
        # "offline" is an alias for Parakeet ASR
        if not parakeet_url:
            raise RuntimeError(
                "Offline transcription provider requested but PARAKEET_ASR_URL not configured"
            )
        logger.info(f"Using offline Parakeet transcription provider in {mode} mode")
        return ParakeetProvider(parakeet_url)
    else:
        return None
