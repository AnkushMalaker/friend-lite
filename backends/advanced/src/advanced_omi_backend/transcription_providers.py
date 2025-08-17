"""
Transcription provider abstraction for multiple online ASR services.
"""

import abc
import logging
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class OnlineTranscriptionProvider(abc.ABC):
    """Abstract base class for online transcription providers."""

    @abc.abstractmethod
    async def transcribe(self, audio_data: bytes, sample_rate: int) -> dict:
        """
        Transcribe audio data to text with optional speaker diarization.

        Args:
            audio_data: Raw audio bytes (PCM format)
            sample_rate: Audio sample rate (Hz)

        Returns:
            Dictionary containing:
            - text: Transcribed text string
            - segments: List of speaker segments (if diarization available)
            - words: List of word-level data with timestamps and speakers
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the provider name for logging."""
        pass


class DeepgramProvider(OnlineTranscriptionProvider):
    """Deepgram transcription provider using Nova-3 model."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.deepgram.com/v1/listen"

    @property
    def name(self) -> str:
        return "Deepgram"

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> dict:
        """Transcribe audio using Deepgram's REST API."""
        try:
            params = {
                "model": "nova-3",
                "language": "multi",
                "smart_format": "true",
                "punctuate": "true",
                "diarize": "false",
                "encoding": "linear16",
                "sample_rate": str(sample_rate),
                "channels": "1",
            }

            headers = {"Authorization": f"Token {self.api_key}", "Content-Type": "audio/raw"}

            logger.info(f"Sending {len(audio_data)} bytes to Deepgram API")

            estimated_duration = len(audio_data) / (sample_rate * 2 * 1)
            processing_timeout = max(120, int(estimated_duration * 3))

            timeout_config = httpx.Timeout(
                connect=30.0,
                read=processing_timeout,
                write=max(180.0, int(len(audio_data) / (sample_rate * 2))),
                pool=10.0,
            )

            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    self.url, params=params, headers=headers, content=audio_data
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get("results", {}).get("channels", []) and result["results"][
                        "channels"
                    ][0].get("alternatives", []):
                        alternative = result["results"]["channels"][0]["alternatives"][0]
                        transcript = alternative.get("transcript", "").strip()
                        if transcript:
                            return {
                                "text": transcript,
                                "words": alternative.get("words", []),
                                "segments": [],
                            }
                    return {"text": "", "words": [], "segments": []}
                else:
                    logger.error(f"Deepgram API error: {response.status_code} - {response.text}")
                    return {"text": "", "words": [], "segments": []}
        except Exception as e:
            logger.error(f"Error calling Deepgram API: {e}")
            return {"text": "", "words": [], "segments": []}


class MistralProvider(OnlineTranscriptionProvider):
    """Mistral transcription provider using Voxtral models."""

    def __init__(self, api_key: str, model: str = "voxtral-mini-2507"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.mistral.ai/v1/audio/transcriptions"

    @property
    def name(self) -> str:
        return "Mistral"

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> dict:
        """Transcribe audio using Mistral's REST API."""
        try:
            wav_data = self._create_wav_header(audio_data, sample_rate) + audio_data
            headers = {"x-api-key": self.api_key}
            files = {"file": ("audio.wav", wav_data, "audio/wav")}
            data = {"model": self.model}
            
            estimated_duration = len(audio_data) / (sample_rate * 2 * 1)
            processing_timeout = max(120, int(estimated_duration * 3))
            timeout_config = httpx.Timeout(connect=30.0, read=processing_timeout)

            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(self.url, headers=headers, files=files, data=data)

                if response.status_code == 200:
                    result = response.json()
                    transcript = result.get("text", "").strip()
                    return {"text": transcript, "words": [], "segments": []}
                else:
                    logger.error(f"Mistral API error: {response.status_code} - {response.text}")
                    return {"text": "", "words": [], "segments": []}
        except Exception as e:
            logger.error(f"Error calling Mistral API: {e}")
            return {"text": "", "words": [], "segments": []}

    def _create_wav_header(self, audio_data: bytes, sample_rate: int) -> bytes:
        """Create a WAV header for raw PCM data."""
        import struct
        channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = len(audio_data)
        file_size = data_size + 36
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", file_size, b"WAVE", b"fmt ", 16, 1,
            channels, sample_rate, byte_rate, block_align,
            bits_per_sample, b"data", data_size,
        )
        return header


class HuggingFaceProvider(OnlineTranscriptionProvider):
    """Hugging Face transcription provider using Inference API."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.url = f"https://api-inference.huggingface.co/models/{model}"

    @property
    def name(self) -> str:
        return "Hugging Face"

    async def transcribe(self, audio_data: bytes, sample_rate: int) -> dict:
        """Transcribe audio using Hugging Face's Inference API."""
        try:
            # HF API works best with WAV files.
            wav_data = self._create_wav_header(audio_data, sample_rate) + audio_data

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "audio/wav",
            }
            logger.info(f"Sending {len(wav_data)} bytes to Hugging Face API with model {self.model}")
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.url, headers=headers, content=wav_data)

                if response.status_code == 200:
                    result = response.json()
                    if "error" in result:
                        logger.error(f"Hugging Face API returned error: {result['error']}")
                        return {"text": "", "words": [], "segments": []}

                    transcript = result.get("text", "").strip()
                    return {"text": transcript, "words": [], "segments": []}
                else:
                    logger.error(f"Hugging Face API error: {response.status_code} - {response.text}")
                    return {"text": "", "words": [], "segments": []}
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {e}", exc_info=True)
            return {"text": "", "words": [], "segments": []}

    def _create_wav_header(self, audio_data: bytes, sample_rate: int) -> bytes:
        """Create a WAV header for raw PCM data."""
        import struct
        channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = len(audio_data)
        file_size = data_size + 36
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", file_size, b"WAVE", b"fmt ", 16, 1,
            channels, sample_rate, byte_rate, block_align,
            bits_per_sample, b"data", data_size,
        )
        return header


def get_transcription_provider(
    provider_name: Optional[str] = None,
) -> Optional[OnlineTranscriptionProvider]:
    """Factory function to get the appropriate transcription provider."""
    import os

    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    mistral_key = os.getenv("MISTRAL_API_KEY")
    mistral_model = os.getenv("MISTRAL_MODEL", "voxtral-mini-2507")
    hf_key = os.getenv("HF_API_KEY")
    hf_model = os.getenv("HF_TRANSCRIPTION_MODEL", "openai/whisper-large-v3")

    if provider_name:
        provider_name = provider_name.lower()

    if provider_name == "deepgram" and deepgram_key:
        return DeepgramProvider(deepgram_key)
    elif provider_name == "mistral" and mistral_key:
        return MistralProvider(mistral_key, mistral_model)
    elif provider_name == "huggingface" and hf_key:
        return HuggingFaceProvider(hf_key, hf_model)
    elif provider_name is None:
        if deepgram_key:
            return DeepgramProvider(deepgram_key)
        elif mistral_key:
            return MistralProvider(mistral_key, mistral_model)
        elif hf_key:
            return HuggingFaceProvider(hf_key, hf_model)

    if provider_name:
        raise RuntimeError(
            f"Requested transcription provider '{provider_name}' is not configured "
            f"(check your API keys in .env)"
        )
    
    return None
