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
    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes (PCM format)
            sample_rate: Audio sample rate (default 16000 Hz)

        Returns:
            Transcribed text string
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

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Transcribe audio using Deepgram's REST API."""
        try:
            params = {
                "model": "nova-3",
                "language": "multi",
                "smart_format": "true",
                "punctuate": "true",
                "diarize": "true",
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
                write=max(180.0, int(len(audio_data) / 16000)),
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
                            # Clean up speaker labels
                            cleaned_transcript = re.sub(
                                r"^[\s\n]*Speaker \d+:\s*", "", transcript, flags=re.MULTILINE
                            )
                            cleaned_transcript = re.sub(
                                r"\n\s*Speaker \d+:\s*", " ", cleaned_transcript
                            )
                            cleaned_transcript = cleaned_transcript.strip()

                            logger.info(
                                f"Cleaned transcript from {len(transcript)} to {len(cleaned_transcript)} characters"
                            )
                            return cleaned_transcript
                        else:
                            logger.warning("Deepgram returned empty transcript")
                            return ""
                    else:
                        logger.warning("Deepgram response missing expected transcript structure")
                        return ""
                else:
                    logger.error(f"Deepgram API error: {response.status_code} - {response.text}")
                    return ""

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
            return ""
        except Exception as e:
            logger.error(f"Error calling Deepgram API: {e}")
            return ""


class MistralProvider(OnlineTranscriptionProvider):
    """Mistral transcription provider using Voxtral models."""

    def __init__(self, api_key: str, model: str = "voxtral-mini-2507"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.mistral.ai/v1/audio/transcriptions"

    @property
    def name(self) -> str:
        return "Mistral"

    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Transcribe audio using Mistral's REST API."""
        try:
            # Mistral expects audio files, so we need to send it as a file upload
            # Convert raw PCM to WAV format by adding WAV header
            wav_data = self._create_wav_header(audio_data, sample_rate) + audio_data

            headers = {
                "x-api-key": self.api_key,
            }

            # Prepare multipart form data
            files = {"file": ("audio.wav", wav_data, "audio/wav")}

            data = {"model": self.model}

            logger.info(f"Sending {len(wav_data)} bytes to Mistral API with model {self.model}")

            # Calculate timeout based on audio duration
            estimated_duration = len(audio_data) / (sample_rate * 2 * 1)  # 16-bit mono
            processing_timeout = max(120, int(estimated_duration * 3))

            timeout_config = httpx.Timeout(
                connect=30.0,
                read=processing_timeout,
                write=max(180.0, int(len(wav_data) / 16000)),
                pool=10.0,
            )

            logger.info(
                f"Estimated audio duration: {estimated_duration:.1f}s, timeout: {processing_timeout}s"
            )

            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(self.url, headers=headers, files=files, data=data)

                if response.status_code == 200:
                    result = response.json()

                    # Extract transcript from response
                    transcript = result.get("text", "").strip()

                    if transcript:
                        logger.info(
                            f"Mistral transcription successful: {len(transcript)} characters"
                        )
                        return transcript
                    else:
                        logger.warning("Mistral returned empty transcript")
                        return ""
                else:
                    logger.error(f"Mistral API error: {response.status_code} - {response.text}")
                    return ""

        except httpx.TimeoutException as e:
            logger.error(f"HTTP timeout during Mistral API call for {len(audio_data)} bytes: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error calling Mistral API: {e}")
            return ""

    def _create_wav_header(self, audio_data: bytes, sample_rate: int) -> bytes:
        """Create a WAV header for raw PCM data."""
        import struct

        # WAV header parameters
        channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = len(audio_data)
        file_size = data_size + 36  # 36 = header size - 8

        # Build WAV header
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",  # ChunkID
            file_size,  # ChunkSize
            b"WAVE",  # Format
            b"fmt ",  # Subchunk1ID
            16,  # Subchunk1Size (16 for PCM)
            1,  # AudioFormat (1 for PCM)
            channels,  # NumChannels
            sample_rate,  # SampleRate
            byte_rate,  # ByteRate
            block_align,  # BlockAlign
            bits_per_sample,  # BitsPerSample
            b"data",  # Subchunk2ID
            data_size,  # Subchunk2Size
        )

        return header


def get_transcription_provider(
    provider_name: Optional[str] = None,
) -> Optional[OnlineTranscriptionProvider]:
    """
    Factory function to get the appropriate transcription provider.

    Args:
        provider_name: Name of the provider ('deepgram', 'mistral').
                      If None, will check environment for available keys.

    Returns:
        An instance of OnlineTranscriptionProvider or None if no provider is available.
    """
    import os

    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    mistral_key = os.getenv("MISTRAL_API_KEY")
    mistral_model = os.getenv("MISTRAL_MODEL", "voxtral-mini-2507")  # Default to voxtral-mini

    if provider_name:
        provider_name = provider_name.lower()

    if provider_name == "deepgram" and deepgram_key:
        return DeepgramProvider(deepgram_key)
    elif provider_name == "mistral" and mistral_key:
        logger.info(f"Using Mistral transcription provider with model: {mistral_model}")
        return MistralProvider(mistral_key, mistral_model)
    elif provider_name is None:
        # Auto-select based on available keys (Deepgram preferred)
        if deepgram_key:
            logger.info("Using Deepgram transcription provider")
            return DeepgramProvider(deepgram_key)
        elif mistral_key:
            logger.info(f"Using Mistral transcription provider with model: {mistral_model}")
            return MistralProvider(mistral_key, mistral_model)

    return None
