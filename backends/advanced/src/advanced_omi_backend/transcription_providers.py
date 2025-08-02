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
                write=max(180.0, int(len(audio_data) / (sample_rate * 2))),  # bytes per second for 16-bit PCM
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

                                # Convert word-level speaker data to speaker segments
                                formatted_transcript = self._format_speaker_segments(words)
                                logger.info(
                                    f"Formatted transcript with speaker segments: {len(formatted_transcript)} characters"
                                )
                                return {
                                    "text": formatted_transcript,
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

    def _format_speaker_segments(self, words: list) -> str:
        """
        Convert word-level speaker data into formatted transcript with speaker labels.

        Args:
            words: List of word objects with speaker information

        Returns:
            Formatted transcript string with speaker labels
        """
        if not words:
            return ""

        segments = []
        current_speaker = None
        current_text = []

        for word in words:
            speaker = word.get("speaker", 0)
            word_text = word.get("punctuated_word", word.get("word", ""))

            if speaker != current_speaker:
                # Save previous segment if it exists
                if current_speaker is not None and current_text:
                    segment_text = " ".join(current_text).strip()
                    if segment_text:
                        segments.append(f"Speaker {current_speaker}: {segment_text}")

                # Start new segment
                current_speaker = speaker
                current_text = [word_text]
            else:
                # Continue current segment
                current_text.append(word_text)

        # Add final segment
        if current_speaker is not None and current_text:
            segment_text = " ".join(current_text).strip()
            if segment_text:
                segments.append(f"Speaker {current_speaker}: {segment_text}")

        return "\n".join(segments)


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
                write=max(180.0, int(len(wav_data) / (sample_rate * 2))),  # bytes per second for 16-bit PCM
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
                        return {"text": transcript, "words": [], "segments": []}
                    else:
                        logger.warning("Mistral returned empty transcript")
                        return {"text": "", "words": [], "segments": []}
                else:
                    logger.error(f"Mistral API error: {response.status_code} - {response.text}")
                    return {"text": "", "words": [], "segments": []}

        except httpx.TimeoutException as e:
            logger.error(f"HTTP timeout during Mistral API call for {len(audio_data)} bytes: {e}")
            return {"text": "", "words": [], "segments": []}
        except Exception as e:
            logger.error(f"Error calling Mistral API: {e}")
            return {"text": "", "words": [], "segments": []}

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
) -> OnlineTranscriptionProvider:
    """
    Factory function to get the appropriate transcription provider.

    Args:
        provider_name: Name of the provider ('deepgram', 'mistral').
                      If None, will check environment for available keys.

    Returns:
        An instance of OnlineTranscriptionProvider.

    Raises:
        RuntimeError: If no transcription provider is configured or requested provider is unavailable.
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

    # No provider available or configured
    if provider_name:
        if provider_name == "deepgram":
            raise RuntimeError(
                f"Deepgram transcription provider requested but DEEPGRAM_API_KEY not configured"
            )
        elif provider_name == "mistral":
            raise RuntimeError(
                f"Mistral transcription provider requested but MISTRAL_API_KEY not configured"
            )
        else:
            raise RuntimeError(
                f"Unknown transcription provider '{provider_name}'. Supported: 'deepgram', 'mistral'"
            )
    else:
        raise RuntimeError(
            "No transcription provider configured. Please set DEEPGRAM_API_KEY or MISTRAL_API_KEY environment variable"
        )
