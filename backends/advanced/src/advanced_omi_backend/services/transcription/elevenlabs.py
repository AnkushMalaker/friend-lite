"""
ElevenLabs transcription provider implementation.

Provides batch transcription using ElevenLabs Scribe v1 model.
"""

import io
import logging
import wave

import httpx

from advanced_omi_backend.models.transcription import BatchTranscriptionProvider

logger = logging.getLogger(__name__)


class ElevenLabsProvider(BatchTranscriptionProvider):
    """ElevenLabs batch transcription provider using Scribe v1 model."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.elevenlabs.io/v1/speech-to-text"

    @property
    def name(self) -> str:
        return "elevenlabs"

    async def transcribe(self, audio_data: bytes, sample_rate: int, diarize: bool = False) -> dict:
        """Transcribe audio using ElevenLabs REST API.

        Args:
            audio_data: Raw audio bytes (will be converted to WAV format)
            sample_rate: Audio sample rate
            diarize: Whether to enable speaker diarization
        """
        try:
            # Convert raw PCM to WAV format for ElevenLabs
            wav_data = self._pcm_to_wav(audio_data, sample_rate)

            # Prepare multipart form data
            files = {
                'file': ('audio.wav', io.BytesIO(wav_data), 'audio/wav')
            }

            data = {
                'model_id': 'scribe_v1',
                'diarize': 'true' if diarize else 'false',
                'timestamps_granularity': 'word',
                'tag_audio_events': 'false',  # Optional: set to true for laughter/applause detection
            }

            headers = {
                'xi-api-key': self.api_key
            }

            logger.info(f"Sending {len(audio_data)} bytes to ElevenLabs API (diarize={diarize})")

            # Calculate timeout based on audio duration
            estimated_duration = len(audio_data) / (sample_rate * 2)  # 16-bit mono
            processing_timeout = max(120, int(estimated_duration * 5))  # 5x audio duration

            timeout_config = httpx.Timeout(
                connect=30.0,
                read=processing_timeout,
                write=180.0,
                pool=10.0,
            )

            logger.info(
                f"Estimated audio duration: {estimated_duration:.1f}s, timeout: {processing_timeout}s"
            )

            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    self.url,
                    headers=headers,
                    data=data,
                    files=files
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"ElevenLabs response: {result}")

                    # Parse ElevenLabs response format
                    transcript = result.get('text', '').strip()

                    # Extract word-level data
                    words = []
                    segments = []

                    if 'words' in result:
                        # Map ElevenLabs format to Friend-Lite format
                        for word_obj in result['words']:
                            if word_obj.get('type') == 'word':  # Skip spacing/audio_events
                                words.append({
                                    'word': word_obj.get('text', ''),
                                    'start': word_obj.get('start', 0),
                                    'end': word_obj.get('end', 0),
                                    'confidence': 1.0 - abs(word_obj.get('logprob', 0)),  # Convert logprob to confidence
                                    'speaker': word_obj.get('speaker_id'),
                                })

                    # Extract speaker segments if diarization is enabled
                    if diarize and words:
                        segments = self._create_speaker_segments(words)

                    logger.info(
                        f"ElevenLabs transcription successful: {len(transcript)} chars, "
                        f"{len(words)} words, {len(segments)} segments"
                    )

                    return {
                        "text": transcript,
                        "words": words,
                        "segments": segments,
                    }
                else:
                    logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                    return {"text": "", "words": [], "segments": []}

        except httpx.TimeoutException as e:
            logger.error(f"Timeout during ElevenLabs API call: {e}")
            return {"text": "", "words": [], "segments": []}
        except Exception as e:
            logger.error(f"Error calling ElevenLabs API: {e}")
            return {"text": "", "words": [], "segments": []}

    def _pcm_to_wav(self, pcm_data: bytes, sample_rate: int) -> bytes:
        """Convert raw PCM data to WAV format."""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)

        return wav_buffer.getvalue()

    def _create_speaker_segments(self, words: list) -> list:
        """Group consecutive words by speaker into segments."""
        segments = []
        current_speaker = None
        current_segment = None

        for word in words:
            speaker = word.get('speaker')
            if speaker is None:
                continue

            if speaker == current_speaker and current_segment:
                # Extend current segment
                current_segment['text'] += ' ' + word['word']
                current_segment['end'] = word['end']
            else:
                # Save previous segment and start new one
                if current_segment:
                    segments.append(current_segment)
                current_segment = {
                    'text': word['word'],
                    'speaker': f"Speaker {speaker}",
                    'start': word['start'],
                    'end': word['end'],
                    'confidence': word.get('confidence'),
                }
                current_speaker = speaker

        # Don't forget the last segment
        if current_segment:
            segments.append(current_segment)

        return segments
