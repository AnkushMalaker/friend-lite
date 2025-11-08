"""
Parakeet stream consumer for Redis Streams architecture.

Reads from: audio:stream:* streams
Writes to: transcription:results:{session_id}
"""

import logging
import os

from advanced_omi_backend.services.audio_stream.consumer import BaseAudioStreamConsumer
from advanced_omi_backend.services.transcription.parakeet import ParakeetProvider

logger = logging.getLogger(__name__)


class ParakeetStreamConsumer:
    """
    Parakeet consumer for Redis Streams architecture.

    Reads from: specified stream (client-specific or provider-specific)
    Writes to: transcription:results:{session_id}

    This inherits from BaseAudioStreamConsumer and implements transcribe_audio().
    """

    def __init__(self, redis_client, service_url: str = None, buffer_chunks: int = 30):
        """
        Initialize Parakeet consumer.

        Dynamically discovers all audio:stream:* streams and claims them using Redis locks.

        Args:
            redis_client: Connected Redis client
            service_url: Parakeet service URL (defaults to PARAKEET_ASR_URL env var)
            buffer_chunks: Number of chunks to buffer before transcribing (default: 30 = ~7.5s)
        """
        self.service_url = service_url or os.getenv("PARAKEET_ASR_URL") or os.getenv("OFFLINE_ASR_TCP_URI")
        if not self.service_url:
            raise ValueError("PARAKEET_ASR_URL or OFFLINE_ASR_TCP_URI is required")

        # Initialize Parakeet provider
        self.provider = ParakeetProvider(service_url=self.service_url)

        # Create a concrete subclass that implements transcribe_audio
        class _ConcreteConsumer(BaseAudioStreamConsumer):
            def __init__(inner_self, provider_name: str, redis_client, buffer_chunks: int):
                super().__init__(provider_name, redis_client, buffer_chunks)
                inner_self._parakeet_provider = self.provider

            async def transcribe_audio(inner_self, audio_data: bytes, sample_rate: int) -> dict:
                """Transcribe using ParakeetProvider."""
                try:
                    result = await inner_self._parakeet_provider.transcribe(
                        audio_data=audio_data,
                        sample_rate=sample_rate
                    )

                    # Calculate confidence (Parakeet may not provide confidence, default to 0.9)
                    confidence = 0.9
                    if result.get("words"):
                        confidences = [
                            w.get("confidence", 0.9)
                            for w in result["words"]
                            if "confidence" in w
                        ]
                        if confidences:
                            confidence = sum(confidences) / len(confidences)

                    return {
                        "text": result.get("text", ""),
                        "words": result.get("words", []),
                        "segments": result.get("segments", []),
                        "confidence": confidence
                    }

                except Exception as e:
                    logger.error(f"Parakeet transcription failed: {e}", exc_info=True)
                    raise

        # Instantiate the concrete consumer
        self._consumer = _ConcreteConsumer("parakeet", redis_client, buffer_chunks)

    async def start_consuming(self):
        """Delegate to base consumer."""
        return await self._consumer.start_consuming()

    async def stop(self):
        """Delegate to base consumer."""
        return await self._consumer.stop()

