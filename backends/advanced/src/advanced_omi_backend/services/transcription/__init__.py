"""
Transcription provider implementations and factory.

This module contains concrete implementations of transcription providers
for different ASR services (Deepgram, Parakeet, etc.) and a factory function
to instantiate the appropriate provider based on configuration.
"""

import logging
import os
from typing import Optional

from advanced_omi_backend.models.transcription import BaseTranscriptionProvider
from advanced_omi_backend.services.transcription.deepgram import (
    DeepgramProvider,
    DeepgramStreamingProvider,
    DeepgramStreamConsumer,
)
from advanced_omi_backend.services.transcription.parakeet import (
    ParakeetProvider,
    ParakeetStreamingProvider,
)

logger = logging.getLogger(__name__)


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
        if mode == "streaming":
            return ParakeetStreamingProvider(parakeet_url)
        else:
            return ParakeetProvider(parakeet_url)

    elif provider_name == "offline":
        # "offline" is an alias for Parakeet ASR
        if not parakeet_url:
            raise RuntimeError(
                "Offline transcription provider requested but PARAKEET_ASR_URL not configured"
            )
        logger.info(f"Using offline Parakeet transcription provider in {mode} mode")
        if mode == "streaming":
            return ParakeetStreamingProvider(parakeet_url)
        else:
            return ParakeetProvider(parakeet_url)

    # Auto-select provider based on available configuration (when provider_name is None)
    if provider_name is None:
        # Check TRANSCRIPTION_PROVIDER environment variable first
        env_provider = os.getenv("TRANSCRIPTION_PROVIDER")
        if env_provider:
            # Recursively call with the specified provider
            return get_transcription_provider(env_provider, mode)

        # Auto-select: prefer Deepgram if available, fallback to Parakeet
        if deepgram_key:
            logger.info(f"Auto-selected Deepgram transcription provider in {mode} mode")
            if mode == "streaming":
                return DeepgramStreamingProvider(deepgram_key)
            else:
                return DeepgramProvider(deepgram_key)
        elif parakeet_url:
            logger.info(f"Auto-selected Parakeet transcription provider in {mode} mode")
            if mode == "streaming":
                return ParakeetStreamingProvider(parakeet_url)
            else:
                return ParakeetProvider(parakeet_url)
        else:
            logger.warning(
                "No transcription provider configured (DEEPGRAM_API_KEY or PARAKEET_ASR_URL required)"
            )
            return None
    else:
        return None


__all__ = [
    "get_transcription_provider",
    "DeepgramProvider",
    "DeepgramStreamingProvider",
    "DeepgramStreamConsumer",
    "ParakeetProvider",
    "ParakeetStreamingProvider",
]
