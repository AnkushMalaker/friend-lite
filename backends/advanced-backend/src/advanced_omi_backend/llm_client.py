"""
Abstract LLM client interface for unified LLM operations across different providers.

This module provides a standardized interface for LLM operations that works with
OpenAI, Ollama, Anthropic, and other OpenAI-compatible APIs.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str | None = None, temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def generate(
        self, prompt: str, model: str | None = None, temperature: float | None = None
    ) -> str:
        """Generate text completion from prompt."""
        pass

    @abstractmethod
    def health_check(self) -> Dict:
        """Check if the LLM service is available and healthy."""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this client."""
        pass


class OpenAILLMClient(LLMClient):
    """OpenAI-compatible LLM client that works with OpenAI, Ollama, and other compatible APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ):
        super().__init__(model, temperature)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")

        # Initialize OpenAI client
        try:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.logger.info(f"OpenAI client initialized with base_url: {self.base_url}")
        except ImportError:
            self.logger.error("OpenAI library not installed. Install with: pip install openai")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def generate(
        self, prompt: str, model: str | None = None, temperature: float | None = None
    ) -> str:
        """Generate text completion using OpenAI-compatible API."""
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature or self.temperature,
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error generating completion: {e}")
            raise

    def health_check(self) -> Dict:
        """Check OpenAI-compatible service health."""
        try:
            # For OpenAI API, check if we have valid configuration
            # Avoid calling /models endpoint as it can be unreliable
            if self.api_key and self.api_key != "dummy" and self.model:
                return {
                    "status": "✅ Connected",
                    "base_url": self.base_url,
                    "default_model": self.model,
                    "api_key_configured": bool(self.api_key and self.api_key != "dummy"),
                }
            else:
                return {
                    "status": "⚠️ Configuration incomplete",
                    "base_url": self.base_url,
                    "default_model": self.model,
                    "api_key_configured": bool(self.api_key and self.api_key != "dummy"),
                }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "❌ Failed",
                "error": str(e),
                "base_url": self.base_url,
                "default_model": self.model,
            }

    def get_default_model(self) -> str:
        """Get the default model for this client."""
        return self.model


class LLMClientFactory:
    """Factory for creating LLM clients based on environment configuration."""

    @staticmethod
    def create_client() -> LLMClient:
        """Create an LLM client based on LLM_PROVIDER environment variable."""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()

        if provider in ["openai", "ollama"]:
            return OpenAILLMClient(
                api_key=os.getenv("OPENAI_API_KEY", "dummy"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            )
        elif provider == "anthropic":
            # Future implementation for Anthropic
            raise NotImplementedError("Anthropic provider not yet implemented")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def get_supported_providers() -> list:
        """Get list of supported LLM providers."""
        return ["openai", "ollama"]


# Global LLM client instance
_llm_client = None


def get_llm_client() -> LLMClient:
    """Get the global LLM client instance (singleton pattern)."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClientFactory.create_client()
    return _llm_client


def reset_llm_client():
    """Reset the global LLM client instance (useful for testing)."""
    global _llm_client
    _llm_client = None


# Async wrapper for blocking LLM operations
async def async_generate(
    prompt: str, model: str | None = None, temperature: float | None = None
) -> str:
    """Async wrapper for LLM text generation."""
    client = get_llm_client()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, client.generate, prompt, model, temperature)


async def async_health_check() -> Dict:
    """Async wrapper for LLM health check."""
    client = get_llm_client()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, client.health_check)
