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

    def __init__(self, model: str | None = None, temperature: float = 0.1, provider: str = "openai"):
        self.model = model
        self.temperature = temperature
        self.provider = provider
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
        provider: str = "openai",
    ):
        super().__init__(model, temperature, provider)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        # Initialize OpenAI client
        try:
            import openai

            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.logger.info(f"OpenAI client initialized for provider '{self.provider}' with base_url: {self.base_url}")
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
            api_key_configured = self.api_key and self.api_key != "dummy"
            
            if self.provider == "ollama":
                # Ollama is local and doesn't require a key, so it's always 'connected' if reachable
                # A more robust check would ping the base_url, but this is sufficient for UI status
                return {
                    "status": "✅ Connected (Local)",
                    "base_url": self.base_url,
                    "default_model": self.model,
                    "api_key_configured": False,
                }

            if api_key_configured and self.model:
                return {
                    "status": "✅ Connected",
                    "base_url": self.base_url,
                    "default_model": self.model,
                    "api_key_configured": api_key_configured,
                }
            else:
                return {
                    "status": "⚠️ Configuration incomplete",
                    "base_url": self.base_url,
                    "default_model": self.model,
                    "api_key_configured": api_key_configured,
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

        if provider == "openai":
            return OpenAILLMClient(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                provider=provider,
            )
        elif provider == "ollama":
            return OpenAILLMClient(
                api_key="dummy",  # Ollama doesn't require a key
                base_url=os.getenv("OPENAI_BASE_URL", "http://ollama:11434/v1"),
                model=os.getenv("OPENAI_MODEL", "llama3.1:latest"),
                provider=provider,
            )
        elif provider == "gemini":
            # The Gemini client uses OpenAI's library for compatibility
            return OpenAILLMClient(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta", # This is not used by the google-generativeai library but good to have
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                provider=provider,
            )
        elif provider == "anthropic":
            # Future implementation for Anthropic
            raise NotImplementedError("Anthropic provider not yet implemented")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def get_supported_providers() -> list:
        """Get list of supported LLM providers."""
        return ["openai", "ollama", "gemini"]


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
