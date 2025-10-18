"""
Abstract LLM client interface for unified LLM operations across different providers.

This module provides a standardized interface for LLM operations that works with
OpenAI, Ollama, and other OpenAI-compatible APIs.
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
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
    ):
        self.provider = provider
        super().__init__(model, temperature)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL")
        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL must be set")

        # Initialize OpenAI client
        try:
            import langfuse.openai as openai

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
            model_name = model or self.model
            temp = temperature or self.temperature
            
            # Build completion parameters
            params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            
            # Skip temperature for gpt-4o-mini as it only supports default (1)
            if not (model_name and "gpt-4o-mini" in model_name):
                params["temperature"] = temp
            
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error generating completion: {e}")
            raise

    async def health_check(self) -> Dict:
        """Check OpenAI-compatible service health."""
        try:
            if not (self.model and self.base_url):
                return {
                    "status": "⚠️ Configuration incomplete (missing model or base_url)",
                    "base_url": self.base_url,
                    "default_model": self.model,
                    "api_key_configured": bool(self.api_key and self.api_key != "dummy"),
                }

            if self.provider == "ollama":
                import aiohttp
                ollama_health_url = self.base_url.replace("/v1", "") if self.base_url.endswith("/v1") else self.base_url
                
                # Initialize response with main LLM status
                response_data = {
                    "status": "❌ Unknown",
                    "base_url": self.base_url,
                    "default_model": self.model,
                    "api_key_configured": False,
                    "embedder_model": os.getenv("OLLAMA_EMBEDDER_MODEL"),
                    "embedder_status": "❌ Not Checked"
                }

                try:
                    async with aiohttp.ClientSession() as session:
                        # Check main Ollama server health
                        async with session.get(f"{ollama_health_url}/api/version", timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                response_data["status"] = "✅ Connected"
                            else:
                                response_data["status"] = f"⚠️ Ollama Unhealthy: HTTP {response.status}"
                        
                        # Check embedder model availability
                        embedder_model_name = os.getenv("OLLAMA_EMBEDDER_MODEL")
                        if embedder_model_name:
                            try:
                                # Use /api/show to check if model exists
                                async with session.post(f"{ollama_health_url}/api/show", json={"name": embedder_model_name}, timeout=aiohttp.ClientTimeout(total=5)) as embedder_response:
                                    if embedder_response.status == 200:
                                        response_data["embedder_status"] = "✅ Available"
                                    else:
                                        response_data["embedder_status"] = "⚠️ Embedder Model Unhealthy"
                            except aiohttp.ClientError:
                                response_data["embedder_status"] = "❌ Embedder Model Connection Failed"
                            except asyncio.TimeoutError:
                                response_data["embedder_status"] = "❌ Embedder Model Timeout"
                        else:
                            response_data["embedder_status"] = "⚠️ Embedder Model Not Configured"

                except aiohttp.ClientError:
                    response_data["status"] = "❌ Ollama Connection Failed"
                except asyncio.TimeoutError:
                    response_data["status"] = "❌ Ollama Connection Timeout (5s)"
                
                return response_data
            else:
                # For other OpenAI-compatible APIs, check configuration
                if self.api_key and self.api_key != "dummy":
                    return {
                        "status": "✅ Connected",
                        "base_url": self.base_url,
                        "default_model": self.model,
                        "api_key_configured": bool(self.api_key and self.api_key != "dummy"),
                    }
                else:
                    return {
                        "status": "⚠️ Configuration incomplete (missing API key)",
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
        return self.model or "gpt-4o-mini"


class LLMClientFactory:
    """Factory for creating LLM clients based on environment configuration."""

    @staticmethod
    def create_client() -> LLMClient:
        """Create an LLM client based on LLM_PROVIDER environment variable."""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()

        if provider == "openai":
            return OpenAILLMClient(
                provider="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                model=os.getenv("OPENAI_MODEL"),
            )
        elif provider == "ollama":
            return OpenAILLMClient(
                provider="ollama",
                api_key="dummy",  # Ollama doesn't require an API key
                base_url=os.getenv("OLLAMA_BASE_URL"),
                model=os.getenv("OLLAMA_MODEL"),
            )
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
    return await client.health_check()
