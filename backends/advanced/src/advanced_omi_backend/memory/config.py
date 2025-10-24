"""Memory service configuration utilities."""

import os
import logging
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

memory_logger = logging.getLogger("memory_service")


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class VectorStoreProvider(Enum):
    """Supported vector store providers."""
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    CUSTOM = "custom"


class MemoryProvider(Enum):
    """Supported memory service providers."""
    FRIEND_LITE = "friend_lite"      # Default sophisticated implementation
    OPENMEMORY_MCP = "openmemory_mcp"  # OpenMemory MCP backend


@dataclass
class MemoryConfig:
    """Configuration for memory service."""
    memory_provider: MemoryProvider = MemoryProvider.FRIEND_LITE
    llm_provider: LLMProvider = LLMProvider.OPENAI
    vector_store_provider: VectorStoreProvider = VectorStoreProvider.QDRANT
    llm_config: Dict[str, Any] = None
    vector_store_config: Dict[str, Any] = None
    embedder_config: Dict[str, Any] = None
    openmemory_config: Dict[str, Any] = None  # Configuration for OpenMemory MCP
    extraction_prompt: str = None
    extraction_enabled: bool = True
    timeout_seconds: int = 1200


def create_openai_config(
    api_key: str,
    model: str = "gpt-4",
    embedding_model: str = "text-embedding-3-small",
    base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.1,
    max_tokens: int = 2000
) -> Dict[str, Any]:
    """Create OpenAI configuration."""
    return {
        "api_key": api_key,
        "model": model,
        "embedding_model": embedding_model,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens
    }


def create_ollama_config(
    base_url: str,
    model: str = "llama2",
    embedding_model: str = "nomic-embed-text",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    use_qwen_embeddings: bool = True
) -> Dict[str, Any]:
    """Create Ollama configuration."""
    return {
        "api_key": "dummy",  # Ollama doesn't require an API key
        "base_url": base_url,
        "model": model,
        "embedding_model": embedding_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "use_qwen_embeddings": use_qwen_embeddings
    }


def create_qdrant_config(
    host: str = "localhost",
    port: int = 6333,
    collection_name: str = "memories",
    embedding_dims: int = 1536
) -> Dict[str, Any]:
    """Create Qdrant configuration."""
    return {
        "host": host,
        "port": port,
        "collection_name": collection_name,
        "embedding_dims": embedding_dims
    }


def create_openmemory_config(
    server_url: str = "http://localhost:8765",
    client_name: str = "friend_lite",
    user_id: str = "default",
    timeout: int = 30
) -> Dict[str, Any]:
    """Create OpenMemory MCP configuration."""
    return {
        "server_url": server_url,
        "client_name": client_name,
        "user_id": user_id,
        "timeout": timeout
    }


def build_memory_config_from_env() -> MemoryConfig:
    """Build memory configuration from environment variables and YAML config."""
    try:
        # Determine memory provider
        memory_provider = os.getenv("MEMORY_PROVIDER", "friend_lite").lower()
        if memory_provider not in [p.value for p in MemoryProvider]:
            raise ValueError(f"Unsupported memory provider: {memory_provider}")
        
        memory_provider_enum = MemoryProvider(memory_provider)
        
        # For OpenMemory MCP, configuration is much simpler
        if memory_provider_enum == MemoryProvider.OPENMEMORY_MCP:
            openmemory_config = create_openmemory_config(
                server_url=os.getenv("OPENMEMORY_MCP_URL", "http://localhost:8765"),
                client_name=os.getenv("OPENMEMORY_CLIENT_NAME", "friend_lite"),
                user_id=os.getenv("OPENMEMORY_USER_ID", "default"),
                timeout=int(os.getenv("OPENMEMORY_TIMEOUT", "30"))
            )
            
            memory_logger.info(f"🔧 Memory config: Provider=OpenMemory MCP, URL={openmemory_config['server_url']}")
            
            return MemoryConfig(
                memory_provider=memory_provider_enum,
                openmemory_config=openmemory_config,
                timeout_seconds=int(os.getenv("OPENMEMORY_TIMEOUT", "30"))
            )
        
        # For Friend-Lite provider, use existing complex configuration
        # Import config loader
        from advanced_omi_backend.memory_config_loader import get_config_loader
        
        config_loader = get_config_loader()
        memory_config = config_loader.get_memory_extraction_config()
        
        # Get LLM provider from environment
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
        memory_logger.info(f"LLM_PROVIDER: {llm_provider}")
        if llm_provider not in [p.value for p in LLMProvider]:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        llm_config = None
        llm_provider_enum = None
        embedding_dims = 1536 # Default

        # Build LLM configuration
        if llm_provider == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY required for OpenAI provider")
            
            # Use environment variables for model, fall back to config, then defaults
            model = os.getenv("OPENAI_MODEL") or memory_config.get("llm_settings", {}).get("model") or "gpt-4o-mini"
            embedding_model = memory_config.get("llm_settings", {}).get("embedding_model") or "text-embedding-3-small"
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            memory_logger.info(f"🔧 Memory config: LLM={model}, Embedding={embedding_model}, Base URL={base_url}")
            
            llm_config = create_openai_config(
                api_key=openai_api_key,
                model=model,
                embedding_model=embedding_model,
                base_url=base_url,
                temperature=memory_config.get("llm_settings", {}).get("temperature", 0.1),
                max_tokens=memory_config.get("llm_settings", {}).get("max_tokens", 2000)
            )
            llm_provider_enum = LLMProvider.OPENAI
            embedding_dims = get_embedding_dims(llm_config)
            memory_logger.info(f"🔧 Setting Embedder dims {embedding_dims}")
        
        elif llm_provider == "ollama":
            base_url = os.getenv("OLLAMA_BASE_URL")
            if not base_url:
                raise ValueError("OLLAMA_BASE_URL required for Ollama provider")
            
            model = os.getenv("OLLAMA_MODEL")
            if not model:
                raise ValueError("OLLAMA_MODEL required for Ollama provider")
            embedding_model = os.getenv("OLLAMA_EMBEDDER_MODEL")
            if not embedding_model:
                raise ValueError("OLLAMA_EMBEDDER_MODEL required for Ollama provider")
            memory_logger.info(f"🔧 Memory config: LLM={model}, Embedding={embedding_model}, Base URL={base_url}")

            llm_config = create_ollama_config(
                base_url=base_url,
                model=model,
                embedding_model=embedding_model,
            )
            llm_provider_enum = LLMProvider.OLLAMA
            embedding_dims = get_embedding_dims(llm_config)
            memory_logger.info(f"🔧 Setting Embedder dims {embedding_dims}")

        # Build vector store configuration
        vector_store_provider = os.getenv("VECTOR_STORE_PROVIDER", "qdrant").lower()
        
        if vector_store_provider == "qdrant":
            qdrant_host = os.getenv("QDRANT_BASE_URL", "qdrant")
            vector_store_config = create_qdrant_config(
                host=qdrant_host,
                port=int(os.getenv("QDRANT_PORT", "6333")),
                collection_name="omi_memories",
                embedding_dims=embedding_dims
            )
            vector_store_provider_enum = VectorStoreProvider.QDRANT
            
        else:
            raise ValueError(f"Unsupported vector store provider: {vector_store_provider}")
        
        # Get memory extraction settings
        extraction_enabled = config_loader.is_memory_extraction_enabled()
        extraction_prompt = config_loader.get_memory_prompt() if extraction_enabled else None
        
        memory_logger.info(f"🔧 Memory config: Provider=Friend-Lite, LLM={llm_provider}, VectorStore={vector_store_provider}, Extraction={extraction_enabled}")
        
        return MemoryConfig(
            memory_provider=memory_provider_enum,
            llm_provider=llm_provider_enum,
            vector_store_provider=vector_store_provider_enum,
            llm_config=llm_config,
            vector_store_config=vector_store_config,
            embedder_config={},  # Included in llm_config
            extraction_prompt=extraction_prompt,
            extraction_enabled=extraction_enabled,
            timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "1200"))
        )
        
    except ImportError:
        memory_logger.warning("Config loader not available, using environment variables only")
        raise


def get_embedding_dims(llm_config: Dict[str, Any]) -> int:
    """
    Query the embedding endpoint and return the embedding vector length.
    Works for OpenAI and OpenAI-compatible endpoints (e.g., Ollama).
    """
    embedding_model = llm_config.get('embedding_model')
    try:
        import langfuse.openai as openai
        client = openai.OpenAI(
            api_key=llm_config.get('api_key'), 
            base_url=llm_config.get('base_url')
        )
        response = client.embeddings.create(
            model=embedding_model,
            input="hello world"
        )
        embedding = response.data[0].embedding
        if not embedding or not isinstance(embedding, list):
            return 1536
        return len(embedding)

    except (ImportError, KeyError, AttributeError, IndexError, TypeError, ValueError) as e:
        embedding_dims = 1536 # default
        memory_logger.exception(f"Failed to get embedding dimensions for model '{embedding_model}'")
        if embedding_model == "text-embedding-3-small":
            embedding_dims = 1536
        elif embedding_model == "text-embedding-3-large":
            embedding_dims = 3072
        elif embedding_model == "text-embedding-ada-002":
            embedding_dims = 1536
        elif embedding_model == "nomic-embed-text:latest":
            embedding_dims = 768
        else:
            # Default for OpenAI embedding models
            memory_logger.info(f"Unrecognized embedding model '{embedding_model}', using default dimension {embedding_dims}")
        return embedding_dims