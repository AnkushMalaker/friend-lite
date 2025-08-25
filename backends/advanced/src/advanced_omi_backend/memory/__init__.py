"""Memory service package.

This package provides memory management functionality with support for
multiple LLM providers and vector stores for the Omi backend.

The memory service handles extraction, storage, and retrieval of memories
from user conversations and interactions.

Architecture:
- base.py: Abstract base classes and interfaces
- memory_service.py: Core implementation  
- compat_service.py: Backward compatibility wrapper
- providers/: LLM and vector store implementations
- config.py: Configuration management
"""

import logging

memory_logger = logging.getLogger("memory_service")

# Initialize core functions to None
get_memory_service = None
MemoryService = None
shutdown_memory_service = None
init_memory_config = None
test_new_memory_service = None
migrate_from_mem0 = None

memory_logger.info("🆕 Using NEW memory service implementation")
try:
    from .compat_service import (
        get_memory_service, 
        MemoryService, 
        shutdown_memory_service,
        init_memory_config,
        migrate_from_mem0
    )
    # Also import core implementation for direct access
    from .memory_service import MemoryService as CoreMemoryService
    test_new_memory_service = None  # Will be implemented if needed
    memory_logger.info("✅ Successfully imported new memory service")
except ImportError as e:
    memory_logger.error(f"Failed to import new memory service: {e}")
    raise

# Also export the new architecture components for direct access when needed
try:
    from .base import MemoryServiceBase, LLMProviderBase, VectorStoreBase, MemoryEntry
    from .config import (
        MemoryConfig,
        LLMProvider,
        VectorStoreProvider,
        MemoryProvider,  # New memory provider enum
        build_memory_config_from_env,
        create_openai_config,
        create_ollama_config,
        create_qdrant_config,
        create_openmemory_config  # New OpenMemory config function
    )
    from .providers import (
        OpenAIProvider,
        QdrantVectorStore,
        OpenMemoryMCPService,  # New complete memory service
        MCPClient,
        MCPError
    )
    from .service_factory import (
        get_memory_service as get_core_memory_service,
        create_memory_service,
        shutdown_memory_service as shutdown_core_memory_service,
        reset_memory_service,
        get_service_info as get_core_service_info
    )
    # Keep backward compatibility alias
    AbstractMemoryService = CoreMemoryService
except ImportError as e:
    memory_logger.warning(f"Some advanced memory service components not available: {e}")
    MemoryServiceBase = None
    LLMProviderBase = None
    VectorStoreBase = None
    AbstractMemoryService = None
    MemoryConfig = None
    LLMProvider = None
    VectorStoreProvider = None
    MemoryProvider = None
    build_memory_config_from_env = None
    create_openai_config = None
    create_ollama_config = None
    create_qdrant_config = None
    create_openmemory_config = None
    MemoryEntry = None
    OpenAIProvider = None
    QdrantVectorStore = None
    OpenMemoryMCPService = None
    MCPClient = None
    MCPError = None
    get_core_memory_service = None
    create_memory_service = None
    shutdown_core_memory_service = None
    reset_memory_service = None
    get_core_service_info = None

__all__ = [
    # Main interface (compatible with legacy)
    "get_memory_service",
    "MemoryService", 
    "shutdown_memory_service",
    "init_memory_config",
    
    # New service specific (may be None if not available)
    "test_new_memory_service",
    "migrate_from_mem0",
    "CoreMemoryService",
    
    # Base classes (new architecture)
    "MemoryServiceBase",
    "LLMProviderBase", 
    "VectorStoreBase",
    
    # Advanced components (may be None if not available)
    "AbstractMemoryService",  # Backward compatibility alias
    "MemoryConfig",
    "MemoryEntry",
    "LLMProvider",
    "VectorStoreProvider",
    "MemoryProvider",  # New enum
    "build_memory_config_from_env",
    "create_openai_config",
    "create_ollama_config", 
    "create_qdrant_config",
    "create_openmemory_config",  # New function
    "OpenAIProvider",
    "QdrantVectorStore",
    
    # Complete memory service implementations
    "OpenMemoryMCPService",
    
    # MCP client components
    "MCPClient",
    "MCPError",
    
    # Service factory functions
    "get_core_memory_service",
    "create_memory_service",
    "shutdown_core_memory_service",
    "reset_memory_service",
    "get_core_service_info"
]

def get_service_info():
    """Get information about which service is currently active."""
    return {
        "active_service": "new",  # Always use new service
        "new_service_available": CoreMemoryService is not None,
        "legacy_service_available": True,  # Assume always available
        "base_classes_available": MemoryServiceBase is not None,
        "core_service_available": CoreMemoryService is not None
    }