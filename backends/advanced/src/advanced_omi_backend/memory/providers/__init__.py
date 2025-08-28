"""Memory service providers package.

This package contains implementations of LLM providers, vector stores,
and complete memory service implementations for the memory service architecture.
"""

from ..base import LLMProviderBase, VectorStoreBase, MemoryEntry
from .llm_providers import OpenAIProvider
from .vector_stores import QdrantVectorStore

# Import complete memory service implementations
try:
    from .openmemory_mcp_service import OpenMemoryMCPService
except ImportError:
    OpenMemoryMCPService = None

try:
    from .mcp_client import MCPClient, MCPError
except ImportError:
    MCPClient = None
    MCPError = None

__all__ = [
    # Base classes
    "LLMProviderBase",
    "VectorStoreBase", 
    "MemoryEntry",
    
    # LLM providers
    "OpenAIProvider",
    
    # Vector stores
    "QdrantVectorStore",
    
    # Complete memory service implementations
    "OpenMemoryMCPService",
    
    # MCP client components
    "MCPClient",
    "MCPError",
]