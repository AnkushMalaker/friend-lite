"""Memory module for Omi-audio service.

This module handles all memory-related operations including:
- Memory configuration and initialization
- Background memory processing
- Memory API operations (get, search, delete)
"""

from .memory_service import (
    MemoryService,
    # init_memory_config,
    get_memory_service,
    shutdown_memory_service,
)

from .mem0_client import get_memory_client

__all__ = [
    "MemoryService",
    "init_memory_config", 
    "get_memory_service",
    "shutdown_memory_service",
] 