"""Memory service implementation for Omi-audio service.

This module provides:
- Memory configuration and initialization
- Background memory processing with process pool
- Memory operations (add, get, search, delete)
- Process-safe memory handling
"""

import asyncio
import concurrent.futures
import logging
import multiprocessing
import os
import time
from typing import List, Optional

from mem0 import AsyncMemory

# Configure Mem0 telemetry based on environment variable
# Set default to False for privacy unless explicitly enabled
if not os.getenv("MEM0_TELEMETRY"):
    os.environ["MEM0_TELEMETRY"] = "False"

# Logger for memory operations
memory_logger = logging.getLogger("memory_service")

# Memory configuration
MEM0_ORGANIZATION_ID = os.getenv("MEM0_ORGANIZATION_ID", "friend-lite-org")
MEM0_PROJECT_ID = os.getenv("MEM0_PROJECT_ID", "audio-conversations")
MEM0_APP_ID = os.getenv("MEM0_APP_ID", "omi-backend")

# Ollama & Qdrant Configuration (these should match main config)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", "qdrant")

# Global memory configuration
MEM0_CONFIG = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:latest",
            "ollama_base_url": OLLAMA_BASE_URL,
            "temperature": 0,
            "max_tokens": 2000,
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            "embedding_dims": 768,
            "ollama_base_url": OLLAMA_BASE_URL,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "omi_memories",
            "embedding_model_dims": 768,
            "host": QDRANT_BASE_URL,
            "port": 6333,
        },
    },
    "custom_prompt": "Extract action items from the conversation. Don't extract likes and dislikes.",
}

# Global instances
_memory_service = None
_process_memory = None  # For worker processes


def init_memory_config(
    ollama_base_url: Optional[str] = None,
    qdrant_base_url: Optional[str] = None,
    organization_id: Optional[str] = None,
    project_id: Optional[str] = None,
    app_id: Optional[str] = None,
) -> dict:
    """Initialize and return memory configuration with optional overrides."""
    global MEM0_CONFIG, MEM0_ORGANIZATION_ID, MEM0_PROJECT_ID, MEM0_APP_ID
    
    if ollama_base_url:
        MEM0_CONFIG["llm"]["config"]["ollama_base_url"] = ollama_base_url
        MEM0_CONFIG["embedder"]["config"]["ollama_base_url"] = ollama_base_url
    
    if qdrant_base_url:
        MEM0_CONFIG["vector_store"]["config"]["host"] = qdrant_base_url
    
    if organization_id:
        MEM0_ORGANIZATION_ID = organization_id
    
    if project_id:
        MEM0_PROJECT_ID = project_id
        
    if app_id:
        MEM0_APP_ID = app_id
    
    return MEM0_CONFIG


async def _init_process_memory():
    """Initialize memory instance once per worker process."""
    global _process_memory
    if _process_memory is None:
        _process_memory = await AsyncMemory.from_config(MEM0_CONFIG)
    return _process_memory


async def _add_memory_to_store(transcript: str, client_id: str, audio_uuid: str) -> bool:
    """
    Function to add memory in a separate process.
    This function will be pickled and run in a process pool.
    Uses a persistent memory instance per process.
    """
    try:
        # Get or create the persistent memory instance for this process
        process_memory = await _init_process_memory()
        await process_memory.add(
            transcript,
            user_id=client_id,
            metadata={
                "source": "offline_streaming",
                "audio_uuid": audio_uuid,
                "timestamp": int(time.time()),
                "conversation_context": "audio_transcription",
                "device_type": "audio_recording",
                "organization_id": MEM0_ORGANIZATION_ID,
                "project_id": MEM0_PROJECT_ID,
                "app_id": MEM0_APP_ID,
            },
        )
        return True
    except Exception as e:
        memory_logger.error(f"Error adding memory for {audio_uuid}: {e}")
        return False


class MemoryService:
    """Service class for managing memory operations."""
    
    def __init__(self):
        self.memory = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the memory service."""
        if self._initialized:
            return
        
        try:
            # Initialize main memory instance
            self.memory = await AsyncMemory.from_config(MEM0_CONFIG)
            self._initialized = True
            memory_logger.info("Memory service initialized successfully")
            
        except Exception as e:
            memory_logger.error(f"Failed to initialize memory service: {e}")
            raise
    
    async def add_memory_async(self, transcript: str, client_id: str, audio_uuid: str) -> bool:
        """Add memory in background process (non-blocking)."""
        if not self._initialized:
            await self.initialize()
        
        try:
            success = await _add_memory_to_store(transcript, client_id, audio_uuid)
            
            if success:
                memory_logger.info(f"Added transcript for {audio_uuid} to mem0 (client: {client_id})")
            else:
                memory_logger.error(f"Failed to add memory for {audio_uuid}")
            
            return success
            
        except Exception as e:
            memory_logger.error(f"Error adding memory for {audio_uuid}: {e}")
            return False
    
    async def get_all_memories(self, user_id: str, limit: int = 100) -> dict:
        """Get all memories for a user."""
        if not self._initialized:
            await self.initialize()
        
        assert self.memory is not None, "Memory service not initialized"
        try:
            memories = await self.memory.get_all(user_id=user_id, limit=limit)
            return memories
        except Exception as e:
            memory_logger.error(f"Error fetching memories for user {user_id}: {e}")
            raise
    
    async def search_memories(self, query: str, user_id: str, limit: int = 10) -> dict:
        """Search memories using semantic similarity."""
        if not self._initialized:
            await self.initialize()
        
        assert self.memory is not None, "Memory service not initialized"
        try:
            memories = await self.memory.search(query=query, user_id=user_id, limit=limit)
            return memories
        except Exception as e:
            memory_logger.error(f"Error searching memories for user {user_id}: {e}")
            raise
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        if not self._initialized:
            await self.initialize()
        
        assert self.memory is not None, "Memory service not initialized"
        try:
            await self.memory.delete(memory_id=memory_id)
            memory_logger.info(f"Deleted memory {memory_id}")
            return True
        except Exception as e:
            memory_logger.error(f"Error deleting memory {memory_id}: {e}")
            raise
    
    async def delete_all_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user and return count of deleted memories."""
        if not self._initialized:
            await self.initialize()
        
        try:
            assert self.memory is not None, "Memory service not initialized"
            # Get all memories first to count them
            user_memories = await self.memory.get_all(user_id=user_id)
            memory_count = len(user_memories) if user_memories else 0
            
            # Delete all memories for this user
            if memory_count > 0:
                await self.memory.delete_all(user_id=user_id)
                memory_logger.info(f"Deleted {memory_count} memories for user {user_id}")
            
            return memory_count
            
        except Exception as e:
            memory_logger.error(f"Error deleting memories for user {user_id}: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """Test memory service connection."""
        try:
            if not self._initialized:
                await self.initialize()
            return True
        except Exception as e:
            memory_logger.error(f"Memory service connection test failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the memory service."""
        self._initialized = False
        memory_logger.info("Memory service shut down")


# Global service instance
def get_memory_service() -> MemoryService:
    """Get the global memory service instance."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service


async def shutdown_memory_service():
    """Shutdown the global memory service."""
    global _memory_service
    if _memory_service:
        await _memory_service.shutdown()
        _memory_service = None 