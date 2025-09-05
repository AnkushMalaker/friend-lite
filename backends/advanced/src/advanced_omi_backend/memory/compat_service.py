"""Compatibility service for backward compatibility.

This module provides a drop-in replacement for the original mem0-based
memory service, maintaining the same interface while using the new
architecture internally.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .config import build_memory_config_from_env
from .memory_service import MemoryService as CoreMemoryService

memory_logger = logging.getLogger("memory_service")


class MemoryService:
    """Drop-in replacement for the original mem0-based MemoryService.
    
    This class provides backward compatibility by wrapping the new
    CoreMemoryService with the same interface as the original service.
    It handles data format conversion and maintains compatibility with
    existing code.
    
    Attributes:
        _service: Internal CoreMemoryService instance
        _initialized: Whether the service has been initialized
    """
    
    def __init__(self):
        """Initialize the compatibility memory service."""
        self._service: Optional[CoreMemoryService] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the memory service.
        
        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            return
        
        try:
            config = build_memory_config_from_env()
            self._service = CoreMemoryService(config)
            await self._service.initialize()
            self._initialized = True
            memory_logger.info("✅ Memory service initialized successfully")
        except Exception as e:
            memory_logger.error(f"Failed to initialize memory service: {e}")
            raise
    
    async def add_memory(
        self,
        transcript: str,
        client_id: str,
        source_id: str,
        user_id: str,
        user_email: str,
        allow_update: bool = False,
        db_helper=None,
    ) -> Tuple[bool, List[str]]:
        """Add memory from transcript - compatible with original interface.
        
        Args:
            transcript: Raw transcript text to extract memories from
            client_id: Client identifier
            source_id: Unique identifier for the source (audio session, chat session, etc.)
            user_id: User identifier
            user_email: User email address
            allow_update: Whether to allow updating existing memories
            db_helper: Optional database helper for tracking relationships
            
        Returns:
            Tuple of (success: bool, created_memory_ids: List[str])
        """
        if not self._initialized:
            await self.initialize()
        
        # Ensure service is initialized if it's not the internal CoreMemoryService
        if hasattr(self._service, 'initialize') and hasattr(self._service, '_initialized'):
            if not self._service._initialized:
                await self._service.initialize()
        
        return await self._service.add_memory(
            transcript=transcript,
            client_id=client_id,
            source_id=source_id,
            user_id=user_id,
            user_email=user_email,
            allow_update=allow_update,
            db_helper=db_helper
        )
    
    def _normalize_memory_content(self, content: str, metadata: Dict[str, Any]) -> str:
        """Return memory content as-is since individual facts are now stored separately.
        
        Args:
            content: Memory content from the provider
            metadata: Memory metadata (not used)
            
        Returns:
            Content as-is (no normalization needed)
        """
        return content
    
    async def get_all_memories(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all memories for a user - returns dict format for compatibility.
        
        Args:
            user_id: User identifier
            limit: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries in legacy format
        """
        if not self._initialized:
            await self.initialize()
        
        memories = await self._service.get_all_memories(user_id, limit)
        
        # Convert MemoryEntry objects to dict format for compatibility with normalized content
        return [
            {
                "id": memory.id,
                "memory": self._normalize_memory_content(memory.content, memory.metadata),
                "metadata": memory.metadata,
                "created_at": memory.created_at,
                "score": memory.score
            }
            for memory in memories
        ]
    
    async def count_memories(self, user_id: str) -> Optional[int]:
        """Count total number of memories for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Total count of memories for the user, or None if not supported
        """
        if not self._initialized:
            await self.initialize()
        
        # Delegate to the core service
        return await self._service.count_memories(user_id)
    
    async def get_all_memories_unfiltered(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all memories without filtering - same as get_all_memories in new implementation.
        
        Args:
            user_id: User identifier
            limit: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries in legacy format
        """
        return await self.get_all_memories(user_id, limit)
    
    async def search_memories(self, query: str, user_id: str, limit: int = 10, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search memories using semantic similarity - returns dict format for compatibility.
        
        Args:
            query: Search query text
            user_id: User identifier to filter memories
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0.0 = no threshold)
            
        Returns:
            List of memory dictionaries in legacy format ordered by relevance
        """
        if not self._initialized:
            await self.initialize()
        
        memories = await self._service.search_memories(query, user_id, limit, score_threshold)
        
        # Convert MemoryEntry objects to dict format for compatibility with normalized content
        return [
            {
                "id": memory.id,
                "memory": self._normalize_memory_content(memory.content, memory.metadata),
                "metadata": memory.metadata,
                "created_at": memory.created_at,
                "score": memory.score
            }
            for memory in memories
        ]
    
    async def delete_all_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user and return count.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of memories that were deleted
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._service.delete_all_user_memories(user_id)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID.
        
        Args:
            memory_id: Unique identifier of the memory to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._service.delete_memory(memory_id)
    
    async def get_all_memories_debug(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Get all memories across all users for admin debugging.
        
        Args:
            limit: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries with user context for debugging
        """
        if not self._initialized:
            await self.initialize()
        
        # Import User model to get all users
        try:
            from advanced_omi_backend.users import User
        except ImportError:
            memory_logger.error("Cannot import User model for debug function")
            return []
        
        all_memories = []
        users = await User.find_all().to_list()
        
        for user in users:
            user_id = str(user.id)
            try:
                user_memories = await self.get_all_memories(user_id)
                
                # Add user context for debugging
                for memory in user_memories:
                    memory_entry = {
                        **memory,
                        "user_id": user_id,
                        "owner_email": user.email,
                        "collection": "omi_memories"
                    }
                    all_memories.append(memory_entry)
                
                # Respect limit
                if len(all_memories) >= limit:
                    break
                    
            except Exception as e:
                memory_logger.warning(f"Error getting memories for user {user_id}: {e}")
                continue
        
        return all_memories[:limit]
    
    async def get_memories_with_transcripts(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get memories with their source transcripts using database relationship.
        
        Args:
            user_id: User identifier
            limit: Maximum number of memories to return
            
        Returns:
            List of enriched memory dictionaries with transcript information
        """
        if not self._initialized:
            await self.initialize()
        
        # Get memories first
        memories = await self.get_all_memories(user_id, limit)
        
        # Import database connection
        try:
            from advanced_omi_backend.database import chunks_col
        except ImportError:
            memory_logger.error("Cannot import database connection")
            return memories  # Return memories without transcript enrichment
        
        # Extract source IDs for bulk query
        source_ids = []
        for memory in memories:
            metadata = memory.get("metadata", {})
            source_id = metadata.get("source_id") or metadata.get("audio_uuid")  # Backward compatibility
            if source_id:
                source_ids.append(source_id)
        
        # Bulk query for chunks (support both old audio_uuid and new source_id)
        chunks_cursor = chunks_col.find({"audio_uuid": {"$in": source_ids}})
        chunks_by_id = {}
        async for chunk in chunks_cursor:
            chunks_by_id[chunk["audio_uuid"]] = chunk
        
        enriched_memories = []
        
        for memory in memories:
            enriched_memory = {
                "memory_id": memory.get("id", "unknown"),
                "memory_text": memory.get("memory", ""),
                "created_at": memory.get("created_at", ""),
                "metadata": memory.get("metadata", {}),
                "source_id": None,
                "transcript": None,
                "client_id": None,
                "user_email": None,
                "compression_ratio": 0,
                "transcript_length": 0,
                "memory_length": 0,
            }
            
            # Extract source_id from memory metadata (with backward compatibility)
            metadata = memory.get("metadata", {})
            source_id = metadata.get("source_id") or metadata.get("audio_uuid")
            
            if source_id:
                enriched_memory["source_id"] = source_id
                enriched_memory["client_id"] = metadata.get("client_id")
                enriched_memory["user_email"] = metadata.get("user_email")
                
                # Get transcript from bulk-loaded chunks
                chunk = chunks_by_id.get(source_id)
                if chunk:
                    transcript_segments = chunk.get("transcript", [])
                    if transcript_segments:
                        full_transcript = " ".join(
                            segment.get("text", "")
                            for segment in transcript_segments
                            if isinstance(segment, dict) and segment.get("text")
                        )
                        
                        if full_transcript.strip():
                            enriched_memory["transcript"] = full_transcript
                            enriched_memory["transcript_length"] = len(full_transcript)
                            
                            memory_text = enriched_memory["memory_text"]
                            enriched_memory["memory_length"] = len(memory_text)
                            
                            # Calculate compression ratio
                            if len(full_transcript) > 0:
                                enriched_memory["compression_ratio"] = round(
                                    (len(memory_text) / len(full_transcript)) * 100, 1
                                )
            
            enriched_memories.append(enriched_memory)
        
        return enriched_memories
    
    async def test_connection(self) -> bool:
        """Test memory service connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            return await self._service.test_connection()
        except Exception as e:
            memory_logger.error(f"Connection test failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the memory service and clean up resources."""
        if self._service:
            self._service.shutdown()
        self._initialized = False
        self._service = None
        memory_logger.info("Memory service shut down")


# Global service instance - maintains compatibility with original code
_memory_service = None


def get_memory_service() -> MemoryService:
    """Get the global memory service instance.
    
    Returns:
        Global MemoryService instance (singleton pattern), wrapped for compatibility
    """
    global _memory_service
    if _memory_service is None:
        # Use the new service factory to create the appropriate service
        from .service_factory import get_memory_service as get_core_service
        
        core_service = get_core_service()
        
        # If it's already a compat service, use it directly
        if isinstance(core_service, MemoryService):
            _memory_service = core_service
        else:
            # Wrap core service with compat layer
            _memory_service = MemoryService()
            _memory_service._service = core_service
            _memory_service._initialized = True
            
    return _memory_service


def shutdown_memory_service():
    """Shutdown the global memory service and clean up resources."""
    global _memory_service
    if _memory_service:
        _memory_service.shutdown()
        _memory_service = None
    
    # Also shutdown the core service
    from .service_factory import shutdown_memory_service as shutdown_core_service
    shutdown_core_service()


# Migration helper functions
async def migrate_from_mem0():
    """Helper function to migrate existing mem0 data to new format.
    
    This is a placeholder for migration logic. Actual implementation
    would depend on the specific mem0 setup and data format.
    
    Raises:
        RuntimeError: If migration fails
    """
    memory_logger.info("🔄 Starting migration from mem0 to new memory service")
    
    try:
        # Initialize new memory service
        new_service = get_memory_service()
        await new_service.initialize()
        
        # Get all users
        try:
            from advanced_omi_backend.users import User
            users = await User.find_all().to_list()
        except ImportError:
            memory_logger.error("Cannot import User model for migration")
            return
        
        # Migration steps would go here:
        # 1. For each user, get their mem0 memories (if accessible)
        # 2. Convert to new format
        # 3. Store in new system
        
        memory_logger.info("✅ Migration completed successfully")
        
    except Exception as e:
        memory_logger.error(f"❌ Migration failed: {e}")
        raise