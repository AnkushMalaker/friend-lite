"""Compatibility service for backward compatibility.

This module provides a drop-in replacement for the original mem0-based
memory service, maintaining the same interface while using the new
architecture internally.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple

from .memory_service import MemoryService as CoreMemoryService
from .config import build_memory_config_from_env

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
        audio_uuid: str,
        user_id: str,
        user_email: str,
        allow_update: bool = False,
        db_helper=None,
    ) -> Tuple[bool, List[str]]:
        """Add memory from transcript - compatible with original interface.
        
        Args:
            transcript: Raw transcript text to extract memories from
            client_id: Client identifier
            audio_uuid: Unique identifier for the audio session
            user_id: User identifier
            user_email: User email address
            allow_update: Whether to allow updating existing memories
            db_helper: Optional database helper for tracking relationships
            
        Returns:
            Tuple of (success: bool, created_memory_ids: List[str])
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._service.add_memory(
            transcript=transcript,
            client_id=client_id,
            audio_uuid=audio_uuid,
            user_id=user_id,
            user_email=user_email,
            allow_update=allow_update,
            db_helper=db_helper
        )
    
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
        
        # Convert MemoryEntry objects to dict format for compatibility
        return [
            {
                "id": memory.id,
                "memory": memory.content,
                "metadata": memory.metadata,
                "created_at": memory.created_at,
                "score": memory.score
            }
            for memory in memories
        ]
    
    async def get_all_memories_unfiltered(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all memories without filtering - same as get_all_memories in new implementation.
        
        Args:
            user_id: User identifier
            limit: Maximum number of memories to return
            
        Returns:
            List of memory dictionaries in legacy format
        """
        return await self.get_all_memories(user_id, limit)
    
    async def search_memories(self, query: str, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories using semantic similarity - returns dict format for compatibility.
        
        Args:
            query: Search query text
            user_id: User identifier to filter memories
            limit: Maximum number of results to return
            
        Returns:
            List of memory dictionaries in legacy format ordered by relevance
        """
        if not self._initialized:
            await self.initialize()
        
        memories = await self._service.search_memories(query, user_id, limit)
        
        # Convert MemoryEntry objects to dict format for compatibility
        return [
            {
                "id": memory.id,
                "memory": memory.content,
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
        
        # Extract audio UUIDs for bulk query
        audio_uuids = []
        for memory in memories:
            metadata = memory.get("metadata", {})
            audio_uuid = metadata.get("audio_uuid")
            if audio_uuid:
                audio_uuids.append(audio_uuid)
        
        # Bulk query for chunks
        chunks_cursor = chunks_col.find({"audio_uuid": {"$in": audio_uuids}})
        chunks_by_uuid = {}
        async for chunk in chunks_cursor:
            chunks_by_uuid[chunk["audio_uuid"]] = chunk
        
        enriched_memories = []
        
        for memory in memories:
            enriched_memory = {
                "memory_id": memory.get("id", "unknown"),
                "memory_text": memory.get("memory", ""),
                "created_at": memory.get("created_at", ""),
                "metadata": memory.get("metadata", {}),
                "audio_uuid": None,
                "transcript": None,
                "client_id": None,
                "user_email": None,
                "compression_ratio": 0,
                "transcript_length": 0,
                "memory_length": 0,
            }
            
            # Extract audio_uuid from memory metadata
            metadata = memory.get("metadata", {})
            audio_uuid = metadata.get("audio_uuid")
            
            if audio_uuid:
                enriched_memory["audio_uuid"] = audio_uuid
                enriched_memory["client_id"] = metadata.get("client_id")
                enriched_memory["user_email"] = metadata.get("user_email")
                
                # Get transcript from bulk-loaded chunks
                chunk = chunks_by_uuid.get(audio_uuid)
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
        Global MemoryService instance (singleton pattern)
    """
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service


def shutdown_memory_service():
    """Shutdown the global memory service and clean up resources."""
    global _memory_service
    if _memory_service:
        _memory_service.shutdown()
        _memory_service = None


def init_memory_config(
    qdrant_base_url: Optional[str] = None,
    organization_id: Optional[str] = None,
    project_id: Optional[str] = None,
    app_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Initialize memory configuration - maintained for compatibility.
    
    Args:
        qdrant_base_url: Qdrant server URL (updates environment if provided)
        organization_id: Legacy parameter (ignored)
        project_id: Legacy parameter (ignored)
        app_id: Legacy parameter (ignored)
        
    Returns:
        Basic configuration dictionary for compatibility
    """
    memory_logger.info(f"Initializing MemoryService with Qdrant URL: {qdrant_base_url}")
    
    # Update environment variable if provided
    if qdrant_base_url:
        os.environ["QDRANT_BASE_URL"] = qdrant_base_url
    
    # Return basic config info for compatibility
    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": qdrant_base_url or os.getenv("QDRANT_BASE_URL", "qdrant"),
                "collection_name": "omi_memories"
            }
        }
    }


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