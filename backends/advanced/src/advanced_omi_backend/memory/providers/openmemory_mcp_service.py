"""OpenMemory MCP implementation of MemoryServiceBase.

This module provides a concrete implementation of the MemoryServiceBase interface
that uses OpenMemory MCP as the backend for all memory operations. It maintains
compatibility with the existing Friend-Lite memory service API while leveraging
OpenMemory's standardized memory management capabilities.
"""

import logging
import time
import uuid
from typing import Optional, List, Tuple, Any, Dict

from ..base import MemoryServiceBase, MemoryEntry
from .mcp_client import MCPClient, MCPError

memory_logger = logging.getLogger("memory_service")


class OpenMemoryMCPService(MemoryServiceBase):
    """Memory service implementation using OpenMemory MCP as backend.
    
    This class implements the MemoryServiceBase interface by delegating memory
    operations to an OpenMemory MCP server. It handles the translation between
    Friend-Lite's memory service API and the standardized MCP operations.
    
    Key features:
    - Maintains compatibility with existing MemoryServiceBase interface
    - Leverages OpenMemory MCP's deduplication and processing
    - Supports transcript-based memory extraction 
    - Provides user isolation and metadata management
    - Handles memory search and CRUD operations
    
    Attributes:
        server_url: URL of the OpenMemory MCP server
        timeout: Request timeout in seconds
        extract_locally: Whether to extract memories locally before sending to MCP
        mcp_client: Client for communicating with MCP server
        _initialized: Whether the service has been initialized
    """
    
    def __init__(
        self, 
        server_url: str = "http://localhost:8765", 
        client_name: str = "friend_lite",
        user_id: str = "default",
        timeout: int = 30
    ):
        """Initialize OpenMemory MCP service as a thin client.
        
        This service delegates all memory processing to the OpenMemory MCP server:
        - Memory extraction (OpenMemory handles internally)
        - Deduplication (OpenMemory handles internally) 
        - Vector storage (OpenMemory handles internally)
        - User isolation via ACL (OpenMemory handles internally)
        
        Args:
            server_url: URL of the OpenMemory MCP server (default: http://localhost:8765)
            client_name: Client identifier for OpenMemory MCP
            user_id: User identifier for memory isolation via OpenMemory ACL
            timeout: HTTP request timeout in seconds
        """
        self.server_url = server_url
        self.client_name = client_name
        self.user_id = user_id
        self.timeout = timeout
        self.mcp_client: Optional[MCPClient] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the OpenMemory MCP service.
        
        Sets up the MCP client connection and tests connectivity to ensure
        the service is ready for memory operations.
        
        Raises:
            RuntimeError: If initialization or connection test fails
        """
        if self._initialized:
            return
        
        try:
            self.mcp_client = MCPClient(
                server_url=self.server_url,
                client_name=self.client_name,
                user_id=self.user_id,
                timeout=self.timeout
            )
            
            # Test connection to OpenMemory MCP server
            is_connected = await self.mcp_client.test_connection()
            if not is_connected:
                raise RuntimeError(f"Cannot connect to OpenMemory MCP server at {self.server_url}")
            
            self._initialized = True
            memory_logger.info(
                f"✅ OpenMemory MCP service initialized successfully at {self.server_url} "
                f"(client: {self.client_name}, user: {self.user_id})"
            )
            
        except Exception as e:
            memory_logger.error(f"OpenMemory MCP service initialization failed: {e}")
            raise RuntimeError(f"Initialization failed: {e}")
    
    async def add_memory(
        self,
        transcript: str,
        client_id: str,
        source_id: str,
        user_id: str,
        user_email: str,
        allow_update: bool = False,
        db_helper: Any = None
    ) -> Tuple[bool, List[str]]:
        """Add memories extracted from a transcript.
        
        Processes a transcript to extract meaningful memories and stores them
        in the OpenMemory MCP server. Can either extract memories locally first
        or send the raw transcript to MCP for processing.
        
        Args:
            transcript: Raw transcript text to extract memories from
            client_id: Client identifier for tracking
            source_id: Unique identifier for the source (audio session, chat session, etc.)
            user_id: User identifier for memory scoping
            user_email: User email address
            allow_update: Whether to allow updating existing memories (Note: MCP may handle this internally)
            db_helper: Optional database helper for relationship tracking
            
        Returns:
            Tuple of (success: bool, created_memory_ids: List[str])
            
        Raises:
            MCPError: If MCP server communication fails
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Skip empty transcripts
            if not transcript or len(transcript.strip()) < 10:
                memory_logger.info(f"Skipping empty transcript for {source_id}")
                return True, []
            
            # Update MCP client user context for this operation
            original_user_id = self.mcp_client.user_id
            self.mcp_client.user_id = "openmemory"  # Use consistent OpenMemory user ID
            
            try:
                # Thin client approach: Send raw transcript to OpenMemory MCP server
                # OpenMemory handles: extraction, deduplication, vector storage, ACL
                enriched_transcript = f"[Source: {source_id}, Client: {client_id}] {transcript}"
                
                memory_logger.info(f"Delegating memory processing to OpenMemory MCP for {source_id}")
                memory_ids = await self.mcp_client.add_memories(text=enriched_transcript)
                    
            finally:
                # Restore original user_id
                self.mcp_client.user_id = original_user_id
            
            # Update database relationships if helper provided
            if memory_ids and db_helper:
                await self._update_database_relationships(db_helper, source_id, memory_ids)
            
            if memory_ids:
                memory_logger.info(f"✅ OpenMemory MCP processed memory for {source_id}: {len(memory_ids)} memories")
                return True, memory_ids
            
            # NOOP due to deduplication is SUCCESS, not failure
            memory_logger.info(f"✅ OpenMemory MCP processed {source_id}: no new memories needed (likely deduplication)")
            return True, []
            
        except MCPError as e:
            memory_logger.error(f"❌ OpenMemory MCP error for {source_id}: {e}")
            raise e
        except Exception as e:
            memory_logger.error(f"❌ OpenMemory MCP service failed for {source_id}: {e}")
            raise e
    
    async def search_memories(
        self, 
        query: str, 
        user_id: str, 
        limit: int = 10,
        score_threshold: float = 0.0
    ) -> List[MemoryEntry]:
        """Search memories using semantic similarity.
        
        Uses the OpenMemory MCP server to perform semantic search across
        stored memories for the specified user.
        
        Args:
            query: Search query text
            user_id: User identifier to filter memories
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (ignored - OpenMemory MCP server controls filtering)
            
        Returns:
            List of matching MemoryEntry objects ordered by relevance
        """
        if not self._initialized:
            await self.initialize()
        
        # Update MCP client user context for this operation
        original_user_id = self.mcp_client.user_id
        self.mcp_client.user_id = "openmemory"  # Use consistent OpenMemory user ID
        
        try:
            results = await self.mcp_client.search_memory(
                query=query,
                limit=limit
            )
            
            # Convert MCP results to MemoryEntry objects
            memory_entries = []
            for result in results:
                memory_entry = self._mcp_result_to_memory_entry(result, user_id)
                if memory_entry:
                    memory_entries.append(memory_entry)
            
            memory_logger.info(f"🔍 Found {len(memory_entries)} memories for query '{query}' (user: {user_id})")
            return memory_entries
            
        except MCPError as e:
            memory_logger.error(f"Search memories failed: {e}")
            return []
        except Exception as e:
            memory_logger.error(f"Search memories failed: {e}")
            return []
        finally:
            # Restore original user_id
            self.mcp_client.user_id = original_user_id
    
    async def get_all_memories(
        self, 
        user_id: str, 
        limit: int = 100
    ) -> List[MemoryEntry]:
        """Get all memories for a specific user.
        
        Retrieves all stored memories for the given user without
        similarity filtering.
        
        Args:
            user_id: User identifier
            limit: Maximum number of memories to return
            
        Returns:
            List of MemoryEntry objects for the user
        """
        if not self._initialized:
            await self.initialize()
        
        # Update MCP client user context for this operation
        original_user_id = self.mcp_client.user_id
        self.mcp_client.user_id = "openmemory"  # Use consistent OpenMemory user ID
        
        try:
            results = await self.mcp_client.list_memories(limit=limit)
            
            # Convert MCP results to MemoryEntry objects
            memory_entries = []
            for result in results:
                memory_entry = self._mcp_result_to_memory_entry(result, user_id)
                if memory_entry:
                    memory_entries.append(memory_entry)
            
            memory_logger.info(f"📚 Retrieved {len(memory_entries)} memories for user {user_id}")
            return memory_entries
            
        except MCPError as e:
            memory_logger.error(f"Get all memories failed: {e}")
            return []
        except Exception as e:
            memory_logger.error(f"Get all memories failed: {e}")
            return []
        finally:
            # Restore original user_id
            self.mcp_client.user_id = original_user_id
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID.
        
        Args:
            memory_id: Unique identifier of the memory to delete
            
        Returns:
            True if successfully deleted, False otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            success = await self.mcp_client.delete_memory(memory_id)
            if success:
                memory_logger.info(f"🗑️ Deleted memory {memory_id} via MCP")
            return success
        except Exception as e:
            memory_logger.error(f"Delete memory failed: {e}")
            return False
    
    async def delete_all_user_memories(self, user_id: str) -> int:
        """Delete all memories for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of memories that were deleted
        """
        if not self._initialized:
            await self.initialize()
        
        # Update MCP client user context for this operation
        original_user_id = self.mcp_client.user_id
        self.mcp_client.user_id = "openmemory"  # Use consistent OpenMemory user ID
        
        try:
            count = await self.mcp_client.delete_all_memories()
            memory_logger.info(f"🗑️ Deleted {count} memories for user {user_id} via OpenMemory MCP")
            return count
            
        except Exception as e:
            memory_logger.error(f"Delete user memories failed: {e}")
            return 0
        finally:
            # Restore original user_id
            self.mcp_client.user_id = original_user_id
    
    async def test_connection(self) -> bool:
        """Test if the memory service and its dependencies are working.
        
        Returns:
            True if all connections are healthy, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            return await self.mcp_client.test_connection()
        except Exception as e:
            memory_logger.error(f"Connection test failed: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the memory service and clean up resources."""
        if self.mcp_client:
            # Note: MCPClient cleanup handled by async context manager
            pass
        self._initialized = False
        self.mcp_client = None
        memory_logger.info("OpenMemory MCP service shut down")
    
    # Private helper methods
    
    def _ensure_client(self) -> MCPClient:
        """Ensure MCP client is available and return it."""
        if self.mcp_client is None:
            raise RuntimeError("OpenMemory MCP client not initialized")
        return self.mcp_client
    
    def _mcp_result_to_memory_entry(self, mcp_result: Dict[str, Any], user_id: str) -> Optional[MemoryEntry]:
        """Convert OpenMemory MCP server result to MemoryEntry object.
        
        Args:
            mcp_result: Result dictionary from OpenMemory MCP server
            user_id: User identifier to include in metadata
            
        Returns:
            MemoryEntry object or None if conversion fails
        """
        try:
            # OpenMemory MCP results may have different formats, adapt as needed
            memory_id = mcp_result.get('id', str(uuid.uuid4()))
            content = mcp_result.get('content', '') or mcp_result.get('memory', '') or mcp_result.get('text', '') or mcp_result.get('data', '')
            
            if not content:
                memory_logger.warning(f"Empty content in MCP result: {mcp_result}")
                return None
            
            # Build metadata with OpenMemory context
            metadata = mcp_result.get('metadata', {})
            if not metadata:
                metadata = {}
            
            # Ensure we have user context
            metadata.update({
                'user_id': user_id,
                'source': 'openmemory_mcp',
                'client_name': self.client_name,
                'mcp_server': self.server_url
            })
            
            # Extract similarity score if available (for search results)
            score = mcp_result.get('score') or mcp_result.get('similarity') or mcp_result.get('relevance')
            
            # Extract timestamp
            created_at = mcp_result.get('created_at') or mcp_result.get('timestamp') or mcp_result.get('date')
            if created_at is None:
                created_at = str(int(time.time()))
            
            return MemoryEntry(
                id=memory_id,
                content=content,
                metadata=metadata,
                embedding=None,  # OpenMemory MCP server handles embeddings internally
                score=score,
                created_at=str(created_at)
            )
            
        except Exception as e:
            memory_logger.error(f"Failed to convert MCP result to MemoryEntry: {e}")
            return None
    
    async def _update_database_relationships(
        self, 
        db_helper: Any, 
        source_id: str, 
        created_ids: List[str]
    ) -> None:
        """Update database relationships for created memories.
        
        Args:
            db_helper: Database helper instance
            source_id: Source session identifier
            created_ids: List of created memory IDs
        """
        for memory_id in created_ids:
            try:
                await db_helper.add_memory_reference(source_id, memory_id, "created")
            except Exception as db_error:
                memory_logger.error(f"Database relationship update failed: {db_error}")