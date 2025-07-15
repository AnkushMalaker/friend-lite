"""Memory service implementation for Omi-audio service.

This module provides:
- Memory configuration and initialization
- Memory operations (add, get, search, delete)
- Action item extraction and management
"""

import logging
import os
import time
import json
from typing import Optional, List, Dict, Any

from mem0 import Memory
from .mem0_client import get_memory_client

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


# Action item extraction configuration

# Global instances
_memory_service = None
_process_memory = None  # For worker processes

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _init_process_memory():
    """Initialize memory instance once per worker process."""
    global _process_memory
    if _process_memory is None:
        _process_memory = get_memory_client()
    return _process_memory


def _add_memory_to_store(transcript: str, client_id: str, audio_uuid: str) -> bool:
    """
    Function to add memory in a separate process.
    This function will be pickled and run in a process pool.
    Uses a persistent memory instance per process.
    """
    try:
        # Get or create the persistent memory instance for this process
        mem0_client = _init_process_memory()
        response = mem0_client.add(
            transcript,
            user_id=client_id,
            # metadata={
            #     "source": "offline_streaming",
            #     "audio_uuid": audio_uuid,
            #     "timestamp": int(time.time()),
            #     "conversation_context": "audio_transcription",
            #     "device_type": "audio_recording",
            #     "organization_id": MEM0_ORGANIZATION_ID,
            #     "project_id": MEM0_PROJECT_ID,
            #     "app_id": MEM0_APP_ID,
            # },
        )
        memory_logger.info(f"Added transcript {transcript} for {audio_uuid} to mem0 (client: {client_id})")
        memory_logger.info(f"Memory add response: {response}")
        return True
    except Exception as e:
        memory_logger.error(f"Error adding memory for {audio_uuid}: {e}")
        return False






class MemoryService:
    """Service class for managing memory operations."""
    
    def __init__(self):
        self.memory = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the memory service."""
        if self._initialized:
            return
        
        try:
            # Initialize main memory instance
            self.memory = get_memory_client()
            self._initialized = True
            memory_logger.info("Memory service initialized successfully")
            
        except Exception as e:
            memory_logger.error(f"Failed to initialize memory service: {e}")
            raise

    def add_memory(self, transcript: str, client_id: str, audio_uuid: str) -> bool:
        """Add memory in background process (non-blocking)."""
        if not self._initialized:
            self.initialize()
        
        try:
            success = _add_memory_to_store(transcript, client_id, audio_uuid)
            if success:
                memory_logger.info(f"Added transcript for {audio_uuid} to mem0 (client: {client_id})")
            else:
                memory_logger.error(f"Failed to add memory for {audio_uuid}")
            return success
        except Exception as e:
            memory_logger.error(f"Error adding memory for {audio_uuid}: {e}")
            return False
    
    
    def get_action_items(self, user_id: str, limit: int = 50, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get action items for a user with optional status filtering.
        """
        if not self._initialized:
            self.initialize()
        
        assert self.memory is not None, "Memory service not initialized"
        try:
            # First, let's try to get all memories and filter manually to debug the issue
            all_memories = self.memory.get_all(user_id=user_id, limit=200)
            
            memory_logger.info(f"All memories response type: {type(all_memories)}")
            memory_logger.info(f"All memories keys: {list(all_memories.keys()) if isinstance(all_memories, dict) else 'not a dict'}")
            
            # Handle different formats
            if isinstance(all_memories, dict):
                if "results" in all_memories:
                    memories_list = all_memories["results"]
                else:
                    memories_list = list(all_memories.values())
            else:
                memories_list = all_memories if isinstance(all_memories, list) else []
            
            memory_logger.info(f"Found {len(memories_list)} total memories for user {user_id}")
            
            # Filter for action items manually
            action_item_memories = []
            for memory in memories_list:
                if isinstance(memory, dict):
                    metadata = memory.get('metadata', {})
                    memory_logger.info(f"Memory {memory.get('id', 'unknown')}: metadata = {metadata}")
                    
                    if metadata.get('type') == 'action_item':
                        action_item_memories.append(memory)
                        memory_logger.info(f"Found action item memory: {memory.get('memory', '')}")
            
            memory_logger.info(f"Found {len(action_item_memories)} action item memories")
            
            # Extract action item data from memories
            action_items = []
            
            for memory in action_item_memories:
                metadata = memory.get('metadata', {})
                action_item_data = metadata.get('action_item_data', {})
                
                # If no action_item_data, try to parse from memory text
                if not action_item_data:
                    memory_logger.warning(f"No action_item_data found in metadata for memory {memory.get('id')}")
                    # Try to create basic action item from memory text
                    memory_text = memory.get('memory', '')
                    if memory_text.startswith('Action Item:'):
                        action_item_data = {
                            'description': memory_text.replace('Action Item:', '').strip(),
                            'status': 'open',
                            'assignee': 'unassigned',
                            'due_date': 'not_specified',
                            'priority': 'not_specified'
                        }
                
                # Apply status filter if specified
                if status_filter and action_item_data.get('status') != status_filter:
                    continue
                
                # Enrich with memory metadata
                action_item_data.update({
                    "memory_id": memory.get('id'),
                    "memory_text": memory.get('memory'),
                    "created_at": metadata.get('timestamp'),
                    "audio_uuid": metadata.get('audio_uuid')
                })
                
                action_items.append(action_item_data)
            
            memory_logger.info(f"Returning {len(action_items)} action items after filtering")
            return action_items
            
        except Exception as e:
            memory_logger.error(f"Error fetching action items for user {user_id}: {e}")
            raise
    
    def update_action_item_status(self, memory_id: str, new_status: str, user_id: Optional[str] = None) -> bool:
        """
        Update the status of an action item using proper Mem0 API.
        """
        if not self._initialized:
            self.initialize()
        
        assert self.memory is not None, "Memory service not initialized"
        try:
            # First, get the current memory to retrieve its metadata
            target_memory = self.memory.get(memory_id=memory_id)
            
            if not target_memory:
                memory_logger.error(f"Action item with memory_id {memory_id} not found")
                return False
            
            # Extract and update the action item data in metadata
            metadata = target_memory.get('metadata', {})
            action_item_data = metadata.get('action_item_data', {})
            
            if not action_item_data:
                memory_logger.error(f"No action_item_data found in memory {memory_id}")
                return False
            
            # Update the status in action_item_data
            action_item_data['status'] = new_status
            action_item_data['updated_at'] = int(time.time())
            
            # Create updated memory text with the new status
            updated_memory_text = f"Action Item: {action_item_data.get('description', 'No description')} (Status: {new_status})"
            if action_item_data.get('assignee') and action_item_data.get('assignee') != 'unassigned':
                updated_memory_text += f" (Assigned to: {action_item_data['assignee']})"
            if action_item_data.get('due_date') and action_item_data.get('due_date') != 'not_specified':
                updated_memory_text += f" (Due: {action_item_data['due_date']})"
            
            # Use Mem0's proper update method
            result = self.memory.update(
                memory_id=memory_id,
                data=updated_memory_text
            )
            
            memory_logger.info(f"Updated action item {memory_id} status to {new_status}")
            return True
            
        except Exception as e:
            memory_logger.error(f"Error updating action item status for {memory_id}: {e}")
            return False
    
    def search_action_items(self, query: str, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search action items by text query using proper Mem0 search with filters.
        """
        if not self._initialized:
            self.initialize()
        
        assert self.memory is not None, "Memory service not initialized"
        try:
            # Use Mem0's search with filters to find action items
            # According to docs, we can pass custom filters
            memories = self.memory.search(
                query=query,
                user_id=user_id,
                limit=limit,
                filters={"metadata.type": "action_item"}
            )
            
            # Extract action item data
            action_items = []
            
            # Handle different response formats from Mem0 search
            if isinstance(memories, dict) and "results" in memories:
                memories_list = memories["results"]
            elif isinstance(memories, list):
                memories_list = memories
            else:
                memory_logger.warning(f"Unexpected search response format: {type(memories)}")
                memories_list = []
            
            for memory in memories_list:
                if not isinstance(memory, dict):
                    memory_logger.warning(f"Skipping non-dict memory: {type(memory)}")
                    continue
                
                metadata = memory.get('metadata', {})
                
                # Double-check it's an action item
                if metadata.get('type') != 'action_item':
                    continue
                
                action_item_data = metadata.get('action_item_data', {})
                
                # If no structured action item data, try to parse from memory text
                if not action_item_data:
                    memory_text = memory.get('memory', '')
                    if memory_text.startswith('Action Item:'):
                        action_item_data = {
                            'description': memory_text.replace('Action Item:', '').strip(),
                            'status': 'open',
                            'assignee': 'unassigned',
                            'due_date': 'not_specified',
                            'priority': 'not_specified'
                        }
                
                # Enrich with memory metadata
                action_item_data.update({
                    "memory_id": memory.get('id'),
                    "memory_text": memory.get('memory'),
                    "relevance_score": memory.get('score', 0),
                    "created_at": metadata.get('timestamp'),
                    "audio_uuid": metadata.get('audio_uuid')
                })
                
                action_items.append(action_item_data)
            
            memory_logger.info(f"Search found {len(action_items)} action items for query '{query}'")
            return action_items
            
        except Exception as e:
            memory_logger.error(f"Error searching action items for user {user_id} with query '{query}': {e}")
            # Fallback: get all action items and do basic text matching
            try:
                all_action_items = self.get_action_items(user_id=user_id, limit=100)
                
                if not all_action_items:
                    return []
                
                # Simple text matching fallback
                search_results = []
                query_lower = query.lower()
                
                for item in all_action_items:
                    description = item.get('description', '').lower()
                    assignee = item.get('assignee', '').lower()
                    context = item.get('context', '').lower()
                    
                    # Check if query appears in any field
                    if (query_lower in description or 
                        query_lower in assignee or 
                        query_lower in context):
                        
                        # Add relevance score based on where the match was found
                        relevance_score = 0.0
                        if query_lower in description:
                            relevance_score += 0.7
                        if query_lower in assignee:
                            relevance_score += 0.2
                        if query_lower in context:
                            relevance_score += 0.1
                        
                        item['relevance_score'] = relevance_score
                        search_results.append(item)
                
                # Sort by relevance score (highest first) and limit results
                search_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                memory_logger.info(f"Fallback search found {len(search_results)} matches")
                return search_results[:limit]
                
            except Exception as fallback_e:
                memory_logger.error(f"Fallback search also failed: {fallback_e}")
                return []
    
    def delete_action_item(self, memory_id: str) -> bool:
        """Delete a specific action item by memory ID."""
        if not self._initialized:
            self.initialize()
        
        assert self.memory is not None, "Memory service not initialized"
        try:
            self.memory.delete(memory_id=memory_id)
            memory_logger.info(f"Deleted action item with memory_id {memory_id}")
            return True
        except Exception as e:
            memory_logger.error(f"Error deleting action item {memory_id}: {e}")
            return False

    def get_all_memories(self, user_id: str, limit: int = 100) -> dict:
        """Get all memories for a user."""
        if not self._initialized:
            self.initialize()
        
        assert self.memory is not None, "Memory service not initialized"
        try:
            memories = self.memory.get_all(user_id=user_id, limit=limit)
            return memories
        except Exception as e:
            memory_logger.error(f"Error fetching memories for user {user_id}: {e}")
            raise
    
    def search_memories(self, query: str, user_id: str, limit: int = 10) -> dict:
        """Search memories using semantic similarity."""
        if not self._initialized:
            self.initialize()
        
        assert self.memory is not None, "Memory service not initialized"
        try:
            memories = self.memory.search(query=query, user_id=user_id, limit=limit)
            return memories
        except Exception as e:
            memory_logger.error(f"Error searching memories for user {user_id}: {e}")
            raise
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        if not self._initialized:
            self.initialize()
        
        assert self.memory is not None, "Memory service not initialized"
        try:
            self.memory.delete(memory_id=memory_id)
            memory_logger.info(f"Deleted memory {memory_id}")
            return True
        except Exception as e:
            memory_logger.error(f"Error deleting memory {memory_id}: {e}")
            raise
    
    def delete_all_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user and return count of deleted memories."""
        if not self._initialized:
            self.initialize()
        
        try:
            assert self.memory is not None, "Memory service not initialized"
            # Get all memories first to count them
            user_memories_response = self.memory.get_all(user_id=user_id)
            memory_count = 0
            
            # Handle different response formats from get_all
            if isinstance(user_memories_response, dict):
                if "results" in user_memories_response:
                    # New paginated format
                    memory_count = len(user_memories_response["results"])
                else:
                    # Old dict format (deprecated)
                    memory_count = len(user_memories_response)
            elif isinstance(user_memories_response, list):
                # Just in case it returns a list
                memory_count = len(user_memories_response)
            else:
                memory_count = 0
            
            # Delete all memories for this user
            if memory_count > 0:
                self.memory.delete_all(user_id=user_id)
                memory_logger.info(f"Deleted {memory_count} memories for user {user_id}")
            
            return memory_count
            
        except Exception as e:
            memory_logger.error(f"Error deleting memories for user {user_id}: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test memory service connection."""
        try:
            if not self._initialized:
                self.initialize()
            return True
        except Exception as e:
            memory_logger.error(f"Memory service connection test failed: {e}")
            return False
    
    def shutdown(self):
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


def shutdown_memory_service():
    """Shutdown the global memory service."""
    global _memory_service
    if _memory_service:
        _memory_service.shutdown()
        _memory_service = None 