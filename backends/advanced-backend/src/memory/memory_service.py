"""Memory service implementation for Omi-audio service.

This module provides:
- Memory configuration and initialization
- Memory operations (add, get, search, delete)
- Action item extraction and management
- Debug tracking and configurable extraction
"""

import asyncio
import logging
import os
import time
import json
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from mem0 import Memory
import ollama

# Import debug tracker and config loader
from memory_debug import get_debug_tracker
from memory_config_loader import get_config_loader

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

# Timeout configurations
OLLAMA_TIMEOUT_SECONDS = 1200  # Timeout for Ollama operations
MEMORY_INIT_TIMEOUT_SECONDS = 60  # Timeout for memory initialization

# Thread pool for blocking operations
_MEMORY_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory_ops")

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
    "custom_prompt": (
        "Extract anything relevant about this conversation. "
        "Anything from what the conversation was about, the people involved, emotion, etc. In each memory, include: No calls mentioned if no call was mentioned."
    ),
    # "custom_fact_extraction_prompt": (
    #     "Extract anything relevant about this conversation. "
    #     "Anything from what the conversation was about, the people involved, emotion, etc."
    # ),

}

# Action item extraction configuration
ACTION_ITEM_EXTRACTION_PROMPT = """
You are an AI assistant specialized in extracting actionable tasks from meeting transcripts and conversations.

Analyze the following conversation transcript and extract all action items, tasks, and commitments mentioned.

For each action item you find, return a JSON object with these fields:
- "description": A clear, specific description of the task
- "assignee": The person responsible (use "unassigned" if not specified)
- "due_date": The deadline if mentioned (use "not_specified" if not mentioned)
- "priority": The urgency level ("high", "medium", "low", or "not_specified")
- "status": Always set to "open" for new items
- "context": A brief context about when/why this was mentioned

Return ONLY a valid JSON array of action items. If no action items are found, return an empty array [].

Examples of action items to look for:
- "I'll send you the report by Friday"
- "We need to schedule a follow-up meeting"
- "Can you review the document before tomorrow?"
- "Let's get that bug fixed"
- "I'll call the client next week"

Transcript:
{transcript}
"""

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

    memory_logger.info(f"Initializing MemoryService with Qdrant URL: {qdrant_base_url} and Ollama base URL: {ollama_base_url}")
    
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


def _init_process_memory():
    """Initialize memory instance once per worker process."""
    global _process_memory
    if _process_memory is None:
        _process_memory = Memory.from_config(MEM0_CONFIG)
    return _process_memory


def _add_memory_to_store(transcript: str, client_id: str, audio_uuid: str, user_id: str, user_email: str) -> bool:
    """
    Function to add memory in a separate process.
    This function will be pickled and run in a process pool.
    Uses a persistent memory instance per process.
    
    Args:
        transcript: The conversation transcript
        client_id: The client ID that generated the audio
        audio_uuid: Unique identifier for the audio
        user_id: Database user ID to associate the memory with
        user_email: User email for easy identification
    """
    start_time = time.time()
    
    try:
        # Get configuration and debug tracker
        config_loader = get_config_loader()
        debug_tracker = get_debug_tracker()
        
        # Start debug tracking if enabled
        session_id = None
        if config_loader.is_debug_enabled():
            session_id = debug_tracker.start_memory_session(audio_uuid, client_id, user_id, user_email)
            debug_tracker.start_memory_processing(session_id)
        
        # Check if conversation should be skipped
        if config_loader.should_skip_conversation(transcript):
            if session_id:
                debug_tracker.complete_memory_processing(session_id, False, "Conversation skipped due to quality control")
            memory_logger.info(f"Skipping memory processing for {audio_uuid} due to quality control")
            return True  # Not an error, just skipped
        
        # Get memory extraction configuration
        memory_config = config_loader.get_memory_extraction_config()
        if not memory_config.get("enabled", True):
            if session_id:
                debug_tracker.complete_memory_processing(session_id, False, "Memory extraction disabled")
            memory_logger.info(f"Memory extraction disabled for {audio_uuid}")
            return True
        
        # Get or create the persistent memory instance for this process
        process_memory = _init_process_memory()
        
        # Use configured prompt or default
        prompt = memory_config.get("prompt", "Please extract summary of the conversation - any topics or names")
        
        # Add the memory with configured settings
        result = process_memory.add(
            transcript,
            user_id=user_id,  # Use database user_id instead of client_id
            metadata={
                "source": "offline_streaming",
                "client_id": client_id,  # Store client_id in metadata
                "user_email": user_email,  # Store user email for easy identification
                "audio_uuid": audio_uuid,
                "timestamp": int(time.time()),
                "conversation_context": "audio_transcription",
                "device_type": "audio_recording",
                "organization_id": MEM0_ORGANIZATION_ID,
                "project_id": MEM0_PROJECT_ID,
                "app_id": MEM0_APP_ID,
                "extraction_method": "configurable",
                "config_enabled": True,
            },
            prompt=prompt
        )
        
        # Record debug information
        if session_id:
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Record the memory extraction
            memory_id = result.get("id") if isinstance(result, dict) else str(result)
            memory_text = result.get("memory") if isinstance(result, dict) else str(result)
            
            # Ensure we have string values
            if not isinstance(memory_id, str):
                memory_id = str(memory_id) if memory_id is not None else "unknown"
            if not isinstance(memory_text, str):
                memory_text = str(memory_text) if memory_text is not None else "unknown"
            
            debug_tracker.add_memory_extraction(
                session_id=session_id,
                audio_uuid=audio_uuid,
                mem0_memory_id=memory_id,
                memory_text=memory_text,
                memory_type="general",
                extraction_prompt=prompt,
                metadata={
                    "client_id": client_id,
                    "user_email": user_email,
                    "processing_time_ms": processing_time_ms
                }
            )
            
            debug_tracker.add_extraction_attempt(
                session_id=session_id,
                audio_uuid=audio_uuid,
                attempt_type="memory_extraction",
                success=True,
                processing_time_ms=processing_time_ms,
                transcript_length=len(transcript),
                prompt_used=prompt,
                llm_model=memory_config.get("llm_settings", {}).get("model", "llama3.1:latest")
            )
            
            debug_tracker.complete_memory_processing(session_id, True)
        
        return True
        
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        memory_logger.error(f"Error adding memory for {audio_uuid}: {e}")
        
        # Record debug information for failure
        if session_id:
            debug_tracker.add_extraction_attempt(
                session_id=session_id,
                audio_uuid=audio_uuid,
                attempt_type="memory_extraction",
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time_ms,
                transcript_length=len(transcript) if transcript else 0
            )
            
            debug_tracker.complete_memory_processing(session_id, False, str(e))
        
        return False


# Action item extraction functions removed - now handled by ActionItemsService
# See action_items_service.py for the main action item processing logic


# Action item storage functions removed - now handled by ActionItemsService
# See action_items_service.py for the main action item processing logic


class MemoryService:
    """Service class for managing memory operations."""
    
    def __init__(self):
        self.memory = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the memory service with timeout protection."""
        if self._initialized:
            return
        
        try:
            # Log Qdrant and Ollama URLs
            memory_logger.info(f"Initializing MemoryService with Qdrant URL: {MEM0_CONFIG['vector_store']['config']['host']} and Ollama base URL: {MEM0_CONFIG['llm']['config']['ollama_base_url']}")
            
            # Initialize main memory instance with timeout protection
            loop = asyncio.get_running_loop()
            self.memory = await asyncio.wait_for(
                loop.run_in_executor(_MEMORY_EXECUTOR, Memory.from_config, MEM0_CONFIG),
                timeout=MEMORY_INIT_TIMEOUT_SECONDS
            )
            self._initialized = True
            memory_logger.info("Memory service initialized successfully")
            
        except asyncio.TimeoutError:
            memory_logger.error(f"Memory service initialization timed out after {MEMORY_INIT_TIMEOUT_SECONDS}s")
            raise Exception("Memory service initialization timeout")
        except Exception as e:
            memory_logger.error(f"Failed to initialize memory service: {e}")
            raise

    async def add_memory(self, transcript: str, client_id: str, audio_uuid: str, user_id: str, user_email: str) -> bool:
        """Add memory in background process (non-blocking).
        
        Args:
            transcript: The conversation transcript
            client_id: The client ID that generated the audio  
            audio_uuid: Unique identifier for the audio
            user_id: Database user ID to associate the memory with
            user_email: User email for identification
        """
        if not self._initialized:
            try:
                await asyncio.wait_for(
                    self.initialize(),
                    timeout=MEMORY_INIT_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                memory_logger.error(f"Memory initialization timed out for {audio_uuid}")
                return False
        
        try:
            # Run the blocking operation in executor with timeout
            loop = asyncio.get_running_loop()
            success = await asyncio.wait_for(
                loop.run_in_executor(_MEMORY_EXECUTOR, _add_memory_to_store, transcript, client_id, audio_uuid, user_id, user_email),
                timeout=OLLAMA_TIMEOUT_SECONDS
            )
            if success:
                memory_logger.info(f"Added transcript for {audio_uuid} to mem0 (user: {user_email}, client: {client_id})")
            else:
                memory_logger.error(f"Failed to add memory for {audio_uuid}")
            return success
        except asyncio.TimeoutError:
            memory_logger.error(f"Memory addition timed out after {OLLAMA_TIMEOUT_SECONDS}s for {audio_uuid}")
            return False
        except Exception as e:
            memory_logger.error(f"Error adding memory for {audio_uuid}: {e}")
            return False
    
    # Action item methods removed - now handled by ActionItemsService
    # See action_items_service.py for the main action item processing logic
    
    # get_action_items method removed - now handled by ActionItemsService
    
    # update_action_item_status method removed - now handled by ActionItemsService
    
    # search_action_items method removed - now handled by ActionItemsService
    
    # search_action_items and delete_action_item methods removed - now handled by ActionItemsService

    def get_all_memories(self, user_id: str, limit: int = 100) -> list:
        """Get all memories for a user."""
        if not self._initialized:
            # This is a sync method, so we need to handle initialization differently
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we can't call initialize() directly
                # This should be handled by the caller
                raise Exception("Memory service not initialized - call await initialize() first")
            else:
                # We're in a sync context, run the async initialize
                loop.run_until_complete(self.initialize())
        
        assert self.memory is not None, "Memory service not initialized"
        try:
            memories_response = self.memory.get_all(user_id=user_id, limit=limit)
            
            # Handle different response formats from Mem0
            if isinstance(memories_response, dict):
                if "results" in memories_response:
                    # New paginated format - return the results list
                    return memories_response["results"]
                else:
                    # Old format - convert dict values to list
                    return list(memories_response.values()) if memories_response else []
            elif isinstance(memories_response, list):
                # Already a list
                return memories_response
            else:
                memory_logger.warning(f"Unexpected memory response format: {type(memories_response)}")
                return []
                
        except Exception as e:
            memory_logger.error(f"Error fetching memories for user {user_id}: {e}")
            raise
    
    def search_memories(self, query: str, user_id: str, limit: int = 10) -> list:
        """Search memories using semantic similarity."""
        if not self._initialized:
            # This is a sync method, so we need to handle initialization differently
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we can't call initialize() directly
                # This should be handled by the caller
                raise Exception("Memory service not initialized - call await initialize() first")
            else:
                # We're in a sync context, run the async initialize
                loop.run_until_complete(self.initialize())
        
        assert self.memory is not None, "Memory service not initialized"
        try:
            memories_response = self.memory.search(query=query, user_id=user_id, limit=limit)
            
            # Handle different response formats from Mem0
            if isinstance(memories_response, dict):
                if "results" in memories_response:
                    # New paginated format - return the results list
                    return memories_response["results"]
                else:
                    # Old format - convert dict values to list
                    return list(memories_response.values()) if memories_response else []
            elif isinstance(memories_response, list):
                # Already a list
                return memories_response
            else:
                memory_logger.warning(f"Unexpected search response format: {type(memories_response)}")
                return []
                
        except Exception as e:
            memory_logger.error(f"Error searching memories for user {user_id}: {e}")
            raise
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        if not self._initialized:
            # This is a sync method, so we need to handle initialization differently
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we can't call initialize() directly
                # This should be handled by the caller
                raise Exception("Memory service not initialized - call await initialize() first")
            else:
                # We're in a sync context, run the async initialize
                loop.run_until_complete(self.initialize())
        
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
            # This is a sync method, so we need to handle initialization differently
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we can't call initialize() directly
                # This should be handled by the caller
                raise Exception("Memory service not initialized - call await initialize() first")
            else:
                # We're in a sync context, run the async initialize
                loop.run_until_complete(self.initialize())
        
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
    
    async def test_connection(self) -> bool:
        """Test memory service connection with timeout protection."""
        try:
            if not self._initialized:
                await asyncio.wait_for(
                    self.initialize(),
                    timeout=MEMORY_INIT_TIMEOUT_SECONDS
                )
            return True
        except asyncio.TimeoutError:
            memory_logger.error(f"Memory service connection test timed out after {MEMORY_INIT_TIMEOUT_SECONDS}s")
            return False
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