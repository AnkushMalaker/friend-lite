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
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from mem0 import Memory

# Import debug tracker and config loader
from memory_debug import get_debug_tracker
from memory_config_loader import get_config_loader

# Configure Mem0 telemetry based on environment variable
# Set default to False for privacy unless explicitly enabled
if not os.getenv("MEM0_TELEMETRY"):
    os.environ["MEM0_TELEMETRY"] = "False"

# Enable detailed mem0 logging to capture LLM responses
mem0_logger = logging.getLogger("mem0")
mem0_logger.setLevel(logging.DEBUG)

# Also enable detailed ollama client logging
ollama_logger = logging.getLogger("ollama")
ollama_logger.setLevel(logging.DEBUG)

# Enable httpx logging to see raw HTTP requests/responses to Ollama
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.DEBUG)

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

def _build_mem0_config() -> dict:
    """Build Mem0 configuration from YAML config and environment variables."""
    config_loader = get_config_loader()
    memory_config = config_loader.get_memory_extraction_config()
    fact_config = config_loader.get_fact_extraction_config()
    llm_settings = memory_config.get("llm_settings", {})
    
    # Get LLM provider from environment or config
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    # Build LLM configuration based on provider
    if llm_provider == "openai":
        llm_config = {
            "provider": "openai",
            "config": {
                "model": llm_settings.get("model", "gpt-4o-mini"),
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": llm_settings.get("temperature", 0.1),
                "max_tokens": llm_settings.get("max_tokens", 2000),
            },
        }
        # For OpenAI, use OpenAI embeddings
        embedder_config = {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "embedding_dims": 1536,
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        }
        embedding_dims = 1536
    else:  # Default to ollama
        llm_config = {
            "provider": "ollama",
            "config": {
                "model": llm_settings.get("model", "gemma3n:e4b"),
                "ollama_base_url": OLLAMA_BASE_URL,
                "temperature": llm_settings.get("temperature", 0.1),
                "max_tokens": llm_settings.get("max_tokens", 2000),
            },
        }
        # For Ollama, use Ollama embeddings
        embedder_config = {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest",
                "embedding_dims": 768,
                "ollama_base_url": OLLAMA_BASE_URL,
            },
        }
        embedding_dims = 768
    
    mem0_config = {
        "llm": llm_config,
        "embedder": embedder_config,
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "omi_memories",
                "embedding_model_dims": embedding_dims,
                "host": QDRANT_BASE_URL,
                "port": 6333,
            },
        },
        "custom_prompt": memory_config.get("prompt", 
            "Extract anything relevant about this conversation. "
            "Anything from what the conversation was about, the people involved, emotion, etc. In each memory, include: No calls mentioned if no call was mentioned."
        ),
    }
    
    # Configure fact extraction based on YAML config
    fact_enabled = config_loader.is_fact_extraction_enabled()
    memory_logger.info(f"Fact extraction enabled: {fact_enabled}")
    
    if fact_enabled:
        fact_prompt = fact_config.get("prompt", "Extract specific facts from this conversation.")
        mem0_config["custom_fact_extraction_prompt"] = fact_prompt
        memory_logger.info(f"Fact extraction enabled with prompt: {fact_prompt[:50]}...")
    else:
        # Disable fact extraction completely - multiple approaches
        mem0_config["custom_fact_extraction_prompt"] = ""
        mem0_config["fact_retrieval"] = False  # Disable fact retrieval
        mem0_config["enable_fact_extraction"] = False  # Explicit disable
        memory_logger.info("Fact extraction disabled - empty prompt and flags set")
    
    memory_logger.debug(f"Final mem0_config: {json.dumps(mem0_config, indent=2)}")
    return mem0_config

# Global memory configuration - built dynamically from YAML config
MEM0_CONFIG = _build_mem0_config()

# Action item extraction is now handled by ActionItemsService
# using configuration from memory_config.yaml

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
        # Build fresh config to ensure we get latest YAML settings
        config = _build_mem0_config()
        # Log config in chunks to avoid truncation
        memory_logger.info("=== MEM0 CONFIG START ===")
        for key, value in config.items():
            memory_logger.info(f"  {key}: {json.dumps(value, indent=4)}")
        memory_logger.info("=== MEM0 CONFIG END ===")
        _process_memory = Memory.from_config(config)
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
        
        # Get LLM settings for logging and testing
        llm_settings = memory_config.get("llm_settings", {})
        model_name = llm_settings.get('model', 'gemma3n:e4b')
        
        # Add the memory with configured settings and error handling
        memory_logger.info(f"Adding memory for {audio_uuid} with prompt: {prompt[:100]}...")
        memory_logger.info(f"Transcript length: {len(transcript)} chars")
        memory_logger.info(f"Transcript preview: {transcript[:300]}...")
        
        # Check if transcript meets quality control
        if len(transcript.strip()) < 10:
            memory_logger.warning(f"Very short transcript for {audio_uuid}: '{transcript}'")
        
        
        # Log LLM model being used
        memory_logger.info(f"Using LLM model: {model_name}")
        
        # Test LLM directly before mem0 processing
        try:
            import ollama
            test_prompt = f"{prompt}\n\nConversation:\n{transcript[:500]}..."
            memory_logger.info(f"Testing LLM directly with prompt: {test_prompt[:200]}...")
            
            # Use the same Ollama URL as configured for mem0
            client = ollama.Client(host=OLLAMA_BASE_URL)
            response = client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': test_prompt}]
            )
            
            raw_response = response.get('message', {}).get('content', 'No content')
            memory_logger.info(f"Raw LLM response: {raw_response}")
            memory_logger.info(f"LLM response length: {len(raw_response)} chars")
            
            # Log the full response to see what gemma3n:e4b is generating
            memory_logger.debug(f"Full LLM response: {raw_response}")
            
        except Exception as llm_test_error:
            memory_logger.error(f"Direct LLM test failed: {llm_test_error}")
        
        memory_logger.info(f"Starting mem0 processing for {audio_uuid}...")
        mem0_start_time = time.time()
        
        try:
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
            
            mem0_duration = time.time() - mem0_start_time
            memory_logger.info(f"Mem0 processing completed in {mem0_duration:.2f}s")
            memory_logger.info(f"Successfully added memory for {audio_uuid}, result type: {type(result)}")
            
            # Log detailed memory result to understand what's being stored
            memory_logger.info(f"Raw mem0 result for {audio_uuid}: {result}")
            memory_logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
            
            # Check if mem0 returned empty results
            if isinstance(result, dict) and result.get('results') == []:
                memory_logger.error(f"Mem0 returned empty results for {audio_uuid} - LLM may not be generating memories")
                raise Exception(f"Empty results from mem0 - LLM '{model_name}' returned no memories")
            
            if isinstance(result, dict):
                results_list = result.get('results', [])
                if results_list:
                    memory_count = len(results_list)
                    memory_logger.info(f"Successfully created {memory_count} memories for {audio_uuid}")
                    
                    # Log details of each memory
                    for i, memory_item in enumerate(results_list):
                        memory_id = memory_item.get('id', 'unknown')
                        memory_text = memory_item.get('memory', 'unknown')
                        event_type = memory_item.get('event', 'unknown')
                        memory_logger.info(f"Memory {i+1}: ID={memory_id[:8]}..., Event={event_type}, Text={memory_text[:80]}...")
                else:
                    # Check for old format (direct id/memory keys)
                    memory_id = result.get("id", result.get("memory_id", "unknown"))
                    memory_text = result.get("memory", result.get("text", result.get("content", "unknown")))
                    memory_logger.info(f"Single memory - ID: {memory_id}, Text: {memory_text[:100] if isinstance(memory_text, str) else memory_text}...")
                
                memory_logger.info(f"Memory metadata: {result.get('metadata', {})}")
                
                # Check for other possible keys in result
                for key, value in result.items():
                    if key not in ['results', 'id', 'memory', 'metadata']:
                        memory_logger.info(f"Additional result key '{key}': {str(value)[:100]}...")
            else:
                memory_logger.info(f"Memory result (non-dict): {str(result)[:200]}...")
            
            memory_logger.debug(f"Full memory result for {audio_uuid}: {result}")
        except (json.JSONDecodeError, Exception) as error:
            # Handle JSON parsing errors and other mem0 errors
            error_msg = str(error)
            memory_logger.error(f"Mem0 error for {audio_uuid}: {error} (type: {type(error)})")
            
            if "UNIQUE constraint failed" in error_msg:
                memory_logger.error(f"Database constraint error for {audio_uuid}: {error}")
                error_type = "database_constraint_error"
            elif "Empty results from mem0" in error_msg:
                memory_logger.error(f"LLM returned empty results for {audio_uuid}: {error}")
                error_type = "empty_llm_results"
            elif "Expecting ':' delimiter" in error_msg or "JSONDecodeError" in str(type(error)) or "Unterminated string" in error_msg:
                memory_logger.error(f"JSON parsing error in mem0 for {audio_uuid}: {error}")
                error_type = "json_parsing_error"
            elif "'facts'" in error_msg:
                memory_logger.error(f"Fact extraction error (should be disabled) for {audio_uuid}: {error}")
                error_type = "fact_extraction_error"
            else:
                memory_logger.error(f"General mem0 processing error for {audio_uuid}: {error}")
                error_type = "mem0_processing_error"
            
            # Create a fallback memory entry 
            try:
                # Store the transcript as a basic memory without using mem0
                result = {
                    "id": f"fallback_{audio_uuid}_{int(time.time())}",
                    "memory": f"Conversation summary: {transcript[:500]}{'...' if len(transcript) > 500 else ''}",
                    "metadata": {
                        "fallback_reason": error_type,
                        "original_error": str(error),
                        "audio_uuid": audio_uuid,
                        "client_id": client_id,
                        "user_email": user_email,
                        "timestamp": int(time.time()),
                        "mem0_bypassed": True
                    }
                }
                memory_logger.warning(f"Created fallback memory for {audio_uuid} due to mem0 error: {error_type}")
            except Exception as fallback_error:
                memory_logger.error(f"Failed to create fallback memory for {audio_uuid}: {fallback_error}")
                raise error  # Re-raise original error if fallback fails
        
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
            # Build fresh config to ensure we get latest YAML settings
            config = _build_mem0_config()
            self.memory = await asyncio.wait_for(
                loop.run_in_executor(_MEMORY_EXECUTOR, Memory.from_config, config),
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