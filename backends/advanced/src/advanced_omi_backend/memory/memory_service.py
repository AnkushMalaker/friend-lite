"""Memory service implementation for Omi-audio service.

This module provides:
- Memory configuration and initialization
- Memory operations (add, get, search, delete)
- Debug tracking and configurable extraction
"""

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from mem0 import AsyncMemory

# Import config loader
from advanced_omi_backend.memory_config_loader import get_config_loader
from advanced_omi_backend.users import User

# Using synchronous Memory from mem0 main branch
# The fixed main.py file is replaced during Docker build

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


def _parse_mem0_response(response, operation: str) -> list:
    """
    Parse mem0 response with explicit format handling based on mem0ai>=0.1.114 API.

    Args:
        response: Raw mem0 response from add/get_all/search operations
        operation: Operation name for error context ("add", "get_all", "search", "delete")

    Returns:
        list: Standardized list of memory objects with consistent format

    Raises:
        ValueError: Invalid/empty response or missing expected keys
        RuntimeError: Mem0 API error in response
        TypeError: Unexpected response format that cannot be handled

    Expected mem0 response formats:
        # add() - Returns single result or results array:
        {"results": [{"id": "...", "memory": "...", "metadata": {...}}]}
        OR {"id": "...", "memory": "...", "metadata": {...}}

        # get_all() - Returns paginated format or legacy dict:
        {"results": [{"id": "...", "memory": "...", ...}]}
        OR {"memory_id_1": {"memory": "...", ...}, "memory_id_2": {...}}

        # search() - Returns results array or direct list:
        {"results": [{"id": "...", "memory": "...", "score": 0.85, ...}]}
        OR [{"id": "...", "memory": "...", "score": 0.85}]
    """
    if not response:
        raise ValueError(f"Mem0 {operation} returned None/empty response")

    # Handle dict responses (most common format)
    if isinstance(response, dict):
        # Check for explicit error responses
        if "error" in response:
            raise RuntimeError(f"Mem0 {operation} error: {response['error']}")

        # NEW paginated format with results key (mem0ai>=0.1.114)
        if "results" in response:
            memory_logger.debug(
                f"Mem0 {operation} using paginated format with {len(response['results'])} results"
            )
            return response["results"]

        # Legacy format for get_all() - dict values are memory objects
        if operation == "get_all" and all(isinstance(v, dict) for v in response.values() if v):
            memory_logger.debug(
                f"Mem0 {operation} using legacy dict format with {len(response)} entries"
            )
            return list(response.values())

        # Single memory result (common for add operation)
        if "id" in response and "memory" in response:
            memory_logger.debug(f"Mem0 {operation} returned single memory object")
            return [response]

        # Check for single memory with different field names
        if "id" in response and any(key in response for key in ["text", "content"]):
            memory_logger.debug(
                f"Mem0 {operation} returned single memory with alternative field names"
            )
            return [response]

        # Unexpected dict format - provide helpful error
        available_keys = list(response.keys())
        raise ValueError(
            f"Mem0 {operation} returned dict without expected keys. Available keys: {available_keys}, Expected: 'results', 'id'+'memory', or memory dict values"
        )

    # Handle direct list responses (legacy/alternative format)
    if isinstance(response, list):
        memory_logger.debug(f"Mem0 {operation} returned direct list with {len(response)} items")
        return response

    # Handle single memory object (some edge cases)
    if hasattr(response, "get") and response.get("id"):
        memory_logger.debug(f"Mem0 {operation} returned single object with get method")
        return [response]

    # Handle primitive types that shouldn't happen
    if isinstance(response, (str, int, float, bool)):
        raise TypeError(f"Mem0 {operation} returned primitive type {type(response)}: {response}")

    # Completely unexpected format
    raise TypeError(f"Mem0 {operation} returned unexpected type {type(response)}: {response}")


def _extract_memory_ids(parsed_memories: list, audio_uuid: str) -> list:
    """
    Extract memory IDs from parsed memory objects.

    Args:
        parsed_memories: List of memory objects from _parse_mem0_response
        audio_uuid: Audio UUID for logging context

    Returns:
        list: List of extracted memory IDs
    """
    memory_ids = []
    for memory_item in parsed_memories:
        if isinstance(memory_item, dict):
            memory_id = memory_item.get("id")
            if memory_id:
                memory_ids.append(memory_id)
                memory_logger.info(f"Extracted memory ID: {memory_id} for {audio_uuid}")
            else:
                memory_logger.warning(
                    f"Memory item missing 'id' field for {audio_uuid}: {memory_item}"
                )
        else:
            memory_logger.warning(f"Non-dict memory item for {audio_uuid}: {memory_item}")

    return memory_ids


# Memory configuration - Optional for tracking/organization
MEM0_ORGANIZATION_ID = os.getenv("MEM0_ORGANIZATION_ID", "friend-lite-org")
MEM0_PROJECT_ID = os.getenv("MEM0_PROJECT_ID", "audio-conversations")
MEM0_APP_ID = os.getenv("MEM0_APP_ID", "omi-backend")

# Qdrant Configuration - Required for vector storage
QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL")

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

    # Get LLM provider from environment - required
    llm_provider = os.getenv("LLM_PROVIDER")
    if not llm_provider:
        raise ValueError(
            "LLM_PROVIDER environment variable is required. " "Set to 'openai' or 'ollama'"
        )
    llm_provider = llm_provider.lower()

    # Build LLM configuration based on provider using standard environment variables
    if llm_provider == "openai":
        # Get OpenAI API key - required for OpenAI provider
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required when using OpenAI provider"
            )

        # Get model from YAML config or environment variable
        model = llm_settings.get("model") or os.getenv("OPENAI_MODEL")
        if not model:
            raise ValueError(
                "Model must be specified either in memory_config.yaml or OPENAI_MODEL environment variable"
            )

        memory_logger.info(f"Using OpenAI provider with model: {model}")

        llm_config = {
            "provider": "openai",
            "config": {
                "model": model,
                "api_key": openai_api_key,
                "temperature": llm_settings.get(
                    "temperature", 0.1
                ),  # Default from YAML is acceptable
                "max_tokens": llm_settings.get(
                    "max_tokens", 2000
                ),  # Default from YAML is acceptable
            },
        }
        # NOTE: base_url not supported in current mem0 version for OpenAI provider
        # OpenAI provider always uses https://api.openai.com/v1
        # For OpenAI, use OpenAI embeddings
        # Note: embedder uses standard OpenAI API endpoint, base_url only applies to LLM
        # For OpenAI, use OpenAI embeddings - model can be configured via env var
        embedder_model = os.getenv("OPENAI_EMBEDDER_MODEL", "text-embedding-3-small")
        embedder_config = {
            "provider": "openai",
            "config": {
                "model": embedder_model,
                "embedding_dims": (
                    1536 if "small" in embedder_model else 3072
                ),  # Adjust based on model
                "api_key": openai_api_key,
            },
        }
        # NOTE: base_url not supported in embedder config for current mem0 version
        # Embedder will use standard OpenAI API endpoint: https://api.openai.com/v1
        embedding_dims = 1536
    elif llm_provider == "ollama":
        # Get Ollama base URL - required for Ollama provider
        ollama_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
        if not ollama_base_url:
            raise ValueError(
                "OPENAI_BASE_URL or OLLAMA_BASE_URL environment variable is required when using Ollama provider"
            )

        # Get model from YAML config or environment variable
        model = llm_settings.get("model") or os.getenv("OPENAI_MODEL")
        if not model:
            raise ValueError(
                "Model must be specified either in memory_config.yaml or OPENAI_MODEL environment variable"
            )

        memory_logger.info(f"Using Ollama provider with model: {model}")

        # Use OpenAI-compatible configuration for Ollama
        llm_config = {
            "provider": "openai",  # Use OpenAI provider for Ollama compatibility
            "config": {
                "model": model,
                "api_key": os.getenv("OPENAI_API_KEY", "dummy"),  # Ollama doesn't need real key
                "base_url": (
                    f"{ollama_base_url}/v1"
                    if not ollama_base_url.endswith("/v1")
                    else ollama_base_url
                ),
                "temperature": llm_settings.get(
                    "temperature", 0.1
                ),  # Default from YAML is acceptable
                "max_tokens": llm_settings.get(
                    "max_tokens", 2000
                ),  # Default from YAML is acceptable
            },
        }
        # For Ollama, use Ollama embeddings with OpenAI-compatible config
        # For Ollama, use Ollama embeddings - model can be configured via env var
        embedder_model = os.getenv("OLLAMA_EMBEDDER_MODEL", "nomic-embed-text:latest")
        embedder_config = {
            "provider": "ollama",
            "config": {
                "model": embedder_model,
                "embedding_dims": 768,  # Most Ollama embedders use 768
                "ollama_base_url": ollama_base_url.rstrip("/v1"),  # Remove /v1 suffix for embedder
            },
        }
        embedding_dims = 768
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    # Build Neo4j graph store configuration
    neo4j_config = None
    neo4j_host = os.getenv("NEO4J_HOST")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if neo4j_host and neo4j_user and neo4j_password:
        neo4j_config = {
            "provider": "neo4j",
            "config": {
                "url": f"bolt://{neo4j_host}:7687",
                "username": neo4j_user,
                "password": neo4j_password,
                "database": "neo4j",
            },
        }
        memory_logger.info(f"Neo4j graph store configured: {neo4j_host}")
    else:
        memory_logger.warning("Neo4j configuration incomplete - graph store disabled")

    # Valid mem0 configuration format based on official documentation
    # See: https://docs.mem0.ai/platform/quickstart and https://github.com/mem0ai/mem0
    mem0_config = {
        "llm": llm_config,
        "embedder": embedder_config,
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "omi_memories",
                "embedding_model_dims": embedding_dims,
                "host": QDRANT_BASE_URL or "qdrant",  # Fallback to service name for Docker
                "port": 6333,
            },
        },
        "version": "v1.1",
    }

    # Add graph store configuration if available
    if neo4j_config:
        mem0_config["graph_store"] = neo4j_config

    # Configure fact extraction based on YAML config
    fact_enabled = config_loader.is_fact_extraction_enabled()
    memory_logger.info(f"YAML fact extraction enabled: {fact_enabled}")

    # IMPORTANT: mem0 appears to require fact extraction to be enabled for memory creation to work
    # When fact extraction is disabled, mem0 skips memory creation entirely
    # This is a limitation of the mem0 library architecture
    if fact_enabled:
        # Use fact extraction prompt from configuration file
        fact_prompt = config_loader.get_fact_prompt()
        mem0_config["custom_fact_extraction_prompt"] = fact_prompt
        memory_logger.info(f"✅ Fact extraction enabled with config prompt")
        memory_logger.info(f"🔍 FULL FACT EXTRACTION PROMPT:")
        memory_logger.info(f"=== PROMPT START ===")
        memory_logger.info(fact_prompt)
        memory_logger.info(f"=== PROMPT END ===")
        memory_logger.info(f"Prompt length: {len(fact_prompt)} characters")
    else:
        memory_logger.warning(
            f"⚠️ Fact extraction disabled - this may prevent mem0 from creating memories due to library limitations"
        )

    memory_logger.debug(
        f"Final mem0_config: {json.dumps(_filter_sensitive_config_fields(mem0_config), indent=2)}"
    )
    return mem0_config


def _filter_sensitive_config_fields(config_value):
    """Filter sensitive fields from configuration values before logging."""
    if isinstance(config_value, dict):
        filtered = {}
        for key, value in config_value.items():
            # Filter out sensitive field names
            if key.lower() in [
                "api_key",
                "password",
                "token",
                "secret",
                "auth_token",
                "bearer_token",
            ]:
                filtered[key] = "***REDACTED***"
            else:
                filtered[key] = _filter_sensitive_config_fields(value)
        return filtered
    elif isinstance(config_value, list):
        return [_filter_sensitive_config_fields(item) for item in config_value]
    else:
        return config_value


# Global memory configuration - built dynamically from YAML config
MEM0_CONFIG = _build_mem0_config()


# Global instances
_memory_service = None
_process_memory = None  # For worker processes


def init_memory_config(
    qdrant_base_url: Optional[str] = None,
    organization_id: Optional[str] = None,
    project_id: Optional[str] = None,
    app_id: Optional[str] = None,
) -> dict:
    """Initialize and return memory configuration with optional overrides."""
    global MEM0_CONFIG, MEM0_ORGANIZATION_ID, MEM0_PROJECT_ID, MEM0_APP_ID

    memory_logger.info(f"Initializing MemoryService with Qdrant URL: {qdrant_base_url}")

    # Configuration updates would go here if needed

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
        # Log config in chunks to avoid truncation (filter sensitive fields)
        memory_logger.info("=== MEM0 CONFIG START ===")
        for key, value in config.items():
            filtered_value = _filter_sensitive_config_fields(value)
            memory_logger.info(f"  {key}: {json.dumps(filtered_value, indent=4)}")
        memory_logger.info("=== MEM0 CONFIG END ===")
        _process_memory = Memory.from_config(config)
    return _process_memory


def _add_memory_to_store(
    transcript: str,
    client_id: str,
    audio_uuid: str,
    user_id: str,
    user_email: str,
    allow_update: bool = False,
) -> tuple[bool, list[str]]:
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

    Returns:
        tuple: (success: bool, memory_ids: list[str])
    """
    start_time = time.time()
    created_memory_ids = []

    try:
        # Get configuration
        config_loader = get_config_loader()

        # # Check if transcript is empty or too short to be meaningful
        # MODIFIED: Reduced minimum length from 10 to 1 character to process almost all transcripts
        if not transcript or len(transcript.strip()) < 10:
            memory_logger.info(
                f"Skipping memory processing for {audio_uuid} - transcript completely empty: {len(transcript.strip()) if transcript else 0} chars"
            )
            return True, []  # Not an error, just skipped

        # Check if conversation should be skipped - BUT always process if we have any content
        # MODIFIED: Only skip if explicitly disabled, not based on quality control for short transcripts
        if config_loader.should_skip_conversation(transcript):
            # If transcript is very short (< 10 chars), force processing anyway to ensure all transcripts are stored
            if len(transcript.strip()) < 10:
                memory_logger.info(
                    f"Overriding quality control skip for short transcript {audio_uuid} - ensuring all transcripts are stored"
                )
            else:
                memory_logger.info(
                    f"Skipping memory processing for {audio_uuid} due to quality control"
                )
                return True, []  # Not an error, just skipped

        # Get memory extraction configuration
        memory_config = config_loader.get_memory_extraction_config()
        if not memory_config.get("enabled", True):
            memory_logger.info(f"Memory extraction disabled for {audio_uuid}")
            return True, []

        # Get or create the persistent memory instance for this process
        process_memory = _init_process_memory()

        # Use configured prompt or default
        prompt = memory_config.get(
            "prompt", "Please extract summary of the conversation - any topics or names"
        )

        # Get LLM settings for logging and testing
        llm_settings = memory_config.get("llm_settings", {})
        model_name = llm_settings.get("model", "gemma3n:e4b")

        # Add the memory with configured settings and error handling
        memory_logger.info(f"Adding memory for {audio_uuid} with prompt: {prompt[:100]}...")
        memory_logger.info(f"Transcript length: {len(transcript)} chars")
        memory_logger.info(f"Transcript preview: {transcript[:300]}...")

        # Additional validation - transcript quality has already been checked above
        memory_logger.info(f"Processing transcript with {len(transcript.strip())} characters")

        # Log LLM model being used
        memory_logger.info(f"Using LLM model: {model_name}")

        memory_logger.info(f"Starting mem0 processing for {audio_uuid}...")
        mem0_start_time = time.time()

        try:
            memory_logger.info(f"🔍 Now calling Mem0 with the same transcript...")

            # Log the mem0 configuration being used
            memory_logger.info(
                f"🔍 Mem0 config LLM provider: {MEM0_CONFIG.get('llm', {}).get('provider', 'unknown')}"
            )
            memory_logger.info(
                f"🔍 Mem0 config LLM model: {MEM0_CONFIG.get('llm', {}).get('config', {}).get('model', 'unknown')}"
            )
            memory_logger.info(
                f"🔍 Mem0 config custom prompt: {MEM0_CONFIG.get('custom_prompt', 'none')}"
            )
            memory_logger.info(
                f"🔍 Mem0 fact extraction disabled: {MEM0_CONFIG.get('custom_fact_extraction_prompt', 'not_set') == ''}"
            )

            # Log the exact parameters being passed to mem0
            metadata = {
                "source": "offline_streaming",
                "client_id": client_id,
                "user_email": user_email,
                "audio_uuid": audio_uuid,
                "timestamp": int(time.time()),
                "conversation_context": "audio_transcription",
                "device_type": "audio_recording",
                "organization_id": MEM0_ORGANIZATION_ID,
                "project_id": MEM0_PROJECT_ID,
                "app_id": MEM0_APP_ID,
                "extraction_method": "configurable",
                "config_enabled": True,
            }

            memory_logger.info(f"🔍 Mem0 add() parameters:")
            memory_logger.info(f"🔍   - transcript: {transcript}")
            memory_logger.info(f"🔍   - user_id: {user_id}")
            memory_logger.info(f"🔍   - metadata: {json.dumps(metadata, indent=2)}")
            memory_logger.info(f"🔍   - prompt: {prompt}")

            memory_logger.info(f"🧪 TEST_1: Calling with original prompt")

            # Try mem0.add() with retry logic for JSON errors
            try:
                result = process_memory.add(
                    transcript,
                    user_id=user_id,
                    metadata=metadata,
                    prompt=prompt,
                )
            except json.JSONDecodeError as json_error:
                memory_logger.warning(
                    f"JSON parsing error on first attempt for {audio_uuid}: {json_error}"
                )
                memory_logger.info(f"🔄 Retrying mem0.add() once for {audio_uuid}")
                try:
                    # Retry once with same parameters
                    result = process_memory.add(
                        transcript,
                        user_id=user_id,
                        metadata=metadata,
                        prompt=prompt,
                    )
                    memory_logger.info(f"✅ Retry successful for {audio_uuid}")
                except json.JSONDecodeError as retry_json_error:
                    memory_logger.error(
                        f"JSON parsing error on retry for {audio_uuid}: {retry_json_error}"
                    )
                    memory_logger.info(f"🔄 Falling back to infer=False for {audio_uuid}")
                    # Fallback to raw storage without LLM processing
                    result = process_memory.add(
                        transcript,
                        user_id=user_id,
                        metadata={
                            **metadata,
                            "storage_reason": "json_error_fallback",
                            "original_error": f"JSONDecodeError after retry: {str(retry_json_error)}",
                        },
                        infer=False,
                    )

            mem0_duration = time.time() - mem0_start_time
            memory_logger.info(f"Mem0 processing completed in {mem0_duration:.2f}s")
            memory_logger.info(
                f"Successfully added memory for {audio_uuid}, result type: {type(result)}"
            )

            # Log detailed memory result to understand what's being stored
            memory_logger.info(f"Raw mem0 result for {audio_uuid}: {result}")

            # Parse response using standardized parser
            try:
                parsed_memories = _parse_mem0_response(result, "add")
                created_memory_ids = _extract_memory_ids(parsed_memories, audio_uuid)

                # Check if mem0 returned empty results (this can be legitimate)
                if not parsed_memories:
                    memory_logger.info(
                        f"Mem0 returned empty results for {audio_uuid} - LLM determined no memorable content"
                    )

                    # Store using mem0 direct API without LLM processing
                    try:
                        direct_result = process_memory.add(
                            transcript,
                            user_id=user_id,
                            metadata={
                                "source": "offline_streaming",
                                "client_id": client_id,
                                "user_email": user_email,
                                "audio_uuid": audio_uuid,
                                "timestamp": int(time.time()),
                                "conversation_context": "audio_transcription",
                                "device_type": "audio_recording",
                                "organization_id": MEM0_ORGANIZATION_ID,
                                "project_id": MEM0_PROJECT_ID,
                                "app_id": MEM0_APP_ID,
                                "storage_reason": "empty_llm_results",
                                "original_error": "LLM returned no memorable content",
                                "processing_bypassed": True,
                            },
                            infer=False,
                        )
                        # Parse direct result using standardized parser
                        try:
                            direct_parsed = _parse_mem0_response(direct_result, "add")
                            direct_memory_ids = _extract_memory_ids(direct_parsed, audio_uuid)
                            if direct_memory_ids:
                                created_memory_ids.extend(direct_memory_ids)
                                memory_logger.info(
                                    f"Successfully stored direct memory for {audio_uuid} after empty LLM results"
                                )
                                result = direct_result  # Use the successful mem0 result
                            else:
                                memory_logger.warning(
                                    f"Direct memory storage returned no IDs for {audio_uuid}"
                                )
                        except (ValueError, RuntimeError, TypeError) as direct_parse_error:
                            memory_logger.warning(
                                f"Failed to parse direct memory result for {audio_uuid}: {direct_parse_error}"
                            )
                    except Exception as direct_error:
                        memory_logger.error(
                            f"Failed to store direct memory for {audio_uuid} after empty LLM results: {direct_error}"
                        )
                        # Continue with the empty results - this is legitimate when LLM finds no memorable content
            except (ValueError, RuntimeError, TypeError) as parse_error:
                memory_logger.error(
                    f"Failed to parse mem0 response for {audio_uuid}: {parse_error}"
                )
                # Re-raise to surface the actual parsing error instead of hiding it
                raise

            # Log details of created memories (we already parsed them above)
            if created_memory_ids:
                memory_count = len(created_memory_ids)
                memory_logger.info(f"Successfully created {memory_count} memories for {audio_uuid}")

                # Log details of each memory from parsed results
                try:
                    final_parsed = _parse_mem0_response(result, "add")
                    for i, memory_item in enumerate(final_parsed):
                        memory_id = memory_item.get("id", "unknown")
                        memory_text = memory_item.get("memory", "unknown")
                        event_type = memory_item.get("event", "unknown")
                        previous_memory = memory_item.get("previous_memory", None)
                        memory_logger.info(
                            f"Memory {i+1}: ID={memory_id[:8]}..., Event={event_type}, Text={memory_text[:80]}..."
                        )
                        if event_type == "UPDATE" and previous_memory:
                            memory_logger.warning(
                                f"UPDATE Event: Memory {memory_id[:8]} was updated from '{previous_memory[:50]}...' to '{memory_text[:50]}...'"
                            )
                except (ValueError, RuntimeError, TypeError) as detail_parse_error:
                    memory_logger.warning(
                        f"Could not parse result details for logging: {detail_parse_error}"
                    )

            # Log raw metadata for debugging
            if hasattr(result, "get"):
                memory_logger.info(f"Memory metadata: {result.get('metadata', {})}")

                # Check for other possible keys in result
                for key, value in result.items():
                    if key not in ["results", "id", "memory", "metadata"]:
                        memory_logger.info(f"Additional result key '{key}': {str(value)[:100]}...")

        except TimeoutError:
            # Handle timeout gracefully by using direct mem0 storage
            memory_logger.error(f"Timeout while adding memory for {audio_uuid}")

            try:
                # Store using mem0 direct API without LLM processing
                timeout_result = process_memory.add(
                    transcript,
                    user_id=user_id,
                    metadata={
                        "source": "offline_streaming",
                        "client_id": client_id,
                        "user_email": user_email,
                        "audio_uuid": audio_uuid,
                        "timestamp": int(time.time()),
                        "conversation_context": "audio_transcription",
                        "device_type": "audio_recording",
                        "organization_id": MEM0_ORGANIZATION_ID,
                        "project_id": MEM0_PROJECT_ID,
                        "app_id": MEM0_APP_ID,
                        "storage_reason": "timeout_fallback",
                        "original_error": "Timeout during LLM memory processing",
                        "processing_bypassed": True,
                    },
                    infer=False,
                )
                # Parse timeout result using standardized parser
                try:
                    timeout_parsed = _parse_mem0_response(timeout_result, "add")
                    timeout_memory_ids = _extract_memory_ids(timeout_parsed, audio_uuid)
                    if timeout_memory_ids:
                        created_memory_ids.extend(timeout_memory_ids)
                        memory_logger.info(
                            f"Successfully stored direct memory for {audio_uuid} after timeout"
                        )
                        result = timeout_result
                    else:
                        memory_logger.warning(
                            f"Timeout fallback returned no memory IDs for {audio_uuid}"
                        )
                except (ValueError, RuntimeError, TypeError) as timeout_parse_error:
                    memory_logger.warning(
                        f"Failed to parse timeout result for {audio_uuid}: {timeout_parse_error}"
                    )
                else:
                    memory_logger.error(
                        f"Direct memory storage failed for {audio_uuid} after timeout"
                    )
                    raise TimeoutError(f"Memory processing timeout for {audio_uuid}")
            except Exception as fallback_error:
                memory_logger.error(
                    f"Failed to store direct memory for {audio_uuid} after timeout: {fallback_error}"
                )
                raise TimeoutError(f"Memory processing timeout for {audio_uuid}")

        except Exception as error:
            # Handle other errors gracefully by using direct mem0 storage
            error_type = type(error).__name__
            memory_logger.error(f"Error while adding memory for {audio_uuid}: {error}")

            try:
                # Store using mem0 direct API without LLM processing
                error_result = process_memory.add(
                    transcript,
                    user_id=user_id,
                    metadata={
                        "source": "offline_streaming",
                        "client_id": client_id,
                        "user_email": user_email,
                        "audio_uuid": audio_uuid,
                        "timestamp": int(time.time()),
                        "conversation_context": "audio_transcription",
                        "device_type": "audio_recording",
                        "organization_id": MEM0_ORGANIZATION_ID,
                        "project_id": MEM0_PROJECT_ID,
                        "app_id": MEM0_APP_ID,
                        "storage_reason": "error_fallback",
                        "original_error": f"{error_type}: {str(error)}",
                        "processing_bypassed": True,
                    },
                    infer=False,
                )
                # Parse error fallback result using standardized parser
                try:
                    error_parsed = _parse_mem0_response(error_result, "add")
                    error_memory_ids = _extract_memory_ids(error_parsed, audio_uuid)
                    if error_memory_ids:
                        created_memory_ids.extend(error_memory_ids)
                        memory_logger.info(
                            f"Successfully stored direct memory for {audio_uuid} after error: {error_type}"
                        )
                        result = error_result
                    else:
                        memory_logger.warning(
                            f"Error fallback returned no memory IDs for {audio_uuid}"
                        )
                except (ValueError, RuntimeError, TypeError) as error_parse_error:
                    memory_logger.warning(
                        f"Failed to parse error fallback result for {audio_uuid}: {error_parse_error}"
                    )
                else:
                    memory_logger.error(
                        f"Direct memory storage failed for {audio_uuid} after error"
                    )
                    raise error  # Re-raise original error if direct storage fails
            except Exception as fallback_error:
                memory_logger.error(
                    f"Failed to store direct memory for {audio_uuid} after error: {fallback_error}"
                )
                raise error  # Re-raise original error if fallback fails

        # Record successful memory completion
        processing_time_ms = (time.time() - start_time) * 1000

        # Record the memory extraction for logging
        try:
            final_parsed = _parse_mem0_response(result, "add")
            memory_id = final_parsed[0].get("id", "unknown") if final_parsed else "unknown"
            memory_text = final_parsed[0].get("memory", "unknown") if final_parsed else "unknown"
        except (ValueError, RuntimeError, TypeError, IndexError):
            memory_id = str(result) if result else "unknown"
            memory_text = str(result) if result else "unknown"

        memory_logger.info(
            f"Successfully processed memory for {audio_uuid}, created {len(created_memory_ids)} memories: {created_memory_ids}"
        )
        return True, created_memory_ids

    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        memory_logger.error(f"Error adding memory for {audio_uuid}: {e}")

        return False, []


class MemoryService:
    """Service class for managing memory operations."""

    def __init__(self):
        self.memory = None
        self._initialized = False

    async def initialize(self):
        """Initialize the memory service using synchronous Memory (non-blocking lazy init)."""
        if self._initialized:
            return

        try:
            # Check LLM provider configuration for better error messages
            llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
            
            if llm_provider == "openai":
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is required when using OpenAI provider")
                memory_logger.info("Initializing Memory with OpenAI provider")
            elif llm_provider == "ollama":
                ollama_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
                if not ollama_base_url:
                    raise ValueError("OPENAI_BASE_URL or OLLAMA_BASE_URL environment variable is required when using Ollama provider")
                memory_logger.info(f"Initializing Memory with Ollama provider at {ollama_base_url}")
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")

            # Initialize AsyncMemory - auto-detects configuration from environment variables
            self.memory = AsyncMemory()
            self._initialized = True
            memory_logger.info("AsyncMemory initialized successfully (non-blocking)")

        except Exception as e:
            memory_logger.error(f"Failed to initialize AsyncMemory: {e}")
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
    ) -> tuple[bool, list[str]]:
        """Add memory in background process (non-blocking).

        Args:
            transcript: The conversation transcript
            client_id: The client ID that generated the audio
            audio_uuid: Unique identifier for the audio
            user_id: Database user ID to associate the memory with
            user_email: User email for identification
            allow_update: Whether to allow updating existing memories for this audio_uuid
            chunk_repo: ChunkRepo instance to update database relationships (optional)
        """
        if not self._initialized:
            try:
                await asyncio.wait_for(self.initialize(), timeout=MEMORY_INIT_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                memory_logger.error(f"Memory initialization timed out for {audio_uuid}")
                return False, []

        try:
            # Use async memory operations directly (no thread executor needed)
            success, created_memory_ids = await asyncio.wait_for(
                self._add_memory_async(
                    transcript,
                    client_id,
                    audio_uuid,
                    user_id,
                    user_email,
                    allow_update,
                ),
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            if success:
                memory_logger.info(
                    f"Added transcript for {audio_uuid} to mem0 (user: {user_email}, client: {client_id})"
                )
                # Update the database relationship if memories were created and chunk_repo is available
                if created_memory_ids and db_helper:
                    try:
                        for memory_id in created_memory_ids:
                            await db_helper.add_memory_reference(audio_uuid, memory_id, "created")
                            memory_logger.info(
                                f"Added memory reference {memory_id} to audio chunk {audio_uuid}"
                            )
                    except Exception as db_error:
                        memory_logger.error(
                            f"Failed to update database relationship for {audio_uuid}: {db_error}"
                        )
                        # Don't fail the entire operation if database update fails
                elif created_memory_ids and not db_helper:
                    memory_logger.warning(
                        f"Created memories {created_memory_ids} for {audio_uuid} but no chunk_repo provided to update database relationship"
                    )
            else:
                memory_logger.error(f"Failed to add memory for {audio_uuid}")
            return success, created_memory_ids
        except asyncio.TimeoutError:
            memory_logger.error(
                f"Memory addition timed out after {OLLAMA_TIMEOUT_SECONDS}s for {audio_uuid}"
            )
            return False, []
        except Exception as e:
            memory_logger.error(f"Error adding memory for {audio_uuid}: {e}")
            return False, []

    async def _add_memory_async(
        self,
        transcript: str,
        client_id: str,
        audio_uuid: str,
        user_id: str,
        user_email: str,
        allow_update: bool = False,
    ) -> tuple[bool, list[str]]:
        """
        Memory addition using synchronous Memory in background task.
        Converts the synchronous _add_memory_to_store logic to use async operations.
        """
        start_time = time.time()
        created_memory_ids = []

        try:
            # Get configuration
            config_loader = get_config_loader()

            # Check if transcript is empty or too short to be meaningful
            if not transcript or len(transcript.strip()) < 10:
                memory_logger.info(
                    f"Skipping memory processing for {audio_uuid} - transcript completely empty: {len(transcript.strip()) if transcript else 0} chars"
                )
                return True, []  # Not an error, just skipped

            # Check if conversation should be skipped
            if config_loader.should_skip_conversation(transcript):
                if len(transcript.strip()) < 10:
                    memory_logger.info(
                        f"Overriding quality control skip for short transcript {audio_uuid} - ensuring all transcripts are stored"
                    )
                else:
                    memory_logger.info(
                        f"Skipping memory processing for {audio_uuid} due to quality control"
                    )
                    return True, []  # Not an error, just skipped

            # Get memory extraction configuration
            memory_config = config_loader.get_memory_extraction_config()
            if not memory_config.get("enabled", True):
                memory_logger.info(f"Memory extraction disabled for {audio_uuid}")
                return True, []

            # Prepare metadata
            metadata = {
                "source": "offline_streaming",
                "client_id": client_id,
                "audio_uuid": audio_uuid,
                "user_email": user_email,
                "timestamp": int(time.time()),
            }

            # Use configured prompt or default
            fact_config = config_loader.get_fact_extraction_config()
            prompt = fact_config.get("custom_prompt") or "Extract important facts and insights from this conversation."

            memory_logger.info(f"🧪 Adding memory for {audio_uuid} using synchronous Memory")
            memory_logger.info(f"🔍   - transcript: {transcript[:100]}...")
            memory_logger.info(f"🔍   - metadata: {json.dumps(metadata, indent=2)}")
            memory_logger.info(f"🔍   - prompt: {prompt}")

            # Try async memory addition with retry logic for JSON errors
            try:
                result = await self.memory.add(
                    transcript,
                    user_id=user_id,
                    metadata=metadata,
                    prompt=prompt,
                )
            except Exception as json_error:
                memory_logger.warning(
                    f"Error on first attempt for {audio_uuid}: {json_error}"
                )
                memory_logger.info(f"🔄 Retrying Memory.add() once for {audio_uuid}")
                try:
                    # Retry once with same parameters
                    result = await self.memory.add(
                        transcript,
                        user_id=user_id,
                        metadata=metadata,
                        prompt=prompt,
                    )
                except Exception as retry_error:
                    memory_logger.error(
                        f"Error on retry for {audio_uuid}: {retry_error}"
                    )
                    memory_logger.info(f"🔄 Falling back to infer=False for {audio_uuid}")
                    # Fallback to raw storage without LLM processing
                    result = await self.memory.add(
                        transcript,
                        user_id=user_id,
                        metadata={
                            **metadata,
                            "storage_reason": "error_fallback",
                        },
                        infer=False,
                    )

            # Parse the result
            try:
                parsed_memories = _parse_mem0_response(result, "add")
                if parsed_memories:
                    created_memory_ids = _extract_memory_ids(parsed_memories, audio_uuid)
                    processing_time = time.time() - start_time
                    memory_logger.info(
                        f"✅ SUCCESS: Created {len(created_memory_ids)} memories for {audio_uuid} in {processing_time:.2f}s"
                    )
                    return True, created_memory_ids
                else:
                    memory_logger.warning(
                        f"Memory returned empty results for {audio_uuid} - LLM determined no memorable content"
                    )
                    
                    # Store using direct API without LLM processing
                    try:
                        direct_result = await self.memory.add(
                            transcript,
                            user_id=user_id,
                            metadata={
                                "source": "offline_streaming",
                                "client_id": client_id,
                                "audio_uuid": audio_uuid,
                                "user_email": user_email,
                                "timestamp": int(time.time()),
                                "storage_reason": "llm_no_memorable_content",
                            },
                            infer=False,
                        )
                        
                        direct_parsed = _parse_mem0_response(direct_result, "add")
                        if direct_parsed:
                            created_memory_ids = _extract_memory_ids(direct_parsed, audio_uuid)
                            processing_time = time.time() - start_time
                            memory_logger.info(
                                f"✅ FALLBACK SUCCESS: Stored {len(created_memory_ids)} raw memories for {audio_uuid} in {processing_time:.2f}s"
                            )
                            return True, created_memory_ids
                        else:
                            memory_logger.error(f"Failed to store even raw memory for {audio_uuid}")
                            return False, []
                    except Exception as direct_error:
                        memory_logger.error(f"Direct storage failed for {audio_uuid}: {direct_error}")
                        return False, []
                        
            except (ValueError, RuntimeError, TypeError) as parse_error:
                memory_logger.error(f"Failed to parse memory result for {audio_uuid}: {parse_error}")
                return False, []

        except Exception as error:
            error_type = type(error).__name__
            memory_logger.error(f"Error while adding memory for {audio_uuid}: {error}")
            return False, []

    async def get_all_memories(self, user_id: str, limit: int = 100) -> list:
        """Get all memories for a user, filtering and prioritizing semantic memories over fallback transcript memories."""
        if not self._initialized:
            await self.initialize()

        assert self.memory is not None, "Memory service not initialized"
        try:
            # Get more memories than requested to account for filtering
            fetch_limit = min(limit * 3, 500)  # Get up to 3x requested amount for filtering
            memories_response = await self.memory.get_all(user_id=user_id, limit=fetch_limit)

            # Parse response using standardized parser
            try:
                raw_memories = _parse_mem0_response(memories_response, "get_all")
            except (ValueError, RuntimeError, TypeError) as e:
                memory_logger.error(f"Failed to parse get_all response for user {user_id}: {e}")
                raise

            # Filter and prioritize memories
            semantic_memories = []
            fallback_memories = []

            for memory in raw_memories:
                metadata = memory.get("metadata", {})
                memory_id = memory.get("id", "")

                # Check if this is a fallback transcript memory
                is_fallback = (
                    metadata.get("empty_results") == True
                    or metadata.get("reason") == "llm_returned_empty_results"
                    or str(memory_id).startswith("transcript_")
                )

                if is_fallback:
                    fallback_memories.append(memory)
                else:
                    semantic_memories.append(memory)

            # Prioritize semantic memories, but include fallback if no semantic memories exist
            if semantic_memories:
                # Return semantic memories first, up to the limit
                result = semantic_memories[:limit]
                memory_logger.info(
                    f"Returning {len(result)} semantic memories for user {user_id} (filtered out {len(fallback_memories)} fallback memories)"
                )
            else:
                # If no semantic memories, return fallback memories
                result = fallback_memories[:limit]
                memory_logger.info(
                    f"No semantic memories found for user {user_id}, returning {len(result)} fallback memories"
                )

            return result

        except Exception as e:
            memory_logger.error(f"Error fetching memories for user {user_id}: {e}")
            raise

    async def get_all_memories_unfiltered(self, user_id: str, limit: int = 100) -> list:
        """Get all memories for a user without filtering fallback memories (for debugging)."""
        if not self._initialized:
            await self.initialize()

        assert self.memory is not None, "Memory service not initialized"
        try:
            memories_response = await self.memory.get_all(user_id=user_id, limit=limit)

            # Parse response using standardized parser
            try:
                return _parse_mem0_response(memories_response, "get_all")
            except (ValueError, RuntimeError, TypeError) as e:
                memory_logger.error(
                    f"Failed to parse get_all_unfiltered response for user {user_id}: {e}"
                )
                raise

        except Exception as e:
            memory_logger.error(f"Error fetching unfiltered memories for user {user_id}: {e}")
            raise

    async def search_memories(self, query: str, user_id: str, limit: int = 10) -> list:
        """Search memories using semantic similarity, prioritizing semantic memories over fallback."""
        if not self._initialized:
            await self.initialize()

        assert self.memory is not None, "Memory service not initialized"
        try:
            # Get more results than requested to account for filtering
            search_limit = min(limit * 3, 100)
            memories_response = await self.memory.search(query=query, user_id=user_id, limit=search_limit)

            # Parse response using standardized parser
            try:
                raw_memories = _parse_mem0_response(memories_response, "search")
            except (ValueError, RuntimeError, TypeError) as e:
                memory_logger.error(
                    f"Failed to parse search response for user {user_id}, query '{query}': {e}"
                )
                raise

            # Filter and prioritize memories
            semantic_memories = []
            fallback_memories = []

            for memory in raw_memories:
                metadata = memory.get("metadata", {})
                memory_id = memory.get("id", "")

                # Check if this is a fallback transcript memory
                is_fallback = (
                    metadata.get("empty_results") == True
                    or metadata.get("reason") == "llm_returned_empty_results"
                    or str(memory_id).startswith("transcript_")
                )

                if is_fallback:
                    fallback_memories.append(memory)
                else:
                    semantic_memories.append(memory)

            # Prioritize semantic memories in search results
            if semantic_memories:
                result = semantic_memories[:limit]
                memory_logger.info(
                    f"Search returned {len(result)} semantic memories for query '{query}' (filtered out {len(fallback_memories)} fallback memories)"
                )
            else:
                # If no semantic memories match, include fallback memories
                result = fallback_memories[:limit]
                memory_logger.info(
                    f"Search found no semantic memories for query '{query}', returning {len(result)} fallback memories"
                )

            return result

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

    async def get_all_memories_debug(self, limit: int = 200) -> list:
        """Get all memories across all users for admin debugging. Admin only."""
        if not self._initialized:
            await self.initialize()

        assert self.memory is not None, "Memory service not initialized"
        try:
            all_memories = []

            # Get all users from the database
            users = await User.find_all().to_list()
            memory_logger.info(f"🔍 Found {len(users)} users for admin debug")

            for user in users:
                user_id = str(user.id)
                try:
                    # Use the proper memory service method for each user
                    user_memories = await self.get_all_memories(user_id)

                    # Add user metadata to each memory for admin debugging
                    for memory in user_memories:
                        memory_text = memory.get("memory", "No content")
                        memory_logger.info(f"🔍 DEBUG memory structure: {memory}")
                        memory_logger.info(f"🔍 Memory text extracted: '{memory_text}'")

                        memory_entry = {
                            "id": memory.get("id", "unknown"),
                            "memory": memory_text,
                            "user_id": user_id,
                            "client_id": memory.get("metadata", {}).get("client_id", "unknown"),
                            "audio_uuid": memory.get("metadata", {}).get("audio_uuid", "unknown"),
                            "created_at": memory.get("created_at", "unknown"),
                            "owner_email": user.email,
                            "metadata": memory.get("metadata", {}),
                            "collection": "omi_memories",
                        }
                        all_memories.append(memory_entry)

                except Exception as e:
                    memory_logger.warning(f"Error getting memories for user {user_id}: {e}")
                    continue

                # Limit total memories returned
                if len(all_memories) >= limit:
                    break

            memory_logger.info(
                f"Retrieved {len(all_memories)} memories for admin debug view using proper memory service methods"
            )
            return all_memories[:limit]  # Ensure we don't exceed limit

        except Exception as e:
            memory_logger.error(f"Error fetching all memories for admin: {e}")
            # Re-raise to surface real errors instead of hiding them
            raise

    async def delete_all_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user and return count of deleted memories."""
        if not self._initialized:
            await self.initialize()

        try:
            assert self.memory is not None, "Memory service not initialized"
            # Get all memories first to count them
            user_memories_response = await self.memory.get_all(user_id=user_id)

            # Parse response using standardized parser to count memories
            try:
                user_memories = _parse_mem0_response(user_memories_response, "get_all")
                memory_count = len(user_memories)
            except (ValueError, RuntimeError, TypeError) as e:
                memory_logger.error(
                    f"Failed to parse get_all response for user {user_id} during delete: {e}"
                )
                # Continue with deletion attempt even if count failed
                memory_count = 0

            # Delete all memories for this user
            if memory_count > 0:
                await self.memory.delete_all(user_id=user_id)
                memory_logger.info(f"Deleted {memory_count} memories for user {user_id}")

            return memory_count

        except Exception as e:
            memory_logger.error(f"Error deleting memories for user {user_id}: {e}")
            raise

    async def test_connection(self) -> bool:
        """Test memory service connection with timeout protection."""
        try:
            if not self._initialized:
                await asyncio.wait_for(self.initialize(), timeout=MEMORY_INIT_TIMEOUT_SECONDS)
            return True
        except asyncio.TimeoutError:
            memory_logger.error(
                f"Memory service connection test timed out after {MEMORY_INIT_TIMEOUT_SECONDS}s"
            )
            return False
        except Exception as e:
            memory_logger.error(f"Memory service connection test failed: {e}")
            return False

    def shutdown(self):
        """Shutdown the memory service."""
        self._initialized = False
        memory_logger.info("Memory service shut down")

    async def get_memories_with_transcripts(self, user_id: str, limit: int = 100) -> list:
        """Get memories with their source transcripts using database relationship."""
        if not self._initialized:
            await self.initialize()

        assert self.memory is not None, "Memory service not initialized"

        try:
            # Get all memories for the user
            memories = await self.get_all_memories(user_id, limit)

            # Import Motor connection here to avoid circular imports
            from advanced_omi_backend.database import chunks_col

            # PERFORMANCE OPTIMIZATION: Extract all audio_uuids first for bulk query
            audio_uuids = []
            for memory in memories:
                metadata = memory.get("metadata", {})
                audio_uuid = metadata.get("audio_uuid")
                if audio_uuid:
                    audio_uuids.append(audio_uuid)

            # Bulk query for all chunks at once instead of individual queries
            memory_logger.debug(f"🔍 Bulk lookup for {len(audio_uuids)} audio UUIDs")
            chunks_cursor = chunks_col.find({"audio_uuid": {"$in": audio_uuids}})
            chunks_by_uuid = {}
            async for chunk in chunks_cursor:
                chunks_by_uuid[chunk["audio_uuid"]] = chunk
            memory_logger.debug(f"✅ Found {len(chunks_by_uuid)} chunks in bulk query")

            enriched_memories = []

            for memory in memories:
                # Create enriched memory entry
                enriched_memory = {
                    "memory_id": memory.get("id", "unknown"),
                    "memory_text": memory.get("memory", memory.get("text", "")),
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

                    # Get transcript from bulk-loaded chunks (PERFORMANCE OPTIMIZED)
                    chunk = chunks_by_uuid.get(audio_uuid)
                    if chunk:
                        memory_logger.debug(
                            f"🔍 Found chunk for {audio_uuid}, extracting transcript segments"
                        )
                        # Extract transcript from chunk
                        transcript_segments = chunk.get("transcript", [])
                        if transcript_segments:
                            # Combine all transcript segments into a single text
                            full_transcript = " ".join(
                                [
                                    segment.get("text", "")
                                    for segment in transcript_segments
                                    if isinstance(segment, dict) and segment.get("text")
                                ]
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
                                memory_logger.debug(
                                    f"✅ Successfully enriched memory {audio_uuid} with {len(full_transcript)} char transcript"
                                )
                            else:
                                memory_logger.debug(f"⚠️ Empty transcript found for {audio_uuid}")
                        else:
                            memory_logger.debug(f"⚠️ No transcript segments found for {audio_uuid}")
                    else:
                        memory_logger.debug(f"⚠️ No chunk found for audio_uuid: {audio_uuid}")

                enriched_memories.append(enriched_memory)

            transcript_count = sum(1 for m in enriched_memories if m.get("transcript"))
            memory_logger.info(
                f"Enriched {len(enriched_memories)} memories with transcripts for user {user_id} ({transcript_count} with actual transcript data)"
            )
            return enriched_memories

        except Exception as e:
            memory_logger.error(f"Error getting memories with transcripts for user {user_id}: {e}")
            raise


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
