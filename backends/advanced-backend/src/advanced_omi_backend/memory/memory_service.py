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

from mem0 import Memory

# Import debug tracker and config loader
from advanced_omi_backend.debug_system_tracker import PipelineStage, get_debug_tracker
from advanced_omi_backend.memory_config_loader import get_config_loader
from advanced_omi_backend.users import User

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
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

    # Build LLM configuration based on provider using standard environment variables
    if llm_provider == "openai":
        # Use standard OPENAI_MODEL environment variable
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")

        # Allow YAML config to override environment variable
        model = llm_settings.get("model", openai_model)

        memory_logger.info(f"Using OpenAI provider with model: {model}")

        llm_config = {
            "provider": "openai",
            "config": {
                "model": model,
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": llm_settings.get("temperature", 0.1),
                "max_tokens": llm_settings.get("max_tokens", 2000),
            },
        }
        # Only add base_url if it's set
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if openai_base_url:
            llm_config["config"]["base_url"] = openai_base_url
        # For OpenAI, use OpenAI embeddings
        embedder_config = {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "embedding_dims": 1536,
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        }
        # Only add base_url if it's set
        if openai_base_url:
            embedder_config["config"]["base_url"] = openai_base_url
        embedding_dims = 1536
    elif llm_provider == "ollama":
        # Use standard OPENAI_MODEL environment variable (Ollama as OpenAI-compatible)
        ollama_model = os.getenv("OPENAI_MODEL", "llama3.1:latest")

        # Allow YAML config to override environment variable
        model = llm_settings.get("model", ollama_model)

        memory_logger.info(f"Using Ollama provider with model: {model}")

        # Use OpenAI-compatible configuration for Ollama
        llm_config = {
            "provider": "openai",  # Use OpenAI provider for Ollama compatibility
            "config": {
                "model": model,
                "api_key": os.getenv("OPENAI_API_KEY", "dummy"),  # Ollama doesn't need real key
                "base_url": os.getenv("OPENAI_BASE_URL", "http://ollama:11434/v1"),
                "temperature": llm_settings.get("temperature", 0.1),
                "max_tokens": llm_settings.get("max_tokens", 2000),
            },
        }
        # For Ollama, use Ollama embeddings with OpenAI-compatible config
        embedder_config = {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest",
                "embedding_dims": 768,
                "ollama_base_url": os.getenv("OPENAI_BASE_URL", "http://ollama:11434"),
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
                "host": QDRANT_BASE_URL,
                "port": 6333,
            },
        },
        "version": "v1.1",
    }

    # Add graph store configuration if available
    if neo4j_config:
        mem0_config["graph_store"] = neo4j_config

    # Configure fact extraction - ALWAYS ENABLE for proper memory creation
    fact_enabled = config_loader.is_fact_extraction_enabled()
    memory_logger.info(f"YAML fact extraction enabled: {fact_enabled}")

    # FORCE ENABLE fact extraction with working prompt format - UPDATED for more inclusive extraction
    # Using custom_fact_extraction_prompt as documented in mem0 repo: https://github.com/mem0ai/mem0
    #     formatted_fact_prompt = """
    # Please extract ALL relevant facts from the conversation, including topics discussed, activities mentioned, people referenced, emotions expressed, and any other notable details.
    # Extract granular, specific facts rather than broad summaries. Be inclusive and extract multiple facts even from casual conversations.

    # Here are some few shot examples:

    # Input: Hi.
    # Output: {"facts" : ["Greeting exchanged"]}

    # Input: I need to buy groceries tomorrow.
    # Output: {"facts" : ["Need to buy groceries tomorrow", "Shopping task mentioned", "Time reference to tomorrow"]}

    # Input: The meeting is at 3 PM on Friday.
    # Output: {"facts" : ["Meeting scheduled for 3 PM on Friday", "Business meeting mentioned", "Specific time commitment", "Friday scheduling"]}

    # Input: We are talking about unicorns.
    # Output: {"facts" : ["Conversation about unicorns", "Fantasy topic discussed", "Mythical creatures mentioned"]}

    # Input: My alarm keeps ringing.
    # Output: {"facts" : ["Alarm is ringing", "Audio disturbance mentioned", "Repetitive sound issue", "Device malfunction or setting"]}

    # Input: Bro, he just did it for the funny. Every move does not need to be perfect.
    # Output: {"facts" : ["Gaming strategy discussed", "Casual conversation with friend", "Philosophy about game moves", "Humorous game action mentioned", "Perfectionism topic", "Gaming advice given"]}

    # Now extract facts from the following conversation. Return only JSON format with "facts" key. Be thorough and extract multiple specific facts. ALWAYS extract at least one fact unless the input is completely empty or meaningless.
    # """
    #     mem0_config["custom_fact_extraction_prompt"] = formatted_fact_prompt
    memory_logger.info(f"✅ FORCED fact extraction enabled with working JSON prompt format")

    memory_logger.debug(f"Final mem0_config: {json.dumps(mem0_config, indent=2)}")
    return mem0_config


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
        # Log config in chunks to avoid truncation
        memory_logger.info("=== MEM0 CONFIG START ===")
        for key, value in config.items():
            memory_logger.info(f"  {key}: {json.dumps(value, indent=4)}")
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
        # Get configuration and debug tracker
        config_loader = get_config_loader()
        debug_tracker = get_debug_tracker()

        # Create a transaction for memory processing tracking
        transaction_id = debug_tracker.create_transaction(
            user_id=user_id,
            client_id=client_id,
            conversation_id=audio_uuid,  # Use audio_uuid as conversation_id
        )

        # Start memory processing stage
        debug_tracker.track_event(
            transaction_id,
            PipelineStage.MEMORY_STARTED,
            True,
            transcript_length=len(transcript) if transcript else 0,
            user_email=user_email,
            audio_uuid=audio_uuid,
        )

        # Check if transcript is empty or too short to be meaningful
        # MODIFIED: Reduced minimum length from 10 to 1 character to process almost all transcripts
        if not transcript or len(transcript.strip()) < 1:
            debug_tracker.track_event(
                transaction_id,
                PipelineStage.MEMORY_COMPLETED,
                False,
                error_message=f"Transcript empty: {len(transcript.strip()) if transcript else 0} chars",
            )
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
                debug_tracker.track_event(
                    transaction_id,
                    PipelineStage.MEMORY_COMPLETED,
                    False,
                    error_message="Conversation skipped due to quality control",
                )
                memory_logger.info(
                    f"Skipping memory processing for {audio_uuid} due to quality control"
                )
                return True, []  # Not an error, just skipped

        # Get memory extraction configuration
        memory_config = config_loader.get_memory_extraction_config()
        if not memory_config.get("enabled", True):
            debug_tracker.track_event(
                transaction_id,
                PipelineStage.MEMORY_COMPLETED,
                False,
                error_message="Memory extraction disabled",
            )
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

        # DEBUGGING: Test OpenAI directly before Mem0 call
        memory_logger.info(f"🔍 DEBUGGING: Testing OpenAI connection directly...")
        try:
            import os

            import openai

            openai_api_key = os.getenv("OPENAI_API_KEY")
            llm_provider = os.getenv("LLM_PROVIDER", "").lower()
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")

            memory_logger.info(f"🔍 OpenAI API Key present: {bool(openai_api_key)}")
            memory_logger.info(f"🔍 LLM Provider: {llm_provider}")
            memory_logger.info(f"🔍 OpenAI Model: {openai_model}")
            memory_logger.info(f"🔍 Full prompt being sent: {prompt}")
            memory_logger.info(f"🔍 Full transcript being processed: {transcript}")

            if llm_provider == "openai" and openai_api_key:
                # Test direct OpenAI call with same system prompt mem0 uses
                client = openai.OpenAI(api_key=openai_api_key)

                # Try the exact same call that mem0 would make for memory extraction
                memory_extraction_prompt = f"""
                You are an expert at extracting memories from conversations.

                Instructions:
                1. Extract key facts, topics, and insights from the conversation
                2. Focus on memorable information that could be useful later
                3. Include names, places, events, preferences, and important details
                4. Format as clear, concise memories
                5. If the conversation contains meaningful content, always extract something

                Custom prompt: {prompt}

                Extract memories from this conversation:
                """

                test_response = client.chat.completions.create(
                    model=openai_model,
                    messages=[
                        {"role": "system", "content": memory_extraction_prompt},
                        {"role": "user", "content": transcript},
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                )

                response_content = test_response.choices[0].message.content
                memory_logger.info(f"🔍 DIRECT OpenAI Response: {response_content}")
                memory_logger.info(f"🔍 OpenAI Response Usage: {test_response.usage}")
                memory_logger.info(
                    f"🔍 Response Length: {len(response_content) if response_content else 0} chars"
                )

                # Also test with a simpler prompt to see if it's a prompt issue
                simple_response = client.chat.completions.create(
                    model=openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Extract key information from this conversation as bullet points:",
                        },
                        {"role": "user", "content": transcript},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )

                simple_content = simple_response.choices[0].message.content
                memory_logger.info(f"🔍 SIMPLE OpenAI Response: {simple_content}")

            else:
                memory_logger.warning(f"🔍 OpenAI not configured properly for direct test")

        except Exception as e:
            memory_logger.error(f"🔍 Direct OpenAI test failed: {e}")

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

            result = process_memory.add(
                transcript,
                user_id=user_id,  # Use database user_id instead of client_id
                metadata=metadata,
                prompt=prompt,
            )

            mem0_duration = time.time() - mem0_start_time
            memory_logger.info(f"Mem0 processing completed in {mem0_duration:.2f}s")
            memory_logger.info(
                f"Successfully added memory for {audio_uuid}, result type: {type(result)}"
            )

            # Log detailed memory result to understand what's being stored
            memory_logger.info(f"Raw mem0 result for {audio_uuid}: {result}")
            memory_logger.info(
                f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}"
            )

            # Extract memory IDs from the result
            if isinstance(result, dict):
                # Check for multiple memories in results list
                results_list = result.get("results", [])
                if results_list:
                    for memory_item in results_list:
                        memory_id = memory_item.get("id")
                        if memory_id:
                            created_memory_ids.append(memory_id)
                            memory_logger.info(f"Extracted memory ID: {memory_id}")
                else:
                    # Check for single memory (old format or fallback)
                    memory_id = result.get("id")
                    if memory_id:
                        created_memory_ids.append(memory_id)
                        memory_logger.info(f"Extracted single memory ID: {memory_id}")

            # Check if mem0 returned empty results (this can be legitimate)
            if isinstance(result, dict) and result.get("results") == []:
                memory_logger.info(
                    f"Mem0 returned empty results for {audio_uuid} - LLM determined no memorable content"
                )
                # Create a minimal tracking entry for debugging purposes
                # MODIFIED: Enhanced to create a proper memory entry that will be visible in UI
                import uuid

                unique_suffix = str(uuid.uuid4())[:8]

                # Create a more descriptive memory entry for transcripts without memorable content
                memory_text = f"Conversation transcript: {transcript}"
                if len(memory_text) > 200:
                    memory_text = f"Conversation transcript: {transcript[:180]}... (truncated)"

                fallback_memory_id = (
                    f"transcript_{audio_uuid}_{int(time.time() * 1000)}_{unique_suffix}"
                )
                created_memory_ids.append(fallback_memory_id)

                result = {
                    "id": fallback_memory_id,
                    "memory": memory_text,
                    "user_id": user_id,  # Ensure user_id is included for proper retrieval
                    "metadata": {
                        "empty_results": True,
                        "audio_uuid": audio_uuid,
                        "client_id": client_id,
                        "user_email": user_email,
                        "timestamp": int(time.time()),
                        "llm_model": model_name,
                        "reason": "llm_returned_empty_results",
                        "source": "offline_streaming",
                        "conversation_context": "audio_transcription",
                        "device_type": "audio_recording",
                        "organization_id": MEM0_ORGANIZATION_ID,
                        "project_id": MEM0_PROJECT_ID,
                        "app_id": MEM0_APP_ID,
                        "full_transcript": transcript,  # Store full transcript for reference
                        "transcript_length": len(transcript),
                        "processing_forced": True,  # Indicate this was processed despite empty results
                    },
                    "results": [],  # Keep the original empty results for consistency
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                }
                memory_logger.info(
                    f"Created enhanced memory entry for transcript without memorable content: {result['id']}"
                )

                # Also try to store this in the actual mem0 system as a basic memory
                try:
                    # Create a simple memory entry that mem0 can store
                    fallback_result = process_memory.add(
                        f"Transcript recorded: {transcript[:100]}{'...' if len(transcript) > 100 else ''}",
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
                            "forced_storage": True,
                            "original_transcript": transcript,
                            "processing_reason": "ensure_all_transcripts_stored",
                        },
                        prompt="Store this transcript as a basic memory entry.",
                    )
                    if fallback_result and isinstance(fallback_result, dict):
                        fallback_memory_id = fallback_result.get("id")
                        if fallback_memory_id and fallback_memory_id not in created_memory_ids:
                            created_memory_ids.append(fallback_memory_id)
                        memory_logger.info(
                            f"Successfully stored fallback memory entry for {audio_uuid}"
                        )
                        result = fallback_result  # Use the successful mem0 result
                    else:
                        memory_logger.info(
                            f"Fallback memory storage failed, using tracking entry for {audio_uuid}"
                        )
                except Exception as fallback_error:
                    memory_logger.warning(
                        f"Failed to store fallback memory for {audio_uuid}: {fallback_error}"
                    )
                    # Continue with the tracking entry we created above

            if isinstance(result, dict):
                results_list = result.get("results", [])
                if results_list:
                    memory_count = len(results_list)
                    memory_logger.info(
                        f"Successfully created {memory_count} memories for {audio_uuid}"
                    )

                    # Log details of each memory
                    for i, memory_item in enumerate(results_list):
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
                else:
                    # Check for old format (direct id/memory keys)
                    memory_id = result.get("id", result.get("memory_id", "unknown"))
                    memory_text = result.get(
                        "memory", result.get("text", result.get("content", "unknown"))
                    )
                    memory_logger.info(
                        f"Single memory - ID: {memory_id}, Text: {memory_text[:100] if isinstance(memory_text, str) else memory_text}..."
                    )

                memory_logger.info(f"Memory metadata: {result.get('metadata', {})}")

                # Check for other possible keys in result
                for key, value in result.items():
                    if key not in ["results", "id", "memory", "metadata"]:
                        memory_logger.info(f"Additional result key '{key}': {str(value)[:100]}...")

        except TimeoutError:
            # Handle timeout gracefully
            error_type = "TimeoutError"
            memory_logger.error(f"Timeout while adding memory for {audio_uuid}")

            # Create a fallback memory entry
            try:
                # Store the transcript as a basic memory without using mem0
                import uuid

                unique_suffix = str(uuid.uuid4())[:8]
                fallback_memory_id = (
                    f"fallback_{audio_uuid}_{int(time.time() * 1000)}_{unique_suffix}"
                )
                created_memory_ids.append(fallback_memory_id)

                result = {
                    "id": fallback_memory_id,
                    "memory": f"Conversation summary: {transcript[:500]}{'...' if len(transcript) > 500 else ''}",
                    "metadata": {
                        "fallback_reason": error_type,
                        "original_error": "Timeout during memory processing",
                        "audio_uuid": audio_uuid,
                        "client_id": client_id,
                        "user_email": user_email,
                        "timestamp": int(time.time()),
                        "mem0_bypassed": True,
                    },
                }
                memory_logger.warning(f"Created fallback memory for {audio_uuid} due to timeout")
            except Exception as fallback_error:
                memory_logger.error(
                    f"Failed to create fallback memory for {audio_uuid}: {fallback_error}"
                )
                raise TimeoutError(f"Memory processing timeout for {audio_uuid}")

        except Exception as error:
            # Handle other errors gracefully
            error_type = type(error).__name__
            memory_logger.error(f"Error while adding memory for {audio_uuid}: {error}")

            # Create a fallback memory entry
            try:
                # Store the transcript as a basic memory without using mem0
                import uuid

                unique_suffix = str(uuid.uuid4())[:8]
                fallback_memory_id = (
                    f"fallback_{audio_uuid}_{int(time.time() * 1000)}_{unique_suffix}"
                )
                created_memory_ids.append(fallback_memory_id)

                result = {
                    "id": fallback_memory_id,
                    "memory": f"Conversation summary: {transcript[:500]}{'...' if len(transcript) > 500 else ''}",
                    "metadata": {
                        "fallback_reason": error_type,
                        "original_error": str(error),
                        "audio_uuid": audio_uuid,
                        "client_id": client_id,
                        "user_email": user_email,
                        "timestamp": int(time.time()),
                        "mem0_bypassed": True,
                    },
                }
                memory_logger.warning(
                    f"Created fallback memory for {audio_uuid} due to mem0 error: {error_type}"
                )
            except Exception as fallback_error:
                memory_logger.error(
                    f"Failed to create fallback memory for {audio_uuid}: {fallback_error}"
                )
                raise error  # Re-raise original error if fallback fails

        # Record successful memory completion
        processing_time_ms = (time.time() - start_time) * 1000

        # Record the memory extraction
        memory_id = result.get("id") if isinstance(result, dict) else str(result)
        memory_text = result.get("memory") if isinstance(result, dict) else str(result)

        debug_tracker.track_event(
            transaction_id,
            PipelineStage.MEMORY_COMPLETED,
            True,
            processing_time_ms=processing_time_ms,
            memory_id=memory_id,
            memory_text=str(memory_text)[:100] if memory_text else "none",
            transcript_length=len(transcript),
            llm_model=memory_config.get("llm_settings", {}).get("model", "llama3.1:latest"),
        )

        memory_logger.info(
            f"Successfully processed memory for {audio_uuid}, created {len(created_memory_ids)} memories: {created_memory_ids}"
        )
        return True, created_memory_ids

    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        memory_logger.error(f"Error adding memory for {audio_uuid}: {e}")

        # Record debug information for failure
        debug_tracker.track_event(
            transaction_id,
            PipelineStage.MEMORY_COMPLETED,
            False,
            error_message=str(e),
            processing_time_ms=processing_time_ms,
            transcript_length=len(transcript) if transcript else 0,
        )

        return False, []


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
            # Log Qdrant and LLM URLs
            llm_url = MEM0_CONFIG["llm"]["config"].get(
                "ollama_base_url", MEM0_CONFIG["llm"]["config"].get("api_key", "OpenAI")
            )
            memory_logger.info(
                f"Initializing MemoryService with Qdrant URL: {MEM0_CONFIG['vector_store']['config']['host']} and LLM: {llm_url}"
            )

            # Initialize main memory instance with timeout protection
            loop = asyncio.get_running_loop()
            # Build fresh config to ensure we get latest YAML settings
            config = _build_mem0_config()
            self.memory = await asyncio.wait_for(
                loop.run_in_executor(_MEMORY_EXECUTOR, Memory.from_config, config),
                timeout=MEMORY_INIT_TIMEOUT_SECONDS,
            )
            self._initialized = True
            memory_logger.info("Memory service initialized successfully")

        except asyncio.TimeoutError:
            memory_logger.error(
                f"Memory service initialization timed out after {MEMORY_INIT_TIMEOUT_SECONDS}s"
            )
            raise Exception("Memory service initialization timeout")
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
    ) -> bool:
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
                return False

        try:
            # Run the blocking operation in executor with timeout
            loop = asyncio.get_running_loop()
            success, created_memory_ids = await asyncio.wait_for(
                loop.run_in_executor(
                    _MEMORY_EXECUTOR,
                    _add_memory_to_store,
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
            return success
        except asyncio.TimeoutError:
            memory_logger.error(
                f"Memory addition timed out after {OLLAMA_TIMEOUT_SECONDS}s for {audio_uuid}"
            )
            return False
        except Exception as e:
            memory_logger.error(f"Error adding memory for {audio_uuid}: {e}")
            return False

    def get_all_memories(self, user_id: str, limit: int = 100) -> list:
        """Get all memories for a user, filtering and prioritizing semantic memories over fallback transcript memories."""
        if not self._initialized:
            # This is a sync method, so we need to handle initialization differently
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, we can't call initialize() directly
                    # This should be handled by the caller
                    raise Exception(
                        "Memory service not initialized - call await initialize() first"
                    )
                else:
                    # We're in a sync context, run the async initialize
                    loop.run_until_complete(self.initialize())
            except RuntimeError:
                # No event loop in thread pool executor
                # The service should already be initialized before being used in executor
                if not self._initialized:
                    raise Exception(
                        "Memory service not initialized - must be initialized before use in thread pool"
                    )

        assert self.memory is not None, "Memory service not initialized"
        try:
            # Get more memories than requested to account for filtering
            fetch_limit = min(limit * 3, 500)  # Get up to 3x requested amount for filtering
            memories_response = self.memory.get_all(user_id=user_id, limit=fetch_limit)

            # Handle different response formats from Mem0
            raw_memories = []
            if isinstance(memories_response, dict):
                if "results" in memories_response:
                    # New paginated format - return the results list
                    raw_memories = memories_response["results"]
                else:
                    # Old format - convert dict values to list
                    raw_memories = list(memories_response.values()) if memories_response else []
            elif isinstance(memories_response, list):
                # Already a list
                raw_memories = memories_response
            else:
                memory_logger.warning(
                    f"Unexpected memory response format: {type(memories_response)}"
                )
                return []

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

    def get_all_memories_unfiltered(self, user_id: str, limit: int = 100) -> list:
        """Get all memories for a user without filtering fallback memories (for debugging)."""
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
                memory_logger.warning(
                    f"Unexpected memory response format: {type(memories_response)}"
                )
                return []

        except Exception as e:
            memory_logger.error(f"Error fetching unfiltered memories for user {user_id}: {e}")
            raise

    def search_memories(self, query: str, user_id: str, limit: int = 10) -> list:
        """Search memories using semantic similarity, prioritizing semantic memories over fallback."""
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
            # Get more results than requested to account for filtering
            search_limit = min(limit * 3, 100)
            memories_response = self.memory.search(query=query, user_id=user_id, limit=search_limit)

            # Handle different response formats from Mem0
            raw_memories = []
            if isinstance(memories_response, dict):
                if "results" in memories_response:
                    # New paginated format - return the results list
                    raw_memories = memories_response["results"]
                else:
                    # Old format - convert dict values to list
                    raw_memories = list(memories_response.values()) if memories_response else []
            elif isinstance(memories_response, list):
                # Already a list
                raw_memories = memories_response
            else:
                memory_logger.warning(
                    f"Unexpected search response format: {type(memories_response)}"
                )
                return []

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
                    user_memories = self.get_all_memories(user_id)

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
            # Return empty list instead of raising to avoid breaking admin interface
            return []

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
            # Get all memories for the user (this is sync)
            memories = self.get_all_memories(user_id, limit)

            # Import Motor connection here to avoid circular imports
            from advanced_omi_backend.database import chunks_col

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

                    # Get transcript from database using Motor (async)
                    try:
                        memory_logger.debug(
                            f"🔍 Looking up transcript for audio_uuid: {audio_uuid}"
                        )

                        # Use existing Motor connection instead of creating new PyMongo clients
                        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})

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
                                    memory_logger.debug(
                                        f"⚠️ Empty transcript found for {audio_uuid}"
                                    )
                            else:
                                memory_logger.debug(
                                    f"⚠️ No transcript segments found for {audio_uuid}"
                                )
                        else:
                            memory_logger.debug(f"⚠️ No chunk found for audio_uuid: {audio_uuid}")

                    except Exception as db_error:
                        memory_logger.warning(
                            f"Failed to get transcript for audio_uuid {audio_uuid}: {db_error}"
                        )
                        # Continue processing other memories even if one fails

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
