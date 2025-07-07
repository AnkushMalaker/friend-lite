from mem0 import Memory
import os

# Custom instructions for memory processing
# These aren't being used right now but Mem0 does support adding custom prompting
# for handling memory retrieval and processing.
CUSTOM_INSTRUCTIONS = """
Extract the Following Information:  

- Key Information: Identify and save the most important details.
- Context: Capture the surrounding context to understand the memory's relevance.
- Connections: Note any relationships to other topics or memories.
- Importance: Highlight why this information might be valuable in the future.
- Source: Record where this information came from when applicable.
"""

def get_memory_client():
    # Get LLM provider and configuration
    llm_provider = os.getenv('LLM_PROVIDER')
    llm_api_key = os.getenv('LLM_API_KEY')
    llm_model = os.getenv('LLM_CHOICE')
    embedding_model = os.getenv('EMBEDDING_MODEL_CHOICE')
    vector_store_provider = os.getenv('VECTOR_STORE_PROVIDER')
    graph_store_provider = os.getenv('GRAPH_STORE_PROVIDER')
    
    # Initialize config dictionary
    config = {}
    print(f"llm_provider: {llm_provider}")
    
    # Configure LLM based on provider
    if llm_provider == 'openai' or llm_provider == 'openrouter':
        print(f"llm_provider: {llm_provider}")
        config["llm"] = {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "temperature": 0.2,
                "max_tokens": 2000,
            }
        }
        
        print(f"llm_api_key: {llm_api_key}")
        # Set API key in environment if not already set
        if llm_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = llm_api_key
  
        # For OpenRouter, set the specific API key
        if llm_provider == 'openrouter' and llm_api_key:
            os.environ["OPENROUTER_API_KEY"] = llm_api_key
    
    elif llm_provider == 'ollama':
        config["llm"] = {
            "provider": "ollama",
            "config": {
                "model": llm_model,
                "temperature": 0.2,
                "max_tokens": 2000,
            }
        }
        
        # Set base URL for Ollama if provided
        llm_base_url = os.getenv('LLM_BASE_URL')
        if llm_base_url:
            config["llm"]["config"]["llm_base_url"] = llm_base_url
    
    # Configure embedder based on provider
    if llm_provider == 'openai':
        config["embedder"] = {
            "provider": "openai",
            "config": {
                "model": embedding_model or "text-embedding-3-small",
                "embedding_dims": 1536  # Default for text-embedding-3-small
            }
        }
        
        # Set API key in environment if not already set
        if llm_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = llm_api_key
    
    elif llm_provider == 'ollama':
        config["embedder"] = {
            "provider": "ollama",
            "config": {
                "model": embedding_model or "nomic-embed-text",
                "embedding_dims": 768  # Default for nomic-embed-text
            }
        }
        
        # Set base URL for Ollama if provided
        embedding_base_url = os.getenv('LLM_BASE_URL')
        if embedding_base_url:
            config["embedder"]["config"]["llm_base_url"] = embedding_base_url
    
    # Configure Supabase vector store
    if vector_store_provider == 'supabase':
        config["vector_store"] = {
            "provider": "supabase",
            "config": {
                "connection_string": os.environ.get('DATABASE_URL', ''),
                "collection_name": "mem0_memories",
                "embedding_model_dims": 1536 if llm_provider == "openai" else 768
            }
        }
    elif vector_store_provider == 'qdrant':
        config["vector_store"] = {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0_memories",
                "embedding_model_dims": 1536 if llm_provider == "openai" else 768
            }
        }

    if graph_store_provider == 'neo4j':
        config["graph_store"] = {
            "provider": "neo4j",
            "config": {
               "url": os.environ.get('neo4j://'+os.environ.get('NEO4J_HOST', '')+':7687'),
               "username": os.environ.get('NEO4J_USERNAME', ''),
               "password": os.environ.get('NEO4J_PASSWORD', ''),
            }
        }
    elif graph_store_provider == 'memgraph':
        config["graph_store"] = {
            "provider": "memgraph",
            "config": {
                "url": os.environ.get('MEMGRAPH_URI', ''),
                "username": os.environ.get('MEMGRAPH_USERNAME', ''),
                "password": os.environ.get('MEMGRAPH_PASSWORD', ''),
            }
        }

    # config["custom_fact_extraction_prompt"] = {CUSTOM_INSTRUCTIONS}
    print(f"config: {config}")

    # Create and return the Memory client
    return Memory.from_config(config)