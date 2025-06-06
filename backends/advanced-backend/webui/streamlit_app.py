import streamlit as st
from pymongo import MongoClient
from mem0 import Memory
from dotenv import load_dotenv
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ---- mem0 + Ollama Configuration (copied from main.py) ---- #
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434") # Added for completeness
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
            "host": "qdrant", # Make sure this matches your docker-compose setup for streamlit if run in a container
            "port": 6333
        },
    },
}


# ---------- connections ----------
# Ensure MONGODB_URI is set in your environment or .env file
# Example: MONGODB_URI=mongodb://localhost:27017/omi if running locally and mongo is on localhost
# Or: MONGODB_URI=mongodb://mongo:27017/omi if streamlit is in a docker container on the same network as a 'mongo' service
mongo_client = MongoClient(os.getenv("MONGODB_URI_STREAMLIT"))
db = mongo_client.get_default_database() # The blueprint specified "friend-lite", make sure this is consistent
chunks_col = db["audio_chunks"]
logger.info("Connected to MongoDB")

# Ensure the vector store config for mem0 points to the correct Qdrant instance
# If Streamlit runs in a container, 'qdrant' as a hostname might need to be resolvable
# or replaced with localhost if Qdrant is also local and not containerized with Streamlit.
memory_streamlit_config = {
    "llm": MEM0_CONFIG["llm"],
    "embedder": MEM0_CONFIG["embedder"],
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": MEM0_CONFIG["vector_store"]["config"]["collection_name"],
            "embedding_model_dims": MEM0_CONFIG["vector_store"]["config"]["embedding_model_dims"],
            "host": os.getenv("QDRANT_HOST_STREAMLIT", MEM0_CONFIG["vector_store"]["config"]["host"]), # Allow override for streamlit
            "port": int(os.getenv("QDRANT_PORT_STREAMLIT", MEM0_CONFIG["vector_store"]["config"]["port"])) # Allow override for streamlit
        }
    }
}
memory = Memory.from_config(memory_streamlit_config)
logger.info("Connected to Mem0")

# ---------- UI ----------
st.set_page_config(page_title="Friend-Lite Dashboard", layout="wide")

st.title("Friend-Lite Unified Audio Service Dashboard")

tab_chunks, tab_mem = st.tabs(["Audio Chunks (MongoDB)", "Mem0 Memories (Qdrant)"])

with tab_chunks:
    st.header("Latest audio chunks")
    try:
        # Adding a refresh button
        if st.button("Refresh Chunks"):
            st.rerun()

        docs = list(chunks_col.find().sort("timestamp", -1).limit(200))
        if docs:
            df = pd.DataFrame(docs)
            # Ensure all expected columns are present, fill with None if not
            expected_cols = ["audio_uuid", "audio_path", "client_id", "timestamp", "transcription", "speakers_identified", "_id"]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None
            
            # Select and reorder columns for display, dropping mongo's _id by default
            display_cols = ["timestamp", "client_id", "audio_uuid", "transcription", "audio_path", "speakers_identified"]
            df_display = df[display_cols]

            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("No audio chunks found in MongoDB. Ensure the backend service is running and generating data.")
    except Exception as e:
        st.error(f"Error connecting to MongoDB or fetching chunks: {e}")
        st.info("Please ensure MongoDB is running and accessible, and the MONGODB_URI environment variable is correctly set.")


with tab_mem:
    st.header("Recent memories (vector store)")
    try:
        # Adding a refresh button
        if st.button("Refresh Memories"):
            st.rerun()
            
        # Basic search functionality
        search_query = st.text_input("Search memories:", key="mem_search")
        search_k = st.slider("Number of results (limit):", 1, 200, 50, key="mem_k")

        if search_query:
            results = memory.search(search_query, limit=search_k)
        else:
            results = memory.get_all(limit=search_k) # Fetch recent if no query

        if results:
            # Process mem0 results which can have varied metadata
            processed_results = []
            for r_idx, r_val in enumerate(results):
                record = {}
                if isinstance(r_val, dict):
                    record = {"id": r_val.get("id", f"mem_{r_idx}"), "text": r_val.get("text", "")}
                    # Flatten metadata, being mindful of potential nesting or complex types if any
                    metadata = r_val.get("metadata", {})
                    if isinstance(metadata, dict):
                        for mk, mv in metadata.items():
                            record[f"meta_{mk}"] = mv # prefix to avoid clashes
                    else:
                        record["metadata_raw"] = str(metadata) # if not a dict, store as string
                elif isinstance(r_val, str):
                    record = {"id": f"mem_{r_idx}", "text": r_val, "metadata_raw": "N/A"}
                else:
                    # Handle other unexpected types if necessary, or skip
                    record = {"id": f"mem_{r_idx}", "text": str(r_val), "metadata_raw": "Unknown Type"}
                processed_results.append(record)

            mem_df = pd.DataFrame(processed_results)
            
            # Reorder columns to make 'text' and 'id' prominent
            cols = list(mem_df.columns)
            if "text" in cols:
                cols.insert(0, cols.pop(cols.index("text")))
            if "id" in cols:
                 cols.insert(0, cols.pop(cols.index("id")))
            if "meta_audio_uuid" in cols:
                cols.insert(1, cols.pop(cols.index("meta_audio_uuid"))) # Make audio_uuid prominent if present

            st.dataframe(mem_df[cols], use_container_width=True)
        else:
            st.info("No memories found in Mem0. Ensure the backend service is storing memories and Qdrant is accessible.")
            st.text(f"Mem0 configuration used: {memory_streamlit_config}")
    except Exception as e:
        st.error(f"Error connecting to Mem0/Qdrant or fetching memories: {e}")
        st.info("Please ensure Qdrant is running and accessible. You might need to set QDRANT_HOST_STREAMLIT if Streamlit is in a different Docker network.")
        st.text(f"Mem0 configuration attempted: {memory_streamlit_config}") 
logger.info("Streamlit app loaded")