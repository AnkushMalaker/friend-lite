import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# --- Configuration ---
st.set_page_config(page_title="Friend-Lite Dashboard", layout="wide")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# --- UI ---
st.title("Friend-Lite Dashboard")

# --- Helper Functions ---
def get_data(endpoint: str):
    """Helper function to get data from the backend API."""
    try:
        response = requests.get(f"{BACKEND_API_URL}{endpoint}")
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend at `{BACKEND_API_URL}`. Please ensure it's running. Error: {e}")
        return None

# --- Main App ---
tab_convos, tab_mem = st.tabs(["Conversations", "Memories"])

with tab_convos:
    st.header("Latest Conversations")
    if st.button("Refresh Conversations"):
        st.rerun()

    conversations = get_data("/api/conversations")

    if conversations:
        for convo in conversations:
            col1, col2 = st.columns([1, 4])
            with col1:
                # Format timestamp for better readability
                ts = datetime.fromisoformat(convo['timestamp'].replace("Z", "+00:00"))
                st.write(f"**Timestamp:**")
                st.write(ts.strftime('%Y-%m-%d %H:%M:%S'))
                st.write(f"**Client ID:**")
                st.write(f"`{convo.get('client_id', 'N/A')}`")
            with col2:
                st.text_area("Transcription", convo.get("transcription", "No transcript available."), height=100, disabled=True, key=f"transcript_{convo['_id']}")
                
                audio_path = convo.get("audio_path")
                if audio_path:
                    audio_url = f"{BACKEND_API_URL}/audio/{audio_path}"
                    st.audio(audio_url, format="audio/wav")

            st.divider()
    elif conversations is not None:
        st.info("No conversations found. The backend is connected but the database might be empty.")

with tab_mem:
    st.header("Discovered Memories")
    if st.button("Refresh Memories"):
        st.rerun()

    memories = get_data("/api/memories")

    if memories:
        df = pd.DataFrame(memories)
        
        # Make the dataframe more readable
        if "created_at" in df.columns:
             df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Reorder and rename columns for clarity
        display_cols = {
            "id": "Memory ID",
            "text": "Memory",
            "created_at": "Created At"
        }
        
        # Filter for columns that exist in the dataframe
        cols_to_display = [col for col in display_cols.keys() if col in df.columns]
        
        st.dataframe(
            df[cols_to_display].rename(columns=display_cols),
            use_container_width=True,
            hide_index=True
        )

    elif memories is not None:
        st.info("No memories found. The backend is connected but no memories have been generated yet.")