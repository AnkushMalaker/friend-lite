import logging
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Configuration ---- #
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://192.168.0.110:8000")

# ---- Health Check Functions ---- #
@st.cache_data(ttl=30)  # Cache for 30 seconds to avoid too many requests
def get_system_health():
    """Get comprehensive system health from backend."""
    try:
        response = requests.get(f"{BACKEND_API_URL}/health", timeout=10)
        if response.status_code in [200, 503]:  # Both OK and unhealthy responses are valid
            return response.json()
        else:
            return {
                "status": "unhealthy",
                "overall_healthy": False,
                "services": {
                    "backend": {
                        "status": f"❌ Backend API Error: HTTP {response.status_code}",
                        "healthy": False
                    }
                },
                "error": "Backend API returned unexpected status code"
            }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "overall_healthy": False,
            "services": {
                "backend": {
                    "status": f"❌ Backend API Connection Failed: {str(e)}",
                    "healthy": False
                }
            },
            "error": str(e)
        }

# ---- Helper Functions ---- #
def get_data(endpoint: str):
    """Helper function to get data from the backend API."""
    try:
        response = requests.get(f"{BACKEND_API_URL}{endpoint}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend at `{BACKEND_API_URL}`. Please ensure it's running. Error: {e}")
        return None

def post_data(endpoint: str, params: dict | None = None):
    """Helper function to post data to the backend API."""
    try:
        response = requests.post(f"{BACKEND_API_URL}{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error posting to backend: {e}")
        return None

def delete_data(endpoint: str, params: dict | None = None):
    """Helper function to delete data from the backend API."""
    try:
        response = requests.delete(f"{BACKEND_API_URL}{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error deleting from backend: {e}")
        return None

st.set_page_config(
    page_title="Friend-Lite Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Friend-Lite Dashboard")


# ---- Sidebar with Health Checks ---- #
with st.sidebar:
    st.header("🔍 System Health")
    
    with st.expander("Service Status", expanded=True):
        # Get system health from backend
        with st.spinner("Checking system health..."):
            health_data = get_system_health()
            
            if health_data.get("overall_healthy", False):
                st.success(f"🟢 System Status: {health_data.get('status', 'Unknown').title()}")
            else:
                st.error(f"🔴 System Status: {health_data.get('status', 'Unknown').title()}")
            
            # Show individual services
            services = health_data.get("services", {})
            for service_name, service_info in services.items():
                status_text = service_info.get("status", "Unknown")
                st.write(f"**{service_name.title()}:** {status_text}")
                
                # Show additional info if available
                if "models" in service_info:
                    st.caption(f"Models available: {service_info['models']}")
                if "uri" in service_info:
                    st.caption(f"URI: {service_info['uri']}")
    
    if st.button("🔄 Refresh Health Check"):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    
    # Configuration Info  
    with st.expander("Configuration"):
        health_data = get_system_health()
        config = health_data.get("config", {})
        
        st.code(f"""
Backend API: {BACKEND_API_URL}
Active Clients: {config.get('active_clients', 'Unknown')}
MongoDB URI: {config.get('mongodb_uri', 'Unknown')[:30]}...
Ollama URL: {config.get('ollama_url', 'Unknown')}
Qdrant URL: {config.get('qdrant_url', 'Unknown')}
ASR URI: {config.get('asr_uri', 'Unknown')}
Chunk Directory: {config.get('chunk_dir', 'Unknown')}
        """)

# Show warning if system is unhealthy
health_data = get_system_health()
if not health_data.get("overall_healthy", False):
    st.error("⚠️ Some critical services are unavailable. The dashboard may not function properly.")

# ---- Main Content ---- #
tab_convos, tab_mem, tab_users = st.tabs(["Conversations", "Memories", "User Management"])

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
                ts = datetime.fromtimestamp(convo['timestamp'])
                st.write(f"**Timestamp:**")
                st.write(ts.strftime('%Y-%m-%d %H:%M:%S'))
                
                # Show client_id with better formatting
                client_id = convo.get('client_id', 'N/A')
                if client_id.startswith('client_'):
                    st.write(f"**Client ID:**")
                    st.write(f"`{client_id}`")
                else:
                    st.write(f"**User ID:**")
                    st.write(f"👤 `{client_id}`")
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
    
    # Use session state for selected user if available
    default_user = st.session_state.get('selected_user', '')
    
    # User selection for memories
    col1, col2 = st.columns([2, 1])
    with col1:
        user_id_input = st.text_input("User ID (leave empty to view memories for all users):", 
                                    value=default_user,
                                    placeholder="e.g., john_doe, alice123")
    with col2:
        st.write("")  # Spacer
        refresh_mem_btn = st.button("Refresh Memories", key="refresh_memories")
    
    # Clear the session state after using it
    if 'selected_user' in st.session_state:
        del st.session_state['selected_user']

    if refresh_mem_btn:
        st.rerun()

    # Get memories based on user selection
    if user_id_input.strip():
        memories = get_data(f"/api/memories?user_id={user_id_input.strip()}")
        st.info(f"Showing memories for user: **{user_id_input.strip()}**")
    else:
        # Show all users' memories or implement a different endpoint
        st.info("Showing memories for all users (this may require backend modification)")
        # For now, let's show a message about needing a specific user
        memories = None
        st.warning("Please enter a specific User ID to view memories. The backend requires a user_id parameter.")

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
        st.info("No memories found for this user.")

with tab_users:
    st.header("User Management")
    
    # Create User Section
    st.subheader("Create New User")
    col1, col2 = st.columns([3, 1])
    with col1:
        new_user_id = st.text_input("New User ID:", placeholder="e.g., john_doe, alice123")
    with col2:
        st.write("")  # Spacer
        create_user_btn = st.button("Create User", key="create_user")

    if create_user_btn:
        if new_user_id.strip():
            result = post_data("/api/create_user", {"user_id": new_user_id.strip()})
            if result:
                st.success(f"User '{new_user_id.strip()}' created successfully!")
                st.rerun()
        else:
            st.error("Please enter a valid User ID")

    st.divider()

    # List Users Section
    st.subheader("Existing Users")
    col1, col2 = st.columns([1, 1])
    with col1:
        refresh_users_btn = st.button("Refresh Users", key="refresh_users")
    
    if refresh_users_btn:
        st.rerun()

    users = get_data("/api/users")
    
    if users:
        st.write(f"**Total Users:** {len(users)}")
        
        # Display users in a nice format
        for user in users:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"👤 **{user.get('user_id', 'Unknown')}**")
                if '_id' in user:
                    st.caption(f"ID: {user['_id']}")
            with col2:
                delete_btn = st.button("Delete", key=f"delete_{user.get('_id', 'unknown')}")
                if delete_btn:
                    result = delete_data("/api/delete_user", {"user_id": user.get('user_id')})
                    if result:
                        st.success(f"User '{user.get('user_id')}' deleted successfully!")
                        st.rerun()
            st.divider()
    
    elif users is not None:
        st.info("No users found in the system.")
    
    st.divider()
    
    # Quick Actions Section
    st.subheader("Quick Actions")
    st.write("**View User Memories:**")
    col1, col2 = st.columns([3, 1])
    with col1:
        quick_user_id = st.text_input("User ID to view memories:", placeholder="Enter user ID", key="quick_view_user")
    with col2:
        st.write("")  # Spacer
        view_memories_btn = st.button("View Memories", key="view_memories")
    
    if view_memories_btn and quick_user_id.strip():
        # Switch to memories tab with this user
        st.session_state['selected_user'] = quick_user_id.strip()
        st.info(f"Switch to the 'Memories' tab to view memories for user: {quick_user_id.strip()}")
        
    # Tips section
    st.subheader("💡 Tips")
    st.markdown("""
    - **User IDs** should be unique identifiers (e.g., usernames, email prefixes)
    - Users are automatically created when they connect with audio if they don't exist
    - Deleting a user will not delete their memories or conversations
    - Use the 'Memories' tab to view specific user memories
    """)
