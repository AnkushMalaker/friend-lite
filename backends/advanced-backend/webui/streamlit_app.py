import logging
import os
import time
import random
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
# For browser-accessible URLs (audio files), use localhost instead of Docker service name
BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "http://localhost:8000")

# ---- Health Check Functions ---- #
@st.cache_data(ttl=30)  # Cache for 30 seconds to avoid too many requests
def get_system_health():
    """Get comprehensive system health from backend."""
    try:
        # First try the simple readiness check with shorter timeout
        response = requests.get(f"{BACKEND_API_URL}/readiness", timeout=5)
        if response.status_code == 200:
            # Backend is responding, now try the full health check with longer timeout
            try:
                health_response = requests.get(f"{BACKEND_API_URL}/health", timeout=30)
                if health_response.status_code == 200:
                    return health_response.json()
                else:
                    # Health check failed but backend is responsive
                    return {
                        "status": "partial",
                        "overall_healthy": False,
                        "services": {
                            "backend": {
                                "status": f"⚠️ Backend responsive but health check failed: HTTP {health_response.status_code}",
                                "healthy": False
                            }
                        },
                        "error": "Health check endpoint returned unexpected status code"
                    }
            except requests.exceptions.Timeout:
                # Health check timed out but backend is responsive
                return {
                    "status": "partial",
                    "overall_healthy": False,
                    "services": {
                        "backend": {
                            "status": "⚠️ Backend responsive but health check timed out (some services may be slow)",
                            "healthy": False
                        }
                    },
                    "error": "Health check timed out - external services may be unavailable"
                }
            except Exception as e:
                return {
                    "status": "partial",
                    "overall_healthy": False,
                    "services": {
                        "backend": {
                            "status": f"⚠️ Backend responsive but health check failed: {str(e)}",
                            "healthy": False
                        }
                    },
                    "error": str(e)
                }
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
    """Helper function to get data from the backend API with retry logic."""
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{BACKEND_API_URL}{endpoint}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"GET {endpoint} attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                time.sleep(delay)
                continue
            else:
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

# Inject custom CSS for conversation box using Streamlit theme variables
st.markdown(
    """
    <style>
    .conversation-box {
        max-height: 300px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid var(--secondary-background-color);
        border-radius: 5px;
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        font-size: 1.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
Backend Public: {BACKEND_PUBLIC_URL}
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
tab_convos, tab_mem, tab_users, tab_manage = st.tabs(["Conversations", "Memories", "User Management", "Conversation Management"])

with tab_convos:
    st.header("Latest Conversations")
    
    # Initialize session state for refresh tracking
    if 'refresh_timestamp' not in st.session_state:
        st.session_state.refresh_timestamp = 0
    
    # Add debug mode toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Refresh Conversations"):
            st.session_state.refresh_timestamp = int(time.time())
            st.session_state.refresh_random = random.randint(1000, 9999)
            st.rerun()
    with col2:
        debug_mode = st.checkbox("🔧 Debug Mode", 
                                help="Show original audio files instead of cropped versions",
                                key="debug_mode")

    # Generate cache-busting parameter based on session state
    if st.session_state.refresh_timestamp > 0:
        random_component = getattr(st.session_state, 'refresh_random', 0)
        cache_buster = f"?t={st.session_state.refresh_timestamp}&r={random_component}"
        st.info("🔄 Audio files refreshed - cache cleared for latest versions")
    else:
        cache_buster = ""

    conversations = get_data("/api/conversations")

    if conversations:
        # Check if conversations is the new grouped format or old format
        if isinstance(conversations, dict) and "conversations" in conversations:
            # New grouped format
            conversations_data = conversations["conversations"]
            
            for client_id, client_conversations in conversations_data.items():
                st.subheader(f"👤 {client_id}")
                
                for convo in client_conversations:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        # Format timestamp for better readability
                        ts = datetime.fromtimestamp(convo['timestamp'])
                        st.write(f"**Timestamp:**")
                        st.write(ts.strftime('%Y-%m-%d %H:%M:%S'))
                        
                        # Show Audio UUID
                        audio_uuid = convo.get("audio_uuid", "N/A")
                        st.write(f"**Audio UUID:**")
                        st.code(audio_uuid, language=None)
                        
                        # Show identified speakers
                        speakers = convo.get("speakers_identified", [])
                        if speakers:
                            st.write(f"**Speakers:**")
                            for speaker in speakers:
                                st.write(f"🎤 `{speaker}`")
                        
                        # Show audio duration info if available
                        cropped_duration = convo.get("cropped_duration")
                        if cropped_duration:
                            st.write(f"**Cropped Duration:**")
                            st.write(f"⏱️ {cropped_duration:.1f}s")
                            
                            # Show speech segments count
                            speech_segments = convo.get("speech_segments", [])
                            if speech_segments:
                                st.write(f"**Speech Segments:**")
                                st.write(f"🗣️ {len(speech_segments)} segments")
                    
                    with col2:
                        # Display conversation transcript with new format
                        transcript = convo.get("transcript", [])
                        if transcript:
                            st.write("**Conversation:**")
                            conversation_text = ""
                            for segment in transcript:
                                speaker = segment.get("speaker", "Unknown")
                                text = segment.get("text", "")
                                start_time = segment.get("start", 0.0)
                                end_time = segment.get("end", 0.0)
                                
                                # Format timing if available
                                timing_info = ""
                                if start_time > 0 or end_time > 0:
                                    timing_info = f" [{start_time:.1f}s - {end_time:.1f}s]"
                                
                                conversation_text += f"<b>{speaker}</b>{timing_info}: {text}<br><br>"
                            
                            # Display in a scrollable container with max height
                            st.markdown(
                                f'<div class="conversation-box">{conversation_text}</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Smart audio display logic
                        audio_path = convo.get("audio_path")
                        cropped_audio_path = convo.get("cropped_audio_path")
                        
                        if audio_path:
                            # Determine which audio to show
                            if debug_mode:
                                # Debug mode: always show original
                                selected_audio_path = audio_path
                                audio_label = "🔧 **Original Audio** (Debug Mode)"
                            elif cropped_audio_path:
                                # Normal mode: prefer cropped if available
                                selected_audio_path = cropped_audio_path
                                audio_label = "🎵 **Cropped Audio** (Silence Removed)"
                            else:
                                # Fallback: show original if no cropped version
                                selected_audio_path = audio_path
                                audio_label = "🎵 **Original Audio** (No cropped version available)"
                            
                            # Display audio with label and cache-busting
                            st.write(audio_label)
                            audio_url = f"{BACKEND_PUBLIC_URL}/audio/{selected_audio_path}{cache_buster}"
                            st.audio(audio_url, format="audio/wav")
                            
                            # Show additional info in debug mode or when both versions exist
                            if debug_mode and cropped_audio_path:
                                st.caption(f"💡 Cropped version available: {cropped_audio_path}")
                            elif not debug_mode and cropped_audio_path:
                                st.caption(f"💡 Enable debug mode to hear original with silence")

                    st.divider()
        else:
            # Old format - single list of conversations
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
                    
                    # Show Audio UUID
                    audio_uuid = convo.get("audio_uuid", "N/A")
                    st.write(f"**Audio UUID:**")
                    st.code(audio_uuid, language=None)
                    
                    # Show identified speakers
                    speakers = convo.get("speakers_identified", [])
                    if speakers:
                        st.write(f"**Speakers:**")
                        for speaker in speakers:
                            st.write(f"🎤 `{speaker}`")
                
                with col2:
                    # Display conversation transcript with new format
                    transcript = convo.get("transcript", [])
                    if transcript:
                        st.write("**Conversation:**")
                        conversation_text = ""
                        for segment in transcript:
                            speaker = segment.get("speaker", "Unknown")
                            text = segment.get("text", "")
                            start_time = segment.get("start", 0.0)
                            end_time = segment.get("end", 0.0)
                            
                            # Format timing if available
                            timing_info = ""
                            if start_time > 0 or end_time > 0:
                                timing_info = f" [{start_time:.1f}s - {end_time:.1f}s]"
                            
                            conversation_text += f"<b>{speaker}</b>{timing_info}: {text}<br><br>"
                        
                        # Display in a scrollable container with max height
                        st.markdown(
                            f'<div class="conversation-box">{conversation_text}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        # Fallback for old format
                        old_transcript = convo.get("transcription", "No transcript available.")
                        st.text_area("Transcription", old_transcript, height=150, disabled=True, key=f"transcript_{convo['_id']}")
                    
                    # Smart audio display logic (same as above)
                    audio_path = convo.get("audio_path")
                    cropped_audio_path = convo.get("cropped_audio_path")
                    
                    if audio_path:
                        # Determine which audio to show
                        if debug_mode:
                            # Debug mode: always show original
                            selected_audio_path = audio_path
                            audio_label = "🔧 **Original Audio** (Debug Mode)"
                        elif cropped_audio_path:
                            # Normal mode: prefer cropped if available
                            selected_audio_path = cropped_audio_path
                            audio_label = "🎵 **Cropped Audio** (Silence Removed)"
                        else:
                            # Fallback: show original if no cropped version
                            selected_audio_path = audio_path
                            audio_label = "🎵 **Original Audio** (No cropped version available)"
                        
                        # Display audio with label and cache-busting
                        st.write(audio_label)
                        audio_url = f"{BACKEND_PUBLIC_URL}/audio/{selected_audio_path}{cache_buster}"
                        st.audio(audio_url, format="audio/wav")
                        
                        # Show additional info in debug mode or when both versions exist
                        if debug_mode and cropped_audio_path:
                            st.caption(f"💡 Cropped version available: {cropped_audio_path}")
                        elif not debug_mode and cropped_audio_path:
                            st.caption(f"💡 Enable debug mode to hear original with silence")

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
        user_id_input = st.text_input("Enter username to view memories:", 
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
        memories_response = get_data(f"/api/memories?user_id={user_id_input.strip()}")
        st.info(f"Showing memories for user: **{user_id_input.strip()}**")
        
        # Handle the API response format with "results" wrapper
        if memories_response and isinstance(memories_response, dict) and "results" in memories_response:
            memories = memories_response["results"]
        else:
            memories = memories_response
    else:
        # Show instruction to enter a username
        memories = None
        st.info("👆 Please enter a username above to view their memories.")
        st.markdown("💡 **Tip:** You can find existing usernames in the 'User Management' tab.")

    if memories:
        df = pd.DataFrame(memories)
        
        # Make the dataframe more readable
        if "created_at" in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')

        # Reorder and rename columns for clarity - handle both "memory" and "text" fields
        display_cols = {
            "id": "Memory ID",
            "created_at": "Created At"
        }
        
        # Check which memory field exists and add it to display columns
        if "memory" in df.columns:
            display_cols["memory"] = "Memory"
        elif "text" in df.columns:
            display_cols["text"] = "Memory"
        
        # Filter for columns that exist in the dataframe
        cols_to_display = [col for col in display_cols.keys() if col in df.columns]
        
        if cols_to_display:
            st.dataframe(
                df[cols_to_display].rename(columns=display_cols),
                use_container_width=True,
                hide_index=True
            )
            
            # Show additional details
            st.caption(f"📊 Found **{len(memories)}** memories for user **{user_id_input.strip()}**")
        else:
            st.error("⚠️ Unexpected memory data format - missing expected fields")
            st.write("Debug info - Available columns:", list(df.columns))

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
        
        # Initialize session state for delete confirmation
        if 'delete_confirmation' not in st.session_state:
            st.session_state.delete_confirmation = {}
        
        # Display users in a nice format
        for user in users:
            user_id = user.get('user_id', 'Unknown')
            user_db_id = user.get('_id', 'unknown')
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"👤 **{user_id}**")
                if '_id' in user:
                    st.caption(f"ID: {user['_id']}")
            
            with col2:
                # Check if we're in confirmation mode for this user
                if user_id in st.session_state.delete_confirmation:
                    # Show confirmation dialog in a container
                    with st.container():
                        st.error("⚠️ **Confirm Deletion**")
                        st.write(f"Delete user **{user_id}** and optionally:")
                        
                        # Checkboxes for what to delete
                        delete_conversations = st.checkbox(
                            "🗨️ Delete all conversations", 
                            key=f"conv_{user_db_id}",
                            help="Permanently delete all audio recordings and transcripts"
                        )
                        delete_memories = st.checkbox(
                            "🧠 Delete all memories", 
                            key=f"mem_{user_db_id}",
                            help="Permanently delete all extracted memories from conversations"
                        )
                        
                        # Action buttons
                        col_cancel, col_confirm = st.columns([1, 1])
                        
                        with col_cancel:
                            if st.button("❌ Cancel", key=f"cancel_{user_db_id}", use_container_width=True, type="secondary"):
                                del st.session_state.delete_confirmation[user_id]
                                st.rerun()
                        
                        with col_confirm:
                            if st.button("🗑️ Confirm Delete", key=f"confirm_{user_db_id}", use_container_width=True, type="primary"):
                                # Build delete parameters
                                params = {
                                    "user_id": user_id,
                                    "delete_conversations": delete_conversations,
                                    "delete_memories": delete_memories
                                }
                                
                                result = delete_data("/api/delete_user", params)
                                if result:
                                    deleted_data = result.get('deleted_data', {})
                                    message = result.get('message', f"User '{user_id}' deleted")
                                    st.success(message)
                                    
                                    # Show detailed deletion info
                                    if deleted_data.get('conversations_deleted', 0) > 0 or deleted_data.get('memories_deleted', 0) > 0:
                                        st.info(f"📊 Deleted: {deleted_data.get('conversations_deleted', 0)} conversations, {deleted_data.get('memories_deleted', 0)} memories")
                                    
                                    del st.session_state.delete_confirmation[user_id]
                                    st.rerun()
                        
                        if delete_conversations or delete_memories:
                            st.caption("⚠️ Selected data will be **permanently deleted** and cannot be recovered!")
                else:
                    # Show normal delete button
                    delete_btn = st.button("🗑️ Delete", key=f"delete_{user_db_id}", type="secondary")
                    if delete_btn:
                        st.session_state.delete_confirmation[user_id] = True
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
    - **Delete Options:**
      - **User Account**: Always deleted when you click delete
      - **🗨️ Conversations**: Check to delete all audio recordings and transcripts
      - **🧠 Memories**: Check to delete all extracted memories from conversations
      - Mix and match: You can delete just conversations, just memories, or both
    - Use the 'Memories' tab to view specific user memories
    """)

with tab_manage:
    st.header("Conversation Management")
    
    st.subheader("Add Speaker to Conversation")
    st.write("Add speakers to conversations even if they haven't spoken yet.")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        audio_uuid_input = st.text_input("Audio UUID:", placeholder="Enter the audio UUID")
    with col2:
        speaker_id_input = st.text_input("Speaker ID:", placeholder="e.g., speaker_1, john_doe")
    with col3:
        st.write("")  # Spacer
        add_speaker_btn = st.button("Add Speaker", key="add_speaker")
    
    if add_speaker_btn:
        if audio_uuid_input.strip() and speaker_id_input.strip():
            result = post_data(f"/api/conversations/{audio_uuid_input.strip()}/speakers", 
                             {"speaker_id": speaker_id_input.strip()})
            if result:
                st.success(f"Speaker '{speaker_id_input.strip()}' added to conversation!")
        else:
            st.error("Please enter both Audio UUID and Speaker ID")
    
    st.divider()
    
    st.subheader("Update Transcript Segment")
    st.write("Modify speaker identification or timing information for transcript segments.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        update_audio_uuid = st.text_input("Audio UUID:", placeholder="Enter the audio UUID", key="update_uuid")
        segment_index = st.number_input("Segment Index:", min_value=0, value=0, step=1)
        new_speaker = st.text_input("New Speaker ID (optional):", placeholder="Leave empty to keep current")
    
    with col2:
        start_time = st.number_input("Start Time (seconds):", min_value=0.0, value=0.0, step=0.1, format="%.1f")
        end_time = st.number_input("End Time (seconds):", min_value=0.0, value=0.0, step=0.1, format="%.1f")
        update_segment_btn = st.button("Update Segment", key="update_segment")
    
    if update_segment_btn:
        if update_audio_uuid.strip():
            params = {}
            if new_speaker.strip():
                params["speaker_id"] = new_speaker.strip()
            if start_time > 0:
                params["start_time"] = start_time
            if end_time > 0:
                params["end_time"] = end_time
            
            if params:
                # Use requests.put for this endpoint
                try:
                    response = requests.put(
                        f"{BACKEND_API_URL}/api/conversations/{update_audio_uuid.strip()}/transcript/{segment_index}",
                        params=params
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.success("Transcript segment updated successfully!")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error updating segment: {e}")
            else:
                st.warning("Please specify at least one field to update")
        else:
            st.error("Please enter the Audio UUID")
    
    st.divider()
    
    st.subheader("💡 Schema Information")
    st.markdown("""
    **New Conversation Schema:**
    ```json
    {
        "audio_uuid": "unique_identifier",
        "audio_path": "path/to/audio/file.wav",
        "client_id": "user_or_client_id", 
        "timestamp": 1234567890,
        "transcript": [
            {
                "speaker": "speaker_1",
                "text": "Hello, how are you?",
                "start": 0.0,
                "end": 3.2
            },
            {
                "speaker": "speaker_2", 
                "text": "I'm good, thanks!",
                "start": 3.3,
                "end": 5.0
            }
        ],
        "speakers_identified": ["speaker_1", "speaker_2"]
    }
    ```
    """)
    
    st.info("💡 **Tip**: You can find Audio UUIDs in the conversation details on the 'Conversations' tab.")
