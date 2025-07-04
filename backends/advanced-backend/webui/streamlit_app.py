import logging
import os
import time
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Create logs directory for Streamlit app
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure comprehensive logging for Streamlit app
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / 'streamlit.log')
    ]
)

logger = logging.getLogger("streamlit-ui")
logger.info("🚀 Starting Friend-Lite Streamlit Dashboard")

# ---- Configuration ---- #
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://192.168.0.110:8000")
# For browser-accessible URLs (audio files), use localhost instead of Docker service name
BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "http://localhost:8000")

logger.info(f"🔧 Configuration loaded - Backend API: {BACKEND_API_URL}, Public URL: {BACKEND_PUBLIC_URL}")

# ---- Health Check Functions ---- #
@st.cache_data(ttl=30)  # Cache for 30 seconds to avoid too many requests
def get_system_health():
    """Get comprehensive system health from backend."""
    logger.info("🏥 Performing system health check")
    start_time = time.time()
    
    try:
        # First try the simple readiness check with shorter timeout
        logger.debug("🔍 Checking backend readiness...")
        response = requests.get(f"{BACKEND_API_URL}/readiness", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Backend readiness check passed")
            # Backend is responding, now try the full health check with longer timeout
            try:
                logger.debug("🔍 Performing full health check...")
                health_response = requests.get(f"{BACKEND_API_URL}/health", timeout=30)
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    duration = time.time() - start_time
                    logger.info(f"✅ Full health check completed in {duration:.3f}s")
                    logger.debug(f"Health data: {health_data}")
                    return health_data
                else:
                    # Health check failed but backend is responsive
                    duration = time.time() - start_time
                    logger.warning(f"⚠️ Health check failed with status {health_response.status_code} in {duration:.3f}s")
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
                duration = time.time() - start_time
                logger.warning(f"⚠️ Health check timed out in {duration:.3f}s")
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
                duration = time.time() - start_time
                logger.error(f"❌ Health check error in {duration:.3f}s: {e}")
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
            duration = time.time() - start_time
            logger.error(f"❌ Backend readiness check failed with status {response.status_code} in {duration:.3f}s")
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
        duration = time.time() - start_time
        logger.error(f"❌ System health check failed in {duration:.3f}s: {e}")
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
    logger.debug(f"📡 GET request to endpoint: {endpoint}")
    start_time = time.time()
    
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"📡 Attempt {attempt + 1}/{max_retries} for GET {endpoint}")
            response = requests.get(f"{BACKEND_API_URL}{endpoint}")
            response.raise_for_status()
            duration = time.time() - start_time
            logger.info(f"✅ GET {endpoint} successful in {duration:.3f}s")
            return response.json()
        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"⚠️ GET {endpoint} attempt {attempt + 1} failed in {duration:.3f}s, retrying in {delay}s: {str(e)}")
                time.sleep(delay)
                continue
            else:
                logger.error(f"❌ GET {endpoint} failed after {max_retries} attempts in {duration:.3f}s: {e}")
                st.error(f"Could not connect to the backend at `{BACKEND_API_URL}`. Please ensure it's running. Error: {e}")
                return None

def post_data(endpoint: str, params: dict | None = None):
    """Helper function to post data to the backend API."""
    logger.debug(f"📤 POST request to endpoint: {endpoint} with params: {params}")
    start_time = time.time()
    
    try:
        response = requests.post(f"{BACKEND_API_URL}{endpoint}", params=params)
        response.raise_for_status()
        duration = time.time() - start_time
        logger.info(f"✅ POST {endpoint} successful in {duration:.3f}s")
        return response.json()
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        logger.error(f"❌ POST {endpoint} failed in {duration:.3f}s: {e}")
        st.error(f"Error posting to backend: {e}")
        return None

def delete_data(endpoint: str, params: dict | None = None):
    """Helper function to delete data from the backend API."""
    logger.debug(f"🗑️ DELETE request to endpoint: {endpoint} with params: {params}")
    start_time = time.time()
    
    try:
        response = requests.delete(f"{BACKEND_API_URL}{endpoint}", params=params)
        response.raise_for_status()
        duration = time.time() - start_time
        logger.info(f"✅ DELETE {endpoint} successful in {duration:.3f}s")
        return response.json()
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        logger.error(f"❌ DELETE {endpoint} failed in {duration:.3f}s: {e}")
        st.error(f"Error deleting from backend: {e}")
        return None

# ---- Streamlit App Configuration ---- #
logger.info("🎨 Configuring Streamlit app...")
st.set_page_config(
    page_title="Friend-Lite Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Friend-Lite Dashboard")
logger.info("📊 Dashboard initialized")

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
    logger.debug("🔍 Loading system health sidebar...")
    
    with st.expander("Service Status", expanded=True):
        # Get system health from backend
        with st.spinner("Checking system health..."):
            health_data = get_system_health()
            
            if health_data.get("overall_healthy", False):
                st.success(f"🟢 System Status: {health_data.get('status', 'Unknown').title()}")
                logger.info("🟢 System health check passed")
            else:
                st.error(f"🔴 System Status: {health_data.get('status', 'Unknown').title()}")
                logger.warning(f"🔴 System health check failed: {health_data.get('error', 'Unknown error')}")
            
            # Show individual services
            services = health_data.get("services", {})
            for service_name, service_info in services.items():
                status_text = service_info.get("status", "Unknown")
                st.write(f"**{service_name.title()}:** {status_text}")
                logger.debug(f"Service {service_name}: {status_text}")
                
                # Show additional info if available
                if "models" in service_info:
                    st.caption(f"Models available: {service_info['models']}")
                    logger.debug(f"Service {service_name} models: {service_info['models']}")
                if "uri" in service_info:
                    st.caption(f"URI: {service_info['uri']}")
                    logger.debug(f"Service {service_name} URI: {service_info['uri']}")
    
    if st.button("🔄 Refresh Health Check"):
        logger.info("🔄 Manual health check refresh requested")
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    
    # Close Conversation Section
    st.header("🔒 Close Conversation")
    logger.debug("🔒 Loading close conversation section...")
    
    with st.expander("Active Clients & Close Conversation", expanded=True):
        # Get active clients
        logger.debug("📡 Fetching active clients...")
        active_clients_data = get_data("/api/active_clients")
        
        if active_clients_data and active_clients_data.get("clients"):
            clients = active_clients_data["clients"]
            logger.info(f"📊 Found {len(clients)} active clients")
            
            # Show active clients with conversation status
            for client_id, client_info in clients.items():
                logger.debug(f"👤 Processing client: {client_id} - Active conversation: {client_info.get('has_active_conversation', False)}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if client_info.get("has_active_conversation", False):
                        st.write(f"🟢 **{client_id}** (Active conversation)")
                        if client_info.get("current_audio_uuid"):
                            st.caption(f"UUID: {client_info['current_audio_uuid'][:8]}...")
                            logger.debug(f"Client {client_id} has active conversation with UUID: {client_info['current_audio_uuid']}")
                    else:
                        st.write(f"⚪ **{client_id}** (No active conversation)")
                        logger.debug(f"Client {client_id} has no active conversation")
                
                with col2:
                    if client_info.get("has_active_conversation", False):
                        close_btn = st.button(
                            "🔒 Close",
                            key=f"close_{client_id}",
                            help=f"Close current conversation for {client_id}",
                            type="secondary"
                        )
                        
                        if close_btn:
                            logger.info(f"🔒 Closing conversation for client: {client_id}")
                            result = post_data("/api/close_conversation", {"client_id": client_id})
                            if result:
                                st.success(f"✅ Conversation closed for {client_id}")
                                logger.info(f"✅ Successfully closed conversation for {client_id}")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to close conversation for {client_id}")
                                logger.error(f"❌ Failed to close conversation for {client_id}")
                    else:
                        st.caption("No active conversation")
            
            st.info(f"💡 **Total active clients:** {active_clients_data.get('active_clients_count', 0)}")
        else:
            st.info("No active clients found")
            logger.info("📊 No active clients found")
    
    st.divider()
    
    # Configuration Info  
    with st.expander("Configuration"):
        logger.debug("🔧 Loading configuration info...")
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
        logger.debug(f"🔧 Configuration displayed - Backend API: {BACKEND_API_URL}")

# Show warning if system is unhealthy
health_data = get_system_health()
if not health_data.get("overall_healthy", False):
    st.error("⚠️ Some critical services are unavailable. The dashboard may not function properly.")
    logger.warning("⚠️ System is unhealthy - some services unavailable")

# ---- Main Content ---- #
logger.info("📋 Loading main dashboard tabs...")
tab_convos, tab_mem, tab_users, tab_manage = st.tabs(["Conversations", "Memories", "User Management", "Conversation Management"])

with tab_convos:
    logger.debug("🗨️ Loading conversations tab...")
    st.header("Latest Conversations")
    
    # Initialize session state for refresh tracking
    if 'refresh_timestamp' not in st.session_state:
        st.session_state.refresh_timestamp = 0
    
    # Add debug mode toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Refresh Conversations"):
            logger.info("🔄 Manual conversation refresh requested")
            st.session_state.refresh_timestamp = int(time.time())
            st.session_state.refresh_random = random.randint(1000, 9999)
            st.rerun()
    with col2:
        debug_mode = st.checkbox("🔧 Debug Mode", 
                                help="Show original audio files instead of cropped versions",
                                key="debug_mode")
        if debug_mode:
            logger.debug("🔧 Debug mode enabled")

    # Generate cache-busting parameter based on session state
    if st.session_state.refresh_timestamp > 0:
        random_component = getattr(st.session_state, 'refresh_random', 0)
        cache_buster = f"?t={st.session_state.refresh_timestamp}&r={random_component}"
        st.info("🔄 Audio files refreshed - cache cleared for latest versions")
        logger.info("🔄 Audio cache busting applied")
    else:
        cache_buster = ""

    logger.debug("📡 Fetching conversations data...")
    conversations = get_data("/api/conversations")

    if conversations:
        logger.info(f"📊 Loaded {len(conversations) if isinstance(conversations, list) else 'grouped'} conversations")
        
        # Check if conversations is the new grouped format or old format
        if isinstance(conversations, dict) and "conversations" in conversations:
            # New grouped format
            logger.debug("📊 Processing conversations in new grouped format")
            conversations_data = conversations["conversations"]
            
            for client_id, client_conversations in conversations_data.items():
                logger.debug(f"👤 Processing conversations for client: {client_id} ({len(client_conversations)} conversations)")
                st.subheader(f"👤 {client_id}")
                
                for convo in client_conversations:
                    logger.debug(f"🗨️ Processing conversation: {convo.get('audio_uuid', 'unknown')}")
                    
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
                            logger.debug(f"🎤 Speakers identified: {speakers}")
                        
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
                                logger.debug(f"🗣️ Speech segments: {len(speech_segments)}")
                    
                    with col2:
                        # Display conversation transcript with new format
                        transcript = convo.get("transcript", [])
                        if transcript:
                            logger.debug(f"📝 Displaying transcript with {len(transcript)} segments")
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
                                logger.debug(f"🔧 Debug mode: showing original audio: {audio_path}")
                            elif cropped_audio_path:
                                # Normal mode: prefer cropped if available
                                selected_audio_path = cropped_audio_path
                                audio_label = "🎵 **Cropped Audio** (Silence Removed)"
                                logger.debug(f"🎵 Normal mode: showing cropped audio: {cropped_audio_path}")
                            else:
                                # Fallback: show original if no cropped version
                                selected_audio_path = audio_path
                                audio_label = "🎵 **Original Audio** (No cropped version available)"
                                logger.debug(f"🎵 Fallback: showing original audio (no cropped version): {audio_path}")
                            
                            # Display audio with label and cache-busting
                            st.write(audio_label)
                            audio_url = f"{BACKEND_PUBLIC_URL}/audio/{selected_audio_path}{cache_buster}"
                            st.audio(audio_url, format="audio/wav")
                            logger.debug(f"🎵 Audio URL: {audio_url}")
                            
                            # Show additional info in debug mode or when both versions exist
                            if debug_mode and cropped_audio_path:
                                st.caption(f"💡 Cropped version available: {cropped_audio_path}")
                            elif not debug_mode and cropped_audio_path:
                                st.caption(f"💡 Enable debug mode to hear original with silence")

                    st.divider()
        else:
            # Old format - single list of conversations
            logger.debug("📊 Processing conversations in old format")
            for convo in conversations:
                logger.debug(f"🗨️ Processing conversation: {convo.get('audio_uuid', 'unknown')}")
                
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
                        logger.debug(f"📝 Displaying transcript with {len(transcript)} segments")
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
                            logger.debug(f"🔧 Debug mode: showing original audio: {audio_path}")
                        elif cropped_audio_path:
                            # Normal mode: prefer cropped if available
                            selected_audio_path = cropped_audio_path
                            audio_label = "🎵 **Cropped Audio** (Silence Removed)"
                            logger.debug(f"🎵 Normal mode: showing cropped audio: {cropped_audio_path}")
                        else:
                            # Fallback: show original if no cropped version
                            selected_audio_path = audio_path
                            audio_label = "🎵 **Original Audio** (No cropped version available)"
                            logger.debug(f"🎵 Fallback: showing original audio (no cropped version): {audio_path}")
                        
                        # Display audio with label and cache-busting
                        st.write(audio_label)
                        audio_url = f"{BACKEND_PUBLIC_URL}/audio/{selected_audio_path}{cache_buster}"
                        st.audio(audio_url, format="audio/wav")
                        logger.debug(f"🎵 Audio URL: {audio_url}")
                        
                        # Show additional info in debug mode or when both versions exist
                        if debug_mode and cropped_audio_path:
                            st.caption(f"💡 Cropped version available: {cropped_audio_path}")
                        elif not debug_mode and cropped_audio_path:
                            st.caption(f"💡 Enable debug mode to hear original with silence")

                st.divider()
    elif conversations is not None:
        st.info("No conversations found. The backend is connected but the database might be empty.")
        logger.info("📊 No conversations found in database")

with tab_mem:
    logger.debug("🧠 Loading memories tab...")
    st.header("Memories & Action Items")
    
    # Use session state for selected user if available
    default_user = st.session_state.get('selected_user', '')
    
    # User selection for memories and action items
    col1, col2 = st.columns([2, 1])
    with col1:
        user_id_input = st.text_input("Enter username to view memories & action items:", 
                                    value=default_user,
                                    placeholder="e.g., john_doe, alice123")
    with col2:
        st.write("")  # Spacer
        refresh_mem_btn = st.button("Load Data", key="refresh_memories")
    
    # Clear the session state after using it
    if 'selected_user' in st.session_state:
        del st.session_state['selected_user']

    if refresh_mem_btn:
        logger.info("🔄 Manual memories refresh requested")
        st.rerun()

    # Get memories and action items based on user selection
    if user_id_input.strip():
        logger.info(f"🧠 Loading data for user: {user_id_input.strip()}")
        st.info(f"Showing data for user: **{user_id_input.strip()}**")
        
        # Load both memories and action items
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.spinner("Loading memories..."):
                logger.debug(f"📡 Fetching memories for user: {user_id_input.strip()}")
                memories_response = get_data(f"/api/memories?user_id={user_id_input.strip()}")
        
        with col2:
            with st.spinner("Loading action items..."):
                logger.debug(f"📡 Fetching action items for user: {user_id_input.strip()}")
                action_items_response = get_data(f"/api/action-items?user_id={user_id_input.strip()}")
        
        # Handle the API response format with "results" wrapper for memories
        if memories_response and isinstance(memories_response, dict) and "results" in memories_response:
            memories = memories_response["results"]
            logger.debug(f"🧠 Memories response has 'results' wrapper, extracted {len(memories)} memories")
        else:
            memories = memories_response
            logger.debug(f"🧠 Memories response format: {type(memories_response)}")
            
        # Handle action items response
        if action_items_response and isinstance(action_items_response, dict) and "action_items" in action_items_response:
            action_items = action_items_response["action_items"]
            logger.debug(f"🎯 Action items response has 'action_items' wrapper, extracted {len(action_items)} items")
        else:
            action_items = action_items_response if action_items_response else []
            logger.debug(f"🎯 Action items response format: {type(action_items_response)}")
    else:
        # Show instruction to enter a username
        memories = None
        action_items = None
        logger.debug("👆 No user ID provided, showing instructions")
        st.info("👆 Please enter a username above to view their memories and action items.")
        st.markdown("💡 **Tip:** You can find existing usernames in the 'User Management' tab.")

    # Display Memories Section
    if memories is not None:
        logger.debug("🧠 Displaying memories section...")
        st.subheader("🧠 Discovered Memories")
        
        if memories:
            logger.info(f"🧠 Displaying {len(memories)} memories for user {user_id_input.strip()}")
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
                logger.debug("🧠 Using 'memory' field for display")
            elif "text" in df.columns:
                display_cols["text"] = "Memory"
                logger.debug("🧠 Using 'text' field for display")
            
            # Filter for columns that exist in the dataframe
            cols_to_display = [col for col in display_cols.keys() if col in df.columns]
            
            if cols_to_display:
                logger.debug(f"🧠 Displaying columns: {cols_to_display}")
                st.dataframe(
                    df[cols_to_display].rename(columns=display_cols),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show additional details
                st.caption(f"📊 Found **{len(memories)}** memories for user **{user_id_input.strip()}**")
            else:
                logger.error(f"⚠️ Unexpected memory data format - missing expected fields. Available columns: {list(df.columns)}")
                st.error("⚠️ Unexpected memory data format - missing expected fields")
                st.write("Debug info - Available columns:", list(df.columns))
        else:
            logger.info(f"🧠 No memories found for user {user_id_input.strip()}")
            st.info("No memories found for this user.")
    
    # Display Action Items Section
    if action_items is not None:
        logger.debug("🎯 Displaying action items section...")
        st.subheader("🎯 Action Items")
        
        if action_items:
            logger.info(f"🎯 Displaying {len(action_items)} action items for user {user_id_input.strip()}")
            
            # Status filter for action items
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                status_filter = st.selectbox(
                    "Filter by status:",
                    options=["All", "open", "in_progress", "completed", "cancelled"],
                    index=0,
                    key="action_items_filter"
                )
            with col2:
                show_stats = st.button("📊 Show Stats", key="show_action_stats")
            with col3:
                # Manual action item creation button
                if st.button("➕ Add Item", key="add_action_item"):
                    logger.info("➕ Manual action item creation requested")
                    st.session_state['show_add_action_item'] = True
            
            # Filter action items by status
            if status_filter != "All":
                filtered_items = [item for item in action_items if item.get('status') == status_filter]
                logger.debug(f"🎯 Filtered action items by status '{status_filter}': {len(filtered_items)} items")
            else:
                filtered_items = action_items
                logger.debug(f"🎯 Showing all action items: {len(filtered_items)} items")
            
            # Show statistics if requested
            if show_stats:
                logger.info("📊 Action items statistics requested")
                stats_response = get_data(f"/api/action-items/stats?user_id={user_id_input.strip()}")
                if stats_response and "statistics" in stats_response:
                    stats = stats_response["statistics"]
                    logger.debug(f"📊 Action items statistics: {stats}")
                    
                    # Display stats in columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total", stats["total"])
                        st.metric("Open", stats["open"])
                    with col2:
                        st.metric("In Progress", stats["in_progress"])
                        st.metric("Completed", stats["completed"])
                    with col3:
                        st.metric("Cancelled", stats["cancelled"])
                        st.metric("Overdue", stats.get("overdue", 0))
                    with col4:
                        st.write("**By Priority:**")
                        for priority, count in stats.get("by_priority", {}).items():
                            if count > 0:
                                st.write(f"• {priority.title()}: {count}")
                    
                    # Assignee breakdown
                    if stats.get("by_assignee"):
                        st.write("**By Assignee:**")
                        assignee_df = pd.DataFrame(list(stats["by_assignee"].items()), columns=["Assignee", "Count"])
                        st.dataframe(assignee_df, hide_index=True, use_container_width=True)
                else:
                    logger.warning("📊 Action items statistics not available")
            
            # Manual action item creation form
            if st.session_state.get('show_add_action_item', False):
                logger.debug("➕ Showing action item creation form")
                with st.expander("➕ Create New Action Item", expanded=True):
                    with st.form("create_action_item"):
                        description = st.text_input("Description*:", placeholder="e.g., Send quarterly report to management")
                        col1, col2 = st.columns(2)
                        with col1:
                            assignee = st.text_input("Assignee:", placeholder="e.g., john_doe", value="unassigned")
                            priority = st.selectbox("Priority:", options=["high", "medium", "low", "not_specified"], index=1)
                        with col2:
                            due_date = st.text_input("Due Date:", placeholder="e.g., Friday, 2024-01-15", value="not_specified")
                            context = st.text_input("Context:", placeholder="e.g., Mentioned in team meeting")
                        
                        submitted = st.form_submit_button("Create Action Item")
                        
                        if submitted:
                            logger.info(f"➕ Creating action item for user {user_id_input.strip()}")
                            if description.strip():
                                create_data = {
                                    "description": description.strip(),
                                    "assignee": assignee.strip() if assignee.strip() else "unassigned",
                                    "due_date": due_date.strip() if due_date.strip() else "not_specified",
                                    "priority": priority,
                                    "context": context.strip()
                                }
                                
                                try:
                                    logger.debug(f"📤 Creating action item with data: {create_data}")
                                    response = requests.post(
                                        f"{BACKEND_API_URL}/api/action-items",
                                        params={"user_id": user_id_input.strip()},
                                        json=create_data
                                    )
                                    response.raise_for_status()
                                    result = response.json()
                                    st.success(f"✅ Action item created: {result['action_item']['description']}")
                                    logger.info(f"✅ Action item created successfully: {result['action_item']['description']}")
                                    st.session_state['show_add_action_item'] = False
                                    st.rerun()
                                except requests.exceptions.RequestException as e:
                                    logger.error(f"❌ Error creating action item: {e}")
                                    st.error(f"Error creating action item: {e}")
                            else:
                                logger.warning("⚠️ Action item creation attempted without description")
                                st.error("Please enter a description for the action item")
                    
                    if st.button("❌ Cancel", key="cancel_add_action"):
                        logger.debug("❌ Action item creation cancelled")
                        st.session_state['show_add_action_item'] = False
                        st.rerun()
            
            # Display action items
            if filtered_items:
                logger.debug(f"🎯 Displaying {len(filtered_items)} filtered action items")
                st.write(f"**Showing {len(filtered_items)} action items** (filtered by: {status_filter})")
                
                for i, item in enumerate(filtered_items):
                    logger.debug(f"🎯 Processing action item {i+1}: {item.get('description', 'No description')[:50]}...")
                    
                    with st.container():
                        # Create columns for action item display
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            # Description with status badge
                            status = item.get('status', 'open')
                            status_emoji = {
                                'open': '🔵',
                                'in_progress': '🟡', 
                                'completed': '✅',
                                'cancelled': '❌'
                            }.get(status, '🔵')
                            
                            st.write(f"**{status_emoji} {item.get('description', 'No description')}**")
                            
                            # Additional details
                            details = []
                            if item.get('assignee') and item.get('assignee') != 'unassigned':
                                details.append(f"👤 {item['assignee']}")
                            if item.get('due_date') and item.get('due_date') != 'not_specified':
                                details.append(f"📅 {item['due_date']}")
                            if item.get('priority') and item.get('priority') != 'not_specified':
                                priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(item['priority'], '⚪')
                                details.append(f"{priority_emoji} {item['priority']}")
                            if item.get('context'):
                                details.append(f"💭 {item['context']}")
                            
                            if details:
                                st.caption(" | ".join(details))
                            
                            # Creation info
                            created_at = item.get('created_at')
                            if created_at:
                                try:
                                    if isinstance(created_at, (int, float)):
                                        created_time = datetime.fromtimestamp(created_at)
                                    else:
                                        created_time = pd.to_datetime(created_at)
                                    st.caption(f"Created: {created_time.strftime('%Y-%m-%d %H:%M:%S')}")
                                except:
                                    st.caption(f"Created: {created_at}")
                        
                        with col2:
                            # Status update
                            new_status = st.selectbox(
                                "Status:",
                                options=["open", "in_progress", "completed", "cancelled"],
                                index=["open", "in_progress", "completed", "cancelled"].index(status),
                                key=f"status_{i}_{item.get('memory_id', i)}"
                            )
                            
                            if new_status != status:
                                if st.button("Update", key=f"update_{i}_{item.get('memory_id', i)}"):
                                    memory_id = item.get('memory_id')
                                    if memory_id:
                                        logger.info(f"🔄 Updating action item {memory_id} status from {status} to {new_status}")
                                        try:
                                            response = requests.put(
                                                f"{BACKEND_API_URL}/api/action-items/{memory_id}",
                                                json={"status": new_status}
                                            )
                                            response.raise_for_status()
                                            st.success(f"Status updated to {new_status}")
                                            logger.info(f"✅ Action item status updated successfully")
                                            st.rerun()
                                        except requests.exceptions.RequestException as e:
                                            logger.error(f"❌ Error updating action item status: {e}")
                                            st.error(f"Error updating status: {e}")
                                    else:
                                        logger.error(f"❌ No memory ID found for action item")
                                        st.error("No memory ID found for this action item")
                        
                        with col3:
                            # Delete button
                            if st.button("🗑️ Delete", key=f"delete_{i}_{item.get('memory_id', i)}", type="secondary"):
                                memory_id = item.get('memory_id')
                                if memory_id:
                                    logger.info(f"🗑️ Deleting action item {memory_id}")
                                    try:
                                        response = requests.delete(f"{BACKEND_API_URL}/api/action-items/{memory_id}")
                                        response.raise_for_status()
                                        st.success("Action item deleted")
                                        logger.info(f"✅ Action item deleted successfully")
                                        st.rerun()
                                    except requests.exceptions.RequestException as e:
                                        logger.error(f"❌ Error deleting action item: {e}")
                                        st.error(f"Error deleting action item: {e}")
                                else:
                                    logger.error(f"❌ No memory ID found for action item")
                                    st.error("No memory ID found for this action item")
                        
                        st.divider()
                
                st.caption(f"💡 **Tip:** Action items are automatically extracted from conversations at the end of each session")
            else:
                if status_filter == "All":
                    logger.info(f"🎯 No action items found for user {user_id_input.strip()}")
                    st.info("No action items found for this user.")
                else:
                    logger.info(f"🎯 No action items found with status '{status_filter}' for user {user_id_input.strip()}")
                    st.info(f"No action items found with status '{status_filter}' for this user.")
        else:
            logger.info(f"🎯 No action items found for user {user_id_input.strip()}")
            st.info("No action items found for this user.")
            
            # Show option to create manual action item even when none exist
            if user_id_input.strip() and st.button("➕ Create First Action Item", key="create_first_item"):
                logger.info("➕ Creating first action item for user")
                st.session_state['show_add_action_item'] = True
                st.rerun()

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
    
    st.subheader("🔒 Close Current Conversation")
    st.write("Close the current active conversation for any connected client.")
    
    # Get active clients for the dropdown
    active_clients_data = get_data("/api/active_clients")
    
    if active_clients_data and active_clients_data.get("clients"):
        clients = active_clients_data["clients"]
        
        # Filter to only clients with active conversations
        active_conversations = {
            client_id: client_info 
            for client_id, client_info in clients.items() 
            if client_info.get("has_active_conversation", False)
        }
        
        if active_conversations:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_client = st.selectbox(
                    "Select client to close conversation:",
                    options=list(active_conversations.keys()),
                    format_func=lambda x: f"{x} (UUID: {active_conversations[x].get('current_audio_uuid', 'N/A')[:8]}...)"
                )
            
            with col2:
                st.write("")  # Spacer
                close_conversation_btn = st.button("🔒 Close Conversation", key="close_conv_main", type="primary")
            
            if close_conversation_btn and selected_client:
                result = post_data("/api/close_conversation", {"client_id": selected_client})
                if result:
                    st.success(f"✅ Successfully closed conversation for client '{selected_client}'!")
                    st.info(f"📋 {result.get('message', 'Conversation closed')}")
                    time.sleep(1)  # Brief pause before refresh
                    st.rerun()
                else:
                    st.error(f"❌ Failed to close conversation for client '{selected_client}'")
        else:
            st.info("🔍 No clients with active conversations found")
            
        # Show all clients status
        with st.expander("All Connected Clients Status"):
            for client_id, client_info in clients.items():
                status_icon = "🟢" if client_info.get("has_active_conversation", False) else "⚪"
                st.write(f"{status_icon} **{client_id}** - {'Active conversation' if client_info.get('has_active_conversation', False) else 'No active conversation'}")
                if client_info.get("current_audio_uuid"):
                    st.caption(f"   Audio UUID: {client_info['current_audio_uuid']}")
    else:
        st.info("🔍 No active clients found")
    
    st.divider()
    
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
