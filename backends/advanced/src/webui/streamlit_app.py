"""
Friend-Lite Streamlit Dashboard (Modular Version)
Web interface for managing conversations, memories, users, and system monitoring.
"""
import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from components.auth import init_auth_state, check_auth_from_url, show_auth_sidebar
from components.health import get_system_health
from components.tabs.upload import show_upload_tab
from components.tabs.conversations import show_conversations_tab
from components.tabs.memories import show_memories_tab
from components.tabs.user_management import show_user_management_tab
from components.tabs.conversation_management import show_conversation_management_tab
from components.tabs.system_state import show_system_state_tab

load_dotenv()

# Create logs directory for Streamlit app
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure comprehensive logging for Streamlit app
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "streamlit.log")],
)

logger = logging.getLogger("streamlit-ui")
logger.info("🚀 Starting Friend-Lite Streamlit Dashboard (Modular Version)")

# ---- Configuration ---- #
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://192.168.0.110:8000")
BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", BACKEND_API_URL)

logger.info(
    f"🔧 Configuration loaded - Backend API: {BACKEND_API_URL}, Public URL: {BACKEND_PUBLIC_URL}"
)

# Store backend URLs in session state for components
if "backend_api_url" not in st.session_state:
    st.session_state.backend_api_url = BACKEND_API_URL
if "backend_public_url" not in st.session_state:
    st.session_state.backend_public_url = BACKEND_PUBLIC_URL

# ---- Streamlit App Configuration ---- #
logger.info("🎨 Configuring Streamlit app...")

st.set_page_config(
    page_title="Friend-Lite Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": """
        # Friend-Lite Dashboard
        
        **Version**: Modular v1.0
        
        A comprehensive web interface for managing your AI-powered personal audio system.
        
        **Features**:
        - 🗨️ Conversation Management  
        - 🧠 Memory Extraction & Search
        - 👥 User Management
        - 📁 Audio File Upload
        - 🔧 System Monitoring
        
        Built with ❤️ using Streamlit
        """,
    },
)

# ---- Initialize Authentication ---- #
init_auth_state()

# Check for authentication from URL parameters (for external auth flows)
if not st.session_state.get("authenticated", False):
    check_auth_from_url(BACKEND_API_URL)

# ---- Sidebar with Authentication and Health Checks ---- #
logger.info("🔧 Setting up sidebar...")

# Show authentication status and controls
show_auth_sidebar(BACKEND_API_URL)

# Show system health in sidebar
with st.sidebar:
    st.header("🔍 System Health")
    
    # Health check with auto-refresh
    if st.button("🔄 Refresh Health", use_container_width=True):
        st.cache_data.clear()  # Clear health check cache
        st.rerun()

    # Get system health data
    with st.spinner("Checking system health..."):
        health_data = get_system_health(BACKEND_API_URL)
    
    if health_data:
        overall_status = health_data.get("status", "unknown")
        if overall_status == "healthy":
            st.success("✅ System Healthy")
        elif overall_status == "partial":
            st.warning("⚠️ Some Issues")
        else:
            st.error("❌ System Issues")
            
        # Show brief service status
        services = health_data.get("services", {})
        if services:
            healthy_count = sum(1 for s in services.values() if s.get("healthy", False))
            total_count = len(services)
            st.caption(f"Services: {healthy_count}/{total_count} healthy")
    else:
        st.error("❌ Health check failed")

    st.divider()

# ---- Main Content Area ---- #
logger.info("🎨 Setting up main content area...")

# Page header
st.title("🎵 Friend-Lite Dashboard")
st.caption("AI-powered personal audio system management")

# Check if user is authenticated and determine admin status
is_authenticated = st.session_state.get("authenticated", False)
is_admin = False

if is_authenticated:
    user_info = st.session_state.get("user_info", {})
    is_admin = user_info.get("is_superuser", False) if isinstance(user_info, dict) else False
    user_name = user_info.get('name', 'Unknown') if user_info else 'Unknown'
    logger.info(f"👤 User authenticated: {user_name}, Admin: {is_admin}")

# Show authentication token preview if available
if is_authenticated:
    with st.expander("🔐 Authentication Info", expanded=False):
        user_info = st.session_state.get("user_info", {})
        st.info(f"**Welcome:** {user_info.get('name', 'User')}")
        st.info(f"**Auth Method:** {st.session_state.get('auth_method', 'unknown').title()}")
        
        if st.session_state.get("auth_token"):
            token_preview = st.session_state.auth_token[:20] + "..."
            st.caption(f"Token: {token_preview}")

# Create tabs based on admin status
if is_admin:
    tab_convos, tab_mem, tab_users, tab_manage, tab_upload, tab_debug = st.tabs([
        "Conversations",
        "Memories", 
        "User Management",
        "Conversation Management",
        "📁 Upload Audio",
        "🔧 System State",
    ])
else:
    tab_convos, tab_mem, tab_users, tab_manage = st.tabs([
        "Conversations",
        "Memories",
        "User Management", 
        "Conversation Management"
    ])
    tab_upload = None
    tab_debug = None

# Tab implementations using modular components

with tab_convos:
    show_conversations_tab()

with tab_mem:
    show_memories_tab()

with tab_users:
    show_user_management_tab()

with tab_manage:
    show_conversation_management_tab()

# Upload tab (only for admins)
if tab_upload is not None:
    with tab_upload:
        logger.debug("📁 Loading upload tab...")
        show_upload_tab()

# System debug tab (only for admins)
if tab_debug is not None:
    with tab_debug:
        show_system_state_tab()

# Footer
st.divider()
from datetime import datetime
st.caption(f"🚀 Friend-Lite Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

logger.info("✅ Streamlit dashboard loaded successfully")