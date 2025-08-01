"""
Friend-Lite Streamlit Dashboard (Modular Version)
Web interface for managing conversations, memories, users, and system monitoring.
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from advanced_omi_backend.debug_system_tracker import get_debug_tracker
from components.auth import init_auth_state, check_auth_from_url, show_auth_sidebar
from components.health import get_system_health
from components.utils import get_data
from components.tabs.upload import show_upload_tab

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

# Store backend URL in session state for components
if "backend_api_url" not in st.session_state:
    st.session_state.backend_api_url = BACKEND_API_URL

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
    
    # Get and display health status
    health_data = get_system_health(BACKEND_API_URL)
    
    if health_data:
        overall_status = health_data.get("status", "unknown")
        if overall_status == "healthy":
            st.success("✅ All systems operational")
        elif overall_status == "partial":
            st.warning("⚠️ Some services degraded")
        else:
            st.error("❌ System issues detected")
            
        # Show individual service status
        services = health_data.get("services", {})
        if services:
            with st.expander("Service Details"):
                for service, details in services.items():
                    status_text = details.get("status", "Unknown")
                    is_healthy = details.get("healthy", False)
                    
                    if is_healthy:
                        st.success(f"✅ {service.title()}")
                    else:
                        st.error(f"❌ {service.title()}")
                    st.caption(status_text)
    else:
        st.error("❌ Cannot reach backend")

# ---- Main Content ---- #
logger.info("📄 Setting up main content area...")

# Add app header
st.title("🎵 Friend-Lite Dashboard")
st.caption(f"Connected to backend: `{BACKEND_PUBLIC_URL}`")

# Get user info and determine admin status
user_info = st.session_state.get("user_info", {})
is_admin = False

if st.session_state.get("authenticated", False):
    # Simple admin check - in a real app you'd check this properly via the backend
    user_email = user_info.get("email", "")
    is_admin = "admin" in user_email.lower()
    
    welcome_msg = f"Welcome back, **{user_info.get('name', 'User')}**!"
    if is_admin:
        welcome_msg += " (Admin)"
    st.success(welcome_msg)
else:
    st.info("🔒 Please log in using the sidebar to access dashboard features.")

# Show debug user info for admins
if is_admin:
    st.sidebar.caption(f"🔧 Admin status: {'✅ Admin' if is_admin else '❌ Regular user'}")
    
    with st.sidebar.expander("🔧 Debug User Info", expanded=False):
        st.json(user_info)
        
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

# For now, show placeholder content for the existing tabs
# These will be replaced with actual modular components in subsequent commits

with tab_convos:
    logger.debug("🗨️ Loading conversations tab...")
    st.header("Latest Conversations")
    st.info("🚧 This tab will be migrated to a modular component soon.")
    
    if not st.session_state.get("authenticated", False):
        st.warning("🔒 Please log in to view conversations.")
    else:
        # Placeholder - will be replaced with actual conversations component
        st.write("Conversations will be displayed here after migration.")

with tab_mem:
    logger.debug("🧠 Loading memories tab...")
    st.header("Memories")
    st.info("🚧 This tab will be migrated to a modular component soon.")
    
    if not st.session_state.get("authenticated", False):
        st.warning("🔒 Please log in to view memories.")
    else:
        # Placeholder - will be replaced with actual memories component  
        st.write("Memories will be displayed here after migration.")

with tab_users:
    logger.debug("👥 Loading users tab...")
    st.header("User Management")
    st.info("🚧 This tab will be migrated to a modular component soon.")
    
    if not st.session_state.get("authenticated", False):
        st.warning("🔒 Please log in to access user management.")
    else:
        # Placeholder - will be replaced with actual users component
        st.write("User management will be displayed here after migration.")

with tab_manage:
    logger.debug("🔧 Loading management tab...")
    st.header("Conversation Management")
    st.info("🚧 This tab will be migrated to a modular component soon.")
    
    if not st.session_state.get("authenticated", False):
        st.warning("🔒 Please log in to access conversation management.")
    else:
        # Placeholder - will be replaced with actual management component
        st.write("Conversation management will be displayed here after migration.")

# New upload tab (only for admins)
if tab_upload is not None:
    with tab_upload:
        logger.debug("📁 Loading upload tab...")
        show_upload_tab()

# System debug tab (only for admins)
if tab_debug is not None:
    with tab_debug:
        logger.debug("🔧 Loading debug tab...")
        st.header("🔧 System State & Failure Recovery")
        st.info("🚧 This tab will be migrated to a modular component soon.")
        
        if not st.session_state.get("authenticated", False):
            st.warning("🔒 Please log in to access system monitoring features")
        else:
            # Placeholder - will be replaced with actual debug component
            st.write("System monitoring will be displayed here after migration.")

# Footer
st.divider()
st.caption(f"🚀 Friend-Lite Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("📡 Modular architecture - easier to maintain and extend!")

logger.info("✅ Streamlit dashboard loaded successfully")