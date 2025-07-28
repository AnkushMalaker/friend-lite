"""Main Streamlit application for speaker recognition system."""

import streamlit as st
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from database import init_db, get_db_session
from database.queries import UserQueries

# Configuration from environment
SPEAKER_SERVICE_URL = os.getenv("SPEAKER_SERVICE_URL", "http://localhost:8001")
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "0.0.0.0")
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

def init_session_state():
    """Initialize session state variables."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False

def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="🎤 Speaker Recognition System",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def sidebar_user_management():
    """Handle user management in sidebar."""
    st.sidebar.header("👤 User Management")
    
    # Initialize database if not done
    if not st.session_state.db_initialized:
        with st.spinner("Initializing database..."):
            init_db()
            st.session_state.db_initialized = True
        st.sidebar.success("Database initialized!")
    
    # User selection/creation
    with st.sidebar.container():
        st.subheader("Select or Create User")
        
        # Get existing users
        db = get_db_session()
        try:
            users = UserQueries.get_all_users(db)
            user_options = ["Create new user..."] + [user.username for user in users]
        finally:
            db.close()
        
        # User selection dropdown
        selected_option = st.selectbox(
            "Choose user:",
            user_options,
            index=0 if not st.session_state.username else (
                user_options.index(st.session_state.username) 
                if st.session_state.username in user_options 
                else 0
            )
        )
        
        # Handle user creation or selection
        if selected_option == "Create new user...":
            new_username = st.text_input(
                "Username:",
                placeholder="Enter a username",
                help="Choose a unique username to identify your speaker data"
            )
            
            if st.button("Create User", type="primary"):
                if new_username and new_username.strip():
                    db = get_db_session()
                    try:
                        user = UserQueries.get_or_create_user(db, new_username.strip())
                        st.session_state.user_id = user.id
                        st.session_state.username = user.username
                        st.sidebar.success(f"✅ Active user: {user.username}")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error creating user: {str(e)}")
                    finally:
                        db.close()
                else:
                    st.sidebar.error("Please enter a valid username")
        else:
            # Existing user selected
            if st.button("Select User", type="primary"):
                db = get_db_session()
                try:
                    user = UserQueries.get_or_create_user(db, selected_option)
                    st.session_state.user_id = user.id
                    st.session_state.username = user.username
                    st.sidebar.success(f"✅ Active user: {user.username}")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error selecting user: {str(e)}")
                finally:
                    db.close()
    
    # Display current user
    if st.session_state.username:
        st.sidebar.info(f"**Current user:** {st.session_state.username}")
        
        # Show user statistics
        db = get_db_session()
        try:
            stats = UserQueries.get_user_stats(db, st.session_state.user_id)
            st.sidebar.metric("Enrolled Speakers", stats["speaker_count"])
            st.sidebar.metric("Annotations", stats["annotation_count"])
        except Exception as e:
            st.sidebar.error(f"Error loading stats: {str(e)}")
        finally:
            db.close()
        
        # Folder location info
        data_dir = Path(__file__).parent / "data"
        st.sidebar.info(f"**Data location:** `{data_dir.absolute()}`")
        
        # Clear user button
        if st.sidebar.button("Switch User"):
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()

def main_content():
    """Display main content area."""
    st.title("🎤 Speaker Recognition System")
    
    if not st.session_state.username:
        st.warning("👈 Please select or create a user in the sidebar to get started.")
        
        # Getting started information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎵 Audio Viewer")
            st.write("""
            - Upload and visualize audio files
            - View waveforms and spectrograms
            - Select segments for analysis
            - Export audio segments
            """)
        
        with col2:
            st.subheader("📝 Annotation Tool")
            st.write("""
            - Label speaker segments
            - Assign known/unknown speakers
            - Quality assessment (CORRECT/INCORRECT)
            - Batch annotation processing
            """)
        
        with col3:
            st.subheader("👥 Speaker Management")
            st.write("""
            - Enroll new speakers
            - Manage speaker profiles
            - View quality metrics
            - Export speaker data
            """)
        
        st.divider()
        
        st.subheader("📖 Quick Start Guide")
        st.markdown("""
        1. **Select a user** in the sidebar (or create a new one)
        2. **Navigate to pages** using the sidebar menu:
           - 🎵 **Audio Viewer**: Upload and explore audio files
           - 📝 **Annotation**: Label speaker segments in your audio
           - 👤 **Enrollment**: Register new speakers in the system
           - 👥 **Speakers**: Manage enrolled speakers and export data
           - 📊 **Analytics**: View system performance and statistics
        3. **Start with Audio Viewer** to upload your first audio file
        4. **Use Annotation tool** to label who is speaking when
        5. **Enroll speakers** to enable automatic recognition
        """)
        
        st.info("💡 **Tip**: The system supports multiple users, so each person can manage their own speaker data independently.")
        
    else:
        # User is logged in - show dashboard
        st.success(f"Welcome back, **{st.session_state.username}**! 👋")
        
        # Quick stats dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        db = get_db_session()
        try:
            stats = UserQueries.get_user_stats(db, st.session_state.user_id)
            
            with col1:
                st.metric("👥 Enrolled Speakers", stats["speaker_count"])
            
            with col2:
                st.metric("📝 Annotations", stats["annotation_count"])
            
            with col3:
                st.metric("📁 Data Location", "Local SQLite")
            
            with col4:
                st.metric("🔧 Status", "Ready")
        
        except Exception as e:
            st.error(f"Error loading dashboard: {str(e)}")
        finally:
            db.close()
        
        st.divider()
        
        # Recent activity or next steps
        if stats["speaker_count"] == 0:
            st.subheader("🚀 Get Started")
            st.markdown("""
            You don't have any speakers enrolled yet. Here's what you can do:
            
            1. **📄 Upload audio files** in the Audio Viewer to explore your data
            2. **👤 Enroll speakers** to start building your speaker database
            3. **📝 Annotate segments** to improve recognition accuracy
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🎵 Go to Audio Viewer", use_container_width=True):
                    st.switch_page("pages/1_audio_viewer.py")
            with col2:
                if st.button("👤 Go to Enrollment", use_container_width=True):
                    st.switch_page("pages/3_enrollment.py")
            with col3:
                if st.button("📝 Go to Annotation", use_container_width=True):
                    st.switch_page("pages/2_annotation.py")
        else:
            st.subheader("🎯 Quick Actions")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🎵 View Audio", use_container_width=True):
                    st.switch_page("pages/1_audio_viewer.py")
            
            with col2:
                if st.button("📝 Annotate", use_container_width=True):
                    st.switch_page("pages/2_annotation.py")
            
            with col3:
                if st.button("👤 Enroll Speaker", use_container_width=True):
                    st.switch_page("pages/3_enrollment.py")
            
            with col4:
                if st.button("👥 Manage Speakers", use_container_width=True):
                    st.switch_page("pages/4_speakers.py")

def main():
    """Main application entry point."""
    setup_page_config()
    init_session_state()
    
    # Sidebar for user management
    sidebar_user_management()
    
    # Main content area
    main_content()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        🎤 Speaker Recognition System v0.1.0 | 
        Built with Streamlit and PyAnnote Audio | 
        <a href='README.detailed.md' target='_blank'>Documentation</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()