"""
System State tab component for the Streamlit UI
"""
import logging
from datetime import datetime

import streamlit as st

from ..health import get_system_health
from ..utils import get_data

logger = logging.getLogger("streamlit-ui")

def show_system_state_tab():
    """Display the system state tab with full functionality"""
    logger.debug("🔧 Loading debug tab...")
    st.header("🔧 System State & Monitoring")
    st.caption("System monitoring and health information")

    # Check authentication like other tabs
    if not st.session_state.get("authenticated", False):
        st.warning("🔒 Please log in to access system monitoring features")
    else:
        # Show immediate system status
        st.info("💡 **System monitoring and health information**")

        # System Health Overview
        st.subheader("🔍 System Health")
        
        # Get system health data
        backend_api_url = st.session_state.get("backend_api_url", "http://192.168.0.110:8000")
        health_data = get_system_health(backend_api_url)
        
        if health_data:
            overall_status = health_data.get("status", "unknown")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if overall_status == "healthy":
                    st.success("✅ System Healthy")
                elif overall_status == "partial":
                    st.warning("⚠️ Partial Issues")
                else:
                    st.error("❌ System Issues")
            
            with col2:
                services = health_data.get("services", {})
                healthy_services = sum(1 for s in services.values() if s.get("healthy", False))
                total_services = len(services)
                st.metric("Services Healthy", f"{healthy_services}/{total_services}")
            
            with col3:
                uptime = health_data.get("uptime", "Unknown")
                st.metric("System Uptime", uptime)
            
            # Service Details
            st.subheader("🔧 Service Status")
            if services:
                for service, details in services.items():
                    status_text = details.get("status", "Unknown")
                    is_healthy = details.get("healthy", False)
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if is_healthy:
                            st.success(f"✅ {service.title()}")
                        else:
                            st.error(f"❌ {service.title()}")
                    with col2:
                        st.write(status_text)
        else:
            st.error("❌ Cannot retrieve system health data")
        
        st.divider()
        
        # API Metrics
        st.subheader("📊 API Metrics")
        metrics_data = get_data("/api/metrics", require_auth=True)
        
        if metrics_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                active_clients = metrics_data.get("active_clients", 0)
                st.metric("Active Clients", active_clients)
            
            with col2:
                active_conversations = metrics_data.get("active_conversations", 0) 
                st.metric("Active Conversations", active_conversations)
            
            with col3:
                total_users = metrics_data.get("total_users", 0)
                st.metric("Total Users", total_users)
            
            # Additional metrics
            if metrics_data.get("processor_status"):
                st.write("**Processor Status:**")
                processor_status = metrics_data["processor_status"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Audio Queue: {processor_status.get('audio_queue_size', 'N/A')}")
                    st.write(f"Memory Queue: {processor_status.get('memory_queue_size', 'N/A')}")
                with col2:
                    st.write(f"Transcription Queue: {processor_status.get('transcript_queue_size', 'N/A')}")
                    st.write(f"Tasks Running: {processor_status.get('running_tasks', 'N/A')}")
        else:
            st.info("Metrics data not available")
        
        st.divider()
        
        # Recent Activity
        st.subheader("📝 Recent Activity")
        
        # Show recent conversations
        recent_conversations = get_data("/api/conversations?limit=5", require_auth=True)
        if recent_conversations:
            st.write("**Recent Conversations:**")
            if isinstance(recent_conversations, dict) and "conversations" in recent_conversations:
                # Grouped format
                conversations_data = recent_conversations["conversations"]
                count = 0
                for client_id, client_conversations in conversations_data.items():
                    for convo in client_conversations[:3]:  # Show max 3 per client
                        if count >= 5:
                            break
                        ts = datetime.fromtimestamp(convo["timestamp"])
                        st.write(f"- {client_id}: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
                        count += 1
                    if count >= 5:
                        break
            else:
                # List format
                for convo in recent_conversations[:5]:
                    ts = datetime.fromtimestamp(convo["timestamp"])
                    client_id = convo.get("client_id", "Unknown")
                    st.write(f"- {client_id}: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("No recent conversations")
        
        st.divider()
        
        # System Information
        st.subheader("ℹ️ System Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Backend Configuration:**")
            st.write(f"API URL: `{backend_api_url}`")
            backend_public_url = st.session_state.get("backend_public_url", backend_api_url)
            st.write(f"Public URL: `{backend_public_url}`")
        
        with col2:
            st.write("**Authentication:**")
            user_info = st.session_state.get("user_info", {})
            st.write(f"User: {user_info.get('name', 'Unknown')}")
            st.write(f"Auth Method: {st.session_state.get('auth_method', 'unknown').title()}")
        
        # Debug Tools
        st.subheader("🔧 Debug Tools")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Cache", help="Clear Streamlit cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
                st.rerun()
                
        with col2:
            if st.button("📊 Refresh Metrics", help="Refresh all metrics"):
                st.rerun()
        
        # Show current session state for debugging (admin only)
        with st.expander("🔍 Session Debug Info", expanded=False):
            st.write("**Session State Keys:**")
            session_keys = list(st.session_state.keys())
            st.write(session_keys)
            
            st.write("**User Info:**")
            st.json(st.session_state.get("user_info", {}))
            
            if st.session_state.get("auth_token"):
                token_preview = st.session_state.auth_token[:30] + "..."
                st.write(f"**Auth Token Preview:** `{token_preview}`")