"""
Memories tab component for the Streamlit UI
"""
import logging

import pandas as pd
import streamlit as st

from ..utils import get_data

logger = logging.getLogger("streamlit-ui")

def show_memories_tab():
    """Display the memories tab with full functionality"""
    logger.debug("🧠 Loading memories tab...")
    st.header("Memories")

    # Use session state for selected user if available
    default_user = st.session_state.get("selected_user", "")

    # User selection for memories
    col1, col2 = st.columns([2, 1])
    with col1:
        user_id_input = st.text_input(
            "Enter username to view memories:",
            value=default_user,
            placeholder="e.g., john_doe, alice123",
        )
    with col2:
        st.write("")  # Spacer
        refresh_mem_btn = st.button("Load Data", key="refresh_memories")

    # Clear the session state after using it
    if "selected_user" in st.session_state:
        del st.session_state["selected_user"]

    if refresh_mem_btn:
        logger.info("🔄 Manual memories refresh requested")
        st.rerun()

    # Get memories based on user selection
    if user_id_input.strip():
        logger.info(f"🧠 Loading data for user: {user_id_input.strip()}")
        st.info(f"Showing data for user: **{user_id_input.strip()}**")

        # Load memories
        with st.spinner("Loading memories..."):
            logger.debug(f"📡 Fetching memories for user: {user_id_input.strip()}")
            memories_response = get_data(
                f"/api/memories/unfiltered?user_id={user_id_input.strip()}", require_auth=True
            )

        # Handle the API response format with "results" wrapper for memories
        if (memories_response and isinstance(memories_response, dict) and "results" in memories_response):
            memories = memories_response["results"]
            logger.debug(f"🧠 Memories response has 'results' wrapper, extracted {len(memories)} memories")
        else:
            memories = memories_response
            logger.debug(f"🧠 Memories response format: {type(memories_response)}")

        # Show admin debug section for admin users
        _show_admin_debug_section()

        # Display Memories Section
        if memories is not None:
            _display_memories_section(memories, user_id_input.strip())
        
    else:
        # Show instruction to enter a username
        logger.debug("👆 No user ID provided, showing instructions")
        st.info("👆 Please enter a username above to view their memories.")
        st.markdown("💡 **Tip:** You can find existing usernames in the 'User Management' tab.")
        
        # Show admin debug section for admin users
        _show_admin_debug_section()


def _show_admin_debug_section():
    """Show admin debug section for admin users"""
    if not st.session_state.get("authenticated", False):
        return
        
    user_info = st.session_state.get("user_info", {})

    # Check if user is admin
    is_admin = False
    if isinstance(user_info, dict):
        is_admin = user_info.get("is_superuser", False)

    # Alternative: Check if the token has superuser privileges by trying an admin endpoint
    if not is_admin:
        try:
            test_response = get_data("/api/users", require_auth=True)
            is_admin = test_response is not None
        except:
            pass

    if is_admin:
        st.subheader("🔧 Admin Debug: All Memories")
        logger.debug("🔧 Admin user detected, showing admin debug section")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🧠 View All User Memories (Admin)", key="admin_all_memories"):
                logger.info("📋 Admin: Loading all memories for all users")
                st.session_state["show_admin_memories"] = True

        with col2:
            if st.session_state.get("show_admin_memories", False):
                if st.button("❌ Hide Admin View", key="hide_admin_views"):
                    st.session_state["show_admin_memories"] = False
                    st.rerun()

        # Show admin memories view if requested
        if st.session_state.get("show_admin_memories", False):
            _display_admin_memories()

        st.divider()


def _display_admin_memories():
    """Display admin memories view"""
    with st.spinner("Loading memories..."):
        logger.debug("📋 Fetching memories for admin view")

        # Use the working user memories endpoint since admin is a user too
        user_memories_response = get_data("/api/memories/unfiltered?limit=500", require_auth=True)

        if user_memories_response and "memories" in user_memories_response:
            # Get current user info
            user_info = st.session_state.get("user", {})
            user_id = user_info.get("id", "unknown")
            user_email = user_info.get("email", "unknown")

            memories = user_memories_response["memories"]

            # Format as admin response for compatibility with existing UI
            admin_memories_response = {
                "memories": [
                    {
                        "id": memory.get("id"),
                        "memory": memory.get("memory", "No content"),
                        "user_id": user_id,
                        "owner_email": user_email,
                        "created_at": memory.get("created_at"),
                        "client_id": memory.get("metadata", {}).get("client_id", "unknown"),
                        "metadata": memory.get("metadata", {}),
                    }
                    for memory in memories
                ],
                "total_memories": len(memories),
                "total_users": 1 if memories else 0,
            }
        else:
            admin_memories_response = None

    if admin_memories_response:
        logger.info(
            f"📋 Admin memories: Loaded {admin_memories_response.get('total_memories', 0)} memories from {admin_memories_response.get('total_users', 0)} users"
        )

        # Display summary stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Users", admin_memories_response.get("total_users", 0))
        with col2:
            st.metric("Total Memories", admin_memories_response.get("total_memories", 0))

        st.divider()

        # Display all memories in flat view
        memories = admin_memories_response.get("memories", [])

        if memories:
            st.write("### 🧠 All User Memories")

            # Create a searchable/filterable view
            search_term = st.text_input("🔍 Search memories", placeholder="Enter text to search...")

            if search_term:
                filtered_memories = [
                    m for m in memories
                    if search_term.lower() in m.get("memory", "").lower()
                    or search_term.lower() in m.get("owner_email", "").lower()
                    or search_term.lower() in m.get("user_id", "").lower()
                ]
                st.caption(f"Showing {len(filtered_memories)} memories matching '{search_term}'")
            else:
                filtered_memories = memories
                st.caption(f"Showing all {len(memories)} memories")

            # Display memories in a nice format
            for i, memory in enumerate(filtered_memories[:50]):  # Limit to 50 for performance
                with st.container():
                    # Memory header
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**Memory {i+1}**")
                    with col2:
                        st.caption(f"👤 {memory.get('owner_email', memory.get('user_id', 'Unknown'))}")
                    with col3:
                        st.caption(f"📅 {memory.get('created_at', 'Unknown')}")

                    # Memory content
                    memory_text = memory.get("memory", "No content")
                    st.write(memory_text)

                    # Memory metadata
                    with st.expander("🔍 Memory Details", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**User ID:** {memory.get('user_id', 'Unknown')}")
                            st.write(f"**Owner Email:** {memory.get('owner_email', 'Unknown')}")
                            st.write(f"**Client ID:** {memory.get('client_id', 'Unknown')}")
                        with col2:
                            st.write(f"**Memory ID:** {memory.get('id', memory.get('memory_id', 'Unknown'))}")
                            metadata = memory.get("metadata", {})
                            if metadata:
                                st.write(f"**Source:** {metadata.get('source', 'Unknown')}")

                    st.divider()

            if len(filtered_memories) > 50:
                st.info(f"Showing first 50 memories. Total: {len(filtered_memories)}")

        else:
            st.info("No memories found across all users.")

    else:
        logger.error("❌ Failed to load admin memories")
        st.error("❌ Failed to load admin memories. You may not have admin privileges.")


def _display_memories_section(memories, user_id):
    """Display the main memories section"""
    logger.debug("🧠 Displaying memories section...")
    st.subheader("🧠 Discovered Memories")

    if memories:
        logger.info(f"🧠 Displaying {len(memories)} memories for user {user_id}")

        # Add view options
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"Found **{len(memories)}** memories for user **{user_id}**")
        with col2:
            view_mode = st.selectbox(
                "View Mode:", ["Standard View", "Transcript Analysis"], key="memory_view_mode"
            )

        if view_mode == "Standard View":
            _display_standard_view(memories)
        else:  # Transcript Analysis View
            _display_transcript_analysis_view(user_id)
    else:
        logger.info(f"🧠 No memories found for user {user_id}")
        st.info("No memories found for this user.")


def _display_standard_view(memories):
    """Display memories in standard tabular view"""
    df = pd.DataFrame(memories)

    # Make the dataframe more readable
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Reorder and rename columns for clarity - handle both "memory" and "text" fields
    display_cols = {"id": "Memory ID", "created_at": "Created At"}

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
            hide_index=True,
        )
    else:
        logger.error(
            f"⚠️ Unexpected memory data format - missing expected fields. Available columns: {list(df.columns)}"
        )
        st.error("⚠️ Unexpected memory data format - missing expected fields")
        st.write("Debug info - Available columns:", list(df.columns))


def _display_transcript_analysis_view(user_id):
    """Display memories with transcript analysis"""
    with st.spinner("Loading memories with transcript analysis..."):
        enriched_response = get_data(
            f"/api/memories/with-transcripts?user_id={user_id}", require_auth=True
        )

    if enriched_response:
        enriched_memories = enriched_response.get("memories", [])

        if enriched_memories:
            # Create enhanced dataframe for transcript analysis
            analysis_data = []
            for memory in enriched_memories:
                analysis_data.append({
                    "Audio UUID": (
                        memory.get("audio_uuid", "N/A")[:12] + "..."
                        if memory.get("audio_uuid") else "N/A"
                    ),
                    "Memory Text": (
                        memory.get("memory_text", "")[:100] + "..."
                        if len(memory.get("memory_text", "")) > 100
                        else memory.get("memory_text", "")
                    ),
                    "Transcript": (
                        memory.get("transcript", "")[:100] + "..."
                        if memory.get("transcript") and len(memory.get("transcript", "")) > 100
                        else (memory.get("transcript", "N/A")[:100] if memory.get("transcript") else "N/A")
                    ),
                    "Transcript Chars": memory.get("transcript_length", 0),
                    "Memory Chars": memory.get("memory_length", 0),
                    "Compression %": f"{memory.get('compression_ratio', 0)}%",
                    "Client ID": memory.get("client_id", "N/A"),
                    "Created": (
                        memory.get("created_at", "N/A")[:19] if memory.get("created_at") else "N/A"
                    ),
                })

            # Display the enhanced table
            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)
                st.dataframe(analysis_df, use_container_width=True, hide_index=True)

                # Show detailed expandable views
                st.subheader("📋 Detailed Memory Analysis")

                for i, memory in enumerate(enriched_memories):
                    audio_uuid = memory.get("audio_uuid", "unknown")
                    memory_text = memory.get("memory_text", "")
                    transcript = memory.get("transcript", "")
                    compression_ratio = memory.get("compression_ratio", 0)

                    # Create meaningful title
                    title_text = (
                        memory_text[:50] + "..." if len(memory_text) > 50 else memory_text
                    )
                    if not title_text.strip():
                        title_text = f"Memory {i+1}"

                    with st.expander(
                        f"🧠 {title_text} | {compression_ratio}% compression", expanded=False
                    ):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**🎤 Original Transcript**")
                            if transcript and transcript.strip():
                                st.text_area(
                                    f"Transcript ({len(transcript)} chars):",
                                    value=transcript,
                                    height=200,
                                    disabled=True,
                                    key=f"transcript_{i}",
                                )
                            else:
                                st.info("No transcript available")

                        with col2:
                            st.markdown("**🧠 Extracted Memory**")
                            if memory_text and memory_text.strip():
                                st.text_area(
                                    f"Memory ({len(memory_text)} chars):",
                                    value=memory_text,
                                    height=200,
                                    disabled=True,
                                    key=f"memory_text_{i}",
                                )
                            else:
                                st.warning("No memory text")

                        # Additional details
                        st.markdown("**📊 Metadata**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Audio UUID",
                                (
                                    audio_uuid[:12] + "..."
                                    if audio_uuid and len(audio_uuid) > 12
                                    else audio_uuid or "N/A"
                                ),
                            )
                        with col2:
                            st.metric("Client ID", memory.get("client_id", "N/A"))
                        with col3:
                            st.metric("User Email", memory.get("user_email", "N/A"))
            else:
                st.info("No enriched memory data available")
        else:
            st.info("No memories with transcript data found")
    else:
        st.error("Failed to load enriched memory data")