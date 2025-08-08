"""
Conversation Management tab component for the Streamlit UI
"""

import logging
import time

import requests
import streamlit as st

from ..utils import get_data, post_data, get_auth_headers

logger = logging.getLogger("streamlit-ui")


def show_conversation_management_tab():
    """Display the conversation management tab with full functionality"""
    logger.debug("🔧 Loading management tab...")
    st.header("Conversation Management")

    st.subheader("🔒 Close Current Conversation")

    # Check if user is authenticated and show appropriate message
    if st.session_state.get("authenticated", False):
        user_info = st.session_state.get("user_info", {})
        is_admin = user_info.get("is_superuser", False) if isinstance(user_info, dict) else False

        if is_admin:
            st.write("Close the current active conversation for any connected client.")
        else:
            st.write("Close the current active conversation for your connected clients.")

        # Get active clients for the dropdown
        active_clients_data = get_data("/api/clients/active", require_auth=True)

        if active_clients_data and active_clients_data.get("active_clients"):
            clients = active_clients_data["active_clients"]

            # Filter to only clients with active conversations
            active_conversations = {
                client_info.get("client_id"): client_info
                for client_info in clients
                if client_info.get("has_active_conversation", False)
            }

            if active_conversations:
                col1, col2 = st.columns([3, 1])

                with col1:
                    selected_client = st.selectbox(
                        "Select client to close conversation:",
                        options=list(active_conversations.keys()),
                        format_func=lambda x: f"{x} (UUID: {active_conversations[x].get('current_audio_uuid', 'N/A')[:8]}...)",
                    )

                with col2:
                    st.write("")  # Spacer
                    close_conversation_btn = st.button(
                        "🔒 Close Conversation", key="close_conv_main", type="primary"
                    )

                if close_conversation_btn and selected_client:
                    result = post_data(
                        f"/api/conversations/{selected_client}/close", require_auth=True
                    )
                    if result:
                        st.success(
                            f"✅ Successfully closed conversation for client '{selected_client}'!"
                        )
                        st.info(f"📋 {result.get('message', 'Conversation closed')}")
                        time.sleep(1)  # Brief pause before refresh
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to close conversation for client '{selected_client}'")
            else:
                if len(clients) > 0:
                    st.info("🔍 No clients with active conversations found.")
                    st.caption(
                        "💡 Your connected clients don't have active conversations at the moment."
                    )
                else:
                    st.info("🔍 No connected clients found for your account.")
                    st.caption(
                        "💡 Connect an audio client with your user ID to manage conversations."
                    )

            # Show all clients status (only if there are clients)
            if len(clients) > 0:
                with st.expander("All Connected Clients Status"):
                    for client_info in clients:
                        client_id = client_info.get("client_id")
                        status_icon = (
                            "🟢" if client_info.get("has_active_conversation", False) else "⚪"
                        )
                        st.write(
                            f"{status_icon} **{client_id}** - {'Active conversation' if client_info.get('has_active_conversation', False) else 'No active conversation'}"
                        )
                        if client_info.get("current_audio_uuid"):
                            st.caption(f"   Audio UUID: {client_info['current_audio_uuid']}")

                    # Show ownership info for non-admin users
                    if not is_admin:
                        st.caption(
                            "ℹ️ You can only see and manage clients that belong to your account."
                        )
        else:
            st.info("🔍 No accessible clients found for your account.")
            st.markdown(
                """
            **To connect an audio client:**
            1. Use your user ID when connecting: `user_id=YOUR_USER_ID`
            2. Include your authentication token in the WebSocket connection
            3. Example: `ws://localhost:8000/ws?user_id=YOUR_USER_ID&token=YOUR_TOKEN`
            """
            )

            if st.session_state.get("auth_token"):
                st.info(
                    "💡 Your authentication token is available - see the WebSocket connection info below."
                )
            else:
                st.warning(
                    "⚠️ Please authenticate first to get your token for audio client connections."
                )
    else:
        st.warning("🔒 Authentication required to manage conversations.")
        st.markdown(
            """
        **Please authenticate using the sidebar to:**
        - View your active audio clients
        - Close conversations for your clients
        - Manage your conversation data
        """
        )
        st.info("👆 Use the authentication options in the sidebar to get started.")

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
            result = post_data(
                f"/api/conversations/{audio_uuid_input.strip()}/speakers",
                params={"speaker_id": speaker_id_input.strip()},
                require_auth=True,
            )
            if result:
                st.success(f"Speaker '{speaker_id_input.strip()}' added to conversation!")
        else:
            st.error("Please enter both Audio UUID and Speaker ID")

    st.divider()

    st.subheader("Update Transcript Segment")
    st.write("Modify speaker identification or timing information for transcript segments.")

    col1, col2 = st.columns([1, 1])
    with col1:
        update_audio_uuid = st.text_input(
            "Audio UUID:", placeholder="Enter the audio UUID", key="update_uuid"
        )
        segment_index = st.number_input("Segment Index:", min_value=0, value=0, step=1)
        new_speaker = st.text_input(
            "New Speaker ID (optional):", placeholder="Leave empty to keep current"
        )

    with col2:
        start_time = st.number_input(
            "Start Time (seconds):", min_value=0.0, value=0.0, step=0.1, format="%.1f"
        )
        end_time = st.number_input(
            "End Time (seconds):", min_value=0.0, value=0.0, step=0.1, format="%.1f"
        )
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
                backend_api_url = st.session_state.get(
                    "backend_api_url", "http://192.168.0.110:8000"
                )
                try:
                    response = requests.put(
                        f"{backend_api_url}/api/conversations/{update_audio_uuid.strip()}/transcript/{segment_index}",
                        params=params,
                        headers=get_auth_headers(),
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
    st.markdown(
        """
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
    """
    )

    st.info(
        "💡 **Tip**: You can find Audio UUIDs in the conversation details on the 'Conversations' tab."
    )

    st.divider()

    # Authentication info for WebSocket connections
    st.subheader("🔐 Authentication & WebSocket Connections")
    if st.session_state.get("authenticated", False):
        auth_token = st.session_state.get("auth_token", "")
        st.success(
            "✅ You are authenticated. Audio clients can use your token for WebSocket connections."
        )

        with st.expander("WebSocket Connection Info"):
            st.markdown("**For audio clients, use one of these WebSocket URLs:**")
            st.code(
                f"""
# Opus audio stream (with authentication):
ws://localhost:8000/ws?token={auth_token[:20]}...

# PCM audio stream (with authentication):
ws://localhost:8000/ws_pcm?token={auth_token[:20]}...

# Or include in Authorization header:
Authorization: Bearer {auth_token[:20]}...
            """
            )
            st.caption("⚠️ Keep your token secure and don't share it publicly!")

        st.info("🎵 **Audio clients must now authenticate** to connect to WebSocket endpoints.")
    else:
        st.warning("🔒 WebSocket audio connections now require authentication.")
        st.markdown(
            """
        **Important Changes:**
        - All WebSocket endpoints (`/ws` and `/ws_pcm`) now require authentication
        - Audio clients must include a JWT token in the connection
        - Tokens can be passed via query parameter (`?token=...`) or Authorization header
        - Get a token by logging in via the sidebar or using the backend auth endpoints
        """
        )

        st.info(
            "👆 **Log in using the sidebar** to get your authentication token for audio clients."
        )
