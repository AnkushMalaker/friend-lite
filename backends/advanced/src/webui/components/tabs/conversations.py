"""
Conversations tab component for the Streamlit UI
"""

import logging
import random
import time
from datetime import datetime

import requests
import streamlit as st

from ..utils import get_data

logger = logging.getLogger("streamlit-ui")


def show_conversations_tab():
    """Display the conversations tab with full functionality"""
    logger.debug("🗨️ Loading conversations tab...")
    st.header("Latest Conversations")

    # Initialize session state for refresh tracking
    if "refresh_timestamp" not in st.session_state:
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
        debug_mode = st.checkbox(
            "🔧 Debug Mode",
            help="Show original audio files instead of cropped versions",
            key="debug_mode",
        )
        if debug_mode:
            logger.debug("🔧 Debug mode enabled")

    # Generate cache-busting parameter based on session state
    if st.session_state.refresh_timestamp > 0:
        random_component = getattr(st.session_state, "refresh_random", 0)
        cache_buster = f"?t={st.session_state.refresh_timestamp}&r={random_component}"
        st.info("🔄 Audio files refreshed - cache cleared for latest versions")
        logger.info("🔄 Audio cache busting applied")
    else:
        cache_buster = ""

    logger.debug("📡 Fetching conversations data...")
    conversations = get_data("/api/conversations", require_auth=True)

    if conversations:
        logger.info(
            f"📊 Loaded {len(conversations) if isinstance(conversations, list) else 'grouped'} conversations"
        )

        # Check if conversations is the new grouped format or old format
        if isinstance(conversations, dict) and "conversations" in conversations:
            # New grouped format
            logger.debug("📊 Processing conversations in new grouped format")
            conversations_data = conversations["conversations"]

            for client_id, client_conversations in conversations_data.items():
                logger.debug(
                    f"👤 Processing conversations for client: {client_id} ({len(client_conversations)} conversations)"
                )
                st.subheader(f"👤 {client_id}")

                for convo in client_conversations:
                    _display_conversation(convo, debug_mode, cache_buster)
        else:
            # Old format - single list of conversations
            logger.debug("📊 Processing conversations in old format")
            for convo in conversations:
                _display_conversation(convo, debug_mode, cache_buster)
    else:
        st.info("No conversations found. Start a conversation to see results here!")


def _display_conversation(convo, debug_mode, cache_buster):
    """Display a single conversation"""
    logger.debug(f"🗨️ Processing conversation: {convo.get('audio_uuid', 'unknown')}")

    col1, col2 = st.columns([1, 4])
    with col1:
        # Format timestamp for better readability
        ts = datetime.fromtimestamp(convo["timestamp"])
        st.write("**Timestamp:**")
        st.write(ts.strftime("%Y-%m-%d %H:%M:%S"))

        # Show client_id or Audio UUID
        client_id = convo.get("client_id")
        audio_uuid = convo.get("audio_uuid", "N/A")

        if client_id and not client_id.startswith("client_"):
            st.write("**User ID:**")
            st.write(f"👤 `{client_id}`")
        else:
            st.write("**Audio UUID:**")
            st.code(audio_uuid, language=None)

        # Show identified speakers
        speakers = convo.get("speakers_identified", [])
        if speakers:
            st.write("**Speakers:**")
            for speaker in speakers:
                st.write(f"🎤 `{speaker}`")
            logger.debug(f"🎤 Speakers identified: {speakers}")

        # Show audio duration info if available
        cropped_duration = convo.get("cropped_duration")
        if cropped_duration:
            st.write("**Cropped Duration:**")
            st.write(f"⏱️ {cropped_duration:.1f}s")

            # Show speech segments count
            speech_segments = convo.get("speech_segments", [])
            if speech_segments:
                st.write("**Speech Segments:**")
                st.write(f"🗣️ {len(speech_segments)} segments")
                logger.debug(f"🗣️ Speech segments: {len(speech_segments)}")

    with col2:
        # Display conversation transcript
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
                unsafe_allow_html=True,
            )
        else:
            # Fallback for old format
            old_transcript = convo.get("transcription", "No transcript available.")
            st.text_area(
                "Transcription",
                old_transcript,
                height=150,
                disabled=True,
                key=f"transcript_{convo.get('_id', convo.get('audio_uuid', 'unknown'))}",
            )

        # Audio display logic
        _display_audio(convo, debug_mode, cache_buster)

        # Display memory information
        _display_memories(convo)

    st.divider()


def _display_audio(convo, debug_mode, cache_buster):
    """Display audio player for conversation"""
    audio_path = convo.get("audio_path")
    cropped_audio_path = convo.get("cropped_audio_path")
    backend_public_url = st.session_state.get(
        "backend_public_url", st.session_state.get("backend_api_url")
    )

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
        audio_url = f"{backend_public_url}/audio/{selected_audio_path}{cache_buster}"

        # Serve audio directly to browser (no server-side accessibility check)
        try:
            st.audio(audio_url, format="audio/wav")
            logger.debug(f"🎵 Audio URL served: {audio_url}")
        except Exception as e:
            st.error(f"❌ Cannot reach audio file: {str(e)}")
            st.code(f"URL: {audio_url}")
            logger.error(f"🎵 Audio URL error: {audio_url} - {e}")

        # Show additional info in debug mode or when both versions exist
        if debug_mode and cropped_audio_path:
            st.caption(f"💡 Cropped version available: {cropped_audio_path}")
        elif not debug_mode and cropped_audio_path:
            st.caption("💡 Enable debug mode to hear original with silence")


def _display_memories(convo):
    """Display memory information for conversation"""
    memories = convo.get("memories", [])
    if memories:
        st.write("**🧠 Memories Created:**")
        memory_count = len(memories)
        st.write(
            f"📊 {memory_count} memory{'ies' if memory_count != 1 else ''} extracted from this conversation"
        )

        # Show memory details in an expandable section
        with st.expander(f"📋 View Memory Details ({memory_count} items)", expanded=False):
            # Get memory content from API
            user_memories_response = get_data(
                "/api/memories/unfiltered?limit=500", require_auth=True
            )
            memory_contents = {}

            if user_memories_response and "memories" in user_memories_response:
                for mem in user_memories_response["memories"]:
                    memory_contents[mem.get("id")] = mem.get("memory", "No content available")

            for i, memory in enumerate(memories):
                memory_id = memory.get("memory_id", "Unknown")
                status = memory.get("status", "unknown")
                created_at = memory.get("created_at", "Unknown")

                # Get actual memory content
                memory_text = memory_contents.get(memory_id, "Memory content not found")

                # Display each memory with content
                st.write(f"**Memory {i+1}:**")

                # Show memory content in a highlighted box
                if memory_text and memory_text not in [
                    "Memory content not found",
                    "No content available",
                ]:
                    # Check if this is a transcript-based fallback memory
                    if str(memory_id).startswith("transcript_") or memory_text.startswith(
                        "Conversation transcript:"
                    ):
                        st.warning(f"📝 **Transcript-based memory:** {memory_text}")
                        st.caption(
                            "💡 This memory was created from the full transcript when LLM extraction returned no results"
                        )
                    else:
                        st.info(f"💭 {memory_text}")
                else:
                    st.warning(f"🔍 ID: `{memory_id}`")
                    st.caption(
                        "Memory content not available - this may be a transcript-based fallback"
                    )

                st.caption(f"📅 Created: {created_at}")

                # Show status badge
                if status == "created":
                    st.success(f"✅ {status}")
                else:
                    st.info(f"ℹ️ {status}")

                if i < len(memories) - 1:  # Add separator between memories
                    st.markdown("---")
    else:
        # Show when no memories are available
        if convo.get("has_memory") is False:
            st.caption("🔍 No memories extracted from this conversation yet")
