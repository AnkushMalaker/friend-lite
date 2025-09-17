"""
Conversation controller for handling conversation-related business logic.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi.responses import JSONResponse

from advanced_omi_backend.audio_cropping_utils import (
    _process_audio_cropping_with_relative_timestamps,
)
from advanced_omi_backend.client_manager import (
    ClientManager,
    client_belongs_to_user,
    get_user_clients_all,
)
from advanced_omi_backend.database import AudioChunksRepository, chunks_col, conversations_col, ConversationsRepository
from advanced_omi_backend.users import User

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")

# Initialize chunk repository
chunk_repo = AudioChunksRepository(chunks_col)


async def close_current_conversation(client_id: str, user: User, client_manager: ClientManager):
    """Close the current conversation for a specific client. Users can only close their own conversations."""
    # Validate client ownership
    if not user.is_superuser and not client_belongs_to_user(client_id, user.user_id):
        logger.warning(
            f"User {user.user_id} attempted to close conversation for client {client_id} without permission"
        )
        return JSONResponse(
            content={
                "error": "Access forbidden. You can only close your own conversations.",
                "details": f"Client '{client_id}' does not belong to your account.",
            },
            status_code=403,
        )

    if not client_manager.has_client(client_id):
        return JSONResponse(
            content={"error": f"Client '{client_id}' not found or not connected"},
            status_code=404,
        )

    client_state = client_manager.get_client(client_id)
    if client_state is None:
        return JSONResponse(
            content={"error": f"Client '{client_id}' not found or not connected"},
            status_code=404,
        )

    if not client_state.connected:
        return JSONResponse(
            content={"error": f"Client '{client_id}' is not connected"}, status_code=400
        )

    try:
        # Close the current conversation
        await client_state.close_current_conversation()

        # Reset conversation state but keep client connected
        client_state.current_audio_uuid = None
        client_state.conversation_start_time = time.time()
        client_state.last_transcript_time = None

        logger.info(f"Manually closed conversation for client {client_id} by user {user.id}")

        return JSONResponse(
            content={
                "message": f"Successfully closed current conversation for client '{client_id}'",
                "client_id": client_id,
                "timestamp": int(time.time()),
            }
        )

    except Exception as e:
        logger.error(f"Error closing conversation for client {client_id}: {e}")
        return JSONResponse(
            content={"error": f"Failed to close conversation: {str(e)}"},
            status_code=500,
        )


async def get_conversations(user: User):
    """Get conversations with speech only (speech-driven architecture)."""
    try:
        # Import conversations collection and repository
        conversations_repo = ConversationsRepository(conversations_col)

        # Build query based on user permissions
        if not user.is_superuser:
            # Regular users can only see their own conversations
            user_conversations = await conversations_repo.get_user_conversations(str(user.user_id))
        else:
            # Admins see all conversations
            cursor = conversations_col.find({}).sort("created_at", -1)
            user_conversations = await cursor.to_list(length=None)

        # Group conversations by client_id for backwards compatibility
        conversations = {}
        for conversation in user_conversations:
            client_id = conversation["client_id"]
            if client_id not in conversations:
                conversations[client_id] = []

            # Get audio file paths from audio_chunks collection
            audio_chunk = await chunk_repo.get_chunk_by_audio_uuid(conversation["audio_uuid"])
            audio_path = audio_chunk.get("audio_path") if audio_chunk else None
            cropped_audio_path = audio_chunk.get("cropped_audio_path") if audio_chunk else None

            # Convert conversation to API format
            conversations[client_id].append(
                {
                    "conversation_id": conversation["conversation_id"],
                    "audio_uuid": conversation["audio_uuid"],
                    "title": conversation.get("title", "Conversation"),
                    "summary": conversation.get("summary", ""),
                    "timestamp": conversation.get("session_start").timestamp() if conversation.get("session_start") else 0,
                    "created_at": conversation.get("created_at").isoformat() if conversation.get("created_at") else None,
                    "transcript": conversation.get("transcript", []),
                    "speakers_identified": conversation.get("speakers_identified", []),
                    "speaker_names": conversation.get("speaker_names", {}),
                    "duration_seconds": conversation.get("duration_seconds", 0),
                    "memories": conversation.get("memories", []),
                    "has_memory": bool(conversation.get("memories", [])),
                    "memory_processing_status": conversation.get("memory_processing_status", "pending"),
                    "action_items": conversation.get("action_items", []),
                    # Audio file paths for playback
                    "audio_path": audio_path,
                    "cropped_audio_path": cropped_audio_path,
                }
            )

        return {"conversations": conversations}

    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        return JSONResponse(status_code=500, content={"error": "Error fetching conversations"})


async def get_conversation_by_id(conversation_id: str, user: User):
    """Get a specific conversation by conversation_id (speech-driven architecture)."""
    try:
        # Import conversations collection and repository
        conversations_repo = ConversationsRepository(conversations_col)

        # Get the conversation
        conversation = await conversations_repo.get_conversation(conversation_id)
        if not conversation:
            return JSONResponse(
                status_code=404,
                content={"error": "Conversation not found"}
            )

        # Check if user owns this conversation
        if not user.is_superuser and conversation["user_id"] != str(user.user_id):
            return JSONResponse(
                status_code=403,
                content={"error": "Access forbidden. You can only access your own conversations."}
            )

        # Get audio file paths from audio_chunks collection
        audio_chunk = await chunk_repo.get_chunk_by_audio_uuid(conversation["audio_uuid"])
        audio_path = audio_chunk.get("audio_path") if audio_chunk else None
        cropped_audio_path = audio_chunk.get("cropped_audio_path") if audio_chunk else None

        # Format conversation for API response
        formatted_conversation = {
            "conversation_id": conversation["conversation_id"],
            "audio_uuid": conversation["audio_uuid"],
            "title": conversation.get("title", "Conversation"),
            "summary": conversation.get("summary", ""),
            "timestamp": conversation.get("session_start").timestamp() if conversation.get("session_start") else 0,
            "created_at": conversation.get("created_at").isoformat() if conversation.get("created_at") else None,
            "transcript": conversation.get("transcript", []),
            "speakers_identified": conversation.get("speakers_identified", []),
            "speaker_names": conversation.get("speaker_names", {}),
            "duration_seconds": conversation.get("duration_seconds", 0),
            "memories": conversation.get("memories", []),
            "has_memory": bool(conversation.get("memories", [])),
            "memory_processing_status": conversation.get("memory_processing_status", "pending"),
            "action_items": conversation.get("action_items", []),
            # Audio file paths for playback
            "audio_path": audio_path,
            "cropped_audio_path": cropped_audio_path,
        }

        return {"conversation": formatted_conversation}

    except Exception as e:
        logger.error(f"Error fetching conversation {conversation_id}: {e}")
        return JSONResponse(status_code=500, content={"error": "Error fetching conversation"})


async def get_cropped_audio_info(audio_uuid: str, user: User):
    """Get cropped audio information for a conversation. Users can only access their own conversations."""
    try:
        # Find the conversation
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser:
            if not client_belongs_to_user(chunk["client_id"], user.user_id):
                return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        return {
            "audio_uuid": audio_uuid,
            "cropped_audio_path": chunk.get("cropped_audio_path"),
            "speech_segments": chunk.get("speech_segments", []),
            "cropped_duration": chunk.get("cropped_duration"),
            "cropped_at": chunk.get("cropped_at"),
            "original_audio_path": chunk.get("audio_path"),
        }

    except Exception as e:
        logger.error(f"Error fetching cropped audio info: {e}")
        return JSONResponse(status_code=500, content={"error": "Error fetching cropped audio info"})


async def reprocess_audio_cropping(audio_uuid: str, user: User):
    """Reprocess audio cropping for a conversation. Users can only reprocess their own conversations."""
    try:
        # Find the conversation
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser:
            if not client_belongs_to_user(chunk["client_id"], user.user_id):
                return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        audio_path = chunk.get("audio_path")
        if not audio_path:
            return JSONResponse(
                status_code=400, content={"error": "No audio file found for this conversation"}
            )

        # Check if file exists
        if not Path(audio_path).exists():
            return JSONResponse(status_code=404, content={"error": "Audio file not found on disk"})

        # Reprocess the audio cropping
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, _process_audio_cropping_with_relative_timestamps, audio_path, audio_uuid
            )

            if result:
                audio_logger.info(f"Successfully reprocessed audio cropping for {audio_uuid}")
                return JSONResponse(
                    content={"message": f"Audio cropping reprocessed for {audio_uuid}"}
                )
            else:
                audio_logger.error(f"Failed to reprocess audio cropping for {audio_uuid}")
                return JSONResponse(
                    status_code=500, content={"error": "Failed to reprocess audio cropping"}
                )

        except Exception as processing_error:
            audio_logger.error(f"Error during audio cropping reprocessing: {processing_error}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Audio processing failed: {str(processing_error)}"},
            )

    except Exception as e:
        logger.error(f"Error reprocessing audio cropping: {e}")
        return JSONResponse(status_code=500, content={"error": "Error reprocessing audio cropping"})


async def add_speaker_to_conversation(audio_uuid: str, speaker_id: str, user: User):
    """Add a speaker to the speakers_identified list for a conversation. Users can only modify their own conversations."""
    try:
        # Find the conversation first
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser:
            if not client_belongs_to_user(chunk["client_id"], user.user_id):
                return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Update the speakers_identified list
        speakers = chunk.get("speakers_identified", [])
        if speaker_id not in speakers:
            speakers.append(speaker_id)
            await chunks_col.update_one(
                {"audio_uuid": audio_uuid}, {"$set": {"speakers_identified": speakers}}
            )

        return {
            "message": f"Speaker {speaker_id} added to conversation",
            "speakers_identified": speakers,
        }

    except Exception as e:
        logger.error(f"Error adding speaker to conversation: {e}")
        return JSONResponse(
            status_code=500, content={"error": "Error adding speaker to conversation"}
        )


async def update_transcript_segment(
    audio_uuid: str,
    segment_index: int,
    user: User,
    speaker_id: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
):
    """Update a specific transcript segment with speaker or timing information. Users can only modify their own conversations."""
    try:
        # Find the conversation first
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser:
            if not client_belongs_to_user(chunk["client_id"], user.user_id):
                return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        update_doc = {}

        if speaker_id is not None:
            update_doc[f"transcript.{segment_index}.speaker"] = speaker_id
            # Add to speakers_identified if not already present
            speakers = chunk.get("speakers_identified", [])
            if speaker_id not in speakers:
                speakers.append(speaker_id)
                await chunks_col.update_one(
                    {"audio_uuid": audio_uuid}, {"$set": {"speakers_identified": speakers}}
                )

        if start_time is not None:
            update_doc[f"transcript.{segment_index}.start"] = start_time

        if end_time is not None:
            update_doc[f"transcript.{segment_index}.end"] = end_time

        if not update_doc:
            return JSONResponse(status_code=400, content={"error": "No update parameters provided"})

        result = await chunks_col.update_one({"audio_uuid": audio_uuid}, {"$set": update_doc})

        if result.modified_count == 0:
            return JSONResponse(status_code=400, content={"error": "No changes were made"})

        return JSONResponse(content={"message": "Transcript segment updated successfully"})

    except Exception as e:
        audio_logger.error(f"Error updating transcript segment: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
