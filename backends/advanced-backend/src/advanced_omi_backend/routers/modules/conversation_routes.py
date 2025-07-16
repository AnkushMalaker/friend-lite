"""
Conversation management routes for Friend-Lite API.

Handles conversation CRUD operations, audio processing, and transcript management.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from advanced_omi_backend.audio_cropping_utils import (
    _process_audio_cropping_with_relative_timestamps,
)
from advanced_omi_backend.auth import current_active_user
from advanced_omi_backend.client_manager import (
    ClientManager,
    client_belongs_to_user,
    get_client_manager_dependency,
    get_user_clients_all,
)
from advanced_omi_backend.database import AudioChunksCollectionHelper, chunks_col
from advanced_omi_backend.users import User

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")

router = APIRouter(prefix="/conversations", tags=["conversations"])

# Initialize chunk repository
chunk_repo = AudioChunksCollectionHelper(chunks_col)


@router.post("/{client_id}/close")
async def close_current_conversation(
    client_id: str,
    current_user: User = Depends(current_active_user),
    client_manager: ClientManager = Depends(get_client_manager_dependency),
):
    """Close the current conversation for a specific client. Users can only close their own conversations."""
    # Validate client ownership
    if not current_user.is_superuser and not client_belongs_to_user(
        client_id, current_user.user_id
    ):
        logger.warning(
            f"User {current_user.user_id} attempted to close conversation for client {client_id} without permission"
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
        await client_state._close_current_conversation()

        # Reset conversation state but keep client connected
        client_state.current_audio_uuid = None
        client_state.conversation_start_time = time.time()
        client_state.last_transcript_time = None

        logger.info(
            f"Manually closed conversation for client {client_id} by user {current_user.id}"
        )

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


@router.get("")
async def get_conversations(current_user: User = Depends(current_active_user)):
    """Get conversations. Admins see all conversations, users see only their own."""
    try:
        # Build query based on user permissions
        if not current_user.is_superuser:
            # Regular users can only see their own conversations
            user_client_ids = get_user_clients_all(current_user.user_id)
            if not user_client_ids:
                # User has no clients, return empty result
                return {"conversations": {}}
            query = {"client_id": {"$in": user_client_ids}}
        else:
            query = {}

        # Get audio chunks and group by client_id
        cursor = chunks_col.find(query).sort("timestamp", -1)
        conversations = {}

        async for chunk in cursor:
            client_id = chunk["client_id"]
            if client_id not in conversations:
                conversations[client_id] = []

            conversations[client_id].append(
                {
                    "audio_uuid": chunk["audio_uuid"],
                    "audio_path": chunk["audio_path"],
                    "timestamp": chunk["timestamp"],
                    "transcript": chunk.get("transcript", []),
                    "speakers_identified": chunk.get("speakers_identified", []),
                    "cropped_audio_path": chunk.get("cropped_audio_path"),
                    "speech_segments": chunk.get("speech_segments"),
                    "cropped_duration": chunk.get("cropped_duration"),
                    "memories": chunk.get(
                        "memories", []
                    ),  # Include memory references if they exist
                    "has_memory": bool(chunk.get("memories", [])),  # Quick boolean check for UI
                }
            )

        return {"conversations": conversations}

    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        return JSONResponse(status_code=500, content={"error": "Error fetching conversations"})


@router.get("/{audio_uuid}/cropped")
async def get_cropped_audio_info(
    audio_uuid: str, current_user: User = Depends(current_active_user)
):
    """Get cropped audio information for a conversation. Users can only access their own conversations."""
    try:
        # Find the conversation
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not current_user.is_superuser:
            if not client_belongs_to_user(chunk["client_id"], current_user.user_id):
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


@router.post("/{audio_uuid}/reprocess")
async def reprocess_audio_cropping(
    audio_uuid: str, current_user: User = Depends(current_active_user)
):
    """Reprocess audio cropping for a conversation. Users can only reprocess their own conversations."""
    try:
        # Find the conversation
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not current_user.is_superuser:
            if not client_belongs_to_user(chunk["client_id"], current_user.user_id):
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


@router.post("/{audio_uuid}/speakers")
async def add_speaker_to_conversation(
    audio_uuid: str, speaker_id: str, current_user: User = Depends(current_active_user)
):
    """Add a speaker to the speakers_identified list for a conversation. Users can only modify their own conversations."""
    try:
        # Find the conversation first
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not current_user.is_superuser:
            if not client_belongs_to_user(chunk["client_id"], current_user.user_id):
                return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        await chunk_repo.add_speaker(audio_uuid, speaker_id)
        return JSONResponse(
            content={"message": f"Speaker {speaker_id} added to conversation {audio_uuid}"}
        )
    except Exception as e:
        audio_logger.error(f"Error adding speaker: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error adding speaker"})


@router.put("/{audio_uuid}/transcript/{segment_index}")
async def update_transcript_segment(
    audio_uuid: str,
    segment_index: int,
    current_user: User = Depends(current_active_user),
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
        if not current_user.is_superuser:
            if not client_belongs_to_user(chunk["client_id"], current_user.user_id):
                return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        update_doc = {}

        if speaker_id is not None:
            update_doc[f"transcript.{segment_index}.speaker"] = speaker_id
            # Also add to speakers_identified if not already present
            await chunk_repo.add_speaker(audio_uuid, speaker_id)

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
