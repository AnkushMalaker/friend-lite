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
from advanced_omi_backend.database import AudioChunksRepository, chunks_col
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
    """Get conversations. Admins see all conversations, users see only their own."""
    try:
        # Build query based on user permissions
        if not user.is_superuser:
            # Regular users can only see their own conversations
            user_client_ids = get_user_clients_all(user.user_id)
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

            # Get transcript data - prefer segments but fallback to raw transcript
            transcript_segments = chunk.get("transcript", [])
            if not transcript_segments and chunk.get("raw_transcript_data"):
                # No segments but we have raw transcript data - create fallback representation
                raw_data = chunk["raw_transcript_data"]
                if raw_data.get("data", {}).get("text"):
                    transcript_segments = [{
                        "text": raw_data["data"]["text"],
                        "start": 0.0,
                        "end": 0.0,
                        "speaker": "Unknown",
                        "confidence": 0.0,
                        "source": "raw_transcript"  # Indicator this is fallback data
                    }]

            conversations[client_id].append(
                {
                    "audio_uuid": chunk["audio_uuid"],
                    "audio_path": chunk["audio_path"],
                    "timestamp": chunk["timestamp"],
                    "transcript": transcript_segments,
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
