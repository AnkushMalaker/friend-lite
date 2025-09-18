"""
Conversation controller for handling conversation-related business logic.
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

from advanced_omi_backend.audio_cropping_utils import (
    _process_audio_cropping_with_relative_timestamps,
)
from advanced_omi_backend.client_manager import (
    ClientManager,
    client_belongs_to_user,
    get_user_clients_all,
)
from advanced_omi_backend.database import AudioChunksRepository, ProcessingRunsRepository, chunks_col, processing_runs_col, conversations_col, ConversationsRepository
from advanced_omi_backend.users import User
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")

# Initialize repositories
chunk_repo = AudioChunksRepository(chunks_col)
processing_runs_repo = ProcessingRunsRepository(processing_runs_col)
conversations_repo = ConversationsRepository(conversations_col)


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

        # Check if file exists - try multiple possible locations
        possible_paths = [
            Path("/app/data/audio_chunks") / audio_path,
            Path("/app/audio_chunks") / audio_path,
            Path(audio_path),  # fallback to relative path
        ]

        full_audio_path = None
        for path in possible_paths:
            if path.exists():
                full_audio_path = path
                break

        if not full_audio_path:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Audio file not found on disk",
                    "details": f"Conversation exists but audio file '{audio_path}' is missing from expected locations",
                    "searched_paths": [str(p) for p in possible_paths]
                }
            )

        # Reprocess the audio cropping
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, _process_audio_cropping_with_relative_timestamps, str(full_audio_path), audio_uuid
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

async def delete_conversation(audio_uuid: str, user: User):
    """Delete a conversation and its associated audio file. Users can only delete their own conversations."""
    try:
        # Create masked identifier for logging
        masked_uuid = f"{audio_uuid[:8]}...{audio_uuid[-4:]}" if len(audio_uuid) > 12 else "***"
        logger.info(f"Attempting to delete conversation: {masked_uuid}")

        # Detailed debugging only when debug level is enabled
        if logger.isEnabledFor(logging.DEBUG):
            total_count = await chunks_col.count_documents({})
            logger.debug(f"Total conversations in collection: {total_count}")
            logger.debug(f"UUID length: {len(audio_uuid)}, type: {type(audio_uuid)}")

        # First, get the audio chunk record to check ownership and get conversation_id
        audio_chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Audio chunk lookup result: {'found' if audio_chunk else 'not found'}")
            if audio_chunk:
                logger.debug(f"Found audio chunk with client_id: {audio_chunk.get('client_id')}")
                logger.debug(f"Audio chunk has conversation_id: {audio_chunk.get('conversation_id')}")
            else:
                # Try alternative queries for debugging
                regex_result = await chunks_col.find_one({"audio_uuid": {"$regex": f"^{audio_uuid}$", "$options": "i"}})
                contains_result = await chunks_col.find_one({"audio_uuid": {"$regex": audio_uuid}})
                logger.debug(f"Alternative query attempts - case insensitive: {'found' if regex_result else 'not found'}, substring: {'found' if contains_result else 'not found'}")

        if not audio_chunk:
            return JSONResponse(
                status_code=404,
                content={"error": f"Audio chunk with audio_uuid '{audio_uuid}' not found"}
            )

        # Check if user has permission to delete this conversation
        client_id = audio_chunk.get("client_id")
        if not user.is_superuser and not client_belongs_to_user(client_id, user.user_id):
            logger.warning(
                f"User {user.user_id} attempted to delete conversation {audio_uuid} without permission"
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Access forbidden. You can only delete your own conversations.",
                    "details": f"Conversation '{audio_uuid}' does not belong to your account."
                }
            )

        # Get audio file paths for deletion
        audio_path = audio_chunk.get("audio_path")
        cropped_audio_path = audio_chunk.get("cropped_audio_path")

        # Get conversation_id if this audio chunk has an associated conversation
        conversation_id = audio_chunk.get("conversation_id")
        conversation_deleted = False

        # Delete from audio_chunks collection first
        audio_result = await chunks_col.delete_one({"audio_uuid": audio_uuid})

        if audio_result.deleted_count == 0:
            return JSONResponse(
                status_code=404,
                content={"error": f"Failed to delete audio chunk with audio_uuid '{audio_uuid}'"}
            )

        logger.info(f"Deleted audio chunk {audio_uuid}")

        # If this audio chunk has an associated conversation, delete it from conversations collection too
        if conversation_id:
            try:
                conversation_result = await conversations_col.delete_one({"conversation_id": conversation_id})
                if conversation_result.deleted_count > 0:
                    conversation_deleted = True
                    logger.info(f"Deleted conversation {conversation_id} associated with audio chunk {audio_uuid}")
                else:
                    logger.warning(f"Conversation {conversation_id} not found in conversations collection, but audio chunk was deleted")
            except Exception as e:
                logger.warning(f"Failed to delete conversation {conversation_id}: {e}")

        # Delete associated audio files
        deleted_files = []
        if audio_path:
            try:
                # Construct full path to audio file
                full_audio_path = Path("/app/audio_chunks") / audio_path
                if full_audio_path.exists():
                    full_audio_path.unlink()
                    deleted_files.append(str(full_audio_path))
                    logger.info(f"Deleted audio file: {full_audio_path}")
            except Exception as e:
                logger.warning(f"Failed to delete audio file {audio_path}: {e}")

        if cropped_audio_path:
            try:
                # Construct full path to cropped audio file
                full_cropped_path = Path("/app/audio_chunks") / cropped_audio_path
                if full_cropped_path.exists():
                    full_cropped_path.unlink()
                    deleted_files.append(str(full_cropped_path))
                    logger.info(f"Deleted cropped audio file: {full_cropped_path}")
            except Exception as e:
                logger.warning(f"Failed to delete cropped audio file {cropped_audio_path}: {e}")

        logger.info(f"Successfully deleted conversation {audio_uuid} for user {user.user_id}")

        # Prepare response message
        delete_summary = []
        delete_summary.append("audio chunk")
        if conversation_deleted:
            delete_summary.append("conversation record")
        if deleted_files:
            delete_summary.append(f"{len(deleted_files)} audio file(s)")

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully deleted {', '.join(delete_summary)} for '{audio_uuid}'",
                "deleted_files": deleted_files,
                "client_id": client_id,
                "conversation_id": conversation_id,
                "conversation_deleted": conversation_deleted
            }
        )

    except Exception as e:
        logger.error(f"Error deleting conversation {audio_uuid}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to delete conversation: {str(e)}"}
        )


async def reprocess_transcript(conversation_id: str, user: User):
    """Reprocess transcript for a conversation. Users can only reprocess their own conversations."""
    try:
        # Find the conversation in conversations collection
        conversations_repo = ConversationsRepository(conversations_col)
        conversation = await conversations_repo.get_conversation(conversation_id)
        if not conversation:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser and conversation["user_id"] != str(user.user_id):
            return JSONResponse(status_code=403, content={"error": "Access forbidden. You can only reprocess your own conversations."})

        # Get audio_uuid for file access
        audio_uuid = conversation["audio_uuid"]

        # Get audio file path from audio_chunks collection
        chunk = await chunks_col.find_one({"audio_uuid": audio_uuid})
        if not chunk:
            return JSONResponse(status_code=404, content={"error": "Audio session not found"})

        audio_path = chunk.get("audio_path")
        if not audio_path:
            return JSONResponse(
                status_code=400, content={"error": "No audio file found for this conversation"}
            )

        # Check if file exists - try multiple possible locations
        possible_paths = [
            Path("/app/data/audio_chunks") / audio_path,
            Path("/app/audio_chunks") / audio_path,
            Path(audio_path),  # fallback to relative path
        ]

        full_audio_path = None
        for path in possible_paths:
            if path.exists():
                full_audio_path = path
                break

        if not full_audio_path:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Audio file not found on disk",
                    "details": f"Conversation exists but audio file '{audio_path}' is missing from expected locations",
                    "searched_paths": [str(p) for p in possible_paths]
                }
            )

        # Generate configuration hash for duplicate detection
        config_data = {
            "audio_path": str(full_audio_path),
            "transcription_provider": "deepgram",  # This would come from settings
            "trigger": "manual_reprocess"
        }
        config_hash = hashlib.sha256(str(config_data).encode()).hexdigest()[:16]

        # Create processing run
        run_id = await processing_runs_repo.create_run(
            conversation_id=conversation_id,
            audio_uuid=audio_uuid,
            run_type="transcript",
            user_id=user.user_id,
            trigger="manual_reprocess",
            config_hash=config_hash
        )

        # Create new transcript version in conversations collection
        version_id = await conversations_repo.create_transcript_version(
            conversation_id=conversation_id,
            processing_run_id=run_id
        )

        if not version_id:
            return JSONResponse(
                status_code=500, content={"error": "Failed to create transcript version"}
            )

        # TODO: Queue audio for reprocessing with ProcessorManager
        # This is where we would integrate with the existing processor
        # For now, we'll return the version ID for the caller to handle

        logger.info(f"Created transcript reprocessing job {run_id} (version {version_id}) for conversation {conversation_id}")

        return JSONResponse(content={
            "message": f"Transcript reprocessing started for conversation {conversation_id}",
            "run_id": run_id,
            "version_id": version_id,
            "config_hash": config_hash,
            "status": "PENDING"
        })

    except Exception as e:
        logger.error(f"Error starting transcript reprocessing: {e}")
        return JSONResponse(status_code=500, content={"error": "Error starting transcript reprocessing"})


async def reprocess_memory(conversation_id: str, transcript_version_id: str, user: User):
    """Reprocess memory extraction for a specific transcript version. Users can only reprocess their own conversations."""
    try:
        # Find the conversation in conversations collection
        conversations_repo = ConversationsRepository(conversations_col)
        conversation = await conversations_repo.get_conversation(conversation_id)
        if not conversation:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser and conversation["user_id"] != str(user.user_id):
            return JSONResponse(status_code=403, content={"error": "Access forbidden. You can only reprocess your own conversations."})

        # Get audio_uuid for processing run tracking
        audio_uuid = conversation["audio_uuid"]

        # Resolve transcript version ID
        transcript_versions = conversation.get("transcript_versions", [])

        # Handle special "active" version ID
        if transcript_version_id == "active":
            active_version_id = conversation.get("active_transcript_version")
            if not active_version_id:
                return JSONResponse(
                    status_code=404, content={"error": "No active transcript version found"}
                )
            transcript_version_id = active_version_id

        # Find the specific transcript version
        transcript_version = None
        for version in transcript_versions:
            if version["version_id"] == transcript_version_id:
                transcript_version = version
                break

        if not transcript_version:
            return JSONResponse(
                status_code=404, content={"error": f"Transcript version '{transcript_version_id}' not found"}
            )

        # Generate configuration hash for duplicate detection
        config_data = {
            "transcript_version_id": transcript_version_id,
            "memory_provider": "friend_lite",  # This would come from settings
            "trigger": "manual_reprocess"
        }
        config_hash = hashlib.sha256(str(config_data).encode()).hexdigest()[:16]

        # Create processing run
        run_id = await processing_runs_repo.create_run(
            conversation_id=conversation_id,
            audio_uuid=audio_uuid,
            run_type="memory",
            user_id=user.user_id,
            trigger="manual_reprocess",
            config_hash=config_hash
        )

        # Create new memory version in conversations collection
        version_id = await conversations_repo.create_memory_version(
            conversation_id=conversation_id,
            transcript_version_id=transcript_version_id,
            processing_run_id=run_id
        )

        if not version_id:
            return JSONResponse(
                status_code=500, content={"error": "Failed to create memory version"}
            )

        # TODO: Queue memory extraction for processing
        # This is where we would integrate with the existing memory processor

        logger.info(f"Created memory reprocessing job {run_id} (version {version_id}) for conversation {conversation_id}")

        return JSONResponse(content={
            "message": f"Memory reprocessing started for conversation {conversation_id}",
            "run_id": run_id,
            "version_id": version_id,
            "transcript_version_id": transcript_version_id,
            "config_hash": config_hash,
            "status": "PENDING"
        })

    except Exception as e:
        logger.error(f"Error starting memory reprocessing: {e}")
        return JSONResponse(status_code=500, content={"error": "Error starting memory reprocessing"})


async def activate_transcript_version(conversation_id: str, version_id: str, user: User):
    """Activate a specific transcript version. Users can only modify their own conversations."""
    try:
        # Find the conversation in conversations collection
        conversations_repo = ConversationsRepository(conversations_col)
        conversation = await conversations_repo.get_conversation(conversation_id)
        if not conversation:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser and conversation["user_id"] != str(user.user_id):
            return JSONResponse(status_code=403, content={"error": "Access forbidden. You can only modify your own conversations."})

        # Activate the transcript version
        success = await conversations_repo.activate_transcript_version(conversation_id, version_id)
        if not success:
            return JSONResponse(
                status_code=400, content={"error": "Failed to activate transcript version"}
            )

        # TODO: Trigger speaker recognition if configured
        # This would integrate with existing speaker recognition logic

        logger.info(f"Activated transcript version {version_id} for conversation {conversation_id} by user {user.user_id}")

        return JSONResponse(content={
            "message": f"Transcript version {version_id} activated successfully",
            "active_transcript_version": version_id
        })

    except Exception as e:
        logger.error(f"Error activating transcript version: {e}")
        return JSONResponse(status_code=500, content={"error": "Error activating transcript version"})


async def activate_memory_version(conversation_id: str, version_id: str, user: User):
    """Activate a specific memory version. Users can only modify their own conversations."""
    try:
        # Find the conversation in conversations collection
        conversations_repo = ConversationsRepository(conversations_col)
        conversation = await conversations_repo.get_conversation(conversation_id)
        if not conversation:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser and conversation["user_id"] != str(user.user_id):
            return JSONResponse(status_code=403, content={"error": "Access forbidden. You can only modify your own conversations."})

        # Activate the memory version
        success = await conversations_repo.activate_memory_version(conversation_id, version_id)
        if not success:
            return JSONResponse(
                status_code=400, content={"error": "Failed to activate memory version"}
            )

        logger.info(f"Activated memory version {version_id} for conversation {conversation_id} by user {user.user_id}")

        return JSONResponse(content={
            "message": f"Memory version {version_id} activated successfully",
            "active_memory_version": version_id
        })

    except Exception as e:
        logger.error(f"Error activating memory version: {e}")
        return JSONResponse(status_code=500, content={"error": "Error activating memory version"})


async def get_conversation_version_history(conversation_id: str, user: User):
    """Get version history for a conversation. Users can only access their own conversations."""
    try:
        # Find the conversation in conversations collection to check ownership
        conversations_repo = ConversationsRepository(conversations_col)
        conversation = await conversations_repo.get_conversation(conversation_id)
        if not conversation:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser and conversation["user_id"] != str(user.user_id):
            return JSONResponse(status_code=403, content={"error": "Access forbidden. You can only access your own conversations."})

        # Get version history
        history = await conversations_repo.get_version_history(conversation_id)

        return JSONResponse(content=history)

    except Exception as e:
        logger.error(f"Error fetching version history: {e}")
        return JSONResponse(status_code=500, content={"error": "Error fetching version history"})
