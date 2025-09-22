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

        # Debug: Check what's in the conversations collection
        total_conversations = await conversations_col.count_documents({})
        logger.info(f"📊 Total conversations in database: {total_conversations}")

        if total_conversations > 0:
            # Show a sample conversation to debug user_id format
            sample = await conversations_col.find_one({})
            if sample:
                logger.info(f"🔍 Sample conversation user_id: '{sample.get('user_id')}' (type: {type(sample.get('user_id'))})")
                logger.info(f"🔍 Looking for user_id: '{str(user.user_id)}' (type: {type(str(user.user_id))})")

        # Build query based on user permissions
        if not user.is_superuser:
            # Regular users can only see their own conversations
            user_conversations = await conversations_repo.get_user_conversations(str(user.user_id))
            logger.info(f"📊 Found {len(user_conversations)} conversations for user {user.user_id}")
        else:
            # Admins see all conversations
            cursor = conversations_col.find({}).sort("created_at", -1)
            all_conversations = await cursor.to_list(length=None)
            # Populate primary fields for admin conversations too
            user_conversations = [conversations_repo._populate_primary_fields(conv) for conv in all_conversations]
            logger.info(f"📊 Admin found {len(user_conversations)} total conversations")

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

            # Get version counts
            transcript_versions = conversation.get("transcript_versions", [])
            memory_versions = conversation.get("memory_versions", [])
            
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
                    # Debug: Add full URLs for troubleshooting
                    "debug_audio_url": f"/audio/{audio_path}" if audio_path else None,
                    # Version information for UI
                    "version_info": {
                        "transcript_count": len(transcript_versions),
                        "memory_count": len(memory_versions),
                        "active_transcript_version": conversation.get("active_transcript_version"),
                        "active_memory_version": conversation.get("active_memory_version"),
                    }
                }
            )

        # Log final result
        total_grouped = sum(len(convs) for convs in conversations.values())
        logger.info(f"✅ Returning {len(conversations)} client groups with {total_grouped} total conversations")
        logger.info(f"📊 Client groups: {list(conversations.keys())}")
        
        return {"conversations": conversations}

    except Exception as e:
        logger.error(f"❌ Error fetching conversations: {e}", exc_info=True)
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

        # Populate primary fields from active versions
        conversation = conversations_repo._populate_primary_fields(conversation)

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
                # Delete the conversation with all its versions (transcript_versions and memory_versions)
                conversation_result = await conversations_col.delete_one({"conversation_id": conversation_id})
                if conversation_result.deleted_count > 0:
                    conversation_deleted = True
                    logger.info(f"Deleted conversation {conversation_id} with all versions associated with audio chunk {audio_uuid}")
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


async def delete_conversation_version(conversation_id: str, version_type: str, version_id: str, user: User):
    """Delete a specific version (transcript or memory) from a conversation. Users can only modify their own conversations."""
    try:
        conversations_repo = ConversationsRepository(conversations_col)
        
        # Get the conversation first to check ownership
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
                content={"error": "Access forbidden. You can only modify your own conversations."}
            )

        # Validate version type
        if version_type not in ["transcript", "memory"]:
            return JSONResponse(
                status_code=400,
                content={"error": "Version type must be 'transcript' or 'memory'"}
            )

        # Determine field names based on version type
        if version_type == "transcript":
            versions_field = "transcript_versions"
            active_field = "active_transcript_version"
        else:  # memory
            versions_field = "memory_versions"
            active_field = "active_memory_version"

        # Check if this version exists
        versions = conversation.get(versions_field, [])
        version_exists = any(v.get("version_id") == version_id for v in versions)
        
        if not version_exists:
            return JSONResponse(
                status_code=404,
                content={"error": f"{version_type.title()} version {version_id} not found"}
            )

        # Check if there are other versions (can't delete the last one)
        if len(versions) <= 1:
            return JSONResponse(
                status_code=400,
                content={"error": f"Cannot delete the last {version_type} version. Conversation must have at least one version."}
            )

        # Check if this is the active version
        active_version = conversation.get(active_field)
        is_active = (active_version == version_id)

        # If deleting active version, we need to set a new active version
        new_active_version = None
        if is_active:
            # Find the most recent non-deleted version to make active
            remaining_versions = [v for v in versions if v.get("version_id") != version_id]
            if remaining_versions:
                # Sort by created_at and pick the most recent
                remaining_versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                new_active_version = remaining_versions[0]["version_id"]

        # Remove the version from the array
        update_operations = {
            "$pull": {versions_field: {"version_id": version_id}}
        }

        # If we need to update the active version
        if new_active_version:
            update_operations["$set"] = {active_field: new_active_version}

        # Execute the update
        result = await conversations_col.update_one(
            {"conversation_id": conversation_id},
            update_operations
        )

        if result.modified_count == 0:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to delete {version_type} version"}
            )

        # If we updated the active version, also update legacy fields
        if new_active_version:
            # Get the updated conversation and populate primary fields
            updated_conversation = await conversations_repo.get_conversation(conversation_id)
            updated_conversation = conversations_repo._populate_primary_fields(updated_conversation)
            
            # Update legacy fields in database
            legacy_updates = {}
            if version_type == "transcript":
                legacy_updates["transcript"] = updated_conversation.get("transcript", [])
                legacy_updates["speakers_identified"] = updated_conversation.get("speakers_identified", [])
            else:  # memory
                legacy_updates["memories"] = updated_conversation.get("memories", [])
                legacy_updates["memory_processing_status"] = updated_conversation.get("memory_processing_status", "pending")
            
            if legacy_updates:
                await conversations_col.update_one(
                    {"conversation_id": conversation_id},
                    {"$set": legacy_updates}
                )

        logger.info(f"Deleted {version_type} version {version_id} from conversation {conversation_id}")
        
        response_data = {
            "message": f"Successfully deleted {version_type} version {version_id}",
            "conversation_id": conversation_id,
            "version_type": version_type,
            "deleted_version_id": version_id,
            "was_active": is_active
        }
        
        if new_active_version:
            response_data["new_active_version"] = new_active_version

        return JSONResponse(status_code=200, content=response_data)

    except Exception as e:
        logger.error(f"Error deleting {version_type} version {version_id} from conversation {conversation_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to delete {version_type} version: {str(e)}"}
        )


async def _do_transcript_reprocessing(
    conversation_id: str, 
    audio_uuid: str, 
    audio_path: str, 
    version_id: str,
    user_id: str
) -> dict:
    """
    Core transcript reprocessing logic that can be called directly or from jobs.
    
    Args:
        conversation_id: ID of conversation to reprocess
        audio_uuid: UUID of audio session
        audio_path: Full path to audio file
        version_id: Version ID for the new transcript
        user_id: ID of user requesting reprocessing
        
    Returns:
        dict: Processing results with transcript and segment data
    """
    from advanced_omi_backend.transcription import TranscriptionManager
    from advanced_omi_backend.database import AudioChunksRepository
    from advanced_omi_backend.processors import get_processor_manager
    import wave
    import time
    
    start_time = time.time()
    logger.info(f"🎤 Starting core transcript reprocessing for conversation {conversation_id}")
    
    # Initialize transcription manager with existing proven pipeline
    from advanced_omi_backend.database import chunks_col
    chunk_repo = AudioChunksRepository(chunks_col)
    processor_manager = get_processor_manager()
    transcription_manager = TranscriptionManager(
        chunk_repo=chunk_repo,
        processor_manager=processor_manager
    )
    
    # Check if transcription provider is available
    if not transcription_manager.provider:
        raise Exception("No transcription provider configured")
    
    logger.info(f"🎤 Using transcription pipeline with provider: {transcription_manager.provider.name}")
    
    # Load and process the audio file using existing pipeline
    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        audio_frames = wav_file.readframes(wav_file.getnframes())
        
    logger.info(f"🎤 Audio loaded: {len(audio_frames)} bytes, {sample_rate}Hz")
    
    # Process audio directly without using conversation creation pipeline
    
    # Process transcription directly without creating new conversations
    logger.info(f"🎤 Processing transcript directly for reprocessing...")
    
    # Use the transcription provider directly to avoid creating new conversations
    transcript_result = await transcription_manager.provider.transcribe(
        audio_frames, sample_rate, diarize=True
    )
    
    if not transcript_result:
        raise Exception("Transcription failed - no result returned")
    
    # Extract transcript and segments from the provider result
    transcript_text = transcript_result.get("text", "")
    segments = transcript_result.get("segments", [])
    
    # If segments are empty but we have text, create a single segment
    if not segments and transcript_text:
        segments = [{
            "text": transcript_text,
            "start": 0.0,
            "end": len(audio_frames) / (sample_rate * 2),  # Estimate duration
            "speaker": "Speaker 0",
            "confidence": transcript_result.get("confidence", 0.9)
        }]
    
    logger.info(f"🎤 Transcript reprocessing completed:")
    logger.info(f"    - Text length: {len(transcript_text)} characters")
    logger.info(f"    - Segments: {len(segments)}")
    logger.info(f"    - Processing time: {time.time() - start_time:.2f} seconds")

    # Add speaker identification step if segments exist
    final_segments = segments
    if segments and len(segments) > 0:
        try:
            logger.info(f"🎤 Starting speaker identification for {len(segments)} segments...")

            # Initialize speaker client
            from advanced_omi_backend.speaker_recognition_client import SpeakerRecognitionClient
            speaker_client = SpeakerRecognitionClient()

            if speaker_client.enabled:
                # Create transcript data for speaker identification
                words = transcript_result.get("words", [])
                transcript_data = {
                    "text": transcript_text,
                    "words": words
                }

                # Call speaker identification with the audio file and transcript data
                speaker_result = await speaker_client.diarize_identify_match(
                    audio_path, transcript_data, user_id=user_id
                )

                if speaker_result and speaker_result.get("segments"):
                    raw_segments = speaker_result["segments"]
                    logger.info(f"🎤 Speaker identification completed: {len(raw_segments)} raw segments")

                    # Filter out segments with empty or minimal text
                    final_segments = []
                    for seg in raw_segments:
                        text = seg.get("text", "").strip()
                        # Only keep segments with meaningful text (at least 2 characters)
                        if len(text) >= 2:
                            final_segments.append(seg)
                        else:
                            logger.debug(f"🗑️ Filtering out empty/minimal segment: '{text}' (speaker: {seg.get('speaker', 'UNKNOWN')})")

                    logger.info(f"🎤 After filtering: {len(final_segments)} segments with meaningful text (filtered out {len(raw_segments) - len(final_segments)} empty segments)")

                    # Debug: Log first few segments
                    for i, seg in enumerate(final_segments[:3]):
                        logger.info(f"🔍 Segment {i}: speaker='{seg.get('speaker', 'UNKNOWN')}', text='{seg.get('text', '')[:50]}...'")
                else:
                    logger.warning("🎤 Speaker identification returned no segments, using original segments")
            else:
                logger.info("🎤 Speaker recognition disabled, using segments without speaker identification")

        except Exception as e:
            logger.warning(f"Speaker identification failed during reprocessing: {e}")
            # Continue with original segments if speaker identification fails

    # Update the conversation with the new transcript version (using final_segments with speaker identification)
    conversations_repo = ConversationsRepository(conversations_col)
    update_result = await conversations_repo.update_transcript_version(
        conversation_id=conversation_id,
        version_id=version_id,
        transcript=transcript_text,
        segments=final_segments,
        processing_time_seconds=time.time() - start_time,
        provider=transcription_manager.provider.name
    )
    
    if update_result:
        logger.info(f"✅ Updated transcript version {version_id} in database")
        
        # Activate the new transcript version
        activation_result = await conversations_repo.activate_transcript_version(conversation_id, version_id)
        if activation_result:
            logger.info(f"✅ Activated transcript version {version_id}")
        else:
            logger.warning(f"⚠️ Failed to activate transcript version {version_id}")
        
        return {
            "success": True,
            "transcript": transcript_text,
            "segments": final_segments,
            "version_id": version_id,
            "processing_time_seconds": time.time() - start_time,
            "provider": transcription_manager.provider.name
        }
    else:
        raise Exception("Failed to update transcript version in database")


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

        # Get audio file metadata for logging
        try:
            import wave
            
            audio_stats = {
                "file_exists": full_audio_path.exists(),
                "file_size_bytes": full_audio_path.stat().st_size if full_audio_path.exists() else 0,
                "duration_seconds": 0,
                "sample_rate": 0,
                "channels": 0
            }
            
            if full_audio_path.exists() and str(full_audio_path).endswith('.wav'):
                try:
                    with wave.open(str(full_audio_path), 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        sample_rate = wav_file.getframerate()
                        audio_stats["duration_seconds"] = frames / sample_rate if sample_rate > 0 else 0
                        audio_stats["sample_rate"] = sample_rate
                        audio_stats["channels"] = wav_file.getnchannels()
                except Exception as wav_error:
                    logger.warning(f"Failed to read WAV metadata: {wav_error}")
            
            logger.info(f"🎵 Audio file metadata for {conversation_id}: {audio_stats}")
            
        except Exception as metadata_error:
            logger.warning(f"Failed to get audio metadata: {metadata_error}")
            audio_stats = {"error": str(metadata_error)}

        # Queue the reprocessing job
        try:
            from advanced_omi_backend.simple_queue import get_simple_queue
            
            queue = await get_simple_queue()
            job_id = await queue.enqueue_job(
                job_type="reprocess_transcript",
                user_id=str(user.user_id),
                data={
                    "conversation_id": conversation_id,
                    "audio_uuid": audio_uuid,
                    "audio_path": str(full_audio_path),
                    "run_id": run_id,
                    "version_id": version_id,
                    "audio_metadata": audio_stats
                }
            )
            
            logger.info(f"📋 Queued transcript reprocessing job {job_id} for conversation {conversation_id} (run {run_id}, version {version_id})")
            logger.info(f"📋 Job data: audio_path={full_audio_path}, duration={audio_stats.get('duration_seconds', 0)}s")
            
            return JSONResponse(content={
                "message": f"Transcript reprocessing queued for conversation {conversation_id}",
                "job_id": job_id,
                "run_id": run_id,
                "version_id": version_id,
                "config_hash": config_hash,
                "status": "QUEUED",
                "audio_metadata": audio_stats
            })
            
        except Exception as queue_error:
            logger.error(f"Failed to queue transcript reprocessing job: {queue_error}")
            return JSONResponse(
                status_code=500, 
                content={
                    "error": "Failed to queue transcript reprocessing job",
                    "details": str(queue_error)
                }
            )

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

        # Queue the memory processing job
        try:
            from advanced_omi_backend.simple_queue import get_simple_queue
            
            queue = await get_simple_queue()
            job_id = await queue.enqueue_job(
                job_type="reprocess_memory",
                user_id=str(user.user_id),
                data={
                    "conversation_id": conversation_id,
                    "audio_uuid": audio_uuid,
                    "transcript_version_id": transcript_version_id,
                    "run_id": run_id,
                    "version_id": version_id
                }
            )
            
            logger.info(f"Queued memory reprocessing job {job_id} for conversation {conversation_id} (run {run_id}, version {version_id})")
            
            return JSONResponse(content={
                "message": f"Memory reprocessing queued for conversation {conversation_id}",
                "job_id": job_id,
                "run_id": run_id,
                "version_id": version_id,
                "transcript_version_id": transcript_version_id,
                "config_hash": config_hash,
                "status": "QUEUED"
            })
            
        except Exception as queue_error:
            logger.error(f"Failed to queue memory reprocessing job: {queue_error}")
            return JSONResponse(
                status_code=500, 
                content={
                    "error": "Failed to queue memory reprocessing job",
                    "details": str(queue_error)
                }
            )

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
