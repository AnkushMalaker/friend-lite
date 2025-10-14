"""
Conversation controller for handling conversation-related business logic.
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from advanced_omi_backend.audio_utils import (
    _process_audio_cropping_with_relative_timestamps,
)
from advanced_omi_backend.client_manager import (
    ClientManager,
    client_belongs_to_user,
    get_user_clients_all,
)
from advanced_omi_backend.database import AudioChunksRepository, chunks_col
from advanced_omi_backend.models.conversation import Conversation
from advanced_omi_backend.users import User
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")

# Initialize repositories (legacy collections only)
chunk_repo = AudioChunksRepository(chunks_col)
# ProcessingRunsRepository removed - using RQ job tracking instead


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


async def get_conversation(conversation_id: str, user: User):
    """Get a single conversation with full transcript details."""
    try:
        # Find the conversation using Beanie
        conversation = await Conversation.find_one(Conversation.conversation_id == conversation_id)
        if not conversation:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser and conversation.user_id != str(user.user_id):
            return JSONResponse(status_code=403, content={"error": "Access forbidden"})

        # Get audio file paths from audio_chunks collection
        audio_chunk = await chunk_repo.get_chunk_by_audio_uuid(conversation.audio_uuid)
        audio_path = audio_chunk.get("audio_path") if audio_chunk else None
        cropped_audio_path = audio_chunk.get("cropped_audio_path") if audio_chunk else None

        # Format conversation for API response - use model_dump and add computed fields
        formatted_conversation = conversation.model_dump(
            mode='json',  # Automatically converts datetime to ISO strings, handles nested models
            exclude={'id'}  # Exclude MongoDB internal _id
        )

        # Add computed/external fields not in the model
        formatted_conversation.update({
            "timestamp": 0,  # Legacy field - using created_at instead
            "has_memory": bool(conversation.memories),
            "audio_path": audio_path,
            "cropped_audio_path": cropped_audio_path,
            "version_info": {
                "transcript_count": len(conversation.transcript_versions),
                "memory_count": len(conversation.memory_versions),
                "active_transcript_version": conversation.active_transcript_version,
                "active_memory_version": conversation.active_memory_version
            }
        })

        return {"conversation": formatted_conversation}

    except Exception as e:
        logger.error(f"Error fetching conversation {conversation_id}: {e}")
        return JSONResponse(status_code=500, content={"error": "Error fetching conversation"})


async def get_conversations(user: User):
    """Get conversations with speech only (speech-driven architecture)."""
    try:
        # Build query based on user permissions using Beanie
        if not user.is_superuser:
            # Regular users can only see their own conversations
            user_conversations = await Conversation.find(
                Conversation.user_id == str(user.user_id)
            ).sort(-Conversation.created_at).to_list()
        else:
            # Admins see all conversations
            user_conversations = await Conversation.find_all().sort(-Conversation.created_at).to_list()

        # Batch fetch all audio chunks in one query to avoid N+1 problem
        audio_uuids = [conv.audio_uuid for conv in user_conversations]
        audio_chunks_dict = {}
        if audio_uuids:
            # Fetch all audio chunks at once
            chunks_cursor = chunk_repo.col.find({"audio_uuid": {"$in": audio_uuids}})
            async for chunk in chunks_cursor:
                audio_chunks_dict[chunk["audio_uuid"]] = chunk

        # Convert conversations to API format
        conversations = []
        for conv in user_conversations:
            # Get audio file paths from pre-fetched chunks
            audio_chunk = audio_chunks_dict.get(conv.audio_uuid)
            audio_path = audio_chunk.get("audio_path") if audio_chunk else None
            cropped_audio_path = audio_chunk.get("cropped_audio_path") if audio_chunk else None

            # Format conversation for list - use model_dump with exclusions
            conv_dict = conv.model_dump(
                mode='json',  # Automatically converts datetime to ISO strings
                exclude={'id', 'transcript', 'segments'}  # Exclude large fields for list view
            )

            # Add computed/external fields
            conv_dict.update({
                "timestamp": 0,  # Legacy field - using created_at instead
                "segment_count": len(conv.segments) if conv.segments else 0,
                "has_memory": bool(conv.memories),
                "audio_path": audio_path,
                "cropped_audio_path": cropped_audio_path,
                "version_info": {
                    "transcript_count": len(conv.transcript_versions),
                    "memory_count": len(conv.memory_versions),
                    "active_transcript_version": conv.active_transcript_version,
                    "active_memory_version": conv.active_memory_version
                }
            })

            conversations.append(conv_dict)

        return {"conversations": conversations}

    except Exception as e:
        logger.exception(f"Error fetching conversations: {e}")
        return JSONResponse(status_code=500, content={"error": "Error fetching conversations"})


async def get_conversation_by_id(conversation_id: str, user: User):
    """Get a specific conversation by conversation_id (speech-driven architecture)."""
    try:
        # Get the conversation using Beanie
        conversation_model = await Conversation.find_one(Conversation.conversation_id == conversation_id)
        if not conversation_model:
            return JSONResponse(
                status_code=404,
                content={"error": "Conversation not found"}
            )

        # Check if user owns this conversation
        if not user.is_superuser and conversation_model.user_id != str(user.user_id):
            return JSONResponse(
                status_code=403,
                content={"error": "Access forbidden. You can only access your own conversations."}
            )

        # Get audio file paths from audio_chunks collection
        audio_chunk = await chunk_repo.get_chunk_by_audio_uuid(conversation_model.audio_uuid)
        audio_path = audio_chunk.get("audio_path") if audio_chunk else None
        cropped_audio_path = audio_chunk.get("cropped_audio_path") if audio_chunk else None

        # Format conversation for API response - use model_dump and add computed fields
        formatted_conversation = conversation_model.model_dump(
            mode='json',  # Automatically converts datetime to ISO strings, handles nested models
            exclude={'id'}  # Exclude MongoDB internal _id
        )

        # Add computed/external fields not in the model
        formatted_conversation.update({
            "timestamp": 0,  # Legacy field - using created_at instead
            "has_memory": bool(conversation_model.memories),
            "audio_path": audio_path,
            "cropped_audio_path": cropped_audio_path,
            "version_info": {
                "transcript_count": len(conversation_model.transcript_versions),
                "memory_count": len(conversation_model.memory_versions),
                "active_transcript_version": conversation_model.active_transcript_version,
                "active_memory_version": conversation_model.active_memory_version
            }
        })

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

        # Get speech segments from the chunk
        speech_segments = chunk.get("speech_segments", [])
        if not speech_segments:
            return JSONResponse(
                status_code=400,
                content={"error": "No speech segments found for this conversation"}
            )

        # Generate output path for cropped audio
        cropped_filename = f"cropped_{audio_uuid}.wav"
        output_path = Path("/app/audio_chunks") / cropped_filename

        # Get repository for database updates
        chunk_repo = AudioChunksRepository(chunks_col)

        # Reprocess the audio cropping
        try:
            result = await _process_audio_cropping_with_relative_timestamps(
                str(full_audio_path),
                speech_segments,
                str(output_path),
                audio_uuid,
                chunk_repo
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

        # If this audio chunk has an associated conversation, delete it using Beanie
        if conversation_id:
            try:
                conversation_model = await Conversation.find_one(Conversation.conversation_id == conversation_id)
                if conversation_model:
                    await conversation_model.delete()
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
        # Find the conversation using Beanie
        conversation_model = await Conversation.find_one(Conversation.conversation_id == conversation_id)
        if not conversation_model:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser and conversation_model.user_id != str(user.user_id):
            return JSONResponse(status_code=403, content={"error": "Access forbidden. You can only reprocess your own conversations."})

        # Get audio_uuid for file access
        audio_uuid = conversation_model.audio_uuid

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

        # Create new transcript version ID
        import uuid
        version_id = str(uuid.uuid4())

        # Enqueue job chain with RQ (transcription -> speaker recognition -> memory)
        from advanced_omi_backend.workers.transcription_jobs import transcribe_full_audio_job, recognise_speakers_job
        from advanced_omi_backend.workers.memory_jobs import process_memory_job
        from advanced_omi_backend.controllers.queue_controller import transcription_queue, memory_queue, JOB_RESULT_TTL

        # Job 1: Transcribe audio to text
        transcript_job = transcription_queue.enqueue(
            transcribe_full_audio_job,
            conversation_id,
            audio_uuid,
            str(full_audio_path),
            version_id,
            str(user.user_id),
            "reprocess",
            job_timeout=600,
            result_ttl=JOB_RESULT_TTL,
            job_id=f"reprocess_{conversation_id[:8]}",
            description=f"Transcribe audio for {conversation_id[:8]}",
            meta={'audio_uuid': audio_uuid}
        )
        logger.info(f"📥 RQ: Enqueued transcription job {transcript_job.id}")

        # Job 2: Recognize speakers (depends on transcription)
        speaker_job = transcription_queue.enqueue(
            recognise_speakers_job,
            conversation_id,
            version_id,
            str(full_audio_path),
            str(user.user_id),
            "",  # transcript_text - will be read from DB
            [],  # words - will be read from DB
            depends_on=transcript_job,
            job_timeout=600,
            result_ttl=JOB_RESULT_TTL,
            job_id=f"speaker_{conversation_id[:8]}",
            description=f"Recognize speakers for {conversation_id[:8]}",
            meta={'audio_uuid': audio_uuid}
        )
        logger.info(f"📥 RQ: Enqueued speaker recognition job {speaker_job.id} (depends on {transcript_job.id})")

        # Job 3: Extract memories (depends on speaker recognition)
        memory_job = memory_queue.enqueue(
            process_memory_job,
            None,  # client_id - will be read from conversation in DB
            str(user.user_id),
            "",  # user_email - will be read from user in DB
            conversation_id,
            depends_on=speaker_job,
            job_timeout=1800,
            result_ttl=JOB_RESULT_TTL,
            job_id=f"memory_{conversation_id[:8]}",
            description=f"Extract memories for {conversation_id[:8]}",
            meta={'audio_uuid': audio_uuid}
        )
        logger.info(f"📥 RQ: Enqueued memory job {memory_job.id} (depends on {speaker_job.id})")

        job = transcript_job  # For backward compatibility with return value
        logger.info(f"Created transcript reprocessing job {job.id} (version: {version_id}) for conversation {conversation_id}")

        return JSONResponse(content={
            "message": f"Transcript reprocessing started for conversation {conversation_id}",
            "job_id": job.id,
            "version_id": version_id,
            "status": "queued"
        })

    except Exception as e:
        logger.error(f"Error starting transcript reprocessing: {e}")
        return JSONResponse(status_code=500, content={"error": "Error starting transcript reprocessing"})


async def reprocess_memory(conversation_id: str, transcript_version_id: str, user: User):
    """Reprocess memory extraction for a specific transcript version. Users can only reprocess their own conversations."""
    try:
        # Find the conversation using Beanie
        conversation_model = await Conversation.find_one(Conversation.conversation_id == conversation_id)
        if not conversation_model:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser and conversation_model.user_id != str(user.user_id):
            return JSONResponse(status_code=403, content={"error": "Access forbidden. You can only reprocess your own conversations."})

        # Get audio_uuid for processing run tracking
        audio_uuid = conversation_model.audio_uuid

        # Resolve transcript version ID
        # Handle special "active" version ID
        if transcript_version_id == "active":
            active_version_id = conversation_model.active_transcript_version
            if not active_version_id:
                return JSONResponse(
                    status_code=404, content={"error": "No active transcript version found"}
                )
            transcript_version_id = active_version_id

        # Find the specific transcript version
        transcript_version = None
        for version in conversation_model.transcript_versions:
            if version.version_id == transcript_version_id:
                transcript_version = version
                break

        if not transcript_version:
            return JSONResponse(
                status_code=404, content={"error": f"Transcript version '{transcript_version_id}' not found"}
            )

        # Create new memory version ID
        import uuid
        version_id = str(uuid.uuid4())

        # Enqueue memory processing job with RQ (RQ handles job tracking)
        from advanced_omi_backend.workers.memory_jobs import enqueue_memory_processing
        from advanced_omi_backend.models.job import JobPriority

        job = enqueue_memory_processing(
            client_id=conversation_model.client_id,
            user_id=str(user.user_id),
            user_email=user.email,
            conversation_id=conversation_id,
            priority=JobPriority.NORMAL
        )

        logger.info(f"Created memory reprocessing job {job.id} (version {version_id}) for conversation {conversation_id}")

        return JSONResponse(content={
            "message": f"Memory reprocessing started for conversation {conversation_id}",
            "job_id": job.id,
            "version_id": version_id,
            "transcript_version_id": transcript_version_id,
            "status": "queued"
        })

    except Exception as e:
        logger.error(f"Error starting memory reprocessing: {e}")
        return JSONResponse(status_code=500, content={"error": "Error starting memory reprocessing"})


async def activate_transcript_version(conversation_id: str, version_id: str, user: User):
    """Activate a specific transcript version. Users can only modify their own conversations."""
    try:
        # Find the conversation using Beanie
        conversation_model = await Conversation.find_one(Conversation.conversation_id == conversation_id)
        if not conversation_model:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser and conversation_model.user_id != str(user.user_id):
            return JSONResponse(status_code=403, content={"error": "Access forbidden. You can only modify your own conversations."})

        # Activate the transcript version using Beanie model method
        success = conversation_model.set_active_transcript_version(version_id)
        if not success:
            return JSONResponse(
                status_code=400, content={"error": "Failed to activate transcript version"}
            )

        await conversation_model.save()

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
        # Find the conversation using Beanie
        conversation_model = await Conversation.find_one(Conversation.conversation_id == conversation_id)
        if not conversation_model:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser and conversation_model.user_id != str(user.user_id):
            return JSONResponse(status_code=403, content={"error": "Access forbidden. You can only modify your own conversations."})

        # Activate the memory version using Beanie model method
        success = conversation_model.set_active_memory_version(version_id)
        if not success:
            return JSONResponse(
                status_code=400, content={"error": "Failed to activate memory version"}
            )

        await conversation_model.save()

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
        # Find the conversation using Beanie to check ownership
        conversation_model = await Conversation.find_one(Conversation.conversation_id == conversation_id)
        if not conversation_model:
            return JSONResponse(status_code=404, content={"error": "Conversation not found"})

        # Check ownership for non-admin users
        if not user.is_superuser and conversation_model.user_id != str(user.user_id):
            return JSONResponse(status_code=403, content={"error": "Access forbidden. You can only access your own conversations."})

        # Get version history from model
        # Convert datetime objects to ISO strings for JSON serialization
        transcript_versions = []
        for v in conversation_model.transcript_versions:
            version_dict = v.model_dump()
            if version_dict.get('created_at'):
                version_dict['created_at'] = version_dict['created_at'].isoformat()
            transcript_versions.append(version_dict)

        memory_versions = []
        for v in conversation_model.memory_versions:
            version_dict = v.model_dump()
            if version_dict.get('created_at'):
                version_dict['created_at'] = version_dict['created_at'].isoformat()
            memory_versions.append(version_dict)

        history = {
            "conversation_id": conversation_id,
            "active_transcript_version": conversation_model.active_transcript_version,
            "active_memory_version": conversation_model.active_memory_version,
            "transcript_versions": transcript_versions,
            "memory_versions": memory_versions
        }

        return JSONResponse(content=history)

    except Exception as e:
        logger.error(f"Error fetching version history: {e}")
        return JSONResponse(status_code=500, content={"error": "Error fetching version history"})
