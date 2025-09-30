"""
Conversation management routes for Friend-Lite API.

Handles conversation CRUD operations, audio processing, and transcript management.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query

from advanced_omi_backend.auth import current_active_user
from advanced_omi_backend.client_manager import (
    ClientManager,
    get_client_manager_dependency,
)
from advanced_omi_backend.controllers import conversation_controller
from advanced_omi_backend.users import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("/{client_id}/close")
async def close_current_conversation(
    client_id: str,
    current_user: User = Depends(current_active_user),
    client_manager: ClientManager = Depends(get_client_manager_dependency),
):
    """Close the current conversation for a specific client. Users can only close their own conversations."""
    return await conversation_controller.close_current_conversation(
        client_id, current_user, client_manager
    )


@router.get("")
async def get_conversations(current_user: User = Depends(current_active_user)):
    """Get conversations. Admins see all conversations, users see only their own."""
    return await conversation_controller.get_conversations(current_user)


@router.get("/{conversation_id}")
async def get_conversation_detail(
    conversation_id: str,
    current_user: User = Depends(current_active_user)
):
    """Get a specific conversation with full transcript details."""
    return await conversation_controller.get_conversation(conversation_id, current_user)


@router.get("/{audio_uuid}/cropped")
async def get_cropped_audio_info(
    audio_uuid: str, current_user: User = Depends(current_active_user)
):
    """Get cropped audio information for a conversation. Users can only access their own conversations."""
    return await conversation_controller.get_cropped_audio_info(audio_uuid, current_user)


# Deprecated
@router.post("/{audio_uuid}/reprocess")
async def reprocess_audio_cropping(
    audio_uuid: str, current_user: User = Depends(current_active_user)
):
    """Reprocess audio cropping for a conversation. Users can only reprocess their own conversations."""
    return await conversation_controller.reprocess_audio_cropping(audio_uuid, current_user)


@router.post("/{audio_uuid}/speakers")
async def add_speaker_to_conversation(
    audio_uuid: str, speaker_id: str, current_user: User = Depends(current_active_user)
):
    """Add a speaker to the speakers_identified list for a conversation. Users can only modify their own conversations."""
    return await conversation_controller.add_speaker_to_conversation(
        audio_uuid, speaker_id, current_user
    )


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
    return await conversation_controller.update_transcript_segment(
        audio_uuid, segment_index, current_user, speaker_id, start_time, end_time
    )


# New reprocessing endpoints
@router.post("/{conversation_id}/reprocess-transcript")
async def reprocess_transcript(
    conversation_id: str, current_user: User = Depends(current_active_user)
):
    """Reprocess transcript for a conversation. Users can only reprocess their own conversations."""
    return await conversation_controller.reprocess_transcript(conversation_id, current_user)


@router.post("/{conversation_id}/reprocess-memory")
async def reprocess_memory(
    conversation_id: str,
    current_user: User = Depends(current_active_user),
    transcript_version_id: str = Query(default="active")
):
    """Reprocess memory extraction for a specific transcript version. Users can only reprocess their own conversations."""
    return await conversation_controller.reprocess_memory(conversation_id, transcript_version_id, current_user)


@router.post("/{conversation_id}/activate-transcript/{version_id}")
async def activate_transcript_version(
    conversation_id: str,
    version_id: str,
    current_user: User = Depends(current_active_user)
):
    """Activate a specific transcript version. Users can only modify their own conversations."""
    return await conversation_controller.activate_transcript_version(conversation_id, version_id, current_user)


@router.post("/{conversation_id}/activate-memory/{version_id}")
async def activate_memory_version(
    conversation_id: str,
    version_id: str,
    current_user: User = Depends(current_active_user)
):
    """Activate a specific memory version. Users can only modify their own conversations."""
    return await conversation_controller.activate_memory_version(conversation_id, version_id, current_user)


@router.get("/{conversation_id}/versions")
async def get_conversation_version_history(
    conversation_id: str, current_user: User = Depends(current_active_user)
):
    """Get version history for a conversation. Users can only access their own conversations."""
    return await conversation_controller.get_conversation_version_history(conversation_id, current_user)


@router.delete("/{audio_uuid}")
async def delete_conversation(
    audio_uuid: str, current_user: User = Depends(current_active_user)
):
    """Delete a conversation and its associated audio file. Users can only delete their own conversations."""
    return await conversation_controller.delete_conversation(audio_uuid, current_user)
