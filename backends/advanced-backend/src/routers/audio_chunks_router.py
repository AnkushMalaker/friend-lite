from fastapi import APIRouter, Depends, Query
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient

from utils.audio_chunk_utils import AudioChunkUtils
from database import get_db_client

router = APIRouter()

def get_audio_chunk_utils(
    db_client: AsyncIOMotorClient = Depends(get_db_client)
):
    chunks_col = db_client.get_default_database("friend-lite")["audio_chunks"]
    return AudioChunkUtils(chunks_col)

@router.get("/api/conversations")
async def get_conversations(audio_chunk_utils: AudioChunkUtils = Depends(get_audio_chunk_utils)):
    """Get all conversations grouped by client_id."""
    return await audio_chunk_utils.get_conversations()

@router.get("/api/conversations/{audio_uuid}/cropped")
async def get_cropped_audio_info(audio_uuid: str, audio_chunk_utils: AudioChunkUtils = Depends(get_audio_chunk_utils)):
    """Get cropped audio information for a specific conversation."""
    return await audio_chunk_utils.get_cropped_audio_info(audio_uuid)

@router.post("/api/conversations/{audio_uuid}/reprocess")
async def reprocess_audio_cropping(audio_uuid: str, audio_chunk_utils: AudioChunkUtils = Depends(get_audio_chunk_utils)):
    """Trigger reprocessing of audio cropping for a specific conversation."""
    return await audio_chunk_utils.reprocess_audio_cropping(audio_uuid)

@router.post("/api/conversations/{audio_uuid}/speakers")
async def add_speaker_to_conversation(audio_uuid: str, speaker_id: str, audio_chunk_utils: AudioChunkUtils = Depends(get_audio_chunk_utils)):
    """Add a speaker to the speakers_identified list for a conversation."""
    return await audio_chunk_utils.add_speaker_to_conversation(audio_uuid, speaker_id)

@router.put("/api/conversations/{audio_uuid}/transcript/{segment_index}")
async def update_transcript_segment(
    audio_uuid: str,
    segment_index: int,
    speaker_id: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    audio_chunk_utils: AudioChunkUtils = Depends(get_audio_chunk_utils)
):
    """Update a specific transcript segment with speaker or timing information."""
    return await audio_chunk_utils.update_transcript_segment(
        audio_uuid, segment_index, speaker_id, start_time, end_time
    )