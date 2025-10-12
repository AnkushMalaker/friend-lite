"""
Memory-related RQ job functions.

This module contains jobs related to memory extraction and processing.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Dict, Any

from advanced_omi_backend.models.job import JobPriority, BaseRQJob, async_job
from advanced_omi_backend.controllers.queue_controller import (
    memory_queue,
    JOB_RESULT_TTL,
)

logger = logging.getLogger(__name__)


@async_job(redis=True, beanie=True)
async def process_memory_job(
    client_id: str,
    user_id: str,
    user_email: str,
    conversation_id: str,
    redis_client=None
) -> Dict[str, Any]:
    """
    RQ job function for memory extraction and processing from conversations.

    Args:
        client_id: Client identifier
        user_id: User ID
        user_email: User email
        conversation_id: Conversation ID to process
        redis_client: Redis client (injected by decorator)

    Returns:
        Dict with processing results
    """
    from advanced_omi_backend.models.conversation import Conversation
    from advanced_omi_backend.memory import get_memory_service
    from advanced_omi_backend.users import get_user_by_id

    start_time = time.time()
    logger.info(f"🔄 Starting memory processing for conversation {conversation_id}")

    # Get conversation data
    conversation_model = await Conversation.find_one(Conversation.conversation_id == conversation_id)
    if not conversation_model:
        logger.warning(f"No conversation found for {conversation_id}")
        return {"success": False, "error": "Conversation not found"}

    # Read client_id and user_email from conversation/user if not provided
    # (Parameters may be empty if called via job dependency)
    actual_client_id = client_id or conversation_model.client_id
    actual_user_email = user_email

    if not actual_user_email:
        user = await get_user_by_id(user_id)
        if user:
            actual_user_email = user.email
        else:
            logger.warning(f"Could not find user {user_id}")
            actual_user_email = ""

    logger.info(f"🔄 Processing memory for conversation {conversation_id}, client={actual_client_id}, user={user_id}")

    # Extract conversation text from transcript segments
    full_conversation = ""
    segments = conversation_model.segments
    if segments:
        dialogue_lines = []
        for segment in segments:
            # Handle both dict and object segments
            if isinstance(segment, dict):
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "Unknown")
            else:
                text = getattr(segment, "text", "").strip()
                speaker = getattr(segment, "speaker", "Unknown")

            if text:
                dialogue_lines.append(f"{speaker}: {text}")
        full_conversation = "\n".join(dialogue_lines)
    elif conversation_model.transcript and isinstance(conversation_model.transcript, str):
        # Fallback: if segments are empty but transcript text exists
        full_conversation = conversation_model.transcript

    if len(full_conversation) < 10:
        logger.warning(f"Conversation too short for memory processing: {conversation_id}")
        return {"success": False, "error": "Conversation too short"}

    # Check primary speakers filter
    user = await get_user_by_id(user_id)
    if user and user.primary_speakers:
        transcript_speakers = set()
        for segment in conversation_model.segments:
            # Handle both dict and object segments
            if isinstance(segment, dict):
                identified_as = segment.get('identified_as')
            else:
                identified_as = getattr(segment, 'identified_as', None)

            if identified_as and identified_as != 'Unknown':
                transcript_speakers.add(identified_as.strip().lower())

        primary_speaker_names = {ps['name'].strip().lower() for ps in user.primary_speakers}

        if transcript_speakers and not transcript_speakers.intersection(primary_speaker_names):
            logger.info(f"Skipping memory - no primary speakers found in conversation {conversation_id}")
            return {"success": True, "skipped": True, "reason": "No primary speakers"}

    # Process memory
    memory_service = get_memory_service()
    memory_result = await memory_service.add_memory(
        full_conversation,
        actual_client_id,
        conversation_id,
        user_id,
        actual_user_email,
        allow_update=True,
    )

    if memory_result:
        success, created_memory_ids = memory_result

        if success and created_memory_ids:
            # Add memory references to conversation
            conversation_model = await Conversation.find_one(Conversation.conversation_id == conversation_id)
            if conversation_model:
                memory_refs = [
                    {"memory_id": mid, "created_at": datetime.now(UTC).isoformat(), "status": "created"}
                    for mid in created_memory_ids
                ]
                conversation_model.memories.extend(memory_refs)
                await conversation_model.save()

            processing_time = time.time() - start_time
            logger.info(f"✅ Completed memory processing for conversation {conversation_id} - created {len(created_memory_ids)} memories in {processing_time:.2f}s")

            # Mark session as complete in Redis (this is the last job in the chain)
            if conversation_model and conversation_model.audio_uuid:
                session_key = f"audio:session:{conversation_model.audio_uuid}"
                try:
                    await redis_client.hset(session_key, mapping={
                        "status": "complete",
                        "completed_at": str(time.time())
                    })
                    logger.info(f"✅ Marked session {conversation_model.audio_uuid} as complete (all jobs finished)")
                except Exception as e:
                    logger.warning(f"⚠️ Could not mark session as complete: {e}")

            return {
                "success": True,
                "memories_created": len(created_memory_ids),
                "processing_time": processing_time
            }
        else:
            # Mark session as complete even if no memories created
            if conversation_model and conversation_model.audio_uuid:
                session_key = f"audio:session:{conversation_model.audio_uuid}"
                try:
                    await redis_client.hset(session_key, mapping={
                        "status": "complete",
                        "completed_at": str(time.time())
                    })
                    logger.info(f"✅ Marked session {conversation_model.audio_uuid} as complete (no memories)")
                except Exception as e:
                    logger.warning(f"⚠️ Could not mark session as complete: {e}")

            return {"success": True, "memories_created": 0, "skipped": True}
    else:
        return {"success": False, "error": "Memory service returned False"}


def enqueue_memory_processing(
    client_id: str,
    user_id: str,
    user_email: str,
    conversation_id: str,
    priority: JobPriority = JobPriority.NORMAL
):
    """
    Enqueue a memory processing job.

    Returns RQ Job object for tracking.
    """
    timeout_mapping = {
        JobPriority.URGENT: 3600,  # 60 minutes
        JobPriority.HIGH: 2400,    # 40 minutes
        JobPriority.NORMAL: 1800,  # 30 minutes
        JobPriority.LOW: 900       # 15 minutes
    }

    job = memory_queue.enqueue(
        process_memory_job,
        client_id,
        user_id,
        user_email,
        conversation_id,
        job_timeout=timeout_mapping.get(priority, 1800),
        result_ttl=JOB_RESULT_TTL,
        job_id=f"memory_{conversation_id[:8]}",
        description=f"Process memory for conversation {conversation_id[:8]}"
    )

    logger.info(f"📥 RQ: Enqueued memory job {job.id} for conversation {conversation_id}")
    return job
