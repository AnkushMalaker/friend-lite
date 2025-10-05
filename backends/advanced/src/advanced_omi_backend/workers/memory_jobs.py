"""
Memory-related RQ job functions.

This module contains jobs related to memory extraction and processing.
"""

import logging
from typing import Dict, Any

from advanced_omi_backend.models.job import JobPriority

from advanced_omi_backend.controllers.queue_controller import (
    memory_queue,
    _ensure_beanie_initialized,
    JOB_RESULT_TTL,
)

logger = logging.getLogger(__name__)


def process_memory_job(
    client_id: str,
    user_id: str,
    user_email: str,
    conversation_id: str
) -> Dict[str, Any]:
    """
    RQ job function for memory extraction and processing.

    This function is executed by RQ workers and can survive server restarts.
    """
    import asyncio
    import time
    from datetime import UTC, datetime
    from advanced_omi_backend.models.conversation import Conversation
    from advanced_omi_backend.memory import get_memory_service
    from advanced_omi_backend.users import get_user_by_id

    try:
        logger.info(f"🔄 RQ: Starting memory processing for conversation {conversation_id}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def process():
                # Initialize Beanie in this worker process
                await _ensure_beanie_initialized()

                start_time = time.time()

                # Get conversation data
                conversation_model = await Conversation.find_one(Conversation.conversation_id == conversation_id)
                if not conversation_model:
                    logger.warning(f"No conversation found for {conversation_id}")
                    return {"success": False, "error": "Conversation not found"}

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
                    client_id,
                    conversation_id,
                    user_id,
                    user_email,
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
                        logger.info(f"✅ RQ: Completed memory processing for conversation {conversation_id} - created {len(created_memory_ids)} memories in {processing_time:.2f}s")

                        return {
                            "success": True,
                            "memories_created": len(created_memory_ids),
                            "processing_time": processing_time
                        }
                    else:
                        return {"success": True, "memories_created": 0, "skipped": True}
                else:
                    return {"success": False, "error": "Memory service returned False"}

            result = loop.run_until_complete(process())
            return result

        finally:
            loop.close()

    except Exception as e:
        logger.error(f"❌ RQ: Memory processing failed for conversation {conversation_id}: {e}")
        raise


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
