"""
Queue Controller - RQ queue configuration, management and monitoring.

This module provides:
- Queue setup and configuration
- Job statistics and monitoring
- Queue health checks
- Beanie initialization for workers
"""

import asyncio
import os
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

import redis
from rq import Queue, Worker
from rq.job import Job
from rq.registry import ScheduledJobRegistry, DeferredJobRegistry

from advanced_omi_backend.models.job import JobPriority
from advanced_omi_backend.models.conversation import Conversation

logger = logging.getLogger(__name__)

# Redis connection configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = redis.from_url(REDIS_URL)

# Queue name constants
TRANSCRIPTION_QUEUE = "transcription"
MEMORY_QUEUE = "memory"
AUDIO_QUEUE = "audio"
DEFAULT_QUEUE = "default"

# Centralized list of all queue names
QUEUE_NAMES = [DEFAULT_QUEUE, TRANSCRIPTION_QUEUE, MEMORY_QUEUE, AUDIO_QUEUE]

# Job retention configuration
JOB_RESULT_TTL = int(os.getenv("RQ_RESULT_TTL", 3600))  # 1 hour default

# Create queues with custom result TTL
transcription_queue = Queue(TRANSCRIPTION_QUEUE, connection=redis_conn, default_timeout=300)
memory_queue = Queue(MEMORY_QUEUE, connection=redis_conn, default_timeout=300)
audio_queue = Queue(AUDIO_QUEUE, connection=redis_conn, default_timeout=3600)  # 1 hour timeout for long sessions
default_queue = Queue(DEFAULT_QUEUE, connection=redis_conn, default_timeout=300)


def get_queue(queue_name: str = DEFAULT_QUEUE) -> Queue:
    """Get an RQ queue by name."""
    queues = {
        TRANSCRIPTION_QUEUE: transcription_queue,
        MEMORY_QUEUE: memory_queue,
        AUDIO_QUEUE: audio_queue,
        DEFAULT_QUEUE: default_queue,
    }
    return queues.get(queue_name, default_queue)


def get_job_stats() -> Dict[str, Any]:
    """Get statistics about jobs in all queues matching frontend expectations."""
    total_jobs = 0
    queued_jobs = 0
    processing_jobs = 0
    completed_jobs = 0
    failed_jobs = 0
    cancelled_jobs = 0
    deferred_jobs = 0  # Jobs waiting for dependencies (depends_on)

    for queue_name in QUEUE_NAMES:
        queue = get_queue(queue_name)

        queued_jobs += len(queue)
        processing_jobs += len(queue.started_job_registry)
        completed_jobs += len(queue.finished_job_registry)
        failed_jobs += len(queue.failed_job_registry)
        cancelled_jobs += len(queue.canceled_job_registry)
        deferred_jobs += len(queue.deferred_job_registry)

    total_jobs = queued_jobs + processing_jobs + completed_jobs + failed_jobs + cancelled_jobs + deferred_jobs

    return {
        "total_jobs": total_jobs,
        "queued_jobs": queued_jobs,
        "processing_jobs": processing_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "cancelled_jobs": cancelled_jobs,
        "deferred_jobs": deferred_jobs,
        "timestamp": datetime.utcnow().isoformat()
    }


def get_jobs(limit: int = 20, offset: int = 0, queue_name: str = None) -> Dict[str, Any]:
    """
    Get jobs from a specific queue or all queues.

    Args:
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        queue_name: Specific queue name or None for all queues

    Returns:
        Dict with jobs list and pagination metadata matching frontend expectations
    """
    all_jobs = []

    queues_to_check = [queue_name] if queue_name else QUEUE_NAMES

    for qname in queues_to_check:
        queue = get_queue(qname)

        # Collect jobs from all registries
        registries = [
            (queue.job_ids, "queued"),
            (queue.started_job_registry.get_job_ids(), "processing"),
            (queue.finished_job_registry.get_job_ids(), "completed"),
            (queue.failed_job_registry.get_job_ids(), "failed"),
            (queue.deferred_job_registry.get_job_ids(), "deferred"),  # Jobs waiting for dependencies
        ]

        for job_ids, status in registries:
            for job_id in job_ids:
                try:
                    job = Job.fetch(job_id, connection=redis_conn)

                    # Extract user_id from kwargs if present
                    user_id = job.kwargs.get("user_id", "") if job.kwargs else ""

                    # Extract just the function name (e.g., "listen_for_speech_job" from "module.listen_for_speech_job")
                    job_type = job.func_name.split('.')[-1] if job.func_name else "unknown"

                    all_jobs.append({
                        "job_id": job.id,
                        "job_type": job_type,
                        "user_id": user_id,
                        "status": status,
                        "priority": "normal",  # RQ doesn't track priority in metadata
                        "data": {
                            "description": job.description or "",
                            "queue": qname,
                        },
                        "result": job.result if hasattr(job, 'result') else None,
                        "meta": job.meta if job.meta else {},  # Include job metadata
                        "error_message": str(job.exc_info) if job.exc_info else None,
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                        "started_at": job.started_at.isoformat() if job.started_at else None,
                        "completed_at": job.ended_at.isoformat() if job.ended_at else None,
                        "retry_count": job.retries_left if hasattr(job, 'retries_left') else 0,
                        "max_retries": 3,  # Default max retries
                        "progress_percent": 0,  # RQ doesn't track progress by default
                        "progress_message": "",
                    })
                except Exception as e:
                    logger.error(f"Error fetching job {job_id}: {e}")

    # Sort by created_at (most recent first)
    all_jobs.sort(key=lambda x: x.get("created_at") or "", reverse=True)

    # Paginate
    total_jobs = len(all_jobs)
    paginated_jobs = all_jobs[offset:offset + limit]
    has_more = (offset + limit) < total_jobs

    return {
        "jobs": paginated_jobs,
        "pagination": {
            "total": total_jobs,
            "limit": limit,
            "offset": offset,
            "has_more": has_more,
        }
    }


def all_jobs_complete_for_session(session_id: str) -> bool:
    """
    Check if all jobs associated with a session are in terminal states.

    Only checks jobs with audio_uuid in job.meta (no backward compatibility).
    Traverses dependency chains to include dependent jobs.

    Args:
        session_id: The audio_uuid (session ID) to check jobs for

    Returns:
        True if all jobs are complete (or no jobs found), False if any job is still processing
    """
    processed_job_ids = set()

    def is_job_complete(job):
        """Recursively check if job and all its dependents are terminal."""
        if job.id in processed_job_ids:
            return True
        processed_job_ids.add(job.id)

        # Check if this job is terminal
        if not (job.is_finished or job.is_failed or job.is_canceled):
            logger.debug(f"Job {job.id} ({job.func_name}) is not terminal")
            return False

        # Check dependent jobs
        for dep_id in (job.dependent_ids or []):
            try:
                dep_job = Job.fetch(dep_id, connection=redis_conn)
                if not is_job_complete(dep_job):
                    return False
            except Exception as e:
                logger.debug(f"Error fetching dependent job {dep_id}: {e}")

        return True

    # Find all jobs for this session
    all_queues = [transcription_queue, memory_queue, audio_queue, default_queue]
    for queue in all_queues:
        registries = [
            queue.job_ids,
            queue.started_job_registry.get_job_ids(),
            queue.finished_job_registry.get_job_ids(),
            queue.failed_job_registry.get_job_ids(),
            queue.canceled_job_registry.get_job_ids(),
            ScheduledJobRegistry(queue=queue).get_job_ids(),
            DeferredJobRegistry(queue=queue).get_job_ids(),
        ]

        for job_ids in registries:
            for job_id in job_ids:
                try:
                    job = Job.fetch(job_id, connection=redis_conn)

                    # Only check jobs with audio_uuid in meta
                    if job.meta and job.meta.get('audio_uuid') == session_id:
                        if not is_job_complete(job):
                            return False
                except Exception as e:
                    logger.debug(f"Error checking job {job_id}: {e}")

    return True


def start_streaming_jobs(
    session_id: str,
    user_id: str,
    client_id: str
) -> Dict[str, str]:
    """
    Enqueue jobs for streaming audio session (initial session setup).

    This starts the parallel job processing for a NEW streaming session:
    1. Speech detection job - monitors transcription results for speech
    2. Audio persistence job - writes audio chunks to WAV file (file rotation per conversation)

    Args:
        session_id: Stream session ID (audio_uuid)
        user_id: User identifier
        client_id: Client identifier

    Returns:
        Dict with job IDs: {'speech_detection': job_id, 'audio_persistence': job_id}

    Note: user_email is fetched from the database when needed.
    """
    from advanced_omi_backend.workers.transcription_jobs import stream_speech_detection_job
    from advanced_omi_backend.workers.audio_jobs import audio_streaming_persistence_job

    # Enqueue speech detection job
    speech_job = transcription_queue.enqueue(
        stream_speech_detection_job,
        session_id,
        user_id,
        client_id,
        job_timeout=3600,  # 1 hour for long recordings
        result_ttl=JOB_RESULT_TTL,
        job_id=f"speech-detect_{session_id[:12]}",
        description=f"Listening for speech...",
        meta={'audio_uuid': session_id, 'client_id': client_id, 'session_level': True}
    )
    logger.info(f"📥 RQ: Enqueued speech detection job {speech_job.id}")

    # Store job ID for cleanup (keyed by client_id for easy WebSocket cleanup)
    try:
        redis_conn.set(f"speech_detection_job:{client_id}", speech_job.id, ex=3600)  # 1 hour TTL
        logger.info(f"📌 Stored speech detection job ID for client {client_id}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to store job ID for {client_id}: {e}")

    # Enqueue audio persistence job on dedicated audio queue
    # NOTE: This job handles file rotation for multiple conversations automatically
    # Runs for entire session, not tied to individual conversations
    audio_job = audio_queue.enqueue(
        audio_streaming_persistence_job,
        session_id,
        user_id,
        client_id,
        job_timeout=3600,  # 1 hour for long recordings
        result_ttl=JOB_RESULT_TTL,
        job_id=f"audio-persist_{session_id[:12]}",
        description=f"Audio persistence for session {session_id[:12]}",
        meta={'audio_uuid': session_id, 'session_level': True}  # Mark as session-level job
    )
    logger.info(f"📥 RQ: Enqueued audio persistence job {audio_job.id} on audio queue")

    return {
        'speech_detection': speech_job.id,
        'audio_persistence': audio_job.id
    }


def start_post_conversation_jobs(
    conversation_id: str,
    audio_uuid: str,
    audio_file_path: str,
    user_id: str,
    post_transcription: bool = True,
    transcript_version_id: Optional[str] = None,
    depends_on_job = None
) -> Dict[str, str]:
    """
    Start post-conversation processing jobs after conversation is created.

    This creates the standard processing chain after a conversation is created:
    1. [Optional] Transcription job - Batch transcription (if post_transcription=True)
    2. Audio cropping job - Removes silence from audio
    3. Speaker recognition job - Identifies speakers in audio
    4. Memory extraction job - Extracts memories from conversation (parallel)
    5. Title/summary generation job - Generates title and summary (parallel)

    Args:
        conversation_id: Conversation identifier
        audio_uuid: Audio UUID for job tracking
        audio_file_path: Path to audio file
        user_id: User identifier
        post_transcription: If True, run batch transcription step (for uploads)
                           If False, skip transcription (streaming already has it)
        transcript_version_id: Transcript version ID (auto-generated if None)
        depends_on_job: Optional job dependency for cropping job

    Returns:
        Dict with job IDs (transcription will be None if post_transcription=False)
    """
    from advanced_omi_backend.workers.transcription_jobs import transcribe_full_audio_job
    from advanced_omi_backend.workers.speaker_jobs import recognise_speakers_job
    from advanced_omi_backend.workers.audio_jobs import process_cropping_job
    from advanced_omi_backend.workers.memory_jobs import process_memory_job
    from advanced_omi_backend.workers.conversation_jobs import generate_title_summary_job

    version_id = transcript_version_id or str(uuid.uuid4())

    # Step 1: Batch transcription job (ALWAYS run to get correct conversation-relative timestamps)
    # Even for streaming, we need batch transcription before cropping to fix cumulative timestamps
    transcribe_job_id = f"transcribe_{conversation_id[:12]}"
    logger.info(f"🔍 DEBUG: Creating transcribe job with job_id={transcribe_job_id}, conversation_id={conversation_id[:12]}, audio_uuid={audio_uuid[:12]}")

    transcription_job = transcription_queue.enqueue(
        transcribe_full_audio_job,
        conversation_id,
        audio_uuid,
        audio_file_path,
        version_id,
        "batch",  # trigger
        job_timeout=1800,  # 30 minutes
        result_ttl=JOB_RESULT_TTL,
        depends_on=depends_on_job,
        job_id=transcribe_job_id,
        description=f"Transcribe conversation {conversation_id[:8]}",
        meta={'audio_uuid': audio_uuid, 'conversation_id': conversation_id}
    )
    logger.info(f"📥 RQ: Enqueued transcription job {transcription_job.id}, meta={transcription_job.meta}")
    crop_depends_on = transcription_job

    # Step 2: Audio cropping job (depends on transcription if it ran, otherwise depends_on_job)
    crop_job_id = f"crop_{conversation_id[:12]}"
    logger.info(f"🔍 DEBUG: Creating crop job with job_id={crop_job_id}, conversation_id={conversation_id[:12]}, audio_uuid={audio_uuid[:12]}")

    cropping_job = default_queue.enqueue(
        process_cropping_job,
        conversation_id,
        audio_file_path,
        job_timeout=300,  # 5 minutes
        result_ttl=JOB_RESULT_TTL,
        depends_on=crop_depends_on,
        job_id=crop_job_id,
        description=f"Crop audio for conversation {conversation_id[:8]}",
        meta={'audio_uuid': audio_uuid, 'conversation_id': conversation_id}
    )
    logger.info(f"📥 RQ: Enqueued cropping job {cropping_job.id}, meta={cropping_job.meta}")

    # Speaker recognition depends on cropping
    speaker_depends_on = cropping_job

    # Step 3: Speaker recognition job
    speaker_job_id = f"speaker_{conversation_id[:12]}"
    logger.info(f"🔍 DEBUG: Creating speaker job with job_id={speaker_job_id}, conversation_id={conversation_id[:12]}, audio_uuid={audio_uuid[:12]}")

    speaker_job = transcription_queue.enqueue(
        recognise_speakers_job,
        conversation_id,
        version_id,
        audio_file_path,
        "",  # transcript_text - will be read from DB
        [],  # words - will be read from DB
        job_timeout=1200,  # 20 minutes
        result_ttl=JOB_RESULT_TTL,
        depends_on=speaker_depends_on,
        job_id=speaker_job_id,
        description=f"Speaker recognition for conversation {conversation_id[:8]}",
        meta={'audio_uuid': audio_uuid, 'conversation_id': conversation_id}
    )
    logger.info(f"📥 RQ: Enqueued speaker recognition job {speaker_job.id}, meta={speaker_job.meta} (depends on {speaker_depends_on.id})")

    # Step 4: Memory extraction job (parallel with title/summary)
    memory_job_id = f"memory_{conversation_id[:12]}"
    logger.info(f"🔍 DEBUG: Creating memory job with job_id={memory_job_id}, conversation_id={conversation_id[:12]}, audio_uuid={audio_uuid[:12]}")

    memory_job = memory_queue.enqueue(
        process_memory_job,
        conversation_id,
        job_timeout=900,  # 15 minutes
        result_ttl=JOB_RESULT_TTL,
        depends_on=speaker_job,
        job_id=memory_job_id,
        description=f"Memory extraction for conversation {conversation_id[:8]}",
        meta={'audio_uuid': audio_uuid, 'conversation_id': conversation_id}
    )
    logger.info(f"📥 RQ: Enqueued memory extraction job {memory_job.id}, meta={memory_job.meta} (depends on {speaker_job.id})")

    # Step 5: Title/summary generation job (parallel with memory, independent)
    # This ensures conversations always get titles/summaries even if memory job fails
    title_job_id = f"title_summary_{conversation_id[:12]}"
    logger.info(f"🔍 DEBUG: Creating title/summary job with job_id={title_job_id}, conversation_id={conversation_id[:12]}, audio_uuid={audio_uuid[:12]}")

    title_summary_job = default_queue.enqueue(
        generate_title_summary_job,
        conversation_id,
        job_timeout=300,  # 5 minutes
        result_ttl=JOB_RESULT_TTL,
        depends_on=speaker_job,  # Depends on speaker job, NOT memory job
        job_id=title_job_id,
        description=f"Generate title and summary for conversation {conversation_id[:8]}",
        meta={'audio_uuid': audio_uuid, 'conversation_id': conversation_id}
    )
    logger.info(f"📥 RQ: Enqueued title/summary job {title_summary_job.id}, meta={title_summary_job.meta} (depends on {speaker_job.id})")

    return {
        'cropping': cropping_job.id,
        'transcription': transcription_job.id if transcription_job else None,
        'speaker_recognition': speaker_job.id,
        'memory': memory_job.id,
        'title_summary': title_summary_job.id
    }




def get_queue_health() -> Dict[str, Any]:
    """Get health status of all queues and workers."""
    health = {
        "queues": {},
        "workers": [],
        "redis_connection": "unknown",
        "total_workers": 0,
        "active_workers": 0,
        "idle_workers": 0,
    }

    # Check Redis connection
    try:
        redis_conn.ping()
        health["redis_connection"] = "healthy"
    except Exception as e:
        health["redis_connection"] = f"unhealthy: {e}"
        return health

    # Check each queue
    for queue_name in QUEUE_NAMES:
        queue = get_queue(queue_name)
        health["queues"][queue_name] = {
            "count": len(queue),
            "failed_count": len(queue.failed_job_registry),
            "finished_count": len(queue.finished_job_registry),
            "started_count": len(queue.started_job_registry),
        }

    # Check workers
    workers = Worker.all(connection=redis_conn)
    health["total_workers"] = len(workers)

    for worker in workers:
        state = worker.get_state()
        current_job = worker.get_current_job_id()

        # Count active vs idle workers
        if current_job or state == "busy":
            health["active_workers"] += 1
        else:
            health["idle_workers"] += 1

        health["workers"].append({
            "name": worker.name,
            "state": state,
            "queues": [q.name for q in worker.queues],
            "current_job": current_job,
        })

    return health

# needs tidying but works for now
async def cleanup_stuck_stream_workers(request):
    """Clean up stuck Redis Stream consumers and pending messages from all active streams."""
    import time
    from fastapi.responses import JSONResponse

    try:
        # Get Redis client from request.app.state (initialized during startup)
        redis_client = request.app.state.redis_audio_stream

        if not redis_client:
            return JSONResponse(
                status_code=503,
                content={"error": "Redis client for audio streaming not initialized"}
            )

        cleanup_results = {}
        total_cleaned = 0
        total_deleted_consumers = 0
        total_deleted_streams = 0
        current_time = time.time()

        # Discover all audio streams (per-client streams)
        stream_keys = await redis_client.keys("audio:stream:*")

        for stream_key in stream_keys:
            stream_name = stream_key.decode() if isinstance(stream_key, bytes) else stream_key

            try:
                # First check stream age - delete old streams (>1 hour) immediately
                stream_info = await redis_client.execute_command('XINFO', 'STREAM', stream_name)

                # Parse stream info
                info_dict = {}
                for i in range(0, len(stream_info), 2):
                    key_name = stream_info[i].decode() if isinstance(stream_info[i], bytes) else str(stream_info[i])
                    info_dict[key_name] = stream_info[i+1]

                stream_length = int(info_dict.get("length", 0))
                last_entry = info_dict.get("last-entry")

                # Check if stream is old
                should_delete_stream = False
                stream_age = 0

                if stream_length == 0:
                    should_delete_stream = True
                    stream_age = 0
                elif last_entry and isinstance(last_entry, list) and len(last_entry) > 0:
                    try:
                        last_id = last_entry[0]
                        if isinstance(last_id, bytes):
                            last_id = last_id.decode()
                        last_timestamp_ms = int(last_id.split('-')[0])
                        last_timestamp_s = last_timestamp_ms / 1000
                        stream_age = current_time - last_timestamp_s

                        # Delete streams older than 1 hour (3600 seconds)
                        if stream_age > 3600:
                            should_delete_stream = True
                    except (ValueError, IndexError):
                        pass

                if should_delete_stream:
                    await redis_client.delete(stream_name)
                    total_deleted_streams += 1
                    cleanup_results[stream_name] = {
                        "message": f"Deleted old stream (age: {stream_age:.0f}s, length: {stream_length})",
                        "cleaned": 0,
                        "deleted_consumers": 0,
                        "deleted_stream": True,
                        "stream_age": stream_age
                    }
                    continue

                # Get consumer groups
                groups = await redis_client.execute_command('XINFO', 'GROUPS', stream_name)

                if not groups:
                    cleanup_results[stream_name] = {"message": "No consumer groups found", "cleaned": 0, "deleted_stream": False}
                    continue

                # Parse first group
                group_dict = {}
                group = groups[0]
                for i in range(0, len(group), 2):
                    key = group[i].decode() if isinstance(group[i], bytes) else str(group[i])
                    value = group[i+1]
                    if isinstance(value, bytes):
                        try:
                            value = value.decode()
                        except UnicodeDecodeError:
                            value = str(value)
                    group_dict[key] = value

                group_name = group_dict.get("name", "unknown")
                if isinstance(group_name, bytes):
                    group_name = group_name.decode()

                pending_count = int(group_dict.get("pending", 0))

                # Get consumers for this group to check per-consumer pending
                consumers = await redis_client.execute_command('XINFO', 'CONSUMERS', stream_name, group_name)

                cleaned_count = 0
                total_consumer_pending = 0

                # Clean up pending messages for each consumer AND delete dead consumers
                deleted_consumers = 0
                for consumer in consumers:
                    consumer_dict = {}
                    for i in range(0, len(consumer), 2):
                        key = consumer[i].decode() if isinstance(consumer[i], bytes) else str(consumer[i])
                        value = consumer[i+1]
                        if isinstance(value, bytes):
                            try:
                                value = value.decode()
                            except UnicodeDecodeError:
                                value = str(value)
                        consumer_dict[key] = value

                    consumer_name = consumer_dict.get("name", "unknown")
                    if isinstance(consumer_name, bytes):
                        consumer_name = consumer_name.decode()

                    consumer_pending = int(consumer_dict.get("pending", 0))
                    consumer_idle_ms = int(consumer_dict.get("idle", 0))
                    total_consumer_pending += consumer_pending

                    # Check if consumer is dead (idle > 5 minutes = 300000ms)
                    is_dead = consumer_idle_ms > 300000

                    if consumer_pending > 0:
                        logger.info(f"Found {consumer_pending} pending messages for consumer {consumer_name} (idle: {consumer_idle_ms}ms)")

                        # Get pending messages for this specific consumer
                        try:
                            pending_messages = await redis_client.execute_command(
                                'XPENDING', stream_name, group_name, '-', '+', str(consumer_pending), consumer_name
                            )

                            # XPENDING returns flat list: [msg_id, consumer, idle_ms, delivery_count, msg_id, ...]
                            # Parse in groups of 4
                            for i in range(0, len(pending_messages), 4):
                                if i < len(pending_messages):
                                    msg_id = pending_messages[i]
                                    if isinstance(msg_id, bytes):
                                        msg_id = msg_id.decode()

                                    # Claim the message to a cleanup worker
                                    try:
                                        await redis_client.execute_command(
                                            'XCLAIM', stream_name, group_name, 'cleanup-worker', '0', msg_id
                                        )

                                        # Acknowledge it immediately
                                        await redis_client.xack(stream_name, group_name, msg_id)
                                        cleaned_count += 1
                                    except Exception as claim_error:
                                        logger.warning(f"Failed to claim/ack message {msg_id}: {claim_error}")

                        except Exception as consumer_error:
                            logger.error(f"Error processing consumer {consumer_name}: {consumer_error}")

                    # Delete dead consumers (idle > 5 minutes with no pending messages)
                    if is_dead and consumer_pending == 0:
                        try:
                            await redis_client.execute_command(
                                'XGROUP', 'DELCONSUMER', stream_name, group_name, consumer_name
                            )
                            deleted_consumers += 1
                            logger.info(f"🧹 Deleted dead consumer {consumer_name} (idle: {consumer_idle_ms}ms)")
                        except Exception as delete_error:
                            logger.warning(f"Failed to delete consumer {consumer_name}: {delete_error}")

                if total_consumer_pending == 0 and deleted_consumers == 0:
                    cleanup_results[stream_name] = {"message": "No pending messages or dead consumers", "cleaned": 0, "deleted_consumers": 0, "deleted_stream": False}
                    continue

                total_cleaned += cleaned_count
                total_deleted_consumers += deleted_consumers
                cleanup_results[stream_name] = {
                    "message": f"Cleaned {cleaned_count} pending messages, deleted {deleted_consumers} dead consumers",
                    "cleaned": cleaned_count,
                    "deleted_consumers": deleted_consumers,
                    "deleted_stream": False,
                    "original_pending": pending_count
                }

            except Exception as e:
                cleanup_results[stream_name] = {
                    "error": str(e),
                    "cleaned": 0
                }

        return {
            "success": True,
            "total_cleaned": total_cleaned,
            "total_deleted_consumers": total_deleted_consumers,
            "total_deleted_streams": total_deleted_streams,
            "streams": cleanup_results,  # New key for per-stream results
            "providers": cleanup_results,  # Keep for backward compatibility with frontend
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error cleaning up stuck workers: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": f"Failed to cleanup stuck workers: {str(e)}"}
        )
