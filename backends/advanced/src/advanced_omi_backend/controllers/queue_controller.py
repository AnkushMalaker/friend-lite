"""
Queue Controller - RQ queue configuration, management and monitoring.

This module provides:
- Queue setup and configuration
- Job statistics and monitoring
- Queue health checks
- Beanie initialization for workers
"""

import os
import logging
from typing import Dict, Any, Optional

import redis
from rq import Queue, Worker
from rq.job import Job

from advanced_omi_backend.models.job import JobPriority

logger = logging.getLogger(__name__)

# Global flag to track if Beanie is initialized in this process
_beanie_initialized = False

# Redis connection configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = redis.from_url(REDIS_URL)

# Queue name constants
TRANSCRIPTION_QUEUE = "transcription"
MEMORY_QUEUE = "memory"
DEFAULT_QUEUE = "default"

# Job retention configuration
JOB_RESULT_TTL = int(os.getenv("RQ_RESULT_TTL", 3600))  # 1 hour default

# Create queues with custom result TTL
transcription_queue = Queue(TRANSCRIPTION_QUEUE, connection=redis_conn, default_timeout=300)
memory_queue = Queue(MEMORY_QUEUE, connection=redis_conn, default_timeout=300)
default_queue = Queue(DEFAULT_QUEUE, connection=redis_conn, default_timeout=300)


def get_queue(queue_name: str = DEFAULT_QUEUE) -> Queue:
    """Get an RQ queue by name."""
    queues = {
        TRANSCRIPTION_QUEUE: transcription_queue,
        MEMORY_QUEUE: memory_queue,
        DEFAULT_QUEUE: default_queue,
    }
    return queues.get(queue_name, default_queue)


async def _ensure_beanie_initialized():
    """Ensure Beanie is initialized in the current process (for RQ workers)."""
    global _beanie_initialized

    if _beanie_initialized:
        return

    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        from beanie import init_beanie
        from advanced_omi_backend.models.conversation import Conversation
        from advanced_omi_backend.models.audio_file import AudioFile
        from advanced_omi_backend.models.user import User

        # Get MongoDB URI from environment
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

        # Create MongoDB client
        client = AsyncIOMotorClient(mongodb_uri)
        database = client.get_default_database("friend-lite")

        # Initialize Beanie
        await init_beanie(
            database=database,
            document_models=[User, Conversation, AudioFile],
        )

        _beanie_initialized = True
        logger.info("✅ Beanie initialized in RQ worker process")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Beanie in RQ worker: {e}")
        raise


def get_job_stats() -> Dict[str, Any]:
    """Get statistics about jobs in all queues matching frontend expectations."""
    from datetime import datetime

    total_jobs = 0
    queued_jobs = 0
    processing_jobs = 0
    completed_jobs = 0
    failed_jobs = 0
    cancelled_jobs = 0
    deferred_jobs = 0  # Jobs waiting for dependencies (depends_on)

    for queue_name in [TRANSCRIPTION_QUEUE, MEMORY_QUEUE, DEFAULT_QUEUE]:
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

    queues_to_check = [queue_name] if queue_name else [TRANSCRIPTION_QUEUE, MEMORY_QUEUE, DEFAULT_QUEUE]

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

    A session is considered complete only when all its jobs are in terminal states
    (completed, failed, or cancelled). Jobs that are queued or processing keep the
    session in active state.

    This function now traverses dependency chains to find dependent jobs that may
    not be in any registry yet (they're stored via job.dependent_ids).

    Args:
        session_id: The audio_uuid (session ID) to check jobs for

    Returns:
        True if all jobs are complete (or no jobs found), False if any job is still processing
    """
    from rq.registry import ScheduledJobRegistry, DeferredJobRegistry
    from advanced_omi_backend.models.conversation import Conversation
    import asyncio

    # First, get conversation_id(s) for this session (for memory jobs)
    conversation_ids = set()
    try:
        # Run async query in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        conversations = loop.run_until_complete(
            Conversation.find(Conversation.audio_uuid == session_id).to_list()
        )
        conversation_ids = {conv.conversation_id for conv in conversations}
        loop.close()
    except Exception as e:
        logger.debug(f"Error fetching conversations for session {session_id}: {e}")

    processed_job_ids = set()  # Track which jobs we've already checked
    session_jobs_found = []  # Track all jobs found for this session

    def check_job_and_dependents(job):
        """
        Recursively check a job and all its dependents.
        Returns True if all are terminal, False if any are non-terminal.
        """
        if job.id in processed_job_ids:
            return True

        processed_job_ids.add(job.id)

        # Check if this job is in a terminal state
        is_terminal = job.is_finished or job.is_failed or job.is_canceled

        if not is_terminal:
            # Job is still queued, processing, or scheduled - session not complete
            logger.debug(f"Job {job.id} ({job.func_name}) is not terminal (queued/processing/scheduled)")
            return False

        # Check dependent jobs (jobs that depend on this one)
        try:
            dependent_ids = job.dependent_ids
            if dependent_ids:
                logger.debug(f"Job {job.id} has {len(dependent_ids)} dependents")
                for dep_id in dependent_ids:
                    try:
                        dep_job = Job.fetch(dep_id, connection=redis_conn)
                        # Recursively check dependent job
                        if not check_job_and_dependents(dep_job):
                            return False
                    except Exception as e:
                        logger.debug(f"Error fetching dependent job {dep_id}: {e}")
        except Exception as e:
            logger.debug(f"Error checking dependents for job {job.id}: {e}")

        return True

    # Check all queues and registries
    for queue in [transcription_queue, memory_queue, default_queue]:
        # Check all job registries for this queue (including scheduled/deferred)
        registries = [
            queue.job_ids,  # Queued jobs
            queue.started_job_registry.get_job_ids(),  # Processing jobs
            queue.finished_job_registry.get_job_ids(),  # Completed
            queue.failed_job_registry.get_job_ids(),  # Failed
            queue.canceled_job_registry.get_job_ids(),  # Cancelled
            ScheduledJobRegistry(queue=queue).get_job_ids(),  # Scheduled (dependent jobs)
            DeferredJobRegistry(queue=queue).get_job_ids(),  # Deferred (retrying)
        ]

        for job_ids in registries:
            for job_id in job_ids:
                try:
                    job = Job.fetch(job_id, connection=redis_conn)
                    matches_session = False

                    # Check job.meta first (preferred method for all new jobs)
                    if job.meta and 'audio_uuid' in job.meta:
                        if job.meta['audio_uuid'] == session_id:
                            matches_session = True
                    # FALLBACK: Check args for backward compatibility
                    elif job.args and len(job.args) > 0:
                        # Check args[0] first (most common for streaming jobs)
                        if job.args[0] == session_id:
                            matches_session = True
                        # Check args[1] for transcription jobs
                        elif len(job.args) > 1 and job.args[1] == session_id:
                            matches_session = True
                        # Check args[3] for memory jobs (conversation_id)
                        elif len(job.args) > 3 and job.args[3] in conversation_ids:
                            matches_session = True

                    if matches_session:
                        session_jobs_found.append(job.id)
                        # Check this job and all its dependents
                        if not check_job_and_dependents(job):
                            logger.debug(f"Session {session_id} has incomplete jobs (found {len(session_jobs_found)} jobs)")
                            return False

                except Exception as e:
                    logger.debug(f"Error checking job {job_id}: {e}")
                    continue

    # All jobs are in terminal states (or no jobs found)
    logger.debug(f"Session {session_id} all jobs complete ({len(session_jobs_found)} jobs checked)")
    return True


def start_streaming_jobs(
    session_id: str,
    user_id: str,
    user_email: str,
    client_id: str
) -> Dict[str, str]:
    """
    Enqueue jobs for streaming audio session.

    This starts the parallel job processing for a streaming session:
    1. Speech detection job - monitors transcription results for speech
    2. Audio persistence job - writes audio chunks to WAV file

    Args:
        session_id: Stream session ID (audio_uuid)
        user_id: User identifier
        user_email: User email
        client_id: Client identifier

    Returns:
        Dict with job IDs: {'speech_detection': job_id, 'audio_persistence': job_id}
    """
    from advanced_omi_backend.workers.transcription_jobs import stream_speech_detection_job
    from advanced_omi_backend.workers.audio_jobs import audio_streaming_persistence_job

    # Enqueue speech detection job
    speech_job = transcription_queue.enqueue(
        stream_speech_detection_job,
        session_id,
        user_id,
        user_email,
        client_id,
        job_timeout=3600,  # 1 hour for long recordings
        result_ttl=JOB_RESULT_TTL,
        job_id=f"speech-detect_{session_id[:12]}",
        description=f"Stream speech detection for {session_id[:12]}",
        meta={'audio_uuid': session_id}
    )
    logger.info(f"📥 RQ: Enqueued speech detection job {speech_job.id}")

    # Enqueue audio persistence job in parallel
    audio_job = transcription_queue.enqueue(
        audio_streaming_persistence_job,
        session_id,
        user_id,
        user_email,
        client_id,
        job_timeout=3600,  # 1 hour for long recordings
        result_ttl=JOB_RESULT_TTL,
        job_id=f"audio-persist_{session_id[:12]}",
        description=f"Audio persistence for {session_id[:12]}",
        meta={'audio_uuid': session_id}
    )
    logger.info(f"📥 RQ: Enqueued audio persistence job {audio_job.id}")

    return {
        'speech_detection': speech_job.id,
        'audio_persistence': audio_job.id
    }


def start_batch_processing_jobs(
    conversation_id: str,
    audio_uuid: str,
    user_id: str,
    user_email: str,
    audio_file_path: str
) -> Dict[str, str]:
    """
    Enqueue complete batch processing job chain with dependencies.

    This creates the full processing pipeline:
    1. Transcription job (transcribe audio file)
    2. Speaker recognition job (depends on transcription)
    3. Memory extraction job (depends on speaker recognition)

    Args:
        conversation_id: Conversation identifier
        audio_uuid: Audio file UUID
        user_id: User identifier
        user_email: User email
        audio_file_path: Path to audio file

    Returns:
        Dict with job IDs: {
            'transcription': job_id,
            'speaker_recognition': job_id,
            'memory': job_id
        }
    """
    import uuid
    from advanced_omi_backend.workers.transcription_jobs import transcribe_full_audio_job
    from advanced_omi_backend.workers.transcription_jobs import recognise_speakers_job
    from advanced_omi_backend.workers.memory_jobs import process_memory_job

    # Generate version IDs for transcript and speaker processing
    transcript_version_id = str(uuid.uuid4())

    # Step 1: Transcription job (no dependencies)
    # Signature: transcribe_full_audio_job(conversation_id, audio_uuid, audio_path, version_id, user_id, trigger, redis_client)
    transcription_job = transcription_queue.enqueue(
        transcribe_full_audio_job,
        conversation_id,
        audio_uuid,
        audio_file_path,
        transcript_version_id,
        user_id,
        "batch",  # trigger
        job_timeout=getattr(transcribe_full_audio_job, 'job_timeout', 1800),  # Use decorator default or 30 min
        result_ttl=getattr(transcribe_full_audio_job, 'result_ttl', JOB_RESULT_TTL),
        job_id=f"transcribe_{audio_uuid[:12]}",
        description=f"Transcribe audio {audio_uuid[:12]}",
        meta={'audio_uuid': audio_uuid, 'conversation_id': conversation_id}
    )
    logger.info(f"📥 RQ: Enqueued transcription job {transcription_job.id}")

    # Step 2: Speaker recognition job (depends on transcription)
    # Signature: recognise_speakers_job(conversation_id, version_id, audio_path, user_id, transcript_text, words, redis_client)
    speaker_job = transcription_queue.enqueue(
        recognise_speakers_job,
        conversation_id,
        transcript_version_id,
        audio_file_path,
        user_id,
        "",  # transcript_text - will be read from DB
        [],  # words - will be read from DB
        job_timeout=getattr(recognise_speakers_job, 'job_timeout', 1200),  # Use decorator default or 20 min
        result_ttl=getattr(recognise_speakers_job, 'result_ttl', JOB_RESULT_TTL),
        depends_on=transcription_job,
        job_id=f"speaker_{audio_uuid[:12]}",
        description=f"Speaker recognition for {audio_uuid[:12]}",
        meta={'audio_uuid': audio_uuid, 'conversation_id': conversation_id}
    )
    logger.info(f"📥 RQ: Enqueued speaker recognition job {speaker_job.id} (depends on {transcription_job.id})")

    # Step 3: Memory extraction job (depends on speaker recognition)
    # Signature: process_memory_job(client_id, user_id, user_email, conversation_id, redis_client)
    memory_job = memory_queue.enqueue(
        process_memory_job,
        None,  # client_id - will be read from conversation in DB
        user_id,
        user_email,
        conversation_id,
        job_timeout=getattr(process_memory_job, 'job_timeout', 900),  # Use decorator default or 15 min
        result_ttl=getattr(process_memory_job, 'result_ttl', JOB_RESULT_TTL),
        depends_on=speaker_job,
        job_id=f"memory_{audio_uuid[:12]}",
        description=f"Memory extraction for {audio_uuid[:12]}",
        meta={'audio_uuid': audio_uuid, 'conversation_id': conversation_id}
    )
    logger.info(f"📥 RQ: Enqueued memory extraction job {memory_job.id} (depends on {speaker_job.id})")

    return {
        'transcription': transcription_job.id,
        'speaker_recognition': speaker_job.id,
        'memory': memory_job.id
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
    for queue_name in [TRANSCRIPTION_QUEUE, MEMORY_QUEUE, DEFAULT_QUEUE]:
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
