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
    retrying_jobs = 0

    for queue_name in [TRANSCRIPTION_QUEUE, MEMORY_QUEUE, DEFAULT_QUEUE]:
        queue = get_queue(queue_name)

        queued_jobs += len(queue)
        processing_jobs += len(queue.started_job_registry)
        completed_jobs += len(queue.finished_job_registry)
        failed_jobs += len(queue.failed_job_registry)
        cancelled_jobs += len(queue.canceled_job_registry)
        retrying_jobs += len(queue.deferred_job_registry)

    total_jobs = queued_jobs + processing_jobs + completed_jobs + failed_jobs + cancelled_jobs + retrying_jobs

    return {
        "total_jobs": total_jobs,
        "queued_jobs": queued_jobs,
        "processing_jobs": processing_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "cancelled_jobs": cancelled_jobs,
        "retrying_jobs": retrying_jobs,
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
            (queue.deferred_job_registry.get_job_ids(), "retrying"),
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
