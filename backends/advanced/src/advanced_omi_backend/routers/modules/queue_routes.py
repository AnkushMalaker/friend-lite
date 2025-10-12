"""
Simple queue API routes for job monitoring.
Provides basic endpoints for viewing job status and statistics.
"""

import logging
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from advanced_omi_backend.auth import current_active_user
from advanced_omi_backend.controllers.queue_controller import get_jobs, get_job_stats, get_queue_health, redis_conn
from advanced_omi_backend.users import User
from rq.job import Job
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/queue", tags=["queue"])


@router.get("/jobs")
async def list_jobs(
    limit: int = Query(20, ge=1, le=100, description="Number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    queue_name: str = Query(None, description="Filter by queue name"),
    current_user: User = Depends(current_active_user)
):
    """List jobs with pagination and filtering."""
    try:
        result = get_jobs(limit=limit, offset=offset, queue_name=queue_name)

        # Filter jobs by user if not admin
        if not current_user.is_superuser:
            # Filter based on user_id in job kwargs (where RQ stores job parameters)
            user_jobs = []
            for job in result["jobs"]:
                job_kwargs = job.get("kwargs", {})
                if job_kwargs.get("user_id") == str(current_user.user_id):
                    user_jobs.append(job)

            result["jobs"] = user_jobs
            result["pagination"]["total"] = len(user_jobs)

        return result

    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        return {"error": "Failed to list jobs", "jobs": [], "pagination": {"total": 0, "limit": limit, "offset": offset, "has_more": False}}


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    current_user: User = Depends(current_active_user)
):
    """Get detailed job information including result."""
    try:
        job = Job.fetch(job_id, connection=redis_conn)

        # Check user permission (non-admins can only see their own jobs)
        if not current_user.is_superuser:
            job_user_id = job.kwargs.get("user_id") if job.kwargs else None
            if job_user_id != str(current_user.user_id):
                raise HTTPException(status_code=403, detail="Access forbidden")

        # Determine status from registries
        status = "unknown"
        if job.is_queued:
            status = "queued"
        elif job.is_started:
            status = "processing"
        elif job.is_finished:
            status = "completed"
        elif job.is_failed:
            status = "failed"
        elif job.is_deferred:
            status = "deferred"

        return {
            "job_id": job.id,
            "status": status,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
            "description": job.description or "",
            "func_name": job.func_name if hasattr(job, 'func_name') else "",
            "args": job.args,
            "kwargs": job.kwargs,
            "result": job.result,
            "error_message": str(job.exc_info) if job.exc_info else None,
        }

    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {e}")
        raise HTTPException(status_code=404, detail="Job not found")


@router.get("/jobs/by-session/{session_id}")
async def get_jobs_by_session(
    session_id: str,
    current_user: User = Depends(current_active_user)
):
    """Get all jobs associated with a specific streaming session."""
    try:
        from rq.registry import FinishedJobRegistry, FailedJobRegistry, StartedJobRegistry, CanceledJobRegistry, DeferredJobRegistry, ScheduledJobRegistry
        from advanced_omi_backend.controllers.queue_controller import get_queue
        from advanced_omi_backend.models.conversation import Conversation

        # First, get conversation_id(s) for this session (for memory jobs)
        conversation_ids = set()
        conversations = await Conversation.find(Conversation.audio_uuid == session_id).to_list()
        conversation_ids = {conv.conversation_id for conv in conversations}

        all_jobs = []
        processed_job_ids = set()  # Track which jobs we've already processed
        queues = ["default", "transcription", "memory"]

        def get_job_status(job, registries_map):
            """Determine job status from registries."""
            if job.is_queued:
                return "queued"
            elif job.is_started:
                return "processing"
            elif job.is_finished:
                return "completed"
            elif job.is_failed:
                return "failed"
            elif job.is_deferred:
                return "deferred"
            elif job.is_scheduled:
                return "waiting"
            else:
                return "unknown"

        def process_job_and_dependents(job, queue_name, base_status):
            """Process a job and recursively find all its dependents."""
            if job.id in processed_job_ids:
                return

            processed_job_ids.add(job.id)

            # Check user permission (non-admins can only see their own jobs)
            if not current_user.is_superuser:
                job_user_id = job.kwargs.get("user_id") if job.kwargs else None
                if job_user_id != str(current_user.user_id):
                    return

            # Get accurate status
            status = get_job_status(job, {})

            # Add this job to results
            all_jobs.append({
                "job_id": job.id,
                "job_type": job.func_name.split('.')[-1] if job.func_name else "unknown",
                "queue": queue_name,
                "status": status,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "ended_at": job.ended_at.isoformat() if job.ended_at else None,
                "description": job.description or "",
                "result": job.result,
                "error_message": str(job.exc_info) if job.exc_info else None,
            })

            # Check for dependent jobs (jobs that depend on this one)
            try:
                dependent_ids = job.dependent_ids
                if dependent_ids:
                    logger.debug(f"Job {job.id} has {len(dependent_ids)} dependents: {dependent_ids}")

                    for dep_id in dependent_ids:
                        try:
                            dep_job = Job.fetch(dep_id, connection=redis_conn)
                            # Recursively process dependent job
                            process_job_and_dependents(dep_job, queue_name, "waiting")
                        except Exception as e:
                            logger.debug(f"Error fetching dependent job {dep_id}: {e}")
            except Exception as e:
                logger.debug(f"Error checking dependents for job {job.id}: {e}")

        # Find all jobs that match the session
        for queue_name in queues:
            queue = get_queue(queue_name)

            # Check all registries
            registries = [
                ("queued", queue.job_ids),
                ("processing", StartedJobRegistry(queue=queue).get_job_ids()),
                ("completed", FinishedJobRegistry(queue=queue).get_job_ids()),
                ("failed", FailedJobRegistry(queue=queue).get_job_ids()),
                ("cancelled", CanceledJobRegistry(queue=queue).get_job_ids()),
                ("waiting", DeferredJobRegistry(queue=queue).get_job_ids()),
                ("waiting", ScheduledJobRegistry(queue=queue).get_job_ids())
            ]

            for status_name, job_ids in registries:
                for job_id in job_ids:
                    try:
                        job = Job.fetch(job_id, connection=redis_conn)

                        # Check if this job belongs to the requested session
                        matches_session = False

                        # NEW: Check job.meta first (preferred method for all new jobs)
                        if job.meta and 'audio_uuid' in job.meta:
                            if job.meta['audio_uuid'] == session_id:
                                matches_session = True
                        # FALLBACK: Check args for backward compatibility with existing queued jobs
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
                            # Process this job and all its dependents
                            process_job_and_dependents(job, queue_name, status_name)

                    except Exception as e:
                        logger.debug(f"Error fetching job {job_id}: {e}")
                        continue

        # Sort by created_at
        all_jobs.sort(key=lambda x: x["created_at"] or "", reverse=False)

        logger.info(f"Found {len(all_jobs)} jobs for session {session_id} (including dependents)")

        return {
            "session_id": session_id,
            "jobs": all_jobs,
            "total": len(all_jobs)
        }

    except Exception as e:
        logger.error(f"Failed to get jobs for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get jobs for session: {str(e)}")


@router.get("/stats")
async def get_queue_stats_endpoint(
    current_user: User = Depends(current_active_user)
):
    """Get queue statistics."""
    try:
        stats = get_job_stats()
        return stats

    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        return {"total_jobs": 0, "queued_jobs": 0, "processing_jobs": 0, "completed_jobs": 0, "failed_jobs": 0, "cancelled_jobs": 0, "deferred_jobs": 0}


@router.get("/health")
async def get_queue_health_endpoint():
    """Get queue system health status."""
    try:
        health = get_queue_health()
        return health

    except Exception as e:
        logger.error(f"Failed to get queue health: {e}")
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}"
        }


@router.get("/streams")
async def get_stream_stats(
    limit: int = Query(default=10, ge=1, le=100),  # Max 100 streams to prevent timeouts
    current_user: User = Depends(current_active_user)
):
    """Get Redis Streams statistics with consumer group information."""
    try:
        from advanced_omi_backend.services.audio_service import get_audio_stream_service
        audio_service = get_audio_stream_service()

        if not audio_service.redis:
            return {
                "error": "Audio stream service not connected",
                "streams": []
            }

        # Get audio streams with limit
        stream_keys = []
        cursor = b"0"
        while cursor and len(stream_keys) < limit:
            cursor, keys = await audio_service.redis.scan(
                cursor, match=f"{audio_service.audio_stream_prefix}*", count=limit
            )
            stream_keys.extend(keys[:limit - len(stream_keys)])

        # Use asyncio.gather to fetch stream info in parallel
        import asyncio

        async def get_stream_info(stream_key):
            try:
                stream_name = stream_key.decode() if isinstance(stream_key, bytes) else stream_key

                # Get basic stream info
                info = await audio_service.redis.xinfo_stream(stream_name)

                # Get consumer groups info
                groups_info = []
                try:
                    groups = await audio_service.redis.xinfo_groups(stream_name)
                    for group in groups:
                        group_dict = {}
                        # Parse group info (alternating key-value pairs)
                        for i in range(0, len(group), 2):
                            if i+1 < len(group):
                                key = group[i].decode() if isinstance(group[i], bytes) else str(group[i])
                                value = group[i+1]
                                if isinstance(value, bytes):
                                    try:
                                        value = value.decode()
                                    except:
                                        value = str(value)
                                group_dict[key] = value

                        # Get consumers for this group
                        consumers = []
                        try:
                            consumers_raw = await audio_service.redis.xinfo_consumers(stream_name, group_dict.get('name', ''))
                            for consumer in consumers_raw:
                                consumer_dict = {}
                                for i in range(0, len(consumer), 2):
                                    if i+1 < len(consumer):
                                        key = consumer[i].decode() if isinstance(consumer[i], bytes) else str(consumer[i])
                                        value = consumer[i+1]
                                        if isinstance(value, bytes):
                                            try:
                                                value = value.decode()
                                            except:
                                                value = str(value)
                                        consumer_dict[key] = value
                                consumers.append(consumer_dict)
                        except Exception as ce:
                            logger.debug(f"Could not fetch consumers for group {group_dict.get('name')}: {ce}")

                        groups_info.append({
                            "name": group_dict.get('name', 'unknown'),
                            "consumers": group_dict.get('consumers', 0),
                            "pending": group_dict.get('pending', 0),
                            "last_delivered_id": group_dict.get('last-delivered-id', 'N/A'),
                            "consumer_details": consumers
                        })
                except Exception as ge:
                    logger.debug(f"No consumer groups for stream {stream_name}: {ge}")

                return {
                    "stream_name": stream_name,
                    "length": info[b"length"],
                    "first_entry_id": info[b"first-entry"][0].decode() if info[b"first-entry"] else None,
                    "last_entry_id": info[b"last-entry"][0].decode() if info[b"last-entry"] else None,
                    "groups": groups_info
                }
            except Exception as e:
                logger.error(f"Error getting info for stream {stream_key}: {e}")
                return None

        # Fetch all stream info in parallel
        streams_info_results = await asyncio.gather(*[get_stream_info(key) for key in stream_keys])
        streams_info = [info for info in streams_info_results if info is not None]

        return {
            "total_streams": len(streams_info),
            "streams": streams_info,
            "limited": len(stream_keys) >= limit
        }

    except Exception as e:
        logger.error(f"Failed to get stream stats: {e}", exc_info=True)
        return {
            "error": str(e),
            "total_streams": 0,
            "streams": []
        }


class FlushJobsRequest(BaseModel):
    older_than_hours: int = 24
    statuses: List[str] = ["completed", "failed", "cancelled"]


class FlushAllJobsRequest(BaseModel):
    confirm: bool


@router.post("/flush")
async def flush_jobs(
    request: FlushJobsRequest,
    current_user: User = Depends(current_active_user)
):
    """Flush old inactive jobs based on age and status."""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        from datetime import datetime, timedelta
        from rq.registry import FinishedJobRegistry, FailedJobRegistry, CanceledJobRegistry
        from advanced_omi_backend.controllers.queue_controller import get_queue

        cutoff_time = datetime.utcnow() - timedelta(hours=request.older_than_hours)
        total_removed = 0

        # Get all queues
        queues = ["default", "transcription", "memory"]

        for queue_name in queues:
            queue = get_queue(queue_name)

            # Flush from appropriate registries based on requested statuses
            if "completed" in request.statuses:
                registry = FinishedJobRegistry(queue=queue)
                for job_id in registry.get_job_ids():
                    try:
                        job = Job.fetch(job_id, connection=redis_conn)
                        if job.ended_at and job.ended_at < cutoff_time:
                            job.delete()
                            total_removed += 1
                    except Exception as e:
                        logger.error(f"Error deleting job {job_id}: {e}")

            if "failed" in request.statuses:
                registry = FailedJobRegistry(queue=queue)
                for job_id in registry.get_job_ids():
                    try:
                        job = Job.fetch(job_id, connection=redis_conn)
                        if job.ended_at and job.ended_at < cutoff_time:
                            job.delete()
                            total_removed += 1
                    except Exception as e:
                        logger.error(f"Error deleting job {job_id}: {e}")

            if "cancelled" in request.statuses:
                registry = CanceledJobRegistry(queue=queue)
                for job_id in registry.get_job_ids():
                    try:
                        job = Job.fetch(job_id, connection=redis_conn)
                        if job.ended_at and job.ended_at < cutoff_time:
                            job.delete()
                            total_removed += 1
                    except Exception as e:
                        logger.error(f"Error deleting job {job_id}: {e}")

        return {
            "total_removed": total_removed,
            "cutoff_time": cutoff_time.isoformat(),
            "statuses": request.statuses
        }

    except Exception as e:
        logger.error(f"Failed to flush jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to flush jobs: {str(e)}")


@router.post("/flush-all")
async def flush_all_jobs(
    request: FlushAllJobsRequest,
    current_user: User = Depends(current_active_user)
):
    """Flush ALL jobs (DANGER - requires confirmation)."""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Admin access required")

    if not request.confirm:
        raise HTTPException(status_code=400, detail="Confirmation required")

    try:
        from rq.registry import (
            FinishedJobRegistry,
            FailedJobRegistry,
            CanceledJobRegistry,
            StartedJobRegistry,
            DeferredJobRegistry,
            ScheduledJobRegistry
        )
        from advanced_omi_backend.controllers.queue_controller import get_queue

        total_removed = 0
        queues = ["default", "transcription", "memory"]

        for queue_name in queues:
            queue = get_queue(queue_name)

            # Remove from all registries
            registries = [
                FinishedJobRegistry(queue=queue),
                FailedJobRegistry(queue=queue),
                CanceledJobRegistry(queue=queue),
                StartedJobRegistry(queue=queue),
                DeferredJobRegistry(queue=queue),
                ScheduledJobRegistry(queue=queue)
            ]

            for registry in registries:
                for job_id in registry.get_job_ids():
                    try:
                        job = Job.fetch(job_id, connection=redis_conn)
                        job.delete()
                        total_removed += 1
                    except Exception as e:
                        logger.error(f"Error deleting job {job_id}: {e}")

            # Also empty the queue itself
            queue.empty()

        return {
            "total_removed": total_removed,
            "message": "All jobs have been flushed"
        }

    except Exception as e:
        logger.error(f"Failed to flush all jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to flush all jobs: {str(e)}")


@router.get("/sessions")
async def get_redis_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(current_active_user)
):
    """Get Redis session tracking information."""
    try:
        import redis.asyncio as aioredis
        from advanced_omi_backend.controllers.queue_controller import REDIS_URL

        redis_client = aioredis.from_url(REDIS_URL)

        # Get session keys
        session_keys = []
        cursor = b"0"
        while cursor and len(session_keys) < limit:
            cursor, keys = await redis_client.scan(
                cursor, match="audio:session:*", count=limit
            )
            session_keys.extend(keys[:limit - len(session_keys)])

        # Get session info
        sessions = []
        for key in session_keys:
            try:
                session_data = await redis_client.hgetall(key)
                if session_data:
                    session_id = key.decode().replace("audio:session:", "")
                    sessions.append({
                        "session_id": session_id,
                        "user_id": session_data.get(b"user_id", b"").decode(),
                        "client_id": session_data.get(b"client_id", b"").decode(),
                        "stream_name": session_data.get(b"stream_name", b"").decode(),
                        "provider": session_data.get(b"provider", b"").decode(),
                        "mode": session_data.get(b"mode", b"").decode(),
                        "status": session_data.get(b"status", b"").decode(),
                        "started_at": session_data.get(b"started_at", b"").decode(),
                        "chunks_published": int(session_data.get(b"chunks_published", b"0").decode() or 0),
                        "last_chunk_at": session_data.get(b"last_chunk_at", b"").decode()
                    })
            except Exception as e:
                logger.error(f"Error getting session info for {key}: {e}")

        await redis_client.close()

        return {
            "total_sessions": len(sessions),
            "sessions": sessions
        }

    except Exception as e:
        logger.error(f"Failed to get sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


@router.post("/sessions/clear")
async def clear_old_sessions(
    older_than_seconds: int = Query(default=3600, description="Clear sessions older than N seconds"),
    current_user: User = Depends(current_active_user)
):
    """Clear old Redis sessions that are stuck or inactive."""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        import redis.asyncio as aioredis
        import time
        from advanced_omi_backend.controllers.queue_controller import REDIS_URL

        redis_client = aioredis.from_url(REDIS_URL)
        current_time = time.time()
        cutoff_time = current_time - older_than_seconds

        # Get all session keys
        session_keys = []
        cursor = b"0"
        while cursor:
            cursor, keys = await redis_client.scan(cursor, match="audio:session:*", count=100)
            session_keys.extend(keys)

        # Check each session and delete if old
        deleted_count = 0
        for key in session_keys:
            try:
                session_data = await redis_client.hgetall(key)
                if session_data:
                    last_chunk_at = session_data.get(b"last_chunk_at", b"").decode()
                    if last_chunk_at:
                        last_chunk_time = float(last_chunk_at)
                        if last_chunk_time < cutoff_time:
                            await redis_client.delete(key)
                            deleted_count += 1
                            logger.info(f"Deleted old session: {key.decode()}")
            except Exception as e:
                logger.error(f"Error processing session {key}: {e}")

        await redis_client.close()

        return {
            "deleted_count": deleted_count,
            "cutoff_seconds": older_than_seconds
        }

    except Exception as e:
        logger.error(f"Failed to clear sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear sessions: {str(e)}")