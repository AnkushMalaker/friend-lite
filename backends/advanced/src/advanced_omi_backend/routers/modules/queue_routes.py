"""
Simple queue API routes for job monitoring.
Provides basic endpoints for viewing job status and statistics.
"""

import logging
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from advanced_omi_backend.auth import current_active_user
from advanced_omi_backend.rq_queue import get_jobs, get_job_stats, get_queue_health, redis_conn
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
            status = "retrying"

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
        return {"total_jobs": 0, "queued_jobs": 0, "processing_jobs": 0, "completed_jobs": 0, "failed_jobs": 0, "cancelled_jobs": 0, "retrying_jobs": 0}


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
    """Get Redis Streams statistics (limited to prevent performance issues)."""
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
                # Get basic stream info only (skip detailed consumer group info for performance)
                info = await audio_service.redis.xinfo_stream(stream_key)
                stream_name = stream_key.decode()

                return {
                    "stream_name": stream_name,
                    "length": info[b"length"],
                    "first_entry_id": info[b"first-entry"][0].decode() if info[b"first-entry"] else None,
                    "last_entry_id": info[b"last-entry"][0].decode() if info[b"last-entry"] else None,
                }
            except Exception as e:
                logger.error(f"Error getting info for stream {stream_key.decode()}: {e}")
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
        from advanced_omi_backend.rq_queue import get_queue

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
        from advanced_omi_backend.rq_queue import get_queue

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