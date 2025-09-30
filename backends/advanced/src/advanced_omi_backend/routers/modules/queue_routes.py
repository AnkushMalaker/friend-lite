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


# Note: RQ handles job cleanup automatically through job registries
# Completed jobs are automatically moved to finished_job_registry
# and can be configured to auto-expire. Manual flush endpoints
# can be added later if needed.