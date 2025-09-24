"""
Simple queue API routes for job monitoring.
Provides basic endpoints for viewing job status and statistics.
"""

import logging
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from advanced_omi_backend.auth import current_active_user
from advanced_omi_backend.simple_queue import get_simple_queue
from advanced_omi_backend.users import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/queue", tags=["queue"])


@router.get("/jobs")
async def list_jobs(
    limit: int = Query(20, ge=1, le=100, description="Number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    status: str = Query(None, description="Filter by job status"),
    job_type: str = Query(None, description="Filter by job type"),
    priority: str = Query(None, description="Filter by job priority"),
    current_user: User = Depends(current_active_user)
):
    """List jobs with pagination and filtering."""
    try:
        # Build filters dict
        filters = {}
        if status:
            filters["status"] = status
        if job_type:
            filters["job_type"] = job_type
        if priority:
            filters["priority"] = priority

        queue = await get_simple_queue()
        result = await queue.get_jobs(limit=limit, offset=offset, filters=filters)
        
            # Filter jobs by user if not admin
        if not current_user.is_superuser:
            result["jobs"] = [
                job for job in result["jobs"]
                if job["user_id"] == str(current_user.user_id)
            ]
            result["pagination"]["total"] = len(result["jobs"])
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        return {"error": "Failed to list jobs", "jobs": [], "pagination": {"total": 0, "limit": limit, "offset": offset, "has_more": False}}


@router.get("/stats")
async def get_queue_stats(
    current_user: User = Depends(current_active_user)
):
    """Get queue statistics."""
    try:
        queue = await get_simple_queue()
        stats = await queue.get_job_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        return {"queued": 0, "processing": 0, "completed": 0, "failed": 0}


@router.get("/health")
async def get_queue_health():
    """Get queue system health status."""
    try:
        queue = await get_simple_queue()
        
        return {
            "status": "healthy" if queue.running else "stopped",
            "worker_running": queue.running,
            "message": "Simple queue is operational" if queue.running else "Simple queue worker not running"
        }
        
    except Exception as e:
        logger.error(f"Failed to get queue health: {e}")
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}"
        }


class FlushJobsRequest(BaseModel):
    older_than_hours: int = 24
    statuses: Optional[List[str]] = None


class FlushAllJobsRequest(BaseModel):
    confirm: bool = False


@router.post("/flush")
async def flush_inactive_jobs(
    request: FlushJobsRequest,
    current_user: User = Depends(current_active_user)
):
    """Flush inactive jobs from the database (admin only)."""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        queue = await get_simple_queue()
        result = await queue.flush_inactive_jobs(
            older_than_hours=request.older_than_hours,
            statuses=request.statuses
        )
        return result

    except Exception as e:
        logger.error(f"Failed to flush inactive jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to flush jobs: {str(e)}")


@router.post("/flush-all")
async def flush_all_jobs(
    request: FlushAllJobsRequest,
    current_user: User = Depends(current_active_user)
):
    """Flush ALL jobs from the database (admin only). USE WITH EXTREME CAUTION!"""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        if not request.confirm:
            raise HTTPException(
                status_code=400,
                detail="Must set confirm=true to flush all jobs. This is a destructive operation."
            )

        queue = await get_simple_queue()
        result = await queue.flush_all_jobs(confirm=request.confirm)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to flush all jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to flush all jobs: {str(e)}")