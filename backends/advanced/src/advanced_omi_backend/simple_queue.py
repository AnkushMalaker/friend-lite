"""
Simple MongoDB-based queue system.
Lightweight replacement for complex queue_manager.py that just calls existing controller methods.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, Any, Optional

from advanced_omi_backend.database import mongo_client

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SimpleQueue:
    """Simple MongoDB queue that calls controller methods directly."""
    
    def __init__(self):
        self.db = mongo_client.get_database("friend-lite")
        self.jobs_collection = self.db["simple_jobs"]
        self.running = False
        self.worker_task = None
        
    async def enqueue_job(self, job_type: str, user_id: str, data: Dict[str, Any], priority: str = "normal") -> str:
        """Add a job to the queue."""
        # Get next job ID using auto-increment
        next_id = await self._get_next_job_id()
        job_id = str(next_id)
        job = {
            "job_id": job_id,
            "job_type": job_type,
            "user_id": user_id,
            "data": data,
            "status": JobStatus.QUEUED,
            "priority": priority,
            "created_at": datetime.now(timezone.utc),
            "attempts": 0,
            "max_attempts": 3
        }
        
        await self.jobs_collection.insert_one(job)
        logger.info(f"📥 Enqueued {job_type} job {job_id}")
        return job_id
    
    async def start_worker(self):
        """Start the background worker."""
        if self.running:
            return
            
        self.running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info("🔄 Simple queue worker started")
    
    async def stop_worker(self):
        """Stop the background worker."""
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("⏹️ Simple queue worker stopped")
    
    async def _worker_loop(self):
        """Main worker loop that processes jobs."""
        while self.running:
            try:
                # Get next job
                job = await self.jobs_collection.find_one_and_update(
                    {"status": JobStatus.QUEUED},
                    {"$set": {"status": JobStatus.PROCESSING, "started_at": datetime.now(timezone.utc)}},
                    sort=[("created_at", 1)]
                )
                
                if job:
                    await self._process_job(job)
                else:
                    # No jobs, sleep for a bit
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(5)
    
    async def _process_job(self, job: Dict[str, Any]):
        """Process a single job by calling the appropriate controller method."""
        job_id = job["job_id"]
        job_type = job["job_type"]
        
        try:
            logger.info(f"🔄 Processing {job_type} job {job_id}")
            
            result = None
            if job_type == "reprocess_transcript":
                result = await self._handle_reprocess_transcript(job)
            elif job_type == "reprocess_memory":
                result = await self._handle_reprocess_memory(job)
            else:
                raise Exception(f"Unknown job type: {job_type}")
            
            # Mark as completed with result
            await self.jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": {
                    "status": JobStatus.COMPLETED,
                    "completed_at": datetime.now(timezone.utc),
                    "result": result if result else {}
                }}
            )
            logger.info(f"✅ Completed {job_type} job {job_id}")
            
        except Exception as e:
            logger.error(f"❌ Job {job_id} failed: {e}")
            
            # Increment attempts
            attempts = job.get("attempts", 0) + 1
            max_attempts = job.get("max_attempts", 3)
            
            if attempts >= max_attempts:
                # Mark as failed
                await self.jobs_collection.update_one(
                    {"job_id": job_id},
                    {"$set": {
                        "status": JobStatus.FAILED,
                        "error": str(e),
                        "attempts": attempts,
                        "failed_at": datetime.now(timezone.utc)
                    }}
                )
            else:
                # Retry
                await self.jobs_collection.update_one(
                    {"job_id": job_id},
                    {"$set": {
                        "status": JobStatus.QUEUED,
                        "attempts": attempts,
                        "last_error": str(e)
                    }}
                )
    
    async def _handle_reprocess_transcript(self, job: Dict[str, Any]):
        """Handle transcript reprocessing by calling the controller method."""
        from advanced_omi_backend.controllers.conversation_controller import _do_transcript_reprocessing
        
        data = job["data"]
        result = await _do_transcript_reprocessing(
            conversation_id=data["conversation_id"],
            audio_uuid=data["audio_uuid"],
            audio_path=data["audio_path"],
            version_id=data["version_id"],
            user_id=job["user_id"]
        )
        
        # Format result for queue job storage
        if result and result.get("success"):
            # Count identified speakers from segments
            segments = result.get("segments", [])
            identified_speakers = set()
            for segment in segments:
                speaker = segment.get("speaker")
                if speaker and not speaker.startswith("unknown_speaker") and speaker != "Unknown":
                    identified_speakers.add(speaker)

            return {
                "conversation_id": data["conversation_id"],
                "version_id": data["version_id"],
                "transcript_segments": len(segments),
                "speakers_identified": len(identified_speakers),
                "identified_speaker_names": list(identified_speakers),
                "processing_time": result.get("processing_time_seconds", 0),
                "provider": result.get("provider", "unknown"),
                "transcript_length": len(result.get("transcript", "")),
                "success": True
            }
        else:
            return {"success": False, "error": "No result returned"}
    
    async def _handle_reprocess_memory(self, job: Dict[str, Any]):
        """Handle memory reprocessing by calling the controller method."""
        # TODO: Implement when needed
        pass
    
    async def get_job_stats(self) -> Dict[str, int]:
        """Get simple job statistics."""
        pipeline = [
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]

        status_counts = {"queued": 0, "processing": 0, "completed": 0, "failed": 0}
        async for stat in self.jobs_collection.aggregate(pipeline):
            status_counts[stat["_id"]] = stat["count"]

        # Format to match frontend expectations
        total = sum(status_counts.values())
        return {
            "total_jobs": total,
            "queued_jobs": status_counts["queued"],
            "processing_jobs": status_counts["processing"],
            "completed_jobs": status_counts["completed"],
            "failed_jobs": status_counts["failed"],
            "cancelled_jobs": 0,  # Not implemented yet
            "retrying_jobs": 0    # Not implemented yet
        }
    
    async def get_jobs(self, limit: int = 20, offset: int = 0, filters: Dict[str, str] = None) -> Dict[str, Any]:
        """Get jobs with pagination and filtering."""
        # Build filter query
        query = {}
        if filters:
            if filters.get("status"):
                query["status"] = filters["status"]
            if filters.get("job_type"):
                query["job_type"] = filters["job_type"]
            if filters.get("priority"):
                query["priority"] = filters["priority"]

        total = await self.jobs_collection.count_documents(query)

        cursor = self.jobs_collection.find(query).sort("created_at", -1).skip(offset).limit(limit)
        jobs = []
        async for job in cursor:
            # Convert ObjectId to string and format dates
            job["_id"] = str(job["_id"])
            job["created_at"] = job["created_at"].isoformat()
            if "started_at" in job:
                job["started_at"] = job["started_at"].isoformat()
            if "completed_at" in job:
                job["completed_at"] = job["completed_at"].isoformat()
            if "failed_at" in job:
                job["failed_at"] = job["failed_at"].isoformat()
            jobs.append(job)
        
        return {
            "jobs": jobs,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total
            }
        }

    async def _get_next_job_id(self) -> int:
        """Get the next job ID using auto-increment."""
        # Use MongoDB's findOneAndUpdate to atomically increment counter
        counter_doc = await self.db["job_counters"].find_one_and_update(
            {"_id": "job_id"},
            {"$inc": {"sequence_value": 1}},
            upsert=True,
            return_document=True
        )
        return counter_doc["sequence_value"]

    async def flush_inactive_jobs(self, older_than_hours: int = 24, statuses: list = None) -> Dict[str, int]:
        """Flush inactive jobs from the database.

        Args:
            older_than_hours: Remove jobs older than this many hours (default: 24)
            statuses: List of statuses to remove. If None, removes completed and failed jobs
                     Options: ['completed', 'failed', 'cancelled']

        Returns:
            Dictionary with count of removed jobs by status
        """
        if statuses is None:
            statuses = [JobStatus.COMPLETED, JobStatus.FAILED]

        # Calculate cutoff time
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)

        # Build query to find jobs to remove
        query = {
            "status": {"$in": statuses},
            "created_at": {"$lt": cutoff_time}
        }

        # Get count by status before deletion
        pipeline = [
            {"$match": query},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]

        removal_stats = {status: 0 for status in statuses}
        async for stat in self.jobs_collection.aggregate(pipeline):
            removal_stats[stat["_id"]] = stat["count"]

        # Remove the jobs
        result = await self.jobs_collection.delete_many(query)

        total_removed = result.deleted_count

        logger.info(
            f"🧹 Flushed {total_removed} inactive jobs older than {older_than_hours} hours. "
            f"Stats: {removal_stats}"
        )

        return {
            "total_removed": total_removed,
            "older_than_hours": older_than_hours,
            "removed_by_status": removal_stats,
            "cutoff_time": cutoff_time.isoformat()
        }

    async def flush_all_jobs(self, confirm: bool = False) -> Dict[str, int]:
        """Flush ALL jobs from the database. USE WITH CAUTION!

        Args:
            confirm: Must be True to actually delete jobs (safety check)

        Returns:
            Dictionary with count of removed jobs by status
        """
        if not confirm:
            raise ValueError("Must set confirm=True to flush all jobs. This is a destructive operation.")

        # Get stats before deletion
        stats = await self.get_job_stats()

        # Remove all jobs
        result = await self.jobs_collection.delete_many({})

        # Reset job counter
        await self.db["job_counters"].delete_one({"_id": "job_id"})

        total_removed = result.deleted_count

        logger.warning(
            f"🚨 FLUSHED ALL {total_removed} jobs from database and reset job counter! "
            f"Previous stats: {stats}"
        )

        return {
            "total_removed": total_removed,
            "previous_stats": stats,
            "job_counter_reset": True
        }

# Global instance
_simple_queue = None

async def get_simple_queue() -> SimpleQueue:
    """Get the global simple queue instance."""
    global _simple_queue
    if _simple_queue is None:
        _simple_queue = SimpleQueue()
        await _simple_queue.start_worker()
    return _simple_queue

# Convenience functions for direct access
async def flush_inactive_jobs(older_than_hours: int = 24, statuses: list = None) -> Dict[str, int]:
    """Convenience function to flush inactive jobs."""
    queue = await get_simple_queue()
    return await queue.flush_inactive_jobs(older_than_hours=older_than_hours, statuses=statuses)

async def flush_all_jobs(confirm: bool = False) -> Dict[str, int]:
    """Convenience function to flush all jobs. USE WITH CAUTION!"""
    queue = await get_simple_queue()
    return await queue.flush_all_jobs(confirm=confirm)