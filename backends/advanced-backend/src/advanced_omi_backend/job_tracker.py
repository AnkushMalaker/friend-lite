"""
Job tracking system for async file processing operations.

Provides in-memory job tracking for file upload and processing operations.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FileStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class FileProcessingInfo:
    filename: str
    duration_seconds: Optional[float] = None
    size_bytes: Optional[int] = None
    status: FileStatus = FileStatus.PENDING
    client_id: Optional[str] = None
    audio_uuid: Optional[str] = None
    transcription_status: Optional[str] = None
    memory_status: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ProcessingJob:
    job_id: str
    user_id: str
    device_name: str
    status: JobStatus = JobStatus.QUEUED
    files: List[FileProcessingInfo] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    current_file_index: int = 0

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def processed_files(self) -> int:
        return len(
            [
                f
                for f in self.files
                if f.status in [FileStatus.COMPLETED, FileStatus.FAILED, FileStatus.SKIPPED]
            ]
        )

    @property
    def progress_percent(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100

    @property
    def current_file(self) -> Optional[FileProcessingInfo]:
        if 0 <= self.current_file_index < len(self.files):
            return self.files[self.current_file_index]
        return None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "current_file": self.current_file.filename if self.current_file else None,
            "progress_percent": round(self.progress_percent, 1),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "files": [
                {
                    "filename": f.filename,
                    "duration_seconds": f.duration_seconds,
                    "size_bytes": f.size_bytes,
                    "status": f.status.value,
                    "client_id": f.client_id,
                    "audio_uuid": f.audio_uuid,
                    "transcription_status": f.transcription_status,
                    "memory_status": f.memory_status,
                    "error_message": f.error_message,
                    "started_at": f.started_at.isoformat() if f.started_at else None,
                    "completed_at": f.completed_at.isoformat() if f.completed_at else None,
                }
                for f in self.files
            ],
        }


class JobTracker:
    """In-memory job tracking system."""

    def __init__(self):
        self.jobs: Dict[str, ProcessingJob] = {}
        self._lock = asyncio.Lock()

        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background task to clean up old jobs."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_old_jobs())

    async def _cleanup_old_jobs(self):
        """Remove jobs older than 1 hour to prevent memory leaks."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                cutoff_time = datetime.now(timezone.utc).timestamp() - 3600  # 1 hour ago

                async with self._lock:
                    jobs_to_remove = []
                    for job_id, job in self.jobs.items():
                        job_age = job.created_at.timestamp()
                        if job_age < cutoff_time and job.status in [
                            JobStatus.COMPLETED,
                            JobStatus.FAILED,
                            JobStatus.CANCELLED,
                        ]:
                            jobs_to_remove.append(job_id)

                    for job_id in jobs_to_remove:
                        del self.jobs[job_id]
                        logger.info(f"Cleaned up old job: {job_id}")

            except Exception as e:
                logger.error(f"Error in job cleanup task: {e}")

    async def create_job(self, user_id: str, device_name: str, files: List[str]) -> str:
        """Create a new processing job."""
        job_id = str(uuid.uuid4())

        file_infos = []
        for filename in files:
            file_infos.append(FileProcessingInfo(filename=filename))

        job = ProcessingJob(
            job_id=job_id, user_id=user_id, device_name=device_name, files=file_infos
        )

        async with self._lock:
            self.jobs[job_id] = job

        logger.info(f"Created job {job_id} with {len(files)} files for user {user_id}")
        return job_id

    async def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get job by ID."""
        async with self._lock:
            return self.jobs.get(job_id)

    async def update_job_status(self, job_id: str, status: JobStatus, error_message: str = None):
        """Update job status."""
        async with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = status
                if error_message:
                    job.error_message = error_message

                if status == JobStatus.PROCESSING and job.started_at is None:
                    job.started_at = datetime.now(timezone.utc)
                elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    job.completed_at = datetime.now(timezone.utc)

    async def update_file_status(
        self,
        job_id: str,
        filename: str,
        status: FileStatus,
        client_id: str = None,
        audio_uuid: str = None,
        transcription_status: str = None,
        memory_status: str = None,
        error_message: str = None,
    ):
        """Update status of a specific file in the job."""
        async with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                for file_info in job.files:
                    if file_info.filename == filename:
                        file_info.status = status
                        if client_id:
                            file_info.client_id = client_id
                        if audio_uuid:
                            file_info.audio_uuid = audio_uuid
                        if transcription_status:
                            file_info.transcription_status = transcription_status
                        if memory_status:
                            file_info.memory_status = memory_status
                        if error_message:
                            file_info.error_message = error_message

                        if status == FileStatus.PROCESSING and file_info.started_at is None:
                            file_info.started_at = datetime.now(timezone.utc)
                        elif status in [
                            FileStatus.COMPLETED,
                            FileStatus.FAILED,
                            FileStatus.SKIPPED,
                        ]:
                            file_info.completed_at = datetime.now(timezone.utc)
                        break

    async def set_current_file(self, job_id: str, filename: str):
        """Set the currently processing file."""
        async with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                for i, file_info in enumerate(job.files):
                    if file_info.filename == filename:
                        job.current_file_index = i
                        break

    async def get_active_jobs(self) -> List[ProcessingJob]:
        """Get all active (non-completed) jobs."""
        async with self._lock:
            return [
                job
                for job in self.jobs.values()
                if job.status in [JobStatus.QUEUED, JobStatus.PROCESSING]
            ]


# Global job tracker instance
_job_tracker: Optional[JobTracker] = None


def get_job_tracker() -> JobTracker:
    """Get the global job tracker instance."""
    global _job_tracker
    if _job_tracker is None:
        _job_tracker = JobTracker()
    return _job_tracker
