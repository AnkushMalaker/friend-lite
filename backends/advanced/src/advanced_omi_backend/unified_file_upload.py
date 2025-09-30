"""Unified file upload processing using the new pipeline architecture.

This module demonstrates how file upload handlers integrate with the unified
pipeline using AudioProcessingItem and job tracking.
"""

import asyncio
import io
import logging
import wave
from pathlib import Path
from typing import Dict, List, Tuple

from advanced_omi_backend.audio_processing_types import AudioProcessingItem
from advanced_omi_backend.job_tracker import FileStatus, JobStatus, get_job_tracker
from advanced_omi_backend.processors import get_processor_manager
from advanced_omi_backend.users import User
from fastapi import BackgroundTasks, HTTPException, UploadFile

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")


def get_audio_duration(content: bytes) -> float:
    """Get duration of audio file in seconds."""
    try:
        with wave.open(io.BytesIO(content), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)
    except Exception as e:
        raise ValueError(f"Could not determine audio duration: {e}")


async def save_uploaded_file(content: bytes, filename: str, user_id: str) -> str:
    """Save uploaded file content to persistent storage.

    Args:
        content: Raw file content
        filename: Original filename
        user_id: User ID for directory organization

    Returns:
        Path to saved file
    """
    # Use volume-mounted audio_chunks directory
    upload_dir = Path("/app/audio_chunks")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename with timestamp
    timestamp = int(asyncio.get_event_loop().time())
    safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    unique_filename = f"{timestamp}_{safe_filename}"

    file_path = upload_dir / unique_filename

    # Save file content
    with open(file_path, "wb") as f:
        f.write(content)

    audio_logger.info(f"💾 Saved uploaded file: {file_path} ({len(content)} bytes)")
    return str(file_path)


async def process_audio_files_unified(
    background_tasks: BackgroundTasks,
    user: User,
    files: List[UploadFile],
    device_name: str = "file-upload"
) -> Dict:
    """Process uploaded audio files using unified pipeline.

    This is the new unified entry point that:
    1. Creates a batch job for tracking file uploads (existing functionality)
    2. Creates individual pipeline jobs for each file processing
    3. Submits files to the unified pipeline

    Args:
        background_tasks: FastAPI background tasks
        user: Current user
        files: List of uploaded files
        device_name: Device identifier for this upload session

    Returns:
        Response with batch_job_id and pipeline_job_ids
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Read all file contents immediately to avoid file handle issues
    file_data = []
    for file in files:
        try:
            content = await file.read()
            file_data.append((file.filename, content))
            audio_logger.info(f"📥 Read file: {file.filename} ({len(content)} bytes)")
        except Exception as e:
            audio_logger.error(f"❌ Failed to read file {file.filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read file {file.filename}: {str(e)}"
            )

    # Create batch job for tracking file uploads (maintains existing functionality)
    job_tracker = get_job_tracker()
    filenames = [filename for filename, _ in file_data]
    batch_job_id = await job_tracker.create_job(user.user_id, device_name, filenames)

    # Start background processing with file contents
    background_tasks.add_task(
        process_files_unified_background,
        batch_job_id,
        file_data,
        user,
        device_name
    )

    audio_logger.info(f"🚀 Started unified file processing: batch_job_id={batch_job_id}, files={len(files)}")

    return {
        "batch_job_id": batch_job_id,
        "message": f"Started processing {len(files)} files using unified pipeline",
        "status_url": f"/api/process-audio-files/jobs/{batch_job_id}",
        "total_files": len(files),
        "pipeline_type": "unified"
    }


async def process_files_unified_background(
    batch_job_id: str,
    file_data: List[Tuple[str, bytes]],
    user: User,
    device_name: str
) -> None:
    """Background task to process files using unified pipeline.

    This function:
    1. Updates the batch job status
    2. Processes each file individually using the unified pipeline
    3. Creates AudioProcessingItem for each file
    4. Submits to ProcessorManager.submit_audio_for_processing()
    """
    audio_logger.info(f"🚀 Starting unified background processing: batch_job_id={batch_job_id}, files={len(file_data)}")

    job_tracker = get_job_tracker()
    processor_manager = get_processor_manager()

    try:
        # Update batch job status to processing
        await job_tracker.update_job_status(batch_job_id, JobStatus.PROCESSING)

        pipeline_job_ids = []

        # Process files one by one
        for file_index, (filename, content) in enumerate(file_data):
            try:
                audio_logger.info(f"🔧 [Batch {batch_job_id}] Processing file {file_index + 1}/{len(file_data)}: {filename}")

                # Update file status in batch job
                await job_tracker.update_file_status(batch_job_id, filename, FileStatus.PROCESSING)

                # Validate file
                await validate_audio_file(filename, content)

                # Save file to persistent storage
                file_path = await save_uploaded_file(content, filename, user.user_id)

                # Generate client_id for this file
                file_device_name = f"{device_name}-{file_index + 1:03d}"
                from advanced_omi_backend.client_manager import generate_client_id
                client_id = generate_client_id(user, file_device_name)

                # Create AudioProcessingItem for unified pipeline
                processing_item = AudioProcessingItem.from_file_upload(
                    audio_file_path=file_path,
                    client_id=client_id,
                    device_name=file_device_name,
                    user_id=user.user_id,
                    user_email=user.email
                )

                # Submit to unified pipeline
                pipeline_job_id = await processor_manager.submit_audio_for_processing(processing_item)
                pipeline_job_ids.append(pipeline_job_id)

                # Update batch job file status
                await job_tracker.update_file_status(
                    batch_job_id,
                    filename,
                    FileStatus.COMPLETED
                )

                audio_logger.info(
                    f"✅ [Batch {batch_job_id}] File {filename} submitted to unified pipeline: {pipeline_job_id}"
                )

            except Exception as e:
                audio_logger.error(f"❌ [Batch {batch_job_id}] Failed to process file {filename}: {e}")
                await job_tracker.update_file_status(
                    batch_job_id,
                    filename,
                    FileStatus.FAILED,
                    error_message=str(e)
                )

        # Wait for all pipeline jobs to complete before marking batch job as complete
        audio_logger.info(f"⏳ [Batch {batch_job_id}] Waiting for {len(pipeline_job_ids)} pipeline jobs to complete...")

        completed_count = 0
        max_wait_time = 1800  # 30 minutes total timeout
        check_interval = 5  # Check every 5 seconds
        elapsed_time = 0

        while completed_count < len(pipeline_job_ids) and elapsed_time < max_wait_time:
            completed_count = 0

            for pipeline_job_id in pipeline_job_ids:
                try:
                    pipeline_job = await job_tracker.get_job(pipeline_job_id)
                    if pipeline_job and pipeline_job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                        completed_count += 1
                except Exception as e:
                    audio_logger.warning(f"⚠️ [Batch {batch_job_id}] Error checking pipeline job {pipeline_job_id}: {e}")

            if completed_count < len(pipeline_job_ids):
                audio_logger.info(f"⏳ [Batch {batch_job_id}] Pipeline progress: {completed_count}/{len(pipeline_job_ids)} jobs completed")
                await asyncio.sleep(check_interval)
                elapsed_time += check_interval

        # Check final status and mark batch job accordingly
        if completed_count == len(pipeline_job_ids):
            audio_logger.info(f"✅ [Batch {batch_job_id}] All {len(pipeline_job_ids)} pipeline jobs completed")
            audio_logger.info(f"📊 [Batch {batch_job_id}] Marking batch job as COMPLETED")
            await job_tracker.update_job_status(batch_job_id, JobStatus.COMPLETED)
            audio_logger.info(f"✅ [Batch {batch_job_id}] Batch job status updated to COMPLETED")
        else:
            error_msg = f"Pipeline processing timeout: {completed_count}/{len(pipeline_job_ids)} jobs completed after {elapsed_time}s"
            audio_logger.error(f"⏰ [Batch {batch_job_id}] {error_msg}")
            await job_tracker.update_job_status(batch_job_id, JobStatus.FAILED, error_msg)

        audio_logger.info(
            f"🏁 [Batch {batch_job_id}] Unified processing finished: "
            f"{completed_count}/{len(pipeline_job_ids)} pipeline jobs completed"
        )

    except Exception as e:
        audio_logger.error(f"❌ [Batch {batch_job_id}] Unified background processing failed: {e}")
        await job_tracker.update_job_status(batch_job_id, JobStatus.FAILED, str(e))


async def validate_audio_file(filename: str, content: bytes) -> None:
    """Validate uploaded audio file.

    Args:
        filename: Original filename
        content: File content

    Raises:
        ValueError: If file is invalid
    """
    # Check file extension
    if not filename or not filename.lower().endswith(".wav"):
        raise ValueError("Only WAV files are currently supported")

    # Check file size (reasonable limits)
    if len(content) > 500 * 1024 * 1024:  # 500MB limit
        raise ValueError("File too large (max 500MB)")

    if len(content) < 1024:  # 1KB minimum
        raise ValueError("File too small (min 1KB)")

    # Validate WAV format and duration
    try:
        duration = get_audio_duration(content)
        audio_logger.info(f"📊 File validation passed: {filename}, duration: {duration/60:.1f} minutes")

        # Optional: duration limits
        if duration > 3600:  # 1 hour limit
            audio_logger.warning(f"⚠️ Long file detected: {duration/60:.1f} minutes")

    except Exception as e:
        raise ValueError(f"Invalid WAV file: {e}")


# Enhanced job tracking endpoints for unified pipeline
async def get_unified_job_status(job_id: str) -> Dict:
    """Get status of a unified pipeline job (batch or pipeline).

    Args:
        job_id: Job ID (either batch job or pipeline job)

    Returns:
        Job status with enhanced pipeline information
    """
    job_tracker = get_job_tracker()
    job = await job_tracker.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.to_dict()

    # If it's a batch job, also include pipeline job information
    if job.job_type.value == "batch":
        # Get pipeline job IDs from file metadata
        pipeline_jobs = []
        for file_info in job.files:
            if hasattr(file_info, 'metadata') and file_info.metadata:
                pipeline_job_id = file_info.metadata.get("pipeline_job_id")
                if pipeline_job_id:
                    pipeline_job = await job_tracker.get_job(pipeline_job_id)
                    if pipeline_job:
                        pipeline_jobs.append(pipeline_job.to_dict())

        result["pipeline_jobs"] = pipeline_jobs

    return result


async def list_unified_jobs() -> Dict:
    """List all jobs with enhanced pipeline information."""
    job_tracker = get_job_tracker()

    # Get regular active jobs
    active_jobs = await job_tracker.get_active_jobs()

    # Get pipeline metrics
    pipeline_metrics = await job_tracker.get_pipeline_metrics()

    # Get active pipeline jobs
    pipeline_jobs = await job_tracker.get_active_pipeline_jobs()

    return {
        "active_batch_jobs": [job.to_dict() for job in active_jobs if job.job_type.value == "batch"],
        "active_pipeline_jobs": [job.to_dict() for job in pipeline_jobs],
        "pipeline_metrics": pipeline_metrics,
        "total_active_jobs": len(active_jobs) + len(pipeline_jobs)
    }