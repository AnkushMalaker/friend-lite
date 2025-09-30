"""Enhanced system routes with unified pipeline support.

This module demonstrates how to integrate unified pipeline functionality
with existing routes while maintaining backward compatibility.
"""

import logging

from advanced_omi_backend.auth import current_superuser
from advanced_omi_backend.controllers import system_controller
from advanced_omi_backend.job_tracker import get_job_tracker
from advanced_omi_backend.unified_file_upload import (
    get_unified_job_status,
    list_unified_jobs,
    process_audio_files_unified,
)
from advanced_omi_backend.users import User
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["unified-system"])


# Enhanced file upload endpoints with unified pipeline
@router.post("/process-audio-files-unified")
async def process_audio_files_unified_route(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    device_name: str = Form("unified-upload"),
    current_user: User = Depends(current_superuser),
):
    """Process uploaded audio files using unified pipeline. Admin only.

    This endpoint:
    - Uses the new AudioProcessingItem and unified pipeline
    - Creates both batch jobs (for file tracking) and pipeline jobs (for processing)
    - Provides enhanced monitoring and debugging capabilities
    """
    try:
        return await process_audio_files_unified(
            background_tasks, current_user, files, device_name
        )
    except Exception as e:
        logger.error(f"Unified file processing failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process files: {str(e)}"}
        )


@router.get("/jobs/{job_id}/unified")
async def get_unified_job_status_route(
    job_id: str,
    current_user: User = Depends(current_superuser)
):
    """Get enhanced status of a job (batch or pipeline) with unified pipeline info. Admin only."""
    try:
        return await get_unified_job_status(job_id)
    except Exception as e:
        logger.error(f"Failed to get unified job status: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get job status: {str(e)}"}
        )


@router.get("/jobs/unified")
async def list_unified_jobs_route(current_user: User = Depends(current_superuser)):
    """List all jobs with enhanced pipeline information. Admin only."""
    try:
        return await list_unified_jobs()
    except Exception as e:
        logger.error(f"Failed to list unified jobs: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list jobs: {str(e)}"}
        )


# Pipeline monitoring endpoints (from Phase 7)
@router.get("/pipeline/jobs")
async def get_pipeline_jobs(current_user: User = Depends(current_superuser)):
    """Get all active pipeline jobs. Admin only."""
    try:
        job_tracker = get_job_tracker()
        jobs = await job_tracker.get_active_pipeline_jobs()

        return {
            "pipeline_jobs": [job.to_dict() for job in jobs],
            "total_jobs": len(jobs)
        }
    except Exception as e:
        logger.error(f"Failed to get pipeline jobs: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get pipeline jobs: {str(e)}"}
        )


@router.get("/pipeline/metrics")
async def get_pipeline_metrics(current_user: User = Depends(current_superuser)):
    """Get pipeline performance metrics. Admin only."""
    try:
        job_tracker = get_job_tracker()
        metrics = await job_tracker.get_pipeline_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get pipeline metrics: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get pipeline metrics: {str(e)}"}
        )


@router.get("/pipeline/bottlenecks")
async def get_pipeline_bottlenecks(current_user: User = Depends(current_superuser)):
    """Identify pipeline bottlenecks with recommendations. Admin only."""
    try:
        job_tracker = get_job_tracker()
        metrics = await job_tracker.get_pipeline_metrics()

        # Analyze bottlenecks
        bottlenecks = []
        stage_metrics = metrics.get("stage_metrics", {})

        for stage, data in stage_metrics.items():
            avg_queue_lag = data.get("avg_queue_lag_seconds", 0)
            avg_processing_lag = data.get("avg_processing_lag_seconds", 0)

            # Flag stages with high lag (configurable thresholds)
            if avg_queue_lag > 10:  # 10 second threshold
                bottlenecks.append({
                    "stage": stage,
                    "type": "queue_lag",
                    "value": avg_queue_lag,
                    "severity": "high" if avg_queue_lag > 30 else "medium",
                    "description": f"High queue lag in {stage} stage ({avg_queue_lag:.1f}s)",
                    "recommendation": f"Consider increasing {stage} processor capacity"
                })

            if avg_processing_lag > 30:  # 30 second threshold
                bottlenecks.append({
                    "stage": stage,
                    "type": "processing_lag",
                    "value": avg_processing_lag,
                    "severity": "high" if avg_processing_lag > 120 else "medium",
                    "description": f"Slow processing in {stage} stage ({avg_processing_lag:.1f}s)",
                    "recommendation": f"Optimize {stage} processing algorithms or increase resources"
                })

        # Generate overall recommendations
        recommendations = generate_recommendations(bottlenecks, metrics)

        return {
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "metrics_summary": {
                "total_pipeline_jobs": metrics.get("total_pipeline_jobs", 0),
                "active_pipeline_jobs": metrics.get("active_pipeline_jobs", 0),
                "stages_analyzed": len(stage_metrics)
            }
        }
    except Exception as e:
        logger.error(f"Failed to analyze pipeline bottlenecks: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to analyze bottlenecks: {str(e)}"}
        )


def generate_recommendations(bottlenecks: list, metrics: dict) -> list:
    """Generate recommendations based on bottleneck analysis."""
    recommendations = []

    if not bottlenecks:
        recommendations.append({
            "type": "success",
            "message": "No significant bottlenecks detected",
            "action": "Monitor regularly to maintain performance"
        })
        return recommendations

    # Count bottlenecks by type
    queue_issues = len([b for b in bottlenecks if b["type"] == "queue_lag"])
    processing_issues = len([b for b in bottlenecks if b["type"] == "processing_lag"])

    if queue_issues > processing_issues:
        recommendations.append({
            "type": "scaling",
            "message": "Multiple queue lag issues detected",
            "action": "Consider horizontal scaling - add more processor instances"
        })
    elif processing_issues > queue_issues:
        recommendations.append({
            "type": "optimization",
            "message": "Multiple processing lag issues detected",
            "action": "Consider algorithm optimization or vertical scaling (more CPU/memory)"
        })

    # Stage-specific recommendations
    stages_with_issues = set(b["stage"] for b in bottlenecks)
    if "transcription" in stages_with_issues:
        recommendations.append({
            "type": "transcription",
            "message": "Transcription bottleneck detected",
            "action": "Check Deepgram API limits or switch to local ASR services"
        })

    if "memory" in stages_with_issues:
        recommendations.append({
            "type": "memory",
            "message": "Memory processing bottleneck detected",
            "action": "Review LLM provider limits or optimize memory extraction prompts"
        })

    return recommendations


# Enhanced processor status with unified pipeline metrics
@router.get("/processor/status/unified")
async def get_unified_processor_status(current_user: User = Depends(current_superuser)):
    """Get processor status with unified pipeline metrics. Admin only."""
    try:
        # Get traditional processor status
        traditional_status = await system_controller.get_processor_status()

        # Get unified pipeline metrics
        job_tracker = get_job_tracker()
        pipeline_metrics = await job_tracker.get_pipeline_metrics()

        return {
            "traditional_processor": traditional_status,
            "unified_pipeline": {
                "metrics": pipeline_metrics,
                "status": "active" if pipeline_metrics["active_pipeline_jobs"] > 0 else "idle"
            },
            "integration_status": {
                "unified_pipeline_enabled": True,
                "backward_compatibility": True,
                "pipeline_version": "v1.0"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get unified processor status: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get processor status: {str(e)}"}
        )


# WebSocket integration endpoint
@router.get("/websocket/unified-status")
async def get_websocket_unified_status(current_user: User = Depends(current_superuser)):
    """Get WebSocket integration status with unified pipeline. Admin only."""
    from advanced_omi_backend.client_manager import get_client_manager

    try:
        client_manager = get_client_manager()
        active_clients = client_manager.get_active_clients()

        # Check which clients are using unified pipeline features
        unified_clients = []
        for client_id in active_clients:
            client_state = client_manager.get_client_state(client_id)
            if client_state and hasattr(client_state, 'is_recording'):
                unified_clients.append({
                    "client_id": client_id,
                    "is_recording": client_state.is_recording,
                    "has_audio_buffer": len(getattr(client_state, 'audio_buffer', [])) > 0,
                    "unified_pipeline_ready": True
                })

        return {
            "total_clients": len(active_clients),
            "unified_enabled_clients": len(unified_clients),
            "client_details": unified_clients,
            "websocket_status": "operational"
        }
    except Exception as e:
        logger.error(f"Failed to get WebSocket unified status: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get WebSocket status: {str(e)}"}
        )