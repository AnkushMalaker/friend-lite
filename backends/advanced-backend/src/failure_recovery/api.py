"""
API endpoints for Failure Recovery System

This module provides REST API endpoints for monitoring and managing
the failure recovery system in the Friend-Lite backend.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from auth import current_superuser, current_active_user
from users import User

from .queue_tracker import QueueTracker, QueueType, QueueStatus, get_queue_tracker
from .persistent_queue import PersistentQueue, MessagePriority, get_persistent_queue
from .recovery_manager import RecoveryManager, get_recovery_manager
from .health_monitor import HealthMonitor, get_health_monitor
from .circuit_breaker import CircuitBreakerManager, CircuitBreakerConfig, get_circuit_breaker_manager

logger = logging.getLogger(__name__)

# Pydantic models for API responses
class QueueStatsResponse(BaseModel):
    queue_type: str
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    retry: int = 0
    dead_letter: int = 0

class ServiceHealthResponse(BaseModel):
    name: str
    status: str
    last_check: float
    response_time: float
    consecutive_failures: int
    error_message: Optional[str] = None

class RecoveryStatsResponse(BaseModel):
    recoveries_attempted: int
    recoveries_successful: int
    items_requeued: int
    items_escalated: int

class CircuitBreakerResponse(BaseModel):
    name: str
    state: str
    failure_count: int
    success_count: int
    total_calls: int
    successful_calls: int
    failed_calls: int

class PipelineStatusResponse(BaseModel):
    audio_uuid: str
    overall_status: str
    started_at: Optional[float]
    completed_at: Optional[float]
    has_failures: bool
    stages: Dict[str, Any]

# Create router
router = APIRouter(prefix="/api/failure-recovery", tags=["failure-recovery"])

# Queue Management Endpoints

@router.get("/queue-stats", response_model=List[QueueStatsResponse])
async def get_queue_stats(
    user: User = Depends(current_active_user),
    queue_tracker: QueueTracker = Depends(get_queue_tracker)
):
    """Get statistics for all processing queues"""
    try:
        stats = queue_tracker.get_queue_stats()
        
        response = []
        for queue_type in QueueType:
            queue_name = queue_type.value
            queue_stats = stats.get(queue_name, {})
            
            response.append(QueueStatsResponse(
                queue_type=queue_name,
                pending=queue_stats.get("pending", 0),
                processing=queue_stats.get("processing", 0),
                completed=queue_stats.get("completed", 0),
                failed=queue_stats.get("failed", 0),
                retry=queue_stats.get("retry", 0),
                dead_letter=queue_stats.get("dead_letter", 0)
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get queue statistics")

@router.get("/queue-stats/{queue_type}")
async def get_queue_stats_by_type(
    queue_type: str,
    user: User = Depends(current_active_user),
    queue_tracker: QueueTracker = Depends(get_queue_tracker)
):
    """Get statistics for a specific queue type"""
    try:
        # Validate queue type
        queue_enum = QueueType(queue_type.upper())
        
        stats = queue_tracker.get_queue_stats()
        queue_stats = stats.get(queue_enum.value, {})
        
        return QueueStatsResponse(
            queue_type=queue_enum.value,
            pending=queue_stats.get("pending", 0),
            processing=queue_stats.get("processing", 0),
            completed=queue_stats.get("completed", 0),
            failed=queue_stats.get("failed", 0),
            retry=queue_stats.get("retry", 0),
            dead_letter=queue_stats.get("dead_letter", 0)
        )
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid queue type: {queue_type}")
    except Exception as e:
        logger.error(f"Error getting queue stats for {queue_type}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get queue statistics")

@router.get("/pipeline-status/{audio_uuid}", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    audio_uuid: str,
    user: User = Depends(current_active_user),
    queue_tracker: QueueTracker = Depends(get_queue_tracker)
):
    """Get processing pipeline status for an audio UUID"""
    try:
        pipeline_status = queue_tracker.get_processing_pipeline_status(audio_uuid)
        
        return PipelineStatusResponse(
            audio_uuid=pipeline_status["audio_uuid"],
            overall_status=pipeline_status["overall_status"],
            started_at=pipeline_status["started_at"],
            completed_at=pipeline_status["completed_at"],
            has_failures=pipeline_status["has_failures"],
            stages=pipeline_status["stages"]
        )
        
    except Exception as e:
        logger.error(f"Error getting pipeline status for {audio_uuid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pipeline status")

@router.get("/client-stats/{client_id}")
async def get_client_stats(
    client_id: str,
    user: User = Depends(current_active_user),
    queue_tracker: QueueTracker = Depends(get_queue_tracker)
):
    """Get processing statistics for a specific client"""
    try:
        # Check if user can access this client's data
        if user.id != client_id and not user.is_superuser:
            raise HTTPException(status_code=403, detail="Access denied")
        
        stats = queue_tracker.get_client_stats(client_id)
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting client stats for {client_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get client statistics")

# Health Monitoring Endpoints

@router.get("/health")
async def get_overall_health(
    user: User = Depends(current_active_user),
    health_monitor: HealthMonitor = Depends(get_health_monitor)
):
    """Get overall system health"""
    try:
        health = health_monitor.get_overall_health()
        return health
        
    except Exception as e:
        logger.error(f"Error getting overall health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health")

@router.get("/health/{service_name}")
async def get_service_health(
    service_name: str,
    user: User = Depends(current_active_user),
    health_monitor: HealthMonitor = Depends(get_health_monitor)
):
    """Get health status for a specific service"""
    try:
        health = health_monitor.get_service_health(service_name)
        
        if not health:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
        
        return ServiceHealthResponse(
            name=health.name,
            status=health.status.value,
            last_check=health.last_check,
            response_time=health.response_time,
            consecutive_failures=health.consecutive_failures,
            error_message=health.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting service health for {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service health")

@router.post("/health/{service_name}/check")
async def manual_health_check(
    service_name: str,
    user: User = Depends(current_superuser),
    health_monitor: HealthMonitor = Depends(get_health_monitor)
):
    """Manually trigger a health check for a service"""
    try:
        result = await health_monitor.manual_health_check(service_name)
        return result
        
    except Exception as e:
        logger.error(f"Error in manual health check for {service_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to perform health check")

# Recovery Management Endpoints

@router.get("/recovery-stats", response_model=RecoveryStatsResponse)
async def get_recovery_stats(
    user: User = Depends(current_active_user),
    recovery_manager: RecoveryManager = Depends(get_recovery_manager)
):
    """Get recovery system statistics"""
    try:
        stats = recovery_manager.get_stats()
        recovery_stats = stats.get("recovery_stats", {})
        
        return RecoveryStatsResponse(
            recoveries_attempted=recovery_stats.get("recoveries_attempted", 0),
            recoveries_successful=recovery_stats.get("recoveries_successful", 0),
            items_requeued=recovery_stats.get("items_requeued", 0),
            items_escalated=recovery_stats.get("items_escalated", 0)
        )
        
    except Exception as e:
        logger.error(f"Error getting recovery stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recovery statistics")

@router.post("/recovery/{queue_type}/trigger")
async def trigger_manual_recovery(
    queue_type: str,
    item_id: Optional[str] = Query(None),
    user: User = Depends(current_superuser),
    recovery_manager: RecoveryManager = Depends(get_recovery_manager)
):
    """Manually trigger recovery for a queue or specific item"""
    try:
        # Validate queue type
        queue_enum = QueueType(queue_type.upper())
        
        result = await recovery_manager.manual_recovery(queue_enum, item_id)
        return result
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid queue type: {queue_type}")
    except Exception as e:
        logger.error(f"Error triggering manual recovery: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger recovery")

# Circuit Breaker Endpoints

@router.get("/circuit-breakers")
async def get_circuit_breaker_stats(
    user: User = Depends(current_active_user),
    circuit_manager: CircuitBreakerManager = Depends(get_circuit_breaker_manager)
):
    """Get statistics for all circuit breakers"""
    try:
        stats = circuit_manager.get_all_stats()
        
        response = []
        for name, breaker_stats in stats.items():
            response.append(CircuitBreakerResponse(
                name=name,
                state=breaker_stats["state"],
                failure_count=breaker_stats["failure_count"],
                success_count=breaker_stats["success_count"],
                total_calls=breaker_stats["stats"]["total_calls"],
                successful_calls=breaker_stats["stats"]["successful_calls"],
                failed_calls=breaker_stats["stats"]["failed_calls"]
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting circuit breaker stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get circuit breaker statistics")

@router.get("/circuit-breakers/{name}")
async def get_circuit_breaker_stats_by_name(
    name: str,
    user: User = Depends(current_active_user),
    circuit_manager: CircuitBreakerManager = Depends(get_circuit_breaker_manager)
):
    """Get statistics for a specific circuit breaker"""
    try:
        all_stats = circuit_manager.get_all_stats()
        
        if name not in all_stats:
            raise HTTPException(status_code=404, detail=f"Circuit breaker {name} not found")
        
        breaker_stats = all_stats[name]
        
        return CircuitBreakerResponse(
            name=name,
            state=breaker_stats["state"],
            failure_count=breaker_stats["failure_count"],
            success_count=breaker_stats["success_count"],
            total_calls=breaker_stats["stats"]["total_calls"],
            successful_calls=breaker_stats["stats"]["successful_calls"],
            failed_calls=breaker_stats["stats"]["failed_calls"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting circuit breaker stats for {name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get circuit breaker statistics")

@router.post("/circuit-breakers/{name}/reset")
async def reset_circuit_breaker(
    name: str,
    user: User = Depends(current_superuser),
    circuit_manager: CircuitBreakerManager = Depends(get_circuit_breaker_manager)
):
    """Reset a specific circuit breaker"""
    try:
        success = circuit_manager.reset_circuit_breaker(name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Circuit breaker {name} not found")
        
        return {"message": f"Circuit breaker {name} reset successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting circuit breaker {name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset circuit breaker")

@router.post("/circuit-breakers/reset-all")
async def reset_all_circuit_breakers(
    user: User = Depends(current_superuser),
    circuit_manager: CircuitBreakerManager = Depends(get_circuit_breaker_manager)
):
    """Reset all circuit breakers"""
    try:
        circuit_manager.reset_all()
        return {"message": "All circuit breakers reset successfully"}
        
    except Exception as e:
        logger.error(f"Error resetting all circuit breakers: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset circuit breakers")

# Persistent Queue Endpoints

@router.get("/persistent-queues")
async def get_persistent_queue_stats(
    user: User = Depends(current_active_user),
    persistent_queue: PersistentQueue = Depends(get_persistent_queue)
):
    """Get statistics for all persistent queues"""
    try:
        stats = await persistent_queue.get_all_queue_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting persistent queue stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get persistent queue statistics")

@router.get("/persistent-queues/{queue_name}")
async def get_persistent_queue_stats_by_name(
    queue_name: str,
    user: User = Depends(current_active_user),
    persistent_queue: PersistentQueue = Depends(get_persistent_queue)
):
    """Get statistics for a specific persistent queue"""
    try:
        stats = await persistent_queue.get_queue_stats(queue_name)
        return {"queue_name": queue_name, "stats": stats}
        
    except Exception as e:
        logger.error(f"Error getting persistent queue stats for {queue_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get persistent queue statistics")

@router.get("/persistent-queues/{queue_name}/dead-letter")
async def get_dead_letter_messages(
    queue_name: str,
    limit: int = Query(100, ge=1, le=1000),
    user: User = Depends(current_superuser),
    persistent_queue: PersistentQueue = Depends(get_persistent_queue)
):
    """Get dead letter messages for a queue"""
    try:
        messages = await persistent_queue.get_dead_letter_messages(queue_name, limit)
        
        return {
            "queue_name": queue_name,
            "count": len(messages),
            "messages": [
                {
                    "id": msg.id,
                    "payload": msg.payload,
                    "retry_count": msg.retry_count,
                    "error_message": msg.error_message,
                    "created_at": msg.created_at,
                    "client_id": msg.client_id,
                    "user_id": msg.user_id,
                    "audio_uuid": msg.audio_uuid
                }
                for msg in messages
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting dead letter messages for {queue_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dead letter messages")

@router.post("/persistent-queues/{queue_name}/dead-letter/{message_id}/requeue")
async def requeue_dead_letter_message(
    queue_name: str,
    message_id: str,
    max_retries: int = Query(3, ge=1, le=10),
    user: User = Depends(current_superuser),
    persistent_queue: PersistentQueue = Depends(get_persistent_queue)
):
    """Requeue a dead letter message"""
    try:
        success = await persistent_queue.requeue_dead_letter_message(message_id, max_retries)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Dead letter message {message_id} not found")
        
        return {"message": f"Message {message_id} requeued successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error requeuing dead letter message {message_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to requeue message")

# Maintenance Endpoints

@router.post("/maintenance/cleanup")
async def cleanup_old_data(
    queue_days: int = Query(7, ge=1, le=30),
    persistent_hours: int = Query(24, ge=1, le=168),
    user: User = Depends(current_superuser),
    queue_tracker: QueueTracker = Depends(get_queue_tracker),
    persistent_queue: PersistentQueue = Depends(get_persistent_queue)
):
    """Clean up old completed data"""
    try:
        # Cleanup queue tracker
        queue_deleted = queue_tracker.cleanup_old_items(queue_days)
        
        # Cleanup persistent queue
        persistent_deleted = await persistent_queue.cleanup_completed_messages(persistent_hours)
        
        return {
            "queue_items_deleted": queue_deleted,
            "persistent_messages_deleted": persistent_deleted,
            "cleanup_completed": True
        }
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup old data")

@router.get("/system-overview")
async def get_system_overview(
    user: User = Depends(current_active_user),
    queue_tracker: QueueTracker = Depends(get_queue_tracker),
    health_monitor: HealthMonitor = Depends(get_health_monitor),
    recovery_manager: RecoveryManager = Depends(get_recovery_manager),
    circuit_manager: CircuitBreakerManager = Depends(get_circuit_breaker_manager)
):
    """Get comprehensive system overview"""
    try:
        # Get all system stats
        queue_stats = queue_tracker.get_queue_stats()
        health_stats = health_monitor.get_overall_health()
        recovery_stats = recovery_manager.get_stats()
        circuit_stats = circuit_manager.get_all_stats()
        
        # Calculate summary metrics
        total_queue_items = sum(
            sum(queue_data.values()) for queue_data in queue_stats.values()
        )
        
        healthy_services = sum(
            1 for service in health_stats["services"].values()
            if service["status"] == "healthy"
        )
        
        open_circuits = sum(
            1 for circuit in circuit_stats.values()
            if circuit["state"] == "open"
        )
        
        return {
            "system_status": health_stats["overall_status"],
            "summary": {
                "total_queue_items": total_queue_items,
                "healthy_services": healthy_services,
                "total_services": health_stats["total_services"],
                "open_circuits": open_circuits,
                "total_circuits": len(circuit_stats),
                "recoveries_attempted": recovery_stats["recovery_stats"]["recoveries_attempted"],
                "recoveries_successful": recovery_stats["recovery_stats"]["recoveries_successful"]
            },
            "queue_stats": queue_stats,
            "health_stats": health_stats,
            "recovery_stats": recovery_stats["recovery_stats"],
            "circuit_stats": {
                name: {
                    "state": stats["state"],
                    "failure_count": stats["failure_count"],
                    "total_calls": stats["stats"]["total_calls"]
                }
                for name, stats in circuit_stats.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system overview")

# Include the router in the main application
def get_failure_recovery_router():
    """Get the failure recovery API router"""
    return router