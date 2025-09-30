"""Pipeline tracker for monitoring audio processing pipeline performance.

This module tracks pipeline events by audio_uuid to provide visibility into
queue depths, processing lag, and bottlenecks across the entire audio pipeline.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Information about a tracked task."""

    task: asyncio.Task
    name: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    cancelled: bool = False


@dataclass
class PipelineEvent:
    """Pipeline event for tracking audio processing flow."""

    audio_uuid: str
    conversation_id: Optional[str]
    event_type: Literal["enqueue", "dequeue", "complete", "failed"]
    stage: Literal["audio", "transcription", "memory", "cropping"]
    timestamp: float
    queue_size: int
    processing_time_ms: Optional[float] = None
    client_id: Optional[str] = None  # For debugging only
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueMetrics:
    """Aggregated metrics for a pipeline stage."""

    stage: str
    current_depth: int = 0
    total_enqueued: int = 0
    total_dequeued: int = 0
    total_completed: int = 0
    total_failed: int = 0
    avg_queue_time_ms: float = 0.0
    avg_processing_time_ms: float = 0.0
    last_updated: float = field(default_factory=time.time)


class PipelineTracker:
    """Tracks pipeline events and performance across audio processing stages."""

    def __init__(self):
        # Task tracking (still needed for memory/cropping async tasks)
        self.tasks: Dict[str, TaskInfo] = {}
        self.completed_tasks: List[TaskInfo] = []
        self.max_completed_history = 1000  # Keep last N completed tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Pipeline event tracking by audio_uuid
        self.audio_sessions: Dict[str, deque[PipelineEvent]] = defaultdict(lambda: deque(maxlen=100))
        self.conversation_mapping: Dict[str, str] = {}  # conversation_id -> audio_uuid
        self.queue_metrics: Dict[str, QueueMetrics] = {
            "audio": QueueMetrics("audio"),
            "transcription": QueueMetrics("transcription"),
            "memory": QueueMetrics("memory"),
            "cropping": QueueMetrics("cropping")
        }

        # Event history cleanup
        self.max_events_per_session = 100
        self.session_cleanup_age_hours = 6

    async def start(self):
        """Start the pipeline tracker."""
        logger.info("Starting PipelineTracker")
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def shutdown(self):
        """Shutdown the pipeline tracker and cancel all tasks."""
        logger.info("Shutting down PipelineTracker")
        self._shutdown = True

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel all active tasks
        active_tasks = list(self.tasks.values())
        logger.info(f"Cancelling {len(active_tasks)} active tasks")

        for task_info in active_tasks:
            if not task_info.task.done():
                task_info.task.cancel()
                task_info.cancelled = True

        # Wait for all tasks to complete with timeout
        if active_tasks:
            tasks = [info.task for info in active_tasks]
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete within shutdown timeout")

        logger.info("PipelineTracker shutdown complete")

    def track_task(
        self, task: asyncio.Task, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track a background task and return the task ID."""
        if metadata is None:
            metadata = {}

        task_id = f"{name}_{id(task)}"
        task_info = TaskInfo(task=task, name=name, metadata=metadata)
        self.tasks[task_id] = task_info

        # Add done callback to track completion
        task.add_done_callback(lambda t: self._task_done(task_id))

        logger.debug(f"Tracking task: {name} (id: {task_id})")
        return task_id

    def _task_done(self, task_id: str):
        """Handle task completion."""
        if task_id not in self.tasks:
            return

        task_info = self.tasks[task_id]
        task_info.completed_at = time.time()

        # Check if task failed
        if task_info.task.cancelled():
            task_info.cancelled = True
            logger.debug(f"Task cancelled: {task_info.name}")
        elif task_info.task.exception():
            try:
                exc = task_info.task.exception()
                task_info.error = str(exc)
                logger.error(f"Task failed: {task_info.name} - {exc}")
            except Exception:
                task_info.error = "Unknown error"
        else:
            # DEBUG: Add more visible logging for memory task completion
            if "memory_" in task_info.name:
                logger.info(
                    f"✅ MEMORY TASK COMPLETED: {task_info.name} at {task_info.completed_at}"
                )
            else:
                logger.debug(f"Task completed: {task_info.name}")

        # Move to completed list
        del self.tasks[task_id]
        self.completed_tasks.append(task_info)

        # Trim completed history
        if len(self.completed_tasks) > self.max_completed_history:
            self.completed_tasks = self.completed_tasks[-self.max_completed_history :]

    async def _periodic_cleanup(self):
        """Periodically clean up old pipeline events and completed tasks."""
        logger.info("Started periodic pipeline cleanup")

        try:
            while not self._shutdown:
                try:
                    # Wait 30 seconds between cleanups
                    await asyncio.sleep(30)

                    current_time = time.time()
                    cleanup_age_seconds = self.session_cleanup_age_hours * 3600

                    # Clean up old pipeline events
                    sessions_to_remove = []
                    for audio_uuid, events in list(self.audio_sessions.items()):
                        if events and (current_time - events[-1].timestamp) > cleanup_age_seconds:
                            sessions_to_remove.append(audio_uuid)

                    for audio_uuid in sessions_to_remove:
                        del self.audio_sessions[audio_uuid]
                        logger.debug(f"Cleaned up old pipeline events for audio session {audio_uuid}")

                    # Clean up old conversation mappings
                    conversations_to_remove = []
                    for conv_id, audio_uuid in list(self.conversation_mapping.items()):
                        if audio_uuid not in self.audio_sessions:
                            conversations_to_remove.append(conv_id)

                    for conv_id in conversations_to_remove:
                        del self.conversation_mapping[conv_id]

                    # Log statistics
                    active_count = len(self.tasks)
                    completed_count = len(self.completed_tasks)
                    active_sessions = len(self.audio_sessions)

                    if active_count > 0 or active_sessions > 10:
                        logger.info(
                            f"Pipeline tracker stats: {active_count} active tasks, "
                            f"{completed_count} completed tasks, "
                            f"{active_sessions} active audio sessions"
                        )

                        # Log details of long-running tasks
                        long_running = []
                        for task_info in self.tasks.values():
                            age = current_time - task_info.created_at
                            if age > 60:  # Tasks older than 1 minute
                                long_running.append(f"{task_info.name} ({age:.0f}s)")

                        if long_running:
                            logger.info(f"Long-running tasks: {', '.join(long_running[:5])}")

                except Exception as e:
                    logger.error(f"Error in periodic cleanup: {e}", exc_info=True)

        except asyncio.CancelledError:
            logger.info("Periodic cleanup cancelled")
        finally:
            logger.info("Periodic cleanup stopped")

    def get_active_tasks(self) -> List[TaskInfo]:
        """Get list of currently active tasks."""
        return list(self.tasks.values())

    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """Get task info by task ID from both active and completed tasks."""
        # First check active tasks
        task_info = self.tasks.get(task_id)
        if task_info:
            return task_info

        # Then check completed tasks
        for completed_task in self.completed_tasks:
            if f"{completed_task.name}_{id(completed_task.task)}" == task_id:
                return completed_task

        return None

    def get_task_count_by_type(self) -> Dict[str, int]:
        """Get count of active tasks grouped by type."""
        counts: Dict[str, int] = {}
        for task_info in self.tasks.values():
            task_type = task_info.metadata.get("type", "unknown")
            counts[task_type] = counts.get(task_type, 0) + 1
        return counts

    # Pipeline event tracking methods

    def track_enqueue(self, stage: str, audio_uuid: str, queue_size: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track when an item is enqueued to a processing stage."""
        if metadata is None:
            metadata = {}

        event = PipelineEvent(
            audio_uuid=audio_uuid,
            conversation_id=self.conversation_mapping.get(audio_uuid),
            event_type="enqueue",
            stage=stage,
            timestamp=time.time(),
            queue_size=queue_size,
            client_id=metadata.get("client_id"),
            user_id=metadata.get("user_id"),
            metadata=metadata
        )

        self.audio_sessions[audio_uuid].append(event)

        # Update queue metrics
        if stage in self.queue_metrics:
            metrics = self.queue_metrics[stage]
            metrics.total_enqueued += 1
            metrics.current_depth = queue_size
            metrics.last_updated = event.timestamp

        logger.debug(f"📥 Pipeline enqueue: {stage} for {audio_uuid} (queue depth: {queue_size})")

    def track_dequeue(self, stage: str, audio_uuid: str, queue_size: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track when an item is dequeued from a processing stage."""
        if metadata is None:
            metadata = {}

        event = PipelineEvent(
            audio_uuid=audio_uuid,
            conversation_id=self.conversation_mapping.get(audio_uuid),
            event_type="dequeue",
            stage=stage,
            timestamp=time.time(),
            queue_size=queue_size,
            client_id=metadata.get("client_id"),
            user_id=metadata.get("user_id"),
            metadata=metadata
        )

        self.audio_sessions[audio_uuid].append(event)

        # Calculate queue time if we have an enqueue event
        events = self.audio_sessions[audio_uuid]
        enqueue_event = None
        for e in reversed(events):
            if e.stage == stage and e.event_type == "enqueue":
                enqueue_event = e
                break

        queue_time_ms = 0.0
        if enqueue_event:
            queue_time_ms = (event.timestamp - enqueue_event.timestamp) * 1000
            event.metadata["queue_time_ms"] = queue_time_ms

        # Update queue metrics
        if stage in self.queue_metrics:
            metrics = self.queue_metrics[stage]
            metrics.total_dequeued += 1
            metrics.current_depth = queue_size
            metrics.last_updated = event.timestamp

            # Update average queue time
            if queue_time_ms > 0:
                if metrics.avg_queue_time_ms == 0:
                    metrics.avg_queue_time_ms = queue_time_ms
                else:
                    metrics.avg_queue_time_ms = (metrics.avg_queue_time_ms + queue_time_ms) / 2

        logger.debug(f"📤 Pipeline dequeue: {stage} for {audio_uuid} (queue time: {queue_time_ms:.1f}ms)")

    def track_complete(self, stage: str, audio_uuid: str, processing_time_ms: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track when processing completes for a stage."""
        if metadata is None:
            metadata = {}

        event = PipelineEvent(
            audio_uuid=audio_uuid,
            conversation_id=self.conversation_mapping.get(audio_uuid),
            event_type="complete",
            stage=stage,
            timestamp=time.time(),
            queue_size=0,  # Not applicable for completion
            processing_time_ms=processing_time_ms,
            client_id=metadata.get("client_id"),
            user_id=metadata.get("user_id"),
            metadata=metadata
        )

        self.audio_sessions[audio_uuid].append(event)

        # Update queue metrics
        if stage in self.queue_metrics:
            metrics = self.queue_metrics[stage]
            metrics.total_completed += 1
            metrics.last_updated = event.timestamp

            # Update average processing time
            if processing_time_ms is not None:
                if metrics.avg_processing_time_ms == 0:
                    metrics.avg_processing_time_ms = processing_time_ms
                else:
                    metrics.avg_processing_time_ms = (metrics.avg_processing_time_ms + processing_time_ms) / 2

        logger.debug(f"✅ Pipeline complete: {stage} for {audio_uuid} (processing time: {processing_time_ms or 0:.1f}ms)")

    def track_failed(self, stage: str, audio_uuid: str, error: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track when processing fails for a stage."""
        if metadata is None:
            metadata = {}

        metadata["error"] = error

        event = PipelineEvent(
            audio_uuid=audio_uuid,
            conversation_id=self.conversation_mapping.get(audio_uuid),
            event_type="failed",
            stage=stage,
            timestamp=time.time(),
            queue_size=0,  # Not applicable for failure
            client_id=metadata.get("client_id"),
            user_id=metadata.get("user_id"),
            metadata=metadata
        )

        self.audio_sessions[audio_uuid].append(event)

        # Update queue metrics
        if stage in self.queue_metrics:
            metrics = self.queue_metrics[stage]
            metrics.total_failed += 1
            metrics.last_updated = event.timestamp

        logger.warning(f"❌ Pipeline failed: {stage} for {audio_uuid} - {error}")

    def link_conversation(self, audio_uuid: str, conversation_id: str) -> None:
        """Link a conversation ID to an audio UUID for tracking."""
        self.conversation_mapping[conversation_id] = audio_uuid

        # Update all events for this audio session to include conversation_id
        if audio_uuid in self.audio_sessions:
            for event in self.audio_sessions[audio_uuid]:
                event.conversation_id = conversation_id

        logger.debug(f"🔗 Linked conversation {conversation_id} to audio {audio_uuid}")

    def get_pipeline_events(self, audio_uuid: str) -> List[PipelineEvent]:
        """Get all pipeline events for a specific audio session."""
        return list(self.audio_sessions.get(audio_uuid, []))

    def get_conversation_events(self, conversation_id: str) -> List[PipelineEvent]:
        """Get all pipeline events for a specific conversation."""
        audio_uuid = self.conversation_mapping.get(conversation_id)
        if audio_uuid:
            return self.get_pipeline_events(audio_uuid)
        return []

    def get_queue_lag(self, stage: str) -> float:
        """Get average queue lag in milliseconds for a stage."""
        metrics = self.queue_metrics.get(stage)
        return metrics.avg_queue_time_ms if metrics else 0.0

    def get_processing_lag(self, stage: str) -> float:
        """Get average processing lag in milliseconds for a stage."""
        metrics = self.queue_metrics.get(stage)
        return metrics.avg_processing_time_ms if metrics else 0.0

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Analyze pipeline bottlenecks and return recommendations."""
        bottlenecks = []
        slowest_stage = None
        slowest_time = 0.0

        for stage, metrics in self.queue_metrics.items():
            total_time = metrics.avg_queue_time_ms + metrics.avg_processing_time_ms

            if total_time > slowest_time:
                slowest_time = total_time
                slowest_stage = stage

            # Identify bottlenecks (arbitrary thresholds for now)
            if metrics.avg_queue_time_ms > 5000:  # 5 second queue time
                severity = "high" if metrics.avg_queue_time_ms > 15000 else "medium"
                bottlenecks.append({
                    "stage": stage,
                    "type": "queue_lag",
                    "severity": severity,
                    "avg_queue_time_ms": metrics.avg_queue_time_ms,
                    "current_depth": metrics.current_depth
                })

            if metrics.avg_processing_time_ms > 10000:  # 10 second processing time
                severity = "high" if metrics.avg_processing_time_ms > 30000 else "medium"
                bottlenecks.append({
                    "stage": stage,
                    "type": "processing_lag",
                    "severity": severity,
                    "avg_processing_time_ms": metrics.avg_processing_time_ms
                })

        return {
            "bottlenecks": bottlenecks,
            "slowest_stage": slowest_stage,
            "slowest_stage_total_time_ms": slowest_time,
            "overall_health": "healthy" if not bottlenecks else "degraded"
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the pipeline tracker including pipeline metrics."""
        current_time = time.time()
        active_tasks = self.get_active_tasks()

        # Calculate task ages
        task_ages = []
        oldest_task = None
        for task_info in active_tasks:
            age = current_time - task_info.created_at
            task_ages.append(age)
            if oldest_task is None or age > oldest_task[1]:
                oldest_task = (task_info.name, age)

        # Count errors in recent completed tasks
        recent_errors = 0
        recent_cancelled = 0
        for task_info in self.completed_tasks[-100:]:  # Last 100 tasks
            if task_info.error:
                recent_errors += 1
            elif task_info.cancelled:
                recent_cancelled += 1

        # Pipeline health
        bottleneck_analysis = self.get_bottleneck_analysis()
        pipeline_health = {
            stage: {
                "queue_depth": metrics.current_depth,
                "avg_queue_time_ms": metrics.avg_queue_time_ms,
                "avg_processing_time_ms": metrics.avg_processing_time_ms,
                "total_processed": metrics.total_completed,
                "total_failed": metrics.total_failed
            }
            for stage, metrics in self.queue_metrics.items()
        }

        status = {
            # Legacy task tracking
            "active_tasks": len(active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "task_counts_by_type": self.get_task_count_by_type(),
            "oldest_task": oldest_task[0] if oldest_task else None,
            "oldest_task_age": oldest_task[1] if oldest_task else 0,
            "average_task_age": sum(task_ages) / len(task_ages) if task_ages else 0,
            "recent_errors": recent_errors,
            "recent_cancelled": recent_cancelled,

            # Pipeline tracking
            "active_sessions": len(self.audio_sessions),
            "active_conversations": len(self.conversation_mapping),
            "pipeline_health": pipeline_health,
            "bottlenecks": bottleneck_analysis["bottlenecks"],
            "overall_pipeline_health": bottleneck_analysis["overall_health"],

            # Overall health
            "healthy": len(active_tasks) < 1000
            and (oldest_task[1] < 3600 if oldest_task else True)
            and bottleneck_analysis["overall_health"] == "healthy",
        }

        return status


# Global pipeline tracker instance
_pipeline_tracker: Optional[PipelineTracker] = None


def init_pipeline_tracker() -> PipelineTracker:
    """Initialize the global pipeline tracker."""
    global _pipeline_tracker
    _pipeline_tracker = PipelineTracker()
    return _pipeline_tracker


def get_pipeline_tracker() -> PipelineTracker:
    """Get the global pipeline tracker instance."""
    if _pipeline_tracker is None:
        raise RuntimeError("PipelineTracker not initialized. Call init_pipeline_tracker first.")
    return _pipeline_tracker


# Backward compatibility aliases
init_task_manager = init_pipeline_tracker
get_task_manager = get_pipeline_tracker
