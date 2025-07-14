"""
Recovery Manager for Friend-Lite Backend

This module provides automatic recovery mechanisms for failed processing tasks,
service restarts, and system failures.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .persistent_queue import PersistentMessage, PersistentQueue, get_persistent_queue
from .queue_tracker import (
    QueueItem,
    QueueStatus,
    QueueTracker,
    QueueType,
    get_queue_tracker,
)

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Types of recovery actions"""

    RETRY = "retry"
    REQUEUE = "requeue"
    SKIP = "skip"
    ESCALATE = "escalate"


@dataclass
class RecoveryRule:
    """Rule for handling recovery scenarios"""

    queue_type: QueueType
    max_stale_time: int  # seconds
    max_retry_count: int
    action: RecoveryAction
    escalation_callback: Optional[Callable] = None


class RecoveryManager:
    """
    Manages automatic recovery of failed processing tasks

    Features:
    - Detects stale processing tasks
    - Automatically retries failed operations
    - Requeues failed messages
    - Escalates persistent failures
    - Handles service restart recovery
    """

    def __init__(
        self,
        queue_tracker: Optional[QueueTracker] = None,
        persistent_queue: Optional[PersistentQueue] = None,
    ):
        self.queue_tracker = queue_tracker or get_queue_tracker()
        self.persistent_queue = persistent_queue or get_persistent_queue()
        self.recovery_rules: Dict[QueueType, RecoveryRule] = {}
        self.recovery_callbacks: Dict[QueueType, Callable] = {}
        self.running = False
        self.recovery_task: Optional[asyncio.Task] = None
        self.stats = {
            "recoveries_attempted": 0,
            "recoveries_successful": 0,
            "items_requeued": 0,
            "items_escalated": 0,
        }

        # Default recovery rules
        self._init_default_rules()

    def _init_default_rules(self):
        """Initialize default recovery rules"""
        self.recovery_rules = {
            QueueType.CHUNK: RecoveryRule(
                queue_type=QueueType.CHUNK,
                max_stale_time=300,  # 5 minutes
                max_retry_count=3,
                action=RecoveryAction.RETRY,
            ),
            QueueType.TRANSCRIPTION: RecoveryRule(
                queue_type=QueueType.TRANSCRIPTION,
                max_stale_time=600,  # 10 minutes
                max_retry_count=3,
                action=RecoveryAction.RETRY,
            ),
            QueueType.MEMORY: RecoveryRule(
                queue_type=QueueType.MEMORY,
                max_stale_time=900,  # 15 minutes
                max_retry_count=2,
                action=RecoveryAction.REQUEUE,
            ),
            QueueType.ACTION_ITEM: RecoveryRule(
                queue_type=QueueType.ACTION_ITEM,
                max_stale_time=300,  # 5 minutes
                max_retry_count=3,
                action=RecoveryAction.RETRY,
            ),
        }

    def set_recovery_rule(self, rule: RecoveryRule):
        """Set a custom recovery rule for a queue type"""
        self.recovery_rules[rule.queue_type] = rule
        logger.info(f"Set recovery rule for {rule.queue_type.value}: {rule.action.value}")

    def set_recovery_callback(self, queue_type: QueueType, callback: Callable):
        """Set a recovery callback for a specific queue type"""
        self.recovery_callbacks[queue_type] = callback
        logger.info(f"Set recovery callback for {queue_type.value}")

    async def start(self, recovery_interval: int = 30):
        """Start the recovery manager"""
        if self.running:
            logger.warning("Recovery manager already running")
            return

        self.running = True
        self.recovery_task = asyncio.create_task(self._recovery_loop(recovery_interval))
        logger.info(f"Started recovery manager with {recovery_interval}s interval")

    async def stop(self):
        """Stop the recovery manager"""
        if not self.running:
            return

        self.running = False
        if self.recovery_task:
            self.recovery_task.cancel()
            try:
                await self.recovery_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped recovery manager")

    async def _recovery_loop(self, interval: int):
        """Main recovery loop"""
        while self.running:
            try:
                await self._run_recovery_cycle()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery loop: {e}")
                await asyncio.sleep(interval)

    async def _run_recovery_cycle(self):
        """Run one recovery cycle"""
        logger.debug("Running recovery cycle")

        for queue_type, rule in self.recovery_rules.items():
            try:
                # Find stale processing items
                stale_items = self.queue_tracker.get_stale_processing_items(
                    queue_type, rule.max_stale_time
                )

                for item in stale_items:
                    await self._recover_item(item, rule)

                # Check for items that need retry
                retry_items = self.queue_tracker.get_pending_items(queue_type)
                retry_items = [item for item in retry_items if item.status == QueueStatus.RETRY]

                for item in retry_items:
                    await self._process_retry_item(item, rule)

            except Exception as e:
                logger.error(f"Error recovering {queue_type.value} queue: {e}")

    async def _recover_item(self, item: QueueItem, rule: RecoveryRule):
        """Recover a stale processing item"""
        self.stats["recoveries_attempted"] += 1

        logger.warning(f"Recovering stale item {item.id} from {item.queue_type.value} queue")

        try:
            if rule.action == RecoveryAction.RETRY:
                await self._retry_item(item, rule)
            elif rule.action == RecoveryAction.REQUEUE:
                await self._requeue_item(item, rule)
            elif rule.action == RecoveryAction.SKIP:
                await self._skip_item(item, rule)
            elif rule.action == RecoveryAction.ESCALATE:
                await self._escalate_item(item, rule)

            self.stats["recoveries_successful"] += 1

        except Exception as e:
            logger.error(f"Failed to recover item {item.id}: {e}")

    async def _retry_item(self, item: QueueItem, rule: RecoveryRule):
        """Retry a failed item"""
        if item.retry_count >= rule.max_retry_count:
            logger.warning(f"Item {item.id} exceeded max retries, escalating")
            await self._escalate_item(item, rule)
            return

        # Update status to retry
        success = self.queue_tracker.update_item_status(
            item.id, QueueStatus.RETRY, f"Recovered from stale processing state"
        )

        if success:
            logger.info(f"Marked item {item.id} for retry")

            # Trigger recovery callback if available
            if item.queue_type in self.recovery_callbacks:
                try:
                    await self.recovery_callbacks[item.queue_type](item)
                except Exception as e:
                    logger.error(f"Recovery callback failed for {item.id}: {e}")

    async def _requeue_item(self, item: QueueItem, rule: RecoveryRule):
        """Requeue an item to persistent queue"""
        try:
            queue_name = item.queue_type.value.lower()

            # Add to persistent queue
            await self.persistent_queue.put(
                queue_name=queue_name,
                payload=item.data,
                client_id=item.client_id,
                user_id=item.user_id,
                audio_uuid=item.audio_uuid,
                max_retries=rule.max_retry_count,
            )

            # Update status to pending
            self.queue_tracker.update_item_status(
                item.id, QueueStatus.PENDING, "Requeued for processing"
            )

            self.stats["items_requeued"] += 1
            logger.info(f"Requeued item {item.id} to {queue_name}")

        except Exception as e:
            logger.error(f"Failed to requeue item {item.id}: {e}")
            await self._escalate_item(item, rule)

    async def _skip_item(self, item: QueueItem, rule: RecoveryRule):
        """Skip a failed item"""
        success = self.queue_tracker.update_item_status(
            item.id, QueueStatus.FAILED, "Skipped due to recovery rule"
        )

        if success:
            logger.info(f"Skipped item {item.id} from {item.queue_type.value} queue")

    async def _escalate_item(self, item: QueueItem, rule: RecoveryRule):
        """Escalate a persistently failing item"""
        self.stats["items_escalated"] += 1

        # Update status to dead letter
        success = self.queue_tracker.update_item_status(
            item.id, QueueStatus.DEAD_LETTER, "Escalated due to persistent failures"
        )

        if success:
            logger.warning(f"Escalated item {item.id} to dead letter queue")

            # Call escalation callback if available
            if rule.escalation_callback:
                try:
                    await rule.escalation_callback(item)
                except Exception as e:
                    logger.error(f"Escalation callback failed for {item.id}: {e}")

    async def _process_retry_item(self, item: QueueItem, rule: RecoveryRule):
        """Process an item marked for retry"""
        if item.retry_count >= rule.max_retry_count:
            logger.warning(f"Retry item {item.id} exceeded max retries, escalating")
            await self._escalate_item(item, rule)
            return

        # Check if enough time has passed for retry
        retry_delay = min(2**item.retry_count, 300)  # Exponential backoff, max 5 minutes

        if time.time() - item.updated_at < retry_delay:
            return  # Not ready for retry yet

        # Update status to pending for reprocessing
        success = self.queue_tracker.update_item_status(
            item.id, QueueStatus.PENDING, "Ready for retry"
        )

        if success:
            logger.info(f"Marked retry item {item.id} as pending for reprocessing")

            # Trigger recovery callback if available
            if item.queue_type in self.recovery_callbacks:
                try:
                    await self.recovery_callbacks[item.queue_type](item)
                except Exception as e:
                    logger.error(f"Recovery callback failed for {item.id}: {e}")

    async def recover_from_startup(self):
        """Recover processing state after service restart"""
        logger.info("Running startup recovery")

        for queue_type in QueueType:
            try:
                # Find items that were processing when service stopped
                stale_items = self.queue_tracker.get_stale_processing_items(
                    queue_type, 0  # Any processing item is stale on startup
                )

                for item in stale_items:
                    logger.info(f"Recovering processing item {item.id} from startup")

                    # Reset to pending for reprocessing
                    self.queue_tracker.update_item_status(
                        item.id, QueueStatus.PENDING, "Reset from processing state on startup"
                    )

                    # Trigger recovery callback if available
                    if item.queue_type in self.recovery_callbacks:
                        try:
                            await self.recovery_callbacks[item.queue_type](item)
                        except Exception as e:
                            logger.error(f"Startup recovery callback failed for {item.id}: {e}")

            except Exception as e:
                logger.error(f"Error in startup recovery for {queue_type.value}: {e}")

        logger.info("Completed startup recovery")

    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        queue_stats = self.queue_tracker.get_queue_stats()
        persistent_stats = asyncio.create_task(self.persistent_queue.get_all_queue_stats())

        return {
            "recovery_stats": self.stats,
            "queue_stats": queue_stats,
            "running": self.running,
            "recovery_rules": {
                queue_type.value: {
                    "max_stale_time": rule.max_stale_time,
                    "max_retry_count": rule.max_retry_count,
                    "action": rule.action.value,
                }
                for queue_type, rule in self.recovery_rules.items()
            },
        }

    async def manual_recovery(
        self, queue_type: QueueType, item_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Manually trigger recovery for a queue or specific item"""
        result = {
            "queue_type": queue_type.value,
            "item_id": item_id,
            "recovered_items": 0,
            "errors": [],
        }

        try:
            if item_id:
                # Recover specific item
                item = self.queue_tracker.get_item(item_id)
                if item and item.queue_type == queue_type:
                    rule = self.recovery_rules.get(queue_type)
                    if rule:
                        await self._recover_item(item, rule)
                        result["recovered_items"] = 1
                    else:
                        result["errors"].append(f"No recovery rule for {queue_type.value}")
                else:
                    result["errors"].append(f"Item {item_id} not found or wrong queue type")
            else:
                # Recover all items in queue
                rule = self.recovery_rules.get(queue_type)
                if rule:
                    stale_items = self.queue_tracker.get_stale_processing_items(
                        queue_type, rule.max_stale_time
                    )

                    for item in stale_items:
                        try:
                            await self._recover_item(item, rule)
                            result["recovered_items"] += 1
                        except Exception as e:
                            result["errors"].append(f"Failed to recover {item.id}: {str(e)}")
                else:
                    result["errors"].append(f"No recovery rule for {queue_type.value}")

        except Exception as e:
            result["errors"].append(f"Manual recovery failed: {str(e)}")

        return result


# Global recovery manager instance
_recovery_manager: Optional[RecoveryManager] = None


def get_recovery_manager() -> RecoveryManager:
    """Get the global recovery manager instance"""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = RecoveryManager()
    return _recovery_manager


def init_recovery_manager(
    queue_tracker: Optional[QueueTracker] = None, persistent_queue: Optional[PersistentQueue] = None
):
    """Initialize the global recovery manager"""
    global _recovery_manager
    _recovery_manager = RecoveryManager(queue_tracker, persistent_queue)
    logger.info("Initialized recovery manager")


def shutdown_recovery_manager():
    """Shutdown the global recovery manager"""
    global _recovery_manager
    if _recovery_manager:
        asyncio.create_task(_recovery_manager.stop())
    _recovery_manager = None
    logger.info("Shutdown recovery manager")
