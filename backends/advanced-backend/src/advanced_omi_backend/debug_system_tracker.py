"""
Debug System Tracker - Single source for all system monitoring and debugging

This module provides centralized tracking for the audio processing pipeline:
Audio → Transcription → Memory → Action Items

Tracks transactions and highlights issues like "transcription finished but memory creation error"
"""

import asyncio
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import uuid4


class PipelineStage(Enum):
    """Pipeline stages for tracking audio processing flow"""

    AUDIO_RECEIVED = "audio_received"
    TRANSCRIPTION_STARTED = "transcription_started"
    TRANSCRIPTION_COMPLETED = "transcription_completed"
    MEMORY_STARTED = "memory_started"
    MEMORY_COMPLETED = "memory_completed"
    CONVERSATION_ENDED = "conversation_ended"


class TransactionStatus(Enum):
    """Status of a pipeline transaction"""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    STALLED = "stalled"  # Started but no progress in reasonable time


@dataclass
class PipelineEvent:
    """Single event in the pipeline"""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    stage: PipelineStage = PipelineStage.AUDIO_RECEIVED
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Transaction:
    """Complete transaction through the pipeline"""

    user_id: str
    client_id: str
    transaction_id: str = field(default_factory=lambda: str(uuid4()))
    conversation_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    events: List[PipelineEvent] = field(default_factory=list)
    status: TransactionStatus = TransactionStatus.IN_PROGRESS
    current_stage: Optional[PipelineStage] = None

    def add_event(
        self,
        stage: PipelineStage,
        success: bool = True,
        error_message: Optional[str] = None,
        **metadata,
    ):
        """Add an event to this transaction"""
        event = PipelineEvent(
            stage=stage, success=success, error_message=error_message, metadata=metadata
        )
        self.events.append(event)
        self.current_stage = stage
        self.updated_at = datetime.now(UTC)

        if not success:
            self.status = TransactionStatus.FAILED
        elif stage == PipelineStage.CONVERSATION_ENDED and success:
            self.status = TransactionStatus.COMPLETED

    def get_stage_status(self, stage: PipelineStage) -> Optional[bool]:
        """Get success status for a specific stage, None if not reached"""
        for event in reversed(self.events):
            if event.stage == stage:
                return event.success
        return None

    def get_issue_description(self) -> Optional[str]:
        """Get human-readable description of any pipeline issues"""
        if self.status == TransactionStatus.COMPLETED:
            return None

        # Check for specific failure patterns
        transcription_done = self.get_stage_status(PipelineStage.TRANSCRIPTION_COMPLETED)
        memory_done = self.get_stage_status(PipelineStage.MEMORY_COMPLETED)

        if transcription_done and memory_done is False:
            return "Transcription completed but memory creation failed"

        if transcription_done and memory_done is None:
            elapsed = (datetime.now(UTC) - self.updated_at).total_seconds()
            if elapsed > 30:  # 30 seconds without memory processing
                return "Transcription completed but memory processing stalled"

        # Check for other patterns
        for event in self.events:
            if not event.success:
                return f"Failed at {event.stage.value}: {event.error_message or 'Unknown error'}"

        return None


@dataclass
class SystemMetrics:
    """Current system metrics and status"""

    system_start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    total_transactions: int = 0
    active_transactions: int = 0
    completed_transactions: int = 0
    failed_transactions: int = 0
    stalled_transactions: int = 0
    active_websockets: int = 0
    total_audio_chunks_processed: int = 0
    total_transcriptions: int = 0
    total_memories_created: int = 0
    last_activity: Optional[datetime] = None

    def uptime_hours(self) -> float:
        """Get system uptime in hours"""
        return (datetime.now(UTC) - self.system_start_time).total_seconds() / 3600


class DebugSystemTracker:
    """
    Single source for all system monitoring and debugging.

    Thread-safe tracker for the audio processing pipeline with real-time issue detection.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.metrics = SystemMetrics()

        # Transaction tracking
        self.transactions: Dict[str, Transaction] = {}
        self.active_websockets: Set[str] = set()

        # Recent activity for dashboard
        self.recent_transactions = deque(maxlen=100)  # Last 100 transactions
        self.recent_issues = deque(maxlen=50)  # Last 50 issues

        # Per-user tracking
        self.user_activity: Dict[str, datetime] = {}

        # Debug dump directory
        self.debug_dir = Path(os.getenv("DEBUG_DUMP_DIR", "debug_dumps"))
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Background task for stalled transaction detection
        self._monitor_task = None
        self._monitoring = False

    def start_monitoring(self):
        """Start background monitoring for stalled transactions"""
        if self._monitoring:
            return
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_stalled_transactions())

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()

    async def _monitor_stalled_transactions(self):
        """Background task to detect stalled transactions"""
        while self._monitoring:
            try:
                now = datetime.now(UTC)
                with self.lock:
                    for transaction in self.transactions.values():
                        if transaction.status == TransactionStatus.IN_PROGRESS:
                            elapsed = (now - transaction.updated_at).total_seconds()
                            if elapsed > 60:  # 1 minute without progress
                                transaction.status = TransactionStatus.STALLED
                                self.metrics.stalled_transactions += 1
                                self.metrics.active_transactions -= 1

                                issue = f"Transaction {transaction.transaction_id[:8]} stalled after {transaction.current_stage.value if transaction.current_stage else 'unknown stage'}"
                                self.recent_issues.append(
                                    {
                                        "timestamp": now.isoformat(),
                                        "transaction_id": transaction.transaction_id,
                                        "user_id": transaction.user_id,
                                        "issue": issue,
                                    }
                                )

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue monitoring
                pass

    def create_transaction(
        self, user_id: str, client_id: str, conversation_id: Optional[str] = None
    ) -> str:
        """Create a new pipeline transaction"""
        with self.lock:
            transaction = Transaction(
                user_id=user_id, client_id=client_id, conversation_id=conversation_id
            )

            self.transactions[transaction.transaction_id] = transaction
            self.recent_transactions.append(transaction.transaction_id)

            self.metrics.total_transactions += 1
            self.metrics.active_transactions += 1
            self.metrics.last_activity = datetime.now(UTC)
            self.user_activity[user_id] = datetime.now(UTC)

            return transaction.transaction_id

    def track_event(
        self,
        transaction_id: str,
        stage: PipelineStage,
        success: bool = True,
        error_message: Optional[str] = None,
        **metadata,
    ):
        """Track an event in a transaction"""
        with self.lock:
            if transaction_id not in self.transactions:
                return

            transaction = self.transactions[transaction_id]
            transaction.add_event(stage, success, error_message, **metadata)

            # Update metrics based on stage
            if success:
                if stage == PipelineStage.TRANSCRIPTION_COMPLETED:
                    self.metrics.total_transcriptions += 1
                elif stage == PipelineStage.MEMORY_COMPLETED:
                    self.metrics.total_memories_created += 1
                elif stage == PipelineStage.CONVERSATION_ENDED:
                    self.metrics.completed_transactions += 1
                    self.metrics.active_transactions -= 1
            else:
                # Track failure
                if transaction.status == TransactionStatus.FAILED:
                    self.metrics.failed_transactions += 1
                    self.metrics.active_transactions -= 1

                # Add to recent issues
                issue_desc = transaction.get_issue_description()
                if issue_desc:
                    self.recent_issues.append(
                        {
                            "timestamp": datetime.now(UTC).isoformat(),
                            "transaction_id": transaction_id,
                            "user_id": transaction.user_id,
                            "issue": issue_desc,
                        }
                    )

            self.metrics.last_activity = datetime.now(UTC)

    def track_audio_chunk(self, transaction_id: str, chunk_size: int = 0):
        """Track audio chunk processing"""
        with self.lock:
            self.metrics.total_audio_chunks_processed += 1
        self.track_event(
            transaction_id, PipelineStage.AUDIO_RECEIVED, metadata={"chunk_size": chunk_size}
        )

    def track_websocket_connected(self, user_id: str, client_id: str):
        """Track WebSocket connection"""
        with self.lock:
            self.active_websockets.add(client_id)
            self.metrics.active_websockets = len(self.active_websockets)
            self.user_activity[user_id] = datetime.now(UTC)

    def track_websocket_disconnected(self, client_id: str):
        """Track WebSocket disconnection"""
        with self.lock:
            self.active_websockets.discard(client_id)
            self.metrics.active_websockets = len(self.active_websockets)

    def get_dashboard_data(self) -> Dict:
        """Get formatted data for Streamlit dashboard"""
        with self.lock:
            # Update stalled count
            now = datetime.now(UTC)
            stalled_count = 0
            for transaction in self.transactions.values():
                if (
                    transaction.status == TransactionStatus.IN_PROGRESS
                    and (now - transaction.updated_at).total_seconds() > 60
                ):
                    stalled_count += 1

            return {
                "system_metrics": {
                    "uptime_hours": self.metrics.uptime_hours(),
                    "total_transactions": self.metrics.total_transactions,
                    "active_transactions": self.metrics.active_transactions,
                    "completed_transactions": self.metrics.completed_transactions,
                    "failed_transactions": self.metrics.failed_transactions,
                    "stalled_transactions": stalled_count,
                    "active_websockets": self.metrics.active_websockets,
                    "total_audio_chunks": self.metrics.total_audio_chunks_processed,
                    "total_transcriptions": self.metrics.total_transcriptions,
                    "total_memories": self.metrics.total_memories_created,
                    "last_activity": (
                        self.metrics.last_activity.isoformat()
                        if self.metrics.last_activity
                        else None
                    ),
                },
                "recent_transactions": [
                    {
                        "id": t_id[:8],
                        "user_id": (
                            self.transactions[t_id].user_id[-6:]
                            if t_id in self.transactions
                            else "unknown"
                        ),
                        "status": (
                            self.transactions[t_id].status.value
                            if t_id in self.transactions
                            else "unknown"
                        ),
                        "current_stage": (
                            self.transactions[t_id].current_stage.value
                            if t_id in self.transactions
                            and self.transactions[t_id].current_stage is not None
                            else "none"
                        ),
                        "created_at": (
                            self.transactions[t_id].created_at.isoformat()
                            if t_id in self.transactions
                            else "unknown"
                        ),
                        "issue": (
                            self.transactions[t_id].get_issue_description()
                            if t_id in self.transactions
                            else None
                        ),
                    }
                    for t_id in list(self.recent_transactions)[-10:]  # Last 10 transactions
                    if t_id in self.transactions
                ],
                "recent_issues": list(self.recent_issues)[-10:],  # Last 10 issues
                "active_users": len(
                    [
                        uid
                        for uid, last_seen in self.user_activity.items()
                        if (now - last_seen).total_seconds() < 300  # Active in last 5 minutes
                    ]
                ),
            }

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction by ID"""
        with self.lock:
            return self.transactions.get(transaction_id)

    def get_user_transactions(self, user_id: str, limit: int = 10) -> List[Transaction]:
        """Get recent transactions for a user"""
        with self.lock:
            user_transactions = [t for t in self.transactions.values() if t.user_id == user_id]
            user_transactions.sort(key=lambda x: x.created_at, reverse=True)
            return user_transactions[:limit]

    def export_debug_dump(self) -> Path:
        """Export comprehensive debug data to JSON file"""
        with self.lock:
            dump_data = {
                "export_metadata": {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "system_start_time": self.metrics.system_start_time.isoformat(),
                    "uptime_hours": self.metrics.uptime_hours(),
                },
                "system_metrics": self.get_dashboard_data()["system_metrics"],
                "transactions": [
                    {
                        "transaction_id": t.transaction_id,
                        "user_id": t.user_id,
                        "client_id": t.client_id,
                        "conversation_id": t.conversation_id,
                        "created_at": t.created_at.isoformat(),
                        "updated_at": t.updated_at.isoformat(),
                        "status": t.status.value,
                        "current_stage": t.current_stage.value if t.current_stage else None,
                        "issue": t.get_issue_description(),
                        "events": [
                            {
                                "timestamp": e.timestamp.isoformat(),
                                "stage": e.stage.value,
                                "success": e.success,
                                "error_message": e.error_message,
                                "metadata": e.metadata,
                            }
                            for e in t.events
                        ],
                    }
                    for t in self.transactions.values()
                ],
                "recent_issues": list(self.recent_issues),
                "active_websockets": list(self.active_websockets),
                "user_activity": {
                    uid: last_seen.isoformat() for uid, last_seen in self.user_activity.items()
                },
            }

            dump_file = self.debug_dir / f"debug_dump_{int(time.time())}.json"
            with open(dump_file, "w") as f:
                json.dump(dump_data, f, indent=2)

            return dump_file


# Global instance
_debug_tracker: Optional[DebugSystemTracker] = None


def get_debug_tracker() -> DebugSystemTracker:
    """Get the global debug tracker instance"""
    global _debug_tracker
    if _debug_tracker is None:
        _debug_tracker = DebugSystemTracker()
    return _debug_tracker


def init_debug_tracker():
    """Initialize the debug tracker (called at startup)"""
    global _debug_tracker
    _debug_tracker = DebugSystemTracker()
    _debug_tracker.start_monitoring()
    return _debug_tracker


def shutdown_debug_tracker():
    """Shutdown the debug tracker (called at shutdown)"""
    global _debug_tracker
    if _debug_tracker:
        _debug_tracker.stop_monitoring()
