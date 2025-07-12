"""
Queue Status Tracking System for Friend-Lite Backend

This module provides persistent tracking of processing queues and enables
failure recovery by maintaining state across service restarts.
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class QueueStatus(Enum):
    """Processing status for queue items"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    DEAD_LETTER = "dead_letter"

class QueueType(Enum):
    """Types of processing queues"""
    CHUNK = "chunk"
    TRANSCRIPTION = "transcription"
    MEMORY = "memory"
    ACTION_ITEM = "action_item"

@dataclass
class QueueItem:
    """Represents an item in a processing queue"""
    id: str
    queue_type: QueueType
    client_id: str
    user_id: str
    audio_uuid: str
    data: Dict[str, Any]
    status: QueueStatus
    created_at: float
    updated_at: float
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    processing_started_at: Optional[float] = None
    processing_completed_at: Optional[float] = None

class QueueTracker:
    """
    Persistent queue tracking system using SQLite
    
    Tracks all processing items across queues and enables recovery
    from failures and service restarts.
    """
    
    def __init__(self, db_path: str = "queue_tracker.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queue_items (
                    id TEXT PRIMARY KEY,
                    queue_type TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    audio_uuid TEXT NOT NULL,
                    data TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    error_message TEXT,
                    processing_started_at REAL,
                    processing_completed_at REAL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_queue_status 
                ON queue_items(queue_type, status, created_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_client_items 
                ON queue_items(client_id, status, created_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audio_uuid 
                ON queue_items(audio_uuid, status)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queue_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
    
    def add_item(self, item: QueueItem) -> bool:
        """Add a new item to the queue tracking system"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO queue_items 
                    (id, queue_type, client_id, user_id, audio_uuid, data, status, 
                     created_at, updated_at, retry_count, max_retries, error_message,
                     processing_started_at, processing_completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.id,
                    item.queue_type.value,
                    item.client_id,
                    item.user_id,
                    item.audio_uuid,
                    json.dumps(item.data),
                    item.status.value,
                    item.created_at,
                    item.updated_at,
                    item.retry_count,
                    item.max_retries,
                    item.error_message,
                    item.processing_started_at,
                    item.processing_completed_at
                ))
                conn.commit()
                logger.debug(f"Added queue item {item.id} to {item.queue_type.value}")
                return True
        except Exception as e:
            logger.error(f"Failed to add queue item {item.id}: {e}")
            return False
    
    def update_item_status(self, item_id: str, status: QueueStatus, 
                          error_message: Optional[str] = None) -> bool:
        """Update the status of a queue item"""
        try:
            now = time.time()
            with sqlite3.connect(self.db_path) as conn:
                # Get current item for retry count management
                cursor = conn.execute(
                    "SELECT retry_count, max_retries FROM queue_items WHERE id = ?", 
                    (item_id,)
                )
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Queue item {item_id} not found")
                    return False
                
                retry_count, max_retries = row
                
                # Handle retry logic
                if status == QueueStatus.FAILED:
                    if retry_count < max_retries:
                        status = QueueStatus.RETRY
                        retry_count += 1
                    else:
                        status = QueueStatus.DEAD_LETTER
                
                # Set processing timestamps
                processing_started_at = now if status == QueueStatus.PROCESSING else None
                processing_completed_at = now if status in [QueueStatus.COMPLETED, QueueStatus.DEAD_LETTER] else None
                
                conn.execute("""
                    UPDATE queue_items 
                    SET status = ?, updated_at = ?, retry_count = ?, 
                        error_message = ?, processing_started_at = ?, 
                        processing_completed_at = ?
                    WHERE id = ?
                """, (
                    status.value,
                    now,
                    retry_count,
                    error_message,
                    processing_started_at,
                    processing_completed_at,
                    item_id
                ))
                conn.commit()
                
                logger.debug(f"Updated queue item {item_id} to {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update queue item {item_id}: {e}")
            return False
    
    def get_item(self, item_id: str) -> Optional[QueueItem]:
        """Get a specific queue item by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM queue_items WHERE id = ?", 
                    (item_id,)
                )
                row = cursor.fetchone()
                if row:
                    return self._row_to_item(row)
                return None
        except Exception as e:
            logger.error(f"Failed to get queue item {item_id}: {e}")
            return None
    
    def get_pending_items(self, queue_type: QueueType, 
                         client_id: Optional[str] = None,
                         limit: int = 100) -> List[QueueItem]:
        """Get pending items for processing"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if client_id:
                    cursor = conn.execute("""
                        SELECT * FROM queue_items 
                        WHERE queue_type = ? AND client_id = ? AND status IN ('pending', 'retry')
                        ORDER BY created_at ASC
                        LIMIT ?
                    """, (queue_type.value, client_id, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM queue_items 
                        WHERE queue_type = ? AND status IN ('pending', 'retry')
                        ORDER BY created_at ASC
                        LIMIT ?
                    """, (queue_type.value, limit))
                
                return [self._row_to_item(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get pending items: {e}")
            return []
    
    def get_stale_processing_items(self, queue_type: QueueType, 
                                  timeout_seconds: int = 300) -> List[QueueItem]:
        """Get items that have been processing for too long"""
        try:
            cutoff_time = time.time() - timeout_seconds
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM queue_items 
                    WHERE queue_type = ? AND status = 'processing' 
                    AND processing_started_at < ?
                    ORDER BY processing_started_at ASC
                """, (queue_type.value, cutoff_time))
                
                return [self._row_to_item(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get stale processing items: {e}")
            return []
    
    def get_queue_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all queues"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT queue_type, status, COUNT(*) as count
                    FROM queue_items
                    GROUP BY queue_type, status
                    ORDER BY queue_type, status
                """)
                
                stats = {}
                for row in cursor.fetchall():
                    queue_type, status, count = row
                    if queue_type not in stats:
                        stats[queue_type] = {}
                    stats[queue_type][status] = count
                
                return stats
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    def get_client_stats(self, client_id: str) -> Dict[str, int]:
        """Get processing statistics for a specific client"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT status, COUNT(*) as count
                    FROM queue_items
                    WHERE client_id = ?
                    GROUP BY status
                """, (client_id,))
                
                stats = {}
                for row in cursor.fetchall():
                    status, count = row
                    stats[status] = count
                
                return stats
        except Exception as e:
            logger.error(f"Failed to get client stats for {client_id}: {e}")
            return {}
    
    def cleanup_old_items(self, days_old: int = 7) -> int:
        """Remove old completed/failed items"""
        try:
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM queue_items
                    WHERE status IN ('completed', 'dead_letter')
                    AND updated_at < ?
                """, (cutoff_time,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old queue items")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old items: {e}")
            return 0
    
    def get_processing_pipeline_status(self, audio_uuid: str) -> Dict[str, Any]:
        """Get the complete processing status for an audio UUID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT queue_type, status, created_at, updated_at, 
                           retry_count, error_message
                    FROM queue_items
                    WHERE audio_uuid = ?
                    ORDER BY created_at ASC
                """, (audio_uuid,))
                
                pipeline_status = {
                    "audio_uuid": audio_uuid,
                    "stages": {},
                    "overall_status": "unknown",
                    "started_at": None,
                    "completed_at": None,
                    "has_failures": False
                }
                
                all_completed = True
                has_failures = False
                started_at = None
                completed_at = None
                
                for row in cursor.fetchall():
                    queue_type, status, created_at, updated_at, retry_count, error_message = row
                    
                    pipeline_status["stages"][queue_type] = {
                        "status": status,
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "retry_count": retry_count,
                        "error_message": error_message
                    }
                    
                    if started_at is None or created_at < started_at:
                        started_at = created_at
                    
                    if status == "completed":
                        if completed_at is None or updated_at > completed_at:
                            completed_at = updated_at
                    else:
                        all_completed = False
                    
                    if status in ["failed", "dead_letter"]:
                        has_failures = True
                
                pipeline_status["started_at"] = started_at
                pipeline_status["completed_at"] = completed_at if all_completed else None
                pipeline_status["has_failures"] = has_failures
                
                if all_completed:
                    pipeline_status["overall_status"] = "completed"
                elif has_failures:
                    pipeline_status["overall_status"] = "failed"
                else:
                    pipeline_status["overall_status"] = "processing"
                
                return pipeline_status
                
        except Exception as e:
            logger.error(f"Failed to get pipeline status for {audio_uuid}: {e}")
            return {"audio_uuid": audio_uuid, "error": str(e)}
    
    def _row_to_item(self, row: Tuple) -> QueueItem:
        """Convert database row to QueueItem object"""
        return QueueItem(
            id=row[0],
            queue_type=QueueType(row[1]),
            client_id=row[2],
            user_id=row[3],
            audio_uuid=row[4],
            data=json.loads(row[5]),
            status=QueueStatus(row[6]),
            created_at=row[7],
            updated_at=row[8],
            retry_count=row[9],
            max_retries=row[10],
            error_message=row[11],
            processing_started_at=row[12],
            processing_completed_at=row[13]
        )

# Global queue tracker instance
_queue_tracker: Optional[QueueTracker] = None

def get_queue_tracker() -> QueueTracker:
    """Get the global queue tracker instance"""
    global _queue_tracker
    if _queue_tracker is None:
        _queue_tracker = QueueTracker()
    return _queue_tracker

def init_queue_tracker(db_path: str = "queue_tracker.db"):
    """Initialize the global queue tracker"""
    global _queue_tracker
    _queue_tracker = QueueTracker(db_path)
    logger.info(f"Initialized queue tracker with database: {db_path}")

def shutdown_queue_tracker():
    """Shutdown the global queue tracker"""
    global _queue_tracker
    _queue_tracker = None
    logger.info("Shutdown queue tracker")