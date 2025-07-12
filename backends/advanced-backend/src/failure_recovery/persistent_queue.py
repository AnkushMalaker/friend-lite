"""
Persistent Queue System for Friend-Lite Backend

This module provides SQLite-based persistent queues that survive service restarts
and enable reliable message processing with retry mechanisms.
"""

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum

from .queue_tracker import QueueTracker, QueueItem, QueueStatus, QueueType, get_queue_tracker

logger = logging.getLogger(__name__)

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class PersistentMessage:
    """Message in a persistent queue"""
    id: str
    queue_name: str
    payload: Dict[str, Any]
    priority: MessagePriority
    created_at: float
    scheduled_at: float
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    client_id: Optional[str] = None
    user_id: Optional[str] = None
    audio_uuid: Optional[str] = None

class PersistentQueue:
    """
    SQLite-based persistent queue implementation
    
    Features:
    - Survives service restarts
    - Message retry with exponential backoff
    - Priority-based message ordering
    - Dead letter queue for failed messages
    - Atomic operations for reliability
    """
    
    def __init__(self, db_path: str = "persistent_queues.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue_tracker = get_queue_tracker()
        self._init_database()
        self._processing_lock = asyncio.Lock()
    
    def _init_database(self):
        """Initialize the SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    queue_name TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    scheduled_at REAL NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    error_message TEXT,
                    client_id TEXT,
                    user_id TEXT,
                    audio_uuid TEXT,
                    status TEXT DEFAULT 'pending'
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_queue_priority 
                ON messages(queue_name, status, priority DESC, scheduled_at ASC)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_client_messages 
                ON messages(client_id, queue_name, status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scheduled_messages 
                ON messages(scheduled_at, status)
            """)
    
    async def put(self, 
                  queue_name: str, 
                  payload: Dict[str, Any],
                  priority: MessagePriority = MessagePriority.NORMAL,
                  delay_seconds: float = 0,
                  max_retries: int = 3,
                  client_id: Optional[str] = None,
                  user_id: Optional[str] = None,
                  audio_uuid: Optional[str] = None) -> str:
        """Add a message to the queue"""
        
        message_id = str(uuid.uuid4())
        now = time.time()
        scheduled_at = now + delay_seconds
        
        message = PersistentMessage(
            id=message_id,
            queue_name=queue_name,
            payload=payload,
            priority=priority,
            created_at=now,
            scheduled_at=scheduled_at,
            max_retries=max_retries,
            client_id=client_id,
            user_id=user_id,
            audio_uuid=audio_uuid
        )
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO messages 
                    (id, queue_name, payload, priority, created_at, scheduled_at, 
                     retry_count, max_retries, client_id, user_id, audio_uuid, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.id,
                    message.queue_name,
                    json.dumps(message.payload),
                    message.priority.value,
                    message.created_at,
                    message.scheduled_at,
                    message.retry_count,
                    message.max_retries,
                    message.client_id,
                    message.user_id,
                    message.audio_uuid,
                    'pending'
                ))
                conn.commit()
            
            # Track in queue tracker if audio_uuid is provided
            if audio_uuid and queue_name in ['chunk', 'transcription', 'memory', 'action_item']:
                queue_type = QueueType(queue_name.upper())
                queue_item = QueueItem(
                    id=message_id,
                    queue_type=queue_type,
                    client_id=client_id or "",
                    user_id=user_id or "",
                    audio_uuid=audio_uuid,
                    data=payload,
                    status=QueueStatus.PENDING,
                    created_at=now,
                    updated_at=now,
                    max_retries=max_retries
                )
                self.queue_tracker.add_item(queue_item)
            
            logger.debug(f"Added message {message_id} to queue {queue_name}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to add message to queue {queue_name}: {e}")
            raise
    
    async def get(self, 
                  queue_name: str, 
                  timeout: Optional[float] = None) -> Optional[PersistentMessage]:
        """Get the next message from the queue"""
        
        async with self._processing_lock:
            try:
                now = time.time()
                
                with sqlite3.connect(self.db_path) as conn:
                    # Get highest priority message that's ready to process
                    cursor = conn.execute("""
                        SELECT id, queue_name, payload, priority, created_at, scheduled_at,
                               retry_count, max_retries, error_message, client_id, user_id, audio_uuid
                        FROM messages
                        WHERE queue_name = ? AND status = 'pending' AND scheduled_at <= ?
                        ORDER BY priority DESC, scheduled_at ASC
                        LIMIT 1
                    """, (queue_name, now))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    message_id = row[0]
                    
                    # Mark message as processing
                    conn.execute("""
                        UPDATE messages 
                        SET status = 'processing'
                        WHERE id = ?
                    """, (message_id,))
                    
                    conn.commit()
                    
                    # Update queue tracker
                    if row[11]:  # audio_uuid exists
                        self.queue_tracker.update_item_status(message_id, QueueStatus.PROCESSING)
                    
                    # Create message object
                    message = PersistentMessage(
                        id=row[0],
                        queue_name=row[1],
                        payload=json.loads(row[2]),
                        priority=MessagePriority(row[3]),
                        created_at=row[4],
                        scheduled_at=row[5],
                        retry_count=row[6],
                        max_retries=row[7],
                        error_message=row[8],
                        client_id=row[9],
                        user_id=row[10],
                        audio_uuid=row[11]
                    )
                    
                    logger.debug(f"Retrieved message {message_id} from queue {queue_name}")
                    return message
                    
            except Exception as e:
                logger.error(f"Failed to get message from queue {queue_name}: {e}")
                return None
    
    async def ack(self, message_id: str) -> bool:
        """Acknowledge successful processing of a message"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE messages 
                    SET status = 'completed'
                    WHERE id = ?
                """, (message_id,))
                
                if cursor.rowcount == 0:
                    logger.warning(f"Message {message_id} not found for ack")
                    return False
                
                conn.commit()
                
                # Update queue tracker
                self.queue_tracker.update_item_status(message_id, QueueStatus.COMPLETED)
                
                logger.debug(f"Acknowledged message {message_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to ack message {message_id}: {e}")
            return False
    
    async def nack(self, message_id: str, error_message: str = "", 
                   delay_seconds: float = 0) -> bool:
        """Negative acknowledge - retry or move to dead letter queue"""
        try:
            now = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                # Get current message details
                cursor = conn.execute("""
                    SELECT retry_count, max_retries FROM messages WHERE id = ?
                """, (message_id,))
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Message {message_id} not found for nack")
                    return False
                
                retry_count, max_retries = row
                new_retry_count = retry_count + 1
                
                if new_retry_count <= max_retries:
                    # Retry with exponential backoff
                    backoff_delay = min(delay_seconds + (2 ** retry_count), 300)  # Max 5 minutes
                    new_scheduled_at = now + backoff_delay
                    
                    conn.execute("""
                        UPDATE messages 
                        SET status = 'pending', retry_count = ?, error_message = ?, 
                            scheduled_at = ?
                        WHERE id = ?
                    """, (new_retry_count, error_message, new_scheduled_at, message_id))
                    
                    # Update queue tracker
                    self.queue_tracker.update_item_status(message_id, QueueStatus.RETRY, error_message)
                    
                    logger.info(f"Retrying message {message_id} in {backoff_delay}s (attempt {new_retry_count})")
                    
                else:
                    # Move to dead letter queue
                    conn.execute("""
                        UPDATE messages 
                        SET status = 'dead_letter', retry_count = ?, error_message = ?
                        WHERE id = ?
                    """, (new_retry_count, error_message, message_id))
                    
                    # Update queue tracker
                    self.queue_tracker.update_item_status(message_id, QueueStatus.DEAD_LETTER, error_message)
                    
                    logger.warning(f"Message {message_id} moved to dead letter queue after {new_retry_count} attempts")
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to nack message {message_id}: {e}")
            return False
    
    async def get_queue_stats(self, queue_name: str) -> Dict[str, int]:
        """Get statistics for a specific queue"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT status, COUNT(*) as count
                    FROM messages
                    WHERE queue_name = ?
                    GROUP BY status
                """, (queue_name,))
                
                stats = {}
                for row in cursor.fetchall():
                    status, count = row
                    stats[status] = count
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get queue stats for {queue_name}: {e}")
            return {}
    
    async def get_all_queue_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all queues"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT queue_name, status, COUNT(*) as count
                    FROM messages
                    GROUP BY queue_name, status
                    ORDER BY queue_name, status
                """)
                
                stats = {}
                for row in cursor.fetchall():
                    queue_name, status, count = row
                    if queue_name not in stats:
                        stats[queue_name] = {}
                    stats[queue_name][status] = count
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get all queue stats: {e}")
            return {}
    
    async def cleanup_completed_messages(self, hours_old: int = 24) -> int:
        """Clean up old completed messages"""
        try:
            cutoff_time = time.time() - (hours_old * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM messages
                    WHERE status = 'completed' AND created_at < ?
                """, (cutoff_time,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} completed messages older than {hours_old} hours")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup completed messages: {e}")
            return 0
    
    async def get_dead_letter_messages(self, queue_name: str, 
                                      limit: int = 100) -> List[PersistentMessage]:
        """Get messages in dead letter queue"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, queue_name, payload, priority, created_at, scheduled_at,
                           retry_count, max_retries, error_message, client_id, user_id, audio_uuid
                    FROM messages
                    WHERE queue_name = ? AND status = 'dead_letter'
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (queue_name, limit))
                
                messages = []
                for row in cursor.fetchall():
                    message = PersistentMessage(
                        id=row[0],
                        queue_name=row[1],
                        payload=json.loads(row[2]),
                        priority=MessagePriority(row[3]),
                        created_at=row[4],
                        scheduled_at=row[5],
                        retry_count=row[6],
                        max_retries=row[7],
                        error_message=row[8],
                        client_id=row[9],
                        user_id=row[10],
                        audio_uuid=row[11]
                    )
                    messages.append(message)
                
                return messages
                
        except Exception as e:
            logger.error(f"Failed to get dead letter messages for {queue_name}: {e}")
            return []
    
    async def requeue_dead_letter_message(self, message_id: str, 
                                         max_retries: int = 3) -> bool:
        """Requeue a message from dead letter queue"""
        try:
            now = time.time()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE messages 
                    SET status = 'pending', retry_count = 0, max_retries = ?, 
                        scheduled_at = ?, error_message = NULL
                    WHERE id = ? AND status = 'dead_letter'
                """, (max_retries, now, message_id))
                
                if cursor.rowcount == 0:
                    logger.warning(f"Dead letter message {message_id} not found")
                    return False
                
                conn.commit()
                
                # Update queue tracker
                self.queue_tracker.update_item_status(message_id, QueueStatus.PENDING)
                
                logger.info(f"Requeued dead letter message {message_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to requeue dead letter message {message_id}: {e}")
            return False

# Global persistent queue instance
_persistent_queue: Optional[PersistentQueue] = None

def get_persistent_queue() -> PersistentQueue:
    """Get the global persistent queue instance"""
    global _persistent_queue
    if _persistent_queue is None:
        _persistent_queue = PersistentQueue()
    return _persistent_queue

def init_persistent_queue(db_path: str = "persistent_queues.db"):
    """Initialize the global persistent queue"""
    global _persistent_queue
    _persistent_queue = PersistentQueue(db_path)
    logger.info(f"Initialized persistent queue with database: {db_path}")

def shutdown_persistent_queue():
    """Shutdown the global persistent queue"""
    global _persistent_queue
    _persistent_queue = None
    logger.info("Shutdown persistent queue")