"""
Memory Debug Tracking System

This module provides detailed tracking of the transcript -> memories conversion process
to help debug and understand what memories are being created from which transcripts.
"""

import sqlite3
import json
import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

# Logger for memory debugging
debug_logger = logging.getLogger("memory_debug")

class MemoryDebugTracker:
    """
    Tracks the transcript -> memories conversion process for debugging purposes.
    
    SQLite tables:
    - memory_sessions: High-level session info (audio_uuid, client_id, user_id, etc.)
    - transcript_segments: Individual transcript segments within a session
    - memory_extractions: Memories extracted from transcripts
    - extraction_attempts: Log of all extraction attempts (success/failure)
    """
    
    def __init__(self, db_path: str = "/app/debug/memory_debug.db"):
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_database()
    
    def _ensure_db_directory(self):
        """Ensure the debug directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Memory sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_uuid TEXT UNIQUE NOT NULL,
                    client_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    user_email TEXT NOT NULL,
                    session_start_time REAL NOT NULL,
                    session_end_time REAL,
                    transcript_count INTEGER DEFAULT 0,
                    full_conversation TEXT,
                    memory_processing_started REAL,
                    memory_processing_completed REAL,
                    memory_processing_success BOOLEAN,
                    memory_processing_error TEXT,
                    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Individual transcript segments
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcript_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    audio_uuid TEXT NOT NULL,
                    segment_order INTEGER NOT NULL,
                    speaker TEXT,
                    transcript_text TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    transcription_method TEXT,
                    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (session_id) REFERENCES memory_sessions (id)
                )
            """)
            
            # Memory extractions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_extractions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    audio_uuid TEXT NOT NULL,
                    mem0_memory_id TEXT UNIQUE,
                    memory_text TEXT NOT NULL,
                    memory_type TEXT DEFAULT 'general',
                    extraction_prompt TEXT,
                    llm_response TEXT,
                    metadata_json TEXT,
                    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (session_id) REFERENCES memory_sessions (id)
                )
            """)
            
            # Extraction attempts (for debugging failures)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extraction_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    audio_uuid TEXT NOT NULL,
                    attempt_type TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    processing_time_ms REAL,
                    transcript_length INTEGER,
                    prompt_used TEXT,
                    llm_model TEXT,
                    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (session_id) REFERENCES memory_sessions (id)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_audio_uuid ON memory_sessions(audio_uuid)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON memory_sessions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_segments_session_id ON transcript_segments(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_extractions_session_id ON memory_extractions(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_session_id ON extraction_attempts(session_id)")
            
            conn.commit()
    
    def start_memory_session(self, audio_uuid: str, client_id: str, user_id: str, user_email: str) -> int:
        """
        Start tracking a new memory session.
        
        Returns:
            Session ID for subsequent tracking calls
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            session_start_time = time.time()
            
            cursor.execute("""
                INSERT INTO memory_sessions 
                (audio_uuid, client_id, user_id, user_email, session_start_time)
                VALUES (?, ?, ?, ?, ?)
            """, (audio_uuid, client_id, user_id, user_email, session_start_time))
            
            session_id = cursor.lastrowid
            conn.commit()
            
            debug_logger.info(f"Started memory session {session_id} for {audio_uuid} (user: {user_email})")
            return session_id
    
    def add_transcript_segment(self, session_id: int, audio_uuid: str, segment_order: int, 
                              transcript_text: str, speaker: str = None, 
                              transcription_method: str = None):
        """Add a transcript segment to the session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO transcript_segments 
                (session_id, audio_uuid, segment_order, speaker, transcript_text, timestamp, transcription_method)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, audio_uuid, segment_order, speaker, transcript_text, time.time(), transcription_method))
            
            # Update transcript count in session
            cursor.execute("""
                UPDATE memory_sessions 
                SET transcript_count = transcript_count + 1 
                WHERE id = ?
            """, (session_id,))
            
            conn.commit()
            
            debug_logger.debug(f"Added transcript segment {segment_order} to session {session_id}: {transcript_text[:50]}...")
    
    def update_full_conversation(self, session_id: int, full_conversation: str):
        """Update the full conversation text for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE memory_sessions 
                SET full_conversation = ?, session_end_time = ?
                WHERE id = ?
            """, (full_conversation, time.time(), session_id))
            
            conn.commit()
            
            debug_logger.info(f"Updated full conversation for session {session_id} ({len(full_conversation)} chars)")
    
    def start_memory_processing(self, session_id: int):
        """Mark the start of memory processing for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE memory_sessions 
                SET memory_processing_started = ?
                WHERE id = ?
            """, (time.time(), session_id))
            
            conn.commit()
            
            debug_logger.info(f"Started memory processing for session {session_id}")
    
    def complete_memory_processing(self, session_id: int, success: bool, error_message: str = None):
        """Mark the completion of memory processing for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE memory_sessions 
                SET memory_processing_completed = ?, memory_processing_success = ?, memory_processing_error = ?
                WHERE id = ?
            """, (time.time(), success, error_message, session_id))
            
            conn.commit()
            
            status = "successfully" if success else f"with error: {error_message}"
            debug_logger.info(f"Completed memory processing for session {session_id} {status}")
    
    def add_memory_extraction(self, session_id: int, audio_uuid: str, mem0_memory_id: str,
                             memory_text: str, memory_type: str = "general",
                             extraction_prompt: str = None, llm_response: str = None,
                             metadata: Dict[str, Any] = None):
        """Record a successful memory extraction."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                INSERT INTO memory_extractions 
                (session_id, audio_uuid, mem0_memory_id, memory_text, memory_type, 
                 extraction_prompt, llm_response, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, audio_uuid, mem0_memory_id, memory_text, memory_type,
                  extraction_prompt, llm_response, metadata_json))
            
            conn.commit()
            
            debug_logger.info(f"Recorded memory extraction for session {session_id}: {memory_text[:50]}...")
    
    def add_extraction_attempt(self, session_id: int, audio_uuid: str, attempt_type: str,
                              success: bool, error_message: str = None, processing_time_ms: float = None,
                              transcript_length: int = None, prompt_used: str = None,
                              llm_model: str = None):
        """Record an extraction attempt (success or failure)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO extraction_attempts 
                (session_id, audio_uuid, attempt_type, success, error_message, processing_time_ms,
                 transcript_length, prompt_used, llm_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, audio_uuid, attempt_type, success, error_message, processing_time_ms,
                  transcript_length, prompt_used, llm_model))
            
            conn.commit()
            
            status = "succeeded" if success else f"failed: {error_message}"
            debug_logger.debug(f"Recorded {attempt_type} attempt for session {session_id}: {status}")
    
    def get_session_summary(self, audio_uuid: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a memory session by audio_uuid."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute("""
                SELECT id, audio_uuid, client_id, user_id, user_email, session_start_time,
                       session_end_time, transcript_count, full_conversation,
                       memory_processing_started, memory_processing_completed,
                       memory_processing_success, memory_processing_error
                FROM memory_sessions WHERE audio_uuid = ?
            """, (audio_uuid,))
            
            session_row = cursor.fetchone()
            if not session_row:
                return None
            
            session_id = session_row[0]
            
            # Get transcript segments
            cursor.execute("""
                SELECT segment_order, speaker, transcript_text, timestamp, transcription_method
                FROM transcript_segments WHERE session_id = ?
                ORDER BY segment_order
            """, (session_id,))
            
            segments = []
            for row in cursor.fetchall():
                segments.append({
                    "order": row[0],
                    "speaker": row[1],
                    "text": row[2],
                    "timestamp": row[3],
                    "method": row[4]
                })
            
            # Get memory extractions
            cursor.execute("""
                SELECT mem0_memory_id, memory_text, memory_type, extraction_prompt,
                       llm_response, metadata_json
                FROM memory_extractions WHERE session_id = ?
            """, (session_id,))
            
            extractions = []
            for row in cursor.fetchall():
                metadata = json.loads(row[5]) if row[5] else {}
                extractions.append({
                    "mem0_id": row[0],
                    "text": row[1],
                    "type": row[2],
                    "prompt": row[3],
                    "llm_response": row[4],
                    "metadata": metadata
                })
            
            # Get extraction attempts
            cursor.execute("""
                SELECT attempt_type, success, error_message, processing_time_ms,
                       transcript_length, prompt_used, llm_model
                FROM extraction_attempts WHERE session_id = ?
            """, (session_id,))
            
            attempts = []
            for row in cursor.fetchall():
                attempts.append({
                    "type": row[0],
                    "success": row[1],
                    "error": row[2],
                    "processing_time_ms": row[3],
                    "transcript_length": row[4],
                    "prompt": row[5],
                    "model": row[6]
                })
            
            return {
                "session_id": session_id,
                "audio_uuid": session_row[1],
                "client_id": session_row[2],
                "user_id": session_row[3],
                "user_email": session_row[4],
                "session_start_time": session_row[5],
                "session_end_time": session_row[6],
                "transcript_count": session_row[7],
                "full_conversation": session_row[8],
                "memory_processing_started": session_row[9],
                "memory_processing_completed": session_row[10],
                "memory_processing_success": session_row[11],
                "memory_processing_error": session_row[12],
                "transcript_segments": segments,
                "memory_extractions": extractions,
                "extraction_attempts": attempts
            }
    
    def get_recent_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent memory sessions with basic info."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT audio_uuid, client_id, user_id, user_email, session_start_time,
                       transcript_count, memory_processing_success, memory_processing_error
                FROM memory_sessions 
                ORDER BY session_start_time DESC 
                LIMIT ?
            """, (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    "audio_uuid": row[0],
                    "client_id": row[1],
                    "user_id": row[2],
                    "user_email": row[3],
                    "session_start_time": row[4],
                    "transcript_count": row[5],
                    "memory_processing_success": row[6],
                    "memory_processing_error": row[7]
                })
            
            return sessions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall memory debugging statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total sessions
            cursor.execute("SELECT COUNT(*) FROM memory_sessions")
            total_sessions = cursor.fetchone()[0]
            
            # Successful memory processing
            cursor.execute("SELECT COUNT(*) FROM memory_sessions WHERE memory_processing_success = 1")
            successful_sessions = cursor.fetchone()[0]
            
            # Failed memory processing
            cursor.execute("SELECT COUNT(*) FROM memory_sessions WHERE memory_processing_success = 0")
            failed_sessions = cursor.fetchone()[0]
            
            # Total transcripts
            cursor.execute("SELECT COUNT(*) FROM transcript_segments")
            total_transcripts = cursor.fetchone()[0]
            
            # Total memories extracted
            cursor.execute("SELECT COUNT(*) FROM memory_extractions")
            total_memories = cursor.fetchone()[0]
            
            # Average processing time
            cursor.execute("""
                SELECT AVG(memory_processing_completed - memory_processing_started) 
                FROM memory_sessions 
                WHERE memory_processing_completed IS NOT NULL 
                AND memory_processing_started IS NOT NULL
            """)
            avg_processing_time = cursor.fetchone()[0]
            
            return {
                "total_sessions": total_sessions,
                "successful_sessions": successful_sessions,
                "failed_sessions": failed_sessions,
                "success_rate": (successful_sessions / total_sessions * 100) if total_sessions > 0 else 0,
                "total_transcripts": total_transcripts,
                "total_memories": total_memories,
                "avg_processing_time_seconds": avg_processing_time,
                "memories_per_session": (total_memories / total_sessions) if total_sessions > 0 else 0
            }


# Global instance
_debug_tracker = None

def get_debug_tracker() -> MemoryDebugTracker:
    """Get the global debug tracker instance."""
    global _debug_tracker
    if _debug_tracker is None:
        _debug_tracker = MemoryDebugTracker()
    return _debug_tracker