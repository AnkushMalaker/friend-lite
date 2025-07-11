"""
Memory Debug API Endpoints

This module provides API endpoints for accessing memory debug information.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any, Optional
import logging

from users import User
from auth import current_active_user
from memory_debug import get_debug_tracker
from memory_config_loader import get_config_loader

# Logger
debug_api_logger = logging.getLogger("memory_debug_api")

# Router for debug endpoints
debug_router = APIRouter(prefix="/api/debug", tags=["Memory Debug"])

@debug_router.get("/memory/stats")
async def get_memory_debug_stats(current_user: User = Depends(current_active_user)):
    """
    Get overall memory debugging statistics.
    Available to all authenticated users.
    """
    try:
        debug_tracker = get_debug_tracker()
        stats = debug_tracker.get_stats()
        return {"stats": stats}
    except Exception as e:
        debug_api_logger.error(f"Error getting memory debug stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get debug stats")

@debug_router.get("/memory/sessions")
async def get_recent_memory_sessions(
    limit: int = 20,
    current_user: User = Depends(current_active_user)
):
    """
    Get recent memory sessions.
    Admins see all sessions, users see only their own.
    """
    try:
        debug_tracker = get_debug_tracker()
        sessions = debug_tracker.get_recent_sessions(limit)
        
        # Filter sessions for non-admin users
        if not current_user.is_superuser:
            sessions = [s for s in sessions if s.get("user_id") == current_user.user_id]
        
        return {"sessions": sessions}
    except Exception as e:
        debug_api_logger.error(f"Error getting recent memory sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get memory sessions")

@debug_router.get("/memory/session/{audio_uuid}")
async def get_memory_session_detail(
    audio_uuid: str,
    current_user: User = Depends(current_active_user)
):
    """
    Get detailed information about a specific memory session.
    Users can only see their own sessions, admins can see all.
    """
    try:
        debug_tracker = get_debug_tracker()
        session = debug_tracker.get_session_summary(audio_uuid)
        
        if not session:
            raise HTTPException(status_code=404, detail="Memory session not found")
        
        # Check permission for non-admin users
        if not current_user.is_superuser and session.get("user_id") != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {"session": session}
    except HTTPException:
        raise
    except Exception as e:
        debug_api_logger.error(f"Error getting memory session detail for {audio_uuid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session detail")

@debug_router.get("/memory/config")
async def get_memory_config(current_user: User = Depends(current_active_user)):
    """
    Get current memory extraction configuration.
    Available to all authenticated users.
    """
    try:
        config_loader = get_config_loader()
        
        return {
            "memory_extraction": config_loader.get_memory_extraction_config(),
            "fact_extraction": config_loader.get_fact_extraction_config(),
            "action_item_extraction": config_loader.get_action_item_extraction_config(),
            "categorization": config_loader.get_categorization_config(),
            "quality_control": config_loader.get_quality_control_config(),
            "processing": config_loader.get_processing_config(),
            "debug": config_loader.get_debug_config()
        }
    except Exception as e:
        debug_api_logger.error(f"Error getting memory config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get memory config")

@debug_router.post("/memory/config/reload")
async def reload_memory_config(current_user: User = Depends(current_active_user)):
    """
    Reload memory extraction configuration from file.
    Available to all authenticated users.
    """
    try:
        config_loader = get_config_loader()
        success = config_loader.reload_config()
        
        if success:
            return {"message": "Configuration reloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload configuration")
    except HTTPException:
        raise
    except Exception as e:
        debug_api_logger.error(f"Error reloading memory config: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload configuration")

@debug_router.get("/memory/config/test")
async def test_memory_config(
    test_text: str = "This is a test conversation about planning a meeting for next week.",
    current_user: User = Depends(current_active_user)
):
    """
    Test memory configuration with sample text.
    Available to all authenticated users.
    """
    try:
        config_loader = get_config_loader()
        
        # Test quality control
        should_skip = config_loader.should_skip_conversation(test_text)
        
        # Test trigger detection
        has_action_triggers = config_loader.has_action_item_triggers(test_text)
        
        # Get relevant prompts
        memory_prompt = config_loader.get_memory_prompt() if config_loader.is_memory_extraction_enabled() else None
        fact_prompt = config_loader.get_fact_prompt() if config_loader.is_fact_extraction_enabled() else None
        action_prompt = config_loader.get_action_item_prompt() if config_loader.is_action_item_extraction_enabled() else None
        
        return {
            "test_text": test_text,
            "should_skip": should_skip,
            "has_action_triggers": has_action_triggers,
            "memory_extraction_enabled": config_loader.is_memory_extraction_enabled(),
            "fact_extraction_enabled": config_loader.is_fact_extraction_enabled(),
            "action_item_extraction_enabled": config_loader.is_action_item_extraction_enabled(),
            "categorization_enabled": config_loader.is_categorization_enabled(),
            "prompts": {
                "memory": memory_prompt,
                "fact": fact_prompt,
                "action_item": action_prompt
            }
        }
    except Exception as e:
        debug_api_logger.error(f"Error testing memory config: {e}")
        raise HTTPException(status_code=500, detail="Failed to test configuration")

@debug_router.get("/memory/pipeline/{audio_uuid}")
async def get_memory_pipeline_trace(
    audio_uuid: str,
    current_user: User = Depends(current_active_user)
):
    """
    Get a detailed trace of the memory processing pipeline for a specific audio session.
    Shows transcript -> memory conversion flow.
    """
    try:
        debug_tracker = get_debug_tracker()
        session = debug_tracker.get_session_summary(audio_uuid)
        
        if not session:
            raise HTTPException(status_code=404, detail="Memory session not found")
        
        # Check permission for non-admin users
        if not current_user.is_superuser and session.get("user_id") != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Build pipeline trace
        pipeline_trace = {
            "audio_uuid": audio_uuid,
            "session_info": {
                "client_id": session.get("client_id"),
                "user_id": session.get("user_id"),
                "user_email": session.get("user_email"),
                "session_start": session.get("session_start_time"),
                "session_end": session.get("session_end_time"),
                "processing_success": session.get("memory_processing_success"),
                "processing_error": session.get("memory_processing_error")
            },
            "input": {
                "transcript_segments": session.get("transcript_segments", []),
                "full_conversation": session.get("full_conversation", ""),
                "transcript_count": session.get("transcript_count", 0),
                "conversation_length": len(session.get("full_conversation", ""))
            },
            "processing": {
                "attempts": session.get("extraction_attempts", []),
                "processing_time": None,
                "success": session.get("memory_processing_success")
            },
            "output": {
                "memories": session.get("memory_extractions", []),
                "memory_count": len(session.get("memory_extractions", []))
            }
        }
        
        # Calculate processing time
        if session.get("memory_processing_started") and session.get("memory_processing_completed"):
            processing_time = session.get("memory_processing_completed") - session.get("memory_processing_started")
            pipeline_trace["processing"]["processing_time"] = processing_time
        
        return {"pipeline_trace": pipeline_trace}
        
    except HTTPException:
        raise
    except Exception as e:
        debug_api_logger.error(f"Error getting memory pipeline trace for {audio_uuid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pipeline trace")

@debug_router.get("/memory/analysis")
async def get_memory_analysis(
    days: int = 7,
    current_user: User = Depends(current_active_user)
):
    """
    Get analysis of memory processing over the last N days.
    Admins see all data, users see only their own.
    """
    try:
        debug_tracker = get_debug_tracker()
        
        # Get recent sessions
        recent_sessions = debug_tracker.get_recent_sessions(limit=1000)
        
        # Filter for time period and user permissions
        import time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        filtered_sessions = []
        for session in recent_sessions:
            if session.get("session_start_time", 0) >= cutoff_time:
                if current_user.is_superuser or session.get("user_id") == current_user.user_id:
                    filtered_sessions.append(session)
        
        # Calculate analysis
        total_sessions = len(filtered_sessions)
        successful_sessions = sum(1 for s in filtered_sessions if s.get("memory_processing_success"))
        failed_sessions = total_sessions - successful_sessions
        
        success_rate = (successful_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        # Group by user
        user_stats = {}
        for session in filtered_sessions:
            user_id = session.get("user_id", "unknown")
            if user_id not in user_stats:
                user_stats[user_id] = {
                    "user_id": user_id,
                    "user_email": session.get("user_email", "unknown"),
                    "total_sessions": 0,
                    "successful_sessions": 0,
                    "failed_sessions": 0,
                    "total_transcripts": 0
                }
            
            user_stats[user_id]["total_sessions"] += 1
            user_stats[user_id]["total_transcripts"] += session.get("transcript_count", 0)
            
            if session.get("memory_processing_success"):
                user_stats[user_id]["successful_sessions"] += 1
            else:
                user_stats[user_id]["failed_sessions"] += 1
        
        # Calculate success rates for each user
        for user_data in user_stats.values():
            total = user_data["total_sessions"]
            user_data["success_rate"] = (user_data["successful_sessions"] / total * 100) if total > 0 else 0
        
        return {
            "analysis": {
                "period_days": days,
                "total_sessions": total_sessions,
                "successful_sessions": successful_sessions,
                "failed_sessions": failed_sessions,
                "success_rate": success_rate,
                "user_stats": list(user_stats.values())
            }
        }
        
    except Exception as e:
        debug_api_logger.error(f"Error getting memory analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to get memory analysis")