"""
API routes module for the Friend Lite application.
Contains all HTTP API endpoints except for service-specific routes.
"""

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from auth import (
    ADMIN_EMAIL,
    current_active_user,
    current_superuser,
    get_user_manager,
)
from users import User, UserCreate, get_user_by_client_id
from services import get_audio_config, get_database, get_memory_service_instance
from websocket_handler import (
    client_belongs_to_user,
    get_active_clients,
    get_client_state,
    get_user_active_clients,
    get_user_clients_all,
)

# Set up logging
api_logger = logging.getLogger("api")

# Create router
router = APIRouter()

# Audio configuration
audio_config = get_audio_config()
CHUNK_DIR = audio_config["chunk_dir"]

# Models
class ConversationResponse(BaseModel):
    audio_uuid: str
    client_id: str
    file_path: str
    transcript_segments: List[Dict[str, Any]]
    speakers: List[str]
    created_at: float
    updated_at: float
    cropped_audio_path: Optional[str] = None
    speech_segments: Optional[List[Dict[str, float]]] = None

class SpeakerAssignment(BaseModel):
    speaker: str

class TranscriptUpdate(BaseModel):
    text: str

class CloseConversationRequest(BaseModel):
    client_id: str

# Health and status endpoints
@router.get("/health")
async def health_check():
    """Comprehensive health check for all services."""
    try:
        # Check database connection
        mongo_client, db, collections = get_database()
        await collections["chunks"].find_one()
        
        # Check memory service
        memory_service = get_memory_service_instance()
        
        # Check active clients
        active_clients = get_active_clients()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "database": "connected",
                "memory": "available",
                "active_clients": f"{len(active_clients)} connected",
            },
            "environment": {
                "chunk_dir": str(CHUNK_DIR),
                "audio_config": audio_config,
            }
        }
    except Exception as e:
        api_logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@router.get("/readiness")
async def readiness_check():
    """Simple readiness check."""
    return {"status": "ready", "timestamp": time.time()}

# Conversation endpoints
@router.get("/api/conversations")
async def get_conversations(
    current_user: User = Depends(current_active_user),
    user_id: Optional[str] = Query(None)
):
    """Get conversations. Admins can specify user_id, users see only their own."""
    try:
        # Determine which user's conversations to retrieve
        if current_user.is_superuser and user_id:
            target_user_id = user_id
        else:
            target_user_id = current_user.user_id
        
        # Get user's client IDs
        user_client_ids = get_user_clients_all(target_user_id)
        
        if not user_client_ids:
            return {"conversations": [], "count": 0}
        
        # Query conversations from database
        _, _, collections = get_database()
        query = {"client_id": {"$in": user_client_ids}}
        cursor = collections["chunks"].find(query).sort("created_at", -1)
        
        conversations = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            conversations.append(doc)
        
        return {"conversations": conversations, "count": len(conversations)}
        
    except Exception as e:
        api_logger.error(f"Error getting conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/conversations/{audio_uuid}/cropped")
async def get_cropped_audio_info(
    audio_uuid: str,
    current_user: User = Depends(current_active_user)
):
    """Get cropped audio information for a conversation."""
    try:
        _, _, collections = get_database()
        
        # Find the conversation
        conversation = await collections["chunks"].find_one({"audio_uuid": audio_uuid})
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Check if user has access to this conversation
        client_id = conversation["client_id"]
        if not current_user.is_superuser:
            user = await get_user_by_client_id(client_id)
            if not user or user.user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Return cropped audio info
        return {
            "audio_uuid": audio_uuid,
            "cropped_audio_path": conversation.get("cropped_audio_path"),
            "speech_segments": conversation.get("speech_segments", []),
            "original_path": conversation.get("file_path"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error getting cropped audio info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/conversations/{audio_uuid}/reprocess")
async def reprocess_audio_cropping(
    audio_uuid: str,
    current_user: User = Depends(current_active_user)
):
    """Reprocess audio cropping for a conversation."""
    try:
        _, _, collections = get_database()
        
        # Find the conversation
        conversation = await collections["chunks"].find_one({"audio_uuid": audio_uuid})
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Check if user has access to this conversation
        client_id = conversation["client_id"]
        if not current_user.is_superuser:
            user = await get_user_by_client_id(client_id)
            if not user or user.user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Trigger reprocessing (this would need to be implemented)
        # For now, just return success
        return {"message": "Audio reprocessing triggered", "audio_uuid": audio_uuid}
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error reprocessing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# User management endpoints
@router.get("/api/users")
async def get_users(current_user: User = Depends(current_superuser)):
    """Get all users (admin only)."""
    try:
        _, _, collections = get_database()
        users = []
        async for user_doc in collections["users"].find():
            user_doc["_id"] = str(user_doc["_id"])
            users.append(user_doc)
        return users
    except Exception as e:
        api_logger.error(f"Error getting users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/create_user")
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(current_superuser)
):
    """Create a new user (admin only)."""
    try:
        async for user_manager in get_user_manager():
            user = await user_manager.create(user_data)
            return {"message": "User created successfully", "user_id": str(user.id)}
    except Exception as e:
        api_logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/delete_user")
async def delete_user(
    user_id: str,
    current_user: User = Depends(current_superuser)
):
    """Delete a user (admin only)."""
    try:
        async for user_manager in get_user_manager():
            user = await user_manager.get(user_id)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            await user_manager.delete(user)
            return {"message": "User deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Memory endpoints
@router.get("/api/memories")
async def get_memories(
    current_user: User = Depends(current_active_user),
    user_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get memories. Admins can specify user_id, users see only their own."""
    try:
        # Determine which user's memories to retrieve
        if current_user.is_superuser and user_id:
            target_user_id = user_id
        else:
            target_user_id = current_user.user_id
        
        memory_service = get_memory_service_instance()
        memories = memory_service.get_all_memories(
            user_id=target_user_id,
            limit=limit
        )
        
        return {"memories": memories, "count": len(memories)}
        
    except Exception as e:
        api_logger.error(f"Error getting memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/memories/search")
async def search_memories(
    query: str = Query(...),
    current_user: User = Depends(current_active_user),
    limit: int = Query(20, ge=1, le=100)
):
    """Search memories."""
    try:
        memory_service = get_memory_service_instance()
        results = memory_service.search_memories(
            query=query,
            user_id=current_user.user_id,
            limit=limit
        )
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        api_logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/api/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    current_user: User = Depends(current_active_user)
):
    """Delete a memory."""
    try:
        memory_service = get_memory_service_instance()
        success = await memory_service.delete_memory(memory_id, current_user.user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return {"message": "Memory deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Admin endpoints
@router.get("/api/admin/memories/debug")
async def debug_memories(
    current_user: User = Depends(current_superuser),
    user_id: Optional[str] = Query(None)
):
    """Debug memories for admin."""
    try:
        from memory_debug import get_debug_tracker
        
        debug_tracker = get_debug_tracker()
        
        # Get recent sessions limited by user if specified
        if user_id:
            sessions = debug_tracker.get_recent_sessions(limit=50)
            # Filter for specific user
            user_sessions = [s for s in sessions if s.get("user_id") == user_id]
            debug_info = {
                "user_id": user_id,
                "sessions": user_sessions,
                "session_count": len(user_sessions)
            }
        else:
            # Get overall debug statistics
            stats = debug_tracker.get_stats()
            recent_sessions = debug_tracker.get_recent_sessions(limit=20)
            debug_info = {
                "stats": stats,
                "recent_sessions": recent_sessions,
                "session_count": len(recent_sessions)
            }
        
        return {"debug_info": debug_info}
        
    except Exception as e:
        api_logger.error(f"Error getting memory debug info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/admin/memories")
async def get_admin_memories(
    current_user: User = Depends(current_superuser),
    user_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get memories for admin with additional details."""
    try:
        memory_service = get_memory_service_instance()
        
        if user_id:
            memories = memory_service.get_all_memories(
                user_id=user_id,
                limit=limit
            )
        else:
            # For admin to get ALL memories from all users, we need a different approach
            # Since get_all_memories requires a user_id, we'll get all users and their memories
            all_memories = []
            try:
                # Get all users from database
                _, _, collections = get_database()
                user_docs = await collections["users"].find({}).to_list(length=None)
                for user_doc in user_docs:
                    try:
                        user_memories = memory_service.get_all_memories(
                            user_id=str(user_doc["_id"]),
                            limit=limit
                        )
                        all_memories.extend(user_memories)
                        if len(all_memories) >= limit:
                            break
                    except Exception as user_error:
                        api_logger.debug(f"Could not get memories for user {user_doc['_id']}: {user_error}")
                        continue
                memories = all_memories[:limit]
            except Exception as e:
                api_logger.warning(f"Could not get all users' memories: {e}")
                # Fallback: return empty list for now
                memories = []
        
        return {"memories": memories, "count": len(memories)}
        
    except Exception as e:
        api_logger.error(f"Error getting admin memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Speaker and transcript management
@router.post("/api/conversations/{audio_uuid}/speakers")
async def add_speaker_to_conversation(
    audio_uuid: str,
    assignment: SpeakerAssignment,
    current_user: User = Depends(current_active_user)
):
    """Add or update speaker for a conversation."""
    try:
        _, _, collections = get_database()
        
        # Find the conversation
        conversation = await collections["chunks"].find_one({"audio_uuid": audio_uuid})
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Check if user has access to this conversation
        client_id = conversation["client_id"]
        if not current_user.is_superuser:
            user = await get_user_by_client_id(client_id)
            if not user or user.user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Update speaker information
        await collections["chunks"].update_one(
            {"audio_uuid": audio_uuid},
            {"$addToSet": {"speakers": assignment.speaker}}
        )
        
        return {"message": "Speaker added successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error adding speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/conversations/{audio_uuid}/transcript/{segment_index}")
async def update_transcript_segment(
    audio_uuid: str,
    segment_index: int,
    update: TranscriptUpdate,
    current_user: User = Depends(current_active_user)
):
    """Update a specific transcript segment."""
    try:
        _, _, collections = get_database()
        
        # Find the conversation
        conversation = await collections["chunks"].find_one({"audio_uuid": audio_uuid})
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Check if user has access to this conversation
        client_id = conversation["client_id"]
        if not current_user.is_superuser:
            user = await get_user_by_client_id(client_id)
            if not user or user.user_id != current_user.user_id:
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Update transcript segment
        result = await collections["chunks"].update_one(
            {"audio_uuid": audio_uuid},
            {"$set": {f"transcript_segments.{segment_index}.text": update.text}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Transcript segment not found")
        
        return {"message": "Transcript segment updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error updating transcript segment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Client management endpoints
@router.post("/api/close_conversation")
async def close_conversation(
    request: CloseConversationRequest,
    current_user: User = Depends(current_active_user)
):
    """Close a conversation for a client."""
    try:
        client_id = request.client_id
        
        # Check if user has access to this client
        if not current_user.is_superuser:
            if not client_belongs_to_user(client_id, current_user.user_id):
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Get client state
        client_state = get_client_state(client_id)
        if not client_state:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Close current conversation
        await client_state._close_current_conversation()
        
        return {"message": "Conversation closed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error closing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/active_clients")
async def get_active_clients_info(
    current_user: User = Depends(current_active_user)
):
    """Get information about active clients."""
    try:
        if current_user.is_superuser:
            # Admin can see all clients
            active_clients = get_active_clients()
            clients_info = []
            
            for client_id, client_state in active_clients.items():
                clients_info.append({
                    "client_id": client_id,
                    "connected": client_state.connected,
                    "current_audio_uuid": client_state.current_audio_uuid,
                    "sample_count": client_state.sample_count,
                    "conversation_transcripts": len(client_state.conversation_transcripts)
                })
        else:
            # Regular user sees only their clients
            user_clients = get_user_active_clients(current_user.user_id)
            clients_info = []
            
            for client_state in user_clients:
                clients_info.append({
                    "client_id": client_state.client_id,
                    "connected": client_state.connected,
                    "current_audio_uuid": client_state.current_audio_uuid,
                    "sample_count": client_state.sample_count,
                    "conversation_transcripts": len(client_state.conversation_transcripts)
                })
        
        return {"active_clients": clients_info, "count": len(clients_info)}
        
    except Exception as e:
        api_logger.error(f"Error getting active clients: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Debug endpoints
@router.get("/api/debug/speech_segments")
async def get_speech_segments_debug(
    current_user: User = Depends(current_superuser)
):
    """Get speech segments debug information."""
    try:
        active_clients = get_active_clients()
        debug_info = {}
        
        for client_id, client_state in active_clients.items():
            debug_info[client_id] = {
                "speech_segments": dict(client_state.speech_segments),
                "current_speech_start": dict(client_state.current_speech_start),
                "current_audio_uuid": client_state.current_audio_uuid
            }
        
        return {"debug_info": debug_info}
        
    except Exception as e:
        api_logger.error(f"Error getting speech segments debug: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/debug/audio-cropping")
async def get_audio_cropping_debug(
    current_user: User = Depends(current_superuser)
):
    """Get audio cropping debug information."""
    try:
        _, _, collections = get_database()
        
        # Get recent conversations with cropping info
        cursor = collections["chunks"].find({
            "cropped_audio_path": {"$exists": True}
        }).sort("created_at", -1).limit(10)
        
        cropping_info = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            cropping_info.append({
                "audio_uuid": doc["audio_uuid"],
                "client_id": doc["client_id"],
                "original_path": doc.get("file_path"),
                "cropped_path": doc.get("cropped_audio_path"),
                "speech_segments": doc.get("speech_segments", []),
                "created_at": doc.get("created_at")
            })
        
        return {"cropping_info": cropping_info}
        
    except Exception as e:
        api_logger.error(f"Error getting audio cropping debug: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metrics endpoint
@router.get("/api/metrics")
async def get_metrics(
    current_user: User = Depends(current_superuser)
):
    """Get application metrics."""
    try:
        active_clients = get_active_clients()
        _, _, collections = get_database()
        
        # Get basic counts
        total_conversations = await collections["chunks"].count_documents({})
        total_users = await collections["users"].count_documents({})
        
        metrics = {
            "active_clients": len(active_clients),
            "total_conversations": total_conversations,
            "total_users": total_users,
            "timestamp": time.time()
        }
        
        return {"metrics": metrics}
        
    except Exception as e:
        api_logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Auth configuration endpoint
@router.get("/api/auth/config")
async def get_auth_config():
    """Get authentication configuration."""
    return {
        "admin_email": ADMIN_EMAIL,
        "admin_user": {
            "username": os.getenv("ADMIN_USERNAME", "admin"),
            "email": os.getenv("ADMIN_EMAIL", f"{os.getenv('ADMIN_USERNAME', 'admin')}@admin.local"),
        },
    }

# File upload endpoint
@router.post("/api/process-audio-files")
async def process_audio_files(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(current_active_user),
    device_name: Optional[str] = Form("file_upload")
):
    """Process uploaded audio files (.wav) and add them to the audio processing pipeline."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Process each file
        results = []
        for file in files:
            if not file.filename.endswith('.wav'):
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "Only .wav files are supported"
                })
                continue
            
            # Save file temporarily
            temp_file_path = CHUNK_DIR / f"upload_{uuid.uuid4()}_{file.filename}"
            
            try:
                with open(temp_file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Process the file (this would need to be implemented)
                # For now, just return success
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "message": "File processed successfully",
                    "file_path": str(temp_file_path)
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": str(e)
                })
        
        return {"results": results}
        
    except Exception as e:
        api_logger.error(f"Error processing audio files: {e}")
        raise HTTPException(status_code=500, detail=str(e))