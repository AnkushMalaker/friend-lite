"""
Memory management routes for Friend-Lite API.

Handles memory CRUD operations, search, and debug functionality.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from advanced_omi_backend.auth import current_active_user, current_superuser
from advanced_omi_backend.client_manager import get_user_clients_all
from advanced_omi_backend.debug_system_tracker import get_debug_tracker
from advanced_omi_backend.memory import get_memory_service
from advanced_omi_backend.users import User

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")

router = APIRouter(prefix="/memories", tags=["memories"])


@router.get("")
async def get_memories(
    current_user: User = Depends(current_active_user),
    limit: int = Query(default=50, ge=1, le=1000),
    user_id: Optional[str] = Query(default=None, description="User ID filter (admin only)"),
):
    """Get memories. Users see only their own memories, admins can see all or filter by user."""
    try:
        memory_service = get_memory_service()

        # Determine which user's memories to fetch
        target_user_id = current_user.user_id
        if current_user.is_superuser and user_id:
            target_user_id = user_id

        # Execute memory retrieval in thread pool to avoid blocking
        memories = await asyncio.get_running_loop().run_in_executor(
            None, memory_service.get_all_memories, target_user_id, limit
        )

        return {"memories": memories, "count": len(memories), "user_id": target_user_id}

    except Exception as e:
        audio_logger.error(f"Error fetching memories: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error fetching memories"})


@router.get("/with-transcripts")
async def get_memories_with_transcripts(
    current_user: User = Depends(current_active_user),
    limit: int = Query(default=50, ge=1, le=1000),
    user_id: Optional[str] = Query(default=None, description="User ID filter (admin only)"),
):
    """Get memories with their source transcripts. Users see only their own memories, admins can see all or filter by user."""
    try:
        memory_service = get_memory_service()

        # Determine which user's memories to fetch
        target_user_id = current_user.user_id
        if current_user.is_superuser and user_id:
            target_user_id = user_id

        # Execute memory retrieval directly (now async)
        memories_with_transcripts = await memory_service.get_memories_with_transcripts(
            target_user_id, limit
        )

        return {
            "memories": memories_with_transcripts,  # Streamlit expects 'memories' key
            "count": len(memories_with_transcripts),
            "user_id": target_user_id,
        }

    except Exception as e:
        audio_logger.error(f"Error fetching memories with transcripts: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error fetching memories with transcripts"}
        )


@router.get("/search")
async def search_memories(
    query: str = Query(..., description="Search query"),
    current_user: User = Depends(current_active_user),
    limit: int = Query(default=20, ge=1, le=100),
    user_id: Optional[str] = Query(default=None, description="User ID filter (admin only)"),
):
    """Search memories by text query. Users can only search their own memories, admins can search all or filter by user."""
    try:
        memory_service = get_memory_service()

        # Determine which user's memories to search
        target_user_id = current_user.user_id
        if current_user.is_superuser and user_id:
            target_user_id = user_id

        # Execute search in thread pool to avoid blocking
        search_results = await asyncio.get_running_loop().run_in_executor(
            None, memory_service.search_memories, query, target_user_id, limit
        )

        return {
            "query": query,
            "results": search_results,
            "count": len(search_results),
            "user_id": target_user_id,
        }

    except Exception as e:
        audio_logger.error(f"Error searching memories: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error searching memories"})


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, current_user: User = Depends(current_active_user)):
    """Delete a memory by ID. Users can only delete their own memories, admins can delete any."""
    try:
        memory_service = get_memory_service()

        # For non-admin users, verify memory ownership before deletion
        if not current_user.is_superuser:
            # Check if memory belongs to current user
            user_memories = await asyncio.get_running_loop().run_in_executor(
                None, memory_service.get_all_memories, current_user.user_id, 1000
            )

            memory_ids = [str(mem.get("id", mem.get("memory_id", ""))) for mem in user_memories]
            if memory_id not in memory_ids:
                return JSONResponse(status_code=404, content={"message": "Memory not found"})

        # Delete the memory
        success = await asyncio.get_running_loop().run_in_executor(
            None, memory_service.delete_memory, memory_id
        )

        if success:
            return JSONResponse(content={"message": f"Memory {memory_id} deleted successfully"})
        else:
            return JSONResponse(status_code=404, content={"message": "Memory not found"})

    except Exception as e:
        audio_logger.error(f"Error deleting memory: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error deleting memory"})


@router.get("/unfiltered")
async def get_memories_unfiltered(
    current_user: User = Depends(current_active_user),
    limit: int = Query(default=50, ge=1, le=1000),
    user_id: Optional[str] = Query(default=None, description="User ID filter (admin only)"),
):
    """Get all memories including fallback transcript memories (for debugging). Users see only their own memories, admins can see all or filter by user."""
    try:
        memory_service = get_memory_service()

        # Determine which user's memories to fetch
        target_user_id = current_user.user_id
        if current_user.is_superuser and user_id:
            target_user_id = user_id

        # Execute memory retrieval in thread pool to avoid blocking
        memories = await asyncio.get_running_loop().run_in_executor(
            None, memory_service.get_all_memories_unfiltered, target_user_id, limit
        )

        return {
            "memories": memories,
            "count": len(memories),
            "user_id": target_user_id,
            "includes_fallback": True,
        }

    except Exception as e:
        audio_logger.error(f"Error fetching unfiltered memories: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": "Error fetching unfiltered memories"}
        )


@router.get("/admin")
async def get_all_memories_admin(current_user: User = Depends(current_superuser), limit: int = 200):
    """Get all memories across all users for admin review. Admin only."""
    try:
        memory_service = get_memory_service()

        # Get debug tracker for additional context
        debug_tracker = get_debug_tracker()

        # Get all memories without user filtering
        all_memories = await memory_service.get_all_memories_debug(limit)

        # Group by user for easier admin review
        user_memories = {}
        users_with_memories = set()
        client_ids_with_memories = set()

        for memory in all_memories:
            user_id = memory.get("user_id", "unknown")
            client_id = memory.get("client_id", "unknown")

            if user_id not in user_memories:
                user_memories[user_id] = []
            user_memories[user_id].append(memory)

            # Track users and clients for debug info
            users_with_memories.add(user_id)
            client_ids_with_memories.add(client_id)

        # Enhanced stats combining both admin and debug information
        stats = {
            "total_memories": len(all_memories),
            "total_users": len(user_memories),
            "debug_tracker_initialized": debug_tracker is not None,
            "users_with_memories": sorted(list(users_with_memories)),
            "client_ids_with_memories": sorted(list(client_ids_with_memories)),
        }

        return {
            "memories": all_memories,  # Flat list for compatibility
            "user_memories": user_memories,  # Grouped by user
            "stats": stats,
            "total_users": len(user_memories),
            "total_memories": len(all_memories),
            "limit": limit,
        }

    except Exception as e:
        audio_logger.error(f"Error fetching admin memories: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"message": "Error fetching admin memories"})
