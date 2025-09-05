"""
Memory controller for handling memory-related business logic.
"""

import asyncio
import logging
from typing import Optional

from fastapi.responses import JSONResponse

from advanced_omi_backend.memory import get_memory_service
from advanced_omi_backend.users import User

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")


async def get_memories(user: User, limit: int, user_id: Optional[str] = None):
    """Get memories. Users see only their own memories, admins can see all or filter by user."""
    try:
        memory_service = get_memory_service()

        # Determine which user's memories to fetch
        target_user_id = user.user_id
        if user.is_superuser and user_id:
            target_user_id = user_id

        # Execute memory retrieval directly (now async)
        memories = await memory_service.get_all_memories(target_user_id, limit)
        
        # Get total count (service returns None on failure)
        total_count = await memory_service.count_memories(target_user_id)

        return {
            "memories": memories, 
            "count": len(memories), 
            "total_count": total_count,
            "user_id": target_user_id
        }

    except Exception as e:
        audio_logger.error(f"Error fetching memories: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": f"Error fetching memories: {str(e)}"}
        )


async def get_memories_with_transcripts(user: User, limit: int, user_id: Optional[str] = None):
    """Get memories with their source transcripts. Users see only their own memories, admins can see all or filter by user."""
    try:
        memory_service = get_memory_service()

        # Determine which user's memories to fetch
        target_user_id = user.user_id
        if user.is_superuser and user_id:
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
            status_code=500,
            content={"message": f"Error fetching memories with transcripts: {str(e)}"},
        )


async def search_memories(query: str, user: User, limit: int, score_threshold: float = 0.0, user_id: Optional[str] = None):
    """Search memories by text query. Users can only search their own memories, admins can search all or filter by user."""
    try:
        memory_service = get_memory_service()

        # Determine which user's memories to search
        target_user_id = user.user_id
        if user.is_superuser and user_id:
            target_user_id = user_id

        # Execute search directly (now async)
        search_results = await memory_service.search_memories(query, target_user_id, limit, score_threshold)

        return {
            "query": query,
            "results": search_results,
            "count": len(search_results),
            "user_id": target_user_id,
        }

    except Exception as e:
        audio_logger.error(f"Error searching memories: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": f"Error searching memories: {str(e)}"}
        )


async def delete_memory(memory_id: str, user: User):
    """Delete a memory by ID. Users can only delete their own memories, admins can delete any."""
    try:
        memory_service = get_memory_service()

        # For non-admin users, verify memory ownership before deletion
        if not user.is_superuser:
            # Check if memory belongs to current user
            user_memories = await memory_service.get_all_memories(user.user_id, 1000)

            memory_ids = [str(mem.get("id", mem.get("memory_id", ""))) for mem in user_memories]
            if memory_id not in memory_ids:
                return JSONResponse(status_code=404, content={"message": "Memory not found"})

        # Delete the memory
        success = await memory_service.delete_memory(memory_id)

        if success:
            return JSONResponse(content={"message": f"Memory {memory_id} deleted successfully"})
        else:
            return JSONResponse(status_code=404, content={"message": "Memory not found"})

    except Exception as e:
        audio_logger.error(f"Error deleting memory: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": f"Error deleting memory: {str(e)}"}
        )


async def get_memories_unfiltered(user: User, limit: int, user_id: Optional[str] = None):
    """Get all memories including fallback transcript memories (for debugging). Users see only their own memories, admins can see all or filter by user."""
    try:
        memory_service = get_memory_service()

        # Determine which user's memories to fetch
        target_user_id = user.user_id
        if user.is_superuser and user_id:
            target_user_id = user_id

        # Execute memory retrieval directly (now async)
        memories = await memory_service.get_all_memories_unfiltered(target_user_id, limit)

        return {
            "memories": memories,
            "count": len(memories),
            "user_id": target_user_id,
            "includes_fallback": True,
        }

    except Exception as e:
        audio_logger.error(f"Error fetching unfiltered memories: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"message": f"Error fetching unfiltered memories: {str(e)}"}
        )


async def get_all_memories_admin(user: User, limit: int):
    """Get all memories across all users for admin review. Admin only."""
    try:
        memory_service = get_memory_service()

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
        return JSONResponse(
            status_code=500, content={"message": f"Error fetching admin memories: {str(e)}"}
        )
