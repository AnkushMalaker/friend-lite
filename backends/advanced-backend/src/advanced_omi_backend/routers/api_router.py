"""
Main API router for Friend-Lite backend.

This module aggregates all the functional router modules and provides
a single entry point for the API endpoints.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from advanced_omi_backend.auth import current_active_user, current_superuser
from advanced_omi_backend.debug_system_tracker import get_debug_tracker
from advanced_omi_backend.memory import get_memory_service
from advanced_omi_backend.users import User

from .modules import (
    client_router,
    conversation_router,
    memory_router,
    system_router,
    user_router,
)

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")

# Create main API router
router = APIRouter(prefix="/api", tags=["api"])

# Include all sub-routers
router.include_router(user_router)
router.include_router(client_router)
router.include_router(conversation_router)
router.include_router(memory_router)
router.include_router(system_router)


# Admin endpoints for backward compatibility with Streamlit UI
@router.get("/admin/memories")
async def get_admin_memories(current_user: User = Depends(current_superuser), limit: int = 200):
    """Get all memories across all users for admin review. Admin only. Compatibility endpoint."""
    try:
        memory_service = get_memory_service()

        # Get debug tracker for additional context
        debug_tracker = get_debug_tracker()

        # Get all memories without user filtering
        all_memories = await asyncio.get_running_loop().run_in_executor(
            None, memory_service.get_all_memories_debug, limit
        )

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


@router.get("/admin/memories/debug")
async def get_admin_memories_debug(
    current_user: User = Depends(current_superuser), limit: int = 200
):
    """Get all memories across all users for debugging. Admin only. Compatibility endpoint that redirects to main admin endpoint."""
    # This is now just a redirect to the main admin endpoint for compatibility
    return await get_admin_memories(current_user, limit)


# Active clients compatibility endpoint
@router.get("/active_clients")
async def get_active_clients_compat(current_user: User = Depends(current_active_user)):
    """Get active clients. Compatibility endpoint for Streamlit UI."""
    try:
        from advanced_omi_backend.client_manager import (
            get_client_manager,
            get_user_clients_active,
        )

        client_manager = get_client_manager()

        if not client_manager.is_initialized():
            return JSONResponse(
                status_code=503,
                content={"error": "Client manager not available"},
            )

        if current_user.is_superuser:
            # Admin: return all active clients
            clients_info = client_manager.get_client_info_summary()
        else:
            # Regular user: return only their own clients
            user_active_clients = get_user_clients_active(current_user.user_id)
            all_clients = client_manager.get_client_info_summary()

            # Filter to only the user's clients
            clients_info = [
                client for client in all_clients if client["client_id"] in user_active_clients
            ]

        return {
            "clients": clients_info,
            "active_clients_count": len(clients_info),
            "total_count": len(clients_info),
        }

    except Exception as e:
        audio_logger.error(f"Error getting active clients: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get active clients"},
        )


logger.info("API router initialized with all sub-modules")
