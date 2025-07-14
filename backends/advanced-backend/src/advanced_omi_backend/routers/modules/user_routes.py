"""
User management routes for Friend-Lite API.

Handles user CRUD operations and admin user management.
"""

import asyncio
import logging

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from advanced_omi_backend.auth import (
    ADMIN_EMAIL,
    current_active_user,
    current_superuser,
    get_user_db,
    get_user_manager,
)
from advanced_omi_backend.client_manager import get_user_clients_all
from advanced_omi_backend.database import chunks_col, db, users_col
from advanced_omi_backend.memory import get_memory_service
from advanced_omi_backend.users import User, UserCreate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[User])
async def get_users(current_user: User = Depends(current_superuser)):
    """Get all users. Admin only."""
    try:
        users = []
        async for user_doc in users_col.find():
            user = User(**user_doc)
            users.append(user)
        return users
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(status_code=500, detail="Error fetching users")


@router.post("/create")
async def create_user(user_data: UserCreate, current_user: User = Depends(current_superuser)):
    """Create a new user. Admin only."""
    try:
        user_db = get_user_db()
        user_manager = get_user_manager()

        # Check if user already exists
        existing_user = await user_manager.get_by_email(user_data.email)
        if existing_user is not None:
            return JSONResponse(
                status_code=409,
                content={"message": f"User with email {user_data.email} already exists"},
            )

        # Create the user through the user manager
        user = await user_manager.create(user_data)

        return JSONResponse(
            status_code=201,
            content={
                "message": f"User {user.email} created successfully",
                "user_id": str(user.id),
                "user_email": user.email,
            },
        )

    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Error creating user: {str(e)}"},
        )


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(current_superuser),
    delete_conversations: bool = False,
    delete_memories: bool = False,
):
    """Delete a user and optionally their associated data. Admin only."""
    try:
        # Validate ObjectId format
        try:
            object_id = ObjectId(user_id)
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Invalid user_id format: {user_id}. Must be a valid ObjectId."
                },
            )

        # Check if user exists
        existing_user = await users_col.find_one({"_id": object_id})
        if not existing_user:
            return JSONResponse(status_code=404, content={"message": f"User {user_id} not found"})

        # Prevent deletion of administrator user
        user_email = existing_user.get("email", "")
        is_superuser = existing_user.get("is_superuser", False)

        if is_superuser or user_email == ADMIN_EMAIL:
            return JSONResponse(
                status_code=403,
                content={
                    "message": f"Cannot delete administrator user. Admin users are protected from deletion."
                },
            )

        deleted_data = {}

        # Delete user from users collection
        user_result = await users_col.delete_one({"_id": object_id})
        deleted_data["user_deleted"] = user_result.deleted_count > 0

        if delete_conversations:
            # Delete all conversations (audio chunks) for this user
            conversations_result = await chunks_col.delete_many({"client_id": user_id})
            deleted_data["conversations_deleted"] = conversations_result.deleted_count

        if delete_memories:
            # Delete all memories for this user using the memory service
            try:
                memory_service = get_memory_service()
                memory_count = await asyncio.get_running_loop().run_in_executor(
                    None, memory_service.delete_all_user_memories, user_id
                )
                deleted_data["memories_deleted"] = memory_count
            except Exception as mem_error:
                logger.error(f"Error deleting memories for user {user_id}: {mem_error}")
                deleted_data["memories_deleted"] = 0
                deleted_data["memory_deletion_error"] = str(mem_error)

        # Build message based on what was deleted
        message = f"User {user_id} deleted successfully"
        deleted_items = []
        if delete_conversations and deleted_data.get("conversations_deleted", 0) > 0:
            deleted_items.append(f"{deleted_data['conversations_deleted']} conversations")
        if delete_memories and deleted_data.get("memories_deleted", 0) > 0:
            deleted_items.append(f"{deleted_data['memories_deleted']} memories")

        if deleted_items:
            message += f" along with {', '.join(deleted_items)}"

        return JSONResponse(
            content={
                "message": message,
                "deleted_data": deleted_data,
            }
        )

    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Error deleting user: {str(e)}"},
        )
