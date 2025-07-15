from fastapi import APIRouter, Depends, Query
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient

from utils.user_utils import UserUtils
from memory.memory_service import get_memory_service # Assuming this is how memory_service is obtained
from database import get_db_client # Assuming this is how the database client is obtained

router = APIRouter()

def get_user_utils(
    db_client: AsyncIOMotorClient = Depends(get_db_client),
    memory_service = Depends(get_memory_service) # Assuming get_memory_service is a dependency
):
    users_col = db_client.get_default_database("friend-lite")["users"]
    return UserUtils(users_col)

@router.get("/api/users")
async def get_users(user_utils: UserUtils = Depends(get_user_utils)):
    """Retrieves all users from the database."""
    return await user_utils.get_all_users()

@router.get("/api/users/{user_id}")
async def get_user_by_id(user_id: str, user_utils: UserUtils = Depends(get_user_utils)):
    """Retrieves a single user by their user_id."""
    return await user_utils.get_user_by_name(user_id)

@router.post("/api/create_user")
async def create_user(user_id: str, user_utils: UserUtils = Depends(get_user_utils)):
    """Creates a new user in the database."""
    return await user_utils.create_new_user(user_id)

@router.delete("/api/delete_user")
async def delete_user(
    user_id: str,
    delete_conversations: bool = False,
    delete_memories: bool = False,
    user_utils: UserUtils = Depends(get_user_utils),
    db_client: AsyncIOMotorClient = Depends(get_db_client)
):
    """Deletes a user from the database with optional data cleanup."""
    chunks_col = db_client.get_default_database("friend-lite")["audio_chunks"]
    return await user_utils.delete_user_data(user_id, delete_conversations, delete_memories, chunks_col)
