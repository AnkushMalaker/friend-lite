import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.results import DeleteResult, InsertOneResult
from fastapi.responses import JSONResponse

logger = logging.getLogger("advanced-backend")

class UserUtils:
    def __init__(self, users_collection):
        self.users_col = users_collection

    async def get_all_users(self):
        """Retrieves all users from the database."""
        try:
            cursor = self.users_col.find()
            users = []
            for doc in await cursor.to_list(length=100):
                doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
                users.append(doc)
            return JSONResponse(content=users)
        except Exception as e:
            logger.error(f"Error fetching users: {e}", exc_info=True)
            return JSONResponse(
                status_code=500, content={"message": "Error fetching users"}
            )

    async def get_user_by_name(self, user_id: str):
        """Retrieves a single user by their user_id."""
        try:
            user = await self.users_col.find_one({"user_id": user_id})
            if user:
                user["_id"] = str(user["_id"]) # Convert ObjectId to string
                return JSONResponse(content=user)
            else:
                return JSONResponse(status_code=404, content={"message": f"User {user_id} not found"})
        except Exception as e:
            logger.error(f"Error fetching user {user_id}: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"message": "Error fetching user"})

    async def create_new_user(self, user_id: str):
        """Creates a new user in the database."""
        try:
            # Check if user already exists using get_user_by_name
            existing_user_response = await self.get_user_by_name(user_id)
            if existing_user_response.status_code == 200:
                return JSONResponse(
                    status_code=409, content={"message": f"User {user_id} already exists"}
                )

            # Create new user
            result: InsertOneResult = await self.users_col.insert_one({"user_id": user_id})
            return JSONResponse(
                status_code=201,
                content={
                    "message": f"User {user_id} created successfully",
                    "id": str(result.inserted_id),
                },
            )
        except Exception as e:
            logger.error(f"Error creating user: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"message": "Error creating user"})

    async def delete_user_data(
        self, user_id: str, delete_conversations: bool = False, delete_memories: bool = False, chunks_col=None, memory_service=None
    ):
        """Deletes a user from the database with optional data cleanup."""
        try:
            # Check if user exists using get_user_by_name
            existing_user_response = await self.get_user_by_name(user_id)
            if existing_user_response.status_code == 404:
                return JSONResponse(
                    status_code=404, content={"message": f"User {user_id} not found"}
                )

            deleted_data = {}

            # Delete user from users collection
            user_result: DeleteResult = await self.users_col.delete_one({"user_id": user_id})
            deleted_data["user_deleted"] = user_result.deleted_count > 0

            if delete_conversations and chunks_col:
                # Delete all conversations (audio chunks) for this user
                conversations_result: DeleteResult = await chunks_col.delete_many({"client_id": user_id})
                deleted_data["conversations_deleted"] = conversations_result.deleted_count

            if delete_memories and memory_service:
                # Delete all memories for this user using the memory service
                try:
                    memory_count = memory_service.delete_all_user_memories(user_id)
                    deleted_data["memories_deleted"] = memory_count
                except Exception as mem_error:
                    logger.error(
                        f"Error deleting memories for user {user_id}: {mem_error}"
                    )
                    deleted_data["memories_deleted"] = 0
                    deleted_data["memory_deletion_error"] = str(mem_error)

            # Build message based on what was deleted
            message = f"User {user_id} deleted successfully"
            deleted_items = []
            if delete_conversations and deleted_data.get("conversations_deleted", 0) > 0:
                deleted_items.append(
                    f"{deleted_data['conversations_deleted']} conversations"
                )
            if delete_memories and deleted_data.get("memories_deleted", 0) > 0:
                deleted_items.append(f"{deleted_data['memories_deleted']} memories")

            if deleted_items:
                message += f" along with {' and '.join(deleted_items)}"

            return JSONResponse(
                status_code=200, content={"message": message, "deleted_data": deleted_data}
            )
        except Exception as e:
            logger.error(f"Error deleting user: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"message": "Error deleting user"})