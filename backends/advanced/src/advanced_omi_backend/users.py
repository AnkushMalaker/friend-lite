"""User models for fastapi-users integration with Beanie and MongoDB."""

import logging
from datetime import UTC, datetime
from typing import Optional

from beanie import Document, PydanticObjectId
from fastapi_users.db import BeanieBaseUser, BeanieUserDatabase
from fastapi_users.schemas import BaseUser, BaseUserCreate, BaseUserUpdate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UserCreate(BaseUserCreate):
    """Schema for creating new users."""

    display_name: Optional[str] = None


class UserRead(BaseUser[PydanticObjectId]):
    """Schema for reading user data."""
    
    display_name: Optional[str] = None
    registered_clients: dict[str, dict] = Field(default_factory=dict)
    primary_speakers: list[dict] = Field(default_factory=list)


class UserUpdate(BaseUserUpdate):
    """Schema for updating user data."""
    
    display_name: Optional[str] = None


class User(BeanieBaseUser, Document):
    """User model extending fastapi-users BeanieBaseUser with custom fields."""

    display_name: Optional[str] = None
    # Client tracking for audio devices
    registered_clients: dict[str, dict] = Field(default_factory=dict)
    # Speaker processing filter configuration
    primary_speakers: list[dict] = Field(default_factory=list)

    @property
    def user_id(self) -> str:
        """Return string representation of MongoDB ObjectId for backward compatibility."""
        return str(self.id)

    def register_client(self, client_id: str, device_name: Optional[str] = None) -> None:
        """Register a new client for this user."""
        # Check if client already exists
        if client_id in self.registered_clients:
            # Update existing client
            logger.info(f"Updating existing client {client_id} for user {self.user_id}")
            self.registered_clients[client_id]["last_seen"] = datetime.now(UTC)
            self.registered_clients[client_id]["device_name"] = (
                device_name or self.registered_clients[client_id].get("device_name")
            )
            return

        # Add new client
        self.registered_clients[client_id] = {
            "client_id": client_id,
            "device_name": device_name,
            "first_seen": datetime.now(UTC),
            "last_seen": datetime.now(UTC),
            "is_active": True,
        }

    def get_client_ids(self) -> list[str]:
        """Get all client IDs registered to this user."""
        return list(self.registered_clients.keys())

    # def has_client(self, client_id: str) -> bool:
    #     """Check if a client is registered to this user."""
    #     return client_id in self.registered_clients

    class Settings:
        name = "users"  # Collection name in MongoDB - standardized from "fastapi_users"
        email_collation = {"locale": "en", "strength": 2}  # Case-insensitive comparison


async def get_user_db():
    """Get the user database instance for dependency injection."""
    yield BeanieUserDatabase(User)  # type: ignore


async def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by MongoDB ObjectId string."""
    try:
        return await User.get(PydanticObjectId(user_id))
    except Exception as e:
        logger.error(f"Failed to get user by ID {user_id}: {e}")
        # Re-raise for proper error handling upstream
        raise


async def get_user_by_client_id(client_id: str) -> Optional[User]:
    """Find the user that owns a specific client_id."""
    return await User.find_one({"registered_clients.client_id": client_id})


async def register_client_to_user(
    user: User, client_id: str, device_name: Optional[str] = None
) -> None:
    """Register a client to a user and save to database."""
    user.register_client(client_id, device_name)
    await user.save()
