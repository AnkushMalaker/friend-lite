"""User models for fastapi-users integration with Beanie and MongoDB."""

import logging
import random
import string
from datetime import UTC, datetime
from typing import Optional

from beanie import Document, PydanticObjectId
from fastapi_users.db import BeanieBaseUser, BeanieUserDatabase
from fastapi_users.schemas import BaseUserCreate
from pydantic import Field

logger = logging.getLogger(__name__)


class UserCreate(BaseUserCreate):
    """Schema for creating new users."""

    display_name: Optional[str] = None


class User(BeanieBaseUser, Document):
    """User model extending fastapi-users BeanieBaseUser with custom fields."""

    display_name: Optional[str] = None
    # Client tracking for audio devices
    registered_clients: dict[str, dict] = Field(default_factory=dict)

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
    except Exception:
        return None


async def get_user_by_client_id(client_id: str) -> Optional[User]:
    """Find the user that owns a specific client_id."""
    return await User.find_one({"registered_clients.client_id": client_id})


async def register_client_to_user(
    user: User, client_id: str, device_name: Optional[str] = None
) -> None:
    """Register a client to a user and save to database."""
    user.register_client(client_id, device_name)
    await user.save()


def generate_client_id(user: User, device_name: Optional[str] = None) -> str:
    """
    Generate a unique client_id in the format: user_id_suffix-device_suffix[-counter]

    Args:
        user: The User object
        device_name: Optional device name (e.g., 'havpe', 'phone', 'tablet')

    Returns:
        client_id in format: user_id_suffix-device_suffix or user_id_suffix-device_suffix-N for duplicates
    """
    # Use last 6 characters of MongoDB ObjectId as user identifier
    user_id_suffix = str(user.id)[-6:]

    if device_name:
        # Sanitize device name: lowercase, alphanumeric + hyphens only, max 10 chars
        sanitized_device = "".join(c for c in device_name.lower() if c.isalnum() or c == "-")[:10]
        base_client_id = f"{user_id_suffix}-{sanitized_device}"

        # Check for existing client IDs to avoid conflicts
        existing_client_ids = user.get_client_ids()

        # If base client_id doesn't exist, use it
        if base_client_id not in existing_client_ids:
            return base_client_id

        # If it exists, find the next available counter
        counter = 2
        while f"{base_client_id}-{counter}" in existing_client_ids:
            counter += 1

        return f"{base_client_id}-{counter}"
    else:
        # Generate random 4-character suffix if no device name provided
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return f"{user_id_suffix}-{suffix}"
