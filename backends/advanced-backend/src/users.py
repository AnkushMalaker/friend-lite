"""User models for fastapi-users integration with Beanie and MongoDB."""

import random
import string
from typing import Optional

from beanie import Document, PydanticObjectId
from fastapi_users.db import BaseOAuthAccount, BeanieBaseUser, BeanieUserDatabase
from fastapi_users.schemas import BaseUser, BaseUserCreate, BaseUserUpdate
from pydantic import Field


class OAuthAccount(BaseOAuthAccount):
    """OAuth account model for storing third-party authentication info."""
    pass


class User(BeanieBaseUser, Document):
    """User model extending fastapi-users BeanieBaseUser with custom fields."""
    
    # Custom fields for your application
    display_name: Optional[str] = None
    profile_picture: Optional[str] = None
    oauth_accounts: list[OAuthAccount] = Field(default_factory=list)
    
    @property
    def user_id(self) -> str:
        """Return string representation of MongoDB ObjectId for backward compatibility."""
        return str(self.id)
    
    class Settings:
        name = "fastapi_users"  # Collection name in MongoDB
        email_collation = {
            "locale": "en", 
            "strength": 2  # Case-insensitive comparison
        }


class UserRead(BaseUser[PydanticObjectId]):
    """Schema for reading user data."""
    display_name: Optional[str] = None
    profile_picture: Optional[str] = None


class UserCreate(BaseUserCreate):
    """Schema for creating user data."""
    display_name: Optional[str] = None


class UserUpdate(BaseUserUpdate):
    """Schema for updating user data."""
    display_name: Optional[str] = None
    profile_picture: Optional[str] = None


async def get_user_db():
    """Get the user database instance for dependency injection."""
    yield BeanieUserDatabase(User, OAuthAccount)


async def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by MongoDB ObjectId string."""
    try:
        return await User.get(PydanticObjectId(user_id))
    except Exception:
        return None


def generate_client_id(user: User, device_name: Optional[str] = None) -> str:
    """
    Generate a client_id in the format: user_id_suffix-device_suffix
    
    Args:
        user: The User object
        device_name: Optional device name (e.g., 'havpe', 'phone', 'tablet')
    
    Returns:
        client_id in format: user_id_suffix-device_suffix
    """
    # Use last 6 characters of MongoDB ObjectId as user identifier
    user_id_suffix = str(user.id)[-6:]
    
    if device_name:
        # Sanitize device name: lowercase, alphanumeric + hyphens only, max 10 chars
        sanitized_device = ''.join(c for c in device_name.lower() if c.isalnum() or c == '-')[:10]
        return f"{user_id_suffix}-{sanitized_device}"
    else:
        # Generate random 4-character suffix if no device name provided
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return f"{user_id_suffix}-{suffix}" 