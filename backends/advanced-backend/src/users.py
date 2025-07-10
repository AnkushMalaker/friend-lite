"""User models for fastapi-users integration with Beanie and MongoDB."""

import random
import string
from typing import Optional

from beanie import Document, PydanticObjectId, Indexed
from fastapi_users.db import BaseOAuthAccount, BeanieBaseUser, BeanieUserDatabase
from fastapi_users.schemas import BaseUser, BaseUserCreate, BaseUserUpdate
from pydantic import Field, validator


def generate_user_id() -> str:
    """Generate a unique 6-character alphanumeric user ID."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))


class OAuthAccount(BaseOAuthAccount):
    """OAuth account model for storing third-party authentication info."""
    pass


class User(BeanieBaseUser, Document):
    """User model extending fastapi-users BeanieBaseUser with custom fields."""
    
    # Primary identifier - 6-character alphanumeric
    user_id: Indexed(str, unique=True) = Field(default_factory=generate_user_id)
    
    # Custom fields for your application
    display_name: Optional[str] = None
    profile_picture: Optional[str] = None
    oauth_accounts: list[OAuthAccount] = Field(default_factory=list)
    
    @validator('user_id')
    def validate_user_id(cls, v):
        """Validate user_id format: 6-character alphanumeric."""
        if not v:
            return generate_user_id()
        if len(v) != 6 or not v.isalnum() or not v.islower():
            raise ValueError('user_id must be 6 lowercase alphanumeric characters')
        return v
    
    class Settings:
        name = "fastapi_users"  # Collection name in MongoDB
        email_collation = {
            "locale": "en", 
            "strength": 2  # Case-insensitive comparison
        }


class UserRead(BaseUser[PydanticObjectId]):
    """Schema for reading user data."""
    user_id: str
    display_name: Optional[str] = None
    profile_picture: Optional[str] = None


class UserCreate(BaseUserCreate):
    """Schema for creating user data."""
    user_id: Optional[str] = None  # Optional - will be auto-generated if not provided
    display_name: Optional[str] = None


class UserUpdate(BaseUserUpdate):
    """Schema for updating user data."""
    display_name: Optional[str] = None
    profile_picture: Optional[str] = None


async def get_user_db():
    """Get the user database instance for dependency injection."""
    yield BeanieUserDatabase(User, OAuthAccount)


async def get_user_by_user_id(user_id: str) -> Optional[User]:
    """Get user by user_id (for user_id+password authentication)."""
    return await User.find_one(User.user_id == user_id)


def generate_client_id(user_id: str, device_name: Optional[str] = None) -> str:
    """
    Generate a client_id in the format: user_id-device_suffix
    
    Args:
        user_id: The user's 6-character identifier
        device_name: Optional device name (e.g., 'havpe', 'phone', 'tablet')
    
    Returns:
        client_id in format: user_id-device_suffix
    """
    if device_name:
        # Sanitize device name: lowercase, alphanumeric + hyphens only, max 10 chars
        sanitized_device = ''.join(c for c in device_name.lower() if c.isalnum() or c == '-')[:10]
        return f"{user_id}-{sanitized_device}"
    else:
        # Generate random 4-character suffix if no device name provided
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return f"{user_id}-{suffix}" 