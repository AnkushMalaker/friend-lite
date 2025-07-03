"""User models for fastapi-users integration with Beanie and MongoDB."""

from typing import Optional

from beanie import Document
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
    
    class Settings:
        name = "fastapi_users"  # Collection name in MongoDB
        email_collation = {
            "locale": "en", 
            "strength": 2  # Case-insensitive comparison
        }


class UserRead(BaseUser[str]):
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