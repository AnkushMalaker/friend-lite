"""Authentication configuration for fastapi-users with Google OAuth."""

import os
from typing import Optional

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    CookieTransport,
    JWTStrategy,
)
from httpx_oauth.clients.google import GoogleOAuth2

from models import User, UserCreate, UserRead, UserUpdate, get_user_db

# Configuration from environment variables
SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# Validate required environment variables
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    raise ValueError("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in environment variables")


class UserManager(BaseUserManager[User, str]):
    """Custom user manager for handling user operations."""
    
    reset_password_token_secret = SECRET_KEY
    verification_token_secret = SECRET_KEY

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        """Called after a user registers."""
        print(f"User {user.id} has registered.")

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Called after a user requests password reset."""
        print(f"User {user.id} has forgot their password. Reset token: {token}")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Called after a user requests verification."""
        print(f"Verification requested for user {user.id}. Verification token: {token}")


async def get_user_manager(user_db=Depends(get_user_db)):
    """Get user manager instance for dependency injection."""
    yield UserManager(user_db)


# Google OAuth client
google_oauth_client = GoogleOAuth2(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)


# Transport configurations
cookie_transport = CookieTransport(
    cookie_max_age=3600,  # 1 hour
    cookie_secure=True,   # Set to False in development if not using HTTPS
    cookie_httponly=True,
    cookie_samesite="lax",
)

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    """Get JWT strategy for token generation and validation."""
    return JWTStrategy(secret=SECRET_KEY, lifetime_seconds=3600)


# Authentication backends
cookie_backend = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_jwt_strategy,
)

bearer_backend = AuthenticationBackend(
    name="bearer", 
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

# FastAPI Users instance
fastapi_users = FastAPIUsers[User, str](
    get_user_manager,
    [cookie_backend, bearer_backend],
)

# User dependency for protecting endpoints
current_active_user = fastapi_users.current_user(active=True)
optional_current_user = fastapi_users.current_user(optional=True) 