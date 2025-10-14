"""Authentication configuration for fastapi-users with email/password and JWT."""

import logging
import os
import re
from typing import Literal, Optional, overload

from beanie import PydanticObjectId
from dotenv import load_dotenv
from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    CookieTransport,
    JWTStrategy,
)

from advanced_omi_backend.users import User, UserCreate, get_user_db

logger = logging.getLogger(__name__)

load_dotenv()


@overload
def _verify_configured(var_name: str, *, optional: Literal[False] = False) -> str: ...
@overload
def _verify_configured(var_name: str, *, optional: Literal[True]) -> Optional[str]: ...


def _verify_configured(var_name: str, *, optional: bool = False) -> Optional[str]:
    value = os.getenv(var_name)
    if not optional and not value:
        raise ValueError(f"{var_name} is not set")
    return value


# Configuration from environment variables
SECRET_KEY = _verify_configured("AUTH_SECRET_KEY")
COOKIE_SECURE = _verify_configured("COOKIE_SECURE", optional=True) == "true"

# Admin user configuration
ADMIN_PASSWORD = _verify_configured("ADMIN_PASSWORD")
ADMIN_EMAIL = _verify_configured("ADMIN_EMAIL", optional=True) or "admin@example.com"


class UserManager(BaseUserManager[User, PydanticObjectId]):
    """User manager with minimal customization for fastapi-users."""

    reset_password_token_secret = SECRET_KEY
    verification_token_secret = SECRET_KEY

    def parse_id(self, value: str) -> PydanticObjectId:
        """Parse string ID to PydanticObjectId for MongoDB compatibility."""
        try:
            return PydanticObjectId(value)
        except Exception as e:
            raise ValueError(f"Invalid ObjectId format: {value}") from e

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        """Called after a user registers."""
        logger.info(f"User {user.user_id} ({user.email}) has registered.")

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Called after a user requests password reset."""
        logger.info(f"User {user.user_id} ({user.email}) has requested password reset")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Called after a user requests verification."""
        logger.info(f"Verification requested for user {user.user_id} ({user.email})")


async def get_user_manager(user_db=Depends(get_user_db)):
    """Get user manager instance for dependency injection."""
    yield UserManager(user_db)


# Transport configurations
cookie_transport = CookieTransport(
    cookie_max_age=86400,  # 24 hours (matches JWT lifetime)
    cookie_secure=COOKIE_SECURE,  # Set to False in development if not using HTTPS
    cookie_httponly=True,
    cookie_samesite="lax",
)

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    """Get JWT strategy for token generation and validation."""
    return JWTStrategy(
        secret=SECRET_KEY, lifetime_seconds=86400
    )  # 24 hours for device compatibility


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
fastapi_users = FastAPIUsers[User, PydanticObjectId](
    get_user_manager,
    [cookie_backend, bearer_backend],
)

# User dependencies for protecting endpoints
current_active_user = fastapi_users.current_user(active=True)
current_superuser = fastapi_users.current_user(active=True, superuser=True)


def get_accessible_user_ids(user: User) -> list[str] | None:
    """
    Get list of user IDs that the current user can access data for.
    Returns None for superusers (can access all), or [user.id] for regular users.
    """
    if user.is_superuser:
        return None  # Can access all data
    else:
        return [str(user.id)]  # Can only access own data


async def create_admin_user_if_needed():
    """Create admin user during startup if it doesn't exist and credentials are provided."""
    if not ADMIN_PASSWORD:
        logger.warning("Skipping admin user creation - ADMIN_PASSWORD not set")
        return

    try:
        # Get user database
        user_db_gen = get_user_db()
        user_db = await user_db_gen.__anext__()

        # Check if admin user already exists by email
        existing_admin = await user_db.get_by_email(ADMIN_EMAIL)

        if existing_admin:
            logger.info(
                f"✅ Admin user already exists: {existing_admin.user_id} ({existing_admin.email})"
            )
            return

        # Create admin user
        user_manager_gen = get_user_manager(user_db)
        user_manager = await user_manager_gen.__anext__()

        admin_create = UserCreate(
            email=ADMIN_EMAIL,
            password=ADMIN_PASSWORD,
            is_superuser=True,
            is_verified=True,
            display_name="Administrator",
        )

        admin_user = await user_manager.create(admin_create)
        logger.info(
            f"✅ Created admin user: {admin_user.user_id} ({admin_user.email}) (ID: {admin_user.id})"
        )

    except Exception as e:
        logger.error(f"Failed to create admin user: {e}", exc_info=True)


async def websocket_auth(websocket, token: Optional[str] = None) -> Optional[User]:
    """
    WebSocket authentication that supports both cookie and token-based auth.
    Returns None if authentication fails (allowing graceful handling).
    """
    strategy = get_jwt_strategy()

    # Try JWT token from query parameter first
    if token:
        logger.debug("Attempting WebSocket auth with query token.")
        try:
            user_db_gen = get_user_db()
            user_db = await user_db_gen.__anext__()
            user_manager = UserManager(user_db)
            user = await strategy.read_token(token, user_manager)
            if user and user.is_active:
                logger.info(f"WebSocket auth successful for user {user.user_id} using query token.")
                return user
        except Exception as e:
            logger.warning(f"WebSocket auth with query token failed: {e}")

    # Try cookie authentication
    logger.debug("Attempting WebSocket auth with cookie.")
    try:
        cookie_header = next(
            (v.decode() for k, v in websocket.headers.items() if k.lower() == b"cookie"), None
        )
        if cookie_header:
            match = re.search(r"fastapiusersauth=([^;]+)", cookie_header)
            if match:
                user_db_gen = get_user_db()
                user_db = await user_db_gen.__anext__()
                user_manager = UserManager(user_db)
                user = await strategy.read_token(match.group(1), user_manager)
                if user and user.is_active:
                    logger.info(f"WebSocket auth successful for user {user.user_id} using cookie.")
                    return user
    except Exception as e:
        logger.warning(f"WebSocket auth with cookie failed: {e}")

    logger.warning("WebSocket authentication failed.")
    return None
