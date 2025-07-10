"""Authentication configuration for fastapi-users with Google OAuth."""

import os
from typing import Optional, overload, Literal, Union

from beanie import PydanticObjectId
from fastapi import Depends, Request, HTTPException, status
from fastapi_users import BaseUserManager, FastAPIUsers
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    CookieTransport,
    JWTStrategy,
)
from fastapi_users.password import PasswordHelper

import re
from httpx_oauth.clients.google import GoogleOAuth2
import logging
from users import User, UserCreate, get_user_db
from dotenv import load_dotenv

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
GOOGLE_CLIENT_ID = _verify_configured("GOOGLE_CLIENT_ID", optional=True)
GOOGLE_CLIENT_SECRET = _verify_configured("GOOGLE_CLIENT_SECRET", optional=True)
COOKIE_SECURE = _verify_configured("COOKIE_SECURE", optional=True) == "true"

# Admin user configuration
ADMIN_USERNAME = _verify_configured("ADMIN_USERNAME", optional=True) or "admin"
ADMIN_PASSWORD = _verify_configured("ADMIN_PASSWORD")
ADMIN_EMAIL = _verify_configured("ADMIN_EMAIL", optional=True) or f"{ADMIN_USERNAME}@example.com"

# Check if Google OAuth is available
GOOGLE_OAUTH_ENABLED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

if GOOGLE_OAUTH_ENABLED:
    print("✅ Google OAuth enabled")
else:
    print("⚠️ Google OAuth disabled - GOOGLE_CLIENT_ID and/or GOOGLE_CLIENT_SECRET not provided")
    print("   Authentication will work with email/password only")

# Check admin configuration
if ADMIN_PASSWORD:
    print(f"✅ Admin user configured: {ADMIN_USERNAME}")
else:
    print("⚠️ ADMIN_PASSWORD not set - admin user will not be created automatically")
    print("   Set ADMIN_PASSWORD in environment to enable automatic admin creation")


class UserManager(BaseUserManager[User, PydanticObjectId]):
    """Custom user manager for handling user operations with both email and user_id authentication."""
    
    reset_password_token_secret = SECRET_KEY
    verification_token_secret = SECRET_KEY

    def parse_id(self, value: str) -> PydanticObjectId:
        """Parse string ID to PydanticObjectId for MongoDB."""
        if isinstance(value, PydanticObjectId):
            return value
        return PydanticObjectId(value)

    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            email: Email address
            
        Returns:
            User if found, None otherwise
        """
        return await self.user_db.get_by_email(email)

    async def authenticate(self, credentials) -> Optional[User]:
        """
        Authenticate user with email+password.
        
        Args:
            credentials: OAuth2PasswordRequestForm with username and password
                        'username' should be email address
        
        Returns:
            User if authentication successful, None otherwise
        """
        # Handle both dict and OAuth2PasswordRequestForm
        if hasattr(credentials, 'username'):
            username = credentials.username
            password = credentials.password
        else:
            username = credentials.get("username")
            password = credentials.get("password")
        
        if not username or not password:
            return None
        
        user = await self.get_by_email(username)
        if not user:
            return None
        
        # Verify password
        password_helper = PasswordHelper()
        verified, updated_password_hash = password_helper.verify_and_update(
            password, user.hashed_password
        )
        
        if not verified:
            return None
        
        # Update password hash if needed
        if updated_password_hash is not None:
            await self.user_db.update(user, {"hashed_password": updated_password_hash})
        
        return user

    async def create(self, user_create: UserCreate, safe: bool = True, request: Optional[Request] = None) -> User:
        """Create user using standard fastapi-users approach with proper superuser handling."""
        # Call parent create method - MongoDB ObjectId will be auto-generated
        user = await super().create(user_create, safe=safe, request=request)
        
        # Update user with superuser and verified status if needed
        # This is required because the base implementation may not preserve these fields
        update_needed = False
        if user_create.is_superuser != user.is_superuser:
            user.is_superuser = user_create.is_superuser
            update_needed = True
        if user_create.is_verified != user.is_verified:
            user.is_verified = user_create.is_verified
            update_needed = True
        
        if update_needed:
            await user.save()
        
        return user

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        """Called after a user registers."""
        print(f"User {user.user_id} ({user.email}) has registered.")

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Called after a user requests password reset."""
        print(f"User {user.user_id} ({user.email}) has forgot their password. Reset token: {token}")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Called after a user requests verification."""
        print(f"Verification requested for user {user.user_id} ({user.email}). Verification token: {token}")


async def get_user_manager(user_db=Depends(get_user_db)):
    """Get user manager instance for dependency injection."""
    yield UserManager(user_db)


# Google OAuth client (only if enabled)
google_oauth_client = None
if GOOGLE_OAUTH_ENABLED:
    assert GOOGLE_CLIENT_ID is not None
    assert GOOGLE_CLIENT_SECRET is not None
    google_oauth_client = GoogleOAuth2(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET)


# Transport configurations
cookie_transport = CookieTransport(
    cookie_max_age=3600,  # 1 hour
    cookie_secure=COOKIE_SECURE,   # Set to False in development if not using HTTPS
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
        print("⚠️ Skipping admin user creation - ADMIN_PASSWORD not set")
        return

    try:
        # Get user database
        user_db_gen = get_user_db()
        user_db = await user_db_gen.__anext__()
        
        # Check if admin user already exists by email
        existing_admin = await user_db.get_by_email(ADMIN_EMAIL)
        
        if existing_admin:
            print(f"✅ Admin user already exists: {existing_admin.user_id} ({existing_admin.email})")
            return

        # Create admin user
        user_manager_gen = get_user_manager(user_db)
        user_manager = await user_manager_gen.__anext__()
        
        admin_create = UserCreate(
            email=ADMIN_EMAIL,
            password=ADMIN_PASSWORD,
            is_superuser=True,
            is_verified=True,
            display_name="Administrator"
        )
        
        admin_user = await user_manager.create(admin_create)
        print(f"✅ Created admin user: {admin_user.user_id} ({admin_user.email}) (ID: {admin_user.id})")
        
    except Exception as e:
        print(f"❌ Failed to create admin user: {e}")


async def websocket_auth(websocket, token: Optional[str] = None) -> Optional[User]:
    """
    WebSocket authentication that supports both cookie and token-based auth.
    Returns None if authentication fails (allowing graceful handling).
    
    Args:
        websocket: The WebSocket connection
        token: Optional JWT token from query parameter
    """
    # Try to get user from JWT token in query parameter first
    if token:
        logger.debug("Attempting WebSocket auth with token from query parameter.")
        try:
            strategy = get_jwt_strategy()
            # Create a dummy user manager instance for token validation
            user_db = await get_user_db().__anext__()
            user_manager = UserManager(user_db)
            user = await strategy.read_token(token, user_manager)
            if user and user.is_active:
                logger.info(f"WebSocket auth successful for user {user.user_id} ({user.email}) using query token.")
                return user
        except Exception as e:
            logger.warning(f"WebSocket auth with query token failed: {e}")
            pass  # Fall through to cookie auth
    
    # Try to get user from cookie
    logger.debug("Attempting WebSocket auth with cookie.")
    try:
        # Extract cookies from WebSocket headers
        cookie_header = None
        for name, value in websocket.headers.items():
            if name.lower() == b'cookie':
                cookie_header = value.decode()
                break
        
        if cookie_header:
            # Parse cookies to find our auth cookie
            cookie_pattern = r'fastapiusersauth=([^;]+)'
            match = re.search(cookie_pattern, cookie_header)
            if match:
                cookie_value = match.group(1)
                strategy = get_jwt_strategy()
                # Create a dummy user manager instance for token validation
                user_db = await get_user_db().__anext__()
                user_manager = UserManager(user_db)
                user = await strategy.read_token(cookie_value, user_manager)
                if user and user.is_active:
                    logger.info(f"WebSocket auth successful for user {user.user_id} ({user.email}) using cookie.")
                    return user
    except Exception as e:
        logger.warning(f"WebSocket auth with cookie failed: {e}")
        pass
    
    logger.warning("WebSocket authentication failed.")
    return None 