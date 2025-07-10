# Authentication Architecture

## Overview

Friend-Lite uses a comprehensive authentication system built on `fastapi-users` with support for multiple authentication methods including JWT tokens, cookies, and Google OAuth. The system provides secure user management with proper data isolation and role-based access control.

## Architecture Components

### 1. User Model (`users.py`)

```python
class User(BeanieBaseUser, Document):
    # Primary identifier - 6-character alphanumeric
    user_id: str = Field(default_factory=generate_user_id)
    
    # Standard fastapi-users fields
    email: str
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    is_verified: bool = False
    
    # Custom fields
    display_name: Optional[str] = None
    profile_picture: Optional[str] = None
    oauth_accounts: list[OAuthAccount] = []
```

**Key Features:**
- **Dual Identity**: Users have both `email` and `user_id` for authentication
- **user_id**: 6-character lowercase alphanumeric identifier (e.g., `abc123`)
- **Email**: Standard email address for authentication
- **MongoDB Integration**: Uses Beanie ODM for document storage
- **OAuth Support**: Integrated Google OAuth account linking

### 2. Authentication Manager (`auth.py`)

```python
class UserManager(BaseUserManager[User, PydanticObjectId]):
    async def authenticate(self, credentials: dict) -> Optional[User]:
        """Authenticate with either email or user_id"""
        username = credentials.get("username")
        # Supports both email and user_id authentication
        
    async def get_by_email_or_user_id(self, identifier: str) -> Optional[User]:
        """Get user by either email or 6-character user_id"""
```

**Key Features:**
- **Flexible Authentication**: Login with either email or user_id
- **Auto-detection**: Automatically detects if identifier is user_id or email
- **Password Management**: Secure password hashing and verification
- **User Creation**: Auto-generates unique user_id if not provided

### 3. Authentication Backends

#### JWT Bearer Token
- **Endpoint**: `/auth/jwt/login`
- **Transport**: Authorization header (`Bearer <token>`)
- **Lifetime**: 1 hour
- **Usage**: API calls, WebSocket authentication

#### Cookie Authentication
- **Endpoint**: `/auth/cookie/login`
- **Transport**: HTTP cookies
- **Lifetime**: 1 hour
- **Usage**: Web dashboard, browser-based clients

#### Google OAuth (Optional)
- **Endpoint**: `/auth/google/login`
- **Transport**: OAuth2 flow with cookies
- **Features**: Auto-registration, email verification
- **Usage**: Social login integration

## Authentication Flow

### 1. User Registration

**Admin-Only Registration:**
```bash
# Create user with auto-generated user_id
curl -X POST "http://localhost:8000/api/create_user" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "userpass",
    "display_name": "John Doe"
  }'

# Response: {"user_id": "abc123", "email": "user@example.com", ...}
```

**User ID Specification:**
```bash
# Create user with specific user_id
curl -X POST "http://localhost:8000/api/create_user" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "userpass",
    "user_id": "user01",
    "display_name": "John Doe"
  }'
```

### 2. Authentication Methods

#### Email-based Login
```bash
curl -X POST "http://localhost:8000/auth/jwt/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=userpass"
```

#### User ID-based Login
```bash
curl -X POST "http://localhost:8000/auth/jwt/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=abc123&password=userpass"
```

#### Google OAuth Login
```bash
# Redirect to Google OAuth
curl "http://localhost:8000/auth/google/login"
```

### 3. WebSocket Authentication

#### Token-based (Recommended)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws_pcm?token=JWT_TOKEN&device_name=phone');
```

#### Cookie-based
```javascript
// Requires existing cookie from web login
const ws = new WebSocket('ws://localhost:8000/ws_pcm?device_name=phone');
```

## Client ID Management

### Format: `user_id-device_name`

The system automatically generates client IDs by combining:
- **user_id**: 6-character user identifier
- **device_name**: Sanitized device identifier

**Examples:**
- `abc123-phone` (user: abc123, device: phone)
- `admin-desktop` (user: admin, device: desktop)
- `user01-havpe` (user: user01, device: havpe)

### Benefits:
- **User Association**: Clear mapping between clients and users
- **Device Tracking**: Multiple devices per user
- **Data Isolation**: Each user only accesses their own data
- **Audit Trail**: Track activity by user and device

## Security Features

### 1. Password Security
- **Bcrypt Hashing**: Secure password storage with salt
- **Password Verification**: Constant-time comparison
- **Hash Updates**: Automatic rehashing on login when needed

### 2. Token Security
- **JWT Tokens**: Signed with secret key
- **Short Lifetime**: 1-hour expiration
- **Secure Transport**: HTTPS recommended for production

### 3. Data Isolation
- **User Scoping**: All data scoped to user_id
- **Client Filtering**: Users only see their own clients
- **Admin Override**: Superusers can access all data

### 4. WebSocket Security
- **Authentication Required**: All WebSocket connections require auth
- **Token Validation**: JWT tokens validated on connection
- **Graceful Rejection**: Unauthenticated connections rejected with reason

## Environment Configuration

### Required Variables
```bash
# JWT secret key (minimum 32 characters)
AUTH_SECRET_KEY=your-super-secret-jwt-key-here-make-it-long-and-random

# Admin user credentials
ADMIN_PASSWORD=your-secure-admin-password
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@example.com
```

### Optional Variables
```bash
# Cookie security (set to true for HTTPS)
COOKIE_SECURE=false

# Google OAuth (optional)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

## API Endpoints

### Authentication
- `POST /auth/jwt/login` - JWT token authentication
- `POST /auth/cookie/login` - Cookie-based authentication
- `POST /auth/logout` - Logout (clear cookies)
- `GET /auth/google/login` - Google OAuth login (if enabled)

### User Management
- `POST /api/create_user` - Create new user (admin only)
- `GET /api/users/me` - Get current user info
- `PATCH /api/users/me` - Update user profile

### WebSocket Endpoints
- `ws://host/ws` - Opus audio stream with auth
- `ws://host/ws_pcm` - PCM audio stream with auth

## Error Handling

### Authentication Errors
- **401 Unauthorized**: Invalid credentials or expired token
- **403 Forbidden**: Insufficient permissions
- **422 Validation Error**: Invalid request format

### WebSocket Errors
- **1008 Policy Violation**: Authentication required
- **1011 Server Error**: Internal authentication error

## Best Practices

### 1. Client Implementation
```python
# Prefer user_id for programmatic access
AUTH_USERNAME = "abc123"  # 6-character user_id
AUTH_PASSWORD = "secure_password"

# Use single AUTH_USERNAME variable
# System auto-detects if it's email or user_id
```

### 2. Token Management
```python
# Store tokens securely
# Refresh tokens before expiry
# Handle 401 errors gracefully
```

### 3. Production Deployment
```bash
# Use strong secrets
AUTH_SECRET_KEY=$(openssl rand -base64 32)

# Enable HTTPS
COOKIE_SECURE=true

# Use environment variables
# Never commit secrets to version control
```

### 4. Admin User Setup
```bash
# Create admin during startup
ADMIN_PASSWORD=secure_admin_password
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@yourdomain.com
```

## Troubleshooting

### Common Issues

#### 1. Authentication Failures
```bash
# Check credentials
curl -X POST "http://localhost:8000/auth/jwt/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test&password=test"

# Verify user exists
# Check password is correct
# Ensure user is active
```

#### 2. WebSocket Connection Issues
```javascript
// Check token validity
// Verify URL format
// Test with curl first
```

#### 3. Admin User Creation
```bash
# Check logs for admin creation
docker compose logs friend-backend | grep -i admin

# Verify environment variables
echo $ADMIN_PASSWORD
```

### Debug Commands
```bash
# Check user database
docker exec -it mongo-container mongosh friend-lite

# View authentication logs
docker compose logs friend-backend | grep -i auth

# Test API endpoints
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/users/me
```

## Migration Guide

### From Basic Auth to fastapi-users

1. **Update Environment Variables**
   ```bash
   # Old
   AUTH_EMAIL=user@example.com
   AUTH_USER_ID=abc123
   
   # New
   AUTH_USERNAME=abc123  # Can be email or user_id
   ```

2. **Update Client Code**
   ```python
   # Old
   username = AUTH_USER_ID if AUTH_USER_ID else AUTH_EMAIL
   
   # New
   username = AUTH_USERNAME
   ```

3. **Test Authentication**
   ```bash
   # Verify both email and user_id work
   curl -X POST "http://localhost:8000/auth/jwt/login" \
     -d "username=user@example.com&password=pass"
   
   curl -X POST "http://localhost:8000/auth/jwt/login" \
     -d "username=abc123&password=pass"
   ```

## Advanced Features

### 1. Role-Based Access Control
```python
# Regular user - can only access own data
@app.get("/api/data")
async def get_data(user: User = Depends(current_active_user)):
    return get_user_data(user.user_id)

# Admin user - can access all data
@app.get("/api/admin/data")
async def get_all_data(user: User = Depends(current_superuser)):
    return get_all_data()
```

### 2. OAuth Integration
```python
# Google OAuth with automatic user creation
# Users can link Google accounts to existing accounts
# Automatic email verification for OAuth users
```

### 3. Multi-Device Support
```python
# Single user, multiple devices
# Client IDs: user123-phone, user123-tablet, user123-desktop
# Separate conversation streams per device
# Unified user dashboard
```

This authentication system provides enterprise-grade security with developer-friendly APIs, supporting both simple email/password authentication and modern OAuth flows while maintaining proper data isolation and user management capabilities.