# Authentication Setup Guide

This backend now supports Google OAuth authentication using fastapi-users. Both HTTP and WebSocket endpoints are protected.

## 🔧 Required Environment Variables

Add these to your `.env` file:

```bash
# Authentication Configuration (REQUIRED)
AUTH_SECRET_KEY=your-super-secret-key-change-me-in-production
GOOGLE_CLIENT_ID=your-google-oauth-client-id
GOOGLE_CLIENT_SECRET=your-google-oauth-client-secret
```

## 🏗️ Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Google+ API
4. Create OAuth 2.0 credentials:
   - Application type: Web application
   - Authorized redirect URIs: `http://localhost:8000/auth/google/callback` (adjust for your domain)
5. Copy the Client ID and Client Secret to your environment variables

## 🚀 Installation

Install the new dependencies:

```bash
pip install -r requirements.txt
```

## 🔐 Authentication Endpoints

The following authentication endpoints are now available:

### Google OAuth Flow
- `GET /auth/google/login` - Redirect to Google OAuth
- `GET /auth/google/callback` - OAuth callback (sets cookie & returns JWT)

### Cookie Authentication (for browsers)
- `POST /auth/cookie/login` - Email/password login (sets cookie)
- `POST /auth/cookie/logout` - Logout (clears cookie)

### JWT Token Authentication (for API clients)
- `POST /auth/jwt/login` - Email/password login (returns JWT)

### User Registration
- `POST /auth/register` - Register new user with email/password

## 🔒 Protected Endpoints

### WebSocket Endpoints
Both WebSocket endpoints now require authentication:

- `/ws` - Opus audio streaming (requires auth)
- `/ws_pcm` - PCM audio streaming (requires auth)

**For browsers:** Authentication cookie is sent automatically with WebSocket connections.

**For programmatic clients:** Include JWT token in query string or Authorization header:
```
ws://localhost:8000/ws?token=your-jwt-token
# OR
ws://localhost:8000/ws (with Authorization: Bearer your-jwt-token header)
```

### Protected HTTP Endpoints
The following endpoints now require authentication:

- `POST /api/create_user` - Create new user
- `DELETE /api/delete_user` - Delete user and optionally their data

Other endpoints remain public for now. You can protect additional endpoints by adding the `current_active_user` dependency.

## 🔄 Authentication Flow

### For Web UI (Cookie-based)
1. User visits `/auth/google/login`
2. Redirected to Google OAuth consent screen
3. After consent, redirected to `/auth/google/callback`
4. Authentication cookie is set automatically
5. WebSocket connections work automatically

### For API Clients (Token-based)
1. Call `POST /auth/jwt/login` with email/password or get token from Google flow
2. Include JWT token in Authorization header: `Bearer your-jwt-token`
3. Or append to WebSocket URL: `ws://localhost:8000/ws?token=your-jwt-token`

## 📊 User Database

User data is stored in a new MongoDB collection called `fastapi_users` using Beanie ODM, separate from your existing collections. The user model includes:

- Standard fields: id, email, hashed_password, is_active, is_superuser, is_verified
- Custom fields: display_name, profile_picture
- OAuth accounts for Google sign-in

## 🛠️ Development Notes

- Set `AUTH_SECRET_KEY` to a secure random string in production
- For local development without HTTPS, you may need to set `cookie_secure=False` in `auth.py`
- The authentication system runs alongside your existing motor-based MongoDB collections
- User management is handled by fastapi-users, while your application data remains in the existing collections

## 🔧 Customization

To protect additional endpoints, add the authentication dependency:

```python
from auth import current_active_user

@app.get("/api/my-protected-endpoint")
async def my_endpoint(user: User = Depends(current_active_user)):
    # Endpoint logic here
    pass
```

To make an endpoint optional (user can be None):

```python
from auth import optional_current_user

@app.get("/api/my-optional-endpoint")
async def my_endpoint(user: Optional[User] = Depends(optional_current_user)):
    # user will be None if not authenticated
    pass
``` 