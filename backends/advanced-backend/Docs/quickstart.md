# Friend-Lite Backend Quickstart Guide

> 📖 **New to friend-lite?** This is your starting point! After reading this, continue with [architecture.md](./architecture.md) for technical details.

## Overview

Friend-Lite is a real-time conversation processing system that captures audio, transcribes speech, extracts memories, and generates action items. The system includes a FastAPI backend with WebSocket audio streaming, a Streamlit web dashboard, and comprehensive user management.

**Core Implementation**: See `src/main.py` for the complete FastAPI application and WebSocket handling.

## Prerequisites

- Docker and Docker Compose
- (Optional) Deepgram API key /Local ASR for cloud transcription
- (Optional) Ollama/OpenAI for local Speech-to-Text processing

## Quick Start

### 1. Environment Setup

Create a `.env` file in `backends/advanced-backend/`:

```bash
# Required Authentication
AUTH_SECRET_KEY=your-super-secret-jwt-key-here-make-it-long-and-random
ADMIN_PASSWORD=your-secure-admin-password

# Optional Configuration
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@example.com
COOKIE_SECURE=false

# Required for Memory Processing (if using Ollama)
OLLAMA_BASE_URL=http://ollama:11434 # if within same compose build, can access by container name

# ASR Configuration (choose one)
DEEPGRAM_API_KEY=your-deepgram-api-key
# OR for self-hosted ASR
OFFLINE_ASR_TCP_URI=tcp://host.docker.internal:8765 # if within same compose build, can access by container name, or here, for example another docker container running on the same machine but different compose (thus network)

# Optional Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Optional Services
HF_TOKEN=your-huggingface-token # For speaker service
NGROK_AUTHTOKEN=your-ngrok-token
```

### 2. Start the System

**Recommended: Docker Compose (using uv)**
```bash
cd backends/advanced-backend
docker compose up --build -d
```

This starts:
- **Backend API**: `http://localhost:8000`
- **Web Dashboard**: `http://localhost:8501`
- **MongoDB**: `localhost:27017`
- **Qdrant**: `localhost:6333`
- (optional) **Ollama**: # commented out

**Implementation**: See `docker-compose.yml` for complete service configuration and `src/main.py` for FastAPI application setup.

### 3. Optional: Start ASR Service

For self-hosted speech recognition, see instructions in `extras/asr-services/`:

## Using the System

### Web Dashboard

1. Open `http://localhost:8501`
2. **Login** using the sidebar:
   - **Admin**: `admin@example.com` / `your-admin-password`
   - **Google OAuth** (if configured)
   - **Create new users** via admin interface

### Dashboard Features

- **Conversations**: View audio recordings, transcripts, and cropped audio
- **Memories**: Search extracted conversation memories
- **Action Items**: Manage automatically detected tasks
- **User Management**: Create/delete users and their data
- **Client Management**: View active connections and close conversations

### Audio Client Connection

Connect audio clients via WebSocket with authentication:

**WebSocket URLs:**
```javascript
// Opus audio stream
ws://your-server-ip:8000/ws?token=YOUR_JWT_TOKEN&device_name=YOUR_DEVICE_NAME

// PCM audio stream  
ws://your-server-ip:8000/ws_pcm?token=YOUR_JWT_TOKEN&device_name=YOUR_DEVICE_NAME
```

**Authentication Methods:**
The system supports authentication with either email or 6-character user_id. The backend automatically detects the format:

```bash
# Login with email (admin user)
curl -X POST "http://localhost:8000/auth/jwt/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin@example.com&password=your-admin-password"

# Login with user_id (6-character alphanumeric)
curl -X POST "http://localhost:8000/auth/jwt/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=abc123&password=user-password"

# Response: {"access_token": "eyJhbGciOiJIUzI1NiIs...", "token_type": "bearer"}
```

**Authentication Flow:**
1. **User Registration**: Admin creates users via API or dashboard
2. **Login**: Users authenticate with email or user_id
3. **Token Usage**: Include JWT token in API calls and WebSocket connections
4. **Data Access**: Users can only access their own data (admins see all)

For detailed authentication documentation, see [`auth.md`](./auth.md).

**Create User Account:**
```bash
export ADMIN_TOKEN="your-admin-token"

# Create user with auto-generated user_id
curl -X POST "http://localhost:8000/api/create_user" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "userpass", "display_name": "John Doe"}'

# Create user with specific user_id (6 chars, lowercase alphanumeric)
curl -X POST "http://localhost:8000/api/create_user" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "userpass", "user_id": "user01", "display_name": "John Doe"}'
```

**Client ID Format:**
The system automatically generates client IDs as `user_id-device_name` (e.g., `abc123-phone`, `admin-desktop`). This ensures proper user-client association and data isolation.

## Add Existing Data

### Audio File Upload & Processing

The system supports processing existing audio files through the file upload API. This allows you to import and process pre-recorded conversations without requiring a live WebSocket connection.

**Upload and Process WAV Files:**
```bash
export USER_TOKEN="your-jwt-token"

# Upload single WAV file
curl -X POST "http://localhost:8000/api/process-audio-files" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -F "files=@/path/to/audio.wav" \
  -F "device_name=file_upload"

# Upload multiple WAV files
curl -X POST "http://localhost:8000/api/process-audio-files" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -F "files=@/path/to/recording1.wav" \
  -F "files=@/path/to/recording2.wav" \
  -F "device_name=import_batch"
```

**Response Example:**
```json
{
  "message": "Successfully processed 2 audio files",
  "processed_files": [
    {
      "filename": "recording1.wav",
      "sample_rate": 16000,
      "channels": 1,
      "duration_seconds": 120.5,
      "size_bytes": 3856000
    },
    {
      "filename": "recording2.wav", 
      "sample_rate": 44100,
      "channels": 2,
      "duration_seconds": 85.2,
      "size_bytes": 7532800
    }
  ],
  "client_id": "user01-import_batch"
}
```

## System Features

### Audio Processing
- **Real-time streaming**: WebSocket audio ingestion
- **Multiple formats**: Opus and PCM audio support
- **Per-client processing**: Isolated conversation management
- **Speech detection**: Automatic silence removal
- **Audio cropping**: Extract only speech segments

**Implementation**: See `src/main.py:1562+` for WebSocket endpoints and `src/main.py:895-1340` for audio processing pipeline.

### Transcription Options
- **Deepgram API**: Cloud-based, high accuracy (recommended)
- **Self-hosted ASR**: Local Wyoming protocol services
- **Real-time processing**: Live transcription with conversation tracking

### Conversation Management
- **Automatic chunking**: 60-second audio segments
- **Conversation timeouts**: Auto-close after 1.5 minutes of silence
- **Speaker identification**: Track multiple speakers per conversation
- **Manual controls**: Close conversations via API or dashboard

### Memory & Intelligence
- **User-centric storage**: All memories and action items keyed by database user_id
- **Memory extraction**: Automatic conversation summaries using LLM
- **Semantic search**: Vector-based memory retrieval
- **Action item detection**: Automatic task extraction with "Simon says" triggers
- **Configurable extraction**: YAML-based configuration for memory and action item extraction
- **Debug tracking**: SQLite-based tracking of transcript → memory/action item conversion
- **Client metadata**: Device information preserved for debugging and reference
- **User isolation**: All data scoped to individual users with multi-device support

**Implementation**: 
- **Memory System**: `src/memory/memory_service.py` + `main.py:1047-1065, 1163-1195`
- **Action Items**: `src/action_items_service.py` + `main.py:1341-1378`
- **Configuration**: `memory_config.yaml` + `src/memory_config_loader.py`
- **Debug Tracking**: `src/memory_debug.py` + API endpoints at `/api/debug/memory/*`

### Authentication & Security
- **Flexible Authentication**: Login with either email or 6-character user_id
- **JWT tokens**: Secure API and WebSocket authentication with 1-hour expiration
- **Google OAuth**: Optional social login integration
- **Role-based access**: Admin vs regular user permissions
- **Data isolation**: Users can only access their own data
- **Client ID Management**: Automatic client-user association via `user_id-device_name` format
- **Multi-device support**: Single user can connect multiple devices
- **Security headers**: Proper CORS, cookie security, and token validation

**Implementation**: See `src/auth.py` for authentication logic, `src/users.py` for user management, and [`auth.md`](./auth.md) for comprehensive documentation.

## Verification

```bash
# System health check
curl http://localhost:8000/health

# Web dashboard
open http://localhost:8501

# View active clients (requires auth token)
curl -H "Authorization: Bearer your-token" http://localhost:8000/api/active_clients
```

## HAVPE Relay Configuration

For ESP32 audio streaming using the HAVPE relay (`extras/havpe-relay/`):

```bash
# Environment variables for HAVPE relay
export AUTH_USERNAME="abc123"                 # Can be email or user_id
export AUTH_PASSWORD="your-password"
export DEVICE_NAME="havpe"                    # Device identifier

# Run the relay
cd extras/havpe-relay
python main.py --backend-url http://your-server:8000 --backend-ws-url ws://your-server:8000
```

The relay will automatically:
- Authenticate using `AUTH_USERNAME` (email or 6-character user_id)
- Generate client ID as `user_id-havpe`
- Forward ESP32 audio to the backend with proper authentication
- Handle token refresh and reconnection

## Development tip
docker compose down && docker compose up --build -d && docker compose logs friend-backend -f
lmao
Once the build is cached it takes 29 seconds on my rasp pi 4, thats enough delay I think. 
If you would like to use the debugger, you can use the following command:
uv sync --group (whatever group you want to sync) 
(for example, deepgram, etc.)

## Troubleshooting

**Service Issues:**
- Check logs: `docker compose logs friend-backend`
- Restart services: `docker compose restart`
- View all services: `docker compose ps`

**Authentication Issues:**
- Verify `AUTH_SECRET_KEY` is set and long enough
- Check admin credentials match `.env` file
- For Google OAuth, verify client ID/secret are correct

**ASR Issues:**
- **Deepgram**: Verify API key is valid
- **Self-hosted**: Ensure ASR service is running on port 8765
- Check ASR connection in health endpoint

**Memory Issues:**
- Ensure Ollama is running and model is pulled
- Check Qdrant connection in health endpoint
- Memory processing happens at conversation end

**Connection Issues:**
- Use server's IP address, not localhost for mobile clients
- Ensure WebSocket connections include authentication token
- Check firewall/port settings for remote connections

## Data Architecture

The friend-lite backend uses a **user-centric data architecture**:

- **All memories and action items are keyed by database user_id** (not client_id)
- **Client information is stored in metadata** for reference and debugging
- **User email is included** for easy identification in admin interfaces
- **Multi-device support**: Users can access their data from any registered device

For detailed information, see [User Data Architecture](user-data-architecture.md).

## Memory & Action Item Configuration

The system supports configurable memory and action item extraction via `memory_config.yaml`:

### Basic Configuration
```yaml
# Enable/disable different extraction types
memory_extraction:
  enabled: true
  
action_item_extraction:
  enabled: true
  trigger_phrases:
    - "simon says"
    - "action item"
    - "todo"
    - "follow up"
```

### Debug & Monitoring
```yaml
debug:
  enabled: true
  db_path: "/app/debug/memory_debug.db"
  log_level: "INFO"
```

### API Endpoints for Debugging
- `GET /api/debug/memory/stats` - Processing statistics
- `GET /api/debug/memory/sessions` - Recent memory sessions
- `GET /api/debug/memory/session/{audio_uuid}` - Detailed session info
- `GET /api/debug/memory/config` - Current configuration
- `GET /api/debug/memory/pipeline/{audio_uuid}` - Pipeline trace

**Implementation**: See `src/memory_debug_api.py` for debug endpoints and `../MEMORY_DEBUG_IMPLEMENTATION.md` for complete debug system documentation.

## Next Steps

- **Configure Google OAuth** for easy user login
- **Set up Ollama** for local memory processing
- **Deploy ASR service** for self-hosted transcription
- **Connect audio clients** using the WebSocket API
- **Explore the dashboard** to manage conversations and users
- **Review the user data architecture** for understanding data organization
- **Customize memory extraction** by editing `memory_config.yaml`
- **Monitor processing performance** using debug API endpoints