# API Reference

## Health & Status Endpoints
- **GET /health**: Basic application health check
- **GET /readiness**: Service dependency validation (MongoDB, Qdrant, etc.)
- **GET /api/metrics**: System metrics and debug tracker status (Admin only)
- **GET /api/processor/status**: Processor queue status and health (Admin only)
- **GET /api/processor/tasks**: All active processing tasks (Admin only)
- **GET /api/processor/tasks/{client_id}**: Processing task status for specific client (Admin only)

## WebSocket Endpoints
- **WS /ws_pcm**: Primary audio streaming endpoint (Wyoming protocol + raw PCM fallback)
- **WS /ws**: Simple audio streaming endpoint (Opus packets + Wyoming audio-chunk events)

## Memory & Conversation Debugging
- **GET /api/admin/memories**: All memories across all users with debug stats (Admin only)
- **GET /api/memories/unfiltered**: User's memories without filtering
- **GET /api/memories/search**: Semantic memory search with relevance scoring
- **GET /api/conversations**: User's conversations with transcripts
- **GET /api/conversations/{conversation_id}**: Specific conversation details
- **POST /api/conversations/{conversation_id}/reprocess-transcript**: Re-run transcript processing
- **POST /api/conversations/{conversation_id}/reprocess-memory**: Re-extract memories with different parameters
- **GET /api/conversations/{conversation_id}/versions**: Get all transcript and memory versions
- **POST /api/conversations/{conversation_id}/activate-transcript**: Switch to a different transcript version
- **POST /api/conversations/{conversation_id}/activate-memory**: Switch to a different memory version

## Client Management
- **GET /api/clients/active**: Currently active WebSocket clients
- **GET /api/users**: List all users (Admin only)

## File Processing
- **POST /api/process-audio-files**: Upload and process audio files (Admin only)
  - Note: Processes files sequentially, may timeout for large files
  - Client timeout: 5 minutes, Server processing: up to 3x audio duration + 60s
  - Example usage:
    ```bash
    # Step 1: Read .env file for ADMIN_EMAIL and ADMIN_PASSWORD
    # Step 2: Get auth token
    # Step 3: Use token in file upload
    curl -X POST \
      -H "Authorization: Bearer YOUR_TOKEN_HERE" \
      -F "files=@/path/to/audio.wav" \
      -F "device_name=test-upload" \
      http://localhost:8000/api/process-audio-files
    ```

## Authentication
- **POST /auth/jwt/login**: Email-based login (returns JWT token)
- **GET /users/me**: Get current authenticated user
- **GET /api/auth/config**: Authentication configuration

## Step-by-Step API Testing Guide

When testing API endpoints that require authentication, follow these steps:

### Step 1: Read credentials from .env file
```bash
# Use the Read tool to view the .env file and identify credentials
# Look for:
# ADMIN_EMAIL=admin@example.com
# ADMIN_PASSWORD=your-password-here
```

### Step 2: Get authentication token
```bash
curl -s -X POST \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin@example.com&password=your-password-here" \
  http://localhost:8000/auth/jwt/login
```
This returns:
```json
{"access_token":"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...","token_type":"bearer"}
```

### Step 3: Use the token in API calls
```bash
# Extract the token from the response above and use it:
curl -s -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  http://localhost:8000/api/conversations

# For reprocessing endpoints:
curl -s -X POST \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  http://localhost:8000/api/conversations/{conversation_id}/reprocess-transcript
```

**Important**: Always read the .env file first using the Read tool rather than using shell commands like `grep` or `cut`. This ensures you see the exact values and can copy them accurately.

### Step 4: Testing Reprocessing Endpoints
Once you have the auth token, you can test the reprocessing functionality:

```bash
# Get list of conversations to find a conversation_id
curl -s -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/conversations

# Test transcript reprocessing (uses conversation_id)
curl -s -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  http://localhost:8000/api/conversations/YOUR_CONVERSATION_ID/reprocess-transcript

# Test memory reprocessing (uses conversation_id and transcript_version_id)
curl -s -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"transcript_version_id": "VERSION_ID"}' \
  http://localhost:8000/api/conversations/YOUR_CONVERSATION_ID/reprocess-memory

# Get transcript and memory versions
curl -s -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/conversations/YOUR_CONVERSATION_ID/versions

# Activate a specific transcript version
curl -s -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"transcript_version_id": "VERSION_ID"}' \
  http://localhost:8000/api/conversations/YOUR_CONVERSATION_ID/activate-transcript

# Activate a specific memory version
curl -s -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"memory_version_id": "VERSION_ID"}' \
  http://localhost:8000/api/conversations/YOUR_CONVERSATION_ID/activate-memory
```

## Development Reset Endpoints
Useful endpoints for resetting state during development:

### Data Cleanup
- **DELETE /api/admin/memory/delete-all**: Delete all memories for the current user
- **DELETE /api/memories/{memory_id}**: Delete a specific memory
- **DELETE /api/conversations/{conversation_id}**: Delete a specific conversation (keeps original audio file in audio_chunks)
- **DELETE /api/chat/sessions/{session_id}**: Delete a chat session and all its messages
- **DELETE /api/users/{user_id}**: Delete a user (Admin only)
  - Optional query params: `delete_conversations=true`, `delete_memories=true`

### Quick Reset Commands
```bash
# Reset all data (development only)
cd backends/advanced
sudo rm -rf data/

# Reset Docker volumes
docker compose down -v
docker compose up --build -d
```