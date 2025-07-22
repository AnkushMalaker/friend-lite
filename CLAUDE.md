# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Friend-Lite is at the core an AI-powered personal system - various devices, incuding but not limited to wearables from OMI can be used for at the very least audio capture, speaker specific transcription, memory extraction and retriaval.
On top of that - it is being designed to support other services, that can help a user with these inputs such as reminders, action items, personal diagnosis etc.

This supports a comprehensive web dashboard for management.

## Development Commands

### Backend Development (Advanced Backend - Primary)
```bash
cd backends/advanced-backend

# Start full stack with Docker
docker compose up --build -d

uv run python src/main.py

# Code formatting and linting
uv run black src/
uv run isort src/

# Run tests
uv run pytest
uv run pytest tests/test_memory_service.py  # Single test file
uv run pytest test_integration.py  # End-to-end integration test (requires Docker)
uv run pytest test_endpoints.py  # Integration tests
uv run pytest test_memory_debug.py  # Memory debug tests

# Environment setup
cp .env.template .env  # Configure environment variables

# Reset data (development)
sudo rm -rf backends/advanced-backend/data/
```

### Integration Test Development and Debugging

#### Running Integration Tests
These needs keys from .env file if running locally and not through CI.
This can be done by exporting these keys before hand - 
```bash
# Basic integration test (requires API keys in .env)
source .env && export DEEPGRAM_API_KEY && export OPENAI_API_KEY && uv run pytest tests/test_integration.py::test_full_pipeline_integration -v -s

# For debugging: Use cached mode to keep containers running
# 1. Edit tests/test_integration.py: Set CACHED_MODE = True
# 2. Run test (containers will persist after test ends)
uv run pytest tests/test_integration.py::test_full_pipeline_integration -v -s --tb=short

# Check running test containers
docker ps
docker logs advanced-backend-friend-backend-test-1 --tail=50
docker logs advanced-backend-mongo-test-1 --tail=20
docker logs advanced-backend-qdrant-test-1 --tail=20

# Clean up test containers
docker compose -f docker-compose-test.yml down -v
```

#### Integration Test Iteration Methodology
1. **Set CACHED_MODE = True** in `tests/test_integration.py` for debugging. If CACHE_MODE is false, the test containers will be created fresh before the test, and removed after - so they won't be running after the test to view logs/debug.
2. **Run the test** - containers will start and persist even if test times out
3. **Check container logs** to see where the test is hanging:
   ```bash
   docker logs advanced-backend-friend-backend-test-1 --tail=100
   ```
4. **Test API endpoints manually** while containers are running:
   ```bash
   curl -X GET http://localhost:8001/health
   curl -X POST http://localhost:8001/auth/jwt/login \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=test-admin@example.com&password=test-admin-password-123"
   ```
5. **Exec into containers** for detailed debugging:
   ```bash
   docker exec -it advanced-backend-friend-backend-test-1 /bin/bash
   ```
6. **Clean up** when done:
   ```bash
   docker compose -f docker-compose-test.yml down -v
   ```
7. **Set CACHED_MODE = False** when committing changes for CI compatibility

### Mobile App Development
```bash
cd app

# Start Expo development server
npm start

# Platform-specific builds
npm run android
npm run ios
npm run web
```

### Additional Services
```bash
# ASR Services
cd extras/asr-services
docker compose up moonshine  # Offline ASR with Moonshine
docker compose up parakeet   # Offline ASR with Parakeet

# Speaker Recognition
cd extras/speaker-recognition
docker compose up --build

# HAVPE Relay (ESP32 bridge)
cd extras/havpe-relay
docker compose up --build
```

## Architecture Overview

### Core Structure
- **backends/advanced-backend/**: Primary FastAPI backend with real-time audio processing
  - `src/main.py`: Central FastAPI application with WebSocket audio streaming
  - `src/auth.py`: Email-based authentication with JWT tokens
  - `src/memory/`: LLM-powered conversation memory system using mem0
  - `webui/streamlit_app.py`: Web dashboard for conversation and user management

### Key Components
- **Audio Pipeline**: Real-time Opus/PCM → Application-level processing → Deepgram transcription → memory extraction
- **Wyoming Protocol**: WebSocket communication uses Wyoming protocol (JSONL + binary) for structured audio sessions
- **Application-Level Processing**: Centralized processors for audio, transcription, memory, and cropping
- **Task Management**: BackgroundTaskManager tracks all async tasks to prevent orphaned processes
- **Transcription**: Deepgram Nova-3 model with Wyoming ASR fallback, auto-reconnection
- **Authentication**: Email-based login with MongoDB ObjectId user system
- **Client Management**: Auto-generated client IDs as `{user_id_suffix}-{device_name}`, centralized ClientManager
- **Data Storage**: MongoDB (`audio_chunks` collection for conversations), Qdrant (vector memory)
- **Web Interface**: Streamlit dashboard with authentication and real-time monitoring

### Service Dependencies
```yaml
Required:
  - MongoDB: User data and conversations
  - FastAPI Backend: Core audio processing
  - LLM Service: Memory extraction and action items (OpenAI or Ollama)

Recommended:
  - Qdrant: Vector storage for semantic memory
  - Deepgram: Primary transcription service (Nova-3 WebSocket)
  - Wyoming ASR: Fallback transcription service (offline)

Optional:
  - Speaker Recognition: Voice identification service
  - Nginx Proxy: Load balancing and routing
```

## Data Flow Architecture

1. **Audio Ingestion**: OMI devices stream audio via WebSocket using Wyoming protocol with JWT auth
2. **Wyoming Protocol Session Management**: Clients send audio-start/audio-stop events for session boundaries
3. **Application-Level Processing**: Global queues and processors handle all audio/transcription/memory tasks
4. **Conversation Storage**: Transcripts saved to MongoDB `audio_chunks` collection with segments array
5. **Conversation Management**: Session-based conversation segmentation using Wyoming protocol events
6. **Memory Extraction**: Background LLM processing (decoupled from conversation storage)
7. **Action Items**: Automatic task detection with "Simon says" trigger phrases
8. **Audio Optimization**: Speech segment extraction removes silence automatically
9. **Task Tracking**: BackgroundTaskManager ensures proper cleanup of all async operations

### Database Schema Details
- **Conversations**: Stored in `audio_chunks` collection (not `conversations`)
- **Transcript Format**: Array of segment objects with `text`, `speaker`, `start`, `end` fields
- **Service Decoupling**: Conversation creation independent of memory processing
- **Error Isolation**: Memory service failures don't affect conversation storage

## Authentication & Security

- **User System**: Email-based authentication with MongoDB ObjectId user IDs
- **Client Registration**: Automatic `{objectid_suffix}-{device_name}` format
- **Data Isolation**: All data scoped by user_id with efficient permission checking
- **API Security**: JWT tokens required for all endpoints and WebSocket connections
- **Admin Bootstrap**: Automatic admin account creation with ADMIN_EMAIL/ADMIN_PASSWORD

## Configuration

### Required Environment Variables
```bash
AUTH_SECRET_KEY=your-super-secret-jwt-key-here
ADMIN_PASSWORD=your-secure-admin-password
ADMIN_EMAIL=admin@example.com

# For integration tests (CI/CD)
DEEPGRAM_API_KEY=your-deepgram-key-here
OPENAI_API_KEY=your-openai-key-here

# CI Environment Requirements:
# - Docker must be available (GitHub Actions ubuntu-latest includes Docker)
# - User must have Docker permissions (no sudo required)
# - For macOS local development, run ./mac-os-docker-ci-quickfix.sh if needed
```

### Optional Service Configuration
```bash
# Transcription (Deepgram primary, Wyoming fallback)
DEEPGRAM_API_KEY=your-deepgram-key-here
OFFLINE_ASR_TCP_URI=tcp://host.docker.internal:8765

# LLM Processing
OLLAMA_BASE_URL=http://ollama:11434

# Vector Storage
QDRANT_BASE_URL=qdrant

# Speaker Recognition
SPEAKER_SERVICE_URL=http://speaker-recognition:8001
```

## Wyoming Protocol Implementation

### Overview
The system uses Wyoming protocol for WebSocket communication between mobile apps and backends. Wyoming is a peer-to-peer protocol for voice assistants that combines JSONL headers with binary audio payloads.

### Protocol Format
```
{JSON_HEADER}\n
<binary_payload>
```

### Supported Events

#### Audio Session Events
- **audio-start**: Signals the beginning of an audio recording session
  ```json
  {"type": "audio-start", "data": {"rate": 16000, "width": 2, "channels": 1}, "payload_length": null}
  ```

- **audio-chunk**: Contains raw audio data with format metadata
  ```json
  {"type": "audio-chunk", "data": {"rate": 16000, "width": 2, "channels": 1}, "payload_length": 320}
  <320 bytes of PCM/Opus audio data>
  ```

- **audio-stop**: Signals the end of an audio recording session
  ```json
  {"type": "audio-stop", "data": {"timestamp": 1234567890}, "payload_length": null}
  ```

### Backend Implementation

#### Advanced Backend (`/ws_pcm`)
- **Full Wyoming Protocol Support**: Parses all Wyoming events for session management
- **Session Tracking**: Only processes audio chunks when session is active (after audio-start)
- **Conversation Boundaries**: Uses audio-start/stop events to define conversation segments
- **Backward Compatibility**: Fallback to raw binary audio for older clients

#### Simple Backend (`/ws`)
- **Minimal Wyoming Support**: Parses audio-chunk events, ignores others
- **Opus Processing**: Handles Opus-encoded audio chunks from Wyoming protocol
- **Graceful Degradation**: Falls back to raw Opus packets for compatibility

### Mobile App Integration

Mobile apps should implement Wyoming protocol for proper session management:

```javascript
// Start audio session
const audioStart = {
  type: "audio-start",
  data: { rate: 16000, width: 2, channels: 1 },
  payload_length: null
};
websocket.send(JSON.stringify(audioStart) + '\n');

// Send audio chunks
const audioChunk = {
  type: "audio-chunk",
  data: { rate: 16000, width: 2, channels: 1 },
  payload_length: audioData.byteLength
};
websocket.send(JSON.stringify(audioChunk) + '\n');
websocket.send(audioData);

// End audio session
const audioStop = {
  type: "audio-stop",
  data: { timestamp: Date.now() },
  payload_length: null
};
websocket.send(JSON.stringify(audioStop) + '\n');
```

### Benefits
- **Clear Session Boundaries**: No timeout-based conversation detection needed
- **Structured Communication**: Consistent protocol across all audio streaming
- **Future Extensibility**: Room for additional event types (pause, resume, metadata)
- **Backward Compatibility**: Works with existing raw audio streaming clients

## Development Notes

### Package Management
- **Backend**: Uses `uv` for Python dependency management (faster than pip)
- **Mobile**: Uses `npm` with React Native and Expo
- **Docker**: Primary deployment method with docker-compose

### Testing Strategy
- **End-to-End Integration**: `test_integration.py` validates complete audio processing pipeline
- **API Integration Tests**: `test_endpoints.py` covers API functionality
- **Unit Tests**: Individual service tests in `tests/` directory (NOT IMPLEMENTED)

### Code Style
- **Python**: Black formatter with 100-character line length, isort for imports
- **TypeScript**: Standard React Native conventions

### Health Monitoring
The system includes comprehensive health checks:
- `/readiness`: Service dependency validation
- `/health`: Basic application status
- Memory debug system for transcript processing monitoring

### Integration Test Infrastructure
- **Test Environment**: `docker-compose-test.yml` provides isolated services on separate ports
- **Test Database**: Uses `test_db` database with isolated collections
- **Service Ports**: Backend (8001), MongoDB (27018), Qdrant (6335/6336), Streamlit (8502)
- **Test Credentials**: Pre-configured in `.env.test` for repeatable testing
- **Ground Truth**: Expected transcript established via `scripts/test_deepgram_direct.py`
- **AI Validation**: OpenAI-powered transcript similarity comparison
- **Test Audio**: 4-minute glass blowing tutorial (`extras/test-audios/DIY*mono*.wav`)

### Cursor Rule Integration
Project includes `.cursor/rules/always-plan-first.mdc` requiring understanding before coding. Always explain the task and confirm approach before implementation.


## Essential Debugging Endpoints

### System Health & Monitoring
- **GET /health**: Basic application health check
- **GET /readiness**: Service dependency validation (MongoDB, Qdrant, etc.)
- **GET /api/metrics**: System metrics and debug tracker status (Admin only)
- **GET /api/processor/status**: Processor queue status and health (Admin only)
- **GET /api/processor/tasks**: All active processing tasks (Admin only)
- **GET /api/processor/tasks/{client_id}**: Processing task status for specific client (Admin only)

### Memory & Conversation Debugging
- **GET /api/admin/memories**: All memories across all users with debug stats (Admin only)
- **GET /api/memories/unfiltered**: User's memories without filtering
- **GET /api/conversations**: User's conversations with transcripts
- **GET /api/conversations/{audio_uuid}**: Specific conversation details

### Client Management
- **GET /api/clients/active**: Currently active WebSocket clients
- **GET /api/users**: List all users (Admin only)

### File Processing
- **POST /api/process-audio-files**: Upload and process audio files (Admin only)
  - Note: Processes files sequentially, may timeout for large files
  - Client timeout: 5 minutes, Server processing: up to 3x audio duration + 60s

### Authentication
- **POST /auth/jwt/login**: Email-based login (returns JWT token)
- **GET /api/auth/config**: Authentication configuration

## Notes for Claude
Check if the src/ is volume mounted. If not, do compose build so that code changes are reflected.