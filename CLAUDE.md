# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Friend-Lite is an AI-powered wearable ecosystem for audio capture, transcription, memory extraction, and action item detection. The system features real-time audio streaming from OMI devices via Bluetooth, intelligent conversation processing, and a comprehensive web dashboard for management.

## Development Commands

### Backend Development (Advanced Backend - Primary)
```bash
cd backends/advanced-backend

# Start full stack with Docker
docker compose up --build -d

# Development with live reload
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
sudo rm -rf ./audio_chunks/ ./mongo_data/ ./qdrant_data/
```

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
- **Audio Pipeline**: Real-time Opus/PCM → Deepgram WebSocket transcription → memory extraction
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

1. **Audio Ingestion**: OMI devices stream Opus audio via WebSocket with JWT auth
2. **Real-time Processing**: Per-client queues handle transcription and buffering
3. **Conversation Storage**: Transcripts saved to MongoDB `audio_chunks` collection with segments array
4. **Conversation Management**: Automatic timeout-based conversation segmentation
5. **Memory Extraction**: Background LLM processing (decoupled from conversation storage)
6. **Action Items**: Automatic task detection with "Simon says" trigger phrases
7. **Audio Optimization**: Speech segment extraction removes silence automatically

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