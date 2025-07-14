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
uv run pytest test_endpoints.py  # Integration tests
uv run pytest test_failure_recovery.py  # Failure recovery tests
uv run pytest test_memory_debug.py  # Memory debug tests

# Environment setup
cp .env.template .env  # Configure environment variables

# Reset data (development)
sudo rm -rf ./audio_chunks/ ./mongo_data/ ./qdrant_data/
```

### Mobile App Development
```bash
cd friend-lite

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
  - `src/failure_recovery/`: Robust processing pipeline with SQLite tracking
  - `webui/streamlit_app.py`: Web dashboard for conversation and user management

### Key Components
- **Audio Pipeline**: Real-time Opus/PCM → Deepgram WebSocket transcription → memory extraction
- **Transcription**: Deepgram Nova-3 model with Wyoming ASR fallback, auto-reconnection
- **Authentication**: Email-based login with MongoDB ObjectId user system
- **Client Management**: Auto-generated client IDs as `{user_id_suffix}-{device_name}`, centralized ClientManager
- **Data Storage**: MongoDB (conversations), Qdrant (vector memory), SQLite (failure recovery)
- **Web Interface**: Streamlit dashboard with authentication and real-time monitoring

### Service Dependencies
```yaml
Required:
  - MongoDB: User data and conversations
  - FastAPI Backend: Core audio processing

Recommended:
  - Qdrant: Vector storage for semantic memory
  - Ollama: LLM for memory extraction and action items
  - Deepgram: Primary transcription service (Nova-3 WebSocket)
  - Wyoming ASR: Fallback transcription service (offline)

Optional:
  - Speaker Recognition: Voice identification service
  - Nginx Proxy: Load balancing and routing
```

## Data Flow Architecture

1. **Audio Ingestion**: OMI devices stream Opus audio via WebSocket with JWT auth
2. **Real-time Processing**: Per-client queues handle transcription and buffering
3. **Conversation Management**: Automatic timeout-based conversation segmentation
4. **Memory Extraction**: LLM processes completed conversations for semantic storage
5. **Action Items**: Automatic task detection with "Simon says" trigger phrases
6. **Audio Optimization**: Speech segment extraction removes silence automatically

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
- **Integration Tests**: `test_endpoints.py` covers API functionality
- **Unit Tests**: Individual service tests in `tests/` directory
- **System Tests**: `test_failure_recovery.py` and `test_memory_debug.py`

### Code Style
- **Python**: Black formatter with 100-character line length, isort for imports
- **TypeScript**: Standard React Native conventions

### Health Monitoring
The system includes comprehensive health checks:
- `/readiness`: Service dependency validation
- `/health`: Basic application status
- Failure recovery system with SQLite tracking
- Memory debug system for transcript processing monitoring

### Cursor Rule Integration
Project includes `.cursor/rules/always-plan-first.mdc` requiring understanding before coding. Always explain the task and confirm approach before implementation.