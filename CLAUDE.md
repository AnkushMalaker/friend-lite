# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Friend-Lite is at the core an AI-powered personal system - various devices, incuding but not limited to wearables from OMI can be used for at the very least audio capture, speaker specific transcription, memory extraction and retriaval.
On top of that - it is being designed to support other services, that can help a user with these inputs such as reminders, action items, personal diagnosis etc.

This supports a comprehensive web dashboard for management.

**⚠️ Active Development Notice**: This project is under active development. Do not create migration scripts or assume stable APIs. Only offer suggestions and improvements when requested.

**❌ No Backward Compatibility**: Do NOT add backward compatibility code unless explicitly requested. This includes fallback logic, legacy field support, or compatibility layers. Always ask before adding backward compatibility - in most cases the answer is no during active development.

## Development Commands

### Backend Development (Advanced Backend - Primary)
```bash
cd backends/advanced

# Start full stack with Docker
docker compose up --build -d

uv run python src/main.py

# Code formatting and linting
uv run black src/
uv run isort src/

# Run tests
uv run pytest
uv run pytest tests/test_memory_service.py  # Single test file

# Run integration tests (local script mirrors CI)
./run-test.sh  # Complete integration test suite

# Environment setup
cp .env.template .env  # Configure environment variables

# Reset data (development)
sudo rm -rf backends/advanced/data/
```

### Testing Infrastructure

#### Local Test Scripts
The project includes simplified test scripts that mirror CI workflows:

```bash
# Run all tests from project root
./run-test.sh [advanced-backend|speaker-recognition|all]

# Advanced backend tests only
./run-test.sh advanced-backend

# Speaker recognition tests only
./run-test.sh speaker-recognition

# Run all test suites (default)
./run-test.sh all
```

#### Advanced Backend Integration Tests
```bash
cd backends/advanced

# Requires .env file with DEEPGRAM_API_KEY and OPENAI_API_KEY
cp .env.template .env  # Configure API keys

# Run full integration test suite
./run-test.sh

# Manual test execution (for debugging)
source .env && export DEEPGRAM_API_KEY && export OPENAI_API_KEY
uv run pytest tests/test_integration.py::test_full_pipeline_integration -v -s
```

#### Speaker Recognition Tests
```bash
cd extras/speaker-recognition

# Requires .env file with HF_TOKEN and DEEPGRAM_API_KEY
cp .env.template .env  # Configure tokens

# Run speaker recognition test suite
./run-test.sh
```

#### Test Script Features
- **Environment Compatibility**: Works with both local .env files and CI environment variables
- **Simplified Configuration**: Uses environment variables directly, no temporary .env.test files
- **Docker Cleanup**: Uses lightweight Alpine container for reliable permission-free cleanup
- **Automatic Cleanup**: Stops and removes test containers after execution
- **Colored Output**: Clear progress indicators and error reporting
- **Timeout Protection**: 15-minute timeout for advanced backend, 30-minute for speaker recognition
- **Fresh Testing**: Uses CACHED_MODE=False for clean test environments

#### Debugging Integration Tests
For advanced debugging, you can still use the cached mode approach:

1. **Edit tests/test_integration.py**: Set CACHED_MODE = True
2. **Run test manually**: `uv run pytest tests/test_integration.py -v -s --tb=short`
3. **Debug containers**: `docker logs advanced-backend-friend-backend-test-1 --tail=100`
4. **Test endpoints**: `curl -X GET http://localhost:8001/health`
5. **Clean up**: `docker compose -f docker-compose-test.yml down -v`

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
docker compose up parakeet-asr   # Offline ASR with Parakeet

# Speaker Recognition (with tests)
cd extras/speaker-recognition
docker compose up --build
./run-test.sh  # Run speaker recognition integration tests

# HAVPE Relay (ESP32 bridge)
cd extras/havpe-relay
docker compose up --build
```

## Architecture Overview

### Key Components
- **Audio Pipeline**: Real-time Opus/PCM → Application-level processing → Deepgram/Mistral transcription → memory extraction
- **Wyoming Protocol**: WebSocket communication uses Wyoming protocol (JSONL + binary) for structured audio sessions
- **Application-Level Processing**: Centralized processors for audio, transcription, memory, and cropping
- **Task Management**: BackgroundTaskManager tracks all async tasks to prevent orphaned processes
- **Unified Transcription**: Deepgram/Mistral transcription with fallback to offline ASR services
- **Memory System**: Pluggable providers (Friend-Lite native or OpenMemory MCP)
- **Authentication**: Email-based login with MongoDB ObjectId user system
- **Client Management**: Auto-generated client IDs as `{user_id_suffix}-{device_name}`, centralized ClientManager
- **Data Storage**: MongoDB (`audio_chunks` collection for conversations), vector storage (Qdrant or OpenMemory)
- **Web Interface**: React-based web dashboard with authentication and real-time monitoring

### Service Dependencies
```yaml
Required:
  - MongoDB: User data and conversations
  - FastAPI Backend: Core audio processing
  - LLM Service: Memory extraction and action items (OpenAI or Ollama)

Recommended:
  - Vector Storage: Qdrant (Friend-Lite provider) or OpenMemory MCP server
  - Transcription: Deepgram, Mistral, or offline ASR services

Optional:
  - Parakeet ASR: Offline transcription service
  - Speaker Recognition: Voice identification service
  - Nginx Proxy: Load balancing and routing
  - OpenMemory MCP: For cross-client memory compatibility
```

## Data Flow Architecture

1. **Audio Ingestion**: OMI devices stream audio via WebSocket using Wyoming protocol with JWT auth
2. **Wyoming Protocol Session Management**: Clients send audio-start/audio-stop events for session boundaries
3. **Application-Level Processing**: Global queues and processors handle all audio/transcription/memory tasks
4. **Speech-Driven Conversation Creation**: User-facing conversations only created when speech is detected
5. **Dual Storage System**: Audio sessions always stored in `audio_chunks`, conversations created in `conversations` collection only with speech
6. **Versioned Processing**: Transcript and memory versions tracked with active version pointers
7. **Memory Processing**: Pluggable providers (Friend-Lite native with individual facts or OpenMemory MCP delegation)
8. **Memory Storage**: Direct Qdrant (Friend-Lite) or OpenMemory server (MCP provider)
9. **Action Items**: Automatic task detection with "Simon says" trigger phrases
10. **Audio Optimization**: Speech segment extraction removes silence automatically
11. **Task Tracking**: BackgroundTaskManager ensures proper cleanup of all async operations

### Speech-Driven Architecture

**Core Principle**: Conversations are only created when speech is detected, eliminating noise-only sessions from user interfaces.

**Storage Architecture**:
- **`audio_chunks` Collection**: Always stores audio sessions by `audio_uuid` (raw audio capture)
- **`conversations` Collection**: Only created when speech is detected, identified by `conversation_id`
- **Speech Detection**: Analyzes transcript content, duration, and meaningfulness before conversation creation
- **Automatic Filtering**: No user-facing conversations for silence, noise, or brief audio without speech

**Benefits**:
- Clean user experience with only meaningful conversations displayed
- Reduced noise in conversation lists and memory processing
- Efficient storage utilization for speech-only content
- Automatic quality filtering without manual intervention

### Versioned Transcript and Memory System

**Version Architecture**:
- **`transcript_versions`**: Array of transcript processing attempts with timestamps and providers
- **`memory_versions`**: Array of memory extraction attempts with different models/prompts
- **`active_transcript_version`**: Pointer to currently displayed transcript
- **`active_memory_version`**: Pointer to currently active memory extraction

**Reprocessing Capabilities**:
- **Transcript Reprocessing**: Re-run speech-to-text with different providers or settings
- **Memory Reprocessing**: Re-extract memories using different LLM models or prompts
- **Version Management**: Switch between different processing results
- **Backward Compatibility**: Legacy fields auto-populated from active versions

**Data Consistency**:
- All reprocessing operations use `conversation_id` (not `audio_uuid`)
- DateTime objects stored as ISO strings for MongoDB/JSON compatibility
- Legacy field support ensures existing integrations continue working

### Database Schema Details

**Collections Overview**:
- **`audio_chunks`**: All audio sessions by `audio_uuid` (always created)
- **`conversations`**: Speech-detected conversations by `conversation_id` (created conditionally)
- **`users`**: User accounts and authentication data

**Speech-Driven Schema**:
```javascript
// audio_chunks collection (always created)
{
  "_id": ObjectId,
  "audio_uuid": "uuid",  // Primary identifier
  "user_id": ObjectId,
  "client_id": "user_suffix-device_name",
  "audio_file_path": "/path/to/audio.wav",
  "created_at": ISODate,
  "transcript": "fallback transcript",  // For non-speech audio
  "segments": [...],  // Speaker segments
  "has_speech": boolean,  // Speech detection result
  "speech_analysis": {...},  // Detection metadata
  "conversation_id": "conv_id" | null  // Link to conversations collection
}

// conversations collection (speech-detected only)
{
  "_id": ObjectId,
  "conversation_id": "conv_uuid",  // Primary identifier for user-facing operations
  "audio_uuid": "audio_uuid",  // Link to audio_chunks
  "user_id": ObjectId,
  "client_id": "user_suffix-device_name",
  "created_at": ISODate,

  // Versioned Transcript System
  "transcript_versions": [
    {
      "version_id": "uuid",
      "transcript": "text content",
      "segments": [...],  // Speaker diarization
      "provider": "deepgram|mistral|parakeet",
      "model": "nova-3|voxtral-mini-2507",
      "created_at": ISODate,
      "processing_time_seconds": 12.5,
      "metadata": {...}
    }
  ],
  "active_transcript_version": "uuid",  // Points to current version

  // Versioned Memory System
  "memory_versions": [
    {
      "version_id": "uuid",
      "memory_count": 5,
      "transcript_version_id": "uuid",  // Which transcript was used
      "provider": "friend_lite|openmemory_mcp",
      "model": "gpt-4o-mini|ollama-llama3",
      "created_at": ISODate,
      "processing_time_seconds": 45.2,
      "metadata": {...}
    }
  ],
  "active_memory_version": "uuid",  // Points to current version

  // Legacy Fields (auto-populated from active versions)
  "transcript": "text",  // From active_transcript_version
  "segments": [...],     // From active_transcript_version
  "memories": [...],     // From active_memory_version
  "memory_count": 5      // From active_memory_version
}
```

**Key Architecture Benefits**:
- **Clean Separation**: Raw audio storage vs user-facing conversations
- **Speech Filtering**: Only meaningful conversations appear in UI
- **Version History**: Complete audit trail of processing attempts
- **Backward Compatibility**: Legacy fields ensure existing code works
- **Reprocessing Support**: Easy to re-run with different providers/models
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
# Authentication
AUTH_SECRET_KEY=your-super-secret-jwt-key-here
ADMIN_PASSWORD=your-secure-admin-password
ADMIN_EMAIL=admin@example.com

# LLM Configuration
LLM_PROVIDER=openai  # or ollama
OPENAI_API_KEY=your-openai-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

# Speech-to-Text
DEEPGRAM_API_KEY=your-deepgram-key-here
# Optional: PARAKEET_ASR_URL=http://host.docker.internal:8767
# Optional: TRANSCRIPTION_PROVIDER=deepgram

# Memory Provider (New)
MEMORY_PROVIDER=friend_lite  # or openmemory_mcp

# Database
MONGODB_URI=mongodb://mongo:27017
# Database name: friend-lite
QDRANT_BASE_URL=qdrant

# Network Configuration
HOST_IP=localhost
BACKEND_PUBLIC_PORT=8000
WEBUI_PORT=5173
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Memory Provider Configuration

Friend-Lite now supports two pluggable memory backends:

#### Friend-Lite Memory Provider (Default)
```bash
# Use Friend-Lite memory provider (default)
MEMORY_PROVIDER=friend_lite

# LLM Configuration for memory extraction
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-key-here
OPENAI_MODEL=gpt-4o-mini

# Vector Storage
QDRANT_BASE_URL=qdrant
```

#### OpenMemory MCP Provider
```bash
# Use OpenMemory MCP provider
MEMORY_PROVIDER=openmemory_mcp

# OpenMemory MCP Server Configuration
OPENMEMORY_MCP_URL=http://host.docker.internal:8765
OPENMEMORY_CLIENT_NAME=friend_lite
OPENMEMORY_USER_ID=openmemory
OPENMEMORY_TIMEOUT=30

# OpenAI key for OpenMemory server
OPENAI_API_KEY=your-openai-key-here
```

#### OpenMemory MCP Interface Patterns

**Important**: OpenMemory MCP stores memories **per-app**, not globally. Understanding this architecture is critical for proper integration.

**App-Based Storage Architecture:**
- All memories are stored under specific "apps" (namespaces)
- Generic endpoints (`/api/v1/memories/`) return empty results
- App-specific endpoints (`/api/v1/apps/{app_id}/memories`) contain the actual memories

**Hardcoded Values and Configuration:**
```bash
# Default app name (configurable via OPENMEMORY_CLIENT_NAME)
Default: "friend_lite"

# Hardcoded metadata (NOT configurable)
"source": "friend_lite"  # Always hardcoded in Friend-Lite

# User ID for OpenMemory MCP server
OPENMEMORY_USER_ID=openmemory  # Configurable
```

**API Interface Pattern:**
```python
# 1. App Discovery - Find app by client_name
GET /api/v1/apps/
# Response: {"apps": [{"id": "uuid", "name": "friend_lite", ...}]}

# 2. Memory Creation - Uses generic endpoint but assigns to app
POST /api/v1/memories/
{
  "user_id": "openmemory",
  "text": "memory content",
  "app": "friend_lite",  # Uses OPENMEMORY_CLIENT_NAME
  "metadata": {
    "source": "friend_lite",    # Hardcoded
    "client": "friend_lite"     # Uses OPENMEMORY_CLIENT_NAME
  }
}

# 3. Memory Retrieval - Must use app-specific endpoint
GET /api/v1/apps/{app_id}/memories?user_id=openmemory&page=1&size=10

# 4. Memory Search - Must use app-specific endpoint with search_query
GET /api/v1/apps/{app_id}/memories?user_id=openmemory&search_query=keyword&page=1&size=10
```

**Friend-Lite Integration Flow:**
1. **App Discovery**: Query `/api/v1/apps/` to find app matching `OPENMEMORY_CLIENT_NAME`
2. **Fallback**: If client app not found, use first available app
3. **Operations**: All memory operations use the app-specific endpoints with discovered `app_id`

**Testing OpenMemory MCP Integration:**
```bash
# Configure .env file with OpenMemory MCP settings
cp .env.template .env
# Edit .env to set MEMORY_PROVIDER=openmemory_mcp and configure OPENMEMORY_* variables

# Start OpenMemory MCP server
cd extras/openmemory-mcp && docker compose up -d

# Run integration tests (reads configuration from .env file)
cd backends/advanced && ./run-test.sh

# Manual testing - Check app structure
curl -s "http://localhost:8765/api/v1/apps/" | jq

# Test memory creation
curl -X POST "http://localhost:8765/api/v1/memories/" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "openmemory", "text": "test memory", "app": "friend_lite"}'

# Retrieve memories (replace app_id with actual ID from apps endpoint)
curl -s "http://localhost:8765/api/v1/apps/{app_id}/memories?user_id=openmemory" | jq
```

### Transcription Provider Configuration

Friend-Lite supports multiple transcription services:

```bash
# Option 1: Deepgram (High quality, recommended)
TRANSCRIPTION_PROVIDER=deepgram
DEEPGRAM_API_KEY=your-deepgram-key-here

# Option 2: Mistral (Voxtral models)
TRANSCRIPTION_PROVIDER=mistral
MISTRAL_API_KEY=your-mistral-key-here
MISTRAL_MODEL=voxtral-mini-2507

# Option 3: Local ASR (Parakeet)
PARAKEET_ASR_URL=http://host.docker.internal:8767
```

### Additional Service Configuration
```bash
# LLM Processing
OLLAMA_BASE_URL=http://ollama:11434

# Speaker Recognition
SPEAKER_SERVICE_URL=http://speaker-recognition:8085
```

## Transcription Architecture

### Provider System
Friend-Lite supports multiple transcription providers:

**Online Providers (API-based):**
- **Deepgram**: High-quality transcription using Nova-3 model with real-time streaming
- **Mistral**: Voxtral models for transcription with REST API processing

**Offline Providers (Local processing):**
- **Parakeet**: Local speech recognition service available in extras/asr-services

**Provider Interface:**
The transcription system handles:
- Connection management and health checks
- Audio format handling (streaming vs batch)
- Error handling and reconnection
- Unified transcript format normalization

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

## Memory System Architecture

### Overview
Friend-Lite supports two pluggable memory backends that can be selected via configuration:

#### 1. Friend-Lite Memory Provider (`friend_lite`)
The sophisticated in-house memory implementation with full control and customization:

**Features:**
- Custom LLM-powered memory extraction with enhanced prompts
- Individual fact storage (no JSON blobs)
- Smart deduplication algorithms
- Intelligent memory updates (ADD/UPDATE/DELETE decisions)
- **Semantic search** with relevance threshold filtering
- **Memory count API** with total count tracking from native Qdrant
- Direct Qdrant vector storage with accurate similarity scoring
- Custom memory prompts and processing
- No external dependencies

**Architecture Flow:**
1. **Audio Input** → Transcription via Deepgram/Parakeet
2. **Memory Extraction** → LLM processes transcript using custom prompts
3. **Fact Parsing** → XML/JSON parsing into individual memory entries
4. **Deduplication** → Smart algorithms prevent duplicate memories
5. **Vector Storage** → Direct Qdrant storage with embeddings
6. **Memory Updates** → LLM-driven action proposals (ADD/UPDATE/DELETE)

#### 2. OpenMemory MCP Provider (`openmemory_mcp`)
Thin client that delegates all memory processing to external OpenMemory MCP server:

**Features:**
- Professional memory extraction (handled by OpenMemory)
- Battle-tested deduplication (handled by OpenMemory)
- Semantic vector search (handled by OpenMemory)
- ACL-based user isolation (handled by OpenMemory)
- Cross-client compatibility (Claude Desktop, Cursor, Windsurf)
- Web UI for memory management at http://localhost:8765

**Architecture Flow:**
1. **Audio Input** → Transcription via Deepgram/Parakeet
2. **MCP Delegation** → Send enriched transcript to OpenMemory MCP server
3. **External Processing** → OpenMemory handles extraction, deduplication, storage
4. **Result Mapping** → Convert MCP results to Friend-Lite MemoryEntry format
5. **Client Management** → Automatic user context switching via MCP client

### Memory Provider Comparison

| Feature | Friend-Lite | OpenMemory MCP |
|---------|-------------|----------------|
| **Processing** | Custom LLM extraction | Delegates to OpenMemory |
| **Deduplication** | Custom algorithms | OpenMemory handles |
| **Vector Storage** | Direct Qdrant | OpenMemory handles |
| **Search Features** | Semantic search with threshold filtering | Semantic search with relevance scoring |
| **Memory Count** | Native Qdrant count API | Varies by OpenMemory support |
| **Dependencies** | Qdrant + MongoDB | External OpenMemory server |
| **Customization** | Full control | Limited to OpenMemory features |
| **Cross-client** | Friend-Lite only | Works with Claude Desktop, Cursor, etc |
| **Web UI** | Friend-Lite WebUI with advanced search | OpenMemory UI + Friend-Lite WebUI |
| **Memory Format** | Individual facts | OpenMemory format |
| **Setup Complexity** | Medium | High (external server required) |

### Switching Memory Providers

You can switch providers by changing the `MEMORY_PROVIDER` environment variable:

```bash
# Switch to OpenMemory MCP
echo "MEMORY_PROVIDER=openmemory_mcp" >> .env

# Switch back to Friend-Lite
echo "MEMORY_PROVIDER=friend_lite" >> .env
```

**Note:** Existing memories are not automatically migrated between providers. Each provider maintains its own memory storage.

### OpenMemory MCP Setup

To use the OpenMemory MCP provider:

```bash
# 1. Start external OpenMemory MCP server
cd extras/openmemory-mcp
docker compose up -d

# 2. Configure Friend-Lite to use OpenMemory MCP
cd backends/advanced
echo "MEMORY_PROVIDER=openmemory_mcp" >> .env

# 3. Start Friend-Lite backend
docker compose up --build -d
```

### When to Use Each Provider

**Use Friend-Lite when:**
- You want full control over memory processing
- You need custom memory extraction logic
- You prefer fewer external dependencies
- You want to customize memory prompts and algorithms
- You need individual fact-based memory storage

**Use OpenMemory MCP when:**
- You want professional, battle-tested memory processing
- You need cross-client compatibility (Claude Desktop, Cursor, etc.)
- You prefer to leverage external expertise rather than maintain custom logic
- You want access to OpenMemory's web interface
- You're already using OpenMemory in other tools

## Versioned Processing System

### Overview

Friend-Lite implements a comprehensive versioning system for both transcript and memory processing, allowing multiple processing attempts with different providers, models, or settings while maintaining a clean user experience.

### Version Data Structure

**Transcript Versions**:
```json
{
  "transcript_versions": [
    {
      "version_id": "uuid",
      "transcript": "processed text",
      "segments": [...],
      "provider": "deepgram|mistral|parakeet",
      "model": "nova-3|voxtral-mini-2507",
      "created_at": "2025-01-15T10:30:00Z",
      "processing_time_seconds": 12.5,
      "metadata": {
        "confidence_scores": [...],
        "speaker_diarization": true
      }
    }
  ],
  "active_transcript_version": "uuid"
}
```

**Memory Versions**:
```json
{
  "memory_versions": [
    {
      "version_id": "uuid",
      "memory_count": 5,
      "transcript_version_id": "uuid",
      "provider": "friend_lite|openmemory_mcp",
      "model": "gpt-4o-mini|ollama-llama3",
      "created_at": "2025-01-15T10:32:00Z",
      "processing_time_seconds": 45.2,
      "metadata": {
        "prompt_version": "v2.1",
        "extraction_quality": "high"
      }
    }
  ],
  "active_memory_version": "uuid"
}
```

### Reprocessing Workflows

**Transcript Reprocessing**:
1. Trigger via API: `POST /api/conversations/{conversation_id}/reprocess-transcript`
2. System creates new transcript version with different provider/model
3. New version added to `transcript_versions` array
4. User can activate any version via `activate-transcript` endpoint
5. Legacy `transcript` field automatically updated from active version

**Memory Reprocessing**:
1. Trigger via API: `POST /api/conversations/{conversation_id}/reprocess-memory`
2. Specify which transcript version to use as input
3. System creates new memory version using specified transcript
4. New version added to `memory_versions` array
5. User can activate any version via `activate-memory` endpoint
6. Legacy `memories` field automatically updated from active version

### Legacy Field Compatibility

**Automatic Population**:
- `transcript`: Auto-populated from active transcript version
- `segments`: Auto-populated from active transcript version
- `memories`: Auto-populated from active memory version
- `memory_count`: Auto-populated from active memory version

**Backward Compatibility**:
- Existing API clients continue working without modification
- WebUI displays active versions by default
- Advanced users can access version history and switch between versions

## Development Notes

### Package Management
- **Backend**: Uses `uv` for Python dependency management (faster than pip)
- **Mobile**: Uses `npm` with React Native and Expo
- **Docker**: Primary deployment method with docker-compose

### Testing Strategy
- **Local Test Scripts**: Simplified scripts (`./run-test.sh`) mirror CI workflows for local development
- **End-to-End Integration**: `test_integration.py` validates complete audio processing pipeline
- **Speaker Recognition Tests**: `test_speaker_service_integration.py` validates speaker identification
- **Environment Flexibility**: Tests work with both local .env files and CI environment variables
- **Automated Cleanup**: Test containers are automatically removed after execution
- **CI/CD Integration**: GitHub Actions use the same local test scripts for consistency

### Code Style
- **Python**: Black formatter with 100-character line length, isort for imports
- **TypeScript**: Standard React Native conventions
- **Import Guidelines**:
  - NEVER import modules in the middle of functions or files
  - ALL imports must be at the top of the file after the docstring
  - Use lazy imports sparingly and only when absolutely necessary for circular import issues
  - Group imports: standard library, third-party, local imports
- **Error Handling Guidelines**:
  - **Always raise errors, never silently ignore**: Use explicit error handling with proper exceptions rather than silent failures
  - **Understand data structures**: Research and understand input/response or class structure instead of adding defensive `hasattr()` checks

### Docker Build Cache Management
- **Default Behavior**: Docker automatically detects file changes in Dockerfile COPY/ADD instructions and invalidates cache as needed
- **No --no-cache by Default**: Only use `--no-cache` when explicitly needed (e.g., package updates, dependency issues)
- **Smart Caching**: Docker checks file modification times and content hashes to determine when rebuilds are necessary
- **Development Efficiency**: Trust Docker's cache system - it handles most development scenarios correctly

### Health Monitoring
The system includes comprehensive health checks:
- `/readiness`: Service dependency validation
- `/health`: Basic application status
- Memory debug system for transcript processing monitoring

### Integration Test Infrastructure
- **Unified Test Scripts**: Local `./run-test.sh` scripts mirror GitHub Actions workflows
- **Test Environment**: `docker-compose-test.yml` provides isolated services on separate ports
- **Test Database**: Uses `test_db` database with isolated collections
- **Service Ports**: Backend (8001), MongoDB (27018), Qdrant (6335/6336), WebUI (5174)
- **Test Credentials**: Auto-generated `.env.test` files with secure test configurations
- **Ground Truth**: Expected transcript established via `scripts/test_deepgram_direct.py`
- **AI Validation**: OpenAI-powered transcript similarity comparison
- **Test Audio**: 4-minute glass blowing tutorial (`extras/test-audios/DIY*mono*.wav`)
- **CI Compatibility**: Same test logic runs locally and in GitHub Actions

### Cursor Rule Integration
Project includes `.cursor/rules/always-plan-first.mdc` requiring understanding before coding. Always explain the task and confirm approach before implementation.


## API Reference

### Health & Status Endpoints
- **GET /health**: Basic application health check
- **GET /readiness**: Service dependency validation (MongoDB, Qdrant, etc.)
- **GET /api/metrics**: System metrics and debug tracker status (Admin only)
- **GET /api/processor/status**: Processor queue status and health (Admin only)  
- **GET /api/processor/tasks**: All active processing tasks (Admin only)
- **GET /api/processor/tasks/{client_id}**: Processing task status for specific client (Admin only)

### WebSocket Endpoints
- **WS /ws_pcm**: Primary audio streaming endpoint (Wyoming protocol + raw PCM fallback)
- **WS /ws**: Simple audio streaming endpoint (Opus packets + Wyoming audio-chunk events)

### Memory & Conversation Debugging
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

### Client Management
- **GET /api/clients/active**: Currently active WebSocket clients
- **GET /api/users**: List all users (Admin only)

### File Processing
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

### Authentication
- **POST /auth/jwt/login**: Email-based login (returns JWT token)
- **GET /users/me**: Get current authenticated user
- **GET /api/auth/config**: Authentication configuration

### Step-by-Step API Testing Guide

When testing API endpoints that require authentication, follow these steps:

#### Step 1: Read credentials from .env file
```bash
# Use the Read tool to view the .env file and identify credentials
# Look for:
# ADMIN_EMAIL=admin@example.com
# ADMIN_PASSWORD=your-password-here
```

#### Step 2: Get authentication token
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

#### Step 3: Use the token in API calls
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

#### Step 4: Testing Reprocessing Endpoints
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

### Development Reset Endpoints
Useful endpoints for resetting state during development:

#### Data Cleanup
- **DELETE /api/admin/memory/delete-all**: Delete all memories for the current user
- **DELETE /api/memories/{memory_id}**: Delete a specific memory
- **DELETE /api/conversations/{conversation_id}**: Delete a specific conversation (keeps original audio file in audio_chunks)
- **DELETE /api/chat/sessions/{session_id}**: Delete a chat session and all its messages
- **DELETE /api/users/{user_id}**: Delete a user (Admin only)
  - Optional query params: `delete_conversations=true`, `delete_memories=true`

#### Quick Reset Commands
```bash
# Reset all data (development only)
cd backends/advanced
sudo rm -rf data/

# Reset Docker volumes
docker compose down -v
docker compose up --build -d
```


## Speaker Recognition Service Features

### Speaker Analysis & Visualization
The speaker recognition service now includes advanced analysis capabilities:

#### Embedding Analysis (/speakers/analysis endpoint)
- **2D/3D Visualization**: Interactive embedding plots using UMAP, t-SNE, or PCA
- **Clustering Analysis**: Automatic clustering using DBSCAN or K-means
- **Speaker Similarity Detection**: Identifies speakers with similar embeddings
- **Quality Metrics**: Embedding separation quality and confidence scores
- **Interactive Controls**: Adjustable analysis parameters and visualization options

Access via: `extras/speaker-recognition/webui` → Speakers → Embedding Analysis tab

#### Live Inference Feature (/infer-live page)
Real-time speaker identification and transcription:
- **WebRTC Audio Capture**: Live microphone access with waveform visualization
- **Deepgram Streaming**: Real-time transcription with speaker diarization
- **Live Speaker ID**: Identifies enrolled speakers in real-time using internal service
- **Session Statistics**: Live metrics for words, speakers, and confidence scores
- **Configurable Settings**: Adjustable confidence thresholds and audio parameters

Access via: `extras/speaker-recognition/webui` → Live Inference

### Technical Implementation

#### Backend (Python)
- **Analysis Utils**: `src/simple_speaker_recognition/utils/analysis.py`
  - UMAP/t-SNE dimensionality reduction
  - DBSCAN/K-means clustering
  - Cosine similarity analysis
  - Quality metrics calculation
- **API Endpoint**: `/speakers/analysis` - Returns processed embedding analysis
- **Dependencies**: Added `umap-learn` for dimensionality reduction

#### Frontend (React/TypeScript)
- **EmbeddingPlot Component**: Interactive Plotly.js visualizations
- **LiveAudioCapture Component**: WebRTC audio recording with waveform
- **DeepgramStreaming Service**: WebSocket integration for real-time transcription
- **InferLive Page**: Complete live inference interface

### Usage Instructions

#### Setting up Live Inference
1. Navigate to Live Inference page
2. Configure Deepgram API key in settings
3. Adjust speaker identification settings (confidence threshold)
4. Start live session to begin real-time transcription and speaker ID

**Technical Details:**
- **Audio Processing**: Uses browser's native sample rate (typically 44.1kHz or 48kHz)
- **Buffer Retention**: 120 seconds of audio for improved utterance capture
- **Real-time Updates**: Live transcription with speaker identification results

#### Using Speaker Analysis
1. Go to Speakers page → Embedding Analysis tab
2. Select analysis method (UMAP, t-SNE, PCA)
3. Choose clustering algorithm (DBSCAN, K-means)
4. Adjust similarity threshold for speaker detection
5. View interactive plots and quality metrics

### Deployment Notes
- Requires Docker rebuild to pick up new Python dependencies
- Frontend dependencies (Plotly.js) already included
- Live inference requires Deepgram API key for streaming transcription
- Speaker identification uses existing enrolled speakers from database

### Live Inference Troubleshooting
- **"NaN:NaN" timestamps**: Fixed in recent updates, ensure you're using latest version
- **Poor speaker identification**: Try adjusting confidence threshold or re-enrolling speakers
- **Audio processing delays**: Check browser console for sample rate detection logs
- **Buffer overflow issues**: Extended to 120-second retention for better performance
- **"extraction_failed" errors**: Usually indicates audio buffer timing issues - check console logs for buffer availability

## Distributed Self-Hosting Architecture

Friend-Lite supports distributed deployment across multiple machines, allowing you to separate GPU-intensive services from lightweight backend components. This is ideal for scenarios where you have a dedicated GPU machine and want to run the main backend on a VPS or Raspberry Pi.

### Architecture Patterns

#### Single Machine (Default)
All services run on one machine using Docker Compose - ideal for development and simple deployments.

#### Distributed GPU Setup
**GPU Machine (High-performance):**
- LLM services (Ollama with GPU acceleration)
- ASR services (Parakeet with GPU)
- Speaker recognition service
- Deepgram fallback can remain on backend machine

**Backend Machine (Lightweight - VPS/RPi):**
- Friend-Lite backend (FastAPI)
- React WebUI
- MongoDB
- Qdrant vector database

### Networking with Tailscale

Tailscale VPN provides secure, encrypted networking between distributed services:

**Benefits:**
- **Zero configuration networking**: Services discover each other automatically
- **Encrypted communication**: All inter-service traffic is encrypted
- **Firewall friendly**: Works behind NATs and firewalls
- **Access control**: Granular permissions for service access
- **CORS support**: Built-in support for Tailscale IP ranges (100.x.x.x)

**Installation:**
```bash
# On each machine
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

### Distributed Service Configuration

#### GPU Machine Services
```bash
# .env on GPU machine
OLLAMA_BASE_URL=http://0.0.0.0:11434  # Expose to Tailscale network
SPEAKER_SERVICE_URL=http://0.0.0.0:8085

# Enable GPU acceleration for Ollama
docker run -d --gpus=all -p 11434:11434 ollama/ollama:latest
```

#### Backend Machine Configuration
```bash
# .env on backend machine  
OLLAMA_BASE_URL=http://100.x.x.x:11434  # GPU machine Tailscale IP
SPEAKER_SERVICE_URL=http://100.x.x.x:8085  # GPU machine Tailscale IP

# Parakeet ASR services can also be distributed (if using offline ASR)
# PARAKEET_ASR_URL=http://100.x.x.x:8767

# CORS automatically supports Tailscale IPs (no configuration needed)
```

#### Service URL Examples

**Common remote service configurations:**
```bash
# LLM Processing (GPU machine)
OLLAMA_BASE_URL=http://100.64.1.100:11434
OPENAI_BASE_URL=http://100.64.1.100:8080  # For vLLM/OpenAI-compatible APIs

# Speech Recognition (GPU machine)
# PARAKEET_ASR_URL=http://100.64.1.100:8767  # If using Parakeet ASR
SPEAKER_SERVICE_URL=http://100.64.1.100:8085

# Database services (can be on separate machine)
MONGODB_URI=mongodb://100.64.1.200:27017  # Database name: friend-lite
QDRANT_BASE_URL=http://100.64.1.200:6333
```

### Deployment Steps

#### 1. Set up Tailscale on all machines
```bash
# Install and connect each machine to your Tailscale network
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

#### 2. Deploy GPU services
```bash
# On GPU machine - start GPU-accelerated services
cd extras/asr-services && docker compose up parakeet -d
cd extras/speaker-recognition && docker compose up --build -d

# Start Ollama with GPU support
docker run -d --gpus=all -p 11434:11434 \
  -v ollama:/root/.ollama \
  ollama/ollama:latest
```

#### 3. Configure backend machine
```bash
# Update .env with Tailscale IPs of GPU machine
OLLAMA_BASE_URL=http://[gpu-machine-tailscale-ip]:11434
SPEAKER_SERVICE_URL=http://[gpu-machine-tailscale-ip]:8085

# Start lightweight backend services
docker compose up --build -d
```

#### 4. Verify connectivity
```bash
# Test service connectivity from backend machine
curl http://[gpu-machine-ip]:11434/api/tags  # Ollama
curl http://[gpu-machine-ip]:8085/health     # Speaker recognition
```

### Performance Considerations

**Network Latency:**
- Tailscale adds minimal latency (typically <5ms between nodes)
- LLM inference: Network time negligible compared to GPU processing
- ASR streaming: Use local fallback for latency-sensitive applications

**Bandwidth Usage:**
- Audio streaming: ~128kbps for Opus, ~512kbps for PCM
- LLM requests: Typically <1MB per conversation
- Memory embeddings: ~3KB per memory vector

**Processing Time Expectations:**
- Transcription (Deepgram): 2-5 seconds for 4-minute audio
- Transcription (Parakeet): 5-10 seconds for 4-minute audio
- Memory extraction (OpenAI GPT-4o-mini): 30-40 seconds for typical conversation
- Memory extraction (Ollama local): 45-90 seconds depending on model and GPU
- Full pipeline (4-min audio): 40-60 seconds with cloud services, 60-120 seconds with local models

### Security Best Practices

**Tailscale Access Control:**
```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["tag:backend"],
      "dst": ["tag:gpu:11434", "tag:gpu:8085", "tag:gpu:8767"]
    }
  ],
  "tagOwners": {
    "tag:backend": ["your-email@example.com"],
    "tag:gpu": ["your-email@example.com"]
  }
}
```

**Service Isolation:**
- Run GPU services in containers with limited network access
- Use Tailscale subnet routing for additional security
- Monitor service access logs for unauthorized requests

### Troubleshooting Distributed Setup

**Debugging Commands:**
```bash
# Check Tailscale connectivity
tailscale ping [machine-name]
tailscale status

# Test service endpoints
curl http://[tailscale-ip]:11434/api/tags
curl http://[tailscale-ip]:8085/health

# Check Docker networks
docker network ls
docker ps --format "table {{.Names}}\t{{.Ports}}"
```

## Notes for Claude
Check if the src/ is volume mounted. If not, do compose build so that code changes are reflected. Do not simply run `docker compose restart` as it will not rebuild the image.
Check backends/advanced/Docs for up to date information on advanced backend.
All docker projects have .dockerignore following the exclude pattern. That means files need to be included for them to be visible to docker.
The uv package manager is used for all python projects. Wherever you'd call `python3 main.py` you'd call `uv run python main.py`

**Docker Build Guidelines:**
- Use `docker compose build` without `--no-cache` by default for faster builds
- Only use `--no-cache` when explicitly needed (e.g., if cached layers are causing issues or when troubleshooting build problems)
- Docker's build cache is efficient and saves significant time during development

- Remember that whenever there's a python command, you should use uv run python3 instead