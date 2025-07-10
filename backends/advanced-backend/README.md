# Advanced Omi Backend

## Transcription Configuration

This backend supports conditional transcription methods:

### 1. Deepgram API (Not Yet Implemented)
When `DEEPGRAM_API_KEY` is provided, the system is designed to use Deepgram's cloud API for transcription. However, this feature is not yet implemented and will fall back to offline ASR with a warning.

### 2. Offline ASR (Current Implementation)
The system uses the offline ASR service specified by `OFFLINE_ASR_TCP_URI`.

```bash
export OFFLINE_ASR_TCP_URI="tcp://192.168.0.110:8765/"
```

## Environment Variables

```bash
# For future Deepgram implementation (currently not implemented)
DEEPGRAM_API_KEY="your_api_key"

# Required for offline ASR (current implementation)
OFFLINE_ASR_TCP_URI="tcp://192.168.0.110:8765/"
```

The system automatically detects which transcription method to use based on the availability of `DEEPGRAM_API_KEY`, but currently always falls back to offline ASR.

# Setup

To setup the backend, you need to do the following:
0. Clone the repository
1. Change the directory to the backend,  
`cd backends/advanced-backend`
2. Fill out the .env variables as you require (check the .env.template for the required variables)
3. Run the backend with `docker compose up --build -d`. This will take a couple minutes, be patient.


# Backend Walkthrough

## Architecture Overview

This is a real-time audio processing backend built with FastAPI that handles continuous audio streams, transcription, memory storage, and conversation management. The system is being designed for 24/7 operation with robust recovery mechanisms.

## Core Services (Docker Compose)

- **friend-backend**: Main FastAPI application serving the audio processing pipeline
- **streamlit**: Web UI for conversation management, speaker enrollment, and system monitoring  
- **proxy**: Nginx reverse proxy handling external requests
- **qdrant**: Vector database for semantic memory storage and retrieval
- **mongo**: Document database for conversations, users, speakers, and action items
- **Optional services**: speaker-recognition (GPU-based), ollama (LLM inference)

## Audio Processing Flow

### 1. Audio Ingestion
- Clients connect via WebSocket endpoints:
  - `/ws`: Opus-encoded audio streams (from mobile apps)
  - `/ws_pcm`: Raw PCM audio streams (from desktop clients)
- Each client gets a `ClientState` managing their processing pipeline
- Audio data flows into central queues to decouple ingestion from processing

### 2. Parallel Processing Pipeline
The system runs multiple async consumers processing audio in parallel:

**Audio Saver Consumer** (`_audio_saver`):
- Buffers incoming PCM audio data
- Writes 60-second WAV chunks to `./audio_chunks/` directory
- Tracks speech segments for audio cropping
- Generates unique audio UUIDs for each chunk

**Transcription Consumer** (`_transcription_processor`):
- Sends audio chunks to Wyoming ASR service via TCP
- Supports fallback to Deepgram API (not yet implemented)
- Handles real-time transcription with segment timing
- Processes voice activity detection (VAD) events

**Memory Consumer** (`_memory_processor`):
- Stores completed transcripts in mem0 vector database
- Creates semantic memories for long-term retrieval
- Manages conversation context and user associations
- Handles background memory processing

### 3. Advanced Features

**Speaker Recognition**:
- Voice enrollment via audio samples
- Real-time speaker identification during conversations
- Speaker diarization and transcript attribution

**Audio Cropping**:
- Removes silence using speech segment detection
- Preserves only voice activity with configurable padding
- Reduces storage requirements and improves processing efficiency

**Action Items Extraction**:
- Uses LLM (Ollama) to extract tasks from conversations
- Tracks action item status and assignments
- Provides API for task management

**Conversation Management**:
- Automatic conversation segmentation based on silence timeouts
- Session state management across client connections
- Conversation closing and archival

### 4. Data Storage

**MongoDB Collections**:
- `audio_chunks`: Audio file metadata, transcripts, timing, speakers
- `users`: User profiles and settings
- `speakers`: Voice enrollment data and models
- `action_items`: Extracted tasks with status tracking

**File System**:
- `./audio_chunks/`: Raw and cropped WAV files
- `./qdrant_data/`: Vector database storage
- `./mongo_data/`: Document database storage

### 5. Health & Monitoring

Current health checks verify:
- MongoDB connectivity (critical service)
- ASR service availability (Wyoming protocol)
- Memory service (mem0 + Qdrant + Ollama)
- Speaker recognition service
- File system access

## Key Classes & Components

- `ClientState`: Per-client audio processing state and queues
- `TranscriptionManager`: ASR service management and reconnection logic
- `ChunkRepo`: MongoDB operations for audio chunks and metadata
- `MemoryService`: mem0 integration for semantic memory
- `SpeakerService`: speaker recognition and enrollment
- `ActionItemsService`: LLM-based task extraction and management

## Recovery & Reliability
TODO

## Metrics & Monitoring Plan

### Target: 24 Hours Uninterrupted Audio Processing

The primary goal is to achieve at least 24 hours of continuous audio recording and processing without interruptions. The metrics system will track:

### Core Metrics to Implement

**System Uptime Metrics**:
- Total system uptime vs. total recording time
- Service-level uptime for each component (friend-backend, mongo, qdrant, ASR, etc.)
- Connection uptime per client
- WebSocket connection stability and reconnection events

**Audio Processing Metrics**:
- Total audio recorded (duration in hours/minutes)
- Total voice activity detected vs. silence
- Audio chunks successfully processed vs. failed
- Transcription success rate and latency
- Memory storage success rate

On the happy path, you could do `sudo rm -rf ./audio_chunks/ ./mongo_data/ ./qdrant_data/` to reset the system.