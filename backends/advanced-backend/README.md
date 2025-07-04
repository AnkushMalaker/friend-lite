# Advanced Backend

An AI-powered backend service for real-time audio processing and transcription.

## Features

- Real-time audio transcription using offline ASR
- Memory storage and retrieval
- Action items extraction
- Speaker enrollment and recognition
- Web UI for monitoring and management

## Quick Setup

1. **Clone and navigate to the backend:**
   ```bash
   cd backends/advanced-backend
   ```

2. **Configure environment variables:**
   Copy `.env.template` to `.env` and fill in the required values:
   ```bash
   cp .env.template .env
   ```

3. **Start the service:**
   ```bash
   docker compose up --build -d
   ```

## Configuration

### Required Environment Variables

- `OFFLINE_ASR_TCP_URI` - TCP URI for the offline ASR service (e.g., `tcp://192.168.0.110:8765/`)

### Optional Environment Variables

- `DEEPGRAM_API_KEY` - For future Deepgram integration (not yet implemented)

## Services

The backend includes several services accessible via different endpoints:

- **Main API** - Core functionality and endpoints
- **Web UI** - Management interface accessible via browser
- **Memory Service** - Storage and retrieval of conversation data
- **Speaker Recognition** - Voice identification capabilities

## Notes

- The system currently uses offline ASR for transcription
- Deepgram API integration is planned but not yet implemented
- Initial startup may take a few minutes while services initialize