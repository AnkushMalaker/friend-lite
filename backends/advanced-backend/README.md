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