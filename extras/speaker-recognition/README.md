# Speaker Recognition Service

A standalone FastAPI service for speaker diarization, enrollment, and identification. This service handles the heavy GPU computations separately from the main backend.

## Features

- **Speaker Diarization**: Identify who spoke when in audio recordings
- **Speaker Enrollment**: Register known speakers with audio samples
- **Speaker Identification**: Identify enrolled speakers in new audio
- **GPU Support**: Optimized for CUDA when available
- **RESTful API**: Easy integration with other services

## Requirements

- Python 3.10+
- GPU (NVIDIA) for optimal performance
- Hugging Face token for gated models

## Setup

1. **Environment Variables**:
   ```bash
   export HF_TOKEN="your_huggingface_token"
   export SIMILARITY_THRESHOLD="0.85"
   ```

2. **Docker (Recommended)**:
   ```bash
   docker-compose up --build
   ```

3. **Local Development**:
   ```bash
   uv sync
   uv run python speaker_service.py
   ```

## API Endpoints

### Health Check
```bash
GET /health
```

### Speaker Enrollment
```bash
POST /enroll
Content-Type: application/json

{
  "speaker_id": "john_doe",
  "speaker_name": "John Doe", 
  "audio_file_path": "/path/to/audio.wav",
  "start_time": 0.0,  // optional
  "end_time": 5.0     // optional
}
```

### Speaker Identification
```bash
POST /identify
Content-Type: application/json

{
  "audio_file_path": "/path/to/audio.wav",
  "start_time": 0.0,  // optional
  "end_time": 5.0     // optional
}
```

### Speaker Diarization
```bash
POST /diarize
Content-Type: application/json

{
  "audio_file_path": "/path/to/audio.wav"
}
```

### List Speakers
```bash
GET /speakers
```

### Remove Speaker
```bash
DELETE /speakers/{speaker_id}
```

## Integration with Advanced Backend

The advanced backend communicates with this service through the `client.py` module, which provides both async and sync interfaces for backward compatibility.

## Performance Notes

- First inference may be slow due to model loading
- GPU memory usage scales with model size (~2-4GB)
- Audio files should be accessible from both services (use shared volumes) 