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

## Laptop Client

The `laptop_client.py` provides a command-line interface for recording from your microphone and interacting with the speaker recognition service.

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements-laptop.txt
   ```

2. **Start the speaker service** (using Docker or locally):
   ```bash
   docker-compose up
   # or
   python speaker_service.py
   ```

### Usage

#### Enroll a Speaker
Record 10 seconds of audio to enroll a new speaker:
```bash
python laptop_client.py enroll --speaker-id "john" --speaker-name "John Doe" --duration 10
```

#### Identify a Speaker
Record 5 seconds of audio to identify who is speaking:
```bash
python laptop_client.py identify --duration 5
```

#### Verify a Speaker
Check if the voice matches a specific enrolled speaker:
```bash
python laptop_client.py verify --speaker-id "john" --duration 3
```

#### List Enrolled Speakers
```bash
python laptop_client.py list
```

#### Remove a Speaker
```bash
python laptop_client.py remove --speaker-id "john"
```

### Options

- `--service-url`: Change the service URL (default: http://localhost:8001)
- `--duration`: Recording duration in seconds
- `--speaker-id`: Unique identifier for speaker
- `--speaker-name`: Display name for speaker

### Example Workflow

1. Start the service:
   ```bash
   docker-compose up
   ```

2. Enroll yourself:
   ```bash
   python laptop_client.py enroll --speaker-id "myself" --speaker-name "My Name" --duration 15
   ```

3. Test identification:
   ```bash
   python laptop_client.py identify --duration 5
   ```

## Laptop Client

A command-line client (`laptop_client.py`) that can record from your microphone and interact with the speaker recognition service.

### Setup for Laptop Client

The laptop client requires PyAudio for microphone access:

```bash
# On Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio

# On macOS
brew install portaudio
pip install pyaudio

# On Windows
pip install pyaudio
```

### Usage Examples

```bash
# Start the speaker service first
docker-compose up -d

# Enroll a new speaker (records 10 seconds)
python laptop_client.py enroll --speaker-id "john" --speaker-name "John Doe" --duration 10

# Identify a speaker (records 5 seconds)
python laptop_client.py identify --duration 5

# Verify against a specific speaker (records 3 seconds)
python laptop_client.py verify --speaker-id "john" --duration 3

# List all enrolled speakers
python laptop_client.py list

# Remove a speaker
python laptop_client.py remove --speaker-id "john"

# Use different service URL
python laptop_client.py --service-url "http://192.168.1.100:8001" identify
```

### Laptop Client Features

- **Live Microphone Recording**: Records directly from your system microphone
- **Automatic Cleanup**: Temporary audio files are automatically cleaned up
- **Service Health Checks**: Verifies the speaker service is online before operations
- **Real-time Feedback**: Shows recording progress and results with emojis
- **Error Handling**: Graceful handling of network and audio errors

## Integration with Advanced Backend

The advanced backend communicates with this service through the `client.py` module, which provides both async and sync interfaces for backward compatibility.

## Performance Notes

- First inference may be slow due to model loading
- GPU memory usage scales with model size (~2-4GB)
- Audio files should be accessible from both services (use shared volumes)
- Microphone recording uses 16kHz sample rate for optimal compatibility
- Microphone recording requires `pyaudio` and proper audio device setup 