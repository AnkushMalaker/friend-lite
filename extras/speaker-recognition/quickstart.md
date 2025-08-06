# Speaker Recognition Quick Start

Get the speaker recognition system running in 5 minutes and start processing audio files.

## What This System Does

- **Separates speakers** in audio files (who spoke when)
- **Identifies known speakers** by name (if you've enrolled them)
- **Transcribes audio** with speaker labels (using Deepgram)
- **Provides a web UI** for easy interaction

## Setup

### 1. Start the Services

```bash
cd extras/speaker-recognition
docker compose up --build -d
```

### 2. Access the Web UI

Open https://localhost:5173 in your browser (accept the SSL certificate warning).

**Need different ports?** Copy `.env.template` to `.env` and customize:
```bash
REACT_UI_PORT=3000           # Web UI port
SPEAKER_SERVICE_PORT=8086    # API port
```

## First Steps

### 1. Upload Audio
- Go to **Audio Viewer** page
- Upload a WAV, MP3, or FLAC file (< 10 minutes recommended)

### 2. Try Processing Modes

| What You Want | Mode | Requirements |
|---------------|------|--------------|
| See who spoke when | **Diarization Only** | Just audio |
| Identify known speakers | **Speaker Identification** | Enrolled speakers |
| Get transcription + speakers | **Deepgram Enhanced** | Deepgram API key |

**Don't have enrolled speakers yet?** Go to **Enrollment** page and record/upload samples for each person.

### 3. Process Your Audio
- Go to **Inference** page
- Select your processing mode
- Upload audio file
- Click process and wait for results

## Processing Modes Explained

### Diarization Only
- **What**: Separates speakers → "Speaker A said this, Speaker B said that"
- **When**: You just want to see speaker changes
- **Output**: Generic speaker labels (Speaker A, Speaker B, etc.)

### Speaker Identification  
- **What**: Separates speakers + identifies them by name
- **When**: You have enrolled speakers and want to know who said what
- **Output**: Named speakers ("John said this, Mary said that")

### Deepgram Enhanced
- **What**: Full transcription + speaker names
- **When**: You want both transcript and speaker identification
- **Requires**: Deepgram API key + enrolled speakers
- **Output**: "John: Hello there. Mary: How are you?"

### Deepgram + Internal Speakers
- **What**: Best quality transcription + most accurate speaker identification
- **When**: Maximum accuracy is needed
- **Requires**: Deepgram API key + enrolled speakers

## Common Issues

### "No speakers identified"
1. Make sure you have enrolled speakers (go to **Enrollment** page)
2. Lower confidence threshold (try 0.10-0.15)
3. Try **Diarization Only** first to see if speakers are detected

### "Processing failed" or timeouts
1. Use shorter audio files (< 5 minutes)
2. Check audio format (WAV works best)
3. Check Docker logs: `docker compose logs speaker-recognition`

### "404 errors" or "endpoint not found"
1. Ensure services are running: `docker compose ps`
2. Rebuild if needed: `docker compose up --build -d`

### Port conflicts
1. Check what's using the port: `netstat -tulpn | grep :5173`
2. Customize ports in `.env` file
3. Restart: `docker compose down && docker compose up -d`

## API Usage

Once services are running, you can use the REST API:

```bash
# Health check
curl http://localhost:8085/health

# Basic speaker separation
curl -X POST http://localhost:8085/v1/diarize-only \
  -F "file=@your-audio.wav"

# Speaker identification (requires enrolled speakers)
curl -X POST http://localhost:8085/diarize-and-identify \
  -F "file=@your-audio.wav" \
  -F "user_id=1"

# Transcription + speaker ID (requires Deepgram API key)
curl -X POST "http://localhost:8085/v1/listen?user_id=1" \
  -H "Authorization: Token YOUR_DEEPGRAM_API_KEY" \
  -F "file=@your-audio.wav"
```

## Configuration Options

Copy `.env.template` to `.env` to customize:

```bash
# Ports
SPEAKER_SERVICE_PORT=8085    # Backend API
REACT_UI_PORT=5173          # Web UI

# Settings  
SIMILARITY_THRESHOLD=0.15    # Speaker ID confidence (0.1-0.3)
REACT_UI_HTTPS=true         # Enable HTTPS (needed for microphone)

# Optional APIs
DEEPGRAM_API_KEY=your_key   # For transcription modes
HF_TOKEN=your_token         # Required for first-time model download
```

## Next Steps

1. **Enroll speakers** - Go to Enrollment page and add voice samples
2. **Try live inference** - Real-time transcription with speaker ID
3. **Explore speaker analysis** - See how similar your enrolled speakers are
4. **Integrate with your app** - Use the REST API endpoints

## Need Help?

- **Service not starting?** → `docker compose logs`
- **Web UI not loading?** → Check https://localhost:5173 and accept SSL cert
- **API not responding?** → Verify http://localhost:8085/health returns `{"status": "ok"}`
- **Still stuck?** → Check the main README for detailed documentation