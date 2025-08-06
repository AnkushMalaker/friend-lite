# Speaker Recognition System

A comprehensive speaker recognition system with web-based UI for audio annotation, speaker enrollment, and data management.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Hugging Face account (for model access)
- 8GB+ RAM, 10GB+ disk space

### 1. Set up your Hugging Face token (use .env recommended)
```bash
export HF_TOKEN="your_huggingface_token"
```
Get your token from https://huggingface.co/settings/tokens

### 2. Start the system
```bash
cd extras/speaker-recognition
docker-compose up --build -d
```

This starts two services:
- **FastAPI backend** on http://localhost:8085 (speaker recognition API)
- **React web UI** on https://localhost:5173 (Modern React interface with HTTPS)

### 3. Access the Web UI
- **React UI**: https://localhost:5173 (with HTTPS enabled for microphone access)

### 4. Get Started
1. **Create a user** using the sidebar
2. **Upload audio** in the "Audio Viewer" page
3. **Annotate segments** in the "Annotation" page
4. **Enroll speakers** in the "Enrollment" page
5. **Manage & export** data in the "Speakers" page

## 🎯 What You Can Do

- **📁 Upload & Visualize**: Interactive waveforms, spectrograms, segment selection
- **📝 Annotate Audio**: Label speaker segments, handle unknown speakers
- **👤 Enroll Speakers**: Register speakers with quality assessment
- **👥 Manage Speakers**: View statistics, compare quality, bulk operations
- **📤 Export Data**: Download audio in organized folders or concatenated files
- **📊 Analytics**: Track quality trends and system usage

## 🛠️ System Architecture

- **Multi-user Support**: Each user manages their own speaker data
- **SQLite Database**: Local storage for annotations, speakers, and sessions  
- **Quality Assessment**: Automatic audio quality scoring with recommendations
- **Export Formats**: 
  - Concatenated audio (max 10min per file)
  - Segmented files: `./exported_data/speaker-1/audio001.wav`
  - Metadata and annotations as JSON
- **Enrollment Tracking**: Tracks audio sample counts and total duration per speaker
- **Weighted Embeddings**: Smart speaker updates using weighted averaging

## 🎯 Processing Modes Comparison

The system offers multiple processing modes for different use cases:

| Mode | Name | Transcription | Diarization | Speaker ID | Use Case |
|------|------|--------------|-------------|------------|----------|
| **diarization-only** | Diarization Only | ❌ | ✅ Internal | ❌ | Basic speaker separation without identification |
| **speaker-identification** | Speaker Identification | ❌ | ✅ Internal | ✅ Enrolled | Identify known speakers without transcripts |
| **deepgram-enhanced** | Deepgram Enhanced | ✅ Deepgram | ✅ Deepgram | ✅ Replace | Full transcription with enhanced speaker names |
| **deepgram-transcript-internal-speakers** | Deepgram + Internal | ✅ Deepgram | ✅ Internal | ✅ Enrolled | Best transcription + precise speaker identification |
| **plain** | Plain (Legacy) | ❌ | ✅ Internal | ✅ Enrolled | Same as Speaker Identification |

### Mode Details

#### 🔹 Diarization Only
- **What it does**: Separates speakers into generic labels (Speaker A, Speaker B, etc.)
- **Best for**: Understanding speaker changes without needing names
- **Output**: Timestamped segments with generic speaker labels
- **Requirements**: Audio file only

#### 🔹 Speaker Identification  
- **What it does**: Separates speakers AND identifies enrolled speakers by name
- **Best for**: Identifying known speakers in meetings or conversations
- **Output**: Timestamped segments with actual speaker names
- **Requirements**: Enrolled speakers in the system

#### 🔹 Deepgram Enhanced
- **What it does**: Uses Deepgram for both transcription and diarization, then replaces generic speaker labels with enrolled speaker names
- **Best for**: High-quality transcription with speaker identification
- **Output**: Full transcript with identified speaker names
- **Requirements**: Deepgram API key + enrolled speakers

#### 🔹 Deepgram + Internal Speakers
- **What it does**: Uses Deepgram for transcription only, internal system for precise speaker diarization and identification
- **Best for**: Maximum accuracy for both transcription and speaker identification
- **Output**: Deepgram transcript with precisely identified speakers
- **Requirements**: Deepgram API key + enrolled speakers

### Quick Mode Selection Guide

- **Need transcription?** → Use Deepgram modes
- **Only need speaker names?** → Use Speaker Identification
- **Just want to see speaker changes?** → Use Diarization Only
- **Maximum accuracy?** → Use Deepgram + Internal Speakers
- **Best balance?** → Use Deepgram Enhanced

## 🖥️ React Web UI

The modern React interface provides an enhanced user experience with:

### Pages
- **Audio Viewer**: Interactive waveform visualization with click-to-play
- **Annotation**: Label speaker segments with Deepgram transcript support
- **Enrollment**: Record or upload audio with real-time quality assessment
- **Speakers**: Manage enrolled speakers with sample counts and duration metrics
- **Inference**: Identify speakers in new audio files with confidence scores

### Key Features
- **Recording Support**: Direct microphone recording with WebM to WAV conversion
- **Enrollment Options**: 
  - Create new speaker enrollment
  - Append to existing speaker (weighted embedding averaging)
  - Direct enrollment from annotation segments
- **Real-time Metrics**: Track sample counts and total audio duration
- **Quality Assessment**: SNR-based quality scoring with visual indicators
- **Export Options**: Download processed audio and annotation data

### Confidence Threshold Control

The React UI includes an adjustable confidence threshold on the Inference page that controls speaker identification strictness:

- **Range**: 0.00 to 1.00 (adjustable in 0.05 increments)
- **Default**: 0.50 (balanced accuracy vs coverage)
- **"Less Strict" (lower values)**: More segments identified as known speakers, but potentially more false positives
- **"More Strict" (higher values)**: Fewer segments identified, but higher accuracy for identified speakers
- **Typical ECAPA-TDNN values**: 0.10-0.30 for most use cases
- **Additional filtering**: Results can be filtered by confidence after processing

## 📖 Deepgram Terminology in Speaker Recognition Context

When importing Deepgram transcripts, you'll encounter different data structures:

### Words
- **What**: Individual word-level transcription data with precise timestamps
- **Use**: Most accurate for speaker boundaries and timing
- **Example**: `{"word": "hello", "start": 1.23, "end": 1.45, "speaker": 0}`

### Segments  
- **What**: Groups of consecutive words from the same speaker
- **Use**: Natural speech chunks for annotation and training
- **Generated by**: DeepgramParser groups words by speaker changes
- **Example**: Speaker 1 talks from 0-5s, then Speaker 2 from 5-10s

### Paragraphs/Sentences
- **What**: Deepgram's paragraph-level grouping based on pauses/context
- **Use**: Better for readability but less precise for speaker boundaries
- **Note**: May combine multiple speakers if they talk quickly
- **Example**: A paragraph might span 30-60 seconds with multiple speakers

### Best Practice
The speaker recognition system uses **word-level data** to create segments because:
- Exact speaker change points (critical for training)
- No overlapping speakers in a segment
- Accurate timing for audio extraction

When you see mismatched text in the UI, it's often because:
- Old imports used paragraph-level data
- Display shows cached/different parsing method
- Solution: Re-import using current word-level parser

## 📚 Documentation

For detailed documentation, API reference, and advanced usage:
- **[README.detailed.md](README.detailed.md)** - Comprehensive guide
- **[plan.md](plan.md)** - Implementation details

## 🔧 Configuration

Environment variables:
```bash
# Required
HF_TOKEN="your_token"                    # Required: Hugging Face token for PyAnnote models

# Speaker Service Configuration  
SPEAKER_SERVICE_HOST="0.0.0.0"          # Speaker service bind host
SPEAKER_SERVICE_PORT="8085"             # Speaker service port (default: 8085)
SPEAKER_SERVICE_URL="http://speaker-service:8085"  # URL for internal Docker communication
SIMILARITY_THRESHOLD="0.15"             # Speaker similarity threshold (0.1-0.3 typical for ECAPA-TDNN)

# React Web UI Configuration
REACT_UI_HOST="0.0.0.0"                # Web UI bind host  
REACT_UI_PORT="5173"                    # Web UI port (default: 5173)
REACT_UI_HTTPS="true"                   # Enable HTTPS for microphone access (default: true)

# Optional
DEEPGRAM_API_KEY="your_key"             # For transcript import features
DEV="false"                             # Enable development mode with reload
```

Copy `.env.template` to `.env` and configure your settings:
```bash
cp .env.template .env
# Edit .env with your configuration
```

## 🔒 HTTPS and Microphone Access

**For Internal VPN/Network Usage:**

The React UI is configured with HTTPS enabled by default (`REACT_UI_HTTPS=true`) to support microphone recording features, which require secure contexts in modern browsers.

### **First-time Setup:**
1. **Access**: Navigate to https://localhost:5173
2. **Certificate Warning**: Your browser will show a security warning for the self-signed certificate
3. **Accept Certificate**: Click "Advanced" → "Proceed to localhost (unsafe)" or similar
4. **One-time Setup**: This only needs to be done once per browser

### **Why HTTPS is Required:**
- **Browser Security**: Modern browsers require HTTPS for microphone access via `getUserMedia()` API
- **Internal Networks**: Self-signed certificates are acceptable for VPN/internal tools
- **Recording Features**: Both Enrollment and Inference pages need microphone access for live recording

### **Alternative Access:**
- **HTTP Fallback**: Set `REACT_UI_HTTPS=false` in `.env` and use http://localhost:5173
- **Limitation**: Microphone recording will not work without HTTPS (file upload still works)

## 🚨 Troubleshooting

**Can't access the web UI?**
- Check if services are running: `docker-compose ps`
- View logs: `docker-compose logs web-ui`

**Speaker service not responding?**
- Check backend logs: `docker-compose logs speaker-service`
- Verify HF_TOKEN is set correctly

**Models not downloading?**
- Ensure HF_TOKEN has access to PyAnnote models
- Check network connection and disk space

## 🔄 Development

For local development without Docker:
```bash
# Terminal 1 - Backend
uv sync
uv run python speaker_service.py

# Terminal 2 - React Web UI  
cd webui && npm run dev
```

## API Endpoints

### Health Check
```bash
GET /health
```
**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "speakers": 5
}
```

### Speaker Enrollment - Single File
```bash
POST /enroll/upload
Content-Type: multipart/form-data
```
**Form Fields:**
- `file`: Audio file (WAV/FLAC, max 3 minutes)
- `speaker_id`: Unique speaker identifier
- `speaker_name`: Speaker display name  
- `start`: Start time in seconds (optional)
- `end`: End time in seconds (optional)

**Response:**
```json
{
  "updated": false,
  "speaker_id": "john_doe"
}
```

### Speaker Enrollment - Batch
```bash
POST /enroll/batch
Content-Type: multipart/form-data
```
**Form Fields:**
- `files`: Multiple audio files for same speaker
- `speaker_id`: Unique speaker identifier
- `speaker_name`: Speaker display name

**Response:**
```json
{
  "updated": false,
  "speaker_id": "john_doe",
  "num_segments": 3,
  "num_files": 3,
  "total_duration": 45.2
}
```

### Speaker Enrollment - Append
```bash
POST /enroll/append
Content-Type: multipart/form-data
```
**Form Fields:**
- `files`: Multiple audio files to append to existing speaker
- `speaker_id`: Existing speaker identifier (must exist)

**Description:**
Appends new audio samples to an existing speaker enrollment using weighted embedding averaging. The system:
- Retrieves the existing speaker's embedding and sample count
- Processes new audio files to generate embeddings
- Computes weighted average: `(old_embedding * old_count + new_embeddings * new_count) / (old_count + new_count)`
- Updates the speaker with the combined embedding and new counts

**Response:**
```json
{
  "updated": true,
  "speaker_id": "john_doe",
  "previous_samples": 3,
  "new_samples": 2,
  "total_samples": 5,
  "previous_duration": 45.2,
  "new_duration": 28.7,
  "total_duration": 73.9
}
```

### Processing Mode Endpoints

#### Diarization Only
```bash
POST /v1/diarize-only
Content-Type: multipart/form-data
```
**Form Fields:**
- `file`: Audio file for diarization
- `min_duration`: Minimum segment duration (optional, default: 0.5)

**Response:**
```json
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 1.234,
      "end": 5.678,
      "duration": 4.444,
      "speaker_label": "SPEAKER_00",
      "identified": false,
      "status": "diarized_only"
    }
  ],
  "summary": {
    "total_duration": 120.5,
    "num_segments": 15,
    "num_speakers": 2,
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "processing_mode": "diarization_only"
  }
}
```

#### Speaker Identification
```bash
POST /diarize-and-identify
Content-Type: multipart/form-data
```
**Form Fields:**
- `file`: Audio file for processing
- `min_duration`: Minimum segment duration (optional, default: 0.5)
- `similarity_threshold`: Speaker similarity threshold (optional, default: 0.15)
- `identify_only_enrolled`: Return only identified speakers (optional, default: false)
- `user_id`: User ID for speaker identification (optional)

**Response:**
```json
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 1.234,
      "end": 5.678,
      "duration": 4.444,
      "identified_as": "John Doe",
      "identified_id": "john_doe",
      "confidence": 0.892,
      "status": "identified"
    }
  ],
  "summary": {
    "total_duration": 120.5,
    "num_segments": 15,
    "num_diarized_speakers": 3,
    "identified_speakers": ["John Doe", "Jane Smith"],
    "unknown_speakers": ["SPEAKER_02"],
    "similarity_threshold": 0.15,
    "filtered": false
  }
}
```

#### Deepgram Enhanced
```bash
POST /v1/listen
Content-Type: multipart/form-data
Authorization: Token YOUR_DEEPGRAM_API_KEY
```
**Query Parameters:**
- `model`: Deepgram model (default: nova-3)
- `language`: Language code (default: multi)
- `diarize`: Enable diarization (default: true)
- `enhance_speakers`: Enable speaker identification (default: true)
- `user_id`: User ID for speaker identification
- `speaker_confidence_threshold`: Speaker confidence threshold (default: 0.15)

**Response:** Deepgram response with enhanced speaker identification

#### Deepgram + Internal Speakers (Hybrid)
```bash
POST /v1/transcribe-and-diarize
Content-Type: multipart/form-data
Authorization: Token YOUR_DEEPGRAM_API_KEY
```
**Query Parameters:**
- Same as Deepgram Enhanced, plus:
- `similarity_threshold`: Internal speaker matching threshold (default: 0.15)
- `min_duration`: Minimum segment duration (default: 1.0)

**Response:** Deepgram transcription with internal speaker diarization and identification

### List Speakers
```bash
GET /speakers
```
**Response:**
```json
{
  "speakers": [
    {
      "id": "john_doe",
      "name": "John Doe",
      "user_id": 1,
      "created_at": "2024-01-15T10:30:00",
      "updated_at": "2024-01-15T14:20:00",
      "audio_sample_count": 5,
      "total_audio_duration": 73.9
    }
  ]
}
```

### Reset All Speakers
```bash
POST /speakers/reset
```
**Response:**
```json
{
  "reset": true
}
```

### Delete Speaker
```bash
DELETE /speakers/{speaker_id}
```
**Response:**
```json
{
  "deleted": true
}
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