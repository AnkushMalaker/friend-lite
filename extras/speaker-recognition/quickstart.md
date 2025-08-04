# Speaker Recognition Quick Start Guide

This guide will help you quickly get started with the new speaker analysis and live inference features.

## Prerequisites

- Docker and Docker Compose installed
- Microphone access for live inference
- (Optional) Deepgram API key for live transcription

## Getting Started

### 1. Start the Speaker Recognition Service

```bash
cd extras/speaker-recognition
docker compose up --build -d
```

This will start:
- **Speaker Recognition API** on http://localhost:8001
- **Web UI** on http://localhost:3000

### 2. Access the Web Interface

Open http://localhost:3000 in your browser. You'll see the Speaker Recognition System interface with navigation for:

- **Audio Viewer**: Upload and view audio files
- **Annotation**: Annotate speaker segments
- **Enrollment**: Enroll new speakers
- **Speakers**: Manage speakers + **NEW: Embedding Analysis**
- **Inference**: Process audio files
- **Live Inference**: **NEW: Real-time transcription and speaker ID**

## New Features

### 🔬 Speaker Embedding Analysis

**Location**: Speakers page → "Embedding Analysis" tab

**What it does**: Visualizes how similar or different your enrolled speakers are by plotting their voice embeddings in 2D/3D space.

**Quick Start**:
1. Make sure you have at least 2-3 enrolled speakers
2. Go to Speakers page → Embedding Analysis tab
3. Click "Refresh Analysis" to generate the plot
4. Explore the interactive visualization:
   - **Points** = individual speakers
   - **Colors** = clusters of similar speakers
   - **Hover** = speaker details and confidence

**Settings**:
- **Reduction Method**: UMAP (recommended), t-SNE, or PCA
- **Clustering Method**: DBSCAN (recommended) or K-means
- **Similarity Threshold**: 0.8 (adjust to find more/fewer similar speakers)

**What to look for**:
- ✅ **Well-separated speakers**: Good speaker recognition quality
- ⚠️ **Clustered speakers**: May indicate similar voices or enrollment issues
- 📊 **Quality Metrics**: Shows separation quality and confidence scores

### 🎙️ Live Inference

**Location**: Live Inference page (new navigation item)

**What it does**: Real-time transcription with speaker identification as you speak.

**Quick Start**:
1. **Get a Deepgram API Key**:
   - Sign up at https://console.deepgram.com/
   - Copy your API key from the dashboard

2. **Configure Live Inference**:
   - Go to Live Inference page
   - Click Settings (⚙️ icon)
   - Paste your Deepgram API key
   - Enable "Speaker Identification" (uses your enrolled speakers)
   - Adjust confidence threshold (0.15 = balanced, higher = stricter)

3. **Start Live Session**:
   - Click "Start Live Session"
   - Allow microphone access when prompted
   - Start speaking - you'll see real-time transcription
   - Enrolled speakers will be identified automatically

**Features**:
- **Live transcription** with word-level timestamps
- **Speaker diarization** (separates different speakers)
- **Real-time speaker identification** using your enrolled speakers
- **Session statistics** (words, speakers, confidence)
- **Audio waveform visualization**

## Troubleshooting

### Speaker Analysis Issues

**Problem**: "No speakers found" or empty analysis
- **Solution**: Make sure you have enrolled speakers first (go to Enrollment page)

**Problem**: Analysis shows all speakers clustered together
- **Solution**: Check speaker enrollment quality - may need more diverse audio samples

**Problem**: Plot doesn't load
- **Solution**: Check browser console for errors, try refreshing the page

### Live Inference Issues

**Problem**: "Failed to start session" error
- **Solution**: 
  1. Check Deepgram API key is correct
  2. Ensure you have internet connection
  3. Verify microphone permissions in browser

**Problem**: Transcription works but no speakers identified
- **Solution**:
  1. Check "Enable Speaker Identification" is turned on
  2. Make sure you have enrolled speakers for the selected user
  3. Lower the confidence threshold in settings

**Problem**: Audio capture fails
- **Solution**:
  1. Allow microphone access in browser
  2. Check if another application is using the microphone
  3. Try refreshing the page and allowing permissions again

### General Issues

**Problem**: Services not starting
```bash
# Check service status
docker compose ps

# View logs
docker compose logs speaker-recognition
docker compose logs webui

# Rebuild if needed
docker compose down
docker compose up --build -d
```

**Problem**: API requests failing
- **Solution**: Ensure speaker recognition service is running on port 8001
- Check: http://localhost:8001/health should return {"status": "ok"}

## API Endpoints

### New Endpoints

#### Speaker Analysis
```bash
# Get embedding analysis
curl "http://localhost:8001/speakers/analysis?user_id=1&method=umap&cluster_method=dbscan"
```

#### Health Check
```bash
# Check service status
curl http://localhost:8001/health
```

## Development Notes

### Backend Changes
- Added `src/simple_speaker_recognition/utils/analysis.py` - analysis utilities
- Added `/speakers/analysis` endpoint in API service
- Added `umap-learn` dependency for dimensionality reduction

### Frontend Changes
- New `EmbeddingPlot` component for interactive visualizations
- New `LiveAudioCapture` component for WebRTC audio
- Extended Deepgram service with streaming support
- New `InferLive` page for real-time inference
- Updated navigation to include new features

### Dependencies Added
- **Python**: `umap-learn>=0.5.3`
- **JavaScript**: Plotly.js (already included)

## Advanced Usage

### Custom Analysis Parameters

You can customize the analysis via URL parameters:
```bash
# Use t-SNE with K-means clustering
http://localhost:3000/speakers?analysis=true&method=tsne&cluster=kmeans

# Adjust similarity threshold
http://localhost:3000/speakers?analysis=true&similarity=0.7
```

### Programmatic Access

```python
import requests

# Get speaker analysis
response = requests.get(
    "http://localhost:8001/speakers/analysis",
    params={
        "user_id": 1,
        "method": "umap",
        "cluster_method": "dbscan",
        "similarity_threshold": 0.8
    }
)

analysis = response.json()
print(f"Found {analysis['clustering']['n_clusters']} clusters")
print(f"Quality score: {analysis['quality_metrics']['separation_quality']:.3f}")
```

## Next Steps

1. **Enroll diverse speakers** to see better analysis results
2. **Experiment with different analysis methods** (UMAP vs t-SNE vs PCA)
3. **Try live inference** with different confidence thresholds
4. **Monitor speaker quality** using the embedding analysis
5. **Integrate with your application** using the REST API

## Support

For issues or questions:
1. Check the main project README
2. Review Docker logs: `docker compose logs`
3. Verify all services are healthy: `docker compose ps`

The speaker recognition system is now ready for both analysis and real-time inference!