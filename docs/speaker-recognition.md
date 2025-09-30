# Speaker Recognition Service Features

## Speaker Analysis & Visualization
The speaker recognition service now includes advanced analysis capabilities:

### Embedding Analysis (/speakers/analysis endpoint)
- **2D/3D Visualization**: Interactive embedding plots using UMAP, t-SNE, or PCA
- **Clustering Analysis**: Automatic clustering using DBSCAN or K-means
- **Speaker Similarity Detection**: Identifies speakers with similar embeddings
- **Quality Metrics**: Embedding separation quality and confidence scores
- **Interactive Controls**: Adjustable analysis parameters and visualization options

Access via: `extras/speaker-recognition/webui` → Speakers → Embedding Analysis tab

### Live Inference Feature (/infer-live page)
Real-time speaker identification and transcription:
- **WebRTC Audio Capture**: Live microphone access with waveform visualization
- **Deepgram Streaming**: Real-time transcription with speaker diarization
- **Live Speaker ID**: Identifies enrolled speakers in real-time using internal service
- **Session Statistics**: Live metrics for words, speakers, and confidence scores
- **Configurable Settings**: Adjustable confidence thresholds and audio parameters

Access via: `extras/speaker-recognition/webui` → Live Inference

## Technical Implementation

### Backend (Python)
- **Analysis Utils**: `src/simple_speaker_recognition/utils/analysis.py`
  - UMAP/t-SNE dimensionality reduction
  - DBSCAN/K-means clustering
  - Cosine similarity analysis
  - Quality metrics calculation
- **API Endpoint**: `/speakers/analysis` - Returns processed embedding analysis
- **Dependencies**: Added `umap-learn` for dimensionality reduction

### Frontend (React/TypeScript)
- **EmbeddingPlot Component**: Interactive Plotly.js visualizations
- **LiveAudioCapture Component**: WebRTC audio recording with waveform
- **DeepgramStreaming Service**: WebSocket integration for real-time transcription
- **InferLive Page**: Complete live inference interface

## Usage Instructions

### Setting up Live Inference
1. Navigate to Live Inference page
2. Configure Deepgram API key in settings
3. Adjust speaker identification settings (confidence threshold)
4. Start live session to begin real-time transcription and speaker ID

**Technical Details:**
- **Audio Processing**: Uses browser's native sample rate (typically 44.1kHz or 48kHz)
- **Buffer Retention**: 120 seconds of audio for improved utterance capture
- **Real-time Updates**: Live transcription with speaker identification results

### Using Speaker Analysis
1. Go to Speakers page → Embedding Analysis tab
2. Select analysis method (UMAP, t-SNE, PCA)
3. Choose clustering algorithm (DBSCAN, K-means)
4. Adjust similarity threshold for speaker detection
5. View interactive plots and quality metrics

## Deployment Notes
- Requires Docker rebuild to pick up new Python dependencies
- Frontend dependencies (Plotly.js) already included
- Live inference requires Deepgram API key for streaming transcription
- Speaker identification uses existing enrolled speakers from database

## Live Inference Troubleshooting
- **"NaN:NaN" timestamps**: Fixed in recent updates, ensure you're using latest version
- **Poor speaker identification**: Try adjusting confidence threshold or re-enrolling speakers
- **Audio processing delays**: Check browser console for sample rate detection logs
- **Buffer overflow issues**: Extended to 120-second retention for better performance
- **"extraction_failed" errors**: Usually indicates audio buffer timing issues - check console logs for buffer availability