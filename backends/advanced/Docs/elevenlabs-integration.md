# ElevenLabs Speech-to-Text Integration Guide

## Overview

This document outlines the integration of ElevenLabs Speech-to-Text (Scribe v1 model) as a transcription provider for Friend-Lite.

## ElevenLabs Capabilities

### Core Features
- **Model**: Scribe v1 with state-of-the-art accuracy
- **API Endpoint**: `https://api.elevenlabs.io/v1/speech-to-text`
- **Authentication**: API key via `xi-api-key` header
- **Languages**: 99 languages with automatic detection
- **Speaker Diarization**: Up to 32 speakers
- **Word-Level Timestamps**: Precise timing for each word
- **Audio Events**: Optional detection of laughter, applause, etc.

### Technical Specifications
- **Mode**: Batch processing only (no streaming support)
- **Format**: Multipart/form-data file upload
- **Max File Size**: 3 GB
- **Max Duration**: 10 hours
- **Supported Formats**: 18+ audio formats (AAC, MP3, WAV, FLAC, Opus, WebM, etc.)

### Output Format
```json
{
  "text": "Full transcript text",
  "language_code": "en",
  "language_probability": 0.95,
  "words": [
    {
      "text": "word",
      "start": 0.5,
      "end": 1.2,
      "type": "word",
      "speaker_id": "speaker_1",
      "logprob": -0.05
    }
  ]
}
```

## Pricing

| Tier | Price/Month | Hours Included | Cost per Hour |
|------|-------------|----------------|---------------|
| Starter | $5 | 12.5 | $0.40 |
| Creator | $22 | 62.85 | $0.35 |
| Pro | $99 | 300 | $0.33 |
| Scale | $330 | 1,100 | $0.30 |

**Comparison**: Deepgram Nova-3 costs ~$0.36/hour (pay-as-you-go)

## Integration Architecture

### Provider Implementation

Create `backends/advanced/src/advanced_omi_backend/services/transcription/elevenlabs.py`:

```python
"""
ElevenLabs transcription provider implementation.

Provides batch transcription using ElevenLabs Scribe v1 model.
"""

import io
import logging
from typing import Dict, Optional

import httpx

from advanced_omi_backend.models.transcription import BatchTranscriptionProvider

logger = logging.getLogger(__name__)


class ElevenLabsProvider(BatchTranscriptionProvider):
    """ElevenLabs batch transcription provider using Scribe v1 model."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.elevenlabs.io/v1/speech-to-text"

    @property
    def name(self) -> str:
        return "elevenlabs"

    async def transcribe(self, audio_data: bytes, sample_rate: int, diarize: bool = False) -> dict:
        """Transcribe audio using ElevenLabs REST API.

        Args:
            audio_data: Raw audio bytes (will be converted to WAV format)
            sample_rate: Audio sample rate
            diarize: Whether to enable speaker diarization
        """
        try:
            # Convert raw PCM to WAV format for ElevenLabs
            wav_data = self._pcm_to_wav(audio_data, sample_rate)

            # Prepare multipart form data
            files = {
                'file': ('audio.wav', io.BytesIO(wav_data), 'audio/wav')
            }

            data = {
                'model_id': 'scribe_v1',
                'diarize': 'true' if diarize else 'false',
                'timestamps_granularity': 'word',
                'tag_audio_events': 'false',  # Optional: set to true for laughter/applause detection
            }

            headers = {
                'xi-api-key': self.api_key
            }

            logger.info(f"Sending {len(audio_data)} bytes to ElevenLabs API (diarize={diarize})")

            # Calculate timeout based on audio duration
            estimated_duration = len(audio_data) / (sample_rate * 2)  # 16-bit mono
            processing_timeout = max(120, int(estimated_duration * 5))  # 5x audio duration

            timeout_config = httpx.Timeout(
                connect=30.0,
                read=processing_timeout,
                write=180.0,
                pool=10.0,
            )

            logger.info(
                f"Estimated audio duration: {estimated_duration:.1f}s, timeout: {processing_timeout}s"
            )

            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    self.url,
                    headers=headers,
                    data=data,
                    files=files
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"ElevenLabs response: {result}")

                    # Parse ElevenLabs response format
                    transcript = result.get('text', '').strip()

                    # Extract word-level data
                    words = []
                    segments = []

                    if 'words' in result:
                        # Map ElevenLabs format to Friend-Lite format
                        for word_obj in result['words']:
                            if word_obj.get('type') == 'word':  # Skip spacing/audio_events
                                words.append({
                                    'word': word_obj.get('text', ''),
                                    'start': word_obj.get('start', 0),
                                    'end': word_obj.get('end', 0),
                                    'confidence': 1.0 - abs(word_obj.get('logprob', 0)),  # Convert logprob to confidence
                                    'speaker': word_obj.get('speaker_id'),
                                })

                    # Extract speaker segments if diarization is enabled
                    if diarize and words:
                        segments = self._create_speaker_segments(words)

                    logger.info(
                        f"ElevenLabs transcription successful: {len(transcript)} chars, "
                        f"{len(words)} words, {len(segments)} segments"
                    )

                    return {
                        "text": transcript,
                        "words": words,
                        "segments": segments,
                    }
                else:
                    logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                    return {"text": "", "words": [], "segments": []}

        except httpx.TimeoutException as e:
            logger.error(f"Timeout during ElevenLabs API call: {e}")
            return {"text": "", "words": [], "segments": []}
        except Exception as e:
            logger.error(f"Error calling ElevenLabs API: {e}")
            return {"text": "", "words": [], "segments": []}

    def _pcm_to_wav(self, pcm_data: bytes, sample_rate: int) -> bytes:
        """Convert raw PCM data to WAV format."""
        import wave
        import io

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)

        return wav_buffer.getvalue()

    def _create_speaker_segments(self, words: list) -> list:
        """Group consecutive words by speaker into segments."""
        segments = []
        current_speaker = None
        current_segment = None

        for word in words:
            speaker = word.get('speaker')
            if speaker is None:
                continue

            if speaker == current_speaker and current_segment:
                # Extend current segment
                current_segment['text'] += ' ' + word['word']
                current_segment['end'] = word['end']
            else:
                # Save previous segment and start new one
                if current_segment:
                    segments.append(current_segment)
                current_segment = {
                    'text': word['word'],
                    'speaker': f"Speaker {speaker}",
                    'start': word['start'],
                    'end': word['end'],
                    'confidence': word.get('confidence'),
                }
                current_speaker = speaker

        # Don't forget the last segment
        if current_segment:
            segments.append(current_segment)

        return segments
```

### Factory Integration

Update `backends/advanced/src/advanced_omi_backend/services/transcription/__init__.py`:

#### 1. Add Import
```python
from advanced_omi_backend.services.transcription.elevenlabs import ElevenLabsProvider
```

#### 2. Update `get_transcription_provider()` Function

Add after line 46:
```python
def get_transcription_provider(
    provider_name: Optional[str] = None,
    mode: Optional[str] = None,
) -> Optional[BaseTranscriptionProvider]:
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    parakeet_url = os.getenv("PARAKEET_ASR_URL")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")  # Add this line

    # ... existing code ...

    # Add after parakeet/offline sections (around line 88):
    elif provider_name == "elevenlabs":
        if not elevenlabs_key:
            raise RuntimeError(
                "ElevenLabs transcription provider requested but ELEVENLABS_API_KEY not configured"
            )
        logger.info(f"Using ElevenLabs transcription provider in {mode} mode")
        if mode == "streaming":
            raise RuntimeError("ElevenLabs does not support streaming mode - use batch mode")
        return ElevenLabsProvider(elevenlabs_key)
```

#### 3. Update `__all__` Export
```python
__all__ = [
    "get_transcription_provider",
    "DeepgramProvider",
    "DeepgramStreamingProvider",
    "DeepgramStreamConsumer",
    "ParakeetProvider",
    "ParakeetStreamingProvider",
    "ElevenLabsProvider",  # Add this
]
```

### Model Update

Update `backends/advanced/src/advanced_omi_backend/models/transcription.py`:

```python
class TranscriptionProvider(Enum):
    """Available transcription providers for audio stream routing."""
    DEEPGRAM = "deepgram"
    PARAKEET = "parakeet"
    MISTRAL = "mistral"
    ELEVENLABS = "elevenlabs"  # Add this line
```

## Configuration

### Environment Variables

Update `backends/advanced/.env.template` (around line 48):

```bash
# ========================================
# SPEECH-TO-TEXT CONFIGURATION (Choose one)
# ========================================

# Option 1: Deepgram (recommended for best transcription quality)
DEEPGRAM_API_KEY=

# Option 2: ElevenLabs (high quality with 99 language support)
# Get your API key from: https://elevenlabs.io/app/settings/api-keys
# ELEVENLABS_API_KEY=

# Option 3: Mistral (Voxtral models)
# MISTRAL_API_KEY=
# MISTRAL_MODEL=voxtral-mini-2507

# Option 4: Parakeet ASR service from extras/asr-services
# PARAKEET_ASR_URL=http://host.docker.internal:8767

# Optional: Specify which provider to use ('deepgram', 'elevenlabs', 'mistral', or 'parakeet')
# If not set, will auto-select based on available configuration (Deepgram preferred)
# TRANSCRIPTION_PROVIDER=elevenlabs
```

### Usage Example

```bash
# In .env file
ELEVENLABS_API_KEY=sk_your_api_key_here
TRANSCRIPTION_PROVIDER=elevenlabs

# Start the backend
docker compose up --build -d
```

## Implementation Checklist

- [ ] Create `elevenlabs.py` provider implementation
- [ ] Update `__init__.py` factory function
- [ ] Add `ELEVENLABS` to `TranscriptionProvider` enum
- [ ] Update `.env.template` with configuration
- [ ] Update `CLAUDE.md` documentation
- [ ] Run integration tests
- [ ] Update API documentation

## Testing

### Unit Tests
```bash
cd backends/advanced
uv run pytest tests/test_transcription_providers.py -k elevenlabs
```

### Integration Tests
```bash
cd backends/advanced

# Set environment variables
export ELEVENLABS_API_KEY=sk_your_key_here
export TRANSCRIPTION_PROVIDER=elevenlabs

# Run full integration test
./run-test.sh
```

### Manual Testing
```bash
# Test with audio file upload
curl -X POST http://localhost:8000/api/audio/upload \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "audio_file=@test_audio.wav" \
  -F "client_id=test-client"
```

## Key Implementation Notes

### 1. File Format Conversion
ElevenLabs requires proper audio file formats (not raw bytes). The provider converts raw PCM to WAV:
```python
def _pcm_to_wav(self, pcm_data: bytes, sample_rate: int) -> bytes:
    """Convert raw PCM data to WAV format."""
```

### 2. Batch-Only Processing
ElevenLabs does not support streaming transcription. All audio must be sent as complete files.

### 3. Confidence Score Mapping
ElevenLabs returns `logprob` (log probability) which needs conversion to confidence (0-1):
```python
'confidence': 1.0 - abs(word_obj.get('logprob', 0))
```

### 4. Speaker Diarization
Automatic speaker identification is built-in and returns `speaker_id` in word-level data. The provider groups consecutive words from the same speaker into segments.

### 5. Timeout Configuration
Processing timeout is dynamically calculated based on audio duration:
```python
processing_timeout = max(120, int(estimated_duration * 5))  # 5x audio duration
```

## Comparison with Other Providers

| Feature | ElevenLabs | Deepgram | Parakeet |
|---------|------------|----------|----------|
| **Streaming** | ❌ No | ✅ Yes | ✅ Yes |
| **Batch** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Languages** | 99 languages | Multi-language | English-focused |
| **Diarization** | ✅ 32 speakers | ✅ Yes | ❌ No |
| **Word Timestamps** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Cost** | $0.30-0.40/hr | ~$0.36/hr | Free (self-hosted) |
| **Setup** | API key | API key | Docker service |

## Limitations

1. **No Streaming**: Real-time transcription not supported
2. **File Size**: Maximum 3 GB per file
3. **Duration**: Maximum 10 hours per file
4. **API Dependency**: Requires internet connection and ElevenLabs service availability
5. **Confidence Scores**: Log probability conversion may not be as accurate as native confidence scores

## Future Enhancements

1. **Async Processing**: Use webhooks for long audio files
2. **Audio Events**: Enable `tag_audio_events` for laughter/applause detection
3. **Multi-Channel**: Support `use_multi_channel` for separate channel transcription
4. **Custom Formats**: Support additional output formats (SRT, DOCX, PDF)

## Speaker Recognition Service Integration

The speaker recognition service (`extras/speaker-recognition`) also needs ElevenLabs support to enhance transcriptions with speaker identification.

### Current Architecture

The speaker recognition service:
1. Acts as a proxy/wrapper for transcription services (currently Deepgram only)
2. Forwards transcription requests to the ASR provider
3. Enhances responses with speaker identification using enrolled speakers
4. Returns enriched transcripts with `identified_speaker_id` and `identified_speaker_name`

### Integration Components Required

#### 1. ElevenLabs Parser (`elevenlabs_parser.py`)
Similar to `deepgram_parser.py`, this will:
- Parse ElevenLabs JSON response format
- Extract speaker segments from word-level data
- Group consecutive words by speaker_id
- Convert logprob to confidence scores
- Provide speaker statistics

#### 2. ElevenLabs Wrapper Endpoint (`elevenlabs_wrapper.py`)
Similar to `deepgram_wrapper.py`, this will:
- Accept audio file uploads
- Forward to ElevenLabs API with diarization enabled
- Extract speaker segments from response
- Identify speakers using enrolled voice embeddings
- Return enhanced response with speaker identification

#### 3. Service Configuration Updates
- Add `elevenlabs_api_key` to Settings
- Register ElevenLabs router
- Update environment templates

### Key Differences from Deepgram Integration

| Aspect | Deepgram | ElevenLabs |
|--------|----------|------------|
| **Speaker Field** | `speaker` (integer) | `speaker_id` (string) |
| **Confidence** | Native `confidence` field | Derived from `logprob` |
| **Word Filtering** | All words included | Filter by `type == "word"` |
| **API Endpoint** | `/v1/listen` | `/v1/speech-to-text` |
| **Auth Header** | `Authorization: Token` | `xi-api-key` |
| **Request Format** | Query params + raw audio | Multipart form data |

### Implementation Plan

1. **Create Parser** (`utils/elevenlabs_parser.py`)
   - Parse JSON response
   - Group words by `speaker_id`
   - Filter word-type entries only
   - Convert logprob to confidence

2. **Create Wrapper** (`api/routers/elevenlabs_wrapper.py`)
   - `/elevenlabs/v1/transcribe` endpoint
   - Forward to ElevenLabs API
   - Enhance with speaker identification
   - Return enriched response

3. **Update Configuration**
   - Add `ELEVENLABS_API_KEY` to `.env.template`
   - Add field to Settings class
   - Register router in main app

4. **Testing**
   - Unit tests for parser
   - Integration tests for wrapper
   - End-to-end with real audio

### Usage Example

```bash
# Transcribe with speaker identification
curl -X POST http://localhost:8085/elevenlabs/v1/transcribe \
  -H "xi-api-key: YOUR_ELEVENLABS_KEY" \
  -F "file=@audio.wav" \
  -F "diarize=true" \
  -F "model_id=scribe_v1" \
  "?user_id=1&enhance_speakers=true&speaker_confidence_threshold=0.15"
```

Expected response with enhancement:
```json
{
  "text": "Hello, how are you today?",
  "language_code": "en",
  "words": [
    {
      "text": "Hello",
      "start": 0.1,
      "end": 0.5,
      "type": "word",
      "speaker_id": "speaker_1",
      "identified_speaker_id": 42,
      "identified_speaker_name": "John Doe",
      "speaker_identification_confidence": 0.87,
      "speaker_status": "IDENTIFIED"
    }
  ],
  "speaker_enhancement": {
    "enabled": true,
    "provider": "elevenlabs",
    "user_id": 1,
    "identified_speakers": {
      "speaker_1": {
        "speaker_id": 42,
        "speaker_name": "John Doe",
        "confidence": 0.87
      }
    },
    "total_segments": 3,
    "identified_segments": 2
  }
}
```

## Part 3: Web UI Integration (Batch Inference)

### Overview

The Speaker Recognition Web UI provides a batch inference page where users can upload audio files and process them with different transcription providers. This integration adds ElevenLabs as a processing mode option alongside existing Deepgram modes.

**Note**: ElevenLabs is **batch-only** and does not support real-time streaming. The live inference pages will continue to use Deepgram WebSocket for real-time transcription.

### Architecture

```
User uploads audio file
        ↓
[Inference Page UI]
        ↓
Select "ElevenLabs Transcribe" mode
        ↓
Frontend service: elevenlabs.ts
        ↓
POST /elevenlabs/v1/transcribe
        ↓
Backend wrapper → ElevenLabs API
        ↓
Enhanced response with speaker IDs
        ↓
Display results with transcription + speakers
```

### Implementation Details

#### 1. ElevenLabs Service (`webui/src/services/elevenlabs.ts`)

Create new service module similar to `deepgram.ts`:

```typescript
// Type definitions
export interface ElevenLabsTranscriptionOptions {
  model_id?: string               // Default: 'scribe_v1'
  diarize?: boolean               // Default: true
  timestamps_granularity?: string // Default: 'word'
  tag_audio_events?: boolean      // Default: false
  enhanceSpeakers?: boolean       // Enable speaker identification
  userId?: number                 // For speaker enhancement
  speakerConfidenceThreshold?: number
}

export interface ElevenLabsWord {
  text: string
  start: number
  end: number
  type: 'word' | 'spacing' | 'audio_event'
  speaker_id?: string
  logprob?: number
  identified_speaker_id?: string
  identified_speaker_name?: string
  speaker_identification_confidence?: number
  speaker_status?: string
}

export interface ElevenLabsResponse {
  text: string
  language_code?: string
  language_probability?: number
  words: ElevenLabsWord[]
  speaker_enhancement?: {
    enabled: boolean
    provider: string
    user_id?: number
    identified_speakers: Record<string, any>
    total_segments: number
  }
}

// Main transcription function
export async function transcribeWithElevenLabs(
  file: File | Blob,
  options: ElevenLabsTranscriptionOptions = {}
): Promise<ElevenLabsResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('model_id', options.model_id || 'scribe_v1')
  formData.append('diarize', String(options.diarize !== false))
  formData.append('timestamps_granularity', options.timestamps_granularity || 'word')
  formData.append('tag_audio_events', String(options.tag_audio_events || false))

  const params = new URLSearchParams()
  if (options.enhanceSpeakers && options.userId) {
    params.append('user_id', String(options.userId))
    params.append('enhance_speakers', 'true')
    params.append('speaker_confidence_threshold', String(options.speakerConfidenceThreshold || 0.15))
  }

  const response = await apiService.post(
    `/elevenlabs/v1/transcribe?${params}`,
    formData
  )
  return response.data
}

// Process response into segments
export function processElevenLabsResponse(response: ElevenLabsResponse) {
  // Filter only words (skip spacing and audio events)
  const words = response.words.filter(w => w.type === 'word')

  // Group consecutive words by speaker_id
  const segments = []
  let currentSegment = null

  for (const word of words) {
    if (!word.speaker_id) continue

    if (currentSegment && currentSegment.speaker_id === word.speaker_id) {
      // Extend current segment
      currentSegment.text += ' ' + word.text
      currentSegment.end = word.end
    } else {
      // Save previous and start new
      if (currentSegment) segments.push(currentSegment)
      currentSegment = {
        speaker_id: word.speaker_id,
        speaker: word.identified_speaker_name || `Speaker ${word.speaker_id}`,
        text: word.text,
        start: word.start,
        end: word.end,
        confidence: 1.0 - Math.abs(word.logprob || 0),
        identifiedSpeakerId: word.identified_speaker_id,
        identifiedSpeakerName: word.identified_speaker_name,
        speakerIdentificationConfidence: word.speaker_identification_confidence,
        speakerStatus: word.speaker_status
      }
    }
  }
  if (currentSegment) segments.push(currentSegment)

  return segments
}

// Calculate confidence summary
export function calculateConfidenceSummary(segments) {
  const total = segments.length
  const high = segments.filter(s => s.confidence >= 0.8).length
  const medium = segments.filter(s => s.confidence >= 0.5 && s.confidence < 0.8).length
  const low = segments.filter(s => s.confidence < 0.5).length

  return {
    total_segments: total,
    high_confidence: high,
    medium_confidence: medium,
    low_confidence: low
  }
}
```

#### 2. Speaker Identification Service Updates

**File**: `webui/src/services/speakerIdentification.ts`

Add ElevenLabs processing mode:

```typescript
// Update type
export type ProcessingMode =
  | 'diarization-only'
  | 'speaker-identification'
  | 'deepgram-enhanced'
  | 'deepgram-transcript-internal-speakers'
  | 'diarize-identify-match'
  | 'elevenlabs-enhanced'  // NEW

// Add processing method
private async processWithElevenLabs(
  audioFile: File | Blob,
  options: ProcessingOptions
): Promise<ProcessingResult> {
  try {
    const filename = audioFile instanceof File ? audioFile.name : 'Audio'

    const elevenlabsResponse = await transcribeWithElevenLabs(audioFile, {
      enhanceSpeakers: options.enhanceSpeakers !== false,
      userId: options.userId,
      speakerConfidenceThreshold: options.confidenceThreshold || 0.15,
    })

    const elevenlabsSegments = processElevenLabsResponse(elevenlabsResponse)

    const speakerSegments: SpeakerSegment[] = elevenlabsSegments.map(segment => ({
      start: segment.start,
      end: segment.end,
      speaker_id: segment.speaker_id,
      speaker_name: segment.identifiedSpeakerName || segment.speaker,
      confidence: segment.confidence,
      text: segment.text,
      identified_speaker_id: segment.identifiedSpeakerId,
      identified_speaker_name: segment.identifiedSpeakerName,
      speaker_identification_confidence: segment.speakerIdentificationConfidence,
      speaker_status: segment.speakerStatus
    }))

    const confidenceSummary = calculateConfidenceSummary(elevenlabsSegments)

    return {
      id: Math.random().toString(36),
      filename,
      duration: this.estimateDuration(speakerSegments),
      status: 'completed',
      created_at: new Date().toISOString(),
      mode: 'elevenlabs-enhanced',
      speakers: speakerSegments,
      confidence_summary: confidenceSummary,
    }
  } catch (error) {
    throw new Error(`ElevenLabs processing failed: ${error.message}`)
  }
}

// Update processAudio switch
async processAudio(audioFile: File | Blob, options: ProcessingOptions): Promise<ProcessingResult> {
  const startTime = Date.now()

  try {
    let result: ProcessingResult

    switch (options.mode) {
      case 'elevenlabs-enhanced':  // NEW
        result = await this.processWithElevenLabs(audioFile, options)
        break
      case 'deepgram-enhanced':
        result = await this.processWithDeepgram(audioFile, options)
        break
      // ... other cases
    }

    result.processing_time = Date.now() - startTime
    return result
  } catch (error) {
    // ... error handling
  }
}
```

#### 3. Processing Mode Selector Updates

**File**: `webui/src/components/ProcessingModeSelector.tsx`

Add ElevenLabs to modes array:

```typescript
const PROCESSING_MODES: ProcessingModeConfig[] = [
  {
    mode: 'speaker-identification',
    name: 'Speaker Identification',
    description: 'Diarization + speaker recognition only',
    icon: '🎯',
    color: 'bg-blue-600 hover:bg-blue-700',
    features: ['Speaker diarization', 'Speaker identification', 'Confidence scoring']
  },
  {
    mode: 'deepgram-enhanced',
    name: 'Transcribe + Identify',
    description: 'Full transcription with enhanced speaker ID',
    icon: '🚀',
    color: 'bg-green-600 hover:bg-green-700',
    requirements: ['Deepgram API key'],
    features: ['High-quality transcription', 'Speaker diarization', 'Enhanced speaker identification', 'Word-level timing']
  },
  {
    mode: 'elevenlabs-enhanced',  // NEW
    name: 'ElevenLabs Transcribe',
    description: '99 languages with speaker diarization',
    icon: '🌐',
    color: 'bg-indigo-600 hover:bg-indigo-700',
    requirements: ['ElevenLabs API key (configured in backend)'],
    features: [
      '99 language support',
      'Built-in speaker diarization',
      'Enhanced speaker identification',
      'Word-level timestamps'
    ]
  },
  // ... other modes
]
```

### Usage Flow

1. **User Navigation**: Navigate to `/inference` page
2. **Audio Input**: Upload an audio file or record audio
3. **Mode Selection**: Select "🌐 ElevenLabs Transcribe" from dropdown
4. **Processing**: Click "Process Audio" button
5. **Backend Flow**:
   - Frontend → `POST /elevenlabs/v1/transcribe`
   - Backend wrapper → ElevenLabs API
   - Speaker enhancement adds `identified_speaker_name` fields
6. **Results Display**: View transcription with speaker identification

### Limitations

**Batch-Only Processing:**
- ✅ Works: `/inference` page (batch file upload)
- ❌ Doesn't work: `/infer-live-simple` and `/infer-live` (require WebSocket streaming)
- ElevenLabs does not support real-time streaming, so live inference pages will continue using Deepgram

**Language Support:**
- Automatic language detection (99 languages)
- No need to specify language code

**Speaker Diarization:**
- Maximum 32 speakers
- Automatic speaker detection
- No manual speaker count configuration needed

### Testing

```bash
# 1. Ensure backend is running with ElevenLabs configured
cd extras/speaker-recognition
docker compose up -d

# 2. Navigate to web UI
open https://your-host:8444/inference

# 3. Test workflow:
# - Upload a WAV file with multiple speakers
# - Select "ElevenLabs Transcribe" mode
# - Set user_id for speaker identification
# - Click "Process Audio"
# - Verify transcription and speaker names appear

# 4. Check network requests:
# POST /elevenlabs/v1/transcribe
# Response should include speaker_enhancement metadata
```

### Integration Checklist

**Advanced Backend:**
- [x] Create `services/transcription/elevenlabs.py`
- [x] Update `services/transcription/__init__.py`
- [x] Update `models/transcription.py` enum
- [x] Update `.env.template`
- [x] Update `init.py` wizard
- [x] Update root `wizard.py`
- [x] Configure API key in `.env`
- [x] Update CLAUDE.md
- [ ] Run integration tests

**Speaker Recognition Service (Backend):**
- [x] Create `utils/elevenlabs_parser.py`
- [x] Create `api/routers/elevenlabs_wrapper.py`
- [x] Update `api/service.py` Settings class
- [x] Register ElevenLabs router in main app
- [x] Update `.env.template`
- [x] Update `init.py` wizard
- [x] Configure API key in `.env`
- [ ] Add parser unit tests
- [ ] Add wrapper integration tests

**Speaker Recognition Web UI:**
- [x] Create `webui/src/services/elevenlabs.ts`
- [x] Update `webui/src/services/speakerIdentification.ts`
- [x] Update `webui/src/components/ProcessingModeSelector.tsx`
- [ ] Test batch inference with ElevenLabs mode

## References

- [ElevenLabs Speech-to-Text Docs](https://elevenlabs.io/docs/capabilities/speech-to-text)
- [API Reference](https://elevenlabs.io/docs/api-reference/speech-to-text/convert)
- [Quickstart Guide](https://elevenlabs.io/docs/cookbooks/speech-to-text/quickstart)
- [Friend-Lite Transcription Architecture](./transcription-architecture.md)
