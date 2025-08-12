# ASR Services

Offline Automatic Speech Recognition (ASR) services for Friend-Lite using the Wyoming protocol.

## Overview

The ASR services provide offline speech recognition capabilities using:
- **Moonshine ASR**: Fast, efficient speech recognition
- **Parakeet ASR**: Alternative ASR model with different characteristics
- **Wyoming Protocol**: Standardized audio processing communication
- **Real-time Processing**: Stream audio chunks and receive transcriptions

## Quick Start

### Start Parakeet ASR Service (Available)
```bash
# Start the Parakeet ASR service
docker compose up parakeet-asr -d

# Check service health
docker compose logs parakeet-asr
```

### Start Moonshine ASR Service (Commented Out)
```bash
# Note: Moonshine service is currently commented out in docker-compose.yml
# To enable, uncomment the moonshine-asr service and change port to avoid conflicts

# Uncomment lines 25-47 in docker-compose.yml, then:
# docker compose up moonshine-asr -d
```

## Architecture

### Wyoming Protocol Support
The ASR services implement Wyoming AsyncTCPServer that:
- Accepts Wyoming AudioChunk events
- Returns StreamingTranscript events (Wyoming Transcript with final flag)
- Supports real-time audio processing
- Handles dynamic sample rates and audio formats

### Audio Processing
- **Dynamic Sample Rate Support**: Adapts to input audio sample rates
- **VAD Integration**: Voice Activity Detection for better processing
- **Streaming Transcription**: Real-time transcript generation
- **Multiple Audio Formats**: PCM, Opus, and other common formats

## Services

### Parakeet ASR (Available)
- **Port**: 8765 (default)
- **Protocol**: Wyoming TCP
- **Features**: GPU-accelerated speech recognition
- **Resource**: GPU acceleration required (NVIDIA)
- **Status**: Active service

### Moonshine ASR (Commented Out)
- **Port**: 8765 (conflicts with Parakeet, needs port change)
- **Protocol**: Wyoming TCP
- **Features**: Fast transcription, low latency
- **Resource**: CPU optimized
- **Status**: Commented out in docker-compose.yml

## Configuration

### Environment Variables
```bash
# Service configuration
MOONSHINE_PORT=8765
PARAKEET_PORT=8766

# Audio settings (auto-detected from input)
DEFAULT_SAMPLE_RATE=16000
SUPPORTED_RATES=8000,16000,44100,48000

# Resource allocation
GPU_ACCELERATION=true
WORKER_THREADS=4
```

### Docker Configuration
```yaml
# docker-compose.yml
services:
  moonshine:
    build: .
    ports:
      - "8765:8765"
    environment:
      - SERVICE_TYPE=moonshine
      
  parakeet:
    build: .
    ports:
      - "8766:8766"
    environment:
      - SERVICE_TYPE=parakeet
      - GPU_ACCELERATION=true
```

## Integration

### With Friend-Lite Backend
The ASR services integrate as fallback transcription when Deepgram is unavailable:
```bash
# Backend configuration
OFFLINE_ASR_TCP_URI=tcp://localhost:8765  # Moonshine
# or
OFFLINE_ASR_TCP_URI=tcp://localhost:8766  # Parakeet
```

### Client Usage
```python
# Example Wyoming client
import asyncio
from wyoming.client import AsyncTcpClient
from wyoming.audio import AudioChunk

async def transcribe_audio(audio_data):
    client = AsyncTcpClient("localhost", 8765)
    await client.connect()
    
    # Send audio chunk
    chunk = AudioChunk(
        audio=audio_data,
        rate=16000,
        width=2,
        channels=1
    )
    await client.send(chunk)
    
    # Receive transcript
    transcript = await client.receive()
    return transcript.text
```

## Performance

| Service | Latency | Throughput | Memory Usage | GPU Required |
|---------|---------|------------|--------------|---------------|
| Moonshine | ~200ms | 10x realtime | 2GB RAM | No |
| Parakeet | ~150ms | 15x realtime | 4GB RAM | Optional |

## Troubleshooting

### Common Issues

1. **Service not starting**
   ```bash
   # Check container logs
   docker compose logs moonshine
   
   # Verify port availability
   netstat -an | grep 8765
   ```

2. **Audio format errors**
   - Ensure audio is in supported format (PCM, 16-bit)
   - Check sample rate compatibility
   - Verify Wyoming protocol message format

3. **Low transcription quality**
   - Check audio quality and noise levels
   - Verify correct sample rate settings
   - Consider switching between Moonshine and Parakeet

4. **Performance issues**
   - Monitor CPU/GPU usage
   - Adjust worker thread count
   - Consider hardware acceleration

### Debug Mode
```bash
# Enable debug logging
docker compose up moonshine -d --env DEBUG=true

# Monitor real-time logs
docker compose logs -f moonshine
```

## Development

### Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run service directly
python moonshine-online.py --port 8765 --debug
```

### Testing
```bash
# Test with audio file
python client.py --host localhost --port 8765 --audio test.wav

# Run Wyoming protocol tests
python -m pytest tests/
```

## Notes

- Streaming is fully supported for real-time transcription
- Services auto-detect optimal sample rates from input audio
- GPU acceleration is optional but recommended for Parakeet
- Multiple clients can connect simultaneously
- Automatic fallback between services when one is unavailable 