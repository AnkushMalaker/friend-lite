# Wyoming Protocol Implementation

## Overview
The system uses Wyoming protocol for WebSocket communication between mobile apps and backends. Wyoming is a peer-to-peer protocol for voice assistants that combines JSONL headers with binary audio payloads.

## Protocol Format
```
{JSON_HEADER}\n
<binary_payload>
```

## Supported Events

### Audio Session Events
- **audio-start**: Signals the beginning of an audio recording session
  ```json
  {"type": "audio-start", "data": {"rate": 16000, "width": 2, "channels": 1}, "payload_length": null}
  ```

- **audio-chunk**: Contains raw audio data with format metadata
  ```json
  {"type": "audio-chunk", "data": {"rate": 16000, "width": 2, "channels": 1}, "payload_length": 320}
  <320 bytes of PCM/Opus audio data>
  ```

- **audio-stop**: Signals the end of an audio recording session
  ```json
  {"type": "audio-stop", "data": {"timestamp": 1234567890}, "payload_length": null}
  ```

## Backend Implementation

### Advanced Backend (`/ws_pcm`)
- **Full Wyoming Protocol Support**: Parses all Wyoming events for session management
- **Session Tracking**: Only processes audio chunks when session is active (after audio-start)
- **Conversation Boundaries**: Uses audio-start/stop events to define conversation segments
- **Backward Compatibility**: Fallback to raw binary audio for older clients

### Simple Backend (`/ws`)
- **Minimal Wyoming Support**: Parses audio-chunk events, ignores others
- **Opus Processing**: Handles Opus-encoded audio chunks from Wyoming protocol
- **Graceful Degradation**: Falls back to raw Opus packets for compatibility

## Mobile App Integration

Mobile apps should implement Wyoming protocol for proper session management:

```javascript
// Start audio session
const audioStart = {
  type: "audio-start",
  data: { rate: 16000, width: 2, channels: 1 },
  payload_length: null
};
websocket.send(JSON.stringify(audioStart) + '\n');

// Send audio chunks
const audioChunk = {
  type: "audio-chunk",
  data: { rate: 16000, width: 2, channels: 1 },
  payload_length: audioData.byteLength
};
websocket.send(JSON.stringify(audioChunk) + '\n');
websocket.send(audioData);

// End audio session
const audioStop = {
  type: "audio-stop",
  data: { timestamp: Date.now() },
  payload_length: null
};
websocket.send(JSON.stringify(audioStop) + '\n');
```

## Benefits
- **Clear Session Boundaries**: No timeout-based conversation detection needed
- **Structured Communication**: Consistent protocol across all audio streaming
- **Future Extensibility**: Room for additional event types (pause, resume, metadata)
- **Backward Compatibility**: Works with existing raw audio streaming clients