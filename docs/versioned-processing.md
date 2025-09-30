# Versioned Processing System

## Overview

Friend-Lite implements a comprehensive versioning system for both transcript and memory processing, allowing multiple processing attempts with different providers, models, or settings while maintaining a clean user experience.

## Version Data Structure

### Transcript Versions
```json
{
  "transcript_versions": [
    {
      "version_id": "uuid",
      "transcript": "processed text",
      "segments": [...],
      "provider": "deepgram|mistral|parakeet",
      "model": "nova-3|voxtral-mini-2507",
      "created_at": "2025-01-15T10:30:00Z",
      "processing_time_seconds": 12.5,
      "metadata": {
        "confidence_scores": [...],
        "speaker_diarization": true
      }
    }
  ],
  "active_transcript_version": "uuid"
}
```

### Memory Versions
```json
{
  "memory_versions": [
    {
      "version_id": "uuid",
      "memory_count": 5,
      "transcript_version_id": "uuid",
      "provider": "friend_lite|openmemory_mcp",
      "model": "gpt-4o-mini|ollama-llama3",
      "created_at": "2025-01-15T10:32:00Z",
      "processing_time_seconds": 45.2,
      "metadata": {
        "prompt_version": "v2.1",
        "extraction_quality": "high"
      }
    }
  ],
  "active_memory_version": "uuid"
}
```

## Database Schema Details

### Collections Overview
- **`audio_chunks`**: All audio sessions by `audio_uuid` (always created)
- **`conversations`**: Speech-detected conversations by `conversation_id` (created conditionally)
- **`users`**: User accounts and authentication data

### Speech-Driven Schema
```javascript
// audio_chunks collection (always created)
{
  "_id": ObjectId,
  "audio_uuid": "uuid",  // Primary identifier
  "user_id": ObjectId,
  "client_id": "user_suffix-device_name",
  "audio_file_path": "/path/to/audio.wav",
  "created_at": ISODate,
  "transcript": "fallback transcript",  // For non-speech audio
  "segments": [...],  // Speaker segments
  "has_speech": boolean,  // Speech detection result
  "speech_analysis": {...},  // Detection metadata
  "conversation_id": "conv_id" | null  // Link to conversations collection
}

// conversations collection (speech-detected only)
{
  "_id": ObjectId,
  "conversation_id": "conv_uuid",  // Primary identifier for user-facing operations
  "audio_uuid": "audio_uuid",  // Link to audio_chunks
  "user_id": ObjectId,
  "client_id": "user_suffix-device_name",
  "created_at": ISODate,

  // Versioned Transcript System
  "transcript_versions": [
    {
      "version_id": "uuid",
      "transcript": "text content",
      "segments": [...],  // Speaker diarization
      "provider": "deepgram|mistral|parakeet",
      "model": "nova-3|voxtral-mini-2507",
      "created_at": ISODate,
      "processing_time_seconds": 12.5,
      "metadata": {...}
    }
  ],
  "active_transcript_version": "uuid",  // Points to current version

  // Versioned Memory System
  "memory_versions": [
    {
      "version_id": "uuid",
      "memory_count": 5,
      "transcript_version_id": "uuid",  // Which transcript was used
      "provider": "friend_lite|openmemory_mcp",
      "model": "gpt-4o-mini|ollama-llama3",
      "created_at": ISODate,
      "processing_time_seconds": 45.2,
      "metadata": {...}
    }
  ],
  "active_memory_version": "uuid",  // Points to current version

  // Legacy Fields (auto-populated from active versions)
  "transcript": "text",  // From active_transcript_version
  "segments": [...],     // From active_transcript_version
  "memories": [...],     // From active_memory_version
  "memory_count": 5      // From active_memory_version
}
```

## Reprocessing Workflows

### Transcript Reprocessing
1. Trigger via API: `POST /api/conversations/{conversation_id}/reprocess-transcript`
2. System creates new transcript version with different provider/model
3. New version added to `transcript_versions` array
4. User can activate any version via `activate-transcript` endpoint
5. Legacy `transcript` field automatically updated from active version

### Memory Reprocessing
1. Trigger via API: `POST /api/conversations/{conversation_id}/reprocess-memory`
2. Specify which transcript version to use as input
3. System creates new memory version using specified transcript
4. New version added to `memory_versions` array
5. User can activate any version via `activate-memory` endpoint
6. Legacy `memories` field automatically updated from active version

## Legacy Field Compatibility

### Automatic Population
- `transcript`: Auto-populated from active transcript version
- `segments`: Auto-populated from active transcript version
- `memories`: Auto-populated from active memory version
- `memory_count`: Auto-populated from active memory version

### Backward Compatibility
- Existing API clients continue working without modification
- WebUI displays active versions by default
- Advanced users can access version history and switch between versions

## Data Consistency
- All reprocessing operations use `conversation_id` (not `audio_uuid`)
- DateTime objects stored as ISO strings for MongoDB/JSON compatibility
- Legacy field support ensures existing integrations continue working

## Key Architecture Benefits
- **Clean Separation**: Raw audio storage vs user-facing conversations
- **Speech Filtering**: Only meaningful conversations appear in UI
- **Version History**: Complete audit trail of processing attempts
- **Backward Compatibility**: Legacy fields ensure existing code works
- **Reprocessing Support**: Easy to re-run with different providers/models
- **Service Decoupling**: Conversation creation independent of memory processing
- **Error Isolation**: Memory service failures don't affect conversation storage