# Debug System Tracker

The **Debug System Tracker** provides centralized monitoring and debugging for the audio processing pipeline in the Friend-Lite backend. It tracks transactions through the complete pipeline from audio reception to memory creation, giving you comprehensive visibility into system health and bottlenecks.

## Overview

The Debug System Tracker replaces scattered debug systems with a unified approach that:
- **Tracks complete pipeline transactions** from audio → transcription → memory
- **Provides real-time monitoring** via the Streamlit dashboard  
- **Captures detailed failure information** for debugging
- **Detects stalled transactions** automatically
- **Thread-safe and performant** with background monitoring
- **Exports debug dumps** for detailed analysis

## Architecture

```
Audio Ingestion → Transcription → Memory
      ↓               ↓            ↓
  AUDIO_RECEIVED → TRANSCRIPTION_* → MEMORY_*
      ↓               ↓            ↓          ↓
         Debug System Tracker Events
              ↓
     Dashboard Visualization
```

## Core Components

### Pipeline Stages

The tracker monitors these stages in the audio processing pipeline:

```python
class PipelineStage(Enum):
    AUDIO_RECEIVED = "audio_received"
    TRANSCRIPTION_STARTED = "transcription_started" 
    TRANSCRIPTION_COMPLETED = "transcription_completed"
    MEMORY_STARTED = "memory_started"
    MEMORY_COMPLETED = "memory_completed"
    CONVERSATION_ENDED = "conversation_ended"
```

### Transaction States

```python
class TransactionStatus(Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    STALLED = "stalled"  # Started but no progress in reasonable time
```

## Usage

### Getting the Debug Tracker

```python
from advanced_omi_backend.debug_system_tracker import get_debug_tracker, PipelineStage

# Get the global tracker instance
tracker = get_debug_tracker()
```

### Basic Transaction Tracking

```python
# Create a new transaction
transaction_id = tracker.create_transaction(
    user_id="507f1f77bcf86cd799439011",
    client_id="cd7994-laptop", 
    conversation_id="conv_123"
)

# Track events through the pipeline
tracker.track_event(transaction_id, PipelineStage.AUDIO_RECEIVED, True, 
                   chunk_size=1024)

tracker.track_event(transaction_id, PipelineStage.TRANSCRIPTION_STARTED)

# Mark successful completion
tracker.track_event(transaction_id, PipelineStage.TRANSCRIPTION_COMPLETED, True,
                   transcript_length=500, processing_time=2.5)

# Mark failure with error
tracker.track_event(transaction_id, PipelineStage.MEMORY_STARTED)
tracker.track_event(transaction_id, PipelineStage.MEMORY_COMPLETED, False,
                   error_message="Ollama connection timeout")
```

### Convenience Methods

```python
# Track audio chunks
tracker.track_audio_chunk(transaction_id, chunk_size=1024)

# Track WebSocket connections
tracker.track_websocket_connected(user_id, client_id)
tracker.track_websocket_disconnected(client_id)
```

### Real Usage Example (Memory Processing)

```python
# From memory_service.py
debug_tracker = get_debug_tracker()

# Start memory session tracking
session_id = debug_tracker.start_memory_session(
    audio_uuid, client_id, user_id, user_email
)
debug_tracker.start_memory_processing(session_id)

try:
    # Process memory
    result = process_memory.add(transcript, user_id=user_id, ...)
    
    # Record successful completion
    debug_tracker.complete_memory_processing(session_id, True)
    
except Exception as e:
    # Record failure
    debug_tracker.complete_memory_processing(session_id, False, str(e))
```

## Dashboard Integration

The Debug System Tracker integrates with the Streamlit dashboard to provide real-time monitoring:

### System Metrics
- **Uptime** - System running time in hours
- **Total Transactions** - All transactions processed
- **Active Transactions** - Currently in progress
- **Completed/Failed/Stalled** - Transaction outcomes
- **Active WebSockets** - Current connections
- **Processing Counts** - Audio chunks, transcriptions, memories

### Recent Activity
- **Recent Transactions** - Last 10 transactions with status and timing
- **Recent Issues** - Last 10 problems detected with descriptions
- **Active Users** - Users active in the last 5 minutes

### Transaction Details
Each transaction shows:
- **Transaction ID** (first 8 characters)
- **User ID** (last 6 characters for privacy)
- **Current Status** and **Stage**
- **Creation Time**
- **Issue Description** (if any problems detected)

## Advanced Features

### Automatic Stall Detection

The tracker automatically detects stalled transactions:

```python
# Background monitoring detects transactions stuck for >60 seconds
async def _monitor_stalled_transactions(self):
    while self._monitoring:
        for transaction in self.transactions.values():
            if transaction.status == TransactionStatus.IN_PROGRESS:
                elapsed = (now - transaction.updated_at).total_seconds()
                if elapsed > 60:  # 1 minute without progress
                    transaction.status = TransactionStatus.STALLED
```

### Issue Pattern Detection

The tracker identifies common failure patterns:

```python
def get_issue_description(self) -> Optional[str]:
    # Detects patterns like:
    # - "Transcription completed but memory creation failed"
    # - "Transcription completed but memory processing stalled"
    # - Stage-specific failures with error messages
```

### Debug Data Export

Export comprehensive debug information:

```python
# Export all system state to JSON file
debug_file = tracker.export_debug_dump()
# Creates: debug_dumps/debug_dump_<timestamp>.json

# Contains:
# - All transactions with complete event history
# - System metrics and timing
# - Recent issues and patterns
# - Active WebSocket connections
# - User activity tracking
```

## Configuration

### Environment Variables

```bash
# Debug dump directory (optional)
DEBUG_DUMP_DIR=debug_dumps
```

### Initialization

The tracker is automatically initialized in `main.py`:

```python
# Startup
init_debug_tracker()

# Shutdown  
shutdown_debug_tracker()
```

## API Reference

### Core Methods

#### `get_debug_tracker() -> DebugSystemTracker`
Get the global debug tracker singleton instance.

#### `create_transaction(user_id: str, client_id: str, conversation_id: Optional[str] = None) -> str`
Create a new pipeline transaction and return its ID.

#### `track_event(transaction_id: str, stage: PipelineStage, success: bool = True, error_message: Optional[str] = None, **metadata)`
Track an event in a transaction with optional metadata.

#### `track_audio_chunk(transaction_id: str, chunk_size: int = 0)`
Convenience method to track audio chunk processing.

#### `track_websocket_connected(user_id: str, client_id: str)`
Track WebSocket connection establishment.

#### `track_websocket_disconnected(client_id: str)`
Track WebSocket disconnection.

### Dashboard Data

#### `get_dashboard_data() -> Dict`
Get formatted data for the Streamlit dashboard including:
- System metrics
- Recent transactions (last 10)
- Recent issues (last 10)  
- Active user count

#### `get_transaction(transaction_id: str) -> Optional[Transaction]`
Get a specific transaction by ID.

#### `get_user_transactions(user_id: str, limit: int = 10) -> List[Transaction]`
Get recent transactions for a specific user.

### Debug Export

#### `export_debug_dump() -> Path`
Export comprehensive debug data to a timestamped JSON file.

## Integration Points

The Debug System Tracker is currently integrated into:

### WebSocket Audio Handling (`main.py:1782+`)
```python
tracker = get_debug_tracker()
tracker.track_websocket_connected(user.user_id, client_id)
# ... on disconnect:
tracker.track_websocket_disconnected(client_id)
```

### Audio Processing Pipeline (`main.py:1039+`)
```python
tracker = get_debug_tracker()
transaction_id = tracker.create_transaction(user.user_id, client_id)
tracker.track_event(transaction_id, PipelineStage.AUDIO_RECEIVED)
```

### Memory Processing (`memory_service.py:230+`)
```python
debug_tracker = get_debug_tracker()
session_id = debug_tracker.start_memory_session(audio_uuid, client_id, user_id, user_email)
debug_tracker.start_memory_processing(session_id)
```

### API Router (`api_router.py:415+`)
```python
debug_tracker = get_debug_tracker()
session_summary = debug_tracker.get_session_summary(audio_uuid)
```

## Best Practices

### 1. Track All Critical Pipeline Stages
```python
# Good - Complete pipeline tracking
transaction_id = tracker.create_transaction(user_id, client_id)
tracker.track_event(transaction_id, PipelineStage.AUDIO_RECEIVED)
tracker.track_event(transaction_id, PipelineStage.TRANSCRIPTION_STARTED)
# ... continue through all stages
```

### 2. Include Rich Metadata
```python
# Good - Detailed metadata for debugging
tracker.track_event(transaction_id, PipelineStage.TRANSCRIPTION_COMPLETED, True,
                   transcript_length=len(transcript),
                   processing_time_ms=elapsed_ms,
                   model_used="deepgram",
                   audio_duration=duration_seconds)
```

### 3. Handle Both Success and Failure
```python
try:
    result = await process_transcription()
    tracker.track_event(transaction_id, PipelineStage.TRANSCRIPTION_COMPLETED, True,
                       result_length=len(result))
except Exception as e:
    tracker.track_event(transaction_id, PipelineStage.TRANSCRIPTION_COMPLETED, False,
                       error_message=str(e), retry_count=attempt_num)
```

### 4. Use Proper Transaction Lifecycle
```python
# Create transaction when pipeline starts
transaction_id = tracker.create_transaction(user_id, client_id, conversation_id)

# Track through all stages
# Always end with CONVERSATION_ENDED for completion
tracker.track_event(transaction_id, PipelineStage.CONVERSATION_ENDED, True)
```

## Troubleshooting

### Common Issues

**Q: Transactions stuck in IN_PROGRESS**  
A: Check that your code calls `track_event()` with success/failure for all pipeline stages. Stalled transactions are automatically detected after 60 seconds.

**Q: Missing transactions in dashboard**  
A: Ensure you're importing from the correct module: `from advanced_omi_backend.debug_system_tracker import get_debug_tracker`

**Q: Memory usage growing**  
A: The tracker automatically limits to 100 recent transactions and 50 recent issues. For high volume, consider the cleanup mechanisms.

**Q: Background monitoring not working**  
A: Ensure `init_debug_tracker()` is called at startup. Check logs for monitoring task errors.

### Debug Tips

1. **Check recent issues**: Use `get_dashboard_data()["recent_issues"]` to see detected problems
2. **Monitor transaction patterns**: Use `get_user_transactions()` to see user-specific pipeline flow
3. **Export debug dumps**: Use `export_debug_dump()` for detailed offline analysis
4. **Watch stall detection**: Transactions with no progress for >60 seconds are automatically flagged

## Migration Notes

This system replaces various old debug tracking approaches:

### From Old Memory Debug System
```python
# Old approach (if any existed)
memory_debug.start_session(audio_uuid)
memory_debug.log_processing(...)

# New approach
tracker = get_debug_tracker()
transaction_id = tracker.create_transaction(user_id, client_id)
tracker.track_event(transaction_id, PipelineStage.MEMORY_STARTED)
```

### From Scattered Logging
```python
# Old approach
logger.info(f"Processing audio for {user_id}")
logger.info(f"Transcription completed: {len(result)}")

# New approach (includes logging + structured tracking)
tracker.track_event(transaction_id, PipelineStage.TRANSCRIPTION_COMPLETED, True,
                   transcript_length=len(result))
```

The Debug System Tracker provides comprehensive visibility into the audio processing pipeline while maintaining performance and thread safety.