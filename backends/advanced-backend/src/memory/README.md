# Disclaimer - AI generated during development
# Memory Service

The Memory Service is a core component of the Friend-Lite backend that provides persistent memory and action item management capabilities using [Mem0](https://mem0.ai/) with local storage.

## Features

### 🧠 **Memory Management**
- **Persistent Memory**: Store and retrieve conversation memories across sessions
- **Semantic Search**: Find relevant memories using vector similarity search
- **User-Scoped Storage**: Memories are isolated per user for privacy and organization
- **Metadata Support**: Rich metadata storage for enhanced filtering and retrieval

### 🎯 **Action Items**
- **Automatic Extraction**: Extract action items from conversation transcripts using LLM
- **Manual Creation**: Create action items directly via API
- **Status Management**: Track action item progress (open, in_progress, completed, cancelled)
- **Priority & Assignment**: Set priority levels and assign to specific users
- **Due Date Tracking**: Basic due date support for task management
- **Search & Filtering**: Find action items by text content or filter by status
- **Statistics**: Get comprehensive statistics and breakdowns by status, priority, assignee

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  Memory Service │    │     Mem0        │
│   Endpoints     │───▶│                 │───▶│    (Local)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │    Ollama       │    │    Qdrant       │
                       │  (LLM & Embed)  │    │  (Vector DB)    │
                       └─────────────────┘    └─────────────────┘
```

## Configuration

The memory service is configured via environment variables:

```bash
# Mem0 Configuration
MEM0_ORGANIZATION_ID=friend-lite-org      # Organization identifier
MEM0_PROJECT_ID=audio-conversations       # Project identifier  
MEM0_APP_ID=omi-backend                   # Application identifier
MEM0_TELEMETRY=False                      # Disable telemetry for privacy

# Backend Services
OLLAMA_BASE_URL=http://ollama:11434       # Ollama server URL
QDRANT_BASE_URL=qdrant                    # Qdrant server host
```

## API Usage

### Memory Operations

```python
from memory.memory_service import get_memory_service

memory_service = get_memory_service()

# Add memory from conversation
success = memory_service.add_memory(
    transcript="User discussed their preferences...",
    client_id="user123",
    audio_uuid="conv_456"
)

# Search memories
memories = memory_service.search_memories(
    query="What are the user's preferences?",
    user_id="user123",
    limit=10
)

# Get all memories
all_memories = memory_service.get_all_memories(
    user_id="user123",
    limit=100
)
```

### Action Item Operations

```python
# Extract action items from transcript
count = memory_service.extract_and_store_action_items(
    transcript="I'll send the report by Friday and schedule a meeting",
    client_id="user123", 
    audio_uuid="conv_456"
)

# Get action items
action_items = memory_service.get_action_items(
    user_id="user123",
    status_filter="open",  # Optional: filter by status
    limit=50
)

# Search action items
results = memory_service.search_action_items(
    query="report",
    user_id="user123",
    limit=20
)

# Update action item status
success = memory_service.update_action_item_status(
    memory_id="mem_789",
    new_status="completed"
)

# Delete action item
success = memory_service.delete_action_item(memory_id="mem_789")
```

## REST API Endpoints

### Memory Endpoints
- `GET /api/memories?user_id={user_id}` - Get all memories
- `GET /api/memories/search?user_id={user_id}&query={query}` - Search memories
- `DELETE /api/memories/{memory_id}` - Delete specific memory

### Action Item Endpoints
- `GET /api/action-items?user_id={user_id}` - Get action items
- `POST /api/action-items?user_id={user_id}` - Create action item
- `PUT /api/action-items/{memory_id}` - Update action item status
- `DELETE /api/action-items/{memory_id}` - Delete action item
- `GET /api/action-items/search?user_id={user_id}&query={query}` - Search action items
- `GET /api/action-items/stats?user_id={user_id}` - Get statistics
- `POST /api/conversations/{audio_uuid}/extract-action-items` - Extract from conversation

## Action Item Schema

Action items are stored with the following structure:

```json
{
  "memory_id": "unique_memory_identifier",
  "description": "Task description",
  "assignee": "person_responsible", 
  "due_date": "deadline_or_not_specified",
  "priority": "high|medium|low|not_specified",
  "status": "open|in_progress|completed|cancelled",
  "context": "when/why_mentioned",
  "created_at": 1234567890,
  "updated_at": 1234567890,
  "source": "manual_creation|transcript_extraction",
  "audio_uuid": "conversation_id_if_extracted"
}
```

## Local Storage Stack

The service uses a completely local storage stack:

- **Ollama**: Local LLM for action item extraction and embeddings
  - Model: `llama3.1:latest` for text processing
  - Embeddings: `nomic-embed-text:latest` for vector representations
- **Qdrant**: Local vector database for memory storage and semantic search
- **No External APIs**: Everything runs locally for privacy and control

## Action Item Extraction

Action items are automatically extracted from conversations using a specialized LLM prompt that looks for:

- Task commitments ("I'll send the report by Friday")
- Requests ("Can you review the document?")
- Scheduling needs ("Let's schedule a meeting")
- Follow-up actions ("We need to contact the client")

The extraction happens at the end of each conversation session and is stored with full context.

## ⚠️ **Important Limitations**

### Mem0 Update Method Warning

**CRITICAL**: The Mem0 `memory.update()` method has a significant limitation - **it destroys all metadata** when updating memory content. This breaks action item functionality since we rely on metadata for:

- Action item identification (`metadata.type = "action_item"`)
- Structured data storage (`metadata.action_item_data`)
- Source tracking and timestamps

**Current Workaround**: The `update_action_item_status()` method uses Mem0's `update()` which will work for text content but will break subsequent retrievals due to metadata loss.

**Symptoms of this issue**:
- Update appears successful (HTTP 200)
- Verification fails (HTTP 500) 
- Action items become unretievable
- Search returns 0 results

**Future Solution**: When Mem0 fixes the update method or provides metadata-preserving updates, the implementation should be updated accordingly.

### Recommended Practices

1. **Testing Updates**: Always test action item retrieval after status updates
2. **Backup Strategy**: Consider implementing a backup/restore mechanism for critical action items
3. **Alternative Approach**: For production use, consider implementing a custom update that:
   - Creates new memory with updated data
   - Preserves all metadata
   - Deletes old memory
   - Updates references to use new memory_id

## Development

### Running Tests

```bash
# Run the comprehensive API test suite
cd backends/advanced-backend
uv run python3 test_action_items.py
```

The test suite covers:
- Backend health checks
- User management
- Action item CRUD operations
- Status updates and verification
- Search functionality
- Statistics generation
- Data cleanup

### Monitoring

Enable debug logging to monitor memory operations:

```python
import logging
logging.getLogger("memory_service").setLevel(logging.DEBUG)
```

Key metrics to monitor:
- Memory creation success rate
- Action item extraction accuracy
- Search response times
- Update operation failures

## Troubleshooting

### Common Issues

1. **Memory Service Not Initialized**
   - Check Ollama and Qdrant connectivity
   - Verify environment variables
   - Check service startup logs

2. **Action Items Not Found**
   - Verify metadata structure
   - Check for update method metadata loss
   - Confirm user_id consistency

3. **Search Returns Empty Results** 
   - Check embedding model availability
   - Verify Qdrant collection health
   - Confirm query format

4. **Update Operations Failing**
   - See metadata loss warning above
   - Check memory_id validity
   - Verify user permissions

### Debug Commands

```bash
# Check Ollama models
curl http://localhost:11434/api/tags

# Check Qdrant health
curl http://localhost:6333/collections

# Test memory service
curl http://localhost:8000/health
```

## Future Enhancements

- [ ] Fix Mem0 update metadata preservation
- [ ] Add bulk operations for action items
- [ ] Implement action item dependencies
- [ ] Add deadline notifications
- [ ] Enhanced search with filters
- [ ] Export/import functionality
- [ ] Integration with calendar systems
- [ ] Action item templates

## License

This memory service is part of the Friend-Lite project. See the main project LICENSE for details. 