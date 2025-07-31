# Disclaimer - AI generated during development
# Memory Service

The Memory Service is a core component of the Friend-Lite backend that provides persistent memory capabilities using [Mem0](https://mem0.ai/) with local storage.

## Features

### 🧠 **Memory Management**
- **Persistent Memory**: Store and retrieve conversation memories across sessions
- **Semantic Search**: Find relevant memories using vector similarity search
- **User-Scoped Storage**: Memories are isolated per user for privacy and organization
- **Metadata Support**: Rich metadata storage for enhanced filtering and retrieval


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


## REST API Endpoints

### Memory Endpoints
- `GET /api/memories?user_id={user_id}` - Get all memories
- `GET /api/memories/search?user_id={user_id}&query={query}` - Search memories
- `DELETE /api/memories/{memory_id}` - Delete specific memory



## Local Storage Stack

The service uses a completely local storage stack:

- **Ollama**: Local LLM for embeddings
  - Model: `llama3.1:latest` for text processing
  - Embeddings: `nomic-embed-text:latest` for vector representations
- **Qdrant**: Local vector database for memory storage and semantic search
- **No External APIs**: Everything runs locally for privacy and control


## ⚠️ **Important Limitations**

### Mem0 Update Method Warning


## Development

### Running Tests

```bash
# Run the comprehensive API test suite
cd backends/advanced-backend
uv run python3 test_memory_service.py
```

The test suite covers:
- Backend health checks
- User management
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
- Search response times
- Update operation failures

## Troubleshooting

### Common Issues

1. **Memory Service Not Initialized**
   - Check Ollama and Qdrant connectivity
   - Verify environment variables
   - Check service startup logs


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

- [ ] Enhanced search with filters
- [ ] Export/import functionality
- [ ] Integration with calendar systems

## License

This memory service is part of the Friend-Lite project. See the main project LICENSE for details. 