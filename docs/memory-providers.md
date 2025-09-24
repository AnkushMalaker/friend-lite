# Memory System Architecture

## Overview
Friend-Lite supports two pluggable memory backends that can be selected via configuration:

## 1. Friend-Lite Memory Provider (`friend_lite`)
The sophisticated in-house memory implementation with full control and customization:

### Features
- Custom LLM-powered memory extraction with enhanced prompts
- Individual fact storage (no JSON blobs)
- Smart deduplication algorithms
- Intelligent memory updates (ADD/UPDATE/DELETE decisions)
- **Semantic search** with relevance threshold filtering
- **Memory count API** with total count tracking from native Qdrant
- Direct Qdrant vector storage with accurate similarity scoring
- Custom memory prompts and processing
- No external dependencies

### Architecture Flow
1. **Audio Input** → Transcription via Deepgram/Parakeet
2. **Memory Extraction** → LLM processes transcript using custom prompts
3. **Fact Parsing** → XML/JSON parsing into individual memory entries
4. **Deduplication** → Smart algorithms prevent duplicate memories
5. **Vector Storage** → Direct Qdrant storage with embeddings
6. **Memory Updates** → LLM-driven action proposals (ADD/UPDATE/DELETE)

## 2. OpenMemory MCP Provider (`openmemory_mcp`)
Thin client that delegates all memory processing to external OpenMemory MCP server:

### Features
- Professional memory extraction (handled by OpenMemory)
- Battle-tested deduplication (handled by OpenMemory)
- Semantic vector search (handled by OpenMemory)
- ACL-based user isolation (handled by OpenMemory)
- Cross-client compatibility (Claude Desktop, Cursor, Windsurf)
- Web UI for memory management at http://localhost:8765

### Architecture Flow
1. **Audio Input** → Transcription via Deepgram/Parakeet
2. **MCP Delegation** → Send enriched transcript to OpenMemory MCP server
3. **External Processing** → OpenMemory handles extraction, deduplication, storage
4. **Result Mapping** → Convert MCP results to Friend-Lite MemoryEntry format
5. **Client Management** → Automatic user context switching via MCP client

## Memory Provider Comparison

| Feature | Friend-Lite | OpenMemory MCP |
|---------|-------------|----------------|
| **Processing** | Custom LLM extraction | Delegates to OpenMemory |
| **Deduplication** | Custom algorithms | OpenMemory handles |
| **Vector Storage** | Direct Qdrant | OpenMemory handles |
| **Search Features** | Semantic search with threshold filtering | Semantic search with relevance scoring |
| **Memory Count** | Native Qdrant count API | Varies by OpenMemory support |
| **Dependencies** | Qdrant + MongoDB | External OpenMemory server |
| **Customization** | Full control | Limited to OpenMemory features |
| **Cross-client** | Friend-Lite only | Works with Claude Desktop, Cursor, etc |
| **Web UI** | Friend-Lite WebUI with advanced search | OpenMemory UI + Friend-Lite WebUI |
| **Memory Format** | Individual facts | OpenMemory format |
| **Setup Complexity** | Medium | High (external server required) |

## Switching Memory Providers

You can switch providers by changing the `MEMORY_PROVIDER` environment variable:

```bash
# Switch to OpenMemory MCP
echo "MEMORY_PROVIDER=openmemory_mcp" >> .env

# Switch back to Friend-Lite
echo "MEMORY_PROVIDER=friend_lite" >> .env
```

**Note:** Existing memories are not automatically migrated between providers. Each provider maintains its own memory storage.

## OpenMemory MCP Setup

To use the OpenMemory MCP provider:

```bash
# 1. Start external OpenMemory MCP server
cd extras/openmemory-mcp
docker compose up -d

# 2. Configure Friend-Lite to use OpenMemory MCP
cd backends/advanced
echo "MEMORY_PROVIDER=openmemory_mcp" >> .env

# 3. Start Friend-Lite backend
docker compose up --build -d
```

## OpenMemory MCP Interface Patterns

**Important**: OpenMemory MCP stores memories **per-app**, not globally. Understanding this architecture is critical for proper integration.

### App-Based Storage Architecture
- All memories are stored under specific "apps" (namespaces)
- Generic endpoints (`/api/v1/memories/`) return empty results
- App-specific endpoints (`/api/v1/apps/{app_id}/memories`) contain the actual memories

### Hardcoded Values and Configuration
```bash
# Default app name (configurable via OPENMEMORY_CLIENT_NAME)
Default: "friend_lite"

# Hardcoded metadata (NOT configurable)
"source": "friend_lite"  # Always hardcoded in Friend-Lite

# User ID for OpenMemory MCP server
OPENMEMORY_USER_ID=openmemory  # Configurable
```

### API Interface Pattern
```python
# 1. App Discovery - Find app by client_name
GET /api/v1/apps/
# Response: {"apps": [{"id": "uuid", "name": "friend_lite", ...}]}

# 2. Memory Creation - Uses generic endpoint but assigns to app
POST /api/v1/memories/
{
  "user_id": "openmemory",
  "text": "memory content",
  "app": "friend_lite",  # Uses OPENMEMORY_CLIENT_NAME
  "metadata": {
    "source": "friend_lite",    # Hardcoded
    "client": "friend_lite"     # Uses OPENMEMORY_CLIENT_NAME
  }
}

# 3. Memory Retrieval - Must use app-specific endpoint
GET /api/v1/apps/{app_id}/memories?user_id=openmemory&page=1&size=10

# 4. Memory Search - Must use app-specific endpoint with search_query
GET /api/v1/apps/{app_id}/memories?user_id=openmemory&search_query=keyword&page=1&size=10
```

### Friend-Lite Integration Flow
1. **App Discovery**: Query `/api/v1/apps/` to find app matching `OPENMEMORY_CLIENT_NAME`
2. **Fallback**: If client app not found, use first available app
3. **Operations**: All memory operations use the app-specific endpoints with discovered `app_id`

### Testing OpenMemory MCP Integration
```bash
# Configure .env file with OpenMemory MCP settings
cp .env.template .env
# Edit .env to set MEMORY_PROVIDER=openmemory_mcp and configure OPENMEMORY_* variables

# Start OpenMemory MCP server
cd extras/openmemory-mcp && docker compose up -d

# Run integration tests (reads configuration from .env file)
cd backends/advanced && ./run-test.sh

# Manual testing - Check app structure
curl -s "http://localhost:8765/api/v1/apps/" | jq

# Test memory creation
curl -X POST "http://localhost:8765/api/v1/memories/" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "openmemory", "text": "test memory", "app": "friend_lite"}'

# Retrieve memories (replace app_id with actual ID from apps endpoint)
curl -s "http://localhost:8765/api/v1/apps/{app_id}/memories?user_id=openmemory" | jq
```

## When to Use Each Provider

### Use Friend-Lite when:
- You want full control over memory processing
- You need custom memory extraction logic
- You prefer fewer external dependencies
- You want to customize memory prompts and algorithms
- You need individual fact-based memory storage

### Use OpenMemory MCP when:
- You want professional, battle-tested memory processing
- You need cross-client compatibility (Claude Desktop, Cursor, etc.)
- You prefer to leverage external expertise rather than maintain custom logic
- You want access to OpenMemory's web interface
- You're already using OpenMemory in other tools