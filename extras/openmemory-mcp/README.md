# OpenMemory MCP Service

This directory contains a local deployment of the OpenMemory MCP (Model Context Protocol) server, which can be used as an alternative memory provider for Friend-Lite.

## What is OpenMemory MCP?

OpenMemory MCP is a memory service from mem0.ai that provides:
- Automatic memory extraction from conversations
- Vector-based memory storage with Qdrant
- Semantic search across memories
- MCP protocol support for AI integrations
- Built-in deduplication and memory management

## Quick Start

### 1. Configure Environment

```bash
cp .env.template .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Start Services

```bash
# Start backend only (recommended)
./run.sh

# Or start with UI (optional)
./run.sh --with-ui
```

### 3. Configure Friend-Lite

In your Friend-Lite backend `.env` file:

```bash
# Use OpenMemory MCP instead of built-in memory processing
MEMORY_PROVIDER=openmemory_mcp
OPENMEMORY_MCP_URL=http://localhost:8765
```

## Architecture

The deployment includes:

1. **OpenMemory MCP Server** (port 8765)
   - FastAPI backend with MCP protocol support
   - Memory extraction using OpenAI
   - REST API and MCP endpoints

2. **Qdrant Vector Database** (port 6334)
   - Stores memory embeddings
   - Enables semantic search
   - Isolated from main Friend-Lite Qdrant

3. **OpenMemory UI** (port 3001, optional)
   - Web interface for memory management
   - View and search memories
   - Debug and testing interface

## Service Endpoints

- **MCP Server**: http://localhost:8765
  - REST API: `/api/v1/memories`
  - MCP SSE: `/mcp/{client_name}/sse/{user_id}`
  
- **Qdrant Dashboard**: http://localhost:6334/dashboard

- **UI** (if enabled): http://localhost:3001

## How It Works with Friend-Lite

When configured with `MEMORY_PROVIDER=openmemory_mcp`, Friend-Lite will:

1. Send raw conversation transcripts to OpenMemory MCP
2. OpenMemory extracts memories using OpenAI
3. Memories are stored in the dedicated Qdrant instance
4. Friend-Lite can search memories via the MCP protocol

This replaces Friend-Lite's built-in memory processing with OpenMemory's implementation.

## Managing Services

```bash
# View logs
docker compose logs -f

# Stop services
docker compose down

# Stop and remove data
docker compose down -v

# Restart services
docker compose restart
```

## Testing

### Standalone Test (No Friend-Lite Dependencies)

Test the OpenMemory MCP server directly:

```bash
# From extras/openmemory-mcp directory
./test_standalone.py

# Or with custom server URL
OPENMEMORY_MCP_URL=http://localhost:8765 python test_standalone.py
```

This test verifies:
- Server connectivity
- Memory creation via REST API
- Memory listing and search
- Memory deletion
- MCP protocol endpoints

### Integration Test (With Friend-Lite)

Test the integration between Friend-Lite and OpenMemory MCP:

```bash
# From backends/advanced directory
cd backends/advanced
uv run python tests/test_openmemory_integration.py

# Or with custom server URL
OPENMEMORY_MCP_URL=http://localhost:8765 uv run python tests/test_openmemory_integration.py
```

This test verifies:
- MCP client functionality
- OpenMemoryMCPService implementation
- Service factory integration
- Memory operations through Friend-Lite interface

## Troubleshooting

### Port Conflicts

If ports are already in use, edit `docker-compose.yml`:
- Change `8765:8765` to another port for MCP server
- Change `6334:6333` to another port for Qdrant
- Update Friend-Lite's `OPENMEMORY_MCP_URL` accordingly

### Memory Not Working

1. Check OpenMemory logs: `docker compose logs openmemory-mcp`
2. Verify OPENAI_API_KEY is set correctly
3. Ensure Friend-Lite backend is configured with correct URL
4. Test MCP endpoint: `curl http://localhost:8765/api/v1/memories?user_id=test`

### Connection Issues

- Ensure containers are on same network if running Friend-Lite in Docker
- Use `host.docker.internal` instead of `localhost` when connecting from Docker containers

## Advanced Configuration

### Using with Docker Network

If Friend-Lite backend is also running in Docker:

```yaml
# In Friend-Lite docker-compose.yml
networks:
  default:
    external:
      name: openmemory-mcp_openmemory-network
```

Then use container names in Friend-Lite .env:
```bash
OPENMEMORY_MCP_URL=http://openmemory-mcp:8765
```

### Custom Models

OpenMemory uses OpenAI by default. To use different models, you would need to modify the OpenMemory source code and build a custom image.

## Resources

- [OpenMemory Documentation](https://docs.mem0.ai/open-memory/introduction)
- [MCP Protocol Spec](https://github.com/mem0ai/mem0/tree/main/openmemory)
- [Friend-Lite Memory Docs](../../backends/advanced/MEMORY_PROVIDERS.md)