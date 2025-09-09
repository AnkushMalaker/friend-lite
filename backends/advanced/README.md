# Friend-Lite Advanced Backend

A FastAPI backend with pluggable memory providers, real-time audio processing, and comprehensive conversation management.

[QuickStart](https://github.com/AnkushMalaker/friend-lite/blob/main/backends/advanced-backend/Docs/quickstart.md) | [Memory Providers](./MEMORY_PROVIDERS.md) | [Configuration Guide](./Docs/memory-configuration-guide.md)

## Key Features

### Memory System
- **Pluggable Memory Providers**: Choose between Friend-Lite native or OpenMemory MCP
- **Enhanced Memory Extraction**: Individual facts instead of generic transcripts
- **Smart Memory Updates**: LLM-driven ADD/UPDATE/DELETE actions
- **Cross-client Compatibility**: Use OpenMemory with Claude Desktop, Cursor, etc.

### Web Interface
Modern React-based web dashboard located in `./webui/` with:
- Live audio recording and real-time streaming
- Chat interface with conversation management
- **Advanced memory search** with semantic search and relevance threshold filtering
- **Memory count display** showing total memories with live filtering
- **Dual-layer filtering** combining semantic and text search
- System monitoring and debugging tools

### Quick Start

#### 1. Interactive Setup (Recommended)
```bash
# Run interactive setup wizard
./init.sh
```

**The setup wizard guides you through:**
- **Authentication**: Admin email/password setup with secure keys
- **Transcription Provider**: Choose between Deepgram, Mistral, or Offline (Parakeet)
- **LLM Provider**: Choose between OpenAI (recommended) or Ollama for memory extraction
- **Memory Provider**: Choose between Friend-Lite Native or OpenMemory MCP
- **Optional Services**: Speaker Recognition, network configuration
- **API Keys**: Prompts for all required keys with helpful links

**HTTPS Setup (Optional):**
```bash
# For microphone access and secure connections
./setup-https.sh your-tailscale-ip
```

#### 2. Start Services 

**HTTP Mode (Default - No SSL required):**
```bash
# Direct service access without nginx proxy
docker compose up --build -d
```
- **Web Dashboard**: http://localhost:5173
- **Backend API**: http://localhost:8000

**HTTPS Mode (For network access and microphone features):**
```bash
# Start with nginx SSL proxy - requires SSL setup first (see below)
docker compose up --build -d
```
- **Web Dashboard**: https://localhost/ or https://your-ip/
- **Backend API**: https://localhost/api/ or https://your-ip/api/

#### 3. HTTPS Setup (Optional - For Network Access & Microphone Features)

For network access and microphone features, HTTPS can be configured during initialization or separately:

```bash
# If not done during init.sh, run HTTPS setup
./init-https.sh 100.83.66.30  # Replace with your IP

# Start with HTTPS proxy
docker compose up --build -d
```

#### Access URLs

**Friend-Lite Advanced Backend (Primary - ports 80/443):**
- **HTTPS Dashboard**: https://localhost/ or https://your-ip/
- **HTTP**: http://localhost/ (redirects to HTTPS)
- **Live Recording**: Available at `/live-record` page

**Speaker Recognition Service (Secondary - ports 8081/8444):**
- **HTTPS Dashboard**: https://localhost:8444/ or https://your-ip:8444/
- **HTTP**: http://localhost:8081/ (redirects to HTTPS)
- **Features**: Speaker enrollment, audio analysis, live inference

**Features available with HTTPS:**
- 🎤 **Live Recording** - Real-time audio streaming with WebSocket
- 🔒 **Secure WebSocket** connections (WSS)
- 🌐 **Network Access** from other devices via Tailscale/LAN
- 🔄 **Automatic protocol detection** - Frontend auto-configures for HTTP/HTTPS

See [Docs/HTTPS_SETUP.md](Docs/HTTPS_SETUP.md) for detailed configuration.

## Testing

### Integration Tests

To run integration tests with different transcription providers:

```bash
# Test with Parakeet ASR (offline transcription)
# Automatically starts test ASR service - no manual setup required
source .env && export DEEPGRAM_API_KEY && export OPENAI_API_KEY && TRANSCRIPTION_PROVIDER=parakeet uv run pytest tests/test_integration.py::test_full_pipeline_integration -v -s --tb=short

# Test with Deepgram (default)
source .env && export DEEPGRAM_API_KEY && export OPENAI_API_KEY && uv run pytest tests/test_integration.py::test_full_pipeline_integration -v -s --tb=short
```

**Prerequisites:**
- API keys configured in `.env` file
- For debugging: Set `CACHED_MODE = True` in test file to keep containers running
