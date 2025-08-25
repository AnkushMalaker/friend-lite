# Friend-Lite Advanced Backend

[QuickStart](https://github.com/AnkushMalaker/friend-lite/blob/main/backends/advanced-backend/Docs/quickstart.md)

## Web Interface

The backend includes a modern React-based web dashboard located in `./webui/` with features including live audio recording, chat interface, conversation management, and system monitoring.

### Quick Start (HTTP)
```bash
# Start with hot reload development server
docker compose up --build -d
```

- **Web Dashboard**: http://localhost:5173

### HTTPS Setup (Required for Microphone Access)

For network access and microphone features, set up HTTPS:

```bash
# Initialize HTTPS with your Tailscale/network IP
./init.sh 100.83.66.30  # Replace with your IP

# Start with HTTPS proxy
docker compose --profile https up --build -d
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

## Legacy Streamlit UI

The original Streamlit interface has been moved to `src/_webui_original/` for reference.

