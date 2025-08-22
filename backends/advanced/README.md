# Friend-Lite Advanced Backend

[QuickStart](https://github.com/AnkushMalaker/friend-lite/blob/main/backends/advanced-backend/Docs/quickstart.md)

## Web Interface

The backend includes a modern React-based web dashboard located in `./webui/`. 

### Quick Start
```bash
# Production
docker compose up webui friend-backend mongo qdrant

# Development (with hot reload)
docker compose --profile dev up
```

- **Production**: http://localhost:3000
- **Development**: http://localhost:5173

See `./webui/README.md` for detailed documentation.

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

