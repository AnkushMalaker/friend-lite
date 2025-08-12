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

## Legacy Streamlit UI

The original Streamlit interface has been moved to `src/_webui_original/` for reference.

