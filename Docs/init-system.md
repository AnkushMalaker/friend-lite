# Friend-Lite Initialization System

## Quick Links

- **👉 [Start Here: Quick Start Guide](../quickstart.md)** - Main setup path for new users
- **📚 [Full Documentation](../CLAUDE.md)** - Comprehensive reference  
- **🏗️ [Architecture Details](features.md)** - Technical deep dive

---

## Overview

Friend-Lite uses a unified initialization system with clean separation of concerns:

- **Configuration** (`init.py`) - Set up service configurations, API keys, and .env files
- **Service Management** (`services.py`) - Start, stop, and manage running services

The root orchestrator handles service selection and delegates configuration to individual service scripts. In general, setup scripts only configure and do not start services automatically. Exceptions: `extras/asr-services` and `extras/openmemory-mcp` are startup scripts. This prevents unnecessary resource usage and gives you control over when services actually run.

> **New to Friend-Lite?** Most users should start with the [Quick Start Guide](../quickstart.md) instead of this detailed reference.

## Architecture

### Root Orchestrator
- **Location**: `/init.py`
- **Purpose**: Service selection and delegation only
- **Does NOT**: Handle service-specific configuration or duplicate setup logic

### Service Scripts
- **Backend**: `backends/advanced/init.py` - Complete Python-based interactive setup
- **Speaker Recognition**: `extras/speaker-recognition/setup.sh` - Simple bash setup
- **ASR Services**: `extras/asr-services/setup.sh` - Service startup script  
- **OpenMemory MCP**: `extras/openmemory-mcp/setup.sh` - External server startup

## Usage

### Orchestrated Setup (Recommended)
Set up multiple services together with automatic URL coordination:

```bash
# From project root
uv run --with-requirements setup-requirements.txt python init.py
```

The orchestrator will:
1. Show service status and availability
2. Let you select which services to configure
3. Automatically pass service URLs between services
4. Display next steps for starting services

### Individual Service Setup
Each service can be configured independently:

```bash
# Advanced Backend only
cd backends/advanced
uv run --with-requirements setup-requirements.txt python init.py

# Speaker Recognition only  
cd extras/speaker-recognition
./setup.sh

# ASR Services only
cd extras/asr-services  
./setup.sh

# OpenMemory MCP only
cd extras/openmemory-mcp
./setup.sh
```

## Service Details

### Advanced Backend
- **Interactive setup** for authentication, LLM, transcription, and memory providers
- **Accepts arguments**: `--speaker-service-url`, `--parakeet-asr-url`
- **Generates**: Complete `.env` file with all required configuration
- **Default ports**: Backend (8000), WebUI (5173)

### Speaker Recognition  
- **Prompts for**: Hugging Face token, compute mode (cpu/gpu)
- **Service port**: 8085
- **WebUI port**: 5173
- **Requires**: HF_TOKEN for pyannote models

### ASR Services
- **Starts**: Parakeet ASR service via Docker Compose  
- **Service port**: 8767
- **Purpose**: Offline speech-to-text processing
- **No configuration required**

### OpenMemory MCP
- **Starts**: External OpenMemory MCP server
- **Service port**: 8765  
- **WebUI**: Available at http://localhost:8765
- **Purpose**: Cross-client memory compatibility

## Automatic URL Coordination

When using the orchestrated setup, service URLs are automatically configured:

| Service Selected     | Backend Gets Configured With                                     |
|----------------------|-------------------------------------------------------------------|
| Speaker Recognition  | `SPEAKER_SERVICE_URL=http://host.docker.internal:8085`           |
| ASR Services         | `PARAKEET_ASR_URL=http://host.docker.internal:8767`              |

This eliminates the need to manually configure service URLs when running services on the same machine.
Note (Linux): If `host.docker.internal` is unavailable, add `extra_hosts: - "host.docker.internal:host-gateway"` to the relevant services in `docker-compose.yml`.

## Key Benefits

✅ **No Unnecessary Building** - Services are only started when you explicitly request them  
✅ **Resource Efficient** - Parakeet ASR won't start if you're using cloud transcription  
✅ **Clean Separation** - Configuration vs service management are separate concerns  
✅ **Unified Control** - Single command to start/stop all services  
✅ **Selective Starting** - Choose which services to run based on your current needs

## Service URLs

### Default Service Endpoints
- **Backend API**: http://localhost:8000
- **Backend WebUI**: http://localhost:5173  
- **Speaker Recognition**: http://localhost:8085
- **Speaker Recognition WebUI**: http://localhost:5173
- **Parakeet ASR**: http://localhost:8767
- **OpenMemory MCP**: http://localhost:8765

### Container-to-Container Communication
Services use `host.docker.internal` for inter-container communication:
- `http://host.docker.internal:8085` - Speaker Recognition
- `http://host.docker.internal:8767` - Parakeet ASR  
- `http://host.docker.internal:8765` - OpenMemory MCP

## Service Management

Friend-Lite now separates **configuration** from **service lifecycle management**:

### Unified Service Management
Use the `services.py` script for all service operations:

```bash
# Start all configured services
uv run --with-requirements setup-requirements.txt python services.py start --all --build

# Start specific services
uv run --with-requirements setup-requirements.txt python services.py start backend speaker-recognition

# Check service status
uv run --with-requirements setup-requirements.txt python services.py status

# Stop all services
uv run --with-requirements setup-requirements.txt python services.py stop --all

# Stop specific services  
uv run --with-requirements setup-requirements.txt python services.py stop asr-services openmemory-mcp
```

### Manual Service Management
You can also manage services individually:

```bash
# Advanced Backend
cd backends/advanced && docker compose up --build -d

# Speaker Recognition  
cd extras/speaker-recognition && docker compose up --build -d

# ASR Services (only if using offline transcription)
cd extras/asr-services && docker compose up --build -d

# OpenMemory MCP (only if using openmemory_mcp provider)
cd extras/openmemory-mcp && docker compose up --build -d
```

## Configuration Files

### Generated Files
- `backends/advanced/.env` - Backend configuration with all services
- `extras/speaker-recognition/.env` - Speaker service configuration
- All services backup existing `.env` files automatically

### Required Dependencies
- **Root**: `setup-requirements.txt` (rich>=13.0.0)
- **Backend**: `setup-requirements.txt` (rich>=13.0.0, pyyaml>=6.0.0)
- **Extras**: No additional setup dependencies required

## Troubleshooting

### Common Issues
- **Port conflicts**: Check if services are already running on default ports
- **Permission errors**: Ensure scripts are executable (`chmod +x setup.sh`)
- **Missing dependencies**: Install uv and ensure setup-requirements.txt dependencies available
- **Service startup failures**: Check Docker is running and has sufficient resources

### Service Health Checks
```bash
# Backend health
curl http://localhost:8000/health

# Speaker Recognition health  
curl http://localhost:8085/health

# ASR service health
curl http://localhost:8767/health
```

### Logs and Debugging
```bash
# View service logs
docker compose logs [service-name]

# Backend logs
cd backends/advanced && docker compose logs friend-backend

# Speaker Recognition logs
cd extras/speaker-recognition && docker compose logs speaker-service
```