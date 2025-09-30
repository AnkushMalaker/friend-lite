# Distributed Self-Hosting Architecture

Friend-Lite supports distributed deployment across multiple machines, allowing you to separate GPU-intensive services from lightweight backend components. This is ideal for scenarios where you have a dedicated GPU machine and want to run the main backend on a VPS or Raspberry Pi.

## Architecture Patterns

### Single Machine (Default)
All services run on one machine using Docker Compose - ideal for development and simple deployments.

### Distributed GPU Setup
**GPU Machine (High-performance):**
- LLM services (Ollama with GPU acceleration)
- ASR services (Parakeet with GPU)
- Speaker recognition service
- Deepgram fallback can remain on backend machine

**Backend Machine (Lightweight - VPS/RPi):**
- Friend-Lite backend (FastAPI)
- React WebUI
- MongoDB
- Qdrant vector database

## Networking with Tailscale

Tailscale VPN provides secure, encrypted networking between distributed services:

**Benefits:**
- **Zero configuration networking**: Services discover each other automatically
- **Encrypted communication**: All inter-service traffic is encrypted
- **Firewall friendly**: Works behind NATs and firewalls
- **Access control**: Granular permissions for service access
- **CORS support**: Built-in support for Tailscale IP ranges (100.x.x.x)

**Installation:**
```bash
# On each machine
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

## Distributed Service Configuration

### GPU Machine Services
```bash
# .env on GPU machine
OLLAMA_BASE_URL=http://0.0.0.0:11434  # Expose to Tailscale network
SPEAKER_SERVICE_URL=http://0.0.0.0:8085

# Enable GPU acceleration for Ollama
docker run -d --gpus=all -p 11434:11434 ollama/ollama:latest
```

### Backend Machine Configuration
```bash
# .env on backend machine
OLLAMA_BASE_URL=http://100.x.x.x:11434  # GPU machine Tailscale IP
SPEAKER_SERVICE_URL=http://100.x.x.x:8085  # GPU machine Tailscale IP

# Parakeet ASR services can also be distributed (if using offline ASR)
# PARAKEET_ASR_URL=http://100.x.x.x:8767

# CORS automatically supports Tailscale IPs (no configuration needed)
```

### Service URL Examples

**Common remote service configurations:**
```bash
# LLM Processing (GPU machine)
OLLAMA_BASE_URL=http://100.64.1.100:11434
OPENAI_BASE_URL=http://100.64.1.100:8080  # For vLLM/OpenAI-compatible APIs

# Speech Recognition (GPU machine)
# PARAKEET_ASR_URL=http://100.64.1.100:8767  # If using Parakeet ASR
SPEAKER_SERVICE_URL=http://100.64.1.100:8085

# Database services (can be on separate machine)
MONGODB_URI=mongodb://100.64.1.200:27017  # Database name: friend-lite
QDRANT_BASE_URL=http://100.64.1.200:6333
```

## Deployment Steps

### 1. Set up Tailscale on all machines
```bash
# Install and connect each machine to your Tailscale network
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

### 2. Deploy GPU services
```bash
# On GPU machine - start GPU-accelerated services
cd extras/asr-services && docker compose up parakeet -d
cd extras/speaker-recognition && docker compose up --build -d

# Start Ollama with GPU support
docker run -d --gpus=all -p 11434:11434 \
  -v ollama:/root/.ollama \
  ollama/ollama:latest
```

### 3. Configure backend machine
```bash
# Update .env with Tailscale IPs of GPU machine
OLLAMA_BASE_URL=http://[gpu-machine-tailscale-ip]:11434
SPEAKER_SERVICE_URL=http://[gpu-machine-tailscale-ip]:8085

# Start lightweight backend services
docker compose up --build -d
```

### 4. Verify connectivity
```bash
# Test service connectivity from backend machine
curl http://[gpu-machine-ip]:11434/api/tags  # Ollama
curl http://[gpu-machine-ip]:8085/health     # Speaker recognition
```

## Performance Considerations

**Network Latency:**
- Tailscale adds minimal latency (typically <5ms between nodes)
- LLM inference: Network time negligible compared to GPU processing
- ASR streaming: Use local fallback for latency-sensitive applications

**Bandwidth Usage:**
- Audio streaming: ~128kbps for Opus, ~512kbps for PCM
- LLM requests: Typically <1MB per conversation
- Memory embeddings: ~3KB per memory vector

**Processing Time Expectations:**
- Transcription (Deepgram): 2-5 seconds for 4-minute audio
- Transcription (Parakeet): 5-10 seconds for 4-minute audio
- Memory extraction (OpenAI GPT-4o-mini): 30-40 seconds for typical conversation
- Memory extraction (Ollama local): 45-90 seconds depending on model and GPU
- Full pipeline (4-min audio): 40-60 seconds with cloud services, 60-120 seconds with local models

## Security Best Practices

**Tailscale Access Control:**
```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["tag:backend"],
      "dst": ["tag:gpu:11434", "tag:gpu:8085", "tag:gpu:8767"]
    }
  ],
  "tagOwners": {
    "tag:backend": ["your-email@example.com"],
    "tag:gpu": ["your-email@example.com"]
  }
}
```

**Service Isolation:**
- Run GPU services in containers with limited network access
- Use Tailscale subnet routing for additional security
- Monitor service access logs for unauthorized requests

## Troubleshooting Distributed Setup

**Debugging Commands:**
```bash
# Check Tailscale connectivity
tailscale ping [machine-name]
tailscale status

# Test service endpoints
curl http://[tailscale-ip]:11434/api/tags
curl http://[tailscale-ip]:8085/health

# Check Docker networks
docker network ls
docker ps --format "table {{.Names}}\t{{.Ports}}"
```