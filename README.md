# Own your Friend DevKit, DevKit 2, OMI

## Quick Start

**Interactive setup (recommended):**
1. From the project root, run `uv run --with-requirements setup-requirements.txt python init.py` to configure all services with guided prompts
2. Run `python services.py start --all --build` to start all configured services
3. Visit `http://localhost:5173` for the React web dashboard

**Manual setup (alternative):**
1. Go to `backends/advanced/` and copy `.env.template` to `.env`
2. Configure your API keys and service settings manually
3. Start with `docker compose up --build -d`

**Mobile App + Phone Audio (Latest Feature):**
1. **Setup Backend**: Follow Advanced Backend setup above
2. **Install Mobile App**: Go to `app/` directory and run `npm install && npm start`
3. **Configure Mobile App**: Point to your backend IP in app settings
4. **Enable Phone Audio**: Tap "Stream Phone Audio" in app for direct microphone streaming
5. **Grant Permissions**: Allow microphone access when prompted
6. **Start Streaming**: Speak into phone for real-time processing with live audio visualization

**Documentation:** See `CLAUDE.md` and `Docs/init-system.md` for detailed setup guide

## Overview
Friend-Lite provides essential components for developers working with OMI-compatible audio devices:
It should provide the minimal requirements to either:
1. Provide firmware, sdks, examples to make your own software
2. Advanced solution or tutorial for one such that you can get full use of your devices.

Instead of going the way of making OMI's ecosystem compatible and more open -
This will attempt to make it easy to roll your own - and then try to make it compatible with OMI's ecosystem.

The app uses react native sdk (A fork of it, since the original wasn't updated fast enough)
The backend uses the python sdk (A fork of it, since the original isn't pushed to pypi)

# Vision
This fits as a small part of the larger idea of
"Have various sensors feeding the state of YOUR world to computers/AI and get some use out of it"

Usecases are numerous - OMI Mentor is one of them
Friend/Omi/pendants are a small but important part of this, since they record personal spoken context the best.
OMI-like devices with a camera can also capture visual context - or smart glasses - which also double as a display.

Regardless - this repo will try to do the minimal of this - multiple OMI-like audio devices feeding audio data - and from it,
- Memories
- Action items
- Home automation

# Use Cases
Friend-Lite supports AI-powered personal systems through multiple OMI-compatible audio devices:

**Core Features:**
- **Advanced memory system** with pluggable providers (Friend-Lite native or OpenMemory MCP)
- **Memory extraction** from conversations with individual fact storage
- **Semantic memory search** with relevance threshold filtering and live results
- **Memory count display** with total count tracking from native providers
- **Speaker-based memory filtering** to control processing based on participant presence
- **Action item detection** and tracking  
- **Home automation** integration
- **Multi-device support** for comprehensive audio capture
- **Cross-client compatibility** (optional with OpenMemory MCP)

**Device Support:**
- OMI pendants and wearables
- Smart glasses with audio capture
- Any Bluetooth-enabled audio device

# Architecture
![Architecture Diagram](.assets/plan.png)

## Architecture Overview
DevKit2 streams audio via Bluetooth using OPUS codec. The processing pipeline includes:

**Audio Processing:**
- Bluetooth audio capture from OMI devices
- OPUS codec streaming to backend services
- WebSocket-based real-time audio transport

**Transcription Services:**
- Cloud-based: Deepgram API for high-quality transcription
- Self-hosted: Local ASR services (Parakeet, Moonshine)

**AI Processing:**
- LLM-based conversation analysis (OpenAI or local Ollama)
- **Dual memory system**: Friend-Lite native or OpenMemory MCP integration
- Enhanced memory extraction with individual fact storage
- **Semantic search** with relevance scoring and threshold filtering
- Smart deduplication and memory updates (ADD/UPDATE/DELETE)
- Action item detection

**Data Storage:**
- MongoDB: User data, conversations, and transcripts
- Qdrant: Vector storage for semantic memory search
- Audio files: Optional conversation recording

# Repository Structure

## Core Components

### 📱 Mobile App (`app/`)
- **React Native app** for connecting to OMI devices via Bluetooth
- Streams audio in OPUS format to selected backend
- Cross-platform (iOS/Android) support
- Uses React Native Bluetooth SDK

### 🖥️ Backends (`backends/`)
Choose one based on your needs:

#### **Simple Backend** (`backends/simple-backend/`)
**Use case:** Getting started, basic audio processing, learning

**Features:**
- ✅ Basic audio ingestion (OPUS → PCM → WAV chunks)
- ✅ File-based storage (30-second segments)
- ✅ Minimal dependencies
- ✅ Quick setup

**Requirements:**
- Minimal resource usage
- No external services

**Limitations:**
- No transcription
- No memory/conversation management
- No speaker recognition
- Manual file management

---

#### **Advanced Backend** (`backends/advanced/`) **RECOMMENDED**
**Use case:** Production use, full feature set

**Features:**
- Audio processing pipeline with real-time WebSocket support
- **Pluggable memory system**: Choose between Friend-Lite native or OpenMemory MCP
- Enhanced memory extraction with individual fact storage (no generic fallbacks)
- **Semantic memory search** with relevance threshold filtering and total count display
- **Speaker-based memory filtering**: Optional control over processing based on participant presence
- Smart memory updates with LLM-driven action proposals (ADD/UPDATE/DELETE)
- Speaker recognition and enrollment
- Action items extraction from conversations
- Audio cropping (removes silence, keeps speech)
- Conversation management with session timeouts
- Modern React web UI with live recording and advanced search
- Multiple ASR options (Deepgram API + offline ASR)
- MongoDB for structured data storage
- RESTful API for all operations
- **Cross-client compatibility** (with OpenMemory MCP provider)

**Requirements:**
- Multiple services (MongoDB, Qdrant, Ollama)
- Higher resource usage
- Authentication configuration

---

#### **OMI-Webhook-Compatible Backend** (`backends/omi-webhook-compatible/`)
**Use case:** Existing OMI users, migration from official OMI backend

**Features:**
- ✅ Compatible with official OMI app webhook system
- ✅ Drop-in replacement for OMI backend
- ✅ Audio file storage
- ✅ ngrok integration for public endpoints

**Requirements:**
- ngrok for public access

**Limitations:**
- Limited features compared to advanced backend
- No built-in AI features

---

#### **Example Satellite Backend** (`backends/example-satellite/`)
**Use case:** Distributed setups, external ASR integration

**Features:**
- ✅ Audio streaming satellite
- ✅ Streams audio to remote ASR servers
- ✅ Bluetooth OMI device discovery
- ✅ Integration with external voice processing systems

**Requirements:**
- Separate ASR server

**Limitations:**
- Limited standalone functionality

### 🔧 Additional Services (`extras/`)

#### **ASR Services** (`extras/asr-services/`)
- **Self-hosted** ASR services
- **Moonshine** - Fast offline ASR
- **Parakeet** - Alternative offline ASR
- Self-hosted transcription options

#### **Speaker Recognition Service** (`extras/speaker-recognition/`)
- Standalone speaker identification service
- Used by advanced backend
- REST API for speaker operations

#### **HAVPE Relay** (`extras/havpe-relay/`)
- Audio relay service
- Protocol bridging capabilities

# Audio Streaming Protocol

Backends and ASR services use standardized audio streaming:
- Consistent audio streaming format
- Interoperable with external systems
- Modular ASR service architecture
- Easy to swap ASR providers

# Quick Start Recommendations

## For Beginners
1. Start with **Simple Backend** to understand the basics
2. Use **mobile app** to connect your OMI device
3. Examine saved audio chunks in `./audio_chunks/`

## For Production Use
1. Use **Advanced Backend** for full features
2. Run the orchestrated setup: `uv run --with-requirements setup-requirements.txt python init.py`
3. Start all services: `python services.py start --all --build`
4. Access the Web UI at http://localhost:5173 for conversation management

## For OMI Users
1. Use **OMI-Webhook-Compatible Backend** for easy migration
2. Configure ngrok for public webhook access
3. Point your OMI app to the webhook URL

## For Home Assistant Users
1. Use **Example Satellite Backend** for audio streaming
2. Set up ASR services from `extras/asr-services/`
3. Configure external voice processing integration

## For Distributed/Self-Hosting Users
1. Use **Advanced Backend** for full feature set
2. **Separate GPU services**: Run LLM/ASR on dedicated GPU machine
3. **Lightweight backend**: Deploy FastAPI/WebUI on VPS or Raspberry Pi
4. **Tailscale networking**: Secure VPN connection between services (automatic CORS support)
5. **Service examples**: Ollama on GPU machine, backend on lightweight server

# Getting Started

## Deployment Scenarios

### Single Machine (Recommended for beginners)
1. **Clone the repository**
2. **Run interactive setup**: `uv run --with-requirements setup-requirements.txt python init.py`
3. **Start all services**: `python services.py start --all --build`
4. **Access WebUI**: `http://localhost:5173` for the React web dashboard

### Distributed Setup (Advanced users with multiple machines)
1. **GPU Machine**: Deploy LLM services (Ollama, ASR, Speaker Recognition)
   ```bash
   # Ollama with GPU
   docker run -d --gpus=all -p 11434:11434 ollama/ollama:latest
   
   # ASR services
   cd extras/asr-services && docker compose up moonshine -d
   
   # Speaker recognition
   cd extras/speaker-recognition && docker compose up --build -d
   ```

2. **Backend Machine**: Deploy lightweight services
   ```bash
   cd backends/advanced
   
   # Configure distributed services in .env
   OLLAMA_BASE_URL=http://[gpu-machine-tailscale-ip]:11434
   SPEAKER_SERVICE_URL=http://[gpu-machine-tailscale-ip]:8001
   
   docker compose up --build -d
   ```

3. **Tailscale Networking**: Connect machines securely
   ```bash
   # On each machine
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```

### Mobile App Connection
4. **Configure the mobile app** to connect to your backend
5. **Start streaming audio** from your OMI device

Each backend directory contains detailed setup instructions and docker-compose files for easy deployment.

**Choosing a backend:** Start with **Advanced Backend** for complete functionality. See feature comparison above for specific requirements.

