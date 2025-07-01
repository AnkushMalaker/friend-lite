# Own your Friend DevKit, DevKit 2, OMI

# Intro
The idea of this repo is to provide just enough to be useful for developers.
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

# Arch
![Architecture Diagram](.assets/plan.png)

## Arch description
The current DevKit2 streams audio via Bluetooth to some device in the OPUS codec.
Once you have audio, you need trascription (you need speech to text AKA STT or automatic speech recognition AKA ASR. Deepgram is an API based service where you stream your audio to them and they give you transcripts. You can host this locally) from it and then any other things you want, such as -
Conversation summarization (typically done via LLMs, so ollama or call OpenAI)
You also need to store these things somewhere - and you need to store different things -
1. Transcript
2. Conversation summary
3. Maybe the audio itself?
4. Memories

Memories are stored in qdrant
Conversation, and like, general logging is done on mongodb.

Its a little complicated to turn that into PCM, which most Apps use.

# Repository Structure

## Core Components

### 📱 Mobile App (`friend-lite/`)
- **React Native app** for connecting to OMI devices via Bluetooth
- Streams audio in OPUS format to selected backend
- Cross-platform (iOS/Android) support
- Uses React Native Bluetooth SDK

### 🖥️ Backends (`backends/`)
Choose one based on your needs:

#### **Simple Backend** (`backends/simple-backend/`)
**Best for:** Getting started, basic audio processing, learning

**Features:**
- ✅ Basic audio ingestion (OPUS → PCM → WAV chunks)
- ✅ File-based storage (30-second segments)
- ✅ Minimal dependencies
- ✅ Quick setup

**Pros:**
- Easiest to understand and modify
- Minimal resource requirements
- No external services needed
- Good for prototyping

**Cons:**
- No transcription built-in
- No memory/conversation management
- No speaker recognition
- Manual file management required

---

#### **Advanced Backend** (`backends/advanced-backend/`) ⭐ **RECOMMENDED**
**Best for:** Production use, full feature set, comprehensive AI features

**Features:**
- ✅ Full audio processing pipeline
- ✅ **Memory system** (mem0 + Qdrant vector storage)
- ✅ **Speaker recognition & enrollment**
- ✅ **Action items extraction** from conversations
- ✅ **Audio cropping** (removes silence, keeps speech)
- ✅ **Conversation management** with timeouts
- ✅ **Web UI** for management and monitoring
- ✅ **Multiple ASR options** (Deepgram API + offline ASR)
- ✅ **MongoDB** for structured data storage
- ✅ **RESTful API** for all operations
- ✅ **Real-time processing** with WebSocket support

**Pros:**
- Complete AI-powered solution
- Scalable architecture
- Rich feature set
- Web interface included
- Speaker identification
- Memory and action item extraction
- Audio optimization

**Cons:**
- More complex setup
- Requires multiple services (MongoDB, Qdrant, Ollama)
- Higher resource requirements
- Steeper learning curve

---

#### **OMI-Webhook-Compatible Backend** (`backends/omi-webhook-compatible/`)
**Best for:** Existing OMI users, migration from official OMI backend

**Features:**
- ✅ Compatible with official OMI app webhook system
- ✅ Drop-in replacement for OMI backend
- ✅ Audio file storage
- ✅ ngrok integration for public endpoints

**Pros:**
- Easy migration from official OMI
- Works with existing OMI mobile app
- Simple webhook-based architecture

**Cons:**
- Limited features compared to advanced backend
- Depends on ngrok for public access
- No built-in AI features

---

#### **Example Satellite Backend** (`backends/example-satellite/`)
**Best for:** Distributed setups, Wyoming protocol integration

**Features:**
- ✅ Wyoming protocol satellite
- ✅ Streams audio to remote Wyoming servers
- ✅ Bluetooth OMI device discovery
- ✅ Integration with Home Assistant/Wyoming ecosystem

**Pros:**
- Integrates with existing Wyoming setups
- Good for distributed architectures
- Home Assistant compatible

**Cons:**
- Requires separate Wyoming ASR server
- Limited standalone functionality

### 🔧 Additional Services (`extras/`)

#### **ASR Services** (`extras/asr-services/`)
- **Wyoming-compatible** ASR services
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

# Wyoming Protocol Compatibility

Both backends and ASR services use the **Wyoming protocol** for standardized communication:
- Consistent audio streaming format
- Interoperable with Home Assistant
- Modular ASR service architecture
- Easy to swap ASR providers

# Quick Start Recommendations

## For Beginners
1. Start with **Simple Backend** to understand the basics
2. Use **friend-lite mobile app** to connect your OMI device
3. Examine saved audio chunks in `./audio_chunks/`

## For Production Use
1. Use **Advanced Backend** for full features
2. Set up the complete stack: MongoDB + Qdrant + Ollama
3. Access the Web UI for conversation management
4. Configure speaker enrollment for multi-user scenarios

## For OMI Users
1. Use **OMI-Webhook-Compatible Backend** for easy migration
2. Configure ngrok for public webhook access
3. Point your OMI app to the webhook URL

## For Home Assistant Users
1. Use **Example Satellite Backend** with Wyoming integration
2. Set up ASR services from `extras/asr-services/`
3. Configure Home Assistant Wyoming integration

# Getting Started

1. **Clone the repository**
2. **Choose your backend** based on the recommendations above
3. **Follow the README** in your chosen backend directory
4. **Configure the mobile app** to connect to your backend
5. **Start streaming audio** from your OMI device

Each backend directory contains detailed setup instructions and docker-compose files for easy deployment.