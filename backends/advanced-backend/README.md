# Advanced Omi Backend

## Transcription Configuration

This backend supports conditional transcription methods:

### 1. Deepgram API (Not Yet Implemented)
When `DEEPGRAM_API_KEY` is provided, the system is designed to use Deepgram's cloud API for transcription. However, this feature is not yet implemented and will fall back to offline ASR with a warning.

### 2. Offline ASR (Current Implementation)
The system uses the offline ASR service specified by `OFFLINE_ASR_TCP_URI`.

```bash
export OFFLINE_ASR_TCP_URI="tcp://192.168.0.110:8765/"
```

## Environment Variables


### ASR configuration

You can either use deepgram, or connect your own service.  One of these env values must be present
```bash
# For future Deepgram implementation (currently not implemented)
DEEPGRAM_API_KEY="your_api_key"

# Required for offline ASR (current implementation)
OFFLINE_ASR_TCP_URI="tcp://192.168.0.110:8765/"
```

The system automatically detects which transcription method to use based on the availability of `DEEPGRAM_API_KEY`, but currently always falls back to offline ASR.

### LLM config
We need an LLM for mem0 and others
`LLM_PROVIDER=openai`
This can be ollama, openrouter or openAI.  You can add more configs for other LLMs in utils.py
Check [https://docs.mem0.ai/components/llms/models/openai](mem0 docs) for more details.
`LLM_BASE_URL=https://api.openai.com/v1`
`LLM_API_KEY=sk----`
`LLM_CHOICE=gpt-4o-mini`

### Databases
We use MongoDB for all the Omi stuff as well as qdrant & Neo4J for memories.
These all have env params in case you already have such services installed and want to use them
These shouldn't need touching if you spin up using docker compose, aside from setting a password for neo4J
`MONGODB_URI=mongo`
`QDRANT_BASE_URL=qdrant`
`NEO4J_HOST=neo4j-mem0`
`NEO4J_USER=neo4j`
`NEO4J_PASSWORD=XXXX`

### Tunelling 
We need to route an external domain to our backend.  You can either use NGROK or Cloudflared to do this.
By default, there's an NGROK docker image, so if you use this you need
`NGROK_URL=yourngrokurl.app`
`NGROK_AUTHTOKEN=xxxxx`

### Other
We pull models from hugging face and so
`HF_TOKEN=xxxx`

### Speaker service
If this is on, you need to provide a URL
`SPEAKER_SERVICE_URL=http://speaker-omi`

# Setup

To setup the backend, you need to do the following:
0. Clone the repository
1. Change the directory to the backend,  
`cd backends/advanced-backend`
2. Fill out the .env variables as you require (check the .env.template for the required variables)
3. Run the backend with `docker compose up --build -d`. This will take a couple minutes, be patient.