#!/bin/bash
set -e

# Friend-Lite Advanced Backend Interactive Setup Script
# Interactive configuration for all services and API keys

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${CYAN}===============================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}===============================================${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${MAGENTA}► $1${NC}"
    echo -e "${MAGENTA}$(echo "$1" | sed 's/./-/g')${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src" ]; then
    print_error "Please run this script from the backends/advanced directory"
    exit 1
fi

print_header "🚀 Friend-Lite Interactive Setup"
echo "This wizard will help you configure Friend-Lite with all necessary services."
echo "We'll ask for your API keys and preferences step by step."
echo ""

# Function to prompt yes/no
prompt_yes_no() {
    local prompt="$1"
    local default="$2"
    local response
    
    if [ "$default" = "y" ]; then
        prompt="$prompt [Y/n]: "
    else
        prompt="$prompt [y/N]: "
    fi
    
    read -p "$prompt" response
    response=${response:-$default}
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Function to prompt for value with default
prompt_value() {
    local prompt="$1"
    local default="$2"
    local response
    
    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " response
        response=${response:-$default}
    else
        read -p "$prompt: " response
    fi
    
    echo "$response"
}

# Function to prompt for password (hidden input)
prompt_password() {
    local prompt="$1"
    local password
    read -s -p "$prompt: " password
    echo ""
    echo "$password"
}

# Backup existing .env if it exists
if [ -f ".env" ]; then
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
    print_info "Backed up existing .env file"
fi

# Start with template
cp .env.template .env.new
print_success "Starting from .env.template"

# =============================================================================
# AUTHENTICATION SETUP
# =============================================================================
print_section "Authentication Setup"
echo "Configure admin account for the dashboard"
echo ""

ADMIN_EMAIL=$(prompt_value "Admin email" "admin@example.com")
ADMIN_PASSWORD=$(prompt_password "Admin password (min 8 chars)")
AUTH_SECRET=$(openssl rand -hex 32 2>/dev/null || cat /dev/urandom | head -c 32 | base64)

sed -i "s|ADMIN_EMAIL=.*|ADMIN_EMAIL=$ADMIN_EMAIL|" .env.new
sed -i "s|ADMIN_PASSWORD=.*|ADMIN_PASSWORD=$ADMIN_PASSWORD|" .env.new
sed -i "s|AUTH_SECRET_KEY=.*|AUTH_SECRET_KEY=$AUTH_SECRET|" .env.new

print_success "Admin account configured"

# =============================================================================
# TRANSCRIPTION PROVIDER
# =============================================================================
print_section "Speech-to-Text Configuration"
echo "Choose your transcription provider:"
echo "  1) Deepgram (recommended - high quality, requires API key)"
echo "  2) Mistral (Voxtral models - requires API key)"
echo "  3) Offline (Parakeet ASR - requires GPU, runs locally)"
echo "  4) None (skip transcription setup)"
echo ""

TRANSCRIPTION_CHOICE=$(prompt_value "Enter choice (1-4)" "1")

case $TRANSCRIPTION_CHOICE in
    1)
        TRANSCRIPTION_PROVIDER="deepgram"
        print_info "Deepgram selected"
        echo "Get your API key from: https://console.deepgram.com/"
        DEEPGRAM_API_KEY=$(prompt_value "Deepgram API key" "")
        if [ -n "$DEEPGRAM_API_KEY" ]; then
            sed -i "s|DEEPGRAM_API_KEY=.*|DEEPGRAM_API_KEY=$DEEPGRAM_API_KEY|" .env.new
            sed -i "s|TRANSCRIPTION_PROVIDER=.*|TRANSCRIPTION_PROVIDER=deepgram|" .env.new
            print_success "Deepgram configured"
        else
            print_warning "No API key provided - transcription will not work"
        fi
        ;;
    2)
        TRANSCRIPTION_PROVIDER="mistral"
        print_info "Mistral selected"
        echo "Get your API key from: https://console.mistral.ai/"
        MISTRAL_API_KEY=$(prompt_value "Mistral API key" "")
        MISTRAL_MODEL=$(prompt_value "Mistral model" "voxtral-mini-2507")
        if [ -n "$MISTRAL_API_KEY" ]; then
            sed -i "s|# MISTRAL_API_KEY=.*|MISTRAL_API_KEY=$MISTRAL_API_KEY|" .env.new
            sed -i "s|# MISTRAL_MODEL=.*|MISTRAL_MODEL=$MISTRAL_MODEL|" .env.new
            sed -i "s|TRANSCRIPTION_PROVIDER=.*|TRANSCRIPTION_PROVIDER=mistral|" .env.new
            print_success "Mistral configured"
        else
            print_warning "No API key provided - transcription will not work"
        fi
        ;;
    3)
        TRANSCRIPTION_PROVIDER="offline"
        print_info "Offline Parakeet ASR selected"
        PARAKEET_URL=$(prompt_value "Parakeet ASR URL" "http://host.docker.internal:8767")
        echo "PARAKEET_ASR_URL=$PARAKEET_URL" >> .env.new
        print_warning "Remember to start Parakeet service: cd ../../extras/asr-services && docker compose up parakeet"
        ;;
    4)
        print_info "Skipping transcription setup"
        ;;
esac

# =============================================================================
# LLM PROVIDER
# =============================================================================
print_section "LLM Provider Configuration"
echo "Choose your LLM provider for memory extraction:"
echo "  1) OpenAI (GPT-4, GPT-3.5 - requires API key)"
echo "  2) Ollama (local models - requires Ollama server)"
echo "  3) Skip (no memory extraction)"
echo ""

LLM_CHOICE=$(prompt_value "Enter choice (1-3)" "1")

case $LLM_CHOICE in
    1)
        LLM_PROVIDER="openai"
        print_info "OpenAI selected"
        echo "Get your API key from: https://platform.openai.com/api-keys"
        OPENAI_API_KEY=$(prompt_value "OpenAI API key" "")
        OPENAI_MODEL=$(prompt_value "OpenAI model" "gpt-4o-mini")
        OPENAI_BASE_URL=$(prompt_value "OpenAI base URL (for proxies/compatible APIs)" "https://api.openai.com/v1")
        
        if [ -n "$OPENAI_API_KEY" ]; then
            sed -i "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$OPENAI_API_KEY|" .env.new
            sed -i "s|OPENAI_MODEL=.*|OPENAI_MODEL=$OPENAI_MODEL|" .env.new
            sed -i "s|OPENAI_BASE_URL=.*|OPENAI_BASE_URL=$OPENAI_BASE_URL|" .env.new
            sed -i "s|LLM_PROVIDER=.*|LLM_PROVIDER=openai|" .env.new
            print_success "OpenAI configured"
        else
            print_warning "No API key provided - memory extraction will not work"
        fi
        ;;
    2)
        LLM_PROVIDER="ollama"
        print_info "Ollama selected"
        OLLAMA_BASE_URL=$(prompt_value "Ollama server URL" "http://host.docker.internal:11434")
        OLLAMA_MODEL=$(prompt_value "Ollama model" "llama3.2")
        
        sed -i "s|# OLLAMA_BASE_URL=.*|OLLAMA_BASE_URL=$OLLAMA_BASE_URL|" .env.new
        sed -i "s|# OLLAMA_MODEL=.*|OLLAMA_MODEL=$OLLAMA_MODEL|" .env.new
        sed -i "s|LLM_PROVIDER=.*|LLM_PROVIDER=ollama|" .env.new
        print_success "Ollama configured"
        print_warning "Make sure Ollama is running and the model is pulled"
        ;;
    3)
        print_info "Skipping LLM setup - memory extraction disabled"
        ;;
esac

# =============================================================================
# MEMORY PROVIDER
# =============================================================================
print_section "Memory Storage Configuration"
echo "Choose your memory storage backend:"
echo "  1) Friend-Lite Native (Qdrant + custom extraction)"
echo "  2) OpenMemory MCP (cross-client compatible, external server)"
echo ""

MEMORY_CHOICE=$(prompt_value "Enter choice (1-2)" "1")

case $MEMORY_CHOICE in
    1)
        MEMORY_PROVIDER="friend_lite"
        print_info "Friend-Lite Native memory provider selected"
        sed -i "s|MEMORY_PROVIDER=.*|MEMORY_PROVIDER=friend_lite|" .env.new
        
        # Qdrant configuration
        QDRANT_URL=$(prompt_value "Qdrant URL" "qdrant")
        sed -i "s|QDRANT_BASE_URL=.*|QDRANT_BASE_URL=$QDRANT_URL|" .env.new
        print_success "Friend-Lite memory provider configured"
        ;;
    2)
        MEMORY_PROVIDER="openmemory_mcp"
        print_info "OpenMemory MCP selected"
        OPENMEMORY_URL=$(prompt_value "OpenMemory MCP server URL" "http://host.docker.internal:8765")
        OPENMEMORY_CLIENT=$(prompt_value "OpenMemory client name" "friend_lite")
        OPENMEMORY_USER=$(prompt_value "OpenMemory user ID" "openmemory")
        
        sed -i "s|MEMORY_PROVIDER=.*|MEMORY_PROVIDER=openmemory_mcp|" .env.new
        sed -i "s|# OPENMEMORY_MCP_URL=.*|OPENMEMORY_MCP_URL=$OPENMEMORY_URL|" .env.new
        sed -i "s|# OPENMEMORY_CLIENT_NAME=.*|OPENMEMORY_CLIENT_NAME=$OPENMEMORY_CLIENT|" .env.new
        sed -i "s|# OPENMEMORY_USER_ID=.*|OPENMEMORY_USER_ID=$OPENMEMORY_USER|" .env.new
        print_success "OpenMemory MCP configured"
        print_warning "Remember to start OpenMemory: cd ../../extras/openmemory-mcp && docker compose up -d"
        ;;
esac

# =============================================================================
# OPTIONAL SERVICES
# =============================================================================
print_section "Optional Services"

if prompt_yes_no "Enable Speaker Recognition?" "n"; then
    SPEAKER_URL=$(prompt_value "Speaker Recognition service URL" "http://host.docker.internal:8001")
    echo "SPEAKER_SERVICE_URL=$SPEAKER_URL" >> .env.new
    print_success "Speaker Recognition configured"
    print_info "Start with: cd ../../extras/speaker-recognition && docker compose up -d"
fi

# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================
print_section "Network Configuration"

HOST_IP=$(prompt_value "Host IP for services" "localhost")
BACKEND_PORT=$(prompt_value "Backend port" "8000")
WEBUI_PORT=$(prompt_value "Web UI port" "5173")

sed -i "s|HOST_IP=.*|HOST_IP=$HOST_IP|" .env.new
sed -i "s|BACKEND_PUBLIC_PORT=.*|BACKEND_PUBLIC_PORT=$BACKEND_PORT|" .env.new
sed -i "s|WEBUI_PORT=.*|WEBUI_PORT=$WEBUI_PORT|" .env.new

# =============================================================================
# FINALIZE
# =============================================================================
print_header "Configuration Complete!"

# Move new .env into place
mv .env.new .env
print_success ".env file created successfully"

# Copy other configuration files
if [ ! -f "memory_config.yaml" ]; then
    cp memory_config.yaml.template memory_config.yaml
    print_success "memory_config.yaml created"
fi

if [ ! -f "diarization_config.json" ]; then
    cp diarization_config.json.template diarization_config.json
    print_success "diarization_config.json created"
fi

# =============================================================================
# SUMMARY
# =============================================================================
print_section "Configuration Summary"
echo ""
echo "✅ Admin Account: $ADMIN_EMAIL"
echo "✅ Transcription: $TRANSCRIPTION_PROVIDER"
echo "✅ LLM Provider: $LLM_PROVIDER"
echo "✅ Memory Provider: $MEMORY_PROVIDER"
echo "✅ Backend URL: http://$HOST_IP:$BACKEND_PORT"
echo "✅ Dashboard URL: http://$HOST_IP:$WEBUI_PORT"

# =============================================================================
# NEXT STEPS
# =============================================================================
print_section "Next Steps"
echo ""
echo "1. Start the main services:"
echo "   ${CYAN}docker compose up --build -d${NC}"
echo ""
echo "2. Access the dashboard:"
echo "   ${CYAN}http://$HOST_IP:$WEBUI_PORT${NC}"
echo ""
echo "3. Check service health:"
echo "   ${CYAN}curl http://$HOST_IP:$BACKEND_PORT/health${NC}"

if [ "$MEMORY_PROVIDER" = "openmemory_mcp" ]; then
    echo ""
    echo "4. Start OpenMemory MCP:"
    echo "   ${CYAN}cd ../../extras/openmemory-mcp && docker compose up -d${NC}"
fi

if [ "$TRANSCRIPTION_PROVIDER" = "offline" ]; then
    echo ""
    echo "5. Start Parakeet ASR:"
    echo "   ${CYAN}cd ../../extras/asr-services && docker compose up parakeet -d${NC}"
fi

echo ""
print_success "Setup complete! 🎉"
echo ""
echo "For detailed documentation, see:"
echo "  • Docs/quickstart.md"
echo "  • MEMORY_PROVIDERS.md"
echo "  • CLAUDE.md"