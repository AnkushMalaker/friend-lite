#!/bin/bash
set -e

# Friend-Lite Advanced Backend Initialization Script
# Comprehensive setup for all configuration files and optional services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src" ]; then
    print_error "Please run this script from the backends/advanced directory"
    exit 1
fi

print_header "Friend-Lite Advanced Backend Initialization"
echo "This script will help you set up the Friend-Lite backend with all necessary configurations."
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

# Step 1: Handle .env file
print_header "Step 1: Environment Configuration"
if [ -f ".env" ]; then
    print_info ".env file already exists"
    if prompt_yes_no "Do you want to update it from template?" "n"; then
        cp .env .env.backup
        print_info "Backed up existing .env to .env.backup"
        cp .env.template .env
        print_success ".env created from template"
        print_warning "Please edit .env to add your API keys and configuration"
    fi
else
    if [ -f ".env.template" ]; then
        cp .env.template .env
        print_success ".env file created from template"
        print_warning "IMPORTANT: Edit .env file to add your API keys:"
        echo "  - DEEPGRAM_API_KEY (for speech-to-text)"
        echo "  - OPENAI_API_KEY (for memory extraction)"
        echo "  - ADMIN_EMAIL and ADMIN_PASSWORD"
        echo ""
        if prompt_yes_no "Would you like to edit .env now?" "y"; then
            ${EDITOR:-nano} .env
        fi
    else
        print_error ".env.template not found!"
        exit 1
    fi
fi

# Step 2: Memory configuration
print_header "Step 2: Memory Configuration"
if [ -f "memory_config.yaml" ]; then
    print_info "memory_config.yaml already exists"
    if prompt_yes_no "Do you want to reset it from template?" "n"; then
        cp memory_config.yaml memory_config.yaml.backup
        print_info "Backed up existing memory_config.yaml to memory_config.yaml.backup"
        cp memory_config.yaml.template memory_config.yaml
        print_success "memory_config.yaml reset from template"
    fi
else
    if [ -f "memory_config.yaml.template" ]; then
        cp memory_config.yaml.template memory_config.yaml
        print_success "memory_config.yaml created from template"
    else
        print_error "memory_config.yaml.template not found!"
        exit 1
    fi
fi

# Step 3: Diarization configuration
print_header "Step 3: Diarization Configuration"
if [ -f "diarization_config.json" ]; then
    print_info "diarization_config.json already exists"
    if prompt_yes_no "Do you want to reset it from template?" "n"; then
        cp diarization_config.json diarization_config.json.backup
        print_info "Backed up existing diarization_config.json to diarization_config.json.backup"
        cp diarization_config.json.template diarization_config.json
        print_success "diarization_config.json reset from template"
    fi
else
    if [ -f "diarization_config.json.template" ]; then
        cp diarization_config.json.template diarization_config.json
        print_success "diarization_config.json created from template"
    else
        print_error "diarization_config.json.template not found!"
        exit 1
    fi
fi

# Step 4: HTTPS Setup (optional)
print_header "Step 4: HTTPS Configuration (Optional)"
echo "HTTPS is required for:"
echo "  - Microphone access from browsers"
echo "  - Remote access via network/Tailscale"
echo "  - Secure WebSocket connections"
echo ""

if prompt_yes_no "Do you want to set up HTTPS?" "n"; then
    if [ -f "init-https.sh" ]; then
        echo ""
        print_info "Please enter your Tailscale IP or network IP"
        print_info "Example: 100.83.66.30"
        read -p "IP Address: " TAILSCALE_IP
        
        if [ -n "$TAILSCALE_IP" ]; then
            ./init-https.sh "$TAILSCALE_IP"
            HTTPS_ENABLED=true
        else
            print_warning "Skipping HTTPS setup - no IP provided"
            HTTPS_ENABLED=false
        fi
    else
        print_warning "init-https.sh not found, skipping HTTPS setup"
        HTTPS_ENABLED=false
    fi
else
    print_info "Skipping HTTPS setup"
    HTTPS_ENABLED=false
fi

# Step 5: Optional Services
print_header "Step 5: Optional Services (extras/)"

echo "Configure additional services from extras/:"
echo ""

# OpenMemory MCP (Memory Provider)
if prompt_yes_no "Use OpenMemory MCP for memory management? (cross-client compatible)" "n"; then
    sed -i 's/MEMORY_PROVIDER=.*/MEMORY_PROVIDER=openmemory_mcp/' .env
    print_success "Configured for OpenMemory MCP"
    OPENMEMORY_ENABLED=true
else
    OPENMEMORY_ENABLED=false
fi

# Parakeet ASR (Offline Transcription)
if prompt_yes_no "Use Parakeet for offline transcription? (requires GPU)" "n"; then
    if ! grep -q "PARAKEET_ASR_URL" .env; then
        echo "PARAKEET_ASR_URL=http://host.docker.internal:8767" >> .env
    fi
    print_success "Configured for Parakeet ASR"
    PARAKEET_ENABLED=true
else
    PARAKEET_ENABLED=false
fi

# Speaker Recognition
if prompt_yes_no "Enable Speaker Recognition service?" "n"; then
    if ! grep -q "SPEAKER_SERVICE_URL" .env; then
        echo "SPEAKER_SERVICE_URL=http://host.docker.internal:8001" >> .env
    fi
    print_success "Configured for Speaker Recognition"
    SPEAKER_ENABLED=true
else
    SPEAKER_ENABLED=false
fi

# Step 6: Summary and Next Steps
print_header "Setup Complete!"

echo "Configuration Summary:"
echo "----------------------"
echo "✅ Environment file (.env) configured"
echo "✅ Memory configuration (memory_config.yaml) ready"
echo "✅ Diarization configuration (diarization_config.json) ready"

if [ "$HTTPS_ENABLED" = true ]; then
    echo "✅ HTTPS configured with SSL certificates"
fi

echo ""
echo "Next Steps:"
echo "-----------"

if [ "$HTTPS_ENABLED" = true ]; then
    echo "1. Start the services with HTTPS:"
    echo "   ${CYAN}docker compose --profile https up --build -d${NC}"
    echo ""
    echo "2. Access the dashboard:"
    echo "   🌐 https://localhost/"
    echo "   🌐 https://$TAILSCALE_IP/"
else
    echo "1. Start the services:"
    echo "   ${CYAN}docker compose up --build -d${NC}"
    echo ""
    echo "2. Access the dashboard:"
    echo "   🌐 http://localhost:5173"
fi

echo ""
echo "3. Check service health:"
echo "   ${CYAN}curl http://localhost:8000/health${NC}"

echo ""
if [ "$OPENMEMORY_ENABLED" = true ] || [ "$PARAKEET_ENABLED" = true ] || [ "$SPEAKER_ENABLED" = true ]; then
    echo "Start Optional Services:"
    echo "------------------------"
    
    if [ "$OPENMEMORY_ENABLED" = true ]; then
        echo "OpenMemory MCP:"
        echo "  ${CYAN}cd ../../extras/openmemory-mcp && docker compose up -d${NC}"
    fi
    
    if [ "$PARAKEET_ENABLED" = true ]; then
        echo "Parakeet ASR:"
        echo "  ${CYAN}cd ../../extras/asr-services && docker compose up parakeet -d${NC}"
    fi
    
    if [ "$SPEAKER_ENABLED" = true ]; then
        echo "Speaker Recognition:"
        echo "  ${CYAN}cd ../../extras/speaker-recognition && docker compose up --build -d${NC}"
    fi
fi

echo ""
echo "For more information, see:"
echo "  - Docs/quickstart.md"
echo "  - Docs/memory-configuration-guide.md"
echo "  - MEMORY_PROVIDERS.md"

echo ""
print_success "Initialization complete! 🎉"