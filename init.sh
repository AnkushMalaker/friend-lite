#!/bin/bash

# Friend-Lite Interactive Setup Script
# This script helps you configure Friend-Lite by asking a few key questions
# and setting up the environment files and Docker services.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Generate secure random string
generate_secret() {
    openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | base64 | tr -d "=+/" | cut -c1-64
}

# Validate email format
validate_email() {
    if [[ $1 =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
        return 0
    else
        return 1
    fi
}

# Validate URL format
validate_url() {
    if [[ $1 =~ ^https?:// ]]; then
        return 0
    else
        return 1
    fi
}

# Welcome message
clear
print_header "🎉 Welcome to Friend-Lite Setup!"

cat << 'EOF'
This script will help you configure Friend-Lite, an AI-powered personal system
that captures audio from OMI-compatible devices and extracts:

📝 Memories and conversations
🎯 Action items and tasks  
🏠 Home automation triggers
🧠 Personal insights and patterns

Let's get you set up with a few quick questions...
EOF

echo
read -p "Press Enter to continue..."
echo

# Backend selection
print_header "🖥️  Backend Selection"
echo "Choose your backend:"
echo "1. Advanced Backend (Recommended) - Full features with memory extraction"
echo "2. Simple Backend - Basic audio processing only"
echo

while true; do
    read -p "Select backend [1-2] (default: 1): " backend_choice
    backend_choice=${backend_choice:-1}
    
    case $backend_choice in
        1)
            BACKEND_DIR="backends/advanced"
            BACKEND_NAME="Advanced Backend"
            break
            ;;
        2)
            BACKEND_DIR="backends/simple-backend"
            BACKEND_NAME="Simple Backend"
            print_warning "Simple backend has limited features (no memory extraction)"
            break
            ;;
        *)
            print_error "Please select 1 or 2"
            ;;
    esac
done

print_success "Selected: $BACKEND_NAME"
echo

# LLM Provider Configuration
print_header "🤖 AI Language Model Configuration"
echo "Choose your LLM provider:"
echo "1. OpenAI (Recommended) - GPT-4o models, reliable API"
echo "2. Ollama - Local/self-hosted models, privacy-focused"
echo

while true; do
    read -p "Select LLM provider [1-2] (default: 1): " llm_choice
    llm_choice=${llm_choice:-1}
    
    case $llm_choice in
        1)
            LLM_PROVIDER="openai"
            break
            ;;
        2)
            LLM_PROVIDER="ollama"
            break
            ;;
        *)
            print_error "Please select 1 or 2"
            ;;
    esac
done

# Configure chosen LLM provider
if [[ $LLM_PROVIDER == "openai" ]]; then
    print_info "OpenAI Configuration (sets LLM_PROVIDER=openai)"
    echo
    while true; do
        read -p "Enter your OpenAI API key (sets OPENAI_API_KEY): " openai_key
        if [[ ${#openai_key} -ge 20 && $openai_key == sk-* ]]; then
            OPENAI_API_KEY="$openai_key"
            break
        else
            print_error "Please enter a valid OpenAI API key (starts with 'sk-')"
        fi
    done
    
    read -p "OpenAI model (sets OPENAI_MODEL) [gpt-4o-mini]: " openai_model
    OPENAI_MODEL=${openai_model:-gpt-4o-mini}
    
    OPENAI_BASE_URL="https://api.openai.com/v1"
else
    print_info "Ollama Configuration (sets LLM_PROVIDER=ollama)"
    echo
    while true; do
        read -p "Enter Ollama base URL (sets OPENAI_BASE_URL): " ollama_url
        if validate_url "$ollama_url"; then
            OPENAI_BASE_URL="$ollama_url"
            break
        else
            print_error "Please enter a valid URL (e.g., http://localhost:11434/v1)"
        fi
    done
    
    read -p "Ollama model (sets OPENAI_MODEL) [llama3.1:latest]: " ollama_model
    OPENAI_MODEL=${ollama_model:-llama3.1:latest}
    
    # Ollama uses dummy API key
    OPENAI_API_KEY="dummy"
fi

print_success "LLM provider configured: $LLM_PROVIDER with model $OPENAI_MODEL"
echo

# Speech-to-Text Configuration
print_header "🎤 Speech-to-Text Configuration"
echo "Choose your transcription provider:"
echo "1. Deepgram API (Recommended) - High accuracy, cloud-based"
echo "2. Custom ASR Service - Self-hosted transcription"
echo "3. Skip - Configure later"
echo

while true; do
    read -p "Select transcription [1-3] (default: 1): " asr_choice
    asr_choice=${asr_choice:-1}
    
    case $asr_choice in
        1)
            TRANSCRIPTION_PROVIDER="deepgram"
            while true; do
                read -p "Enter Deepgram API key (sets DEEPGRAM_API_KEY): " deepgram_key
                if [[ ${#deepgram_key} -ge 20 ]]; then
                    DEEPGRAM_API_KEY="$deepgram_key"
                    break
                else
                    print_error "Please enter a valid Deepgram API key"
                fi
            done
            break
            ;;
        2)
            TRANSCRIPTION_PROVIDER="parakeet"
            read -p "Enter ASR service URL (sets PARAKEET_ASR_URL) [http://localhost:8767]: " asr_url
            PARAKEET_ASR_URL=${asr_url:-http://localhost:8767}
            break
            ;;
        3)
            print_warning "Transcription provider will need to be configured later"
            break
            ;;
        *)
            print_error "Please select 1, 2, or 3"
            ;;
    esac
done

echo

# Admin Configuration
print_header "👤 Admin Account Configuration"
while true; do
    read -p "Admin email address (sets ADMIN_EMAIL): " admin_email
    if validate_email "$admin_email"; then
        ADMIN_EMAIL="$admin_email"
        break
    else
        print_error "Please enter a valid email address"
    fi
done

print_info "Generating secure admin password and JWT secret (sets ADMIN_PASSWORD, AUTH_SECRET_KEY)..."
ADMIN_PASSWORD=$(generate_secret | cut -c1-20)
AUTH_SECRET_KEY=$(generate_secret)

print_success "Admin credentials configured"
echo "  📧 Email: $ADMIN_EMAIL"
echo "  🔑 Password: $ADMIN_PASSWORD"
print_warning "Save this password - it won't be shown again!"
echo

# Network Configuration
print_header "🌐 Network Configuration"
print_info "Configuring network access (sets HOST_IP, CORS_ORIGINS)"

# Auto-detect IP
if command -v ip >/dev/null 2>&1; then
    LOCAL_IP=$(ip route get 8.8.8.8 | awk '{print $7; exit}' 2>/dev/null || echo "localhost")
else
    LOCAL_IP="localhost"
fi

read -p "Backend host IP (sets HOST_IP) [${LOCAL_IP}]: " host_ip
HOST_IP=${host_ip:-$LOCAL_IP}

BACKEND_PORT="8000"
WEBUI_PORT="3000" 
CORS_ORIGINS="http://${HOST_IP}:${WEBUI_PORT},http://localhost:${WEBUI_PORT},http://127.0.0.1:${WEBUI_PORT}"

print_success "Network configured for access at http://${HOST_IP}:${WEBUI_PORT}"
echo

# Speaker Recognition Service Configuration
print_header "🗣️ Speaker Recognition Service (Optional)"
echo "Speaker recognition enables speaker identification for conversations."
echo "1. Set up speaker recognition service"
echo "2. Skip - Configure later"
echo

while true; do
    read -p "Configure speaker recognition? [1-2] (default: 2): " speaker_choice
    speaker_choice=${speaker_choice:-2}
    
    case $speaker_choice in
        1)
            SETUP_SPEAKER_RECOGNITION="yes"
            
            print_info "Speaker recognition requires a Hugging Face token for pyannote models"
            while true; do
                read -p "Enter your Hugging Face token (sets HF_TOKEN): " hf_token
                if [[ ${#hf_token} -ge 10 ]]; then
                    HF_TOKEN="$hf_token"
                    break
                else
                    print_error "Please enter a valid Hugging Face token"
                fi
            done
            
            echo "Choose compute mode for speaker recognition:"
            echo "1. CPU (Recommended for development)"
            echo "2. GPU (Requires NVIDIA GPU with CUDA)"
            echo
            
            while true; do
                read -p "Select compute mode [1-2] (default: 1): " compute_choice
                compute_choice=${compute_choice:-1}
                
                case $compute_choice in
                    1)
                        SPEAKER_COMPUTE_MODE="cpu"
                        break
                        ;;
                    2)
                        SPEAKER_COMPUTE_MODE="gpu"
                        print_warning "GPU mode requires NVIDIA GPU with CUDA drivers"
                        break
                        ;;
                    *)
                        print_error "Please select 1 or 2"
                        ;;
                esac
            done
            
            read -p "Speaker similarity threshold (0.0-1.0, higher=more strict) [0.15]: " similarity_threshold
            SIMILARITY_THRESHOLD=${similarity_threshold:-0.15}
            
            print_success "Speaker recognition configured with $SPEAKER_COMPUTE_MODE mode"
            break
            ;;
        2)
            print_warning "Speaker recognition setup skipped - can be configured later"
            break
            ;;
        *)
            print_error "Please select 1 or 2"
            ;;
    esac
done

echo

# HTTPS Configuration
print_header "🔒 HTTPS Configuration (Optional)"
echo "HTTPS setup provides secure access and works well with Tailscale networks."
echo "1. Set up HTTPS with Tailscale IP"
echo "2. Set up HTTPS with localhost only"
echo "3. Skip - Use HTTP only"
echo

while true; do
    read -p "Configure HTTPS? [1-3] (default: 3): " https_choice
    https_choice=${https_choice:-3}
    
    case $https_choice in
        1)
            SETUP_HTTPS="yes"
            read -p "Enter your Tailscale IP address: " tailscale_ip
            if [[ $tailscale_ip =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
                TAILSCALE_IP="$tailscale_ip"
                print_success "HTTPS will be configured for localhost and $TAILSCALE_IP"
            else
                print_error "Invalid IP format, using localhost only"
                TAILSCALE_IP="127.0.0.1"
            fi
            break
            ;;
        2)
            SETUP_HTTPS="yes"
            TAILSCALE_IP="127.0.0.1"
            print_success "HTTPS will be configured for localhost only"
            break
            ;;
        3)
            print_warning "HTTPS setup skipped - using HTTP only"
            break
            ;;
        *)
            print_error "Please select 1, 2, or 3"
            ;;
    esac
done

echo

# Configuration Summary
print_header "📋 Configuration Summary"
echo "Backend: $BACKEND_NAME"
echo "LLM Provider: $LLM_PROVIDER ($OPENAI_MODEL)"
if [[ -n $DEEPGRAM_API_KEY ]]; then
    echo "Speech-to-Text: Deepgram API"
elif [[ -n $PARAKEET_ASR_URL ]]; then
    echo "Speech-to-Text: Custom ASR ($PARAKEET_ASR_URL)"
else
    echo "Speech-to-Text: Not configured"
fi
if [[ $SETUP_SPEAKER_RECOGNITION == "yes" ]]; then
    echo "Speaker Recognition: Enabled ($SPEAKER_COMPUTE_MODE mode)"
else
    echo "Speaker Recognition: Not configured"
fi
if [[ $SETUP_HTTPS == "yes" ]]; then
    echo "HTTPS: Enabled (localhost + $TAILSCALE_IP)"
    echo "Access URL: https://${HOST_IP} (or https://${TAILSCALE_IP})"
else
    echo "HTTPS: Not configured"
    echo "Access URL: http://${HOST_IP}:${WEBUI_PORT}"
fi
echo "Admin Email: $ADMIN_EMAIL"
echo

read -p "Continue with setup? [Y/n]: " confirm
if [[ $confirm == "n" || $confirm == "N" ]]; then
    print_info "Setup cancelled"
    exit 0
fi

# File Creation
print_header "📁 Creating Configuration Files"

# Navigate to backend directory
cd "$BACKEND_DIR"

# Create .env file
print_info "Creating .env file with all configured environment variables..."
cat > .env << EOF
# Friend-Lite Configuration
# Generated by init.sh on $(date)

# Authentication (sets AUTH_SECRET_KEY, ADMIN_PASSWORD, ADMIN_EMAIL)
AUTH_SECRET_KEY=$AUTH_SECRET_KEY
ADMIN_PASSWORD=$ADMIN_PASSWORD
ADMIN_EMAIL=$ADMIN_EMAIL

# LLM Configuration (sets LLM_PROVIDER, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL)
LLM_PROVIDER=$LLM_PROVIDER
OPENAI_API_KEY=$OPENAI_API_KEY
OPENAI_BASE_URL=$OPENAI_BASE_URL
OPENAI_MODEL=$OPENAI_MODEL

EOF

# Add transcription config if configured
if [[ -n $DEEPGRAM_API_KEY ]]; then
    cat >> .env << EOF
# Speech-to-Text Configuration (sets DEEPGRAM_API_KEY, TRANSCRIPTION_PROVIDER)
DEEPGRAM_API_KEY=$DEEPGRAM_API_KEY
TRANSCRIPTION_PROVIDER=$TRANSCRIPTION_PROVIDER

EOF
elif [[ -n $PARAKEET_ASR_URL ]]; then
    cat >> .env << EOF
# Speech-to-Text Configuration (sets PARAKEET_ASR_URL, TRANSCRIPTION_PROVIDER)
PARAKEET_ASR_URL=$PARAKEET_ASR_URL
TRANSCRIPTION_PROVIDER=$TRANSCRIPTION_PROVIDER

EOF
fi

# Add standard database and network config
cat >> .env << EOF
# Database Configuration
MONGODB_URI=mongodb://mongo:27017
QDRANT_BASE_URL=qdrant

# Network Configuration (sets HOST_IP, BACKEND_PUBLIC_PORT, WEBUI_PORT, CORS_ORIGINS)
HOST_IP=$HOST_IP
BACKEND_PUBLIC_PORT=$BACKEND_PORT
WEBUI_PORT=$WEBUI_PORT
CORS_ORIGINS=$CORS_ORIGINS
EOF

# Add speaker recognition config if configured
if [[ $SETUP_SPEAKER_RECOGNITION == "yes" ]]; then
    cat >> .env << EOF
# Speaker Recognition Configuration (sets HF_TOKEN, SPEAKER_SERVICE_URL)
HF_TOKEN=$HF_TOKEN
SPEAKER_SERVICE_URL=http://host.docker.internal:8085
EOF
fi

print_success ".env file created"

# Copy memory config for advanced backend
if [[ $BACKEND_DIR == "backends/advanced" ]]; then
    if [[ -f "memory_config.yaml.template" ]]; then
        print_info "Creating memory_config.yaml..."
        cp memory_config.yaml.template memory_config.yaml
        print_success "memory_config.yaml created from template"
    fi
fi

# Docker Setup
print_header "🐳 Starting Docker Services"
print_info "Building and starting services..."

# Start Docker services
if docker compose up --build -d; then
    print_success "Docker services started successfully!"
    echo
    
    print_info "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    if curl -s http://localhost:$BACKEND_PORT/health >/dev/null 2>&1; then
        print_success "Backend service is healthy"
    else
        print_warning "Backend may still be starting up"
    fi
    
else
    print_error "Failed to start Docker services"
    print_info "You can try running 'docker compose up --build -d' manually"
    exit 1
fi

# Additional Service Setup
cd ..  # Back to project root

# Set up speaker recognition service if requested
if [[ $SETUP_SPEAKER_RECOGNITION == "yes" ]]; then
    print_header "🗣️ Setting up Speaker Recognition Service"
    
    if [[ -d "extras/speaker-recognition" ]]; then
        cd extras/speaker-recognition
        
        # Create .env file for speaker recognition
        print_info "Creating speaker recognition .env file..."
        cat > .env << EOF
# Speaker Recognition Service Environment Configuration
# Generated by init.sh on $(date)

# Required: Hugging Face token for pyannote models
HF_TOKEN=$HF_TOKEN

# Docker build configuration
COMPUTE_MODE=$SPEAKER_COMPUTE_MODE

# Speaker recognition similarity threshold
SIMILARITY_THRESHOLD=$SIMILARITY_THRESHOLD

# Service Configuration
SPEAKER_SERVICE_HOST=speaker-service
SPEAKER_SERVICE_PORT=8085
SPEAKER_SERVICE_URL=http://speaker-service:8085

# React Web UI Configuration
REACT_UI_HOST=0.0.0.0
REACT_UI_PORT=5173
REACT_UI_HTTPS=false

# Optional: External Services
DEEPGRAM_API_KEY=$DEEPGRAM_API_KEY

# Test Configuration
SPEAKER_SERVICE_TEST_PORT=8086
EOF
        
        # Set up HTTPS for speaker recognition if requested
        if [[ $SETUP_HTTPS == "yes" ]]; then
            print_info "Setting up HTTPS for speaker recognition..."
            
            # Generate SSL certificates for speaker recognition
            if [[ -f "ssl/generate-ssl.sh" ]]; then
                chmod +x ssl/generate-ssl.sh
                ./ssl/generate-ssl.sh "$TAILSCALE_IP"
                print_success "SSL certificates generated for speaker recognition"
            else
                print_warning "SSL generation script not found for speaker recognition"
            fi
            
            # Configure nginx.conf with Tailscale IP
            if [[ -f "nginx.conf.template" ]]; then
                sed "s/TAILSCALE_IP/$TAILSCALE_IP/g" nginx.conf.template > nginx.conf
                print_success "nginx.conf configured for speaker recognition with Tailscale IP: $TAILSCALE_IP"
            else
                print_warning "nginx.conf.template not found for speaker recognition"
            fi
        fi
        
        print_info "Starting speaker recognition services..."
        if docker compose up --build -d; then
            print_success "Speaker recognition service started!"
        else
            print_warning "Failed to start speaker recognition service - check logs with 'docker compose logs'"
        fi
        
        cd ../..  # Back to project root
    else
        print_error "Speaker recognition directory not found at extras/speaker-recognition"
    fi
fi

# Set up HTTPS if requested
if [[ $SETUP_HTTPS == "yes" ]]; then
    print_header "🔒 Setting up HTTPS"
    
    cd "$BACKEND_DIR"
    
    # Generate SSL certificates
    print_info "Generating SSL certificates..."
    if [[ -f "ssl/generate-ssl.sh" ]]; then
        chmod +x ssl/generate-ssl.sh
        ./ssl/generate-ssl.sh "$TAILSCALE_IP"
        print_success "SSL certificates generated"
    else
        print_error "SSL generation script not found"
    fi
    
    # Configure nginx.conf with Tailscale IP
    print_info "Configuring nginx for HTTPS..."
    if [[ -f "nginx.conf.template" ]]; then
        sed "s/TAILSCALE_IP/$TAILSCALE_IP/g" nginx.conf.template > nginx.conf
        print_success "nginx.conf configured with Tailscale IP: $TAILSCALE_IP"
    else
        print_error "nginx.conf.template not found"
    fi
    
    # Restart services to enable nginx
    print_info "Restarting services to enable HTTPS..."
    docker compose restart
    
    # Wait for services to be ready
    sleep 5
    
    cd ..  # Back to project root
fi

# Final Success Message
print_header "🎉 Setup Complete!"

cat << EOF
Friend-Lite is now running! Here's how to access it:

EOF

if [[ $SETUP_HTTPS == "yes" ]]; then
    cat << EOF
🌐 Web Interface: https://${HOST_IP} or https://${TAILSCALE_IP}
🔧 API Endpoint: https://${HOST_IP}/api or https://${TAILSCALE_IP}/api

EOF
else
    cat << EOF
🌐 Web Interface: http://${HOST_IP}:${WEBUI_PORT}
🔧 API Endpoint: http://${HOST_IP}:${BACKEND_PORT}

EOF
fi

cat << EOF
👤 Admin Login:
   Email: ${ADMIN_EMAIL}
   Password: ${ADMIN_PASSWORD}

EOF

if [[ $SETUP_SPEAKER_RECOGNITION == "yes" ]]; then
    if [[ $SETUP_HTTPS == "yes" ]]; then
        cat << EOF
🗣️ Speaker Recognition: https://${HOST_IP}:8444 or https://${TAILSCALE_IP}:8444 (Web UI)
   Service API: https://${HOST_IP}:8444/api or https://${TAILSCALE_IP}:8444/api

EOF
    else
        cat << EOF
🗣️ Speaker Recognition: http://localhost:5173 (Web UI)
   Service API: http://localhost:8085

EOF
    fi
fi

cat << EOF
📖 Next Steps:
   1. Open the web interface and log in
   2. Connect your OMI device via the mobile app
   3. Start recording conversations!

📚 Documentation:
   - See CLAUDE.md for developer information
   - Visit backends/advanced/Docs/ for detailed guides
   
🧪 Testing:
   Run './run-test.sh' to test the full pipeline

🔧 Configuration:
   - Edit .env files to modify settings
   - Run 'docker compose restart' after changes
   - Use 'docker compose logs' to view service logs

EOF

if [[ -n $DEEPGRAM_API_KEY && -n $OPENAI_API_KEY ]]; then
    echo "🧪 To test the full pipeline:"
    echo "   cd backends/advanced && ./run-test.sh"
    echo
fi

print_success "Enjoy using Friend-Lite! 🚀"