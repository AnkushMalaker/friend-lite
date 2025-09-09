#!/bin/bash

# Enable strict error handling
set -euo pipefail

# Error trap for better diagnostics
trap 'echo "Error on line $LINENO"; exit 1' ERR

# Parse command line arguments
ENABLE_HTTPS=false
SERVER_IP=""
DEEPGRAM_API_KEY=""
HF_TOKEN=""
COMPUTE_MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-https)
            ENABLE_HTTPS=true
            shift
            ;;
        --server-ip)
            SERVER_IP="$2"
            shift 2
            ;;
        --deepgram-api-key)
            DEEPGRAM_API_KEY="$2"
            shift 2
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --compute-mode)
            COMPUTE_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "🗣️ Speaker Recognition Setup"
echo "=============================="

# Check if already configured
if [ -f ".env" ]; then
    echo "⚠️  .env already exists. Backing up..."
    cp .env ".env.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Start from template - check existence first
if [ ! -r ".env.template" ]; then
    echo "Error: .env.template not found or not readable" >&2
    exit 1
fi

# Copy template and set secure permissions
if ! cp .env.template .env; then
    echo "Error: Failed to copy .env.template to .env" >&2
    exit 1
fi

# Set restrictive permissions (owner read/write only)
chmod 600 .env

# Get HF Token (prompt only if not provided via command line)
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "🔑 Hugging Face Token (required for pyannote models)"
    echo "Get yours from: https://huggingface.co/settings/tokens"
    read -s -r -p "HF Token: " HF_TOKEN
    echo  # Print newline after silent input
else
    echo "✅ HF Token configured from command line"
fi

# Validate token is not empty
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF token is required" >&2
    exit 1
fi

# Get Compute Mode (prompt only if not provided via command line)
if [ -z "$COMPUTE_MODE" ]; then
    echo ""
    echo "🖥️ Compute Mode"
    echo "  1) CPU-only (works everywhere)"  
    echo "  2) GPU acceleration (requires NVIDIA+CUDA)"
    read -p "Enter choice [1-2] (1): " COMPUTE_CHOICE
    COMPUTE_CHOICE=${COMPUTE_CHOICE:-1}

    if [ "$COMPUTE_CHOICE" = "2" ]; then
        COMPUTE_MODE="gpu"
    else
        COMPUTE_MODE="cpu"
    fi
else
    echo "✅ Compute mode configured from command line: $COMPUTE_MODE"
fi

# Validate compute mode
if [ "$COMPUTE_MODE" != "cpu" ] && [ "$COMPUTE_MODE" != "gpu" ]; then
    echo "Error: Invalid compute mode '$COMPUTE_MODE'. Must be 'cpu' or 'gpu'" >&2
    exit 1
fi

# HTTPS Configuration
if [ "$ENABLE_HTTPS" = true ]; then
    # Use command line arguments
    HTTPS_MODE="https"
    if [ -z "$SERVER_IP" ]; then
        SERVER_IP="localhost"
    fi
    echo "✅ HTTPS configured via command line: $SERVER_IP"
else
    # Interactive configuration
    echo ""
    echo "🔒 HTTPS Configuration (required for microphone access)"
    echo "  1) HTTP mode (development, localhost only)"
    echo "  2) HTTPS mode with SSL (production, remote access, microphone access)"
    read -p "Enter choice [1-2] (1): " HTTPS_CHOICE
    HTTPS_CHOICE=${HTTPS_CHOICE:-1}

    if [ "$HTTPS_CHOICE" = "2" ]; then
        HTTPS_MODE="https"
    else
        HTTPS_MODE="http"
    fi
fi

# Update .env file
sed -i "s|HF_TOKEN=.*|HF_TOKEN=$HF_TOKEN|" .env
sed -i "s|COMPUTE_MODE=.*|COMPUTE_MODE=$COMPUTE_MODE|" .env

# Update Deepgram API key if provided via command line
if [ -n "$DEEPGRAM_API_KEY" ]; then
    sed -i "s|DEEPGRAM_API_KEY=.*|DEEPGRAM_API_KEY=$DEEPGRAM_API_KEY|" .env
    echo "✅ Deepgram API key configured from command line"
fi

if [ "$HTTPS_MODE" = "https" ]; then
    # Get SERVER_IP if not already set (from command line)
    if [ -z "$SERVER_IP" ]; then
        echo ""
        echo "🌐 Server Configuration for HTTPS"
        echo "Enter your server IP/domain for SSL certificate"
        echo "Examples: localhost, 192.168.1.100, your-domain.com"
        read -p "Server IP/Domain (localhost): " SERVER_IP
        SERVER_IP=${SERVER_IP:-localhost}
    fi
    
    # Update .env for HTTPS mode
    sed -i "s|REACT_UI_HTTPS=.*|REACT_UI_HTTPS=true|" .env
    sed -i "s|REACT_UI_PORT=.*|REACT_UI_PORT=5175|" .env
    
    # Generate SSL certificates
    echo ""
    echo "📄 Generating SSL certificates..."
    if [ -f "ssl/generate-ssl.sh" ]; then
        ./ssl/generate-ssl.sh "$SERVER_IP"
        echo "✅ SSL certificates generated"
    else
        echo "⚠️  ssl/generate-ssl.sh not found, SSL setup skipped"
    fi
    
    # Create nginx.conf from template
    echo "📄 Creating nginx configuration..."
    if [ -f "nginx.conf.template" ]; then
        sed "s/TAILSCALE_IP/$SERVER_IP/g" nginx.conf.template > nginx.conf
        echo "✅ nginx.conf created for: $SERVER_IP"
    else
        echo "⚠️  nginx.conf.template not found, nginx config skipped"
    fi
    
    echo ""
    echo "✅ Speaker Recognition configured (HTTPS mode)!"
    echo "📁 Configuration saved to .env"
    echo "🔒 HTTPS configured for: $SERVER_IP"
    echo ""
    echo "🚀 To start: docker compose up --build -d"
    echo "🌐 HTTPS Access: https://localhost:8444/"
    echo "🌐 HTTP Redirect: http://localhost:8081/ → HTTPS"
    echo "📱 Service API: https://localhost:8444/api/"
    echo "💡 Accept SSL certificate in browser"
else
    # HTTP Configuration
    sed -i "s|REACT_UI_HTTPS=.*|REACT_UI_HTTPS=false|" .env
    sed -i "s|REACT_UI_PORT=.*|REACT_UI_PORT=5174|" .env
    
    echo ""
    echo "✅ Speaker Recognition configured (HTTP mode)!"
    echo "📁 Configuration saved to .env"
    echo ""
    echo "🚀 To start: docker compose up --build -d speaker-service web-ui"
    echo "📱 Service API: http://localhost:8085"
    echo "📱 Web Interface: http://localhost:5174"
    echo "⚠️  Note: Microphone access may not work over HTTP on remote connections"
fi