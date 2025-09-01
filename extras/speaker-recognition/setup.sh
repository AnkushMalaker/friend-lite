#!/bin/bash
set -e

echo "🗣️ Speaker Recognition Setup"
echo "=============================="

# Check if already configured
if [ -f ".env" ]; then
    echo "⚠️  .env already exists. Backing up..."
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
fi

# Start from template
cp .env.template .env

# Prompt for required settings
echo ""
echo "🔑 Hugging Face Token (required for pyannote models)"
echo "Get yours from: https://huggingface.co/settings/tokens"
read -p "HF Token: " HF_TOKEN

echo ""
echo "🖥️ Compute Mode"
echo "  cpu - CPU-only (works everywhere)"  
echo "  gpu - GPU acceleration (requires NVIDIA+CUDA)"
read -p "Compute Mode [cpu/gpu]: " COMPUTE_MODE
COMPUTE_MODE=${COMPUTE_MODE:-cpu}

echo ""
echo "🔒 HTTPS Configuration (required for microphone access)"
echo "  http  - Simple HTTP mode (development, localhost only)"
echo "  https - HTTPS mode with SSL (production, remote access, microphone access)"
read -p "Mode [http/https]: " HTTPS_MODE
HTTPS_MODE=${HTTPS_MODE:-http}

# Update .env file
sed -i "s|HF_TOKEN=.*|HF_TOKEN=$HF_TOKEN|" .env
sed -i "s|COMPUTE_MODE=.*|COMPUTE_MODE=$COMPUTE_MODE|" .env

if [ "$HTTPS_MODE" = "https" ]; then
    # HTTPS Configuration
    echo ""
    echo "🌐 Server Configuration for HTTPS"
    echo "Enter your server IP/domain for SSL certificate"
    echo "Examples: localhost, 192.168.1.100, your-domain.com"
    read -p "Server IP/Domain (localhost): " SERVER_IP
    SERVER_IP=${SERVER_IP:-localhost}
    
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