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

# Update .env file
sed -i "s|HF_TOKEN=.*|HF_TOKEN=$HF_TOKEN|" .env
sed -i "s|COMPUTE_MODE=.*|COMPUTE_MODE=$COMPUTE_MODE|" .env

echo ""
echo "✅ Speaker Recognition configured!"
echo "📁 Configuration saved to .env"
echo ""
echo "🚀 To start: docker compose up --build -d"