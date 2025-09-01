#!/bin/bash
set -e

echo "🧠 OpenMemory MCP Setup"
echo "======================"

# Check if already configured
if [ -f ".env" ]; then
    echo "⚠️  .env already exists. Backing up..."
    cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
fi

# Start from template
cp .env.template .env

# Clone the custom fork of mem0 with OpenMemory fixes
echo ""
echo "📦 Setting up custom mem0 fork with OpenMemory..."
if [ -d "cache/mem0" ]; then
    echo "  Removing existing mem0 directory..."
    rm -rf cache/mem0
fi

echo "  Cloning mem0 fork from AnkushMalaker/mem0..."
mkdir -p cache
git clone https://github.com/AnkushMalaker/mem0.git cache/mem0
cd cache/mem0
echo "  Checking out fix/get-endpoint branch..."
git checkout fix/get-endpoint
cd ../..

echo "✅ Custom mem0 fork ready with OpenMemory improvements"

# Prompt for OpenAI API key
echo ""
echo "🔑 OpenAI API Key (required for memory extraction)"
echo "Get yours from: https://platform.openai.com/api-keys"
read -p "OpenAI API Key: " OPENAI_API_KEY

# Update .env file
sed -i "s|OPENAI_API_KEY=.*|OPENAI_API_KEY=$OPENAI_API_KEY|" .env

echo ""
echo "✅ OpenMemory MCP configured!"
echo "📁 Configuration saved to .env"
echo ""
echo "🚀 To start: docker compose up --build -d"
echo "🌐 MCP Server: http://localhost:8765"
echo "📱 Web Interface: http://localhost:8765"
echo "🔧 UI (optional): docker compose --profile ui up -d"
echo ""
echo "💡 Note: Using custom mem0 fork from AnkushMalaker/mem0:fix/get-endpoint"