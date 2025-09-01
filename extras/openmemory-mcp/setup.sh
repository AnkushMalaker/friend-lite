#!/bin/bash
set -e

echo "🧠 OpenMemory MCP Setup"
echo "======================"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📄 Creating .env file from template..."
    cp .env.template .env
    echo "✅ .env file created"
else
    echo "ℹ️  .env file already exists, using existing configuration"
fi

echo "✅ OpenMemory MCP configured!"
echo "📁 Configuration saved to .env"
echo ""
echo "🚀 To start: docker compose up --build -d"
echo "  🌐 Server: http://host.docker.internal:8765"
echo "  📱 Web UI: http://localhost:8765"
echo ""
echo "💡 Only start if you selected openmemory_mcp as memory provider"