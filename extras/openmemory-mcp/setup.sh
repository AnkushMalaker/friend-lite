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

echo "Starting OpenMemory MCP server..."
echo ""

# Start external server
docker compose up -d

echo "✅ OpenMemory MCP running!"
echo "  🌐 Server: http://host.docker.internal:8765"
echo "  📱 Web UI: http://localhost:8765"
echo ""
echo "💡 Set MEMORY_PROVIDER=openmemory_mcp in your backend configuration"