#!/bin/bash
set -e

echo "🧠 OpenMemory MCP Setup"
echo "======================"

echo "Starting OpenMemory MCP server..."
echo ""

# Start external server
docker compose up -d

echo "✅ OpenMemory MCP running!"
echo "  🌐 Server: http://host.docker.internal:8765"
echo "  📱 Web UI: http://localhost:8765"
echo ""
echo "💡 Set MEMORY_PROVIDER=openmemory_mcp in your backend configuration"