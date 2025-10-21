#!/bin/bash

set -e

echo "🚀 Starting OpenMemory MCP installation for Friend-Lite..."

# Set environment variables
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
USER="${USER:-$(whoami)}"

# Check for .env file first, load if exists
if [ -f .env ]; then
    echo "📝 Loading configuration from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
else
    # Create .env from template if it doesn't exist
    if [ -f .env.template ]; then
        echo "📝 Creating .env from template..."
        cp .env.template .env
    fi
fi

# If OPENAI_API_KEY is provided via environment but not in .env, write it
if [ -n "$OPENAI_API_KEY" ] && [ -f .env ]; then
    # Update or add OPENAI_API_KEY in .env file
    if grep -q "^OPENAI_API_KEY=" .env 2>/dev/null; then
        # Key exists, update it
        sed -i "s|^OPENAI_API_KEY=.*|OPENAI_API_KEY=${OPENAI_API_KEY}|" .env
    else
        # Key doesn't exist, append it
        echo "OPENAI_API_KEY=${OPENAI_API_KEY}" >> .env
    fi
    echo "✅ Updated .env with provided OPENAI_API_KEY"
fi

# Final check
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set."
    echo "   Option 1: Edit .env file and add your key"
    echo "   Option 2: Run with: OPENAI_API_KEY=your_api_key ./run.sh"
    echo "   Option 3: Export it: export OPENAI_API_KEY=your_api_key"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if docker compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose V2."
    exit 1
fi

# Export required variables for Compose
export OPENAI_API_KEY
export USER

# Parse command line arguments
PROFILE=""
if [ "$1" = "--with-ui" ]; then
    PROFILE="--profile ui"
    echo "🎨 UI will be enabled at http://localhost:3001"
fi

# Start services
echo "🚀 Starting OpenMemory MCP services..."
docker compose up -d $PROFILE

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check if services are running
if docker ps | grep -q openmemory-mcp; then
    echo "✅ OpenMemory MCP Backend: http://localhost:8765"
    echo "✅ OpenMemory Qdrant:      http://localhost:6334"
    if [ "$1" = "--with-ui" ]; then
        echo "✅ OpenMemory UI:          http://localhost:3001"
        echo "✅ OpenMemory MCP API:     http://localhost:8765/openapi.json"
        echo "   Available endpoints:"
        curl -s http://localhost:8765/openapi.json | jq '.paths | keys[]'
    fi
    echo ""
    echo "📚 Integration with Friend-Lite:"
    echo "   Set MEMORY_PROVIDER=openmemory_mcp in your Friend-Lite .env"
    echo "   Set OPENMEMORY_MCP_URL=http://localhost:8765 in your Friend-Lite .env"
    echo ""
    echo "🔍 Check logs: docker compose logs -f"
    echo "🛑 Stop services: docker compose down"
else
    echo "❌ Failed to start OpenMemory MCP services"
    echo "   Check logs: docker compose logs"
    exit 1
fi