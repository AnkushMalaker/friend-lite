#!/bin/bash

# Friend-Lite Backend Startup Script
# Starts both the FastAPI backend and RQ workers

set -e

echo "🚀 Starting Friend-Lite Backend..."

# Function to handle shutdown
shutdown() {
    echo "🛑 Shutting down services..."
    pkill -TERM -P $$
    wait
    echo "✅ All services stopped"
    exit 0
}

# Set up signal handlers
trap shutdown SIGTERM SIGINT

# Start RQ workers in the background
echo "🔧 Starting RQ workers..."
uv run --extra deepgram rq worker transcription memory default --url "${REDIS_URL:-redis://localhost:6379/0}" &
WORKER_PID=$!

# Give workers a moment to start
sleep 2

# Start the main FastAPI application
echo "🌐 Starting FastAPI backend..."
uv run --extra deepgram python3 src/advanced_omi_backend/main.py &
BACKEND_PID=$!

# Wait for any process to exit
wait -n

# If we get here, one process has exited - kill the others
echo "⚠️  One service exited, stopping all services..."
kill $WORKER_PID $BACKEND_PID 2>/dev/null || true
wait

echo "🔄 All services stopped"
exit 1