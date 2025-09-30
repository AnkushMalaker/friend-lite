#!/bin/bash

# Friend-Lite Backend Kubernetes Startup Script
# Starts both the FastAPI backend and RQ workers for K8s deployment

set -e

echo "🚀 Starting Friend-Lite Backend (Kubernetes)..."

# Debug environment variables
echo "🔍 Environment check:"
echo "  REDIS_URL: ${REDIS_URL:-NOT_SET}"
echo "  MONGODB_URI: ${MONGODB_URI:-NOT_SET}"

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

# Test Redis connectivity first
echo "🔍 Testing Redis connectivity..."
if [ -n "${REDIS_URL}" ]; then
    echo "  Using Redis URL: ${REDIS_URL}"
    # Try to ping Redis to verify connectivity
    timeout 5 python3 -c "
import redis
import sys
import os
try:
    r = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
    r.ping()
    print('  ✅ Redis connection successful')
except Exception as e:
    print(f'  ❌ Redis connection failed: {e}')
    sys.exit(1)
" || {
        echo "  ❌ Redis connectivity test failed, continuing anyway..."
    }
else
    echo "  ⚠️  REDIS_URL not set, using default"
fi

# Start RQ workers in the background
echo "🔧 Starting RQ workers..."
if uv run --no-sync rq worker transcription memory default --url "${REDIS_URL:-redis://localhost:6379/0}" &
then
    WORKER_PID=$!
    echo "  ✅ RQ workers started with PID: $WORKER_PID"
else
    echo "  ❌ Failed to start RQ workers"
    exit 1
fi

# Give workers a moment to start
sleep 3

# Start the main FastAPI application
echo "🌐 Starting FastAPI backend..."
if uv run --no-sync python3 src/advanced_omi_backend/main.py &
then
    BACKEND_PID=$!
    echo "  ✅ FastAPI backend started with PID: $BACKEND_PID"
else
    echo "  ❌ Failed to start FastAPI backend"
    kill $WORKER_PID 2>/dev/null || true
    exit 1
fi

echo "🎉 All services started successfully!"
echo "  - RQ Workers: $WORKER_PID"
echo "  - FastAPI Backend: $BACKEND_PID"

# Wait for any process to exit
wait -n

# If we get here, one process has exited - kill the others
echo "⚠️  One service exited, stopping all services..."
kill $WORKER_PID $BACKEND_PID 2>/dev/null || true
wait

echo "🔄 All services stopped"
exit 1