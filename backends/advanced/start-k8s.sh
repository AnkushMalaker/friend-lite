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
    kill $AUDIO_WORKER_1_PID 2>/dev/null || true
    kill $RQ_WORKER_1_PID 2>/dev/null || true
    kill $RQ_WORKER_2_PID 2>/dev/null || true
    kill $RQ_WORKER_3_PID 2>/dev/null || true
    kill $BACKEND_PID 2>/dev/null || true
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

# OLD WORKERS - Disabled for testing new Redis Streams architecture
# These have been renamed to old_audio_stream_worker.py and old_transcription_stream_worker.py
# echo "🎵 Starting Redis Streams audio workers (2 workers)..."
# if uv run --no-sync python3 -m advanced_omi_backend.workers.old_audio_stream_worker &
# then
#     AUDIO_WORKER_1_PID=$!
#     echo "  ✅ Audio stream worker 1 started with PID: $AUDIO_WORKER_1_PID"
# else
#     echo "  ❌ Failed to start audio stream worker 1"
#     exit 1
# fi

# if uv run --no-sync python3 -m advanced_omi_backend.workers.old_audio_stream_worker &
# then
#     AUDIO_WORKER_2_PID=$!
#     echo "  ✅ Audio stream worker 2 started with PID: $AUDIO_WORKER_2_PID"
# else
#     echo "  ❌ Failed to start audio stream worker 2"
#     kill $AUDIO_WORKER_1_PID 2>/dev/null || true
#     exit 1
# fi

# echo "📝 Starting transcription stream workers (2 workers)..."
# if uv run --no-sync python3 -m advanced_omi_backend.workers.old_transcription_stream_worker &
# then
#     TRANSCRIPTION_WORKER_1_PID=$!
#     echo "  ✅ Transcription stream worker 1 started with PID: $TRANSCRIPTION_WORKER_1_PID"
# else
#     echo "  ❌ Failed to start transcription stream worker 1"
#     kill $AUDIO_WORKER_1_PID $AUDIO_WORKER_2_PID 2>/dev/null || true
#     exit 1
# fi

# if uv run --no-sync python3 -m advanced_omi_backend.workers.old_transcription_stream_worker &
# then
#     TRANSCRIPTION_WORKER_2_PID=$!
#     echo "  ✅ Transcription stream worker 2 started with PID: $TRANSCRIPTION_WORKER_2_PID"
# else
#     echo "  ❌ Failed to start transcription stream worker 2"
#     kill $AUDIO_WORKER_1_PID $AUDIO_WORKER_2_PID $TRANSCRIPTION_WORKER_1_PID 2>/dev/null || true
#     exit 1
# fi

# NEW WORKERS - Redis Streams multi-provider architecture
# Single worker ensures sequential processing of audio chunks (matching start-workers.sh)
echo "🎵 Starting audio stream Deepgram worker (1 worker for sequential processing)..."
if uv run --no-sync python3 -m advanced_omi_backend.workers.audio_stream_deepgram_worker &
then
    AUDIO_WORKER_1_PID=$!
    echo "  ✅ Deepgram stream worker started with PID: $AUDIO_WORKER_1_PID"
else
    echo "  ❌ Failed to start Deepgram stream worker"
    exit 1
fi

# Start 3 RQ workers listening to ALL queues (matching start-workers.sh)
echo "🔧 Starting RQ workers (3 workers, all queues: transcription, memory, default)..."
if uv run --no-sync rq worker transcription memory default --url "${REDIS_URL:-redis://localhost:6379/0}" --verbose --logging_level INFO &
then
    RQ_WORKER_1_PID=$!
    echo "  ✅ RQ worker 1 started with PID: $RQ_WORKER_1_PID"
else
    echo "  ❌ Failed to start RQ worker 1"
    kill $AUDIO_WORKER_1_PID 2>/dev/null || true
    exit 1
fi

if uv run --no-sync rq worker transcription memory default --url "${REDIS_URL:-redis://localhost:6379/0}" --verbose --logging_level INFO &
then
    RQ_WORKER_2_PID=$!
    echo "  ✅ RQ worker 2 started with PID: $RQ_WORKER_2_PID"
else
    echo "  ❌ Failed to start RQ worker 2"
    kill $AUDIO_WORKER_1_PID $RQ_WORKER_1_PID 2>/dev/null || true
    exit 1
fi

if uv run --no-sync rq worker transcription memory default --url "${REDIS_URL:-redis://localhost:6379/0}" --verbose --logging_level INFO &
then
    RQ_WORKER_3_PID=$!
    echo "  ✅ RQ worker 3 started with PID: $RQ_WORKER_3_PID"
else
    echo "  ❌ Failed to start RQ worker 3"
    kill $AUDIO_WORKER_1_PID $RQ_WORKER_1_PID $RQ_WORKER_2_PID 2>/dev/null || true
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
    kill $AUDIO_WORKER_1_PID $RQ_WORKER_1_PID $RQ_WORKER_2_PID $RQ_WORKER_3_PID 2>/dev/null || true
    exit 1
fi

echo "🎉 All services started successfully!"
echo "  - Audio stream worker: $AUDIO_WORKER_1_PID (Redis Streams consumer - sequential processing)"
echo "  - RQ worker 1: $RQ_WORKER_1_PID (transcription, memory, default)"
echo "  - RQ worker 2: $RQ_WORKER_2_PID (transcription, memory, default)"
echo "  - RQ worker 3: $RQ_WORKER_3_PID (transcription, memory, default)"
echo "  - FastAPI Backend: $BACKEND_PID"

# Wait for any process to exit
wait -n

# If we get here, one process has exited - kill the others
echo "⚠️  One service exited, stopping all services..."
# Kill only non-empty PIDs
[ -n "$AUDIO_WORKER_1_PID" ] && kill $AUDIO_WORKER_1_PID 2>/dev/null || true
[ -n "$RQ_WORKER_1_PID" ] && kill $RQ_WORKER_1_PID 2>/dev/null || true
[ -n "$RQ_WORKER_2_PID" ] && kill $RQ_WORKER_2_PID 2>/dev/null || true
[ -n "$RQ_WORKER_3_PID" ] && kill $RQ_WORKER_3_PID 2>/dev/null || true
[ -n "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null || true
wait

echo "🔄 All services stopped"
exit 1