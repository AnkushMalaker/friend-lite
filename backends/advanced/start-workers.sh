#!/bin/bash
# Unified worker startup script
# Starts all workers in a single container for efficiency

set -e

echo "🚀 Starting Friend-Lite Workers..."

# Clean up any stale worker registrations from previous runs
echo "🧹 Cleaning up stale worker registrations from Redis..."
# Use RQ's cleanup command to remove dead workers
uv run python -c "
from rq import Worker
from redis import Redis
import os
import socket

redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
redis_conn = Redis.from_url(redis_url)
hostname = socket.gethostname()

# Only clean up workers from THIS hostname (pod)
workers = Worker.all(connection=redis_conn)
cleaned = 0
for worker in workers:
    if hostname in worker.name:
        worker.register_death()
        cleaned += 1
print(f'Cleaned up {cleaned} stale workers from {hostname}')
" 2>/dev/null || echo "No stale workers to clean"

sleep 1

# Function to handle shutdown
shutdown() {
    echo "🛑 Shutting down workers..."
    kill $RQ_WORKER_1_PID 2>/dev/null || true
    kill $RQ_WORKER_2_PID 2>/dev/null || true
    kill $RQ_WORKER_3_PID 2>/dev/null || true
    kill $RQ_WORKER_4_PID 2>/dev/null || true
    kill $RQ_WORKER_5_PID 2>/dev/null || true
    kill $RQ_WORKER_6_PID 2>/dev/null || true
    kill $AUDIO_PERSISTENCE_WORKER_PID 2>/dev/null || true
    [ -n "$AUDIO_STREAM_WORKER_PID" ] && kill $AUDIO_STREAM_WORKER_PID 2>/dev/null || true
    [ -n "$PARAKEET_STREAM_WORKER_PID" ] && kill $PARAKEET_STREAM_WORKER_PID 2>/dev/null || true
    wait
    echo "✅ All workers stopped"
    exit 0
}

# Set up signal handlers
trap shutdown SIGTERM SIGINT

# Configure Python logging for RQ workers
export PYTHONUNBUFFERED=1

# Start 6 RQ workers listening to ALL queues
echo "🔧 Starting RQ workers (6 workers, all queues: transcription, memory, default)..."
uv run python -m advanced_omi_backend.workers.rq_worker_entry transcription memory default &
RQ_WORKER_1_PID=$!
uv run python -m advanced_omi_backend.workers.rq_worker_entry transcription memory default &
RQ_WORKER_2_PID=$!
uv run python -m advanced_omi_backend.workers.rq_worker_entry transcription memory default &
RQ_WORKER_3_PID=$!
uv run python -m advanced_omi_backend.workers.rq_worker_entry transcription memory default &
RQ_WORKER_4_PID=$!
uv run python -m advanced_omi_backend.workers.rq_worker_entry transcription memory default &
RQ_WORKER_5_PID=$!
uv run python -m advanced_omi_backend.workers.rq_worker_entry transcription memory default &
RQ_WORKER_6_PID=$!

# Start 1 dedicated audio persistence worker
# Single worker for audio persistence jobs (file rotation)
echo "💾 Starting audio persistence worker (1 worker for audio queue)..."
uv run python -m advanced_omi_backend.workers.rq_worker_entry audio &
AUDIO_PERSISTENCE_WORKER_PID=$!

# Start 1 audio stream worker for Deepgram (only if DEEPGRAM_API_KEY is set)
# Single worker ensures sequential processing of audio chunks
if [ -n "$DEEPGRAM_API_KEY" ]; then
    echo "🎵 Starting audio stream Deepgram worker (1 worker for sequential processing)..."
    uv run python -m advanced_omi_backend.workers.audio_stream_deepgram_worker &
    AUDIO_STREAM_WORKER_PID=$!
else
    echo "⏭️  Skipping Deepgram audio stream worker (DEEPGRAM_API_KEY not set)"
    AUDIO_STREAM_WORKER_PID=""
fi

# Start 1 audio stream worker for Parakeet (only if PARAKEET_ASR_URL or OFFLINE_ASR_TCP_URI is set)
# Single worker ensures sequential processing of audio chunks
PARAKEET_URL="${PARAKEET_ASR_URL:-${OFFLINE_ASR_TCP_URI:-}}"
if [ -n "$PARAKEET_URL" ]; then
    echo "🎤 Starting audio stream Parakeet worker (1 worker for sequential processing)..."
    uv run python -m advanced_omi_backend.workers.audio_stream_parakeet_worker &
    PARAKEET_STREAM_WORKER_PID=$!
else
    echo "⏭️  Skipping Parakeet audio stream worker (PARAKEET_ASR_URL or OFFLINE_ASR_TCP_URI not set)"
    PARAKEET_STREAM_WORKER_PID=""
fi

echo "✅ All workers started:"
echo "  - RQ worker 1: PID $RQ_WORKER_1_PID (transcription, memory, default)"
echo "  - RQ worker 2: PID $RQ_WORKER_2_PID (transcription, memory, default)"
echo "  - RQ worker 3: PID $RQ_WORKER_3_PID (transcription, memory, default)"
echo "  - RQ worker 4: PID $RQ_WORKER_4_PID (transcription, memory, default)"
echo "  - RQ worker 5: PID $RQ_WORKER_5_PID (transcription, memory, default)"
echo "  - RQ worker 6: PID $RQ_WORKER_6_PID (transcription, memory, default)"
echo "  - Audio persistence worker: PID $AUDIO_PERSISTENCE_WORKER_PID (audio queue - file rotation)"
if [ -n "$AUDIO_STREAM_WORKER_PID" ]; then
    echo "  - Audio stream Deepgram worker: PID $AUDIO_STREAM_WORKER_PID (Redis Streams consumer - sequential processing)"
fi
if [ -n "$PARAKEET_STREAM_WORKER_PID" ]; then
    echo "  - Audio stream Parakeet worker: PID $PARAKEET_STREAM_WORKER_PID (Redis Streams consumer - sequential processing)"
fi

# Wait for any process to exit
wait -n

# If we get here, one process has exited - kill the others
echo "⚠️  One worker exited, stopping all workers..."
kill $RQ_WORKER_1_PID 2>/dev/null || true
kill $RQ_WORKER_2_PID 2>/dev/null || true
kill $RQ_WORKER_3_PID 2>/dev/null || true
kill $RQ_WORKER_4_PID 2>/dev/null || true
kill $RQ_WORKER_5_PID 2>/dev/null || true
kill $RQ_WORKER_6_PID 2>/dev/null || true
kill $AUDIO_PERSISTENCE_WORKER_PID 2>/dev/null || true
[ -n "$AUDIO_STREAM_WORKER_PID" ] && kill $AUDIO_STREAM_WORKER_PID 2>/dev/null || true
wait

echo "🔄 All workers stopped"
exit 1
