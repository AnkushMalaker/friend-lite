#!/bin/bash
set -e

echo "🎤 Offline ASR Setup"
echo "==================="

echo "Starting Parakeet ASR service..."
echo ""

# Start Parakeet ASR service
docker compose up parakeet -d

echo "✅ Parakeet ASR running:"
echo "  📝 Service URL: http://host.docker.internal:8767"
echo ""
echo "💡 Configure PARAKEET_ASR_URL in your backend's transcription settings"