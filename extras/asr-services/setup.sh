#!/bin/bash
set -e

echo "🎤 Offline ASR Setup"
echo "==================="

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📄 Creating .env file from template..."
    cp .env.template .env
    echo "✅ .env file created"
else
    echo "ℹ️  .env file already exists, using existing configuration"
fi

echo "Starting Parakeet ASR service..."
echo ""

# Start Parakeet ASR service
docker compose up parakeet-asr -d

echo "✅ Parakeet ASR running:"
echo "  📝 Service URL: http://host.docker.internal:8767"
echo ""
echo "💡 Configure PARAKEET_ASR_URL in your backend's transcription settings"