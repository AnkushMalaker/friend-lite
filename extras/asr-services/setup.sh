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

echo "✅ Parakeet ASR configured!"
echo "📁 Configuration saved to .env"
echo ""
echo "🚀 To start: docker compose up --build -d parakeet-asr"
echo "  📝 Service will be available at: http://host.docker.internal:8767"
echo ""
echo "💡 Only start if you want to use offline ASR instead of cloud providers"