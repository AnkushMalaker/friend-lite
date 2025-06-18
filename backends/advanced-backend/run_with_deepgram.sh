#!/bin/bash

echo "Installing Deepgram SDK for advanced transcription features..."

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv to install Deepgram SDK..."
    uv sync --group deepgram
else
    echo "uv not found, using pip..."
    pip install deepgram-sdk
fi

echo "Deepgram SDK installation complete!"
echo "Don't forget to set your DEEPGRAM_API_KEY environment variable." 