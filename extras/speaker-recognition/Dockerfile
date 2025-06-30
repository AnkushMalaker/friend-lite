FROM python:3.12-bookworm

# Install system dependencies including PortAudio for pyaudio
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --no-dev

# Copy application code
COPY speaker_service.py ./
# COPY client.py ./

# Create directories
RUN mkdir -p /app/audio_chunks /app/debug /app/data /models

# Set environment variables
ENV HF_HOME=/models
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Run the service
CMD ["uv", "run", "python", "speaker_service.py"] 