FROM python:3.12-bookworm

# Install system dependencies including PortAudio for pyaudio
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*
    # portaudio19-dev \
    # libasound2-dev \

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files first (for better caching)
COPY pyproject.toml ./
COPY uv.lock ./

# Create minimal package structure for dependency installation
RUN mkdir -p src/simple_speaker_recognition
COPY src/simple_speaker_recognition/__init__.py src/simple_speaker_recognition/

# Install dependencies and package  
# Use build arg to control CPU vs GPU mode
ARG COMPUTE_MODE=cpu
RUN if [ "$COMPUTE_MODE" = "gpu" ]; then \
        uv sync --no-default-groups --group gpu; \
    else \
        uv sync --no-default-groups --group cpu; \
    fi

# Create directories
RUN mkdir -p /app/audio_chunks /app/debug /app/data /models

# Set environment variables
ENV HF_HOME=/models
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8085

# Run the service
CMD ["uv", "run", "--no-dev", "simple-speaker-service"] 
