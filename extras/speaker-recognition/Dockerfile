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
t 
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files first
COPY pyproject.toml ./
COPY uv.lock ./

# Copy the source code maintaining the src/ structure
COPY src/ ./src/

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
