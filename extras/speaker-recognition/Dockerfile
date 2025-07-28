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

# Copy dependency files and source code
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies and package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev
# RUN uv pip install -e .

# Create directories
RUN mkdir -p /app/audio_chunks /app/debug /app/data /models

# Set environment variables
ENV HF_HOME=/models
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Run the service
CMD ["uv", "run", "simple-speaker-service"] 