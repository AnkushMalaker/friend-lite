FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
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
COPY client.py ./

# Create directories
RUN mkdir -p /app/audio_chunks /app/debug /models

# Set environment variables
ENV HF_HOME=/models
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8001

# Run the service
CMD ["uv", "run", "python", "speaker_service.py"] 