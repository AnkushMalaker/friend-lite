FROM python:3.12-slim-bookworm AS builder

# Install system dependencies for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
    # portaudio19-dev \

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/

# Set up the working directory
WORKDIR /app

# Copy package structure and dependency files first
COPY pyproject.toml README.md ./
COPY uv.lock .
RUN mkdir -p src/advanced_omi_backend
COPY src/advanced_omi_backend/__init__.py src/advanced_omi_backend/

# Install dependencies using uv with deepgram extra
# Use cache mount for BuildKit, fallback for legacy builds
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --extra deepgram
# Fallback for legacy Docker builds (CI compatibility)
RUN uv sync --extra deepgram

# Copy all application code
COPY . .

# Copy configuration files if they exist, otherwise they will be created from templates at runtime
# The files are expected to exist, but we handle the case where they don't gracefully
COPY memory_config.yaml* ./
COPY diarization_config.json* ./


# Copy and make startup scripts executable
COPY start.sh ./
COPY start-workers.sh ./
RUN chmod +x start.sh start-workers.sh

# Run the application with workers
CMD ["./start.sh"]
