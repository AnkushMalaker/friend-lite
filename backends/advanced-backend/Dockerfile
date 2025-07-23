FROM python:3.12-slim-bookworm AS builder

# Install system dependencies for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    git \
    curl \
    ffmpeg \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/

# Set up the working directory
WORKDIR /app

# Copy package structure and dependency files first
COPY pyproject.toml README.md ./
RUN mkdir -p src/advanced_omi_backend
COPY src/advanced_omi_backend/__init__.py src/advanced_omi_backend/

# Install dependencies using uv with deepgram extra
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --extra deepgram
# Somehow the above is breaking consecutive builds.
RUN uv sync --extra deepgram

# Copy all application code
COPY . .

# Copy memory config to the expected location
COPY memory_config.yaml ./


# Run the application
CMD ["uv", "run", "python3", "src/advanced_omi_backend/main.py"]
