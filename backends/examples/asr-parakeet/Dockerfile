# syntax=docker/dockerfile:1

#########################  builder #################################
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

# NeMo and texterrors need libs and C++ compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependency manifest first for cache‑friendly installs
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --compile-bytecode

# Add source and install project itself
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --compile-bytecode

#########################  runtime #################################
FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080
CMD ["python", "online.py"]
