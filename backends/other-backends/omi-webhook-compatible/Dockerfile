# syntax=docker/dockerfile:1

#########################  builder #################################
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

# NeMo and texterrors need libs and C++ compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Dependency manifest first for cache‑friendly installs
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --compile-bytecode

# Add source and install project itself
COPY . .

#########################  runtime #################################
FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

EXPOSE 8000
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
