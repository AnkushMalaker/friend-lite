FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/

COPY pyproject.toml .

# Install dependencies using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

COPY . .


CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
