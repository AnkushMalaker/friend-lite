#!/bin/sh
# Health check script for web-ui service
# Checks HTTP or HTTPS based on REACT_UI_HTTPS environment variable

PORT="${REACT_UI_PORT:-5173}"
USE_HTTPS="${REACT_UI_HTTPS:-false}"

if [ "$USE_HTTPS" = "true" ]; then
    curl -f -k "https://localhost:${PORT}"
else
    curl -f "http://localhost:${PORT}"
fi
