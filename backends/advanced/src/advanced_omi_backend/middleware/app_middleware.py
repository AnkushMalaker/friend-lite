"""
Middleware configuration for Friend-Lite backend.

Centralizes CORS configuration and global exception handlers.
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo.errors import ConnectionFailure, PyMongoError

from advanced_omi_backend.app_config import get_app_config

logger = logging.getLogger(__name__)


def setup_cors_middleware(app: FastAPI) -> None:
    """Configure CORS middleware for the FastAPI application."""
    config = get_app_config()

    logger.info(f"🌐 CORS configured with origins: {config.allowed_origins}")
    logger.info(f"🌐 CORS also allows Tailscale IPs via regex: {config.tailscale_regex}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_origin_regex=config.tailscale_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Configure global exception handlers for the FastAPI application."""

    @app.exception_handler(ConnectionFailure)
    @app.exception_handler(PyMongoError)
    async def database_exception_handler(request: Request, exc: Exception):
        """Handle database connection failures and return structured error response."""
        logger.error(f"Database connection error: {type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Unable to connect to server. Please check your connection and try again.",
                "error_type": "connection_failure",
                "error_category": "database"
            }
        )

    @app.exception_handler(ConnectionError)
    async def connection_exception_handler(request: Request, exc: ConnectionError):
        """Handle general connection errors and return structured error response."""
        logger.error(f"Connection error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Unable to connect to server. Please check your connection and try again.",
                "error_type": "connection_failure",
                "error_category": "network"
            }
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with structured error response."""
        # For authentication failures (401), add error_type
        if exc.status_code == 401:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "detail": exc.detail,
                    "error_type": "authentication_failure",
                    "error_category": "security"
                }
            )

        # For other HTTP exceptions, return as normal
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )


def setup_middleware(app: FastAPI) -> None:
    """Set up all middleware for the FastAPI application."""
    setup_cors_middleware(app)
    setup_exception_handlers(app)