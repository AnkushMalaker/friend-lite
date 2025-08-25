"""
Main API router for Friend-Lite backend.

This module aggregates all the functional router modules and provides
a single entry point for the API endpoints.
"""

import logging

from fastapi import APIRouter

from .modules import (
    chat_router,
    client_router,
    conversation_router,
    memory_router,
    system_router,
    user_router,
)

logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")

# Create main API router
router = APIRouter(prefix="/api", tags=["api"])

# Include all sub-routers
router.include_router(user_router)
router.include_router(chat_router)
router.include_router(client_router)
router.include_router(conversation_router)
router.include_router(memory_router)
router.include_router(system_router)


logger.info("API router initialized with all sub-modules")
