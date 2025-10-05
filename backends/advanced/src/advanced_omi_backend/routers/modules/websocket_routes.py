"""
WebSocket routes for Friend-Lite backend.

This module handles WebSocket connections for audio streaming.
"""

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional

from advanced_omi_backend.controllers.websocket_controller import (
    handle_omi_websocket,
    handle_pcm_websocket
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["websocket"])

@router.websocket("/ws_omi")
async def ws_endpoint_omi(
    ws: WebSocket,
    token: Optional[str] = Query(None),
    device_name: Optional[str] = Query(None),
):
    """Accepts WebSocket connections with Wyoming protocol, decodes OMI Opus audio, and processes per-client."""
    await handle_omi_websocket(ws, token, device_name)


@router.websocket("/ws_pcm")
async def ws_endpoint_pcm(
    ws: WebSocket,
    token: Optional[str] = Query(None),
    device_name: Optional[str] = Query(None)
):
    """Accepts WebSocket connections, processes PCM audio per-client."""
    await handle_pcm_websocket(ws, token, device_name)