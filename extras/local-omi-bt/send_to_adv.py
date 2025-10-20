import json
import logging
import os
from typing import AsyncGenerator, Optional

import httpx
import websockets
from dotenv import load_dotenv
from wyoming.audio import AudioChunk

# Configuration
websocket_uri = "ws://100.83.66.30:8000/ws_pcm"
backend_url = "http://100.83.66.30:8000"
env_path = ".env"
load_dotenv(env_path)

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
DEVICE_NAME = "omi-bt"  # Device name for client identification

logger = logging.getLogger(__name__)


async def get_jwt_token(username: str, password: str) -> Optional[str]:
    """
    Get JWT token from backend using username and password.

    Args:
        username: User email/username
        password: User password

    Returns:
        JWT token string or None if authentication failed
    """
    try:
        logger.info(f"Authenticating with backend as: {username}")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{backend_url}/auth/jwt/login",
                data={'username': username, 'password': password},
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )

        if response.status_code == 200:
            auth_data = response.json()
            token = auth_data.get('access_token')

            if token:
                logger.info("JWT authentication successful")
                return token
            else:
                logger.error("No access token in response")
                return None
        else:
            error_msg = "Invalid credentials"
            try:
                error_data = response.json()
                error_msg = error_data.get('detail', error_msg)
            except:
                pass
            logger.error(f"Authentication failed: {error_msg}")
            return None

    except httpx.TimeoutException:
        logger.error("Authentication request timed out")
        return None
    except httpx.RequestError as e:
        logger.error(f"Authentication request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected authentication error: {e}")
        return None


async def stream_to_backend(stream: AsyncGenerator[AudioChunk, None]):
    """
    Stream audio to backend using Wyoming protocol with JWT authentication.

    Args:
        stream: AsyncGenerator yielding AudioChunk objects
    """
    # Get JWT token for authentication
    token = await get_jwt_token(ADMIN_EMAIL, ADMIN_PASSWORD)
    if not token:
        logger.error("Failed to get JWT token, cannot stream audio")
        return

    # Connect with JWT token as query parameter
    uri_with_token = f"{websocket_uri}?token={token}&device_name={DEVICE_NAME}"

    logger.info(f"Connecting to WebSocket: {websocket_uri}")
    async with websockets.connect(uri_with_token) as websocket:
        # Wait for ready message from backend
        ready_msg = await websocket.recv()
        logger.info(f"Backend ready: {ready_msg}")

        # Send audio-start event (Wyoming protocol)
        audio_start = {
            "type": "audio-start",
            "data": {
                "rate": 16000,
                "width": 2,
                "channels": 1,
                "mode": "streaming"  # or "batch"
            },
            "payload_length": None
        }
        await websocket.send(json.dumps(audio_start) + '\n')
        logger.info("Sent audio-start event")

        # Stream audio chunks
        chunk_count = 0
        async for chunk in stream:
            chunk_count += 1
            audio_data = chunk.audio  # Extract bytes from AudioChunk

            # Send audio-chunk header (Wyoming protocol)
            audio_chunk_header = {
                "type": "audio-chunk",
                "data": {
                    "rate": chunk.rate,
                    "width": chunk.width,
                    "channels": chunk.channels
                },
                "payload_length": len(audio_data)
            }
            await websocket.send(json.dumps(audio_chunk_header) + '\n')

            # Send audio data as binary
            await websocket.send(audio_data)

            if chunk_count % 100 == 0:
                logger.info(f"Sent {chunk_count} chunks")

        # Send audio-stop event
        audio_stop = {
            "type": "audio-stop",
            "data": {},
            "payload_length": None
        }
        await websocket.send(json.dumps(audio_stop) + '\n')
        logger.info(f"Sent audio-stop event. Total chunks: {chunk_count}")
