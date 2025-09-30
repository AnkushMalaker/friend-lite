"""Unified WebSocket handlers for Wyoming protocol with job tracking.

This module demonstrates how WebSocket handlers integrate with the unified
pipeline architecture using job tracking.
"""

import json
import logging
from typing import Optional

from fastapi import WebSocket

from advanced_omi_backend.client import ClientState
from advanced_omi_backend.client_manager import get_client_manager
from advanced_omi_backend.processors import get_processor_manager

logger = logging.getLogger(__name__)


async def handle_audio_start(
    websocket: WebSocket,
    client_state: ClientState,
    message: dict
) -> None:
    """Handle audio-start event."""
    # Extract audio configuration from message
    data = message.get("data", {})
    sample_rate = data.get("rate", 16000)
    channels = data.get("channels", 1)
    sample_width = data.get("width", 2)

    # Update client state with audio configuration
    client_state.sample_rate = sample_rate
    client_state.channels = channels
    client_state.sample_width = sample_width

    # Start new audio session
    audio_uuid = client_state.start_audio_session()

    logger.info(f"🎙️ Audio session started: {audio_uuid} for client {client_state.client_id}")
    logger.debug(f"    Audio config: {sample_rate}Hz, {channels} channels, {sample_width} bytes/sample")


async def handle_audio_chunk(
    websocket: WebSocket,
    client_state: ClientState,
    audio_data: bytes
) -> None:
    """Handle incoming audio chunk."""
    # Add audio chunk to buffer
    client_state.add_audio_chunk(audio_data)

    # Log periodically to avoid spam
    if len(client_state.audio_buffer) % 100 == 0:
        total_bytes = sum(len(chunk) for chunk in client_state.audio_buffer)
        logger.debug(f"📦 Buffered {len(client_state.audio_buffer)} chunks, {total_bytes} bytes for client {client_state.client_id}")


async def handle_audio_stop(
    websocket: WebSocket,
    client_state: ClientState,
    message: dict
) -> Optional[str]:
    """Handle audio-stop event and submit to unified pipeline.

    Returns:
        job_id if audio was submitted for processing, None otherwise
    """
    logger.info(f"🛑 Audio session stopping for client {client_state.client_id}")

    # Get processing item from client state
    processing_item = await client_state.signal_audio_end()

    if processing_item:
        # Submit to unified pipeline
        processor_manager = get_processor_manager()
        job_id = await processor_manager.submit_audio_for_processing(processing_item)

        logger.info(f"✅ WebSocket audio submitted for processing: job_id={job_id}, client={client_state.client_id}")

        # Send job_id back to client for tracking
        await websocket.send_json({
            "type": "processing-started",
            "data": {
                "job_id": job_id,
                "audio_uuid": processing_item.audio_uuid
            }
        })

        return job_id
    else:
        logger.warning(f"⚠️ No audio data to process for client {client_state.client_id}")
        return None


async def handle_client_disconnect(
    client_id: str,
    force_close: bool = False
) -> Optional[str]:
    """Handle client disconnection.

    Args:
        client_id: Client identifier
        force_close: If True, force close any active audio session

    Returns:
        job_id if audio was submitted for processing, None otherwise
    """
    logger.info(f"🔌 Handling disconnection for client {client_id}")

    client_manager = get_client_manager()
    client_state = client_manager.get_client_state(client_id)

    if not client_state:
        logger.warning(f"Client state not found for {client_id}")
        return None

    job_id = None

    # If recording is active, force end the audio session
    if client_state.is_recording and force_close:
        logger.info(f"⚠️ Force ending active recording for disconnected client {client_id}")

        processing_item = await client_state.signal_audio_end()
        if processing_item:
            processor_manager = get_processor_manager()
            job_id = await processor_manager.submit_audio_for_processing(processing_item)
            logger.info(f"✅ Disconnect triggered audio processing: job_id={job_id}")

    # Clean up client state
    await client_state.disconnect()
    client_manager.remove_client(client_id)

    return job_id


async def handle_wyoming_message(
    websocket: WebSocket,
    client_state: ClientState,
    raw_data: bytes
) -> None:
    """Process Wyoming protocol message.

    Wyoming protocol format:
    {JSON_HEADER}\n
    <optional_binary_payload>
    """
    try:
        # Try to parse as Wyoming protocol
        if b'\n' in raw_data:
            header_bytes, payload = raw_data.split(b'\n', 1)
            header = json.loads(header_bytes.decode('utf-8'))
        else:
            # Header only, no payload
            header = json.loads(raw_data.decode('utf-8'))
            payload = b''

        # Route to appropriate handler based on event type
        event_type = header.get("type", "")

        if event_type == "audio-start":
            await handle_audio_start(websocket, client_state, header)

        elif event_type == "audio-chunk":
            # Audio data is in the payload
            if payload:
                await handle_audio_chunk(websocket, client_state, payload)
            else:
                logger.warning(f"audio-chunk event without payload from {client_state.client_id}")

        elif event_type == "audio-stop":
            await handle_audio_stop(websocket, client_state, header)

        else:
            logger.debug(f"Unhandled Wyoming event type: {event_type}")

    except json.JSONDecodeError:
        # Not Wyoming protocol, might be raw audio
        # This provides backward compatibility
        if client_state.is_recording:
            await handle_audio_chunk(websocket, client_state, raw_data)
        else:
            logger.warning(f"Received raw audio without active session from {client_state.client_id}")

    except Exception as e:
        logger.error(f"Error processing Wyoming message: {e}")


# Example WebSocket endpoint integration
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Example WebSocket endpoint using unified handlers."""
    await websocket.accept()

    # Get or create client state
    client_manager = get_client_manager()
    client_state = client_manager.get_or_create_client(client_id)

    try:
        while True:
            # Receive data from WebSocket
            data = await websocket.receive_bytes()

            # Process Wyoming protocol message
            await handle_wyoming_message(websocket, client_state, data)

    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")

    finally:
        # Handle disconnection
        await handle_client_disconnect(client_id, force_close=True)