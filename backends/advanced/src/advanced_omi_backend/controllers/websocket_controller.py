
"""
WebSocket controller for Friend-Lite backend.

This module handles WebSocket connections for audio streaming.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import time
import uuid
from functools import partial
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect, Query
from friend_lite.decoder import OmiOpusDecoder

from advanced_omi_backend.auth import websocket_auth
from advanced_omi_backend.client_manager import generate_client_id, get_client_manager
from advanced_omi_backend.constants import OMI_CHANNELS, OMI_SAMPLE_RATE, OMI_SAMPLE_WIDTH
from advanced_omi_backend.audio_utils import process_audio_chunk
from advanced_omi_backend.services.audio_stream import AudioStreamProducer
from advanced_omi_backend.services.audio_stream.producer import get_audio_stream_producer

# Thread pool executors for audio decoding
_DEC_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=os.cpu_count() or 4,
    thread_name_prefix="opus_io",
)

# Logging setup
logger = logging.getLogger(__name__)
application_logger = logging.getLogger("audio_processing")

# Track pending WebSocket connections to prevent race conditions
pending_connections: set[str] = set()


async def parse_wyoming_protocol(ws: WebSocket) -> tuple[dict, Optional[bytes]]:
    """Parse Wyoming protocol: JSON header line followed by optional binary payload.

    Returns:
        Tuple of (header_dict, payload_bytes or None)
    """
    # Read data from WebSocket
    logger.debug(f"parse_wyoming_protocol: About to call ws.receive()")
    message = await ws.receive()
    logger.debug(f"parse_wyoming_protocol: Received message with keys: {message.keys() if message else 'None'}")

    # Handle WebSocket close frame
    if "type" in message and message["type"] == "websocket.disconnect":
        # This is a normal WebSocket close event
        code = message.get("code", 1000)
        reason = message.get("reason", "")
        logger.info(f"📴 WebSocket disconnect received in parse_wyoming_protocol. Code: {code}, Reason: {reason}")
        raise WebSocketDisconnect(code=code, reason=reason)

    # Handle text message (JSON header)
    if "text" in message:
        header_text = message["text"]
        # Wyoming protocol uses newline-terminated JSON
        if not header_text.endswith("\n"):
            header_text += "\n"

        # Parse JSON header
        json_line = header_text.strip()
        header = json.loads(json_line)

        # If payload is expected, read binary data
        payload = None
        payload_length = header.get("payload_length")
        if payload_length is not None and payload_length > 0:
            payload_msg = await ws.receive()
            if "bytes" in payload_msg:
                payload = payload_msg["bytes"]
            else:
                logger.warning(f"Expected binary payload but got: {payload_msg.keys()}")

        return header, payload

    # Handle binary message (invalid - Wyoming protocol requires JSONL headers)
    elif "bytes" in message:
        raise ValueError(
            "Raw binary messages not supported - Wyoming protocol requires JSONL headers"
        )

    else:
        raise ValueError(f"Unexpected WebSocket message type: {message.keys()}")


async def create_client_state(client_id: str, user, device_name: Optional[str] = None):
    """Create and register a new client state."""
    # Get client manager and repository
    client_manager = get_client_manager()
    from advanced_omi_backend.database import AudioChunksRepository
    from motor.motor_asyncio import AsyncIOMotorClient
    
    # MongoDB Configuration
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
    mongo_client = AsyncIOMotorClient(MONGODB_URI)
    db = mongo_client.get_default_database("friend-lite")
    chunks_col = db["audio_chunks"]
    
    # Initialize repository
    ac_repository = AudioChunksRepository(chunks_col)
    
    # Directory where WAV chunks are written
    from pathlib import Path
    CHUNK_DIR = Path("./audio_chunks")  # This will be mounted to ./data/audio_chunks by Docker
    
    # Use ClientManager for atomic client creation and registration
    client_state = client_manager.create_client(
        client_id, ac_repository, CHUNK_DIR, user.user_id, user.email
    )

    # Also track in persistent mapping (for database queries)
    from advanced_omi_backend.client_manager import track_client_user_relationship
    track_client_user_relationship(client_id, user.user_id)

    # Register client in user model (persistent)
    from advanced_omi_backend.users import register_client_to_user
    await register_client_to_user(user, client_id, device_name)

    return client_state


async def cleanup_client_state(client_id: str):
    """Clean up and remove client state."""
    # Use ClientManager for atomic client removal with cleanup
    client_manager = get_client_manager()
    removed = await client_manager.remove_client_with_cleanup(client_id)

    if removed:
        logger.info(f"Client {client_id} cleaned up successfully")
    else:
        logger.warning(f"Client {client_id} was not found for cleanup")


# Shared helper functions for WebSocket handlers
async def _setup_websocket_connection(
    ws: WebSocket,
    token: Optional[str],
    device_name: Optional[str],
    pending_client_id: str,
    connection_type: str
) -> tuple[Optional[str], Optional[object], Optional[object]]:
    """
    Setup WebSocket connection: accept, authenticate, create client state.

    Args:
        ws: WebSocket connection
        token: JWT authentication token
        device_name: Optional device name for client ID
        pending_client_id: Temporary tracking ID
        connection_type: "OMI" or "PCM" for logging

    Returns:
        tuple: (client_id, client_state, user) or (None, None, None) on failure
    """
    # Accept WebSocket first (required before any send/close operations)
    await ws.accept()

    # Authenticate user after accepting connection
    user = await websocket_auth(ws, token)
    if not user:
        await ws.close(code=1008, reason="Authentication required")
        return None, None, None

    # Generate proper client_id using user and device_name
    client_id = generate_client_id(user, device_name)

    # Remove from pending now that we have real client_id
    pending_connections.discard(pending_client_id)
    application_logger.info(
        f"🔌 {connection_type} WebSocket connection accepted - User: {user.user_id} ({user.email}), Client: {client_id}"
    )

    # Send ready message for PCM clients
    if connection_type == "PCM":
        try:
            ready_msg = json.dumps({"type": "ready", "message": "WebSocket connection established"}) + "\n"
            await ws.send_text(ready_msg)
            application_logger.debug(f"✅ Sent ready message to {client_id}")
        except Exception as e:
            application_logger.error(f"Failed to send ready message to {client_id}: {e}")

    # Create client state
    client_state = await create_client_state(client_id, user, device_name)

    return client_id, client_state, user


async def _initialize_streaming_session(
    client_state,
    audio_stream_producer,
    user_id: str,
    user_email: str,
    client_id: str,
    audio_format: dict
) -> None:
    """
    Initialize streaming session with Redis and enqueue processing jobs.

    Args:
        client_state: Client state object
        audio_stream_producer: Audio stream producer instance
        user_id: User ID
        user_email: User email
        client_id: Client ID
        audio_format: Audio format dict from audio-start event
    """
    if hasattr(client_state, 'stream_session_id'):
        application_logger.debug(f"Session already initialized for {client_id}")
        return

    # Initialize stream session
    client_state.stream_session_id = str(uuid.uuid4())
    client_state.stream_chunk_count = 0
    client_state.stream_audio_format = audio_format
    application_logger.info(f"🆔 Created stream session: {client_state.stream_session_id}")

    # Initialize session tracking in Redis
    await audio_stream_producer.init_session(
        session_id=client_state.stream_session_id,
        user_id=user_id,
        client_id=client_id,
        mode="streaming",
        provider="deepgram"
    )

    # Enqueue speech detection job
    from advanced_omi_backend.workers.transcription_jobs import stream_speech_detection_job
    from advanced_omi_backend.workers.audio_jobs import audio_streaming_persistence_job
    from advanced_omi_backend.controllers.queue_controller import transcription_queue, JOB_RESULT_TTL

    speech_job = transcription_queue.enqueue(
        stream_speech_detection_job,
        client_state.stream_session_id,
        user_id,
        user_email,
        client_id,
        job_timeout=3600,  # 1 hour for long recordings
        result_ttl=JOB_RESULT_TTL,
        job_id=f"speech-detect_{client_state.stream_session_id[:12]}",
        description=f"Stream speech detection for {client_state.stream_session_id[:12]}"
    )
    client_state.speech_detection_job_id = speech_job.id
    application_logger.info(f"✅ Enqueued speech detection job {speech_job.id}")

    # Enqueue audio persistence job in parallel
    audio_job = transcription_queue.enqueue(
        audio_streaming_persistence_job,
        client_state.stream_session_id,
        user_id,
        user_email,
        client_id,
        job_timeout=3600,  # 1 hour for long recordings
        result_ttl=JOB_RESULT_TTL,
        job_id=f"audio-persist_{client_state.stream_session_id[:12]}",
        description=f"Audio persistence for {client_state.stream_session_id[:12]}"
    )
    client_state.audio_persistence_job_id = audio_job.id
    application_logger.info(f"✅ Enqueued audio persistence job {audio_job.id}")


async def _finalize_streaming_session(
    client_state,
    audio_stream_producer,
    user_id: str,
    user_email: str,
    client_id: str
) -> None:
    """
    Finalize streaming session: flush buffer, signal workers, enqueue finalize job, cleanup.

    Args:
        client_state: Client state object
        audio_stream_producer: Audio stream producer instance
        user_id: User ID
        user_email: User email
        client_id: Client ID
    """
    if not hasattr(client_state, 'stream_session_id'):
        application_logger.debug(f"No active session to finalize for {client_id}")
        return

    session_id = client_state.stream_session_id

    try:
        # Flush any remaining buffered audio
        audio_format = getattr(client_state, 'stream_audio_format', {})
        await audio_stream_producer.flush_session_buffer(
            session_id=session_id,
            sample_rate=audio_format.get("rate", 16000),
            channels=audio_format.get("channels", 1),
            sample_width=audio_format.get("width", 2)
        )

        # Send end-of-session signal to workers
        await audio_stream_producer.send_session_end_signal(session_id)

        # Mark session as finalizing
        await audio_stream_producer.finalize_session(session_id)

        # Enqueue finalize job
        from advanced_omi_backend.workers.transcription_jobs import enqueue_streaming_finalization

        finalize_job = enqueue_streaming_finalization(
            session_id=session_id,
            user_id=user_id,
            user_email=user_email,
            client_id=client_id
        )

        application_logger.info(
            f"✅ Enqueued finalization job {finalize_job.id} for session {session_id[:12]}"
        )

        # Clear session state
        for attr in ['stream_session_id', 'stream_chunk_count', 'stream_audio_format',
                     'speech_detection_job_id', 'audio_persistence_job_id']:
            if hasattr(client_state, attr):
                delattr(client_state, attr)

    except Exception as finalize_error:
        application_logger.error(
            f"❌ Failed to finalize streaming session: {finalize_error}",
            exc_info=True
        )


async def _publish_audio_to_stream(
    client_state,
    audio_stream_producer,
    audio_data: bytes,
    user_id: str,
    client_id: str,
    sample_rate: int,
    channels: int,
    sample_width: int
) -> None:
    """
    Publish audio chunk to Redis Stream with chunk tracking.

    Args:
        client_state: Client state object
        audio_stream_producer: Audio stream producer instance
        audio_data: Raw PCM audio bytes
        user_id: User ID
        client_id: Client ID
        sample_rate: Sample rate (Hz)
        channels: Number of channels
        sample_width: Bytes per sample
    """
    if not hasattr(client_state, 'stream_session_id'):
        application_logger.warning(f"⚠️ Received audio chunk before session initialized for {client_id}")
        return

    # Increment chunk count and format chunk ID
    client_state.stream_chunk_count += 1
    chunk_id = f"{client_state.stream_chunk_count:05d}"

    # Publish to Redis Stream using producer
    await audio_stream_producer.add_audio_chunk(
        audio_data=audio_data,
        session_id=client_state.stream_session_id,
        chunk_id=chunk_id,
        user_id=user_id,
        client_id=client_id,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width
    )


async def handle_omi_websocket(
    ws: WebSocket,
    token: Optional[str] = None,
    device_name: Optional[str] = None,
):
    """Handle OMI WebSocket connections with Opus decoding."""
    # Generate pending client_id to track connection even if auth fails
    pending_client_id = f"pending_{uuid.uuid4()}"
    pending_connections.add(pending_client_id)

    client_id = None
    client_state = None

    try:
        # Setup connection (accept, auth, create client state)
        client_id, client_state, user = await _setup_websocket_connection(
            ws, token, device_name, pending_client_id, "OMI"
        )
        if not user:
            return

        # OMI-specific: Setup Opus decoder
        decoder = OmiOpusDecoder()
        _decode_packet = partial(decoder.decode_packet, strip_header=False)

        # Get singleton audio stream producer
        audio_stream_producer = get_audio_stream_producer()

        packet_count = 0
        total_bytes = 0

        while True:
            # Parse Wyoming protocol
            header, payload = await parse_wyoming_protocol(ws)

            if header["type"] == "audio-start":
                # Handle audio session start
                application_logger.info(f"🎙️ OMI audio session started for {client_id}")
                await _initialize_streaming_session(
                    client_state,
                    audio_stream_producer,
                    user.user_id,
                    user.email,
                    client_id,
                    header.get("data", {"rate": OMI_SAMPLE_RATE, "width": OMI_SAMPLE_WIDTH, "channels": OMI_CHANNELS})
                )

            elif header["type"] == "audio-chunk" and payload:
                packet_count += 1
                total_bytes += len(payload)

                # Log progress
                if packet_count <= 5 or packet_count % 1000 == 0:
                    application_logger.info(
                        f"🎵 Received OMI audio chunk #{packet_count}: {len(payload)} bytes"
                    )

                # OMI-specific: Decode Opus to PCM
                start_time = time.time()
                loop = asyncio.get_running_loop()
                pcm_data = await loop.run_in_executor(_DEC_IO_EXECUTOR, _decode_packet, payload)
                decode_time = time.time() - start_time

                if pcm_data:
                    if packet_count <= 5 or packet_count % 1000 == 0:
                        application_logger.debug(
                            f"🎵 Decoded OMI packet #{packet_count}: {len(payload)} bytes -> "
                            f"{len(pcm_data)} PCM bytes (took {decode_time:.3f}s)"
                        )

                    # Publish decoded PCM to Redis Stream
                    await _publish_audio_to_stream(
                        client_state,
                        audio_stream_producer,
                        pcm_data,
                        user.user_id,
                        client_id,
                        OMI_SAMPLE_RATE,
                        OMI_CHANNELS,
                        OMI_SAMPLE_WIDTH
                    )

                    # Log every 1000th packet
                    if packet_count % 1000 == 0:
                        application_logger.info(
                            f"📊 Processed {packet_count} OMI packets ({total_bytes} bytes total)"
                        )
                else:
                    # Log decode failures for first 5 packets
                    if packet_count <= 5:
                        application_logger.warning(
                            f"❌ Failed to decode OMI packet #{packet_count}: {len(payload)} bytes"
                        )

            elif header["type"] == "audio-stop":
                # Handle audio session stop
                application_logger.info(
                    f"🛑 OMI audio session stopped for {client_id} - "
                    f"Total chunks: {packet_count}, Total bytes: {total_bytes}"
                )

                # Finalize session
                await _finalize_streaming_session(
                    client_state,
                    audio_stream_producer,
                    user.user_id,
                    user.email,
                    client_id
                )

                # Reset counters for next session
                packet_count = 0
                total_bytes = 0

            else:
                # Unknown event type
                application_logger.debug(
                    f"Ignoring Wyoming event type '{header['type']}' for OMI client {client_id}"
                )

    except WebSocketDisconnect:
        application_logger.info(
            f"🔌 WebSocket disconnected - Client: {client_id}, Packets: {packet_count}, Total bytes: {total_bytes}"
        )
    except Exception as e:
        application_logger.error(f"❌ WebSocket error for client {client_id}: {e}", exc_info=True)
    finally:
        # Clean up pending connection tracking
        pending_connections.discard(pending_client_id)

        # Ensure cleanup happens even if client_id is None
        if client_id:
            try:
                # Clean up client state
                await cleanup_client_state(client_id)
            except Exception as cleanup_error:
                application_logger.error(
                    f"Error during cleanup for client {client_id}: {cleanup_error}", exc_info=True
                )


async def handle_pcm_websocket(
    ws: WebSocket,
    token: Optional[str] = None,
    device_name: Optional[str] = None
):
    """Handle PCM WebSocket connections with batch and streaming mode support."""
    # Generate pending client_id to track connection even if auth fails
    pending_client_id = f"pending_{uuid.uuid4()}"
    pending_connections.add(pending_client_id)

    client_id = None
    client_state = None

    try:
        # Setup connection (accept, auth, create client state)
        client_id, client_state, user = await _setup_websocket_connection(
            ws, token, device_name, pending_client_id, "PCM"
        )
        if not user:
            return

        # Get singleton audio stream producer
        audio_stream_producer = get_audio_stream_producer()

        packet_count = 0
        total_bytes = 0
        audio_streaming = False  # Track if audio session is active

        while True:
            try:
                if not audio_streaming:
                    # Control message mode - parse Wyoming protocol
                    application_logger.debug(f"🔄 Control mode for {client_id}, WebSocket state: {ws.client_state if hasattr(ws, 'client_state') else 'unknown'}")
                    application_logger.debug(f"📨 About to receive control message for {client_id}")
                    header, payload = await parse_wyoming_protocol(ws)
                    application_logger.debug(f"✅ Received message type: {header.get('type')} for {client_id}")

                    if header["type"] == "audio-start":
                        application_logger.debug(f"🎙️ Processing audio-start for {client_id}")
                        # Handle audio session start
                        audio_streaming = True
                        audio_format = header.get("data", {})

                        # Extract recording mode (batch vs streaming)
                        recording_mode = audio_format.get("mode", "batch")
                        client_state.recording_mode = recording_mode  # Store mode in client state

                        application_logger.info(
                            f"🎙️ Audio session started for {client_id} - "
                            f"Format: {audio_format.get('rate')}Hz, "
                            f"{audio_format.get('width')}bytes, "
                            f"{audio_format.get('channels')}ch, "
                            f"Mode: {recording_mode}"
                        )

                        # Log mode selection (no transcription manager needed - both modes use job queue now)
                        if recording_mode == "batch":
                            application_logger.info(f"📦 Batch mode enabled for {client_id} - will accumulate and process on stop")
                        else:
                            application_logger.info(f"🎵 Streaming mode enabled for {client_id} - using Redis Streams")

                        application_logger.info(f"🎵 Switching to audio streaming mode for {client_id}")
                        continue  # Continue to audio streaming mode
                    
                    elif header["type"] == "ping":
                        # Handle keepalive ping from frontend
                        application_logger.debug(f"🏓 Received ping from {client_id}")
                        continue
                    
                    else:
                        # Unknown control message type
                        application_logger.debug(
                            f"Ignoring Wyoming control event type '{header['type']}' for {client_id}"
                        )
                        continue
                        
                else:
                    # Audio streaming mode - receive raw bytes (like speaker recognition)
                    application_logger.debug(f"🎵 Audio streaming mode for {client_id} - waiting for audio data")
                    
                    try:
                        # Receive raw audio bytes or check for control messages
                        message = await ws.receive()
                        
                        
                        # Check if it's a disconnect
                        if "type" in message and message["type"] == "websocket.disconnect":
                            code = message.get("code", 1000)
                            reason = message.get("reason", "")
                            application_logger.info(f"🔌 WebSocket disconnect during audio streaming for {client_id}. Code: {code}, Reason: {reason}")
                            break
                        
                        # Check if it's a text message (control message like audio-stop)
                        if "text" in message:
                            try:
                                control_header = json.loads(message["text"].strip())
                                if control_header.get("type") == "audio-stop":
                                    application_logger.info(f"🛑 Audio session stopped for {client_id}")
                                    audio_streaming = False

                                    # Check if this was a streaming mode session
                                    recording_mode = getattr(client_state, 'recording_mode', 'batch')

                                    if recording_mode == "streaming":
                                        # Streaming mode: finalize session using helper
                                        await _finalize_streaming_session(
                                            client_state,
                                            audio_stream_producer,
                                            user.user_id,
                                            user.email,
                                            client_id
                                        )
                                    else:
                                        # Batch mode: write complete file and enqueue job (like upload)
                                        if hasattr(client_state, 'batch_audio_chunks') and client_state.batch_audio_chunks:
                                            try:
                                                from datetime import UTC, datetime
                                                from advanced_omi_backend.audio_utils import write_audio_file
                                                from advanced_omi_backend.models.conversation import Conversation
                                                from advanced_omi_backend.workers.transcription_jobs import enqueue_transcript_processing
                                                from advanced_omi_backend.workers.memory_jobs import enqueue_memory_processing
                                                from advanced_omi_backend.models.job import JobPriority

                                                # Combine all chunks
                                                complete_audio = b''.join(client_state.batch_audio_chunks)
                                                application_logger.info(
                                                    f"📦 Batch mode: Combined {len(client_state.batch_audio_chunks)} chunks into {len(complete_audio)} bytes"
                                                )

                                                # Generate audio UUID and timestamp
                                                audio_uuid = str(uuid.uuid4())
                                                timestamp = int(time.time() * 1000)

                                                # Write audio file and create AudioFile entry
                                                wav_filename, file_path, duration = await write_audio_file(
                                                    raw_audio_data=complete_audio,
                                                    audio_uuid=audio_uuid,
                                                    client_id=client_id,
                                                    user_id=user.user_id,
                                                    user_email=user.email,
                                                    timestamp=timestamp,
                                                    validate=False  # PCM data, not WAV
                                                )

                                                application_logger.info(
                                                    f"✅ Batch mode: Wrote audio file {wav_filename} ({duration:.1f}s)"
                                                )

                                                # Create conversation immediately for batch audio
                                                conversation_id = str(uuid.uuid4())
                                                version_id = str(uuid.uuid4())

                                                conversation = Conversation(
                                                    conversation_id=conversation_id,
                                                    audio_uuid=audio_uuid,
                                                    user_id=user.user_id,
                                                    client_id=client_id,
                                                    title="Batch Recording",
                                                    summary="Processing batch audio...",
                                                    transcript_versions=[],
                                                    active_transcript_version=None,
                                                    memory_versions=[],
                                                    active_memory_version=None,
                                                    created_at=datetime.now(UTC),
                                                    updated_at=datetime.now(UTC),
                                                    session_start=datetime.fromtimestamp(timestamp / 1000, tz=UTC),
                                                    session_end=datetime.fromtimestamp(timestamp / 1000, tz=UTC),
                                                    duration_seconds=duration,
                                                    speech_start_time=0.0,
                                                    speech_end_time=duration,
                                                    speaker_names={},
                                                    action_items=[]
                                                )
                                                await conversation.insert()

                                                application_logger.info(f"📝 Batch mode: Created conversation {conversation_id}")

                                                # Enqueue transcript processing job
                                                transcript_job = enqueue_transcript_processing(
                                                    conversation_id=conversation_id,
                                                    audio_uuid=audio_uuid,
                                                    audio_path=file_path,
                                                    version_id=version_id,
                                                    user_id=user.user_id,
                                                    priority=JobPriority.HIGH,
                                                    trigger="batch_audio"
                                                )

                                                # Enqueue memory processing job
                                                memory_job = enqueue_memory_processing(
                                                    client_id=client_id,
                                                    user_id=user.user_id,
                                                    user_email=user.email,
                                                    conversation_id=conversation_id,
                                                    priority=JobPriority.NORMAL
                                                )

                                                application_logger.info(
                                                    f"✅ Batch mode: Enqueued jobs for {conversation_id} - "
                                                    f"transcript: {transcript_job.id}, memory: {memory_job.id}"
                                                )

                                                # Clear accumulated chunks
                                                client_state.batch_audio_chunks = []

                                            except Exception as batch_error:
                                                application_logger.error(
                                                    f"❌ Batch mode processing failed: {batch_error}",
                                                    exc_info=True
                                                )
                                        else:
                                            application_logger.warning(f"⚠️ Batch mode: No audio chunks accumulated for {client_id}")

                                    # Reset counters for next session
                                    packet_count = 0
                                    total_bytes = 0
                                    continue
                                elif control_header.get("type") == "ping":
                                    application_logger.debug(f"🏓 Received ping during streaming from {client_id}")
                                    continue
                                elif control_header.get("type") == "audio-start":
                                    # Handle duplicate audio-start messages gracefully (idempotent behavior)
                                    application_logger.info(f"🔄 Ignoring duplicate audio-start message during streaming for {client_id}")
                                    continue
                                elif control_header.get("type") == "audio-chunk":
                                    # Handle Wyoming protocol audio-chunk with binary payload
                                    payload_length = control_header.get("payload_length")
                                    if payload_length and payload_length > 0:
                                        # Receive the binary audio data
                                        payload_msg = await ws.receive()
                                        if "bytes" in payload_msg:
                                            audio_data = payload_msg["bytes"]
                                            packet_count += 1
                                            total_bytes += len(audio_data)
                                            
                                            application_logger.debug(f"🎵 Received audio chunk #{packet_count}: {len(audio_data)} bytes")

                                            # Route based on recording mode
                                            recording_mode = getattr(client_state, 'recording_mode', 'batch')
                                            audio_format = control_header.get("data", {})

                                            if recording_mode == "streaming":
                                                # Streaming mode: initialize session if needed, then publish
                                                try:
                                                    # Initialize streaming session (idempotent)
                                                    if not hasattr(client_state, 'stream_session_id'):
                                                        await _initialize_streaming_session(
                                                            client_state,
                                                            audio_stream_producer,
                                                            user.user_id,
                                                            user.email,
                                                            client_id,
                                                            audio_format
                                                        )

                                                    # Publish audio chunk to Redis Stream
                                                    await _publish_audio_to_stream(
                                                        client_state,
                                                        audio_stream_producer,
                                                        audio_data,
                                                        user.user_id,
                                                        client_id,
                                                        audio_format.get("rate", 16000),
                                                        audio_format.get("channels", 1),
                                                        audio_format.get("width", 2)
                                                    )

                                                except Exception as stream_error:
                                                    application_logger.error(
                                                        f"❌ Failed to publish to Redis Stream: {stream_error}",
                                                        exc_info=True
                                                    )
                                                    # Don't break - continue trying to process chunks
                                            else:
                                                # Batch mode: accumulate audio in memory
                                                if not hasattr(client_state, 'batch_audio_chunks'):
                                                    client_state.batch_audio_chunks = []
                                                    client_state.batch_audio_format = audio_format
                                                    application_logger.info(f"📦 Started batch audio accumulation for {client_id}")

                                                client_state.batch_audio_chunks.append(audio_data)
                                                application_logger.debug(
                                                    f"📦 Accumulated chunk #{len(client_state.batch_audio_chunks)} ({len(audio_data)} bytes) for {client_id}"
                                                )
                                        else:
                                            application_logger.warning(f"Expected binary payload for audio-chunk, got: {payload_msg.keys()}")
                                    else:
                                        application_logger.warning(f"audio-chunk missing payload_length: {payload_length}")
                                    continue
                                else:
                                    application_logger.warning(f"Unknown control message during streaming: {control_header.get('type')}")
                                    continue
                                            
                            except json.JSONDecodeError:
                                application_logger.warning(f"Invalid control message during streaming for {client_id}")
                                continue
                        
                        # Check if it's binary data (raw audio without Wyoming protocol)
                        elif "bytes" in message:
                            # Raw binary audio data (legacy support)
                            audio_data = message["bytes"]
                            packet_count += 1
                            total_bytes += len(audio_data)

                            application_logger.debug(f"🎵 Received raw audio chunk #{packet_count}: {len(audio_data)} bytes")

                            # Route based on recording mode
                            recording_mode = getattr(client_state, 'recording_mode', 'batch')

                            if recording_mode == "streaming":
                                # Streaming mode: publish to Redis Streams (legacy raw binary support)
                                try:
                                    # Initialize streaming session if needed (fallback for clients not sending audio-start)
                                    if not hasattr(client_state, 'stream_session_id'):
                                        default_format = {"rate": 16000, "width": 2, "channels": 1}
                                        await _initialize_streaming_session(
                                            client_state,
                                            audio_stream_producer,
                                            user.user_id,
                                            user.email,
                                            client_id,
                                            default_format
                                        )

                                    # Publish audio chunk to Redis Stream
                                    await _publish_audio_to_stream(
                                        client_state,
                                        audio_stream_producer,
                                        audio_data,
                                        user.user_id,
                                        client_id,
                                        16000,  # Default sample rate
                                        1,      # Default channels
                                        2       # Default sample width
                                    )
                                except Exception as stream_error:
                                    application_logger.error(
                                        f"❌ Failed to publish to Redis Stream: {stream_error}",
                                        exc_info=True
                                    )
                            else:
                                # Batch mode: use existing flow
                                if not hasattr(client_state, 'batch_audio_chunks'):
                                    client_state.batch_audio_chunks = []
                                    application_logger.info(f"📦 Started batch audio accumulation for {client_id}")
                                
                                client_state.batch_audio_chunks.append(audio_data)
                                application_logger.debug(
                                    f"📦 Accumulated chunk #{len(client_state.batch_audio_chunks)} ({len(audio_data)} bytes) for {client_id}"
                                )
                        
                        else:
                            application_logger.warning(f"Unexpected message format in streaming mode: {message.keys()}")
                            continue
                            
                    except Exception as streaming_error:
                        application_logger.error(f"Error in audio streaming mode: {streaming_error}")
                        if "disconnect" in str(streaming_error).lower():
                            break
                        continue

            except WebSocketDisconnect as e:
                application_logger.info(
                    f"🔌 WebSocket disconnected during message processing for {client_id}. "
                    f"Code: {e.code}, Reason: {e.reason}"
                )
                break  # Exit the loop on disconnect
            except json.JSONDecodeError as e:
                application_logger.error(
                    f"❌ JSON decode error in Wyoming protocol for {client_id}: {e}"
                )
                continue  # Skip this message but don't disconnect
            except ValueError as e:
                application_logger.error(
                    f"❌ Protocol error for {client_id}: {e}"
                )
                continue  # Skip this message but don't disconnect
            except RuntimeError as e:
                # Handle "Cannot call receive once a disconnect message has been received"
                if "disconnect" in str(e).lower():
                    application_logger.info(
                        f"🔌 WebSocket already disconnected for {client_id}: {e}"
                    )
                    break  # Exit the loop on disconnect
                else:
                    application_logger.error(
                        f"❌ Runtime error for {client_id}: {e}", exc_info=True
                    )
                    continue
            except Exception as e:
                application_logger.error(
                    f"❌ Unexpected error processing message for {client_id}: {e}", exc_info=True
                )
                # Check if it's a connection-related error
                error_msg = str(e).lower()
                if "disconnect" in error_msg or "closed" in error_msg or "receive" in error_msg:
                    application_logger.info(
                        f"🔌 Connection issue detected for {client_id}, exiting loop"
                    )
                    break
                else:
                    continue  # Skip this message for other errors
                
    except WebSocketDisconnect:
        application_logger.info(
            f"🔌 PCM WebSocket disconnected - Client: {client_id}, Packets: {packet_count}, Total bytes: {total_bytes}"
        )
    except Exception as e:
        application_logger.error(
            f"❌ PCM WebSocket error for client {client_id}: {e}", exc_info=True
        )
    finally:
        # Clean up pending connection tracking
        pending_connections.discard(pending_client_id)

        # Ensure cleanup happens even if client_id is None
        if client_id:
            try:
                # Clean up client state
                await cleanup_client_state(client_id)
            except Exception as cleanup_error:
                application_logger.error(
                    f"Error during cleanup for client {client_id}: {cleanup_error}", exc_info=True
                )
