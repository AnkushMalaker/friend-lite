"""
Audio Stream Worker - Consumes audio chunks from Redis Streams.

This worker:
1. Consumes audio chunks from Redis Streams (audio:{client_id})
2. Accumulates chunks and writes them to WAV files
3. Creates database entries for audio sessions
4. Enqueues transcription jobs when audio is complete

Architecture:
- Uses Redis Streams consumer groups for distributed processing
- Maintains state per client_id for audio accumulation
- Coordinates with RQ for downstream processing (transcription, memory)
"""

import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import redis.asyncio as aioredis
from wyoming.audio import AudioChunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
audio_logger = logging.getLogger("audio_processing")


@dataclass
class AudioAccumulator:
    """Accumulates audio chunks for a client session."""
    client_id: str
    user_id: str
    user_email: str
    audio_uuid: str
    audio_rate: int
    audio_width: int
    audio_channels: int
    chunks: dict  # message_id -> audio_data for ordering
    session_timestamp: int  # Session start time
    last_message_time: int  # Last chunk received time
    total_bytes: int = 0

    def add_chunk(self, message_id: str, audio_data: bytes):
        """Add an audio chunk to the accumulator with message_id ordering."""
        self.chunks[message_id] = audio_data
        self.total_bytes += len(audio_data)
        self.last_message_time = int(time.time() * 1000)

    def get_ordered_audio(self) -> bytes:
        """Get audio chunks in message_id (Redis Stream) order."""
        # Redis Stream message IDs are in format: timestamp-sequence
        # They are lexicographically sortable
        ordered_chunks = [self.chunks[msg_id] for msg_id in sorted(self.chunks.keys())]
        return b"".join(ordered_chunks)


class AudioStreamWorker:
    """
    Worker that consumes audio chunks from Redis Streams and processes them.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        consumer_name: Optional[str] = None,
        block_ms: int = 5000,
        batch_size: int = 10
    ):
        """Initialize audio stream worker.

        Args:
            redis_url: Redis connection URL (defaults to REDIS_URL env var)
            consumer_name: Unique consumer name (auto-generated if not provided)
            block_ms: Block time in milliseconds when waiting for messages
            batch_size: Maximum messages to read per batch
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.consumer_name = consumer_name or f"worker-{uuid.uuid4().hex[:8]}"
        self.block_ms = block_ms
        self.batch_size = batch_size

        self.redis: Optional[aioredis.Redis] = None
        self.running = False

        # Audio accumulation state
        self.accumulators: Dict[str, AudioAccumulator] = {}

        # Stream and consumer group configuration
        self.audio_stream_prefix = "audio:"
        self.audio_writer = "audio-file-writer"

        # Track last processed message ID per stream (for XREAD)
        self.stream_positions: Dict[bytes, bytes] = {}  # stream_key -> last_message_id

        # Session timeout (ms) - audio sessions older than this are considered complete
        self.session_timeout_ms = 5000  # 5 seconds

        logger.info(f"Initialized AudioStreamWorker: {self.consumer_name}")

    async def connect(self):
        """Connect to Redis with connection pooling."""
        self.redis = await aioredis.from_url(
            self.redis_url,
            decode_responses=False,
            max_connections=10,  # Worker processes fewer concurrent operations
            socket_keepalive=True,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )
        logger.info(f"Connected to Redis at {self.redis_url}")

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis")

    async def _get_all_audio_streams(self) -> list:
        """Get all audio stream keys."""
        stream_keys = []
        cursor = b"0"

        while cursor:
            cursor, keys = await self.redis.scan(
                cursor, match=f"{self.audio_stream_prefix}*"
            )
            stream_keys.extend(keys)

        return stream_keys

    async def _process_audio_message(self, stream_name: bytes, message_id: bytes, message_data: dict):
        """Process a single audio message from Redis Stream.

        Args:
            stream_name: Name of the stream (e.g., b'audio:client_id')
            message_id: Message ID from Redis Stream
            message_data: Message data containing audio chunk
        """
        try:
            # Extract client_id from stream name
            client_id = stream_name.decode().split(":", 1)[1]

            # Parse message data
            user_id = message_data[b"user_id"].decode()
            user_email = message_data[b"user_email"].decode()
            audio_data = message_data[b"audio_data"]
            audio_rate = int(message_data[b"audio_rate"].decode())
            audio_width = int(message_data[b"audio_width"].decode())
            audio_channels = int(message_data[b"audio_channels"].decode())
            session_timestamp = int(message_data[b"session_timestamp"].decode())
            audio_uuid = message_data.get(b"audio_uuid", b"").decode() or None

            # Decode message_id for ordering
            msg_id_str = message_id.decode()

            audio_logger.debug(
                f"Processing audio chunk for client {client_id}: "
                f"{len(audio_data)} bytes, message_id={msg_id_str}"
            )

            # Get or create accumulator for this client
            if client_id not in self.accumulators:
                # New audio session
                final_audio_uuid = audio_uuid or uuid.uuid4().hex
                self.accumulators[client_id] = AudioAccumulator(
                    client_id=client_id,
                    user_id=user_id,
                    user_email=user_email,
                    audio_uuid=final_audio_uuid,
                    audio_rate=audio_rate,
                    audio_width=audio_width,
                    audio_channels=audio_channels,
                    chunks={},  # dict for message_id ordering
                    session_timestamp=session_timestamp,
                    last_message_time=int(time.time() * 1000)
                )
                audio_logger.info(
                    f"Started new audio session for client {client_id}, "
                    f"audio_uuid={final_audio_uuid}, session_timestamp={session_timestamp}"
                )

            # Add chunk to accumulator using message_id for ordering
            accumulator = self.accumulators[client_id]
            accumulator.add_chunk(msg_id_str, audio_data)

            audio_logger.debug(
                f"Accumulated audio for {client_id}: "
                f"{len(accumulator.chunks)} chunks, {accumulator.total_bytes} bytes"
            )

        except Exception as e:
            logger.error(f"Error processing audio message {message_id.decode()}: {e}", exc_info=True)
            raise

    async def _flush_completed_sessions(self):
        """Flush audio sessions that have timed out."""
        current_time_ms = int(time.time() * 1000)
        completed_clients = []

        for client_id, accumulator in self.accumulators.items():
            time_since_last_chunk = current_time_ms - accumulator.last_message_time

            if time_since_last_chunk >= self.session_timeout_ms:
                # Session has timed out, flush it
                audio_logger.info(
                    f"Flushing completed audio session for {client_id}: "
                    f"{len(accumulator.chunks)} chunks, {accumulator.total_bytes} bytes"
                )

                try:
                    await self._write_audio_file(accumulator)
                    completed_clients.append(client_id)
                except Exception as e:
                    logger.error(f"Error flushing audio session for {client_id}: {e}", exc_info=True)

        # Remove completed sessions
        for client_id in completed_clients:
            del self.accumulators[client_id]

    async def _write_audio_file(self, accumulator: AudioAccumulator):
        """Write accumulated audio chunks to WAV file and create database entry.

        Args:
            accumulator: Audio accumulator with chunks to write
        """
        from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
        from advanced_omi_backend.config import CHUNK_DIR
        from advanced_omi_backend.models.audio_file import AudioFile

        try:
            # Ensure directory exists
            chunk_dir = CHUNK_DIR
            chunk_dir.mkdir(parents=True, exist_ok=True)

            # Create filename
            wav_filename = (
                f"{accumulator.session_timestamp}_{accumulator.client_id}_"
                f"{accumulator.audio_uuid}.wav"
            )
            file_path = chunk_dir / wav_filename

            # Combine all audio chunks in message_id order
            combined_audio = accumulator.get_ordered_audio()
            audio_logger.info(
                f"Combined {len(accumulator.chunks)} chunks in message_id order, total: {len(combined_audio)} bytes"
            )

            # Create file sink and write audio
            sink = LocalFileSink(
                file_path=str(file_path),
                sample_rate=accumulator.audio_rate,
                channels=accumulator.audio_channels,
                sample_width=accumulator.audio_width
            )

            await sink.open()
            audio_chunk = AudioChunk(
                rate=accumulator.audio_rate,
                width=accumulator.audio_width,
                channels=accumulator.audio_channels,
                audio=combined_audio
            )
            await sink.write(audio_chunk)
            await sink.close()

            audio_logger.info(
                f"✅ Wrote audio file: {wav_filename} "
                f"({accumulator.total_bytes} bytes, {len(accumulator.chunks)} chunks)"
            )

            # Create AudioFile database entry using Beanie model
            audio_file = AudioFile(
                audio_uuid=accumulator.audio_uuid,
                audio_path=wav_filename,
                client_id=accumulator.client_id,
                timestamp=accumulator.session_timestamp,
                user_id=accumulator.user_id,
                user_email=accumulator.user_email,
                has_speech=False,  # Will be updated by transcription
                speech_analysis={}
            )
            await audio_file.insert()

            audio_logger.info(f"✅ Created AudioFile entry for {accumulator.audio_uuid}")

            # Enqueue initial transcription job
            from advanced_omi_backend.rq_queue import enqueue_initial_transcription
            from advanced_omi_backend.models.job import JobPriority

            job = enqueue_initial_transcription(
                audio_uuid=accumulator.audio_uuid,
                audio_path=wav_filename,
                client_id=accumulator.client_id,
                user_id=accumulator.user_id,
                user_email=accumulator.user_email,
                priority=JobPriority.NORMAL
            )

            audio_logger.info(
                f"✅ Enqueued initial transcription job {job.id} for audio {accumulator.audio_uuid}"
            )

        except Exception as e:
            logger.error(f"Error writing audio file for {accumulator.client_id}: {e}", exc_info=True)
            raise

    async def _get_worker_count(self) -> int:
        """Get the number of active audio workers."""
        # Store worker registration in Redis with TTL
        worker_key = f"audio_workers:{self.consumer_name}"
        await self.redis.setex(worker_key, 30, "active")  # 30 second TTL

        # Get all active workers
        cursor = b"0"
        worker_count = 0
        while cursor:
            cursor, keys = await self.redis.scan(cursor, match="audio_workers:*")
            worker_count += len(keys)

        return max(worker_count, 1)  # At least 1 worker

    def _should_process_stream(self, stream_key: bytes, worker_count: int) -> bool:
        """Determine if this worker should process a given stream using consistent hashing.

        This ensures each client stream is processed by exactly one worker.
        """
        # Extract client_id from stream key (format: b"audio:client_id")
        client_id = stream_key.decode().split(":", 1)[1]

        # Use hash of client_id modulo worker_count to assign to a specific worker slot
        client_slot = hash(client_id) % worker_count

        # Use hash of worker name modulo worker_count to get this worker's slot
        worker_slot = hash(self.consumer_name) % worker_count

        # This worker handles this stream if the slots match
        should_process = client_slot == worker_slot

        if should_process:
            audio_logger.debug(
                f"Worker {self.consumer_name} (slot {worker_slot}/{worker_count}) "
                f"will process {client_id} (slot {client_slot})"
            )

        return should_process

    async def _consume_loop(self):
        """Main consumption loop."""
        logger.info(f"Starting audio stream consumer: {self.consumer_name}")

        last_flush_time = time.time()
        flush_interval = 1.0  # Check for completed sessions every second

        while self.running:
            try:
                # Get worker count for consistent hashing
                worker_count = await self._get_worker_count()

                # Get all audio streams
                all_stream_keys = await self._get_all_audio_streams()

                if not all_stream_keys:
                    audio_logger.debug("No audio streams found, waiting...")
                    await asyncio.sleep(1)
                    continue

                # Filter to only streams this worker should process
                # This ensures each client's stream is processed by exactly one worker
                stream_keys = [key for key in all_stream_keys if self._should_process_stream(key, worker_count)]

                if not stream_keys:
                    if all_stream_keys:  # Only log if there are streams but none assigned
                        audio_logger.debug(f"No streams assigned to {self.consumer_name}, {len(all_stream_keys)} total streams, {worker_count} workers")
                    await asyncio.sleep(1)
                    continue

                # Log assigned streams (only when they change)
                if stream_keys:
                    client_ids = [key.decode().split(':', 1)[1] for key in stream_keys]
                    audio_logger.info(f"Worker {self.consumer_name} processing {len(stream_keys)} streams: {client_ids[:3]}{'...' if len(client_ids) > 3 else ''}")

                # Use XREAD (not XREADGROUP) to independently consume from assigned streams
                # Each worker tracks its own position per stream
                # Format: {stream_key: last_message_id}
                streams_dict = {}
                for stream_key in stream_keys:
                    # Get last processed message ID for this stream
                    # If never processed, use $ to only get new messages
                    last_id = self.stream_positions.get(stream_key, b"$")
                    streams_dict[stream_key] = last_id

                # Use XREAD to get new messages from assigned streams
                messages = await self.redis.xread(
                    streams_dict,
                    count=self.batch_size,
                    block=self.block_ms
                )

                # Process messages
                if messages:
                    for stream_name, stream_messages in messages:
                        for message_id, message_data in stream_messages:
                            try:
                                # Process message
                                await self._process_audio_message(
                                    stream_name, message_id, message_data
                                )

                                # Update stream position to this message ID
                                self.stream_positions[stream_name] = message_id

                            except Exception as e:
                                logger.error(
                                    f"Error processing message {message_id.decode()}: {e}",
                                    exc_info=True
                                )

                # Periodically flush completed sessions
                current_time = time.time()
                if current_time - last_flush_time >= flush_interval:
                    await self._flush_completed_sessions()
                    last_flush_time = current_time

            except Exception as e:
                logger.error(f"Error in consume loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause before retrying

    async def start(self):
        """Start the worker."""
        await self.connect()

        # Initialize Beanie for database operations
        from advanced_omi_backend.rq_queue import _ensure_beanie_initialized
        await _ensure_beanie_initialized()

        self.running = True

        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start consumption
        await self._consume_loop()

        # Cleanup
        await self.disconnect()
        logger.info("Worker stopped")

    async def stop(self):
        """Stop the worker."""
        self.running = False


async def main():
    """Main entry point for audio stream worker."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    worker = AudioStreamWorker(redis_url=redis_url)

    logger.info("Starting audio stream worker...")
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
