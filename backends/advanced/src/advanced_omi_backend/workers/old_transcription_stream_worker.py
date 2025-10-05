"""
Transcription Stream Worker - Consumes audio chunks from Redis Streams for real-time transcription.

This worker:
1. Consumes audio chunks from Redis Streams (audio:{client_id})
2. Accumulates audio chunks per client
3. Sends accumulated audio to transcription service periodically
4. Updates conversation with streaming transcription results

Architecture:
- Runs in parallel with audio_stream_worker
- Uses same Redis Streams but different consumer group
- Provides real-time transcription feedback
"""

import asyncio
import logging
import os
import signal
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

import redis.asyncio as aioredis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
transcription_logger = logging.getLogger("transcription_processing")


@dataclass
class TranscriptionAccumulator:
    """Accumulates audio chunks for streaming transcription."""
    client_id: str
    user_id: str
    user_email: str
    audio_rate: int
    audio_width: int
    audio_channels: int
    audio_buffer: bytearray  # Accumulated audio data
    last_transcription_time: float  # Last time we sent for transcription
    total_chunks: int = 0


class TranscriptionStreamWorker:
    """
    Worker that consumes audio chunks from Redis Streams and performs streaming transcription.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        consumer_name: Optional[str] = None,
        block_ms: int = 1000,
        batch_size: int = 10,
        transcription_interval: float = 2.0  # Transcribe every 2 seconds
    ):
        """Initialize transcription stream worker.

        Args:
            redis_url: Redis connection URL (defaults to REDIS_URL env var)
            consumer_name: Unique consumer name (auto-generated if not provided)
            block_ms: Block time in milliseconds when waiting for messages
            batch_size: Maximum messages to read per batch
            transcription_interval: Seconds between transcription updates
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.consumer_name = consumer_name or f"transcription-{uuid.uuid4().hex[:8]}"
        self.block_ms = block_ms
        self.batch_size = batch_size
        self.transcription_interval = transcription_interval

        self.redis: Optional[aioredis.Redis] = None
        self.running = False

        # Transcription accumulation state
        self.accumulators: Dict[str, TranscriptionAccumulator] = {}

        # Stream and consumer group configuration
        self.audio_stream_prefix = "audio:"
        self.transcription_consumer_group = "transcription-processor"

        # Track last processed message ID per stream (for XREAD)
        self.stream_positions: Dict[bytes, bytes] = {}

        logger.info(f"Initialized TranscriptionStreamWorker: {self.consumer_name}")

    async def connect(self):
        """Connect to Redis with connection pooling."""
        self.redis = await aioredis.from_url(
            self.redis_url,
            decode_responses=False,
            max_connections=10,
            socket_keepalive=True,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )
        logger.info(f"Transcription worker connected to Redis at {self.redis_url}")

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            logger.info("Transcription worker disconnected from Redis")

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

    async def _get_worker_count(self) -> int:
        """Get the number of active transcription workers."""
        worker_key = f"transcription_workers:{self.consumer_name}"
        await self.redis.setex(worker_key, 30, "active")

        cursor = b"0"
        worker_count = 0
        while cursor:
            cursor, keys = await self.redis.scan(cursor, match="transcription_workers:*")
            worker_count += len(keys)

        return max(worker_count, 1)

    def _should_process_stream(self, stream_key: bytes, worker_count: int) -> bool:
        """Determine if this worker should process a given stream using consistent hashing."""
        client_id = stream_key.decode().split(":", 1)[1]
        client_slot = hash(client_id) % worker_count
        worker_slot = hash(self.consumer_name) % worker_count
        return client_slot == worker_slot

    async def _process_audio_message(self, stream_name: bytes, message_id: bytes, message_data: dict):
        """Process a single audio message for transcription."""
        try:
            client_id = stream_name.decode().split(":", 1)[1]

            user_id = message_data[b"user_id"].decode()
            user_email = message_data[b"user_email"].decode()
            audio_data = message_data[b"audio_data"]
            audio_rate = int(message_data[b"audio_rate"].decode())
            audio_width = int(message_data[b"audio_width"].decode())
            audio_channels = int(message_data[b"audio_channels"].decode())

            # Get or create accumulator for this client
            if client_id not in self.accumulators:
                self.accumulators[client_id] = TranscriptionAccumulator(
                    client_id=client_id,
                    user_id=user_id,
                    user_email=user_email,
                    audio_rate=audio_rate,
                    audio_width=audio_width,
                    audio_channels=audio_channels,
                    audio_buffer=bytearray(),
                    last_transcription_time=time.time()
                )
                transcription_logger.info(
                    f"Started transcription session for client {client_id}"
                )

            # Add chunk to buffer
            accumulator = self.accumulators[client_id]
            accumulator.audio_buffer.extend(audio_data)
            accumulator.total_chunks += 1

        except Exception as e:
            logger.error(f"Error processing audio message for transcription: {e}", exc_info=True)

    async def _process_transcriptions(self):
        """Check accumulators and process transcription for clients that have accumulated enough audio."""
        current_time = time.time()

        for client_id, accumulator in list(self.accumulators.items()):
            time_since_last = current_time - accumulator.last_transcription_time

            # Transcribe if enough time has passed and we have audio
            if time_since_last >= self.transcription_interval and len(accumulator.audio_buffer) > 0:
                try:
                    await self._transcribe_audio(accumulator)
                    accumulator.last_transcription_time = current_time
                except Exception as e:
                    logger.error(f"Error transcribing audio for {client_id}: {e}", exc_info=True)

    async def _transcribe_audio(self, accumulator: TranscriptionAccumulator):
        """Send accumulated audio to transcription service."""
        from advanced_omi_backend.services.transcription import get_transcription_provider

        audio_bytes = bytes(accumulator.audio_buffer)

        transcription_logger.info(
            f"Transcribing {len(audio_bytes)} bytes for {accumulator.client_id} "
            f"({accumulator.total_chunks} chunks)"
        )

        # Get transcription provider
        provider = get_transcription_provider()
        if not provider:
            logger.warning("No transcription provider available")
            return

        # Transcribe audio
        result = await provider.transcribe(audio_bytes, accumulator.audio_rate, diarize=True)

        if result:
            # Extract transcript text
            transcript_text = ""
            segments = []

            if hasattr(result, "text"):
                transcript_text = result.text
                segments = getattr(result, "segments", [])
            elif isinstance(result, dict):
                transcript_text = result.get("text", "")
                segments = result.get("segments", [])

            if transcript_text:
                transcription_logger.info(
                    f"✅ Transcription for {accumulator.client_id}: "
                    f"{len(transcript_text)} chars, {len(segments)} segments"
                )

                # TODO: Publish transcription result to Redis for real-time updates
                # This could be consumed by WebSocket to send to clients

        # Clear buffer after transcription
        accumulator.audio_buffer.clear()

    async def _consume_loop(self):
        """Main consumption loop."""
        logger.info(f"Starting transcription stream consumer: {self.consumer_name}")

        last_process_time = time.time()

        while self.running:
            try:
                # Get worker count for consistent hashing
                worker_count = await self._get_worker_count()

                # Get all audio streams
                all_stream_keys = await self._get_all_audio_streams()

                if not all_stream_keys:
                    await asyncio.sleep(1)
                    continue

                # Filter to only streams this worker should process
                stream_keys = [key for key in all_stream_keys if self._should_process_stream(key, worker_count)]

                if not stream_keys:
                    await asyncio.sleep(1)
                    continue

                # Use XREAD to independently consume from assigned streams
                streams_dict = {}
                for stream_key in stream_keys:
                    last_id = self.stream_positions.get(stream_key, b"$")
                    streams_dict[stream_key] = last_id

                # Read new messages
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
                                await self._process_audio_message(
                                    stream_name, message_id, message_data
                                )
                                self.stream_positions[stream_name] = message_id
                            except Exception as e:
                                logger.error(
                                    f"Error processing message {message_id.decode()}: {e}",
                                    exc_info=True
                                )

                # Periodically process transcriptions
                current_time = time.time()
                if current_time - last_process_time >= 0.5:  # Check every 500ms
                    await self._process_transcriptions()
                    last_process_time = current_time

            except Exception as e:
                logger.error(f"Error in transcription consume loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def start(self):
        """Start the worker."""
        await self.connect()
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
        logger.info("Transcription worker stopped")

    async def stop(self):
        """Stop the worker."""
        self.running = False


async def main():
    """Main entry point for transcription stream worker."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    worker = TranscriptionStreamWorker(redis_url=redis_url)

    logger.info("Starting transcription stream worker...")
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
