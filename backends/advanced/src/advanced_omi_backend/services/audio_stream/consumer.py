"""
Base audio stream consumer - reads from Redis Streams and transcribes.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod

import redis.asyncio as redis
from redis import exceptions as redis_exceptions

logger = logging.getLogger(__name__)


class BaseAudioStreamConsumer(ABC):
    """
    Base class for audio stream consumers.

    Reads from audio:stream:{provider} and transcribes using the provider.
    Writes results to transcription:results:{session_id}.
    """

    def __init__(self, provider_name: str, redis_client: redis.Redis, buffer_chunks: int = 30):
        """
        Initialize consumer.

        Args:
            provider_name: Provider name (e.g., "deepgram", "parakeet")
            redis_client: Connected Redis client
            buffer_chunks: Number of chunks to accumulate before transcribing (default: 5 = ~1 second)
        """
        self.provider_name = provider_name
        self.redis_client = redis_client
        self.buffer_chunks = buffer_chunks

        # Stream and consumer group names
        self.input_stream = f"audio:stream:{provider_name}"
        self.group_name = f"{provider_name}_workers"
        self.consumer_name = f"{provider_name}-worker-{os.getpid()}"

        self.running = False

        # Buffering: accumulate chunks per session
        self.session_buffers = {}  # {session_id: {"chunks": [], "chunk_ids": [], "sample_rate": int}}

    async def setup_consumer_group(self):
        """Create consumer group if it doesn't exist."""
        # Create consumer group (ignore error if already exists)
        try:
            await self.redis_client.xgroup_create(
                self.input_stream,
                self.group_name,
                "0",
                mkstream=True
            )
            logger.info(f"➡️ Created consumer group: {self.group_name}")
        except redis_exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
            logger.info(f"➡️ Consumer group already exists: {self.group_name}")

    async def cleanup_dead_consumers(self, idle_threshold_ms: int = 30000):
        """
        Clean up dead consumers from the consumer group.

        Removes consumers that are idle > threshold (default 30 seconds) and have no pending messages.
        Claims and ACKs any pending messages from dead consumers first.

        Args:
            idle_threshold_ms: Idle time threshold in milliseconds (default 30 seconds)
        """
        try:
            # Get all consumers in the group
            consumers = await self.redis_client.execute_command(
                'XINFO', 'CONSUMERS', self.input_stream, self.group_name
            )

            if not consumers:
                return

            deleted_count = 0
            claimed_count = 0

            # Parse consumer info - each consumer is a nested list
            for consumer_info in consumers:
                consumer_dict = {}

                # Parse consumer fields (flat key-value pairs within each consumer)
                for j in range(0, len(consumer_info), 2):
                    if j+1 < len(consumer_info):
                        key = consumer_info[j].decode() if isinstance(consumer_info[j], bytes) else str(consumer_info[j])
                        value = consumer_info[j+1]
                        if isinstance(value, bytes):
                            try:
                                value = value.decode()
                            except UnicodeDecodeError:
                                value = str(value)
                        consumer_dict[key] = value

                consumer_name = consumer_dict.get("name", "")
                if isinstance(consumer_name, bytes):
                    consumer_name = consumer_name.decode()

                consumer_pending = int(consumer_dict.get("pending", 0))
                consumer_idle_ms = int(consumer_dict.get("idle", 0))

                # Skip our own consumer
                if consumer_name == self.consumer_name:
                    continue

                # Check if consumer is dead
                is_dead = consumer_idle_ms > idle_threshold_ms

                if is_dead:
                    # If consumer has pending messages, claim and ACK them first
                    if consumer_pending > 0:
                        logger.info(f"🔄 Claiming {consumer_pending} pending messages from dead consumer {consumer_name}")

                        try:
                            pending_messages = await self.redis_client.execute_command(
                                'XPENDING', self.input_stream, self.group_name, '-', '+', str(consumer_pending), consumer_name
                            )

                            # Parse pending messages (groups of 4: msg_id, consumer, idle_ms, delivery_count)
                            for k in range(0, len(pending_messages), 4):
                                if k < len(pending_messages):
                                    msg_id = pending_messages[k]
                                    if isinstance(msg_id, bytes):
                                        msg_id = msg_id.decode()

                                    # Claim to ourselves and ACK immediately
                                    try:
                                        await self.redis_client.execute_command(
                                            'XCLAIM', self.input_stream, self.group_name, self.consumer_name, '0', msg_id
                                        )
                                        await self.redis_client.xack(self.input_stream, self.group_name, msg_id)
                                        claimed_count += 1
                                    except Exception as claim_error:
                                        logger.warning(f"Failed to claim/ack message {msg_id}: {claim_error}")

                        except Exception as pending_error:
                            logger.warning(f"Failed to process pending messages for {consumer_name}: {pending_error}")

                    # Delete the dead consumer
                    try:
                        await self.redis_client.execute_command(
                            'XGROUP', 'DELCONSUMER', self.input_stream, self.group_name, consumer_name
                        )
                        deleted_count += 1
                        logger.info(f"🧹 Deleted dead consumer {consumer_name} (idle: {consumer_idle_ms}ms)")
                    except Exception as delete_error:
                        logger.warning(f"Failed to delete consumer {consumer_name}: {delete_error}")

            if deleted_count > 0 or claimed_count > 0:
                logger.info(f"✅ Cleanup complete: deleted {deleted_count} dead consumers, claimed {claimed_count} pending messages")

        except Exception as e:
            logger.error(f"❌ Failed to cleanup dead consumers: {e}", exc_info=True)

    @abstractmethod
    async def transcribe_audio(self, audio_data: bytes, sample_rate: int) -> dict:
        """
        Transcribe audio using the provider.

        Must be implemented by subclasses.

        Args:
            audio_data: Raw PCM audio bytes
            sample_rate: Audio sample rate (Hz)

        Returns:
            Dict with "text", "words", "segments", "confidence"
        """
        pass

    async def start_consuming(self):
        """Start consuming messages from the stream."""
        # Setup consumer group
        await self.setup_consumer_group()

        # Clean up dead consumers from previous runs
        await self.cleanup_dead_consumers()

        self.running = True
        logger.info(f"➡️ Starting consumer: {self.consumer_name}")

        while self.running:
            try:
                # Read messages from stream
                messages = await self.redis_client.xreadgroup(
                    self.group_name,
                    self.consumer_name,
                    {self.input_stream: ">"},
                    count=1,
                    block=1000  # Block for 1 second
                )

                if not messages:
                    continue

                for _, msgs in messages:
                    for message_id, fields in msgs:
                        await self.process_message(message_id, fields)

            except Exception as e:
                logger.error(f"➡️ [{self.consumer_name}] Error in consume loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def process_message(self, message_id: bytes, fields: dict):
        """
        Process a single message from the stream.
        Accumulates chunks and transcribes when buffer is full.

        Args:
            message_id: Redis message ID
            fields: Message fields
        """
        try:
            # Extract message data
            audio_data = fields[b"audio_data"]
            session_id = fields[b"session_id"].decode()
            chunk_id = fields[b"chunk_id"].decode()
            sample_rate = int(fields[b"sample_rate"].decode())

            # Check for end-of-session signal
            if chunk_id == "END":
                logger.info(f"➡️ [{self.consumer_name}] {self.provider_name}: Received END signal for session {session_id}")

                # Flush buffer for this session if it has any chunks
                if session_id in self.session_buffers:
                    buffer = self.session_buffers[session_id]

                    if len(buffer["chunks"]) > 0:
                        start_time = time.time()

                        # Combine buffered chunks
                        combined_audio = b"".join(buffer["chunks"])
                        combined_chunk_id = f"{buffer['chunk_ids'][0]}-{buffer['chunk_ids'][-1]}"

                        logger.info(
                            f"➡️ [{self.consumer_name}] {self.provider_name}: Flushing {len(buffer['chunks'])} remaining chunks "
                            f"({len(combined_audio)} bytes, ~{len(combined_audio)/32000:.1f}s) as {combined_chunk_id}"
                        )

                        # Transcribe remaining audio
                        result = await self.transcribe_audio(combined_audio, buffer["sample_rate"])

                        # Store result
                        processing_time = time.time() - start_time
                        await self.store_result(
                            session_id=session_id,
                            chunk_id=combined_chunk_id,
                            text=result.get("text", ""),
                            confidence=result.get("confidence", 0.0),
                            words=result.get("words", []),
                            segments=result.get("segments", []),
                            processing_time=processing_time
                        )

                        # ACK all buffered messages
                        for msg_id in buffer["message_ids"]:
                            await self.redis_client.xack(self.input_stream, self.group_name, msg_id)

                        logger.info(
                            f"➡️ [{self.consumer_name}] {self.provider_name}: Flushed buffer for session {session_id} "
                            f"in {processing_time:.2f}s (transcript: {len(result.get('text', ''))} chars)"
                        )

                    # Clean up session buffer
                    del self.session_buffers[session_id]

                # ACK the END message
                await self.redis_client.xack(self.input_stream, self.group_name, message_id)
                return

            # Initialize buffer for this session if needed
            if session_id not in self.session_buffers:
                self.session_buffers[session_id] = {
                    "chunks": [],
                    "chunk_ids": [],
                    "sample_rate": sample_rate,
                    "message_ids": []
                }

            # Add to buffer (skip empty audio data from END signals)
            if len(audio_data) > 0:
                buffer = self.session_buffers[session_id]
                buffer["chunks"].append(audio_data)
                buffer["chunk_ids"].append(chunk_id)
                buffer["message_ids"].append(message_id)
            else:
                # ACK and skip empty chunks
                await self.redis_client.xack(self.input_stream, self.group_name, message_id)
                return

            logger.debug(
                f"➡️ [{self.consumer_name}] {self.provider_name}: Buffered chunk {chunk_id} ({len(buffer['chunks'])}/{self.buffer_chunks})"
            )

            # Transcribe when buffer is full
            if len(buffer["chunks"]) >= self.buffer_chunks:
                start_time = time.time()

                # Combine buffered chunks
                combined_audio = b"".join(buffer["chunks"])
                combined_chunk_id = f"{buffer['chunk_ids'][0]}-{buffer['chunk_ids'][-1]}"

                # Log individual chunk IDs to detect duplicates
                chunk_list = ", ".join(buffer['chunk_ids'][:5] + ['...'] + buffer['chunk_ids'][-5:]) if len(buffer['chunk_ids']) > 10 else ", ".join(buffer['chunk_ids'])

                logger.info(
                    f"➡️ [{self.consumer_name}] {self.provider_name}: Transcribing {len(buffer['chunks'])} chunks "
                    f"({len(combined_audio)} bytes, ~{len(combined_audio)/32000:.1f}s) as {combined_chunk_id} [{chunk_list}]"
                )

                # Transcribe combined audio
                result = await self.transcribe_audio(combined_audio, sample_rate)

                # Store result
                processing_time = time.time() - start_time
                await self.store_result(
                    session_id=session_id,
                    chunk_id=combined_chunk_id,
                    text=result.get("text", ""),
                    confidence=result.get("confidence", 0.0),
                    words=result.get("words", []),
                    segments=result.get("segments", []),
                    processing_time=processing_time
                )

                # ACK all buffered messages
                for msg_id in buffer["message_ids"]:
                    await self.redis_client.xack(self.input_stream, self.group_name, msg_id)

                logger.info(
                    f"➡️ [{self.consumer_name}] {self.provider_name}: Completed {combined_chunk_id} in {processing_time:.2f}s "
                    f"(transcript: {len(result.get('text', ''))} chars)"
                )

                # Clear buffer
                buffer["chunks"] = []
                buffer["chunk_ids"] = []
                buffer["message_ids"] = []

        except Exception as e:
            logger.error(
                f"➡️ [{self.consumer_name}] {self.provider_name}: Failed to process chunk {fields.get(b'chunk_id', b'unknown').decode()}: {e}",
                exc_info=True
            )

    async def store_result(
        self,
        session_id: str,
        chunk_id: str,
        text: str,
        confidence: float,
        words: list,
        segments: list,
        processing_time: float
    ):
        """
        Store transcription result in Redis Stream.

        Args:
            session_id: Session identifier
            chunk_id: Chunk identifier
            text: Transcribed text
            confidence: Confidence score
            words: Word-level data
            segments: Speaker segments
            processing_time: Processing time in seconds
        """
        result_data = {
            b"text": text.encode(),
            b"chunk_id": chunk_id.encode(),
            b"provider": self.provider_name.encode(),
            b"confidence": str(confidence).encode(),
            b"processing_time": str(processing_time).encode(),
            b"timestamp": str(time.time()).encode(),
        }

        # Add optional JSON fields
        if words:
            result_data[b"words"] = json.dumps(words).encode()
        if segments:
            result_data[b"segments"] = json.dumps(segments).encode()

        # Write to session results stream
        session_results_stream = f"transcription:results:{session_id}"
        message_id = await self.redis_client.xadd(session_results_stream, result_data)

        logger.info(
            f"➡️ Stored result {chunk_id} in {session_results_stream}: "
            f"text_len={len(text)}, msg_id={message_id.decode()}"
        )

    async def stop(self):
        """Stop consuming messages."""
        self.running = False
        logger.info(f"➡️ Stopping consumer: {self.consumer_name}")
