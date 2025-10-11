"""
Audio extraction utilities for getting audio chunks from Redis.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def parse_chunk_range(chunk_id: str) -> Tuple[int, int]:
    """
    Parse chunk ID range like "00001-00030" into (1, 30).

    Args:
        chunk_id: Chunk ID string (e.g., "00001-00030" or "00005")

    Returns:
        Tuple of (start_chunk, end_chunk)
    """
    if "-" in chunk_id:
        start, end = chunk_id.split("-")
        return int(start), int(end)
    else:
        # Single chunk
        chunk_num = int(chunk_id)
        return chunk_num, chunk_num


async def extract_audio_for_results(
    redis_client,
    client_id: str,
    session_id: str,
    transcription_results: List[dict]
) -> bytes:
    """
    Extract audio chunks for transcription results.

    Reads the chunk_id from each result to determine which audio chunks to fetch.

    Args:
        redis_client: Redis client
        client_id: Client identifier
        session_id: Session identifier
        transcription_results: List of transcription results from aggregator

    Returns:
        Combined audio bytes for all chunks in results
    """
    if not transcription_results:
        return b""

    # Parse chunk ranges from all results
    chunk_ranges = []
    for result in transcription_results:
        chunk_id = result.get("chunk_id", "")
        if chunk_id:
            start, end = parse_chunk_range(chunk_id)
            chunk_ranges.append((start, end))

    if not chunk_ranges:
        logger.warning("No chunk ranges found in transcription results")
        return b""

    # Find overall range
    min_chunk = min(start for start, _ in chunk_ranges)
    max_chunk = max(end for _, end in chunk_ranges)

    logger.info(
        f"Extracting audio chunks {min_chunk:05d}-{max_chunk:05d} "
        f"for session {session_id} ({max_chunk - min_chunk + 1} chunks)"
    )

    # Read from audio stream
    stream_name = f"audio:stream:{client_id}"

    # Get all messages (we'll filter by session and chunk)
    messages = await redis_client.xrange(stream_name)

    # Collect audio chunks
    audio_chunks = {}  # {chunk_num: audio_data}

    for _, fields in messages:
        # Check if this message belongs to our session
        msg_session_id = fields.get(b"session_id", b"").decode()
        if msg_session_id != session_id:
            continue

        # Get chunk number
        msg_chunk_id = fields.get(b"chunk_id", b"").decode()
        if not msg_chunk_id or msg_chunk_id == "END":
            continue

        try:
            chunk_num = int(msg_chunk_id)
        except ValueError:
            continue

        # Check if this chunk is in our range
        if min_chunk <= chunk_num <= max_chunk:
            audio_data = fields.get(b"audio_data", b"")
            audio_chunks[chunk_num] = audio_data

    # Combine chunks in order
    sorted_chunks = sorted(audio_chunks.items())
    combined_audio = b"".join(data for _, data in sorted_chunks)

    logger.info(
        f"Extracted {len(sorted_chunks)} audio chunks "
        f"({len(combined_audio)} bytes, ~{len(combined_audio)/32000:.1f}s)"
    )

    return combined_audio

