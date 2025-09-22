###############################################################################
# AUDIO PROCESSING FUNCTIONS
###############################################################################

import asyncio
import logging
import os
import time
import wave
import io
import numpy as np
from pathlib import Path

# Type import to avoid circular imports
from typing import TYPE_CHECKING, Optional

from wyoming.audio import AudioChunk

if TYPE_CHECKING:
    from advanced_omi_backend.client import ClientState
    from advanced_omi_backend.database import AudioChunksRepository

logger = logging.getLogger(__name__)

# Import constants from main.py (these are defined there)
MIN_SPEECH_SEGMENT_DURATION = float(os.getenv("MIN_SPEECH_SEGMENT_DURATION", "1.0"))  # seconds
CROPPING_CONTEXT_PADDING = float(os.getenv("CROPPING_CONTEXT_PADDING", "0.1"))  # seconds


async def process_audio_chunk(
    audio_data: bytes,
    client_id: str,
    user_id: str,
    user_email: str,
    audio_format: dict,
    client_state: Optional["ClientState"] = None
) -> None:
    """Process a single audio chunk through the standard pipeline.

    This function encapsulates the common pattern used across all audio input sources:
    1. Create AudioChunk with format details
    2. Queue AudioProcessingItem to processor
    3. Update client state if provided

    Args:
        audio_data: Raw audio bytes
        client_id: Client identifier
        user_id: User identifier
        user_email: User email
        audio_format: Dict containing {rate, width, channels, timestamp}
        client_state: Optional ClientState for state updates
    """

    from advanced_omi_backend.processors import (
        AudioProcessingItem,
        get_processor_manager,
    )

    # Extract format details
    rate = audio_format.get("rate", 16000)
    width = audio_format.get("width", 2)
    channels = audio_format.get("channels", 1)
    timestamp = audio_format.get("timestamp")

    # Use current time if no timestamp provided
    if timestamp is None:
        timestamp = int(time.time() * 1000)

    # Create AudioChunk with format details
    chunk = AudioChunk(
        audio=audio_data,
        rate=rate,
        width=width,
        channels=channels,
        timestamp=timestamp
    )

    # Create AudioProcessingItem and queue for processing
    processor_manager = get_processor_manager()
    processing_item = AudioProcessingItem(
        client_id=client_id,
        user_id=user_id,
        user_email=user_email,
        audio_chunk=chunk,
        timestamp=timestamp
    )

    await processor_manager.queue_audio(processing_item)


async def load_audio_file_as_chunk(audio_path: Path) -> AudioChunk:
    """Load existing audio file into Wyoming AudioChunk format for reprocessing.

    Args:
        audio_path: Path to the audio file on disk

    Returns:
        AudioChunk object ready for processing

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio file format is invalid
    """
    try:
        # Read the audio file
        with open(audio_path, 'rb') as f:
            file_content = f.read()

        # Process WAV file using existing pattern from system_controller.py
        with wave.open(io.BytesIO(file_content), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            channels = wav_file.getnchannels()
            audio_data = wav_file.readframes(wav_file.getnframes())

            # Convert to mono if stereo (same logic as system_controller.py)
            if channels == 2:
                if sample_width == 2:
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_array = audio_array.reshape(-1, 2)
                    audio_data = np.mean(audio_array, axis=1, dtype=np.int16).tobytes()
                    channels = 1
                else:
                    raise ValueError(f"Unsupported sample width for stereo: {sample_width}")

            # Validate format matches expected (16kHz, mono, 16-bit)
            if sample_rate != 16000:
                raise ValueError(f"Audio file has sample rate {sample_rate}Hz, expected 16kHz")
            if channels != 1:
                raise ValueError(f"Audio file has {channels} channels, expected mono")
            if sample_width != 2:
                raise ValueError(f"Audio file has {sample_width}-byte samples, expected 2 bytes")

            # Create AudioChunk with current timestamp
            chunk = AudioChunk(
                audio=audio_data,
                rate=sample_rate,
                width=sample_width,
                channels=channels,
                timestamp=int(time.time() * 1000)
            )

            logger.info(f"Loaded audio file {audio_path} as AudioChunk ({len(audio_data)} bytes)")
            return chunk

    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading audio file {audio_path}: {e}")
        raise ValueError(f"Invalid audio file format: {e}")


async def _process_audio_cropping_with_relative_timestamps(
    original_path: str,
    speech_segments: list[tuple[float, float]],
    output_path: str,
    audio_uuid: str,
    chunk_repo: Optional['AudioChunksRepository'] = None,
) -> bool:
    """
    Process audio cropping with automatic relative timestamp conversion.
    This function handles both live processing and reprocessing scenarios.
    """
    try:
        # Convert absolute timestamps to relative timestamps
        # Extract file start time from filename: timestamp_client_uuid.wav
        filename = original_path.split("/")[-1]
        logger.info(f"🕐 Parsing filename: {filename}")
        filename_parts = filename.split("_")
        if len(filename_parts) < 3:
            logger.error(
                f"Invalid filename format: {filename}. Expected format: timestamp_client_id_audio_uuid.wav"
            )
            return False

        try:
            file_start_timestamp = float(filename_parts[0])
        except ValueError as e:
            logger.error(f"Cannot parse timestamp from filename {filename}: {e}")
            return False

        # Convert speech segments to relative timestamps
        relative_segments = []
        for start_abs, end_abs in speech_segments:
            # Validate input timestamps
            if start_abs >= end_abs:
                logger.warning(
                    f"⚠️ Invalid speech segment: start={start_abs} >= end={end_abs}, skipping"
                )
                continue

            start_rel = start_abs - file_start_timestamp
            end_rel = end_abs - file_start_timestamp

            # Ensure relative timestamps are positive (sanity check)
            if start_rel < 0:
                logger.warning(
                    f"⚠️ Negative start timestamp: {start_rel} (absolute: {start_abs}, file_start: {file_start_timestamp}), clamping to 0.0"
                )
                start_rel = 0.0
            if end_rel < 0:
                logger.warning(
                    f"⚠️ Negative end timestamp: {end_rel} (absolute: {end_abs}, file_start: {file_start_timestamp}), skipping segment"
                )
                continue

            relative_segments.append((start_rel, end_rel))

        logger.info(f"🕐 Converting timestamps for {audio_uuid}: file_start={file_start_timestamp}")
        logger.info(f"🕐 Absolute segments: {speech_segments}")
        logger.info(f"🕐 Relative segments: {relative_segments}")

        # Validate that we have valid relative segments after conversion
        if not relative_segments:
            logger.warning(
                f"No valid relative segments after timestamp conversion for {audio_uuid}"
            )
            return False

        success = await _crop_audio_with_ffmpeg(original_path, relative_segments, output_path)
        if success:
            # Update database with cropped file info (keep original absolute timestamps for reference)
            cropped_filename = output_path.split("/")[-1]
            if chunk_repo is not None:
                await chunk_repo.update_cropped_audio(audio_uuid, cropped_filename, speech_segments)
            logger.info(f"Successfully processed cropped audio: {cropped_filename}")
            return True
        else:
            logger.error(f"Failed to crop audio for {audio_uuid}")
            return False
    except Exception as e:
        logger.error(f"Error in audio cropping task for {audio_uuid}: {e}", exc_info=True)
        return False


async def _crop_audio_with_ffmpeg(
    original_path: str, speech_segments: list[tuple[float, float]], output_path: str
) -> bool:
    """Use ffmpeg to crop audio - runs as async subprocess, no GIL issues"""
    logger.info(f"Cropping audio {original_path} with {len(speech_segments)} speech segments")

    if not speech_segments:
        logger.warning(f"No speech segments to crop for {original_path}")
        return False

    # Check if the original file exists
    if not os.path.exists(original_path):
        logger.error(f"Original audio file does not exist: {original_path}")
        return False

    # Filter out segments that are too short
    filtered_segments = []
    for start, end in speech_segments:
        duration = end - start
        if duration >= MIN_SPEECH_SEGMENT_DURATION:
            # Add padding around speech segments
            padded_start = max(0, start - CROPPING_CONTEXT_PADDING)
            padded_end = end + CROPPING_CONTEXT_PADDING
            filtered_segments.append((padded_start, padded_end))
        else:
            logger.debug(
                f"Skipping short segment: {start}-{end} ({duration:.2f}s < {MIN_SPEECH_SEGMENT_DURATION}s)"
            )

    if not filtered_segments:
        logger.warning(
            f"No segments meet minimum duration ({MIN_SPEECH_SEGMENT_DURATION}s) for {original_path}"
        )
        return False

    logger.info(
        f"Cropping audio {original_path} with {len(filtered_segments)} speech segments (filtered from {len(speech_segments)})"
    )

    try:
        # Build ffmpeg filter for concatenating speech segments
        filter_parts = []
        for i, (start, end) in enumerate(filtered_segments):
            duration = end - start
            filter_parts.append(
                f"[0:a]atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS[seg{i}]"
            )

        # Concatenate all segments
        inputs = "".join(f"[seg{i}]" for i in range(len(filtered_segments)))
        concat_filter = f"{inputs}concat=n={len(filtered_segments)}:v=0:a=1[out]"

        full_filter = ";".join(filter_parts + [concat_filter])

        # Run ffmpeg as async subprocess
        cmd = [
            "ffmpeg",
            "-y",  # -y = overwrite output
            "-i",
            original_path,
            "-filter_complex",
            full_filter,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",  # Keep same format as original
            output_path,
        ]

        logger.info(f"Running ffmpeg command: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        if stdout:
            logger.debug(f"FFMPEG stdout: {stdout.decode()}")

        if process.returncode == 0:
            # Calculate cropped duration
            cropped_duration = sum(end - start for start, end in filtered_segments)
            logger.info(
                f"Successfully cropped {original_path} -> {output_path} ({cropped_duration:.1f}s from {len(filtered_segments)} segments)"
            )
            return True
        else:
            error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
            logger.error(f"ffmpeg failed for {original_path}: {error_msg}")
            return False

    except Exception as e:
        logger.error(f"Error running ffmpeg on {original_path}: {e}", exc_info=True)
        return False
