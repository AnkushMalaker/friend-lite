###############################################################################
# AUDIO PROCESSING FUNCTIONS
###############################################################################

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


async def _process_audio_cropping_with_relative_timestamps(
    original_path: str,
    speech_segments: list[tuple[float, float]],
    output_path: str,
    audio_uuid: str,
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
