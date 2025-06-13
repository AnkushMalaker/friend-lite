#!/usr/bin/env python3
"""
Test script for audio cropping functionality.
This script tests the ffmpeg-based audio cropping without requiring the full backend.
"""

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio_cropping_test")

async def create_test_audio(duration_seconds: float, output_path: str) -> bool:
    """Create a test audio file using ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration_seconds}",
            "-c:a", "pcm_s16le",
            "-ar", "16000",
            output_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        await process.communicate()
        return process.returncode == 0
    except Exception as e:
        logger.error(f"Error creating test audio: {e}")
        return False

async def crop_audio_with_ffmpeg(original_path: str, speech_segments: List[Tuple[float, float]], output_path: str) -> bool:
    """Test version of the audio cropping function."""
    if not speech_segments:
        logger.warning(f"No speech segments to crop for {original_path}")
        return False
        
    logger.info(f"Cropping audio {original_path} with {len(speech_segments)} speech segments")
    
    try:
        # Build ffmpeg filter for concatenating speech segments
        filter_parts = []
        for i, (start, end) in enumerate(speech_segments):
            duration = end - start
            filter_parts.append(f"[0:a]atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS[seg{i}]")
        
        # Concatenate all segments
        inputs = "".join(f"[seg{i}]" for i in range(len(speech_segments)))
        concat_filter = f"{inputs}concat=n={len(speech_segments)}:v=0:a=1[out]"
        
        full_filter = ";".join(filter_parts + [concat_filter])
        
        # Run ffmpeg as async subprocess
        cmd = [
            "ffmpeg", "-y",  # -y = overwrite output
            "-i", original_path,
            "-filter_complex", full_filter,
            "-map", "[out]",
            "-c:a", "pcm_s16le",  # Keep same format as original
            output_path
        ]
        
        logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            cropped_duration = sum(end - start for start, end in speech_segments)
            logger.info(f"Successfully cropped {original_path} -> {output_path} ({cropped_duration:.1f}s)")
            return True
        else:
            error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
            logger.error(f"ffmpeg failed for {original_path}: {error_msg}")
            return False
            
    except Exception as e:
        logger.error(f"Error running ffmpeg on {original_path}: {e}")
        return False

def get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
    return 0.0

async def test_audio_cropping():
    """Test the audio cropping functionality."""
    logger.info("Starting audio cropping test...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a 10-second test audio file
        original_file = temp_path / "test_original.wav"
        cropped_file = temp_path / "test_cropped.wav"
        
        logger.info("Creating test audio file...")
        success = await create_test_audio(10.0, str(original_file))
        if not success:
            logger.error("Failed to create test audio file")
            return False
        
        original_duration = get_audio_duration(str(original_file))
        logger.info(f"Created test audio: {original_duration:.1f}s")
        
        # Define speech segments to extract (simulate VAD output)
        speech_segments = [
            (1.0, 3.0),   # 2 seconds of speech
            (5.0, 7.5),   # 2.5 seconds of speech  
            (8.5, 9.5)    # 1 second of speech
        ]
        
        expected_duration = sum(end - start for start, end in speech_segments)
        logger.info(f"Testing cropping with segments: {speech_segments}")
        logger.info(f"Expected cropped duration: {expected_duration:.1f}s")
        
        # Test the cropping
        success = await crop_audio_with_ffmpeg(
            str(original_file), 
            speech_segments, 
            str(cropped_file)
        )
        
        if not success:
            logger.error("Audio cropping failed")
            return False
        
        # Verify the result
        if not cropped_file.exists():
            logger.error("Cropped file was not created")
            return False
        
        cropped_duration = get_audio_duration(str(cropped_file))
        logger.info(f"Actual cropped duration: {cropped_duration:.1f}s")
        
        # Allow small tolerance for ffmpeg precision
        if abs(cropped_duration - expected_duration) < 0.1:
            logger.info("✅ Audio cropping test PASSED!")
            return True
        else:
            logger.error(f"❌ Duration mismatch: expected {expected_duration:.1f}s, got {cropped_duration:.1f}s")
            return False

if __name__ == "__main__":
    asyncio.run(test_audio_cropping()) 