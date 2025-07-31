#!/usr/bin/env python3
"""
Speaker enrollment script for the OMI backend.

This script helps enroll speakers by:
1. Recording audio from microphone
2. Using existing audio files
3. Calling the enrollment API

Usage examples:
    # Enroll from an existing audio file
    python enroll_speaker.py --id john_doe --name "John Doe" --file audio_chunks/sample.wav

    # Enroll from a specific segment of an audio file
    python enroll_speaker.py --id jane_smith --name "Jane Smith" --file audio_chunks/sample.wav --start 10.0 --end 15.0

    # Record new audio for enrollment (requires microphone)
    python enroll_speaker.py --id bob_jones --name "Bob Jones" --record --duration 5.0

    # List enrolled speakers
    python enroll_speaker.py --list
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import aiohttp
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default server settings
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000


async def enroll_speaker_api(
    host: str,
    port: int,
    speaker_id: str,
    speaker_name: str,
    audio_file_path: str,
    start_time=None,
    end_time=None,
):
    """Call the API to enroll a speaker."""
    url = f"http://{host}:{port}/api/speakers/enroll"

    data = {
        "speaker_id": speaker_id,
        "speaker_name": speaker_name,
        "audio_file_path": audio_file_path,
    }

    if start_time is not None:
        data["start_time"] = start_time
    if end_time is not None:
        data["end_time"] = end_time

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            result = await response.json()
            if response.status == 200:
                logger.info(f"✅ Successfully enrolled speaker: {result}")
                return True
            else:
                logger.error(f"❌ Failed to enroll speaker: {result}")
                return False


async def list_speakers_api(host: str, port: int):
    """List all enrolled speakers."""
    url = f"http://{host}:{port}/api/speakers"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            result = await response.json()
            if response.status == 200:
                speakers = result.get("speakers", [])
                if speakers:
                    print("\n📋 Enrolled Speakers:")
                    print("-" * 60)
                    for speaker in speakers:
                        enrolled_time = ""
                        if speaker.get("enrolled_at"):
                            enrolled_time = time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.localtime(speaker["enrolled_at"])
                            )
                        print(f"ID: {speaker['id']}")
                        print(f"Name: {speaker['name']}")
                        print(f"Audio File: {speaker.get('audio_file_path', 'N/A')}")
                        print(f"Enrolled: {enrolled_time}")
                        print("-" * 60)
                else:
                    print("No speakers enrolled yet.")
                return True
            else:
                logger.error(f"❌ Failed to list speakers: {result}")
                return False


async def identify_speaker_api(
    host: str, port: int, audio_file_path: str, start_time=None, end_time=None
):
    """Test speaker identification."""
    url = f"http://{host}:{port}/api/speakers/identify"

    data = {"audio_file_path": audio_file_path}

    if start_time is not None:
        data["start_time"] = start_time
    if end_time is not None:
        data["end_time"] = end_time

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as response:
            result = await response.json()
            if response.status == 200:
                if result.get("identified"):
                    print(f"🎯 Identified speaker: {result['speaker_id']}")
                    if result.get("speaker_info"):
                        print(f"   Name: {result['speaker_info'].get('speaker_name')}")
                else:
                    print("❓ Speaker not recognized")
                return True
            else:
                logger.error(f"❌ Failed to identify speaker: {result}")
                return False


def record_audio(duration: float, output_file: Path):
    """Record audio from microphone."""
    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        logger.info(f"🎤 Recording audio for {duration} seconds...")
        logger.info("💡 Speak clearly into your microphone now!")

        # Record audio
        sample_rate = 16000  # Same as backend configuration
        audio_data = sd.rec(
            int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32
        )
        sd.wait()  # Wait until recording is finished

        # Save to file
        sf.write(output_file, audio_data, sample_rate)
        logger.info(f"✅ Audio saved to: {output_file}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to record audio: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Speaker enrollment for OMI backend")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")

    # Speaker enrollment options
    parser.add_argument("--id", help="Speaker ID (unique identifier)")
    parser.add_argument("--name", help="Speaker name (human readable)")
    parser.add_argument("--file", help="Audio file path (relative to audio_chunks/)")
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds")

    # Recording option
    parser.add_argument("--record", action="store_true", help="Record new audio")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds")

    # Utility options
    parser.add_argument("--list", action="store_true", help="List enrolled speakers")
    parser.add_argument("--identify", help="Test speaker identification on audio file")

    args = parser.parse_args()

    # Check server connection
    try:
        response = requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
        if response.status_code != 200:
            logger.error(f"❌ Server not responding properly at {args.host}:{args.port}")
            return
    except requests.exceptions.RequestException:
        logger.error(f"❌ Cannot connect to server at {args.host}:{args.port}")
        logger.error("   Make sure the backend is running!")
        return

    logger.info(f"✅ Connected to server at {args.host}:{args.port}")

    # Handle different operations
    if args.list:
        await list_speakers_api(args.host, args.port)

    elif args.identify:
        await identify_speaker_api(args.host, args.port, args.identify, args.start, args.end)

    elif args.record:
        if not args.id or not args.name:
            logger.error("❌ --id and --name are required for recording")
            return

        # Generate filename based on speaker ID and timestamp
        timestamp = int(time.time())
        audio_file = Path(f"speaker_enrollment_{args.id}_{timestamp}.wav")

        if record_audio(args.duration, audio_file):
            # Enroll speaker using recorded audio
            await enroll_speaker_api(
                args.host, args.port, args.id, args.name, str(audio_file), args.start, args.end
            )

    elif args.file:
        if not args.id or not args.name:
            logger.error("❌ --id and --name are required for enrollment")
            return

        await enroll_speaker_api(
            args.host, args.port, args.id, args.name, args.file, args.start, args.end
        )

    else:
        parser.print_help()
        print("\n💡 Quick start examples:")
        print(f"   # List speakers:")
        print(f"   python {parser.prog} --list")
        print(f"   ")
        print(f"   # Enroll from existing audio:")
        print(f'   python {parser.prog} --id alice --name "Alice" --file sample.wav')
        print(f"   ")
        print(f"   # Record and enroll:")
        print(f'   python {parser.prog} --id bob --name "Bob" --record --duration 5')


if __name__ == "__main__":
    asyncio.run(main())
