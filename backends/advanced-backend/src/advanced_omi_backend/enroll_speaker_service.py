#!/usr/bin/env python3
"""
Speaker enrollment script for the OMI backend with separate speaker recognition service.

This script helps enroll speakers by communicating with the speaker recognition service.

Usage examples:
    # Enroll from an existing audio file
    python enroll_speaker_service.py --id john_doe --name "John Doe" --file audio_chunks/sample.wav

    # Enroll from a specific segment of an audio file
    python enroll_speaker_service.py --id jane_smith --name "Jane Smith" --file audio_chunks/sample.wav --start 10.0 --end 15.0

    # List enrolled speakers
    python enroll_speaker_service.py --list

    # Remove a speaker
    python enroll_speaker_service.py --remove john_doe
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service settings
DEFAULT_BACKEND_HOST = "localhost"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_SPEAKER_HOST = "localhost"
DEFAULT_SPEAKER_PORT = 8001


async def enroll_speaker_api(
    speaker_host: str,
    speaker_port: int,
    speaker_id: str,
    speaker_name: str,
    audio_file_path: str,
    start_time=None,
    end_time=None,
):
    """Call the speaker service API to enroll a speaker."""
    url = f"http://{speaker_host}:{speaker_port}/enroll"

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
        try:
            async with session.post(url, json=data) as response:
                result = await response.json()
                if response.status == 200:
                    logger.info(f"✅ Successfully enrolled speaker: {result}")
                    return True
                else:
                    logger.error(f"❌ Failed to enroll speaker: {result}")
                    return False
        except aiohttp.ClientError as e:
            logger.error(f"❌ Connection error: {e}")
            return False


async def list_speakers_api(speaker_host: str, speaker_port: int):
    """List all enrolled speakers."""
    url = f"http://{speaker_host}:{speaker_port}/speakers"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                result = await response.json()
                if response.status == 200:
                    speakers = result.get("speakers", [])
                    if speakers:
                        print("\n📋 Enrolled Speakers:")
                        print("-" * 60)
                        for speaker in speakers:
                            print(f"ID: {speaker['id']}")
                            print(f"Name: {speaker['name']}")
                            print("-" * 60)
                    else:
                        print("No speakers enrolled yet.")
                    return True
                else:
                    logger.error(f"❌ Failed to list speakers: {result}")
                    return False
        except aiohttp.ClientError as e:
            logger.error(f"❌ Connection error: {e}")
            return False


async def remove_speaker_api(speaker_host: str, speaker_port: int, speaker_id: str):
    """Remove a speaker."""
    url = f"http://{speaker_host}:{speaker_port}/speakers/{speaker_id}"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.delete(url) as response:
                result = await response.json()
                if response.status == 200:
                    logger.info(f"✅ Successfully removed speaker: {speaker_id}")
                    return True
                else:
                    logger.error(f"❌ Failed to remove speaker: {result}")
                    return False
        except aiohttp.ClientError as e:
            logger.error(f"❌ Connection error: {e}")
            return False


async def identify_speaker_api(
    speaker_host: str, speaker_port: int, audio_file_path: str, start_time=None, end_time=None
):
    """Test speaker identification."""
    url = f"http://{speaker_host}:{speaker_port}/identify"

    data = {"audio_file_path": audio_file_path}

    if start_time is not None:
        data["start_time"] = start_time
    if end_time is not None:
        data["end_time"] = end_time

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=data) as response:
                result = await response.json()
                if response.status == 200:
                    if result.get("identified"):
                        print(f"🎯 Identified speaker: {result['speaker_id']}")
                        if result.get("speaker_info"):
                            print(f"   Name: {result['speaker_info'].get('name')}")
                    else:
                        print("❓ Speaker not recognized")
                    return True
                else:
                    logger.error(f"❌ Failed to identify speaker: {result}")
                    return False
        except aiohttp.ClientError as e:
            logger.error(f"❌ Connection error: {e}")
            return False


async def check_service_health(speaker_host: str, speaker_port: int):
    """Check if the speaker recognition service is running."""
    url = f"http://{speaker_host}:{speaker_port}/health"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Speaker service is healthy: {result}")
                    return True
                else:
                    logger.error(f"❌ Speaker service not healthy: {response.status}")
                    return False
        except asyncio.TimeoutError:
            logger.error(f"❌ Speaker service timeout at {speaker_host}:{speaker_port}")
            return False
        except aiohttp.ClientError as e:
            logger.error(f"❌ Cannot connect to speaker service: {e}")
            return False


async def main():
    parser = argparse.ArgumentParser(
        description="Speaker enrollment for OMI backend via speaker service"
    )
    parser.add_argument("--speaker-host", default=DEFAULT_SPEAKER_HOST, help="Speaker service host")
    parser.add_argument(
        "--speaker-port", type=int, default=DEFAULT_SPEAKER_PORT, help="Speaker service port"
    )

    # Speaker enrollment options
    parser.add_argument("--id", help="Speaker ID (unique identifier)")
    parser.add_argument("--name", help="Speaker name (human readable)")
    parser.add_argument("--file", help="Audio file path (relative to shared audio directory)")
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds")

    # Utility options
    parser.add_argument("--list", action="store_true", help="List enrolled speakers")
    parser.add_argument("--identify", help="Test speaker identification on audio file")
    parser.add_argument("--remove", help="Remove a speaker by ID")

    args = parser.parse_args()

    # Check speaker service connection
    if not await check_service_health(args.speaker_host, args.speaker_port):
        logger.error("   Make sure the speaker recognition service is running!")
        logger.error("   Try: docker-compose up speaker-recognition")
        return

    # Handle different operations
    if args.list:
        await list_speakers_api(args.speaker_host, args.speaker_port)

    elif args.identify:
        await identify_speaker_api(
            args.speaker_host, args.speaker_port, args.identify, args.start, args.end
        )

    elif args.remove:
        await remove_speaker_api(args.speaker_host, args.speaker_port, args.remove)

    elif args.id and args.name and args.file:
        # Convert relative path to absolute path
        audio_file_path = os.path.abspath(args.file)
        if not os.path.exists(audio_file_path):
            logger.error(f"❌ Audio file not found: {audio_file_path}")
            return

        await enroll_speaker_api(
            args.speaker_host,
            args.speaker_port,
            args.id,
            args.name,
            audio_file_path,
            args.start,
            args.end,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
