#!/usr/bin/env python3
"""
Speaker Recognition Laptop Client

A client application that uses microphone input to enroll speakers or identify them
using the speaker recognition service.

Usage:
    # Enroll a speaker
    python laptop_client.py enroll --speaker-id "john_doe" --speaker-name "John Doe" --duration 10

    # Identify a speaker
    python laptop_client.py identify --duration 5

    # List enrolled speakers
    python laptop_client.py list

    # Remove a speaker
    python laptop_client.py remove --speaker-id "john_doe"

Requirements:
    # Main dependencies are included in pyproject.toml
    # Only need PyAudio for microphone access:
    # Ubuntu/Debian: sudo apt-get install portaudio19-dev python3-pyaudio
    # macOS: brew install portaudio && pip install pyaudio
    # Windows: pip install pyaudio
"""

import argparse
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

import aiohttp

from easy_audio_interfaces.extras.local_audio import InputMicStream  # type: ignore
from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink  # type: ignore

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("laptop_client")


class SpeakerClient:
    """Client for speaker recognition service with microphone recording."""
    
    def __init__(self, service_url: str = "http://localhost:8001"):
        self.service_url = service_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, endpoint: str, **kwargs):
        """Make HTTP request to speaker service."""
        if not self.session:
            raise RuntimeError("Client not initialized")
        
        url = f"{self.service_url}{endpoint}"
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {e}")
    
    async def health_check(self):
        """Check service health."""
        return await self._request("GET", "/health")
    
    async def enroll_speaker(self, speaker_id: str, speaker_name: str, audio_path: str):
        """Enroll a speaker from audio file."""
        data = {
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "audio_path": audio_path
        }
        return await self._request("POST", "/enroll", json=data)
    
    async def identify_speaker(self, audio_path: str):
        """Identify speaker from audio file."""
        data = {"audio_path": audio_path}
        return await self._request("POST", "/identify", json=data)
    
    async def verify_speaker(self, speaker_id: str, audio_path: str):
        """Verify if audio matches specific speaker."""
        data = {
            "speaker_id": speaker_id,
            "audio_path": audio_path
        }
        return await self._request("POST", "/verify", json=data)
    
    async def list_speakers(self):
        """List all enrolled speakers."""
        result = await self._request("GET", "/speakers")
        return result.get("speakers", [])
    
    async def remove_speaker(self, speaker_id: str):
        """Remove an enrolled speaker."""
        return await self._request("DELETE", f"/speakers/{speaker_id}")


async def record_audio(duration: float, sample_rate: int = 16000) -> str:
    """Record audio from microphone and save to temporary file."""
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()
    
    logger.info(f"Recording {duration} seconds of audio...")
    logger.info("Speak now!")
    
    try:
        async with InputMicStream(sample_rate=sample_rate) as mic, \
                   LocalFileSink(temp_path) as sink:
            
            # Record for specified duration
            start_time = asyncio.get_event_loop().time()
            async for chunk in mic.iter_frames():
                await sink.write(chunk)
                
                # Check if duration elapsed
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= duration:
                    break
        
        logger.info(f"Recording complete. Saved to: {temp_path}")
        return temp_path
        
    except Exception as e:
        # Clean up on error
        Path(temp_path).unlink(missing_ok=True)
        raise e


async def cmd_enroll(args):
    """Enroll a new speaker."""
    logger.info(f"Enrolling speaker: {args.speaker_name} (ID: {args.speaker_id})")
    
    # Record audio
    audio_path = await record_audio(args.duration)
    
    try:
        async with SpeakerClient(args.service_url) as client:
            # Check service health
            await client.health_check()
            logger.info("Speaker service is online")
            
            # Enroll speaker
            result = await client.enroll_speaker(args.speaker_id, args.speaker_name, audio_path)
            
            if result.get("success"):
                logger.info(f"✅ Successfully enrolled {args.speaker_name}")
            else:
                logger.error("❌ Failed to enroll speaker")
                
    except Exception as e:
        logger.error(f"Error during enrollment: {e}")
    finally:
        # Clean up temporary file
        Path(audio_path).unlink(missing_ok=True)


async def cmd_identify(args):
    """Identify a speaker from microphone."""
    logger.info("Recording audio for speaker identification...")
    
    # Record audio
    audio_path = await record_audio(args.duration)
    
    try:
        async with SpeakerClient(args.service_url) as client:
            # Check service health
            await client.health_check()
            logger.info("Speaker service is online")
            
            # Identify speaker
            result = await client.identify_speaker(audio_path)
            
            if result.get("identified"):
                speaker = result.get("speaker", {})
                score = result.get("score", 0.0)
                logger.info(f"✅ Identified speaker: {speaker.get('name', 'Unknown')} (ID: {speaker.get('id', 'Unknown')}) - Score: {score:.3f}")
            else:
                score = result.get("score", 0.0)
                logger.info(f"❓ Speaker not recognized - Score: {score:.3f}")
                
    except Exception as e:
        logger.error(f"Error during identification: {e}")
    finally:
        # Clean up temporary file
        Path(audio_path).unlink(missing_ok=True)


async def cmd_verify(args):
    """Verify if audio matches a specific speaker."""
    logger.info(f"Recording audio for verification against speaker: {args.speaker_id}")
    
    # Record audio
    audio_path = await record_audio(args.duration)
    
    try:
        async with SpeakerClient(args.service_url) as client:
            # Check service health
            await client.health_check()
            logger.info("Speaker service is online")
            
            # Verify speaker
            result = await client.verify_speaker(args.speaker_id, audio_path)
            
            match = result.get("match", False)
            score = result.get("score", 0.0)
            
            if match:
                logger.info(f"✅ Voice matches {args.speaker_id} - Score: {score:.3f}")
            else:
                logger.info(f"❌ Voice does not match {args.speaker_id} - Score: {score:.3f}")
                
    except Exception as e:
        logger.error(f"Error during verification: {e}")
    finally:
        # Clean up temporary file
        Path(audio_path).unlink(missing_ok=True)


async def cmd_list(args):
    """List all enrolled speakers."""
    try:
        async with SpeakerClient(args.service_url) as client:
            # Check service health
            await client.health_check()
            logger.info("Speaker service is online")
            
            # List speakers
            speakers = await client.list_speakers()
            
            if speakers:
                logger.info(f"📋 Enrolled speakers ({len(speakers)}):")
                for speaker in speakers:
                    logger.info(f"  • {speaker.get('name', 'Unknown')} (ID: {speaker.get('id', 'Unknown')})")
            else:
                logger.info("📋 No speakers enrolled")
                
    except Exception as e:
        logger.error(f"Error listing speakers: {e}")


async def cmd_remove(args):
    """Remove an enrolled speaker."""
    logger.info(f"Removing speaker: {args.speaker_id}")
    
    try:
        async with SpeakerClient(args.service_url) as client:
            # Check service health
            await client.health_check()
            logger.info("Speaker service is online")
            
            # Remove speaker
            result = await client.remove_speaker(args.speaker_id)
            
            if result.get("deleted"):
                logger.info(f"✅ Successfully removed speaker: {args.speaker_id}")
            else:
                logger.error(f"❌ Failed to remove speaker: {args.speaker_id}")
                
    except Exception as e:
        logger.error(f"Error removing speaker: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Speaker Recognition Laptop Client",
        epilog="Examples:\n"
               "  python laptop_client.py enroll --speaker-id john --speaker-name 'John Doe' --duration 10\n"
               "  python laptop_client.py identify --duration 5\n"
               "  python laptop_client.py verify --speaker-id john --duration 3\n"
               "  python laptop_client.py list\n"
               "  python laptop_client.py remove --speaker-id john",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--service-url",
        default="http://localhost:8001",
        help="Speaker recognition service URL (default: http://localhost:8001)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Enroll command
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new speaker")
    enroll_parser.add_argument("--speaker-id", required=True, help="Unique speaker ID")
    enroll_parser.add_argument("--speaker-name", required=True, help="Speaker display name")
    enroll_parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds (default: 10)")
    
    # Identify command
    identify_parser = subparsers.add_parser("identify", help="Identify a speaker")
    identify_parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds (default: 5)")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a specific speaker")
    verify_parser.add_argument("--speaker-id", required=True, help="Speaker ID to verify against")
    verify_parser.add_argument("--duration", type=float, default=3.0, help="Recording duration in seconds (default: 3)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List enrolled speakers")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove an enrolled speaker")
    remove_parser.add_argument("--speaker-id", required=True, help="Speaker ID to remove")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    

    
    # Run the appropriate command
    try:
        if args.command == "enroll":
            asyncio.run(cmd_enroll(args))
        elif args.command == "identify":
            asyncio.run(cmd_identify(args))
        elif args.command == "verify":
            asyncio.run(cmd_verify(args))
        elif args.command == "list":
            asyncio.run(cmd_list(args))
        elif args.command == "remove":
            asyncio.run(cmd_remove(args))
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main() 