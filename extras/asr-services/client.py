#!/usr/bin/env python3
"""
ASR client using Wyoming protocol.
Captures audio from microphone or file and sends to ASR service for transcription.
"""

import argparse
import asyncio
import logging
from pathlib import Path

from easy_audio_interfaces.extras.local_audio import InputMicStream
from easy_audio_interfaces.filesystem import LocalFileStreamer
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart
from wyoming.client import AsyncTcpClient

logger = logging.getLogger(__name__)

SAMP_RATE = 16000
CHANNELS = 1
SAMP_WIDTH = 2  # bytes (16-bit)


async def run_mic_transcription(asr_url: str, device_index: int | None = None):
    """Run ASR transcription from microphone input."""
    print(f"Connecting to ASR service: {asr_url}")
    async with AsyncTcpClient.from_uri(asr_url) as client:
        print("Connected to ASR service")
        
        # Initialize ASR session
        await client.write_event(Transcribe().event())
        await client.write_event(
            AudioStart(rate=SAMP_RATE, width=SAMP_WIDTH, channels=CHANNELS).event()
        )

        async def mic():
            try:
                async with InputMicStream(
                    sample_rate=SAMP_RATE,
                    channels=CHANNELS,
                    chunk_size=512,
                    device_index=device_index
                ) as stream:
                    logger.info(f"Starting microphone capture from device: {device_index or 'default'}")
                    while True:
                        data = await stream.read()
                        await client.write_event(data.event())
                        logger.debug(f"Sent audio chunk: {len(data.audio)} bytes")
                        await asyncio.sleep(0.01)
            except KeyboardInterrupt:
                logger.info("Stopping microphone capture...")
                raise

        async def transcriptions():
            while True:
                event = await client.read_event()
                if event is None:
                    break
                if Transcript.is_type(event.type):
                    transcript = Transcript.from_event(event)
                    print(f"Transcript: {transcript.text}")
                else:
                    logger.debug(f"Received event: {event}")

        try:
            await asyncio.gather(mic(), transcriptions())
        except KeyboardInterrupt:
            print("\nStopping ASR client...")


async def run_file_transcription(asr_url: str, file_path: str | Path):
    """Run ASR transcription from audio file input."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    print(f"Connecting to ASR service: {asr_url}")
    async with AsyncTcpClient.from_uri(asr_url) as client:
        print("Connected to ASR service")
        
        # Initialize file streamer to get audio properties
        async with LocalFileStreamer(
            file_path=file_path,
            chunk_size_samples=512
        ) as streamer:
            print(f"Loaded audio file: {file_path}")
            print(f"Sample rate: {streamer.sample_rate}, Channels: {streamer.channels}")
            
            # Initialize ASR session with file's audio properties
            await client.write_event(Transcribe().event())
            await client.write_event(
                AudioStart(
                    rate=streamer.sample_rate, 
                    width=SAMP_WIDTH, 
                    channels=streamer.channels
                ).event()
            )

            async def file_reader():
                try:
                    logger.info(f"Starting file transcription: {file_path}")
                    async for chunk in streamer.iter_frames():
                        await client.write_event(chunk.event())
                        logger.debug(f"Sent audio chunk: {len(chunk.audio)} bytes")
                        await asyncio.sleep(0.01)  # Small delay to avoid overwhelming the server
                    logger.info("Finished sending audio file")
                except Exception as e:
                    logger.error(f"Error reading file: {e}")
                    raise

            async def transcriptions():
                while True:
                    event = await client.read_event()
                    if event is None:
                        break
                    if Transcript.is_type(event.type):
                        transcript = Transcript.from_event(event)
                        print(f"Transcript: {transcript.text}")
                    else:
                        logger.debug(f"Received event: {event}")

            try:
                await asyncio.gather(file_reader(), transcriptions())
            except KeyboardInterrupt:
                print("\nStopping ASR client...")


async def main():
    parser = argparse.ArgumentParser(
        description="ASR client using Wyoming protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s mic                    # Use default microphone
  %(prog)s mic --device 1         # Use specific audio device
  %(prog)s file audio.wav         # Transcribe audio file
        """
    )
    
    parser.add_argument(
        "--asr-url",
        type=str,
        default="tcp://192.168.0.110:8765",
        help="ASR service URL (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="count", 
        default=0, 
        help="Increase verbosity (-v: INFO, -vv: DEBUG)"
    )
    
    # Create subparsers for different input modes
    subparsers = parser.add_subparsers(
        dest="command", 
        help="Input source for audio",
        required=True
    )
    
    # Microphone subcommand
    mic_parser = subparsers.add_parser(
        "mic", 
        help="Capture audio from microphone"
    )
    mic_parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (default: system default)"
    )
    
    # File subcommand
    file_parser = subparsers.add_parser(
        "file",
        help="Transcribe audio from file"
    )
    file_parser.add_argument(
        "file_path",
        type=str,
        help="Path to audio file (e.g., audio.wav)"
    )
    
    args = parser.parse_args()

    # Set up logging
    loglevel = logging.WARNING - (10 * min(args.verbose, 2))
    logging.basicConfig(
        format="%(asctime)s  %(levelname)s  %(message)s", 
        level=loglevel
    )

    try:
        if args.command == "mic":
            await run_mic_transcription(args.asr_url, args.device)
        elif args.command == "file":
            await run_file_transcription(args.asr_url, args.file_path)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
