#!/usr/bin/env python3
"""
ASR client using Wyoming protocol.
Captures audio from microphone and sends to ASR service for transcription.
"""

import argparse
import asyncio
import logging

from easy_audio_interfaces.extras.local_audio import InputMicStream
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart
from wyoming.client import AsyncTcpClient

logger = logging.getLogger(__name__)

SAMP_RATE = 16000
CHANNELS = 1
SAMP_WIDTH = 2  # bytes (16-bit)


async def main():
    ap = argparse.ArgumentParser(description="ASR client using Wyoming protocol")
    ap.add_argument(
        "--asr-url",
        type=str,
        default="tcp://192.168.0.110:8765",
        help="ASR service URL (default: tcp://192.168.0.110:8765)",
    )
    ap.add_argument("-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG")
    args = ap.parse_args()

    loglevel = logging.WARNING - (10 * min(args.verbose, 2))
    logging.basicConfig(format="%(asctime)s  %(levelname)s  %(message)s", level=loglevel)

    print(f"Connecting to ASR service: {args.asr_url}")
    async with AsyncTcpClient.from_uri(args.asr_url) as client:
        print("Connected to ASR service")
        
        # Initialize ASR session
        await client.write_event(Transcribe().event())
        await client.write_event(
            AudioStart(rate=SAMP_RATE, width=SAMP_WIDTH, channels=CHANNELS).event()
        )

        async def mic():
            try:
                async with InputMicStream(chunk_size=512) as stream:
                    logger.info("Starting microphone capture...")
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


if __name__ == "__main__":
    asyncio.run(main())
