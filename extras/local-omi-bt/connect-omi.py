import asyncio
import logging
import os
import sys
from asyncio import Queue
from typing import Any, AsyncGenerator

import asyncstdlib as asyncstd
from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from dotenv import load_dotenv, set_key
from easy_audio_interfaces.filesystem import RollingFileSink
from friend_lite.bluetooth import listen_to_omi, print_devices
from friend_lite.decoder import OmiOpusDecoder
from wyoming.audio import AudioChunk

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

env_path = ".env"
load_dotenv(env_path)

sys.path.append(os.path.dirname(__file__))
from send_to_adv import stream_to_backend

OMI_MAC = os.getenv("OMI_MAC")
if not OMI_MAC:
    logger.info("OMI_MAC not found in .env. Will try to find and set.")
    
# Standard Omi audio characteristic UUID
OMI_CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"

async def source_bytes(audio_queue: Queue[bytes]) -> AsyncGenerator[bytes, None]:
    """Single source iterator from the queue."""
    while True:
        chunk = await audio_queue.get()
        try:
            yield chunk
        finally:
            audio_queue.task_done()


async def as_audio_chunks(it) -> AsyncGenerator[AudioChunk, None]:
    """Convert bytes iterator to AudioChunk iterator."""
    async for data in it:
        yield AudioChunk(audio=data, rate=16000, width=2, channels=1)

# Add this to friend-lite sdk
async def list_devices(prefix: str = "OMI") -> list[BLEDevice]:
    devices = await BleakScanner.discover()
    filtered_devices = []
    for d in devices:
        if d.name:
            if prefix.casefold() in d.name.casefold():
                filtered_devices.append(d)
    return filtered_devices


def main() -> None:
    # api_key: str | None = os.getenv("DEEPGRAM_API_KEY")
    # if not api_key:
    #     print("Set your Deepgram API Key in the DEEPGRAM_API_KEY environment variable.")
    #     return

    audio_queue: Queue[bytes] = Queue()
    decoder = OmiOpusDecoder()

    def handle_ble_data(sender: Any, data: bytes) -> None:
        decoded_pcm: bytes = decoder.decode_packet(data)
        if decoded_pcm:
            try:
                audio_queue.put_nowait(decoded_pcm)
            except Exception as e:
                logger.error("Queue Error: %s", e)
                

    async def find_and_set_omi_mac() -> str:
        devices = await list_devices()
        assert len(devices) == 1, "Expected 1 Omi device, got %d" % len(devices)
        discovered_mac = devices[0].address
        set_key(env_path, "OMI_MAC", discovered_mac)
        logger.info("OMI_MAC set to %s and saved to .env" % discovered_mac)
        return discovered_mac

    async def run() -> None:
        logger.info("Starting OMI Bluetooth connection and audio streaming")
        if not OMI_MAC:
            mac_address = await find_and_set_omi_mac()
        else:
            mac_address = OMI_MAC
            logger.info("using OMI_MAC from .env: %s" % mac_address)

        # First, verify device is available by attempting to connect
        logger.info("Checking if device is available...")
        try:
            async with BleakClient(mac_address) as test_client:
                logger.info(f"Successfully connected to device {mac_address}")
        except Exception as e:
            logger.error(f"Failed to connect to device {mac_address}: {e}")
            logger.error("Exiting without creating audio sink or backend connection")
            return

        # Device is available, now setup audio sink and backend connection
        logger.info("Device found and connected, setting up audio pipeline...")

        # Create file sink now that we know device is available
        file_sink = RollingFileSink(
            directory="./audio_chunks",
            prefix="omi_audio",
            segment_duration_seconds=30,
            sample_rate=16000,
            channels=1,
            sample_width=2,
        )

        # Queue for backend streaming
        backend_queue = asyncio.Queue()

        async def process_audio():
            """Process audio chunks from BLE queue and duplicate to file + backend"""
            async for chunk_bytes in source_bytes(audio_queue):
                # Create AudioChunk for both destinations
                chunk = AudioChunk(audio=chunk_bytes, rate=16000, width=2, channels=1)

                # Write to file sink
                await file_sink.write(chunk)

                # Send to backend queue (non-blocking)
                await backend_queue.put(chunk)

        async def backend_stream_wrapper():
            """Wrapper to stream from backend_queue to backend"""
            async def queue_to_stream():
                while True:
                    chunk = await backend_queue.get()
                    if chunk is None:  # Sentinel value to stop
                        break
                    yield chunk

            try:
                await stream_to_backend(queue_to_stream())
            except Exception as e:
                logger.error(f"Backend streaming error: {e}", exc_info=True)

        async with file_sink:
            try:
                await asyncio.gather(
                    listen_to_omi(mac_address, OMI_CHAR_UUID, handle_ble_data),
                    process_audio(),
                    backend_stream_wrapper(),
                )
            except Exception as e:
                logger.error(f"Error in audio processing: {e}", exc_info=True)
            finally:
                # Signal backend stream to stop
                await backend_queue.put(None)

    asyncio.run(run())

if __name__ == '__main__':
    main()