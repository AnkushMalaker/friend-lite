import asyncio
import logging

from easy_audio_interfaces.extras.local_audio import InputMicStream
from wyoming.audio import AudioChunk
from wyoming.client import AsyncTcpClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

URI = "ws://192.168.0.110:8080/"
URI = "tcp://localhost:8765/"


async def main():
    print(f"Connecting to {URI}")
    async with AsyncTcpClient.from_uri(URI) as client:
        print("Connected")
        async def mic():
            async with InputMicStream(chunk_size=512) as stream:
                while True:
                    data = await stream.read()
                    await client.write_event(
                        AudioChunk(
                            audio=data.raw_data, # type: ignore
                            width=2,
                            rate=16_000,
                            channels=1,
                        ).event()
                    )
                    logger.debug(f"Sent audio chunk: {len(data.raw_data)} bytes")
                    await asyncio.sleep(0.01)

        async def captions():
            while True:
                event = await client.read_event()
                if event is None:
                    break
                print(f"Received event: {event}")

        await asyncio.gather(mic(), captions())

asyncio.run(main())
