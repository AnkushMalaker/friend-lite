import asyncio
import json
import sys

import numpy as np
import sounddevice as sd
from websockets.asyncio.client import connect

URI = "ws://192.168.0.110:8080/"

async def main():
    print(f"Connecting to {URI}")
    async with connect(URI) as ws:
        print("Connected")
        async def mic():
            with sd.InputStream(
                samplerate=16_000, channels=1, dtype=np.float32, blocksize=2048
            ) as stream:
                while True:
                    data, _ = stream.read(2048)
                    await ws.send(data.tobytes())
                    await asyncio.sleep(0.01)

        async def captions():
            async for msg in ws:
                if json.loads(msg)['final']:
                    payload = json.loads(msg)
                    sys.stdout.write("\r" + " " * 90 + "\r")
                    sys.stdout.write(payload["text"])
                    sys.stdout.flush()
                await asyncio.sleep(0.01)

        await asyncio.gather(mic(), captions())

asyncio.run(main())
