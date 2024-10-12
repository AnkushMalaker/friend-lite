import asyncio
import logging
import wave
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
from easy_audio_interfaces.audio_interfaces import RechunkingBlock
from easy_audio_interfaces.types import NumpyFrame
from ez_wearable_backend.helper import FriendSocketReceiver
from opuslib import Decoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

decoder = Decoder(fs=16000, channels=1)

RECORDING_DIR = Path("recordings")
RECORDING_DIR.mkdir(parents=True, exist_ok=True)


def save_segment(segment: NumpyFrame, path: Path | str):
    path = Path(path)
    with wave.open(str(path), "wb") as wav_file:
        # Set the WAV file parameters
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample for 16-bit audio
        wav_file.setframerate(16000)  # Sample rate of 16kHz

        # Ensure the audio data is in the correct format (16-bit int)
        audio_data = segment.astype(np.int16).tobytes()

        # Write the audio data to the WAV file
        wav_file.writeframes(audio_data)

    logger.info(f"Saved audio segment to {path}")


async def main_async():
    RECORD_MINUTES = 1
    rechunking_block = RechunkingBlock(chunk_size=512)
    socket_receiver = FriendSocketReceiver(
        host="0.0.0.0",
        port=8081,
        sample_rate=16000,
    )
    await socket_receiver.open()

    # each frame is 512 samples at 16000 Hz
    # meaning  seconds per frame
    audio_stream = rechunking_block.rechunk(socket_receiver)
    async for audio_chunk in audio_stream:
        # Calculate the duration of the segment in seconds
        duration = len(audio_chunk) / 16000  # 16000 is the sample rate

        if duration >= RECORD_MINUTES * 60:
            logger.info(f"Audio shape: {audio_chunk.shape}")
            logger.info(f"Duration: {duration:.2f} seconds")

            save_segment(audio_chunk, RECORDING_DIR / f"segment_{datetime.now()}.wav")
            logger.info("---")
        else:
            logger.info(f"Ignoring short segment (duration: {duration:.2f} seconds)")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    fire.Fire(main)
