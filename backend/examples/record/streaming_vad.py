import asyncio
import logging
import wave
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
from easy_audio_interfaces.audio_interfaces import RechunkingBlock
from easy_audio_interfaces.extras.models import SileroVad, VoiceGate, WhisperBlock
from easy_audio_interfaces.types import NumpySegment
from ez_wearable_backend.helper import FriendSocketReceiver
from opuslib import Decoder
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

decoder = Decoder(fs=16000, channels=1)

RECORDING_DIR = Path("recordings")
RECORDING_DIR.mkdir(parents=True, exist_ok=True)


class SpeechSegment(BaseModel, arbitrary_types_allowed=True):
    audio: NumpySegment
    start_time: datetime
    end_time: datetime


def save_segment(segment: NumpySegment, path: Path | str):
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
    silero = SileroVad(sampling_rate=16000)
    rechunking_block = RechunkingBlock(chunk_size=512)
    socket_receiver = FriendSocketReceiver(
        host="0.0.0.0",
        port=8081,
        sample_rate=16000,
    )
    await socket_receiver.open()

    # each frame is 512 samples at 16000 Hz
    # meaning  seconds per frame
    voice_gate = VoiceGate.init_from_seconds(
        sample_rate=16000,
        starting_patience_seconds=1,
        stopping_patience_seconds=2,
        cool_down_seconds=1,
        threshold=0.5,
    )
    whisper_block = WhisperBlock(model_description="distil-large-v3", language="en")

    audio_stream = rechunking_block.rechunk(socket_receiver)
    voice_segments = silero.iter_segments(audio_stream, voice_gate=voice_gate)
    async for voice_segment in voice_segments:
        # Calculate the duration of the segment in seconds
        duration = len(voice_segment) / 16000  # 16000 is the sample rate

        if duration >= 1.5:
            logger.info(f"VoiceSegment object created")
            logger.info(f"Audio shape: {voice_segment.shape}")
            logger.info(f"Duration: {duration:.2f} seconds")

            save_segment(voice_segment, RECORDING_DIR / f"segment_{datetime.now()}.wav")
            transcription_segments, _ = whisper_block.transcribe(
                np.array(voice_segment, dtype=np.float32),
            )
            logger.info(list(transcription_segments))
            logger.info("---")
        else:
            logger.info(f"Ignoring short segment (duration: {duration:.2f} seconds)")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    fire.Fire(main)
