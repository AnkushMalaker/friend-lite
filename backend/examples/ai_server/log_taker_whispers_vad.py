import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

import fire
from dotenv import load_dotenv
from easy_audio_interfaces.audio_interfaces import RechunkingBlock, SocketReceiver
from easy_audio_interfaces.extras.models import WhisperBlock
from ez_wearable_backend.helper import FriendSocketReceiver
from faster_whisper.vad import VadOptions
from utils import llm_refactor, save_segment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()

RECORDING_DIR = (
    Path("friend-recordings")
    if (rec_dir := os.getenv("RECORDING_DIR")) is None
    else Path(rec_dir)
)
RECORDING_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = (
    Path("friend-logs") if (log_dir := os.getenv("LOG_DIR")) is None else Path(log_dir)
)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def chunk_size_from_seconds(seconds: int, sample_rate: int = 16000) -> int:
    return int(seconds * sample_rate)


async def main_async():
    SECONDS_TO_SEND_WHISPER = 40
    rechunking_block = RechunkingBlock(
        chunk_size=chunk_size_from_seconds(SECONDS_TO_SEND_WHISPER)
    )
    socket_receiver = FriendSocketReceiver(
        host="0.0.0.0",
        port=8081,
        sample_rate=16000,
    )
    await socket_receiver.open()
    whisper_block = WhisperBlock(
        model_description="large-v3", language="en", compute_type="int8"
    )
    # porcupine = Porcupine()

    audio_stream = rechunking_block.rechunk(socket_receiver)
    async for audio_chunk in audio_stream:
        logger.debug(f"Received audio chunk of length {len(audio_chunk)}")
        # Process the recorded audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        await save_segment(audio_chunk, RECORDING_DIR / f"recording_{timestamp}.wav")

        # Transcribe the audio
        transcription, _ = whisper_block.transcribe(
            audio_chunk,
            vad_filter=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2,
            vad_parameters=VadOptions(
                threshold=0.5,
                min_speech_duration_ms=250,
                max_speech_duration_s=float("inf"),
                min_silence_duration_ms=2000,
                speech_pad_ms=400,
            ),
            word_timestamps=True,
        )
        transcription_text = " ".join([t.text for t in transcription])
        logger.info(f"Transcription: {transcription_text}")

        # Refactor the transcription using LLM
        refactored_log = llm_refactor(
            transcription_text,
            chores=[],
        )
        logger.info("Refactored log:")
        print(refactored_log)

        # Save the refactored log
        log_file_path = LOG_DIR / f"log_{timestamp}.md"
        with open(log_file_path, "w") as log_file:
            log_file.write(refactored_log)
        logger.info(f"Saved refactored log to {log_file_path}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    fire.Fire(main)
