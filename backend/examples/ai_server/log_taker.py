import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import fire
import numpy as np
import requests
from dotenv import load_dotenv
from easy_audio_interfaces.audio_interfaces import RechunkingBlock, SocketReceiver
from easy_audio_interfaces.extras.models import SileroVad, WhisperBlock
from easy_audio_interfaces.types import NumpyFrame
from ez_wearable_backend.helper import FriendSocketReceiver
from prompt_template import log_taker_prompt_tmpl_str
from utils import llm_refactor, record_till_silence, save_segment, wait_for_voice

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

SERVER_URL = os.getenv("SERVER_URL") or "http://127.0.0.1:8080"


def llm_refactor(transcription_text: str, chores: Optional[List[str]] = None) -> str:
    # Prepare the prompt for the LLM
    prompt = log_taker_prompt_tmpl_str.format(
        transcript=transcription_text, chores=chores
    )

    # Prepare the payload for the API request
    payload = {"query": prompt}

    try:
        response = requests.post(f"{SERVER_URL}/predict", json=payload)
        response.raise_for_status()

        response_data = response.json()

        # Extract the refactored text from the response
        if isinstance(response_data, dict):
            result = response_data.get(
                "response", response_data.get("output", response_data)
            )
        else:
            result = response_data

        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending request to LLM service: {e}")
        return f"Error: Unable to refactor transcription. {str(e)}"


async def main_async():
    rechunking_block = RechunkingBlock(chunk_size=512)
    socket_receiver = FriendSocketReceiver(
        host="0.0.0.0",
        port=8081,
        sample_rate=16000,
    )
    await socket_receiver.open()
    vad = SileroVad(sampling_rate=16000)
    whisper_block = WhisperBlock(
        model_description="distil-large-v3", language="en", compute_type="int8"
    )
    # porcupine = Porcupine()
    silence_threshold_seconds = 3
    timeout_seconds = 120

    audio_stream = rechunking_block.rechunk(socket_receiver)
    while True:
        voice_chunk = await wait_for_voice(
            audio_stream, vad, min_seconds_with_voice=1, threshold=0.5
        )
        if voice_chunk is None:
            continue
        logger.info("Voice detected, starting to record")
        recorded_audio = await record_till_silence(
            audio_stream,
            vad,
            silence_threshold_seconds=silence_threshold_seconds,
            timeout_seconds=timeout_seconds,
        )
        recorded_audio = NumpyFrame(np.concatenate([voice_chunk, recorded_audio]))

        # Process the recorded audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        await save_segment(recorded_audio, RECORDING_DIR / f"recording_{timestamp}.wav")

        # Transcribe the audio
        transcription, _ = whisper_block.transcribe(
            recorded_audio,
            # hotwords="whisper"
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
