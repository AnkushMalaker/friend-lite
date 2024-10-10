import logging
from pathlib import Path
from typing import AsyncIterable, List, Optional

import numpy as np
import requests
import torch
from constants import SERVER_URL
from easy_audio_interfaces.audio_interfaces import LocalFileSink
from easy_audio_interfaces.extras.models import SileroVad
from easy_audio_interfaces.types import NumpyFrame
from prompt_template import log_taker_prompt_tmpl_str

logger = logging.getLogger(__name__)


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


async def save_segment(segment: np.ndarray, path: Path | str):
    async with LocalFileSink(path, sample_rate=16000, channels=1) as sink:
        await sink.write(NumpyFrame(segment))

    logger.info(f"Saved audio segment to {path}")


async def record_till_silence(
    audio_stream: AsyncIterable[NumpyFrame],
    vad: SileroVad,
    silence_threshold_seconds: float = 2,
    vad_threshold: float = 0.5,
    sample_rate: int = 16000,
    timeout_seconds: float = 360,
) -> NumpyFrame:
    allowed_silence_samples = int(silence_threshold_seconds * sample_rate)
    silence_count = 0
    audio_chunks: List[NumpyFrame] = []

    total_duration = 0

    async for audio_chunk in audio_stream:
        logger.info("audio chunk in record_till_silence")

        audio_chunks.append(audio_chunk)
        total_duration += len(audio_chunk) / sample_rate

        with torch.inference_mode():
            # input_tensor = torch.tensor(audio_chunk.normalize(), dtype=torch.float32)
            prob = vad(audio_chunk)
        logger.info(f"Waiting for silence: {prob}. Silence count: {silence_count}")
        if prob < vad_threshold:
            silence_count += len(audio_chunk)
        else:
            silence_count = 0

        # Use inertia threshold if recording is longer than 5 second
        current_silence_threshold = allowed_silence_samples

        if silence_count > current_silence_threshold:
            logger.info("Silence detected, stopping recording")
            break

        if total_duration > timeout_seconds:
            logger.info("Recording timed out")
            break

    return NumpyFrame(np.concatenate(audio_chunks))


async def wait_for_voice(
    audio_stream: AsyncIterable[NumpyFrame],
    vad: SileroVad,
    min_seconds_with_voice: int = 1,  # Maybe we should make this min samples in a window
    threshold: float = 0.5,
) -> Optional[NumpyFrame]:
    min_samples_with_voice = int(min_seconds_with_voice * 16000)
    samples_with_voice = 0
    voice_chunks: List[NumpyFrame] = []
    async for audio_chunk in audio_stream:
        with torch.inference_mode():
            prob = vad(audio_chunk)
        logger.info(f"Koi kuch bola kya?: I'm {prob}% sure.")
        if prob > threshold:
            samples_with_voice += len(audio_chunk)
            voice_chunks.append(audio_chunk)
        if samples_with_voice > min_samples_with_voice:
            return NumpyFrame(np.concatenate(voice_chunks))
