#!/usr/bin/env python3
"""
Wyoming ASR server using Nemo Parakeet ASR + Silero VAD.

Dependencies
------------
pip install numpy silero-vad nemo_toolkit[asr] soundfile wyoming easy_audio_interfaces pydub
"""

import argparse
import asyncio
import logging
import os
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Sequence, cast

import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
from easy_audio_interfaces.filesystem import LocalFileSink
from pydub import AudioSegment
from silero_vad import VADIterator, load_silero_vad
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncTcpServer
from nemo.collections.asr.models import ASRModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
SAMPLING_RATE = 16_000
CHUNK_SAMPLES = 512                       # Silero requirement (32 ms @ 16 kHz)
MAX_SPEECH_SECS = 30                     # Max duration of a speech segment
MIN_SPEECH_SECS = 0.2                    # Min duration for transcription

# --------------------------------------------------------------------------- #
class Transcriber:
    """Thin wrapper around Nemo ASR for synchronous calls."""

    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v2"):
        logger.info(f"Loading Nemo ASR model: {model_name}")
        self.model= cast(nemo_asr.models.ASRModel, nemo_asr.models.ASRModel.from_pretrained(model_name=model_name))
        self.rate = SAMPLING_RATE
        # Warm-up call
        logger.info("Warming up Nemo ASR model...")
        try:
            self.__call__(np.zeros(self.rate // 10, np.float32))  # 0.1s silence
            logger.info("Nemo ASR model warmed up successfully.")
        except Exception as e:
            logger.error(f"Error during ASR model warm-up: {e}")

    def __call__(self, pcm: np.ndarray) -> str:
        tmpfile_name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile_name = tmpfile.name
            
            sf.write(tmpfile_name, pcm, self.rate)
            
            results = self.model.transcribe([tmpfile_name], batch_size=1, timestamps=True)
            logger.debug(f"Transcription results: {results}")
            
            if results and len(results) > 0:
                # Check if the first result is an object with a .text attribute
                if hasattr(results[0], 'text') and results[0].text is not None:
                    return results[0].text
                # Check if the first result is directly a string (some models/configs)
                elif isinstance(results[0], str):
                    return results[0]
            return "" # Return empty string if transcription failed or no text
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""
        finally:
            if tmpfile_name and os.path.exists(tmpfile_name):
                os.remove(tmpfile_name)

# --------------------------------------------------------------------------- #
class ParakeetTranscriptionHandler(AsyncEventHandler):
    """
    Wyoming ASR handler for Parakeet offline transcription.
    """

    def __init__(self, *args, model_name: str = "nvidia/parakeet-tdt-0.6b-v2", **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name
        self._transcriber = Transcriber(model_name)
        self._vad_model = load_silero_vad(onnx=True)
        self._vad_iterator = VADIterator(
            model=self._vad_model,
            sampling_rate=SAMPLING_RATE,
            threshold=0.4,  # VAD sensitivity
        )
        
        self._speech_buf: list[AudioChunk] = []
        self._recording = False

        self._debug_dir = Path("debug")
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug file handling
        self._recording_debug_handle = None
        self._DEBUG_LENGTH = 30  # seconds
        self._cur_seg_duration = 0

    def soft_reset(self) -> None:
        """Reset only the iterator's state, not the underlying model."""
        self._vad_iterator.triggered = False
        self._vad_iterator.temp_end = 0
        self._vad_iterator.current_sample = 0
        logger.debug("VAD iterator soft reset.")

    def _chunk_to_numpy(self, chunk: AudioChunk) -> np.ndarray:
        if chunk.width == 2:
            logger.debug(f"Converting chunk to int16")
            return np.array(np.frombuffer(chunk.audio, dtype=np.int16), dtype=np.float32) / 32768.0
        elif chunk.width == 4:
            logger.debug(f"Converting chunk to float32")
            return np.array(np.frombuffer(chunk.audio, dtype=np.float32), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported width: {chunk.width}")

    async def _write_debug_file(self, event: AudioChunk) -> None:
        """Logic to create debug files"""
        create_audio_segment = partial(
            AudioSegment,
            sample_width=event.width,
            frame_rate=event.rate,
            channels=event.channels,
        )
        if self._recording_debug_handle is None:
            self._cur_seg_duration = 0
            self._recording_debug_handle = LocalFileSink(
                file_path=self._debug_dir / f"{time.time()}.wav",
                sample_rate=event.rate,
                channels=event.channels,
                sample_width=event.width,
            )
            await self._recording_debug_handle.open()
        await self._recording_debug_handle.write(create_audio_segment(event.audio))
        logger.debug(f"Wrote debug file: {self._recording_debug_handle._file_path}")
        self._cur_seg_duration += event.samples / event.rate
        logger.debug(f"Current segment duration: {self._cur_seg_duration} seconds")
        if self._cur_seg_duration > self._DEBUG_LENGTH:
            await self._recording_debug_handle.close()
            self._recording_debug_handle = None

    async def _process_audio_chunk(self, chunk: AudioChunk) -> None:
        """Process audio chunk with VAD and transcription."""
        # Write debug file
        await self._write_debug_file(chunk)
        
        chunk_array = self._chunk_to_numpy(chunk) # 512 samples, MAYBE?
        
        # Run VAD on this chunk
        try:
            vad_event = self._vad_iterator(chunk_array)
            logger.debug(f"VAD event: {vad_event}")
        except Exception as e:
            logger.error(f"Error during VAD: {e}")
            return

        if vad_event:
            logger.info(f"VAD event: {vad_event}")
            
            if "start" in vad_event and not self._recording:
                # Start recording - initialize buffer with current chunk
                self._recording = True
                self._speech_buf.append(chunk)
                logger.info("Started recording speech")

            elif "end" in vad_event and self._recording:
                # End recording - add final chunk and transcribe
                self._speech_buf.append(chunk)
                self._recording = False
                
                speech_duration = sum(chunk.seconds for chunk in self._speech_buf)
                logger.info(f"VAD end detected. Speech duration: {speech_duration:.2f}s")
                
                if speech_duration >= MIN_SPEECH_SECS:
                    await self._transcribe_and_send(self._speech_buf)
                else:
                    logger.info("Speech too short for transcription. Discarding.")
                
                # Clear buffer and reset
                self._speech_buf.clear()
                self.soft_reset()
                return
        
        # If we're recording, accumulate the chunk
        if self._recording:
            self._speech_buf.append(chunk)
            
            # Safety check: if speech buffer gets too long, force transcription
            if len(self._speech_buf) >= MAX_SPEECH_SECS * SAMPLING_RATE:
                logger.warning(f"Max speech length {MAX_SPEECH_SECS}s reached. Forcing transcription.")
                await self._transcribe_and_send(self._speech_buf)
                self._speech_buf.clear()
                self._recording = False
                self.soft_reset()

    async def _transcribe_and_send(self, speech: Sequence[AudioChunk]) -> None:
        """Transcribe speech and send Wyoming transcript event."""
        try:
            # Run blocking ASR call in a separate thread
            speech_array = np.concatenate([self._chunk_to_numpy(chunk) for chunk in speech])
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self._transcriber, speech_array)
            
            if not text:  # If transcription is empty or failed
                logger.warning("Transcription resulted in empty text. Not sending.")
                return

            logger.info(f"Transcription result: '{text}'")

            # Send Wyoming transcript event
            transcript = Transcript(text=text)
            await self.write_event(transcript.event())
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming protocol events"""
        if Transcribe.is_type(event.type):
            # Reset for new transcription request
            self._recording = False
            self._speech_buf.clear()
            self.soft_reset()
            return True
        elif AudioStart.is_type(event.type):
            # Start a new audio stream
            logger.info("Audio stream started")
            return True
        elif AudioChunk.is_type(event.type):
            # Process audio chunk
            audio_chunk = AudioChunk.from_event(event)
            await self._process_audio_chunk(audio_chunk)
            return True
        elif AudioStop.is_type(event.type):
            # End of audio stream
            logger.info("Audio stream stopped")
            # If we have accumulated speech, transcribe it
            if self._recording and len(self._speech_buf) >= MIN_SPEECH_SECS * SAMPLING_RATE:
                logger.info("Audio stream ended. Transcribing remaining speech.")
                await self._transcribe_and_send(self._speech_buf)
            
            # Reset state
            self._speech_buf.clear()
            self._recording = False
            self.soft_reset()
            return True
        elif Describe.is_type(event.type):
            # Respond with service capabilities
            model = AsrModel(
                name=self._model_name,
                attribution=Attribution(
                    name="Nemo Parakeet ASR",
                    url="https://github.com/NVIDIA/NeMo"
                ),
                installed=True,
                description="Nemo Parakeet ASR model",
                version="1.0.0",
                languages=["en"]
            )
            
            program = AsrProgram(
                name="parakeet-asr",
                attribution=Attribution(
                    name="Nemo Parakeet ASR",
                    url="https://github.com/NVIDIA/NeMo"
                ),
                installed=True,
                description="Nemo Parakeet ASR with Silero VAD",
                version="1.0.0",
                models=[model]
            )
            
            info = Info(asr=[program])
            await self.write_event(info.event())
            return True
        
        return False

    async def disconnect(self) -> None:
        """Clean up debug file handle on disconnect"""
        if self._recording_debug_handle is not None:
            await self._recording_debug_handle.close()

# --------------------------------------------------------------------------- #
async def main() -> None:
    parser = argparse.ArgumentParser(description="Nemo Parakeet ASR Wyoming Server")
    parser.add_argument(
        "--model_name",
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="Nemo ASR model name from HuggingFace or NGC (default: nvidia/parakeet-tdt-0.6b-v2)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Interface to bind the TCP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Port to bind the TCP server (default: 8765)"
    )
    args = parser.parse_args()

    server = AsyncTcpServer(host=args.host, port=args.port)
    logger.info(
        f"Parakeet ASR service starting on {args.host}:{args.port} (model={args.model_name})"
    )
    await server.run(
        handler_factory=lambda *_args, **_kwargs: ParakeetTranscriptionHandler(
            *_args, model_name=args.model_name, **_kwargs
        )
    )

if __name__ == "__main__":
    # Set up logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger(__name__)  # Re-initialize logger with new format for main scope
    
    asyncio.run(main())
