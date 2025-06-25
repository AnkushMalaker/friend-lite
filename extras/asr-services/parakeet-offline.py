#!/usr/bin/env python3
"""
Wyoming ASR server using Nemo Parakeet ASR + Silero VAD.

Dependencies
------------
pip install numpy silero-vad nemo_toolkit[asr] soundfile wyoming easy_audio_interfaces pydub
"""

import argparse
import asyncio
import csv
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Sequence, cast

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from easy_audio_interfaces.filesystem import LocalFileSink
from silero_vad import VADIterator, load_silero_vad
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncTcpServer
from wyoming.vad import VoiceStarted, VoiceStopped

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
SAMPLING_RATE = 16_000
CHUNK_SAMPLES = 512  # Silero requirement (32 ms @ 16 kHz)
MAX_SPEECH_SECS = 30  # Max duration of a speech segment
MIN_SPEECH_SECS = 0.5  # Min duration for transcription


def _chunk_to_numpy_float(chunk: AudioChunk) -> np.ndarray:
    if chunk.width == 2:
        logger.debug(f"Converting chunk to float32")
        return (
            np.array(np.frombuffer(chunk.audio, dtype=np.int16), dtype=np.float32)
            / np.iinfo(np.int16).max
        )
    elif chunk.width == 4:
        logger.debug(f"Converting chunk to float32")
        return (
            np.array(np.frombuffer(chunk.audio, dtype=np.int32), dtype=np.float32)
            / np.iinfo(np.int32).max
        )
    else:
        raise ValueError(f"Unsupported width: {chunk.width}")


# --------------------------------------------------------------------------- #
class SharedTranscriber:
    """Shared transcriber instance that can be used by multiple clients."""

    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v2"):
        logger.info(f"Loading shared Nemo ASR model: {model_name}")
        self.model = cast(
            nemo_asr.models.ASRModel,
            nemo_asr.models.ASRModel.from_pretrained(model_name=model_name),
        )
        self._rate = SAMPLING_RATE
        self._model_name = model_name
        self._lock = asyncio.Lock()

        # Note: warmup will be called separately after initialization

    async def warmup(self) -> None:
        """Warm up the ASR model."""
        logger.info("Warming up shared Nemo ASR model...")
        try:
            await self._transcribe(
                [
                    AudioChunk(
                        audio=np.zeros(self._rate // 10, np.int32).tobytes(),
                        rate=self._rate,
                        channels=1,
                        width=2,
                    )
                ]
            )  # 0.1s silence
            logger.info("Shared Nemo ASR model warmed up successfully.")
        except Exception as e:
            logger.error(f"Error during ASR model warm-up: {e}")

    async def _transcribe(self, speech: Sequence[AudioChunk]) -> str:
        """Internal transcription method."""

        assert len(speech) > 0
        sample_rate = speech[0].rate
        channels = speech[0].channels
        sample_width = speech[0].width
        tmpfile_name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile_name = tmpfile.name

            logger.info(f"Writing to file: {tmpfile_name}")
            sink = LocalFileSink(
                file_path=tmpfile_name,
                sample_rate=sample_rate,
                channels=channels,
                sample_width=sample_width,
            )
            assert self._rate == sample_rate
            await sink.open()
            for chunk in speech:
                await sink.write(chunk)
            await sink.close()

            with torch.no_grad():
                results = self.model.transcribe( # type: ignore
                    [tmpfile_name], batch_size=1, timestamps=True
                )
            logger.debug(f"Transcription results: {results}")

            if results and len(results) > 0:
                # Check if the first result is an object with a .text attribute
                if hasattr(results[0], "text") and results[0].text is not None:
                    return results[0].text
                # Check if the first result is directly a string (some models/configs)
                elif isinstance(results[0], str):
                    return results[0]
            return ""  # Return empty string if transcription failed or no text
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return ""
        finally:
            if tmpfile_name and os.path.exists(tmpfile_name):
                # os.remove(tmpfile_name)
                logger.warning(f"Not removing tmpfile: {tmpfile_name}")
                pass

    async def transcribe_async(self, speech: Sequence[AudioChunk]) -> str:
        """Thread-safe async transcription method."""

        async with self._lock:
            return await self._transcribe(speech)


# --------------------------------------------------------------------------- #
class ParakeetTranscriptionHandler(AsyncEventHandler):
    """
    Wyoming ASR handler for Parakeet offline transcription.
    """

    def __init__(
        self,
        *args,
        shared_transcriber: SharedTranscriber,
        vad_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._transcriber = shared_transcriber
        self._model_name = shared_transcriber._model_name
        self._vad_enabled = vad_enabled

        # VAD-related initialization
        if self._vad_enabled:
            self._vad_model = load_silero_vad(onnx=True)
            self._vad_iterator = VADIterator(
                model=self._vad_model,
                sampling_rate=SAMPLING_RATE,
                threshold=0.4,  # VAD sensitivity
                min_silence_duration_ms=1000,  # Minimum silence duration in ms
            )
        else:
            self._vad_model = None
            self._vad_iterator = None

        self._speech_buf: list[AudioChunk] = []
        self._recording = False

        # Non-VAD mode: collect everything from AudioStart to AudioStop
        self._collecting_audio = False

        # VAD sample buffering - ensure exactly 512 samples per VAD call
        self._vad_sample_buffer = np.array([], dtype=np.float32)
        self._vad_buffer_size = CHUNK_SAMPLES

        self._debug_dir = Path("debug")
        self._debug_dir.mkdir(parents=True, exist_ok=True)

        # Results directory for CSV logging
        self._results_dir = Path("results")
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # CSV file for logging transcriptions
        self._csv_file = self._results_dir / "transcription_results.csv"
        self._csv_lock = asyncio.Lock()
        self._init_csv_file()

        # Debug file handling
        self._recording_debug_handle = None
        self._current_debug_file_path = None
        self._DEBUG_LENGTH = 30  # seconds
        self._cur_seg_duration = 0

    def _init_csv_file(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not self._csv_file.exists():
            with open(self._csv_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ["timestamp", "audio_path", "transcription", "ground_truth"]
                )
                logger.info(f"Created CSV file: {self._csv_file}")

    async def _log_to_csv(self, audio_path: str, transcription: str) -> None:
        """Log transcription result to CSV file."""
        async with self._csv_lock:
            try:
                with open(self._csv_file, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [time.time(), audio_path, transcription, ""]
                    )  # Empty ground_truth
                    logger.info(f"Logged to CSV: {audio_path} -> '{transcription}'")
            except Exception as e:
                logger.error(f"Error writing to CSV: {e}")

    def soft_reset(self) -> None:
        """Reset only the iterator's state, not the underlying model."""
        if self._vad_enabled and self._vad_iterator:
            self._vad_iterator.triggered = False
            self._vad_iterator.temp_end = 0
            self._vad_iterator.current_sample = 0
            # Clear the VAD sample buffer
            self._vad_sample_buffer = np.array([], dtype=np.float32)
            logger.debug("VAD iterator soft reset.")

    async def _write_debug_file(self, event: AudioChunk) -> None:
        """Logic to create debug files"""
        if self._recording_debug_handle is None:
            self._cur_seg_duration = 0
            self._current_debug_file_path = self._debug_dir / f"{time.time()}.wav"
            logger.info(f"Writing debug file: {self._current_debug_file_path}\n\
                        with rate: {event.rate}\n\
                        channels: {event.channels}\n\
                        width: {event.width}\n\
                        samples: {event.samples}\n\
                        seconds: {event.seconds}\n\
                        ")
            self._recording_debug_handle = LocalFileSink(
                file_path=self._current_debug_file_path,
                sample_rate=event.rate,
                channels=event.channels,
                sample_width=event.width,
            )
            await self._recording_debug_handle.open()
        await self._recording_debug_handle.write(event)
        logger.debug(f"Wrote debug file: {self._recording_debug_handle._file_path}")
        self._cur_seg_duration += event.samples / event.rate
        logger.debug(f"Current segment duration: {self._cur_seg_duration} seconds")
        if self._cur_seg_duration > self._DEBUG_LENGTH:
            await self._recording_debug_handle.close()
            self._recording_debug_handle = None
            self._current_debug_file_path = None

    async def _process_audio_chunk(self, chunk: AudioChunk) -> None:
        """Process audio chunk with VAD and transcription."""
        if self._vad_enabled:
            # VAD-enabled mode: use VAD to detect speech segments
            await self._buffer_and_process_vad(chunk)
        else:
            # VAD-disabled mode: collect all audio from start to stop
            await self._write_debug_file(chunk)
            await self._process_without_vad(chunk)

    async def _process_without_vad(self, chunk: AudioChunk) -> None:
        """Process audio chunk without VAD - collect everything from start to stop."""
        if self._collecting_audio:
            self._speech_buf.append(chunk)

            # Safety check: if speech buffer gets too long, force transcription
            speech_duration = sum(c.seconds for c in self._speech_buf)
            if speech_duration >= MAX_SPEECH_SECS:
                logger.warning(
                    f"Max speech length {MAX_SPEECH_SECS}s reached. Forcing transcription."
                )
                await self._transcribe_and_send(self._speech_buf)
                self._speech_buf.clear()

    async def _transcribe_and_send(self, speech: Sequence[AudioChunk]) -> None:
        """Transcribe speech and send Wyoming transcript event."""
        try:
            # Run blocking ASR call using the shared transcriber
            logger.info(
                f"audio_chunk_details:\
                   audio length: {len(speech[0].audio)}\
                   audio rate: {speech[0].rate}\
                   audio channels: {speech[0].channels}\
                   audio width: {speech[0].width}\
                   audio samples: {speech[0].samples}\
                   audio seconds: {speech[0].seconds}\
                   total samples: {sum(chunk.samples for chunk in speech)}\
                   total chunks: {len(speech)}\
                   "
            )
            text = await self._transcriber.transcribe_async(speech)
            sample_rate = speech[0].rate
            channels = speech[0].channels
            sample_width = speech[0].width

            if not text:  # If transcription is empty or failed
                logger.warning("Transcription resulted in empty text. Not sending.")
                return

            logger.info(f"Transcription result: '{text}'")

            # Log to CSV if we have a current debug file path
            if self._current_debug_file_path:
                await self._log_to_csv(str(self._current_debug_file_path), text)
            else:
                # Create a temporary audio file for this transcription segment
                temp_audio_path = self._debug_dir / f"transcription_{time.time()}.wav"
                try:
                    # Save the audio segment for CSV logging
                    sink = LocalFileSink(
                        file_path=temp_audio_path,
                        sample_rate=sample_rate,
                        channels=channels,
                        sample_width=sample_width,
                    )
                    await sink.open()
                    for chunk in speech:
                        await sink.write(chunk)
                    await sink.close()
                    await self._log_to_csv(str(temp_audio_path), text)
                except Exception as e:
                    logger.error(f"Error saving temporary audio file: {e}")
                    # Log with timestamp as fallback
                    await self._log_to_csv(f"temp_audio_{time.time()}", text)

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
            self._collecting_audio = False
            if len(self._speech_buf) > 0:
                logger.warning(
                    f"Clearing speech buffer of {len(self._speech_buf)} chunks"
                )
            self._speech_buf.clear()
            self.soft_reset()
            return True
        elif AudioStart.is_type(event.type):
            # Start a new audio stream
            logger.info("Audio stream started")
            if not self._vad_enabled:
                # In non-VAD mode, start collecting all audio
                self._collecting_audio = True
                self._speech_buf.clear()
                logger.info("Started collecting audio (VAD disabled)")
            return True
        elif AudioChunk.is_type(event.type):
            # Process audio chunk
            audio_chunk = AudioChunk.from_event(event)
            await self._process_audio_chunk(audio_chunk)
            return True
        elif AudioStop.is_type(event.type):
            # End of audio stream
            logger.info("Audio stream stopped")

            if self._vad_enabled:
                # Flush any remaining samples in the VAD buffer
                await self._flush_vad_buffer()

                # VAD mode: transcribe any remaining speech if we were recording
                if (
                    self._recording
                    and len(self._speech_buf) >= MIN_SPEECH_SECS * SAMPLING_RATE
                ):
                    logger.info("Audio stream ended. Transcribing remaining speech.")
                    await self._transcribe_and_send(self._speech_buf)
            else:
                # Non-VAD mode: transcribe the entire collected audio
                if self._collecting_audio and self._speech_buf:
                    speech_duration = sum(chunk.seconds for chunk in self._speech_buf)
                    logger.info(
                        f"Audio stream ended. Transcribing entire segment (duration: {speech_duration:.2f}s)"
                    )
                    if speech_duration >= MIN_SPEECH_SECS:
                        await self._transcribe_and_send(self._speech_buf)
                    else:
                        logger.info(
                            "Audio segment too short for transcription. Discarding."
                        )

            # Reset state
            self._speech_buf.clear()
            self._recording = False
            self._collecting_audio = False
            self.soft_reset()
            return True
        elif Describe.is_type(event.type):
            # Respond with service capabilities
            model = AsrModel(
                name=self._model_name,
                attribution=Attribution(
                    name="Nemo Parakeet ASR", url="https://github.com/NVIDIA/NeMo"
                ),
                installed=True,
                description="Nemo Parakeet ASR model",
                version="1.0.0",
                languages=["en"],
            )

            program = AsrProgram(
                name="parakeet-asr",
                attribution=Attribution(
                    name="Nemo Parakeet ASR", url="https://github.com/NVIDIA/NeMo"
                ),
                installed=True,
                description="Nemo Parakeet ASR with Silero VAD",
                version="1.0.0",
                models=[model],
            )

            info = Info(asr=[program])
            await self.write_event(info.event())
            return True

        return False

    async def disconnect(self) -> None:
        """Clean up debug file handle on disconnect"""
        if self._recording_debug_handle is not None:
            await self._recording_debug_handle.close()

    async def _process_vad_samples(self, samples: np.ndarray) -> None:
        """Process exactly 512 samples with VAD iterator."""
        if len(samples) != self._vad_buffer_size:
            logger.warning(
                f"Expected {self._vad_buffer_size} samples, got {len(samples)}"
            )
            return

        try:
            assert self._vad_iterator is not None
            vad_event = self._vad_iterator(samples)
            logger.debug(f"VAD event: {vad_event}")
        except Exception as e:
            logger.error(f"Error during VAD: {e}")
            return

        if vad_event:
            logger.info(f"VAD event: {vad_event}")

            if "start" in vad_event and not self._recording:
                # Start recording
                self._recording = True
                await self.write_event(VoiceStarted().event())
                logger.info("Started recording speech")

            elif "end" in vad_event and self._recording:
                # End recording and transcribe
                self._recording = False
                await self.write_event(VoiceStopped().event())

                speech_duration = sum(chunk.seconds for chunk in self._speech_buf)
                logger.info(
                    f"VAD end detected. Speech duration: {speech_duration:.2f}s"
                )

                if speech_duration >= MIN_SPEECH_SECS:
                    await self._transcribe_and_send(self._speech_buf)
                else:
                    logger.info("Speech too short for transcription. Discarding.")

                # Clear buffer and reset
                self._speech_buf.clear()
                self.soft_reset()

    async def _buffer_and_process_vad(
        self, chunk: AudioChunk
    ) -> None:
        """Buffer audio samples and process VAD with exactly 512 samples at a time."""
        # Add new samples to buffer
        chunk_array = _chunk_to_numpy_float(chunk)
        self._vad_sample_buffer = np.concatenate([self._vad_sample_buffer, chunk_array])

        # Write debug file once per chunk (not per VAD iteration)
        await self._write_debug_file(chunk)

        # Process complete 512-sample chunks
        while len(self._vad_sample_buffer) >= self._vad_buffer_size:
            # Extract exactly 512 samples
            samples_to_process = self._vad_sample_buffer[: self._vad_buffer_size]

            # Process with VAD
            await self._process_vad_samples(samples_to_process)

            # Remove processed samples from buffer
            self._vad_sample_buffer = self._vad_sample_buffer[self._vad_buffer_size :]

        # If we're recording, accumulate the original chunk once per chunk
        if self._recording:
            self._speech_buf.append(chunk)

            # Safety check: if speech buffer gets too long, force transcription
            speech_duration = sum(c.seconds for c in self._speech_buf)
            if speech_duration >= MAX_SPEECH_SECS:
                logger.warning(
                    f"Max speech length {MAX_SPEECH_SECS}s reached. Forcing transcription."
                )
                await self._transcribe_and_send(self._speech_buf)
                self._speech_buf.clear()
                self._recording = False
                self.soft_reset()

    async def _flush_vad_buffer(self) -> None:
        """Process any remaining samples in the VAD buffer at stream end."""
        if len(self._vad_sample_buffer) > 0:
            logger.debug(
                f"Flushing {len(self._vad_sample_buffer)} remaining samples from VAD buffer"
            )

            # Pad with zeros to reach 512 samples if needed
            if len(self._vad_sample_buffer) < self._vad_buffer_size:
                padding_needed = self._vad_buffer_size - len(self._vad_sample_buffer)
                self._vad_sample_buffer = np.concatenate(
                    [
                        self._vad_sample_buffer,
                        np.zeros(padding_needed, dtype=np.float32),
                    ]
                )

            # Process the final chunk
            await self._process_vad_samples(self._vad_sample_buffer)

            # Clear the buffer
            self._vad_sample_buffer = np.array([], dtype=np.float32)


# --------------------------------------------------------------------------- #
async def main() -> None:
    parser = argparse.ArgumentParser(description="Nemo Parakeet ASR Wyoming Server")
    parser.add_argument(
        "--model_name",
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="Nemo ASR model name from HuggingFace or NGC (default: nvidia/parakeet-tdt-0.6b-v2)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface to bind the TCP server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind the TCP server (default: 8765)",
    )
    parser.add_argument(
        "--disable-vad",
        action="store_true",
        help="Disable VAD and use normal Wyoming protocol (default: VAD enabled)",
    )
    args = parser.parse_args()

    # Create shared transcriber instance once
    logger.info("Initializing shared transcriber...")
    shared_transcriber = SharedTranscriber(args.model_name)
    logger.info("Shared transcriber initialized successfully")
    
    # Perform async warmup
    try:
        await shared_transcriber.warmup()
    except Exception as e:
        logger.error(f"Error during ASR model warm-up: {e}")

    server = AsyncTcpServer(host=args.host, port=args.port)
    vad_enabled = not args.disable_vad
    logger.info(
        f"Parakeet ASR service starting on {args.host}:{args.port} (model={args.model_name}, VAD={'enabled' if vad_enabled else 'disabled'})"
    )
    await server.run(
        handler_factory=lambda *_args, **_kwargs: ParakeetTranscriptionHandler(
            *_args,
            shared_transcriber=shared_transcriber,
            vad_enabled=vad_enabled,
            **_kwargs,
        )
    )


if __name__ == "__main__":
    # Set up logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger(
        __name__
    )  # Re-initialize logger with new format for main scope

    asyncio.run(main())
