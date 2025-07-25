#!/usr/bin/env python3
"""
WebSocket live-caption server using Moonshine ONNX + Silero VAD.

▸  Client  →  binary frames: 16-kHz mono float32 PCM
▸  Server  →  text frames   : {"text": "<caption>", "final": true|false}

Dependencies
------------
pip install websockets sounddevice numpy silero-vad moonshine-onnx
"""

import argparse
import asyncio
import logging
import time
from asyncio import Queue
from pathlib import Path

import numpy as np
from easy_audio_interfaces.filesystem import LocalFileSink
from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
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

LOOKBACK_CHUNKS = 5  # prepend a little context
MAX_SPEECH_SECS = 5
MIN_REFRESH_SECS = 0.20


class StreamingTranscript(Transcript):
    def __init__(self, text: str, final: bool):
        super().__init__(text=text)
        self.final = final

    def event(self) -> Event:
        data = super().event().data
        data["final"] = self.final
        return Event(type=super().event().type, data=data)


# --------------------------------------------------------------------------- #
class Transcriber:
    """Thin wrapper around Moonshine ONNX for synchronous calls."""

    def __init__(self, model_name: str):
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.tokenizer = load_tokenizer()

        self.rate = SAMPLING_RATE
        self.__call__(np.zeros(self.rate, np.float32))  # warm-up

    def __call__(self, pcm: np.ndarray) -> str:
        tokens = self.model.generate(pcm[np.newaxis, :].astype(np.float32))
        return self.tokenizer.decode_batch(tokens)[0]
        
    def transcribe_with_vad(self, pcm: np.ndarray, vad_event: dict) -> str:
        """Transcribe audio with VAD event context."""
        return self.__call__(pcm)



# --------------------------------------------------------------------------- #
# Currently the decision that a final transcript is found is done via VAD.
# Could be some other thing later
class StreamingTranscriptionHandler(AsyncEventHandler):
    """
    One TCP connection = one live-caption session.

    Incoming: AudioChunks of arbitrary length (16-kHz float32 PCM).
    Outgoing: Transcripts with keys:
              • text  - entire current line (incl. cached context)
              • final - True when VAD says the utterance ended
    """

    def __init__(self, *args, model_name: str = "moonshine/base", **kwargs):
        super().__init__(*args, **kwargs)
        self._model_name = model_name
        self._transcriber = Transcriber(model_name)
        self._vad_model = load_silero_vad(onnx=True)
        self._vad_iterator = VADIterator(
            model=self._vad_model,
            sampling_rate=SAMPLING_RATE,
            threshold=0.5,
            min_silence_duration_ms=500,
        )
        self._speech_buf: Queue[AudioChunk] = Queue(maxsize=100000)
        self._recording = False
        self._transcriber_task_handle = asyncio.create_task(self._transcriber_task())
        self._recording_debug_handle = None
        self._speech_samples = np.empty(0, np.float32)
        self._last_refresh_t = 0
        if True:
        # if "debug_dir" in kwargs:
            self._debug_dir = Path("debug")
            self._debug_dir.mkdir(parents=True, exist_ok=True)
        

        self._DEBUG_LENGTH = 30 # seconds
        self._cur_seg_duration = 0

        # VAD sample buffering - ensure exactly 512 samples per VAD call
        self._vad_sample_buffer = np.array([], dtype=np.float32)
        self._vad_buffer_size = CHUNK_SAMPLES

    def soft_reset(self) -> None:
        """Reset only the iterator's state, not the underlying model."""
        self._vad_iterator.triggered = False
        self._vad_iterator.temp_end = 0
        self._vad_iterator.current_sample = 0
        # Clear the VAD sample buffer
        self._vad_sample_buffer = np.array([], dtype=np.float32)
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

    async def _transcriber_task(self) -> None:
        while True:
            chunk = await self._speech_buf.get()
            
            # Convert chunk to numpy array
            chunk_array = self._chunk_to_numpy(chunk)
            
            # Add chunk to speech buffer for transcription
            self._speech_samples = np.concatenate((
                self._speech_samples, 
                chunk_array
            ))
            
            # Keep only lookback context while idle
            if not self._recording:
                max_idle = LOOKBACK_CHUNKS * CHUNK_SAMPLES
                self._speech_samples = self._speech_samples[-max_idle:]
            
            # Use buffered VAD processing to ensure exactly 512 samples per VAD call
            await self._buffer_and_process_vad(chunk, chunk_array)
            
            self._speech_buf.task_done()

    async def _buffer_and_process_vad(self, chunk: AudioChunk, chunk_array: np.ndarray) -> None:
        """Buffer audio samples and process VAD with exactly 512 samples at a time."""
        # Add new samples to buffer
        self._vad_sample_buffer = np.concatenate([self._vad_sample_buffer, chunk_array])
        
        # Process complete 512-sample chunks
        while len(self._vad_sample_buffer) >= self._vad_buffer_size:
            # Extract exactly 512 samples
            samples_to_process = self._vad_sample_buffer[:self._vad_buffer_size]
            
            # Process with VAD
            await self._process_vad_samples(samples_to_process)
            
            # Remove processed samples from buffer
            self._vad_sample_buffer = self._vad_sample_buffer[self._vad_buffer_size:]

    async def _process_vad_samples(self, samples: np.ndarray) -> None:
        """Process exactly 512 samples with VAD iterator."""
        if len(samples) != self._vad_buffer_size:
            logger.warning(f"Expected {self._vad_buffer_size} samples, got {len(samples)}")
            return
            
        try:
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
                self._last_refresh_t = time.time()
                logger.info("Started recording speech")
                await self.write_event(VoiceStarted().event())

            elif "end" in vad_event and self._recording:
                # End recording and transcribe
                self._recording = False
                await self.write_event(VoiceStopped().event())
                text = self._transcriber(self._speech_samples)
                transcript = StreamingTranscript(text=text, final=True)
                await self.write_event(transcript.event())
                self._speech_samples = np.empty(0, np.float32)
                self.soft_reset()
                logger.info(f"VAD end detected. Transcribed: {text}")

        # If we are inside speech, push incremental refreshes
        if self._recording:
            now = time.time()
            speech_len_sec = len(self._speech_samples) / SAMPLING_RATE
            
            if (speech_len_sec > MAX_SPEECH_SECS or 
                now - self._last_refresh_t > MIN_REFRESH_SECS):
                text = self._transcriber(self._speech_samples)
                transcript = StreamingTranscript(text=text, final=False)
                await self.write_event(transcript.event())
                self._last_refresh_t = now

    async def _flush_vad_buffer(self) -> None:
        """Process any remaining samples in the VAD buffer at stream end."""
        if len(self._vad_sample_buffer) > 0:
            logger.debug(f"Flushing {len(self._vad_sample_buffer)} remaining samples from VAD buffer")
            
            # Pad with zeros to reach 512 samples if needed
            if len(self._vad_sample_buffer) < self._vad_buffer_size:
                padding_needed = self._vad_buffer_size - len(self._vad_sample_buffer)
                self._vad_sample_buffer = np.concatenate([
                    self._vad_sample_buffer, 
                    np.zeros(padding_needed, dtype=np.float32)
                ])
            
            # Process the final chunk
            await self._process_vad_samples(self._vad_sample_buffer)
            
            # Clear the buffer
            self._vad_sample_buffer = np.array([], dtype=np.float32)

    async def _write_debug_file(self, event: AudioChunk) -> None:
        """Logic to create debug files"""
        if self._recording_debug_handle is None:
            self._cur_seg_duration = 0
            self._recording_debug_handle = LocalFileSink(
                file_path=self._debug_dir / f"{time.time()}.wav",
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

    async def _handle_audio_chunk(self, event: AudioChunk) -> None:
        await self._write_debug_file(event)
        self._speech_buf.put_nowait(event)

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming protocol events"""
        if Transcribe.is_type(event.type):
            # Reset for new transcription request
            self._recording = False
            self._speech_samples = np.empty(0, np.float32)
            self.soft_reset()
            return True
        elif AudioStart.is_type(event.type):
            # Start a new audio stream
            logger.info("Audio stream started")
            # Note: _recording is controlled by VAD events, not AudioStart
            return True
        elif AudioChunk.is_type(event.type):
            # Process audio chunk
            audio_chunk = AudioChunk.from_event(event)
            await self._handle_audio_chunk(audio_chunk)
            return True
        elif AudioStop.is_type(event.type):
            # End of audio stream
            logger.info("Audio stream stopped")
            
            # Flush any remaining samples in the VAD buffer
            await self._flush_vad_buffer()
            
            # If we have speech samples, process the final transcript
            if len(self._speech_samples) > 0:
                text = self._transcriber(self._speech_samples)
                transcript = StreamingTranscript(text=text, final=True)
                await self.write_event(transcript.event())
                self._speech_samples = np.empty(0, np.float32)
            
            self._recording = False
            self.soft_reset()
            return True
        elif Describe.is_type(event.type):
            # Respond with service capabilities
            model = AsrModel(
                name=self._model_name,
                attribution=Attribution(
                    name="Moonshine ONNX",
                    url="https://github.com/usefulsensors/moonshine"
                ),
                installed=True,
                description="Moonshine ASR model",
                version="1.0.0",
                languages=["en"]
            )
            
            program = AsrProgram(
                name="moonshine-asr",
                attribution=Attribution(
                    name="Wyoming Moonshine Streaming ASR",
                    url="https://github.com/usefulsensors/moonshine"
                ),
                installed=True,
                description="Moonshine ASR with Silero VAD",
                version="1.0.0",
                models=[model]
            )
            
            info = Info(asr=[program])
            await self.write_event(info.event())
            return True
        
        return False
    
    async def disconnect(self) -> None:
        if self._recording_debug_handle is not None:
            await self._recording_debug_handle.close()

# --------------------------------------------------------------------------- #
async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="moonshine/base",
        choices=["moonshine/base", "moonshine/tiny"],
        help="Moonshine model to load",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Interface to bind the TCP server"
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Port to bind the TCP server"
    )
    args = parser.parse_args()

    server = AsyncTcpServer(host=args.host, port=args.port)
    logger.info(
        f"Moonshine ASR service starting on {args.host}:{args.port} (model={args.model_name})"
    )
    await server.run(
        handler_factory=lambda *_args, **_kwargs: StreamingTranscriptionHandler(
            *_args, model_name=args.model_name, **_kwargs
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
