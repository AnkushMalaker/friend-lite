"""
Audio processing module for handling audio recording, transcription, and processing.
Contains the main classes and functions for audio pipeline management.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from motor.motor_asyncio import AsyncIOMotorCollection
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient

from services import (
    get_audio_config,
    get_conversation_config,
)

# Set up logging
audio_logger = logging.getLogger("audio")

# Audio processing configuration
audio_config = get_audio_config()
conversation_config = get_conversation_config()

# Audio constants
OMI_SAMPLE_RATE = audio_config["sample_rate"]
OMI_CHANNELS = audio_config["channels"]
OMI_SAMPLE_WIDTH = audio_config["sample_width"]
SEGMENT_SECONDS = audio_config["segment_seconds"]
TARGET_SAMPLES = audio_config["target_samples"]
CHUNK_DIR = audio_config["chunk_dir"]
USE_DEEPGRAM = audio_config["use_deepgram"]
OFFLINE_ASR_TCP_URI = audio_config["offline_asr_uri"]
DEEPGRAM_API_KEY = audio_config["deepgram_api_key"]

# Conversation configuration
NEW_CONVERSATION_TIMEOUT_MINUTES = conversation_config["timeout_minutes"]
AUDIO_CROPPING_ENABLED = conversation_config["audio_cropping_enabled"]
MIN_SPEECH_SEGMENT_DURATION = conversation_config["min_speech_duration"]
CROPPING_CONTEXT_PADDING = conversation_config["context_padding"]

@dataclass
class AudioProcessingResult:
    """Result of audio processing operations."""
    success: bool
    message: str
    audio_uuid: Optional[str] = None
    file_path: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None

async def _process_audio_cropping_with_relative_timestamps(
    original_path: str | Path,
    speech_segments: list[tuple[float, float]],
    output_path: str | Path,
    audio_uuid: str,
) -> bool:
    """
    Process audio cropping with automatic relative timestamp conversion.
    This function handles both live processing and reprocessing scenarios.
    """
    try:
        # Convert absolute timestamps to relative timestamps
        # Extract file start time from filename: timestamp_client_uuid.wav
        original_path = Path(original_path)
        filename = original_path.name
        audio_logger.info(f"🕐 Parsing filename: {filename}")
        filename_parts = filename.split("_")
        if len(filename_parts) < 3:
            audio_logger.error(f"Invalid filename format: {filename}. Expected format: timestamp_client_id_audio_uuid.wav")
            return False
        
        try:
            file_start_timestamp = float(filename_parts[0])
        except ValueError as e:
            audio_logger.error(f"Cannot parse timestamp from filename {filename}: {e}")
            return False

        # Convert speech segments to relative timestamps
        relative_segments = []
        for start_abs, end_abs in speech_segments:
            # Validate input timestamps
            if start_abs >= end_abs:
                audio_logger.warning(f"⚠️ Invalid speech segment: start={start_abs} >= end={end_abs}, skipping")
                continue
                
            start_rel = start_abs - file_start_timestamp
            end_rel = end_abs - file_start_timestamp

            # Ensure relative timestamps are positive (sanity check)
            if start_rel < 0:
                audio_logger.warning(f"⚠️ Negative start timestamp: {start_rel} (absolute: {start_abs}, file_start: {file_start_timestamp}), clamping to 0.0")
                start_rel = 0.0
            if end_rel < 0:
                audio_logger.warning(f"⚠️ Negative end timestamp: {end_rel} (absolute: {end_abs}, file_start: {file_start_timestamp}), skipping segment")
                continue

            relative_segments.append((start_rel, end_rel))

        if not relative_segments:
            audio_logger.warning(f"⚠️ No valid speech segments found after timestamp conversion for {audio_uuid}")
            return False

        audio_logger.info(f"🕐 Converted {len(speech_segments)} absolute segments to {len(relative_segments)} relative segments")
        
        # Perform the actual cropping
        success = await _crop_audio_with_ffmpeg(original_path, relative_segments, output_path)
        
        if success:
            audio_logger.info(f"✅ Audio cropping completed for {audio_uuid}")
        else:
            audio_logger.error(f"❌ Audio cropping failed for {audio_uuid}")
            
        return success

    except Exception as e:
        audio_logger.error(f"❌ Error in audio cropping with relative timestamps for {audio_uuid}: {e}")
        return False

async def _crop_audio_with_ffmpeg(
    original_path: str | Path,
    speech_segments: list[tuple[float, float]],
    output_path: str | Path,
) -> bool:
    """
    Use ffmpeg to crop audio as async subprocess.
    """
    try:
        # Filter segments by minimum duration and add padding
        filtered_segments = []
        for start, end in speech_segments:
            duration = end - start
            if duration >= MIN_SPEECH_SEGMENT_DURATION:
                # Add padding
                padded_start = max(0, start - CROPPING_CONTEXT_PADDING)
                padded_end = end + CROPPING_CONTEXT_PADDING
                filtered_segments.append((padded_start, padded_end))
                audio_logger.debug(f"📏 Segment: {start:.2f}s-{end:.2f}s (duration: {duration:.2f}s) -> {padded_start:.2f}s-{padded_end:.2f}s")
            else:
                audio_logger.debug(f"⏭️ Skipping short segment: {start:.2f}s-{end:.2f}s (duration: {duration:.2f}s < {MIN_SPEECH_SEGMENT_DURATION}s)")

        if not filtered_segments:
            audio_logger.warning("⚠️ No segments meet minimum duration requirement")
            return False

        # Build the ffmpeg filter for concatenating speech segments
        filter_parts = []
        for i, (start, end) in enumerate(filtered_segments):
            filter_parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]")
        
        # Concatenate all segments
        if len(filtered_segments) > 1:
            input_labels = "".join(f"[a{i}]" for i in range(len(filtered_segments)))
            concat_filter = f"{input_labels}concat=n={len(filtered_segments)}:v=0:a=1[out]"
            filter_parts.append(concat_filter)
            output_map = "-map [out]"
        else:
            output_map = "-map [a0]"

        filter_complex = ";".join(filter_parts)

        # Build the ffmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-i", str(original_path),
            "-filter_complex", filter_complex,
            output_map,
            "-c:a", "pcm_s16le",  # Preserve audio format
            "-y",  # Overwrite output file
            str(output_path),
        ]

        audio_logger.info(f"🎬 Running ffmpeg command: {' '.join(ffmpeg_cmd)}")

        # Run ffmpeg as subprocess
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            audio_logger.info(f"✅ Audio cropping completed successfully: {str(output_path)}")
            return True
        else:
            audio_logger.error(f"❌ ffmpeg failed with return code {process.returncode}")
            audio_logger.error(f"STDERR: {stderr.decode()}")
            return False

    except Exception as e:
        audio_logger.error(f"❌ Error in audio cropping: {e}")
        return False

def _new_local_file_sink(file_path: str) -> LocalFileSink:
    """
    Create a properly configured LocalFileSink with all wave parameters set.
    """
    return LocalFileSink(
        file_path=file_path,
        sample_rate=OMI_SAMPLE_RATE,
        channels=OMI_CHANNELS,
        sample_width=OMI_SAMPLE_WIDTH,
    )

class ChunkRepo:
    """Async helpers for the audio_chunks collection."""

    def __init__(self, collection: AsyncIOMotorCollection):
        self.collection = collection

    async def create_chunk(self, audio_uuid: str, client_id: str, file_path: str) -> dict[str, Any]:
        """Create a new chunk document."""
        chunk_doc = {
            "audio_uuid": audio_uuid,
            "client_id": client_id,
            "file_path": file_path,
            "transcript_segments": [],
            "speakers": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        await self.collection.insert_one(chunk_doc)
        return chunk_doc

    async def add_transcript_segment(self, audio_uuid: str, segment: dict[str, Any]) -> None:
        """Add a transcript segment to the chunk."""
        await self.collection.update_one(
            {"audio_uuid": audio_uuid}, {"$push": {"transcript_segments": segment}}
        )

    async def add_speaker(self, audio_uuid: str, speaker_name: str) -> None:
        """Add a speaker to the chunk."""
        await self.collection.update_one(
            {"audio_uuid": audio_uuid}, {"$addToSet": {"speakers": speaker_name}}
        )

    async def update_transcript(self, audio_uuid: str, transcript: str) -> None:
        """Update the full transcript for a chunk."""
        await self.collection.update_one(
            {"audio_uuid": audio_uuid}, {"$set": {"transcript": transcript}}
        )

    async def update_segment_timing(self, audio_uuid: str, segment_index: int, start_time: float, end_time: float) -> bool:
        """Update segment timing information."""
        result = await self.collection.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    f"transcript_segments.{segment_index}.start_time": start_time,
                    f"transcript_segments.{segment_index}.end_time": end_time,
                }
            },
        )
        return result.modified_count > 0

    async def update_segment_speaker(self, audio_uuid: str, segment_index: int, speaker: str) -> bool:
        """Update segment speaker information."""
        result = await self.collection.update_one(
            {"audio_uuid": audio_uuid},
            {"$set": {f"transcript_segments.{segment_index}.speaker": speaker}},
        )
        return result.modified_count > 0

    async def update_cropped_audio(self, audio_uuid: str, cropped_path: str, speech_segments: list[tuple[float, float]]) -> bool:
        """Update cropped audio information."""
        result = await self.collection.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    "cropped_audio_path": cropped_path,
                    "speech_segments": [
                        {"start": start, "end": end} for start, end in speech_segments
                    ],
                    "updated_at": time.time(),
                }
            },
        )
        return result.modified_count > 0

class TranscriptionManager:
    """Manages transcription using either Deepgram or offline ASR service."""

    def __init__(self, action_item_callback=None):
        self.client = None
        self._current_audio_uuid = None
        self.use_deepgram = USE_DEEPGRAM
        self.deepgram_client = None  # WebSocket client for Deepgram
        self._audio_buffer = []  # Buffer for Deepgram batch processing
        self.action_item_callback = action_item_callback  # Callback to queue action items

    async def connect(self, client_id: Optional[str] = None):
        """Establish connection to ASR service."""
        self._client_id = client_id

        if self.use_deepgram:
            await self._connect_deepgram()
        else:
            # For offline ASR, we connect on-demand for each chunk
            audio_logger.info("Using offline ASR - connections established per chunk")

    async def flush_final_transcript(self, audio_uuid: str) -> None:
        """Flush any remaining transcript data for the audio UUID."""
        if self.use_deepgram and self._audio_buffer:
            audio_logger.info(f"🔄 Flushing final transcript for {audio_uuid}")
            
            # Process any remaining buffered audio
            if self._audio_buffer:
                # Send buffered audio to Deepgram
                try:
                    buffered_audio = b"".join(self._audio_buffer)
                    if self.deepgram_client and not self.deepgram_client.is_closing():
                        await self.deepgram_client.send(buffered_audio)
                        
                    # Clear buffer
                    self._audio_buffer = []
                    
                    # Send close signal
                    if self.deepgram_client and not self.deepgram_client.is_closing():
                        await self.deepgram_client.send(b"")
                        
                except Exception as e:
                    audio_logger.error(f"Error flushing final transcript: {e}")

    async def disconnect(self):
        """Disconnect from ASR service."""
        if self.use_deepgram:
            await self._disconnect_deepgram()
        else:
            if self.client:
                await self.client.disconnect()
                self.client = None

    async def _connect_deepgram(self):
        """Connect to Deepgram WebSocket API."""
        try:
            import websockets

            # Build WebSocket URL
            url = f"wss://api.deepgram.com/v1/listen?model=nova-2&language=en-US&encoding=linear16&sample_rate={OMI_SAMPLE_RATE}&channels={OMI_CHANNELS}&endpointing=300"
            
            headers = {
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
            }
            
            self.deepgram_client = await websockets.connect(url, extra_headers=headers)
            
            # Start listening for responses
            asyncio.create_task(self._listen_for_deepgram_responses())
            
            audio_logger.info("🎤 Connected to Deepgram WebSocket API")
            
        except Exception as e:
            audio_logger.error(f"Failed to connect to Deepgram: {e}")
            raise

    async def _disconnect_deepgram(self):
        """Disconnect from Deepgram WebSocket API."""
        if self.deepgram_client:
            try:
                await self.deepgram_client.close()
                audio_logger.info("🔌 Disconnected from Deepgram WebSocket API")
            except Exception as e:
                audio_logger.error(f"Error disconnecting from Deepgram: {e}")
            finally:
                self.deepgram_client = None

    async def _listen_for_deepgram_responses(self):
        """Listen for responses from Deepgram WebSocket."""
        try:
            async for message in self.deepgram_client:
                await self._handle_deepgram_response(message)
        except Exception as e:
            audio_logger.error(f"Error listening for Deepgram responses: {e}")

    async def _handle_deepgram_response(self, message: str):
        """Handle response from Deepgram WebSocket."""
        try:
            data = json.loads(message)
            
            if data.get("type") == "Results":
                channel = data.get("channel", {})
                alternatives = channel.get("alternatives", [])
                
                if alternatives:
                    transcript = alternatives[0].get("transcript", "")
                    is_final = channel.get("is_final", False)
                    
                    if transcript and is_final:
                        # Process final transcript
                        if self.action_item_callback and self._current_audio_uuid:
                            await self.action_item_callback(
                                transcript, self._client_id, self._current_audio_uuid
                            )
                        
                        audio_logger.info(f"📝 Final transcript: {transcript}")
                        return transcript
                        
        except Exception as e:
            audio_logger.error(f"Error handling Deepgram response: {e}")
        
        return None

    async def transcribe_chunk(self, audio_uuid: str, chunk_data: bytes) -> Optional[str]:
        """Transcribe audio chunk."""
        self._current_audio_uuid = audio_uuid
        
        if self.use_deepgram:
            return await self._transcribe_chunk_deepgram(chunk_data)
        else:
            return await self._transcribe_chunk_offline(chunk_data)

    async def _transcribe_chunk_deepgram(self, chunk_data: bytes) -> Optional[str]:
        """Transcribe audio chunk using Deepgram WebSocket."""
        try:
            if not self.deepgram_client or self.deepgram_client.is_closing():
                await self._connect_deepgram()
            
            # Send audio data
            await self.deepgram_client.send(chunk_data)
            
            # Buffer audio for potential final flush
            self._audio_buffer.append(chunk_data)
            
            # Keep buffer size manageable
            if len(self._audio_buffer) > 10:
                self._audio_buffer.pop(0)
                
            return None  # Transcripts are handled in the response listener
            
        except Exception as e:
            audio_logger.error(f"Error transcribing chunk with Deepgram: {e}")
            return None

    async def _transcribe_chunk_offline(self, chunk_data: bytes) -> Optional[str]:
        """Transcribe audio chunk using offline ASR service."""
        try:
            if not self.client:
                self.client = AsyncTcpClient(OFFLINE_ASR_TCP_URI)
                await self.client.connect()
                
                # Send transcription request
                await self.client.send(
                    Transcribe(
                        language="en",
                        model="",
                    )
                )
                
                # Send audio parameters
                await self.client.send(
                    AudioStart(
                        rate=OMI_SAMPLE_RATE,
                        width=OMI_SAMPLE_WIDTH,
                        channels=OMI_CHANNELS,
                    )
                )

            # Send audio chunk
            await self.client.send(AudioChunk(audio=chunk_data))
            
            # Send stop signal
            await self.client.send(AudioStop())
            
            # Wait for transcript
            while True:
                event = await self.client.receive()
                if isinstance(event, Transcript):
                    transcript_text = event.text.strip()
                    if transcript_text:
                        audio_logger.info(f"📝 Transcript: {transcript_text}")
                        
                        # Process for action items
                        if self.action_item_callback and self._current_audio_uuid:
                            await self.action_item_callback(
                                transcript_text, self._client_id, self._current_audio_uuid
                            )
                        
                        return transcript_text
                    break
                    
        except Exception as e:
            audio_logger.error(f"Error transcribing chunk with offline ASR: {e}")
            # Try to reconnect
            await self._reconnect()
            return None

    async def _reconnect_deepgram(self):
        """Reconnect to Deepgram WebSocket."""
        try:
            await self._disconnect_deepgram()
            await self._connect_deepgram()
            audio_logger.info("🔄 Reconnected to Deepgram")
        except Exception as e:
            audio_logger.error(f"Failed to reconnect to Deepgram: {e}")

    async def _reconnect(self):
        """Reconnect to ASR service."""
        if self.use_deepgram:
            await self._reconnect_deepgram()
        else:
            try:
                if self.client:
                    await self.client.disconnect()
                self.client = None
                audio_logger.info("🔄 Reconnected to offline ASR")
            except Exception as e:
                audio_logger.error(f"Failed to reconnect to offline ASR: {e}")

def get_audio_processing_result(success: bool, message: str, **kwargs) -> AudioProcessingResult:
    """Create an AudioProcessingResult object."""
    return AudioProcessingResult(
        success=success,
        message=message,
        **kwargs
    )