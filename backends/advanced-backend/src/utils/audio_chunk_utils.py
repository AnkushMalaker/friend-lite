import asyncio
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink

from models.audio_chunk import AudioChunk, TranscriptSegment, SpeechSegment

logger = logging.getLogger("advanced-backend")
audio_logger = logging.getLogger("audio_processing")
audio_cropper_logger = logging.getLogger("audio_cropper")

# Configuration values that were previously in main.py
OMI_SAMPLE_RATE = 16_000  # Hz
OMI_CHANNELS = 1
OMI_SAMPLE_WIDTH = 2  # bytes (16‑bit)
SEGMENT_SECONDS = 60  # length of each stored chunk
CHUNK_DIR = Path("./audio_chunks")
AUDIO_CROPPING_ENABLED = os.getenv("AUDIO_CROPPING_ENABLED", "true").lower() == "true"
MIN_SPEECH_SEGMENT_DURATION = float(os.getenv("MIN_SPEECH_SEGMENT_DURATION", "1.0"))
CROPPING_CONTEXT_PADDING = float(os.getenv("CROPPING_CONTEXT_PADDING", "0.1"))

# Ensure CHUNK_DIR exists
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

def _new_local_file_sink(file_path):
    """Create a properly configured LocalFileSink with all wave parameters set."""
    sink = LocalFileSink(
        file_path=file_path,
        sample_rate=int(OMI_SAMPLE_RATE),
        channels=int(OMI_CHANNELS),
        sample_width=int(OMI_SAMPLE_WIDTH),
    )
    return sink

async def _process_audio_cropping_with_relative_timestamps(
    original_path: str,
    speech_segments: List[Tuple[float, float]],
    output_path: str,
    audio_uuid: str,
    chunk_repo_instance, # Pass chunk_repo instance
) -> bool:
    """
    Process audio cropping with automatic relative timestamp conversion.
    This function handles both live processing and reprocessing scenarios.
    """
    try:
        # Convert absolute timestamps to relative timestamps
        # Extract file start time from filename: timestamp_client_uuid.wav
        filename = original_path.split("/")[-1]
        file_start_timestamp = float(filename.split("_")[0])

        # Convert speech segments to relative timestamps
        relative_segments = []
        for start_abs, end_abs in speech_segments:
            start_rel = start_abs - file_start_timestamp
            end_rel = end_abs - file_start_timestamp

            # Ensure relative timestamps are positive (sanity check)
            if start_rel < 0:
                audio_logger.warning(
                    f"⚠️ Negative start timestamp: {start_rel}, clamping to 0.0"
                )
                start_rel = 0.0
            if end_rel < 0:
                audio_logger.warning(
                    f"⚠️ Negative end timestamp: {end_rel}, skipping segment"
                )
                continue

            relative_segments.append((start_rel, end_rel))

        audio_logger.info(
            f"🕐 Converting timestamps for {audio_uuid}: file_start={file_start_timestamp}"
        )
        audio_logger.info(f"🕐 Absolute segments: {speech_segments}")
        audio_logger.info(f"🕐 Relative segments: {relative_segments}")

        success = await _crop_audio_with_ffmpeg(
            original_path, relative_segments, output_path
        )
        if success:
            # Update database with cropped file info (keep original absolute timestamps for reference)
            cropped_filename = output_path.split("/")[-1]
            await chunk_repo_instance.update_cropped_audio( # Use passed instance
                audio_uuid, cropped_filename, speech_segments
            )
            audio_logger.info(
                f"Successfully processed cropped audio: {cropped_filename}"
            )
            return True
        else:
            audio_logger.error(f"Failed to crop audio for {audio_uuid}")
            return False
    except Exception as e:
        audio_logger.error(f"Error in audio cropping task for {audio_uuid}: {e}")
        return False

async def _crop_audio_with_ffmpeg(
    original_path: str, speech_segments: List[Tuple[float, float]], output_path: str
) -> bool:
    """Use ffmpeg to crop audio - runs as async subprocess, no GIL issues"""
    audio_cropper_logger.info(
        f"Cropping audio {original_path} with {len(speech_segments)} speech segments"
    )

    if not AUDIO_CROPPING_ENABLED:
        audio_cropper_logger.info(f"Audio cropping disabled, skipping {original_path}")
        return False

    if not speech_segments:
        audio_cropper_logger.warning(f"No speech segments to crop for {original_path}")
        return False

    # Filter out segments that are too short
    filtered_segments = []
    for start, end in speech_segments:
        duration = end - start
        if duration >= MIN_SPEECH_SEGMENT_DURATION:
            # Add padding around speech segments
            padded_start = max(0, start - CROPPING_CONTEXT_PADDING)
            padded_end = end + CROPPING_CONTEXT_PADDING
            filtered_segments.append((padded_start, padded_end))
        else:
            audio_cropper_logger.debug(
                f"Skipping short segment: {start}-{end} ({duration:.2f}s < {MIN_SPEECH_SEGMENT_DURATION}s)"
            )

    if not filtered_segments:
        audio_cropper_logger.warning(
            f"No segments meet minimum duration ({MIN_SPEECH_SEGMENT_DURATION}s) for {original_path}"
        )
        return False

    audio_cropper_logger.info(
        f"Cropping audio {original_path} with {len(filtered_segments)} speech segments (filtered from {len(speech_segments)})"
    )

    try:
        # Build ffmpeg filter for concatenating speech segments
        filter_parts = []
        for i, (start, end) in enumerate(filtered_segments):
            duration = end - start
            filter_parts.append(
                f"[0:a]atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS[seg{i}]"
            )

        # Concatenate all segments
        inputs = "".join(f"[seg{i}]" for i in range(len(filtered_segments)))
        concat_filter = f"{inputs}concat=n={len(filtered_segments)}:v=0:a=1[out]"

        full_filter = ";".join(filter_parts + [concat_filter])

        # Run ffmpeg as async subprocess
        cmd = [
            "ffmpeg",
            "-y",  # -y = overwrite output
            "-i",
            original_path,
            "-filter_complex",
            full_filter,
            "-map",
            "[out]",
            "-c:a",
            "pcm_s16le",  # Keep same format as original
            output_path,
        ]

        audio_cropper_logger.info(f"Running ffmpeg command: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        if stdout:
            audio_cropper_logger.debug(f"FFMPEG stdout: {stdout.decode()}")

        if process.returncode == 0:
            # Calculate cropped duration
            cropped_duration = sum(end - start for start, end in filtered_segments)
            audio_cropper_logger.info(
                f"Successfully cropped {original_path} -> {output_path} ({cropped_duration:.1f}s from {len(filtered_segments)} segments)"
            )
            return True
        else:
            error_msg = stderr.decode() if stderr else "Unknown ffmpeg error"
            audio_logger.error(f"ffmpeg failed for {original_path}: {error_msg}")
            return False

    except Exception as e:
        audio_logger.error(f"Error running ffmpeg on {original_path}: {e}")
        return False

class ChunkRepo:
    """Async helpers for the audio_chunks collection."""

    def __init__(self, collection):
        self.col = collection

    async def create_chunk(
        self,
        *,
        audio_uuid,
        audio_path,
        client_id,
        timestamp,
        transcript=None,
        speakers_identified=None,
    ):
        doc = {
            "audio_uuid": audio_uuid,
            "audio_path": audio_path,
            "client_id": client_id,
            "timestamp": timestamp,
            "transcript": transcript or [],  # List of conversation segments
            "speakers_identified": speakers_identified
            or [],  # List of identified speakers
        }
        await self.col.insert_one(doc)

    async def add_transcript_segment(self, audio_uuid, transcript_segment):
        """Add a single transcript segment to the conversation."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid}, {"$push": {"transcript": transcript_segment}}
        )

    async def add_speaker(self, audio_uuid, speaker_id):
        """Add a speaker to the speakers_identified list if not already present."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$addToSet": {"speakers_identified": speaker_id}},
        )

    async def update_transcript(self, audio_uuid, full_transcript):
        """Update the entire transcript list (for compatibility)."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid}, {"$set": {"transcript": full_transcript}}
        )

    async def update_segment_timing(
        self, audio_uuid, segment_index, start_time, end_time
    ):
        """Update timing information for a specific transcript segment."""
        await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    f"transcript.{segment_index}.start": start_time,
                    f"transcript.{segment_index}.end": end_time,
                }
            },
        )

    async def update_segment_speaker(self, audio_uuid, segment_index, speaker_id):
        """Update the speaker for a specific transcript segment."""
        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {"$set": {f"transcript.{segment_index}.speaker": speaker_id}},
        )
        if result.modified_count > 0:
            audio_logger.info(
                f"Updated segment {segment_index} speaker to {speaker_id} for {audio_uuid}"
            )
        return result.modified_count > 0

    async def update_cropped_audio(
        self,
        audio_uuid: str,
        cropped_path: str,
        speech_segments: List[Tuple[float, float]],
    ):
        """Update the chunk with cropped audio information."""
        cropped_duration = sum(end - start for start, end in speech_segments)

        result = await self.col.update_one(
            {"audio_uuid": audio_uuid},
            {
                "$set": {
                    "cropped_audio_path": cropped_path,
                    "speech_segments": [
                        {"start": start, "end": end} for start, end in speech_segments
                    ],
                    "cropped_duration": cropped_duration,
                    "cropped_at": time.time(),
                }
            },
        )
        if result.modified_count > 0:
            audio_logger.info(
                f"Updated cropped audio info for {audio_uuid}: {cropped_path}"
            )
        return result.modified_count > 0

class AudioChunkUtils:
    def __init__(self, chunks_collection: AsyncIOMotorClient):
        self.chunks_col = chunks_collection
        self.chunk_repo = ChunkRepo(chunks_collection)

    async def get_conversations(self):
        """Get all conversations grouped by client_id."""
        try:
            # Get all audio chunks and group by client_id
            cursor = self.chunks_col.find({}).sort("timestamp", -1)
            conversations = {}

            async for chunk in cursor:
                client_id = chunk.get("client_id", "unknown")
                if client_id not in conversations:
                    conversations[client_id] = []

                conversations[client_id].append(
                    {
                        "audio_uuid": chunk["audio_uuid"],
                        "audio_path": chunk["audio_path"],
                        "cropped_audio_path": chunk.get("cropped_audio_path"),
                        "timestamp": chunk["timestamp"],
                        "transcript": chunk.get("transcript", []),
                        "speakers_identified": chunk.get("speakers_identified", []),
                        "speech_segments": chunk.get("speech_segments", []),
                        "cropped_duration": chunk.get("cropped_duration"),
                    }
                )

            return JSONResponse(content={"conversations": conversations})
        except Exception as e:
            audio_logger.error(f"Error getting conversations: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    async def get_cropped_audio_info(self, audio_uuid: str):
        """Get cropped audio information for a specific conversation."""
        try:
            chunk = await self.chunks_col.find_one({"audio_uuid": audio_uuid})
            if not chunk:
                return JSONResponse(
                    status_code=404, content={"error": "Conversation not found"}
                )

            return JSONResponse(content={
                "audio_uuid": audio_uuid,
                "original_audio_path": chunk["audio_path"],
                "cropped_audio_path": chunk.get("cropped_audio_path"),
                "speech_segments": chunk.get("speech_segments", []),
                "cropped_duration": chunk.get("cropped_duration"),
                "cropped_at": chunk.get("cropped_at"),
                "has_cropped_version": bool(chunk.get("cropped_audio_path")),
            })
        except Exception as e:
            audio_logger.error(f"Error getting cropped audio info: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    async def reprocess_audio_cropping(self, audio_uuid: str):
        """Trigger reprocessing of audio cropping for a specific conversation."""
        try:
            chunk = await self.chunks_col.find_one({"audio_uuid": audio_uuid})
            if not chunk:
                return JSONResponse(
                    status_code=404, content={"error": "Conversation not found"}
                )

            original_path = f"{CHUNK_DIR}/{chunk['audio_path']}"
            if not Path(original_path).exists():
                return JSONResponse(
                    status_code=404, content={"error": "Original audio file not found"}
                )

            # Check if we have speech segments
            speech_segments = chunk.get("speech_segments", [])
            if not speech_segments:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No speech segments available for cropping"},
                )

            # Convert speech segments from dict format to tuple format
            speech_segments_tuples = [(seg["start"], seg["end"]) for seg in speech_segments]

            cropped_filename = chunk["audio_path"].replace(".wav", "_cropped.wav")
            cropped_path = f"{CHUNK_DIR}/{cropped_filename}"

            # Process in background using shared logic
            async def reprocess_task():
                audio_logger.info(f"🔄 Starting reprocess for {audio_uuid}")
                await _process_audio_cropping_with_relative_timestamps(
                    original_path, speech_segments_tuples, cropped_path, audio_uuid, self.chunk_repo
                )

            asyncio.create_task(reprocess_task())

            return JSONResponse(content={"message": "Reprocessing started", "audio_uuid": audio_uuid})
        except Exception as e:
            audio_logger.error(f"Error reprocessing audio: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    async def add_speaker_to_conversation(self, audio_uuid: str, speaker_id: str):
        """Add a speaker to the speakers_identified list for a conversation."""
        try:
            await self.chunk_repo.add_speaker(audio_uuid, speaker_id)
            return JSONResponse(
                content={
                    "message": f"Speaker {speaker_id} added to conversation {audio_uuid}"
                }
            )
        except Exception as e:
            audio_logger.error(f"Error adding speaker: {e}", exc_info=True)
            return JSONResponse(
                status_code=500, content={"message": "Error adding speaker"}
            )

    async def update_transcript_segment(
        self,
        audio_uuid: str,
        segment_index: int,
        speaker_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        """Update a specific transcript segment with speaker or timing information."""
        try:
            update_doc = {}

            if speaker_id is not None:
                update_doc[f"transcript.{segment_index}.speaker"] = speaker_id
                # Also add to speakers_identified if not already present
                await self.chunk_repo.add_speaker(audio_uuid, speaker_id)

            if start_time is not None:
                update_doc[f"transcript.{segment_index}.start"] = start_time

            if end_time is not None:
                update_doc[f"transcript.{segment_index}.end"] = end_time

            if not update_doc:
                return JSONResponse(
                    status_code=400, content={"error": "No update parameters provided"}
                )

            result = await self.chunks_col.update_one(
                {"audio_uuid": audio_uuid}, {"$set": update_doc}
            )

            if result.matched_count == 0:
                return JSONResponse(
                    status_code=404, content={"error": "Conversation not found"}
                )

            return JSONResponse(
                content={"message": "Transcript segment updated successfully"}
            )

        except Exception as e:
            audio_logger.error(f"Error updating transcript segment: {e}")
            return JSONResponse(status_code=500, content={"error": "Internal server error"})