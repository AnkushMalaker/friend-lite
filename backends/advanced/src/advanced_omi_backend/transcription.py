import asyncio
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Optional

from advanced_omi_backend.client_manager import get_client_manager
from advanced_omi_backend.config import (
    get_conversation_stop_settings,
    get_speech_detection_settings,
    load_diarization_settings_from_file,
)
from advanced_omi_backend.conversation_manager import get_conversation_manager
from advanced_omi_backend.database import ConversationsRepository, conversations_col
from advanced_omi_backend.llm_client import async_generate
from advanced_omi_backend.processors import get_processor_manager
from advanced_omi_backend.audio_processing_types import MemoryProcessingItem, CroppingItem
from advanced_omi_backend.speaker_recognition_client import SpeakerRecognitionClient
from advanced_omi_backend.transcription_providers import (
    BaseTranscriptionProvider,
    get_transcription_provider,
)
from wyoming.audio import AudioChunk

# ASR Configuration
TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER")  # Optional: 'deepgram' or 'parakeet'

logger = logging.getLogger(__name__)


class AudioTimeline:
    """Track audio timeline for proper speech gap detection."""

    def __init__(self):
        self.start_time = time.time()
        self.total_samples = 0
        self.sample_rate = None

    def add_chunk(self, chunk: AudioChunk):
        """Add audio chunk and update timeline."""
        if chunk.audio and len(chunk.audio) > 0:
            # Assuming 16-bit PCM audio (2 bytes per sample)
            self.total_samples += len(chunk.audio) // 2
            self.sample_rate = chunk.rate

    @property
    def current_position(self) -> float:
        """Get current position in audio stream (seconds)."""
        if not self.sample_rate:
            return 0.0
        return self.total_samples / self.sample_rate

    def reset(self):
        """Reset timeline for new audio session."""
        self.start_time = time.time()
        self.total_samples = 0


class SpeechActivityAnalyzer:
    """Analyze transcripts for speech activity after transcription."""

    def __init__(self, audio_timeline: AudioTimeline):
        config = get_conversation_stop_settings()
        self.speech_inactivity_threshold = config["speech_inactivity_threshold"]
        self.min_word_confidence = config["min_word_confidence"]
        self.audio_timeline = audio_timeline

    def analyze_transcript_activity(self, transcript_data: dict) -> dict:
        """
        Analyze transcript for speech activity.

        Returns:
            dict: {
                "has_speech": bool,
                "last_word_time": float or None,
                "speech_gap_seconds": float or None,
                "word_count": int
            }
        """
        words = transcript_data.get("words", [])

        # Filter by confidence
        valid_words = [
            w for w in words
            if w.get("confidence", 0) >= self.min_word_confidence
        ]

        if not valid_words:
            return {
                "has_speech": False,
                "last_word_time": None,
                "speech_gap_seconds": None,
                "word_count": 0
            }

        # Find last word timestamp
        last_word = valid_words[-1]
        last_word_end = last_word.get("end", 0)

        # Calculate speech gap using audio timeline
        current_audio_position = self.audio_timeline.current_position
        speech_gap = current_audio_position - last_word_end if current_audio_position else None

        return {
            "has_speech": True,
            "last_word_time": last_word_end,
            "speech_gap_seconds": speech_gap,
            "word_count": len(valid_words)
        }


class TranscriptionManager:
    """Manages transcription using any configured transcription provider."""

    def __init__(self, chunk_repo=None, processor_manager=None, skip_memory_queuing=False):
        self.provider: Optional[BaseTranscriptionProvider] = get_transcription_provider(
            TRANSCRIPTION_PROVIDER
        )
        self._current_audio_uuid = None
        self._audio_buffer = []  # Buffer for collecting audio chunks
        self._audio_start_time = None  # Track when audio collection started
        self._max_collection_time = 600.0  # 10 minutes timeout - allow longer conversations
        self._current_transaction_id = None  # Track current debug transaction
        self.chunk_repo = chunk_repo  # Database repository for chunks
        self.client_manager = get_client_manager()  # Cached client manager instance
        self.processor_manager = (
            processor_manager  # Reference to processor manager for completion tracking
        )
        self._client_id = None
        self._skip_memory_queuing = skip_memory_queuing  # For unified pipeline integration

        # Collection state tracking
        self._collecting = False
        self._collection_task = None

        # Audio timeline for speech gap detection
        self._audio_timeline = AudioTimeline()

        # Buffer monitoring for periodic transcription (batch providers)
        self._buffer_start_time = None
        config = get_conversation_stop_settings()
        self._max_buffer_duration = config["transcription_buffer_seconds"]
        self._transcribing = False  # Track transcription state
        self._last_word_time = None  # Track last word for conversation closure

        # Optional speaker recognition
        self.speaker_client = SpeakerRecognitionClient()
        if self.speaker_client.enabled:
            logger.info("Speaker recognition integration enabled")

    def _get_current_client(self):
        """Get the current client state using ClientManager."""
        if not self._client_id:
            return None
        return self.client_manager.get_client(self._client_id)

    async def connect(self, client_id: str | None = None):
        """Initialize transcription service for the client."""
        self._client_id = client_id

        if not self.provider:
            raise Exception("No transcription provider configured")

        try:
            await self.provider.connect(client_id)
            logger.info(
                f"{self.provider.name} transcription initialized for client {self._client_id}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to {self.provider.name} transcription service: {e}")
            raise

    async def process_collected_audio(self) -> Optional[str]:
        """Unified processing for all transcription providers.

        Returns:
            conversation_id if conversation was created, None otherwise
        """
        logger.info(f"🚀 process_collected_audio called for client {self._client_id}")
        logger.info(
            f"📊 Current state - buffer size: {len(self._audio_buffer) if self._audio_buffer else 0}, collecting: {self._collecting}"
        )

        if not self.provider:
            logger.error("No transcription provider available")
            return None

        # Cancel collection timeout task first to prevent interference
        if self._collection_task and not self._collection_task.done():
            logger.info(f"🛑 Cancelling collection timeout task before processing")
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                logger.info(f"✅ Collection task cancelled successfully")
            except Exception as e:
                logger.error(f"❌ Error cancelling collection task: {e}")

        # Get transcript from provider
        try:
            transcript_result = await self._get_transcript()
            # Process the result uniformly and get conversation_id
            conversation_id = await self._process_transcript_result(transcript_result)
            return conversation_id
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Signal transcription failure
            logger.exception(f"Transcription failed for {self._current_audio_uuid}: {e}")
            if self._current_audio_uuid:
                # Update database status to FAILED
                if self.chunk_repo:
                    await self.chunk_repo.update_transcription_status(
                        self._current_audio_uuid, "FAILED", error_message=str(e)
                    )
                # Transcription failed
                logger.error(f"Transcript failed for {self._current_audio_uuid}: {str(e)}")
            return None

    async def _get_transcript(self):
        """Get transcript from any provider using unified interface."""
        if not self.provider:
            logger.error("No transcription provider available")
            return None

        try:
            # For all providers, combine collected audio and call transcribe
            combined_audio = b"".join(chunk.audio for chunk in self._audio_buffer if chunk.audio)
            sample_rate = self._get_sample_rate()

            if not combined_audio or not sample_rate:
                logger.warning("No audio data or sample rate available for transcription")
                return None

            # Check if we should request diarization based on configuration
            config = load_diarization_settings_from_file()
            diarization_source = config.get("diarization_source", "pyannote")
            
            # Request diarization if using Deepgram as diarization source
            should_diarize = (diarization_source == "deepgram" and 
                            self.provider.name in ["Deepgram", "Deepgram-Streaming"])
            
            if should_diarize:
                logger.info(f"Requesting diarization from {self.provider.name} (diarization_source=deepgram)")
            
            return await self.provider.transcribe(combined_audio, sample_rate, diarize=should_diarize)

        except Exception as e:
            logger.error(f"Error getting transcript from {self.provider.name}: {e}")
            # Clean up buffer before re-raising
            self._audio_buffer.clear()
            self._audio_start_time = None
            self._collecting = False
            raise e
        finally:
            # Clear the buffer for all provider types (in case of success)
            self._audio_buffer.clear()
            self._audio_start_time = None
            self._collecting = False

    def _get_sample_rate(self):
        """Get sample rate from client state or audio buffer."""
        current_client = self._get_current_client()
        if current_client and current_client.sample_rate:
            return current_client.sample_rate
        elif self._audio_buffer:
            return self._audio_buffer[0].rate
        return None

    async def _process_transcript_result(self, transcript_result) -> Optional[str]:
        """Process transcript result uniformly for all providers.

        Returns:
            conversation_id if conversation was created, None otherwise
        """
        if not transcript_result or not self._current_audio_uuid:
            logger.info(f"⚠️ No transcript result to process for {self._current_audio_uuid}")
            # No transcript to process
            if self._current_audio_uuid:
                logger.info(
                    f"⚠️ No transcript data for {self._current_audio_uuid}"
                )
            return None

        start_time = time.time()

        try:
            # Store raw transcript data
            provider_name = self.provider.name if self.provider else "unknown"
            logger.info(f"transcript_result type={type(transcript_result)}, content preview: {str(transcript_result)[:200]}")
            if self.chunk_repo:
                await self.chunk_repo.store_raw_transcript_data(
                    self._current_audio_uuid, transcript_result, provider_name
                )
                logger.info(f"Successfully stored raw transcript data for {self._current_audio_uuid}")

            # Normalize transcript result
            normalized_result = self._normalize_transcript_result(transcript_result)
            if not normalized_result.get("text"):
                logger.warning(
                    f"No text in normalized transcript result for {self._current_audio_uuid}"
                )
                # Empty transcript text
                logger.warning(
                    f"⚠️ Empty transcript text for {self._current_audio_uuid}"
                )
                return None

            # Get speaker diarization with word matching (if available)
            final_segments = []
            # Prepare transcript data for speaker service (define early so it's available for fallback)
            transcript_data = {
                "words": normalized_result.get("words", []),
                "text": normalized_result.get("text", ""),
            }
            # SPEECH DETECTION: Analyze transcript for meaningful speech
            speech_analysis = self._analyze_speech(transcript_data)
            logger.info(f"🎯 Speech analysis for {self._current_audio_uuid}: {speech_analysis}")

            # Mark audio_chunks with speech detection results
            if self.chunk_repo:
                await self.chunk_repo.update_speech_detection(
                    self._current_audio_uuid,
                    has_speech=speech_analysis["has_speech"],
                    **{k: v for k, v in speech_analysis.items() if k != "has_speech"}
                )

            # Speech detection check - conversation will be created after speaker recognition
            conversation_id = None
            if not speech_analysis["has_speech"]:
                logger.info(f"⏭️ No speech detected in {self._current_audio_uuid}: {speech_analysis.get('reason', 'Unknown reason')}")
                # Update transcript status to EMPTY for silent audio
                if self.chunk_repo:
                    await self.chunk_repo.update_transcription_status(
                        self._current_audio_uuid, "EMPTY", provider=provider_name
                    )
                # No speech detected, not queuing memory processing
                logger.info(f"No speech detected for {self._current_audio_uuid}")
                return None

            # SPEECH GAP ANALYSIS: Check for conversation closure (only if speech detected)
            if speech_analysis["has_speech"]:
                analyzer = SpeechActivityAnalyzer(self._audio_timeline)
                activity = analyzer.analyze_transcript_activity(transcript_data)

                last_word_str = f"{activity['last_word_time']:.1f}s" if activity['last_word_time'] else 'N/A'
                gap_str = f"{activity['speech_gap_seconds']:.1f}s" if activity['speech_gap_seconds'] else 'N/A'
                logger.info(
                    f"📊 Speech activity analysis for {self._client_id}: "
                    f"words={activity['word_count']}, "
                    f"last_word={last_word_str}, "
                    f"gap={gap_str}"
                )

                # Check if we should close due to inactivity
                if activity['speech_gap_seconds'] and \
                   activity['speech_gap_seconds'] > analyzer.speech_inactivity_threshold:
                    logger.info(
                        f"💤 No speech for {activity['speech_gap_seconds']:.1f}s, "
                        f"closing conversation for {self._client_id}"
                    )
                    await self._trigger_conversation_close()
                    # Conversation closed due to inactivity
                    logger.info(f"Conversation closed for {self._current_audio_uuid}")
                    return None
                else:
                    # Update last word time for next analysis
                    if activity['last_word_time']:
                        self._last_word_time = activity['last_word_time']
                    logger.debug(f"Speech detected, continuing collection for {self._client_id}")

            # ONLY process speaker diarization if speech was detected
            final_segments = []
            if self.speaker_client.enabled and self._current_audio_uuid and self.chunk_repo:
                try:
                    # Get audio file path from database
                    chunk_data = await self.chunk_repo.get_chunk(self._current_audio_uuid)
                    if chunk_data and "audio_path" in chunk_data:
                        audio_path = chunk_data["audio_path"]
                        full_audio_path = f"/app/audio_chunks/{audio_path}"

                        logger.info(
                            f"🎤 Getting speaker diarization with word matching for: {full_audio_path}"
                        )

                        # Get user_id from client state
                        current_client = self._get_current_client()
                        user_id = current_client.user_id if current_client else None

                        # Call new speaker service endpoint
                        speaker_result = await self.speaker_client.diarize_identify_match(
                            full_audio_path, transcript_data, user_id=user_id
                        )

                        if speaker_result and speaker_result.get("segments"):
                            final_segments = speaker_result["segments"]
                            logger.info(
                                f"🎤 Speaker service returned {len(final_segments)} segments with matched text"
                            )
                            # Debug: Log first few segments to see text content
                            for i, seg in enumerate(final_segments[:3]):
                                logger.info(f"🔍 DEBUG: Segment {i}: text='{seg.get('text', 'MISSING')}', speaker={seg.get('speaker', 'UNKNOWN')}")
                        else:
                            logger.info("🎤 Speaker service returned no segments")
                    else:
                        logger.warning("No audio path found for speaker diarization")

                except Exception as e:
                    logger.warning(f"Speaker diarization with matching failed: {e}")

            # Only store segments if we got them from speaker service
            if not final_segments:
                logger.info(
                    f"📝 No diarization available - creating single segment from raw transcript for {self._current_audio_uuid}"
                )
                # Create a single segment from the raw transcript when speaker recognition is disabled
                if transcript_data and transcript_data.get("text"):
                    final_segments = [{
                        "text": transcript_data["text"],
                        "start": 0.0,
                        "end": 0.0,  # Duration unknown without audio analysis
                        "speaker": "",
                        "speaker_id": "",
                        "confidence": 1.0,
                    }]

            # Store final segments with required fields
            if self.chunk_repo and final_segments:
                # Process segments for storage
                segments_to_store = []
                speakers_found = set()
                speaker_names = {}

                for segment in final_segments:
                    # Add required fields for database storage
                    segment_to_store = {
                        "text": segment.get("text", ""),
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                        "speaker": segment.get("identified_as") or segment.get("speaker", ""),
                        "speaker_id": segment.get("speaker_id", ""),
                        "confidence": segment.get("confidence", 0.0),
                        "chunk_sequence": 0,
                        "absolute_timestamp": time.time() + segment.get("start", 0.0),
                    }
                    segments_to_store.append(segment_to_store)

                    # Store segments in audio_chunks (legacy support)
                    await self.chunk_repo.add_transcript_segment(
                        self._current_audio_uuid, segment_to_store
                    )

                    # Collect speaker information
                    speaker_name = segment.get("identified_as") or segment.get("speaker")
                    if speaker_name:
                        speakers_found.add(speaker_name)
                        # Map speaker_id to name if available
                        speaker_id = segment.get("speaker_id", "")
                        if speaker_id:
                            speaker_names[speaker_id] = speaker_name

                # Add speakers to audio_chunks (legacy support)
                for speaker in speakers_found:
                    await self.chunk_repo.add_speaker(self._current_audio_uuid, speaker)

                conversation_manager = get_conversation_manager()
                conversation_id = await conversation_manager.create_conversation_with_processing(
                    audio_uuid=self._current_audio_uuid,
                    transcript_data=transcript_data,
                    speech_analysis=speech_analysis,
                    speaker_segments=segments_to_store,
                    chunk_repo=self.chunk_repo
                )

                if not conversation_id:
                    logger.error(f"❌ Failed to create conversation for {self._current_audio_uuid}")
                    # Continue processing even if conversation creation fails
            else:
                # Edge case: speech detected but no segments processed
                logger.warning(f"🚨 EDGE CASE: Speech detected but no segments processed for {self._current_audio_uuid}. Developer felt this edge case can never happen. Developer wants to sleep. 😴")
                # If this actually happens, we should investigate why final_segments was empty

            # Update client state
            current_client = self._get_current_client()
            if current_client:
                current_client.update_transcript_received()

            # Transcript processing completed
            logger.info(f"Transcript completed for {self._current_audio_uuid}")

            # Queue memory processing now that transcription is complete (only for conversations with speech)
            if conversation_id:
                await self._queue_memory_processing(conversation_id)

            # Queue audio cropping if we have diarization segments and cropping is enabled
            if final_segments and os.getenv("AUDIO_CROPPING_ENABLED", "true").lower() == "true":
                await self._queue_diarization_based_cropping(final_segments)

            # Update database transcription status
            if self.chunk_repo:
                status = "EMPTY" if not normalized_result.get("text").strip() else "COMPLETED"
                await self.chunk_repo.update_transcription_status(
                    self._current_audio_uuid, status, provider=provider_name
                )

            # Legacy track_processing_stage call removed - unified pipeline uses job-based tracking

        except Exception as e:
            logger.error(f"Error processing transcript result: {e}")
            # Update database transcription status to failed
            if self.chunk_repo and self._current_audio_uuid:
                await self.chunk_repo.update_transcription_status(
                    self._current_audio_uuid, "FAILED", error_message=str(e)
                )
        finally:
            # Log total processing time
            total_duration = time.time() - start_time
            logger.info(
                f"⏱️ Total transcript processing time: {total_duration:.2f}s for client {self._client_id}"
            )

        return conversation_id

    def _normalize_transcript_result(self, transcript_result):
        """Normalize transcript result to consistent format."""
        if isinstance(transcript_result, str):
            # Handle string response (legacy offline ASR)
            return {"text": transcript_result, "words": [], "segments": []}
        elif isinstance(transcript_result, dict):
            # Handle dict response (modern providers)
            return {
                "text": transcript_result.get("text", ""),
                "words": transcript_result.get("words", []),
                "segments": transcript_result.get("segments", []),
            }
        else:
            # Invalid format
            return {"text": "", "words": [], "segments": []}

    # REMOVED: All segment creation methods have been removed.
    # Segments are now only created by the speaker service endpoint /v1/diarize-identify-match
    # which handles diarization, speaker identification, and word-to-speaker matching.
    # This keeps all the segment creation logic in one place (speaker service).

    def _analyze_speech(self, transcript_data: dict):
        """Analyze transcript for meaningful speech to determine if conversation should be created."""

        settings = get_speech_detection_settings()
        words = transcript_data.get("words", [])

        # Filter by confidence
        valid_words = [
            w for w in words
            if w.get("confidence", 0) >= settings["min_confidence"]
        ]

        if len(valid_words) < settings["min_words"]:
            return {"has_speech": False, "reason": f"Not enough valid words ({len(valid_words)} < {settings['min_words']})"}

        # Calculate speech duration
        if valid_words:
            speech_duration = valid_words[-1].get("end", 0) - valid_words[0].get("start", 0)

            return {
                "has_speech": True,
                "word_count": len(valid_words),
                "speech_start": valid_words[0].get("start", 0),
                "speech_end": valid_words[-1].get("end", 0),
                "duration": speech_duration
            }

        # Fallback for transcripts without detailed word timing
        text = transcript_data.get("text", "").strip()
        if text:
            word_count = len(text.split())
            if word_count >= settings["min_words"]:
                return {
                    "has_speech": True,
                    "word_count": word_count,
                    "speech_start": 0.0,
                    "speech_end": 0.0,  # Duration unknown
                    "duration": 0.0,
                    "fallback": True
                }

        return {"has_speech": False, "reason": "No meaningful speech content detected"}

    async def _queue_memory_processing(self, conversation_id: str):
        """Queue memory processing for a speech-detected conversation.

        Args:
            conversation_id: The conversation ID to process (not audio_uuid)
        """
        # Skip if running within unified pipeline (it handles memory queuing)
        if self._skip_memory_queuing:
            logger.info(f"⏭️  Skipping internal memory queuing for {conversation_id} (unified pipeline handles it)")
            return

        try:
            # Get conversation data from conversations collection
            conversations_repo = ConversationsRepository(conversations_col)
            conversation = await conversations_repo.get_conversation(conversation_id)
            if not conversation:
                logger.warning(
                    f"No conversation found for memory processing {conversation_id}"
                )
                return

            # Get audio session data to get user info
            audio_session = await self.chunk_repo.get_chunk(conversation["audio_uuid"])
            if not audio_session:
                logger.warning(
                    f"No audio session found for conversation {conversation_id}"
                )
                return

            # Check if we have required data
            if not all(
                [conversation_id, conversation.get("user_id"), audio_session.get("user_email")]
            ):
                logger.warning(
                    f"Memory processing skipped - missing required data for conversation {conversation_id}"
                )
                logger.warning(f"    - conversation_id: {bool(conversation_id)}")
                logger.warning(
                    f"    - user_id: {bool(conversation.get('user_id'))}"
                )
                logger.warning(
                    f"    - user_email: {bool(audio_session.get('user_email'))}"
                )
                return

            logger.info(
                f"💭 Queuing memory processing for conversation {conversation_id} (audio: {conversation['audio_uuid']})"
            )

            # Import here to avoid circular imports

            # Queue memory processing for conversation
            processor_manager = get_processor_manager()
            await processor_manager.queue_memory(
                MemoryProcessingItem(
                    conversation_id=conversation_id,
                    user_id=conversation["user_id"],
                    user_email=audio_session["user_email"],
                    client_id=self._client_id,
                    transcript_version_id=None  # Use active version
                )
            )

        except Exception as e:
            logger.error(f"Error queuing memory processing for conversation {conversation_id}: {e}")

    async def _queue_diarization_based_cropping(self, segments):
        """Queue audio cropping based on diarization segments."""
        try:
            # Import here to avoid circular imports

            # Get current client for user info
            current_client = self._get_current_client()
            if not current_client:
                logger.warning(f"No client state available for cropping {self._current_audio_uuid}")
                return

            # Get audio file path from database
            if not self.chunk_repo:
                logger.warning(
                    f"No chunk repository available for cropping {self._current_audio_uuid}"
                )
                return

            chunk_data = await self.chunk_repo.get_chunk(self._current_audio_uuid)
            if not chunk_data or "audio_path" not in chunk_data:
                logger.warning(f"No audio path found for cropping {self._current_audio_uuid}")
                return

            # Build file paths
            audio_filename = chunk_data["audio_path"]
            original_path = f"/app/audio_chunks/{audio_filename}"
            cropped_path = original_path.replace(".wav", "_cropped.wav")

            # Convert segments to cropping format (start, end tuples)
            cropping_segments = []
            for seg in segments:
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                if end > start:  # Only include valid segments
                    cropping_segments.append((start, end))

            if not cropping_segments:
                logger.debug(
                    f"No valid cropping segments from diarization for {self._current_audio_uuid}"
                )
                return

            logger.info(
                f"✂️ Queuing diarization-based cropping for {self._current_audio_uuid} "
                f"with {len(cropping_segments)} segments"
            )

            # Queue cropping with processor manager
            processor_manager = get_processor_manager()
            await processor_manager.queue_cropping(
                CroppingItem(
                    client_id=self._client_id,
                    user_id=current_client.user_id,
                    audio_uuid=self._current_audio_uuid,
                    original_path=original_path,
                    speech_segments=cropping_segments,
                    output_path=cropped_path,
                )
            )

        except Exception as e:
            logger.error(
                f"Error queuing diarization-based cropping for {self._current_audio_uuid}: {e}"
            )

    async def disconnect(self):
        """Cleanly disconnect from transcription service."""
        logger.info(
            f"🔌 disconnect called for client {self._client_id} - provider: {self.provider.name if self.provider else 'None'}"
        )

        if not self.provider:
            logger.warning("No provider to disconnect")
            return

        # Cancel collection timeout task first to prevent interference
        if self._collection_task and not self._collection_task.done():
            logger.info(f"🛑 Cancelling collection timeout task for client {self._client_id}")
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                logger.info(f"✅ Collection task cancelled successfully")
            except Exception as e:
                logger.error(f"❌ Error cancelling collection task: {e}")

        # Process any remaining audio before disconnect
        if self._collecting or self._audio_buffer:
            logger.info(
                f"📊 Processing remaining audio on disconnect - buffer size: {len(self._audio_buffer)}"
            )
            await self.process_collected_audio()

        # Disconnect the provider
        try:
            await self.provider.disconnect()
            logger.info(
                f"{self.provider.name} transcription disconnected for client {self._client_id}"
            )
        except Exception as e:
            logger.error(f"Error disconnecting from {self.provider.name}: {e}")

    async def _collection_timeout_handler(self):
        """Handle collection timeout - process audio after timeout period."""
        logger.info(
            f"⏰ Collection timeout handler started for client {self._client_id} ({self._max_collection_time}s)"
        )
        try:
            await asyncio.sleep(self._max_collection_time)
            if self._collecting and self._audio_buffer:
                logger.info(
                    f"⏰ Collection timeout reached for client {self._client_id}, processing audio (buffer: {len(self._audio_buffer)} chunks)"
                )
                await self.process_collected_audio()
            else:
                logger.info(
                    f"⏰ Collection timeout reached but no audio to process (collecting: {self._collecting}, buffer: {len(self._audio_buffer) if self._audio_buffer else 0})"
                )
        except asyncio.CancelledError:
            logger.info(f"⏰ Collection timeout cancelled for client {self._client_id}")
        except Exception as e:
            logger.error(f"❌ Error in collection timeout handler: {e}", exc_info=True)

    async def transcribe_chunk(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Process audio chunk using the configured transcription provider."""
        if not self.provider:
            logger.error("No transcription provider available")
            return

        # Clean mode-based dispatch - no exception handling for control flow
        if self.provider.mode == "streaming":
            # Streaming providers process chunks immediately
            try:
                await self.provider.process_streaming_chunk(audio_uuid, chunk, client_id)
            except Exception as e:
                logger.error(f"Error in streaming processing for {audio_uuid}: {e}")
                await self._reconnect()
        else:
            # Batch providers collect chunks for later processing
            await self._collect_audio_chunk(audio_uuid, chunk, client_id)

    async def _collect_audio_chunk(self, audio_uuid: str, chunk: AudioChunk, client_id: str):
        """Collect audio chunk for batch processing."""
        logger.debug(
            f"📥 _collect_audio_chunk called for client {client_id}, audio_uuid: {audio_uuid}"
        )
        try:
            # Update current audio UUID
            if self._current_audio_uuid != audio_uuid:
                self._current_audio_uuid = audio_uuid
                logger.info(
                    f"🆕 New audio_uuid for {self.provider.name if self.provider else 'online'} batch: {audio_uuid}"
                )

                # Reset collection state for new audio session
                self._audio_buffer.clear()
                self._audio_start_time = time.time()
                self._collecting = True

                # Reset audio timeline for new session
                self._audio_timeline.reset()
                self._buffer_start_time = time.time()
                self._last_word_time = None

                # Start collection timeout task
                if self._collection_task and not self._collection_task.done():
                    self._collection_task.cancel()
                self._collection_task = asyncio.create_task(self._collection_timeout_handler())

            # Add chunk to buffer if we have audio data
            if chunk.audio and len(chunk.audio) > 0:
                # Get sample rate from client state (set by audio processor)
                current_client = self._get_current_client()
                if current_client and current_client.sample_rate:
                    # Use sample rate from client state
                    expected_rate = current_client.sample_rate
                    if chunk.rate != expected_rate:
                        logger.warning(
                            f"⚠️ Sample rate mismatch for {client_id}: expected {expected_rate}Hz, got {chunk.rate}Hz"
                        )
                else:
                    # Fallback: no client state available, just log chunk rate
                    logger.info(
                        f"📊 Processing chunk with sample rate {chunk.rate}Hz for client {client_id} (no client state)"
                    )

                self._audio_buffer.append(chunk)

                # Update audio timeline
                self._audio_timeline.add_chunk(chunk)

                logger.debug(
                    f"📦 Collected {len(chunk.audio)} bytes for {audio_uuid} (total chunks: {len(self._audio_buffer)})"
                )

                # Track buffer start time for periodic transcription
                if not self._buffer_start_time:
                    self._buffer_start_time = time.time()

                # Check if buffer duration exceeded and safe to transcribe
                buffer_duration = time.time() - self._buffer_start_time
                if buffer_duration >= self._max_buffer_duration and not self._transcribing:
                    logger.info(
                        f"📊 Buffer duration limit reached ({buffer_duration:.1f}s), "
                        f"triggering transcription for {client_id}"
                    )
                    await self._trigger_periodic_transcription()
            else:
                logger.warning(f"⚠️ Empty audio chunk received for {audio_uuid}")

        except Exception as e:
            logger.error(f"Error collecting audio chunk for {audio_uuid}: {e}")

    async def _trigger_periodic_transcription(self):
        """Safely trigger periodic transcription with state management."""
        # Check if already transcribing or not collecting
        if self._transcribing or not self._collecting:
            logger.debug("Skipping periodic trigger - transcribing or not collecting")
            return

        # Mark as transcribing to prevent concurrent triggers
        self._transcribing = True
        try:
            await self.process_collected_audio()
        finally:
            self._transcribing = False
            self._buffer_start_time = time.time()  # Reset for next period

    async def _trigger_conversation_close(self):
        """Trigger conversation close due to inactivity but continue audio collection."""
        if not self._client_id:
            logger.warning("Cannot close conversation - missing client_id")
            return

        logger.info(f"🔚 Closing conversation for {self._client_id} due to speech inactivity")

        try:
            # Reset conversation-specific state but continue audio collection
            # Setting _current_audio_uuid to None will trigger "new session" logic
            # on next audio chunk, which will reset buffer and timeline
            self._current_audio_uuid = None
            self._last_word_time = None

            # Keep audio collection active:
            # - _collecting=True (audio stream continues)
            # - _audio_buffer (will be cleared on next chunk due to _current_audio_uuid=None)
            # - _audio_timeline (will be reset on next chunk)
            # - _buffer_start_time (will be reset on next chunk)

            logger.info(f"✅ Conversation closed for {self._client_id}, audio collection continues for new conversations")

        except Exception as e:
            logger.error(f"❌ Error closing conversation for {self._client_id}: {e}")

    async def _reconnect(self):
        """Attempt to reconnect to transcription service."""
        if not self.provider:
            logger.warning("No provider to reconnect")
            return

        logger.info("Attempting to reconnect to transcription service...")

        try:
            await self.provider.disconnect()
            await asyncio.sleep(2)  # Brief delay before reconnecting
            await self.provider.connect(self._client_id)
            logger.info(f"Successfully reconnected to {self.provider.name}")
        except Exception as e:
            logger.error(f"Reconnection to {self.provider.name} failed: {e}")
