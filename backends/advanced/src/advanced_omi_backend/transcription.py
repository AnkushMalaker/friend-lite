import asyncio
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Optional

from wyoming.audio import AudioChunk

from advanced_omi_backend.client_manager import get_client_manager
from advanced_omi_backend.config import get_speech_detection_settings, get_conversation_stop_settings, load_diarization_settings_from_file
from advanced_omi_backend.conversation_repository import get_conversation_repository
from advanced_omi_backend.database import conversations_col, ConversationsRepository
from advanced_omi_backend.llm_client import async_generate
from advanced_omi_backend.processors import AudioCroppingItem, MemoryProcessingItem, get_processor_manager
from advanced_omi_backend.speaker_recognition_client import SpeakerRecognitionClient
from advanced_omi_backend.transcript_coordinator import get_transcript_coordinator
from advanced_omi_backend.transcription_providers import (
    BaseTranscriptionProvider,
    get_transcription_provider,
)

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

    def __init__(self, chunk_repo=None, processor_manager=None):
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

    # REMOVED: Memory processing is now handled exclusively by conversation closure
    # to prevent duplicate processing. The _queue_memory_processing method has been
    # removed as part of the fix for double memory generation issue.

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

    async def process_collected_audio(self, audio_duration_seconds: Optional[float] = None):
        """Unified processing for all transcription providers."""
        logger.info(f"🚀 process_collected_audio called for client {self._client_id}")
        logger.info(
            f"📊 Current state - buffer size: {len(self._audio_buffer) if self._audio_buffer else 0}, collecting: {self._collecting}"
        )

        if not self.provider:
            logger.error("No transcription provider available")
            return

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
        transcript_result = await self._get_transcript(audio_duration_seconds)

        # Process the result uniformly
        await self._process_transcript_result(transcript_result)

    async def _get_transcript(self, audio_duration_seconds: Optional[float] = None):
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
            return None
        finally:
            # Clear the buffer for all provider types
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

    async def _process_transcript_result(self, transcript_result):
        """Process transcript result uniformly for all providers."""
        if not transcript_result or not self._current_audio_uuid:
            logger.info(f"⚠️ No transcript result to process for {self._current_audio_uuid}")
            # Even with no transcript, signal completion to unblock memory processing
            if self._current_audio_uuid:
                coordinator = get_transcript_coordinator()
                coordinator.signal_transcript_ready(self._current_audio_uuid)
                logger.info(
                    f"⚠️ Signaled transcript completion (no data) for {self._current_audio_uuid}"
                )
            return

        start_time = time.time()

        try:
            # Store raw transcript data
            provider_name = self.provider.name if self.provider else "unknown"
            if self.chunk_repo:
                await self.chunk_repo.store_raw_transcript_data(
                    self._current_audio_uuid, transcript_result, provider_name
                )

            # Normalize transcript result
            normalized_result = self._normalize_transcript_result(transcript_result)
            if not normalized_result.get("text"):
                logger.warning(
                    f"No text in normalized transcript result for {self._current_audio_uuid}"
                )
                # Signal completion even with empty text to unblock memory processing
                coordinator = get_transcript_coordinator()
                coordinator.signal_transcript_ready(self._current_audio_uuid)
                logger.warning(
                    f"⚠️ Signaled transcript completion (empty text) for {self._current_audio_uuid}"
                )
                return

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

            # Create conversation only if speech is detected
            conversation_id = None
            if speech_analysis["has_speech"]:
                conversation_id = await self._create_conversation(
                    self._current_audio_uuid, transcript_data, speech_analysis
                )
                if conversation_id:
                    logger.info(f"✅ Created conversation {conversation_id} for detected speech in {self._current_audio_uuid}")
                else:
                    logger.error(f"❌ Failed to create conversation for {self._current_audio_uuid}")
            else:
                logger.info(f"⏭️ No speech detected in {self._current_audio_uuid}: {speech_analysis.get('reason', 'Unknown reason')}")
                # Update transcript status to EMPTY for silent audio
                if self.chunk_repo:
                    await self.chunk_repo.update_transcription_status(
                        self._current_audio_uuid, "EMPTY", provider=provider_name
                    )
                # Signal completion but don't queue memory processing
                coordinator = get_transcript_coordinator()
                coordinator.signal_transcript_ready(self._current_audio_uuid)
                return

            # SPEECH GAP ANALYSIS: Check for conversation closure (only if conversation exists)
            if conversation_id:
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
                    # Signal completion and return (conversation closed)
                    coordinator = get_transcript_coordinator()
                    coordinator.signal_transcript_ready(self._current_audio_uuid)
                    return
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

                # CRITICAL: Update conversation with transcript data
                if conversation_id:
                    try:
                        conversations_repo = ConversationsRepository(conversations_col)

                        # Update conversation with transcript segments and speaker info
                        update_data = {
                            "transcript": segments_to_store,
                            "speakers_identified": list(speakers_found),
                            "speaker_names": speaker_names,
                            "updated_at": datetime.now(UTC)
                        }
                        await conversations_repo.update_conversation(conversation_id, update_data)

                        logger.info(f"✅ Updated conversation {conversation_id} with {len(segments_to_store)} transcript segments and {len(speakers_found)} speakers")
                    except Exception as e:
                        logger.error(f"Failed to update conversation {conversation_id} with transcript data: {e}")

            # Update client state
            current_client = self._get_current_client()
            if current_client:
                current_client.update_transcript_received()

            # Signal transcript coordinator
            coordinator = get_transcript_coordinator()
            coordinator.signal_transcript_ready(self._current_audio_uuid)

            # Queue memory processing now that transcription is complete (only for conversations with speech)
            if conversation_id:
                await self._queue_memory_processing(conversation_id)

            # Queue audio cropping if we have diarization segments and cropping is enabled
            if final_segments and os.getenv("AUDIO_CROPPING_ENABLED", "false").lower() == "true":
                await self._queue_diarization_based_cropping(final_segments)

            # Update database transcription status
            if self.chunk_repo:
                status = "EMPTY" if not normalized_result.get("text").strip() else "COMPLETED"
                await self.chunk_repo.update_transcription_status(
                    self._current_audio_uuid, status, provider=provider_name
                )

            # Mark transcription as completed
            if self.processor_manager and self._client_id:
                self.processor_manager.track_processing_stage(
                    self._client_id,
                    "transcription",
                    "completed",
                    {
                        "audio_uuid": self._current_audio_uuid,
                        "segments": len(final_segments),
                        "provider": provider_name,
                    },
                )

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
            if speech_duration < settings["min_duration"]:
                return {"has_speech": False, "reason": f"Speech too short ({speech_duration:.1f}s < {settings['min_duration']}s)"}

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

    async def _create_conversation(self, audio_uuid: str, transcript_data: dict, speech_analysis: dict):
        """Create conversation entry for detected speech."""
        try:
            # Get audio session info from audio_chunks
            audio_session = await self.chunk_repo.get_chunk(audio_uuid)
            if not audio_session:
                logger.error(f"No audio session found for {audio_uuid}")
                return None

            # Generate title and summary from transcript
            title = await self._generate_title(transcript_data.get("text", ""))
            summary = await self._generate_summary(transcript_data.get("text", ""))

            # Create conversation data
            conversation_id = str(uuid.uuid4())
            conversation_data = {
                "conversation_id": conversation_id,
                "audio_uuid": audio_uuid,
                "user_id": audio_session["user_id"],
                "client_id": audio_session["client_id"],
                "title": title,
                "summary": summary,
                "transcript": [],  # Will be populated by existing segment processing
                "duration_seconds": speech_analysis.get("duration", 0.0),
                "speech_start_time": speech_analysis.get("speech_start", 0.0),
                "speech_end_time": speech_analysis.get("speech_end", 0.0),
                "speakers_identified": [],
                "speaker_names": {},
                "memories": [],
                "memory_processing_status": "pending",
                "action_items": [],
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
                "session_start": datetime.fromtimestamp(audio_session.get("timestamp", 0), tz=UTC),
                "session_end": datetime.now(UTC),
            }

            # Create conversation in conversations collection
            conversations_repo = ConversationsRepository(conversations_col)
            await conversations_repo.create_conversation(conversation_data)

            # Mark audio_chunks as having speech and link to conversation
            await self.chunk_repo.mark_conversation_created(audio_uuid, conversation_id)

            logger.info(f"✅ Created conversation {conversation_id} for audio {audio_uuid} (speech detected)")
            return conversation_id

        except Exception as e:
            logger.error(f"Failed to create conversation for {audio_uuid}: {e}", exc_info=True)
            return None

    async def _generate_title(self, text: str) -> str:
        """Generate an LLM-powered title from conversation text."""
        if not text or len(text.strip()) < 10:
            return "Conversation"

        try:
            prompt = f"""Generate a concise, descriptive title (3-6 words) for this conversation transcript:

"{text[:500]}"

Rules:
- Maximum 6 words
- Capture the main topic or theme
- No quotes or special characters
- Examples: "Planning Weekend Trip", "Work Project Discussion", "Medical Appointment"

Title:"""

            title = await async_generate(prompt, temperature=0.3)
            return title.strip().strip('"').strip("'") or "Conversation"

        except Exception as e:
            logger.warning(f"Failed to generate LLM title: {e}")
            # Fallback to simple title generation
            words = text.split()[:6]
            title = " ".join(words)
            return title[:40] + "..." if len(title) > 40 else title or "Conversation"

    async def _generate_summary(self, text: str) -> str:
        """Generate an LLM-powered summary from conversation text."""
        if not text or len(text.strip()) < 10:
            return "No content"

        try:
            prompt = f"""Generate a brief, informative summary (1-2 sentences, max 120 characters) for this conversation:

"{text[:1000]}"

Rules:
- Maximum 120 characters
- 1-2 complete sentences
- Capture key topics and outcomes
- Use present tense
- Be specific and informative

Summary:"""

            summary = await async_generate(prompt, temperature=0.3)
            return summary.strip().strip('"').strip("'") or "No content"

        except Exception as e:
            logger.warning(f"Failed to generate LLM summary: {e}")
            # Fallback to simple summary generation
            return text[:120] + "..." if len(text) > 120 else text or "No content"

    async def _queue_memory_processing(self, conversation_id: str):
        """Queue memory processing for a speech-detected conversation.

        Args:
            conversation_id: The conversation ID to process (not audio_uuid)
        """
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
                    client_id=self._client_id,
                    user_id=conversation["user_id"],
                    user_email=audio_session["user_email"],
                    conversation_id=conversation_id,
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
                AudioCroppingItem(
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
