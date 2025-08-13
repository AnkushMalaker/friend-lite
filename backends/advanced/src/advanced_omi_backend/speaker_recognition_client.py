"""
Speaker recognition client for integrating with the speaker recognition service.

This module provides an optional integration with the speaker recognition service
to enhance transcripts with actual speaker names instead of generic labels.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class SpeakerRecognitionClient:
    """Client for communicating with the speaker recognition service."""

    def __init__(self, service_url: Optional[str] = None):
        """
        Initialize the speaker recognition client.

        Args:
            service_url: URL of the speaker recognition service (e.g., http://speaker-service:8085)
                        If not provided, uses SPEAKER_SERVICE_URL env var
        """
        self.service_url = service_url or os.getenv("SPEAKER_SERVICE_URL")
        self.enabled = bool(self.service_url)

        if self.enabled:
            logger.info(f"Speaker recognition client initialized with URL: {self.service_url}")
        else:
            logger.info("Speaker recognition client disabled (no service URL configured)")

    async def diarize_identify_match(
        self, audio_path: str, transcript_data: Dict, user_id: Optional[str] = None
    ) -> Dict:
        """
        Perform diarization, speaker identification, and word-to-speaker matching.

        Args:
            audio_path: Path to the audio file
            transcript_data: Dict containing words array and text from transcription
            user_id: Optional user ID for speaker identification

        Returns:
            Dictionary containing segments with matched text and speaker identification
        """
        if not self.enabled:
            return {}

        try:
            logger.info(f"Diarizing, identifying, and matching words for {audio_path}")

            # Call the new speaker recognition service endpoint
            async with aiohttp.ClientSession() as session:
                # Prepare the audio file for upload
                with open(audio_path, "rb") as audio_file:
                    form_data = aiohttp.FormData()
                    form_data.add_field(
                        "file", audio_file, filename=Path(audio_path).name, content_type="audio/wav"
                    )
                    # Add transcript data as JSON string
                    form_data.add_field("transcript_data", json.dumps(transcript_data))
                    # Get current diarization settings
                    from advanced_omi_backend.controllers.system_controller import _diarization_settings
                    
                    # Add configurable parameters
                    form_data.add_field("min_duration", str(_diarization_settings["min_duration"]))
                    form_data.add_field("similarity_threshold", str(_diarization_settings["similarity_threshold"]))
                    form_data.add_field("collar", str(_diarization_settings["collar"]))
                    form_data.add_field("min_duration_off", str(_diarization_settings["min_duration_off"]))
                    if _diarization_settings.get("min_speakers"):
                        form_data.add_field("min_speakers", str(_diarization_settings["min_speakers"]))
                    if _diarization_settings.get("max_speakers"):
                        form_data.add_field("max_speakers", str(_diarization_settings["max_speakers"]))
                    
                    # TODO: Implement proper user mapping between MongoDB ObjectIds and speaker service integer IDs
                    # For now, hardcode to admin user (ID=1) since speaker service expects integer user_id
                    form_data.add_field("user_id", "1")

                    # Make the request
                    async with session.post(
                        f"{self.service_url}/v1/diarize-identify-match",
                        data=form_data,
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as response:
                        if response.status != 200:
                            logger.warning(
                                f"Speaker service returned status {response.status}: {await response.text()}"
                            )
                            return {}

                        result = await response.json()
                        logger.info(
                            f"Speaker service returned {len(result.get('segments', []))} segments with matched text"
                        )
                        return result

        except aiohttp.ClientError as e:
            logger.warning(f"Failed to connect to speaker recognition service: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error during diarize-identify-match: {e}")
            return {}

    async def diarize_and_identify(
        self, audio_path: str, words: None, user_id: Optional[str] = None  # NOT IMPLEMENTED
    ) -> Dict:
        """
        Perform diarization and speaker identification using the speaker recognition service.

        Args:
            audio_path: Path to the audio file
            words: Optional word-level data from transcription provider (for hints)
            user_id: Optional user ID for speaker identification

        Returns:
            Dictionary containing segments with speaker identification results
        """
        if words:
            logger.warning("Words parameter is not implemented yet")

        if not self.enabled:
            return {}

        try:
            logger.info(f"Diarizing and identifying speakers in {audio_path}")

            # Call the speaker recognition service
            async with aiohttp.ClientSession() as session:
                # Prepare the audio file for upload
                with open(audio_path, "rb") as audio_file:
                    form_data = aiohttp.FormData()
                    form_data.add_field(
                        "file", audio_file, filename=Path(audio_path).name, content_type="audio/wav"
                    )
                    # Get current diarization settings
                    from advanced_omi_backend.controllers.system_controller import _diarization_settings
                    
                    # Add parameters for the diarize-and-identify endpoint
                    form_data.add_field("min_duration", str(_diarization_settings["min_duration"]))
                    form_data.add_field("identify_only_enrolled", "false")
                    # TODO: Implement proper user mapping between MongoDB ObjectIds and speaker service integer IDs
                    # For now, hardcode to admin user (ID=1) since speaker service expects integer user_id
                    form_data.add_field("user_id", "1")

                    # Make the request
                    async with session.post(
                        f"{self.service_url}/diarize-and-identify",
                        data=form_data,
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as response:
                        if response.status != 200:
                            logger.warning(
                                f"Speaker recognition service returned status {response.status}: {await response.text()}"
                            )
                            return {}

                        result = await response.json()
                        logger.info(
                            f"Speaker service returned {len(result.get('segments', []))} segments"
                        )
                        return result

        except aiohttp.ClientError as e:
            logger.warning(f"Failed to connect to speaker recognition service: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error during speaker diarization and identification: {e}")
            return {}

    async def identify_speakers(self, audio_path: str, segments: List[Dict]) -> Dict[str, str]:
        """
        Identify speakers in audio segments using the speaker recognition service.

        Args:
            audio_path: Path to the audio file
            segments: List of transcript segments with speaker labels

        Returns:
            Dictionary mapping generic speaker labels to identified names
            e.g., {"Speaker 0": "ankush", "Speaker 1": "unknown_speaker_0"}
        """
        if not self.enabled:
            return {}

        try:
            # Extract unique speakers from segments
            unique_speakers = set()
            for segment in segments:
                if "speaker" in segment:
                    unique_speakers.add(segment["speaker"])

            logger.info(f"Identifying {len(unique_speakers)} speakers in {audio_path}")

            # Call the speaker recognition service
            async with aiohttp.ClientSession() as session:
                # Prepare the audio file for upload
                with open(audio_path, "rb") as audio_file:
                    form_data = aiohttp.FormData()
                    form_data.add_field(
                        "file", audio_file, filename=Path(audio_path).name, content_type="audio/wav"
                    )
                    # Add parameters for the diarize-and-identify endpoint
                    form_data.add_field("min_duration", "0.5")
                    form_data.add_field("identify_only_enrolled", "false")

                    # Make the request
                    async with session.post(
                        f"{self.service_url}/diarize-and-identify",
                        data=form_data,
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as response:
                        if response.status != 200:
                            logger.warning(
                                f"Speaker recognition service returned status {response.status}: {await response.text()}"
                            )
                            return {}

                        result = await response.json()

                        # Process the response to create speaker mapping
                        speaker_mapping = self._process_diarization_result(result, segments)

                        if speaker_mapping:
                            logger.info(f"Speaker mapping created: {speaker_mapping}")
                        else:
                            logger.warning(
                                "No speaker mapping could be created from diarization result"
                            )

                        return speaker_mapping

        except aiohttp.ClientError as e:
            logger.warning(f"Failed to connect to speaker recognition service: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error during speaker identification: {e}")
            return {}

    def _process_diarization_result(
        self, diarization_result: Dict, original_segments: List[Dict]
    ) -> Dict[str, str]:
        """
        Process the diarization result to create a mapping from generic to identified speakers.

        Args:
            diarization_result: Response from the diarize-and-identify endpoint
            original_segments: Original transcript segments with generic speaker labels

        Returns:
            Dictionary mapping generic speaker labels to identified names
        """
        try:
            identified_segments = diarization_result.get("segments", [])

            # Create a mapping based on temporal overlap between segments
            speaker_mapping = {}
            unknown_counter = 0

            # Group diarization segments by their original speaker label
            diar_speakers = {}
            for seg in identified_segments:
                speaker_label = f"Speaker {seg.get('speaker', 0)}"
                if speaker_label not in diar_speakers:
                    diar_speakers[speaker_label] = []
                diar_speakers[speaker_label].append(seg)

            # Map each generic speaker to the most common identified speaker
            for generic_speaker in diar_speakers:
                segments_for_speaker = diar_speakers[generic_speaker]

                # Count identified names for this speaker
                name_counts = {}
                for seg in segments_for_speaker:
                    identified_name = seg.get("identified_as")
                    if identified_name and identified_name != "Unknown":
                        name_counts[identified_name] = name_counts.get(identified_name, 0) + 1

                # Assign the most common identified name, or unknown if none found
                if name_counts:
                    # Get the name with the highest count
                    best_name = max(name_counts.items(), key=lambda x: x[1])[0]
                    speaker_mapping[generic_speaker] = best_name
                else:
                    # Assign unknown speaker label
                    speaker_mapping[generic_speaker] = f"unknown_speaker_{unknown_counter}"
                    unknown_counter += 1

            return speaker_mapping

        except Exception as e:
            logger.error(f"Error processing diarization result: {e}")
            return {}
