"""ElevenLabs API wrapper endpoints with speaker enhancement."""

import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import APIRouter, File, Form, Header, HTTPException, Query, UploadFile

from simple_speaker_recognition.api.core.utils import (
    safe_format_confidence,
    validate_confidence
)
from simple_speaker_recognition.core.models import SpeakerStatus
from simple_speaker_recognition.utils.audio_processing import get_audio_info
from simple_speaker_recognition.utils.elevenlabs_parser import ElevenLabsParser

router = APIRouter()
log = logging.getLogger("speaker_service")


# Dependency functions - will be resolved during integration
async def get_db():
    """Get speaker database dependency."""
    from .. import service
    return await service.get_db()


def get_audio_backend():
    """Get audio backend."""
    from .. import service
    return service.audio_backend


def get_speaker_db():
    """Get speaker database."""
    from .. import service
    return service.speaker_db


def get_auth():
    """Get auth settings."""
    from .. import service
    return service.auth


async def forward_to_elevenlabs(
    audio_data: bytes,
    params: Dict[str, Any],
    elevenlabs_api_key: str
) -> Dict[str, Any]:
    """Forward audio to ElevenLabs API and return response."""
    url = "https://api.elevenlabs.io/v1/speech-to-text"

    headers = {
        "xi-api-key": elevenlabs_api_key
    }

    # Prepare multipart form data
    form_data = aiohttp.FormData()
    form_data.add_field('file', io.BytesIO(audio_data), filename='audio.wav', content_type='audio/wav')

    # Add text form fields
    for key, value in params.items():
        if value is not None:
            # Convert boolean to lowercase string
            if isinstance(value, bool):
                form_data.add_field(key, str(value).lower())
            else:
                form_data.add_field(key, str(value))

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers=headers,
            data=form_data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                log.error(f"ElevenLabs API error: {response.status} - {error_text}")
                raise HTTPException(
                    status_code=response.status,
                    detail=f"ElevenLabs API error: {error_text}"
                )

            result = await response.json()
            log.info("Successfully received ElevenLabs response")
            return result


async def enhance_elevenlabs_response_with_speaker_id(
    audio_data: bytes,
    elevenlabs_response: Dict[str, Any],
    user_id: Optional[int],
    confidence_threshold: float = 0.15
) -> Dict[str, Any]:
    """Extract speaker segments and identify speakers from ElevenLabs response."""
    enhanced_response = elevenlabs_response.copy()

    if not user_id:
        log.warning("No user_id provided, skipping speaker identification")
        enhanced_response["speaker_enhancement"] = {
            "enabled": False,
            "provider": "elevenlabs",
            "reason": "No user_id provided for speaker identification"
        }
        return enhanced_response

    try:
        # Create temporary file for audio processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = Path(tmp_file.name)

        try:
            # Get actual audio duration for boundary validation
            audio_info = get_audio_info(str(tmp_path))
            audio_duration = audio_info["duration_seconds"]
            log.info(f"Audio file duration: {audio_duration:.6f}s")

            # Extract words from ElevenLabs response
            words = elevenlabs_response.get("words", [])

            if not words:
                log.warning("No words found in ElevenLabs response")
                enhanced_response["speaker_enhancement"] = {
                    "enabled": False,
                    "provider": "elevenlabs",
                    "reason": "No words found in response"
                }
                return enhanced_response

            # Filter only actual words (skip spacing and audio events)
            filtered_words = [w for w in words if w.get('type') == 'word']

            # Group consecutive words by speaker_id to create segments
            speaker_segments = []
            if filtered_words:
                current_segment = None

                for word in filtered_words:
                    speaker_id = word.get('speaker_id')
                    if speaker_id is None:
                        continue

                    if current_segment is None or current_segment['speaker_id'] != speaker_id:
                        # Save previous segment
                        if current_segment:
                            speaker_segments.append(current_segment)

                        # Start new segment
                        current_segment = {
                            'speaker_id': speaker_id,
                            'start_time': word.get('start', 0.0),
                            'end_time': word.get('end', 0.0),
                            'word_indices': [filtered_words.index(word)]
                        }
                    else:
                        # Extend current segment
                        current_segment['end_time'] = word.get('end', 0.0)
                        current_segment['word_indices'].append(filtered_words.index(word))

                # Don't forget the last segment
                if current_segment:
                    speaker_segments.append(current_segment)

            log.info(f"Found {len(speaker_segments)} speaker segments to identify")

            # Create enhanced words list
            enhanced_words = words.copy()

            # Get audio backend and speaker DB
            audio_backend = get_audio_backend()
            speaker_db = get_speaker_db()

            # Identify each segment
            for segment_idx, segment_info in enumerate(speaker_segments):
                try:
                    start_time = segment_info["start_time"]
                    end_time = segment_info["end_time"]

                    # Validate segment boundaries
                    if start_time >= audio_duration:
                        log.warning(f"Segment {segment_idx} start_time {start_time:.6f}s >= audio duration {audio_duration:.6f}s, skipping")
                        continue
                    if end_time > audio_duration:
                        log.warning(f"Segment {segment_idx} end_time {end_time:.6f}s > audio duration {audio_duration:.6f}s, clamping to {audio_duration:.6f}s")
                        end_time = audio_duration

                    # Load and extract segment
                    wav = audio_backend.load_wave(tmp_path, start_time, end_time)

                    # Get embedding
                    emb = await audio_backend.async_embed(wav)

                    # Identify speaker
                    found, speaker_info, confidence = await speaker_db.identify(emb, user_id=user_id)
                    confidence = validate_confidence(confidence, "elevenlabs_enhancement")

                    # Store identification result for this segment
                    segment_result = None

                    if found and confidence >= confidence_threshold:
                        segment_result = {
                            "speaker_id": speaker_info["id"],
                            "speaker_name": speaker_info["name"],
                            "confidence": confidence,
                            "status": SpeakerStatus.IDENTIFIED.value
                        }
                        confidence_str = safe_format_confidence(confidence, "elevenlabs_speaker_identification")
                        log.info(f"Identified segment {segment_idx} (speaker_id {segment_info['speaker_id']}) as {speaker_info['name']} (confidence: {confidence_str})")
                    else:
                        segment_result = {
                            "speaker_id": None,
                            "speaker_name": None,
                            "confidence": confidence if confidence is not None else 0.0,
                            "status": SpeakerStatus.UNKNOWN.value
                        }
                        confidence_str = safe_format_confidence(confidence, "elevenlabs_speaker_unknown")
                        log.info(f"Segment {segment_idx} (speaker_id {segment_info['speaker_id']}) not identified (confidence: {confidence_str})")

                    # Apply identification to all words in this segment
                    for word_idx in segment_info["word_indices"]:
                        if word_idx < len(filtered_words):
                            # Find the original index in enhanced_words
                            original_word = filtered_words[word_idx]
                            for i, w in enumerate(enhanced_words):
                                if w is original_word or (w.get('start') == original_word.get('start') and w.get('text') == original_word.get('text')):
                                    enhanced_words[i].update({
                                        "identified_speaker_id": segment_result["speaker_id"],
                                        "identified_speaker_name": segment_result["speaker_name"],
                                        "speaker_identification_confidence": segment_result["confidence"],
                                        "speaker_status": segment_result["status"]
                                    })
                                    break

                    # Store result for summary
                    segment_info["identification"] = segment_result

                except Exception as e:
                    log.warning(f"Error identifying segment {segment_idx}: {e}")
                    # Apply error status to all words in this segment
                    for word_idx in segment_info["word_indices"]:
                        if word_idx < len(filtered_words):
                            original_word = filtered_words[word_idx]
                            for i, w in enumerate(enhanced_words):
                                if w is original_word or (w.get('start') == original_word.get('start') and w.get('text') == original_word.get('text')):
                                    enhanced_words[i].update({
                                        "identified_speaker_id": None,
                                        "identified_speaker_name": None,
                                        "speaker_identification_confidence": 0.0,
                                        "speaker_status": SpeakerStatus.ERROR.value
                                    })
                                    break

            # Update the response with enhanced words
            enhanced_response["words"] = enhanced_words

            # Collect unique identified speakers
            identified_speakers = {}
            for segment in speaker_segments:
                if "identification" in segment:
                    result = segment["identification"]
                    if result["status"] == SpeakerStatus.IDENTIFIED.value:
                        # Use the ElevenLabs speaker_id as key
                        speaker_key = str(segment["speaker_id"])
                        # Only store the first occurrence of each identified speaker
                        if speaker_key not in identified_speakers:
                            identified_speakers[speaker_key] = result

            # Add speaker enhancement metadata
            enhanced_response["speaker_enhancement"] = {
                "enabled": True,
                "provider": "elevenlabs",
                "user_id": user_id,
                "confidence_threshold": confidence_threshold,
                "identified_speakers": identified_speakers,
                "total_segments": len(speaker_segments),
                "identified_segments": len([s for s in speaker_segments if s.get("identification", {}).get("status") == SpeakerStatus.IDENTIFIED.value]),
                "total_speakers": len(set(s["speaker_id"] for s in speaker_segments)),
                "identified_count": len(identified_speakers)
            }

        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)

    except Exception as e:
        log.error(f"Error during speaker identification: {e}")
        # Add error info to response but don't fail the request
        enhanced_response["speaker_enhancement"] = {
            "enabled": True,
            "provider": "elevenlabs",
            "error": str(e),
            "status": "failed"
        }

    return enhanced_response


@router.post("/elevenlabs/v1/transcribe")
async def elevenlabs_transcription_with_speaker_id(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    # ElevenLabs API parameters
    model_id: str = Form(default="scribe_v1", description="Model to use for transcription"),
    diarize: bool = Form(default=True, description="Enable speaker diarization"),
    timestamps_granularity: str = Form(default="word", description="Timestamp granularity"),
    tag_audio_events: bool = Form(default=False, description="Tag audio events like laughter"),
    # Speaker identification parameters
    user_id: Optional[int] = Query(default=None, description="User ID for speaker identification"),
    enhance_speakers: bool = Query(default=True, description="Enable speaker identification enhancement"),
    speaker_confidence_threshold: float = Query(default=0.15, ge=0.0, le=1.0, description="Minimum confidence threshold for speaker identification"),
    # API key
    xi_api_key: Optional[str] = Header(default=None, description="ElevenLabs API key")
):
    """
    Transcribe audio using ElevenLabs with speaker identification.

    This endpoint forwards the audio to ElevenLabs API for transcription with
    speaker diarization, then enhances the response with speaker identification
    from enrolled speakers.

    **Authentication**: Provide `xi-api-key` header with your ElevenLabs API key.

    **Speaker Enhancement**: If `user_id` and `enhance_speakers=true`, the service will:
    1. Forward audio to ElevenLabs for transcription with diarization
    2. Extract speaker segments from the diarized response
    3. Identify each speaker using enrolled voice embeddings
    4. Add `identified_speaker_name` and related fields to each word

    **Response Format**: Returns ElevenLabs JSON format with additional `speaker_enhancement` metadata.
    """
    # Get ElevenLabs API key from header or settings
    auth = get_auth()
    api_key = xi_api_key or auth.elevenlabs_api_key

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="ElevenLabs API key required (provide via xi-api-key header or ELEVENLABS_API_KEY env var)"
        )

    try:
        # Read audio data
        audio_data = await file.read()
        log.info(f"Received audio file: {file.filename}, size: {len(audio_data)} bytes")

        # Prepare ElevenLabs API parameters
        elevenlabs_params = {
            "model_id": model_id,
            "diarize": diarize,
            "timestamps_granularity": timestamps_granularity,
            "tag_audio_events": tag_audio_events
        }

        # Forward to ElevenLabs
        log.info(f"Forwarding to ElevenLabs API with params: {elevenlabs_params}")
        elevenlabs_response = await forward_to_elevenlabs(audio_data, elevenlabs_params, api_key)

        # Enhance with speaker identification if requested
        if enhance_speakers and user_id:
            log.info(f"Enhancing response with speaker identification for user_id={user_id}")
            enhanced_response = await enhance_elevenlabs_response_with_speaker_id(
                audio_data,
                elevenlabs_response,
                user_id,
                speaker_confidence_threshold
            )
            return enhanced_response
        else:
            # Return ElevenLabs response without enhancement
            if not enhance_speakers:
                log.info("Speaker enhancement disabled")
            elif not user_id:
                log.info("No user_id provided, skipping speaker identification")

            elevenlabs_response["speaker_enhancement"] = {
                "enabled": False,
                "provider": "elevenlabs",
                "reason": "Enhancement not requested or no user_id provided"
            }
            return elevenlabs_response

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
