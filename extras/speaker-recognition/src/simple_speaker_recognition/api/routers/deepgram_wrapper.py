"""Deepgram API wrapper endpoints with speaker enhancement."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, UploadFile

from simple_speaker_recognition.api.core.utils import (
    safe_format_confidence,
    validate_confidence
)
from simple_speaker_recognition.core.models import SpeakerStatus
from simple_speaker_recognition.core.unified_speaker_db import UnifiedSpeakerDB
from simple_speaker_recognition.utils.audio_processing import get_audio_info

# These will be imported from the main service.py when we integrate
# from ..service import get_db, audio_backend, speaker_db, auth

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


def sanitize_params_for_deepgram(params: Dict[str, Any]) -> Dict[str, str]:
    """Convert parameters to string format expected by Deepgram API."""
    sanitized = {}
    for key, value in params.items():
        if value is None:
            continue
        elif isinstance(value, bool):
            # Convert boolean to lowercase string
            sanitized[key] = str(value).lower()
        elif isinstance(value, (int, float)):
            sanitized[key] = str(value)
        elif isinstance(value, str):
            sanitized[key] = value
        elif isinstance(value, list):
            # Handle list parameters (like keywords, search terms)
            sanitized[key] = ','.join(str(item) for item in value)
        else:
            # Convert other types to string
            sanitized[key] = str(value)
    return sanitized


async def forward_to_deepgram(
    audio_data: bytes,
    content_type: str,
    params: Dict[str, Any],
    deepgram_api_key: str
) -> Dict[str, Any]:
    """Forward audio to Deepgram API and return response."""
    auth = get_auth()
    url = f"{auth.deepgram_base_url}/v1/listen"
    
    headers = {
        "Authorization": f"Token {deepgram_api_key}",
        "Content-Type": content_type
    }
    
    # Sanitize parameters for Deepgram API
    sanitized_params = sanitize_params_for_deepgram(params)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            headers=headers,
            data=audio_data,
            params=sanitized_params
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                log.error(f"Deepgram API error: {response.status} - {error_text}")
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Deepgram API error: {error_text}"
                )
            
            result = await response.json()
            log.info("Successfully received Deepgram response")
            return result


async def enhance_deepgram_response_with_speaker_id(
    audio_data: bytes,
    deepgram_response: Dict[str, Any],
    user_id: Optional[int],
    confidence_threshold: float = 0.15
) -> Dict[str, Any]:
    """Extract speaker segments and identify speakers from Deepgram response."""
    enhanced_response = deepgram_response.copy()
    
    if not user_id:
        log.warning("No user_id provided, skipping speaker identification")
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
            
            # Extract words from Deepgram response
            results = deepgram_response.get("results", {})
            channels = results.get("channels", [])
            
            if not channels:
                log.warning("No channels found in Deepgram response")
                return enhanced_response
            
            channel = channels[0]
            alternatives = channel.get("alternatives", [])
            
            if not alternatives:
                log.warning("No alternatives found in Deepgram response")
                return enhanced_response
            
            words = alternatives[0].get("words", [])
            
            # Group consecutive words by speaker to create segments
            speaker_segments = []
            enhanced_words = []
            current_segment = None
            
            for word_info in words:
                speaker = word_info.get("speaker", 0)
                start_time = word_info.get("start", 0)
                end_time = word_info.get("end", 0)
                
                # Create enhanced word with default values
                enhanced_word = word_info.copy()
                enhanced_word.update({
                    "identified_speaker_id": None,
                    "identified_speaker_name": None,
                    "speaker_identification_confidence": None,
                    "speaker_status": SpeakerStatus.UNKNOWN.value
                })
                
                # Check if we need to start a new segment (speaker changed)
                if current_segment is None or current_segment["speaker"] != speaker:
                    # Save previous segment
                    if current_segment is not None:
                        speaker_segments.append(current_segment)
                    
                    # Start new segment
                    current_segment = {
                        "speaker": speaker,
                        "start": start_time,
                        "end": end_time,
                        "words": [word_info],
                        "word_indices": [len(enhanced_words)],  # Track word indices for later update
                        "identified": False
                    }
                else:
                    # Continue current segment
                    current_segment["end"] = end_time
                    current_segment["words"].append(word_info)
                    current_segment["word_indices"].append(len(enhanced_words))
                
                enhanced_words.append(enhanced_word)
            
            # Don't forget the last segment
            if current_segment is not None:
                speaker_segments.append(current_segment)
            
            # Process each speaker segment for identification
            audio_backend = get_audio_backend()
            speaker_db = get_speaker_db()
            
            for segment_idx, segment_info in enumerate(speaker_segments):
                try:
                    # Extract audio segment
                    start_time = segment_info["start"]
                    end_time = segment_info["end"]
                    
                    # Validate segment boundaries against actual audio duration
                    if start_time >= audio_duration:
                        log.warning(f"Segment {segment_idx} start time {start_time:.6f}s exceeds audio duration {audio_duration:.6f}s, skipping")
                        # Apply error status to all words in this segment
                        for word_idx in segment_info["word_indices"]:
                            enhanced_words[word_idx].update({
                                "identified_speaker_id": None,
                                "identified_speaker_name": None,
                                "speaker_identification_confidence": 0.0,
                                "speaker_status": SpeakerStatus.ERROR.value
                            })
                        continue
                        
                    if end_time > audio_duration:
                        log.warning(f"Segment {segment_idx} end time {end_time:.6f}s exceeds audio duration {audio_duration:.6f}s, clipping to {audio_duration:.6f}s")
                        end_time = audio_duration
                        
                    if end_time <= start_time:
                        log.warning(f"Segment {segment_idx} has invalid duration (start: {start_time:.6f}s, end: {end_time:.6f}s), skipping")
                        # Apply error status to all words in this segment
                        for word_idx in segment_info["word_indices"]:
                            enhanced_words[word_idx].update({
                                "identified_speaker_id": None,
                                "identified_speaker_name": None,
                                "speaker_identification_confidence": 0.0,
                                "speaker_status": SpeakerStatus.ERROR.value
                            })
                        continue
                    
                    # Load and extract segment
                    wav = audio_backend.load_wave(tmp_path, start_time, end_time)
                    
                    # Get embedding
                    emb = await audio_backend.async_embed(wav)
                    
                    # Identify speaker
                    found, speaker_info, confidence = await speaker_db.identify(emb, user_id=user_id)
                    confidence = validate_confidence(confidence, "deepgram_enhancement")
                    
                    # Store identification result for this segment
                    segment_result = None
                    
                    if found and confidence >= confidence_threshold:
                        segment_result = {
                            "speaker_id": speaker_info["id"],
                            "speaker_name": speaker_info["name"],
                            "confidence": confidence,
                            "status": SpeakerStatus.IDENTIFIED.value
                        }
                        confidence_str = safe_format_confidence(confidence, "deepgram_speaker_identification")
                        log.info(f"Identified segment {segment_idx} (speaker {segment_info['speaker']}) as {speaker_info['name']} (confidence: {confidence_str})")
                    else:
                        segment_result = {
                            "speaker_id": None,
                            "speaker_name": None,
                            "confidence": confidence if confidence is not None else 0.0,
                            "status": SpeakerStatus.UNKNOWN.value
                        }
                        confidence_str = safe_format_confidence(confidence, "deepgram_speaker_unknown")
                        log.info(f"Segment {segment_idx} (speaker {segment_info['speaker']}) not identified (confidence: {confidence_str})")
                    
                    # Apply identification to all words in this segment
                    for word_idx in segment_info["word_indices"]:
                        enhanced_words[word_idx].update({
                            "identified_speaker_id": segment_result["speaker_id"],
                            "identified_speaker_name": segment_result["speaker_name"],
                            "speaker_identification_confidence": segment_result["confidence"],
                            "speaker_status": segment_result["status"]
                        })
                    
                    # Store result for summary
                    segment_info["identification"] = segment_result
                        
                except Exception as e:
                    log.warning(f"Error identifying segment {segment_idx}: {e}")
                    # Apply error status to all words in this segment
                    for word_idx in segment_info["word_indices"]:
                        enhanced_words[word_idx].update({
                            "identified_speaker_id": None,
                            "identified_speaker_name": None,
                            "speaker_identification_confidence": 0.0,
                            "speaker_status": SpeakerStatus.ERROR.value
                        })
            
            # Update the response with enhanced words
            enhanced_response["results"]["channels"][0]["alternatives"][0]["words"] = enhanced_words
            
            # Collect unique identified speakers
            identified_speakers = {}
            for segment in speaker_segments:
                if "identification" in segment:
                    result = segment["identification"]
                    if result["status"] == SpeakerStatus.IDENTIFIED.value:
                        # Use the original Deepgram speaker ID as key
                        speaker_key = str(segment["speaker"])
                        # Only store the first occurrence of each identified speaker
                        if speaker_key not in identified_speakers:
                            identified_speakers[speaker_key] = result
            
            # Add speaker enhancement metadata
            enhanced_response["speaker_enhancement"] = {
                "enabled": True,
                "user_id": user_id,
                "confidence_threshold": confidence_threshold,
                "identified_speakers": identified_speakers,
                "total_segments": len(speaker_segments),
                "identified_segments": len([s for s in speaker_segments if s.get("identification", {}).get("status") == SpeakerStatus.IDENTIFIED.value]),
                "total_speakers": len(set(s["speaker"] for s in speaker_segments)),
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
            "error": str(e),
            "status": "failed"
        }
    
    return enhanced_response


@router.post("/v1/listen")
async def deepgram_compatible_transcription(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    # Deepgram-compatible query parameters
    model: str = Query(default="nova-3", description="Model to use for transcription"),
    language: str = Query(default="multi", description="Language code"),
    version: str = Query(default="latest", description="Model version"),
    punctuate: bool = Query(default=True, description="Add punctuation"),
    profanity_filter: bool = Query(default=False, description="Filter profanity"),
    diarize: bool = Query(default=True, description="Enable speaker diarization"),
    diarize_version: str = Query(default="latest", description="Diarization model version"),
    multichannel: bool = Query(default=False, description="Process multiple channels"),
    alternatives: int = Query(default=1, description="Number of alternative transcripts"),
    numerals: bool = Query(default=True, description="Convert numbers to numerals"),
    smart_format: bool = Query(default=True, description="Enable smart formatting"),
    paragraphs: bool = Query(default=True, description="Organize into paragraphs"),
    utterances: bool = Query(default=True, description="Organize into utterances"),
    detect_language: bool = Query(default=False, description="Detect language automatically"),
    summarize: bool = Query(default=False, description="Generate summary"),
    sentiment: bool = Query(default=False, description="Analyze sentiment"),
    # Speaker enhancement parameters
    enhance_speakers: bool = Query(default=True, description="Enable speaker identification enhancement"),
    user_id: Optional[int] = Query(default=None, description="User ID for speaker identification"),
    speaker_confidence_threshold: float = Query(default=0.15, description="Minimum confidence for speaker identification"),
    # Authentication
    authorization: Optional[str] = Header(default=None, description="Authorization header"),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """Deepgram-compatible transcription endpoint with speaker enhancement."""
    
    log.info("Processing /v1/listen request")
    log.info(f"Parameters - user_id: {user_id}, enhance_speakers: {enhance_speakers}, speaker_confidence_threshold: {speaker_confidence_threshold}")
    log.info(f"File - name: {file.filename}, content_type: {file.content_type}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
    
    # Extract API key from authorization header
    api_key = None
    if authorization:
        if authorization.startswith("Token "):
            api_key = authorization[6:]
        elif authorization.startswith("Bearer "):
            api_key = authorization[7:]
        else:
            api_key = authorization
    
    # Use provided API key or fall back to service default
    auth = get_auth()
    deepgram_key = api_key or auth.deepgram_api_key
    if not deepgram_key:
        raise HTTPException(
            status_code=401,
            detail="Deepgram API key required. Provide via Authorization header or DEEPGRAM_API_KEY environment variable."
        )
    
    log.info(f"Processing Deepgram-compatible transcription request: {file.filename}")
    
    try:
        # Read audio file
        audio_data = await file.read()
        content_type = file.content_type or "audio/wav"
        
        # Prepare parameters for Deepgram API
        deepgram_params = {
            "model": model,
            "language": language,
            "version": version,
            "punctuate": punctuate,
            "profanity_filter": profanity_filter,
            "diarize": diarize,
            "diarize_version": diarize_version,
            "multichannel": multichannel,
            "alternatives": alternatives,
            "numerals": numerals,
            "smart_format": smart_format,
            "paragraphs": paragraphs,
            "utterances": utterances,
            "detect_language": detect_language,
            "summarize": summarize,
            "sentiment": sentiment,
        }
        
        # Remove None values and custom parameters not for Deepgram
        deepgram_params = {k: v for k, v in deepgram_params.items() if v is not None}
        
        # Remove our custom parameters that shouldn't go to Deepgram
        custom_params = ['enhance_speakers', 'user_id', 'speaker_confidence_threshold']
        for param in custom_params:
            deepgram_params.pop(param, None)
        
        log.info(f"Forwarding request to Deepgram API with params: {deepgram_params}")
        
        # Forward to Deepgram
        deepgram_response = await forward_to_deepgram(
            audio_data=audio_data,
            content_type=content_type,
            params=deepgram_params,
            deepgram_api_key=deepgram_key
        )
        
        # Enhance with speaker identification if enabled
        if enhance_speakers and diarize:
            log.info("Enhancing transcription with speaker identification")
            enhanced_response = await enhance_deepgram_response_with_speaker_id(
                audio_data=audio_data,
                deepgram_response=deepgram_response,
                user_id=user_id,
                confidence_threshold=speaker_confidence_threshold
            )
            return enhanced_response
        else:
            log.info("Returning original Deepgram response (speaker enhancement disabled)")
            return deepgram_response
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error in Deepgram-compatible transcription: {e}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")


@router.post("/v1/transcribe-and-diarize")
async def deepgram_transcribe_and_diarize(
    file: UploadFile = File(..., description="Audio file to transcribe and diarize"),
    # Deepgram-compatible query parameters
    model: str = Query(default="nova-3", description="Model to use for transcription"),
    language: str = Query(default="multi", description="Language code"),
    version: str = Query(default="latest", description="Model version"),
    punctuate: bool = Query(default=True, description="Add punctuation"),
    profanity_filter: bool = Query(default=False, description="Filter profanity"),
    diarize: bool = Query(default=True, description="Enable speaker diarization"),
    diarize_version: str = Query(default="latest", description="Diarization model version"),
    multichannel: bool = Query(default=False, description="Process multiple channels"),
    alternatives: int = Query(default=1, description="Number of alternative transcripts"),
    numerals: bool = Query(default=True, description="Convert numbers to numerals"),
    smart_format: bool = Query(default=True, description="Enable smart formatting"),
    paragraphs: bool = Query(default=True, description="Organize into paragraphs"),
    utterances: bool = Query(default=True, description="Organize into utterances"),
    detect_language: bool = Query(default=False, description="Detect language automatically"),
    summarize: bool = Query(default=False, description="Generate summary"),
    sentiment: bool = Query(default=False, description="Analyze sentiment"),
    # Hybrid mode parameters  
    enhance_speakers: bool = Query(default=True, description="Enable speaker identification enhancement"),
    user_id: Optional[int] = Query(default=None, description="User ID for speaker identification"),
    speaker_confidence_threshold: float = Query(default=0.15, description="Minimum confidence for speaker identification"),
    similarity_threshold: float = Query(default=0.15, description="Similarity threshold for internal speaker matching"),
    min_duration: float = Query(default=1.0, description="Minimum segment duration for processing"),
    # Authentication
    authorization: Optional[str] = Header(default=None, description="Authorization header"),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """
    Hybrid transcription endpoint: Deepgram transcription + internal diarization + speaker identification.
    
    This endpoint provides hybrid mode processing:
    1. Uses Deepgram for high-quality transcription only
    2. Uses internal pyannote system for speaker diarization 
    3. Applies internal speaker identification to segments
    4. Returns transcribed text with identified speakers
    
    This is different from /v1/listen which uses Deepgram for both transcription and diarization.
    """
    
    # Extract API key from authorization header
    api_key = None
    if authorization:
        if authorization.startswith("Token "):
            api_key = authorization[6:]
        elif authorization.startswith("Bearer "):
            api_key = authorization[7:]
        else:
            api_key = authorization
    
    # Use provided API key or fall back to service default
    auth = get_auth()
    deepgram_key = api_key or auth.deepgram_api_key
    if not deepgram_key:
        raise HTTPException(
            status_code=401,
            detail="Deepgram API key required. Provide via Authorization header or DEEPGRAM_API_KEY environment variable."
        )
    
    log.info(f"Processing hybrid transcribe-and-diarize request: {file.filename}")
    
    try:
        # Read audio file
        audio_data = await file.read()
        content_type = file.content_type or "audio/wav"
        
        # Step 1: Get transcription from Deepgram (no diarization)
        deepgram_params = {
            "model": model,
            "language": language,
            "version": version,
            "punctuate": punctuate,
            "profanity_filter": profanity_filter,
            "diarize": False,  # Disable Deepgram diarization for hybrid mode
            "multichannel": multichannel,
            "alternatives": alternatives,
            "numerals": numerals,
            "smart_format": smart_format,
            "paragraphs": paragraphs,
            "utterances": utterances,
            "detect_language": detect_language,
            "summarize": summarize,
            "sentiment": sentiment,
        }
        
        # Remove None values
        deepgram_params = {k: v for k, v in deepgram_params.items() if v is not None}
        
        log.info(f"Getting Deepgram transcription with params: {deepgram_params}")
        
        # Get transcription from Deepgram
        deepgram_response = await forward_to_deepgram(
            audio_data=audio_data,
            content_type=content_type,
            params=deepgram_params,
            deepgram_api_key=deepgram_key
        )
        
        # Step 2: Perform internal diarization and speaker identification
        if enhance_speakers and user_id:
            log.info("Performing internal diarization and speaker identification")
            
            # Save audio to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = Path(tmp_file.name)
            
            try:
                # Get audio backend for diarization
                audio_backend = get_audio_backend()
                
                # Perform diarization using internal system
                segments = await audio_backend.async_diarize(tmp_path)
                
                # Apply minimum duration filter
                if min_duration > 0:
                    original_count = len(segments)
                    segments = [s for s in segments if s["duration"] >= min_duration]
                    if len(segments) < original_count:
                        log.info(f"Filtered out {original_count - len(segments)} segments shorter than {min_duration}s")
                
                # Identify speakers for each segment
                speaker_mappings = {}  # Map diarization speaker -> identified speaker
                enhanced_segments = []
                
                for i, segment in enumerate(segments):
                    try:
                        speaker_label = segment["speaker"]
                        start_time = segment["start"]
                        end_time = segment["end"]
                        
                        # Load audio segment
                        wav = audio_backend.load_wave(tmp_path, start_time, end_time)
                        
                        # Generate embedding
                        emb = await audio_backend.async_embed(wav)
                        
                        # Identify speaker
                        found, speaker_info, confidence = await db.identify(emb, user_id=user_id)
                        
                        # Store mapping for this diarization speaker
                        if found and confidence >= speaker_confidence_threshold:
                            if speaker_label not in speaker_mappings:
                                speaker_mappings[speaker_label] = {
                                    "speaker_id": speaker_info["id"],
                                    "speaker_name": speaker_info["name"],
                                    "confidence": confidence
                                }
                            
                            enhanced_segments.append({
                                "speaker": speaker_label,
                                "start": round(start_time, 3),
                                "end": round(end_time, 3),
                                "duration": round(end_time - start_time, 3),
                                "identified_speaker_id": speaker_info["id"],
                                "identified_speaker_name": speaker_info["name"],
                                "speaker_identification_confidence": round(float(confidence), 3),
                                "speaker_status": "identified"
                            })
                        else:
                            enhanced_segments.append({
                                "speaker": speaker_label,
                                "start": round(start_time, 3),
                                "end": round(end_time, 3),
                                "duration": round(end_time - start_time, 3),
                                "identified_speaker_id": None,
                                "identified_speaker_name": None,
                                "speaker_identification_confidence": round(float(confidence), 3) if confidence else 0.0,
                                "speaker_status": "unknown"
                            })
                            
                    except Exception as e:
                        log.warning(f"Error processing segment {i+1}: {str(e)}")
                        continue
                
                # Enhance Deepgram response with speaker identification
                # Create a hybrid response that combines Deepgram transcription with internal speaker mapping
                enhanced_response = deepgram_response.copy()
                
                # Add speaker enhancement metadata
                enhanced_response["speaker_enhancement"] = {
                    "enabled": True,
                    "method": "hybrid",
                    "user_id": user_id,
                    "confidence_threshold": speaker_confidence_threshold,
                    "similarity_threshold": similarity_threshold,
                    "identified_speakers": speaker_mappings,
                    "total_segments": len(enhanced_segments),
                    "identified_segments": len([s for s in enhanced_segments if s["speaker_status"] == "identified"]),
                    "diarization_speakers": list(set(s["speaker"] for s in enhanced_segments)),
                    "processing_mode": "hybrid"
                }
                
                # Add segment information for reference
                enhanced_response["internal_diarization"] = {
                    "segments": enhanced_segments,
                    "speaker_mappings": speaker_mappings
                }
                
                log.info(f"Hybrid processing complete: {len(enhanced_segments)} segments, {len(speaker_mappings)} identified speakers")
                return enhanced_response
                
            finally:
                # Clean up temporary file
                tmp_path.unlink(missing_ok=True)
        else:
            log.info("Returning Deepgram transcription only (speaker enhancement disabled)")
            # Add metadata to indicate no speaker enhancement
            deepgram_response["speaker_enhancement"] = {
                "enabled": False,
                "method": "hybrid",
                "reason": "No user_id provided or enhance_speakers disabled"
            }
            return deepgram_response
    
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error in hybrid transcribe-and-diarize: {e}")
        raise HTTPException(500, f"Hybrid processing failed: {str(e)}")


@router.post("/v1/diarize-only")
async def diarize_only(
    file: UploadFile = File(..., description="Audio file for speaker diarization only"),
    min_duration: float = Query(default=0.5, description="Minimum segment duration in seconds"),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """
    Pure speaker diarization without speaker identification.
    
    This endpoint provides diarization-only processing:
    1. Uses internal pyannote system for speaker diarization
    2. Returns speaker segments with generic labels (SPEAKER_00, SPEAKER_01, etc.)
    3. No speaker identification or transcription performed
    4. Useful for understanding speaker changes without needing enrolled speakers
    """
    log.info(f"Processing diarization-only request: {file.filename}")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = Path(tmp_file.name)
    
    try:
        # Perform diarization using internal system
        log.info(f"Performing speaker diarization on {tmp_path}")
        audio_backend = get_audio_backend()
        segments = await audio_backend.async_diarize(tmp_path)
        
        # Apply minimum duration filter
        if min_duration > 0:
            original_count = len(segments)
            segments = [s for s in segments if s["duration"] >= min_duration]
            if len(segments) < original_count:
                log.info(f"Filtered out {original_count - len(segments)} segments shorter than {min_duration}s")
        
        # Format segments for response (no speaker identification)
        diarization_segments = []
        for segment in segments:
            diarization_segments.append({
                "speaker": segment["speaker"],  # SPEAKER_00, SPEAKER_01, etc.
                "start": round(segment["start"], 3),
                "end": round(segment["end"], 3),
                "duration": round(segment["duration"], 3),
                "speaker_label": segment["speaker"],  # Generic label
                "identified": False,  # No identification performed
                "status": "diarized_only"
            })
        
        # Calculate summary
        total_duration = max(s["end"] for s in diarization_segments) if diarization_segments else 0
        unique_speakers = list(set(s["speaker"] for s in diarization_segments))
        
        log.info(f"Diarization complete: {len(diarization_segments)} segments, {len(unique_speakers)} speakers")
        
        return {
            "segments": diarization_segments,
            "summary": {
                "total_duration": round(total_duration, 2),
                "num_segments": len(diarization_segments),
                "num_speakers": len(unique_speakers),
                "speakers": unique_speakers,
                "min_duration_filter": min_duration,
                "processing_mode": "diarization_only"
            },
            "processing_info": {
                "method": "pyannote",
                "speaker_identification": False,
                "transcription": False,
                "description": "Speaker diarization only - generic speaker labels without identification"
            }
        }
        
    except Exception as e:
        log.error(f"Error during diarization-only processing: {e}")
        raise HTTPException(500, f"Diarization failed: {str(e)}")
    finally:
        # Clean up temporary file
        tmp_path.unlink(missing_ok=True)


@router.get("/v1/health")
async def deepgram_health():
    """Health check endpoint for Deepgram wrapper compatibility."""
    auth = get_auth()
    speaker_db = get_speaker_db()
    
    return {
        "status": "ok",
        "service": "Deepgram Speaker Enhancement Wrapper",
        "deepgram_configured": bool(auth.deepgram_api_key),
        "speaker_recognition_available": True,
        "enrolled_speakers": speaker_db.get_speaker_count()
    }


@router.get("/deepgram/config")
async def get_deepgram_config():
    """Get Deepgram configuration for frontend.
    
    Returns the Deepgram API key if configured, allowing the frontend
    to make direct WebSocket connections to Deepgram for streaming.
    """
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    
    if not deepgram_api_key:
        raise HTTPException(
            status_code=404,
            detail="Deepgram API key not configured on server"
        )
    
    return {
        "api_key": deepgram_api_key,
        "status": "configured"
    }