"""Speaker identification and diarization endpoints."""

import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from simple_speaker_recognition.api.core.utils import (
    safe_format_confidence,
    secure_temp_file,
    validate_confidence
)
from simple_speaker_recognition.core.models import (
    DiarizeAndIdentifyRequest,
    IdentifyResponse,
    SpeakerStatus
)
from simple_speaker_recognition.core.unified_speaker_db import UnifiedSpeakerDB
from simple_speaker_recognition.utils.audio_processing import get_audio_info
from simple_speaker_recognition.utils.analysis import create_speaker_analysis

# These will be imported from the main service.py when we integrate
# from ..service import get_db, audio_backend

router = APIRouter()
log = logging.getLogger("speaker_service")


# Dependency functions - will be resolved during integration
def get_db():
    """Get speaker database dependency."""
    from .. import service
    return service.get_db()


def get_audio_backend():
    """Get audio backend."""
    from .. import service
    return service.audio_backend


class AnnotationSegment(BaseModel):
    """Annotation segment for analysis."""
    start: float
    end: float
    speaker_label: str
    audio_file_hash: Optional[str] = None


class AnalyzeSegmentsRequest(BaseModel):
    """Request model for analyzing annotation segments."""
    segments: List[AnnotationSegment]
    method: str = "umap"
    cluster_method: str = "dbscan"
    similarity_threshold: float = 0.8


@router.post("/diarize-and-identify")
async def diarize_and_identify(
    file: UploadFile = File(..., description="Audio file for diarization and speaker identification"),
    req: DiarizeAndIdentifyRequest = Depends(),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """
    Perform speaker diarization and identify enrolled speakers in one step.
    
    This endpoint:
    1. Runs pyannote diarization to segment audio by speaker
    2. For each segment, extracts embeddings and identifies enrolled speakers
    3. Returns segments with both diarization labels and identified speaker names
    """
    log.info("Processing diarize-and-identify request")
    
    with secure_temp_file() as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    
    try:
        # Step 1: Perform diarization
        log.info(f"Step 1: Performing speaker diarization on {tmp_path}")
        audio_backend = get_audio_backend()
        segments = await audio_backend.async_diarize(tmp_path)
        
        # Apply minimum duration filter if specified
        if req.min_duration is not None:
            original_count = len(segments)
            segments = [s for s in segments if s["duration"] >= req.min_duration]
            if len(segments) < original_count:
                log.info(f"Filtered out {original_count - len(segments)} segments shorter than {req.min_duration}s")
        
        # Step 2: Identify speakers for each segment
        log.info(f"Step 2: Identifying speakers for {len(segments)} segments")
        enhanced_segments = []
        identified_speakers = set()
        unknown_speakers = set()
        
        # Use custom threshold if provided, otherwise use default
        threshold = req.similarity_threshold if req.similarity_threshold is not None else db.similarity_thr
        
        for i, segment in enumerate(segments):
            try:
                speaker_label = segment["speaker"]
                start_time = segment["start"]
                end_time = segment["end"]
                
                # Skip very short segments (less than min_duration)
                if end_time - start_time < req.min_duration:
                    log.debug(f"Skipping segment {i+1}: too short ({end_time - start_time:.2f}s)")
                    continue
                
                # Load audio segment
                wav = audio_backend.load_wave(tmp_path, start_time, end_time)
                
                # Generate embedding
                emb = await audio_backend.async_embed(wav)
                
                # Identify speaker with custom threshold
                found = False
                speaker_info = None
                confidence = 0.0
                
                # Try to identify speaker (UnifiedSpeakerDB handles speaker existence check internally)
                # Temporarily override threshold for this identification
                original_threshold = db.similarity_thr
                db.similarity_thr = threshold
                try:
                    found, speaker_info, confidence = await db.identify(emb, user_id=req.user_id)
                    confidence = validate_confidence(confidence, "diarize_and_identify")
                finally:
                    db.similarity_thr = original_threshold
                
                # Build enhanced segment
                enhanced_segment = {
                    "speaker": speaker_label,
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "duration": round(end_time - start_time, 3),
                    "identified_as": speaker_info["name"] if found and speaker_info else None,
                    "identified_id": speaker_info["id"] if found and speaker_info else None,
                    "confidence": round(float(confidence), 3) if confidence else 0.0,
                    "status": "identified" if found else "unknown"
                }
                
                # Track identified vs unknown speakers
                if found and speaker_info:
                    identified_speakers.add(speaker_info["name"])
                    confidence_str = safe_format_confidence(confidence, "diarization_segment")
                    log.debug(f"Segment {i+1}: Identified as {speaker_info['name']} (confidence: {confidence_str})")
                else:
                    unknown_speakers.add(speaker_label)
                    log.debug(f"Segment {i+1}: Unknown speaker {speaker_label}")
                
                # Only add segment if it's identified or we're not filtering
                if not req.identify_only_enrolled or found:
                    enhanced_segments.append(enhanced_segment)
                    
            except Exception as e:
                log.warning(f"Error processing segment {i+1}: {str(e)}")
                # Add segment with error status unless filtering
                if not req.identify_only_enrolled:
                    enhanced_segments.append({
                        "speaker": segment["speaker"],
                        "start": segment["start"],
                        "end": segment["end"],
                        "duration": segment["duration"],
                        "identified_as": None,
                        "identified_id": None,
                        "confidence": 0.0,
                        "status": "error",
                        "error": str(e)
                    })
        
        # Calculate summary statistics
        total_duration = max(s["end"] for s in segments) if segments else 0
        
        log.info(f"Diarization and identification complete - {len(identified_speakers)} identified, {len(unknown_speakers)} unknown")
        
        return {
            "segments": enhanced_segments,
            "summary": {
                "total_duration": round(total_duration, 2),
                "num_segments": len(enhanced_segments),
                "num_diarized_speakers": len(set(s["speaker"] for s in segments)),
                "identified_speakers": sorted(list(identified_speakers)),
                "unknown_speakers": sorted(list(unknown_speakers)),
                "similarity_threshold": threshold,
                "filtered": req.identify_only_enrolled
            }
        }
        
    except Exception as e:
        log.error(f"Error during diarize-and-identify: {e}")
        raise HTTPException(500, f"Diarize and identify failed: {str(e)}") from e
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/identify", response_model=IdentifyResponse)
async def identify(
    file: UploadFile = File(..., description="Audio file for speaker identification"),
    req: DiarizeAndIdentifyRequest = Depends(),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """
    Identify the speaker in an audio file.
    
    This endpoint is optimized for real-time processing:
    1. Assumes the audio contains speech from a single speaker
    2. Extracts embedding from the entire audio chunk
    3. Identifies the enrolled speaker
    4. Returns a single identification result
    
    Designed for use with utterance boundaries in real-time transcription.
    """
    log.info("Processing identify request")
    
    with secure_temp_file() as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
        
        # Debug: Copy WAV file to debug directory
        try:
            debug_dir = Path("/app/debug")
            if not debug_dir.exists():
                log.error(f"Debug directory does not exist, creating: {debug_dir}")
            
            # Create filename with timestamp and original filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            original_name = getattr(file, 'filename', 'utterance.wav') or 'utterance.wav'
            debug_filename = f"{timestamp}_{original_name}"
            debug_path = debug_dir / debug_filename
            
            # Copy the temp file to debug location
            shutil.copy2(tmp_path, debug_path)
            
            log.info(f"🐛 [DEBUG] WAV file dumped to: {debug_path}")
        except Exception as e:
            log.warning(f"Failed to dump debug WAV file: {e}")
    
    try:
        # Get audio info for duration
        audio_info = get_audio_info(str(tmp_path))
        duration = audio_info.get('duration', 0.0)
        
        log.info(f"Processing audio: {duration:.2f}s duration")
        
        # Load the entire audio file (no segmentation needed)
        audio_backend = get_audio_backend()
        wav = audio_backend.load_wave(tmp_path)
        
        # Generate embedding for the entire utterance
        emb = await audio_backend.async_embed(wav)
        
        # Use custom threshold if provided, otherwise use default
        threshold = req.similarity_threshold if req.similarity_threshold is not None else db.similarity_thr
        
        # Identify speaker with custom threshold
        found = False
        speaker_info = None
        confidence = 0.0
        
        # Temporarily override threshold for this identification
        original_threshold = db.similarity_thr
        db.similarity_thr = threshold
        try:
            found, speaker_info, confidence = await db.identify(emb, user_id=req.user_id)
            confidence = validate_confidence(confidence, "speaker_identification")
        finally:
            db.similarity_thr = original_threshold
        
        # Build response
        if found and speaker_info:
            confidence_str = safe_format_confidence(confidence, "speaker_identification")
            log.info(f"Speaker identified as {speaker_info['name']} (confidence: {confidence_str})")
            
            return IdentifyResponse(
                found=True,
                speaker_id=speaker_info["id"],
                speaker_name=speaker_info["name"],
                confidence=round(float(confidence), 3),
                status=SpeakerStatus.IDENTIFIED,
                similarity_threshold=threshold,
                duration=round(duration, 3)
            )
        else:
            log.info(f"Speaker not identified (confidence: {confidence:.3f} < threshold: {threshold:.3f})")
            
            return IdentifyResponse(
                found=False,
                speaker_id=None,
                speaker_name=None,
                confidence=round(float(confidence), 3),
                status=SpeakerStatus.UNKNOWN,
                similarity_threshold=threshold,
                duration=round(duration, 3)
            )
        
    except Exception as e:
        log.error(f"Error during speaker identification: {e}")
        raise HTTPException(500, f"Speaker identification failed: {str(e)}") from e
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/annotations/analyze-segments")
async def analyze_annotation_segments(
    audio_file: UploadFile = File(..., description="Audio file containing the segments"),
    segments: str = Form(..., description="JSON string of segments to analyze"),
    method: str = Form(default="umap", description="Dimensionality reduction method"),
    cluster_method: str = Form(default="dbscan", description="Clustering method"),
    similarity_threshold: float = Form(default=0.8, description="Similarity threshold"),
    db: UnifiedSpeakerDB = Depends(get_db),
):
    """
    Analyze speaker embedding clustering for annotation segments.
    
    This endpoint extracts embeddings from specific segments in an audio file
    and performs clustering analysis to help visualize speaker separation.
    """
    import json
    
    # Parse segments JSON
    try:
        segments_data = json.loads(segments)
        request_segments = [AnnotationSegment(**seg) for seg in segments_data]
    except Exception as e:
        raise HTTPException(400, f"Invalid segments JSON: {str(e)}")
    
    log.info(f"Processing segment analysis request for {len(request_segments)} segments")
    
    if len(request_segments) == 0:
        return {
            "status": "success",
            "message": "No segments provided",
            "visualization": {
                "speakers": [],
                "embeddings_2d": [],
                "embeddings_3d": [],
                "cluster_labels": [],
                "colors": []
            },
            "clustering": {"n_clusters": 0, "method": cluster_method},
            "similar_speakers": [],
            "quality_metrics": {"n_speakers": 0},
            "parameters": {
                "reduction_method": method,
                "cluster_method": cluster_method,
                "similarity_threshold": similarity_threshold
            }
        }
    
    with secure_temp_file() as tmp:
        tmp.write(await audio_file.read())
        tmp_path = Path(tmp.name)
    
    try:
        # Extract embeddings for each segment
        audio_backend = get_audio_backend()
        embeddings_dict = {}
        
        for i, segment in enumerate(request_segments):
            try:
                # Load audio segment
                wav = audio_backend.load_wave(tmp_path, segment.start, segment.end)
                
                # Generate embedding
                emb = await audio_backend.async_embed(wav)
                
                # Create unique identifier for this segment
                segment_id = f"{segment.speaker_label}_seg_{i}_{segment.start:.2f}s"
                embeddings_dict[segment_id] = emb
                
                log.debug(f"Extracted embedding for segment {i}: {segment.speaker_label} ({segment.start:.2f}s - {segment.end:.2f}s)")
                
            except Exception as e:
                log.warning(f"Failed to extract embedding for segment {i}: {e}")
                continue
        
        if not embeddings_dict:
            return {
                "status": "error",
                "message": "Failed to extract embeddings from any segments",
                "error": "No valid segments could be processed"
            }
        
        # Perform analysis on extracted embeddings
        log.info(f"Analyzing {len(embeddings_dict)} segment embeddings")
        analysis_result = create_speaker_analysis(
            embeddings_dict=embeddings_dict,
            method=method,
            cluster_method=cluster_method,
            similarity_threshold=similarity_threshold
        )
        
        if analysis_result.get("status") == "failed":
            log.error(f"Analysis failed: {analysis_result.get('error')}")
            return {
                "status": "error",
                "message": "Embedding analysis failed",
                "error": analysis_result.get("error")
            }
        
        # Add metadata about segments
        analysis_result["segment_info"] = {
            "total_segments": len(request_segments),
            "processed_segments": len(embeddings_dict),
            "unique_speakers": list(set(seg.speaker_label for seg in request_segments)),
            "total_duration": sum(seg.end - seg.start for seg in request_segments)
        }
        
        log.info(f"Segment analysis completed successfully for {len(embeddings_dict)} segments")
        return analysis_result
        
    except Exception as e:
        log.error(f"Error during segment analysis: {e}")
        raise HTTPException(500, f"Segment analysis failed: {str(e)}") from e
    finally:
        tmp_path.unlink(missing_ok=True)