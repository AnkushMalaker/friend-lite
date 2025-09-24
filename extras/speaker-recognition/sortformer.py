#!/usr/bin/env python3
"""
Test script for NVIDIA SortFormer diarization model with speaker enrollment.
Tests on conversation and enrollment audio files, then maps diarized tracks to enrolled speakers.
"""
import os
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from nemo.collections.asr.models import SortformerEncLabelModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000

def get_audio_duration(file_path):
    """Get audio duration using wave module."""
    try:
        with wave.open(file_path, 'r') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / float(sample_rate)
            return duration
    except Exception as e:
        return 0.0

def load_audio_16k_mono(path: str) -> Tuple[torch.Tensor, int]:
    """Load audio file and convert to 16kHz mono."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # convert to mono
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0), TARGET_SR

def write_temp_wav(path: str, wav: torch.Tensor, sr: int = TARGET_SR) -> None:
    """Write temporary wav file for embedding extraction."""
    sf.write(path, wav.cpu().numpy(), sr)

def get_embedding_from_file(speaker_model, file_path: str) -> Optional[torch.Tensor]:
    """Extract normalized speaker embedding from audio file."""
    try:
        with torch.no_grad():
            emb = speaker_model.get_embedding(file_path)
        
        # Handle different return types from get_embedding
        if isinstance(emb, (list, tuple)):
            emb = emb[0]
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        
        emb = emb.float().squeeze().cpu()
        # Normalize embedding
        return emb / (emb.norm(p=2) + 1e-9)
    except Exception as e:
        print(f"    ERROR extracting embedding from {file_path}: {e}")
        return None

def create_speaker_enrollment(speaker_model, enrollment_files: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
    """Create speaker enrollment centroids from multiple audio files per speaker."""
    enrollment = {}
    
    print("\n" + "="*60)
    print("SPEAKER ENROLLMENT")
    print("="*60)
    
    for speaker_name, file_list in enrollment_files.items():
        print(f"\nEnrolling {speaker_name}...")
        embeddings = []
        
        for file_path in file_list:
            if not os.path.exists(file_path):
                print(f"  WARNING: {file_path} not found")
                continue
            
            duration = get_audio_duration(file_path)
            print(f"  Processing {os.path.basename(file_path)} ({duration:.1f}s)...")
            
            emb = get_embedding_from_file(speaker_model, file_path)
            if emb is not None:
                embeddings.append(emb)
                print(f"    ✓ Embedding extracted (shape: {emb.shape})")
        
        if embeddings:
            # Average embeddings to create centroid
            centroid = torch.stack(embeddings, dim=0).mean(dim=0)
            centroid = centroid / (centroid.norm(p=2) + 1e-9)  # normalize
            enrollment[speaker_name] = centroid
            print(f"  ✓ {speaker_name} enrolled with {len(embeddings)} samples")
            print(f"    Centroid shape: {centroid.shape}")
        else:
            print(f"  ✗ Failed to enroll {speaker_name} - no valid embeddings")
    
    return enrollment

def extract_segments_embeddings(speaker_model, audio_file: str, segments: List) -> Dict[int, torch.Tensor]:
    """Extract embeddings for each diarized speaker track."""
    print("\n" + "="*60)
    print("EXTRACTING TRACK EMBEDDINGS")
    print("="*60)
    
    # Load full audio
    full_wav, sr = load_audio_16k_mono(audio_file)
    
    # Group segments by speaker
    speaker_segments = {}
    for seg in segments:
        start, end, spk_idx = float(seg[0]), float(seg[1]), int(seg[2])
        speaker_segments.setdefault(spk_idx, []).append((start, end))
    
    # Create temp directory for segment files
    temp_dir = "tmp_segments"
    os.makedirs(temp_dir, exist_ok=True)
    
    track_embeddings = {}
    
    for spk_idx, seg_list in speaker_segments.items():
        print(f"\nProcessing Speaker Track {spk_idx}...")
        print(f"  Found {len(seg_list)} segments")
        
        seg_embeddings = []
        
        for i, (start_sec, end_sec) in enumerate(seg_list):
            # Extract audio segment
            start_samp = int(start_sec * TARGET_SR)
            end_samp = int(end_sec * TARGET_SR)
            segment_wav = full_wav[start_samp:end_samp].clone()
            
            # Skip very short segments
            if segment_wav.numel() < TARGET_SR // 10:  # < 0.1 seconds
                print(f"    Skipping segment {i+1} (too short: {len(segment_wav)/TARGET_SR:.2f}s)")
                continue
            
            # Write temporary file
            temp_path = os.path.join(temp_dir, f"spk{spk_idx}_{i:03d}.wav")
            write_temp_wav(temp_path, segment_wav, TARGET_SR)
            
            # Extract embedding
            emb = get_embedding_from_file(speaker_model, temp_path)
            if emb is not None:
                seg_embeddings.append(emb)
                print(f"    ✓ Segment {i+1}: {start_sec:.2f}-{end_sec:.2f}s -> embedding extracted")
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
        
        if seg_embeddings:
            # Average embeddings for this speaker track
            track_emb = torch.stack(seg_embeddings, dim=0).mean(dim=0)
            track_emb = track_emb / (track_emb.norm(p=2) + 1e-9)  # normalize
            track_embeddings[spk_idx] = track_emb
            print(f"  ✓ Track {spk_idx}: {len(seg_embeddings)} segments -> final embedding")
        else:
            print(f"  ✗ Track {spk_idx}: No valid embeddings extracted")
    
    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    return track_embeddings

def map_speakers_to_enrollment(track_embeddings: Dict[int, torch.Tensor], 
                              enrollment: Dict[str, torch.Tensor],
                              similarity_threshold: float = 0.0) -> Dict[int, str]:
    """Map diarized speaker tracks to enrolled speaker identities."""
    print("\n" + "="*60)
    print("SPEAKER IDENTITY MAPPING")
    print("="*60)
    
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(torch.dot(a, b) / ((a.norm(p=2) + 1e-9) * (b.norm(p=2) + 1e-9)))
    
    speaker_mapping = {}
    
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Available enrolled speakers: {list(enrollment.keys())}")
    
    for track_idx, track_emb in track_embeddings.items():
        print(f"\nMapping Track {track_idx}:")
        
        best_match = None
        best_similarity = -1.0
        similarities = {}
        
        # Compare with all enrolled speakers
        for speaker_name, enrolled_emb in enrollment.items():
            similarity = cosine_similarity(track_emb, enrolled_emb)
            similarities[speaker_name] = similarity
            print(f"  vs {speaker_name}: {similarity:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_name
        
        # Assign identity based on threshold
        if best_similarity >= similarity_threshold and best_match:
            speaker_mapping[track_idx] = best_match
            print(f"  → Track {track_idx} mapped to: {best_match} (confidence: {best_similarity:.4f})")
        else:
            speaker_mapping[track_idx] = f"unknown_spk{track_idx}"
            print(f"  → Track {track_idx} mapped to: unknown_spk{track_idx} (low confidence: {best_similarity:.4f})")
    
    return speaker_mapping

def generate_labeled_segments(segments: List, speaker_mapping: Dict[int, str]) -> List[Dict]:
    """Generate final segments with speaker labels."""
    labeled_segments = []
    
    for seg in segments:
        start, end, spk_idx = float(seg[0]), float(seg[1]), int(seg[2])
        speaker_name = speaker_mapping.get(spk_idx, f"spk{spk_idx}")
        
        labeled_segments.append({
            "start": start,
            "end": end,
            "speaker": speaker_name,
            "duration": end - start
        })
    
    return labeled_segments

def test_sortformer_with_enrollment():
    """Test SortFormer diarization with speaker enrollment and mapping."""
    # Audio file paths
    test_files = {
        "conversation": "tests/assets/conversation_evan_katelyn_2min.wav",
        "evan_enrollment": [
            "tests/assets/evan/evan_001.wav",
            "tests/assets/evan/evan_002.wav",
            "tests/assets/evan/evan_003.wav",
            "tests/assets/evan/evan_004.wav"
        ],
        "katelyn_enrollment": [
            "tests/assets/katelyn/katelyn_001.wav",
            "tests/assets/katelyn/katelyn_002.wav"
        ]
    }
    
    # Check if files exist
    print("Checking audio files...")
    for category, files in test_files.items():
        if isinstance(files, str):
            files = [files]
        for file_path in files:
            if not os.path.exists(file_path):
                print(f"WARNING: {file_path} not found")
            else:
                duration = get_audio_duration(file_path)
                print(f"✓ {file_path} (duration: {duration:.1f}s)")
    
    print(f"\nLoading models on {DEVICE}...")
    try:
        # Load diarization model
        diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2").to(DEVICE)
        diar_model.eval()
        print("✓ SortFormer diarization model loaded")
        
        # Load speaker verification model  
        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large").to(DEVICE)
        speaker_model.eval()
        print("✓ TitaNet speaker embedding model loaded")
        
    except Exception as e:
        print(f"ERROR loading models: {e}")
        return
    
    # Test basic diarization first
    conversation_file = test_files["conversation"]
    if not os.path.exists(conversation_file):
        print(f"ERROR: Conversation file not found: {conversation_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"BASIC DIARIZATION TEST: {conversation_file}")
    print('='*60)
    
    try:
        segments = diar_model.diarize(audio=conversation_file, batch_size=1)
        print(f"\nFound {len(segments)} diarized segments:")
        for i, segment in enumerate(segments):
            start, end, spk = float(segment[0]), float(segment[1]), int(segment[2])
            print(f"  {i+1:2d}: {start:6.2f}-{end:6.2f}s | Speaker {spk} | Duration: {end-start:.2f}s")
        
    except Exception as e:
        print(f"ERROR during diarization: {e}")
        return
    
    # Create speaker enrollment
    enrollment_files = {
        "Evan": test_files["evan_enrollment"],
        "Katelyn": test_files["katelyn_enrollment"]
    }
    
    enrollment = create_speaker_enrollment(speaker_model, enrollment_files)
    
    if not enrollment:
        print("ERROR: No speakers enrolled successfully")
        return
    
    # Extract embeddings for diarized tracks
    track_embeddings = extract_segments_embeddings(speaker_model, conversation_file, segments)
    
    if not track_embeddings:
        print("ERROR: No track embeddings extracted")
        return
    
    # Map speaker tracks to enrolled identities
    speaker_mapping = map_speakers_to_enrollment(track_embeddings, enrollment, similarity_threshold=0.3)
    
    # Generate final labeled segments
    labeled_segments = generate_labeled_segments(segments, speaker_mapping)
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULTS WITH SPEAKER LABELS")
    print("="*60)
    
    print(f"\nLabeled segments ({len(labeled_segments)} total):")
    for i, seg in enumerate(labeled_segments):
        print(f"  {i+1:2d}: {seg['start']:6.2f}-{seg['end']:6.2f}s | {seg['speaker']:12s} | {seg['duration']:.2f}s")
    
    # Summary by speaker
    print(f"\nSpeaker summary:")
    speaker_stats = {}
    for seg in labeled_segments:
        speaker = seg['speaker']
        speaker_stats.setdefault(speaker, {'count': 0, 'total_duration': 0.0})
        speaker_stats[speaker]['count'] += 1
        speaker_stats[speaker]['total_duration'] += seg['duration']
    
    for speaker, stats in speaker_stats.items():
        print(f"  {speaker:12s}: {stats['count']:2d} segments, {stats['total_duration']:6.1f}s total")

if __name__ == "__main__":
    print("SortFormer Diarization + Speaker Enrollment Test Script")
    print("=" * 60)
    test_sortformer_with_enrollment()
    print("\nTest completed!")