"""Speaker enrollment page with guided recording and file-based enrollment."""

import streamlit as st
import numpy as np
import tempfile
import os
import time
import requests
import json
import threading
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import logging
import soundfile as sf

# Configuration from environment
SPEAKER_SERVICE_URL = os.getenv("SPEAKER_SERVICE_URL", "http://localhost:8001")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class AudioRecorder:
    """Handles real-time audio recording and quality assessment."""
    
    def __init__(self):
        self.audio_frames = []
        self.is_recording = False
        self.sample_rate = 16000  # Standard rate for speech processing
        self.quality_queue = queue.Queue()
        self.recording_duration = 0.0
        self.min_recording_time = 30.0  # Minimum 30 seconds
        self.target_recording_time = 120.0  # Target 2 minutes
        
    def audio_frame_callback(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Process incoming audio frames with proper PyAV property handling."""
        if self.is_recording:
            # Get proper PyAV frame properties
            frame_sample_rate = frame.sample_rate
            frame_samples = frame.samples  # samples per channel
            frame_format_name = frame.format.name
            frame_channels = len(frame.layout.channels)
            
            # Convert frame to numpy array
            audio_array = frame.to_ndarray()
            original_dtype = audio_array.dtype
            
            # Log detailed frame information every 50 frames
            if len(self.audio_frames) % 50 == 0:
                logger.info(f"WebRTC Frame #{len(self.audio_frames)}: "
                           f"rate={frame_sample_rate}Hz, samples={frame_samples}, "
                           f"channels={frame_channels}, format={frame_format_name}, "
                           f"shape={audio_array.shape}, dtype={audio_array.dtype}, "
                           f"range=[{audio_array.min()}, {audio_array.max()}]")
            
            # Convert to float32 for processing to avoid integer overflow
            if audio_array.dtype == np.int16:
                # Convert int16 to float32 normalized to [-1.0, 1.0]
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype in [np.float32, np.float64]:
                # Already float, just ensure float32
                audio_array = audio_array.astype(np.float32)
            else:
                # Unknown dtype, try to convert
                logger.warning(f"Unexpected audio dtype: {audio_array.dtype}, converting to float32")
                audio_array = audio_array.astype(np.float32)
            
            # Handle stereo-to-mono conversion based on PyAV format type
            original_shape = audio_array.shape
            if frame_channels > 1:
                # Determine if this is planar or packed format
                if frame_format_name.endswith('p'):
                    # PLANAR format (e.g., s16p, fltp): Shape is (channels, samples)
                    if len(audio_array.shape) == 2 and audio_array.shape[0] == frame_channels:
                        audio_array = np.mean(audio_array, axis=0, dtype=np.float32)  # Average across channels
                        if len(self.audio_frames) % 50 == 0:
                            logger.info(f"PLANAR {frame_format_name}: Converted {original_shape} to mono {audio_array.shape}, dtype={audio_array.dtype}")
                    else:
                        logger.warning(f"PLANAR format {frame_format_name} but unexpected shape: {audio_array.shape}")
                else:
                    # PACKED format (e.g., s16, flt): Shape is (1, total_samples) where total_samples = channels * samples
                    if len(audio_array.shape) == 2 and audio_array.shape[1] == frame_channels * frame_samples:
                        # Reshape to (samples, channels) then average
                        audio_array = audio_array.reshape(frame_samples, frame_channels)
                        audio_array = np.mean(audio_array, axis=1, dtype=np.float32)  # Average across channels
                        if len(self.audio_frames) % 50 == 0:
                            logger.info(f"PACKED {frame_format_name}: Reshaped {original_shape} to ({frame_samples}, {frame_channels}), then mono {audio_array.shape}, dtype={audio_array.dtype}")
                    elif len(audio_array.shape) == 1 and len(audio_array) == frame_channels * frame_samples:
                        # 1D array with interleaved samples
                        audio_array = audio_array.reshape(frame_samples, frame_channels)
                        audio_array = np.mean(audio_array, axis=1, dtype=np.float32)  # Average across channels
                        if len(self.audio_frames) % 50 == 0:
                            logger.info(f"PACKED {frame_format_name}: Reshaped 1D {original_shape} to ({frame_samples}, {frame_channels}), then mono {audio_array.shape}, dtype={audio_array.dtype}")
                    else:
                        logger.warning(f"PACKED format {frame_format_name} but unexpected shape: {audio_array.shape}")
            else:
                # Already mono - just ensure 1D
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.flatten()
                if len(self.audio_frames) % 50 == 0:
                    logger.info(f"MONO {frame_format_name}: Flattened {original_shape} to {audio_array.shape}, dtype={audio_array.dtype}")
            
            # Now convert back to int16 for consistent storage
            # Clip to prevent overflow and scale properly
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_array = (audio_array * 32767).astype(np.int16)
            
            if len(self.audio_frames) % 50 == 0:
                logger.info(f"Final conversion: from {original_dtype} → float32 processing → int16 storage, "
                           f"final range=[{audio_array.min()}, {audio_array.max()}]")
            
            # Ensure we have a 1D array for final storage (should already be 1D)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.flatten()
                logger.warning(f"Had to flatten array after conversion: {audio_array.shape}")
            
            # Validate final audio range
            if len(self.audio_frames) % 100 == 0:
                max_val = np.abs(audio_array).max()
                if max_val >= 32767:
                    logger.warning(f"Audio may be clipping! Max absolute value: {max_val}")
            
            # Store the frame
            self.audio_frames.append(audio_array)
            
            # Use actual sample rate from WebRTC frame for duration calculation
            frame_duration = len(audio_array) / frame_sample_rate
            
            # Update duration using actual sample rate
            old_duration = self.recording_duration
            self.recording_duration += frame_duration
            
            if len(self.audio_frames) % 50 == 0:
                logger.info(f"Frame duration: {frame_duration:.4f}s (using actual {frame_sample_rate}Hz), "
                           f"Total: {old_duration:.3f}s -> {self.recording_duration:.3f}s")
            
            # Perform quality assessment every 2 seconds worth of frames
            frames_for_assessment = max(1, int(2 * frame_sample_rate / max(len(audio_array), 1)))
            if len(self.audio_frames) % frames_for_assessment == 0:
                if len(self.audio_frames) % 50 == 0:
                    logger.info(f"Triggering quality assessment at frame {len(self.audio_frames)}")
                self._assess_quality(frame_sample_rate)
        
        return frame
    
    def _assess_quality(self, actual_sample_rate: int):
        """Assess current recording quality using actual sample rate."""
        if not self.audio_frames:
            logger.warning("No audio frames available for quality assessment")
            return
            
        # Use representative samples across entire recording instead of just last 20 frames
        if len(self.audio_frames) <= 50:
            # For short recordings, use all frames
            recent_audio = np.concatenate(self.audio_frames)
        else:
            # For longer recordings, sample evenly across the entire recording
            # Take every Nth frame to get a representative sample
            step = max(1, len(self.audio_frames) // 20)  # Sample 20 frames evenly distributed
            sampled_frames = self.audio_frames[::step][:20]  # Take up to 20 frames
            recent_audio = np.concatenate(sampled_frames)
        
        samples_used = min(20, len(self.audio_frames)) if len(self.audio_frames) > 50 else len(self.audio_frames)
        logger.info(f"Quality assessment: {len(self.audio_frames)} total frames, "
                   f"using {samples_used} representative samples, "
                   f"audio length={len(recent_audio)}, "
                   f"using_sample_rate={actual_sample_rate}Hz")
        
        try:
            from utils.quality import assess_audio_quality
            quality_metrics = assess_audio_quality(recent_audio, actual_sample_rate)
            
            logger.info(f"Quality metrics: snr_db={quality_metrics.get('snr_db', 0):.1f}, "
                       f"overall_quality={quality_metrics.get('overall_quality', 0):.2f}")
            
            # Put quality metrics in queue for UI update
            quality_data = {
                'duration': self.recording_duration,
                'snr_db': quality_metrics.get('snr_db', 0),
                'quality_score': quality_metrics.get('overall_quality', 0)
            }
            self.quality_queue.put(quality_data)
            
        except Exception as e:
            logger.error(f"Quality assessment error: {e}", exc_info=True)
    
    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.audio_frames = []
        self.recording_duration = 0.0
        logger.info(f"Recording started with sample_rate={self.sample_rate}Hz")
    
    def stop_recording(self):
        """Stop recording and return audio data with actual sample rate."""
        self.is_recording = False
        
        if not self.audio_frames:
            logger.warning("No audio frames captured during recording")
            return None, 0
        
        # Combine all frames and ensure int16 dtype
        full_audio = np.concatenate(self.audio_frames).astype(np.int16)
        
        # Use the duration and sample count to determine the effective sample rate
        # This should match the WebRTC frame sample rate
        if self.recording_duration > 0:
            calculated_rate = len(full_audio) / self.recording_duration
            effective_sample_rate = round(calculated_rate)
            
            # Validate sample rate is reasonable (8kHz to 96kHz range)
            if effective_sample_rate < 8000 or effective_sample_rate > 96000:
                logger.warning(f"Calculated sample rate {effective_sample_rate}Hz seems unreasonable, using fallback")
                effective_sample_rate = 48000  # Common WebRTC default
        else:
            logger.warning("No recording duration available, using fallback sample rate")
            effective_sample_rate = 48000  # Common WebRTC default
        
        # Calculate duration using different methods for verification
        duration_from_frames = self.recording_duration
        duration_from_samples_effective = len(full_audio) / effective_sample_rate
        
        logger.info(f"Recording stopped. "
                   f"Frames: {len(self.audio_frames)}, "
                   f"Total samples: {len(full_audio)}, "
                   f"Audio dtype: {full_audio.dtype}, "
                   f"Audio range: [{full_audio.min()}, {full_audio.max()}], "
                   f"Duration from frame timing: {duration_from_frames:.3f}s, "
                   f"Duration from samples@{effective_sample_rate}Hz: {duration_from_samples_effective:.3f}s, "
                   f"Final sample rate: {effective_sample_rate}Hz")
        
        return full_audio, effective_sample_rate
    
    def get_latest_quality(self):
        """Get the latest quality metrics without consuming the queue."""
        latest_data = None
        # Consume all available items to get the most recent one
        while not self.quality_queue.empty():
            try:
                latest_data = self.quality_queue.get_nowait()
            except queue.Empty:
                break
        
        if latest_data:
            logger.info(f"Retrieved latest quality data: {latest_data}")
            # Put the latest data back in the queue for consistency
            self.quality_queue.put(latest_data)
        else:
            logger.debug("No quality data available")
        
        return latest_data
    
    def clear_quality_queue(self):
        """Clear the quality queue."""
        while not self.quality_queue.empty():
            try:
                self.quality_queue.get_nowait()
            except queue.Empty:
                break

from utils.audio_processing import (
    load_audio, get_audio_info, audio_to_bytes,
    detect_speech_segments, create_temp_audio_file
)
from utils.visualization import create_waveform_plot, create_quality_metrics_plot
from utils.quality import assess_audio_quality, is_quality_sufficient_for_enrollment, get_quality_feedback_message
from database import get_db_session
from database.queries import SpeakerQueries, EnrollmentQueries

def init_session_state():
    """Initialize session state for enrollment."""
    if "enrollment_mode" not in st.session_state:
        st.session_state.enrollment_mode = "guided"
    if "recording_prompts" not in st.session_state:
        st.session_state.recording_prompts = [
            "Tell me about your typical morning routine and what you enjoy most about starting your day.",
            "Describe your favorite hobby in detail - what got you interested and why do you love it?",
            "What's the most interesting place you've visited? What made it special to you?",
            "Explain something you're passionate about and why it matters to you.",
            "Share a memorable experience from your childhood that still makes you smile.",
            "Describe your ideal weekend - what would you do and who would you spend it with?",
            "Tell me about a book, movie, or TV show that had a big impact on you.",
            "What's a skill you'd like to learn and why? How would you go about learning it?",
            "Describe your perfect meal - what would you eat and where would you have it?",
            "Tell me about someone who has been a positive influence in your life."
        ]
    if "current_prompt_index" not in st.session_state:
        st.session_state.current_prompt_index = 0
    if "audio_recorder" not in st.session_state:
        st.session_state.audio_recorder = AudioRecorder()
    if "recording_state" not in st.session_state:
        st.session_state.recording_state = "stopped"  # stopped, recording, processing
    if "recorded_audio" not in st.session_state:
        st.session_state.recorded_audio = None
    if "quality_metrics" not in st.session_state:
        st.session_state.quality_metrics = {}
    if "recorded_audio" not in st.session_state:
        st.session_state.recorded_audio = None
    if "recording_quality" not in st.session_state:
        st.session_state.recording_quality = None
    if "enrollment_files" not in st.session_state:
        st.session_state.enrollment_files = []

def enrollment_mode_selection():
    """Allow user to select enrollment mode."""
    st.subheader("🎯 Choose Enrollment Method")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Guided Recording", use_container_width=True, type="primary" if st.session_state.enrollment_mode == "guided" else "secondary"):
            st.session_state.enrollment_mode = "guided"
            st.rerun()
        
        st.markdown("""
        **Guided Recording:**
        - Live microphone recording
        - Conversation prompts to help you speak
        - Real-time quality feedback
        - Minimum 2-minute recording
        """)
    
    with col2:
        if st.button("📁 File Upload", use_container_width=True, type="primary" if st.session_state.enrollment_mode == "file" else "secondary"):
            st.session_state.enrollment_mode = "file"
            st.rerun()
        
        st.markdown("""
        **File Upload:**
        - Upload existing audio files
        - Batch processing multiple files
        - Automatic quality assessment
        - Select best segments
        """)

def guided_recording_interface():
    """Interface for guided audio recording."""
    st.subheader("🎤 Guided Recording Session")
    
    # Speaker info input
    col1, col2 = st.columns(2)
    
    with col1:
        speaker_id = st.text_input(
            "Speaker ID:",
            placeholder="e.g., john_doe",
            help="Unique identifier for this speaker",
            key="guided_speaker_id"
        )
    
    with col2:
        speaker_name = st.text_input(
            "Speaker Name:",
            placeholder="e.g., John Doe",
            help="Display name for this speaker",
            key="guided_speaker_name"
        )
    
    if not speaker_id or not speaker_name:
        st.warning("Please enter both Speaker ID and Name to continue.")
        return
    
    # Check if speaker already exists
    db = get_db_session()
    try:
        existing_speaker = SpeakerQueries.get_speaker(db, speaker_id)
        if existing_speaker:
            st.warning(f"⚠️ Speaker '{speaker_id}' already exists. This will add to their existing enrollment data.")
    except Exception as e:
        st.error(f"Error checking speaker: {str(e)}")
    finally:
        db.close()
    
    st.divider()
    
    # Recording prompts
    st.subheader("💬 Conversation Prompts")
    st.info("Speak naturally and continuously. These prompts will help you provide varied speech samples.")
    
    # Prompt navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("◀️ Previous") and st.session_state.current_prompt_index > 0:
            st.session_state.current_prompt_index -= 1
            st.rerun()
    
    with col2:
        st.markdown(f"**Prompt {st.session_state.current_prompt_index + 1} of {len(st.session_state.recording_prompts)}:**")
        
        current_prompt = st.session_state.recording_prompts[st.session_state.current_prompt_index]
        st.markdown(f"*{current_prompt}*")
    
    with col3:
        if st.button("Next ▶️") and st.session_state.current_prompt_index < len(st.session_state.recording_prompts) - 1:
            st.session_state.current_prompt_index += 1
            st.rerun()
    
    st.divider()
    
    # Recording interface
    st.subheader("🔴 Live Recording")
    
    # WebRTC audio streamer - use device default settings
    webrtc_ctx = webrtc_streamer(
        key="audio-recording",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True, "video": False},  # Let device use optimal settings
        audio_frame_callback=st.session_state.audio_recorder.audio_frame_callback,
    )
    
    # Recording controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔴 Start Recording", 
                     disabled=st.session_state.recording_state == "recording",
                     type="primary" if st.session_state.recording_state == "stopped" else "secondary"):
            if webrtc_ctx.state.playing:
                st.session_state.audio_recorder.start_recording()
                st.session_state.recording_state = "recording"
                st.session_state.audio_recorder.clear_quality_queue()
                st.rerun()
            else:
                st.error("Please allow microphone access and ensure WebRTC is connected.")
    
    with col2:
        if st.button("⏹️ Stop Recording", 
                     disabled=st.session_state.recording_state != "recording",
                     type="primary" if st.session_state.recording_state == "recording" else "secondary"):
            audio_data, sample_rate = st.session_state.audio_recorder.stop_recording()
            st.session_state.recording_state = "processing"
            st.session_state.recorded_audio = (audio_data, sample_rate)
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset", 
                     disabled=st.session_state.recording_state == "recording"):
            st.session_state.audio_recorder = AudioRecorder()
            st.session_state.recording_state = "stopped"
            st.session_state.recorded_audio = None
            st.session_state.quality_metrics = {}
            st.rerun()
    
    # WebRTC connection status
    if webrtc_ctx.state.playing:
        st.info("🎤 Microphone connected and ready")
    else:
        st.warning("🎤 Please allow microphone access to start recording")
    
    # Recording status and quality monitoring
    if st.session_state.recording_state == "recording":
        # Get latest quality metrics
        latest_quality = st.session_state.audio_recorder.get_latest_quality()
        if latest_quality:
            st.session_state.quality_metrics = latest_quality
        
        duration = st.session_state.quality_metrics.get('duration', 0)
        progress = min(duration / st.session_state.audio_recorder.target_recording_time, 1.0)
        
        # Recording timer
        min_time = st.session_state.audio_recorder.min_recording_time
        target_time = st.session_state.audio_recorder.target_recording_time
        
        # Stable timer display (no dynamic containers)
        st.markdown(f"### 🔴 LIVE Recording: {duration:.1f} seconds")
        
        if duration < min_time:
            progress_text = f"Continue recording... {duration:.1f}s / {min_time:.0f}s minimum (target: {target_time:.0f}s)"
            st.warning(f"⏱️ {progress_text}")
        elif duration < target_time:
            progress_text = f"Minimum reached! {duration:.1f}s / {target_time:.0f}s target"
            st.info(f"✅ {progress_text}")
        else:
            progress_text = f"Excellent duration! {duration:.1f}s (target achieved)"
            st.success(f"🎉 {progress_text}")
        
        # Progress bar with current status
        st.progress(progress, text=f"Progress: {progress:.1%}")
        
        # Manual refresh button for live updates
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Refresh Progress", key="refresh_progress"):
                st.rerun()
        
        # Debug information display
        with st.expander("🔧 Debug Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Recording Status:**")
                st.write(f"• Total frames captured: {len(st.session_state.audio_recorder.audio_frames)}")
                st.write(f"• Recording duration: {st.session_state.audio_recorder.recording_duration:.3f}s")
                if st.session_state.audio_recorder.audio_frames:
                    total_samples = sum(len(frame) for frame in st.session_state.audio_recorder.audio_frames)
                    st.write(f"• Total samples: {total_samples:,}")
            
            with col2:
                st.write("**Audio Properties:**")
                st.write("• Sample rate: Auto-detected from device")
                st.write("• Format: Adaptive to device capabilities")
                st.write("• Check docker logs for detected properties")
        
        # Real-time quality indicators
        if st.session_state.quality_metrics:
            st.subheader("📊 Quality Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                quality_score = st.session_state.quality_metrics.get('quality_score', 0)
                st.metric("Quality Score", f"{quality_score:.1%}")
            
            with col2:
                snr_db = st.session_state.quality_metrics.get('snr_db', 0)
                st.metric("SNR", f"{snr_db:.1f} dB")
            
            with col3:
                st.metric("Duration", f"{duration:.1f}s")
            
            # Quality feedback with error handling
            try:
                if quality_score > 0:
                    # Convert quality_metrics to the format expected by get_quality_feedback_message
                    quality_assessment = {
                        'overall_quality': quality_score,
                        'quality_level': 'Excellent' if quality_score >= 0.8 else 'Good' if quality_score >= 0.6 else 'Acceptable' if quality_score >= 0.4 else 'Poor'
                    }
                    
                    from utils.quality import get_quality_feedback_message
                    feedback = get_quality_feedback_message(quality_assessment)
                    if "excellent" in feedback.lower() or "good" in feedback.lower():
                        st.success(f"✅ {feedback}")
                    elif "acceptable" in feedback.lower():
                        st.info(f"ℹ️ {feedback}")
                    else:
                        st.warning(f"⚠️ {feedback}")
            except Exception as e:
                logger.error(f"Error generating quality feedback: {e}")
                st.info("ℹ️ Quality assessment in progress...")
    
    elif st.session_state.recording_state == "processing":
        st.info("🔄 Processing recorded audio...")
        
        # Process the recorded audio
        if st.session_state.recorded_audio:
            audio_data, sample_rate = st.session_state.recorded_audio
            
            if audio_data is not None and len(audio_data) > 0:
                # Assess final quality
                try:
                    from utils.quality import assess_audio_quality, is_quality_sufficient_for_enrollment
                    
                    quality_metrics = assess_audio_quality(audio_data, sample_rate)
                    st.session_state.recording_quality = quality_metrics
                    
                    # Display final results
                    st.subheader("📊 Recording Quality Assessment")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Duration", f"{len(audio_data)/sample_rate:.1f}s")
                    with col2:
                        st.metric("Quality Score", f"{quality_metrics.get('overall_quality', 0):.1%}")
                    with col3:
                        st.metric("SNR", f"{quality_metrics.get('snr_db', 0):.1f} dB")
                    
                    # Quality assessment
                    if is_quality_sufficient_for_enrollment(quality_metrics['overall_quality']):
                        st.success("✅ Recording quality is excellent for speaker enrollment!")
                        
                        # Enroll button
                        if st.button("🎯 Enroll Speaker with This Recording", type="primary"):
                            enroll_speaker_with_audio(speaker_id, speaker_name, audio_data, sample_rate, quality_metrics)
                    else:
                        st.warning("⚠️ Recording quality could be improved. You can still enroll or try recording again.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🎯 Enroll Anyway", type="secondary"):
                                enroll_speaker_with_audio(speaker_id, speaker_name, audio_data, sample_rate, quality_metrics)
                        with col2:
                            if st.button("🔄 Record Again", type="primary"):
                                st.session_state.recording_state = "stopped"
                                st.session_state.recorded_audio = None
                                st.rerun()
                    
                    # Audio preview
                    st.subheader("🎵 Preview Recording")
                    audio_bytes = audio_to_bytes(audio_data, sample_rate, 'wav')
                    st.audio(audio_bytes, format='audio/wav')
                    
                except Exception as e:
                    st.error(f"Error processing recording: {str(e)}")
                    st.session_state.recording_state = "stopped"
            else:
                st.error("No audio data recorded. Please try again.")
                st.session_state.recording_state = "stopped"
    
    else:
        # Stopped state - show instructions
        st.info("""
        **📋 Recording Instructions:**
        
        1. **Allow microphone access** when prompted by your browser
        2. **Click Start Recording** when ready
        3. **Speak naturally** following the conversation prompt above
        4. **Continue for at least 30 seconds** (target 2 minutes for best results)
        5. **Click Stop Recording** when finished
        
        **Tips for better quality:**
        - Speak in a quiet environment
        - Maintain consistent distance from microphone
        - Speak clearly and naturally
        - Avoid background noise and interruptions
        """)
    
def enroll_speaker_with_audio(speaker_id: str, speaker_name: str, audio_data: np.ndarray, sample_rate: int, quality_metrics: dict):
    """Enroll a speaker using recorded audio data."""
    try:
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Save audio to temporary file
        sf.write(temp_path, audio_data, sample_rate)
        
        # Enroll with speaker service
        response = call_speaker_service_enrollment(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            audio_data=audio_data,
            sample_rate=sample_rate,
            filename=f"{speaker_id}_live_recording.wav"
        )
        
        if response and response.get('success'):
            st.success(f"✅ Successfully enrolled speaker '{speaker_name}' with live recording!")
            
            # Reset recording state
            st.session_state.recording_state = "stopped"
            st.session_state.recorded_audio = None
            st.session_state.audio_recorder = AudioRecorder()
            
            # Show success details
            st.info(f"📊 Enrollment completed with {quality_metrics.get('overall_quality', quality_metrics.get('quality_score', 0)):.1%} quality score")
            
        else:
            error_msg = response.get('error', 'Unknown error') if response else 'No response from service'
            st.error(f"❌ Failed to enroll speaker: {error_msg}")
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
            
    except Exception as e:
        st.error(f"Error during enrollment: {str(e)}")
        logger.error(f"Enrollment error: {e}")
        
        # Reset state on error
        st.session_state.recording_state = "stopped"

def file_upload_interface():
    """Interface for file-based enrollment."""
    st.subheader("📁 File Upload Enrollment")
    
    # Speaker info input
    col1, col2 = st.columns(2)
    
    with col1:
        speaker_id = st.text_input(
            "Speaker ID:",
            placeholder="e.g., john_doe",
            help="Unique identifier for this speaker",
            key="file_speaker_id"
        )
    
    with col2:
        speaker_name = st.text_input(
            "Speaker Name:",
            placeholder="e.g., John Doe",
            help="Display name for this speaker",
            key="file_speaker_name"
        )
    
    if not speaker_id or not speaker_name:
        st.warning("Please enter both Speaker ID and Name to continue.")
        return
    
    # Check if speaker already exists
    db = get_db_session()
    try:
        existing_speaker = SpeakerQueries.get_speaker(db, speaker_id)
        if existing_speaker:
            st.info(f"ℹ️ Speaker '{speaker_id}' already exists. New files will be added to their enrollment data.")
    except Exception as e:
        st.error(f"Error checking speaker: {str(e)}")
        return
    finally:
        db.close()
    
    st.divider()
    
    # File upload
    st.subheader("📤 Upload Audio Files")
    
    uploaded_files = st.file_uploader(
        "Choose audio files for enrollment:",
        type=['wav', 'flac', 'mp3', 'm4a', 'ogg'],
        accept_multiple_files=True,
        help="Upload one or more audio files containing the speaker's voice"
    )
    
    if uploaded_files:
        st.session_state.enrollment_files = uploaded_files
        
        # Process uploaded files
        st.subheader("📊 File Analysis")
        
        file_analyses = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"📁 {uploaded_file.name}", expanded=True):
                
                try:
                    # Save file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    # Load and analyze audio
                    with st.spinner(f"Analyzing {uploaded_file.name}..."):
                        audio_data, sample_rate = load_audio(temp_path)
                        audio_info = get_audio_info(temp_path)
                        quality_assessment = assess_audio_quality(audio_data, sample_rate)
                    
                    # Display basic info
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Duration", f"{audio_info['duration_seconds']:.1f}s")
                    with col2:
                        st.metric("Quality", f"{quality_assessment['overall_quality']:.1%}")
                    with col3:
                        st.metric("SNR", f"{quality_assessment['snr_db']:.1f} dB")
                    with col4:
                        st.metric("Sample Rate", f"{sample_rate/1000:.1f} kHz")
                    
                    # Quality feedback
                    quality_message = get_quality_feedback_message(quality_assessment)
                    if quality_assessment['quality_color'] == 'green':
                        st.success(quality_message)
                    elif quality_assessment['quality_color'] == 'blue':
                        st.info(quality_message)
                    elif quality_assessment['quality_color'] == 'orange':
                        st.warning(quality_message)
                    else:
                        st.error(quality_message)
                    
                    # Waveform visualization
                    if st.checkbox(f"Show waveform", key=f"waveform_{i}"):
                        fig = create_waveform_plot(
                            audio_data, sample_rate,
                            title=f"Waveform - {uploaded_file.name}",
                            height=200
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Audio playback
                    audio_bytes = audio_to_bytes(audio_data, sample_rate)
                    st.audio(audio_bytes, format='audio/wav')
                    
                    # Store analysis results
                    file_analyses.append({
                        'filename': uploaded_file.name,
                        'temp_path': temp_path,
                        'audio_data': audio_data,
                        'sample_rate': sample_rate,
                        'audio_info': audio_info,
                        'quality_assessment': quality_assessment,
                        'include_in_enrollment': is_quality_sufficient_for_enrollment(quality_assessment['overall_quality'])
                    })
                    
                    # Recommendations
                    if quality_assessment['recommendations']:
                        with st.expander("💡 Quality Recommendations"):
                            for rec in quality_assessment['recommendations']:
                                st.write(f"• {rec}")
                
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue
        
        st.divider()
        
        # Enrollment summary and controls
        if file_analyses:
            enrollment_summary_and_controls(speaker_id, speaker_name, file_analyses)

def enrollment_summary_and_controls(speaker_id: str, speaker_name: str, file_analyses: list):
    """Display enrollment summary and process enrollment."""
    st.subheader("📋 Enrollment Summary")
    
    # Filter files for enrollment
    suitable_files = [f for f in file_analyses if f['include_in_enrollment']]
    total_duration = sum(f['audio_info']['duration_seconds'] for f in suitable_files)
    avg_quality = np.mean([f['quality_assessment']['overall_quality'] for f in suitable_files]) if suitable_files else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Files Selected", f"{len(suitable_files)}/{len(file_analyses)}")
    with col2:
        st.metric("Total Duration", f"{total_duration:.1f}s")
    with col3:
        st.metric("Average Quality", f"{avg_quality:.1%}")
    with col4:
        enrollment_ready = len(suitable_files) > 0 and total_duration >= 10
        st.metric("Ready for Enrollment", "✅ Yes" if enrollment_ready else "❌ No")
    
    # File selection table
    st.subheader("📁 File Selection")
    
    for i, analysis in enumerate(file_analyses):
        col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 1])
        
        with col1:
            include = st.checkbox(
                "",
                value=analysis['include_in_enrollment'],
                key=f"include_{i}",
                help="Include this file in enrollment"
            )
            analysis['include_in_enrollment'] = include
        
        with col2:
            st.write(f"**{analysis['filename']}**")
        
        with col3:
            st.write(f"{analysis['audio_info']['duration_seconds']:.1f}s")
        
        with col4:
            quality = analysis['quality_assessment']['overall_quality']
            quality_color = analysis['quality_assessment']['quality_color']
            st.markdown(f":{quality_color}[{quality:.1%}]")
        
        with col5:
            level = analysis['quality_assessment']['quality_level']
            st.write(level)
    
    # Update metrics based on current selection
    selected_files = [f for f in file_analyses if f['include_in_enrollment']]
    if selected_files:
        total_selected_duration = sum(f['audio_info']['duration_seconds'] for f in selected_files)
        avg_selected_quality = np.mean([f['quality_assessment']['overall_quality'] for f in selected_files])
        
        st.info(f"Selected: {len(selected_files)} files, {total_selected_duration:.1f}s total, {avg_selected_quality:.1%} avg quality")
    
    st.divider()
    
    # Enrollment controls
    st.subheader("🚀 Process Enrollment")
    
    if not selected_files:
        st.warning("No files selected for enrollment. Please select at least one file with acceptable quality.")
        return
    
    if total_selected_duration < 10:
        st.warning("Total selected audio is less than 10 seconds. Consider adding more audio for better results.")
    
    # Enrollment button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🎯 Enroll Speaker", type="primary", use_container_width=True):
            process_enrollment(speaker_id, speaker_name, selected_files)

def process_enrollment(speaker_id: str, speaker_name: str, selected_files: list):
    """Process the speaker enrollment with selected files."""
    st.subheader("⚙️ Processing Enrollment")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Create or get speaker
        status_text.text("Creating speaker profile...")
        progress_bar.progress(0.1)
        
        db = get_db_session()
        try:
            existing_speaker = SpeakerQueries.get_speaker(db, speaker_id)
            if not existing_speaker:
                speaker = SpeakerQueries.create_speaker(db, speaker_id, speaker_name, st.session_state.user_id)
                st.success(f"✅ Created new speaker: {speaker_name}")
            else:
                speaker = existing_speaker
                st.info(f"ℹ️ Adding to existing speaker: {speaker_name}")
        finally:
            db.close()
        
        # Step 2: Process each file
        total_files = len(selected_files)
        
        for i, file_analysis in enumerate(selected_files):
            status_text.text(f"Processing file {i+1}/{total_files}: {file_analysis['filename']}")
            progress_bar.progress(0.1 + (i / total_files) * 0.7)
            
            # Prepare audio data for enrollment
            audio_data = file_analysis['audio_data']
            sample_rate = file_analysis['sample_rate']
            quality_assessment = file_analysis['quality_assessment']
            
            # Call speaker service enrollment endpoint
            try:
                enrollment_result = call_speaker_service_enrollment(
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    filename=file_analysis['filename']
                )
                
                if enrollment_result['success']:
                    # Save enrollment session to database
                    db = get_db_session()
                    try:
                        session = EnrollmentQueries.create_enrollment_session(
                            db=db,
                            speaker_id=speaker_id,
                            audio_file_path=file_analysis['temp_path'],
                            duration_seconds=quality_assessment['duration_seconds'],
                            speech_duration_seconds=quality_assessment['duration_seconds'],  # Use full duration since we removed speech detection
                            quality_score=quality_assessment['overall_quality'],
                            snr_db=quality_assessment['snr_db'],
                            enrollment_method='file_upload'
                        )
                        st.success(f"✅ Enrolled {file_analysis['filename']}")
                    finally:
                        db.close()
                else:
                    st.error(f"❌ Failed to enroll {file_analysis['filename']}: {enrollment_result.get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"❌ Error enrolling {file_analysis['filename']}: {str(e)}")
                continue
        
        # Step 3: Complete
        status_text.text("Enrollment complete!")
        progress_bar.progress(1.0)
        
        st.success(f"🎉 Successfully enrolled speaker '{speaker_name}' with {len(selected_files)} audio files!")
        
        # Cleanup temporary files
        for file_analysis in selected_files:
            try:
                os.unlink(file_analysis['temp_path'])
            except:
                pass
        
        # Clear session state
        st.session_state.enrollment_files = []
        
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Enrollment failed: {str(e)}")
        progress_bar.progress(0)
        status_text.text("Enrollment failed")

def call_speaker_service_enrollment(speaker_id: str, speaker_name: str, audio_data: np.ndarray, sample_rate: int, filename: str) -> dict:
    """
    Call the speaker service enrollment endpoint.
    
    Args:
        speaker_id: Unique speaker identifier
        speaker_name: Speaker display name
        audio_data: Audio signal array
        sample_rate: Audio sample rate
        filename: Original filename
    
    Returns:
        Dictionary with success status and result
    """
    try:
        # Create temporary audio file for upload
        temp_audio_path = create_temp_audio_file(audio_data, sample_rate)
        
        # Prepare multipart form data
        with open(temp_audio_path, 'rb') as audio_file:
            files = {'file': (filename, audio_file, 'audio/wav')}
            data = {
                'speaker_id': speaker_id,
                'speaker_name': speaker_name
            }
            
            # Call the enrollment endpoint
            response = requests.post(
                f'{SPEAKER_SERVICE_URL}/enroll/upload',
                files=files,
                data=data,  # Send as form data
                timeout=60
            )
        
        # Cleanup temp file
        os.unlink(temp_audio_path)
        
        if response.status_code == 200:
            return {'success': True, 'result': response.json()}
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}: {response.text}"}
    
    except requests.exceptions.ConnectionError:
        return {'success': False, 'error': f"Could not connect to speaker service at {SPEAKER_SERVICE_URL}. Is it running?"}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def enrollment_history():
    """Display enrollment history for current user."""
    if not st.session_state.user_id:
        return
    
    st.subheader("📚 Enrollment History")
    
    db = get_db_session()
    try:
        speakers = SpeakerQueries.get_speakers_for_user(db, st.session_state.user_id)
        
        if not speakers:
            st.info("No speakers enrolled yet.")
            return
        
        # Display speakers and their enrollment sessions
        for speaker in speakers:
            with st.expander(f"👤 {speaker.name} ({speaker.id})"):
                sessions = EnrollmentQueries.get_sessions_for_speaker(db, speaker.id)
                
                if not sessions:
                    st.info("No enrollment sessions found.")
                    continue
                
                # Session summary
                total_duration = sum(s.speech_duration_seconds or 0 for s in sessions)
                avg_quality = sum(s.quality_score or 0 for s in sessions) / len(sessions)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sessions", len(sessions))
                with col2:
                    st.metric("Total Duration", f"{total_duration:.1f}s")
                with col3:
                    st.metric("Avg Quality", f"{avg_quality:.1%}")
                
                # Session details
                for session in sessions:
                    st.write(f"**{session.created_at.strftime('%Y-%m-%d %H:%M')}** - "
                            f"{session.enrollment_method} - "
                            f"Quality: {session.quality_score:.1%}")
    
    except Exception as e:
        st.error(f"Error loading enrollment history: {str(e)}")
    finally:
        db.close()

def main():
    """Main enrollment page."""
    st.title("👤 Speaker Enrollment")
    st.markdown("Register new speakers in the system using guided recording or file upload.")
    
    # Check if user is logged in
    if "username" not in st.session_state or not st.session_state.username:
        st.warning("👈 Please select or create a user in the sidebar to continue.")
        return
    
    # Initialize session state
    init_session_state()
    
    # Mode selection
    enrollment_mode_selection()
    
    st.divider()
    
    # Show appropriate interface based on mode
    if st.session_state.enrollment_mode == "guided":
        guided_recording_interface()
    else:
        file_upload_interface()
    
    st.divider()
    
    # Enrollment history
    enrollment_history()
    
    # Help section
    with st.expander("ℹ️ Enrollment Tips & Best Practices"):
        st.markdown("""
        ### For Best Results:
        
        **Audio Quality:**
        - Use a quiet environment with minimal background noise
        - Maintain consistent distance from microphone
        - Speak naturally and continuously
        - Aim for at least 30 seconds of clear speech
        
        **Recording Content:**
        - Include varied speech patterns (questions, statements)
        - Use different emotional tones naturally
        - Avoid reading text - speak conversationally
        - Include pauses and natural speech rhythms
        
        **File Requirements:**
        - Supported formats: WAV, FLAC, MP3, M4A, OGG
        - Minimum 10 seconds, recommended 30+ seconds
        - Signal-to-noise ratio > 15 dB preferred
        - Speech content > 50% of total audio
        
        **Quality Indicators:**
        - 🟢 Excellent (>80%): Perfect for enrollment
        - 🔵 Good (60-80%): Suitable for enrollment  
        - 🟡 Acceptable (40-60%): May work, consider improving
        - 🔴 Poor (<40%): Please improve before enrollment
        """)

if __name__ == "__main__":
    main()