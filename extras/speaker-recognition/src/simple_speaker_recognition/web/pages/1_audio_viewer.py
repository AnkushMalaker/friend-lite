"""Audio Viewer page for uploading and visualizing audio files."""

import streamlit as st
import numpy as np
import tempfile
import os

from utils.audio_processing import (
    load_audio, get_audio_info, load_audio_segment, 
    audio_to_bytes, detect_speech_segments, calculate_snr
)
from utils.visualization import (
    create_waveform_plot, create_spectrogram_plot, create_segment_timeline,
    add_selection_overlay
)

def init_session_state():
    """Initialize session state for audio viewer."""
    if "uploaded_audio" not in st.session_state:
        st.session_state.uploaded_audio = None
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None
    if "sample_rate" not in st.session_state:
        st.session_state.sample_rate = None
    if "audio_info" not in st.session_state:
        st.session_state.audio_info = None
    if "selected_segment" not in st.session_state:
        st.session_state.selected_segment = None
    if "speech_segments" not in st.session_state:
        st.session_state.speech_segments = None

def upload_audio_file():
    """Handle audio file upload."""
    st.subheader("📁 Upload Audio File")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'flac', 'mp3', 'm4a', 'ogg'],
        help="Supported formats: WAV, FLAC, MP3, M4A, OGG"
    )
    
    if uploaded_file is not None:
        if st.session_state.uploaded_audio != uploaded_file.name:
            # New file uploaded
            st.session_state.uploaded_audio = uploaded_file.name
            
            with st.spinner("Loading audio file..."):
                try:
                    # Save uploaded file to temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    # Load audio data
                    audio_data, sample_rate = load_audio(temp_path)
                    audio_info = get_audio_info(temp_path)
                    
                    # Store in session state
                    st.session_state.audio_data = audio_data
                    st.session_state.sample_rate = sample_rate
                    st.session_state.audio_info = audio_info
                    st.session_state.temp_file_path = temp_path
                    
                    # Detect speech segments
                    speech_segments = detect_speech_segments(audio_data, sample_rate)
                    st.session_state.speech_segments = speech_segments
                    
                    st.success(f"✅ Audio loaded successfully: {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error loading audio file: {str(e)}")
                    return False
        
        return True
    
    return False

def display_audio_info():
    """Display basic audio file information."""
    if st.session_state.audio_info is None:
        return
    
    st.subheader("📊 Audio Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{st.session_state.audio_info['duration_seconds']:.2f}s")
    
    with col2:
        st.metric("Sample Rate", f"{st.session_state.audio_info['sample_rate']:,} Hz")
    
    with col3:
        st.metric("Channels", st.session_state.audio_info['channels'])
    
    with col4:
        st.metric("Format", st.session_state.audio_info.get('format', 'Unknown'))
    
    # Additional metrics
    if st.session_state.audio_data is not None:
        col5, col6, col7 = st.columns(3)
        
        with col5:
            snr = calculate_snr(st.session_state.audio_data)
            st.metric("SNR", f"{snr:.1f} dB")
        
        with col6:
            # Show bit depth or format info instead of speech ratio
            format_info = st.session_state.audio_info.get('subtype', st.session_state.audio_info.get('format', 'PCM'))
            st.metric("Audio Format", format_info)
        
        with col7:
            if st.session_state.speech_segments:
                total_speech = sum(end - start for start, end in st.session_state.speech_segments)
                st.metric("Speech Duration", f"{total_speech:.1f}s")

def display_waveform():
    """Display interactive waveform."""
    if st.session_state.audio_data is None:
        return
    
    st.subheader("🌊 Waveform")
    
    # Create waveform plot
    fig = create_waveform_plot(
        st.session_state.audio_data, 
        st.session_state.sample_rate,
        title=f"Waveform - {st.session_state.uploaded_audio}",
        height=300
    )
    
    # Add selection overlay if segment is selected
    if st.session_state.selected_segment:
        start_time, end_time = st.session_state.selected_segment
        fig = add_selection_overlay(fig, start_time, end_time)
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Manual segment selection using input fields
    st.subheader("🎯 Select Audio Segment")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_time = st.number_input(
            "Start time (seconds)",
            min_value=0.0,
            max_value=st.session_state.audio_info['duration_seconds'],
            value=0.0,
            step=0.1,
            key="segment_start"
        )
    
    with col2:
        end_time = st.number_input(
            "End time (seconds)",
            min_value=start_time,
            max_value=st.session_state.audio_info['duration_seconds'],
            value=min(10.0, st.session_state.audio_info['duration_seconds']),
            step=0.1,
            key="segment_end"
        )
    
    with col3:
        if st.button("🎯 Select Segment", type="primary"):
            if start_time < end_time:
                st.session_state.selected_segment = (start_time, end_time)
                st.success(f"Selected segment: {start_time:.1f}s - {end_time:.1f}s")
                st.rerun()
            else:
                st.error("End time must be greater than start time")

def display_spectrogram():
    """Display spectrogram if requested."""
    if st.session_state.audio_data is None:
        return
    
    if st.checkbox("Show Spectrogram", help="Display frequency content over time"):
        st.subheader("🎵 Spectrogram")
        
        with st.spinner("Generating spectrogram..."):
            fig = create_spectrogram_plot(
                st.session_state.audio_data,
                st.session_state.sample_rate,
                title=f"Spectrogram - {st.session_state.uploaded_audio}",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_speech_segments():
    """Display detected speech segments."""
    if st.session_state.speech_segments is None:
        return
    
    st.subheader("🗣️ Detected Speech Segments")
    
    if not st.session_state.speech_segments:
        st.info("No speech segments detected in this audio.")
        return
    
    # Create timeline visualization
    segment_labels = [f"Speech {i+1}" for i in range(len(st.session_state.speech_segments))]
    fig = create_segment_timeline(
        st.session_state.speech_segments,
        labels=segment_labels,
        total_duration=st.session_state.audio_info['duration_seconds'],
        height=150
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display segment details
    with st.expander(f"📋 Segment Details ({len(st.session_state.speech_segments)} segments)"):
        for i, (start, end) in enumerate(st.session_state.speech_segments):
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            
            with col1:
                st.write(f"**{i+1}**")
            
            with col2:
                st.write(f"{start:.2f}s - {end:.2f}s")
            
            with col3:
                st.write(f"Duration: {end-start:.2f}s")
            
            with col4:
                if st.button("Select", key=f"select_segment_{i}"):
                    st.session_state.selected_segment = (start, end)
                    st.rerun()

def segment_controls():
    """Controls for segment selection and playback."""
    if st.session_state.audio_data is None:
        return
    
    st.subheader("✂️ Segment Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Manual segment selection
        st.write("**Manual Selection:**")
        duration = st.session_state.audio_info['duration_seconds']
        
        start_time = st.number_input(
            "Start time (seconds)",
            min_value=0.0,
            max_value=duration,
            value=0.0,
            step=0.1,
            format="%.2f"
        )
        
        end_time = st.number_input(
            "End time (seconds)",
            min_value=start_time,
            max_value=duration,
            value=min(start_time + 5.0, duration),
            step=0.1,
            format="%.2f"
        )
        
        if st.button("Apply Selection"):
            st.session_state.selected_segment = (start_time, end_time)
            st.rerun()
    
    with col2:
        # Selection info and controls
        st.write("**Current Selection:**")
        
        if st.session_state.selected_segment:
            start, end = st.session_state.selected_segment
            st.info(f"📍 {start:.2f}s - {end:.2f}s (Duration: {end-start:.2f}s)")
            
            # Audio playback
            try:
                segment_audio, _ = load_audio_segment(
                    st.session_state.temp_file_path,
                    start, end, 
                    st.session_state.sample_rate
                )
                
                audio_bytes = audio_to_bytes(segment_audio, st.session_state.sample_rate)
                st.audio(audio_bytes, format='audio/wav')
                
                # Download button
                st.download_button(
                    label="📥 Download Segment",
                    data=audio_bytes,
                    file_name=f"segment_{start:.1f}s-{end:.1f}s.wav",
                    mime="audio/wav"
                )
                
            except Exception as e:
                st.error(f"Error loading segment: {str(e)}")
        else:
            st.info("No segment selected. Use the waveform or controls above to select a segment.")
        
        # Clear selection
        if st.button("Clear Selection"):
            st.session_state.selected_segment = None
            st.rerun()

def export_options():
    """Export options for the entire audio or segments."""
    if st.session_state.audio_data is None:
        return
    
    st.subheader("📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export full audio
        if st.button("📁 Export Full Audio", use_container_width=True):
            try:
                audio_bytes = audio_to_bytes(st.session_state.audio_data, st.session_state.sample_rate)
                st.download_button(
                    label="📥 Download Full Audio",
                    data=audio_bytes,
                    file_name=f"full_audio_{st.session_state.uploaded_audio.split('.')[0]}.wav",
                    mime="audio/wav"
                )
            except Exception as e:
                st.error(f"Export error: {str(e)}")
    
    with col2:
        # Export all speech segments
        if st.button("🗣️ Export Speech Segments", use_container_width=True):
            if st.session_state.speech_segments:
                try:
                    # Create ZIP file with all segments
                    import zipfile
                    import io
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        for i, (start, end) in enumerate(st.session_state.speech_segments):
                            segment_audio, _ = load_audio_segment(
                                st.session_state.temp_file_path,
                                start, end,
                                st.session_state.sample_rate
                            )
                            segment_bytes = audio_to_bytes(segment_audio, st.session_state.sample_rate)
                            zip_file.writestr(f"speech_segment_{i+1:03d}_{start:.1f}s-{end:.1f}s.wav", segment_bytes)
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="📥 Download Speech Segments ZIP",
                        data=zip_buffer.getvalue(),
                        file_name=f"speech_segments_{st.session_state.uploaded_audio.split('.')[0]}.zip",
                        mime="application/zip"
                    )
                except Exception as e:
                    st.error(f"Export error: {str(e)}")
            else:
                st.info("No speech segments detected to export.")
    
    with col3:
        # Export metadata
        if st.button("📋 Export Metadata", use_container_width=True):
            try:
                import json
                
                metadata = {
                    "filename": st.session_state.uploaded_audio,
                    "audio_info": st.session_state.audio_info,
                    "speech_segments": st.session_state.speech_segments,
                    "quality_metrics": {
                        "snr_db": calculate_snr(st.session_state.audio_data),
                        "duration_seconds": st.session_state.audio_info['duration_seconds']
                    }
                }
                
                metadata_json = json.dumps(metadata, indent=2)
                st.download_button(
                    label="📥 Download Metadata JSON",
                    data=metadata_json,
                    file_name=f"metadata_{st.session_state.uploaded_audio.split('.')[0]}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Export error: {str(e)}")

def main():
    """Main audio viewer page."""
    st.title("🎵 Audio Viewer")
    st.markdown("Upload and explore audio files with interactive visualization and segment selection.")
    
    # Check if user is logged in
    if "username" not in st.session_state or not st.session_state.username:
        st.warning("👈 Please select or create a user in the sidebar to continue.")
        return
    
    # Initialize session state
    init_session_state()
    
    # Upload section
    if upload_audio_file():
        # Display audio information
        display_audio_info()
        
        st.divider()
        
        # Waveform visualization
        display_waveform()
        
        # Spectrogram (optional)
        display_spectrogram()
        
        st.divider()
        
        # Speech segments
        display_speech_segments()
        
        st.divider()
        
        # Segment controls
        segment_controls()
        
        st.divider()
        
        # Export options
        export_options()
        
        # Cleanup temp file when done
        if hasattr(st.session_state, 'temp_file_path') and os.path.exists(st.session_state.temp_file_path):
            if st.button("🗑️ Clear Audio Data"):
                try:
                    os.unlink(st.session_state.temp_file_path)
                except:
                    pass
                
                # Clear session state
                for key in ['uploaded_audio', 'audio_data', 'sample_rate', 'audio_info', 
                           'selected_segment', 'speech_segments', 'temp_file_path']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("Audio data cleared!")
                st.rerun()
    
    else:
        # Show help when no file is uploaded
        st.info("👆 Upload an audio file to start exploring!")
        
        with st.expander("ℹ️ How to use Audio Viewer"):
            st.markdown("""
            ### Features:
            - **Upload Audio**: Support for WAV, FLAC, MP3, M4A, OGG formats
            - **Waveform Visualization**: Interactive plot with zoom and pan
            - **Segment Selection**: Click and drag on waveform to select regions
            - **Speech Detection**: Automatic detection of speech vs silence
            - **Quality Metrics**: SNR, speech ratio, and duration analysis
            - **Export Options**: Download segments, full audio, or metadata
            
            ### Tips:
            - Use the waveform plot to visually select interesting segments
            - Check the spectrogram to see frequency content
            - Speech segments are automatically detected and can be exported
            - All visualizations are interactive - zoom, pan, and hover for details
            """)

if __name__ == "__main__":
    main()