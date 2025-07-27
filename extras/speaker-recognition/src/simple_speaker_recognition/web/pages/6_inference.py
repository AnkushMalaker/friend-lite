"""Inference page for speech transcription and speaker identification."""

import streamlit as st
import json
import tempfile
import os
import asyncio
import requests
from pathlib import Path

from utils.youtube_transcriber import YouTubeTranscriber
from utils.audio_processing import load_audio, audio_to_bytes
from utils.cache_manager import get_cache_manager, get_file_hash

# Configuration
SPEAKER_SERVICE_URL = os.getenv("SPEAKER_SERVICE_URL", "http://speaker-service:8085")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

def init_session_state():
    """Initialize session state for inference page."""
    if "inference_audio_file" not in st.session_state:
        st.session_state.inference_audio_file = None
    if "inference_transcript" not in st.session_state:
        st.session_state.inference_transcript = None
    if "inference_results" not in st.session_state:
        st.session_state.inference_results = None
    if "temp_audio_path" not in st.session_state:
        st.session_state.temp_audio_path = None

def transcription_section():
    """Handle audio transcription with Deepgram."""
    st.subheader("🎤 Step 1: Audio Transcription")
    
    if not DEEPGRAM_API_KEY:
        st.error("❌ DEEPGRAM_API_KEY environment variable not set. Please configure your Deepgram API key.")
        st.info("Add DEEPGRAM_API_KEY to your .env file to use transcription features.")
        return False
    
    # Audio file upload
    uploaded_file = st.file_uploader(
        "Upload audio file for transcription",
        type=['wav', 'flac', 'mp3', 'm4a', 'ogg'],
        help="Supported formats: WAV, FLAC, MP3, M4A, OGG",
        key="transcription_upload"
    )
    
    if uploaded_file is not None:
        if st.session_state.inference_audio_file != uploaded_file.name:
            # New file uploaded, reset state
            st.session_state.inference_audio_file = uploaded_file.name
            st.session_state.inference_transcript = None
            st.session_state.inference_results = None
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.temp_audio_path = tmp_file.name
            
            st.success(f"✅ Audio file uploaded: {uploaded_file.name}")
        
        # Transcription controls
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox(
                "Transcription Language",
                options=["en", "multi", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"],
                help="Select 'multi' for mixed languages like Hinglish"
            )
        
        with col2:
            # Check if we have a cached transcription
            cache_status = ""
            if st.session_state.temp_audio_path:
                cache_manager = get_cache_manager()
                api_params = {
                    "model": "nova-3",
                    "diarize": True,
                    "multichannel": False,
                    "smart_format": True,
                    "punctuate": True,
                    "language": language,
                    "utterances": True,
                    "paragraphs": True,
                }
                cached_response = cache_manager.get_cached_deepgram_response(st.session_state.temp_audio_path, api_params)
                if cached_response:
                    cache_status = "💾 (Cached)"
                else:
                    cache_status = "💸 (API Call)"
            
            if st.button(f"🎯 Start Transcription {cache_status}", type="primary"):
                if st.session_state.temp_audio_path:
                    start_transcription(st.session_state.temp_audio_path, language)
                else:
                    st.error("No audio file available for transcription")
        
        # Show existing transcript if available
        if st.session_state.inference_transcript:
            st.success("✅ Transcription completed!")
            
            with st.expander("📋 View Transcript Details", expanded=False):
                transcript = st.session_state.inference_transcript
                st.write(f"**Total segments:** {len(transcript)}")
                
                for i, segment in enumerate(transcript[:5]):  # Show first 5 segments
                    st.write(f"**Segment {i+1}:** Speaker {segment['speaker']} ({segment['start']:.1f}s-{segment['end']:.1f}s)")
                    st.write(f"*{segment['text']}*")
                    st.divider()
                
                if len(transcript) > 5:
                    st.write(f"... and {len(transcript) - 5} more segments")
            
            return True
    
    return False

def start_transcription(audio_path: str, language: str):
    """Start the transcription process using YouTubeTranscriber."""
    try:
        with st.spinner("🎯 Transcribing audio with Deepgram..."):
            # Initialize transcriber
            transcriber = YouTubeTranscriber(DEEPGRAM_API_KEY, {"language": language})
            
            # Run transcription
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            response = loop.run_until_complete(transcriber.transcribe_audio(audio_path))
            
            if response:
                # Extract segments from response
                segments = extract_segments_from_response(response)
                st.session_state.inference_transcript = segments
                st.success(f"✅ Transcription completed! Found {len(segments)} speaker segments.")
            else:
                st.error("❌ Transcription failed. Please check your audio file and API key.")
                
    except Exception as e:
        st.error(f"❌ Transcription error: {str(e)}")

def extract_segments_from_response(response) -> list:
    """Extract structured segments from Deepgram response."""
    segments = []
    
    try:
        # Convert response to dict if needed
        if hasattr(response, 'to_dict'):
            response_dict = response.to_dict()
        elif hasattr(response, '__dict__'):
            response_dict = response.__dict__
        else:
            response_dict = dict(response)
        
        # Extract utterances (which include speaker diarization)
        if 'results' in response_dict and 'utterances' in response_dict['results']:
            for utterance in response_dict['results']['utterances']:
                segments.append({
                    'speaker': utterance.get('speaker', 0),
                    'start': utterance.get('start', 0),
                    'end': utterance.get('end', 0),
                    'text': utterance.get('transcript', '')
                })
        else:
            st.warning("⚠️ No diarized utterances found in transcription response")
            
    except Exception as e:
        st.error(f"Error extracting segments: {str(e)}")
    
    return segments

def speaker_identification_section():
    """Handle speaker identification for diarized segments."""
    st.subheader("👥 Step 2: Speaker Identification")
    
    if not st.session_state.inference_transcript:
        st.info("👆 Complete Step 1 (transcription) to proceed with speaker identification.")
        return
    
    # Show transcript summary
    transcript = st.session_state.inference_transcript
    unique_speakers = set(seg['speaker'] for seg in transcript)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Segments", len(transcript))
    with col2:
        st.metric("Unique Speakers", len(unique_speakers))
    with col3:
        total_duration = sum(seg['end'] - seg['start'] for seg in transcript)
        st.metric("Total Duration", f"{total_duration:.1f}s")
    
    # Manual transcript editing option
    with st.expander("✏️ Edit Transcript (Optional)", expanded=False):
        st.info("You can manually edit the transcript JSON before speaker identification.")
        
        transcript_json = st.text_area(
            "Transcript JSON",
            value=json.dumps(transcript, indent=2),
            height=300,
            help="Edit the transcript segments if needed"
        )
        
        if st.button("Update Transcript"):
            try:
                updated_transcript = json.loads(transcript_json)
                st.session_state.inference_transcript = updated_transcript
                st.success("✅ Transcript updated!")
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON: {str(e)}")
    
    # Speaker identification
    if st.button("🔍 Identify Speakers", type="primary"):
        if st.session_state.temp_audio_path and st.session_state.inference_transcript:
            identify_speakers()
        else:
            st.error("Missing audio file or transcript for speaker identification")

def identify_speakers():
    """Call the speaker service to identify speakers in segments."""
    try:
        with st.spinner("🔍 Identifying speakers..."):
            # Prepare the request
            with open(st.session_state.temp_audio_path, 'rb') as audio_file:
                files = {'audio_file': (
                    st.session_state.inference_audio_file, 
                    audio_file, 
                    'audio/wav'
                )}
                data = {
                    'segments': json.dumps(st.session_state.inference_transcript)
                }
                
                # Call the inference endpoint
                response = requests.post(
                    f'{SPEAKER_SERVICE_URL}/infer/diarized',
                    files=files,
                    data=data,
                    timeout=120  # 2 minute timeout for processing
                )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.inference_results = result
                st.success(f"✅ Speaker identification completed! Identified {result.get('identified_segments', 0)} of {result.get('total_segments', 0)} segments.")
            else:
                st.error(f"❌ Speaker identification failed: HTTP {response.status_code}: {response.text}")
                
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Could not connect to speaker service at {SPEAKER_SERVICE_URL}. Is it running?")
    except Exception as e:
        st.error(f"❌ Speaker identification error: {str(e)}")

def results_section():
    """Display and export the final results."""
    st.subheader("📊 Results")
    
    if not st.session_state.inference_results:
        st.info("👆 Complete Steps 1 & 2 to see results here.")
        return
    
    results = st.session_state.inference_results
    segments = results.get('segments', [])
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Segments", results.get('total_segments', 0))
    with col2:
        st.metric("Identified", results.get('identified_segments', 0))
    with col3:
        unknown_segments = len([s for s in segments if s.get('status') == 'unknown'])
        st.metric("Unknown", unknown_segments)
    with col4:
        error_segments = len([s for s in segments if s.get('status', '').endswith('error')])
        st.metric("Errors", error_segments)
    
    # Results table
    st.subheader("📋 Detailed Results")
    
    # Create a formatted table
    table_data = []
    for i, segment in enumerate(segments):
        status_emoji = {
            'identified': '✅',
            'unknown': '❓',
            'error': '❌',
            'audio_load_error': '🔊',
            'embedding_error': '🧠',
            'identification_error': '🔍',
            'skipped_too_short': '⏭️'
        }.get(segment.get('status'), '❓')
        
        confidence = segment.get('confidence', 0)
        confidence_str = f"{confidence:.2f}" if confidence > 0 else "-"
        
        table_data.append({
            "Segment": i + 1,
            "Time": f"{segment['start']:.1f}s - {segment['end']:.1f}s",
            "Speaker": segment.get('identified_speaker', 'Unknown'),
            "Confidence": confidence_str,
            "Status": f"{status_emoji} {segment.get('status', 'unknown')}",
            "Text": segment.get('text', '')[:100] + ('...' if len(segment.get('text', '')) > 100 else '')
        })
    
    st.dataframe(table_data, use_container_width=True)
    
    # Export options
    st.subheader("📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download JSON", use_container_width=True):
            json_data = json.dumps(results, indent=2)
            st.download_button(
                label="💾 Save JSON File",
                data=json_data,
                file_name=f"inference_results_{st.session_state.inference_audio_file.split('.')[0]}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📄 Download Transcript", use_container_width=True):
            transcript_text = generate_transcript_text(segments)
            st.download_button(
                label="💾 Save Transcript",
                data=transcript_text,
                file_name=f"transcript_{st.session_state.inference_audio_file.split('.')[0]}.txt",
                mime="text/plain"
            )
    
    with col3:
        if st.button("🎧 Audio Segments", use_container_width=True):
            st.info("Audio segment export feature coming soon!")

def generate_transcript_text(segments: list) -> str:
    """Generate a formatted transcript text."""
    transcript_lines = []
    transcript_lines.append("SPEAKER-IDENTIFIED TRANSCRIPT")
    transcript_lines.append("=" * 50)
    transcript_lines.append("")
    
    for i, segment in enumerate(segments):
        start_time = segment['start']
        end_time = segment['end']
        speaker = segment.get('identified_speaker', 'Unknown')
        text = segment.get('text', '')
        confidence = segment.get('confidence', 0)
        
        # Format time as MM:SS
        start_min, start_sec = divmod(int(start_time), 60)
        end_min, end_sec = divmod(int(end_time), 60)
        
        confidence_str = f" (confidence: {confidence:.2f})" if confidence > 0 else ""
        
        transcript_lines.append(f"{speaker}{confidence_str} [{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]: {text}")
        transcript_lines.append("")
    
    return "\n".join(transcript_lines)

def cleanup_temp_files():
    """Clean up temporary files."""
    if st.session_state.temp_audio_path and os.path.exists(st.session_state.temp_audio_path):
        try:
            os.unlink(st.session_state.temp_audio_path)
            st.session_state.temp_audio_path = None
        except:
            pass

def main():
    """Main inference page."""
    st.title("🎯 Speech Inference")
    st.markdown("Transcribe audio with speaker diarization and identify speakers from your enrolled database.")
    
    # Check if user is logged in
    if "username" not in st.session_state or not st.session_state.username:
        st.warning("👈 Please select or create a user in the sidebar to continue.")
        return
    
    # Initialize session state
    init_session_state()
    
    # Main workflow
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Step 1: Transcription
        has_transcript = transcription_section()
        
        st.divider()
        
        # Step 2: Speaker Identification
        speaker_identification_section()
        
        st.divider()
        
        # Step 3: Results
        results_section()
    
    with col2:
        st.subheader("ℹ️ Process Overview")
        
        step1_status = "✅" if st.session_state.inference_transcript else "⏳"
        step2_status = "✅" if st.session_state.inference_results else "⏳"
        
        st.markdown(f"""
        **Workflow Steps:**
        
        {step1_status} **Step 1: Transcription**
        - Upload audio file
        - Generate diarized transcript
        - Identify speaker segments
        
        {step2_status} **Step 2: Speaker ID**
        - Match segments to enrolled speakers
        - Generate confidence scores
        - Handle unknown speakers
        
        **📊 Export Options:**
        - JSON format with full data
        - Formatted transcript text
        - Audio segments (coming soon)
        """)
        
        # Cache management section
        st.divider()
        st.subheader("💾 Cache Management")
        
        cache_manager = get_cache_manager()
        cache_stats = cache_manager.get_cache_stats()
        
        st.write("**Cache Statistics:**")
        deepgram_count = cache_stats['deepgram']['count']
        transcript_count = cache_stats['transcripts']['count']
        
        st.write(f"• Deepgram responses: {deepgram_count}")
        st.write(f"• Processed transcripts: {transcript_count}")
        
        if deepgram_count > 0 or transcript_count > 0:
            total_size = cache_stats['deepgram']['total_size'] + cache_stats['transcripts']['total_size']
            st.write(f"• Total cache size: {total_size / 1024 / 1024:.1f} MB")
        
        if st.button("🗑️ Clear Cache", use_container_width=True):
            cleared = cache_manager.clear_cache()
            total_cleared = cleared['deepgram'] + cleared['transcripts']
            st.success(f"✅ Cleared {total_cleared} cache files")
            st.rerun()
        
        if st.session_state.inference_transcript or st.session_state.inference_results:
            st.divider()
            if st.button("🗑️ Clear Session Data", use_container_width=True):
                cleanup_temp_files()
                st.session_state.inference_audio_file = None
                st.session_state.inference_transcript = None
                st.session_state.inference_results = None
                st.success("✅ Session data cleared!")
                st.rerun()
    
    # Help section
    with st.expander("❓ Help & Tips"):
        st.markdown("""
        ### How to Use Speech Inference
        
        1. **Upload Audio**: Support for WAV, FLAC, MP3, M4A, OGG formats
        2. **Transcription**: Uses Deepgram Nova-3 for high-quality speech-to-text with speaker diarization
        3. **Speaker ID**: Matches transcript segments against your enrolled speaker database
        
        ### Tips for Better Results
        - **Audio Quality**: Clear audio with minimal background noise works best
        - **Speaker Enrollment**: Ensure target speakers are enrolled with quality samples
        - **Segment Length**: Very short segments (<0.5s) are automatically skipped
        - **Multiple Speakers**: Works best with 2-10 distinct speakers
        
        ### Caching & Performance
        - **Smart Caching**: Transcriptions are automatically cached to avoid repeat API calls
        - **Cache Indicators**: 💾 means cached result, 💸 means new API call required
        - **Cost Savings**: Reprocessing the same audio uses cache instead of calling Deepgram
        - **Cache Management**: Use sidebar controls to view and clear cache when needed
        
        ### Troubleshooting
        - **Transcription Fails**: Check Deepgram API key and internet connection
        - **No Speaker Matches**: Ensure speakers are enrolled in the system
        - **Poor Identification**: Try enrolling more/better samples for target speakers
        - **Cache Issues**: Clear cache if you get unexpected results from cached data
        """)

if __name__ == "__main__":
    main()