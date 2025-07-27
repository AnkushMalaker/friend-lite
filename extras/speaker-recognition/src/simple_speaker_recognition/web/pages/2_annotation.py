"""Annotation interface for labeling speaker segments."""

import streamlit as st
import numpy as np
import tempfile
import os
import json
import hashlib
from pathlib import Path

from utils.audio_processing import (
    load_audio, get_audio_info, load_audio_segment, 
    audio_to_bytes
)
from utils.visualization import (
    create_waveform_plot, create_segment_timeline,
    add_selection_overlay
)
from utils.deepgram_parser import DeepgramParser
from database import get_db_session
from database.queries import SpeakerQueries, AnnotationQueries, UserQueries
from database.models import Annotation
from utils.filter_components import combined_filters_component, apply_annotation_filters

def init_session_state():
    """Initialize session state for annotation."""
    if "annotation_audio" not in st.session_state:
        st.session_state.annotation_audio = None
    if "annotation_audio_data" not in st.session_state:
        st.session_state.annotation_audio_data = None
    if "annotation_sample_rate" not in st.session_state:
        st.session_state.annotation_sample_rate = None
    if "annotation_info" not in st.session_state:
        st.session_state.annotation_info = None
    if "annotation_segments" not in st.session_state:
        st.session_state.annotation_segments = []
    if "current_segment" not in st.session_state:
        st.session_state.current_segment = None
    if "annotation_temp_file" not in st.session_state:
        st.session_state.annotation_temp_file = None
    if "speaker_mappings" not in st.session_state:
        st.session_state.speaker_mappings = {}  # Maps deepgram speaker labels to actual speakers
    if "deepgram_data" not in st.session_state:
        st.session_state.deepgram_data = None
    if "annotation_file_loaded" not in st.session_state:
        st.session_state.annotation_file_loaded = False  # Track if file is loaded
    if "expander_states" not in st.session_state:
        st.session_state.expander_states = {}  # Track expander open/closed states

def clear_audio_cache():
    """Clear all cached audio segments when file changes."""
    # Clear any cached audio segments
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith("audio_cache_")]
    for key in keys_to_remove:
        del st.session_state[key]
    if keys_to_remove:
        st.info(f"🧹 Cleared {len(keys_to_remove)} audio cache entries")

def get_file_hash(file_content: bytes) -> str:
    """Calculate MD5 hash of file content for consistent identification."""
    return hashlib.md5(file_content).hexdigest()

def upload_annotation_file():
    """Handle audio file upload for annotation."""
    st.subheader("📁 Upload Files for Annotation")
    
    # Tab interface for different upload methods
    tab1, tab2 = st.tabs(["Audio File", "Import Deepgram JSON"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an audio file to annotate",
            type=['wav', 'flac', 'mp3', 'm4a', 'ogg'],
            help="Upload the audio file you want to annotate with speaker labels",
            key="annotation_uploader"
        )
        
        if uploaded_file is not None:
            # Calculate file hash for consistent identification
            file_content = uploaded_file.getvalue()
            file_hash = get_file_hash(file_content)
            
            # Check if this is a different file
            if not hasattr(st.session_state, 'annotation_file_hash') or st.session_state.annotation_file_hash != file_hash:
                # New file uploaded - clear any cached audio
                clear_audio_cache()
                st.session_state.annotation_audio = uploaded_file.name
                st.session_state.annotation_file_hash = file_hash
                
                with st.spinner("Loading audio file for annotation..."):
                    try:
                        # Save uploaded file to temporary location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(file_content)
                            temp_path = tmp_file.name
                        
                        # Load audio data
                        audio_data, sample_rate = load_audio(temp_path)
                        audio_info = get_audio_info(temp_path)
                        
                        # Store in session state
                        st.session_state.annotation_audio_data = audio_data
                        st.session_state.annotation_sample_rate = sample_rate
                        st.session_state.annotation_info = audio_info
                        st.session_state.annotation_temp_file = temp_path
                        st.session_state.annotation_file_loaded = True
                        
                        # Load existing annotations by file hash
                        load_existing_annotations(file_hash, uploaded_file.name)
                        
                        st.success(f"✅ Audio loaded for annotation: {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"❌ Error loading audio file: {str(e)}")
                        return False
            
            return True
    
    with tab2:
        # Deepgram JSON import interface
        col1, col2 = st.columns(2)
        
        with col1:
            audio_file = st.file_uploader(
                "Audio file (matching the Deepgram transcript)",
                type=['wav', 'flac', 'mp3', 'm4a', 'ogg'],
                help="Upload the audio file that was transcribed",
                key="deepgram_audio"
            )
        
        with col2:
            json_file = st.file_uploader(
                "Deepgram JSON transcript",
                type=['json'],
                help="Upload the Deepgram JSON output file",
                key="deepgram_json"
            )
        
        if audio_file and json_file:
            if st.button("Import Deepgram Transcript", type="primary"):
                # Clear any cached audio from previous files
                clear_audio_cache()
                with st.spinner("Processing Deepgram transcript..."):
                    try:
                        # Calculate file hash
                        audio_content = audio_file.getvalue()
                        file_hash = get_file_hash(audio_content)
                        
                        # Save audio file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_audio:
                            tmp_audio.write(audio_content)
                            audio_path = tmp_audio.name
                        
                        # Save JSON file  
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w') as tmp_json:
                            json_content = json_file.read().decode('utf-8')
                            tmp_json.write(json_content)
                            json_path = tmp_json.name
                        
                        # Parse Deepgram JSON
                        parser = DeepgramParser()
                        parsed_data = parser.parse_deepgram_json(json_path)
                        
                        # Store Deepgram data
                        st.session_state.deepgram_data = parsed_data
                        
                        # Convert to annotations
                        annotations = parser.convert_to_annotation_format(
                            parsed_data, 
                            audio_path,
                            st.session_state.user_id
                        )
                        
                        # Load audio data
                        audio_data, sample_rate = load_audio(audio_path)
                        audio_info = get_audio_info(audio_path)
                        
                        # Store in session state
                        st.session_state.annotation_audio = audio_file.name
                        st.session_state.annotation_file_hash = file_hash
                        st.session_state.annotation_audio_data = audio_data
                        st.session_state.annotation_sample_rate = sample_rate
                        st.session_state.annotation_info = audio_info
                        st.session_state.annotation_temp_file = audio_path
                        st.session_state.annotation_segments = annotations
                        st.session_state.annotation_file_loaded = True
                        
                        # Initialize speaker mappings
                        unique_speakers = parsed_data['unique_speakers']
                        for speaker in unique_speakers:
                            if speaker not in st.session_state.speaker_mappings:
                                st.session_state.speaker_mappings[speaker] = None
                        
                        # Display statistics
                        st.success(f"✅ Imported {len(annotations)} segments from Deepgram transcript")
                        
                        stats = parser.get_speaker_statistics(parsed_data)
                        st.subheader("📊 Speaker Statistics")
                        
                        for speaker, data in stats.items():
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(f"{speaker}", f"{data['segment_count']} segments")
                            with col2:
                                st.metric("Duration", f"{data['total_duration']:.1f}s")
                            with col3:
                                st.metric("Words/min", f"{data['words_per_minute']:.1f}")
                        
                        # Clean up JSON file
                        os.unlink(json_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error importing Deepgram transcript: {str(e)}")
                        return False
                
                return True
    
    return False

def load_existing_annotations(file_hash: str, file_name: str):
    """Load existing annotations for the audio file by hash."""
    if not st.session_state.user_id:
        return
    
    db = get_db_session()
    try:
        # First try to load by file hash
        annotations = db.query(Annotation).filter(
            Annotation.audio_file_hash == file_hash,
            Annotation.user_id == st.session_state.user_id
        ).order_by(Annotation.start_time).all()
        
        # If no annotations found by hash, try by filename as fallback
        if not annotations:
            annotations = db.query(Annotation).filter(
                Annotation.audio_file_name == file_name,
                Annotation.user_id == st.session_state.user_id
            ).order_by(Annotation.start_time).all()
        
        # Convert to session state format
        segments = []
        for ann in annotations:
            segments.append({
                'start_time': ann.start_time,
                'end_time': ann.end_time,
                'speaker_id': ann.speaker_id,
                'speaker_label': ann.speaker_label,
                'deepgram_speaker_label': ann.deepgram_speaker_label,
                'label': ann.label,
                'confidence': ann.confidence,
                'transcription': ann.transcription,
                'notes': ann.notes or '',
                'annotation_id': ann.id
            })
        
        st.session_state.annotation_segments = segments
        
        if segments:
            st.info(f"📋 Loaded {len(segments)} existing annotations for {file_name}")
    
    except Exception as e:
        st.error(f"Error loading annotations: {str(e)}")
    finally:
        db.close()

def get_speaker_options():
    """Get available speaker options for annotation."""
    if not st.session_state.user_id:
        return ["No user selected"]
    
    db = get_db_session()
    try:
        speakers = SpeakerQueries.get_speakers_for_user(db, st.session_state.user_id)
        
        options = ["Select speaker..."]
        
        # Add enrolled speakers
        for speaker in speakers:
            options.append(f"👤 {speaker.name} ({speaker.id})")
        
        # Add single unknown speaker option
        options.append("❓ Unknown Speaker")
        
        return options
    
    except Exception as e:
        st.error(f"Error loading speakers: {str(e)}")
        return ["Error loading speakers"]
    finally:
        db.close()

def speaker_mapping_interface():
    """Interface for mapping Deepgram speakers to enrolled speakers."""
    if not st.session_state.speaker_mappings:
        return
    
    st.subheader("🔗 Speaker Mapping")
    st.info("Map Deepgram speaker labels to enrolled speakers. Changes will apply to all segments.")
    
    # Quick speaker creation section - always show if we have Deepgram mappings
    with st.expander("➕ Create New Speaker", expanded=True):
        st.markdown("**Create a new speaker to use in mapping:**")
        
        # Use a form to prevent automatic submission on Enter/Tab
        with st.form("create_speaker_form", clear_on_submit=False):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                new_speaker_name = st.text_input(
                    "Speaker name:",
                    placeholder="e.g., John Smith",
                    key="form_speaker_name"
                )
            
            with col2:
                new_speaker_notes = st.text_input(
                    "Notes (optional):",
                    placeholder="e.g., CEO, main presenter",
                    key="form_speaker_notes"
                )
            
            with col3:
                create_speaker_clicked = st.form_submit_button("Create Speaker", type="primary")
        
        # Handle form submission outside the form
        if create_speaker_clicked:
            if new_speaker_name and new_speaker_name.strip():
                try:
                    # Generate speaker ID based on name
                    import re
                    import time
                    
                    name_clean = re.sub(r'[^a-zA-Z0-9]', '', new_speaker_name.strip().lower())
                    timestamp = str(int(time.time()))[-4:]  # Last 4 digits of timestamp
                    speaker_id = f"{name_clean}_{timestamp}"
                    
                    # Add to database directly
                    db = get_db_session()
                    try:
                        new_speaker = SpeakerQueries.create_speaker(
                            db, 
                            speaker_id=speaker_id,
                            name=new_speaker_name.strip(),
                            user_id=st.session_state.user_id,
                            notes=new_speaker_notes.strip() if new_speaker_notes else None
                        )
                        st.success(f"✅ Created speaker: {new_speaker.name} (ID: {new_speaker.id})")
                        st.info("💡 The new speaker will appear in the dropdowns below. Click 'Refresh Speakers' to see it.")
                    except Exception as e:
                        st.error(f"Error creating speaker: {str(e)}")
                    finally:
                        db.close()
                except Exception as e:
                    st.error(f"Error creating speaker: {str(e)}")
            else:
                st.error("Please enter a speaker name")
    
    # Get enrolled speakers for dropdowns
    enrolled_speakers = []
    db = get_db_session()
    try:
        speakers = SpeakerQueries.get_speakers_for_user(db, st.session_state.user_id)
        for speaker in speakers:
            enrolled_speakers.append((speaker.id, f"{speaker.name} ({speaker.id})"))
    finally:
        db.close()
    
    # Refresh speakers button and status
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        if enrolled_speakers:
            st.info(f"Found {len(enrolled_speakers)} enrolled speakers")
        else:
            st.warning("⚠️ No enrolled speakers found")
    with col2:
        if st.button("🔄 Refresh Speakers", help="Reload speaker list"):
            st.rerun()
    with col3:
        if not enrolled_speakers:
            st.info("Create speakers above to enable mapping")
    
    # Show message if no enrolled speakers
    if not enrolled_speakers:
        return
    
    # Create mapping interface
    mapping_changed = False
    new_mappings = {}
    
    for deepgram_speaker, current_mapping in st.session_state.speaker_mappings.items():
        # Count segments for this speaker
        segment_count = sum(1 for seg in st.session_state.annotation_segments 
                          if seg.get('deepgram_speaker_label') == deepgram_speaker)
        
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.write(f"**{deepgram_speaker}** ({segment_count} segments)")
        
        with col2:
            # Add Unknown Speaker as a mapping option
            options = ["Unmapped", "❓ Unknown Speaker"] + [name for _, name in enrolled_speakers]
            current_index = 0
            
            if current_mapping:
                if current_mapping == "unknown":
                    current_index = 1  # Unknown Speaker option
                else:
                    # Find current mapping in enrolled speakers
                    for i, (speaker_id, _) in enumerate(enrolled_speakers, 2):  # Start at 2 because of Unmapped and Unknown
                        if speaker_id == current_mapping:
                            current_index = i
                            break
            
            selected = st.selectbox(
                f"Map to:",
                options=options,
                index=current_index,
                key=f"map_{deepgram_speaker}",
                label_visibility="collapsed"
            )
            
            if selected == "Unmapped":
                new_mappings[deepgram_speaker] = None
            elif selected == "❓ Unknown Speaker":
                new_mappings[deepgram_speaker] = "unknown"
            else:
                # Extract speaker ID from selection
                for speaker_id, name in enrolled_speakers:
                    if name == selected:
                        new_mappings[deepgram_speaker] = speaker_id
                        break
        
        with col3:
            # Check if mapping changed
            if new_mappings.get(deepgram_speaker) != current_mapping:
                mapping_changed = True
                st.write("✏️ Changed")
    
    # Apply mappings button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("Apply All Mappings", type="primary", disabled=not mapping_changed):
            # Update speaker mappings
            st.session_state.speaker_mappings = new_mappings
            
            # Apply mappings to all segments
            for segment in st.session_state.annotation_segments:
                deepgram_label = segment.get('deepgram_speaker_label')
                if deepgram_label and deepgram_label in new_mappings:
                    mapped_speaker = new_mappings[deepgram_label]
                    if mapped_speaker == "unknown":
                        # Map to unknown speaker
                        segment['speaker_id'] = None
                        segment['speaker_label'] = "unknown"
                    elif mapped_speaker:
                        # Map to enrolled speaker
                        segment['speaker_id'] = mapped_speaker
                        segment['speaker_label'] = None  # Clear unknown label
                    else:
                        # Unmapped - keep original deepgram label
                        segment['speaker_id'] = None
                        segment['speaker_label'] = deepgram_label
            
            st.success("✅ Applied speaker mappings to all segments")
            st.rerun()
    
    with col2:
        if st.button("Clear All Mappings"):
            st.session_state.speaker_mappings = {k: None for k in st.session_state.speaker_mappings}
            
            # Clear all speaker assignments
            for segment in st.session_state.annotation_segments:
                segment['speaker_id'] = None
                segment['speaker_label'] = segment.get('deepgram_speaker_label', 'unknown')
            
            st.info("🔄 Cleared all speaker mappings")
            st.rerun()
    
    st.divider()

def segment_annotation_interface():
    """Interface for annotating individual segments."""
    if st.session_state.annotation_audio_data is None:
        return
    
    st.subheader("✂️ Segment Annotation")
    
    # Manual segment creation
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Create New Segment:**")
        duration = st.session_state.annotation_info['duration_seconds']
        
        start_time = st.number_input(
            "Start time (seconds)",
            min_value=0.0,
            max_value=duration,
            value=0.0,
            step=0.1,
            format="%.2f",
            key="ann_start"
        )
        
        end_time = st.number_input(
            "End time (seconds)",
            min_value=start_time + 0.1,
            max_value=duration,
            value=min(start_time + 2.0, duration),
            step=0.1,
            format="%.2f",
            key="ann_end"
        )
    
    with col2:
        st.write("**Speaker Assignment:**")
        
        speaker_options = get_speaker_options()
        selected_speaker = st.selectbox(
            "Assign to speaker:",
            speaker_options,
            key="speaker_select"
        )
        
        # Label assignment
        label = st.selectbox(
            "Quality label:",
            ["CORRECT", "INCORRECT", "UNCERTAIN"],
            index=2,  # Default to UNCERTAIN
            key="quality_label"
        )
        
        confidence = st.slider(
            "Confidence (optional):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="confidence_slider"
        )
        
        notes = st.text_area(
            "Notes (optional):",
            placeholder="Add any notes about this segment...",
            key="segment_notes"
        )
    
    # Create segment button
    if st.button("➕ Add Annotation", type="primary"):
        if selected_speaker == "Select speaker...":
            st.error("Please select a speaker")
            return
        
        # Parse speaker selection
        speaker_id = None
        speaker_label = None
        
        if selected_speaker.startswith("👤"):
            # Enrolled speaker
            speaker_name = selected_speaker.split("(")[1].rstrip(")")
            speaker_id = speaker_name
        elif selected_speaker == "❓ Unknown Speaker":
            # Unknown speaker - use a simple "unknown" label
            speaker_label = "unknown"
        
        # Add to session state
        new_segment = {
            'start_time': start_time,
            'end_time': end_time,
            'speaker_id': speaker_id,
            'speaker_label': speaker_label,
            'label': label,
            'confidence': confidence,
            'notes': notes,
            'annotation_id': None  # Will be set when saved to DB
        }
        
        st.session_state.annotation_segments.append(new_segment)
        st.success(f"✅ Added annotation: {start_time:.2f}s - {end_time:.2f}s")
        st.rerun()

def display_annotation_timeline():
    """Display timeline of all annotations."""
    if not st.session_state.annotation_segments:
        st.info("No annotations yet. Create segments above to start annotating.")
        return
    
    st.subheader("📅 Annotation Timeline")
    
    # Prepare data for timeline
    segments = [(seg['start_time'], seg['end_time']) for seg in st.session_state.annotation_segments]
    labels = []
    colors = []
    
    color_map = {
        'CORRECT': '#90EE90',  # Light green
        'INCORRECT': '#FFB6C1',  # Light pink
        'UNCERTAIN': '#FFE4B5'  # Light yellow
    }
    
    for seg in st.session_state.annotation_segments:
        if seg['speaker_id']:
            label = f"👤 {seg['speaker_id']}"
        elif seg['speaker_label'] == "unknown":
            label = "❓ Unknown"
        else:
            # For unmapped deepgram labels
            label = f"🎙️ {seg['speaker_label']}"
        
        label += f" ({seg['label']})"
        labels.append(label)
        colors.append(color_map.get(seg['label'], '#D3D3D3'))
    
    # Create timeline
    fig = create_segment_timeline(
        segments,
        labels=labels,
        colors=colors,
        total_duration=st.session_state.annotation_info['duration_seconds'],
        height=max(200, len(segments) * 30),
        clickable=True
    )
    
    # Display interactive timeline
    timeline_selected = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="timeline_plot")
    
    # Add timeline play buttons row
    if len(segments) > 0:
        st.write("**Timeline Segment Players:**")
        timeline_cols = st.columns(min(len(segments), 6))  # Max 6 columns for timeline
        
        for idx, (start_time, end_time) in enumerate(segments):
            col_idx = idx % 6
            with timeline_cols[col_idx]:
                # Create a more descriptive label
                timeline_label = f"T{idx+1}"
                if idx < len(labels):
                    speaker_part = labels[idx].split('(')[0].strip()
                    timeline_label = f"T{idx+1}: {speaker_part[:6]}..."
                
                if st.button(f"▶️ {timeline_label}", key=f"timeline_play_{idx}", help=f"Play {start_time:.1f}s-{end_time:.1f}s"):
                    st.info(f"🔊 Playing timeline segment {idx + 1}: {start_time:.2f}s - {end_time:.2f}s")
                    
                    try:
                        segment_audio, _ = load_audio_segment(
                            st.session_state.annotation_temp_file,
                            start_time, end_time,
                            st.session_state.annotation_sample_rate
                        )
                        audio_bytes = audio_to_bytes(segment_audio, st.session_state.annotation_sample_rate)
                        st.audio(audio_bytes, format='audio/wav', autoplay=True)
                    except Exception as e:
                        st.error(f"Error playing segment: {str(e)}")
    
    # Handle clicks on timeline segments (keeping this as backup)
    if timeline_selected and hasattr(timeline_selected, 'selection') and timeline_selected.selection:
        if 'points' in timeline_selected.selection and timeline_selected.selection['points']:
            for point in timeline_selected.selection['points']:
                if 'customdata' in point and point['customdata']:
                    segment_idx, start_time, end_time = point['customdata']
                    st.info(f"🔊 Playing timeline segment {segment_idx + 1}: {start_time:.2f}s - {end_time:.2f}s")
                    
                    try:
                        segment_audio, _ = load_audio_segment(
                            st.session_state.annotation_temp_file,
                            start_time, end_time,
                            st.session_state.annotation_sample_rate
                        )
                        audio_bytes = audio_to_bytes(segment_audio, st.session_state.annotation_sample_rate)
                        st.audio(audio_bytes, format='audio/wav', autoplay=True)
                    except Exception as e:
                        st.error(f"Error playing segment: {str(e)}")

def annotation_list_management():
    """Manage the list of annotations."""
    if not st.session_state.annotation_segments:
        return
    
    st.subheader("📋 Annotation Management")
    
    # Summary stats
    total_annotations = len(st.session_state.annotation_segments)
    correct_count = sum(1 for seg in st.session_state.annotation_segments if seg['label'] == 'CORRECT')
    incorrect_count = sum(1 for seg in st.session_state.annotation_segments if seg['label'] == 'INCORRECT')
    uncertain_count = sum(1 for seg in st.session_state.annotation_segments if seg['label'] == 'UNCERTAIN')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", total_annotations)
    with col2:
        st.metric("✅ Correct", correct_count)
    with col3:
        st.metric("❌ Incorrect", incorrect_count)
    with col4:
        st.metric("❓ Uncertain", uncertain_count)
    
    # Annotation list with edit/delete options
    with st.expander(f"📝 Edit Annotations ({total_annotations} items)"):
        for i, seg in enumerate(st.session_state.annotation_segments):
            with st.container():
                # Show Deepgram speaker label if present
                if seg.get('deepgram_speaker_label'):
                    st.caption(f"🎙️ Deepgram: {seg['deepgram_speaker_label']}")
                
                col1, col2, col3, col4, col5 = st.columns([3, 0.5, 3, 1.2, 1])
                
                with col1:
                    # Speaker selection dropdown
                    speaker_options = get_speaker_options()
                    
                    # Find current selection
                    current_index = 0
                    if seg['speaker_id']:
                        # Find enrolled speaker
                        for j, opt in enumerate(speaker_options):
                            if f"({seg['speaker_id']})" in opt:
                                current_index = j
                                break
                    elif seg['speaker_label'] == "unknown":
                        # Find Unknown Speaker option
                        for j, opt in enumerate(speaker_options):
                            if opt == "❓ Unknown Speaker":
                                current_index = j
                                break
                    
                    new_speaker = st.selectbox(
                        "Speaker",
                        options=speaker_options,
                        index=current_index,
                        key=f"speaker_select_{i}",
                        label_visibility="collapsed"
                    )
                
                with col2:
                    # Play button for this segment
                    if st.button("▶️", key=f"quick_play_{i}", help="Play segment"):
                        # Set a flag to play this segment
                        st.session_state[f"play_segment_{i}"] = True
                
                # Check if speaker changed (moved outside columns)
                speaker_changed = new_speaker != speaker_options[current_index] and new_speaker != "Select speaker..."
                
                with col3:
                    # Time range and transcription preview
                    st.write(f"⏱️ {seg['start_time']:.1f}s - {seg['end_time']:.1f}s")
                    if seg.get('transcription'):
                        st.caption(f'"{seg["transcription"][:50]}..."' if len(seg.get("transcription", "")) > 50 else f'"{seg.get("transcription", "")}"')
                
                with col4:
                    # Quality label
                    new_label = st.selectbox(
                        "Label",
                        ["CORRECT", "INCORRECT", "UNCERTAIN"],
                        index=["CORRECT", "INCORRECT", "UNCERTAIN"].index(seg['label']),
                        key=f"label_{i}",
                        label_visibility="collapsed"
                    )
                    if new_label != seg['label']:
                        seg['label'] = new_label
                
                with col5:
                    # Action buttons
                    bcol1, bcol2 = st.columns(2)
                    
                    with bcol1:
                        # Edit notes
                        if st.button("📝", key=f"notes_{i}", help="Edit notes"):
                            seg['_edit_notes'] = not seg.get('_edit_notes', False)
                            st.rerun()
                    
                    with bcol2:
                        # Delete button
                        if st.button("🗑️", key=f"delete_{i}", help="Delete annotation"):
                            st.session_state.annotation_segments.pop(i)
                            st.rerun()
                
                # Notes editing
                if seg.get('_edit_notes'):
                    new_notes = st.text_area(
                        "Notes:",
                        value=seg.get('notes', ''),
                        key=f"notes_edit_{i}"
                    )
                    if st.button("Save Notes", key=f"save_notes_{i}"):
                        seg['notes'] = new_notes
                        seg['_edit_notes'] = False
                        st.rerun()
                elif seg.get('notes'):
                    st.caption(f"📝 {seg['notes']}")
                
                # Audio player outside columns (if play button was clicked)
                if st.session_state.get(f"play_segment_{i}", False):
                    try:
                        segment_audio, _ = load_audio_segment(
                            st.session_state.annotation_temp_file,
                            seg['start_time'], seg['end_time'],
                            st.session_state.annotation_sample_rate
                        )
                        audio_bytes = audio_to_bytes(segment_audio, st.session_state.annotation_sample_rate)
                        st.audio(audio_bytes, format='audio/wav', autoplay=True)
                        # Clear the flag after playing
                        st.session_state[f"play_segment_{i}"] = False
                    except Exception as e:
                        st.error(f"Error playing segment: {str(e)}")
                
                # Process speaker change outside columns
                if speaker_changed:
                    # Parse new speaker selection
                    new_speaker_id = None
                    new_speaker_label = None
                    
                    if new_speaker.startswith("👤"):
                        # Enrolled speaker
                        speaker_name = new_speaker.split("(")[1].rstrip(")")
                        new_speaker_id = speaker_name
                    elif new_speaker == "❓ Unknown Speaker":
                        # Unknown speaker
                        new_speaker_label = "unknown"
                    
                    # Update segment
                    seg['speaker_id'] = new_speaker_id
                    seg['speaker_label'] = new_speaker_label
                    
                    # If this is from Deepgram, offer to apply to all
                    if seg.get('deepgram_speaker_label'):
                        deepgram_label = seg['deepgram_speaker_label']
                        matching_count = sum(1 for s in st.session_state.annotation_segments 
                                           if s.get('deepgram_speaker_label') == deepgram_label and s != seg)
                        
                        if matching_count > 0:
                            # Display checkbox in a full-width container below
                            st.markdown("") # Add some spacing
                            
                            # Create expandable section for other segments
                            # Use session state to track expander state, default to True when speaker changes
                            expander_key = f"expander_{i}_{deepgram_label}"
                            if expander_key not in st.session_state.expander_states:
                                st.session_state.expander_states[expander_key] = True  # Open by default when speaker assignment happens
                            
                            with st.expander(f"Apply to all {deepgram_label} ({matching_count} other segments)", expanded=st.session_state.expander_states[expander_key]):
                                # Apply to all button
                                apply_col1, apply_col2 = st.columns([2, 1])
                                with apply_col1:
                                    st.write(f"**Apply this speaker assignment to all {matching_count} other {deepgram_label} segments**")
                                with apply_col2:
                                    if st.button(f"Apply to All", key=f"apply_all_{i}", type="primary"):
                                        # Apply to all matching segments immediately
                                        for other_seg in st.session_state.annotation_segments:
                                            if other_seg.get('deepgram_speaker_label') == deepgram_label:
                                                other_seg['speaker_id'] = new_speaker_id
                                                other_seg['speaker_label'] = new_speaker_label
                                        st.success(f"✅ Applied to all {deepgram_label} segments")
                                        # Don't trigger rerun here to keep expander open
                                
                                st.divider()
                                
                                # Show other segments with audio players and individual apply buttons
                                st.write("**Other segments with this Deepgram label:**")
                                
                                # Pre-load and cache audio segments for this speaker
                                cache_key = f"audio_cache_{deepgram_label}_{i}"
                                if cache_key not in st.session_state:
                                    st.session_state[cache_key] = {}
                                    segments_to_load = [temp_seg for temp_seg in st.session_state.annotation_segments 
                                                      if temp_seg.get('deepgram_speaker_label') == deepgram_label and temp_seg != seg]
                                    
                                    if segments_to_load:
                                        with st.spinner(f"Loading {len(segments_to_load)} audio previews for {deepgram_label}..."):
                                            for j, temp_seg in enumerate(st.session_state.annotation_segments):
                                                if temp_seg.get('deepgram_speaker_label') == deepgram_label and temp_seg != seg:
                                                    try:
                                                        preview_audio, _ = load_audio_segment(
                                                            st.session_state.annotation_temp_file,
                                                            temp_seg['start_time'], 
                                                            temp_seg['end_time'],
                                                            st.session_state.annotation_sample_rate
                                                        )
                                                        preview_bytes = audio_to_bytes(preview_audio, st.session_state.annotation_sample_rate)
                                                        st.session_state[cache_key][j] = preview_bytes
                                                    except Exception as e:
                                                        st.session_state[cache_key][j] = None
                                                        st.warning(f"Could not load audio for segment {j}: {str(e)}")
                                        st.success(f"✅ Loaded audio previews for {deepgram_label}")
                                else:
                                    st.info(f"📱 Using cached audio for {deepgram_label}")
                                
                                other_segment_count = 0
                                for j, other_seg in enumerate(st.session_state.annotation_segments):
                                    if other_seg.get('deepgram_speaker_label') == deepgram_label and other_seg != seg:
                                        other_segment_count += 1
                                        
                                        # Use 3 columns: time, transcript+audio, apply button
                                        preview_col1, preview_col2, preview_col3 = st.columns([2, 4, 1])
                                        
                                        with preview_col1:
                                            st.write(f"**Segment {other_segment_count}**")
                                            st.write(f"{other_seg['start_time']:.1f}s - {other_seg['end_time']:.1f}s")
                                        
                                        with preview_col2:
                                            # Show transcript
                                            if other_seg.get('transcription'):
                                                preview_text = other_seg['transcription'][:80] + "..." if len(other_seg['transcription']) > 80 else other_seg['transcription']
                                                st.caption(f'"{preview_text}"')
                                            
                                            # Show audio player (cached)
                                            cached_audio = st.session_state[cache_key].get(j)
                                            if cached_audio:
                                                st.audio(cached_audio, format='audio/wav')
                                            else:
                                                st.error("Audio not available")
                                        
                                        with preview_col3:
                                            if st.button("Apply", key=f"apply_individual_{i}_{j}", help="Apply speaker to this segment only"):
                                                other_seg['speaker_id'] = new_speaker_id
                                                other_seg['speaker_label'] = new_speaker_label
                                                st.success(f"✅ Applied to segment {other_segment_count}")
                                        
                                        st.divider()
                
                st.divider()

def save_annotations():
    """Save annotations to database."""
    if not st.session_state.annotation_segments or not st.session_state.user_id:
        return
    
    st.subheader("💾 Save Annotations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save to Database", type="primary"):
            db = get_db_session()
            try:
                saved_count = 0
                
                for seg in st.session_state.annotation_segments:
                    if seg.get('annotation_id') is None:  # New annotation
                        # Create new Annotation object
                        new_annotation = Annotation(
                            audio_file_path=st.session_state.annotation_temp_file,
                            audio_file_hash=st.session_state.annotation_file_hash,
                            audio_file_name=st.session_state.annotation_audio,
                            start_time=seg['start_time'],
                            end_time=seg['end_time'],
                            user_id=st.session_state.user_id,
                            speaker_id=seg.get('speaker_id'),
                            speaker_label=seg.get('speaker_label'),
                            deepgram_speaker_label=seg.get('deepgram_speaker_label'),
                            label=seg['label'],
                            confidence=seg.get('confidence', 0.5),
                            transcription=seg.get('transcription'),
                            notes=seg.get('notes') if seg.get('notes') else None
                        )
                        db.add(new_annotation)
                        saved_count += 1
                
                db.commit()
                
                st.success(f"✅ Saved {saved_count} new annotations to database!")
                
                # Reload annotations to get IDs
                load_existing_annotations(st.session_state.annotation_file_hash, st.session_state.annotation_audio)
                
            except Exception as e:
                st.error(f"Error saving annotations: {str(e)}")
            finally:
                db.close()
    
    with col2:
        # Export annotations as JSON
        if st.button("📤 Export JSON"):
            try:
                export_data = {
                    "audio_file": st.session_state.annotation_audio,
                    "total_duration": st.session_state.annotation_info['duration_seconds'],
                    "annotations": st.session_state.annotation_segments
                }
                
                json_data = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📥 Download Annotations JSON",
                    data=json_data,
                    file_name=f"annotations_{st.session_state.annotation_audio.split('.')[0]}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Export error: {str(e)}")
    
    with col3:
        # Clear all annotations
        if st.button("🗑️ Clear All", help="Clear all annotations (not saved ones)"):
            if st.session_state.annotation_segments:
                st.session_state.annotation_segments = []
                st.success("Cleared all annotations from memory")
                st.rerun()

def speaker_enrollment_interface():
    """Interface for enrolling speakers from annotations."""
    if not st.session_state.annotation_segments or not st.session_state.user_id:
        return
    
    st.subheader("👤 Enroll Speakers from Annotations")
    st.markdown("Automatically enroll speakers using quality-based filtering from your annotated segments.")
    
    # Use the shared filter component
    filters = combined_filters_component(st.session_state.user_id, "enrollment")
    
    if filters["is_valid"]:
        # Preview enrollment statistics
        st.write("**📊 Enrollment Preview**")
        
        # Get current annotations as objects for filtering
        temp_annotations = []
        for seg in st.session_state.annotation_segments:
            # Create temporary annotation object for filtering
            class TempAnnotation:
                def __init__(self, seg):
                    self.speaker_id = seg.get('speaker_id')
                    self.label = seg.get('label', 'UNCERTAIN')
                    self.confidence = seg.get('confidence', 0.0)
                    self.start_time = seg.get('start_time')
                    self.end_time = seg.get('end_time')
                    self.transcription = seg.get('transcription', '')
            
            if seg.get('speaker_id'):  # Only include segments with assigned speakers
                temp_annotations.append(TempAnnotation(seg))
        
        # Apply filters
        filtered_annotations = apply_annotation_filters(temp_annotations, filters)
        
        # Group by speaker
        speaker_stats = {}
        for ann in filtered_annotations:
            if ann.speaker_id not in speaker_stats:
                speaker_stats[ann.speaker_id] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'avg_confidence': 0.0,
                    'confidences': []
                }
            
            speaker_stats[ann.speaker_id]['count'] += 1
            speaker_stats[ann.speaker_id]['total_duration'] += (ann.end_time - ann.start_time)
            speaker_stats[ann.speaker_id]['confidences'].append(ann.confidence)
        
        # Calculate averages
        for speaker_id, stats in speaker_stats.items():
            if stats['confidences']:
                stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
        
        # Display preview
        if speaker_stats:
            st.write(f"**{len(speaker_stats)} speakers** will be enrolled with **{sum(s['count'] for s in speaker_stats.values())} segments** total.")
            
            # Show per-speaker breakdown
            with st.expander("📋 Speaker Breakdown", expanded=True):
                for speaker_id, stats in speaker_stats.items():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(f"👤 {speaker_id}", f"{stats['count']} segments")
                    with col2:
                        st.metric("Duration", f"{stats['total_duration']:.1f}s")
                    with col3:
                        st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
                    with col4:
                        # Check if speaker already enrolled
                        db = get_db_session()
                        try:
                            speaker = SpeakerQueries.get_speaker(db, speaker_id)
                            if speaker:
                                st.info("Already exists")
                            else:
                                st.success("New speaker")
                        finally:
                            db.close()
            
            st.divider()
            
            # Enrollment button
            col1, col2, col3 = st.columns(3)
            with col2:
                if st.button("🚀 Start Enrollment", type="primary", use_container_width=True):
                    enroll_speakers_from_annotations(filtered_annotations, speaker_stats)
        else:
            st.info("No segments match the current filter criteria. Adjust your filters to see enrollment candidates.")

def enroll_speakers_from_annotations(filtered_annotations, speaker_stats):
    """Process enrollment from filtered annotations."""
    import requests
    import tempfile
    import os
    from utils.audio_processing import load_audio_segment, audio_to_bytes
    
    SPEAKER_SERVICE_URL = os.getenv("SPEAKER_SERVICE_URL", "http://localhost:8001")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_speakers = len(speaker_stats)
    enrolled_count = 0
    failed_count = 0
    
    try:
        for i, (speaker_id, stats) in enumerate(speaker_stats.items()):
            status_text.text(f"Processing speaker {i+1}/{total_speakers}: {speaker_id}")
            progress_bar.progress(i / total_speakers)
            
            # Get all segments for this speaker
            speaker_annotations = [ann for ann in filtered_annotations if ann.speaker_id == speaker_id]
            
            # Create temporary audio files for each segment
            temp_files = []
            try:
                for j, ann in enumerate(speaker_annotations):
                    # Load audio segment
                    segment_audio, sample_rate = load_audio_segment(
                        st.session_state.annotation_temp_file,
                        ann.start_time,
                        ann.end_time,
                        st.session_state.annotation_sample_rate
                    )
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        audio_bytes = audio_to_bytes(segment_audio, sample_rate, "WAV")
                        temp_file.write(audio_bytes)
                        temp_files.append(temp_file.name)
                
                # Send all segments to batch enrollment endpoint
                if temp_files:
                    # Prepare multiple files for upload
                    files = []
                    for temp_file_path in temp_files:
                        files.append(('files', (f'segment_{temp_files.index(temp_file_path)}.wav', 
                                              open(temp_file_path, 'rb'), 
                                              'audio/wav')))
                    
                    data = {
                        'speaker_name': speaker_id,
                        'speaker_id': speaker_id
                    }
                    
                    try:
                        # Use batch enrollment endpoint for multiple segments
                        response = requests.post(f"{SPEAKER_SERVICE_URL}/enroll/batch", 
                                               files=files, 
                                               data=data)
                        
                        # Close all file handles
                        for _, file_tuple in files:
                            file_tuple[1].close()
                        
                        if response.status_code == 200:
                            result = response.json()
                            enrolled_count += 1
                            st.success(f"✅ Enrolled {speaker_id} successfully using {result['num_segments']} segments")
                        else:
                            failed_count += 1
                            st.error(f"❌ Failed to enroll {speaker_id}: {response.text}")
                    except requests.exceptions.RequestException as e:
                        failed_count += 1
                        st.error(f"❌ Connection error for {speaker_id}: {str(e)}")
                        # Make sure to close file handles on error
                        for _, file_tuple in files:
                            try:
                                file_tuple[1].close()
                            except:
                                pass
                
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
        
        # Final status
        progress_bar.progress(1.0)
        status_text.text(f"Enrollment complete! ✅ {enrolled_count} successful, ❌ {failed_count} failed")
        
        if enrolled_count > 0:
            st.balloons()
    
    except Exception as e:
        st.error(f"❌ Enrollment process failed: {str(e)}")
        progress_bar.progress(0)
        status_text.text("Enrollment failed")

def batch_annotation_tools():
    """Tools for batch annotation operations."""
    if not st.session_state.annotation_segments:
        return
    
    st.subheader("⚡ Batch Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Bulk Label Change:**")
        new_label = st.selectbox(
            "Change all to:",
            ["CORRECT", "INCORRECT", "UNCERTAIN"],
            key="bulk_label"
        )
        
        if st.button("Apply to All"):
            for seg in st.session_state.annotation_segments:
                seg['label'] = new_label
            st.success(f"Changed all annotations to {new_label}")
            st.rerun()
    
    with col2:
        st.write("**Filter by Label:**")
        filter_label = st.selectbox(
            "Show only:",
            ["All", "CORRECT", "INCORRECT", "UNCERTAIN"],
            key="filter_label"
        )
        
        if filter_label != "All":
            filtered_segments = [seg for seg in st.session_state.annotation_segments if seg['label'] == filter_label]
            st.info(f"Showing {len(filtered_segments)} out of {len(st.session_state.annotation_segments)} annotations")
    
    with col3:
        st.write("**Quality Check:**")
        min_duration = st.number_input(
            "Min segment duration (s):",
            min_value=0.1,
            value=0.5,
            step=0.1,
            key="min_duration"
        )
        
        short_segments = [
            seg for seg in st.session_state.annotation_segments 
            if (seg['end_time'] - seg['start_time']) < min_duration
        ]
        
        if short_segments:
            st.warning(f"⚠️ {len(short_segments)} segments shorter than {min_duration}s")

def main():
    """Main annotation page."""
    st.title("📝 Annotation Tool")
    st.markdown("Label speaker segments in your audio files with interactive annotation tools.")
    
    # Check if user is logged in
    if "username" not in st.session_state or not st.session_state.username:
        st.warning("👈 Please select or create a user in the sidebar to continue.")
        return
    
    # Initialize session state
    init_session_state()
    
    # Upload section
    file_uploaded = upload_annotation_file()
    
    # Check if we have files loaded (either just uploaded or from session state)
    if file_uploaded or st.session_state.annotation_file_loaded:
        # Audio info
        if st.session_state.annotation_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{st.session_state.annotation_info['duration_seconds']:.2f}s")
            with col2:
                st.metric("Sample Rate", f"{st.session_state.annotation_info['sample_rate']:,} Hz")
            with col3:
                st.metric("File", st.session_state.annotation_audio)
        
        st.divider()
        
        # Waveform with annotations
        if st.session_state.annotation_audio_data is not None:
            st.subheader("🌊 Waveform with Annotations")
            
            # Prepare segments and colors for waveform
            segments = [(seg['start_time'], seg['end_time']) for seg in st.session_state.annotation_segments]
            colors = ['yellow', 'lightgreen', 'lightcoral', 'lightblue', 'lightgray']
            segment_colors = [colors[i % len(colors)] for i in range(len(segments))]
            
            fig = create_waveform_plot(
                st.session_state.annotation_audio_data,
                st.session_state.annotation_sample_rate,
                title=f"Annotated Waveform - {st.session_state.annotation_audio} (Click segments to play)",
                height=300,
                segments=segments,
                segment_colors=segment_colors
            )
            
            # Display the interactive plot
            selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="waveform_plot")
            
            # Handle clicks on waveform segments
            if selected_points and hasattr(selected_points, 'selection') and selected_points.selection:
                if 'points' in selected_points.selection and selected_points.selection['points']:
                    for point in selected_points.selection['points']:
                        if 'customdata' in point and point['customdata']:
                            segment_idx, start_time, end_time = point['customdata']
                            st.info(f"🔊 Playing segment {segment_idx + 1}: {start_time:.2f}s - {end_time:.2f}s")
                            
                            try:
                                segment_audio, _ = load_audio_segment(
                                    st.session_state.annotation_temp_file,
                                    start_time, end_time,
                                    st.session_state.annotation_sample_rate
                                )
                                audio_bytes = audio_to_bytes(segment_audio, st.session_state.annotation_sample_rate)
                                st.audio(audio_bytes, format='audio/wav', autoplay=True)
                            except Exception as e:
                                st.error(f"Error playing segment: {str(e)}")
        
        st.divider()
        
        # Speaker mapping interface (for Deepgram imports)
        speaker_mapping_interface()
        
        # Annotation interface
        segment_annotation_interface()
        
        st.divider()
        
        # Timeline view
        display_annotation_timeline()
        
        st.divider()
        
        # Annotation management
        annotation_list_management()
        
        # Batch tools
        batch_annotation_tools()
        
        st.divider()
        
        # Save/export
        save_annotations()
        
        st.divider()
        
        # Speaker enrollment from annotations
        speaker_enrollment_interface()
    
    else:
        # Help when no file uploaded
        st.info("👆 Upload an audio file to start annotating!")
        
        with st.expander("ℹ️ How to use the Annotation Tool"):
            st.markdown("""
            ### Features:
            - **Upload Audio**: Support for various audio formats
            - **Visual Timeline**: See all annotations on an interactive timeline
            - **Speaker Assignment**: Assign segments to known speakers or create unknown speakers
            - **Quality Labels**: Mark segments as CORRECT, INCORRECT, or UNCERTAIN
            - **Batch Operations**: Apply changes to multiple annotations at once
            - **Export Options**: Save to database or export as JSON
            
            ### Workflow:
            1. **Upload** your audio file
            2. **Create segments** by specifying start/end times
            3. **Assign speakers** - either enrolled speakers or unknown speakers
            4. **Set quality labels** to indicate annotation confidence
            5. **Review** annotations in the timeline view
            6. **Save** to database or export for later use
            
            ### Speaker Options:
            - **👤 Enrolled Speakers**: Speakers already registered in the system
            - **❓ Unknown Speakers**: Create labels for speakers not yet enrolled
            - **➕ Add New Unknown**: Create custom unknown speaker labels
            
            ### Tips:
            - Use keyboard shortcuts when available for faster annotation
            - Start with UNCERTAIN labels and refine as you review
            - Use the batch operations to quickly update multiple segments
            - Save frequently to avoid losing work
            """)

if __name__ == "__main__":
    main()