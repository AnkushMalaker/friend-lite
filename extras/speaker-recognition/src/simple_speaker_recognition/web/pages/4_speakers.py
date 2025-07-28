"""Speaker management page with export functionality."""

import datetime
import io
import json
import logging
import os
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import streamlit as st

# Configuration from environment
SPEAKER_SERVICE_URL = os.getenv("SPEAKER_SERVICE_URL", "http://localhost:8001")

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from database import get_db_session
from database.queries import (
    AnnotationQueries,
    EnrollmentQueries,
    ExportQueries,
    SpeakerQueries,
    UserQueries,
)
from database.models import Annotation
from sqlalchemy import func
from utils.audio_processing import (
    audio_to_bytes,
    concatenate_audio_segments,
    load_audio_segment,
)
from utils.visualization import (
    create_embedding_visualization,
    create_quality_metrics_plot,
    create_speaker_similarity_matrix,
)
from utils.filter_components import combined_filters_component, apply_annotation_filters


def init_session_state():
    """Initialize session state for speaker management."""
    if "selected_speakers" not in st.session_state:
        st.session_state.selected_speakers = []
    if "export_format" not in st.session_state:
        st.session_state.export_format = "concatenated"

def speaker_list_interface():
    """Display and manage the list of speakers."""
    st.subheader("👥 Your Speakers")
    
    if not st.session_state.user_id:
        st.warning("Please select a user to view speakers.")
        return []
    
    # Get speakers for current user
    db = get_db_session()
    try:
        speakers = SpeakerQueries.get_speakers_for_user(db, st.session_state.user_id)
        
        if not speakers:
            st.info("No speakers enrolled yet. Go to the Enrollment page to add speakers.")
            return []
        
        # Display speakers in expandable sections
        speaker_data = []
        
        for speaker in speakers:
            try:
                # Get speaker statistics
                stats = SpeakerQueries.get_speaker_quality_stats(db, str(speaker.id))
            except Exception as stats_error:
                st.error(f"Error loading stats for speaker {speaker.name}: {str(stats_error)}")
                logger.error(f"Error loading stats for speaker {speaker.name}: {traceback.format_exc()}")
                continue
            
            with st.expander(f"👤 {speaker.name} ({speaker.id})", expanded=False):
                # Basic info
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Sessions", stats['total_sessions'])
                with col2:
                    st.metric("Duration", f"{stats['total_duration']:.1f}s")
                with col3:
                    if stats['total_sessions'] == 0:
                        st.metric("Status", "📝 Created Only")
                        st.caption("Speaker not enrolled")
                    else:
                        st.metric("Avg Quality", f"{stats['avg_quality']:.1%}")
                with col4:
                    if stats['total_sessions'] == 0:
                        # Show annotation count instead
                        db_temp = get_db_session()
                        try:
                            annotation_count = db_temp.query(func.count(Annotation.id)).filter(
                                Annotation.speaker_id == speaker.id
                            ).scalar() or 0
                            st.metric("Annotations", annotation_count)
                        except:
                            st.metric("Annotations", "0")
                        finally:
                            db_temp.close()
                    else:
                        st.metric("Best Quality", f"{stats['best_quality']:.1%}")
                
                # Management buttons
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Select for export
                    selected = st.checkbox(
                        "Select for export",
                        key=f"select_{speaker.id}",
                        value=speaker.id in st.session_state.selected_speakers
                    )
                    
                    if selected and speaker.id not in st.session_state.selected_speakers:
                        st.session_state.selected_speakers.append(speaker.id)
                    elif not selected and speaker.id in st.session_state.selected_speakers:
                        st.session_state.selected_speakers.remove(speaker.id)
                
                with col2:
                    # View details
                    if st.button("📊 Details", key=f"details_{speaker.id}"):
                        try:
                            show_speaker_details(speaker, stats)
                        except Exception as detail_error:
                            st.error(f"Error showing speaker details: {str(detail_error)}")
                            logger.error(f"Error in show_speaker_details for {speaker.name}: {traceback.format_exc()}")
                
                with col3:
                    # Edit speaker
                    if st.button("✏️ Edit", key=f"edit_{speaker.id}"):
                        edit_speaker_dialog(speaker)
                
                with col4:
                    # Delete speaker
                    if st.button("🗑️ Delete", key=f"delete_{speaker.id}"):
                        delete_speaker_dialog(speaker)
                
                # Show enrollment sessions
                sessions = EnrollmentQueries.get_sessions_for_speaker(db, str(speaker.id))
                if sessions:
                    st.write("**Recent Enrollment Sessions:**")
                    for session in sessions[:3]:  # Show last 3 sessions
                        st.caption(
                            f"📅 {session.created_at.strftime('%Y-%m-%d %H:%M')} | "
                            f"Quality: {session.quality_score:.1%} | "
                            f"Duration: {session.speech_duration_seconds:.1f}s | "
                            f"Method: {session.enrollment_method}"
                        )
            
            speaker_data.append({
                'speaker': speaker,
                'stats': stats,
                'sessions': sessions
            })
        
        return speaker_data
    
    except Exception as e:
        st.error(f"Error loading speakers: {str(e)}")
        logger.error(f"Error in speaker_list_interface: {traceback.format_exc()}")
        return []
    finally:
        db.close()

def show_speaker_details(speaker, stats):
    """Show detailed information about a speaker."""
    try:
        st.subheader(f"📊 Details for {speaker.name}")
        
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Speaker Information:**")
            st.write(f"• **ID**: {speaker.id}")
            st.write(f"• **Name**: {speaker.name}")
            
            # Safe datetime formatting
            try:
                created_str = speaker.created_at.strftime('%Y-%m-%d %H:%M') if speaker.created_at else "Unknown"
                st.write(f"• **Created**: {created_str}")
            except Exception as e:
                logger.error(f"Error formatting created_at for speaker {speaker.name}: {e}")
                st.write("• **Created**: Unknown")
            
            try:
                updated_str = speaker.updated_at.strftime('%Y-%m-%d %H:%M') if speaker.updated_at else "Unknown"
                st.write(f"• **Last Updated**: {updated_str}")
            except Exception as e:
                logger.error(f"Error formatting updated_at for speaker {speaker.name}: {e}")
                st.write("• **Last Updated**: Unknown")
                
            if speaker.notes:
                st.write(f"• **Notes**: {speaker.notes}")
        
        with col2:
            st.write("**Quality Metrics:**")
            st.write(f"• **Total Sessions**: {stats.get('total_sessions', 0)}")
            st.write(f"• **Total Duration**: {stats.get('total_duration', 0.0):.1f} seconds")
            st.write(f"• **Average Quality**: {stats.get('avg_quality', 0.0):.1%}")
            st.write(f"• **Best Quality**: {stats.get('best_quality', 0.0):.1%}")
            
            # Safe latest session formatting
            if stats.get('latest_session'):
                try:
                    latest_str = stats['latest_session'].strftime('%Y-%m-%d %H:%M')
                    st.write(f"• **Latest Session**: {latest_str}")
                except Exception as e:
                    logger.error(f"Error formatting latest_session for speaker {speaker.name}: {e}")
                    st.write("• **Latest Session**: Invalid date")
            else:
                st.write("• **Latest Session**: Never enrolled")
    except Exception as e:
        logger.error(f"Error in show_speaker_details basic info for {speaker.name}: {traceback.format_exc()}")
        st.error(f"Error displaying basic speaker information: {str(e)}")
    
    # Get enrollment sessions
    db = get_db_session()
    try:
        sessions = EnrollmentQueries.get_sessions_for_speaker(db, speaker.id)
        
        if sessions:
            st.write("**Enrollment History:**")
            
            # Create quality metrics plot - with safe date formatting
            session_dates = []
            for s in sessions[-10:]:  # Last 10 sessions
                try:
                    date_str = s.created_at.strftime('%m/%d') if s.created_at else 'Unknown'
                    session_dates.append(date_str)
                except Exception as e:
                    logger.error(f"Error formatting session date: {e}")
                    session_dates.append('Invalid')
            
            quality_scores = [s.quality_score or 0.0 for s in sessions[-10:]]
            durations = [s.speech_duration_seconds or 0.0 for s in sessions[-10:]]
            snr_values = [s.snr_db or 0.0 for s in sessions[-10:]]
            
            metrics = {
                'Quality Score': quality_scores,
                'Duration (s)': durations,
                'SNR (dB)': snr_values
            }
            
            fig = create_quality_metrics_plot(
                metrics, 
                session_dates,
                title=f"Quality Metrics for {speaker.name}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Session details table
            st.write("**Session Details:**")
            for session in sessions:
                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        try:
                            date_str = session.created_at.strftime('%Y-%m-%d %H:%M') if session.created_at else 'Unknown'
                            st.write(date_str)
                        except Exception as e:
                            logger.error(f"Error formatting session created_at: {e}")
                            st.write('Invalid Date')
                    with col2:
                        st.write(f"{(session.quality_score or 0.0):.1%}")
                    with col3:
                        st.write(f"{(session.speech_duration_seconds or 0.0):.1f}s")
                    with col4:
                        st.write(session.enrollment_method)
                    
                    st.divider()
    
    except Exception as e:
        st.error(f"Error loading session details: {str(e)}")
        logger.error(f"Error in show_speaker_details sessions for {speaker.name}: {traceback.format_exc()}")
    finally:
        db.close()

def edit_speaker_dialog(speaker):
    """Dialog to edit speaker information."""
    st.subheader(f"✏️ Edit Speaker: {speaker.name}")
    
    with st.form(f"edit_speaker_{speaker.id}"):
        new_name = st.text_input("Speaker Name:", value=speaker.name)
        new_notes = st.text_area("Notes:", value=speaker.notes or "", help="Optional notes about this speaker")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("💾 Save Changes", type="primary"):
                db = get_db_session()
                try:
                    # Update speaker in database
                    speaker.name = new_name
                    speaker.notes = new_notes
                    db.commit()
                    
                    st.success(f"✅ Updated speaker: {new_name}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error updating speaker: {str(e)}")
                    db.rollback()
                finally:
                    db.close()
        
        with col2:
            if st.form_submit_button("❌ Cancel"):
                st.rerun()

def delete_speaker_dialog(speaker):
    """Dialog to confirm speaker deletion."""
    st.subheader(f"🗑️ Delete Speaker: {speaker.name}")
    
    st.warning(f"⚠️ Are you sure you want to delete speaker '{speaker.name}'?")
    st.write("This will permanently remove:")
    st.write("• All enrollment sessions")
    st.write("• All annotations for this speaker")
    st.write("• Speaker embeddings from the recognition system")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("❌ Cancel", key=f"cancel_delete_{speaker.id}"):
            st.rerun()
    
    with col2:
        # Require typing speaker name to confirm
        confirm_name = st.text_input(f"Type '{speaker.name}' to confirm:", key=f"confirm_{speaker.id}")
    
    with col3:
        if st.button(
            "🗑️ DELETE", 
            key=f"confirm_delete_{speaker.id}",
            type="primary",
            disabled=confirm_name != speaker.name
        ):
            if confirm_name == speaker.name:
                db = get_db_session()
                try:
                    # Delete from speaker service first
                    import requests
                    try:
                        response = requests.delete(f"{SPEAKER_SERVICE_URL}/speakers/{speaker.id}")
                        if response.status_code == 200:
                            st.success("✅ Removed from speaker recognition system")
                        else:
                            st.warning("⚠️ Could not remove from speaker service, but will delete from database")
                    except requests.exceptions.ConnectionError:
                        st.warning("⚠️ Speaker service not available, deleting from database only")
                    
                    # Delete from database
                    if SpeakerQueries.delete_speaker(db, speaker.id):
                        st.success(f"✅ Deleted speaker: {speaker.name}")
                        
                        # Remove from selected speakers if present
                        if speaker.id in st.session_state.selected_speakers:
                            st.session_state.selected_speakers.remove(speaker.id)
                        
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete speaker")
                
                except Exception as e:
                    st.error(f"Error deleting speaker: {str(e)}")
                finally:
                    db.close()

def export_interface():
    """Interface for exporting speaker data."""
    st.subheader("📤 Export Speaker Data")
    
    if not st.session_state.selected_speakers:
        st.info("Please select speakers above to enable export options.")
        return
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.radio(
            "Export Format:",
            ["concatenated", "segments", "json_only"],
            format_func=lambda x: {
                "concatenated": "📁 Concatenated (max 10min per file)",
                "segments": "📂 Individual Segments",
                "json_only": "📄 JSON Only (no audio files)"
            }[x],
            key="export_format_radio"
        )
        st.session_state.export_format = export_format
    
    with col2:
        include_metadata = st.checkbox("Include metadata JSON", value=True)
        include_annotations = st.checkbox("Include annotations", value=True)
        if export_format != "json_only":
            audio_format = st.selectbox("Audio format:", ["WAV", "MP3"], index=0)
        else:
            audio_format = "N/A"
    
    st.divider()
    
    # Use shared filter component
    filters = combined_filters_component(st.session_state.user_id, "export")
    
    # Export summary
    st.write(f"**Selected for export:** {len(st.session_state.selected_speakers)} speakers")
    
    selected_speaker_names = []
    db = get_db_session()
    try:
        for speaker_id in st.session_state.selected_speakers:
            speaker = SpeakerQueries.get_speaker(db, speaker_id)
            if speaker:
                selected_speaker_names.append(f"• {speaker.name} ({speaker.id})")
    finally:
        db.close()
    
    for name in selected_speaker_names:
        st.write(name)
    
    # Export buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📥 Export Selected", type="primary", use_container_width=True):
            if filters["is_valid"]:
                export_selected_speakers(
                    st.session_state.selected_speakers,
                    export_format,
                    include_metadata,
                    include_annotations,
                    audio_format,
                    filters
                )
    
    with col2:
        if st.button("📥 Export All", use_container_width=True):
            if filters["is_valid"]:
                export_all_speakers(
                    export_format,
                    include_metadata,
                    include_annotations,
                    audio_format,
                    filters
                )
    
    with col3:
        if st.button("📦 Dataset ZIP", use_container_width=True, help="Download cropped audio dataset"):
            if filters["is_valid"]:
                download_dataset_zip(filters)
    
    with col4:
        if st.button("🗑️ Clear Selection", use_container_width=True):
            st.session_state.selected_speakers = []
            st.rerun()

def export_selected_speakers(
    speaker_ids: List[str],
    export_format: str,
    include_metadata: bool,
    include_annotations: bool,
    audio_format: str,
    filters: Dict[str, Any]
):
    """Export selected speakers data."""
    if not speaker_ids:
        st.error("No speakers selected for export.")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            export_data = {}
            
            total_speakers = len(speaker_ids)
            
            for i, speaker_id in enumerate(speaker_ids):
                status_text.text(f"Processing speaker {i+1}/{total_speakers}: {speaker_id}")
                progress_bar.progress((i / total_speakers) * 0.8)
                
                speaker_export = export_single_speaker(
                    speaker_id, 
                    temp_path, 
                    export_format,
                    include_annotations,
                    audio_format,
                    filters
                )
                
                if speaker_export:
                    export_data[speaker_id] = speaker_export
            
            # Create metadata file if requested
            if include_metadata:
                status_text.text("Creating metadata...")
                progress_bar.progress(0.9)
                
                metadata = {
                    "export_timestamp": st.session_state.get("export_timestamp", "unknown"),
                    "export_format": export_format,
                    "audio_format": audio_format,
                    "include_annotations": include_annotations,
                    "filters_applied": filters,
                    "speakers": export_data
                }
                
                metadata_path = temp_path / "export_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            # Create ZIP file
            status_text.text("Creating ZIP file...")
            progress_bar.progress(0.95)
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in temp_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(temp_path)
                        zip_file.write(file_path, arcname)
            
            # Offer download
            status_text.text("Export complete!")
            progress_bar.progress(1.0)
            
            zip_buffer.seek(0)
            st.download_button(
                label="📥 Download Export ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"speaker_export_{len(speaker_ids)}speakers.zip",
                mime="application/zip"
            )
            
            st.success(f"✅ Successfully exported {len(speaker_ids)} speakers!")
    
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")
        progress_bar.progress(0)
        status_text.text("Export failed")

def export_single_speaker(
    speaker_id: str,
    base_path: Path,
    export_format: str,
    include_annotations: bool,
    audio_format: str,
    filters: Dict[str, Any]
) -> Dict[str, Any]:
    """Export a single speaker's data."""
    db = get_db_session()
    try:
        speaker = SpeakerQueries.get_speaker(db, speaker_id)
        if not speaker:
            return None
        
        speaker_dir = base_path / f"speaker-{speaker.name.replace(' ', '_')}"
        speaker_dir.mkdir(exist_ok=True)
        
        # Get annotations for this speaker
        annotations = AnnotationQueries.get_annotations_for_speaker(db, speaker_id)
        
        # Apply filters using the shared filter function
        annotations = apply_annotation_filters(annotations, filters)
        
        if not annotations:
            # No annotations found, create empty structure
            return {
                "speaker_info": {
                    "id": speaker.id,
                    "name": speaker.name,
                    "notes": speaker.notes
                },
                "audio_files": [],
                "annotations": []
            }
        
        audio_files = []
        
        if export_format == "json_only":
            # Skip audio processing for JSON-only export
            pass
        elif export_format == "concatenated":
            # Concatenate segments up to 10 minutes per file
            audio_files = export_concatenated_audio(annotations, speaker_dir, audio_format)
        else:
            # Export each segment as separate file
            audio_files = export_segmented_audio(annotations, speaker_dir, audio_format)
        
        # Export annotations if requested
        annotation_data = []
        if include_annotations:
            for ann in annotations:
                annotation_data.append({
                    "start_time": ann.start_time,
                    "end_time": ann.end_time,
                    "label": ann.label,
                    "confidence": ann.confidence,
                    "notes": ann.notes,
                    "created_at": ann.created_at.isoformat()
                })
            
            # Save annotations file
            annotations_path = speaker_dir / "annotations.json"
            with open(annotations_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)
        
        return {
            "speaker_info": {
                "id": speaker.id,
                "name": speaker.name,
                "notes": speaker.notes
            },
            "audio_files": audio_files,
            "annotations": annotation_data
        }
    
    except Exception as e:
        st.error(f"Error exporting speaker {speaker_id}: {str(e)}")
        return None
    finally:
        db.close()

def export_concatenated_audio(annotations: list, speaker_dir: Path, audio_format: str) -> List[str]:
    """Export audio as concatenated files (max 10 minutes each)."""
    audio_files = []
    current_segments = []
    current_duration = 0.0
    file_index = 1
    
    for annotation in annotations:
        try:
            # Load audio segment
            segment_audio, sr = load_audio_segment(
                annotation.audio_file_path,
                annotation.start_time,
                annotation.end_time
            )
            
            segment_duration = len(segment_audio) / sr
            
            # Check if adding this segment would exceed 10 minutes
            if current_duration + segment_duration > 600 and current_segments:  # 10 minutes = 600 seconds
                # Save current concatenated file
                if current_segments:
                    concatenated_audio = concatenate_audio_segments(current_segments, sr)
                    
                    file_name = f"audio_part_{file_index:03d}.{audio_format.lower()}"
                    file_path = speaker_dir / file_name
                    
                    audio_bytes = audio_to_bytes(concatenated_audio, sr, audio_format)
                    with open(file_path, 'wb') as f:
                        f.write(audio_bytes)
                    
                    audio_files.append(file_name)
                    file_index += 1
                
                # Start new file
                current_segments = [segment_audio]
                current_duration = segment_duration
            else:
                current_segments.append(segment_audio)
                current_duration += segment_duration
        
        except Exception as e:
            st.warning(f"Could not load segment: {str(e)}")
            continue
    
    # Save remaining segments
    if current_segments:
        concatenated_audio = concatenate_audio_segments(current_segments, sr)
        
        file_name = f"audio_part_{file_index:03d}.{audio_format.lower()}"
        file_path = speaker_dir / file_name
        
        audio_bytes = audio_to_bytes(concatenated_audio, sr, audio_format)
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        
        audio_files.append(file_name)
    
    return audio_files

def export_segmented_audio(annotations: list, speaker_dir: Path, audio_format: str) -> List[str]:
    """Export each annotation as a separate audio file."""
    audio_files = []
    
    for i, annotation in enumerate(annotations):
        try:
            # Load audio segment
            segment_audio, sr = load_audio_segment(
                annotation.audio_file_path,
                annotation.start_time,
                annotation.end_time
            )
            
            # Create filename
            file_name = f"audio{i+1:04d}_{annotation.start_time:.1f}s-{annotation.end_time:.1f}s.{audio_format.lower()}"
            file_path = speaker_dir / file_name
            
            # Save audio file
            audio_bytes = audio_to_bytes(segment_audio, sr, audio_format)
            with open(file_path, 'wb') as f:
                f.write(audio_bytes)
            
            audio_files.append(file_name)
        
        except Exception as e:
            st.warning(f"Could not export segment {i+1}: {str(e)}")
            continue
    
    return audio_files

def export_all_speakers(
    export_format: str,
    include_metadata: bool,
    include_annotations: bool,
    audio_format: str,
    filters: Dict[str, Any]
):
    """Export all speakers for the current user."""
    if not st.session_state.user_id:
        st.error("No user selected.")
        return
    
    db = get_db_session()
    try:
        speakers = SpeakerQueries.get_speakers_for_user(db, st.session_state.user_id)
        speaker_ids = [speaker.id for speaker in speakers]
        
        if not speaker_ids:
            st.info("No speakers to export.")
            return
        
        export_selected_speakers(
            speaker_ids, 
            export_format, 
            include_metadata, 
            include_annotations, 
            audio_format,
            filters
        )
    
    except Exception as e:
        st.error(f"Error exporting all speakers: {str(e)}")
    finally:
        db.close()

def download_dataset_zip(filters: Dict[str, Any]):
    """Create and download a dataset ZIP with cropped audio segments."""
    import datetime
    from utils.audio_processing import load_audio_segment, audio_to_bytes
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Create timestamp for filename
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M")
        
        # Create temporary directory for dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_dir = temp_path / f"dataset_{timestamp}"
            dataset_dir.mkdir(exist_ok=True)
            
            status_text.text("Gathering annotations...")
            progress_bar.progress(0.1)
            
            # Get all annotations for selected speakers
            db = get_db_session()
            try:
                all_annotations = []
                for speaker_id in filters["speaker_filters"]:
                    annotations = AnnotationQueries.get_annotations_for_speaker(db, speaker_id)
                    all_annotations.extend(annotations)
                
                # Apply filters
                filtered_annotations = apply_annotation_filters(all_annotations, filters)
                
                if not filtered_annotations:
                    st.error("No annotations match the current filters.")
                    return
                
                status_text.text(f"Processing {len(filtered_annotations)} segments...")
                progress_bar.progress(0.2)
                
                # Group by speaker
                speaker_segments = {}
                for ann in filtered_annotations:
                    if ann.speaker_id not in speaker_segments:
                        speaker_segments[ann.speaker_id] = []
                    speaker_segments[ann.speaker_id].append(ann)
                
                # Create speaker directories and process segments
                total_segments = len(filtered_annotations)
                processed_segments = 0
                
                dataset_metadata = {
                    "created_at": now.isoformat(),
                    "filters_applied": filters,
                    "total_speakers": len(speaker_segments),
                    "total_segments": total_segments,
                    "speakers": {}
                }
                
                for speaker_id, segments in speaker_segments.items():
                    speaker_dir = dataset_dir / speaker_id
                    speaker_dir.mkdir(exist_ok=True)
                    
                    speaker_metadata = {
                        "speaker_id": speaker_id,
                        "segment_count": len(segments),
                        "segments": []
                    }
                    
                    for i, ann in enumerate(segments):
                        status_text.text(f"Processing {speaker_id}: segment {i+1}/{len(segments)}")
                        progress_bar.progress(0.2 + (processed_segments / total_segments) * 0.7)
                        
                        try:
                            # Load audio segment
                            segment_audio, sample_rate = load_audio_segment(
                                ann.audio_file_path,
                                ann.start_time,
                                ann.end_time
                            )
                            
                            # Create filename
                            duration = ann.end_time - ann.start_time
                            filename = f"segment_{i+1:03d}_{ann.start_time:.1f}s-{ann.end_time:.1f}s_{duration:.1f}s.wav"
                            file_path = speaker_dir / filename
                            
                            # Save audio file
                            audio_bytes = audio_to_bytes(segment_audio, sample_rate, "WAV")
                            with open(file_path, 'wb') as f:
                                f.write(audio_bytes)
                            
                            # Add to metadata
                            segment_metadata = {
                                "filename": filename,
                                "start_time": ann.start_time,
                                "end_time": ann.end_time,
                                "duration": duration,
                                "label": ann.label,
                                "confidence": ann.confidence,
                                "transcription": ann.transcription,
                                "original_audio_file": Path(ann.audio_file_path).name
                            }
                            speaker_metadata["segments"].append(segment_metadata)
                            
                        except Exception as e:
                            st.warning(f"Skipped segment due to error: {str(e)}")
                        
                        processed_segments += 1
                    
                    dataset_metadata["speakers"][speaker_id] = speaker_metadata
                
                # Save dataset metadata
                status_text.text("Creating metadata...")
                progress_bar.progress(0.9)
                
                metadata_path = dataset_dir / "dataset_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(dataset_metadata, f, indent=2, default=str)
                
                # Create README
                readme_content = f"""# Speaker Recognition Dataset
                
Created: {now.strftime("%Y-%m-%d %H:%M:%S")}
Total Speakers: {len(speaker_segments)}
Total Segments: {total_segments}

## Structure
- Each speaker has their own directory: `{speaker_id}/`
- Audio files are named: `segment_XXX_START-END_DURATION.wav`
- See `dataset_metadata.json` for detailed information

## Filters Applied
- Quality Labels: {filters['quality_filters']}
- Confidence Threshold: {filters['confidence_threshold']}
- Speakers: {len(filters['speaker_filters'])} selected

## Usage
This dataset can be used for:
1. Speaker enrollment via the enrollment endpoint
2. Training speaker recognition models
3. Audio analysis and research
"""
                
                readme_path = dataset_dir / "README.md"
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                
                # Create ZIP file
                status_text.text("Creating ZIP file...")
                progress_bar.progress(0.95)
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for file_path in dataset_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(dataset_dir)
                            zip_file.write(file_path, arcname)
                
                # Offer download
                status_text.text("Dataset ready for download!")
                progress_bar.progress(1.0)
                
                zip_buffer.seek(0)
                st.download_button(
                    label="📥 Download Dataset ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"dataset_{timestamp}.zip",
                    mime="application/zip"
                )
                
                st.success(f"✅ Dataset created successfully! {total_segments} segments from {len(speaker_segments)} speakers.")
                
            finally:
                db.close()
                
    except Exception as e:
        st.error(f"❌ Dataset creation failed: {str(e)}")
        progress_bar.progress(0)
        status_text.text("Dataset creation failed")

def bulk_operations():
    """Bulk operations for multiple speakers."""
    if not st.session_state.selected_speakers:
        return
    
    st.subheader("⚡ Bulk Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏷️ Bulk Tag", use_container_width=True):
            st.info("Bulk tagging feature coming soon...")
    
    with col2:
        if st.button("🔄 Bulk Re-enroll", use_container_width=True):
            st.info("Bulk re-enrollment feature coming soon...")
    
    with col3:
        if st.button("📊 Compare Quality", use_container_width=True):
            compare_speaker_quality()

def compare_speaker_quality():
    """Compare quality metrics across selected speakers."""
    if len(st.session_state.selected_speakers) < 2:
        st.warning("Please select at least 2 speakers to compare.")
        return
    
    st.subheader("📊 Speaker Quality Comparison")
    
    db = get_db_session()
    try:
        speaker_names = []
        durations = []
        snr_values = []
        quality_scores = []
        session_counts = []
        
        for speaker_id in st.session_state.selected_speakers:
            speaker = SpeakerQueries.get_speaker(db, speaker_id)
            stats = SpeakerQueries.get_speaker_quality_stats(db, speaker_id)
            sessions = EnrollmentQueries.get_sessions_for_speaker(db, speaker_id)
            
            if speaker and sessions:
                speaker_names.append(speaker.name)
                durations.append(stats['total_duration'])
                session_counts.append(len(sessions))
                
                # Calculate average metrics across all sessions
                avg_snr = np.mean([s.snr_db for s in sessions if s.snr_db])
                avg_quality = np.mean([s.quality_score for s in sessions if s.quality_score])
                
                snr_values.append(avg_snr)
                quality_scores.append(avg_quality)
        
        if speaker_names:
            # Create comparison plot
            metrics = {
                'Duration (s)': durations,
                'SNR (dB)': snr_values,
                'Sessions': session_counts,
                'Quality Score': quality_scores
            }
            
            fig = create_quality_metrics_plot(
                metrics,
                speaker_names,
                title="Speaker Quality Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary table
            st.subheader("📋 Comparison Summary")
            
            comparison_data = []
            for i, name in enumerate(speaker_names):
                comparison_data.append({
                    'Speaker': name,
                    'Duration (s)': f"{durations[i]:.1f}",
                    'SNR (dB)': f"{snr_values[i]:.1f}",
                    'Sessions': f"{session_counts[i]}",
                    'Quality': f"{quality_scores[i]:.1%}"
                })
            
            st.table(comparison_data)
    
    except Exception as e:
        st.error(f"Error comparing speakers: {str(e)}")
    finally:
        db.close()

def main():
    """Main speaker management page."""
    st.title("👥 Speaker Management")
    st.markdown("Manage enrolled speakers, view statistics, and export your data.")
    
    # Check if user is logged in
    if "username" not in st.session_state or not st.session_state.username:
        st.warning("👈 Please select or create a user in the sidebar to continue.")
        return
    
    # Initialize session state
    init_session_state()
    
    # Speaker list and management
    speaker_data = speaker_list_interface()
    
    if speaker_data:
        st.divider()
        
        # Export interface
        export_interface()
        
        st.divider()
        
        # Bulk operations
        if st.session_state.selected_speakers:
            bulk_operations()
    
    # Help section
    with st.expander("ℹ️ Speaker Management Help"):
        st.markdown("""
        ### Features:
        
        **Speaker Overview:**
        - View all enrolled speakers with quality statistics
        - See enrollment history and session details
        - Track quality improvements over time
        
        **Export Options:**
        - **Concatenated**: Combine segments into files (max 10 minutes each)
        - **Segments**: Export each annotation as separate file
        - **Formats**: WAV or MP3 audio output
        - **Metadata**: Include JSON files with timestamps and quality info
        - **Annotations**: Include annotation labels and confidence scores
        
        **Management Actions:**
        - **Edit**: Update speaker name and notes
        - **Delete**: Remove speaker and all associated data
        - **Compare**: Analyze quality metrics across speakers
        - **Bulk Export**: Process multiple speakers at once
        
        ### Export Structure:
        ```
        exported_data/
        ├── speaker-John_Doe/
        │   ├── audio0001.wav
        │   ├── audio0002.wav
        │   └── annotations.json
        ├── speaker-Jane_Smith/
        │   ├── audio_part_001.wav
        │   └── annotations.json
        └── export_metadata.json
        ```
        
        ### Tips:
        - Select multiple speakers for batch operations
        - Review quality metrics before exporting
        - Use concatenated format for training data
        - Use segments format for detailed analysis
        - Always include metadata for reference
        """)

if __name__ == "__main__":
    main()