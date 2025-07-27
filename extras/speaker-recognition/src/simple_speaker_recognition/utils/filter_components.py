"""Shared filter components for export and enrollment functionality."""

import streamlit as st
from typing import List, Dict, Any, Tuple
from database import get_db_session
from database.queries import SpeakerQueries


def quality_filter_component(key_prefix: str = "") -> List[str]:
    """Render quality filter checkboxes and return selected quality labels.
    
    Args:
        key_prefix: Prefix for streamlit widget keys to avoid conflicts
        
    Returns:
        List of selected quality labels
    """
    st.write("**Quality Filter:**")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        include_correct = st.checkbox("✅ CORRECT", value=True, key=f"{key_prefix}_correct")
    with filter_col2:
        include_incorrect = st.checkbox("❌ INCORRECT", value=False, key=f"{key_prefix}_incorrect")
    with filter_col3:
        include_uncertain = st.checkbox("❓ UNCERTAIN", value=False, key=f"{key_prefix}_uncertain")
    
    quality_filters = []
    if include_correct:
        quality_filters.append("CORRECT")
    if include_incorrect:
        quality_filters.append("INCORRECT")
    if include_uncertain:
        quality_filters.append("UNCERTAIN")
    
    return quality_filters


def speaker_filter_component(user_id: int, key_prefix: str = "") -> List[str]:
    """Render speaker filter checkboxes and return selected speaker IDs.
    
    Args:
        user_id: User ID to get speakers for
        key_prefix: Prefix for streamlit widget keys to avoid conflicts
        
    Returns:
        List of selected speaker IDs
    """
    if not user_id:
        st.warning("No user selected for speaker filtering.")
        return []
    
    db = get_db_session()
    try:
        speakers = SpeakerQueries.get_speakers_for_user(db, user_id)
        
        if not speakers:
            st.info("No speakers available for filtering.")
            return []
        
        st.write("**Speaker Filter:**")
        
        # Add "All Speakers" option
        select_all = st.checkbox("Select All Speakers", value=True, key=f"{key_prefix}_select_all")
        
        selected_speakers = []
        
        # Calculate columns needed (max 3 per row)
        num_cols = min(len(speakers), 3)
        if num_cols > 0:
            cols = st.columns(num_cols)
            
            for i, speaker in enumerate(speakers):
                col_idx = i % num_cols
                with cols[col_idx]:
                    # Use select_all to control individual checkbox default values
                    if st.checkbox(
                        f"👤 {speaker.name}",
                        value=select_all,
                        key=f"{key_prefix}_speaker_{speaker.id}"
                    ):
                        selected_speakers.append(speaker.id)
        
        return selected_speakers
        
    except Exception as e:
        st.error(f"Error loading speakers: {str(e)}")
        return []
    finally:
        db.close()


def confidence_threshold_component(key_prefix: str = "") -> float:
    """Render confidence threshold slider and return selected value.
    
    Args:
        key_prefix: Prefix for streamlit widget keys to avoid conflicts
        
    Returns:
        Selected confidence threshold (0.0 to 1.0)
    """
    st.write("**Confidence Threshold:**")
    threshold = st.slider(
        "Minimum confidence level",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        key=f"{key_prefix}_confidence_threshold",
        help="Only include segments with confidence above this threshold"
    )
    
    return threshold


def combined_filters_component(user_id: int, key_prefix: str = "") -> Dict[str, Any]:
    """Render all filter components and return combined filter dictionary.
    
    Args:
        user_id: User ID to get speakers for
        key_prefix: Prefix for streamlit widget keys to avoid conflicts
        
    Returns:
        Dictionary containing all filter values
    """
    st.subheader("🔍 Filter Options")
    
    # Quality filters
    quality_filters = quality_filter_component(key_prefix)
    
    st.divider()
    
    # Speaker filters  
    speaker_filters = speaker_filter_component(user_id, key_prefix)
    
    st.divider()
    
    # Confidence threshold
    confidence_threshold = confidence_threshold_component(key_prefix)
    
    # Validation
    filter_valid = True
    if not quality_filters:
        st.error("Please select at least one quality label.")
        filter_valid = False
    
    if not speaker_filters:
        st.error("Please select at least one speaker.")
        filter_valid = False
    
    return {
        "quality_filters": quality_filters,
        "speaker_filters": speaker_filters,
        "confidence_threshold": confidence_threshold,
        "is_valid": filter_valid
    }


def apply_annotation_filters(annotations: List, filters: Dict[str, Any]) -> List:
    """Apply filters to a list of annotation objects.
    
    Args:
        annotations: List of annotation objects
        filters: Filter dictionary from combined_filters_component
        
    Returns:
        Filtered list of annotations
    """
    if not filters.get("is_valid", False):
        return []
    
    filtered_annotations = []
    
    for ann in annotations:
        # Quality filter
        if ann.label not in filters["quality_filters"]:
            continue
            
        # Speaker filter
        if ann.speaker_id not in filters["speaker_filters"]:
            continue
            
        # Confidence threshold filter
        if (ann.confidence or 0.0) < filters["confidence_threshold"]:
            continue
            
        filtered_annotations.append(ann)
    
    return filtered_annotations