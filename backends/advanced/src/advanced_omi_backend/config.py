"""
Configuration management for Friend-Lite backend.

Currently contains diarization settings because they were used in multiple places 
causing circular imports. Other configurations can be moved here as needed.
"""

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Default diarization settings
DEFAULT_DIARIZATION_SETTINGS = {
    "diarization_source": "pyannote",
    "similarity_threshold": 0.15,
    "min_duration": 0.5,
    "collar": 2.0,
    "min_duration_off": 1.5,
    "min_speakers": 2,
    "max_speakers": 6
}

# Global cache for diarization settings
_diarization_settings = None


def get_diarization_config_path():
    """Get the path to the diarization config file."""
    # Try different locations in order of preference
    # 1. Data directory (for persistence across container restarts)
    data_path = Path("/app/data/diarization_config.json")
    if data_path.parent.exists():
        return data_path
    
    # 2. App root directory
    app_path = Path("/app/diarization_config.json")
    if app_path.parent.exists():
        return app_path
    
    # 3. Local development path
    local_path = Path("diarization_config.json")
    return local_path


def load_diarization_settings_from_file():
    """Load diarization settings from file or create from template."""
    global _diarization_settings
    
    config_path = get_diarization_config_path()
    template_path = Path("/app/diarization_config.json.template")
    
    # If no template, try local development path
    if not template_path.exists():
        template_path = Path("diarization_config.json.template")
    
    # If config doesn't exist, try to copy from template
    if not config_path.exists():
        if template_path.exists():
            try:
                # Ensure parent directory exists
                config_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(template_path, config_path)
                logger.info(f"Created diarization config from template at {config_path}")
            except Exception as e:
                logger.warning(f"Could not copy template to {config_path}: {e}")
    
    # Load from file if it exists
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                _diarization_settings = json.load(f)
                logger.info(f"Loaded diarization settings from {config_path}")
                return _diarization_settings
        except Exception as e:
            logger.error(f"Error loading diarization settings from {config_path}: {e}")
    
    # Fall back to defaults
    _diarization_settings = DEFAULT_DIARIZATION_SETTINGS.copy()
    logger.info("Using default diarization settings")
    return _diarization_settings


def save_diarization_settings_to_file(settings):
    """Save diarization settings to file."""
    global _diarization_settings
    
    config_path = get_diarization_config_path()
    
    try:
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write settings to file
        with open(config_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        # Update cache
        _diarization_settings = settings
        
        logger.info(f"Saved diarization settings to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving diarization settings to {config_path}: {e}")
        return False


# Initialize settings on module load
_diarization_settings = load_diarization_settings_from_file()