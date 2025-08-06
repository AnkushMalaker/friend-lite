"""Core speaker recognition components."""

from .audio_backend import AudioBackend
from .unified_speaker_db import UnifiedSpeakerDB
from .models import *

__all__ = ["AudioBackend", "UnifiedSpeakerDB"]