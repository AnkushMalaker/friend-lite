"""Core speaker recognition components."""

from .audio_backend import AudioBackend
from .speaker_db import SpeakerDB
from .models import *

__all__ = ["AudioBackend", "SpeakerDB"]