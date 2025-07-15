import asyncio
import logging
import time
import uuid
from typing import Optional, Tuple, Any, Type
from pathlib import Path

from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from wyoming.audio import AudioChunk

from memory import get_memory_service
from metrics import get_metrics_collector

# Logging setup
audio_logger = logging.getLogger("audio_processing")
logger = logging.getLogger("client_state")

class ClientState:
    """Manages all state for a single client connection."""

    def __init__(self, client_id: str, audio_chunk_utils, metrics_collector, active_clients, config: dict, transcription_manager_class: Type[Any]):
        self.client_id = client_id
        self.connected = True
        self.audio_chunk_utils = audio_chunk_utils
        self.metrics_collector = metrics_collector
        self.active_clients = active_clients
      

        # Configuration values
        self.CHUNK_DIR = config.get("CHUNK_DIR", Path("./audio_chunks"))
        self.OMI_SAMPLE_RATE = config.get("OMI_SAMPLE_RATE", 16_000)
        self.OMI_CHANNELS = config.get("OMI_CHANNELS", 1)
        self.OMI_SAMPLE_WIDTH = config.get("OMI_SAMPLE_WIDTH", 2)
        self.NEW_CONVERSATION_TIMEOUT_MINUTES = config.get("NEW_CONVERSATION_TIMEOUT_MINUTES", 1.5)
        self.AUDIO_CROPPING_ENABLED = config.get("AUDIO_CROPPING_ENABLED", False)
        self.MIN_SPEECH_SEGMENT_DURATION = config.get("MIN_SPEECH_SEGMENT_DURATION", 1.0)
        self.CROPPING_CONTEXT_PADDING = config.get("CROPPING_CONTEXT_PADDING", 0.1)
        self._DEC_IO_EXECUTOR = config.get("_DEC_IO_EXECUTOR")
        self.memory_service = get_memory_service() # Get global instance
        self.action_items_service = config.get("action_items_service") # Passed from main

        # Per-client queues
        self.chunk_queue = asyncio.Queue[Optional[AudioChunk]]()
        self.transcription_queue = asyncio.Queue[Tuple[Optional[str], Optional[AudioChunk]]]()
        self.memory_queue = asyncio.Queue[Tuple[Optional[str], Optional[str], Optional[str]]]()  # (transcript, client_id, audio_uuid)
        self.action_item_queue = asyncio.Queue[Tuple[Optional[str], Optional[str], Optional[str]]]()  # (transcript_text, client_id, audio_uuid)
        
        # Per-client file sink
        self.file_sink: Optional[LocalFileSink] = None
        self.current_audio_uuid: Optional[str] = None

        # Per-client transcription manager
        self.transcription_manager: Optional[Any] = None
        self.transcription_manager_class: Type[Any] = transcription_manager_class
        # Conversation timeout tracking
        self.last_transcript_time: Optional[float] = None
        self.conversation_start_time: float = time.time()

        # Speech segment tracking for audio cropping
        self.speech_segments: dict[str, list[tuple[float, float]]] = (
            {}
        )  # audio_uuid -> [(start, end), ...]
        self.current_speech_start: dict[str, Optional[float]] = (
            {}
        )  # audio_uuid -> start_time

        # Conversation transcript collection for end-of-conversation memory processing
        self.conversation_transcripts: list[str] = (
            []
        )  # Collect all transcripts for this conversation

        # Tasks for this client
        self.saver_task: Optional[asyncio.Task] = None
        self.transcription_task: Optional[asyncio.Task] = None
        self.memory_task: Optional[asyncio.Task] = None
        self.action_item_task: Optional[asyncio.Task] = None
