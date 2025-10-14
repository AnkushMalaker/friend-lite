"""
Workers package - RQ job definitions and queue utilities.

This package provides modular RQ job functions organized by domain:
- transcription_jobs: Speech-to-text processing
- conversation_jobs: Conversation management and updates
- memory_jobs: Memory extraction and processing
- audio_jobs: Audio file processing and cropping

Queue configuration and utilities are in controllers/queue_controller.py
"""

# Import from transcription_jobs
from .transcription_jobs import (
    transcribe_full_audio_job,
    recognise_speakers_job,
    stream_speech_detection_job,
)

# Import from conversation_jobs
from .conversation_jobs import (
    open_conversation_job,
)

# Import from memory_jobs
from .memory_jobs import (
    process_memory_job,
    enqueue_memory_processing,
)

# Import from audio_jobs
from .audio_jobs import (
    process_audio_job,
    process_cropping_job,
    audio_streaming_persistence_job,
    enqueue_audio_processing,
    enqueue_cropping,
)

# Import from queue_controller
from advanced_omi_backend.controllers.queue_controller import (
    get_queue,
    get_job_stats,
    get_jobs,
    get_queue_health,
    transcription_queue,
    memory_queue,
    default_queue,
    redis_conn,
    REDIS_URL,
    JOB_RESULT_TTL,
    _ensure_beanie_initialized,
    TRANSCRIPTION_QUEUE,
    MEMORY_QUEUE,
    DEFAULT_QUEUE,
)

__all__ = [
    # Transcription jobs
    "transcribe_full_audio_job",
    "recognise_speakers_job",
    "stream_speech_detection_job",

    # Conversation jobs
    "open_conversation_job",
    "audio_streaming_persistence_job",

    # Memory jobs
    "process_memory_job",
    "enqueue_memory_processing",

    # Audio jobs
    "process_audio_job",
    "process_cropping_job",
    "enqueue_audio_processing",
    "enqueue_cropping",

    # Queue utils
    "get_queue",
    "get_job_stats",
    "get_jobs",
    "get_queue_health",
    "transcription_queue",
    "memory_queue",
    "default_queue",
    "redis_conn",
    "REDIS_URL",
    "JOB_RESULT_TTL",
    "_ensure_beanie_initialized",
    "TRANSCRIPTION_QUEUE",
    "MEMORY_QUEUE",
    "DEFAULT_QUEUE",
]
