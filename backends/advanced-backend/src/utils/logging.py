import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advanced-backend")
audio_logger = logging.getLogger("audio_processing")
memory_logger = logging.getLogger("memory_service")
items_logger = logging.getLogger("action_items_service")
audio_cropper_logger = logging.getLogger("audio_cropper")