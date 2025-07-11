"""
Service initialization and configuration module.
Handles database connections, service setup, and dependency injection.
"""
import os
import logging
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
import ollama

# Memory service imports
from memory import get_memory_service, init_memory_config

# Service manager and transcript services
from service_interface import TranscriptServiceManager
from other_services.action_items_service import ActionItemsService
from other_services.coaching_service import CoachingService

# Set up logging
services_logger = logging.getLogger("services")

# Global variables to hold initialized services
_services_initialized = False
_mongo_client = None
_db = None
_collections = {}
_services = {}

def get_config():
    """Get configuration from environment variables."""
    return {
        # MongoDB Configuration
        "MONGODB_URI": os.getenv("MONGODB_URI", "mongodb://mongo:27017"),
        
        # Audio Configuration
        "OMI_SAMPLE_RATE": 16_000,
        "OMI_CHANNELS": 1,
        "OMI_SAMPLE_WIDTH": 2,
        "SEGMENT_SECONDS": 60,
        "NEW_CONVERSATION_TIMEOUT_MINUTES": float(os.getenv("NEW_CONVERSATION_TIMEOUT_MINUTES", "1.5")),
        
        # Audio cropping configuration
        "AUDIO_CROPPING_ENABLED": os.getenv("AUDIO_CROPPING_ENABLED", "true").lower() == "true",
        "MIN_SPEECH_SEGMENT_DURATION": float(os.getenv("MIN_SPEECH_SEGMENT_DURATION", "1.0")),
        "CROPPING_CONTEXT_PADDING": float(os.getenv("CROPPING_CONTEXT_PADDING", "0.1")),
        
        # Directory configuration
        "CHUNK_DIR": Path("./audio_chunks"),
        
        # ASR Configuration
        "OFFLINE_ASR_TCP_URI": os.getenv("OFFLINE_ASR_TCP_URI", "tcp://192.168.0.110:8765/"),
        "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY"),
        
        # AI Services Configuration
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        "QDRANT_BASE_URL": os.getenv("QDRANT_BASE_URL", "qdrant"),
    }

def initialize_database():
    """Initialize MongoDB database and collections."""
    global _mongo_client, _db, _collections
    
    if _mongo_client is not None:
        return _mongo_client, _db, _collections
    
    config = get_config()
    
    # Initialize MongoDB
    _mongo_client = AsyncIOMotorClient(config["MONGODB_URI"])
    _db = _mongo_client.get_default_database("friend-lite")
    
    # Initialize collections
    _collections = {
        "chunks": _db["audio_chunks"],
        "users": _db["users"],
        "speakers": _db["speakers"],
        "action_items": _db["action_items"],
    }
    
    # Ensure chunk directory exists
    config["CHUNK_DIR"].mkdir(parents=True, exist_ok=True)
    
    services_logger.info("Database initialized successfully")
    return _mongo_client, _db, _collections

def initialize_ai_services():
    """Initialize AI and memory services."""
    global _services
    
    if "memory_service" in _services:
        return _services
    
    config = get_config()
    
    # Initialize memory configuration
    init_memory_config(
        ollama_base_url=config["OLLAMA_BASE_URL"],
        qdrant_base_url=config["QDRANT_BASE_URL"],
    )
    
    # Initialize services
    _services["memory_service"] = get_memory_service()
    _services["ollama_client"] = ollama.Client(host=config["OLLAMA_BASE_URL"])
    
    services_logger.info("AI services initialized successfully")
    return _services

def initialize_transcript_services():
    """Initialize transcript processing services."""
    global _services
    
    if "transcript_service_manager" in _services:
        return _services["transcript_service_manager"]
    
    # Ensure database and AI services are initialized
    _, _, collections = initialize_database()
    ai_services = initialize_ai_services()
    
    # Initialize transcript service manager
    transcript_service_manager = TranscriptServiceManager()
    
    # Register services
    action_items_service = ActionItemsService(
        collections["action_items"], 
        ai_services["ollama_client"]
    )
    coaching_service = CoachingService()
    
    transcript_service_manager.register_service(action_items_service)
    transcript_service_manager.register_service(coaching_service)
    
    _services["transcript_service_manager"] = transcript_service_manager
    _services["action_items_service"] = action_items_service
    _services["coaching_service"] = coaching_service
    
    services_logger.info("Transcript services initialized successfully")
    return transcript_service_manager

def initialize_all_services():
    """Initialize all services in the correct order."""
    global _services_initialized
    
    if _services_initialized:
        return get_all_services()
    
    services_logger.info("Initializing all services...")
    
    # Initialize in order
    mongo_client, db, collections = initialize_database()
    ai_services = initialize_ai_services()
    transcript_service_manager = initialize_transcript_services()
    
    _services_initialized = True
    services_logger.info("All services initialized successfully")
    
    return {
        "mongo_client": mongo_client,
        "db": db,
        "collections": collections,
        **ai_services,
        "transcript_service_manager": transcript_service_manager,
    }

def get_all_services():
    """Get all initialized services."""
    if not _services_initialized:
        return initialize_all_services()
    
    return {
        "mongo_client": _mongo_client,
        "db": _db,
        "collections": _collections,
        **_services,
    }

def get_database():
    """Get database client and collections."""
    if _mongo_client is None:
        initialize_database()
    return _mongo_client, _db, _collections

def get_transcript_service_manager():
    """Get the transcript service manager."""
    if "transcript_service_manager" not in _services:
        initialize_transcript_services()
    return _services["transcript_service_manager"]

def get_memory_service_instance():
    """Get the memory service instance."""
    if "memory_service" not in _services:
        initialize_ai_services()
    return _services["memory_service"]

def get_ollama_client():
    """Get the Ollama client."""
    if "ollama_client" not in _services:
        initialize_ai_services()
    return _services["ollama_client"]

# Computed configurations
def get_target_samples():
    """Get target samples for audio processing."""
    config = get_config()
    return config["OMI_SAMPLE_RATE"] * config["SEGMENT_SECONDS"]

def use_deepgram():
    """Check if Deepgram should be used."""
    config = get_config()
    return bool(config["DEEPGRAM_API_KEY"])

def get_audio_config():
    """Get audio processing configuration."""
    config = get_config()
    return {
        "sample_rate": config["OMI_SAMPLE_RATE"],
        "channels": config["OMI_CHANNELS"],
        "sample_width": config["OMI_SAMPLE_WIDTH"],
        "segment_seconds": config["SEGMENT_SECONDS"],
        "target_samples": get_target_samples(),
        "use_deepgram": use_deepgram(),
        "chunk_dir": config["CHUNK_DIR"],
        "offline_asr_uri": config["OFFLINE_ASR_TCP_URI"],
        "deepgram_api_key": config["DEEPGRAM_API_KEY"],
    }

def get_conversation_config():
    """Get conversation configuration."""
    config = get_config()
    return {
        "timeout_minutes": config["NEW_CONVERSATION_TIMEOUT_MINUTES"],
        "audio_cropping_enabled": config["AUDIO_CROPPING_ENABLED"],
        "min_speech_duration": config["MIN_SPEECH_SEGMENT_DURATION"],
        "context_padding": config["CROPPING_CONTEXT_PADDING"],
    }