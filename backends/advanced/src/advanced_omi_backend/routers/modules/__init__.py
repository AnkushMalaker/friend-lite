"""
Router modules for Friend-Lite API.

This package contains organized router modules for different functional areas:
- user_routes: User management and authentication
- chat_routes: Chat interface with memory integration
- client_routes: Active client monitoring and management
- conversation_routes: Conversation CRUD and audio processing
- memory_routes: Memory management, search, and debug
- system_routes: System utilities and metrics
- queue_routes: Job queue management and monitoring
- audio_routes: Audio file uploads and processing
"""

from .audio_routes import router as audio_router
from .chat_routes import router as chat_router
from .client_routes import router as client_router
from .conversation_routes import router as conversation_router
from .memory_routes import router as memory_router
from .queue_routes import router as queue_router
from .system_routes import router as system_router
from .user_routes import router as user_router

__all__ = [
    "audio_router",
    "chat_router",
    "client_router",
    "conversation_router",
    "memory_router",
    "queue_router",
    "system_router",
    "user_router",
]
