"""
Router modules for Friend-Lite API.

This package contains organized router modules for different functional areas:
- user_routes: User management and authentication
- chat_routes: Chat interface with memory integration
- client_routes: Active client monitoring and management
- conversation_routes: Conversation CRUD and audio processing
- memory_routes: Memory management, search, and debug
- system_routes: System utilities, metrics, and file processing
"""

from .chat_routes import router as chat_router
from .client_routes import router as client_router
from .conversation_routes import router as conversation_router
from .memory_routes import router as memory_router
from .system_routes import router as system_router
from .user_routes import router as user_router

__all__ = ["user_router", "chat_router", "client_router", "conversation_router", "memory_router", "system_router"]
