from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Awaitable
import asyncio
import logging
from fastapi import APIRouter

# Set up logging
services_logger = logging.getLogger("services")

class TranscriptService(ABC):
    """
    Abstract base class for services that process transcript data.
    
    This interface allows for extensible transcript processing services
    like action items, coaching, analytics, etc.
    """
    
    @abstractmethod
    async def process_transcript(self, transcript_text: str, client_id: str, 
                               audio_uuid: str, user_id: str, user_email: str) -> Dict[str, Any]:
        """
        Process a transcript segment and return results.
        
        Args:
            transcript_text: The transcript text to process
            client_id: The client ID that generated the audio
            audio_uuid: Unique identifier for the audio
            user_id: Database user ID
            user_email: User email for identification
            
        Returns:
            Dict containing processing results with structure:
            {
                "success": bool,
                "count": int,  # Number of items processed
                "message": str,  # Status message
                "data": Any  # Optional additional data
            }
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service (setup databases, indexes, etc.)"""
        pass
    
    @property
    @abstractmethod
    def service_name(self) -> str:
        """Return the name of this service"""
        pass
    
    def get_router(self) -> Optional[APIRouter]:
        """
        Return FastAPI router for this service's endpoints.
        Override this method to provide REST API endpoints for the service.
        """
        return None


class TranscriptServiceManager:
    """
    Manager for handling multiple transcript processing services.
    
    Provides a unified interface for registering and processing
    transcripts through multiple services.
    """
    
    def __init__(self):
        self._services: Dict[str, TranscriptService] = {}
        self._initialized = False
        services_logger.info("TranscriptServiceManager initialized")
    
    def register_service(self, service: TranscriptService) -> None:
        """Register a new transcript processing service."""
        service_name = service.service_name
        if service_name in self._services:
            services_logger.warning(f"Service {service_name} already registered, overwriting")
        
        self._services[service_name] = service
        services_logger.info(f"Registered service: {service_name}")
    
    def unregister_service(self, service_name: str) -> None:
        """Unregister a transcript processing service."""
        if service_name in self._services:
            del self._services[service_name]
            services_logger.info(f"Unregistered service: {service_name}")
        else:
            services_logger.warning(f"Service {service_name} not found for unregistration")
    
    async def initialize_all(self) -> None:
        """Initialize all registered services."""
        if self._initialized:
            return
        
        for service_name, service in self._services.items():
            try:
                await service.initialize()
                services_logger.info(f"Initialized service: {service_name}")
            except Exception as e:
                services_logger.error(f"Failed to initialize service {service_name}: {e}")
                # Continue with other services
        
        self._initialized = True
        services_logger.info("All services initialized")
    
    async def process_transcript(self, transcript_text: str, client_id: str, 
                               audio_uuid: str, user_id: str, user_email: str) -> Dict[str, Dict[str, Any]]:
        """
        Process transcript through all registered services.
        
        Returns:
            Dict mapping service names to their processing results
        """
        if not self._initialized:
            await self.initialize_all()
        
        results = {}
        
        # Process through all services concurrently
        async def process_service(service_name: str, service: TranscriptService) -> tuple[str, Dict[str, Any]]:
            try:
                result = await service.process_transcript(
                    transcript_text, client_id, audio_uuid, user_id, user_email
                )
                services_logger.debug(f"Service {service_name} processed transcript for {audio_uuid}")
                return service_name, result
            except Exception as e:
                services_logger.error(f"Service {service_name} failed to process transcript for {audio_uuid}: {e}")
                return service_name, {
                    "success": False,
                    "count": 0,
                    "message": f"Service error: {str(e)}",
                    "data": None
                }
        
        # Run all services concurrently
        tasks = [
            process_service(service_name, service) 
            for service_name, service in self._services.items()
        ]
        
        if tasks:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in completed_results:
                if isinstance(result, tuple):
                    service_name, service_result = result
                    results[service_name] = service_result
                else:
                    # Handle exceptions from gather
                    services_logger.error(f"Unexpected error in service processing: {result}")
        
        return results
    
    def get_service(self, service_name: str) -> Optional[TranscriptService]:
        """Get a specific service by name."""
        return self._services.get(service_name)
    
    def list_services(self) -> List[str]:
        """List all registered service names."""
        return list(self._services.keys())
    
    def get_service_count(self) -> int:
        """Get the number of registered services."""
        return len(self._services)
    
    def get_all_routers(self) -> List[tuple[str, APIRouter]]:
        """
        Get all API routers from registered services.
        
        Returns:
            List of tuples containing (service_name, router)
        """
        routers = []
        for service_name, service in self._services.items():
            router = service.get_router()
            if router:
                routers.append((service_name, router))
        return routers