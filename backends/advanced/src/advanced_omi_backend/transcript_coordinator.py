"""Transcript Coordinator for Event-Driven Memory Processing.

This module provides proper async coordination between transcript completion and memory processing,
eliminating polling/retry mechanisms in favor of asyncio events.
"""

import asyncio
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TranscriptCoordinator:
    """Coordinates transcript completion events across the system.

    This replaces polling/retry mechanisms with proper asyncio event coordination.
    When transcription is saved to the database, it signals waiting memory processors.
    """

    def __init__(self):
        self.transcript_events: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        logger.info("TranscriptCoordinator initialized")

    async def wait_for_transcript_completion(self, audio_uuid: str, timeout: float = 30.0) -> bool:
        """Wait for transcript completion for the given audio_uuid.

        Args:
            audio_uuid: The audio UUID to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            True if transcript was completed, False if timeout
        """
        async with self._lock:
            # Create event for this audio_uuid if it doesn't exist
            if audio_uuid not in self.transcript_events:
                self.transcript_events[audio_uuid] = asyncio.Event()
                logger.info(f"Created transcript wait event for {audio_uuid}")

        event = self.transcript_events[audio_uuid]

        try:
            # Wait for the transcript to be ready
            await asyncio.wait_for(event.wait(), timeout=timeout)
            logger.info(f"Transcript ready event received for {audio_uuid}")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Transcript wait timeout ({timeout}s) for {audio_uuid}")
            return False
        finally:
            # Clean up the event
            async with self._lock:
                self.transcript_events.pop(audio_uuid, None)
                logger.debug(f"Cleaned up transcript event for {audio_uuid}")

    def signal_transcript_ready(self, audio_uuid: str):
        """Signal that transcript is ready for the given audio_uuid.

        This should be called by TranscriptionManager after successfully saving
        transcript segments to the database.

        Args:
            audio_uuid: The audio UUID that has completed transcription
        """
        if audio_uuid in self.transcript_events:
            self.transcript_events[audio_uuid].set()
            logger.info(f"Signaled transcript ready for {audio_uuid}")
        else:
            logger.debug(f"No waiting processors for transcript {audio_uuid}")
    
    def cleanup_transcript_events_for_client(self, client_id: str):
        """Clean up any transcript events associated with a disconnected client.
        
        This prevents memory leaks and orphaned events when clients disconnect
        before transcription completes.
        
        Args:
            client_id: The client ID that disconnected
        """
        # Since we don't track client_id -> audio_uuid mapping here,
        # this is a safety method that can be called but currently has limited scope
        # In the future, we could enhance this by tracking client associations
        events_cleaned = 0
        for audio_uuid in list(self.transcript_events.keys()):
            # For now, we'll rely on the timeout mechanism in wait_for_transcript_completion
            # Future enhancement: track client_id associations to enable targeted cleanup
            pass
        
        if events_cleaned > 0:
            logger.info(f"Cleaned up {events_cleaned} transcript events for disconnected client {client_id}")
        else:
            logger.debug(f"No transcript events to clean up for client {client_id}")

    async def cleanup_stale_events(self, max_age_seconds: float = 300.0):
        """Clean up any stale events that might be left over.

        This is a safety mechanism to prevent memory leaks if events are not
        properly cleaned up during normal operation.

        Args:
            max_age_seconds: Maximum age for events before cleanup
        """
        async with self._lock:
            # For now, just log the count - in a real implementation you'd track creation times
            stale_count = len(self.transcript_events)
            if stale_count > 0:
                logger.warning(f"Found {stale_count} potentially stale transcript events")

    def get_waiting_count(self) -> int:
        """Get the number of currently waiting transcript events."""
        return len(self.transcript_events)


# Global singleton instance
_transcript_coordinator: Optional[TranscriptCoordinator] = None


def get_transcript_coordinator() -> TranscriptCoordinator:
    """Get the global TranscriptCoordinator instance."""
    global _transcript_coordinator
    if _transcript_coordinator is None:
        _transcript_coordinator = TranscriptCoordinator()
    return _transcript_coordinator
