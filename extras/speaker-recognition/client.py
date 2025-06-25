#!/usr/bin/env python3
"""
Speaker Recognition Service Client

Client library for communicating with the speaker recognition service
from the advanced backend.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import aiohttp

log = logging.getLogger("speaker_client")

class SpeakerRecognitionClient:
    """Client for speaker recognition service."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make an HTTP request to the service."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
        except aiohttp.ClientError as e:
            raise Exception(f"Connection error: {e}")
    
    async def health_check(self) -> Dict:
        """Check service health."""
        return await self._request("GET", "/health")
    
    async def enroll_speaker(self, speaker_id: str, speaker_name: str, 
                           audio_file_path: str, start_time: Optional[float] = None,
                           end_time: Optional[float] = None) -> bool:
        """Enroll a speaker."""
        data = {
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "audio_file_path": audio_file_path
        }
        
        if start_time is not None:
            data["start_time"] = start_time
        if end_time is not None:
            data["end_time"] = end_time
        
        try:
            result = await self._request("POST", "/enroll", json=data)
            return result.get("success", False)
        except Exception as e:
            log.error(f"Failed to enroll speaker {speaker_id}: {e}")
            return False
    
    async def identify_speaker(self, audio_file_path: str, 
                             start_time: Optional[float] = None,
                             end_time: Optional[float] = None) -> Optional[str]:
        """Identify a speaker from audio."""
        data = {"audio_file_path": audio_file_path}
        
        if start_time is not None:
            data["start_time"] = start_time
        if end_time is not None:
            data["end_time"] = end_time
        
        try:
            result = await self._request("POST", "/identify", json=data)
            if result.get("identified"):
                return result.get("speaker_id")
            return None
        except Exception as e:
            log.error(f"Failed to identify speaker: {e}")
            return None
    
    async def diarize_audio(self, audio_file_path: str) -> Optional[Dict]:
        """Perform speaker diarization."""
        data = {"audio_file_path": audio_file_path}
        
        try:
            return await self._request("POST", "/diarize", json=data)
        except Exception as e:
            log.error(f"Failed to diarize audio: {e}")
            return None
    
    async def list_speakers(self) -> List[Dict]:
        """List all enrolled speakers."""
        try:
            result = await self._request("GET", "/speakers")
            return result.get("speakers", [])
        except Exception as e:
            log.error(f"Failed to list speakers: {e}")
            return []
    
    async def remove_speaker(self, speaker_id: str) -> bool:
        """Remove an enrolled speaker."""
        try:
            result = await self._request("DELETE", f"/speakers/{speaker_id}")
            return result.get("success", False)
        except Exception as e:
            log.error(f"Failed to remove speaker {speaker_id}: {e}")
            return False


# Convenience functions for backward compatibility
class MockSpeakerRecognition:
    """Mock object that provides the same interface as the original speaker_recognition module."""
    
    def __init__(self, client_url: str = "http://localhost:8001"):
        self.client_url = client_url
        # Mock the original attributes for compatibility
        self.audio_loader = None
        self.embedding_model = None
        self.diar = None
        self.SIMILARITY_THRESHOLD = 0.85
    
    async def process_file(self, wav_path: Path, audio_uuid: str, mongo_chunks):
        """Process audio file for speaker diarization and identification."""
        async with SpeakerRecognitionClient(self.client_url) as client:
            # Check if service is available
            try:
                await client.health_check()
            except Exception as e:
                log.warning(f"Speaker recognition service not available: {e}")
                return
            
            # Perform diarization
            result = await client.diarize_audio(str(wav_path))
            if not result:
                log.error(f"Failed to diarize audio for {audio_uuid}")
                return
            
            # Update MongoDB with results
            try:
                await mongo_chunks.update_one(
                    {"audio_uuid": audio_uuid},
                    {
                        "$set": {
                            "diarization_segments": result["segments"],
                            "speaker_embeddings": result["speaker_embeddings"]
                        },
                        "$addToSet": {"speakers_identified": {"$each": result["speakers_identified"]}}
                    },
                )
                log.info(f"Speaker diarization completed for {audio_uuid} with {len(result['speakers_identified'])} speakers")
            except Exception as e:
                log.error(f"Failed to update MongoDB for {audio_uuid}: {e}")
    
    def enroll_speaker(self, speaker_id: str, speaker_name: str, audio_file: str, 
                      start_time: Optional[float] = None, end_time: Optional[float] = None) -> bool:
        """Enroll a new speaker (sync wrapper)."""
        async def _enroll():
            async with SpeakerRecognitionClient(self.client_url) as client:
                return await client.enroll_speaker(speaker_id, speaker_name, audio_file, start_time, end_time)
        
        try:
            return asyncio.run(_enroll())
        except Exception as e:
            log.error(f"Failed to enroll speaker {speaker_id}: {e}")
            return False
    
    def identify_speaker(self, embedding_or_path: Union[str, object]) -> Optional[str]:
        """Identify a speaker (sync wrapper)."""
        # This is a simplified version - in practice, you might need to handle embeddings differently
        if isinstance(embedding_or_path, str):
            audio_path = embedding_or_path
        else:
            log.warning("Direct embedding identification not supported in service mode")
            return None
            
        async def _identify():
            async with SpeakerRecognitionClient(self.client_url) as client:
                return await client.identify_speaker(audio_path)
        
        try:
            return asyncio.run(_identify())
        except Exception as e:
            log.error(f"Failed to identify speaker: {e}")
            return None
    
    def list_enrolled_speakers(self) -> List[Dict]:
        """List enrolled speakers (sync wrapper)."""
        async def _list():
            async with SpeakerRecognitionClient(self.client_url) as client:
                return await client.list_speakers()
        
        try:
            return asyncio.run(_list())
        except Exception as e:
            log.error(f"Failed to list speakers: {e}")
            return []
    
    def remove_speaker(self, speaker_id: str) -> bool:
        """Remove a speaker (sync wrapper)."""
        async def _remove():
            async with SpeakerRecognitionClient(self.client_url) as client:
                return await client.remove_speaker(speaker_id)
        
        try:
            return asyncio.run(_remove())
        except Exception as e:
            log.error(f"Failed to remove speaker {speaker_id}: {e}")
            return False
    
    def normalize_embedding(self, embedding):
        """Placeholder for embedding normalization."""
        log.warning("normalize_embedding called in service mode - this should be handled by the service")
        return embedding 