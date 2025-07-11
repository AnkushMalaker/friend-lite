"""
WebSocket handling and client state management module.
Handles WebSocket connections, client state management, and real-time audio processing.
"""

import asyncio
import concurrent.futures
import json
import logging
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import WebSocket, WebSocketDisconnect
from omi.decoder import OmiOpusDecoder
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.vad import VoiceStarted, VoiceStopped

from audio_processing import (
    AUDIO_CROPPING_ENABLED,
    CHUNK_DIR,
    NEW_CONVERSATION_TIMEOUT_MINUTES,
    TARGET_SAMPLES,
    ChunkRepo,
    TranscriptionManager,
    _new_local_file_sink,
    _process_audio_cropping_with_relative_timestamps,
)
from services import (
    get_audio_config,
    get_database,
    get_memory_service_instance,
    get_transcript_service_manager,
)
from users import get_user_by_client_id

# Set up logging
audio_logger = logging.getLogger("audio")
websocket_logger = logging.getLogger("websocket")

# Thread pool executor for blocking operations
_DEC_IO_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="opus_io",
)

# Audio configuration
audio_config = get_audio_config()

# Global state for client management
active_clients: Dict[str, "ClientState"] = {}
client_user_mapping: Dict[str, str] = {}
user_clients: Dict[str, List[str]] = defaultdict(list)

def register_client_user_mapping(client_id: str, user_id: str):
    """Register the mapping between client_id and user_id."""
    client_user_mapping[client_id] = user_id
    if client_id not in user_clients[user_id]:
        user_clients[user_id].append(client_id)

def unregister_client_user_mapping(client_id: str):
    """Unregister the mapping between client_id and user_id."""
    if client_id in client_user_mapping:
        user_id = client_user_mapping[client_id]
        if client_id in user_clients[user_id]:
            user_clients[user_id].remove(client_id)
        del client_user_mapping[client_id]

def get_user_clients(user_id: str) -> List[str]:
    """Get all client IDs for a user."""
    return user_clients.get(user_id, [])

def track_client_user_relationship(client_id: str, user_id: str):
    """Track the relationship between client and user."""
    register_client_user_mapping(client_id, user_id)

def client_belongs_to_user(client_id: str, user_id: str) -> bool:
    """Check if a client belongs to a user."""
    return client_user_mapping.get(client_id) == user_id

def get_user_clients_all(user_id: str) -> List[str]:
    """Get all client IDs for a user (including inactive ones)."""
    return user_clients.get(user_id, [])

class ClientState:
    """Manages all state for a single client connection."""

    def __init__(self, client_id: str, websocket: WebSocket):
        self.client_id = client_id
        self.websocket = websocket
        self.connected = False
        self.decoder = OmiOpusDecoder()
        
        # Audio processing state
        self.current_audio_uuid: Optional[str] = None
        self.file_sink = None
        self.sample_count = 0
        self.samples_since_last_cropping = 0
        self.transcription_manager: Optional[TranscriptionManager] = None
        
        # Speech tracking
        self.speech_segments: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.current_speech_start: Dict[str, float] = {}
        self.conversation_transcripts: List[str] = []
        
        # Processing queues
        self.chunk_queue = asyncio.Queue()
        self.transcription_queue = asyncio.Queue()
        self.memory_queue = asyncio.Queue()
        self.action_item_queue = asyncio.Queue()
        
        # Background tasks
        self.saver_task: Optional[asyncio.Task] = None
        self.transcription_task: Optional[asyncio.Task] = None
        self.memory_task: Optional[asyncio.Task] = None
        self.action_item_task: Optional[asyncio.Task] = None
        
        # Services
        self.chunk_repo = None
        self.transcript_service_manager = None
        self.memory_service = None
        
        audio_logger.info(f"Created ClientState for {client_id}")

    def record_speech_start(self, audio_uuid: str):
        """Record the start of a speech segment."""
        self.current_speech_start[audio_uuid] = time.time()

    def record_speech_end(self, audio_uuid: str):
        """Record the end of a speech segment."""
        if audio_uuid in self.current_speech_start:
            start_time = self.current_speech_start.pop(audio_uuid)
            end_time = time.time()
            self.speech_segments[audio_uuid].append((start_time, end_time))
            
            audio_logger.info(f"🎤 Speech segment recorded for {audio_uuid}: {start_time:.2f}s - {end_time:.2f}s")
            
            # Process audio cropping if enabled
            if AUDIO_CROPPING_ENABLED and len(self.speech_segments[audio_uuid]) > 0:
                # Process cropping every few speech segments to avoid overloading
                if len(self.speech_segments[audio_uuid]) % 3 == 0:
                    asyncio.create_task(self._process_audio_cropping(audio_uuid))

    async def connect(self):
        """Initialize client connection and start background tasks."""
        if self.connected:
            return
        
        self.connected = True
        
        # Initialize services
        _, _, collections = get_database()
        self.chunk_repo = ChunkRepo(collections["chunks"])
        self.transcript_service_manager = get_transcript_service_manager()
        self.memory_service = get_memory_service_instance()
        
        # Start background processing tasks
        self.saver_task = asyncio.create_task(self._audio_saver())
        self.transcription_task = asyncio.create_task(self._transcription_processor())
        self.memory_task = asyncio.create_task(self._memory_processor())
        self.action_item_task = asyncio.create_task(self._transcript_services_processor())
        
        audio_logger.info(f"Started processing tasks for client {self.client_id}")

    async def disconnect(self):
        """Clean disconnect of client state."""
        if not self.connected:
            return

        self.connected = False
        audio_logger.info(f"Disconnecting client {self.client_id}")

        # Close current conversation with all processing before signaling shutdown
        await self._close_current_conversation()

        # Signal processors to stop
        await self.chunk_queue.put(None)
        await self.transcription_queue.put((None, None))
        await self.memory_queue.put((None, None, None))
        await self.action_item_queue.put((None, None, None))

        # Wait for tasks to complete
        if self.saver_task:
            await self.saver_task
        if self.transcription_task:
            await self.transcription_task
        if self.memory_task:
            await self.memory_task
        if self.action_item_task:
            await self.action_item_task

        # Clean up transcription manager
        if self.transcription_manager:
            await self.transcription_manager.disconnect()
            self.transcription_manager = None

        # Clean up any remaining speech segment tracking
        self.speech_segments.clear()
        self.current_speech_start.clear()
        self.conversation_transcripts.clear()

        audio_logger.info(f"Client {self.client_id} disconnected and cleaned up")

    async def _close_current_conversation(self):
        """Close the current conversation and process any remaining data."""
        if self.current_audio_uuid:
            audio_logger.info(f"Closing conversation {self.current_audio_uuid}")
            
            # Close file sink
            if self.file_sink:
                await self.file_sink.close()
                self.file_sink = None
            
            # Flush transcription manager
            if self.transcription_manager:
                await self.transcription_manager.flush_final_transcript(self.current_audio_uuid)
            
            # Process any remaining speech segments
            if self.current_audio_uuid in self.speech_segments:
                await self._process_audio_cropping(self.current_audio_uuid)
            
            # Process memory for the conversation
            if self.conversation_transcripts:
                full_transcript = " ".join(self.conversation_transcripts)
                await self.memory_queue.put((self.current_audio_uuid, self.client_id, full_transcript))
            
            # Reset state
            self.current_audio_uuid = None
            self.sample_count = 0
            self.samples_since_last_cropping = 0
            self.conversation_transcripts.clear()

    async def _start_new_conversation(self):
        """Start a new conversation with a new audio UUID."""
        # Close current conversation if exists
        await self._close_current_conversation()
        
        # Generate new audio UUID and file path
        timestamp = int(time.time())
        self.current_audio_uuid = str(uuid.uuid4())
        
        file_path = CHUNK_DIR / f"{timestamp}_{self.client_id}_{self.current_audio_uuid}.wav"
        
        # Create new file sink
        self.file_sink = _new_local_file_sink(str(file_path))
        
        # Create chunk entry in database
        # assert self.chunk_repo is not None
        await self.chunk_repo.create_chunk(self.current_audio_uuid, self.client_id, str(file_path))
        
        # Reset counters
        self.sample_count = 0
        self.samples_since_last_cropping = 0
        
        audio_logger.info(f"Started new conversation: {self.current_audio_uuid}")

    async def _process_audio_cropping(self, audio_uuid: str):
        """Process audio cropping for the given audio UUID."""
        if not AUDIO_CROPPING_ENABLED:
            return
        
        if audio_uuid not in self.speech_segments or not self.speech_segments[audio_uuid]:
            return
        
        try:
            # Get original file path
            original_path = None
            if self.file_sink:
                original_path = self.file_sink.file_path
            
            if not original_path:
                audio_logger.warning(f"No original file path for audio cropping: {audio_uuid}")
                return
            
            # Create cropped file path
            cropped_path = str(original_path).replace(".wav", "_cropped.wav")
            
            # Process cropping
            success = await _process_audio_cropping_with_relative_timestamps(
                str(original_path), 
                self.speech_segments[audio_uuid], 
                cropped_path,
                audio_uuid
            )
            
            if success:
                # Update database with cropped audio info
                await self.chunk_repo.update_cropped_audio(
                    audio_uuid, cropped_path, self.speech_segments[audio_uuid]
                )
                
        except Exception as e:
            audio_logger.error(f"Error processing audio cropping for {audio_uuid}: {e}")

    async def _process_memory_background(self, audio_uuid: str, client_id: str, transcript: str):
        """Process memory in the background."""
        try:
            # Resolve client_id to user information
            user = await get_user_by_client_id(client_id)
            if not user:
                audio_logger.error(f"Could not resolve client_id {client_id} to user for memory processing")
                return
            
            # Add to memory service
            await self.memory_service.add_memory(
                client_id=client_id,
                user_id=user.user_id,
                text=transcript,
                audio_uuid=audio_uuid,
                session_id=None,
                metadata={"source": "conversation"}
            )
            
            audio_logger.info(f"Added memory for conversation {audio_uuid}")
            
        except Exception as e:
            audio_logger.error(f"Error processing memory for {audio_uuid}: {e}")

    async def _audio_saver(self):
        """Background task to save audio chunks to disk."""
        try:
            while self.connected:
                chunk = await self.chunk_queue.get()
                if chunk is None:  # Disconnect signal
                    break
                
                try:
                    # Check if we need to start a new conversation
                    if self.current_audio_uuid is None:
                        await self._start_new_conversation()
                    
                    # Decode audio
                    decoded_samples = await asyncio.get_event_loop().run_in_executor(
                        _DEC_IO_EXECUTOR, self.decoder.decode, chunk
                    )
                    
                    # Write to file
                    if self.file_sink:
                        self.file_sink.write(decoded_samples)
                        self.sample_count += len(decoded_samples)
                        self.samples_since_last_cropping += len(decoded_samples)
                    
                    # Queue for transcription
                    await self.transcription_queue.put((self.current_audio_uuid, decoded_samples))
                    
                    # Check if we need to start a new conversation (time-based)
                    if self.sample_count >= TARGET_SAMPLES:
                        await self._start_new_conversation()
                        
                except Exception as e:
                    audio_logger.error(f"Error processing audio chunk for client {self.client_id}: {e}")
                finally:
                    self.chunk_queue.task_done()
                    
        except Exception as e:
            audio_logger.error(f"Error in audio saver for client {self.client_id}: {e}")

    async def _transcription_processor(self):
        """Background task to process transcription."""
        try:
            while self.connected:
                audio_uuid, chunk_data = await self.transcription_queue.get()
                
                if audio_uuid is None or chunk_data is None:  # Disconnect signal
                    self.transcription_queue.task_done()
                    break
                
                try:
                    # Get or create transcription manager
                    if self.transcription_manager is None:
                        # Create callback function to queue transcript processing
                        async def transcript_service_callback(transcript_text, client_id, audio_uuid):
                            await self.action_item_queue.put((transcript_text, client_id, audio_uuid))

                        self.transcription_manager = TranscriptionManager(
                            action_item_callback=transcript_service_callback
                        )
                        try:
                            await self.transcription_manager.connect(self.client_id)
                        except Exception as e:
                            audio_logger.error(f"Failed to connect transcription manager: {e}")
                            continue
                    
                    # Transcribe chunk
                    transcript = await self.transcription_manager.transcribe_chunk(
                        audio_uuid, chunk_data
                    )
                    
                    if transcript:
                        # Add to conversation transcripts
                        self.conversation_transcripts.append(transcript)
                        
                        # Store in database
                        await self.chunk_repo.add_transcript_segment(audio_uuid, {
                            "text": transcript,
                            "timestamp": time.time(),
                        })
                        
                        # Add to active clients tracking
                        if self.client_id in active_clients:
                            active_clients[self.client_id].conversation_transcripts.append(transcript)
                            audio_logger.info(f"✅ Added transcript to conversation: '{transcript}' (total: {len(active_clients[self.client_id].conversation_transcripts)})")
                        else:
                            audio_logger.warning(f"⚠️ Client {self.client_id} not found in active_clients for transcript update")
                        
                        # Queue for action item processing using callback (handled in transcription manager)
                        await self.chunk_repo.add_speaker(audio_uuid, f"speaker_{self.client_id}")
                        audio_logger.info(f"📝 Added transcript segment for {audio_uuid} to DB.")
                        
                except Exception as e:
                    audio_logger.error(f"Error processing transcription for client {self.client_id}: {e}")
                finally:
                    self.transcription_queue.task_done()
                    
        except Exception as e:
            audio_logger.error(f"Error in transcription processor for client {self.client_id}: {e}")

    async def _memory_processor(self):
        """Background task to process memory."""
        try:
            while self.connected:
                audio_uuid, client_id, transcript = await self.memory_queue.get()
                
                if audio_uuid is None or client_id is None or transcript is None:  # Disconnect signal
                    self.memory_queue.task_done()
                    break
                
                try:
                    await self._process_memory_background(audio_uuid, client_id, transcript)
                except Exception as e:
                    audio_logger.error(f"Error processing memory for client {self.client_id}: {e}")
                finally:
                    self.memory_queue.task_done()
                    
        except Exception as e:
            audio_logger.error(f"Error in memory processor for client {self.client_id}: {e}")

    async def _transcript_services_processor(self):
        """
        Processes transcript segments from the per-client transcript services queue.
        
        This processor handles queue management and delegates processing
        to all registered transcript services through the service manager.
        """
        try:
            while self.connected:
                transcript_text, client_id, audio_uuid = await self.action_item_queue.get()

                if (
                    transcript_text is None or client_id is None or audio_uuid is None
                ):  # Disconnect signal
                    self.action_item_queue.task_done()
                    break

                try:
                    # Resolve client_id to user information
                    user = await get_user_by_client_id(client_id)
                    if user:
                        # Process transcript through all registered services
                        results = await self.transcript_service_manager.process_transcript(
                            transcript_text, client_id, audio_uuid, user.user_id, user.email
                        )
                        
                        # Log results from all services
                        total_items = 0
                        for service_name, result in results.items():
                            if result.get("success", False):
                                count = result.get("count", 0)
                                total_items += count
                                if count > 0:
                                    audio_logger.info(
                                        f"🎯 {service_name} service: {count} items processed for {audio_uuid}"
                                    )
                                else:
                                    audio_logger.debug(
                                        f"ℹ️ {service_name} service: no items found for {audio_uuid}"
                                    )
                            else:
                                audio_logger.error(
                                    f"❌ {service_name} service failed: {result.get('message', 'Unknown error')}"
                                )
                        
                        if total_items > 0:
                            audio_logger.info(
                                f"🎯 Total items processed: {total_items} for {audio_uuid}"
                            )
                    else:
                        audio_logger.error(f"Could not resolve client_id {client_id} to user for transcript processing")
                        
                except Exception as e:
                    audio_logger.error(f"Error processing transcript for client {self.client_id}: {e}")
                finally:
                    # Always mark task as done
                    self.action_item_queue.task_done()

        except Exception as e:
            audio_logger.error(
                f"Error in transcript services processor for client {self.client_id}: {e}",
                exc_info=True,
            )

async def handle_websocket_connection(websocket: WebSocket, client_id: str):
    """Handle WebSocket connection for audio streaming."""
    client_state = ClientState(client_id, websocket)
    active_clients[client_id] = client_state
    
    try:
        await client_state.connect()
        websocket_logger.info(f"WebSocket connected for client {client_id}")
        
        while True:
            try:
                # Receive data from WebSocket
                data = await websocket.receive_bytes()
                
                # Queue audio chunk for processing
                await client_state.chunk_queue.put(data)
                
            except WebSocketDisconnect:
                websocket_logger.info(f"WebSocket disconnected for client {client_id}")
                break
            except Exception as e:
                websocket_logger.error(f"Error in WebSocket connection for client {client_id}: {e}")
                break
                
    except Exception as e:
        websocket_logger.error(f"Error handling WebSocket connection for client {client_id}: {e}")
    finally:
        # Clean up client state
        await client_state.disconnect()
        if client_id in active_clients:
            del active_clients[client_id]
        unregister_client_user_mapping(client_id)

async def handle_pcm_websocket_connection(websocket: WebSocket, client_id: str):
    """Handle WebSocket connection for PCM audio streaming."""
    client_state = ClientState(client_id, websocket)
    active_clients[client_id] = client_state
    
    try:
        await client_state.connect()
        websocket_logger.info(f"PCM WebSocket connected for client {client_id}")
        
        while True:
            try:
                # Receive JSON message
                message = await websocket.receive_text()
                data = json.loads(message)
                
                message_type = data.get("type")
                
                if message_type == "audio_chunk":
                    # Handle audio chunk
                    audio_data = bytes(data.get("audio", []))
                    await client_state.chunk_queue.put(audio_data)
                    
                elif message_type == "voice_started":
                    # Handle voice activity detection
                    if client_state.current_audio_uuid:
                        client_state.record_speech_start(client_state.current_audio_uuid)
                        
                elif message_type == "voice_stopped":
                    # Handle voice activity detection
                    if client_state.current_audio_uuid:
                        client_state.record_speech_end(client_state.current_audio_uuid)
                        
            except WebSocketDisconnect:
                websocket_logger.info(f"PCM WebSocket disconnected for client {client_id}")
                break
            except Exception as e:
                websocket_logger.error(f"Error in PCM WebSocket connection for client {client_id}: {e}")
                break
                
    except Exception as e:
        websocket_logger.error(f"Error handling PCM WebSocket connection for client {client_id}: {e}")
    finally:
        # Clean up client state
        await client_state.disconnect()
        if client_id in active_clients:
            del active_clients[client_id]
        unregister_client_user_mapping(client_id)

def get_active_clients() -> Dict[str, ClientState]:
    """Get all active clients."""
    return active_clients.copy()

def get_client_state(client_id: str) -> Optional[ClientState]:
    """Get client state by ID."""
    return active_clients.get(client_id)

def get_user_active_clients(user_id: str) -> List[ClientState]:
    """Get all active clients for a user."""
    user_client_ids = get_user_clients(user_id)
    return [active_clients[client_id] for client_id in user_client_ids if client_id in active_clients]