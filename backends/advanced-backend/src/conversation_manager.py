import asyncio
import logging
import time
import uuid
import re
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Any

from easy_audio_interfaces.filesystem.filesystem_interfaces import LocalFileSink
from wyoming.audio import AudioChunk

# Import services and utilities that ConversationManager depends on
from memory import get_memory_service, shutdown_memory_service
from metrics import get_metrics_collector
from action_items_service import ActionItemsService
from routers import audio_chunks_router # For AudioChunkUtils

# Import the ClientState model
from models.client_state import ClientState

# Global instances (will be managed by ConversationManager)
active_clients: dict[str, ClientState] = {}

# Logging setup
audio_logger = logging.getLogger("audio_processing")
logger = logging.getLogger("conversation_manager")

async def create_client_state(client_id: str, audio_chunk_utils, config: dict, transcription_manager_class: Any) -> ClientState:
    """Create and register a new client state."""
    metrics_collector = get_metrics_collector() # Get metrics collector here
    client_state = ClientState(client_id, audio_chunk_utils, metrics_collector, active_clients, config, transcription_manager_class)
    active_clients[client_id] = client_state
    await start_processing(client_state) # Call the new standalone function

    # Track client connection in metrics
    metrics_collector.record_client_connection(client_id)

    return client_state

async def cleanup_client_state(client_id: str):
    """Clean up and remove client state."""
    if client_id in active_clients:
        client_state = active_clients[client_id]
        await disconnect(client_state) # Call the new standalone function
        del active_clients[client_id]

        # Track client disconnection in metrics
        get_metrics_collector().record_client_disconnection(client_id)

# --- Conversation-related functions (moved from ClientState) ---

def record_speech_start(client_state: ClientState, audio_uuid: str, timestamp: float):
    """Record the start of a speech segment."""
    client_state.current_speech_start[audio_uuid] = timestamp
    audio_logger.info(f"Recorded speech start for {audio_uuid}: {timestamp}")

def record_speech_end(client_state: ClientState, audio_uuid: str, timestamp: float):
    """Record the end of a speech segment."""
    if (
        audio_uuid in client_state.current_speech_start
        and client_state.current_speech_start[audio_uuid] is not None
    ):
        start_time = client_state.current_speech_start[audio_uuid]
        if start_time is not None:  # Type guard
            if audio_uuid not in client_state.speech_segments:
                client_state.speech_segments[audio_uuid] = []
            client_state.speech_segments[audio_uuid].append((start_time, timestamp))
            client_state.current_speech_start[audio_uuid] = None
            duration = timestamp - start_time
            audio_logger.info(
                f"Recorded speech segment for {audio_uuid}: {start_time:.3f} -> {timestamp:.3f} (duration: {duration:.3f}s)"
            )
    else:
        audio_logger.warning(
            f"Speech end recorded for {audio_uuid} but no start time found"
        )

async def start_processing(client_state: ClientState):
    """Start the processing tasks for this client."""
    client_state.saver_task = asyncio.create_task(_audio_saver(client_state))
    client_state.transcription_task = asyncio.create_task(_transcription_processor(client_state))
    client_state.memory_task = asyncio.create_task(_memory_processor(client_state))
    client_state.action_item_task = asyncio.create_task(_action_item_processor(client_state))
    audio_logger.info(f"Started processing tasks for client {client_state.client_id}")

async def disconnect(client_state: ClientState):
    """Clean disconnect of client state."""
    if not client_state.connected:
        return

    client_state.connected = False
    audio_logger.info(f"Disconnecting client {client_state.client_id}")

    # Close current conversation with all processing before signaling shutdown
    await _close_current_conversation(client_state)

    # Signal processors to stop
    await client_state.chunk_queue.put(None)
    await client_state.transcription_queue.put((None, None))
    await client_state.memory_queue.put((None, None, None))
    await client_state.action_item_queue.put((None, None, None))
    
    # Wait for tasks to complete
    if client_state.saver_task:
        await client_state.saver_task
    if client_state.transcription_task:
        await client_state.transcription_task
    if client_state.memory_task:
        await client_state.memory_task
    if client_state.action_item_task:
        await client_state.action_item_task

    # Clean up transcription manager
    if client_state.transcription_manager:
        await client_state.transcription_manager.disconnect()
        client_state.transcription_manager = None

    # Clean up any remaining speech segment tracking
    client_state.speech_segments.clear()
    client_state.current_speech_start.clear()
    client_state.conversation_transcripts.clear()  # Clear conversation transcripts

    audio_logger.info(f"Client {client_state.client_id} disconnected and cleaned up")

def _should_start_new_conversation(client_state: ClientState) -> bool:
    """Check if we should start a new conversation based on timeout."""
    if client_state.last_transcript_time is None:
        return False  # No transcript yet, keep current conversation

    current_time = time.time()
    time_since_last_transcript = current_time - client_state.last_transcript_time
    timeout_seconds = client_state.NEW_CONVERSATION_TIMEOUT_MINUTES * 60

    return time_since_last_transcript > timeout_seconds

async def _close_current_conversation(client_state: ClientState):
    """Close the current conversation with proper cleanup including audio cropping and speaker processing."""
    if client_state.file_sink:
        # Store current audio info before closing
        current_uuid = client_state.current_audio_uuid
        current_path = client_state.file_sink.file_path

        audio_logger.info(
            f"🔒 Closing conversation {current_uuid}, file: {current_path}"
        )

        # Process memory at end of conversation if we have transcripts
        if client_state.conversation_transcripts and current_uuid:
            full_conversation = " ".join(client_state.conversation_transcripts)
            audio_logger.info(
                f"💭 Processing memory for conversation {current_uuid} with {len(client_state.conversation_transcripts)} transcript segments"
            )
            audio_logger.info(
                f"💭 Individual transcripts: {client_state.conversation_transcripts}"
            )
            audio_logger.info(
                f"💭 Full conversation text: {full_conversation[:200]}..."
            )  # Log first 200 chars

            start_time = time.time()
            memories_created = []
            action_items_created = []
            processing_success = True
            error_message = None

            try:
                # Track memory storage request
                client_state.metrics_collector.record_memory_storage_request()

                # Add general memory
                memory_result = client_state.memory_service.add_memory(
                    full_conversation, client_state.client_id, current_uuid
                )
                if memory_result:
                    audio_logger.info(
                        f"✅ Successfully added conversation memory for {current_uuid}"
                    )
                    client_state.metrics_collector.record_memory_storage_result(True)

                    # Use the actual memory objects returned from mem0's add() method
                    memory_results = memory_result.get("results", [])
                    memories_created = []

                    for mem in memory_results:
                        memory_text = mem.get("memory", "Memory text unavailable")
                        memory_id = mem.get("id", "unknown")
                        event = mem.get("event", "UNKNOWN")
                        memories_created.append(
                            {"id": memory_id, "text": memory_text, "event": event}
                        )

                    audio_logger.info(
                        f"Created {len(memories_created)} memory objects: {[m['event'] for m in memories_created]}"
                    )
                else:
                    audio_logger.error(
                        f"❌ Failed to add conversation memory for {current_uuid}"
                    )
                    client_state.metrics_collector.record_memory_storage_result(False)
                    processing_success = False
                    error_message = "Failed to add general memory"

            except Exception as e:
                audio_logger.error(
                    f"❌ Error processing memory and action items for {current_uuid}: {e}"
                )
                processing_success = False
                error_message = str(e)

            # Log debug information
            processing_time_ms = (time.time() - start_time) * 1000
            # memory_debug.log_memory_processing(
            #     user_id=client_state.client_id,
            #     audio_uuid=current_uuid,
            #     transcript_text=full_conversation,
            #     memories_created=memories_created,
            #     action_items_created=action_items_created,
            #     processing_success=processing_success,
            #     error_message=error_message,
            #     processing_time_ms=processing_time_ms,
            # )
        else:
            audio_logger.info(
                f"ℹ️ No transcripts to process for memory in conversation {current_uuid}"
            )
            # Log empty processing for debug
            if current_uuid:
                pass
                # memory_debug.log_memory_processing(
                #     user_id=client_state.client_id,
                #     audio_uuid=current_uuid,
                #     transcript_text="",
                #     memories_created=[],
                #     action_items_created=[],
                #     processing_success=True,
                #     error_message="No transcripts available for processing",
                #     processing_time_ms=0,
                # )

        await client_state.file_sink.close()

        # Track successful audio chunk save in metrics
        try:
            file_path = Path(current_path)
            if file_path.exists():
                # Estimate duration (60 seconds per chunk is TARGET_SAMPLES)
                duration_seconds = client_state.OMI_SAMPLE_RATE * client_state.SEGMENT_SECONDS / client_state.OMI_SAMPLE_RATE # Corrected
                # Calculate voice activity if we have speech segments
                # if current_uuid and current_uuid in client_state.speech_segments:
                #     for start, end in client_state.speech_segments[current_uuid]:
                #         voice_activity_seconds += end - start

                client_state.metrics_collector.record_audio_chunk_saved(
                    duration_seconds, voice_activity_seconds
                )
                audio_logger.debug(
                    f"📊 Recorded audio chunk metrics: {duration_seconds}s total, {voice_activity_seconds}s voice activity"
                )
            else:
                client_state.metrics_collector.record_audio_chunk_failed()
                audio_logger.warning(
                    f"📊 Audio file not found after save: {current_path}"
                )
        except Exception as e:
            audio_logger.error(f"📊 Error recording audio metrics: {e}")

        client_state.file_sink = None

        # Process audio cropping if we have speech segments
        if current_uuid and current_path:
            if current_uuid in client_state.speech_segments:
                speech_segments = client_state.speech_segments[current_uuid]
                audio_logger.info(
                    f"🎯 Found {len(speech_segments)} speech segments for {current_uuid}: {speech_segments}"
                )
                if speech_segments:  # Only crop if we have speech segments
                    cropped_path = str(current_path).replace(".wav", "_cropped.wav")

                    # Process in background - won't block
                    asyncio.create_task(
                        client_state.audio_chunk_utils.reprocess_audio_cropping(
                            audio_uuid=current_uuid
                        )
                    )
                    audio_logger.info(
                        f"✂️ Queued audio cropping for {current_path} with {len(speech_segments)} speech segments"
                    )
                else:
                    audio_logger.info(
                        f"⚠️ Empty speech segments list found for {current_path}, skipping cropping"
                    )
                    
                # Clean up segments for this conversation
                del client_state.speech_segments[current_uuid]
                if current_uuid in client_state.current_speech_start:
                    del client_state.current_speech_start[current_uuid]
            else:
                audio_logger.info(
                    f"⚠️ No speech segments found for {current_path} (uuid: {current_uuid}), skipping cropping"
                )

    else:
        audio_logger.info(
            f"🔒 No active file sink to close for client {client_state.client_id}"
        )
 
async def start_new_conversation(client_state: ClientState):
    """Start a new conversation by closing current conversation and resetting state."""
    await _close_current_conversation(client_state)

    # Reset conversation state
    client_state.current_audio_uuid = None
    client_state.conversation_start_time = time.time()
    client_state.last_transcript_time = None
    client_state.conversation_transcripts.clear()  # Clear collected transcripts for new conversation

    audio_logger.info(
        f"Client {client_state.client_id}: Started new conversation due to {client_state.NEW_CONVERSATION_TIMEOUT_MINUTES}min timeout"
    )


async def _audio_saver(client_state: ClientState):
    """Per-client audio saver consumer."""
    try:
        while client_state.connected:
            audio_chunk = await client_state.chunk_queue.get()

            if audio_chunk is None:  # Disconnect signal
                break

            # Check if we should start a new conversation due to timeout
            if _should_start_new_conversation(client_state):
                await start_new_conversation(client_state)

            if client_state.file_sink is None:
                # Create new file sink for this client
                client_state.current_audio_uuid = uuid.uuid4().hex
                timestamp = audio_chunk.timestamp or int(time.time())
                wav_filename = (
                    f"{timestamp}_{client_state.client_id}_{client_state.current_audio_uuid}.wav"
                )
                audio_logger.info(
                    f"Creating file sink with: rate={int(client_state.OMI_SAMPLE_RATE)}, channels={int(client_state.OMI_CHANNELS)}, width={int(client_state.OMI_SAMPLE_WIDTH)}"
                )
                client_state.file_sink = LocalFileSink(f"{client_state.CHUNK_DIR}/{wav_filename}", client_state.OMI_SAMPLE_RATE, client_state.OMI_CHANNELS, client_state.OMI_SAMPLE_WIDTH)
                await client_state.file_sink.open()

                await client_state.audio_chunk_utils.chunk_repo.create_chunk(
                    audio_uuid=client_state.current_audio_uuid,
                    audio_path=wav_filename,
                    client_id=client_state.client_id,
                    timestamp=timestamp,
                )

            await client_state.file_sink.write(audio_chunk)

            # Queue for transcription
            await client_state.transcription_queue.put(
                (client_state.current_audio_uuid, audio_chunk)
            )

    except Exception as e:
        audio_logger.error(
            f"Error in audio saver for client {client_state.client_id}: {e}", exc_info=True
        )
    finally:
        # Close current conversation with all processing when audio saver ends
        await _close_current_conversation(client_state)

async def _transcription_processor(client_state: ClientState):
    """Per-client transcription processor."""
    try:
        while client_state.connected:
            audio_uuid, chunk = await client_state.transcription_queue.get()

            if audio_uuid is None or chunk is None:  # Disconnect signal
                break

            # Get or create transcription manager
            if client_state.transcription_manager is None:
                # Create callback function to queue action items
                async def action_item_callback(transcript_text, client_id, audio_uuid):
                    await client_state.action_item_queue.put((transcript_text, client_id, audio_uuid))
                
                client_state.transcription_manager = client_state.transcription_manager_class(
                    action_item_callback=action_item_callback,
                    audio_chunk_utils=client_state.audio_chunk_utils,
                    metrics_collector=client_state.metrics_collector,
                    active_clients=client_state.active_clients
                )
                try:
                    await client_state.transcription_manager.connect()
                except Exception as e:
                    audio_logger.error(
                        f"Failed to create transcription manager for client {client_state.client_id}: {e}"
                    )
                    continue

            # Process transcription
            try:
                await client_state.transcription_manager.transcribe_chunk(
                    audio_uuid, chunk, client_state.client_id
                )
            except Exception as e:
                audio_logger.error(
                    f"Error transcribing for client {client_state.client_id}: {e}"
                )
                # Recreate transcription manager on error
                if client_state.transcription_manager:
                    await client_state.transcription_manager.disconnect()
                    client_state.transcription_manager = None

    except Exception as e:
        audio_logger.error(
            f"Error in transcription processor for client {client_state.client_id}: {e}",
            exc_info=True,
        )

async def _memory_processor(client_state: ClientState):
    """Per-client memory processor - currently unused as memory processing happens at conversation end."""
    try:
        while client_state.connected:
            transcript, client_id, audio_uuid = await client_state.memory_queue.get()

            if (
                transcript is None or client_id is None or audio_uuid is None
            ):  # Disconnect signal
                break

            # Memory processing now happens at conversation end, so this is effectively a no-op
            # Keeping the processor running to avoid breaking the queue system
            audio_logger.debug(
                f"Memory processor received item but processing is now done at conversation end"
            )

    except Exception as e:
        audio_logger.error(
            f"Error in memory processor for client {client_state.client_id}: {e}",
            exc_info=True,
        )

async def _action_item_processor(client_state: ClientState):
    """
    Processes transcript segments from the per-client action item queue.

    For each transcript segment, this processor:
    - Checks if the special keyphrase 'Simon says' (case-insensitive, as a phrase) appears in the text.
      - If found, it replaces all occurrences of the keyphrase with 'Simon says' (canonical form) and extracts action items from the modified text.
      - Logs the detection and extraction process for this special case.
    - If the keyphrase is not found, it extracts action items from the original transcript text.
    - All extraction is performed using the action_items_service.
    - Logs the number of action items extracted or any errors encountered.
    """
    try:
        while client_state.connected:
            transcript_text, client_id, audio_uuid = await client_state.action_item_queue.get()
            
            if transcript_text is None or client_id is None or audio_uuid is None:  # Disconnect signal
                break
            
            # Check for the special keyphrase 'simon says' (case-insensitive, any spaces or dots)
            keyphrase_pattern = re.compile(r'\bSimon says\b', re.IGNORECASE)
            if keyphrase_pattern.search(transcript_text):
                # Remove all occurrences of the keyphrase
                modified_text = keyphrase_pattern.sub('Simon says', transcript_text)
                audio_logger.info(f"🔑 'simon says' keyphrase detected in transcript for {audio_uuid}. Extracting action items from: '{modified_text.strip()}'")
                try:
                    action_item_count = await client_state.action_items_service.extract_and_store_action_items(
                        modified_text.strip(), client_id, audio_uuid
                    )
                    if action_item_count > 0:
                        audio_logger.info(f"🎯 Extracted {action_item_count} action items from 'simon says' transcript segment for {audio_uuid}")
                    else:
                        audio_logger.debug(f"ℹ️ No action items found in 'simon says' transcript segment for {audio_uuid}")
                except Exception as e:
                    audio_logger.error(f"❌ Error processing 'simon says' action items for transcript segment in {audio_uuid}: {e}")
                continue  # Skip the normal extraction for this case
            
    except Exception as e:
        audio_logger.error(f"Error in action item processor for client {client_state.client_id}: {e}", exc_info=True)