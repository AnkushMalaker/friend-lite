
@app.get("/api/debug/speech_segments")
async def debug_speech_segments(current_user: User = Depends(current_active_user)):
    """Debug endpoint to check current speech segments. Admins see all clients, users see only their own."""
    filtered_clients = {}

    for client_id, client_state in active_clients.items():
        # Filter clients based on user permissions
        if not current_user.is_superuser:
            # Regular users can only see clients that belong to them
            if not client_belongs_to_user(client_id, current_user.user_id):
                continue

        filtered_clients[client_id] = {
            "current_audio_uuid": client_state.current_audio_uuid,
            "speech_segments": {
                uuid: segments for uuid, segments in client_state.speech_segments.items()
            },
            "current_speech_start": dict(client_state.current_speech_start),
            "connected": client_state.connected,
            "last_transcript_time": client_state.last_transcript_time,
        }

    debug_info = {
        "active_clients": len(filtered_clients),
        "audio_cropping_enabled": AUDIO_CROPPING_ENABLED,
        "min_speech_duration": MIN_SPEECH_SEGMENT_DURATION,
        "cropping_padding": CROPPING_CONTEXT_PADDING,
        "clients": filtered_clients,
    }

    return JSONResponse(content=debug_info)


@app.get("/api/debug/audio-cropping")
async def get_audio_cropping_debug(current_user: User = Depends(current_superuser)):
    """Get detailed debug information about the audio cropping system."""
    # Get speech segments for all active clients
    speech_segments_info = {}
    for client_id, client_state in active_clients.items():
        if client_state.connected:
            speech_segments_info[client_id] = {
                "current_audio_uuid": client_state.current_audio_uuid,
                "speech_segments": dict(client_state.speech_segments),
                "current_speech_start": dict(client_state.current_speech_start),
                "total_segments": sum(
                    len(segments) for segments in client_state.speech_segments.values()
                ),
            }

    # Get recent audio chunks with cropping status
    recent_chunks = []
    try:
        cursor = chunks_col.find().sort("timestamp", -1).limit(10)
        async for chunk in cursor:
            recent_chunks.append(
                {
                    "audio_uuid": chunk["audio_uuid"],
                    "timestamp": chunk["timestamp"],
                    "client_id": chunk["client_id"],
                    "audio_path": chunk["audio_path"],
                    "has_cropped_version": bool(chunk.get("cropped_audio_path")),
                    "cropped_audio_path": chunk.get("cropped_audio_path"),
                    "speech_segments_count": len(chunk.get("speech_segments", [])),
                    "cropped_duration": chunk.get("cropped_duration"),
                }
            )
    except Exception as e:
        audio_logger.error(f"Error getting recent chunks: {e}")
        recent_chunks = []

    return JSONResponse(
        content={
            "timestamp": time.time(),
            "audio_cropping_config": {
                "enabled": AUDIO_CROPPING_ENABLED,
                "min_speech_duration": MIN_SPEECH_SEGMENT_DURATION,
                "cropping_padding": CROPPING_CONTEXT_PADDING,
            },
            "asr_config": {
                "use_deepgram": USE_DEEPGRAM,
                "offline_asr_uri": OFFLINE_ASR_TCP_URI,
                "deepgram_available": DEEPGRAM_AVAILABLE,
            },
            "active_clients_speech_segments": speech_segments_info,
            "recent_audio_chunks": recent_chunks,
        }
    )

