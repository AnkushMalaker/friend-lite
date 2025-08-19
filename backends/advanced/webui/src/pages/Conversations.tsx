import { useState, useEffect, useRef } from 'react'
import { MessageSquare, RefreshCw, Calendar, User, Play, Pause } from 'lucide-react'
import { conversationsApi, BACKEND_URL } from '../services/api'

interface Conversation {
  audio_uuid: string
  timestamp: number
  client_id: string
  transcript: Array<{
    text: string
    speaker: string
    start: number
    end: number
    speaker_id?: string
    confidence?: number
  }>
  audio_path?: string
  cropped_audio_path?: string
  speakers_identified?: string[]
  transcription_status?: string
  memory_processing_status?: string
}

// Speaker color palette for consistent colors across conversations
const SPEAKER_COLOR_PALETTE = [
  'text-blue-600 dark:text-blue-400',
  'text-green-600 dark:text-green-400',
  'text-purple-600 dark:text-purple-400',
  'text-orange-600 dark:text-orange-400',
  'text-pink-600 dark:text-pink-400',
  'text-indigo-600 dark:text-indigo-400',
  'text-red-600 dark:text-red-400',
  'text-yellow-600 dark:text-yellow-400',
  'text-teal-600 dark:text-teal-400',
  'text-cyan-600 dark:text-cyan-400',
];

export default function Conversations() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [debugMode, setDebugMode] = useState(false)
  
  // Audio playback state
  const [playingSegment, setPlayingSegment] = useState<string | null>(null) // Format: "audioUuid-segmentIndex"
  const audioRefs = useRef<{ [key: string]: HTMLAudioElement }>({})
  const segmentTimerRef = useRef<number | null>(null)

  const loadConversations = async () => {
    try {
      setLoading(true)
      const response = await conversationsApi.getAll()
      // Convert the conversations object to an array
      const conversationsData = response.data.conversations || {}
      const conversationsList = Object.entries(conversationsData).flatMap(([clientId, convs]: [string, any]) =>
        convs.map((conv: any) => ({ ...conv, client_id: clientId }))
      )
      setConversations(conversationsList)
      setError(null)
    } catch (err: any) {
      setError(err.message || 'Failed to load conversations')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadConversations()
  }, [])

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString()
  }

  const formatDuration = (start: number, end: number) => {
    const duration = end - start
    const minutes = Math.floor(duration / 60)
    const seconds = Math.floor(duration % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  const handleSegmentPlayPause = (audioUuid: string, segmentIndex: number, segment: any, audioPath: string) => {
    const segmentId = `${audioUuid}-${segmentIndex}`;
    
    // If this segment is already playing, pause it
    if (playingSegment === segmentId) {
      const audio = audioRefs.current[audioUuid];
      if (audio) {
        audio.pause();
      }
      if (segmentTimerRef.current) {
        window.clearTimeout(segmentTimerRef.current);
        segmentTimerRef.current = null;
      }
      setPlayingSegment(null);
      return;
    }
    
    // Stop any currently playing segment
    if (playingSegment) {
      const [currentAudioUuid] = playingSegment.split('-');
      const currentAudio = audioRefs.current[currentAudioUuid];
      if (currentAudio) {
        currentAudio.pause();
      }
      if (segmentTimerRef.current) {
        window.clearTimeout(segmentTimerRef.current);
        segmentTimerRef.current = null;
      }
    }
    
    // Get or create audio element for this conversation
    let audio = audioRefs.current[audioUuid];
    if (!audio) {
      audio = new Audio(`${BACKEND_URL}/audio/${audioPath}`);
      audioRefs.current[audioUuid] = audio;
      
      // Add event listener to handle when audio ends naturally
      audio.addEventListener('ended', () => {
        setPlayingSegment(null);
      });
    }
    
    // Set the start time and play
    audio.currentTime = segment.start;
    audio.play().then(() => {
      setPlayingSegment(segmentId);
      
      // Set a timer to stop at the segment end time
      const duration = (segment.end - segment.start) * 1000; // Convert to milliseconds
      segmentTimerRef.current = window.setTimeout(() => {
        audio.pause();
        setPlayingSegment(null);
        segmentTimerRef.current = null;
      }, duration);
    }).catch(err => {
      console.error('Error playing audio segment:', err);
      setPlayingSegment(null);
    });
  }

  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      // Stop all audio and clear timers
      Object.values(audioRefs.current).forEach(audio => {
        audio.pause();
      });
      if (segmentTimerRef.current) {
        window.clearTimeout(segmentTimerRef.current);
      }
    };
  }, [])


  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600 dark:text-gray-400">Loading conversations...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center">
        <div className="text-red-600 dark:text-red-400 mb-4">{error}</div>
        <button
          onClick={loadConversations}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Try Again
        </button>
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center space-x-2">
          <MessageSquare className="h-6 w-6 text-blue-600" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Latest Conversations
          </h1>
        </div>
        <div className="flex items-center space-x-4">
          <label className="flex items-center space-x-2 text-sm">
            <input
              type="checkbox"
              checked={debugMode}
              onChange={(e) => setDebugMode(e.target.checked)}
              className="rounded border-gray-300"
            />
            <span className="text-gray-700 dark:text-gray-300">Debug Mode</span>
          </label>
          <button
            onClick={loadConversations}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Conversations List */}
      <div className="space-y-6">
        {conversations.length === 0 ? (
          <div className="text-center text-gray-500 dark:text-gray-400 py-12">
            <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No conversations found</p>
          </div>
        ) : (
          conversations.map((conversation) => (
            <div
              key={conversation.audio_uuid}
              className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6 border border-gray-200 dark:border-gray-600"
            >
              {/* Conversation Header */}
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
                    <Calendar className="h-4 w-4" />
                    <span>{formatDate(conversation.timestamp)}</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
                    <User className="h-4 w-4" />
                    <span>{conversation.client_id}</span>
                  </div>
                </div>
                
{/* Audio Player */}
                <div className="space-y-2">
                  {(conversation.audio_path || conversation.cropped_audio_path) && (
                    <>
                      <div className="flex items-center space-x-2 text-sm text-gray-700 dark:text-gray-300">
                        <span className="font-medium">
                          {debugMode ? '🔧 Original Audio' : '🎵 Audio'}
                          {debugMode && conversation.cropped_audio_path && ' (Debug Mode)'}
                        </span>
                      </div>
                      <audio 
                        controls 
                        className="w-full h-10" 
                        preload="metadata"
                        style={{ minWidth: '300px' }}
                        src={`${BACKEND_URL}/audio/${
                          debugMode 
                            ? conversation.audio_path 
                            : conversation.cropped_audio_path || conversation.audio_path
                        }`}
                      >
                        Your browser does not support the audio element.
                      </audio>
                      {debugMode && conversation.cropped_audio_path && (
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          💡 Cropped version available: {conversation.cropped_audio_path}
                        </div>
                      )}
                      {!debugMode && conversation.cropped_audio_path && (
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          💡 Enable debug mode to hear original with silence
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>

              {/* Transcript */}
              <div className="space-y-2">
                <h3 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Transcript:</h3>
                {conversation.transcript && conversation.transcript.length > 0 ? (
                  <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-600">
                    <div className="space-y-1">
                      {(() => {
                        // Build a speaker-to-color map for this conversation
                        const speakerColorMap: { [key: string]: string } = {};
                        let colorIndex = 0;
                        
                        // First pass: assign colors to unique speakers
                        conversation.transcript.forEach(segment => {
                          const speaker = segment.speaker || 'Unknown';
                          if (!speakerColorMap[speaker]) {
                            speakerColorMap[speaker] = SPEAKER_COLOR_PALETTE[colorIndex % SPEAKER_COLOR_PALETTE.length];
                            colorIndex++;
                          }
                        });
                        
                        // Render the transcript
                        return conversation.transcript.map((segment, index) => {
                          const speaker = segment.speaker || 'Unknown';
                          const speakerColor = speakerColorMap[speaker];
                          const segmentId = `${conversation.audio_uuid}-${index}`;
                          const isPlaying = playingSegment === segmentId;
                          const audioPath = debugMode 
                            ? conversation.audio_path 
                            : conversation.cropped_audio_path || conversation.audio_path;
                          
                          return (
                            <div 
                              key={index} 
                              className={`text-sm leading-relaxed flex items-start space-x-2 py-1 px-2 rounded transition-colors ${
                                isPlaying ? 'bg-blue-50 dark:bg-blue-900/20' : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                              }`}
                            >
                              {/* Play/Pause Button */}
                              {audioPath && (
                                <button
                                  onClick={() => handleSegmentPlayPause(conversation.audio_uuid, index, segment, audioPath)}
                                  className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center transition-colors mt-0.5 ${
                                    isPlaying 
                                      ? 'bg-blue-600 text-white hover:bg-blue-700' 
                                      : 'bg-gray-200 dark:bg-gray-600 text-gray-600 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-500'
                                  }`}
                                  title={isPlaying ? 'Pause segment' : 'Play segment'}
                                >
                                  {isPlaying ? (
                                    <Pause className="w-2.5 h-2.5" />
                                  ) : (
                                    <Play className="w-2.5 h-2.5 ml-0.5" />
                                  )}
                                </button>
                              )}
                              
                              <div className="flex-1 min-w-0">
                                {debugMode && (
                                  <span className="text-xs text-gray-400 mr-2" title={`${segment.start.toFixed(1)}s - ${segment.end.toFixed(1)}s`}>
                                    [{formatDuration(segment.start, segment.end)}]
                                  </span>
                                )}
                                <span className={`font-medium ${speakerColor}`}>
                                  {speaker}:
                                </span>
                                <span className="text-gray-900 dark:text-gray-100 ml-1">
                                  {segment.text}
                                </span>
                              </div>
                            </div>
                          );
                        });
                      })()}
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-gray-500 dark:text-gray-400 italic">
                    No transcript available
                  </div>
                )}
              </div>

              {/* Speaker Information */}
              {conversation.speakers_identified && conversation.speakers_identified.length > 0 && (
                <div className="mt-4">
                  <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">🎤 Identified Speakers:</h4>
                  <div className="flex flex-wrap gap-2">
                    {conversation.speakers_identified.map((speaker, index) => (
                      <span 
                        key={index}
                        className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded-md text-sm"
                      >
                        {speaker}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Debug info */}
              {debugMode && (
                <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
                  <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">🔧 Debug Info:</h4>
                  <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                    <div>Audio UUID: {conversation.audio_uuid}</div>
                    <div>Original Audio: {conversation.audio_path || 'N/A'}</div>
                    <div>Cropped Audio: {conversation.cropped_audio_path || 'N/A'}</div>
                    <div>Transcription Status: {conversation.transcription_status || 'N/A'}</div>
                    <div>Memory Processing Status: {conversation.memory_processing_status || 'N/A'}</div>
                    <div>Transcript Segments: {conversation.transcript?.length || 0}</div>
                    <div>Client ID: {conversation.client_id}</div>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  )
}