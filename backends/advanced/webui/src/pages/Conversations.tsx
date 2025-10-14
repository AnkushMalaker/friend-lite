import { useState, useEffect, useRef } from 'react'
import { MessageSquare, RefreshCw, Calendar, User, Play, Pause, MoreVertical, RotateCcw, Zap, ChevronDown, ChevronUp, Trash2 } from 'lucide-react'
import { conversationsApi, BACKEND_URL } from '../services/api'
import ConversationVersionHeader from '../components/ConversationVersionHeader'

interface Conversation {
  conversation_id?: string
  audio_uuid: string
  title?: string
  summary?: string
  timestamp: number
  created_at?: string
  client_id: string
  segment_count?: number  // From list endpoint
  transcript?: string  // Full text transcript (for LLM parsing)
  segments?: Array<{  // Optional - only populated after fetching details
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
  speaker_names?: { [key: string]: string }
  duration_seconds?: number
  memories?: any[]
  has_memory?: boolean
  memory_processing_status?: string
  transcription_status?: string
  action_items?: any[]
  version_info?: {
    transcript_count: number
    memory_count: number
    active_transcript_version?: string
    active_memory_version?: string
  }
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

  // Transcript expand/collapse state
  const [expandedTranscripts, setExpandedTranscripts] = useState<Set<string>>(new Set())
  // Audio playback state
  const [playingSegment, setPlayingSegment] = useState<string | null>(null) // Format: "audioUuid-segmentIndex"
  const audioRefs = useRef<{ [key: string]: HTMLAudioElement }>({})
  const segmentTimerRef = useRef<number | null>(null)

  // Reprocessing state
  const [openDropdown, setOpenDropdown] = useState<string | null>(null)
  const [reprocessingTranscript, setReprocessingTranscript] = useState<Set<string>>(new Set())
  const [reprocessingMemory, setReprocessingMemory] = useState<Set<string>>(new Set())
  const [deletingConversation, setDeletingConversation] = useState<Set<string>>(new Set())

  const loadConversations = async () => {
    try {
      setLoading(true)
      const response = await conversationsApi.getAll()
      // API now returns a flat list with client_id as a field
      const conversationsList = response.data.conversations || []
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

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = () => setOpenDropdown(null)
    document.addEventListener('click', handleClickOutside)
    return () => document.removeEventListener('click', handleClickOutside)
  }, [])

  const formatDate = (timestamp: number | string) => {
    // Handle both Unix timestamp (number) and ISO string
    if (typeof timestamp === 'string') {
      return new Date(timestamp).toLocaleString()
    }
    // If timestamp is 0, return placeholder
    if (timestamp === 0) {
      return 'Unknown date'
    }
    return new Date(timestamp * 1000).toLocaleString()
  }

  const formatDuration = (start: number, end: number) => {
    const duration = end - start
    const minutes = Math.floor(duration / 60)
    const seconds = Math.floor(duration % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  const handleReprocessTranscript = async (conversation: Conversation) => {
    try {
      if (!conversation.conversation_id) {
        setError('Cannot reprocess transcript: Conversation ID is missing. This conversation may be from an older format.')
        return
      }

      setReprocessingTranscript(prev => new Set(prev).add(conversation.conversation_id!))
      setOpenDropdown(null)

      const response = await conversationsApi.reprocessTranscript(conversation.conversation_id)

      if (response.status === 200) {
        // Refresh conversations to show updated data
        await loadConversations()
      } else {
        setError(`Failed to start transcript reprocessing: ${response.data?.error || 'Unknown error'}`)
      }
    } catch (err: any) {
      setError(`Error starting transcript reprocessing: ${err.message || 'Unknown error'}`)
    } finally {
      if (conversation.conversation_id) {
        setReprocessingTranscript(prev => {
          const newSet = new Set(prev)
          newSet.delete(conversation.conversation_id!)
          return newSet
        })
      }
    }
  }

  const handleReprocessMemory = async (conversation: Conversation, transcriptVersionId?: string) => {
    try {
      if (!conversation.conversation_id) {
        setError('Cannot reprocess memory: Conversation ID is missing. This conversation may be from an older format.')
        return
      }

      setReprocessingMemory(prev => new Set(prev).add(conversation.conversation_id!))
      setOpenDropdown(null)

      // For now, use active transcript version. In future, this could be selected from UI
      const response = await conversationsApi.reprocessMemory(conversation.conversation_id, transcriptVersionId || 'active')

      if (response.status === 200) {
        // Refresh conversations to show updated data
        await loadConversations()
      } else {
        setError(`Failed to start memory reprocessing: ${response.data?.error || 'Unknown error'}`)
      }
    } catch (err: any) {
      setError(`Error starting memory reprocessing: ${err.message || 'Unknown error'}`)
    } finally {
      if (conversation.conversation_id) {
        setReprocessingMemory(prev => {
          const newSet = new Set(prev)
          newSet.delete(conversation.conversation_id!)
          return newSet
        })
      }
    }
  }

  const handleDeleteConversation = async (audioUuid: string) => {
    try {
      const confirmed = window.confirm('Are you sure you want to delete this conversation? This action cannot be undone.')
      if (!confirmed) return

      setDeletingConversation(prev => new Set(prev).add(audioUuid))
      setOpenDropdown(null)

      const response = await conversationsApi.delete(audioUuid)

      if (response.status === 200) {
        // Refresh conversations to show updated data
        await loadConversations()
      } else {
        setError(`Failed to delete conversation: ${response.data?.error || 'Unknown error'}`)
      }
    } catch (err: any) {
      setError(`Error deleting conversation: ${err.message || 'Unknown error'}`)
    } finally {
      setDeletingConversation(prev => {
        const newSet = new Set(prev)
        newSet.delete(audioUuid)
        return newSet
      })
    }
  }

  const toggleTranscriptExpansion = async (audioUuid: string) => {
    // If already expanded, just collapse
    if (expandedTranscripts.has(audioUuid)) {
      setExpandedTranscripts(prev => {
        const newSet = new Set(prev)
        newSet.delete(audioUuid)
        return newSet
      })
      return
    }

    // Find the conversation by audio_uuid
    const conversation = conversations.find(c => c.audio_uuid === audioUuid)
    if (!conversation || !conversation.conversation_id) {
      console.error('Cannot expand transcript: conversation_id missing')
      return
    }

    // If segments are already loaded, just expand
    if (conversation.segments && conversation.segments.length > 0) {
      setExpandedTranscripts(prev => new Set(prev).add(audioUuid))
      return
    }

    // Fetch full conversation details including segments
    try {
      const response = await conversationsApi.getById(conversation.conversation_id)
      if (response.status === 200 && response.data.conversation) {
        // Update the conversation in state with full segments and transcript
        setConversations(prev => prev.map(c =>
          c.audio_uuid === audioUuid
            ? { ...c, segments: response.data.conversation.segments, transcript: response.data.conversation.transcript }
            : c
        ))
        // Expand the transcript
        setExpandedTranscripts(prev => new Set(prev).add(audioUuid))
      }
    } catch (err: any) {
      console.error('Failed to fetch conversation details:', err)
      setError(`Failed to load transcript: ${err.message || 'Unknown error'}`)
    }
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
              {/* Version Selector Header - Only show for conversations with conversation_id */}
              {conversation.conversation_id && (
                <ConversationVersionHeader
                  conversationId={conversation.conversation_id}
                  versionInfo={conversation.version_info}
                  onVersionChange={async () => {
                    // Update only this specific conversation without reloading all conversations
                    // This prevents page scroll jump
                    try {
                      const response = await conversationsApi.getById(conversation.conversation_id!)
                      if (response.status === 200 && response.data.conversation) {
                        setConversations(prev => prev.map(c =>
                          c.conversation_id === conversation.conversation_id
                            ? { ...c, ...response.data.conversation }
                            : c
                        ))
                      }
                    } catch (err: any) {
                      console.error('Failed to refresh conversation:', err)
                      // Fallback to full reload on error
                      loadConversations()
                    }
                  }}
                />
              )}
              
              {/* Conversation Header */}
              <div className="flex justify-between items-start mb-4">
                <div className="flex flex-col space-y-2">
                  {/* Conversation Title */}
                  <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                    {conversation.title || "Conversation"}
                  </h2>

                  {/* Summary */}
                  {conversation.summary && (
                    <p className="text-sm text-gray-600 dark:text-gray-400 italic">
                      {conversation.summary}
                    </p>
                  )}

                  {/* Metadata */}
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
                      <Calendar className="h-4 w-4" />
                      <span>{formatDate(conversation.created_at || conversation.timestamp)}</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
                      <User className="h-4 w-4" />
                      <span>{conversation.client_id}</span>
                    </div>
                    {conversation.duration_seconds && conversation.duration_seconds > 0 && (
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        Duration: {Math.floor(conversation.duration_seconds / 60)}:{(conversation.duration_seconds % 60).toFixed(0).padStart(2, '0')}
                      </div>
                    )}
                  </div>
                </div>

                {/* Hamburger Menu */}
                <div className="relative">
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      setOpenDropdown(openDropdown === conversation.audio_uuid ? null : conversation.audio_uuid)
                    }}
                    className="p-1 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                    title="Conversation options"
                  >
                    <MoreVertical className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                  </button>

                  {/* Dropdown Menu */}
                  {openDropdown === conversation.audio_uuid && (
                    <div className="absolute right-0 top-8 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-600 py-2 z-10">
                      <button
                        onClick={() => handleReprocessTranscript(conversation)}
                        disabled={!conversation.conversation_id || reprocessingTranscript.has(conversation.conversation_id)}
                        className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {conversation.conversation_id && reprocessingTranscript.has(conversation.conversation_id) ? (
                          <RefreshCw className="h-4 w-4 animate-spin" />
                        ) : (
                          <RotateCcw className="h-4 w-4" />
                        )}
                        <span>Reprocess Transcript</span>
                        {!conversation.conversation_id && (
                          <span className="text-xs text-red-500 ml-1">(ID missing)</span>
                        )}
                      </button>
                      <button
                        onClick={() => handleReprocessMemory(conversation)}
                        disabled={!conversation.conversation_id || reprocessingMemory.has(conversation.conversation_id)}
                        className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {conversation.conversation_id && reprocessingMemory.has(conversation.conversation_id) ? (
                          <RefreshCw className="h-4 w-4 animate-spin" />
                        ) : (
                          <Zap className="h-4 w-4" />
                        )}
                        <span>Reprocess Memory</span>
                        {!conversation.conversation_id && (
                          <span className="text-xs text-red-500 ml-1">(ID missing)</span>
                        )}
                      </button>
                      <div className="border-t border-gray-200 dark:border-gray-600 my-1"></div>
                      <button
                        onClick={() => handleDeleteConversation(conversation.audio_uuid)}
                        disabled={deletingConversation.has(conversation.audio_uuid)}
                        className="w-full text-left px-4 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {deletingConversation.has(conversation.audio_uuid) ? (
                          <RefreshCw className="h-4 w-4 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4" />
                        )}
                        <span>Delete Conversation</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Audio Player */}
              <div className="mb-4">
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
                {/* Transcript Header with Expand/Collapse */}
                <div
                  className="flex items-center justify-between cursor-pointer p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                  onClick={() => toggleTranscriptExpansion(conversation.audio_uuid)}
                >
                  <h3 className="font-medium text-gray-900 dark:text-gray-100">
                    Transcript {((conversation.segments && conversation.segments.length > 0) || conversation.segment_count) && (
                      <span className="text-sm text-gray-500 dark:text-gray-400 ml-1">
                        ({conversation.segments?.length || conversation.segment_count || 0} segments)
                      </span>
                    )}
                  </h3>
                  <div className="flex items-center space-x-2">
                    {expandedTranscripts.has(conversation.audio_uuid) ? (
                      <ChevronUp className="h-5 w-5 text-gray-500 dark:text-gray-400 transition-transform duration-200" />
                    ) : (
                      <ChevronDown className="h-5 w-5 text-gray-500 dark:text-gray-400 transition-transform duration-200" />
                    )}
                  </div>
                </div>

                {/* Transcript Content - Conditionally Rendered */}
                {expandedTranscripts.has(conversation.audio_uuid) && (
                  <div className="animate-in slide-in-from-top-2 duration-300 ease-out space-y-4">
                    {conversation.segments && conversation.segments.length > 0 ? (
                      <div className="p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-600">
                        <div className="space-y-1">
                      {(() => {
                        // Build a speaker-to-color map for this conversation
                        const speakerColorMap: { [key: string]: string } = {};
                        let colorIndex = 0;

                        // First pass: assign colors to unique speakers
                        conversation.segments.forEach(segment => {
                          const speaker = segment.speaker || 'Unknown';
                          if (!speakerColorMap[speaker]) {
                            speakerColorMap[speaker] = SPEAKER_COLOR_PALETTE[colorIndex % SPEAKER_COLOR_PALETTE.length];
                            colorIndex++;
                          }
                        });

                        // Render the transcript
                        return conversation.segments.map((segment, index) => {
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
                                  <span className="text-xs text-gray-400 mr-2">
                                    [start: {segment.start.toFixed(1)}s, end: {segment.end.toFixed(1)}s, duration: {formatDuration(segment.start, segment.end)}]
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
                      <div className="text-sm text-gray-500 dark:text-gray-400 italic p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-600">
                        No transcript available
                      </div>
                    )}
                    
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
                    <div>Conversation ID: {conversation.conversation_id || 'N/A'}</div>
                    <div>Audio UUID: {conversation.audio_uuid}</div>
                    <div>Original Audio: {conversation.audio_path || 'N/A'}</div>
                    <div>Cropped Audio: {conversation.cropped_audio_path || 'N/A'}</div>
                    <div>Transcription Status: {conversation.transcription_status || 'N/A'}</div>
                    <div>Memory Processing Status: {conversation.memory_processing_status || 'N/A'}</div>
                    <div>Transcript Segments: {conversation.segments?.length || 0}</div>
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