import { useState, useEffect } from 'react'
import { MessageSquare, RefreshCw, Calendar, User, Clock } from 'lucide-react'
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

export default function Conversations() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [debugMode, setDebugMode] = useState(false)

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
                  <div className="space-y-2">
                    {conversation.transcript.map((segment, index) => (
                      <div
                        key={index}
                        className="flex items-start space-x-3 p-3 bg-white dark:bg-gray-800 rounded border"
                      >
                        <div className="flex-shrink-0 text-xs text-gray-500 dark:text-gray-400 flex items-center space-x-1">
                          <Clock className="h-3 w-3" />
                          <span>{formatDuration(segment.start, segment.end)}</span>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <span className="font-medium text-sm text-blue-600 dark:text-blue-400">
                              {segment.speaker || 'Unknown'}
                            </span>
                          </div>
                          <p className="text-sm text-gray-900 dark:text-gray-100">
                            {segment.text}
                          </p>
                        </div>
                      </div>
                    ))}
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