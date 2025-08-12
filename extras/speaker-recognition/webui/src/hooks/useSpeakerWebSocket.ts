/**
 * Hook for using the Speaker WebSocket service
 * Provides a simplified interface for real-time transcription with speaker detection
 */

import { useState, useCallback, useRef, useEffect } from 'react'
import { 
  SpeakerWebSocketService, 
  UtteranceBoundaryEvent, 
  SpeakerWebSocketOptions 
} from '../services/speakerWebSocket'

export interface TranscriptSegment {
  id: string
  timestamp: number
  text: string
  speaker_name: string | null
  speaker_id: string | null
  confidence: number
  duration: number
  status: 'identified' | 'unknown' | 'error'
  audio_segment: {
    start: number
    end: number
    duration: number
  }
}

export interface UseSpeakerWebSocketReturn {
  // Connection state
  isConnected: boolean
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error'
  
  // Streaming state
  isStreaming: boolean
  transcriptSegments: TranscriptSegment[]
  
  // Controls
  connect: () => Promise<void>
  disconnect: () => void
  startStreaming: () => void
  stopStreaming: () => void
  sendAudio: (audioData: ArrayBuffer | Uint8Array) => boolean
  clearTranscripts: () => void
  
  // Settings
  updateSettings: (settings: Partial<SpeakerWebSocketOptions>) => void
  
  // Statistics
  stats: {
    totalSegments: number
    identifiedSpeakers: Set<string>
    sessionDuration: number
    averageConfidence: number
  }
}

export const useSpeakerWebSocket = (
  initialOptions: SpeakerWebSocketOptions = {}
): UseSpeakerWebSocketReturn => {
  // Connection state
  const [isConnected, setIsConnected] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected')
  
  // Streaming state  
  const [isStreaming, setIsStreaming] = useState(false)
  const [transcriptSegments, setTranscriptSegments] = useState<TranscriptSegment[]>([])
  
  // Statistics
  const [stats, setStats] = useState({
    totalSegments: 0,
    identifiedSpeakers: new Set<string>(),
    sessionDuration: 0,
    averageConfidence: 0
  })

  // Refs
  const wsServiceRef = useRef<SpeakerWebSocketService | null>(null)
  const segmentIdRef = useRef(0)
  const sessionStartRef = useRef<number>(0)

  // Session duration timer
  useEffect(() => {
    if (!isStreaming) return

    const interval = setInterval(() => {
      setStats(prev => ({
        ...prev,
        sessionDuration: Date.now() - sessionStartRef.current
      }))
    }, 1000)

    return () => clearInterval(interval)
  }, [isStreaming])

  // Handle utterance boundary events
  const handleUtteranceBoundary = useCallback((event: UtteranceBoundaryEvent) => {
    const segmentId = `segment_${segmentIdRef.current++}`
    const timestamp = Date.now()

    const segment: TranscriptSegment = {
      id: segmentId,
      timestamp,
      text: event.transcript,
      speaker_name: event.speaker_identification?.speaker_name || null,
      speaker_id: event.speaker_identification?.speaker_id || null,
      confidence: event.speaker_identification?.confidence || 0,
      duration: event.audio_segment.duration,
      status: event.speaker_identification?.status || 'unknown',
      audio_segment: event.audio_segment
    }

    // Add segment to list
    setTranscriptSegments(prev => [...prev, segment])

    // Update statistics
    setStats(prev => {
      const totalSegments = prev.totalSegments + 1
      const identifiedSpeakers = new Set(prev.identifiedSpeakers)
      
      if (segment.speaker_name && segment.status === 'identified') {
        identifiedSpeakers.add(segment.speaker_name)
      }

      // Calculate average confidence (only for segments with identification attempts)
      const segments = [...transcriptSegments, segment]
      const segmentsWithConfidence = segments.filter(s => s.confidence > 0)
      const averageConfidence = segmentsWithConfidence.length > 0 
        ? segmentsWithConfidence.reduce((sum, s) => sum + s.confidence, 0) / segmentsWithConfidence.length
        : 0

      return {
        totalSegments,
        identifiedSpeakers,
        sessionDuration: Date.now() - sessionStartRef.current,
        averageConfidence
      }
    })
  }, [transcriptSegments])

  // Initialize WebSocket service
  const initializeService = useCallback(() => {
    if (wsServiceRef.current) return

    wsServiceRef.current = new SpeakerWebSocketService({
      ...initialOptions,
      onUtteranceBoundary: handleUtteranceBoundary,
      onReady: () => {
        console.log('🟢 Speaker WebSocket ready')
      },
      onError: (error) => {
        console.error('❌ Speaker WebSocket error:', error)
      },
      onConnectionStatusChange: (status) => {
        setConnectionStatus(status)
        setIsConnected(status === 'connected')
      }
    })
  }, [initialOptions, handleUtteranceBoundary])

  // Connect to WebSocket
  const connect = useCallback(async () => {
    initializeService()
    
    if (wsServiceRef.current) {
      try {
        await wsServiceRef.current.connect()
      } catch (error) {
        console.error('Failed to connect to Speaker WebSocket:', error)
        throw error
      }
    }
  }, [initializeService])

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (wsServiceRef.current) {
      wsServiceRef.current.disconnect()
      wsServiceRef.current = null
    }
    setIsConnected(false)
    setIsStreaming(false)
    setConnectionStatus('disconnected')
  }, [])

  // Start streaming
  const startStreaming = useCallback(() => {
    if (!isConnected) {
      console.warn('Cannot start streaming: not connected')
      return
    }
    
    sessionStartRef.current = Date.now()
    setIsStreaming(true)
    setTranscriptSegments([])
    setStats({
      totalSegments: 0,
      identifiedSpeakers: new Set(),
      sessionDuration: 0,
      averageConfidence: 0
    })
    console.log('🎙️ Started streaming session')
  }, [isConnected])

  // Stop streaming
  const stopStreaming = useCallback(() => {
    setIsStreaming(false)
    console.log('⏹️ Stopped streaming session')
  }, [])

  // Send audio data
  const sendAudio = useCallback((audioData: ArrayBuffer | Uint8Array): boolean => {
    if (!wsServiceRef.current || !isStreaming) {
      return false
    }
    
    return wsServiceRef.current.sendAudio(audioData)
  }, [isStreaming])

  // Clear transcripts
  const clearTranscripts = useCallback(() => {
    setTranscriptSegments([])
    setStats({
      totalSegments: 0,
      identifiedSpeakers: new Set(),
      sessionDuration: 0,
      averageConfidence: 0
    })
    segmentIdRef.current = 0
  }, [])

  // Update settings
  const updateSettings = useCallback((settings: Partial<SpeakerWebSocketOptions>) => {
    if (wsServiceRef.current) {
      wsServiceRef.current.updateOptions(settings)
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    // Connection state
    isConnected,
    connectionStatus,
    
    // Streaming state
    isStreaming,
    transcriptSegments,
    
    // Controls
    connect,
    disconnect,
    startStreaming,
    stopStreaming,
    sendAudio,
    clearTranscripts,
    
    // Settings
    updateSettings,
    
    // Statistics
    stats
  }
}