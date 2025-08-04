import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Mic, Users, AlertCircle, Volume2, Clock } from 'lucide-react'
import { useUser } from '../contexts/UserContext'
import { apiService } from '../services/api'
import { 
  DeepgramStreaming, 
  StreamingConfig, 
  StreamingTranscript, 
  convertAudioForDeepgram,
  DEFAULT_DEEPGRAM_OPTIONS 
} from '../services/deepgram'

interface TranscriptSegment {
  id: string
  timestamp: number
  speaker: number
  text: string
  confidence: number
  isInterim: boolean
  identifiedSpeaker?: {
    id: string
    name: string
    confidence: number
  }
}

interface SpeakerIdentificationResult {
  found: boolean
  speaker_info?: {
    id: string
    name: string
  }
  confidence: number
}

interface LiveStats {
  totalWords: number
  averageConfidence: number
  identifiedSpeakers: Set<string>
  sessionDuration: number
}

export default function InferLive() {
  const { user } = useUser()
  const [isRecording, setIsRecording] = useState(false)
  const [transcriptSegments, setTranscriptSegments] = useState<TranscriptSegment[]>([])
  const [currentSpeakers, setCurrentSpeakers] = useState<Map<number, string>>(new Map())
  const [stats, setStats] = useState<LiveStats>({
    totalWords: 0,
    averageConfidence: 0,
    identifiedSpeakers: new Set(),
    sessionDuration: 0
  })
  
  // Settings
  const [deepgramApiKey, setDeepgramApiKey] = useState('')
  const [apiKeySource, setApiKeySource] = useState<'server' | 'manual' | 'loading'>('loading')
  const [enableSpeakerIdentification, setEnableSpeakerIdentification] = useState(true)
  
  // Status
  const [deepgramStatus, setDeepgramStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected')
  const [error, setError] = useState<string | null>(null)
  
  // Refs
  const deepgramRef = useRef<DeepgramStreaming | null>(null)
  const sessionStartRef = useRef<number>(0)
  const segmentIdRef = useRef(0)
  const transcriptEndRef = useRef<HTMLDivElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const apiKeyFetchedRef = useRef(false)

  // Fetch Deepgram API key from server on component mount
  useEffect(() => {
    const fetchApiKey = async () => {
      if (apiKeyFetchedRef.current) return
      apiKeyFetchedRef.current = true
      
      console.log('🎙️ [API_KEY] Fetching Deepgram API key from server...')
      try {
        const response = await apiService.get('/deepgram/config')
        console.log('🎙️ [API_KEY] ✅ Server API key retrieved successfully')
        setDeepgramApiKey(response.data.api_key)
        setApiKeySource('server')
      } catch (error) {
        console.log('🎙️ [API_KEY] ❌ Server API key not available:', error.response?.status)
        console.log('🎙️ [API_KEY] User will need to provide API key manually')
        setApiKeySource('manual')
      }
    }
    
    fetchApiKey()
  }, [])

  // Auto-scroll to bottom of transcript
  useEffect(() => {
    if (transcriptEndRef.current) {
      console.log('🎙️ [LIFECYCLE] Auto-scrolling to latest transcript segment')
      transcriptEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [transcriptSegments])

  // Update session duration
  useEffect(() => {
    if (!isRecording) {
      console.log('🎙️ [LIFECYCLE] Session duration timer stopped - not recording')
      return
    }

    console.log('🎙️ [LIFECYCLE] Starting session duration timer')
    const interval = setInterval(() => {
      setStats(prev => ({
        ...prev,
        sessionDuration: Date.now() - sessionStartRef.current
      }))
    }, 1000)

    return () => {
      console.log('🎙️ [LIFECYCLE] Cleaning up session duration timer')
      clearInterval(interval)
    }
  }, [isRecording])

  const identifySpeaker = async (audioBuffer: Float32Array, sampleRate: number): Promise<SpeakerIdentificationResult> => {
    try {
      console.log('🔍 [SPEAKER_ID] Starting speaker identification for audio segment')
      console.log('🔍 [SPEAKER_ID] Audio buffer:', audioBuffer.length, 'samples at', sampleRate, 'Hz')
      
      // Convert Float32Array to WAV blob
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)()
      const audioBufferNode = audioContext.createBuffer(1, audioBuffer.length, sampleRate)
      audioBufferNode.getChannelData(0).set(audioBuffer)
      
      // Create blob from audio buffer
      const length = audioBuffer.length
      const arrayBuffer = new ArrayBuffer(length * 2)
      const view = new DataView(arrayBuffer)
      
      // Convert to 16-bit PCM
      for (let i = 0; i < length; i++) {
        const sample = Math.max(-1, Math.min(1, audioBuffer[i]))
        view.setInt16(i * 2, sample * 0x7FFF, true)
      }
      
      // Create WAV blob
      const wavBlob = new Blob([createWAVHeader(sampleRate, 1, 16, arrayBuffer.byteLength), arrayBuffer], {
        type: 'audio/wav'
      })
      
      console.log('🔍 [SPEAKER_ID] Created WAV blob:', wavBlob.size, 'bytes')
      
      // Call the diarize-and-identify endpoint
      const formData = new FormData()
      formData.append('file', wavBlob, 'segment.wav')
      formData.append('similarity_threshold', '0.15')
      formData.append('min_duration', '1.0')
      formData.append('identify_only_enrolled', 'false')
      
      console.log('🔍 [SPEAKER_ID] Calling /diarize-and-identify endpoint...')
      const response = await apiService.post('/diarize-and-identify', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 seconds for speaker identification
      })
      
      console.log('🔍 [SPEAKER_ID] Response received:', response.data)
      
      // Process the response
      if (response.data.speakers && response.data.speakers.length > 0) {
        const firstSpeaker = response.data.speakers[0]
        return {
          found: !!firstSpeaker.identified_speaker_id,
          speaker_info: firstSpeaker.identified_speaker_id ? {
            id: firstSpeaker.identified_speaker_id,
            name: firstSpeaker.identified_speaker_name || firstSpeaker.speaker_name
          } : undefined,
          confidence: firstSpeaker.speaker_identification_confidence || firstSpeaker.confidence || 0
        }
      }
      
      return { found: false, confidence: 0 }
      
    } catch (error) {
      console.error('🔍 [SPEAKER_ID] Error during speaker identification:', error)
      return { found: false, confidence: 0 }
    }
  }

  // Helper method to create WAV header
  const createWAVHeader = (sampleRate: number, channels: number, bitsPerSample: number, dataLength: number) => {
    const buffer = new ArrayBuffer(44)
    const view = new DataView(buffer)
    
    // WAV file header
    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i))
      }
    }
    
    writeString(0, 'RIFF')
    view.setUint32(4, 36 + dataLength, true)
    writeString(8, 'WAVE')
    writeString(12, 'fmt ')
    view.setUint32(16, 16, true)
    view.setUint16(20, 1, true)
    view.setUint16(22, channels, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * channels * (bitsPerSample / 8), true)
    view.setUint16(32, channels * (bitsPerSample / 8), true)
    view.setUint16(34, bitsPerSample, true)
    writeString(36, 'data')
    view.setUint32(40, dataLength, true)
    
    return buffer
  }

  const startSession = async () => {
    console.log('🎙️ [SESSION] Starting live session...')
    
    if (!deepgramApiKey || !deepgramApiKey.trim()) {
      if (apiKeySource === 'server') {
        setError('Server Deepgram API key not configured. Please contact administrator.')
      } else {
        setError('Please provide a Deepgram API key in Settings')
      }
      return
    }

    if (!deepgramApiKey.match(/^[a-f0-9]{40}$/i)) {
      setError('Invalid Deepgram API key format. Should be a 40-character hex string.')
      return
    }

    setError(null)
    sessionStartRef.current = Date.now()
    
    try {
      // Initialize Deepgram streaming for Opus format
      const config: StreamingConfig = {
        ...DEFAULT_DEEPGRAM_OPTIONS,
        apiKey: deepgramApiKey,
        diarize: false,
        interim_results: true,
        endpointing: 300,
        vad_events: true,
        utterance_end_ms: 1000,
        // Configure for Opus since MediaRecorder uses WebM/Opus
        encoding: 'opus',
        sample_rate: 48000 // Opus typically uses 48kHz
      }

      console.log('🎙️ [SESSION] Deepgram config:', {
        encoding: config.encoding,
        sample_rate: config.sample_rate,
        model: config.model
      })

      deepgramRef.current = new DeepgramStreaming(config)
      deepgramRef.current.onStatus(setDeepgramStatus)
      deepgramRef.current.onError((error) => {
        console.error('🛑 [DEEPGRAM] Error:', error.message)
        setError(`Deepgram error: ${error.message}`)
        setDeepgramStatus('error')
      })

      deepgramRef.current.onTranscript(async (transcript: StreamingTranscript) => {
        if (!transcript.transcript.trim()) return

        const segmentId = `segment_${segmentIdRef.current++}`
        const timestamp = Date.now()

        const segment: TranscriptSegment = {
          id: segmentId,
          timestamp,
          speaker: transcript.words[0]?.speaker ?? 0,
          text: transcript.transcript,
          confidence: transcript.confidence,
          isInterim: !transcript.is_final
        }

        // Speaker identification for final transcripts
        if (transcript.is_final && enableSpeakerIdentification) {
          if (!currentSpeakers.has(segment.speaker)) {
            // For now, use basic speaker labels - actual audio segment capture would be needed for real identification
            const speakerKey = `Speaker ${segment.speaker + 1}`
            segment.identifiedSpeaker = {
              id: `speaker_${segment.speaker}`,
              name: speakerKey,
              confidence: 0.5
            }
            setCurrentSpeakers(prev => new Map(prev.set(segment.speaker, speakerKey)))
          } else {
            const knownSpeakerName = currentSpeakers.get(segment.speaker)!
            segment.identifiedSpeaker = {
              id: `speaker_${segment.speaker}`,
              name: knownSpeakerName,
              confidence: 0.9
            }
          }
        }

        // Update segments
        setTranscriptSegments(prev => {
          if (segment.isInterim) {
            const withoutLastInterim = prev.filter(s => !s.isInterim)
            return [...withoutLastInterim, segment]
          } else {
            const withoutInterim = prev.filter(s => !s.isInterim)
            return [...withoutInterim, segment]
          }
        })

        // Update stats for final transcripts
        if (transcript.is_final) {
          setStats(prev => {
            const newIdentified = new Set(prev.identifiedSpeakers)
            if (segment.identifiedSpeaker) {
              newIdentified.add(segment.identifiedSpeaker.name)
            }

            const wordCount = transcript.words.length
            const totalWords = prev.totalWords + wordCount
            const avgConfidence = totalWords > 0 ? 
              (prev.averageConfidence * prev.totalWords + transcript.confidence * wordCount) / totalWords : 
              transcript.confidence

            return {
              ...prev,
              totalWords,
              averageConfidence: avgConfidence,
              identifiedSpeakers: newIdentified
            }
          })
        }
      })

      // Connect to Deepgram
      await deepgramRef.current.connect()
      console.log('🎙️ [SESSION] Connected to Deepgram')

      // Start MediaRecorder audio capture
      await startAudioCapture()
      setIsRecording(true)
      
      console.log('🎙️ [SESSION] Live session started successfully!')

    } catch (error) {
      console.error('🎙️ [SESSION] Failed to start session:', error)
      const errorMessage = error instanceof Error ? error.message : String(error)
      setError(`Failed to start session: ${errorMessage}`)
      setIsRecording(false)
      setDeepgramStatus('error')
    }
  }

  const startAudioCapture = async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      })

      mediaStreamRef.current = stream

      // Try different formats for better Deepgram compatibility
      let mimeType = 'audio/ogg;codecs=opus'  // Try OGG/Opus first (might work better)
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/wav'  // WAV would be ideal but unlikely supported
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = 'audio/webm;codecs=opus'  // Fallback to WebM/Opus
          if (!MediaRecorder.isTypeSupported(mimeType)) {
            mimeType = 'audio/webm'
            if (!MediaRecorder.isTypeSupported(mimeType)) {
              mimeType = '' // Let browser choose
            }
          }
        }
      }
      
      console.log('🎤 [AUDIO] Selected mimeType:', mimeType)

      const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = async (event) => {
        console.log('🎤 [AUDIO] Received audio chunk:', event.data.size, 'bytes, type:', event.data.type)
        
        if (event.data.size > 0 && deepgramRef.current && deepgramRef.current.getConnectionStatus() === 'connected') {
          try {
            console.log('🎤 [AUDIO] Processing chunk - deepgram connected, format:', event.data.type)
            
            // Convert blob to ArrayBuffer
            const arrayBuffer = await event.data.arrayBuffer()
            console.log('🎤 [AUDIO] ArrayBuffer created:', arrayBuffer.byteLength, 'bytes')
            
            // Send raw WebM/Opus data directly to Deepgram (no decoding needed)
            console.log('🎤 [AUDIO] Sending raw WebM/Opus data to Deepgram (configured for opus/48kHz)...')
            
            // Deepgram is configured with encoding=opus to handle WebM/Opus directly
            deepgramRef.current.sendAudio(arrayBuffer)
            
            console.log('🎤 [AUDIO] ✅ Audio sent to Deepgram successfully')
            
          } catch (error) {
            console.error('🎤 [AUDIO] ❌ Error processing audio chunk:', error)
            console.error('🎤 [AUDIO] Error details:', {
              name: error.name,
              message: error.message,
              chunkSize: event.data.size,
              chunkType: event.data.type,
              mimeType: mimeType
            })
          }
        } else {
          const actualStatus = deepgramRef.current?.getConnectionStatus() || 'no-ref'
          console.log('🎤 [AUDIO] Skipping chunk - size:', event.data.size, 'deepgram ref:', !!deepgramRef.current, 'state status:', deepgramStatus, 'actual status:', actualStatus)
        }
      }

      mediaRecorder.onerror = (error) => {
        console.error('🎤 [AUDIO] MediaRecorder error:', error)
        setError('Audio recording failed. Please check microphone permissions.')
      }

      // Start recording with 250ms chunks for real-time streaming
      mediaRecorder.start(250)
      console.log('🎤 [AUDIO] MediaRecorder started with 250ms chunks')

    } catch (error) {
      console.error('🎤 [AUDIO] Failed to start audio capture:', error)
      let errorMessage = 'Failed to access microphone. '
      if (error instanceof Error) {
        if (error.name === 'NotAllowedError') {
          errorMessage += 'Please allow microphone access and try again.'
        } else if (error.name === 'NotFoundError') {
          errorMessage += 'No microphone found. Please check your device.'
        } else {
          errorMessage += error.message
        }
      }
      throw new Error(errorMessage)
    }
  }

  const stopSession = () => {
    console.log('🛑 [SESSION] Stopping session...')
    
    // Stop MediaRecorder
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current = null
    }
    
    // Stop media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop())
      mediaStreamRef.current = null
    }
    
    // Disconnect Deepgram
    if (deepgramRef.current) {
      deepgramRef.current.disconnect()
      deepgramRef.current = null
    }
    
    setIsRecording(false)
    setDeepgramStatus('disconnected')
    console.log('🎙️ [SESSION] Session stopped successfully')
  }

  const toggleSession = () => {
    if (isRecording) {
      stopSession()
    } else {
      startSession()
    }
  }

  const formatDuration = (ms: number): string => {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)
    
    if (hours > 0) {
      return `${hours}:${(minutes % 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`
    }
    return `${minutes}:${(seconds % 60).toString().padStart(2, '0')}`
  }



  if (!user) {
    return (
      <div className="text-center py-12">
        <Users className="h-16 w-16 text-gray-300 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">User Required</h3>
        <p className="text-gray-500">Please select a user to access live inference.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">🎙️ Live Inference</h1>
          <p className="text-gray-600">Real-time transcription and speaker identification</p>
        </div>
        <div className="flex items-center space-x-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={enableSpeakerIdentification}
              onChange={(e) => setEnableSpeakerIdentification(e.target.checked)}
              className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
            />
            <span className="ml-2 text-sm text-gray-700">
              Speaker Identification
            </span>
          </label>
        </div>
      </div>

      {/* API Key Configuration (only if manual key needed) */}
      {apiKeySource === 'manual' && !deepgramApiKey && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <AlertCircle className="h-5 w-5 text-yellow-600" />
            <h3 className="text-sm font-medium text-yellow-800">API Key Required</h3>
          </div>
          <p className="text-sm text-yellow-700 mb-3">
            Server API key not configured. Please enter your Deepgram API key:
          </p>
          <input
            type="password"
            value={deepgramApiKey}
            onChange={(e) => setDeepgramApiKey(e.target.value)}
            placeholder="Enter your Deepgram API key"
            className="w-full px-3 py-2 border border-gray-300 rounded-md"
          />
          <p className="text-xs text-gray-600 mt-1">
            Get your key from{' '}
            <a href="https://console.deepgram.com/" target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">
              Deepgram Console
            </a>
          </p>
        </div>
      )}

      {/* Session Stats */}
      {isRecording && (
        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <Clock className="h-4 w-4 text-gray-500" />
                <span className="text-sm text-gray-900">{formatDuration(stats.sessionDuration)}</span>
              </div>
              <div className="flex items-center space-x-2">
                <Volume2 className="h-4 w-4 text-gray-500" />
                <span className="text-sm text-gray-900">{stats.totalWords} words</span>
              </div>
              <div className="flex items-center space-x-2">
                <Users className="h-4 w-4 text-gray-500" />
                <span className="text-sm text-gray-900">{stats.identifiedSpeakers.size} speakers</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-600" />
            <div className="flex-1">
              <h4 className="text-red-800 font-medium">Session Error</h4>
              <p className="text-red-600 text-sm">{error}</p>
              {error.includes('API key') && (
                <p className="text-red-500 text-xs mt-1">
                  💡 Make sure your Deepgram API key is valid and has credits available
                </p>
              )}
              {error.includes('WebSocket') && (
                <p className="text-red-500 text-xs mt-1">
                  💡 Check your internet connection and firewall settings
                </p>
              )}
              {error.includes('timeout') && (
                <p className="text-red-500 text-xs mt-1">
                  💡 Connection timed out - try again or check Deepgram service status
                </p>
              )}
            </div>
            <button
              onClick={() => {
                console.log('🎙️ [ERROR] Clear error button clicked')
                setError(null)
              }}
              className="text-red-400 hover:text-red-600 text-sm"
              title="Clear error"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Recording Status */}
      {isRecording && (
        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-600 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-gray-900">Recording</span>
            </div>
            <div className="flex items-center space-x-2">
              <Mic className="h-4 w-4 text-gray-500" />
              <span className="text-sm text-gray-600">
                Status: {deepgramStatus === 'connected' ? '✅ Connected' : deepgramStatus}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Single Control Button */}
      <div className="flex justify-center">
        <button
          onClick={toggleSession}
          disabled={!deepgramApiKey || deepgramStatus === 'connecting' || apiKeySource === 'loading'}
          className={`flex items-center space-x-2 px-8 py-4 rounded-lg font-medium text-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors ${
            isRecording
              ? 'bg-red-600 text-white hover:bg-red-700'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          <Mic className="h-6 w-6" />
          <span>
            {apiKeySource === 'loading' ? '⏳ Loading Config...' :
             deepgramStatus === 'connecting' ? '🔄 Connecting...' : 
             isRecording ? 'Stop Transcribe & Identify' :
             'Start Transcribe & Identify'}
          </span>
        </button>
      </div>

      {/* Live Transcript */}
      <div className="bg-white border rounded-lg">
        <div className="p-4 border-b">
          <h3 className="text-lg font-medium text-gray-900">Live Transcript</h3>
        </div>
        <div className="p-4 max-h-96 overflow-y-auto">
          {transcriptSegments.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Mic className="h-12 w-12 mx-auto mb-2 opacity-50" />
              <p>Start a session to see live transcription</p>
            </div>
          ) : (
            <div className="space-y-2">
              {transcriptSegments.map((segment) => (
                <div 
                  key={segment.id} 
                  className={`p-3 rounded-lg transition-all ${
                    segment.isInterim 
                      ? 'bg-gray-50 border-l-4 border-yellow-400 opacity-70' 
                      : 'bg-blue-50 border-l-4 border-blue-400'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className={`text-sm font-medium ${segment.isInterim ? 'text-gray-600' : 'text-gray-900'}`}>
                        {segment.identifiedSpeaker?.name || `Speaker ${segment.speaker + 1}`}
                      </span>
                      {segment.identifiedSpeaker && !segment.isInterim && (
                        <span className="text-xs text-green-600 bg-green-100 px-2 py-1 rounded-full">
                          ID: {(segment.identifiedSpeaker.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                      {segment.isInterim && (
                        <span className="text-xs text-yellow-600 bg-yellow-100 px-2 py-1 rounded-full">
                          typing...
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-400">
                      {new Date(segment.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                  <p className={`${segment.isInterim ? 'text-gray-600 italic' : 'text-gray-900 font-medium'}`}>
                    {segment.text}
                  </p>
                </div>
              ))}
              <div ref={transcriptEndRef} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}