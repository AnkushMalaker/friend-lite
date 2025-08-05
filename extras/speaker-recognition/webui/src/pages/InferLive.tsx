import React, { useState, useEffect, useRef } from 'react'
import { Mic, Users, AlertCircle, Volume2, Clock } from 'lucide-react'
import { useUser } from '../contexts/UserContext'
import { apiService } from '../services/api'
import { 
  DeepgramStreaming, 
  StreamingConfig, 
  StreamingTranscript,
  SpeechStartedEvent,
  UtteranceEndEvent,
  DEFAULT_DEEPGRAM_OPTIONS 
} from '../services/deepgram'

interface SpeakerPart {
  speaker: string
  text: string
  confidence: number
}

interface TranscriptSegment {
  id: string
  timestamp: number
  speaker: number
  text: string
  confidence: number
  isInterim: boolean
  speakerParts?: SpeakerPart[]  // Inline speaker labels within the text
}





interface IdentifyResult {
  found: boolean
  speaker_id: string | null
  speaker_name: string | null
  confidence: number
  status: string
  similarity_threshold: number
  duration: number
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
  const audioBufferRef = useRef<Float32Array[]>([])
  const lastTranscriptRef = useRef<string>('')

  const utteranceTranscriptsRef = useRef<string[]>([])
  const currentUtteranceSegmentIds = useRef<string[]>([])
  const processingUtteranceRef = useRef(false)
  
  // Utterance timing tracking
  const utteranceStartTimeRef = useRef<number | null>(null)
  const streamStartTimeRef = useRef<number | null>(null) // When the Deepgram stream started
  const audioBufferStartIndexRef = useRef<number>(0)
  
  // Helper function to reset utterance refs
  const _resetRefs = () => {
    utteranceTranscriptsRef.current = []
    currentUtteranceSegmentIds.current = []
    utteranceStartTimeRef.current = null
    processingUtteranceRef.current = false
  }

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
      transcriptEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [transcriptSegments])

  // Update session duration
  useEffect(() => {
    if (!isRecording) return

    const interval = setInterval(() => {
      setStats(prev => ({
        ...prev,
        sessionDuration: Date.now() - sessionStartRef.current
      }))
    }, 1000)

    return () => clearInterval(interval)
  }, [isRecording])

  // Helper function to create WAV blob from Float32Array
  const createWAVBlob = (audioBuffer: Float32Array, sampleRate: number): Blob => {
    const length = audioBuffer.length
    const arrayBuffer = new ArrayBuffer(length * 2)
    const view = new DataView(arrayBuffer)
    
    // Convert to 16-bit PCM
    for (let i = 0; i < length; i++) {
      const sample = Math.max(-1, Math.min(1, audioBuffer[i]))
      view.setInt16(i * 2, sample * 0x7FFF, true)
    }
    
    // Create WAV blob
    return new Blob([createWAVHeader(sampleRate, 1, 16, arrayBuffer.byteLength), arrayBuffer], {
      type: 'audio/wav'
    })
  }

  // Simplified utterance speaker identification
  const identifyUtteranceSpeaker = async (audioBuffer: Float32Array, sampleRate: number): Promise<IdentifyResult> => {
    try {
      console.log('🔍 [UTTERANCE_ID] Starting utterance speaker identification')
      
      // Create WAV blob
      const wavBlob = createWAVBlob(audioBuffer, sampleRate)
      
      // Call the simple identify-utterance endpoint
      const formData = new FormData()
      formData.append('file', wavBlob, 'utterance.wav')
      formData.append('similarity_threshold', '0.15')
      if (user?.id) {
        formData.append('user_id', user.id.toString())
      }
      
      const response = await apiService.post('/identify', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 15000, // 15 seconds for utterance identification
      })
      
      console.log('🔍 [UTTERANCE_ID] API Response:', response.data)
      
      return response.data as IdentifyResult
      
    } catch (error) {
      console.error('🔍 [UTTERANCE_ID] Error during utterance identification:', error)
      return { 
        found: false, 
        speaker_id: null, 
        speaker_name: null, 
        confidence: 0, 
        status: 'error',
        similarity_threshold: 0.15,
        duration: 0
      }
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
      // Initialize Deepgram streaming with VAD events for better speaker detection
      const config: StreamingConfig = {
        apiKey: deepgramApiKey,
        model: 'nova-3',
        language: 'multi',         // Multilingual support
        // Use standard linear16 format for maximum compatibility
        encoding: 'linear16',
        sample_rate: 16000,
        // Enable speech detection events
        interim_results: true,      // Required for UtteranceEnd
        vad_events: true,          // Enables SpeechStarted events  
        utterance_end_ms: 1000,    // UtteranceEnd after 1s gap
        endpointing: 300           // Still use endpointing for speech_final
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

        // Always start with N/A - will be split into speaker parts after identification
        segment.speakerParts = [{
          speaker: 'N/A',
          text: segment.text,
          confidence: 0.0
        }]

        // Store the latest transcript for UtteranceEnd processing
        lastTranscriptRef.current = transcript.transcript
        
        // Collect transcript segments for utterance processing
        if (!segment.isInterim) {
          utteranceTranscriptsRef.current.push(transcript.transcript)
          currentUtteranceSegmentIds.current.push(segment.id)
          
          // Track the start of the utterance (first segment)
          if (utteranceTranscriptsRef.current.length === 1 && transcript.words.length > 0) {
            utteranceStartTimeRef.current = transcript.words[0].start
            console.log('📝 [UTTERANCE] ⏰ Utterance started at:', utteranceStartTimeRef.current)
          }
          
          console.log('📝 [UTTERANCE] Collected segment:', transcript.transcript)
          console.log('📝 [UTTERANCE] Segment ID:', segment.id)
          console.log('📝 [UTTERANCE] Total segments so far:', utteranceTranscriptsRef.current.length)
          console.log('📝 [UTTERANCE] Current segment IDs:', currentUtteranceSegmentIds.current)
        }

        // Add segment to display immediately (for real-time feedback)
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
            // Collect all identified speakers from speaker parts
            if (segment.speakerParts) {
              segment.speakerParts.forEach(part => {
                if (part.speaker !== 'N/A') {
                  newIdentified.add(part.speaker)
                }
              })
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

      // Handle SpeechStarted events - just log for debugging
      deepgramRef.current.onSpeechStarted((event: SpeechStartedEvent) => {
        console.log('🎙️ [SPEECH] Speech started at', event.timestamp, '(VAD detection)')
      })

      // Handle UtteranceEnd events - trigger simple speaker identification
      deepgramRef.current.onUtteranceEnd(async (event: UtteranceEndEvent) => {
        if (!enableSpeakerIdentification) return
        
        // Prevent duplicate processing
        if (processingUtteranceRef.current) {
          console.log('🔚 [UTTERANCE] ⚠️ Already processing an utterance, skipping duplicate event')
          return
        }
        
        console.log('🔚 [UTTERANCE] Utterance ended, triggering speaker identification')
        console.log('🔚 [UTTERANCE] Collected transcripts:', utteranceTranscriptsRef.current)
        console.log('🔚 [UTTERANCE] Collected segment IDs:', currentUtteranceSegmentIds.current)
        
        if (utteranceTranscriptsRef.current.length === 0) {
          console.log('🔚 [UTTERANCE] No transcripts to process')
          return
        }

        if (currentUtteranceSegmentIds.current.length === 0) {
          console.log('🔚 [UTTERANCE] ⚠️ No segment IDs collected - this is unexpected!')
          console.log('🔚 [UTTERANCE] This might be a timing issue or duplicate utterance event')
          return
        }

        processingUtteranceRef.current = true

        // Capture segment IDs locally to prevent race conditions
        const segmentIdsToUpdate = [...currentUtteranceSegmentIds.current]
        const transcriptsToProcess = [...utteranceTranscriptsRef.current]
        
        console.log('🔚 [UTTERANCE] Captured for processing:', {
          segmentIds: segmentIdsToUpdate,
          transcripts: transcriptsToProcess
        })

        try {
          // Calculate utterance timing and extract the exact audio segment
          const utteranceStartTime = utteranceStartTimeRef.current
          const utteranceEndTime = event.last_word_end
          const streamStartTime = streamStartTimeRef.current
          
          if (!utteranceStartTime || !streamStartTime) {
            console.error('🔚 [UTTERANCE] ❌ Missing timing information - cannot extract utterance audio')
            console.error('🔚 [UTTERANCE] utteranceStartTime:', utteranceStartTime, 'streamStartTime:', streamStartTime)
            _resetRefs()
            return
          }
          
          // Calculate which buffers correspond to the utterance timing
          const utteranceDuration = utteranceEndTime - utteranceStartTime
          const bufferDurationMs = 256 // Each buffer is ~256ms (4096 samples at 16kHz)
          
          // Calculate buffer indices based on stream-relative timing
          // utteranceStartTime and utteranceEndTime are in seconds from stream start
          const startBufferIndex = Math.max(0, Math.floor(utteranceStartTime * 1000 / bufferDurationMs))
          const endBufferIndex = Math.min(audioBufferRef.current.length, Math.ceil(utteranceEndTime * 1000 / bufferDurationMs))
          
          console.log('🔚 [UTTERANCE] Timing calculation:', {
            utteranceStartTime,
            utteranceEndTime,
            utteranceDuration,
            streamStartTime,
            startBufferIndex,
            endBufferIndex,
            totalBuffers: audioBufferRef.current.length
          })
          
          if (startBufferIndex >= endBufferIndex || startBufferIndex >= audioBufferRef.current.length) {
            console.error('🔚 [UTTERANCE] ❌ Invalid buffer range - cannot extract utterance audio')
            console.error('🔚 [UTTERANCE] startBufferIndex:', startBufferIndex, 'endBufferIndex:', endBufferIndex, 'totalBuffers:', audioBufferRef.current.length)
            _resetRefs()
            return
          }
          
          // Extract the exact utterance buffers
          const utteranceBuffers = audioBufferRef.current.slice(startBufferIndex, endBufferIndex)
          
          if (utteranceBuffers.length === 0) {
            console.error('🔚 [UTTERANCE] ❌ No utterance buffers in calculated range')
            _resetRefs()
            return
          }
          
          // Concatenate the utterance audio buffers
          const totalLength = utteranceBuffers.reduce((sum, buf) => sum + buf.length, 0)
          const audioBuffer = new Float32Array(totalLength)
          let offset = 0
          for (const buffer of utteranceBuffers) {
            audioBuffer.set(buffer, offset)
            offset += buffer.length
          }
          
          console.log('🔚 [UTTERANCE] ✅ Exact utterance audio buffer created:', {
            bufferCount: utteranceBuffers.length,
            totalSamples: audioBuffer.length,
            durationSeconds: audioBuffer.length / 16000,
            calculatedDuration: utteranceDuration,
            startBufferIndex,
            endBufferIndex
          })
          
          // Call simplified speaker identification API
          const identification = await identifyUtteranceSpeaker(audioBuffer, 16000)
          
          console.log('🔍 [SPEAKER_ID] Identification result:', identification)
          
          // Update transcript segments with speaker information
          setTranscriptSegments(prev => {
            const updatedSegments = [...prev]
            
            console.log('🔍 [SPEAKER_ID] Updating segments with IDs:', segmentIdsToUpdate)
            
            // Apply speaker identification only to segments from this utterance
            segmentIdsToUpdate.forEach((segmentId) => {
              const segmentIndex = updatedSegments.findIndex(seg => seg.id === segmentId)
              
              if (segmentIndex >= 0) {
                const speakerName = identification.found ? 
                  identification.speaker_name : 'Unknown'
                
                updatedSegments[segmentIndex] = {
                  ...updatedSegments[segmentIndex],
                  speakerParts: [{
                    speaker: speakerName || 'Unknown',
                    text: updatedSegments[segmentIndex].text,
                    confidence: identification.confidence
                  }]
                }
                
                console.log(`🔍 [SPEAKER_ID] Updated segment ${segmentId} with speaker: ${speakerName}`)
              } else {
                console.warn(`🔍 [SPEAKER_ID] Could not find segment with ID: ${segmentId}`)
              }
            })
            
            return updatedSegments
          })
          
          // Update current speakers mapping
          if (identification.found && identification.speaker_name) {
            setCurrentSpeakers(prev => {
              const newSpeakers = new Map(prev)
              newSpeakers.set(0, identification.speaker_name!)
              return newSpeakers
            })
          }
          
        } catch (error) {
          console.error('🔍 [SPEAKER_ID] Error during utterance identification:', error)
          _resetRefs()
        }
        
        // Reset for next utterance
        _resetRefs()
      })

      // Connect to Deepgram
      await deepgramRef.current.connect()
      console.log('🎙️ [SESSION] Connected to Deepgram')
      
      // Record when the stream started for timing calculations
      streamStartTimeRef.current = Date.now() / 1000 // Unix timestamp when stream started
      audioBufferStartIndexRef.current = 0

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
      // Request microphone access with standard 16kHz settings
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

      // Use Web Audio API to process audio directly to linear16 format
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000
      })
      
      const source = audioContext.createMediaStreamSource(stream)
      const processor = audioContext.createScriptProcessor(4096, 1, 1)

      console.log('🎤 [AUDIO] Using Web Audio API for linear16 processing at 16kHz')

      processor.onaudioprocess = (event) => {
        if (deepgramRef.current && deepgramRef.current.getConnectionStatus() === 'connected') {
          const inputBuffer = event.inputBuffer
          const inputData = inputBuffer.getChannelData(0)
          
          // Buffer audio for speaker identification (keep last 30 seconds)
          const audioCopy = new Float32Array(inputData)
          audioBufferRef.current.push(audioCopy)
          if (audioBufferRef.current.length > 750) { // 30 seconds at 4096 samples per 250ms
            audioBufferRef.current.shift()
          }
          
          // Convert Float32Array to Int16 (linear16)
          const int16Buffer = new Int16Array(inputData.length)
          for (let i = 0; i < inputData.length; i++) {
            const sample = Math.max(-1, Math.min(1, inputData[i]))
            int16Buffer[i] = sample * 0x7FFF
          }
          
          // Reduced logging - only log every 100 packets to reduce spam
          if (Math.random() < 0.01) {
            console.log('🎤 [AUDIO] Sending audio data...')
          }
          deepgramRef.current.sendAudio(int16Buffer.buffer)
        }
      }

      source.connect(processor)
      processor.connect(audioContext.destination)

      // Store references for cleanup
      ;(mediaStreamRef.current as any).audioContext = audioContext
      ;(mediaStreamRef.current as any).processor = processor

      console.log('🎤 [AUDIO] Web Audio API processing started')

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
    
    // Stop Web Audio API components
    if (mediaStreamRef.current) {
      const stream = mediaStreamRef.current as any
      
      // Stop audio context and processor
      if (stream.audioContext) {
        stream.audioContext.close()
      }
      if (stream.processor) {
        stream.processor.disconnect()
      }
      
      // Stop media stream tracks
      mediaStreamRef.current.getTracks().forEach(track => track.stop())
      mediaStreamRef.current = null
    }
    
    // Stop MediaRecorder if it exists (fallback cleanup)
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current = null
    }
    
    // Disconnect Deepgram
    if (deepgramRef.current) {
      deepgramRef.current.disconnect()
      deepgramRef.current = null
    }
    
    // Clear audio buffer and transcript references
    audioBufferRef.current = []
    lastTranscriptRef.current = ''
    _resetRefs()
    
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
                  <div className={`${segment.isInterim ? 'text-gray-600 italic' : 'text-gray-900'}`}>
                    {segment.speakerParts?.map((part, partIndex) => (
                      <span key={partIndex} className="inline-block mr-2">
                        <span className={`font-semibold ${
                          part.speaker === 'N/A' 
                            ? 'text-gray-500' 
                            : 'text-blue-700'
                        }`}>
                          {part.speaker}:
                        </span>
                        <span className="ml-1">
                          "{part.text}"
                        </span>
                        {part.speaker !== 'N/A' && part.confidence > 0 && (
                          <span className="text-xs text-green-600 bg-green-100 px-1 py-0.5 rounded ml-1">
                            {(part.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </span>
                    )) || (
                      <span className="text-gray-500">
                        <span className="font-semibold">N/A:</span>
                        <span className="ml-1">"{segment.text}"</span>
                      </span>
                    )}
                  </div>
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