import { useState, useRef, useCallback, useEffect } from 'react'

export type RecordingStep = 'idle' | 'mic' | 'websocket' | 'audio-start' | 'streaming' | 'stopping' | 'error'

export interface DebugStats {
  chunksSent: number
  messagesReceived: number
  lastError: string | null
  lastErrorTime: Date | null
  sessionStartTime: Date | null
  connectionAttempts: number
}

export interface SimpleAudioRecordingReturn {
  // Current state
  currentStep: RecordingStep
  isRecording: boolean
  recordingDuration: number
  error: string | null
  
  // Actions
  startRecording: () => Promise<void>
  stopRecording: () => void
  
  // For components
  analyser: AnalyserNode | null
  debugStats: DebugStats
  
  // Utilities
  formatDuration: (seconds: number) => string
  canAccessMicrophone: boolean
}

export const useSimpleAudioRecording = (): SimpleAudioRecordingReturn => {
  // Basic state
  const [currentStep, setCurrentStep] = useState<RecordingStep>('idle')
  const [isRecording, setIsRecording] = useState(false)
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [error, setError] = useState<string | null>(null)
  
  // Debug stats
  const [debugStats, setDebugStats] = useState<DebugStats>({
    chunksSent: 0,
    messagesReceived: 0,
    lastError: null,
    lastErrorTime: null,
    sessionStartTime: null,
    connectionAttempts: 0
  })
  
  // Refs for direct access
  const wsRef = useRef<WebSocket | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const durationIntervalRef = useRef<number>()
  const keepAliveIntervalRef = useRef<number>()
  const chunkCountRef = useRef(0)
  const audioProcessingStartedRef = useRef(false)
  
  // Note: user was unused and removed
  
  // Check if we're on localhost or using HTTPS
  const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  const isHttps = window.location.protocol === 'https:'
  const canAccessMicrophone = isLocalhost || isHttps
  
  // Format duration helper
  const formatDuration = useCallback((seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }, [])
  
  // Cleanup function
  const cleanup = useCallback(() => {
    console.log('🧹 Cleaning up audio recording resources')
    
    // Stop audio processing
    audioProcessingStartedRef.current = false
    
    // Clean up media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop())
      mediaStreamRef.current = null
    }
    
    // Clean up audio context
    if (audioContextRef.current?.state !== 'closed') {
      audioContextRef.current?.close()
    }
    audioContextRef.current = null
    analyserRef.current = null
    processorRef.current = null
    
    // Clean up WebSocket
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    
    // Clear intervals
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current)
      durationIntervalRef.current = undefined
    }
    
    if (keepAliveIntervalRef.current) {
      clearInterval(keepAliveIntervalRef.current)
      keepAliveIntervalRef.current = undefined
    }
    
    // Reset counters
    chunkCountRef.current = 0
  }, [])
  
  // Step 1: Get microphone access
  const getMicrophoneAccess = useCallback(async (): Promise<MediaStream> => {
    console.log('🎤 Step 1: Requesting microphone access')
    
    if (!canAccessMicrophone) {
      throw new Error('Microphone access requires HTTPS or localhost')
    }
    
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
    console.log('✅ Microphone access granted')
    return stream
  }, [canAccessMicrophone])
  
  // Step 2: Connect WebSocket
  const connectWebSocket = useCallback(async (): Promise<WebSocket> => {
    console.log('🔗 Step 2: Connecting to WebSocket')
    
    const token = localStorage.getItem('token')
    if (!token) {
      throw new Error('No authentication token found')
    }
    
    // Build WebSocket URL using same logic as API service
    let wsUrl: string
    const { protocol, port } = window.location
    
    // Check if we have a backend URL from environment
    if (import.meta.env.VITE_BACKEND_URL) {
      const backendUrl = import.meta.env.VITE_BACKEND_URL
      const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:'
      // Fallback logic based on current location
      const isStandardPort = (protocol === 'https:' && (port === '' || port === '443')) || 
                             (protocol === 'http:' && (port === '' || port === '80'))
      
      if (isStandardPort || backendUrl === '') {
        // Use same origin for Ingress access
        wsUrl = `${wsProtocol}//${window.location.host}/ws_pcm?token=${token}&device_name=webui-simple-recorder`
      } else if (backendUrl != undefined && backendUrl != '') {
        wsUrl = `${wsProtocol}//${backendUrl}/ws_pcm?token=${token}&device_name=webui-simple-recorder`
      }    
      else if (port === '5173') {
        // Development mode
        wsUrl = `ws://localhost:8000/ws_pcm?token=${token}&device_name=webui-simple-recorder`
      } else {
        // Fallback - use same origin instead of hardcoded port 8000
        wsUrl = `${wsProtocol}//${window.location.host}/ws_pcm?token=${token}&device_name=webui-simple-recorder`
      }
    } else {
      // No environment variable set, use same origin as fallback
      const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:'
      wsUrl = `${wsProtocol}//${window.location.host}/ws_pcm?token=${token}&device_name=webui-simple-recorder`
    }
    
    return new Promise<WebSocket>((resolve, reject) => {
      const ws = new WebSocket(wsUrl)
      // Don't set binaryType yet - only when needed for audio chunks
      
      ws.onopen = () => {
        console.log('🔌 WebSocket connected')
        
        // Add stabilization delay before resolving
        setTimeout(() => {
          wsRef.current = ws
          setDebugStats(prev => ({ 
            ...prev, 
            connectionAttempts: prev.connectionAttempts + 1,
            sessionStartTime: new Date()
          }))
          
          // Start keepalive ping every 30 seconds
          keepAliveIntervalRef.current = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
              try {
                const ping = { type: 'ping', payload_length: null }
                ws.send(JSON.stringify(ping) + '\n')
              } catch (e) {
                console.error('Failed to send keepalive ping:', e)
              }
            }
          }, 30000)
          
          console.log('✅ WebSocket stabilized and ready')
          resolve(ws)
        }, 100) // 100ms stabilization delay
      }
      
      ws.onclose = (event) => {
        console.log('🔌 WebSocket disconnected:', event.code, event.reason)
        wsRef.current = null
        
        if (keepAliveIntervalRef.current) {
          clearInterval(keepAliveIntervalRef.current)
          keepAliveIntervalRef.current = undefined
        }
      }
      
      ws.onerror = (error) => {
        console.error('🔌 WebSocket error:', error)
        reject(new Error('Failed to connect to backend'))
      }
      
      ws.onmessage = (event) => {
        console.log('📨 Received message from server:', event.data)
        setDebugStats(prev => ({ ...prev, messagesReceived: prev.messagesReceived + 1 }))
      }
    })
  }, [])
  
  // Step 3: Send audio-start message
  const sendAudioStartMessage = useCallback(async (ws: WebSocket): Promise<void> => {
    console.log('📤 Step 3: Sending audio-start message')
    
    if (ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected')
    }
    
    const startMessage = {
      type: 'audio-start',
      data: {
        rate: 16000,
        width: 2,
        channels: 1
      },
      payload_length: null
    }
    
    ws.send(JSON.stringify(startMessage) + '\n')
    console.log('✅ Audio-start message sent')
  }, [])
  
  // Step 4: Start audio streaming
  const startAudioStreaming = useCallback(async (stream: MediaStream, ws: WebSocket): Promise<void> => {
    console.log('🎵 Step 4: Starting audio streaming')
    
    // Set up audio context and analyser for visualization
    const audioContext = new AudioContext({ sampleRate: 16000 })
    const analyser = audioContext.createAnalyser()
    const source = audioContext.createMediaStreamSource(stream)
    
    analyser.fftSize = 256
    source.connect(analyser)
    
    audioContextRef.current = audioContext
    analyserRef.current = analyser
    
    // Wait brief moment for backend to process audio-start
    await new Promise(resolve => setTimeout(resolve, 100))
    
    // Set up audio processing
    const processor = audioContext.createScriptProcessor(4096, 1, 1)
    source.connect(processor)
    processor.connect(audioContext.destination)
    
    processor.onaudioprocess = (event) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        return
      }
      
      if (!audioProcessingStartedRef.current) {
        console.log('🚫 Audio processing not started yet, skipping chunk')
        return
      }
      
      const inputBuffer = event.inputBuffer
      const inputData = inputBuffer.getChannelData(0)
      
      // Convert float32 to int16 PCM
      const pcmBuffer = new Int16Array(inputData.length)
      for (let i = 0; i < inputData.length; i++) {
        const sample = Math.max(-1, Math.min(1, inputData[i]))
        pcmBuffer[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF
      }
      
      try {
        const chunkHeader = {
          type: 'audio-chunk',
          data: {
            rate: 16000,
            width: 2,
            channels: 1
          },
          payload_length: pcmBuffer.byteLength
        }
        
        // Set binary type for WebSocket before sending binary data
        if (ws.binaryType !== 'arraybuffer') {
          ws.binaryType = 'arraybuffer'
          console.log('🔧 Set WebSocket binaryType to arraybuffer for audio chunks')
        }
        
        ws.send(JSON.stringify(chunkHeader) + '\n')
        ws.send(new Uint8Array(pcmBuffer.buffer, pcmBuffer.byteOffset, pcmBuffer.byteLength))
        
        // Update debug stats
        chunkCountRef.current++
        setDebugStats(prev => ({ ...prev, chunksSent: chunkCountRef.current }))
      } catch (error) {
        console.error('Failed to send audio chunk:', error)
        setDebugStats(prev => ({ 
          ...prev, 
          lastError: error instanceof Error ? error.message : 'Chunk send failed',
          lastErrorTime: new Date()
        }))
      }
    }
    
    processorRef.current = processor
    audioProcessingStartedRef.current = true
    
    console.log('✅ Audio streaming started')
  }, [])
  
  // Main start recording function - sequential flow
  const startRecording = useCallback(async () => {
    try {
      setError(null)
      setCurrentStep('mic')
      
      // Step 1: Get microphone access
      const stream = await getMicrophoneAccess()
      
      setCurrentStep('websocket')
      // Step 2: Connect WebSocket (includes stabilization delay)
      const ws = await connectWebSocket()
      
      setCurrentStep('audio-start')
      // Step 3: Send audio-start message
      await sendAudioStartMessage(ws)
      
      setCurrentStep('streaming')
      // Step 4: Start audio streaming (includes processing delay)
      await startAudioStreaming(stream, ws)
      
      // All steps complete - mark as recording
      setIsRecording(true)
      setRecordingDuration(0)
      
      // Start duration timer
      durationIntervalRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1)
      }, 1000)
      
      console.log('🎉 Recording started successfully!')
      
    } catch (error) {
      console.error('❌ Recording failed:', error)
      setCurrentStep('error')
      setError(error instanceof Error ? error.message : 'Recording failed')
      setDebugStats(prev => ({ 
        ...prev, 
        lastError: error instanceof Error ? error.message : 'Recording failed',
        lastErrorTime: new Date()
      }))
      cleanup()
    }
  }, [getMicrophoneAccess, connectWebSocket, sendAudioStartMessage, startAudioStreaming, cleanup])
  
  // Stop recording function
  const stopRecording = useCallback(() => {
    if (!isRecording) return
    
    console.log('🛑 Stopping recording')
    setCurrentStep('stopping')
    
    // Stop audio processing
    audioProcessingStartedRef.current = false
    
    // Send audio-stop message if WebSocket is still open
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        const stopMessage = {
          type: 'audio-stop',
          data: { timestamp: Date.now() },
          payload_length: null
        }
        wsRef.current.send(JSON.stringify(stopMessage) + '\n')
        console.log('📤 Audio-stop message sent')
      } catch (error) {
        console.error('Failed to send audio-stop:', error)
      }
    }
    
    // Cleanup resources
    cleanup()
    
    // Reset state
    setIsRecording(false)
    setRecordingDuration(0)
    setCurrentStep('idle')
    
    console.log('✅ Recording stopped')
  }, [isRecording, cleanup])
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup()
    }
  }, [cleanup])
  
  return {
    currentStep,
    isRecording,
    recordingDuration,
    error,
    startRecording,
    stopRecording,
    analyser: analyserRef.current,
    debugStats,
    formatDuration,
    canAccessMicrophone
  }
}