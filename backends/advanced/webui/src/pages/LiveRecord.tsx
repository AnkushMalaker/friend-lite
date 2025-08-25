import { useState, useRef, useEffect, useCallback } from 'react'
import { Mic, MicOff, Radio, AlertTriangle, Wifi, WifiOff } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

interface AudioVisualizerProps {
  isRecording: boolean
  analyser: AnalyserNode | null
}

function AudioVisualizer({ isRecording, analyser }: AudioVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationIdRef = useRef<number>()

  const drawWaveform = useCallback(() => {
    if (!analyser || !canvasRef.current) return

    const canvas = canvasRef.current
    const canvasCtx = canvas.getContext('2d')
    if (!canvasCtx) return

    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const draw = () => {
      if (!isRecording) return

      analyser.getByteFrequencyData(dataArray)

      canvasCtx.fillStyle = 'rgb(17, 24, 39)' // gray-900
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height)

      const barWidth = (canvas.width / bufferLength) * 2.5
      let barHeight
      let x = 0

      for (let i = 0; i < bufferLength; i++) {
        barHeight = (dataArray[i] / 255) * canvas.height

        // Gradient from blue to green based on intensity
        const intensity = dataArray[i] / 255
        const red = Math.floor(59 * (1 - intensity) + 34 * intensity)
        const green = Math.floor(130 * (1 - intensity) + 197 * intensity)
        const blue = Math.floor(246 * (1 - intensity) + 94 * intensity)
        
        canvasCtx.fillStyle = `rgb(${red},${green},${blue})`
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight)

        x += barWidth + 1
      }

      animationIdRef.current = requestAnimationFrame(draw)
    }

    draw()
  }, [analyser, isRecording])

  useEffect(() => {
    if (isRecording && analyser) {
      drawWaveform()
    } else {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current)
      }
      
      // Clear canvas
      if (canvasRef.current) {
        const canvasCtx = canvasRef.current.getContext('2d')
        if (canvasCtx) {
          canvasCtx.fillStyle = 'rgb(17, 24, 39)'
          canvasCtx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height)
        }
      }
    }

    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current)
      }
    }
  }, [isRecording, analyser, drawWaveform])

  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-700">
      <canvas
        ref={canvasRef}
        width={600}
        height={100}
        className="w-full h-24 bg-gray-900 rounded"
      />
      <p className="text-center text-sm text-gray-400 mt-2">
        {isRecording ? 'Audio Waveform - Recording...' : 'Audio Waveform - Ready'}
      </p>
    </div>
  )
}

export default function LiveRecord() {
  const [isRecording, setIsRecording] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected')
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const durationIntervalRef = useRef<number>()
  const keepAliveIntervalRef = useRef<number>()

  const { user } = useAuth()

  // Check if we're on localhost or using HTTPS
  const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  const isHttps = window.location.protocol === 'https:'
  const canAccessMicrophone = isLocalhost || isHttps

  const connectWebSocket = useCallback(async () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return true
    }

    setConnectionStatus('connecting')
    setError(null)

    try {
      const token = localStorage.getItem('token')
      if (!token) {
        throw new Error('No authentication token found')
      }

      // Use appropriate WebSocket protocol and host based on page access
      const { protocol, hostname, port } = window.location
      const isStandardPort = (protocol === 'https:' && (port === '' || port === '443')) || 
                             (protocol === 'http:' && (port === '' || port === '80'))
      
      let wsUrl: string
      if (isStandardPort) {
        // Accessed through nginx proxy - use same host with secure WebSocket
        const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:'
        wsUrl = `${wsProtocol}//${window.location.host}/ws_pcm?token=${token}&device_name=webui-recorder`
      } else if (port === '5173') {
        // Development mode - direct connection to backend
        wsUrl = `ws://localhost:8000/ws_pcm?token=${token}&device_name=webui-recorder`
      } else {
        // Fallback
        const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:'
        wsUrl = `${wsProtocol}//${hostname}:8000/ws_pcm?token=${token}&device_name=webui-recorder`
      }
      const ws = new WebSocket(wsUrl)
      ws.binaryType = 'arraybuffer'  // Ensure binary data is handled correctly

      return new Promise<boolean>((resolve, reject) => {
        ws.onopen = () => {
          console.log('🎤 WebSocket connected for live recording')
          setConnectionStatus('connected')
          wsRef.current = ws
          
          // Start keepalive ping every 30 seconds
          keepAliveIntervalRef.current = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
              try {
                // Send a Wyoming protocol ping event
                const ping = { type: 'ping', payload_length: null }
                ws.send(JSON.stringify(ping) + '\n')
              } catch (e) {
                console.error('Failed to send keepalive ping:', e)
              }
            }
          }, 30000)
          
          resolve(true)
        }

        ws.onclose = (event) => {
          console.log('🎤 WebSocket disconnected:', event.code, event.reason)
          setConnectionStatus('disconnected')
          wsRef.current = null
          
          // Clear keepalive interval
          if (keepAliveIntervalRef.current) {
            clearInterval(keepAliveIntervalRef.current)
            keepAliveIntervalRef.current = undefined
          }
          
          if (isRecording) {
            stopRecording()
          }
        }

        ws.onerror = (error) => {
          console.error('🎤 WebSocket error:', error)
          setConnectionStatus('error')
          setError('Failed to connect to backend')
          reject(error)
        }
        
        ws.onmessage = (event) => {
          // Handle any messages from the server
          console.log('🎤 Received message from server:', event.data)
        }

        // Timeout after 5 seconds
        setTimeout(() => {
          if (ws.readyState !== WebSocket.OPEN) {
            ws.close()
            reject(new Error('Connection timeout'))
          }
        }, 5000)
      })
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
      setConnectionStatus('error')
      setError(error instanceof Error ? error.message : 'Connection failed')
      return false
    }
  }, [isRecording])

  const startRecording = async () => {
    try {
      setError(null)

      if (!canAccessMicrophone) {
        setError('Microphone access requires either localhost access or HTTPS connection due to browser security restrictions')
        return
      }

      // Connect WebSocket first
      const connected = await connectWebSocket()
      if (!connected) {
        return
      }

      // Get user media
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

      // Set up audio context and analyser for visualization
      const audioContext = new AudioContext({ sampleRate: 16000 })
      const analyser = audioContext.createAnalyser()
      const source = audioContext.createMediaStreamSource(stream)
      
      analyser.fftSize = 256
      source.connect(analyser)
      
      audioContextRef.current = audioContext
      analyserRef.current = analyser

      // Set up audio processing for WebSocket
      const processor = audioContext.createScriptProcessor(4096, 1, 1)
      source.connect(processor)
      processor.connect(audioContext.destination)

      processor.onaudioprocess = (event) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
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
          // Send Wyoming protocol audio chunk
          const chunkHeader = {
            type: 'audio-chunk',
            data: {
              rate: 16000,
              width: 2,
              channels: 1
            },
            payload_length: pcmBuffer.byteLength
          }

          // Send header + binary data
          wsRef.current.send(JSON.stringify(chunkHeader) + '\n')
          // Send the actual Int16Array buffer, not the underlying ArrayBuffer
          wsRef.current.send(new Uint8Array(pcmBuffer.buffer, pcmBuffer.byteOffset, pcmBuffer.byteLength))
        } catch (error) {
          console.error('Failed to send audio chunk:', error)
        }
      }

      processorRef.current = processor

      // Send Wyoming protocol start message
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        try {
          const startMessage = {
            type: 'audio-start',
            data: {
              rate: 16000,
              width: 2,
              channels: 1
            },
            payload_length: null
          }
          wsRef.current.send(JSON.stringify(startMessage) + '\n')
          console.log('🎤 Sent audio-start message')
        } catch (error) {
          console.error('Failed to send audio-start:', error)
          throw error
        }
      } else {
        throw new Error('WebSocket not connected')
      }

      setIsRecording(true)
      setRecordingDuration(0)
      
      // Start duration timer
      durationIntervalRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1)
      }, 1000)

      console.log('🎤 Recording started')

    } catch (error) {
      console.error('Failed to start recording:', error)
      setError(error instanceof Error ? error.message : 'Failed to start recording')
      setConnectionStatus('error')
    }
  }

  const stopRecording = () => {
    try {
      // Send Wyoming protocol stop message
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        try {
          const stopMessage = {
            type: 'audio-stop',
            data: {
              timestamp: Date.now()
            },
            payload_length: null
          }
          wsRef.current.send(JSON.stringify(stopMessage) + '\n')
          console.log('🎤 Sent audio-stop message')
        } catch (error) {
          console.error('Failed to send audio-stop:', error)
        }
      }

      // Clean up media stream
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop())
        mediaStreamRef.current = null
      }

      // Clean up audio context
      if (processorRef.current) {
        processorRef.current.disconnect()
        processorRef.current = null
      }

      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }

      analyserRef.current = null

      // Clear duration timer
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current)
        durationIntervalRef.current = undefined
      }

      setIsRecording(false)
      console.log('🎤 Recording stopped')

    } catch (error) {
      console.error('Error stopping recording:', error)
      setError('Error stopping recording')
    }
  }

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
  }

  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <Wifi className="h-5 w-5 text-green-500" />
      case 'connecting':
        return <Radio className="h-5 w-5 text-yellow-500 animate-pulse" />
      case 'error':
        return <WifiOff className="h-5 w-5 text-red-500" />
      default:
        return <WifiOff className="h-5 w-5 text-gray-500" />
    }
  }

  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected'
      case 'connecting':
        return 'Connecting...'
      case 'error':
        return 'Connection Error'
      default:
        return 'Disconnected'
    }
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isRecording) {
        stopRecording()
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [isRecording])

  return (
    <div>
      {/* Header */}
      <div className="flex items-center space-x-2 mb-6">
        <Radio className="h-6 w-6 text-blue-600" />
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Live Audio Recording
        </h1>
      </div>

      {/* Microphone Access Warning */}
      {!canAccessMicrophone && (
        <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-4 mb-6">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="h-5 w-5 text-orange-600 dark:text-orange-400 mt-0.5 flex-shrink-0" />
            <div>
              <h3 className="font-medium text-orange-800 dark:text-orange-200 mb-2">
                Secure Access Required for Microphone
              </h3>
              <p className="text-sm text-orange-700 dark:text-orange-300">
                For security reasons, microphone access requires either:
              </p>
              <ul className="text-sm text-orange-700 dark:text-orange-300 list-disc ml-4 mt-2">
                <li><strong>Localhost access:</strong> <code className="bg-orange-100 dark:bg-orange-800 px-1 py-0.5 rounded text-xs">http://localhost/live-record</code></li>
                <li><strong>HTTPS connection:</strong> <code className="bg-orange-100 dark:bg-orange-800 px-1 py-0.5 rounded text-xs">https://{window.location.host}/live-record</code></li>
              </ul>
              <p className="text-sm text-orange-700 dark:text-orange-300 mt-2">
                Run <code className="bg-orange-100 dark:bg-orange-800 px-1 py-0.5 rounded text-xs">./init.sh {window.location.hostname}</code> to set up HTTPS access.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Connection Status */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {getStatusIcon()}
            <div>
              <h3 className="font-medium text-gray-900 dark:text-gray-100">
                Backend Connection
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {getStatusText()}
              </p>
            </div>
          </div>
          
          <div className="text-right">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              User: {user?.name || user?.email}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Endpoint: /ws_pcm
            </p>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
          <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}

      {/* Recording Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 mb-6">
        <div className="text-center">
          <div className="mb-6">
            {isRecording ? (
              <button
                onClick={stopRecording}
                className="w-20 h-20 bg-red-600 hover:bg-red-700 text-white rounded-full flex items-center justify-center transition-colors shadow-lg"
              >
                <MicOff className="h-8 w-8" />
              </button>
            ) : (
              <button
                onClick={startRecording}
                disabled={!canAccessMicrophone || connectionStatus === 'connecting'}
                className="w-20 h-20 bg-blue-600 hover:bg-blue-700 text-white rounded-full flex items-center justify-center transition-colors shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Mic className="h-8 w-8" />
              </button>
            )}
          </div>

          <div className="space-y-2">
            <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {isRecording ? 'Recording...' : 'Ready to Record'}
            </p>
            
            {isRecording && (
              <p className="text-2xl font-mono text-blue-600 dark:text-blue-400">
                {formatDuration(recordingDuration)}
              </p>
            )}
            
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {isRecording 
                ? `Audio streaming via ${isHttps ? 'WSS (secure)' : 'WS'} to backend for processing`
                : canAccessMicrophone 
                  ? 'Click the microphone to start recording'
                  : 'Secure connection required for microphone access'}
            </p>
          </div>
        </div>
      </div>

      {/* Audio Visualizer */}
      <AudioVisualizer 
        isRecording={isRecording}
        analyser={analyserRef.current}
      />

      {/* Instructions */}
      <div className="mt-8 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <h3 className="font-medium text-blue-800 dark:text-blue-200 mb-2">
          📝 How it Works
        </h3>
        <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
          <li>• Audio is captured from your microphone and streamed in real-time</li>
          <li>• Uses Wyoming protocol for structured communication with the backend</li>
          <li>• Audio is processed for transcription and memory extraction in the background</li>
          <li>• No real-time transcription display - check Conversations page for results</li>
          <li>• 16kHz mono audio with noise suppression and echo cancellation</li>
        </ul>
      </div>
    </div>
  )
}