/**
 * WebSocket service for real-time transcription with speaker change detection
 * Connects to the new WebSocket wrapper endpoint on the server
 */

export interface UtteranceBoundaryEvent {
  type: 'utterance_boundary'
  timestamp: number
  audio_segment: {
    start: number
    end: number
    duration: number
  }
  transcript: string
  speaker_identification?: {
    speaker_id: string | null
    speaker_name: string | null
    confidence: number
    status: 'identified' | 'unknown' | 'error'
    error?: string
  }
}

export interface SpeakerWebSocketEvent {
  type: 'ready' | 'utterance_boundary' | 'error' | 'raw_deepgram'
  message?: string
  timestamp?: number
  data?: any  // For raw_deepgram events
  audio_segment?: {
    start: number
    end: number
    duration: number
  }
  transcript?: string
  speaker_identification?: {
    speaker_id: string | null
    speaker_name: string | null
    confidence: number
    status: 'identified' | 'unknown' | 'error'
    error?: string
  }
}

export interface SpeakerWebSocketOptions {
  userId?: number
  confidenceThreshold?: number
  deepgramApiKey?: string
  onUtteranceBoundary?: (event: UtteranceBoundaryEvent) => void
  onReady?: () => void
  onError?: (error: string) => void
  onConnectionStatusChange?: (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void
  onRawDeepgram?: (data: any) => void
}

export class SpeakerWebSocketService {
  private ws: WebSocket | null = null
  private options: SpeakerWebSocketOptions
  private connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error' = 'disconnected'
  private baseUrl: string
  private fallbackUrls: string[] = []

  constructor(options: SpeakerWebSocketOptions = {}) {
    this.options = options
    
    // Use nginx proxy for WebSocket connections (HTTPS requirement)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    this.baseUrl = `${protocol}//${host}`
  }

  private setConnectionStatus(status: 'connecting' | 'connected' | 'disconnected' | 'error') {
    console.log(`🔄 [WS Service] Setting connection status: ${this.connectionStatus} → ${status}`)
    this.connectionStatus = status
    this.options.onConnectionStatusChange?.(status)
  }

  async connect(): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected')
      return
    }

    this.setConnectionStatus('connecting')

    // Build WebSocket URL with query parameters (excluding API key)
    const params = new URLSearchParams()
    if (this.options.userId) {
      params.set('user_id', this.options.userId.toString())
    }
    if (this.options.confidenceThreshold !== undefined) {
      params.set('confidence_threshold', this.options.confidenceThreshold.toString())
    }

    const wsUrl = `${this.baseUrl}/ws/streaming-with-scd?${params.toString()}`
    
    console.log(`🔌 Connecting to Speaker WebSocket: ${wsUrl}`)
    console.log(`🔧 Base URL: ${this.baseUrl}`)
    console.log(`🔧 Parameters: ${params.toString()}`)

    return new Promise((resolve, reject) => {
      let connectionResolved = false
      
      try {
        // Use WebSocket subprotocols for API key authentication (like Deepgram does)
        const protocols = this.options.deepgramApiKey ? ['token', this.options.deepgramApiKey] : undefined
        this.ws = new WebSocket(wsUrl, protocols)

        this.ws.onopen = () => {
          console.log('✅ Speaker WebSocket connected - waiting for ready message')
          this.setConnectionStatus('connected')
          // Don't resolve immediately - wait for ready message
        }

        this.ws.onmessage = (event) => {
          try {
            const data: SpeakerWebSocketEvent = JSON.parse(event.data)
            this.handleMessage(data)
            
            // Resolve promise when ready message is received
            if (data.type === 'ready' && !connectionResolved) {
              connectionResolved = true
              resolve()
            }
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error)
            if (!connectionResolved) {
              connectionResolved = true
              reject(error)
            }
          }
        }

        this.ws.onclose = (event) => {
          console.log(`🔌 Speaker WebSocket closed: ${event.code} ${event.reason}`)
          this.setConnectionStatus('disconnected')
          this.ws = null
        }

        this.ws.onerror = (error) => {
          console.error('❌ Speaker WebSocket error:', error)
          this.setConnectionStatus('error')
          if (!connectionResolved) {
            connectionResolved = true
            reject(error)
          }
        }
      } catch (error) {
        this.setConnectionStatus('error')
        reject(error)
      }
    })
  }

  private handleMessage(data: SpeakerWebSocketEvent) {
    console.log('📨 Received message:', data.type)

    switch (data.type) {
      case 'ready':
        console.log('🟢 WebSocket ready for audio streaming')
        this.options.onReady?.()
        break

      case 'utterance_boundary':
        if (data.transcript && data.audio_segment) {
          const event: UtteranceBoundaryEvent = {
            type: 'utterance_boundary',
            timestamp: data.timestamp || Date.now(),
            audio_segment: data.audio_segment,
            transcript: data.transcript,
            speaker_identification: data.speaker_identification
          }
          
          console.log(`🎙️ Utterance boundary: "${event.transcript}" (${event.audio_segment.duration.toFixed(2)}s)`)
          if (event.speaker_identification?.speaker_name) {
            console.log(`👤 Speaker: ${event.speaker_identification.speaker_name} (${event.speaker_identification.confidence.toFixed(3)})`)
          }

          this.options.onUtteranceBoundary?.(event)
        }
        break

      case 'error':
        console.error('❌ Server error:', data.message)
        this.options.onError?.(data.message || 'Unknown server error')
        break

      case 'raw_deepgram':
        console.log('🎤 Raw Deepgram event:', data.data?.type || 'unknown')
        this.options.onRawDeepgram?.(data.data)
        break

      default:
        console.warn('Unknown message type:', data)
    }
  }

  sendAudio(audioData: ArrayBuffer | Uint8Array): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('⚠️ Cannot send audio: WebSocket not connected')
      return false
    }

    try {
      this.ws.send(audioData)
      return true
    } catch (error) {
      console.error('❌ Failed to send audio:', error)
      return false
    }
  }

  disconnect(): void {
    if (this.ws) {
      console.log('🔌 Disconnecting Speaker WebSocket')
      this.ws.close()
      this.ws = null
    }
    this.setConnectionStatus('disconnected')
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  get status(): 'connecting' | 'connected' | 'disconnected' | 'error' {
    return this.connectionStatus
  }

  // Update options
  updateOptions(newOptions: Partial<SpeakerWebSocketOptions>): void {
    this.options = { ...this.options, ...newOptions }
  }
}