import { apiService } from './api'

export interface DeepgramSegment {
  start: number
  end: number
  duration: number
  speaker: number
  speakerId?: string
  speakerName?: string
  confidence: number
  text: string
  identifiedSpeakerId?: string
  identifiedSpeakerName?: string
  speakerIdentificationConfidence?: number
  speakerStatus?: string
}

export interface DeepgramTranscriptionOptions {
  model?: string
  language?: string
  diarize?: boolean
  smartFormat?: boolean
  punctuate?: boolean
  enhanceSpeakers?: boolean
  userId?: number
  speakerConfidenceThreshold?: number
  mode?: 'standard' | 'hybrid'
}

export interface DeepgramResponse {
  results: {
    channels: Array<{
      alternatives: Array<{
        words: Array<{
          start: number
          end: number
          word: string
          punctuated_word?: string
          speaker?: number
          confidence?: number
          identified_speaker_id?: string
          identified_speaker_name?: string
          speaker_identification_confidence?: number
          speaker_status?: string
        }>
        transcript: string
      }>
    }>
  }
  speaker_enhancement?: {
    enabled: boolean
    method?: string
    user_id?: number
    confidence_threshold?: number
    identified_speakers?: any[]
    total_segments?: number
    identified_segments?: number
    total_speakers?: number
    identified_count?: number
  }
}

/**
 * Default options for Deepgram transcription requests
 */
export const DEFAULT_DEEPGRAM_OPTIONS: DeepgramTranscriptionOptions = {
  model: 'nova-3',
  language: 'multi',
  diarize: true,
  smartFormat: true,
  punctuate: true,
  enhanceSpeakers: true,
  speakerConfidenceThreshold: 0.15,
  mode: 'standard'
}

/**
 * Transcribe audio file using Deepgram API
 */
export async function transcribeWithDeepgram(
  file: File | Blob,
  options: DeepgramTranscriptionOptions = {}
): Promise<DeepgramResponse> {
  const opts = { ...DEFAULT_DEEPGRAM_OPTIONS, ...options }
  
  const formData = new FormData()
  formData.append('file', file)
  
  // Choose endpoint based on mode
  const endpoint = opts.mode === 'hybrid' ? '/v1/transcribe-and-diarize' : '/v1/listen'
  
  const params: Record<string, any> = {
    model: opts.model,
    language: opts.language,
    diarize: opts.diarize,
    smart_format: opts.smartFormat,
    punctuate: opts.punctuate
  }
  
  // Add enhancement parameters if enabled
  if (opts.enhanceSpeakers) {
    params.enhance_speakers = true
    params.user_id = opts.userId
    params.speaker_confidence_threshold = opts.speakerConfidenceThreshold
  }
  
  // Additional params for hybrid mode
  if (opts.mode === 'hybrid') {
    params.similarity_threshold = opts.speakerConfidenceThreshold
    params.min_duration = 1.0
  }
  
  const response = await apiService.post(endpoint, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    params,
    timeout: 300000, // 5 minutes for transcription operations
  })
  
  return response.data as DeepgramResponse
}

/**
 * Group words from Deepgram response by speaker into segments
 */
export function groupWordsBySpeaker(words: DeepgramResponse['results']['channels'][0]['alternatives'][0]['words']): DeepgramSegment[] {
  if (!words || words.length === 0) return []
  
  const segments: DeepgramSegment[] = []
  let currentSegment: any = null
  
  for (const word of words) {
    const speaker = word.speaker || 0
    
    if (!currentSegment || currentSegment.speaker !== speaker) {
      // Start new segment
      if (currentSegment) {
        segments.push(currentSegment)
      }
      currentSegment = {
        start: word.start,
        end: word.end,
        duration: 0, // Will be calculated when segment ends
        speaker,
        speakerId: word.identified_speaker_id || `speaker_${speaker}`,
        speakerName: word.identified_speaker_name || `Speaker ${speaker}`,
        confidence: word.speaker_identification_confidence || 0,
        text: word.punctuated_word || word.word,
        identifiedSpeakerId: word.identified_speaker_id,
        identifiedSpeakerName: word.identified_speaker_name,
        speakerIdentificationConfidence: word.speaker_identification_confidence,
        speakerStatus: word.speaker_status
      }
    } else {
      // Continue current segment
      currentSegment.end = word.end
      currentSegment.text += ' ' + (word.punctuated_word || word.word)
    }
  }
  
  // Add final segment
  if (currentSegment) {
    segments.push(currentSegment)
  }
  
  // Calculate durations
  segments.forEach(segment => {
    segment.duration = segment.end - segment.start
  })
  
  return segments
}

/**
 * Process Deepgram response and extract speaker segments
 */
export function processDeepgramResponse(response: DeepgramResponse): DeepgramSegment[] {
  const results = response.results || {}
  const channels = results.channels || []
  
  if (!channels.length) return []
  
  const channel = channels[0]
  const alternatives = channel.alternatives || []
  
  if (!alternatives.length) return []
  
  const words = alternatives[0].words || []
  return groupWordsBySpeaker(words)
}

/**
 * Calculate confidence summary for segments
 */
export function calculateConfidenceSummary(segments: DeepgramSegment[]) {
  const total = segments.length
  const high = segments.filter(s => s.confidence >= 0.8).length
  const medium = segments.filter(s => s.confidence >= 0.6 && s.confidence < 0.8).length
  const low = segments.filter(s => s.confidence >= 0.4 && s.confidence < 0.6).length
  const veryLow = total - high - medium - low
  
  return {
    total_segments: total,
    high_confidence: high,
    medium_confidence: medium,
    low_confidence: low,
    very_low_confidence: veryLow
  }
}

/**
 * Convert Deepgram segments to annotation format
 * Used by the Annotation page to convert transcription results
 */
export function convertToAnnotationSegments(segments: DeepgramSegment[]): Array<{
  id: string
  start: number
  end: number
  duration: number
  speakerLabel?: string
  deepgramSpeakerLabel?: string
  label: 'CORRECT' | 'INCORRECT' | 'UNCERTAIN'
  confidence?: number
  transcription?: string
}> {
  return segments.map(segment => ({
    id: Math.random().toString(36),
    start: segment.start,
    end: segment.end,
    duration: segment.duration,
    speakerLabel: segment.speakerName,
    deepgramSpeakerLabel: `speaker_${segment.speaker}`,
    label: 'UNCERTAIN' as const,
    confidence: segment.confidence,
    transcription: segment.text
  }))
}

// ============================================================================
// STREAMING SUPPORT
// ============================================================================

export interface StreamingConfig {
  apiKey: string
  model?: string
  language?: string
  punctuate?: boolean
  diarize?: boolean
  interim_results?: boolean
  endpointing?: number
  vad_events?: boolean
  utterance_end_ms?: number
  encoding?: string
  sample_rate?: number
}

export interface StreamingTranscript {
  transcript: string
  confidence: number
  words: Array<{
    word: string
    start: number
    end: number
    confidence: number
    speaker?: number
    punctuated_word?: string
  }>
  is_final: boolean
}

export class DeepgramStreaming {
  private ws: WebSocket | null = null
  private isConnected = false
  private config: StreamingConfig
  private onTranscriptCallback?: (transcript: StreamingTranscript) => void
  private onErrorCallback?: (error: Error) => void
  private onStatusCallback?: (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void
  private connectionTime?: number
  private firstAudioTime?: number
  private audioPacketCount = 0

  constructor(config: StreamingConfig) {
    this.config = {
      model: 'nova-2',
      language: 'en-US',
      punctuate: true,
      diarize: true,
      interim_results: true,
      endpointing: 300,
      vad_events: true,
      utterance_end_ms: 1000,
      encoding: 'linear16',
      sample_rate: 16000,
      ...config
    }
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log('🌐 [DEEPGRAM] Connecting to Deepgram...')
        this.onStatusCallback?.('connecting')

        // Build WebSocket URL with parameters (no API key in URL for security)
        const params = new URLSearchParams({
          model: this.config.model!,
          language: this.config.language!,
          smart_format: this.config.punctuate!.toString(),  // Fixed: was 'punctuate'
          diarize: this.config.diarize!.toString(),
          interim_results: this.config.interim_results!.toString(),
          endpointing: this.config.endpointing!.toString(),
          vad_events: this.config.vad_events!.toString(),
          utterance_end_ms: this.config.utterance_end_ms!.toString(),
          encoding: this.config.encoding!,
          sample_rate: this.config.sample_rate!.toString(),
          channels: '1'
        })

        const wsUrl = `wss://api.deepgram.com/v1/listen?${params.toString()}`

        // Create WebSocket using Sec-WebSocket-Protocol header for browser authentication
        // This is the correct way to authenticate with Deepgram from browsers
        this.ws = new WebSocket(wsUrl, ['token', this.config.apiKey])

        this.ws.onopen = () => {
          console.log('🌐 [DEEPGRAM] ✅ Connected!')
          this.isConnected = true
          this.connectionTime = Date.now()
          this.onStatusCallback?.('connected')
          resolve()
        }

        this.ws.onclose = (event) => {
          console.log('🌐 [DEEPGRAM] WebSocket connection closed:', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean
          })
          console.log('🌐 [DEEPGRAM] Setting isConnected = false')
          this.isConnected = false
          console.log('🌐 [DEEPGRAM] Calling status callback with "disconnected"')
          this.onStatusCallback?.('disconnected')
        }

        this.ws.onerror = (error) => {
          console.error('🌐 [DEEPGRAM] ❌ WebSocket error occurred:', error)
          console.error('🌐 [DEEPGRAM] Error event details:', {
            type: error.type,
            target: error.target?.readyState,
            currentTarget: error.currentTarget?.readyState
          })
          console.log('🌐 [DEEPGRAM] Setting isConnected = false due to error')
          this.isConnected = false
          console.log('🌐 [DEEPGRAM] Calling status callback with "error"')
          this.onStatusCallback?.('error')
          
          const err = new Error(`WebSocket connection failed. Check your Deepgram API key and internet connection.`)
          console.error('🌐 [DEEPGRAM] Calling error callback with error:', err.message)
          this.onErrorCallback?.(err)
          console.error('🌐 [DEEPGRAM] Rejecting connection promise with error')
          reject(err)
        }

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            
            // More focused logging for different message types
            if (data.type === 'Results') {
              console.log('📝 [DEEPGRAM] TRANSCRIPT:', data.channel?.alternatives?.[0]?.transcript || '[empty]', {
                is_final: data.is_final,
                confidence: data.channel?.alternatives?.[0]?.confidence,
                words: data.channel?.alternatives?.[0]?.words?.length || 0
              })
            } else if (data.type === 'Metadata') {
              console.log('🌐 [DEEPGRAM] Connected - ready for audio')
            } else {
              console.log('🌐 [DEEPGRAM]', data.type + ':', data)
            }
            
            // Log timing for connection health
            if (this.connectionTime) {
              const timeSinceConnection = Date.now() - this.connectionTime
              console.log('🌐 [DEEPGRAM] ⏱️ Time since connection:', timeSinceConnection + 'ms')
            }
            
            // Handle different message types
            if (data.type === 'Results' && data.channel?.alternatives?.[0]) {
              console.log('🌐 [DEEPGRAM] Processing Results message')
              const transcript = data.channel.alternatives[0]
              console.log('🌐 [DEEPGRAM] Transcript data:', {
                text: transcript.transcript,
                confidence: transcript.confidence,
                is_final: data.is_final,
                words_count: transcript.words?.length || 0
              })
              
              // Only emit if we have actual content
              if (transcript.transcript && transcript.transcript.trim()) {
                console.log('🌐 [DEEPGRAM] Emitting transcript to callback')
                this.onTranscriptCallback?.({
                  transcript: transcript.transcript,
                  confidence: transcript.confidence || 0,
                  words: transcript.words || [],
                  is_final: data.is_final || false
                })
              } else {
                console.log('🌐 [DEEPGRAM] Skipping empty transcript')
              }
            } else if (data.type === 'Metadata') {
              console.log('🌐 [DEEPGRAM] 📋 Received metadata:', {
                request_id: data.request_id,
                created: data.created,
                model_info: data.model_info,
                channels: data.channels,
                models: data.models
              })
              console.log('🌐 [DEEPGRAM] ✅ Connection ready - metadata confirms successful setup')
            } else if (data.type === 'UtteranceEnd') {
              console.log('🌐 [DEEPGRAM] 🔚 Utterance ended at', data.timestamp)
            } else if (data.type === 'SpeechStarted') {
              console.log('🌐 [DEEPGRAM] 🎙️ Speech started at', data.timestamp)
            } else if (data.type === 'Close') {
              console.log('🌐 [DEEPGRAM] 🔒 Close message received:', data)
            } else {
              console.log('🌐 [DEEPGRAM] ❓ Unknown message type:', data.type)
              console.log('🌐 [DEEPGRAM] Full unknown message:', data)
            }
          } catch (error) {
            console.error('🌐 [DEEPGRAM] ❌ Error parsing message:', error)
            console.error('🌐 [DEEPGRAM] Raw message data:', event.data)
            console.error('🌐 [DEEPGRAM] Parse error details:', {
              message: error instanceof Error ? error.message : String(error),
              stack: error instanceof Error ? error.stack : 'No stack'
            })
            const parseError = new Error(`Failed to parse message: ${error}`)
            console.error('🌐 [DEEPGRAM] Calling error callback with parse error')
            this.onErrorCallback?.(parseError)
          }
        }

        // Connection timeout
        console.log('🌐 [DEEPGRAM] Setting 10-second connection timeout')
        setTimeout(() => {
          if (!this.isConnected) {
            console.error('🌐 [DEEPGRAM] ⏰ Connection timeout after 10 seconds')
            console.error('🌐 [DEEPGRAM] WebSocket readyState at timeout:', this.ws?.readyState)
            const timeoutError = new Error('Connection timeout - Deepgram did not respond within 10 seconds')
            console.error('🌐 [DEEPGRAM] Rejecting with timeout error')
            reject(timeoutError)
          } else {
            console.log('🌐 [DEEPGRAM] Connection established before timeout')
          }
        }, 10000)

      } catch (error) {
        console.error('🌐 [DEEPGRAM] ❌ Exception in connect() method:', error)
        console.error('🌐 [DEEPGRAM] Exception details:', {
          message: error instanceof Error ? error.message : String(error),
          stack: error instanceof Error ? error.stack : 'No stack'
        })
        console.error('🌐 [DEEPGRAM] Rejecting with caught exception')
        reject(error)
      }
    })
  }

  disconnect(): void {
    console.log('🌐 [DEEPGRAM] Disconnect called')
    if (this.ws) {
      console.log('🌐 [DEEPGRAM] Closing WebSocket connection, current readyState:', this.ws.readyState)
      this.ws.close()
      this.ws = null
      console.log('🌐 [DEEPGRAM] WebSocket closed and reference cleared')
    } else {
      console.log('🌐 [DEEPGRAM] No WebSocket to close')
    }
    console.log('🌐 [DEEPGRAM] Setting isConnected = false')
    this.isConnected = false
  }

  sendAudio(audioData: ArrayBuffer): void {
    if (this.ws && this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      this.audioPacketCount++
      
      // Track first audio packet timing
      if (!this.firstAudioTime && this.connectionTime) {
        this.firstAudioTime = Date.now()
        const delayMs = this.firstAudioTime - this.connectionTime
        console.log('🌐 [DEEPGRAM] 🎵 First audio packet sent!')
        console.log('🌐 [DEEPGRAM] ⏱️ Time since connection:', delayMs + 'ms')
        console.log('🌐 [DEEPGRAM] Audio packet size:', audioData.byteLength, 'bytes')
        if (delayMs > 8000) {
          console.warn('🌐 [DEEPGRAM] ⚠️ Audio delay > 8s - may cause timeout!')
        }
      }
      
      // Log periodic audio packet info (every 50 packets to avoid spam)
      if (this.audioPacketCount % 50 === 0) {
        console.log('🌐 [DEEPGRAM] 📊 Audio packets sent:', this.audioPacketCount)
      }
      
      this.ws.send(audioData)
    } else {
      console.warn('🌐 [DEEPGRAM] ⚠️ Cannot send audio - WebSocket not connected:', {
        wsExists: !!this.ws,
        isConnected: this.isConnected,
        readyState: this.ws?.readyState,
        audioPacketCount: this.audioPacketCount
      })
    }
  }

  finishStream(): void {
    if (this.ws && this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      // Send empty message to signal end of stream
      this.ws.send(JSON.stringify({ type: 'CloseStream' }))
    }
  }

  onTranscript(callback: (transcript: StreamingTranscript) => void): void {
    this.onTranscriptCallback = callback
  }

  onError(callback: (error: Error) => void): void {
    this.onErrorCallback = callback
  }

  onStatus(callback: (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void): void {
    this.onStatusCallback = callback
  }

  getConnectionStatus(): 'connecting' | 'connected' | 'disconnected' | 'error' {
    if (!this.ws) return 'disconnected'
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting'
      case WebSocket.OPEN:
        return 'connected'
      case WebSocket.CLOSING:
      case WebSocket.CLOSED:
        return 'disconnected'
      default:
        return 'error'
    }
  }
}

/**
 * Utility function to convert audio samples to the format expected by Deepgram
 */
export function convertAudioForDeepgram(audioBuffer: Float32Array, sampleRate: number = 16000): ArrayBuffer {
  // Convert float32 samples to int16
  const int16Array = new Int16Array(audioBuffer.length)
  
  for (let i = 0; i < audioBuffer.length; i++) {
    // Clamp to [-1, 1] and convert to 16-bit integer
    const sample = Math.max(-1, Math.min(1, audioBuffer[i]))
    int16Array[i] = sample * 0x7FFF
  }
  
  return int16Array.buffer
}

/**
 * Resample audio to 16kHz if needed
 */
export function resampleAudio(
  audioBuffer: Float32Array, 
  originalSampleRate: number, 
  targetSampleRate: number = 16000
): Float32Array {
  if (originalSampleRate === targetSampleRate) {
    return audioBuffer
  }

  const ratio = originalSampleRate / targetSampleRate
  const newLength = Math.round(audioBuffer.length / ratio)
  const resampled = new Float32Array(newLength)

  for (let i = 0; i < newLength; i++) {
    const originalIndex = i * ratio
    const index = Math.floor(originalIndex)
    const fraction = originalIndex - index

    if (index + 1 < audioBuffer.length) {
      // Linear interpolation
      resampled[i] = audioBuffer[index] * (1 - fraction) + audioBuffer[index + 1] * fraction
    } else {
      resampled[i] = audioBuffer[index] || 0
    }
  }

  return resampled
}