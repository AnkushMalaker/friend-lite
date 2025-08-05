export interface AudioSegment {
  start: number
  end: number
  duration: number
}

export function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.floor(seconds % 60)
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
}

export function formatTime(seconds: number): string {
  return `${seconds.toFixed(2)}s`
}

export async function loadAudioBuffer(file: File): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as ArrayBuffer)
    reader.onerror = () => reject(new Error('Failed to read audio file'))
    reader.readAsArrayBuffer(file)
  })
}

export function createAudioContext(): AudioContext {
  const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext
  return new AudioContextClass()
}

export async function decodeAudioData(audioContext: AudioContext, arrayBuffer: ArrayBuffer): Promise<AudioBuffer> {
  try {
    return await audioContext.decodeAudioData(arrayBuffer)
  } catch (error) {
    throw new Error('Failed to decode audio data')
  }
}

export function getAudioInfo(audioBuffer: AudioBuffer) {
  return {
    duration: audioBuffer.duration,
    sampleRate: audioBuffer.sampleRate,
    channels: audioBuffer.numberOfChannels,
    length: audioBuffer.length
  }
}

export function extractAudioSamples(audioBuffer: AudioBuffer, channel = 0): Float32Array {
  return audioBuffer.getChannelData(channel)
}

export function calculateRMS(samples: Float32Array): number {
  let sum = 0
  for (let i = 0; i < samples.length; i++) {
    sum += samples[i] * samples[i]
  }
  return Math.sqrt(sum / samples.length)
}

export function calculateSNR(samples: Float32Array): number {
  const rms = calculateRMS(samples)
  
  // Find maximum absolute value iteratively to avoid stack overflow
  let max = 0
  for (let i = 0; i < samples.length; i++) {
    const absValue = Math.abs(samples[i])
    if (absValue > max) {
      max = absValue
    }
  }
  
  if (rms === 0 || max === 0) return 0
  
  const snr = 20 * Math.log10(max / rms)
  return Math.max(0, Math.min(60, snr)) // Clamp between 0-60 dB
}

export function detectSpeechSegments(samples: Float32Array, sampleRate: number): AudioSegment[] {
  const windowSize = Math.floor(sampleRate * 0.1) // 100ms windows
  const threshold = 0.01 // Adjust based on your needs
  const minSegmentLength = 0.5 // Minimum 0.5 seconds
  
  // For very large files, limit processing to reduce memory usage
  const maxProcessingLength = sampleRate * 600 // Max 10 minutes
  const samplesToProcess = samples.length > maxProcessingLength ? 
    samples.slice(0, maxProcessingLength) : samples
  
  if (samples.length > maxProcessingLength) {
    console.warn(`Audio file is very long (${(samples.length / sampleRate / 60).toFixed(1)} minutes). Only processing first 10 minutes for speech detection.`)
  }
  
  const segments: AudioSegment[] = []
  let inSpeech = false
  let segmentStart = 0
  
  for (let i = 0; i < samplesToProcess.length; i += windowSize) {
    const end = Math.min(i + windowSize, samplesToProcess.length)
    const window = samplesToProcess.slice(i, end)
    const rms = calculateRMS(window)
    
    const timePos = i / sampleRate
    
    if (rms > threshold && !inSpeech) {
      // Start of speech
      inSpeech = true
      segmentStart = timePos
    } else if (rms <= threshold && inSpeech) {
      // End of speech
      inSpeech = false
      const duration = timePos - segmentStart
      
      if (duration >= minSegmentLength) {
        segments.push({
          start: segmentStart,
          end: timePos,
          duration
        })
      }
    }
  }
  
  // Handle case where speech continues to end of processed section
  if (inSpeech) {
    const duration = (samplesToProcess.length / sampleRate) - segmentStart
    if (duration >= minSegmentLength) {
      segments.push({
        start: segmentStart,
        end: samplesToProcess.length / sampleRate,
        duration
      })
    }
  }
  
  return segments
}

export function createAudioBlob(samples: Float32Array, sampleRate: number): Blob {
  // Convert float32 samples to 16-bit PCM
  const int16Array = new Int16Array(samples.length)
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]))
    int16Array[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF
  }
  
  // Create WAV file header
  const length = int16Array.length
  const buffer = new ArrayBuffer(44 + length * 2)
  const view = new DataView(buffer)
  
  // WAV header
  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i))
    }
  }
  
  writeString(0, 'RIFF')
  view.setUint32(4, 36 + length * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeString(36, 'data')
  view.setUint32(40, length * 2, true)
  
  // PCM data
  const offset = 44
  for (let i = 0; i < length; i++) {
    view.setInt16(offset + i * 2, int16Array[i], true)
  }
  
  return new Blob([buffer], { type: 'audio/wav' })
}

export function extractAudioSegment(
  audioBuffer: AudioBuffer, 
  startTime: number, 
  endTime: number
): Float32Array {
  /**
   * Extract a segment of audio samples from the audio buffer
   * @param audioBuffer - Source audio buffer
   * @param startTime - Start time in seconds
   * @param endTime - End time in seconds
   * @returns Float32Array containing the extracted samples
   */
  const startSample = Math.floor(startTime * audioBuffer.sampleRate)
  const endSample = Math.ceil(endTime * audioBuffer.sampleRate)
  const segmentLength = endSample - startSample
  
  // Ensure we don't go beyond buffer bounds
  const actualStartSample = Math.max(0, startSample)
  const actualEndSample = Math.min(audioBuffer.length, endSample)
  const actualLength = actualEndSample - actualStartSample
  
  const channelData = audioBuffer.getChannelData(0)
  const segment = new Float32Array(actualLength)
  
  for (let i = 0; i < actualLength; i++) {
    segment[i] = channelData[actualStartSample + i]
  }
  
  return segment
}

export async function convertBlobToWav(blob: Blob): Promise<Blob> {
  /**
   * Convert any audio blob (WebM, MP3, etc.) to WAV format using Web Audio API
   */
  try {
    // Read the blob as ArrayBuffer
    const arrayBuffer = await blob.arrayBuffer()
    
    // Create audio context and decode the audio data
    const audioContext = createAudioContext()
    const audioBuffer = await decodeAudioData(audioContext, arrayBuffer)
    
    // Extract samples and create WAV blob using existing function
    const samples = extractAudioSamples(audioBuffer)
    const wavBlob = createAudioBlob(samples, audioBuffer.sampleRate)
    
    return wavBlob
  } catch (error) {
    console.error('Failed to convert blob to WAV:', error)
    throw new Error('Audio conversion failed')
  }
}