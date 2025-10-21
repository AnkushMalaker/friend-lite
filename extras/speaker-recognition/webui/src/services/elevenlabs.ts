/**
 * ElevenLabs Service - Transcription and speaker diarization via ElevenLabs API
 * Provides batch transcription with speaker diarization (up to 32 speakers)
 * Word-level timestamps with confidence scores
 */

import { apiService } from './api'

export interface ElevenLabsWord {
  text: string
  start: number
  end: number
  type: 'word' | 'spacing' | 'audio_event' | 'punctuation'
  logprob: number
  speaker_id?: string
}

export interface ElevenLabsResponse {
  language_code: string
  language_probability: number
  text: string
  words: ElevenLabsWord[]
  transcription_id: string
}

export interface ElevenLabsSegment {
  start: number
  end: number
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

export interface ElevenLabsOptions {
  enhanceSpeakers?: boolean
  userId?: number
  speakerConfidenceThreshold?: number
  model?: string
  language?: string
  numSpeakers?: number
}

/**
 * Transcribe audio using ElevenLabs Scribe API with optional speaker enhancement
 */
export async function transcribeWithElevenLabs(
  audioFile: File | Blob,
  options: ElevenLabsOptions = {}
): Promise<ElevenLabsResponse> {
  try {
    const formData = new FormData()
    formData.append('file', audioFile, audioFile instanceof File ? audioFile.name : 'audio.wav')
    formData.append('model_id', options.model || 'scribe_v1')

    if (options.language) {
      formData.append('language', options.language)
    }

    // Enable speaker diarization
    formData.append('enable_speaker_diarization', 'true')

    if (options.numSpeakers) {
      formData.append('num_speakers', options.numSpeakers.toString())
    }

    // Determine endpoint based on enhancement option
    const endpoint = options.enhanceSpeakers
      ? '/elevenlabs/v1/transcribe'
      : 'https://api.elevenlabs.io/v1/speech-to-text'

    // Add speaker enhancement parameters if using wrapper endpoint
    const params: Record<string, string> = {}
    if (options.enhanceSpeakers) {
      params.enhance_speakers = 'true'
      if (options.userId) {
        params.user_id = options.userId.toString()
      }
      if (options.speakerConfidenceThreshold !== undefined) {
        params.similarity_threshold = options.speakerConfidenceThreshold.toString()
      }
    }

    const response = await apiService.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        ...(endpoint.includes('elevenlabs.io') && {
          'xi-api-key': import.meta.env.VITE_ELEVENLABS_API_KEY || ''
        })
      },
      params,
      timeout: 180000
    })

    return response.data as ElevenLabsResponse
  } catch (error) {
    throw new Error(`ElevenLabs transcription failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

/**
 * Process ElevenLabs response into speaker segments
 */
export function processElevenLabsResponse(response: ElevenLabsResponse): ElevenLabsSegment[] {
  const segments: ElevenLabsSegment[] = []

  // Filter to only word types and group by speaker_id
  const words = response.words.filter(w => w.type === 'word')

  if (words.length === 0) {
    return segments
  }

  let currentSegment: ElevenLabsSegment | null = null

  for (const word of words) {
    const speakerId = word.speaker_id || 'speaker_0'
    const speakerNum = parseInt(speakerId.replace('speaker_', '')) || 0

    // Convert logprob to confidence (logprob is typically 0 to -1, where 0 is highest confidence)
    const confidence = logprobToConfidence(word.logprob)

    if (!currentSegment || currentSegment.speaker !== speakerNum) {
      // Start new segment
      if (currentSegment) {
        segments.push(currentSegment)
      }

      currentSegment = {
        start: word.start,
        end: word.end,
        speaker: speakerNum,
        speakerId: speakerId,
        speakerName: `Speaker ${speakerNum}`,
        confidence: confidence,
        text: word.text
      }
    } else {
      // Continue current segment
      currentSegment.end = word.end
      currentSegment.text += word.text
      // Update confidence as running average
      currentSegment.confidence = (currentSegment.confidence + confidence) / 2
    }
  }

  // Push last segment
  if (currentSegment) {
    segments.push(currentSegment)
  }

  return segments
}

/**
 * Convert ElevenLabs logprob to confidence score (0-1)
 * logprob is typically in range [0, -1] where 0 is highest confidence
 */
function logprobToConfidence(logprob: number): number {
  // ElevenLabs logprob: 0 = perfect confidence, negative = lower confidence
  // Convert to 0-1 scale where 1 = perfect confidence
  return 1.0 - Math.min(Math.abs(logprob), 1.0)
}

/**
 * Calculate confidence summary statistics for segments
 */
export function calculateConfidenceSummary(segments: ElevenLabsSegment[]): {
  total_segments: number
  high_confidence: number
  medium_confidence: number
  low_confidence: number
} {
  return {
    total_segments: segments.length,
    high_confidence: segments.filter(s => s.confidence >= 0.8).length,
    medium_confidence: segments.filter(s => s.confidence >= 0.6 && s.confidence < 0.8).length,
    low_confidence: segments.filter(s => s.confidence >= 0.4 && s.confidence < 0.6).length
  }
}
