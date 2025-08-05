/**
 * Custom hook for speaker identification functionality
 */

import { useCallback } from 'react'
import { useUser } from '../contexts/UserContext'
import { apiService } from '../services/api'
import { createWAVBlob } from '../utils/audioUtils'
import { speakerLogger } from '../utils/logger'

export interface IdentifyResult {
  found: boolean
  speaker_id: string | null
  speaker_name: string | null
  confidence: number
  status: string
  similarity_threshold: number
  duration: number
}

export interface UseSpeakerIdentificationReturn {
  identifyUtteranceSpeaker: (audioBuffer: Float32Array, sampleRate: number) => Promise<IdentifyResult>
}

export function useSpeakerIdentification(): UseSpeakerIdentificationReturn {
  const { user } = useUser()

  const identifyUtteranceSpeaker = useCallback(async (
    audioBuffer: Float32Array, 
    sampleRate: number
  ): Promise<IdentifyResult> => {
    try {
      speakerLogger.info('Starting utterance speaker identification')
      
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
      
      speakerLogger.info('API Response:', response.data)
      
      return response.data as IdentifyResult
      
    } catch (error) {
      speakerLogger.error('Error during utterance identification:', error)
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
  }, [user])

  return {
    identifyUtteranceSpeaker
  }
}