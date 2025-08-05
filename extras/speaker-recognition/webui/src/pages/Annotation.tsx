import { useState, useCallback, useRef, useEffect } from 'react'
import { Play, Pause, Save, Download, CheckCircle, XCircle, AlertCircle, UserPlus } from 'lucide-react'
import { useUser } from '../contexts/UserContext'
import { calculateFileHash, isAudioFile } from '../utils/fileHash'
import { 
  loadAudioBuffer, 
  createAudioContext, 
  decodeAudioData,
  extractAudioSamples,
  formatDuration,
  createAudioBlob,
  extractAudioSegment,
  AudioSegment
} from '../utils/audioUtils'
import { databaseService } from '../services/database'
import { apiService, type Annotation } from '../services/api'
import { 
  transcribeWithDeepgram, 
  convertToAnnotationSegments,
  DEFAULT_DEEPGRAM_OPTIONS 
} from '../services/deepgram'
import FileUploader from '../components/FileUploader'
import WaveformPlot from '../components/WaveformPlot'

interface AudioData {
  file: File
  buffer: AudioBuffer
  hash: string
  samples: Float32Array
}

interface AnnotationSegment extends AudioSegment {
  id: string
  speakerId?: string
  speakerLabel?: string
  deepgramSpeakerLabel?: string
  label: 'CORRECT' | 'INCORRECT' | 'UNCERTAIN'
  confidence?: number
  transcription?: string
  notes?: string
}

interface AvailableSpeaker {
  id?: string  // Only for enrolled speakers
  label: string
}

export default function Annotation() {
  const { user } = useUser()
  const [audioData, setAudioData] = useState<AudioData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [segments, setSegments] = useState<AnnotationSegment[]>([])
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null)
  const [availableSpeakers, setAvailableSpeakers] = useState<AvailableSpeaker[]>([
    { label: 'Speaker 1' },
    { label: 'Speaker 2' },
    { label: 'Unknown' }
  ])
  const [newSpeakerName, setNewSpeakerName] = useState('')
  const [playingSegmentId, setPlayingSegmentId] = useState<string | null>(null)
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  const [enrollingSpeaker, setEnrollingSpeaker] = useState<string | null>(null)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [deepgramResponse, setDeepgramResponse] = useState<any>(null)
  const [showJsonOutput, setShowJsonOutput] = useState(false)
  
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null)

  // Load enrolled speakers on mount
  useEffect(() => {
    const loadEnrolledSpeakers = async () => {
      if (!user) return
      
      try {
        const response = await apiService.get('/speakers', {
          params: { user_id: user.id }
        })
        
        const enrolledSpeakers = response.data.speakers || []
        
        // Combine enrolled speakers with default temporary labels
        const allSpeakers: AvailableSpeaker[] = [
          ...enrolledSpeakers.map((s: any) => ({ id: s.id, label: s.name })),
          { label: 'Speaker 1' },
          { label: 'Speaker 2' },
          { label: 'Unknown' }
        ]
        
        setAvailableSpeakers(allSpeakers)
      } catch (error) {
        console.error('Failed to load enrolled speakers:', error)
        // Keep default speakers on error
      }
    }
    
    loadEnrolledSpeakers()
  }, [user])

  const handleDeepgramTranscribe = useCallback(async () => {
    if (!audioData || !user) return

    setIsTranscribing(true)
    try {
      // Use shared Deepgram service
      const response = await transcribeWithDeepgram(audioData.file, {
        enhanceSpeakers: true,
        userId: user.id,
        speakerConfidenceThreshold: 0.15,
        mode: 'standard'
      })

      // Store response for JSON debugging
      setDeepgramResponse(response)

      // Convert Deepgram response to annotation segments
      const results = response.results || {}
      const channels = results.channels || []
      
      if (!channels.length) {
        alert('No transcription results found')
        return
      }

      const words = channels[0]?.alternatives?.[0]?.words || []
      
      if (!words.length) {
        alert('No words found in transcription')
        return
      }

      // Debug: Log word count and speaker distribution
      console.log(`Total words received: ${words.length}`)
      const speakerCounts = {}
      words.forEach(word => {
        const speaker = word.speaker || 0
        speakerCounts[speaker] = (speakerCounts[speaker] || 0) + 1
      })
      console.log('Speaker distribution in words:', speakerCounts)

      // Group words by speaker into segments (similar to existing Deepgram JSON import logic)
      const newSegments: AnnotationSegment[] = []
      let currentSpeaker = null
      let currentSegmentStart = null
      let currentWords: any[] = []

      for (const word of words) {
        if (word.speaker !== currentSpeaker) {
          // Save previous segment
          if (currentSpeaker !== null && currentWords.length > 0) {
            const segmentEnd = currentWords[currentWords.length - 1].end
            // FIX: Use the first word of the CURRENT segment, not the new word
            const firstWordOfSegment = currentWords[0]
            newSegments.push({
              id: Math.random().toString(36),
              start: currentSegmentStart,
              end: segmentEnd,
              duration: segmentEnd - currentSegmentStart,
              speakerLabel: firstWordOfSegment.identified_speaker_name || `Speaker ${currentSpeaker + 1}`,
              deepgramSpeakerLabel: `speaker_${currentSpeaker}`,
              label: 'UNCERTAIN' as const,
              confidence: Math.min(...currentWords.map(w => w.confidence || 0)),
              transcription: currentWords.map(w => w.punctuated_word || w.word).join(' ')
            })
            console.log(`Created segment for speaker ${currentSpeaker}: ${currentSegmentStart}s - ${segmentEnd}s`)
          }

          // Start new segment
          currentSpeaker = word.speaker
          currentSegmentStart = word.start
          currentWords = [word]
        } else {
          currentWords.push(word)
        }
      }

      // Add final segment
      if (currentWords.length > 0) {
        const segmentEnd = currentWords[currentWords.length - 1].end
        // FIX: Use the first word of the CURRENT segment for speaker info
        const firstWordOfSegment = currentWords[0]
        newSegments.push({
          id: Math.random().toString(36),
          start: currentSegmentStart,
          end: segmentEnd,
          duration: segmentEnd - currentSegmentStart,
          speakerLabel: firstWordOfSegment.identified_speaker_name || `Speaker ${currentSpeaker + 1}`,
          deepgramSpeakerLabel: `speaker_${currentSpeaker}`,
          label: 'UNCERTAIN' as const,
          confidence: Math.min(...currentWords.map(w => w.confidence || 0)),
          transcription: currentWords.map(w => w.punctuated_word || w.word).join(' ')
        })
        console.log(`Created final segment for speaker ${currentSpeaker}: ${currentSegmentStart}s - ${segmentEnd}s`)
      }

      console.log(`Total segments created: ${newSegments.length}`)

      // Add unique speaker labels to available speakers
      const uniqueSpeakerLabels = new Set(
        newSegments
          .map(s => s.speakerLabel)
          .filter(label => label && !availableSpeakers.some(sp => sp.label === label))
      )

      if (uniqueSpeakerLabels.size > 0) {
        setAvailableSpeakers(prev => [
          ...prev,
          ...Array.from(uniqueSpeakerLabels).map(label => ({ label }))
        ])
      }

      setSegments(newSegments)
      setHasUnsavedChanges(true)
      alert(`Transcribed ${newSegments.length} segments from audio`)

    } catch (error) {
      console.error('Failed to transcribe audio:', error)
      alert('Failed to transcribe audio. Please try again.')
    } finally {
      setIsTranscribing(false)
    }
  }, [audioData, user, availableSpeakers])

  const handleFileUpload = useCallback(async (files: File[]) => {
    if (!files.length || !user) return

    const file = files[0]
    if (!isAudioFile(file)) {
      alert('Please select a valid audio file (WAV, FLAC, MP3, M4A, OGG)')
      return
    }

    setIsLoading(true)
    try {
      // Calculate file hash
      const hash = await calculateFileHash(file)
      
      // Check if annotations exist for this file
      const hasExisting = await databaseService.hasAnnotations(hash, user.id)
      if (hasExisting) {
        const proceed = confirm(
          'Found existing annotations for this file. Load them? (Cancel to start fresh)'
        )
        if (proceed) {
          const existingAnnotations = await databaseService.loadAnnotations(hash, user.id)
          if (existingAnnotations) {
            // Convert database annotations to segments
            const loadedSegments: AnnotationSegment[] = existingAnnotations.map(ann => ({
              id: ann.id?.toString() || Math.random().toString(36),
              start: ann.start_time,
              end: ann.end_time,
              duration: ann.end_time - ann.start_time,
              speakerId: ann.speaker_id,
              speakerLabel: ann.speaker_label,
              deepgramSpeakerLabel: ann.deepgram_speaker_label,
              label: ann.label,
              confidence: ann.confidence,
              transcription: ann.transcription,
              notes: ann.notes
            }))
            setSegments(loadedSegments)
            
            // Also extract any unique speaker labels from loaded annotations
            const uniqueSpeakerLabels = new Set(
              loadedSegments
                .map(s => s.speakerLabel)
                .filter(label => label && !availableSpeakers.some(sp => sp.label === label))
            )
            
            // Add these to available speakers if they don't exist
            if (uniqueSpeakerLabels.size > 0) {
              setAvailableSpeakers(prev => [
                ...prev,
                ...Array.from(uniqueSpeakerLabels).map(label => ({ label }))
              ])
            }
          }
        }
      }
      
      // Load and decode audio
      const arrayBuffer = await loadAudioBuffer(file)
      const audioContext = createAudioContext()
      audioContextRef.current = audioContext
      
      const audioBuffer = await decodeAudioData(audioContext, arrayBuffer)
      const samples = extractAudioSamples(audioBuffer)
      
      setAudioData({
        file,
        buffer: audioBuffer,
        hash,
        samples
      })
      
      setHasUnsavedChanges(false)
    } catch (error) {
      console.error('Failed to process audio file:', error)
      alert('Failed to process audio file. Please try a different file.')
    } finally {
      setIsLoading(false)
    }
  }, [user])

  const handleDeepgramUpload = useCallback(async (files: File[]) => {
    const file = files[0]
    if (!file || !file.name.endsWith('.json')) {
      alert('Please select a JSON file')
      return
    }

    try {
      const text = await file.text()
      const deepgramData = JSON.parse(text)
      
      // Parse Deepgram format
      if (deepgramData.results?.channels?.[0]?.alternatives?.[0]?.words) {
        const words = deepgramData.results.channels[0].alternatives[0].words
        const newSegments: AnnotationSegment[] = []
        
        // Group words by speaker
        let currentSpeaker = null
        let currentSegmentStart = null
        let currentWords: any[] = []
        
        for (const word of words) {
          if (word.speaker !== currentSpeaker) {
            // Save previous segment
            if (currentSpeaker !== null && currentWords.length > 0) {
              const segmentEnd = currentWords[currentWords.length - 1].end
              newSegments.push({
                id: Math.random().toString(36),
                start: currentSegmentStart,
                end: segmentEnd,
                duration: segmentEnd - currentSegmentStart,
                speakerLabel: `Speaker ${currentSpeaker + 1}`,
                deepgramSpeakerLabel: `speaker_${currentSpeaker}`,
                label: 'UNCERTAIN' as const,
                confidence: Math.min(...currentWords.map(w => w.confidence || 0)),
                transcription: currentWords.map(w => w.word).join(' ')
              })
            }
            
            // Start new segment
            currentSpeaker = word.speaker
            currentSegmentStart = word.start
            currentWords = [word]
          } else {
            currentWords.push(word)
          }
        }
        
        // Add final segment
        if (currentWords.length > 0) {
          const segmentEnd = currentWords[currentWords.length - 1].end
          newSegments.push({
            id: Math.random().toString(36),
            start: currentSegmentStart,
            end: segmentEnd,
            duration: segmentEnd - currentSegmentStart,
            speakerLabel: `Speaker ${currentSpeaker + 1}`,
            deepgramSpeakerLabel: `speaker_${currentSpeaker}`,
            label: 'UNCERTAIN' as const,
            confidence: Math.min(...currentWords.map(w => w.confidence || 0)),
            transcription: currentWords.map(w => w.word).join(' ')
          })
        }
        
        setSegments(newSegments)
        setHasUnsavedChanges(true)
        alert(`Loaded ${newSegments.length} segments from Deepgram JSON`)
      } else {
        alert('Invalid Deepgram JSON format')
      }
    } catch (error) {
      console.error('Failed to parse Deepgram JSON:', error)
      alert('Failed to parse Deepgram JSON file')
    }
  }, [])

  const addManualSegment = useCallback(() => {
    if (!audioData) return
    
    const newSegment: AnnotationSegment = {
      id: Math.random().toString(36),
      start: 0,
      end: Math.min(5, audioData.buffer.duration),
      duration: Math.min(5, audioData.buffer.duration),
      label: 'UNCERTAIN',
      speakerLabel: 'Unknown'
    }
    
    setSegments(prev => [...prev, newSegment])
    setSelectedSegmentId(newSegment.id)
    setHasUnsavedChanges(true)
  }, [audioData])

  const updateSegment = useCallback((id: string, updates: Partial<AnnotationSegment>) => {
    setSegments(prev => prev.map(seg => 
      seg.id === id ? { ...seg, ...updates } : seg
    ))
    setHasUnsavedChanges(true)
  }, [])

  const deleteSegment = useCallback((id: string) => {
    setSegments(prev => prev.filter(seg => seg.id !== id))
    if (selectedSegmentId === id) {
      setSelectedSegmentId(null)
    }
    setHasUnsavedChanges(true)
  }, [selectedSegmentId])

  const addSpeaker = useCallback(() => {
    if (!newSpeakerName.trim()) return
    
    // Add as temporary speaker (no ID)
    setAvailableSpeakers(prev => [...prev, { label: newSpeakerName.trim() }])
    setNewSpeakerName('')
  }, [newSpeakerName])

  const playSegment = useCallback(async (segment: AnnotationSegment) => {
    if (!audioData || !audioContextRef.current) return

    // Stop any currently playing audio
    if (audioSourceRef.current) {
      audioSourceRef.current.stop()
      audioSourceRef.current = null
    }

    try {
      const source = audioContextRef.current.createBufferSource()
      source.buffer = audioData.buffer
      source.connect(audioContextRef.current.destination)
      
      source.start(0, segment.start, segment.duration)
      audioSourceRef.current = source
      setPlayingSegmentId(segment.id)
      
      source.onended = () => {
        setPlayingSegmentId(null)
        audioSourceRef.current = null
      }
    } catch (error) {
      console.error('Failed to play segment:', error)
      setPlayingSegmentId(null)
    }
  }, [audioData])

  const stopPlayback = useCallback(() => {
    if (audioSourceRef.current) {
      audioSourceRef.current.stop()
      audioSourceRef.current = null
      setPlayingSegmentId(null)
    }
  }, [])

  const saveAnnotations = useCallback(async () => {
    if (!audioData || !user) return

    try {
      const annotations: Annotation[] = segments.map(seg => ({
        audio_file_path: audioData.file.name,
        audio_file_hash: audioData.hash,
        audio_file_name: audioData.file.name,
        start_time: seg.start,
        end_time: seg.end,
        speaker_id: seg.speakerId,
        speaker_label: seg.speakerLabel,
        deepgram_speaker_label: seg.deepgramSpeakerLabel,
        label: seg.label,
        confidence: seg.confidence,
        transcription: seg.transcription,
        user_id: user.id,
        notes: seg.notes
      }))

      await databaseService.saveAnnotations(
        audioData.hash,
        audioData.file.name,
        annotations,
        user.id
      )

      setHasUnsavedChanges(false)
      alert('Annotations saved successfully!')
    } catch (error) {
      console.error('Failed to save annotations:', error)
      alert('Failed to save annotations')
    }
  }, [audioData, user, segments])

  const exportAnnotations = useCallback(() => {
    if (!segments.length) return

    const exportData = {
      filename: audioData?.file.name,
      hash: audioData?.hash,
      segments: segments.map(seg => ({
        start: seg.start,
        end: seg.end,
        duration: seg.duration,
        speaker: seg.speakerLabel,
        speaker_id: seg.speakerId,
        label: seg.label,
        confidence: seg.confidence,
        transcription: seg.transcription,
        notes: seg.notes
      }))
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `annotations_${audioData?.file.name.split('.')[0]}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [segments, audioData])

  const exportSpeakerSegments = useCallback(async (speakerLabel: string) => {
    if (!audioData || !segments.length) return
    
    const speakerSegments = segments.filter(s => s.speakerLabel === speakerLabel)
    if (speakerSegments.length === 0) {
      alert(`No segments found for ${speakerLabel}`)
      return
    }
    
    // Create export data with segment information
    const exportData = {
      speaker_label: speakerLabel,
      audio_file: audioData.file.name,
      total_segments: speakerSegments.length,
      total_duration: speakerSegments.reduce((sum, seg) => sum + seg.duration, 0),
      segments: speakerSegments.map(seg => ({
        start: seg.start,
        end: seg.end,
        duration: seg.duration,
        transcription: seg.transcription,
        confidence: seg.confidence
      }))
    }
    
    // For now, export as JSON metadata
    // TODO: In future, could create actual audio files with segments
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `speaker_segments_${speakerLabel.replace(/\s+/g, '_')}_${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    
    alert(`Exported ${speakerSegments.length} segments for ${speakerLabel}`)
  }, [segments, audioData])

  const enrollSpeaker = useCallback(async (speakerLabel: string, speakerId?: string) => {
    if (!audioData || !user) return
    
    const speakerSegments = segments.filter(s => s.speakerLabel === speakerLabel)
    if (speakerSegments.length === 0) {
      alert(`No segments found for ${speakerLabel}`)
      return
    }
    
    // If speaker is already enrolled, ask user what to do
    if (speakerId) {
      const action = window.confirm(
        `"${speakerLabel}" is already enrolled.\n\n` +
        `Click OK to add these segments to the existing enrollment.\n` +
        `Click Cancel to create a fresh enrollment (will replace the existing one).`
      )
      
      if (!action) {
        // User chose to create fresh enrollment
        const confirmReplace = window.confirm(
          `Are you sure you want to replace the existing enrollment for "${speakerLabel}"?\n\n` +
          `This will delete the current speaker embeddings and create new ones.`
        )
        if (!confirmReplace) return
        speakerId = undefined // Will create new enrollment
      }
    }
    
    setEnrollingSpeaker(speakerLabel)
    
    try {
      // Extract audio segments and create blobs
      console.log(`Processing ${speakerSegments.length} segments for enrollment...`)
      const audioBlobs: Blob[] = []
      
      for (let i = 0; i < speakerSegments.length; i++) {
        const segment = speakerSegments[i]
        console.log(`Extracting segment ${i + 1}: ${segment.start}s - ${segment.end}s`)
        
        const samples = extractAudioSegment(audioData.buffer, segment.start, segment.end)
        const blob = createAudioBlob(samples, audioData.buffer.sampleRate)
        audioBlobs.push(blob)
      }
      
      // Create FormData for enrollment
      const formData = new FormData()
      const newSpeakerId = speakerId || `user_${user.id}_${speakerLabel.replace(/\s+/g, '_')}_${Date.now()}`
      
      formData.append('speaker_id', newSpeakerId)
      formData.append('speaker_name', speakerLabel)
      
      audioBlobs.forEach((blob, index) => {
        formData.append('files', blob, `segment_${String(index + 1).padStart(3, '0')}.wav`)
      })
      
      console.log(`Enrolling speaker with ID: ${newSpeakerId}`)
      
      // Submit enrollment - use append endpoint if speaker exists, batch if new
      const endpoint = speakerId ? '/enroll/append' : '/enroll/batch'
      console.log(`Using endpoint: ${endpoint}`)
      
      await apiService.post(endpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes for enrollment operations
      })
      
      // Refresh speakers list to get the updated enrolled speakers
      try {
        const response = await apiService.get('/speakers', {
          params: { user_id: user.id }
        })
        
        const enrolledSpeakers = response.data.speakers || []
        
        // Combine enrolled speakers with default temporary labels
        const allSpeakers: AvailableSpeaker[] = [
          ...enrolledSpeakers.map((s: any) => ({ id: s.id, label: s.name })),
          { label: 'Speaker 1' },
          { label: 'Speaker 2' },
          { label: 'Unknown' }
        ]
        
        setAvailableSpeakers(allSpeakers)
      } catch (error) {
        console.error('Failed to refresh speakers list:', error)
      }
      
      alert(
        `✅ Speaker "${speakerLabel}" enrolled successfully!\n\n` +
        `Processed ${speakerSegments.length} segments (${formatDuration(speakerSegments.reduce((sum, seg) => sum + seg.duration, 0))} total)`
      )
      
    } catch (error) {
      console.error('Failed to enroll speaker:', error)
      alert(`❌ Failed to enroll speaker "${speakerLabel}". Please try again.`)
    } finally {
      setEnrollingSpeaker(null)
    }
  }, [audioData, user, segments])

  if (!user) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Please select a user to continue.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">📝 Annotation Tool</h1>
        {hasUnsavedChanges && (
          <div className="flex items-center space-x-2 text-amber-600">
            <AlertCircle className="h-5 w-5" />
            <span className="text-sm font-medium">Unsaved changes</span>
          </div>
        )}
      </div>

      <p className="text-gray-600">
        Upload audio files and create speaker annotations with timeline visualization.
      </p>

      {/* File Upload */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6">
          <h3 className="text-lg font-medium mb-4">📁 Upload Audio File</h3>
          <FileUploader
            onUpload={handleFileUpload}
            accept=".wav,.flac,.mp3,.m4a,.ogg"
            multiple={false}
            disabled={isLoading}
          />
        </div>

        {/* Right column - Split vertically */}
        <div className="space-y-4">
          {/* Top half - Import JSON */}
          <div className="border-2 border-dashed border-blue-300 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-3">📄 Import Deepgram JSON</h3>
            <FileUploader
              onUpload={handleDeepgramUpload}
              accept=".json"
              multiple={false}
              disabled={!audioData}
              title="Upload Transcript JSON"
            />
            {!audioData && (
              <p className="text-sm text-gray-500 mt-2">Upload audio file first</p>
            )}
          </div>

          {/* Bottom half - Transcribe with Deepgram */}
          <div className="border-2 border-dashed border-green-300 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-3">🎤 Transcribe with Deepgram</h3>
            <div className="text-center">
              <button
                onClick={handleDeepgramTranscribe}
                disabled={!audioData || isTranscribing || isLoading}
                className="w-full px-4 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
              >
                {isTranscribing ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Transcribing...</span>
                  </>
                ) : (
                  <>
                    <span>🚀</span>
                    <span>Transcribe Audio</span>
                  </>
                )}
              </button>
              <p className="text-sm text-gray-500 mt-2">
                {!audioData 
                  ? 'Upload audio file first' 
                  : 'Generate transcript with speaker diarization'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {isLoading && (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <p className="mt-2 text-gray-600">Processing audio file...</p>
        </div>
      )}

      {audioData && (
        <>
          {/* Audio Information */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-2">🎵 {audioData.file.name}</h3>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>Duration: {formatDuration(audioData.buffer.duration)}</div>
              <div>Sample Rate: {(audioData.buffer.sampleRate / 1000).toFixed(1)} kHz</div>
              <div>Segments: {segments.length}</div>
            </div>
          </div>

          {/* Waveform with segments */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium">📊 Timeline</h3>
              <button
                onClick={addManualSegment}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Add Segment
              </button>
            </div>
            
            <WaveformPlot
              samples={audioData.samples}
              sampleRate={audioData.buffer.sampleRate}
              segments={segments}
              selectedSegment={selectedSegmentId ? 
                segments.find(s => s.id === selectedSegmentId) ? 
                [segments.find(s => s.id === selectedSegmentId)!.start, segments.find(s => s.id === selectedSegmentId)!.end] : null 
                : null}
              onSegmentSelect={(segment) => {
                // Find closest segment to selection
                const midpoint = (segment[0] + segment[1]) / 2
                const closest = segments.reduce((prev, curr) => {
                  const prevDist = Math.abs((prev.start + prev.end) / 2 - midpoint)
                  const currDist = Math.abs((curr.start + curr.end) / 2 - midpoint)
                  return prevDist < currDist ? prev : curr
                })
                setSelectedSegmentId(closest?.id || null)
              }}
            />
          </div>

          {/* Segments List */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium">🗣️ Speaker Segments</h3>
            
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {segments.map((segment) => (
                <SegmentEditor
                  key={segment.id}
                  segment={segment}
                  isSelected={selectedSegmentId === segment.id}
                  isPlaying={playingSegmentId === segment.id}
                  availableSpeakers={availableSpeakers}
                  onSelect={() => setSelectedSegmentId(segment.id)}
                  onUpdate={(updates) => updateSegment(segment.id, updates)}
                  onDelete={() => deleteSegment(segment.id)}
                  onPlay={() => playSegment(segment)}
                  onStop={stopPlayback}
                />
              ))}
            </div>
          </div>

          {/* Speaker Management */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-4">👥 Manage Speakers</h3>
            <div className="flex space-x-2">
              <input
                type="text"
                value={newSpeakerName}
                onChange={(e) => setNewSpeakerName(e.target.value)}
                placeholder="New speaker name"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md"
                onKeyPress={(e) => e.key === 'Enter' && addSpeaker()}
              />
              <button
                onClick={addSpeaker}
                disabled={!newSpeakerName.trim()}
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
              >
                Add Speaker
              </button>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {availableSpeakers.map((speaker, index) => (
                <span
                  key={index}
                  className={`px-3 py-1 rounded-full text-sm ${
                    speaker.id 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-blue-100 text-blue-800'
                  }`}
                  title={speaker.id ? 'Enrolled speaker' : 'Temporary label'}
                >
                  {speaker.label}
                </span>
              ))}
            </div>
          </div>

          {/* Speaker Segments Summary */}
          {segments.length > 0 && (
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="text-lg font-medium mb-4">📊 Speaker Segments Summary</h3>
              <div className="space-y-2">
                {Array.from(new Set(segments.map(s => s.speakerLabel).filter(Boolean))).map(speakerLabel => {
                  const speakerSegments = segments.filter(s => s.speakerLabel === speakerLabel)
                  const totalDuration = speakerSegments.reduce((sum, seg) => sum + seg.duration, 0)
                  const speaker = availableSpeakers.find(s => s.label === speakerLabel)
                  
                  return (
                    <div key={speakerLabel} className="flex items-center justify-between p-2 bg-white rounded">
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          speaker?.id 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-blue-100 text-blue-800'
                        }`}>
                          {speakerLabel}
                        </span>
                        <span className="text-sm text-gray-600">
                          {speakerSegments.length} segments • {formatDuration(totalDuration)}
                        </span>
                      </div>
                      <div className="flex space-x-2">
                        <button
                          onClick={() => enrollSpeaker(speakerLabel!, speaker?.id)}
                          disabled={enrollingSpeaker === speakerLabel}
                          className="flex items-center space-x-1 px-3 py-1 text-sm bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
                        >
                          <UserPlus className="h-3 w-3" />
                          <span>
                            {enrollingSpeaker === speakerLabel 
                              ? 'Enrolling...' 
                              : speaker?.id 
                                ? 'Update Enrollment' 
                                : 'Enroll Speaker'}
                          </span>
                        </button>
                        <button
                          onClick={() => exportSpeakerSegments(speakerLabel!)}
                          className="flex items-center space-x-1 px-2 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                        >
                          <Download className="h-3 w-3" />
                          <span>Export</span>
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Save and Export Actions */}
          <div className="flex justify-end space-x-3 pt-6 border-t">
            {deepgramResponse && (
              <button
                onClick={() => setShowJsonOutput(!showJsonOutput)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-md ${
                  showJsonOutput 
                    ? 'bg-blue-600 text-white hover:bg-blue-700' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                <span>📄</span>
                <span>{showJsonOutput ? 'Hide' : 'Show'} JSON</span>
              </button>
            )}
            <button
              onClick={saveAnnotations}
              disabled={!hasUnsavedChanges || segments.length === 0}
              className="flex items-center space-x-2 px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Save className="h-4 w-4" />
              <span>Save Annotations</span>
            </button>
            <button
              onClick={exportAnnotations}
              disabled={segments.length === 0}
              className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Download className="h-4 w-4" />
              <span>Export JSON</span>
            </button>
          </div>

          {/* JSON Output Section */}
          {showJsonOutput && deepgramResponse && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-medium text-gray-900">🔍 Debug Output</h4>
                <button
                  onClick={() => setShowJsonOutput(!showJsonOutput)}
                  className="flex items-center space-x-2 px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
                >
                  <span>{showJsonOutput ? 'Hide' : 'Show'} JSON</span>
                </button>
              </div>
              
              <div className="space-y-4">
                {/* Deepgram Response */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h5 className="font-medium text-gray-900 mb-2">Deepgram API Response</h5>
                  <pre className="text-xs bg-white p-3 rounded border overflow-auto max-h-64 text-gray-800 text-left">
                    {JSON.stringify(deepgramResponse, null, 2)}
                  </pre>
                </div>
                
                {/* Annotation Segments */}
                <div className="bg-gray-50 rounded-lg p-4">
                  <h5 className="font-medium text-gray-900 mb-2">Annotation Segments</h5>
                  <pre className="text-xs bg-white p-3 rounded border overflow-auto max-h-64 text-gray-800 text-left">
                    {JSON.stringify(segments, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

interface SegmentEditorProps {
  segment: AnnotationSegment
  isSelected: boolean
  isPlaying: boolean
  availableSpeakers: AvailableSpeaker[]
  onSelect: () => void
  onUpdate: (updates: Partial<AnnotationSegment>) => void
  onDelete: () => void
  onPlay: () => void
  onStop: () => void
}

function SegmentEditor({
  segment,
  isSelected,
  isPlaying,
  availableSpeakers,
  onSelect,
  onUpdate,
  onDelete,
  onPlay,
  onStop
}: SegmentEditorProps) {
  const getLabelIcon = (label: string) => {
    switch (label) {
      case 'CORRECT': return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'INCORRECT': return <XCircle className="h-4 w-4 text-red-500" />
      default: return <AlertCircle className="h-4 w-4 text-yellow-500" />
    }
  }

  const getLabelColor = (label: string) => {
    switch (label) {
      case 'CORRECT': return 'border-green-200 bg-green-50'
      case 'INCORRECT': return 'border-red-200 bg-red-50'
      default: return 'border-yellow-200 bg-yellow-50'
    }
  }

  return (
    <div
      className={`border rounded-lg p-4 cursor-pointer transition-colors ${
        isSelected 
          ? 'border-blue-500 bg-blue-50' 
          : getLabelColor(segment.label)
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          {getLabelIcon(segment.label)}
          <span className="font-medium">
            {segment.start.toFixed(2)}s - {segment.end.toFixed(2)}s
          </span>
          <span className="text-sm text-gray-500">
            ({formatDuration(segment.duration)})
          </span>
        </div>
        
        <div className="flex space-x-2">
          <button
            onClick={(e) => {
              e.stopPropagation()
              isPlaying ? onStop() : onPlay()
            }}
            className={`p-1 rounded ${
              isPlaying 
                ? 'bg-red-100 text-red-700 hover:bg-red-200'
                : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
            }`}
          >
            {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation()
              onDelete()
            }}
            className="p-1 bg-red-100 text-red-700 rounded hover:bg-red-200"
          >
            <XCircle className="h-4 w-4" />
          </button>
        </div>
      </div>

      {isSelected && (
        <div className="space-y-3 border-t pt-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium mb-1">Speaker</label>
              <select
                value={segment.speakerLabel || ''}
                onChange={(e) => {
                  const selectedLabel = e.target.value
                  const selectedSpeaker = availableSpeakers.find(s => s.label === selectedLabel)
                  onUpdate({
                    speakerLabel: selectedLabel,
                    speakerId: selectedSpeaker?.id // Will be undefined for temporary labels
                  })
                }}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              >
                <option value="">Select speaker...</option>
                {availableSpeakers.map((speaker) => (
                  <option key={speaker.label} value={speaker.label}>
                    {speaker.label} {speaker.id ? '✓' : ''}
                  </option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Label</label>
              <select
                value={segment.label}
                onChange={(e) => onUpdate({ label: e.target.value as any })}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              >
                <option value="CORRECT">Correct</option>
                <option value="INCORRECT">Incorrect</option>
                <option value="UNCERTAIN">Uncertain</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Transcription</label>
            <textarea
              value={segment.transcription || ''}
              onChange={(e) => onUpdate({ transcription: e.target.value })}
              placeholder="Enter transcription..."
              className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              rows={2}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Notes</label>
            <input
              type="text"
              value={segment.notes || ''}
              onChange={(e) => onUpdate({ notes: e.target.value })}
              placeholder="Optional notes..."
              className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
            />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-sm font-medium mb-1">Start (s)</label>
              <input
                type="number"
                step="0.1"
                value={segment.start}
                onChange={(e) => {
                  const start = parseFloat(e.target.value)
                  onUpdate({ 
                    start, 
                    duration: segment.end - start 
                  })
                }}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">End (s)</label>
              <input
                type="number"
                step="0.1"
                value={segment.end}
                onChange={(e) => {
                  const end = parseFloat(e.target.value)
                  onUpdate({ 
                    end, 
                    duration: end - segment.start 
                  })
                }}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Confidence</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={segment.confidence || ''}
                onChange={(e) => onUpdate({ confidence: parseFloat(e.target.value) || undefined })}
                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}