import { useState, useRef, useCallback, useEffect } from 'react'
import { Upload, Play, Pause, Download, Users, BarChart3, Clock, CheckCircle, Mic, MicOff } from 'lucide-react'
import { useUser } from '../contexts/UserContext'
import { calculateFileHash, isAudioFile } from '../utils/fileHash'
import { 
  loadAudioBuffer, 
  createAudioContext, 
  decodeAudioData,
  extractAudioSamples,
  formatDuration,
  convertBlobToWav,
  createAudioBlob
} from '../utils/audioUtils'
import { apiService } from '../services/api'
import FileUploader from '../components/FileUploader'
import WaveformPlot from '../components/WaveformPlot'

interface InferenceResult {
  id: string
  filename: string
  duration: number
  status: 'processing' | 'completed' | 'failed'
  created_at: string
  speakers: SpeakerSegment[]
  confidence_summary: {
    total_segments: number
    high_confidence: number
    medium_confidence: number
    low_confidence: number
  }
}

interface SpeakerSegment {
  start: number
  end: number
  speaker_id: string
  speaker_name: string
  confidence: number
  text?: string
}

interface AudioData {
  file: File | Blob
  filename: string
  buffer: AudioBuffer
  samples: Float32Array
}

export default function Inference() {
  const { user } = useUser()
  const [audioData, setAudioData] = useState<AudioData | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [results, setResults] = useState<InferenceResult[]>([])
  const [selectedResult, setSelectedResult] = useState<InferenceResult | null>(null)
  const [playingSegment, setPlayingSegment] = useState<SpeakerSegment | null>(null)
  const [showConfidenceFilter, setShowConfidenceFilter] = useState(false)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.15)
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [deepgramResponse, setDeepgramResponse] = useState<any>(null)
  const [showJsonOutput, setShowJsonOutput] = useState(false)
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null)

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      // Stop recording if active
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop()
      }
      
      // Clear interval
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current)
      }
      
      // Stop audio playback
      if (audioSourceRef.current) {
        audioSourceRef.current.stop()
      }
      
      // Close audio context
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close()
      }
    }
  }, [])

  const handleFileUpload = useCallback(async (files: File[]) => {
    if (!files.length || !user) return

    const file = files[0]
    if (!file.name.toLowerCase().endsWith('.wav')) {
      alert('Please select a WAV audio file. Other formats are not currently supported.')
      return
    }

    try {
      // Load and decode audio for visualization
      const arrayBuffer = await loadAudioBuffer(file)
      const audioContext = createAudioContext()
      audioContextRef.current = audioContext
      const audioBuffer = await decodeAudioData(audioContext, arrayBuffer)
      const samples = extractAudioSamples(audioBuffer)

      setAudioData({
        file,
        filename: file.name,
        buffer: audioBuffer,
        samples
      })

    } catch (error) {
      console.error('Failed to process audio file:', error)
      alert('Failed to process audio file. Please try a different file.')
    }
  }, [user])

  const startRecording = useCallback(async () => {
    try {
      // Check if HTTPS is required
      if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
        alert('Microphone access requires HTTPS. Please use HTTPS or localhost.')
        return
      }

      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
          // Removed sampleRate constraint for better compatibility
        } 
      })
      
      // Try WAV first, fallback to WebM if not supported
      let mimeType = 'audio/wav'
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/webm'
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = '' // Let browser choose
        }
      }
      
      const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      
      audioChunksRef.current = []
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }
      
      mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: mimeType || 'audio/webm' })
        await processRecording(blob)
        stream.getTracks().forEach(track => track.stop())
      }
      
      mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event)
        alert('Recording failed. Please try again.')
        setIsRecording(false)
        stream.getTracks().forEach(track => track.stop())
      }
      
      mediaRecorderRef.current = mediaRecorder
      setIsRecording(true)
      setRecordingTime(0)
      
      // Start timer immediately
      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1)
      }, 1000)
      
      mediaRecorder.start(250) // Increased interval for better stability
      
    } catch (error) {
      console.error('Failed to start recording:', error)
      
      // Clean up if recording failed
      setIsRecording(false)
      setRecordingTime(0)
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current)
        recordingIntervalRef.current = null
      }
      
      let errorMessage = 'Failed to access microphone. '
      if (error.name === 'NotAllowedError') {
        errorMessage += 'Please allow microphone access and try again.'
      } else if (error.name === 'NotFoundError') {
        errorMessage += 'No microphone found. Please check your device.'
      } else if (error.name === 'NotSupportedError') {
        errorMessage += 'Recording not supported in this browser.'
      } else {
        errorMessage += 'Please check permissions and try again.'
      }
      
      alert(errorMessage)
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      try {
        if (mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop()
        }
      } catch (error) {
        console.error('Error stopping MediaRecorder:', error)
      }
      
      setIsRecording(false)
      
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current)
        recordingIntervalRef.current = null
      }
    }
  }, [isRecording])

  const processRecording = useCallback(async (blob: Blob) => {
    try {
      console.log('Processing recording blob:', blob.type, blob.size, 'bytes')
      
      // Convert WebM blob to WAV if needed for better backend compatibility
      let processedBlob = blob
      if (blob.type.includes('webm')) {
        console.log('Converting WebM recording to WAV format...')
        processedBlob = await convertBlobToWav(blob)
        console.log('Conversion successful:', processedBlob.type, processedBlob.size, 'bytes')
      }
      
      // Convert blob to audio buffer for analysis
      const arrayBuffer = await processedBlob.arrayBuffer()
      
      if (arrayBuffer.byteLength === 0) {
        alert('Recording is empty. Please try recording again.')
        return
      }
      
      const audioContext = createAudioContext()
      audioContextRef.current = audioContext
      
      console.log('Decoding audio data...')
      const audioBuffer = await decodeAudioData(audioContext, arrayBuffer)
      
      if (audioBuffer.duration === 0) {
        alert('Recording has no duration. Please try recording again.')
        return
      }
      
      console.log('Audio decoded successfully:', audioBuffer.duration, 'seconds')
      
      const samples = extractAudioSamples(audioBuffer)
      
      setAudioData({
        file: processedBlob,
        filename: `Recording ${new Date().toLocaleString()}`,
        buffer: audioBuffer,
        samples
      })
      
      console.log('Recording processed successfully')
      
    } catch (error) {
      console.error('Failed to process recording:', error)
      
      let errorMessage = 'Failed to process recording. '
      if (error.name === 'EncodingError' || error.message.includes('decode')) {
        errorMessage += 'Audio format not supported. Try using a different browser or check your microphone settings.'
      } else if (error.message.includes('context')) {
        errorMessage += 'Audio processing failed. Please try again.'
      } else if (error.message.includes('conversion')) {
        errorMessage += 'Audio conversion failed. Please try again or use a different browser.'
      } else {
        errorMessage += 'Please try again or refresh the page.'
      }
      
      alert(errorMessage)
    }
  }, [])

  const startInference = useCallback(async (mode: 'diarization' | 'deepgram' = 'diarization') => {
    if (!audioData || !user) return

    setIsProcessing(true)
    try {
      if (mode === 'deepgram') {
        // Use Deepgram wrapper endpoint
        const formData = new FormData()
        formData.append('file', audioData.file)
        
        const response = await apiService.post('/v1/listen', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          params: {
            model: 'nova-3',
            language: 'en',
            diarize: true,
            smart_format: true,
            punctuate: true,
            enhance_speakers: true,
            user_id: user.id,
            speaker_confidence_threshold: confidenceThreshold
          },
          timeout: 300000, // 5 minutes for Deepgram + speaker identification
        })

        setDeepgramResponse(response.data)
        
        // Extract words and convert to speaker segments
        const results = response.data.results || {}
        const channels = results.channels || []
        const words = channels[0]?.alternatives?.[0]?.words || []
        
        // Group words by speaker into segments
        const speakerSegments: SpeakerSegment[] = []
        let currentSegment: any = null
        
        for (const word of words) {
          const speaker = word.speaker || 0
          
          if (!currentSegment || currentSegment.original_speaker !== speaker) {
            // Start new segment
            if (currentSegment) {
              speakerSegments.push(currentSegment)
            }
            currentSegment = {
              start: word.start,
              end: word.end,
              original_speaker: speaker,
              speaker_id: word.identified_speaker_id || `speaker_${speaker}`,
              speaker_name: word.identified_speaker_name || `Speaker ${speaker}`,
              confidence: word.speaker_identification_confidence || 0,
              text: word.punctuated_word || word.word,
              identified_speaker_id: word.identified_speaker_id,
              identified_speaker_name: word.identified_speaker_name,
              speaker_identification_confidence: word.speaker_identification_confidence,
              speaker_status: word.speaker_status
            }
          } else {
            // Continue current segment
            currentSegment.end = word.end
            currentSegment.text += ' ' + (word.punctuated_word || word.word)
          }
        }
        
        if (currentSegment) {
          speakerSegments.push(currentSegment)
        }
        
        // Calculate confidence summary
        const high_confidence = speakerSegments.filter(s => s.confidence >= 0.8).length
        const medium_confidence = speakerSegments.filter(s => s.confidence >= 0.6 && s.confidence < 0.8).length
        const low_confidence = speakerSegments.filter(s => s.confidence >= 0.4 && s.confidence < 0.6).length
        
        const newResult: InferenceResult = {
          id: Math.random().toString(36),
          filename: audioData.filename,
          duration: audioData.buffer.duration,
          status: 'completed',
          created_at: new Date().toISOString(),
          speakers: speakerSegments,
          confidence_summary: {
            total_segments: speakerSegments.length,
            high_confidence,
            medium_confidence,
            low_confidence
          }
        }

        setResults(prev => [newResult, ...prev])
        setSelectedResult(newResult)
        
      } else {
        // Use original diarization endpoint
        const formData = new FormData()
        formData.append('file', audioData.file)
        formData.append('similarity_threshold', confidenceThreshold.toString())
        formData.append('min_duration', '1.0') // Minimum segment duration
        formData.append('identify_only_enrolled', 'false') // Show all speakers

        const response = await apiService.post('/diarize-and-identify', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          timeout: 180000, // 3 minutes for inference operations
        })

        // Process the diarization response directly (no polling needed)
        const backendSegments = response.data.segments || []
        
        // Convert backend format to frontend format
        const speakers: SpeakerSegment[] = backendSegments.map((segment: any) => ({
          start: segment.start,
          end: segment.end,
          speaker_id: segment.identified_id || segment.speaker,
          speaker_name: segment.identified_as || `Unknown (${segment.speaker})`,
          confidence: segment.confidence || 0,
          text: undefined // Backend doesn't provide transcription
        }))
        
        // Calculate confidence summary
        const high_confidence = speakers.filter(s => s.confidence >= 0.8).length
        const medium_confidence = speakers.filter(s => s.confidence >= 0.6 && s.confidence < 0.8).length
        const low_confidence = speakers.filter(s => s.confidence >= 0.4 && s.confidence < 0.6).length
        
        const newResult: InferenceResult = {
          id: Math.random().toString(36),
          filename: audioData.filename,
          duration: audioData.buffer.duration,
          status: 'completed',
          created_at: new Date().toISOString(),
          speakers,
          confidence_summary: {
            total_segments: speakers.length,
            high_confidence,
            medium_confidence,
            low_confidence
          }
        }

        setResults(prev => [newResult, ...prev])
        setSelectedResult(newResult)
      }

    } catch (error) {
      console.error('Failed to start inference:', error)
      alert('Failed to start speaker inference. Please check that the backend service is running and try again.')
    } finally {
      setIsProcessing(false)
    }
  }, [audioData, user, confidenceThreshold])

  const pollInferenceResult = useCallback(async (inferenceId: string) => {
    try {
      const response = await apiService.get(`/inference/${inferenceId}`)
      const updatedResult = response.data

      setResults(prev => prev.map(result => 
        result.id === inferenceId ? updatedResult : result
      ))

      if (updatedResult.status === 'processing') {
        // Continue polling
        setTimeout(() => pollInferenceResult(inferenceId), 2000)
      }
    } catch (error) {
      console.error('Failed to poll inference result:', error)
    }
  }, [])

  const exportResults = useCallback((result: InferenceResult) => {
    const exportData = {
      filename: result.filename,
      duration: result.duration,
      created_at: result.created_at,
      confidence_summary: result.confidence_summary,
      speakers: result.speakers.map(segment => ({
        start: segment.start,
        end: segment.end,
        duration: segment.end - segment.start,
        speaker: segment.speaker_name,
        confidence: segment.confidence,
        text: segment.text
      }))
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `inference_${result.filename.split('.')[0]}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [])

  const getConfidenceColor = useCallback((confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-100'
    if (confidence >= 0.6) return 'text-blue-600 bg-blue-100' 
    if (confidence >= 0.4) return 'text-yellow-600 bg-yellow-100'
    return 'text-red-600 bg-red-100'
  }, [])

  const getConfidenceLabel = useCallback((confidence: number) => {
    if (confidence >= 0.8) return 'High'
    if (confidence >= 0.6) return 'Medium'
    if (confidence >= 0.4) return 'Low'
    return 'Very Low'
  }, [])

  const filteredSegments = useCallback((segments: SpeakerSegment[]) => {
    if (!showConfidenceFilter) return segments
    return segments.filter(segment => segment.confidence >= confidenceThreshold)
  }, [showConfidenceFilter, confidenceThreshold])

  const playSegment = useCallback(async (segment: SpeakerSegment) => {
    if (!audioData) return

    // Create audio context if it doesn't exist
    if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
      audioContextRef.current = createAudioContext()
    }

    // Resume audio context if it's suspended (required by browser policies)
    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume()
    }

    // Stop any currently playing audio
    if (audioSourceRef.current) {
      try {
        audioSourceRef.current.stop()
      } catch (e) {
        // Ignore if already stopped
      }
      audioSourceRef.current = null
    }

    try {
      const source = audioContextRef.current.createBufferSource()
      source.buffer = audioData.buffer
      source.connect(audioContextRef.current.destination)
      
      // Calculate duration to play
      const duration = segment.end - segment.start
      source.start(0, segment.start, duration)
      audioSourceRef.current = source
      setPlayingSegment(segment)
      
      source.onended = () => {
        setPlayingSegment(null)
        audioSourceRef.current = null
      }
    } catch (error) {
      console.error('Failed to play segment:', error)
      setPlayingSegment(null)
    }
  }, [audioData])

  const stopPlayback = useCallback(() => {
    if (audioSourceRef.current) {
      audioSourceRef.current.stop()
      audioSourceRef.current = null
      setPlayingSegment(null)
    }
  }, [])

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
        <h1 className="text-2xl font-bold text-gray-900">🧠 Speaker Inference</h1>
      </div>

      {/* <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <p className="text-blue-800 text-sm">
          <span className="font-medium">🆕 Enhanced Mode:</span> Now with Deepgram transcription + speaker identification! 
          Choose between diarization-only or full transcription with enhanced speaker recognition.
        </p>
      </div> */}
      
      <p className="text-gray-600">
        Upload audio files to identify speakers using trained recognition models.
      </p>

      {/* Audio Input Section */}
      <div className="bg-white border rounded-lg p-6">
        <h3 className="text-lg font-medium mb-4">🎵 Audio Input</h3>
        
        {/* Upload and Recording Options */}
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          {/* File Upload */}
          <div className="border rounded-lg p-4">
            <h4 className="font-medium mb-3">📁 Upload Audio File</h4>
            <FileUploader
              onUpload={handleFileUpload}
              accept=".wav"
              multiple={false}
              disabled={isProcessing || isRecording}
            />
          </div>

          {/* Recording */}
          <div className="border rounded-lg p-4">
            <h4 className="font-medium mb-3">🎤 Record Audio</h4>
            <div className="text-center space-y-4">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  disabled={isProcessing}
                  className="flex items-center space-x-2 px-6 py-3 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 mx-auto"
                >
                  <Mic className="h-5 w-5" />
                  <span>Start Recording</span>
                </button>
              ) : (
                <div className="space-y-3">
                  <div className="flex items-center justify-center space-x-2 text-red-600">
                    <div className="w-3 h-3 bg-red-600 rounded-full animate-pulse"></div>
                    <span className="font-medium">Recording... {formatDuration(recordingTime)}</span>
                  </div>
                  <button
                    onClick={stopRecording}
                    className="flex items-center space-x-2 px-6 py-3 bg-gray-600 text-white rounded-md hover:bg-gray-700 mx-auto"
                  >
                    <MicOff className="h-5 w-5" />
                    <span>Stop Recording</span>
                  </button>
                </div>
              )}
              <div className="text-sm text-gray-500 space-y-1">
                <p>Record audio for speaker identification</p>
                <p className="text-xs">
                  {location.protocol !== 'https:' && location.hostname !== 'localhost' 
                    ? '⚠️ HTTPS required for microphone access'
                    : '✓ Ready to record'}
                </p>
              </div>
            </div>
          </div>
        </div>
        
        {audioData && (
          <div className="space-y-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-medium text-gray-900 mb-2">🎵 {audioData.filename}</h4>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>Duration: {formatDuration(audioData.buffer.duration)}</div>
                <div>Sample Rate: {(audioData.buffer.sampleRate / 1000).toFixed(1)} kHz</div>
                <div>Channels: {audioData.buffer.numberOfChannels}</div>
              </div>
            </div>

            {/* Configuration */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Confidence Threshold: {confidenceThreshold.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="w-full"
                  disabled={isProcessing}
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Less Strict</span>
                  <span>More Strict</span>
                </div>
              </div>
              
              {/* Two Processing Options */}
              <div className="grid md:grid-cols-2 gap-4">
                <button
                  onClick={() => startInference('diarization')}
                  disabled={isProcessing}
                  className="flex flex-col items-center px-6 py-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 space-y-2"
                >
                  <span className="text-lg">🎯</span>
                  <span className="font-medium">{isProcessing ? 'Processing...' : 'Start Speaker Identification'}</span>
                  <span className="text-xs opacity-90">Diarization + speaker recognition only</span>
                </button>
                
                <button
                  onClick={() => startInference('deepgram')}
                  disabled={isProcessing}
                  className="flex flex-col items-center px-6 py-4 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 space-y-2"
                >
                  <span className="text-lg">🚀</span>
                  <span className="font-medium">{isProcessing ? 'Processing...' : 'Transcribe + Identify'}</span>
                  <span className="text-xs opacity-90">Full transcription with enhanced speaker ID</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-6">
          <h2 className="text-xl font-semibold text-gray-900">📊 Inference Results</h2>
          
          {/* Results List */}
          <div className="space-y-4">
            {results.map((result) => (
              <div
                key={result.id}
                className={`border rounded-lg p-4 cursor-pointer transition-colors ${
                  selectedResult?.id === result.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedResult(result)}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium text-gray-900">{result.filename}</h3>
                    <div className="flex items-center space-x-4 mt-1 text-sm text-gray-500">
                      <span>{formatDuration(result.duration)}</span>
                      <span>{new Date(result.created_at).toLocaleString()}</span>
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        result.status === 'completed' ? 'bg-green-100 text-green-800' :
                        result.status === 'processing' ? 'bg-blue-100 text-blue-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {result.status}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4">
                    {result.status === 'completed' && (
                      <>
                        <div className="text-right text-sm">
                          <div className="text-gray-900 font-medium">{result.speakers.length} segments</div>
                          <div className="text-gray-500">
                            {result.confidence_summary.high_confidence} high conf.
                          </div>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            exportResults(result)
                          }}
                          className="p-2 text-green-600 hover:text-green-800"
                          title="Export Results"
                        >
                          <Download className="h-4 w-4" />
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Selected Result Details */}
          {selectedResult && selectedResult.status === 'completed' && (
            <div className="bg-white border rounded-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-medium">Detailed Results: {selectedResult.filename}</h3>
                <div className="flex items-center space-x-4">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={showConfidenceFilter}
                      onChange={(e) => setShowConfidenceFilter(e.target.checked)}
                      className="rounded border-gray-300"
                    />
                    <span className="text-sm text-gray-700">Filter by confidence</span>
                  </label>
                  <div className="flex space-x-2">
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
                    <button
                      onClick={() => exportResults(selectedResult)}
                      className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                    >
                      <Download className="h-4 w-4" />
                      <span>Export</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Summary Stats */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-sm font-medium text-gray-500">Total Segments</div>
                  <div className="text-xl font-bold text-gray-900">{selectedResult.confidence_summary.total_segments}</div>
                </div>
                <div className="bg-green-50 rounded-lg p-3">
                  <div className="text-sm font-medium text-green-600">High Confidence</div>
                  <div className="text-xl font-bold text-green-700">{selectedResult.confidence_summary.high_confidence}</div>
                </div>
                <div className="bg-blue-50 rounded-lg p-3">
                  <div className="text-sm font-medium text-blue-600">Medium Confidence</div>
                  <div className="text-xl font-bold text-blue-700">{selectedResult.confidence_summary.medium_confidence}</div>
                </div>
                <div className="bg-yellow-50 rounded-lg p-3">
                  <div className="text-sm font-medium text-yellow-600">Low Confidence</div>
                  <div className="text-xl font-bold text-yellow-700">{selectedResult.confidence_summary.low_confidence}</div>
                </div>
              </div>

              {/* Timeline Visualization */}
              {audioData && (
                <div className="mb-6">
                  <h4 className="font-medium text-gray-900 mb-3">📊 Speaker Timeline</h4>
                  <WaveformPlot
                    samples={audioData.samples}
                    sampleRate={audioData.buffer.sampleRate}
                    segments={filteredSegments(selectedResult.speakers)}
                    selectedSegment={null}
                    onSegmentSelect={() => {}}
                  />
                </div>
              )}

              {/* Speaker Segments */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium text-gray-900">🗣️ Speaker Segments</h4>
                  {deepgramResponse?.speaker_enhancement && (
                    <div className="text-sm text-gray-500">
                      Enhanced: {deepgramResponse.speaker_enhancement.identified_count || 0} of {deepgramResponse.speaker_enhancement.total_speakers || 0} speakers identified
                    </div>
                  )}
                </div>
                {filteredSegments(selectedResult.speakers).map((segment, index) => (
                  <div
                    key={index}
                    className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                  >
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <span className="font-medium text-gray-900">{segment.speaker_name}</span>
                        {segment.speaker_status && (
                          <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                            segment.speaker_status === 'identified' ? 'bg-green-100 text-green-800' :
                            segment.speaker_status === 'unknown' ? 'bg-gray-100 text-gray-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {segment.speaker_status}
                          </span>
                        )}
                        <span className="text-sm text-gray-500">
                          {segment.start.toFixed(1)}s - {segment.end.toFixed(1)}s
                        </span>
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColor(segment.confidence)}`}>
                          {(segment.confidence * 100).toFixed(0)}% ({getConfidenceLabel(segment.confidence)})
                        </span>
                        {segment.speaker_identification_confidence && (
                          <span className="text-xs text-purple-600">
                            ID: {(segment.speaker_identification_confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                      {segment.text && (
                        <p className="text-sm text-gray-700 italic">"{segment.text}"</p>
                      )}
                      {segment.identified_speaker_id && (
                        <p className="text-xs text-green-600 mt-1">
                          📝 Identified as: {segment.identified_speaker_name} ({segment.identified_speaker_id})
                        </p>
                      )}
                    </div>
                    <div className="flex space-x-2 ml-4">
                      <button
                        onClick={() => {
                          if (playingSegment?.start === segment.start) {
                            stopPlayback()
                          } else {
                            playSegment(segment)
                          }
                        }}
                        className={`p-2 rounded ${
                          playingSegment?.start === segment.start
                            ? 'bg-red-100 text-red-700 hover:bg-red-200'
                            : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                        }`}
                      >
                        {playingSegment?.start === segment.start ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* JSON Output Section */}
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
                
                {showJsonOutput && (
                  <div className="space-y-4">
                    {/* Deepgram Response */}
                    {deepgramResponse && (
                      <div className="bg-gray-50 rounded-lg p-4">
                        <h5 className="font-medium text-gray-900 mb-2">Deepgram API Response</h5>
                        <pre className="text-xs bg-white p-3 rounded border overflow-auto max-h-64 text-gray-800">
                          {JSON.stringify(deepgramResponse, null, 2)}
                        </pre>
                      </div>
                    )}
                    
                    {/* Our System Response */}
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h5 className="font-medium text-gray-900 mb-2">Speaker Recognition Results</h5>
                      <pre className="text-xs bg-white p-3 rounded border overflow-auto max-h-64 text-gray-800">
                        {JSON.stringify({
                          filename: selectedResult.filename,
                          duration: selectedResult.duration,
                          status: selectedResult.status,
                          created_at: selectedResult.created_at,
                          confidence_summary: selectedResult.confidence_summary,
                          speakers: selectedResult.speakers,
                          deepgram_enhancement: deepgramResponse?.speaker_enhancement
                        }, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty State */}
      {results.length === 0 && !audioData && (
        <div className="text-center py-12">
          <BarChart3 className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Inference Results Yet</h3>
          <p className="text-gray-500 mb-4">Upload an audio file to start speaker identification.</p>
        </div>
      )}
    </div>
  )
}