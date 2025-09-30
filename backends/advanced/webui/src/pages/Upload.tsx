import React, { useState, useCallback, useEffect } from 'react'
import { Upload as UploadIcon, File, X, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react'
import { uploadApi, systemApi } from '../services/api'
import { useAuth } from '../contexts/AuthContext'

interface UploadFile {
  file: File
  id: string
  status: 'pending' | 'uploading' | 'success' | 'error'
  error?: string
}

// Legacy JobStatus interface - kept for backward compatibility
interface JobStatus {
  job_id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  total_files: number
  processed_files: number
  current_file?: string
  progress_percent: number
  files?: Array<{
    filename: string
    client_id: string
    status: 'pending' | 'processing' | 'completed' | 'failed'
    transcription_status?: string
    memory_status?: string
    error_message?: string
  }>
}

// New unified processing interfaces
interface ProcessingTask {
  client_id: string
  user_id: string
  status: 'processing' | 'complete'
  stages: Record<string, {
    status?: string
    completed?: boolean
    error?: string
    metadata?: any
    timestamp?: number
  }>
}

// UploadSessionData interface removed - replaced by unified processor tasks polling

interface UploadSession {
  job_id: string
  file_names: string[]
  started_at: number
  upload_completed: boolean
  total_files: number
}

export default function Upload() {
  const [files, setFiles] = useState<UploadFile[]>([])
  const [dragActive, setDragActive] = useState(false)

  // Three-phase state management
  const [uploadPhase, setUploadPhase] = useState<'idle' | 'uploading' | 'completed'>('idle')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [processingPhase, setProcessingPhase] = useState<'idle' | 'starting' | 'active' | 'completed'>('idle')
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)

  // Polling configuration
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [refreshInterval, setRefreshInterval] = useState(2000) // 2s default for upload page
  const [isPolling, setIsPolling] = useState(false)

  const { isAdmin } = useAuth()

  const generateId = () => Math.random().toString(36).substr(2, 9)

  const handleFileSelect = (selectedFiles: FileList | null) => {
    if (!selectedFiles) return

    const audioFiles = Array.from(selectedFiles).filter(file =>
      file.type.startsWith('audio/') ||
      file.name.toLowerCase().endsWith('.wav') ||
      file.name.toLowerCase().endsWith('.mp3') ||
      file.name.toLowerCase().endsWith('.m4a') ||
      file.name.toLowerCase().endsWith('.flac')
    )

    const newFiles: UploadFile[] = audioFiles.map(file => ({
      file,
      id: generateId(),
      status: 'pending'
    }))

    setFiles(prevFiles => [...prevFiles, ...newFiles])

    // Reset phases when adding files after completion
    if (processingPhase === 'completed') {
      setProcessingPhase('idle')
      setUploadPhase('idle')
      setJobStatus(null)
    }
  }

  const removeFile = (id: string) => {
    setFiles(files.filter(f => f.id !== id))
  }

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    handleFileSelect(e.dataTransfer.files)
  }, [])

  // localStorage persistence
  const saveSession = (session: UploadSession) => {
    localStorage.setItem('upload_session', JSON.stringify(session))
  }

  const getStoredSession = (): UploadSession | null => {
    const saved = localStorage.getItem('upload_session')
    return saved ? JSON.parse(saved) : null
  }

  const clearStoredSession = () => {
    localStorage.removeItem('upload_session')
  }

  // Resume session on page load
  useEffect(() => {
    const session = getStoredSession()
    if (session) {
      setProcessingPhase('active')
      setIsPolling(true)
      // Use unified polling without session dependency
      pollProcessingStatus()
    }
  }, [])

  // Polling effect
  useEffect(() => {
    if (!autoRefresh || !isPolling) return

    const interval = setInterval(() => {
      pollProcessingStatus()
    }, refreshInterval)

    return () => clearInterval(interval)
  }, [autoRefresh, refreshInterval, isPolling])

  // Job-based polling - polls active pipeline jobs
  const pollProcessingStatus = async () => {
    try {
      // Get active pipeline jobs (response has {active_jobs: count, jobs: array})
      const jobsResponse = await systemApi.getActivePipelineJobs()
      const jobsArray = jobsResponse.data.jobs || []

      // Check if processing is complete
      const allComplete = jobsArray.length > 0 && jobsArray.every((job: any) => job.status === 'completed')
      const noActiveJobs = jobsArray.length === 0 && processingPhase === 'active'

      if (allComplete || noActiveJobs) {
        setIsPolling(false)
        setProcessingPhase('completed')
        clearStoredSession()

        // Check for errors in completed jobs
        const hasErrors = jobsArray.some((job: any) => job.status === 'failed' || job.failed_files > 0)

        setFiles(prevFiles =>
          prevFiles.map(f => ({
            ...f,
            status: hasErrors ? 'error' : 'success',
            error: hasErrors ? 'Processing failed' : undefined
          }))
        )
      }
    } catch (error) {
      console.error('Failed to poll processing status:', error)
    }
  }

  // Legacy job polling for backward compatibility
  const pollJobStatus = async (jobId: string) => {
    try {
      // Use new unified polling (no session dependency)
      await pollProcessingStatus()

      // Also get legacy job status for progress display (if available)
      try {
        const response = await uploadApi.getJobStatus(jobId)
        const status: JobStatus = response.data
        setJobStatus(status)
      } catch (jobError) {
        console.log('Legacy job status not available, using unified polling only')
      }
    } catch (error) {
      console.error('Failed to poll unified processing status:', error)
      // Fallback to legacy job polling
      try {
        const response = await uploadApi.getJobStatus(jobId)
        const status: JobStatus = response.data
        setJobStatus(status)

        if (status.status === 'completed' || status.status === 'failed') {
          setIsPolling(false)
          setProcessingPhase('completed')
          clearStoredSession()

          setFiles(prevFiles =>
            prevFiles.map(f => ({
              ...f,
              status: status.status === 'completed' ? 'success' : 'error'
            }))
          )
        }
      } catch (fallbackError) {
        console.error('All polling methods failed:', fallbackError)
      }
    }
  }

  const uploadFiles = async () => {
    if (files.length === 0) return

    // Phase 1: File Upload
    setUploadPhase('uploading')
    setUploadProgress(0)

    try {
      const formData = new FormData()
      files.forEach(({ file }) => {
        formData.append('files', file)
      })

      // Update all files to uploading status
      setFiles(prevFiles =>
        prevFiles.map(f => ({ ...f, status: 'uploading' as const }))
      )

      // Phase 1: Upload files and get job ID
      const response = await uploadApi.uploadAudioFilesAsync(formData, (progress) => {
        setUploadProgress(progress)
      })

      // Phase 2: Job Creation
      setUploadPhase('completed')
      setProcessingPhase('starting')

      const jobData = response.data
      const jobId = jobData.job_id || jobData.jobs?.[0]?.job_id

      if (!jobId) {
        throw new Error('No job ID received from server')
      }

      // Save session for disconnection handling
      const session: UploadSession = {
        job_id: jobId,
        file_names: files.map(f => f.file.name),
        started_at: Date.now(),
        upload_completed: true,
        total_files: files.length
      }
      saveSession(session)

      // Phase 3: Start polling for processing status
      setProcessingPhase('active')
      setIsPolling(true)
      pollJobStatus(jobId)

    } catch (error: any) {
      console.error('Upload failed:', error)

      setUploadPhase('idle')
      setProcessingPhase('idle')

      // Mark all files as failed
      setFiles(prevFiles =>
        prevFiles.map(f => ({
          ...f,
          status: 'error' as const,
          error: error.message || 'Upload failed'
        }))
      )
    }
  }

  const clearCompleted = () => {
    setFiles(files.filter(f => f.status === 'pending' || f.status === 'uploading'))
    if (processingPhase === 'completed') {
      setProcessingPhase('idle')
      setUploadPhase('idle')
      setJobStatus(null)
      clearStoredSession()
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getStatusIcon = (status: UploadFile['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-500" />
      case 'uploading':
        return <RefreshCw className="h-5 w-5 text-blue-500 animate-spin" />
      default:
        return <File className="h-5 w-5 text-gray-500" />
    }
  }

  if (!isAdmin) {
    return (
      <div className="text-center">
        <UploadIcon className="h-12 w-12 mx-auto mb-4 text-gray-400" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
          Access Restricted
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          You need administrator privileges to upload audio files.
        </p>
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center space-x-2 mb-6">
        <UploadIcon className="h-6 w-6 text-blue-600" />
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Upload Audio Files
        </h1>
      </div>

      {/* Drop Zone */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive
            ? 'border-blue-400 bg-blue-50 dark:bg-blue-900/10'
            : 'border-gray-300 dark:border-gray-600 hover:border-blue-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <UploadIcon className="h-12 w-12 mx-auto mb-4 text-gray-400" />
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
          Drop audio files here or click to browse
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Supported formats: WAV, MP3, M4A, FLAC
        </p>
        
        <input
          type="file"
          multiple
          accept="audio/*,.wav,.mp3,.m4a,.flac"
          onChange={(e) => handleFileSelect(e.target.files)}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <button
          onClick={() => (document.querySelector('input[type="file"]') as HTMLInputElement)?.click()}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Select Files
        </button>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="mt-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Files ({files.length})
            </h2>
            <div className="flex space-x-2">
              <button
                onClick={clearCompleted}
                className="px-3 py-1 text-sm bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
              >
                Clear Completed
              </button>
              <button
                onClick={uploadFiles}
                disabled={uploadPhase !== 'idle' || processingPhase !== 'idle' || files.every(f => f.status !== 'pending')}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {uploadPhase === 'uploading' ? 'Uploading...' :
                 processingPhase === 'starting' ? 'Starting...' :
                 processingPhase === 'active' ? 'Processing...' :
                 'Upload All'}
              </button>
            </div>
          </div>

          <div className="space-y-2">
            {files.map((uploadFile) => (
              <div
                key={uploadFile.id}
                className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600"
              >
                <div className="flex items-center space-x-3 flex-1">
                  {getStatusIcon(uploadFile.status)}
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-gray-900 dark:text-gray-100 truncate">
                      {uploadFile.file.name}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      {formatFileSize(uploadFile.file.size)}
                      {uploadFile.error && (
                        <span className="text-red-600 dark:text-red-400 ml-2">
                          • {uploadFile.error}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <span className={`text-sm font-medium ${
                    uploadFile.status === 'success' ? 'text-green-600' :
                    uploadFile.status === 'error' ? 'text-red-600' :
                    uploadFile.status === 'uploading' ? 'text-blue-600' :
                    'text-gray-600 dark:text-gray-400'
                  }`}>
                    {uploadFile.status.charAt(0).toUpperCase() + uploadFile.status.slice(1)}
                  </span>
                  
                  {uploadFile.status === 'pending' && (
                    <button
                      onClick={() => removeFile(uploadFile.id)}
                      className="p-1 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Phase 1: Upload Progress */}
      {uploadPhase === 'uploading' && (
        <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
              Uploading files... ({files.length} files)
            </span>
            <span className="text-sm text-blue-600 dark:text-blue-400">
              {uploadProgress}%
            </span>
          </div>
          <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Phase 2: Job Creation */}
      {processingPhase === 'starting' && (
        <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-green-700 dark:text-green-300">
              Files uploaded. Starting processing jobs...
            </span>
            <RefreshCw className="h-4 w-4 text-green-600 animate-spin" />
          </div>
        </div>
      )}

      {/* Phase 3: Processing Status with Configurable Refresh */}
      {processingPhase === 'active' && jobStatus && (
        <div className="mt-6 space-y-4">
          {/* Refresh Controls */}
          <div className="flex items-center justify-between p-4 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg">
            <div className="flex items-center space-x-4">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">Auto-refresh</span>
              </label>

              <select
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(Number(e.target.value))}
                disabled={!autoRefresh}
                className="text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 disabled:opacity-50"
              >
                <option value={500}>0.5s</option>
                <option value={1000}>1s</option>
                <option value={2000}>2s</option>
                <option value={5000}>5s</option>
                <option value={10000}>10s</option>
              </select>
            </div>

            <button
              onClick={() => pollProcessingStatus()}
              className="flex items-center space-x-1 px-3 py-1 text-sm bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Refresh Now</span>
            </button>
          </div>

          {/* Processing Status */}
          <div className="p-4 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-purple-700 dark:text-purple-300">
                Processing file {jobStatus.processed_files + 1}/{jobStatus.total_files}
                {jobStatus.current_file && `: ${jobStatus.current_file}`}
              </span>
              <span className="text-sm text-purple-600 dark:text-purple-400">
                {Math.round(jobStatus.progress_percent)}%
              </span>
            </div>

            <div className="w-full bg-purple-200 dark:bg-purple-800 rounded-full h-2 mb-2">
              <div
                className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${jobStatus.progress_percent}%` }}
              />
            </div>

            <p className="text-xs text-purple-600 dark:text-purple-400 mt-2">
              Processing may take up to 3x audio duration + 60s. Status updates every {refreshInterval/1000}s.
            </p>
          </div>

          {/* Per-File Status */}
          {jobStatus.files && jobStatus.files.length > 0 && (
            <div className="p-4 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg">
              <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">File Processing Status</h4>
              <div className="space-y-2">
                {jobStatus.files.map((file, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <span className="text-sm text-gray-700 dark:text-gray-300 truncate">
                      {file.filename}
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        file.status === 'completed' ? 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300' :
                        file.status === 'processing' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300' :
                        file.status === 'failed' ? 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300' :
                        'bg-gray-100 text-gray-800 dark:bg-gray-900/40 dark:text-gray-300'
                      }`}>
                        {file.status.charAt(0).toUpperCase() + file.status.slice(1)}
                      </span>
                      {file.status === 'processing' && (
                        <RefreshCw className="h-3 w-3 text-blue-500 animate-spin" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Completion Status */}
      {processingPhase === 'completed' && (
        <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
          <div className="flex items-center space-x-2">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="text-sm font-medium text-green-700 dark:text-green-300">
              All files processed successfully! Check the Conversations tab to see results.
            </span>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="mt-8 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
        <h3 className="font-medium text-yellow-800 dark:text-yellow-200 mb-2">
          📝 Upload Instructions
        </h3>
        <ul className="text-sm text-yellow-700 dark:text-yellow-300 space-y-1">
          <li>• <strong>Phase 1:</strong> Files upload quickly to server (progress bar shows transfer)</li>
          <li>• <strong>Phase 2:</strong> Processing jobs created (immediate)</li>
          <li>• <strong>Phase 3:</strong> Audio processing (transcription + memory extraction, ~3x audio duration)</li>
          <li>• You can safely navigate away - processing continues in background</li>
          <li>• Refresh rate is configurable (0.5s to 10s) during processing</li>
          <li>• Check Conversations tab for final results</li>
          <li>• Supported formats: WAV, MP3, M4A, FLAC</li>
        </ul>
      </div>
    </div>
  )
}