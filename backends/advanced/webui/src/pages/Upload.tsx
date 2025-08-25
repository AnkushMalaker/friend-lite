import React, { useState, useCallback } from 'react'
import { Upload as UploadIcon, File, X, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react'
import { uploadApi } from '../services/api'
import { useAuth } from '../contexts/AuthContext'

interface UploadFile {
  file: File
  id: string
  status: 'pending' | 'uploading' | 'success' | 'error'
  error?: string
}

export default function Upload() {
  const [files, setFiles] = useState<UploadFile[]>([])
  const [isUploading, setIsUploading] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

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

  const uploadFiles = async () => {
    if (files.length === 0) return

    setIsUploading(true)
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

      await uploadApi.uploadAudioFiles(formData, (progress) => {
        setUploadProgress(progress)
      })
      
      // Mark all files as successful
      setFiles(prevFiles => 
        prevFiles.map(f => ({ ...f, status: 'success' as const }))
      )

    } catch (error: any) {
      console.error('Upload failed:', error)
      
      // Mark all files as failed
      setFiles(prevFiles => 
        prevFiles.map(f => ({ 
          ...f, 
          status: 'error' as const, 
          error: error.message || 'Upload failed' 
        }))
      )
    } finally {
      setIsUploading(false)
      setUploadProgress(100)
    }
  }

  const clearCompleted = () => {
    setFiles(files.filter(f => f.status === 'pending' || f.status === 'uploading'))
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
                disabled={isUploading || files.every(f => f.status !== 'pending')}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isUploading ? 'Uploading...' : 'Upload All'}
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

      {/* Upload Progress */}
      {isUploading && (
        <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
              Processing audio files...
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
          <p className="text-xs text-blue-600 dark:text-blue-400 mt-2">
            Note: Processing may take up to 5 minutes depending on file size and quantity.
          </p>
        </div>
      )}

      {/* Instructions */}
      <div className="mt-8 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
        <h3 className="font-medium text-yellow-800 dark:text-yellow-200 mb-2">
          📝 Upload Instructions
        </h3>
        <ul className="text-sm text-yellow-700 dark:text-yellow-300 space-y-1">
          <li>• Audio files will be processed sequentially for transcription and memory extraction</li>
          <li>• Processing time varies based on audio length (roughly 3x the audio duration + 60s)</li>
          <li>• Large files or multiple files may cause timeout errors - this is normal</li>
          <li>• Check the Conversations tab to see processed results</li>
          <li>• Supported formats: WAV, MP3, M4A, FLAC</li>
        </ul>
      </div>
    </div>
  )
}