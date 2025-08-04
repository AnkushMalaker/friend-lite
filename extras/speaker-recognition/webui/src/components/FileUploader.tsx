import React, { useRef, useState } from 'react'
import { Upload, X, FileAudio } from 'lucide-react'
import { formatFileSize, isAudioFile } from '../utils/fileHash'

interface FileUploaderProps {
  onUpload: (files: File[]) => void
  accept?: string
  multiple?: boolean
  disabled?: boolean
  className?: string
  title?: string
}

export default function FileUploader({
  onUpload,
  accept = '*',
  multiple = false,
  disabled = false,
  className = '',
  title
}: FileUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    if (!disabled) {
      setIsDragOver(true)
    }
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    
    if (disabled) return

    const files = Array.from(e.dataTransfer.files)
    handleFiles(files)
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return
    
    const files = Array.from(e.target.files)
    handleFiles(files)
  }

  const handleFiles = (files: File[]) => {
    // Filter for audio files if accept is for audio
    const validFiles = files.filter(file => {
      if (accept.includes('audio') || accept.includes('.wav') || accept.includes('.mp3')) {
        return isAudioFile(file)
      }
      return true
    })

    if (validFiles.length === 0) {
      alert('Please select valid audio files')
      return
    }

    // Check for large files and warn user
    const largeFiles = validFiles.filter(file => file.size > 50 * 1024 * 1024) // 50MB
    if (largeFiles.length > 0) {
      const fileNames = largeFiles.map(f => f.name).join(', ')
      const proceed = confirm(
        `Warning: Large files detected (${fileNames}). Processing may take time and use significant memory. Continue?`
      )
      if (!proceed) return
    }

    const filesToProcess = multiple ? validFiles : [validFiles[0]]
    setSelectedFiles(filesToProcess)
    onUpload(filesToProcess)
  }

  const removeFile = (index: number) => {
    const newFiles = selectedFiles.filter((_, i) => i !== index)
    setSelectedFiles(newFiles)
  }

  const openFileDialog = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  return (
    <div className={className}>
      {/* File Input (Hidden) */}
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={handleFileSelect}
        className="hidden"
        disabled={disabled}
      />

      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={openFileDialog}
        className={`
          relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
          ${disabled 
            ? 'border-gray-200 bg-gray-50 cursor-not-allowed' 
            : isDragOver 
              ? 'border-blue-400 bg-blue-50' 
              : 'border-gray-300 hover:border-gray-400'
          }
        `}
      >
        <div className="space-y-3">
          <div className="mx-auto">
            <Upload className={`h-12 w-12 mx-auto ${disabled ? 'text-gray-300' : 'text-gray-400'}`} />
          </div>
          
          <div>
            <p className={`text-lg font-medium ${disabled ? 'text-gray-400' : 'text-gray-900'}`}>
              {isDragOver ? 'Drop files here' : (title || 'Upload Audio Files')}
            </p>
            <p className={`text-sm ${disabled ? 'text-gray-300' : 'text-gray-500'}`}>
              Drag and drop or click to select files
            </p>
            {accept.includes('audio') && (
              <p className="text-xs text-gray-400 mt-1">
                Supported: WAV, FLAC, MP3, M4A, OGG
              </p>
            )}
          </div>
        </div>
        
        {disabled && (
          <div className="absolute inset-0 bg-gray-50 bg-opacity-50 rounded-lg"></div>
        )}
      </div>

      {/* Selected Files List */}
      {selectedFiles.length > 0 && (
        <div className="mt-4 space-y-2">
          <h4 className="text-sm font-medium text-gray-900">Selected Files:</h4>
          <div className="space-y-2">
            {selectedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center space-x-3">
                  <FileAudio className="h-5 w-5 text-blue-500" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">{file.name}</p>
                    <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                  </div>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    removeFile(index)
                  }}
                  className="p-1 text-gray-400 hover:text-gray-600"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}