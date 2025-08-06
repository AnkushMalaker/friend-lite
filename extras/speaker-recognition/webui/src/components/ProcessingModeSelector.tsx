/**
 * ProcessingModeSelector Component - Unified processing mode selection
 * Provides consistent interface for selecting processing modes across all pages
 * Supports diarization, deepgram, hybrid, and plain modes
 */

import React from 'react'
import { Settings, Info } from 'lucide-react'
import { ProcessingMode } from '../services/speakerIdentification'
import { ProcessedAudio } from '../services/audioProcessing'

export interface ProcessingModeConfig {
  mode: ProcessingMode
  name: string
  description: string
  icon?: string
  color: string
  requirements?: string[]
  features: string[]
}

export interface ProcessingModeSelectorProps {
  selectedMode: ProcessingMode
  onModeChange: (mode: ProcessingMode) => void
  onProcessAudio: (mode: ProcessingMode) => Promise<void>
  audioData: ProcessedAudio | null
  isProcessing: boolean
  confidenceThreshold: number
  onConfidenceThresholdChange: (threshold: number) => void
  showSettings?: boolean
  compact?: boolean
  className?: string
}

const PROCESSING_MODES: ProcessingModeConfig[] = [
  {
    mode: 'diarization',
    name: 'Speaker Identification',
    description: 'Diarization + speaker recognition only',
    icon: '🎯',
    color: 'bg-blue-600 hover:bg-blue-700',
    features: ['Speaker diarization', 'Speaker identification', 'Confidence scoring']
  },
  {
    mode: 'deepgram',
    name: 'Transcribe + Identify',
    description: 'Full transcription with enhanced speaker ID',
    icon: '🚀',
    color: 'bg-green-600 hover:bg-green-700',
    requirements: ['Deepgram API key'],
    features: ['High-quality transcription', 'Speaker diarization', 'Enhanced speaker identification', 'Word-level timing']
  },
  {
    mode: 'hybrid',
    name: 'Hybrid Mode',
    description: 'Deepgram transcription + internal diarization',
    icon: '🔄',
    color: 'bg-purple-600 hover:bg-purple-700',
    requirements: ['Deepgram API key'],
    features: ['Best transcription quality', 'Accurate speaker segmentation', 'Enhanced identification', 'Optimal accuracy']
  },
  {
    mode: 'plain',
    name: 'Plain Diarize + Identify',
    description: 'Diarization + identification without Deepgram',
    icon: '⚡',
    color: 'bg-orange-600 hover:bg-orange-700',
    features: ['No external dependencies', 'Fast processing', 'Speaker diarization', 'Speaker identification']
  }
]

export const ProcessingModeSelector: React.FC<ProcessingModeSelectorProps> = ({
  selectedMode,
  onModeChange,
  onProcessAudio,
  audioData,
  isProcessing,
  confidenceThreshold,
  onConfidenceThresholdChange,
  showSettings = true,
  compact = false,
  className = ''
}) => {
  const selectedConfig = PROCESSING_MODES.find(m => m.mode === selectedMode) || PROCESSING_MODES[0]

  const handleProcessAudio = async (mode: ProcessingMode) => {
    if (!audioData) return
    await onProcessAudio(mode)
  }

  if (compact) {
    return (
      <div className={`space-y-3 ${className}`}>
        {/* Mode Selection Dropdown */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Processing Mode
          </label>
          <select
            value={selectedMode}
            onChange={(e) => onModeChange(e.target.value as ProcessingMode)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
            disabled={isProcessing}
          >
            {PROCESSING_MODES.map((config) => (
              <option key={config.mode} value={config.mode}>
                {config.icon} {config.name}
              </option>
            ))}
          </select>
        </div>

        {/* Confidence Threshold */}
        {showSettings && (
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
              onChange={(e) => onConfidenceThresholdChange(parseFloat(e.target.value))}
              className="w-full"
              disabled={isProcessing}
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Less Strict</span>
              <span>More Strict</span>
            </div>
          </div>
        )}

        {/* Process Button */}
        <button
          onClick={() => handleProcessAudio(selectedMode)}
          disabled={!audioData || isProcessing}
          className={`w-full flex items-center justify-center space-x-2 px-4 py-2 text-white rounded-md disabled:opacity-50 disabled:cursor-not-allowed ${selectedConfig.color}`}
        >
          <span>{selectedConfig.icon}</span>
          <span>{isProcessing ? 'Processing...' : `Start ${selectedConfig.name}`}</span>
        </button>
      </div>
    )
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Settings Panel */}
      {showSettings && (
        <div className="bg-gray-50 border rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Settings className="h-4 w-4 text-gray-600" />
            <h4 className="font-medium text-gray-900">Processing Settings</h4>
          </div>
          
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
              onChange={(e) => onConfidenceThresholdChange(parseFloat(e.target.value))}
              className="w-full"
              disabled={isProcessing}
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Less Strict</span>
              <span>More Strict</span>
            </div>
          </div>
        </div>
      )}

      {/* Mode Selection */}
      <div className="space-y-4">
        <h4 className="font-medium text-gray-900">Choose Processing Mode</h4>
        
        {/* Grid Layout for Modes */}
        <div className="grid md:grid-cols-2 gap-4">
          {PROCESSING_MODES.slice(0, 2).map((config) => (
            <div
              key={config.mode}
              className={`border rounded-lg p-4 cursor-pointer transition-colors ${
                selectedMode === config.mode 
                  ? 'border-blue-500 bg-blue-50' 
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => onModeChange(config.mode)}
            >
              <div className="flex items-start space-x-3">
                <span className="text-2xl">{config.icon}</span>
                <div className="flex-1">
                  <h5 className="font-medium text-gray-900">{config.name}</h5>
                  <p className="text-sm text-gray-600 mt-1">{config.description}</p>
                  
                  {/* Requirements */}
                  {config.requirements && (
                    <div className="mt-2">
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-yellow-100 text-yellow-800">
                        Requires: {config.requirements.join(', ')}
                      </span>
                    </div>
                  )}
                  
                  {/* Features */}
                  <div className="mt-2">
                    <div className="flex flex-wrap gap-1">
                      {config.features.slice(0, 2).map((feature, index) => (
                        <span key={index} className="text-xs text-gray-500">
                          • {feature}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Full-width mode for hybrid */}
        <div
          className={`border rounded-lg p-4 cursor-pointer transition-colors ${
            selectedMode === 'hybrid' 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-200 hover:border-gray-300'
          }`}
          onClick={() => onModeChange('hybrid')}
        >
          <div className="flex items-start space-x-3">
            <span className="text-2xl">🔄</span>
            <div className="flex-1">
              <h5 className="font-medium text-gray-900">Hybrid: Deepgram Transcription + Internal Diarization</h5>
              <p className="text-sm text-gray-600 mt-1">Best of both: High-quality transcription with accurate speaker segmentation</p>
              
              <div className="mt-2">
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-yellow-100 text-yellow-800">
                  Requires: Deepgram API key
                </span>
              </div>
              
              <div className="mt-2">
                <div className="flex flex-wrap gap-1">
                  {PROCESSING_MODES[2].features.map((feature, index) => (
                    <span key={index} className="text-xs text-gray-500">
                      • {feature}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Plain mode */}
        <div
          className={`border rounded-lg p-4 cursor-pointer transition-colors ${
            selectedMode === 'plain' 
              ? 'border-blue-500 bg-blue-50' 
              : 'border-gray-200 hover:border-gray-300'
          }`}
          onClick={() => onModeChange('plain')}
        >
          <div className="flex items-start space-x-3">
            <span className="text-2xl">⚡</span>
            <div className="flex-1">
              <h5 className="font-medium text-gray-900">Plain Diarize + Identify</h5>
              <p className="text-sm text-gray-600 mt-1">Diarization + identification without Deepgram dependency</p>
              
              <div className="mt-2">
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 text-green-800">
                  No external dependencies
                </span>
              </div>
              
              <div className="mt-2">
                <div className="flex flex-wrap gap-1">
                  {PROCESSING_MODES[3].features.map((feature, index) => (
                    <span key={index} className="text-xs text-gray-500">
                      • {feature}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Selected Mode Info */}
      {selectedConfig && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Info className="h-4 w-4 text-blue-600" />
            <h5 className="font-medium text-blue-900">Selected: {selectedConfig.name}</h5>
          </div>
          <p className="text-sm text-blue-800 mb-3">{selectedConfig.description}</p>
          
          <div className="space-y-2">
            <div>
              <span className="text-sm font-medium text-blue-900">Features:</span>
              <ul className="text-sm text-blue-800 ml-4 mt-1">
                {selectedConfig.features.map((feature, index) => (
                  <li key={index}>• {feature}</li>
                ))}
              </ul>
            </div>
            
            {selectedConfig.requirements && (
              <div>
                <span className="text-sm font-medium text-blue-900">Requirements:</span>
                <ul className="text-sm text-blue-800 ml-4 mt-1">
                  {selectedConfig.requirements.map((req, index) => (
                    <li key={index}>• {req}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Process Button */}
      <button
        onClick={() => handleProcessAudio(selectedMode)}
        disabled={!audioData || isProcessing}
        className={`w-full flex items-center justify-center space-x-2 px-6 py-4 text-white rounded-md disabled:opacity-50 disabled:cursor-not-allowed text-lg font-medium ${selectedConfig.color}`}
      >
        <span className="text-xl">{selectedConfig.icon}</span>
        <span>{isProcessing ? 'Processing...' : `Start ${selectedConfig.name}`}</span>
      </button>
    </div>
  )
}

export default ProcessingModeSelector