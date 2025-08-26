import { Radio } from 'lucide-react'
import { useSimpleAudioRecording } from '../hooks/useSimpleAudioRecording'
import SimplifiedControls from '../components/audio/SimplifiedControls'
import StatusDisplay from '../components/audio/StatusDisplay'
import AudioVisualizer from '../components/audio/AudioVisualizer'
import SimpleDebugPanel from '../components/audio/SimpleDebugPanel'

export default function LiveRecord() {
  const recording = useSimpleAudioRecording()

  return (
    <div>
      {/* Header */}
      <div className="flex items-center space-x-2 mb-6">
        <Radio className="h-6 w-6 text-blue-600" />
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Live Audio Recording
        </h1>
      </div>

      {/* Main Controls - Single START button */}
      <SimplifiedControls recording={recording} />

      {/* Status Display - Shows setup progress */}
      <StatusDisplay recording={recording} />

      {/* Audio Visualizer - Shows waveform when recording */}
      <AudioVisualizer 
        isRecording={recording.isRecording}
        analyser={recording.analyser}
      />

      {/* Instructions */}
      <div className="mt-8 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <h3 className="font-medium text-blue-800 dark:text-blue-200 mb-2">
          📝 How it Works
        </h3>
        <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
          <li>• <strong>One-click recording:</strong> Single button handles complete setup automatically</li>
          <li>• <strong>Sequential process:</strong> Mic access → WebSocket connection → Audio session → Streaming</li>
          <li>• <strong>Real-time processing:</strong> Audio streams to backend for transcription and memory extraction</li>
          <li>• <strong>Wyoming protocol:</strong> Structured communication ensures reliable data transmission</li>
          <li>• <strong>High quality audio:</strong> 16kHz mono with noise suppression and echo cancellation</li>
          <li>• <strong>View results:</strong> Check Conversations page for transcribed content and memories</li>
        </ul>
      </div>

      {/* Debug Information Panel */}
      <SimpleDebugPanel recording={recording} />
    </div>
  )
}