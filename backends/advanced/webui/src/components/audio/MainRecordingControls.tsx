import { Mic, MicOff } from 'lucide-react'
import { UseAudioRecordingReturn } from '../../hooks/useAudioRecording'

interface MainRecordingControlsProps {
  recording: UseAudioRecordingReturn
}

export default function MainRecordingControls({ recording }: MainRecordingControlsProps) {
  const isHttps = window.location.protocol === 'https:'

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 mb-6">
      <div className="text-center">
        <div className="mb-6">
          {recording.isRecording ? (
            <button
              onClick={recording.stopRecording}
              className="w-20 h-20 bg-red-600 hover:bg-red-700 text-white rounded-full flex items-center justify-center transition-colors shadow-lg"
            >
              <MicOff className="h-8 w-8" />
            </button>
          ) : (
            <button
              onClick={recording.startRecording}
              disabled={!recording.canAccessMicrophone || recording.connectionStatus === 'connecting'}
              className="w-20 h-20 bg-blue-600 hover:bg-blue-700 text-white rounded-full flex items-center justify-center transition-colors shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Mic className="h-8 w-8" />
            </button>
          )}
        </div>

        <div className="space-y-2">
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {recording.isRecording ? 'Recording...' : 'Ready to Record'}
          </p>
          
          {recording.isRecording && (
            <p className="text-2xl font-mono text-blue-600 dark:text-blue-400">
              {recording.formatDuration(recording.recordingDuration)}
            </p>
          )}
          
          <p className="text-sm text-gray-600 dark:text-gray-400">
            {recording.isRecording 
              ? `Audio streaming via ${isHttps ? 'WSS (secure)' : 'WS'} to backend for processing`
              : recording.canAccessMicrophone 
                ? 'Click the microphone to start recording'
                : 'Secure connection required for microphone access'}
          </p>
        </div>
      </div>
    </div>
  )
}