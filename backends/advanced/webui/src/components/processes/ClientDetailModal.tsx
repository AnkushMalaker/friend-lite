import { useState, useEffect } from 'react'
import { X, User, Activity, Clock, CheckCircle, XCircle, RefreshCw, AlertTriangle } from 'lucide-react'
import { systemApi } from '../../services/api'

interface ClientProcessingDetail {
  client_id: string
  client_info: {
    user_id: string
    user_email: string
    current_audio_uuid?: string
    conversation_start_time?: string
    sample_rate?: number
  }
  processing_status: {
    stages: Record<string, {
      status: string
      timestamp?: string
      completed?: boolean
      error?: string
      metadata?: any
    }>
  }
  active_tasks: Array<{
    task_id: string
    task_name: string
    task_type: string
    created_at: string
    completed_at?: string
    error?: string
    cancelled: boolean
  }>
}

interface ClientDetailModalProps {
  clientId: string
  onClose: () => void
}

export default function ClientDetailModal({ clientId, onClose }: ClientDetailModalProps) {
  const [clientDetail, setClientDetail] = useState<ClientProcessingDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadClientDetail = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await systemApi.getClientProcessingDetail(clientId)
      setClientDetail(response.data)
    } catch (err: any) {
      setError(err.message || 'Failed to load client details')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadClientDetail()
  }, [clientId])

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  const getStageIcon = (status: string, completed?: boolean, error?: string) => {
    if (error) return <XCircle className="h-4 w-4 text-red-500" />
    if (completed) return <CheckCircle className="h-4 w-4 text-green-500" />
    if (status === 'started') return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />
    return <Clock className="h-4 w-4 text-gray-400" />
  }

  const getStageStatus = (status: string, completed?: boolean, error?: string) => {
    if (error) return 'Failed'
    if (completed) return 'Completed'
    if (status === 'started') return 'Processing'
    return 'Pending'
  }

  const getStageColor = (status: string, completed?: boolean, error?: string) => {
    if (error) return 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
    if (completed) return 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
    if (status === 'started') return 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-900/20'
    return 'border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800'
  }

  const getTaskStatusIcon = (task: ClientProcessingDetail['active_tasks'][0]) => {
    if (task.cancelled) return <XCircle className="h-4 w-4 text-orange-500" />
    if (task.error) return <XCircle className="h-4 w-4 text-red-500" />
    if (task.completed_at) return <CheckCircle className="h-4 w-4 text-green-500" />
    return <Activity className="h-4 w-4 text-blue-500 animate-pulse" />
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-2">
            <User className="h-6 w-6 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              Client Details
            </h2>
            <code className="text-sm bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
              {clientId}
            </code>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={loadClientDetail}
              disabled={loading}
              className="flex items-center space-x-1 px-3 py-1 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-80px)]">
          {loading && !clientDetail && (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="h-8 w-8 animate-spin text-blue-600 mr-3" />
              <span className="text-lg text-gray-600 dark:text-gray-400">Loading client details...</span>
            </div>
          )}

          {error && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <div className="flex items-center">
                <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
                <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
              </div>
            </div>
          )}

          {clientDetail && (
            <div className="space-y-6">
              {/* Client Information */}
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-3">
                  Client Information
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium text-gray-600 dark:text-gray-400">User ID</label>
                    <p className="text-gray-900 dark:text-gray-100">{clientDetail.client_info.user_id}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-600 dark:text-gray-400">User Email</label>
                    <p className="text-gray-900 dark:text-gray-100">{clientDetail.client_info.user_email}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-600 dark:text-gray-400">Current Audio UUID</label>
                    <p className="text-gray-900 dark:text-gray-100">
                      {clientDetail.client_info.current_audio_uuid ? (
                        <code className="text-xs bg-gray-200 dark:bg-gray-600 px-2 py-1 rounded">
                          {clientDetail.client_info.current_audio_uuid}
                        </code>
                      ) : (
                        'None'
                      )}
                    </p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-600 dark:text-gray-400">Sample Rate</label>
                    <p className="text-gray-900 dark:text-gray-100">
                      {clientDetail.client_info.sample_rate ? `${clientDetail.client_info.sample_rate} Hz` : 'N/A'}
                    </p>
                  </div>
                  {clientDetail.client_info.conversation_start_time && (
                    <div className="md:col-span-2">
                      <label className="text-sm font-medium text-gray-600 dark:text-gray-400">Conversation Started</label>
                      <p className="text-gray-900 dark:text-gray-100">
                        {formatTime(clientDetail.client_info.conversation_start_time)}
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Processing Stages */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-3">
                  Processing Stages
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(clientDetail.processing_status.stages || {}).map(([stageName, stage]) => (
                    <div
                      key={stageName}
                      className={`p-4 rounded-lg border ${getStageColor(stage.status, stage.completed, stage.error)}`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          {getStageIcon(stage.status, stage.completed, stage.error)}
                          <h4 className="font-medium text-gray-900 dark:text-gray-100 capitalize">
                            {stageName}
                          </h4>
                        </div>
                        <span className="text-sm text-gray-600 dark:text-gray-400">
                          {getStageStatus(stage.status, stage.completed, stage.error)}
                        </span>
                      </div>
                      {stage.timestamp && (
                        <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">
                          {formatTime(stage.timestamp)}
                        </p>
                      )}
                      {stage.error && (
                        <p className="text-xs text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/20 p-2 rounded">
                          {stage.error}
                        </p>
                      )}
                      {stage.metadata && Object.keys(stage.metadata).length > 0 && (
                        <div className="mt-2">
                          <details className="text-xs">
                            <summary className="cursor-pointer text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200">
                              View Metadata
                            </summary>
                            <pre className="mt-2 p-2 bg-gray-100 dark:bg-gray-600 rounded text-xs overflow-x-auto">
                              {JSON.stringify(stage.metadata, null, 2)}
                            </pre>
                          </details>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Active Tasks */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-3">
                  Active Tasks ({clientDetail.active_tasks.length})
                </h3>
                {clientDetail.active_tasks.length === 0 ? (
                  <p className="text-gray-500 dark:text-gray-400 text-center py-4">
                    No active tasks
                  </p>
                ) : (
                  <div className="space-y-3">
                    {clientDetail.active_tasks.map((task) => (
                      <div
                        key={task.task_id}
                        className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            {getTaskStatusIcon(task)}
                            <h4 className="font-medium text-gray-900 dark:text-gray-100">
                              {task.task_name}
                            </h4>
                            <span className="px-2 py-1 bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 text-xs rounded">
                              {task.task_type}
                            </span>
                          </div>
                          <code className="text-xs bg-gray-200 dark:bg-gray-600 px-2 py-1 rounded">
                            {task.task_id}
                          </code>
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <label className="text-gray-600 dark:text-gray-400">Created</label>
                            <p className="text-gray-900 dark:text-gray-100">{formatTime(task.created_at)}</p>
                          </div>
                          {task.completed_at && (
                            <div>
                              <label className="text-gray-600 dark:text-gray-400">Completed</label>
                              <p className="text-gray-900 dark:text-gray-100">{formatTime(task.completed_at)}</p>
                            </div>
                          )}
                        </div>
                        {task.error && (
                          <div className="mt-2 p-2 bg-red-100 dark:bg-red-900/20 rounded">
                            <p className="text-xs text-red-600 dark:text-red-400">{task.error}</p>
                          </div>
                        )}
                        {task.cancelled && (
                          <div className="mt-2 p-2 bg-orange-100 dark:bg-orange-900/20 rounded">
                            <p className="text-xs text-orange-600 dark:text-orange-400">Task was cancelled</p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}