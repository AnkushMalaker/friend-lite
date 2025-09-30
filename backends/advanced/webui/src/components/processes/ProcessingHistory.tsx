import { useState, useEffect } from 'react'
import { Clock, CheckCircle, XCircle, ChevronLeft, ChevronRight, RefreshCw, BarChart3 } from 'lucide-react'
import { systemApi } from '../../services/api'

interface ProcessingHistoryItem {
  client_id: string
  conversation_id?: string
  task_type: string
  started_at: string
  completed_at?: string
  duration_ms?: number
  status: string
  error?: string
}

interface ProcessingHistoryProps {
  initialData?: ProcessingHistoryItem[]
  refreshTrigger?: Date | null
}

export default function ProcessingHistory({ initialData = [], refreshTrigger }: ProcessingHistoryProps) {
  const [history, setHistory] = useState<ProcessingHistoryItem[]>(initialData)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [perPage] = useState(10)

  const loadHistory = async (page: number = currentPage) => {
    try {
      setLoading(true)
      setError(null)
      const response = await systemApi.getProcessorHistory(page, perPage)

      setHistory(response.data.history)
      setCurrentPage(response.data.pagination.page)
      setTotalPages(response.data.pagination.total_pages)
    } catch (err: any) {
      setError(err.message || 'Failed to load processing history')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (refreshTrigger) {
      loadHistory(1) // Refresh from first page
    }
  }, [refreshTrigger])

  useEffect(() => {
    if (initialData.length === 0) {
      loadHistory(1)
    }
  }, [])

  const formatDuration = (durationMs?: number) => {
    if (!durationMs) return 'N/A'
    if (durationMs < 1000) return `${Math.round(durationMs)}ms`
    if (durationMs < 60000) return `${(durationMs / 1000).toFixed(1)}s`
    return `${(durationMs / 60000).toFixed(1)}m`
  }

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300'
      case 'failed':
        return 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300'
      default:
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300'
    }
  }

  const getTaskTypeColor = (taskType: string) => {
    const colors = {
      memory: 'bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300',
      transcription_chunk: 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300',
      cropping: 'bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300'
    }
    return colors[taskType as keyof typeof colors] || 'bg-gray-100 text-gray-800 dark:bg-gray-900/40 dark:text-gray-300'
  }

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      loadHistory(newPage)
    }
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <BarChart3 className="h-5 w-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Processing History
          </h3>
        </div>
        <button
          onClick={() => loadHistory(1)}
          disabled={loading}
          className="flex items-center space-x-1 px-3 py-1 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
          <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}

      {/* History List */}
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="h-6 w-6 animate-spin text-blue-600 mr-2" />
            <span className="text-gray-600 dark:text-gray-400">Loading history...</span>
          </div>
        ) : history.length === 0 ? (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            No processing history available
          </div>
        ) : (
          history.map((item, index) => (
            <div
              key={`${item.client_id}-${item.conversation_id || 'no-conv'}-${item.started_at}-${item.task_type}-${index}`}
              className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-md"
            >
              <div className="flex items-center space-x-3 flex-1 min-w-0">
                {getStatusIcon(item.status)}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2 mb-1">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTaskTypeColor(item.task_type)}`}>
                      {item.task_type.replace('_', ' ')}
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(item.status)}`}>
                      {item.status}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 truncate">
                    Client: <code className="text-xs bg-gray-200 dark:bg-gray-600 px-1 rounded">{item.client_id}</code>
                    {item.conversation_id && (
                      <span className="ml-2">
                        Conv: <code className="text-xs bg-gray-200 dark:bg-gray-600 px-1 rounded">{item.conversation_id}</code>
                      </span>
                    )}
                  </div>
                  {item.error && (
                    <div className="text-xs text-red-600 dark:text-red-400 mt-1 truncate">
                      Error: {item.error}
                    </div>
                  )}
                </div>
              </div>
              <div className="text-right text-sm text-gray-600 dark:text-gray-400 ml-4">
                <div>{formatTime(item.started_at)}</div>
                <div className="text-xs">
                  {formatDuration(item.duration_ms)}
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Page {currentPage} of {totalPages}
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage <= 1 || loading}
              className="flex items-center space-x-1 px-3 py-1 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="h-4 w-4" />
              <span>Previous</span>
            </button>
            <button
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage >= totalPages || loading}
              className="flex items-center space-x-1 px-3 py-1 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <span>Next</span>
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}