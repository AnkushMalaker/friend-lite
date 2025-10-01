import { useState, useEffect } from 'react'
import { Activity, RefreshCw } from 'lucide-react'
import { systemApi } from '../services/api'
import { useAuth } from '../contexts/AuthContext'
import ProcessPipelineView from '../components/processes/ProcessPipelineView'
import SystemHealthCards from '../components/processes/SystemHealthCards'
import ActiveTasksTable from '../components/processes/ActiveTasksTable'
import ProcessingHistory from '../components/processes/ProcessingHistory'
import ClientDetailModal from '../components/processes/ClientDetailModal'
import AllJobsView from '../components/processes/AllJobsView'

interface ProcessorOverview {
  pipeline_stats: {
    audio: PipelineStageStats
    transcription: PipelineStageStats
    memory: PipelineStageStats
    cropping: PipelineStageStats
  }
  system_health: {
    total_active_clients: number
    total_processing_tasks: number
    task_manager_healthy: boolean
    error_rate: number
    uptime_hours: number
  }
  queue_health: Record<string, string>
  recent_activity: ProcessingHistoryItem[]
}

interface PipelineStageStats {
  queue_size: number
  active_tasks: number
  avg_processing_time_ms: number
  success_rate: number
  throughput_per_minute: number
}

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


export default function Processes() {
  const [overviewData, setOverviewData] = useState<ProcessorOverview | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [selectedClientId, setSelectedClientId] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const { isAdmin } = useAuth()

  const loadProcessorOverview = async () => {
    if (!isAdmin) return

    try {
      setLoading(true)
      setError(null)

      const response = await systemApi.getProcessorOverview()
      setOverviewData(response.data)
      setLastUpdated(new Date())
    } catch (err: any) {
      setError(err.message || 'Failed to load processor overview')
    } finally {
      setLoading(false)
    }
  }

  // Auto-refresh effect
  useEffect(() => {
    if (!autoRefresh) return

    const interval = setInterval(() => {
      loadProcessorOverview()
    }, 5000) // Refresh every 5 seconds

    return () => clearInterval(interval)
  }, [autoRefresh, isAdmin])

  // Initial load
  useEffect(() => {
    loadProcessorOverview()
  }, [isAdmin])

  if (!isAdmin) {
    return (
      <div className="text-center">
        <Activity className="h-12 w-12 mx-auto mb-4 text-gray-400" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
          Access Restricted
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          You need administrator privileges to view process monitoring.
        </p>
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center space-x-2">
          <Activity className="h-6 w-6 text-blue-600" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Process Monitoring
          </h1>
        </div>
        <div className="flex items-center space-x-4">
          {lastUpdated && (
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Last updated: {lastUpdated.toLocaleTimeString()}
            </span>
          )}

          {/* Auto-refresh toggle */}
          <label className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span>Auto-refresh</span>
          </label>

          <button
            onClick={loadProcessorOverview}
            disabled={loading}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4 mb-6">
          <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}

      {overviewData && (
        <div className="space-y-6">
          {/* System Health Overview */}
          <SystemHealthCards data={overviewData.system_health} />

          {/* Processing Pipeline View */}
          <ProcessPipelineView
            pipelineStats={overviewData.pipeline_stats}
            queueHealth={overviewData.queue_health}
          />

          {/* Active Tasks and History */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <ActiveTasksTable
              onClientSelect={setSelectedClientId}
              refreshTrigger={lastUpdated}
            />
            <ProcessingHistory
              initialData={overviewData.recent_activity}
              refreshTrigger={lastUpdated}
            />
          </div>

          {/* All Jobs from MongoDB */}
          <AllJobsView refreshTrigger={lastUpdated} />
        </div>
      )}

      {/* Loading State */}
      {loading && !overviewData && (
        <div className="flex items-center justify-center py-12">
          <RefreshCw className="h-8 w-8 animate-spin text-blue-600 mr-3" />
          <span className="text-lg text-gray-600 dark:text-gray-400">Loading process data...</span>
        </div>
      )}

      {/* Client Detail Modal */}
      {selectedClientId && (
        <ClientDetailModal
          clientId={selectedClientId}
          onClose={() => setSelectedClientId(null)}
        />
      )}
    </div>
  )
}