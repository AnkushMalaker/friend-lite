import { useState, useEffect } from 'react'
import { Settings, RefreshCw, CheckCircle, XCircle, AlertCircle, Activity, Users, Database, Server } from 'lucide-react'
import { systemApi } from '../services/api'
import { useAuth } from '../contexts/AuthContext'

interface HealthData {
  status: 'healthy' | 'partial' | 'unhealthy'
  services: Record<string, {
    healthy: boolean
    message?: string
  }>
  timestamp?: string
}

interface MetricsData {
  debug_tracker?: {
    total_files: number
    processed_files: number
    failed_files: number
  }
}

interface ProcessorStatus {
  audio_queue_size: number
  transcription_queue_size: number
  memory_queue_size: number
  active_tasks: number
}

interface ActiveClient {
  id: string
  user_id: string
  connected_at: string
  last_activity: string
}

export default function System() {
  const [healthData, setHealthData] = useState<HealthData | null>(null)
  const [readinessData, setReadinessData] = useState<any>(null)
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null)
  const [processorStatus, setProcessorStatus] = useState<ProcessorStatus | null>(null)
  const [activeClients, setActiveClients] = useState<ActiveClient[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const { isAdmin } = useAuth()

  const loadSystemData = async () => {
    if (!isAdmin) return

    try {
      setLoading(true)
      setError(null)

      const [health, readiness, metrics, processor, clients] = await Promise.allSettled([
        systemApi.getHealth(),
        systemApi.getReadiness(),
        systemApi.getMetrics().catch(() => ({ data: null })), // Optional endpoint
        systemApi.getProcessorStatus().catch(() => ({ data: null })), // Optional endpoint
        systemApi.getActiveClients().catch(() => ({ data: [] })), // Optional endpoint
      ])

      if (health.status === 'fulfilled') {
        setHealthData(health.value.data)
      }
      if (readiness.status === 'fulfilled') {
        setReadinessData(readiness.value.data)
      }
      if (metrics.status === 'fulfilled' && metrics.value.data) {
        setMetricsData(metrics.value.data)
      }
      if (processor.status === 'fulfilled' && processor.value.data) {
        setProcessorStatus(processor.value.data)
      }
      if (clients.status === 'fulfilled' && clients.value.data) {
        setActiveClients(clients.value.data)
      }

      setLastUpdated(new Date())
    } catch (err: any) {
      setError(err.message || 'Failed to load system data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadSystemData()
  }, [isAdmin])

  const getStatusIcon = (healthy: boolean) => {
    return healthy 
      ? <CheckCircle className="h-5 w-5 text-green-500" />
      : <XCircle className="h-5 w-5 text-red-500" />
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600'
      case 'partial': return 'text-yellow-600'
      default: return 'text-red-600'
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  if (!isAdmin) {
    return (
      <div className="text-center">
        <Settings className="h-12 w-12 mx-auto mb-4 text-gray-400" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
          Access Restricted
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          You need administrator privileges to view system monitoring.
        </p>
      </div>
    )
  }

  return (
    <div>
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center space-x-2">
          <Settings className="h-6 w-6 text-blue-600" />
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            System Monitoring
          </h1>
        </div>
        <div className="flex items-center space-x-4">
          {lastUpdated && (
            <span className="text-sm text-gray-600 dark:text-gray-400">
              Last updated: {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={loadSystemData}
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

      {/* Overall Health Status */}
      {healthData && (
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6 border border-gray-200 dark:border-gray-600 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Activity className="h-6 w-6 text-blue-600" />
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                System Health
              </h2>
            </div>
            <div className="flex items-center space-x-2">
              {healthData.status === 'healthy' && <CheckCircle className="h-6 w-6 text-green-500" />}
              {healthData.status === 'partial' && <AlertCircle className="h-6 w-6 text-yellow-500" />}
              {healthData.status === 'unhealthy' && <XCircle className="h-6 w-6 text-red-500" />}
              <span className={`font-semibold ${getStatusColor(healthData.status)}`}>
                {healthData.status.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Services Status */}
        {healthData?.services && (
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center">
              <Database className="h-5 w-5 mr-2 text-blue-600" />
              Services Status
            </h3>
            <div className="space-y-3">
              {Object.entries(healthData.services).map(([service, status]) => (
                <div key={service} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(status.healthy)}
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {service.replace('_', ' ').toUpperCase()}
                    </span>
                  </div>
                  {status.message && (
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {status.message}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Processor Status */}
        {processorStatus && (
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center">
              <Server className="h-5 w-5 mr-2 text-blue-600" />
              Processor Status
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
                <div className="text-sm text-gray-600 dark:text-gray-400">Audio Queue</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {processorStatus.audio_queue_size}
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
                <div className="text-sm text-gray-600 dark:text-gray-400">Transcription Queue</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {processorStatus.transcription_queue_size}
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
                <div className="text-sm text-gray-600 dark:text-gray-400">Memory Queue</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {processorStatus.memory_queue_size}
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
                <div className="text-sm text-gray-600 dark:text-gray-400">Active Tasks</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {processorStatus.active_tasks}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Active Clients */}
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center">
            <Users className="h-5 w-5 mr-2 text-blue-600" />
            Active Clients ({activeClients.length})
          </h3>
          {activeClients.length > 0 ? (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {activeClients.map((client) => (
                <div key={client.id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
                  <div>
                    <div className="font-medium text-gray-900 dark:text-gray-100">{client.id}</div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      User: {client.user_id}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Connected: {formatDate(client.connected_at)}
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Last: {formatDate(client.last_activity)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 dark:text-gray-400 text-center py-4">
              No active clients
            </p>
          )}
        </div>

        {/* Debug Metrics */}
        {metricsData?.debug_tracker && (
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
              Debug Metrics
            </h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
                <div className="text-sm text-gray-600 dark:text-gray-400">Total Files</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  {metricsData.debug_tracker.total_files}
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
                <div className="text-sm text-gray-600 dark:text-gray-400">Processed</div>
                <div className="text-2xl font-bold text-green-600">
                  {metricsData.debug_tracker.processed_files}
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
                <div className="text-sm text-gray-600 dark:text-gray-400">Failed</div>
                <div className="text-2xl font-bold text-red-600">
                  {metricsData.debug_tracker.failed_files}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Raw Data (Debug) */}
      {readinessData && (
        <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
          <details>
            <summary className="cursor-pointer text-lg font-semibold text-gray-900 dark:text-gray-100 hover:text-blue-600">
              View Raw Readiness Data
            </summary>
            <pre className="mt-4 p-4 bg-gray-100 dark:bg-gray-700 rounded-md text-sm overflow-x-auto">
              {JSON.stringify(readinessData, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  )
}