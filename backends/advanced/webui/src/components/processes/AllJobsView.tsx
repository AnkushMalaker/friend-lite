import { useState, useEffect } from 'react'
import { Database, RefreshCw } from 'lucide-react'
import { systemApi } from '../../services/api'

interface Job {
  job_id: string
  job_type: string
  status: string
  user_id: string
  device_name: string
  created_at: string
  started_at?: string
  completed_at?: string
  progress_percent: number
  error_message?: string
  audio_uuid?: string
  conversation_id?: string
  client_id?: string
}

interface AllJobsViewProps {
  refreshTrigger?: Date | null
}

export default function AllJobsView({ refreshTrigger }: AllJobsViewProps) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadJobs = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await systemApi.getAllJobs()
      setJobs(response.data.jobs || [])
    } catch (err: any) {
      setError(err.message || 'Failed to load jobs')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadJobs()
  }, [refreshTrigger])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
      case 'processing':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'
      case 'failed':
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
      case 'queued':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-300'
    }
  }

  const getJobTypeLabel = (type: string) => {
    switch (type) {
      case 'pipeline':
        return 'Pipeline'
      case 'batch':
        return 'Batch'
      default:
        return type
    }
  }

  const formatDate = (dateString: string) => {
    if (!dateString) return 'N/A'
    const date = new Date(dateString)
    return date.toLocaleString()
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center space-x-2">
          <Database className="h-5 w-5 text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            All Jobs (MongoDB)
          </h2>
        </div>
        <button
          onClick={loadJobs}
          disabled={loading}
          className="flex items-center space-x-2 px-3 py-1 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded p-3 mb-4">
          <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}

      <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Showing last 100 jobs from MongoDB
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-900">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Job ID
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Type
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Status
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Device
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Progress
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Created
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Completed
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {jobs.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-gray-500 dark:text-gray-400">
                  {loading ? 'Loading...' : 'No jobs found'}
                </td>
              </tr>
            ) : (
              jobs.map((job) => (
                <tr key={job.job_id} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                  <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100 font-mono">
                    {job.job_id.substring(0, 8)}...
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                    {getJobTypeLabel(job.job_type)}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                      {job.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                    {job.device_name || job.client_id || 'N/A'}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                    {job.progress_percent.toFixed(1)}%
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">
                    {formatDate(job.created_at)}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">
                    {job.completed_at ? formatDate(job.completed_at) : '-'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {jobs.length > 0 && (
        <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
          Total jobs: {jobs.length}
        </div>
      )}
    </div>
  )
}