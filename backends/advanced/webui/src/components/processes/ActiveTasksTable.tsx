import { useState, useEffect } from 'react'
import { Users, ExternalLink, ArrowUpDown, Search, RefreshCw } from 'lucide-react'
import { systemApi } from '../../services/api'

interface ProcessingJob {
  job_id: string
  job_type: 'batch' | 'pipeline'
  user_id?: string
  device_name?: string
  client_id?: string
  audio_uuid?: string
  status: string
  created_at: string
  completed_at?: string

  // Batch job fields
  files?: Array<{
    filename: string
    status: string
    pipeline_job_id?: string
  }>
  total_files?: number
  processed_files?: number

  // Pipeline job fields
  pipeline_stages?: Array<{
    stage: string
    status: string
    enqueue_time?: string
    complete_time?: string
  }>
}

interface ActiveTasksTableProps {
  onClientSelect: (clientId: string) => void
  refreshTrigger?: Date | null
}

export default function ActiveTasksTable({ onClientSelect, refreshTrigger }: ActiveTasksTableProps) {
  const [jobs, setJobs] = useState<ProcessingJob[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [sortField, setSortField] = useState<'job_id' | 'user_id' | 'file_count'>('job_id')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')

  const loadActiveJobs = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await systemApi.getActivePipelineJobs()

      // Extract jobs array from response (active_jobs is a count, jobs is the array)
      const jobsArray = response.data.jobs || []
      setJobs(jobsArray)
    } catch (err: any) {
      setError(err.message || 'Failed to load active jobs')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadActiveJobs()
  }, [refreshTrigger])

  const handleSort = (field: typeof sortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('asc')
    }
  }

  const getStatusDisplay = (status: string) => {
    const statusColors = {
      processing: 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300',
      completed: 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300',
      failed: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300',
      pending: 'bg-gray-100 text-gray-800 dark:bg-gray-900/40 dark:text-gray-300'
    }

    const color = statusColors[status as keyof typeof statusColors] || statusColors.pending

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${color}`}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    )
  }

  const getProgressText = (job: ProcessingJob) => {
    if (job.job_type === 'pipeline') {
      // Pipeline jobs show stage progress
      const stages = job.pipeline_stages || []
      const completed = stages.filter(s => s.status === 'completed').length
      return `${completed}/${stages.length} stages`
    }

    // Batch jobs show file progress
    const total = job.total_files || (job.files?.length ?? 0)
    const processed = job.processed_files || 0
    const failed = total - processed

    if (failed > 0 && job.status === 'failed') {
      return `${processed}/${total} (${failed} failed)`
    }
    return `${processed}/${total} files`
  }

  // Filter and sort jobs
  const filteredJobs = jobs.filter(job =>
    job.job_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (job.user_id && job.user_id.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (job.device_name && job.device_name.toLowerCase().includes(searchTerm.toLowerCase())) ||
    (job.client_id && job.client_id.toLowerCase().includes(searchTerm.toLowerCase()))
  )

  const sortedJobs = [...filteredJobs].sort((a, b) => {
    let aValue: any, bValue: any

    switch (sortField) {
      case 'file_count':
        aValue = a.total_files || (a.files?.length ?? 0)
        bValue = b.total_files || (b.files?.length ?? 0)
        break
      case 'user_id':
        aValue = a.user_id || a.client_id || ''
        bValue = b.user_id || b.client_id || ''
        break
      default:
        aValue = a.job_id
        bValue = b.job_id
    }

    if (sortDirection === 'asc') {
      return aValue > bValue ? 1 : -1
    } else {
      return aValue < bValue ? 1 : -1
    }
  })

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Users className="h-5 w-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Active Pipeline Jobs ({sortedJobs.length})
          </h3>
        </div>
        <button
          onClick={loadActiveJobs}
          disabled={loading}
          className="flex items-center space-x-1 px-3 py-1 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Search */}
      <div className="mb-4">
        <div className="relative">
          <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search by job ID, user, or device..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
          <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-600">
              <th className="text-left py-2 px-3">
                <button
                  onClick={() => handleSort('job_id')}
                  className="flex items-center space-x-1 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
                >
                  <span>Job ID</span>
                  <ArrowUpDown className="h-4 w-4" />
                </button>
              </th>
              <th className="text-left py-2 px-3">
                <button
                  onClick={() => handleSort('user_id')}
                  className="flex items-center space-x-1 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
                >
                  <span>User</span>
                  <ArrowUpDown className="h-4 w-4" />
                </button>
              </th>
              <th className="text-left py-2 px-3">Device</th>
              <th className="text-left py-2 px-3">Status</th>
              <th className="text-left py-2 px-3">
                <button
                  onClick={() => handleSort('file_count')}
                  className="flex items-center space-x-1 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
                >
                  <span>Progress</span>
                  <ArrowUpDown className="h-4 w-4" />
                </button>
              </th>
              <th className="text-left py-2 px-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={6} className="text-center py-8">
                  <RefreshCw className="h-6 w-6 animate-spin mx-auto text-blue-600 mb-2" />
                  <span className="text-gray-600 dark:text-gray-400">Loading jobs...</span>
                </td>
              </tr>
            ) : sortedJobs.length === 0 ? (
              <tr>
                <td colSpan={6} className="text-center py-8 text-gray-500 dark:text-gray-400">
                  {jobs.length === 0 ? 'No active jobs' : 'No jobs match your search'}
                </td>
              </tr>
            ) : (
              sortedJobs.map((job) => (
                <tr
                  key={job.job_id}
                  className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50"
                >
                  <td className="py-3 px-3">
                    <code className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                      {job.job_id.substring(0, 8)}...
                    </code>
                  </td>
                  <td className="py-3 px-3 text-gray-700 dark:text-gray-300">
                    {job.user_id || job.client_id || 'N/A'}
                  </td>
                  <td className="py-3 px-3 text-gray-700 dark:text-gray-300">
                    {job.device_name || (job.job_type === 'pipeline' ? 'Pipeline' : 'N/A')}
                  </td>
                  <td className="py-3 px-3">
                    {getStatusDisplay(job.status)}
                  </td>
                  <td className="py-3 px-3 text-gray-700 dark:text-gray-300">
                    {getProgressText(job)}
                  </td>
                  <td className="py-3 px-3">
                    <button
                      onClick={() => onClientSelect(job.job_id)}
                      className="flex items-center space-x-1 text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                    >
                      <ExternalLink className="h-4 w-4" />
                      <span>View Details</span>
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}