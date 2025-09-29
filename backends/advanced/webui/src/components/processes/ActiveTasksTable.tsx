import { useState, useEffect } from 'react'
import { Users, ExternalLink, ArrowUpDown, Search, RefreshCw } from 'lucide-react'
import { systemApi } from '../../services/api'

interface ProcessingTask {
  client_id: string
  user_id: string
  stages: Record<string, {
    status: string
    timestamp?: string
    metadata?: any
    completed?: boolean
    error?: string
  }>
}

interface ActiveTasksTableProps {
  onClientSelect: (clientId: string) => void
  refreshTrigger?: Date | null
}

export default function ActiveTasksTable({ onClientSelect, refreshTrigger }: ActiveTasksTableProps) {
  const [tasks, setTasks] = useState<ProcessingTask[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [sortField, setSortField] = useState<'client_id' | 'user_id' | 'stage_count'>('client_id')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')

  const loadActiveTasks = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await systemApi.getProcessorTasks()

      // Convert the response to our expected format
      const taskList = Object.entries(response.data).map(([clientId, taskData]: [string, any]) => ({
        client_id: clientId,
        user_id: taskData.user_id || 'Unknown',
        stages: taskData.stages || {}
      }))

      setTasks(taskList)
    } catch (err: any) {
      setError(err.message || 'Failed to load active tasks')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadActiveTasks()
  }, [refreshTrigger])

  const handleSort = (field: typeof sortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('asc')
    }
  }

  const getStageCount = (stages: Record<string, any>) => {
    return Object.keys(stages).length
  }

  const getActiveStage = (stages: Record<string, any>) => {
    // Find the most recent active stage
    const stageNames = ['audio', 'transcription', 'memory', 'cropping']
    for (const stageName of stageNames) {
      const stage = stages[stageName]
      if (stage && stage.status === 'started' && !stage.completed) {
        return stageName
      }
    }
    return 'idle'
  }

  const getStageDisplay = (stageName: string) => {
    const stageColors = {
      audio: 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300',
      transcription: 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300',
      memory: 'bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300',
      cropping: 'bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300',
      idle: 'bg-gray-100 text-gray-800 dark:bg-gray-900/40 dark:text-gray-300'
    }

    const color = stageColors[stageName as keyof typeof stageColors] || stageColors.idle

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${color}`}>
        {stageName.charAt(0).toUpperCase() + stageName.slice(1)}
      </span>
    )
  }

  // Filter and sort tasks
  const filteredTasks = tasks.filter(task =>
    task.client_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    task.user_id.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const sortedTasks = [...filteredTasks].sort((a, b) => {
    let aValue: any, bValue: any

    switch (sortField) {
      case 'stage_count':
        aValue = getStageCount(a.stages)
        bValue = getStageCount(b.stages)
        break
      case 'user_id':
        aValue = a.user_id
        bValue = b.user_id
        break
      default:
        aValue = a.client_id
        bValue = b.client_id
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
            Active Tasks ({sortedTasks.length})
          </h3>
        </div>
        <button
          onClick={loadActiveTasks}
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
            placeholder="Search by client ID or user ID..."
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
                  onClick={() => handleSort('client_id')}
                  className="flex items-center space-x-1 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
                >
                  <span>Client ID</span>
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
              <th className="text-left py-2 px-3">Current Stage</th>
              <th className="text-left py-2 px-3">
                <button
                  onClick={() => handleSort('stage_count')}
                  className="flex items-center space-x-1 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100"
                >
                  <span>Stages</span>
                  <ArrowUpDown className="h-4 w-4" />
                </button>
              </th>
              <th className="text-left py-2 px-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={5} className="text-center py-8">
                  <RefreshCw className="h-6 w-6 animate-spin mx-auto text-blue-600 mb-2" />
                  <span className="text-gray-600 dark:text-gray-400">Loading tasks...</span>
                </td>
              </tr>
            ) : sortedTasks.length === 0 ? (
              <tr>
                <td colSpan={5} className="text-center py-8 text-gray-500 dark:text-gray-400">
                  {tasks.length === 0 ? 'No active tasks' : 'No tasks match your search'}
                </td>
              </tr>
            ) : (
              sortedTasks.map((task) => (
                <tr
                  key={task.client_id}
                  className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50"
                >
                  <td className="py-3 px-3">
                    <code className="text-xs bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
                      {task.client_id}
                    </code>
                  </td>
                  <td className="py-3 px-3 text-gray-700 dark:text-gray-300">
                    {task.user_id}
                  </td>
                  <td className="py-3 px-3">
                    {getStageDisplay(getActiveStage(task.stages))}
                  </td>
                  <td className="py-3 px-3 text-gray-700 dark:text-gray-300">
                    {getStageCount(task.stages)}
                  </td>
                  <td className="py-3 px-3">
                    <button
                      onClick={() => onClientSelect(task.client_id)}
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