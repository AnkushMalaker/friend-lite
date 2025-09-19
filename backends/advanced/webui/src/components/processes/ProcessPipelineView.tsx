import { ArrowRight, Volume2, FileText, Brain, Scissors, CheckCircle, AlertTriangle, Clock } from 'lucide-react'

interface PipelineStageStats {
  queue_size: number
  active_tasks: number
  avg_processing_time_ms: number
  success_rate: number
  throughput_per_minute: number
}

interface ProcessPipelineViewProps {
  pipelineStats: {
    audio: PipelineStageStats
    transcription: PipelineStageStats
    memory: PipelineStageStats
    cropping: PipelineStageStats
  }
  queueHealth: Record<string, string>
}

export default function ProcessPipelineView({ pipelineStats, queueHealth }: ProcessPipelineViewProps) {
  const stages = [
    {
      name: 'Audio',
      icon: Volume2,
      key: 'audio' as keyof typeof pipelineStats,
      color: 'blue',
      description: 'Audio chunk processing'
    },
    {
      name: 'Transcription',
      icon: FileText,
      key: 'transcription' as keyof typeof pipelineStats,
      color: 'green',
      description: 'Speech-to-text conversion'
    },
    {
      name: 'Memory',
      icon: Brain,
      key: 'memory' as keyof typeof pipelineStats,
      color: 'purple',
      description: 'Memory extraction'
    },
    {
      name: 'Cropping',
      icon: Scissors,
      key: 'cropping' as keyof typeof pipelineStats,
      color: 'orange',
      description: 'Audio file optimization'
    }
  ]

  const getHealthIcon = (health: string) => {
    switch (health) {
      case 'healthy':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'busy':
        return <Clock className="h-4 w-4 text-yellow-500" />
      case 'overloaded':
        return <AlertTriangle className="h-4 w-4 text-red-500" />
      default:
        return <CheckCircle className="h-4 w-4 text-gray-400" />
    }
  }

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
      case 'busy': return 'border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20'
      case 'overloaded': return 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20'
      default: return 'border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800/20'
    }
  }

  const getStageColor = (color: string) => {
    const colors = {
      blue: 'text-blue-600 bg-blue-100 dark:bg-blue-900/20',
      green: 'text-green-600 bg-green-100 dark:bg-green-900/20',
      purple: 'text-purple-600 bg-purple-100 dark:bg-purple-900/20',
      orange: 'text-orange-600 bg-orange-100 dark:bg-orange-900/20'
    }
    return colors[color as keyof typeof colors] || colors.blue
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6">
        Processing Pipeline
      </h3>

      {/* Pipeline Stages */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 lg:gap-2">
        {stages.map((stage, index) => {
          const stats = pipelineStats[stage.key]
          const health = queueHealth[stage.key] || 'idle'
          const Icon = stage.icon

          return (
            <div key={stage.key} className="flex items-center">
              {/* Stage Card */}
              <div className={`flex-1 p-4 rounded-lg border-2 ${getHealthColor(health)}`}>
                {/* Stage Header */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <div className={`p-2 rounded-full ${getStageColor(stage.color)}`}>
                      <Icon className="h-4 w-4" />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-gray-100">
                        {stage.name}
                      </h4>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        {stage.description}
                      </p>
                    </div>
                  </div>
                  {getHealthIcon(health)}
                </div>

                {/* Stage Stats */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-600 dark:text-gray-400">Queue</span>
                    <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                      {stats.queue_size}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-600 dark:text-gray-400">Active</span>
                    <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                      {stats.active_tasks}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-600 dark:text-gray-400">Avg Time</span>
                    <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                      {stats.avg_processing_time_ms < 1000
                        ? `${Math.round(stats.avg_processing_time_ms)}ms`
                        : `${(stats.avg_processing_time_ms / 1000).toFixed(1)}s`
                      }
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-600 dark:text-gray-400">Success</span>
                    <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                      {(stats.success_rate * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {/* Health Status */}
                <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-600">
                  <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                    health === 'healthy' ? 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300' :
                    health === 'busy' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300' :
                    health === 'overloaded' ? 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300' :
                    'bg-gray-100 text-gray-800 dark:bg-gray-900/40 dark:text-gray-300'
                  }`}>
                    {health.charAt(0).toUpperCase() + health.slice(1)}
                  </span>
                </div>
              </div>

              {/* Arrow (except for last stage) */}
              {index < stages.length - 1 && (
                <div className="hidden lg:flex items-center justify-center w-8">
                  <ArrowRight className="h-5 w-5 text-gray-400" />
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Pipeline Summary */}
      <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-600">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {Object.values(pipelineStats).reduce((sum, stage) => sum + stage.queue_size, 0)}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Total Queued</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {Object.values(pipelineStats).reduce((sum, stage) => sum + stage.active_tasks, 0)}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Total Active</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {Math.round(Object.values(pipelineStats).reduce((sum, stage) => sum + stage.success_rate, 0) / Object.keys(pipelineStats).length * 100)}%
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Avg Success Rate</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {Object.values(pipelineStats).reduce((sum, stage) => sum + stage.throughput_per_minute, 0)}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">Total Throughput/min</div>
          </div>
        </div>
      </div>
    </div>
  )
}