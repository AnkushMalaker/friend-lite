import { Users, Activity, AlertTriangle, CheckCircle, Clock } from 'lucide-react'

interface SystemHealthData {
  total_active_clients: number
  total_processing_tasks: number
  task_manager_healthy: boolean
  error_rate: number
  uptime_hours: number
}

interface SystemHealthCardsProps {
  data: SystemHealthData
}

export default function SystemHealthCards({ data }: SystemHealthCardsProps) {
  const cards = [
    {
      title: 'Active Clients',
      value: data.total_active_clients,
      icon: Users,
      color: 'blue',
      description: 'Currently connected clients'
    },
    {
      title: 'Processing Tasks',
      value: data.total_processing_tasks,
      icon: Activity,
      color: 'green',
      description: 'Tasks in processing queues'
    },
    {
      title: 'Error Rate',
      value: `${(data.error_rate * 100).toFixed(1)}%`,
      icon: data.error_rate > 0.1 ? AlertTriangle : CheckCircle,
      color: data.error_rate > 0.1 ? 'red' : 'green',
      description: 'Recent processing error rate'
    },
    {
      title: 'Uptime',
      value: `${Math.floor(data.uptime_hours)}h`,
      icon: Clock,
      color: 'purple',
      description: 'System uptime'
    }
  ]

  const getCardColors = (color: string) => {
    const colors = {
      blue: {
        bg: 'bg-blue-50 dark:bg-blue-900/20',
        border: 'border-blue-200 dark:border-blue-800',
        icon: 'text-blue-600 bg-blue-100 dark:bg-blue-900/40 dark:text-blue-400',
        text: 'text-blue-900 dark:text-blue-100'
      },
      green: {
        bg: 'bg-green-50 dark:bg-green-900/20',
        border: 'border-green-200 dark:border-green-800',
        icon: 'text-green-600 bg-green-100 dark:bg-green-900/40 dark:text-green-400',
        text: 'text-green-900 dark:text-green-100'
      },
      red: {
        bg: 'bg-red-50 dark:bg-red-900/20',
        border: 'border-red-200 dark:border-red-800',
        icon: 'text-red-600 bg-red-100 dark:bg-red-900/40 dark:text-red-400',
        text: 'text-red-900 dark:text-red-100'
      },
      purple: {
        bg: 'bg-purple-50 dark:bg-purple-900/20',
        border: 'border-purple-200 dark:border-purple-800',
        icon: 'text-purple-600 bg-purple-100 dark:bg-purple-900/40 dark:text-purple-400',
        text: 'text-purple-900 dark:text-purple-100'
      }
    }
    return colors[color as keyof typeof colors] || colors.blue
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card) => {
        const Icon = card.icon
        const colors = getCardColors(card.color)

        return (
          <div
            key={card.title}
            className={`p-6 rounded-lg border ${colors.bg} ${colors.border}`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  {card.title}
                </p>
                <p className={`text-2xl font-bold ${colors.text}`}>
                  {card.value}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {card.description}
                </p>
              </div>
              <div className={`p-3 rounded-full ${colors.icon}`}>
                <Icon className="h-6 w-6" />
              </div>
            </div>

            {/* Health Indicator for Task Manager */}
            {card.title === 'Processing Tasks' && (
              <div className="mt-4 flex items-center">
                <div className={`w-2 h-2 rounded-full mr-2 ${
                  data.task_manager_healthy ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="text-xs text-gray-600 dark:text-gray-400">
                  Task Manager: {data.task_manager_healthy ? 'Healthy' : 'Unhealthy'}
                </span>
              </div>
            )}

            {/* Error Rate Trend */}
            {card.title === 'Error Rate' && (
              <div className="mt-4">
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1">
                  <div
                    className={`h-1 rounded-full ${
                      data.error_rate > 0.1 ? 'bg-red-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${Math.min(data.error_rate * 100, 100)}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}