import React, { useState, useEffect } from 'react'
import { Brain, Search, RefreshCw, Trash2, Calendar, User, Tag } from 'lucide-react'
import { memoriesApi } from '../services/api'

interface Memory {
  id: string
  text: string
  category: string
  created_at: string
  updated_at: string
  user_id: string
  score?: number
  metadata?: any
}

export default function Memories() {
  const [memories, setMemories] = useState<Memory[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [userId, setUserId] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [showUnfiltered, setShowUnfiltered] = useState(false)

  const loadMemories = async () => {
    if (!userId.trim()) return

    try {
      setLoading(true)
      const response = showUnfiltered 
        ? await memoriesApi.getUnfiltered(userId)
        : await memoriesApi.getAll(userId)
      setMemories(response.data)
      setError(null)
    } catch (err: any) {
      setError(err.message || 'Failed to load memories')
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteMemory = async (memoryId: string) => {
    if (!confirm('Are you sure you want to delete this memory?')) return

    try {
      await memoriesApi.delete(memoryId)
      setMemories(memories.filter(m => m.id !== memoryId))
    } catch (err: any) {
      setError(err.message || 'Failed to delete memory')
    }
  }

  const filteredMemories = memories.filter(memory =>
    memory.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
    memory.category?.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  const getCategoryColor = (category: string) => {
    const colors = {
      'personal': 'bg-blue-100 text-blue-800',
      'work': 'bg-green-100 text-green-800', 
      'health': 'bg-red-100 text-red-800',
      'entertainment': 'bg-purple-100 text-purple-800',
      'education': 'bg-yellow-100 text-yellow-800',
      'default': 'bg-gray-100 text-gray-800'
    }
    return colors[category as keyof typeof colors] || colors.default
  }

  return (
    <div>
      {/* Header */}
      <div className="flex items-center space-x-2 mb-6">
        <Brain className="h-6 w-6 text-blue-600" />
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Memory Management
        </h1>
      </div>

      {/* Controls */}
      <div className="space-y-4 mb-6">
        <div className="flex flex-col sm:flex-row gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Enter username to view memories:
            </label>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="e.g., john_doe, alice123"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="flex items-end space-x-2">
            <label className="flex items-center space-x-2 text-sm">
              <input
                type="checkbox"
                checked={showUnfiltered}
                onChange={(e) => setShowUnfiltered(e.target.checked)}
                className="rounded border-gray-300"
              />
              <span className="text-gray-700 dark:text-gray-300">Show unfiltered</span>
            </label>
            <button
              onClick={loadMemories}
              disabled={!userId.trim() || loading}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              <span>Load Data</span>
            </button>
          </div>
        </div>

        {/* Search */}
        {memories.length > 0 && (
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search memories..."
              className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        )}
      </div>

      {/* Status Messages */}
      {userId.trim() && (
        <div className="mb-4">
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-4">
            <p className="text-sm text-blue-700 dark:text-blue-300">
              Showing {showUnfiltered ? 'unfiltered' : 'filtered'} data for user: <strong>{userId}</strong>
            </p>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4 mb-6">
          <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600 dark:text-gray-400">Loading memories...</span>
        </div>
      )}

      {/* Memories List */}
      {!loading && filteredMemories.length > 0 && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Found {filteredMemories.length} memories
              {searchQuery && ` matching "${searchQuery}"`}
            </p>
          </div>

          <div className="space-y-4">
            {filteredMemories.map((memory) => (
              <div
                key={memory.id}
                className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6 border border-gray-200 dark:border-gray-600"
              >
                {/* Memory Header */}
                <div className="flex justify-between items-start mb-4">
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
                      <Calendar className="h-4 w-4" />
                      <span>{formatDate(memory.created_at)}</span>
                    </div>
                    {memory.category && (
                      <div className="flex items-center space-x-2">
                        <Tag className="h-4 w-4 text-gray-400" />
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getCategoryColor(memory.category)}`}>
                          {memory.category}
                        </span>
                      </div>
                    )}
                    {memory.score && (
                      <span className="text-xs text-gray-500">
                        Score: {memory.score.toFixed(3)}
                      </span>
                    )}
                  </div>
                  
                  <button
                    onClick={() => handleDeleteMemory(memory.id)}
                    className="p-2 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-md transition-colors"
                    title="Delete memory"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>

                {/* Memory Content */}
                <div className="prose prose-sm max-w-none">
                  <p className="text-gray-900 dark:text-gray-100 leading-relaxed">
                    {memory.text}
                  </p>
                </div>

                {/* Metadata */}
                {memory.metadata && (
                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
                    <details className="text-sm">
                      <summary className="cursor-pointer text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100">
                        View metadata
                      </summary>
                      <pre className="mt-2 p-2 bg-gray-100 dark:bg-gray-800 rounded text-xs overflow-x-auto">
                        {JSON.stringify(memory.metadata, null, 2)}
                      </pre>
                    </details>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty States */}
      {!loading && !userId.trim() && (
        <div className="text-center text-gray-500 dark:text-gray-400 py-12">
          <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Enter a username above to view their memories</p>
        </div>
      )}

      {!loading && userId.trim() && filteredMemories.length === 0 && !error && (
        <div className="text-center text-gray-500 dark:text-gray-400 py-12">
          <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>
            {searchQuery 
              ? `No memories found matching "${searchQuery}"`
              : `No memories found for user "${userId}"`
            }
          </p>
        </div>
      )}
    </div>
  )
}