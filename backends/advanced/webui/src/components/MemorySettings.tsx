import { useState, useEffect } from 'react'
import { Brain, RefreshCw, CheckCircle, Trash2, Save, RotateCcw, AlertCircle } from 'lucide-react'
import { systemApi, memoriesApi } from '../services/api'

interface MemorySettingsProps {
  className?: string
}

export default function MemorySettings({ className }: MemorySettingsProps) {
  const [configYaml, setConfigYaml] = useState('')
  const [loading, setLoading] = useState(false)
  const [validating, setValidating] = useState(false)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    loadMemoryConfig()
  }, [])

  const loadMemoryConfig = async () => {
    setLoading(true)
    setError('')
    setMessage('')
    
    try {
      const response = await systemApi.getMemoryConfigRaw()
      setConfigYaml(response.data.config_yaml)
      setMessage('Configuration loaded successfully')
      setTimeout(() => setMessage(''), 3000)
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to load memory configuration')
    } finally {
      setLoading(false)
    }
  }

  const validateConfig = async () => {
    if (!configYaml.trim()) {
      setError('Configuration cannot be empty')
      return
    }

    setValidating(true)
    setError('')
    setMessage('')
    
    try {
      const response = await systemApi.validateMemoryConfig(configYaml)
      setMessage('✅ Configuration is valid')
      setTimeout(() => setMessage(''), 3000)
    } catch (err: any) {
      setError(err.response?.data?.error || 'Validation failed')
    } finally {
      setValidating(false)
    }
  }

  const saveConfig = async () => {
    if (!configYaml.trim()) {
      setError('Configuration cannot be empty')
      return
    }

    setSaving(true)
    setError('')
    setMessage('')
    
    try {
      const response = await systemApi.updateMemoryConfigRaw(configYaml)
      setMessage('✅ Configuration saved and reloaded successfully')
      setTimeout(() => setMessage(''), 5000)
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to save configuration')
    } finally {
      setSaving(false)
    }
  }

  const reloadConfig = async () => {
    await loadMemoryConfig()
  }

  const resetConfig = () => {
    loadMemoryConfig()
    setMessage('Configuration reset to file version')
    setTimeout(() => setMessage(''), 3000)
  }

  const deleteAllMemories = async () => {
    const confirmed = window.confirm(
      '⚠️ WARNING: This will permanently delete ALL your memories. This action cannot be undone.\n\nAre you sure you want to continue?'
    )
    
    if (!confirmed) return

    const doubleConfirmed = window.confirm(
      '🚨 FINAL CONFIRMATION: You are about to delete ALL memories permanently.\n\nType "DELETE" in the next dialog to confirm.'
    )
    
    if (!doubleConfirmed) return

    const userInput = window.prompt('Type "DELETE" to confirm memory deletion:')
    if (userInput !== 'DELETE') {
      setMessage('Deletion cancelled - confirmation text did not match')
      return
    }

    setDeleting(true)
    setError('')
    setMessage('')
    
    try {
      const response = await memoriesApi.deleteAll()
      setMessage(`✅ Successfully deleted ${response.data.deleted_count || 'all'} memories`)
      setTimeout(() => setMessage(''), 5000)
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to delete memories')
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className={className}>
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Brain className="h-5 w-5 text-blue-600" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Memory Configuration
            </h3>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={reloadConfig}
              disabled={loading}
              className="flex items-center space-x-1 px-3 py-1 text-sm bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors disabled:opacity-50"
              title="Reload configuration from file"
            >
              <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
              <span>Reload</span>
            </button>
            <button
              onClick={deleteAllMemories}
              disabled={deleting}
              className="flex items-center space-x-1 px-3 py-1 text-sm bg-red-600 text-white rounded hover:bg-red-700 transition-colors disabled:opacity-50"
              title="Delete all memories (cannot be undone)"
            >
              <Trash2 className="h-3 w-3" />
              <span>{deleting ? 'Deleting...' : 'Delete All Memories'}</span>
            </button>
          </div>
        </div>

        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
          Edit the memory extraction configuration. This controls how memories are extracted from conversations, quality
          control settings, and processing parameters.
        </p>

        {/* Status Messages */}
        {message && (
          <div className="mb-4 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md">
            <p className="text-sm text-green-700 dark:text-green-300">{message}</p>
          </div>
        )}

        {error && (
          <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <div className="flex">
              <AlertCircle className="h-5 w-5 text-red-400 mr-2 flex-shrink-0" />
              <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
            </div>
          </div>
        )}

        {/* YAML Editor */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Configuration (YAML format)
            </label>
            <textarea
              value={configYaml}
              onChange={(e) => setConfigYaml(e.target.value)}
              placeholder="Loading configuration..."
              disabled={loading}
              className="w-full h-96 p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              style={{
                fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
              }}
            />
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-600">
            <div className="flex items-center space-x-2">
              <button
                onClick={validateConfig}
                disabled={validating || !configYaml.trim()}
                className="flex items-center space-x-1 px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <CheckCircle className={`h-4 w-4 ${validating ? 'animate-pulse' : ''}`} />
                <span>{validating ? 'Validating...' : 'Validate'}</span>
              </button>

              <button
                onClick={resetConfig}
                disabled={loading}
                className="flex items-center space-x-1 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <RotateCcw className="h-4 w-4" />
                <span>Reset</span>
              </button>
            </div>

            <button
              onClick={saveConfig}
              disabled={saving || !configYaml.trim()}
              className="flex items-center space-x-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Save className={`h-4 w-4 ${saving ? 'animate-pulse' : ''}`} />
              <span>{saving ? 'Saving...' : 'Save Configuration'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}