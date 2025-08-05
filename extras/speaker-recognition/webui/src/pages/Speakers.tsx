import { useState, useEffect, useCallback } from 'react'
import { Search, Download, Trash2, Eye, BarChart3, User, Clock, CheckCircle, XCircle } from 'lucide-react'
import { useUser } from '../contexts/UserContext'
import { apiService } from '../services/api'
import { formatDuration } from '../utils/audioUtils'

interface Speaker {
  id: string
  name: string
  user_id?: string
  created_at?: string
  updated_at?: string
  audio_sample_count?: number
  total_audio_duration?: number
  average_quality?: number
  enrollment_status: 'pending' | 'completed' | 'failed'
  last_enrollment?: string
}

interface SpeakerStats {
  total_speakers: number
  total_audio_samples: number
  total_duration: number
  average_quality: number
  speakers_by_status: {
    pending: number
    completed: number
    failed: number
  }
}

export default function Speakers() {
  const { user } = useUser()
  const [speakers, setSpeakers] = useState<Speaker[]>([])
  const [stats, setStats] = useState<SpeakerStats | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'completed' | 'failed'>('all')
  const [sortBy, setSortBy] = useState<'name' | 'created_at' | 'audio_sample_count' | 'total_audio_duration' | 'average_quality'>('name')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc')
  const [selectedSpeaker, setSelectedSpeaker] = useState<Speaker | null>(null)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null)

  const loadSpeakers = useCallback(async () => {
    if (!user) return

    setIsLoading(true)
    try {
      const speakersResponse = await apiService.get('/speakers')
      
      // Backend returns { speakers: [...] }, extract the speakers array
      const allSpeakers = speakersResponse.data.speakers || []
      
      // Filter speakers by user - treat users as "folders" by using speaker ID prefixes
      const userPrefix = `user_${user.id}_`
      const userSpeakers = allSpeakers.filter((speaker: any) => {
        // Filter speakers that belong to this user based on ID prefix
        return speaker.id?.startsWith(userPrefix)
      }).map((speaker: any) => ({
        // Map backend speaker data to frontend format
        id: speaker.id,
        name: speaker.name,
        user_id: user.id.toString(),
        created_at: speaker.created_at || null,
        updated_at: speaker.updated_at || null,
        audio_sample_count: speaker.audio_sample_count || 0,
        total_audio_duration: speaker.total_audio_duration || 0,
        average_quality: null, // Not provided by backend yet
        enrollment_status: 'completed' as const, // If speaker exists, it's completed
        last_enrollment: null // Not provided by backend yet
      }))
      
      // Calculate stats from filtered speakers
      const stats = {
        total_speakers: userSpeakers.length,
        total_audio_samples: userSpeakers.reduce((sum, s) => sum + (s.audio_sample_count || 0), 0),
        total_duration: userSpeakers.reduce((sum, s) => sum + (s.total_audio_duration || 0), 0),
        average_quality: 0, // Not available from backend yet
        speakers_by_status: {
          pending: 0,
          completed: userSpeakers.length,
          failed: 0
        }
      }

      setSpeakers(userSpeakers)
      setStats(stats)
    } catch (error) {
      console.error('Failed to load speakers:', error)
      // Set empty state when API fails
      setSpeakers([])
      setStats({
        total_speakers: 0,
        total_audio_samples: 0,
        total_duration: 0,
        average_quality: 0,
        speakers_by_status: {
          pending: 0,
          completed: 0,
          failed: 0
        }
      })
    } finally {
      setIsLoading(false)
    }
  }, [user])

  useEffect(() => {
    loadSpeakers()
  }, [loadSpeakers])

  const filteredAndSortedSpeakers = useCallback(() => {
    let filtered = speakers.filter(speaker => {
      const matchesSearch = speaker.name.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesStatus = statusFilter === 'all' || speaker.enrollment_status === statusFilter
      return matchesSearch && matchesStatus
    })

    filtered.sort((a, b) => {
      let aValue = a[sortBy]
      let bValue = b[sortBy]

      if (typeof aValue === 'string') {
        aValue = aValue.toLowerCase()
        bValue = (bValue as string).toLowerCase()
      }

      if (sortOrder === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0
      }
    })

    return filtered
  }, [speakers, searchTerm, statusFilter, sortBy, sortOrder])

  const deleteSpeaker = useCallback(async (speakerId: string) => {
    if (!user) return
    
    // Ensure user can only delete their own speakers
    const userPrefix = `user_${user.id}_`
    if (!speakerId.startsWith(userPrefix)) {
      alert('You can only delete your own speakers.')
      return
    }
    
    try {
      await apiService.delete(`/speakers/${speakerId}`)
      setSpeakers(prev => prev.filter(s => s.id !== speakerId))
      setShowDeleteConfirm(null)
      loadSpeakers() // Reload to update stats
    } catch (error) {
      console.error('Failed to delete speaker:', error)
      alert('Failed to delete speaker. Please try again.')
    }
  }, [loadSpeakers, user])

  const exportSpeakerData = useCallback(async (speaker: Speaker) => {
    try {
      const response = await apiService.get(`/speakers/${speaker.id}/export`, {
        responseType: 'blob'
      })

      const blob = new Blob([response.data], { type: 'application/zip' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `speaker_${speaker.name.replace(/\s+/g, '_')}_data.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to export speaker data:', error)
      alert('Failed to export speaker data. Please try again.')
    }
  }, [])

  const getQualityColor = useCallback((quality: number) => {
    if (quality >= 25) return 'text-green-600 bg-green-100'
    if (quality >= 20) return 'text-blue-600 bg-blue-100'
    if (quality >= 15) return 'text-yellow-600 bg-yellow-100'
    return 'text-red-600 bg-red-100'
  }, [])

  const getQualityLabel = useCallback((quality: number) => {
    if (quality >= 30) return 'Excellent'
    if (quality >= 20) return 'Good'
    if (quality >= 15) return 'Fair'
    return 'Poor'
  }, [])

  const getStatusColor = useCallback((status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100'
      case 'pending': return 'text-yellow-600 bg-yellow-100'
      case 'failed': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }, [])

  const getStatusIcon = useCallback((status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle className="h-4 w-4" />
      case 'failed': return <XCircle className="h-4 w-4" />
      default: return <Clock className="h-4 w-4" />
    }
  }, [])

  if (!user) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Please select a user to continue.</p>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="text-center py-12">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <p className="mt-2 text-gray-600">Loading speakers...</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">👥 Speaker Management</h1>
        <div className="flex space-x-2">
          <button
            onClick={loadSpeakers}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Refresh
          </button>
        </div>
      </div>

      <p className="text-gray-600">
        Manage enrolled speakers and view their quality metrics.
      </p>

      {/* Statistics Cards */}
      {stats && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white border rounded-lg p-4">
            <div className="text-sm font-medium text-gray-500">Total Speakers</div>
            <div className="text-2xl font-bold text-gray-900">{stats.total_speakers}</div>
          </div>
          <div className="bg-white border rounded-lg p-4">
            <div className="text-sm font-medium text-gray-500">Audio Samples</div>
            <div className="text-2xl font-bold text-gray-900">{stats.total_audio_samples}</div>
          </div>
          <div className="bg-white border rounded-lg p-4">
            <div className="text-sm font-medium text-gray-500">Total Duration</div>
            <div className="text-2xl font-bold text-gray-900">{formatDuration(stats.total_duration)}</div>
          </div>
          <div className="bg-white border rounded-lg p-4">
            <div className="text-sm font-medium text-gray-500">Avg Quality</div>
            <div className="text-2xl font-bold text-gray-900">{stats.average_quality.toFixed(1)} dB</div>
          </div>
        </div>
      )}

      {/* Filters and Search */}
      <div className="bg-white border rounded-lg p-4">
        <div className="grid md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Search</label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search speakers..."
                className="pl-10 w-full px-3 py-2 border border-gray-300 rounded-md"
              />
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as any)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option value="all">All Statuses</option>
              <option value="completed">Completed</option>
              <option value="pending">Pending</option>
              <option value="failed">Failed</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Sort By</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option value="name">Name</option>
              <option value="created_at">Created Date</option>
              <option value="audio_sample_count">Audio Count</option>
              <option value="total_audio_duration">Duration</option>
              <option value="average_quality">Quality</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Order</label>
            <select
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value as any)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option value="asc">Ascending</option>
              <option value="desc">Descending</option>
            </select>
          </div>
        </div>
      </div>

      {/* Speakers List */}
      {filteredAndSortedSpeakers().length > 0 ? (
        <div className="bg-white border rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Speaker
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Audio Count
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Duration
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Quality
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Created
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredAndSortedSpeakers().map((speaker) => (
                  <tr key={speaker.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="flex-shrink-0 h-10 w-10">
                          <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center">
                            <User className="h-5 w-5 text-blue-600" />
                          </div>
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900">{speaker.name}</div>
                          <div className="text-sm text-gray-500">ID: {speaker.id}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(speaker.enrollment_status)}`}>
                        {getStatusIcon(speaker.enrollment_status)}
                        <span>{speaker.enrollment_status}</span>
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {speaker.audio_sample_count || 0} samples
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {speaker.total_audio_duration ? formatDuration(speaker.total_audio_duration) : 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {speaker.average_quality ? (
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getQualityColor(speaker.average_quality)}`}>
                          {speaker.average_quality.toFixed(1)} dB ({getQualityLabel(speaker.average_quality)})
                        </span>
                      ) : (
                        <span className="text-gray-500 text-xs">N/A</span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {speaker.created_at ? new Date(speaker.created_at).toLocaleDateString() : 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex justify-end space-x-2">
                        <button
                          onClick={() => setSelectedSpeaker(speaker)}
                          className="p-1 text-blue-600 hover:text-blue-800"
                          title="View Details"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => exportSpeakerData(speaker)}
                          className="p-1 text-green-600 hover:text-green-800"
                          title="Export Data"
                        >
                          <Download className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => setShowDeleteConfirm(speaker.id)}
                          className="p-1 text-red-600 hover:text-red-800"
                          title="Delete Speaker"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="text-center py-12">
          <User className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Speakers Found</h3>
          <p className="text-gray-500">
            {speakers.length === 0 
              ? "No speakers have been enrolled yet." 
              : "No speakers match your current filters."}
          </p>
        </div>
      )}

      {/* Speaker Details Modal */}
      {selectedSpeaker && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">Speaker Details</h2>
                <button
                  onClick={() => setSelectedSpeaker(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ×
                </button>
              </div>
            </div>
            <div className="p-6 space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-500">Name</label>
                  <p className="text-gray-900">{selectedSpeaker.name}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500">Status</label>
                  <span className={`inline-flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(selectedSpeaker.enrollment_status)}`}>
                    {getStatusIcon(selectedSpeaker.enrollment_status)}
                    <span>{selectedSpeaker.enrollment_status}</span>
                  </span>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500">Audio Samples</label>
                  <p className="text-gray-900">{selectedSpeaker.audio_sample_count || 0}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500">Total Duration</label>
                  <p className="text-gray-900">{formatDuration(selectedSpeaker.total_audio_duration || 0)}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500">Average Quality</label>
                  <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getQualityColor(selectedSpeaker.average_quality)}`}>
                    {selectedSpeaker.average_quality.toFixed(1)} dB ({getQualityLabel(selectedSpeaker.average_quality)})
                  </span>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500">Created</label>
                  <p className="text-gray-900">{new Date(selectedSpeaker.created_at).toLocaleString()}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500">Last Updated</label>
                  <p className="text-gray-900">{new Date(selectedSpeaker.updated_at).toLocaleString()}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-500">Last Enrollment</label>
                  <p className="text-gray-900">{new Date(selectedSpeaker.last_enrollment).toLocaleString()}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-md w-full">
            <div className="p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Delete Speaker</h3>
              <p className="text-gray-600 mb-6">
                Are you sure you want to delete this speaker? This action cannot be undone and will remove all associated audio data.
              </p>
              <div className="flex justify-end space-x-3">
                <button
                  onClick={() => setShowDeleteConfirm(null)}
                  className="px-4 py-2 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={() => deleteSpeaker(showDeleteConfirm)}
                  className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}