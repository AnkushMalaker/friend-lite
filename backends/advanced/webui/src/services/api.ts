import axios from 'axios'

// Get backend URL from environment or auto-detect based on current location
const getBackendUrl = () => {
  // If explicitly set in environment, use that
  if (import.meta.env.VITE_BACKEND_URL !== undefined && import.meta.env.VITE_BACKEND_URL !== '') {
    return import.meta.env.VITE_BACKEND_URL
  }
  
  // If accessed through proxy (standard ports), use relative URLs
  const { protocol, hostname, port } = window.location
  const isStandardPort = (protocol === 'https:' && (port === '' || port === '443')) || 
                         (protocol === 'http:' && (port === '' || port === '80'))
  
  if (isStandardPort) {
    // We're being accessed through nginx proxy or Kubernetes Ingress, use same origin
    return ''  // Empty string means use relative URLs (same origin)
  }
  
  // Development mode - direct access to dev server
  if (port === '5173') {
    return 'http://localhost:8000'
  }
  
  // Fallback
  return `${protocol}//${hostname}:8000`
}

const BACKEND_URL = getBackendUrl()
console.log('🌐 API: Backend URL configured as:', BACKEND_URL || 'Same origin (relative URLs)')

// Export BACKEND_URL for use in other components
export { BACKEND_URL }

export const api = axios.create({
  baseURL: BACKEND_URL,
  timeout: 60000,  // Increased to 60 seconds for heavy processing scenarios
})

// Add request interceptor to include auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Add response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Only clear token and redirect on actual 401 responses, not on timeouts
    if (error.response?.status === 401) {
      // Token expired or invalid, redirect to login
      console.warn('🔐 API: 401 Unauthorized - clearing token and redirecting to login')
      localStorage.removeItem('token')
      window.location.href = '/login'
    } else if (error.code === 'ECONNABORTED') {
      // Request timeout - don't logout, just log it
      console.warn('⏱️ API: Request timeout - server may be busy')
    } else if (!error.response) {
      // Network error - don't logout
      console.warn('🌐 API: Network error - server may be unreachable')
    }
    return Promise.reject(error)
  }
)

// API endpoints
export const authApi = {
  login: (email: string, password: string) => {
    const formData = new FormData()
    formData.append('username', email)
    formData.append('password', password)
    return api.post('/auth/jwt/login', formData)
  },
  getMe: () => api.get('/users/me'),
}

export const conversationsApi = {
  getAll: () => api.get('/api/conversations'),
  getById: (id: string) => api.get(`/api/conversations/${id}`),
  delete: (id: string) => api.delete(`/api/conversations/${id}`),

  // Reprocessing endpoints
  reprocessTranscript: (conversationId: string) => api.post(`/api/conversations/${conversationId}/reprocess-transcript`),
  reprocessMemory: (conversationId: string, transcriptVersionId: string = 'active') => api.post(`/api/conversations/${conversationId}/reprocess-memory`, null, {
    params: { transcript_version_id: transcriptVersionId }
  }),

  // Version management
  activateTranscriptVersion: (conversationId: string, versionId: string) => api.post(`/api/conversations/${conversationId}/activate-transcript/${versionId}`),
  activateMemoryVersion: (conversationId: string, versionId: string) => api.post(`/api/conversations/${conversationId}/activate-memory/${versionId}`),
  getVersionHistory: (conversationId: string) => api.get(`/api/conversations/${conversationId}/versions`),
}

export const memoriesApi = {
  getAll: (userId?: string) => api.get('/api/memories', { params: userId ? { user_id: userId } : {} }),
  getUnfiltered: (userId?: string) => api.get('/api/memories/unfiltered', { params: userId ? { user_id: userId } : {} }),
  search: (query: string, userId?: string, limit: number = 20, scoreThreshold?: number) => 
    api.get('/api/memories/search', { 
      params: { 
        query, 
        ...(userId && { user_id: userId }), 
        limit,
        ...(scoreThreshold !== undefined && { score_threshold: scoreThreshold / 100 }) // Convert percentage to decimal
      } 
    }),
  delete: (id: string) => api.delete(`/api/memories/${id}`),
  deleteAll: () => api.delete('/api/admin/memory/delete-all'),
}

export const usersApi = {
  getAll: () => api.get('/api/users'),
  create: (userData: any) => api.post('/api/users', userData),
  update: (id: string, userData: any) => api.put(`/api/users/${id}`, userData),
  delete: (id: string) => api.delete(`/api/users/${id}`),
}

export const systemApi = {
  getHealth: () => api.get('/health'),
  getReadiness: () => api.get('/readiness'),
  getMetrics: () => api.get('/api/metrics'),
  getProcessorStatus: () => api.get('/api/processor/status'),
  getProcessorTasks: () => api.get('/api/processor/tasks'),
  getActiveClients: () => api.get('/api/clients/active'),
  getDiarizationSettings: () => api.get('/api/diarization-settings'),
  saveDiarizationSettings: (settings: any) => api.post('/api/diarization-settings', settings),
  
  // Memory Configuration Management
  getMemoryConfigRaw: () => api.get('/api/admin/memory/config/raw'),
  updateMemoryConfigRaw: (configYaml: string) => 
    api.post('/api/admin/memory/config/raw', configYaml, {
      headers: { 'Content-Type': 'text/plain' }
    }),
  validateMemoryConfig: (configYaml: string) => 
    api.post('/api/admin/memory/config/validate', configYaml, {
      headers: { 'Content-Type': 'text/plain' }
    }),
  reloadMemoryConfig: () => api.post('/api/admin/memory/config/reload'),
}

export const queueApi = {
  getJobs: (params: URLSearchParams) => api.get(`/api/queue/jobs?${params}`),
  getJob: (jobId: string) => api.get(`/api/queue/jobs/${jobId}`),
  getJobsBySession: (sessionId: string) => api.get(`/api/queue/jobs/by-session/${sessionId}`),
  getStats: () => api.get('/api/queue/stats'),
  getStreamingStatus: () => api.get('/api/streaming/status'),
  cleanupStuckWorkers: () => api.post('/api/streaming/cleanup'),
  cleanupOldSessions: (maxAgeSeconds: number = 3600) => api.post(`/api/streaming/cleanup-sessions?max_age_seconds=${maxAgeSeconds}`),
  retryJob: (jobId: string, force: boolean = false) =>
    api.post(`/api/queue/jobs/${jobId}/retry`, { force }),
  cancelJob: (jobId: string) => api.delete(`/api/queue/jobs/${jobId}`),
}

export const uploadApi = {
  uploadAudioFiles: (files: FormData, onProgress?: (progress: number) => void) => 
    api.post('/api/audio/upload', files, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 300000, // 5 minutes
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          onProgress(progress)
        }
      }
    }),
}

export const chatApi = {
  // Session management
  createSession: (title?: string) => api.post('/api/chat/sessions', { title }),
  getSessions: (limit = 50) => api.get('/api/chat/sessions', { params: { limit } }),
  getSession: (sessionId: string) => api.get(`/api/chat/sessions/${sessionId}`),
  updateSession: (sessionId: string, title: string) => api.put(`/api/chat/sessions/${sessionId}`, { title }),
  deleteSession: (sessionId: string) => api.delete(`/api/chat/sessions/${sessionId}`),
  
  // Messages
  getMessages: (sessionId: string, limit = 100) => api.get(`/api/chat/sessions/${sessionId}/messages`, { params: { limit } }),
  
  // Memory extraction
  extractMemories: (sessionId: string) => api.post(`/api/chat/sessions/${sessionId}/extract-memories`),
  
  // Statistics
  getStatistics: () => api.get('/api/chat/statistics'),
  
  // Health check
  getHealth: () => api.get('/api/chat/health'),
  
  // Streaming chat (returns EventSource for Server-Sent Events)
  sendMessage: (message: string, sessionId?: string) => {
    const requestBody: any = { message }
    if (sessionId) {
      requestBody.session_id = sessionId
    }
    
    return fetch(`${BACKEND_URL}/api/chat/send`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify(requestBody)
    })
  }
}

export const speakerApi = {
  // Get current user's speaker configuration
  getSpeakerConfiguration: () => api.get('/api/speaker-configuration'),
  
  // Update current user's speaker configuration
  updateSpeakerConfiguration: (primarySpeakers: Array<{speaker_id: string, name: string, user_id: number}>) => 
    api.post('/api/speaker-configuration', primarySpeakers),
    
  // Get enrolled speakers from speaker recognition service  
  getEnrolledSpeakers: () => api.get('/api/enrolled-speakers'),
  
  // Check speaker service status (admin only)
  getSpeakerServiceStatus: () => api.get('/api/speaker-service-status'),
}