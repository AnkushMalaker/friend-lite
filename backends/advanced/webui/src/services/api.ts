import axios from 'axios'

// Get backend URL from environment or default to localhost
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
console.log('🌐 API: Backend URL configured as:', BACKEND_URL)

// Export BACKEND_URL for use in other components
export { BACKEND_URL }

export const api = axios.create({
  baseURL: BACKEND_URL,
  timeout: 30000,
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
    if (error.response?.status === 401) {
      // Token expired or invalid, redirect to login
      localStorage.removeItem('token')
      window.location.href = '/login'
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
}

export const memoriesApi = {
  getAll: (userId?: string) => api.get('/api/memories', { params: userId ? { user_id: userId } : {} }),
  getUnfiltered: (userId?: string) => api.get('/api/memories/unfiltered', { params: userId ? { user_id: userId } : {} }),
  delete: (id: string) => api.delete(`/api/memories/${id}`),
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
}

export const uploadApi = {
  uploadAudioFiles: (files: FormData) => api.post('/api/process-audio-files', files, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000, // 5 minutes
  }),
}