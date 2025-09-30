import React, { useState, useEffect } from 'react';
import {
  Clock,
  Play,
  CheckCircle,
  XCircle,
  RotateCcw,
  StopCircle,
  Eye,
  Filter,
  X,
  RefreshCw,
  Layers,
  Trash2,
  AlertTriangle
} from 'lucide-react';
import { queueApi } from '../services/api';

interface QueueJob {
  job_id: string;
  job_type: string;
  user_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled' | 'retrying';
  priority: 'low' | 'normal' | 'high';
  data: any;
  result?: any;
  error_message?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  retry_count: number;
  max_retries: number;
  progress_percent: number;
  progress_message: string;
}

interface QueueStats {
  total_jobs: number;
  queued_jobs: number;
  processing_jobs: number;
  completed_jobs: number;
  failed_jobs: number;
  cancelled_jobs: number;
  retrying_jobs: number;
  timestamp: string;
}

interface Filters {
  status: string;
  job_type: string;
  priority: string;
}

const Queue: React.FC = () => {
  const [jobs, setJobs] = useState<QueueJob[]>([]);
  const [stats, setStats] = useState<QueueStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedJob, setSelectedJob] = useState<QueueJob | null>(null);
  const [filters, setFilters] = useState<Filters>({
    status: '',
    job_type: '',
    priority: ''
  });
  const [pagination, setPagination] = useState({
    offset: 0,
    limit: 20,
    total: 0,
    has_more: false
  });
  const [refreshing, setRefreshing] = useState(false);
  const [showFlushModal, setShowFlushModal] = useState(false);
  const [flushSettings, setFlushSettings] = useState({
    older_than_hours: 24,
    statuses: ['completed', 'failed'],
    flush_all: false
  });
  const [flushing, setFlushing] = useState(false);

  // Auto-refresh interval
  useEffect(() => {
    console.log('🔄 Setting up queue auto-refresh interval');
    const interval = setInterval(() => {
      if (!loading) {
        console.log('⏰ Auto-refreshing queue data');
        fetchData();
      }
    }, 5000); // Refresh every 5 seconds

    return () => {
      console.log('🧹 Clearing queue auto-refresh interval');
      clearInterval(interval);
    };
  }, []); // Remove dependencies to prevent interval recreation

  // Initial data fetch
  useEffect(() => {
    fetchData();
  }, [filters, pagination.offset]);

  const fetchData = async () => {
    console.log('📥 fetchData called, refreshing:', refreshing, 'loading:', loading);
    if (!refreshing) setRefreshing(true);
    
    try {
      console.log('🔄 Starting Promise.all for jobs and stats');
      await Promise.all([fetchJobs(), fetchStats()]);
      console.log('✅ Promise.all completed successfully');
    } catch (error) {
      console.error('❌ Error fetching queue data:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
      console.log('🏁 fetchData completed');
    }
  };

  const fetchJobs = async () => {
    try {
      console.log('🔍 fetchJobs starting...');
      const params = new URLSearchParams({
        limit: pagination.limit.toString(),
        offset: pagination.offset.toString(),
        sort: 'created_at',
        order: 'desc'
      });

      if (filters.status) params.append('status', filters.status);
      if (filters.job_type) params.append('job_type', filters.job_type);
      if (filters.priority) params.append('priority', filters.priority);

      console.log('📡 Fetching jobs with params:', params.toString());
      const response = await queueApi.getJobs(params);
      const data = response.data;
      console.log('✅ fetchJobs success, got', data.jobs?.length, 'jobs');
      setJobs(data.jobs);
      setPagination(prev => ({
        ...prev,
        total: data.pagination.total,
        has_more: data.pagination.has_more
      }));
    } catch (error) {
      console.error('❌ Error fetching jobs:', error);
    }
  };

  const fetchStats = async () => {
    try {
      console.log('📊 fetchStats starting...');
      const response = await queueApi.getStats();
      const data = response.data;
      console.log('✅ fetchStats success, total jobs:', data.total_jobs);
      setStats(data);
    } catch (error) {
      console.error('❌ Error fetching stats:', error);
    }
  };

  const retryJob = async (jobId: string) => {
    try {
      await queueApi.retryJob(jobId, false);
      fetchJobs();
    } catch (error) {
      console.error('Error retrying job:', error);
    }
  };

  const cancelJob = async (jobId: string) => {
    if (!confirm('Are you sure you want to cancel this job?')) return;

    try {
      await queueApi.cancelJob(jobId);
      fetchJobs();
    } catch (error) {
      console.error('Error cancelling job:', error);
    }
  };

  const applyFilters = () => {
    setPagination(prev => ({ ...prev, offset: 0 }));
    fetchJobs();
  };

  const clearFilters = () => {
    setFilters({ status: '', job_type: '', priority: '' });
    setPagination(prev => ({ ...prev, offset: 0 }));
  };

  const nextPage = () => {
    if (pagination.has_more) {
      setPagination(prev => ({ ...prev, offset: prev.offset + prev.limit }));
    }
  };

  const prevPage = () => {
    if (pagination.offset > 0) {
      setPagination(prev => ({ 
        ...prev, 
        offset: Math.max(0, prev.offset - prev.limit) 
      }));
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'queued': return <Clock className="w-4 h-4" />;
      case 'processing': return <Play className="w-4 h-4 animate-pulse" />;
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      case 'failed': return <XCircle className="w-4 h-4" />;
      case 'cancelled': return <StopCircle className="w-4 h-4" />;
      case 'retrying': return <RotateCcw className="w-4 h-4 animate-spin" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'queued': return 'text-yellow-600 bg-yellow-100';
      case 'processing': return 'text-blue-600 bg-blue-100';
      case 'completed': return 'text-green-600 bg-green-100';
      case 'failed': return 'text-red-600 bg-red-100';
      case 'cancelled': return 'text-gray-600 bg-gray-100';
      case 'retrying': return 'text-orange-600 bg-orange-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const formatJobType = (type: string) => {
    const typeMap: { [key: string]: string } = {
      'process_audio_files': 'Audio File Processing',
      'process_single_audio_file': 'Single Audio File',
      'reprocess_transcript': 'Reprocess Transcript',
      'reprocess_memory': 'Reprocess Memory'
    };
    return typeMap[type] || type;
  };

  const getJobTypeShort = (type: string) => {
    const typeMap: { [key: string]: string } = {
      'process_audio_files': 'Process',
      'process_single_audio_file': 'Process',
      'reprocess_transcript': 'Reprocess',
      'reprocess_memory': 'Memory'
    };
    return typeMap[type] || type;
  };

  const getJobResult = (job: QueueJob) => {
    if (job.status !== 'completed' || !job.result) {
      return <span className="text-sm text-gray-500">-</span>;
    }

    const result = job.result;

    // Show different results based on job type
    if (job.job_type === 'reprocess_transcript') {
      const segments = result.transcript_segments || 0;
      const speakers = result.speakers_identified || 0;

      return (
        <div className="text-sm text-gray-900">
          <div>{segments} segments</div>
          {speakers > 0 && (
            <div className="text-xs text-green-600">{speakers} speakers identified</div>
          )}
        </div>
      );
    }

    if (job.job_type === 'reprocess_memory') {
      const memories = result.memory_count || 0;
      return (
        <div className="text-sm text-gray-900">
          {memories} memories
        </div>
      );
    }

    return (
      <div className="text-sm text-green-600">
        ✓ Success
      </div>
    );
  };

  const flushJobs = async () => {
    setFlushing(true);
    try {
      const endpoint = flushSettings.flush_all ? '/api/queue/flush-all' : '/api/queue/flush';
      const body = flushSettings.flush_all
        ? { confirm: true }
        : {
            older_than_hours: flushSettings.older_than_hours,
            statuses: flushSettings.statuses
          };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      });

      if (response.ok) {
        const result = await response.json();
        alert(`Successfully flushed ${result.total_removed} jobs!`);
        setShowFlushModal(false);
        fetchData(); // Refresh the data
      } else if (response.status === 403) {
        alert('Admin access required to flush jobs');
      } else if (response.status === 401) {
        localStorage.removeItem('token');
        window.location.href = '/login';
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail || 'Failed to flush jobs'}`);
      }
    } catch (error) {
      console.error('Error flushing jobs:', error);
      alert('Failed to flush jobs');
    } finally {
      setFlushing(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <Layers className="w-6 h-6 text-blue-600" />
          <h1 className="text-2xl font-bold text-gray-900">Queue Management</h1>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowFlushModal(true)}
            className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            <Trash2 className="w-4 h-4" />
            <span>Flush Jobs</span>
          </button>
          <button
            onClick={() => fetchData()}
            disabled={refreshing}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
          <div className="bg-white rounded-lg border p-4">
            <div className="flex items-center space-x-2">
              <Layers className="w-5 h-5 text-gray-600" />
              <div>
                <p className="text-sm text-gray-600">Total</p>
                <p className="text-xl font-semibold">{stats.total_jobs}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border p-4">
            <div className="flex items-center space-x-2">
              <Clock className="w-5 h-5 text-yellow-600" />
              <div>
                <p className="text-sm text-gray-600">Queued</p>
                <p className="text-xl font-semibold text-yellow-600">{stats.queued_jobs}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border p-4">
            <div className="flex items-center space-x-2">
              <Play className={`w-5 h-5 text-blue-600 ${stats.processing_jobs > 0 ? 'animate-pulse' : ''}`} />
              <div>
                <p className="text-sm text-gray-600">Processing</p>
                <p className="text-xl font-semibold text-blue-600">{stats.processing_jobs}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border p-4">
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <div>
                <p className="text-sm text-gray-600">Completed</p>
                <p className="text-xl font-semibold text-green-600">{stats.completed_jobs}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border p-4">
            <div className="flex items-center space-x-2">
              <XCircle className="w-5 h-5 text-red-600" />
              <div>
                <p className="text-sm text-gray-600">Failed</p>
                <p className="text-xl font-semibold text-red-600">{stats.failed_jobs}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border p-4">
            <div className="flex items-center space-x-2">
              <StopCircle className="w-5 h-5 text-gray-600" />
              <div>
                <p className="text-sm text-gray-600">Cancelled</p>
                <p className="text-xl font-semibold text-gray-600">{stats.cancelled_jobs}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border p-4">
            <div className="flex items-center space-x-2">
              <RotateCcw className="w-5 h-5 text-orange-600" />
              <div>
                <p className="text-sm text-gray-600">Retrying</p>
                <p className="text-xl font-semibold text-orange-600">{stats.retrying_jobs}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="bg-white rounded-lg border p-4">
        <h3 className="text-lg font-medium mb-4">Filters</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
            <select
              value={filters.status}
              onChange={(e) => setFilters({ ...filters, status: e.target.value })}
              className="w-full border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="">All Statuses</option>
              <option value="queued">Queued</option>
              <option value="processing">Processing</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="cancelled">Cancelled</option>
              <option value="retrying">Retrying</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Job Type</label>
            <select
              value={filters.job_type}
              onChange={(e) => setFilters({ ...filters, job_type: e.target.value })}
              className="w-full border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="">All Types</option>
              <option value="process_audio_files">Audio File Processing</option>
              <option value="process_single_audio_file">Single Audio File</option>
              <option value="reprocess_transcript">Reprocess Transcript</option>
              <option value="reprocess_memory">Reprocess Memory</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Priority</label>
            <select
              value={filters.priority}
              onChange={(e) => setFilters({ ...filters, priority: e.target.value })}
              className="w-full border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="">All Priorities</option>
              <option value="high">High</option>
              <option value="normal">Normal</option>
              <option value="low">Low</option>
            </select>
          </div>

          <div className="flex items-end space-x-2">
            <button
              onClick={applyFilters}
              className="flex items-center space-x-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              <Filter className="w-4 h-4" />
              <span>Apply</span>
            </button>
            <button
              onClick={clearFilters}
              className="flex items-center space-x-1 px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
            >
              <X className="w-4 h-4" />
              <span>Clear</span>
            </button>
          </div>
        </div>
      </div>

      {/* Jobs Table */}
      <div className="bg-white rounded-lg border overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium">Jobs</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Result</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {jobs.map((job) => (
                <tr key={job.job_id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {formatDate(job.created_at)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">
                      #{job.job_id}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{getJobTypeShort(job.job_type)}</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                      {getStatusIcon(job.status)}
                      <span className="ml-1">{job.status.charAt(0).toUpperCase() + job.status.slice(1)}</span>
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {getJobResult(job)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium space-x-2">
                    {job.status === 'failed' && (
                      <button
                        onClick={() => retryJob(job.job_id)}
                        className="text-blue-600 hover:text-blue-900"
                      >
                        <RotateCcw className="w-4 h-4" />
                      </button>
                    )}
                    <button
                      onClick={() => setSelectedJob(job)}
                      className="text-indigo-600 hover:text-indigo-900"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    {(job.status === 'queued' || job.status === 'processing') && (
                      <button
                        onClick={() => cancelJob(job.job_id)}
                        className="text-red-600 hover:text-red-900"
                      >
                        <StopCircle className="w-4 h-4" />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {pagination.total > pagination.limit && (
          <div className="bg-white px-4 py-3 border-t border-gray-200 flex items-center justify-between">
            <div className="text-sm text-gray-700">
              Showing {pagination.offset + 1} to {Math.min(pagination.offset + pagination.limit, pagination.total)} of {pagination.total} results
            </div>
            <div className="flex space-x-2">
              <button
                onClick={prevPage}
                disabled={pagination.offset === 0}
                className="px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
              >
                Previous
              </button>
              <button
                onClick={nextPage}
                disabled={!pagination.has_more}
                className="px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Job Details Modal */}
      {selectedJob && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-11/12 md:w-3/4 lg:w-1/2 shadow-lg rounded-md bg-white">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-medium text-gray-900">Job Details</h3>
              <button onClick={() => setSelectedJob(null)} className="text-gray-400 hover:text-gray-600">
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Job ID</label>
                  <p className="text-sm text-gray-900">{selectedJob.job_id}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Status</label>
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(selectedJob.status)}`}>
                    {getStatusIcon(selectedJob.status)}
                    <span className="ml-1">{selectedJob.status.charAt(0).toUpperCase() + selectedJob.status.slice(1)}</span>
                  </span>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Type</label>
                  <p className="text-sm text-gray-900">{formatJobType(selectedJob.job_type)}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Priority</label>
                  <p className="text-sm text-gray-900">{selectedJob.priority}</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Created</label>
                  <p className="text-sm text-gray-900">{formatDate(selectedJob.created_at)}</p>
                </div>
                {selectedJob.completed_at && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Completed</label>
                    <p className="text-sm text-gray-900">{formatDate(selectedJob.completed_at)}</p>
                  </div>
                )}
              </div>
              
              {selectedJob.progress_message && (
                <div>
                  <label className="block text-sm font-medium text-gray-700">Progress</label>
                  <p className="text-sm text-gray-900">{selectedJob.progress_message}</p>
                  {selectedJob.progress_percent !== undefined && (
                    <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${selectedJob.progress_percent}%` }}
                      ></div>
                    </div>
                  )}
                </div>
              )}
              
              {selectedJob.error_message && (
                <div>
                  <label className="block text-sm font-medium text-gray-700">Error</label>
                  <p className="text-sm text-red-600 bg-red-50 p-2 rounded">{selectedJob.error_message}</p>
                </div>
              )}
              
              {selectedJob.data && (
                <div>
                  <label className="block text-sm font-medium text-gray-700">Job Data</label>
                  <pre className="text-xs text-gray-900 bg-gray-50 p-2 rounded overflow-auto max-h-64">
                    {JSON.stringify(selectedJob.data, null, 2)}
                  </pre>
                </div>
              )}
              
              {selectedJob.result && (
                <div>
                  <label className="block text-sm font-medium text-gray-700">Result</label>
                  <pre className="text-xs text-gray-900 bg-green-50 p-2 rounded overflow-auto max-h-64">
                    {JSON.stringify(selectedJob.result, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Flush Jobs Modal */}
      {showFlushModal && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-11/12 md:w-1/2 lg:w-1/3 shadow-lg rounded-md bg-white">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-medium text-gray-900 flex items-center">
                <Trash2 className="w-5 h-5 mr-2" />
                Flush Jobs
              </h3>
              <button onClick={() => setShowFlushModal(false)} className="text-gray-400 hover:text-gray-600">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
                <div className="flex items-center">
                  <AlertTriangle className="w-5 h-5 text-yellow-600 mr-2" />
                  <span className="text-sm text-yellow-800">This will permanently remove jobs from the database</span>
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <label className="flex items-center space-x-2">
                    <input
                      type="radio"
                      name="flushType"
                      checked={!flushSettings.flush_all}
                      onChange={() => setFlushSettings(prev => ({ ...prev, flush_all: false }))}
                      className="text-blue-600"
                    />
                    <span className="text-sm font-medium">Flush old inactive jobs (recommended)</span>
                  </label>

                  {!flushSettings.flush_all && (
                    <div className="ml-6 mt-2 space-y-2">
                      <div>
                        <label className="block text-xs text-gray-600 mb-1">Remove jobs older than:</label>
                        <select
                          value={flushSettings.older_than_hours}
                          onChange={(e) => setFlushSettings(prev => ({ ...prev, older_than_hours: parseInt(e.target.value) }))}
                          className="w-full text-sm border border-gray-300 rounded px-2 py-1"
                        >
                          <option value={1}>1 hour</option>
                          <option value={6}>6 hours</option>
                          <option value={12}>12 hours</option>
                          <option value={24}>24 hours</option>
                          <option value={72}>3 days</option>
                          <option value={168}>1 week</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-xs text-gray-600 mb-1">Job statuses to remove:</label>
                        <div className="space-y-1">
                          {['completed', 'failed', 'cancelled'].map(status => (
                            <label key={status} className="flex items-center space-x-2">
                              <input
                                type="checkbox"
                                checked={flushSettings.statuses.includes(status)}
                                onChange={(e) => {
                                  if (e.target.checked) {
                                    setFlushSettings(prev => ({
                                      ...prev,
                                      statuses: [...prev.statuses, status]
                                    }));
                                  } else {
                                    setFlushSettings(prev => ({
                                      ...prev,
                                      statuses: prev.statuses.filter(s => s !== status)
                                    }));
                                  }
                                }}
                                className="text-blue-600"
                              />
                              <span className="text-xs capitalize">{status}</span>
                            </label>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <div>
                  <label className="flex items-center space-x-2">
                    <input
                      type="radio"
                      name="flushType"
                      checked={flushSettings.flush_all}
                      onChange={() => setFlushSettings(prev => ({ ...prev, flush_all: true }))}
                      className="text-red-600"
                    />
                    <span className="text-sm font-medium text-red-600">Flush ALL jobs (DANGER!)</span>
                  </label>

                  {flushSettings.flush_all && (
                    <div className="ml-6 mt-2">
                      <div className="bg-red-50 border border-red-200 rounded p-2">
                        <p className="text-xs text-red-800">
                          ⚠️ This will remove ALL jobs including queued and processing ones, and reset the job counter!
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="flex space-x-2 pt-4 border-t">
                <button
                  onClick={() => setShowFlushModal(false)}
                  className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={flushJobs}
                  disabled={flushing || (!flushSettings.flush_all && flushSettings.statuses.length === 0)}
                  className={`flex-1 px-4 py-2 text-white rounded-lg disabled:opacity-50 ${
                    flushSettings.flush_all
                      ? 'bg-red-600 hover:bg-red-700'
                      : 'bg-blue-600 hover:bg-blue-700'
                  }`}
                >
                  {flushing ? (
                    <>
                      <RotateCcw className="w-4 h-4 animate-spin inline mr-2" />
                      Flushing...
                    </>
                  ) : (
                    <>
                      <Trash2 className="w-4 h-4 inline mr-2" />
                      {flushSettings.flush_all ? 'Flush ALL Jobs' : 'Flush Selected Jobs'}
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Queue;