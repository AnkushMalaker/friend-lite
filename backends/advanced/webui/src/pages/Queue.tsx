import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Clock,
  Play,
  Pause,
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
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  FileAudio,
  FileText,
  Brain,
  Repeat,
  Zap
} from 'lucide-react';
import { queueApi } from '../services/api';

interface QueueJob {
  job_id: string;
  job_type: string;
  user_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled' | 'deferred' | 'waiting';
  priority: 'low' | 'normal' | 'high';
  data: {
    description?: string;
    [key: string]: any;
  };
  result?: any;
  error_message?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  ended_at?: string;  // API returns this field instead of completed_at
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
  deferred_jobs: number;
  timestamp: string;
}

interface Filters {
  status: string;
  job_type: string;
  priority: string;
}

interface StreamingSession {
  session_id: string;
  user_id: string;
  client_id: string;
  provider: string;
  mode: string;
  status: string;
  chunks_published: number;
  started_at: number;
  last_chunk_at: number;
  age_seconds: number;
  idle_seconds: number;
}

interface StreamConsumer {
  name: string;
  pending: number;
  idle_ms: number;
}

interface StreamConsumerGroup {
  name: string;
  consumers: StreamConsumer[];
  pending: number;
}

interface StreamHealth {
  stream_length?: number;
  consumer_groups?: StreamConsumerGroup[];
  total_pending?: number;
  error?: string;
  exists?: boolean;
}

interface CompletedSession {
  session_id: string;
  client_id: string;
  conversation_id: string | null;
  has_conversation: boolean;
  action: string;
  reason: string;
  completed_at: number;
  audio_file: string;
}

interface StreamingStatus {
  active_sessions: StreamingSession[];
  completed_sessions: CompletedSession[];
  stream_health: {
    [provider: string]: StreamHealth;
  };
  rq_queues: {
    [queue: string]: {
      count: number;
      failed_count: number;
    };
  };
  timestamp: number;
}

const Queue: React.FC = () => {
  const [jobs, setJobs] = useState<QueueJob[]>([]);
  const [stats, setStats] = useState<QueueStats | null>(null);
  const [streamingStatus, setStreamingStatus] = useState<StreamingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedJob, setSelectedJob] = useState<any | null>(null);
  const [loadingJobDetails, setLoadingJobDetails] = useState(false);
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
  const [expandedSessions, setExpandedSessions] = useState<Set<string>>(new Set());
  const [sessionJobs, setSessionJobs] = useState<{[sessionId: string]: any[]}>({});
  const [lastUpdate, setLastUpdate] = useState<number>(Date.now());
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState<boolean>(() => {
    // Load from localStorage, default to true
    const saved = localStorage.getItem('queue_auto_refresh');
    return saved !== null ? saved === 'true' : true;
  });

  // Use refs to track current state in interval
  const expandedSessionsRef = useRef<Set<string>>(new Set());
  const streamingStatusRef = useRef<StreamingStatus | null>(null);
  const refreshingRef = useRef<boolean>(false);

  // Update refs when state changes
  useEffect(() => {
    expandedSessionsRef.current = expandedSessions;
  }, [expandedSessions]);

  useEffect(() => {
    streamingStatusRef.current = streamingStatus;
  }, [streamingStatus]);

  useEffect(() => {
    refreshingRef.current = refreshing;
  }, [refreshing]);

  // Refresh jobs for all expanded, active, and completed sessions
  const refreshSessionJobs = useCallback(async () => {
    const currentExpanded = expandedSessionsRef.current;
    const currentStreamingStatus = streamingStatusRef.current;

    // Get all active session IDs
    const activeSessionIds = currentStreamingStatus?.active_sessions
      ?.filter(s => s.status !== 'complete')
      .map(s => s.session_id) || [];

    // Get all completed session IDs
    const completedSessionIds = currentStreamingStatus?.completed_sessions
      ?.map(s => s.session_id) || [];

    // Get all session IDs that should have jobs loaded (expanded, active, or completed)
    const sessionIdsToRefresh = new Set([...currentExpanded, ...activeSessionIds, ...completedSessionIds]);

    if (sessionIdsToRefresh.size === 0) return;

    // Fetch jobs for all sessions in parallel
    const fetchPromises = Array.from(sessionIdsToRefresh).map(async (sessionId) => {
      try {
        const response = await queueApi.getJobsBySession(sessionId);
        return { sessionId, jobs: response.data.jobs };
      } catch (error) {
        console.error(`❌ Failed to refresh jobs for session ${sessionId}:`, error);
        return { sessionId, jobs: [] };
      }
    });

    const results = await Promise.all(fetchPromises);

    // Update session jobs state with all results
    setSessionJobs(prev => {
      const updated = { ...prev };
      results.forEach(({ sessionId, jobs }) => {
        updated[sessionId] = jobs;
      });
      return updated;
    });
  }, []);

  // Main data fetch function
  const fetchData = useCallback(async () => {
    if (refreshingRef.current) {
      return;
    }

    setRefreshing(true);

    try {
      // Fetch all main data in parallel
      await Promise.all([fetchJobs(), fetchStats(), fetchStreamingStatus()]);

      // Then refresh session jobs
      await refreshSessionJobs();

      setLastUpdate(Date.now());
    } catch (error) {
      console.error('❌ Error fetching queue data:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [refreshSessionJobs]);

  // Save auto-refresh preference to localStorage
  useEffect(() => {
    localStorage.setItem('queue_auto_refresh', autoRefreshEnabled.toString());
  }, [autoRefreshEnabled]);

  // Auto-refresh interval using useRef
  useEffect(() => {
    if (!autoRefreshEnabled) {
      return;
    }

    const intervalId = setInterval(() => {
      fetchData();
    }, 2000); // Refresh every 2 seconds

    return () => {
      clearInterval(intervalId);
    };
  }, [fetchData, autoRefreshEnabled]);

  // Initial data fetch
  useEffect(() => {
    fetchData();
  }, [filters, pagination.offset, fetchData]);

  const fetchJobs = async () => {
    try {
      const params = new URLSearchParams({
        limit: pagination.limit.toString(),
        offset: pagination.offset.toString(),
        sort: 'created_at',
        order: 'desc'
      });

      if (filters.status) params.append('status', filters.status);
      if (filters.job_type) params.append('job_type', filters.job_type);
      if (filters.priority) params.append('priority', filters.priority);

      const response = await queueApi.getJobs(params);
      const data = response.data;
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
      const response = await queueApi.getStats();
      const data = response.data;
      setStats(data);
    } catch (error) {
      console.error('❌ Error fetching stats:', error);
    }
  };

  const fetchStreamingStatus = async () => {
    try {
      const response = await queueApi.getStreamingStatus();
      const data = response.data;
      setStreamingStatus(data);

      // Auto-expand active sessions
      if (data.active_sessions && data.active_sessions.length > 0) {
        setExpandedSessions(prev => {
          const newExpanded = new Set(prev);
          let hasChanges = false;

          data.active_sessions.filter((s: StreamingSession) => s.status !== 'complete').forEach((session: StreamingSession) => {
            if (!newExpanded.has(session.session_id)) {
              newExpanded.add(session.session_id);
              hasChanges = true;
            }
          });

          return hasChanges ? newExpanded : prev;
        });
      }
    } catch (error) {
      console.error('❌ Error fetching streaming status:', error);
      // Don't fail the whole page if streaming status fails
      setStreamingStatus(null);
    }
  };

  const viewJobDetails = async (jobId: string) => {
    setLoadingJobDetails(true);
    try {
      const response = await queueApi.getJob(jobId);
      setSelectedJob(response.data);
    } catch (error) {
      console.error('Error fetching job details:', error);
      alert('Failed to fetch job details');
    } finally {
      setLoadingJobDetails(false);
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

  const cleanupStuckWorkers = async () => {
    if (!confirm('This will clean up all stuck workers and pending messages. Continue?')) return;

    try {
      console.log('🧹 Starting cleanup of stuck workers...');
      const response = await queueApi.cleanupStuckWorkers();
      const data = response.data;
      console.log('✅ Cleanup complete:', data);

      alert(`✅ Cleanup complete!\n\nTotal cleaned: ${data.total_cleaned} messages\n\n${
        Object.entries(data.providers).map(([provider, result]: [string, any]) =>
          `${provider}: ${result.message || result.error || 'Unknown'}`
        ).join('\n')
      }`);

      // Refresh streaming status to show updated counts
      fetchStreamingStatus();
    } catch (error: any) {
      console.error('❌ Error during cleanup:', error);
      alert(`Failed to cleanup workers: ${error.response?.data?.error || error.message}`);
    }
  };

  const cleanupOldSessions = async () => {
    if (!confirm('This will remove old and stuck "finalizing" sessions from the dashboard. Continue?')) return;

    try {
      console.log('🧹 Starting cleanup of old sessions...');
      const response = await queueApi.cleanupOldSessions(3600); // 1 hour
      const data = response.data;
      console.log('✅ Cleanup complete:', data);

      alert(`✅ Cleanup complete!\n\nRemoved ${data.cleaned_count} old session(s)`);

      // Refresh streaming status to show updated counts
      fetchStreamingStatus();
    } catch (error: any) {
      console.error('❌ Error during cleanup:', error);
      alert(`Failed to cleanup sessions: ${error.response?.data?.error || error.message}`);
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
      case 'deferred': return <Pause className="w-4 h-4" />;
      case 'waiting': return <Pause className="w-4 h-4" />;
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
      case 'deferred': return 'text-blue-600 bg-blue-100';
      case 'waiting': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
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

  const getJobTypeIcon = (type: string) => {
    const iconClass = "w-3.5 h-3.5";
    switch (type) {
      case 'audio_transcription':
      case 'process_audio_chunk':
        return <FileAudio className={iconClass} />;
      case 'transcript_processing':
      case 'reprocess_transcript':
        return <FileText className={iconClass} />;
      case 'memory_extraction':
      case 'reprocess_memory':
        return <Brain className={iconClass} />;
      case 'process_audio_files':
      case 'process_single_audio_file':
        return <Zap className={iconClass} />;
      default:
        return <Repeat className={iconClass} />;
    }
  };

  const getJobTypeColor = (type: string, status: string) => {
    // Base colors by job type
    let bgColor = 'bg-gray-400';
    let borderColor = 'border-gray-500';

    // Transcription jobs - blue shades
    if (type.includes('transcribe') || type === 'transcribe_full_audio_job') {
      bgColor = 'bg-blue-500';
      borderColor = 'border-blue-600';
    }
    // Speaker recognition - purple shades
    else if (type.includes('speaker') || type.includes('recognise') || type === 'recognise_speakers_job') {
      bgColor = 'bg-purple-500';
      borderColor = 'border-purple-600';
    }
    // Memory jobs - pink shades
    else if (type.includes('memory') || type === 'process_memory_job') {
      bgColor = 'bg-pink-500';
      borderColor = 'border-pink-600';
    }
    // Conversation/open jobs - cyan shades (check this AFTER memory to avoid confusion)
    else if (type.includes('conversation') || type.includes('open_conversation') || type === 'open_conversation_job') {
      bgColor = 'bg-cyan-500';
      borderColor = 'border-cyan-600';
    }
    // Speech detection jobs - green shades
    else if (type.includes('speech') || type.includes('detect')) {
      bgColor = 'bg-green-500';
      borderColor = 'border-green-600';
    }
    // Audio processing - orange shades
    else if (type.includes('audio') || type.includes('persist') || type.includes('cropping')) {
      bgColor = 'bg-orange-500';
      borderColor = 'border-orange-600';
    }
    // Default - gray
    else {
      bgColor = 'bg-gray-400';
      borderColor = 'border-gray-500';
    }

    // Failed jobs - always red
    if (status === 'failed') {
      bgColor = 'bg-red-500';
      borderColor = 'border-red-600';
    }
    // Processing jobs - add pulse animation
    else if (status === 'processing') {
      bgColor = bgColor + ' animate-pulse';
    }

    return { bgColor, borderColor };
  };

  const renderJobTimeline = (jobs: any[], session: StreamingSession | CompletedSession) => {
    if (!jobs || jobs.length === 0) return null;

    // Sort jobs by created_at first
    const sortedJobs = [...jobs].sort((a, b) =>
      new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    );

    // Calculate timeline boundaries
    // For active sessions, use session timestamps
    // For completed sessions without started_at, use earliest job timestamp
    let sessionStart: number;
    let sessionEnd: number;

    if ('started_at' in session) {
      // Active session - use session.started_at
      sessionStart = session.started_at * 1000;
    } else {
      // Completed session - calculate from jobs
      // Use the earliest job timestamp (created_at or started_at)
      const earliestTime = Math.min(...sortedJobs.map(j => {
        const created = new Date(j.created_at).getTime();
        const started = j.started_at ? new Date(j.started_at).getTime() : created;
        return Math.min(created, started);
      }));
      sessionStart = earliestTime;
    }

    if ('completed_at' in session) {
      // Completed session - use the latest job end time (not session.completed_at)
      // This handles batch jobs that run after the session is marked complete
      const latestJobEnd = Math.max(...sortedJobs.map(j => {
        const completed = j.completed_at ? new Date(j.completed_at).getTime() : 0;
        const ended = j.ended_at ? new Date(j.ended_at).getTime() : 0;
        const started = j.started_at ? new Date(j.started_at).getTime() : 0;
        return Math.max(completed, ended, started);
      }));
      // Use the later of: session completion or latest job end
      sessionEnd = Math.max(session.completed_at * 1000, latestJobEnd);
    } else {
      // Active session - use current time
      sessionEnd = Date.now();
    }

    const totalDuration = sessionEnd - sessionStart;

    if (totalDuration <= 0) return null;

    // Smart row assignment - place jobs in rows to avoid overlaps
    const rows: any[][] = [];
    sortedJobs.forEach(job => {
      const jobStart = job.started_at ? new Date(job.started_at).getTime() : new Date(job.created_at).getTime();

      // Find first row where this job doesn't overlap
      let assignedRow = -1;
      for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
        const row = rows[rowIndex];
        const lastJobInRow = row[row.length - 1];
        // Calculate when the last job in this row ends (use Date.now() for active jobs)
        const lastJobEnd = lastJobInRow.completed_at || lastJobInRow.ended_at
          ? new Date((lastJobInRow.completed_at || lastJobInRow.ended_at)!).getTime()
          : (lastJobInRow.status === 'processing' ? Date.now() : new Date(lastJobInRow.started_at || lastJobInRow.created_at).getTime());

        // If this job starts after the last job in this row ends, we can use this row
        if (jobStart >= lastJobEnd) {
          assignedRow = rowIndex;
          break;
        }
      }

      // If no suitable row found, create a new one
      if (assignedRow === -1) {
        assignedRow = rows.length;
        rows.push([]);
      }

      rows[assignedRow].push(job);
      job._assignedRow = assignedRow;
    });

    // Calculate height based on number of rows needed
    const rowCount = rows.length;
    const timelineHeight = Math.max(4, rowCount * 2); // At least 4rem, 2rem per row

    return (
      <div className="mt-3 mb-2">
        <div className="text-xs font-medium text-gray-700 mb-2">Timeline:</div>
        <div className="relative bg-gray-100 rounded-lg border border-gray-200 overflow-visible" style={{ height: `${timelineHeight}rem` }}>
          {/* Timeline grid lines */}
          <div className="absolute inset-0 flex">
            {[0, 25, 50, 75, 100].map(percent => (
              <div
                key={percent}
                className="absolute h-full border-l border-gray-300"
                style={{ left: `${percent}%` }}
              />
            ))}
          </div>

          {/* Job bars */}
          {sortedJobs.map((job) => {
            const jobStart = job.started_at ? new Date(job.started_at).getTime() : new Date(job.created_at).getTime();
            const jobEnd = job.completed_at || job.ended_at
              ? new Date((job.completed_at || job.ended_at)!).getTime()
              : (job.status === 'processing' ? Date.now() : jobStart);

            const startPercent = Math.max(0, ((jobStart - sessionStart) / totalDuration) * 100);
            const duration = jobEnd - jobStart;
            const widthPercent = Math.max(1, (duration / totalDuration) * 100);

            // Color based on job type
            const { bgColor, borderColor } = getJobTypeColor(job.job_type, job.status);

            // Calculate position in assigned row
            const rowIndex = job._assignedRow;
            const rowHeight = 100 / rowCount;
            const barHeight = Math.min(25, rowHeight * 0.6); // 60% of row height, max 25%
            const topPercent = (rowIndex * rowHeight) + (rowHeight - barHeight) / 2;

            // Format duration for display
            const durationMs = jobEnd - jobStart;
            let durationStr = '';
            if (durationMs < 1000) durationStr = `${durationMs}ms`;
            else if (durationMs < 60000) durationStr = `${(durationMs / 1000).toFixed(1)}s`;
            else durationStr = `${Math.floor(durationMs / 60000)}m ${Math.floor((durationMs % 60000) / 1000)}s`;

            return (
              <div
                key={job.job_id}
                className={`absolute ${bgColor} ${borderColor} border rounded shadow-sm group cursor-pointer transition-all hover:z-10 hover:scale-105`}
                style={{
                  left: `${startPercent}%`,
                  width: `${widthPercent}%`,
                  top: `${topPercent}%`,
                  height: `${barHeight}%`
                }}
                title={`${job.job_type} - ${job.status} - ${durationStr}`}
              >
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-white opacity-0 group-hover:opacity-100 transition-opacity">
                    {getJobTypeIcon(job.job_type)}
                  </div>
                </div>

                {/* Tooltip on hover - smart positioning to avoid viewport overflow */}
                <div
                  className="absolute bottom-full mb-1 px-2 py-1 bg-gray-900 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-20"
                  style={{
                    left: startPercent < 20 ? '0' : startPercent > 80 ? 'auto' : '50%',
                    right: startPercent > 80 ? '0' : 'auto',
                    transform: startPercent >= 20 && startPercent <= 80 ? 'translateX(-50%)' : 'none'
                  }}
                >
                  <div className="font-medium">{job.job_type}</div>
                  <div className="text-gray-300">{job.status} • {durationStr}</div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Timeline labels */}
        <div className="flex justify-between text-xs text-gray-500 mt-1 px-1">
          <span>0s</span>
          <span>{(totalDuration / 1000).toFixed(0)}s</span>
        </div>
      </div>
    );
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

  const formatDuration = (job: any) => {
    if (!job.started_at) return '-';

    const start = new Date(job.started_at).getTime();
    // For failed/finished jobs, use completed_at or ended_at. For running jobs, use current time.
    const end = job.completed_at || job.ended_at
      ? new Date((job.completed_at || job.ended_at)!).getTime()
      : (job.status === 'processing' ? Date.now() : start); // Don't show increasing time for failed jobs
    const durationMs = end - start;

    if (durationMs < 1000) return `${durationMs}ms`;
    if (durationMs < 60000) return `${(durationMs / 1000).toFixed(1)}s`;
    if (durationMs < 3600000) return `${Math.floor(durationMs / 60000)}m ${Math.floor((durationMs % 60000) / 1000)}s`;
    return `${Math.floor(durationMs / 3600000)}h ${Math.floor((durationMs % 3600000) / 60000)}m`;
  };

  const toggleSessionExpansion = async (sessionId: string) => {
    const newExpanded = new Set(expandedSessions);

    if (newExpanded.has(sessionId)) {
      // Collapse
      newExpanded.delete(sessionId);
      setExpandedSessions(newExpanded);
    } else {
      // Expand and fetch jobs if not already loaded
      newExpanded.add(sessionId);
      setExpandedSessions(newExpanded);

      if (!sessionJobs[sessionId]) {
        try {
          const response = await queueApi.getJobsBySession(sessionId);
          const data = response.data;
          setSessionJobs(prev => ({ ...prev, [sessionId]: data.jobs }));
        } catch (error) {
          console.error(`❌ Failed to fetch jobs for session ${sessionId}:`, error);
          setSessionJobs(prev => ({ ...prev, [sessionId]: [] }));
        }
      }
    }
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
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Queue Management</h1>
            <p className="text-xs text-gray-500">
              Last updated: {new Date(lastUpdate).toLocaleTimeString()} • Auto-refresh every 2s
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setAutoRefreshEnabled(!autoRefreshEnabled)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
              autoRefreshEnabled
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : 'bg-gray-600 hover:bg-gray-700 text-white'
            }`}
            title={autoRefreshEnabled ? 'Auto-refresh enabled (click to disable)' : 'Auto-refresh disabled (click to enable)'}
          >
            {autoRefreshEnabled ? (
              <>
                <Pause className="w-4 h-4" />
                <span>Auto-refresh ON</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Auto-refresh OFF</span>
              </>
            )}
          </button>
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
              <Pause className="w-5 h-5 text-blue-600" />
              <div>
                <p className="text-sm text-gray-600">Deferred</p>
                <p className="text-xl font-semibold text-blue-600">{stats.deferred_jobs}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Streaming Status */}
      {streamingStatus && (
        <div className="bg-white rounded-lg border overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
            <h3 className="text-lg font-medium">Audio Streaming Status</h3>
            <div className="flex items-center space-x-2">
              <button
                onClick={cleanupOldSessions}
                className="flex items-center space-x-2 px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700 transition-colors text-sm"
                title="Remove old sessions (>1 hour old)"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Cleanup Old Sessions</span>
              </button>
              {streamingStatus?.active_sessions && streamingStatus.active_sessions.length > 0 && (
                <button
                  onClick={async () => {
                    if (!streamingStatus || !confirm(`Remove ALL ${streamingStatus.active_sessions.length} active sessions? This will force-delete all sessions including actively streaming ones.`)) return;

                    try {
                      const response = await queueApi.cleanupOldSessions(0); // 0 seconds = all sessions
                      const data = response.data;
                      alert(`✅ Removed ${data.cleaned_count} session(s)`);
                      fetchStreamingStatus();
                    } catch (error: any) {
                      console.error('❌ Error removing sessions:', error);
                      alert(`Failed to remove sessions: ${error.response?.data?.error || error.message}`);
                    }
                  }}
                  className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors text-sm"
                  title="Force remove ALL active sessions"
                >
                  <Trash2 className="w-4 h-4" />
                  <span>Remove All Sessions ({streamingStatus.active_sessions.length})</span>
                </button>
              )}
              <button
                onClick={cleanupStuckWorkers}
                className="flex items-center space-x-2 px-4 py-2 bg-yellow-600 text-white rounded-md hover:bg-yellow-700 transition-colors text-sm"
                title="Clean up stuck workers and pending messages"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Cleanup Stuck Workers{streamingStatus?.stream_health && Object.values(streamingStatus.stream_health).some((s: any) => s.total_pending > 0) && ` (${
                  Object.values(streamingStatus.stream_health).reduce((sum: number, s: any) => sum + (s.total_pending || 0), 0)
                })`}</span>
              </button>
            </div>
          </div>

          <div className="p-6 space-y-6">
            {/* Active and Completed Sessions Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Active Sessions */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-3">Active Streaming Sessions</h4>
                {streamingStatus?.active_sessions && streamingStatus.active_sessions.filter(s => s.status !== 'complete').length > 0 ? (
                  <div className="space-y-2">
                    {streamingStatus.active_sessions.filter(s => s.status !== 'complete').map((session) => {
                      const isExpanded = expandedSessions.has(session.session_id);
                      const jobs = sessionJobs[session.session_id] || [];

                      return (
                        <div key={session.session_id} className="bg-blue-50 rounded-lg border border-blue-200 overflow-hidden">
                          <div
                            className="flex items-center justify-between p-3 cursor-pointer hover:bg-blue-100 transition-colors"
                            onClick={() => toggleSessionExpansion(session.session_id)}
                          >
                            <div className="flex-1">
                              <div className="flex items-center space-x-2">
                                {isExpanded ? (
                                  <ChevronDown className="w-4 h-4 text-blue-600" />
                                ) : (
                                  <ChevronRight className="w-4 h-4 text-blue-600" />
                                )}
                                <Play className="w-4 h-4 text-blue-600 animate-pulse" />
                                <span className="text-sm font-medium text-gray-900">{session.client_id}</span>
                                <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded">{session.provider}</span>
                                <span className="text-xs px-2 py-0.5 bg-green-100 text-green-700 rounded">{session.status}</span>
                              </div>
                              <div className="mt-1 text-xs text-gray-600">
                                Session: {session.session_id.substring(0, 8)}... •
                                Chunks: {session.chunks_published} •
                                Duration: {Math.floor(session.age_seconds)}s •
                                Idle: {session.idle_seconds.toFixed(1)}s
                              </div>
                            </div>
                          </div>

                          {/* Expanded Jobs Section */}
                          {isExpanded && (
                            <div className="border-t border-blue-200 bg-white p-3">
                              {/* Timeline Visualization */}
                              {renderJobTimeline(jobs, session)}

                              <h5 className="text-xs font-medium text-gray-700 mb-2">Jobs for this session:</h5>
                              {jobs.length > 0 ? (
                                <div className="space-y-1">
                                  {jobs.map((job, index) => (
                                    <div key={job.job_id} className={`flex items-center justify-between p-2 bg-gray-50 rounded border ${getJobTypeColor(job.job_type, job.status).borderColor}`} style={{ borderLeftWidth: '12px' }}>
                                      <div className="flex-1 min-w-0">
                                        <div className="flex items-center space-x-2">
                                          <span className="text-xs font-mono text-gray-500 flex-shrink-0">#{index + 1}</span>
                                          <span className="flex-shrink-0">{getJobTypeIcon(job.job_type)}</span>
                                          <span className="flex-shrink-0">{getStatusIcon(job.status)}</span>
                                          <span className="text-xs font-medium text-gray-900 truncate">{job.job_type}</span>
                                          <span className={`text-xs px-1.5 py-0.5 rounded ${getStatusColor(job.status)}`}>
                                            {job.status}
                                          </span>
                                          <span className="text-xs text-gray-500">{job.queue}</span>
                                        </div>
                                        <div className="mt-1 text-xs text-gray-600">
                                          {job.started_at && (
                                            <span>Started: {new Date(job.started_at).toLocaleTimeString()}</span>
                                          )}
                                          {job.started_at && (
                                            <span> • Duration: {formatDuration(job)}</span>
                                          )}
                                        </div>
                                      </div>
                                      <button
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          viewJobDetails(job.job_id);
                                        }}
                                        className="ml-2 text-indigo-600 hover:text-indigo-900 flex-shrink-0"
                                      >
                                        <Eye className="w-3 h-3" />
                                      </button>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="text-xs text-gray-500 italic">No jobs found for this session</div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500 text-sm bg-gray-50 rounded-lg border border-gray-200">
                    No active sessions
                  </div>
                )}
              </div>

              {/* Completed Sessions */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-3">Completed Sessions (Last Hour)</h4>
                {streamingStatus?.completed_sessions && streamingStatus.completed_sessions.length > 0 ? (
                  <div className="space-y-2">
                    {streamingStatus.completed_sessions.map((session) => {
                      const isExpanded = expandedSessions.has(session.session_id);
                      const jobs = sessionJobs[session.session_id] || [];

                      return (
                        <div key={session.session_id} className={`rounded-lg border overflow-hidden ${
                          session.has_conversation
                            ? 'bg-green-50 border-green-200'
                            : 'bg-gray-50 border-gray-200'
                        }`}>
                          <div
                            className={`flex items-center justify-between p-3 cursor-pointer transition-colors ${
                              session.has_conversation
                                ? 'hover:bg-green-100'
                                : 'hover:bg-gray-100'
                            }`}
                            onClick={() => toggleSessionExpansion(session.session_id)}
                          >
                            <div className="flex-1">
                              <div className="flex items-center space-x-2">
                                {isExpanded ? (
                                  <ChevronDown className={`w-4 h-4 ${session.has_conversation ? 'text-green-600' : 'text-gray-600'}`} />
                                ) : (
                                  <ChevronRight className={`w-4 h-4 ${session.has_conversation ? 'text-green-600' : 'text-gray-600'}`} />
                                )}
                                {session.has_conversation ? (
                                  <CheckCircle className="w-4 h-4 text-green-600" />
                                ) : (
                                  <XCircle className="w-4 h-4 text-gray-600" />
                                )}
                                <span className="text-sm font-medium text-gray-900">{session.client_id}</span>
                                {session.has_conversation ? (
                                  <span className="text-xs px-2 py-0.5 bg-green-100 text-green-700 rounded">Conversation</span>
                                ) : (
                                  <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-600 rounded">{session.reason || 'No speech'}</span>
                                )}
                              </div>
                              <div className="mt-1 text-xs text-gray-600">
                                Session: {session.session_id.substring(0, 8)}... •
                                {new Date(session.completed_at * 1000).toLocaleTimeString()}
                              </div>
                            </div>
                          </div>

                          {/* Expanded Jobs Section */}
                          {isExpanded && (
                            <div className={`border-t bg-white p-3 ${
                              session.has_conversation ? 'border-green-200' : 'border-gray-200'
                            }`}>
                              {/* Timeline Visualization */}
                              {renderJobTimeline(jobs, session)}

                              <h5 className="text-xs font-medium text-gray-700 mb-2">Jobs for this session:</h5>
                              {jobs.length > 0 ? (
                                <div className="space-y-1">
                                  {jobs.map((job, index) => (
                                    <div key={job.job_id} className={`flex items-center justify-between p-2 bg-gray-50 rounded border ${getJobTypeColor(job.job_type, job.status).borderColor}`} style={{ borderLeftWidth: '12px' }}>
                                      <div className="flex-1 min-w-0">
                                        <div className="flex items-center space-x-2">
                                          <span className="text-xs font-mono text-gray-500 flex-shrink-0">#{index + 1}</span>
                                          <span className="flex-shrink-0">{getJobTypeIcon(job.job_type)}</span>
                                          <span className="flex-shrink-0">{getStatusIcon(job.status)}</span>
                                          <span className="text-xs font-medium text-gray-900 truncate">{job.job_type}</span>
                                          <span className={`text-xs px-1.5 py-0.5 rounded ${getStatusColor(job.status)}`}>
                                            {job.status}
                                          </span>
                                          <span className="text-xs text-gray-500">{job.queue}</span>
                                        </div>
                                        <div className="mt-1 text-xs text-gray-600">
                                          {job.started_at && (
                                            <span>Started: {new Date(job.started_at).toLocaleTimeString()}</span>
                                          )}
                                          {job.started_at && (
                                            <span> • Duration: {formatDuration(job)}</span>
                                          )}
                                        </div>
                                      </div>
                                      <button
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          viewJobDetails(job.job_id);
                                        }}
                                        className="ml-2 text-indigo-600 hover:text-indigo-900 flex-shrink-0"
                                      >
                                        <Eye className="w-3 h-3" />
                                      </button>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="text-xs text-gray-500 italic">No jobs found for this session</div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500 text-sm bg-gray-50 rounded-lg border border-gray-200">
                    No completed sessions
                  </div>
                )}
              </div>
            </div>

            {/* Stream Health */}
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Stream Workers</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {streamingStatus?.stream_health && Object.entries(streamingStatus.stream_health).map(([provider, health]) => (
                  <div key={provider} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium capitalize">{provider}</span>
                      {health.error ? (
                        <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-600 rounded">Inactive</span>
                      ) : (
                        <span className="text-xs px-2 py-0.5 bg-green-100 text-green-700 rounded">Active</span>
                      )}
                    </div>

                    {health.error ? (
                      <p className="text-xs text-gray-500">{health.error}</p>
                    ) : (
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-600">Stream Length:</span>
                          <span className="font-medium">{health.stream_length}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-600">Pending:</span>
                          <span className={`font-medium ${health.total_pending && health.total_pending > 0 ? 'text-yellow-600' : 'text-green-600'}`}>
                            {health.total_pending}
                          </span>
                        </div>
                        {health.consumer_groups && health.consumer_groups.map((group) => (
                          <div key={group.name} className="mt-2 pt-2 border-t border-gray-200">
                            <div className="text-xs text-gray-600 mb-1">Consumers:</div>
                            {group.consumers.map((consumer) => (
                              <div key={consumer.name} className="flex justify-between text-xs pl-2">
                                <span className="text-gray-700 truncate">{consumer.name}</span>
                                <span className={consumer.pending > 0 ? 'text-yellow-600' : 'text-green-600'}>
                                  {consumer.pending} pending
                                </span>
                              </div>
                            ))}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
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
              <option value="deferred">Deferred</option>
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
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Date</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Job ID</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Type</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Duration</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Result</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase whitespace-nowrap">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {jobs.map((job) => (
                <tr key={job.job_id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-500 whitespace-nowrap">
                    {new Date(job.created_at).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                  </td>
                  <td className="px-4 py-3 max-w-xs">
                    <div className="text-xs font-mono text-gray-900 truncate" title={job.job_id}>
                      {job.job_id}
                    </div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <div className="text-sm text-gray-900">{getJobTypeShort(job.job_type)}</div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                      {getStatusIcon(job.status)}
                      <span className="ml-1">{job.status.charAt(0).toUpperCase() + job.status.slice(1)}</span>
                    </span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <div className="text-sm text-gray-700 font-mono">
                      {formatDuration(job)}
                    </div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    {getJobResult(job)}
                  </td>
                  <td className="px-4 py-3 text-sm font-medium space-x-2 whitespace-nowrap">
                    {job.status === 'failed' && (
                      <button
                        onClick={() => retryJob(job.job_id)}
                        className="text-blue-600 hover:text-blue-900"
                      >
                        <RotateCcw className="w-4 h-4" />
                      </button>
                    )}
                    <button
                      onClick={() => viewJobDetails(job.job_id)}
                      className="text-indigo-600 hover:text-indigo-900"
                      disabled={loadingJobDetails}
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
          <div className="relative top-20 mx-auto p-5 border w-11/12 max-w-6xl shadow-lg rounded-md bg-white">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-medium text-gray-900">Job Details</h3>
              <button onClick={() => setSelectedJob(null)} className="text-gray-400 hover:text-gray-600">
                <X className="w-5 h-5" />
              </button>
            </div>

            {loadingJobDetails ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Job ID</label>
                    <p className="text-sm text-gray-900 font-mono">{selectedJob.job_id}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Status</label>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(selectedJob.status)}`}>
                      {getStatusIcon(selectedJob.status)}
                      <span className="ml-1">{selectedJob.status.charAt(0).toUpperCase() + selectedJob.status.slice(1)}</span>
                    </span>
                  </div>
                  {selectedJob.description && (
                    <div className="col-span-2">
                      <label className="block text-sm font-medium text-gray-700">Description</label>
                      <p className="text-sm text-gray-900">{selectedJob.description}</p>
                    </div>
                  )}
                  {selectedJob.func_name && (
                    <div className="col-span-2">
                      <label className="block text-sm font-medium text-gray-700">Function Name</label>
                      <p className="text-sm text-gray-900 font-mono">{selectedJob.func_name}</p>
                    </div>
                  )}
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Created</label>
                    <p className="text-sm text-gray-900">{selectedJob.created_at ? formatDate(selectedJob.created_at) : '-'}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Started</label>
                    <p className="text-sm text-gray-900">{selectedJob.started_at ? formatDate(selectedJob.started_at) : '-'}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Ended</label>
                    <p className="text-sm text-gray-900">{selectedJob.ended_at ? formatDate(selectedJob.ended_at) : '-'}</p>
                  </div>
                </div>

                {selectedJob.args && selectedJob.args.length > 0 && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Arguments</label>
                    <pre className="text-xs text-gray-900 bg-gray-50 p-2 rounded overflow-auto max-h-64">
                      {JSON.stringify(selectedJob.args, null, 2)}
                    </pre>
                  </div>
                )}

                {selectedJob.kwargs && Object.keys(selectedJob.kwargs).length > 0 && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Keyword Arguments</label>
                    <pre className="text-xs text-gray-900 bg-gray-50 p-2 rounded overflow-auto max-h-64">
                      {JSON.stringify(selectedJob.kwargs, null, 2)}
                    </pre>
                  </div>
                )}

                {selectedJob.error_message && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Error</label>
                    <pre className="text-xs text-red-600 bg-red-50 p-2 rounded overflow-auto max-h-64">
                      {selectedJob.error_message}
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
            )}
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