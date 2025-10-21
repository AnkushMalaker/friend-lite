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
  active_sessions: StreamingSession[];  // Kept for backward compatibility
  completed_sessions: CompletedSession[];
  stream_health: {
    [streamKey: string]: StreamHealth & {
      stream_age_seconds?: number;
    };
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
  const [expandedJobs, setExpandedJobs] = useState<Set<string>>(new Set());
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

  // Main data fetch function - uses consolidated dashboard endpoint
  const fetchData = useCallback(async () => {
    if (refreshingRef.current) {
      return;
    }

    setRefreshing(true);

    try {
      const currentExpanded = expandedSessionsRef.current;
      const expandedSessionIds = Array.from(currentExpanded);

      // Single API call to get all dashboard data
      const response = await queueApi.getDashboard(expandedSessionIds);
      const dashboardData = response.data;

      // Extract jobs from response
      const queuedJobs = dashboardData.jobs.queued || [];
      const processingJobs = dashboardData.jobs.processing || [];
      const completedJobs = dashboardData.jobs.completed || [];
      const failedJobs = dashboardData.jobs.failed || [];

      // Combine all jobs
      const allFetchedJobs = [...queuedJobs, ...processingJobs, ...completedJobs, ...failedJobs];

      console.log(`📊 Fetched ${allFetchedJobs.length} total jobs via consolidated endpoint`);
      console.log(`  - Queued: ${queuedJobs.length}`);
      console.log(`  - Processing: ${processingJobs.length}`);
      console.log(`  - Completed: ${completedJobs.length}`);
      console.log(`  - Failed: ${failedJobs.length}`);

      // Debug: Log open_conversation_job details
      const openConvJobs = allFetchedJobs.filter(j => j?.job_type === 'open_conversation_job');
      console.log(`🔍 Found ${openConvJobs.length} open_conversation_job(s):`);
      openConvJobs.forEach(job => {
        console.log(`  Job ID: ${job.job_id}`);
        console.log(`  Status: ${job.status}`);
        console.log(`  meta.audio_uuid: ${job.meta?.audio_uuid}`);
        console.log(`  meta.conversation_id: ${job.meta?.conversation_id}`);
      });

      // Group jobs by session_id (use audio_uuid from metadata)
      const jobsBySession: {[sessionId: string]: any[]} = {};

      allFetchedJobs.forEach(job => {
        if (!job || !job.job_id) return; // Skip invalid jobs

        // Extract session_id from meta.audio_uuid
        const sessionId = job.meta?.audio_uuid;
        if (sessionId) {
          if (!jobsBySession[sessionId]) {
            jobsBySession[sessionId] = [];
          }
          jobsBySession[sessionId].push(job);

          // Debug logging for grouping
          if (job.job_type === 'open_conversation_job') {
            console.log(`✅ Grouped open_conversation_job ${job.job_id} under session ${sessionId}`);
          }
        } else {
          // Log jobs that couldn't be grouped
          console.log(`⚠️ Job ${job.job_id} (${job.job_type}) has no session_id - cannot group`);
        }
      });

      // Merge session jobs from dashboard response
      if (dashboardData.session_jobs) {
        Object.entries(dashboardData.session_jobs).forEach(([sessionId, jobs]: [string, any]) => {
          // Merge with existing jobs and deduplicate by job_id
          const existingJobs = jobsBySession[sessionId] || [];
          const existingJobIds = new Set(existingJobs.map((j: any) => j.job_id));
          const newJobs = jobs.filter((j: any) => !existingJobIds.has(j.job_id));
          jobsBySession[sessionId] = [...existingJobs, ...newJobs];
        });
      }

      // Update state
      setSessionJobs(jobsBySession);
      setStats(dashboardData.stats);
      setStreamingStatus(dashboardData.streaming_status);
      setLastUpdate(Date.now());

      // Auto-expand active conversations (those with open_conversation_job in progress)
      const newExpanded = new Set(expandedSessions);
      let expandedCount = 0;

      // Find all conversations with active open_conversation_job
      Object.entries(jobsBySession).forEach(([_sessionId, jobs]) => {
        const openConvJob = jobs.find((j: any) => j.job_type === 'open_conversation_job');
        if (openConvJob && openConvJob.status === 'started') {
          const conversationId = openConvJob.meta?.conversation_id;
          if (conversationId && !expandedSessions.has(conversationId)) {
            newExpanded.add(conversationId);
            expandedCount++;
            console.log(`🔓 Auto-expanding active conversation: ${conversationId}`);
          }
        }
      });

      // Update expanded sessions if any new active conversations found
      if (expandedCount > 0) {
        console.log(`📂 Auto-expanded ${expandedCount} active conversation(s)`);
        setExpandedSessions(newExpanded);
      }
    } catch (error) {
      console.error('❌ Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

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

  // ESC key handler for modals
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (selectedJob) {
          setSelectedJob(null);
        } else if (showFlushModal) {
          setShowFlushModal(false);
        }
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [selectedJob, showFlushModal]);

  // Commented out - keeping for future use
  // const retryJob = async (jobId: string) => {
  //   try {
  //     await queueApi.retryJob(jobId, false);
  //     fetchData();
  //   } catch (error) {
  //     console.error('Error retrying job:', error);
  //   }
  // };

  // const cancelJob = async (jobId: string) => {
  //   if (!confirm('Are you sure you want to cancel this job?')) return;

  //   try {
  //     await queueApi.cancelJob(jobId);
  //     fetchData();
  //   } catch (error) {
  //     console.error('Error cancelling job:', error);
  //   }
  // };

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

      // Refresh data to show updated counts
      fetchData();
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

      // Refresh data to show updated counts
      fetchData();
    } catch (error: any) {
      console.error('❌ Error during cleanup:', error);
      alert(`Failed to cleanup sessions: ${error.response?.data?.error || error.message}`);
    }
  };

  const applyFilters = () => {
    setPagination(prev => ({ ...prev, offset: 0 }));
    fetchData();
  };

  const clearFilters = () => {
    setFilters({ status: '', job_type: '', priority: '' });
    setPagination(prev => ({ ...prev, offset: 0 }));
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
    // Safety check for undefined/null values
    if (!type || !status) {
      return { bgColor: 'bg-gray-400', borderColor: 'border-gray-500' };
    }

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

  const toggleSessionExpansion = (sessionId: string) => {
    const newExpanded = new Set(expandedSessions);

    if (newExpanded.has(sessionId)) {
      // Collapse
      newExpanded.delete(sessionId);
      setExpandedSessions(newExpanded);
    } else {
      // Expand and trigger refresh to fetch jobs via dashboard endpoint
      newExpanded.add(sessionId);
      setExpandedSessions(newExpanded);

      // Trigger a refresh if jobs not already loaded
      if (!sessionJobs[sessionId]) {
        fetchData();
      }
    }
  };

  const toggleJobExpansion = (jobId: string) => {
    const newExpanded = new Set(expandedJobs);
    if (newExpanded.has(jobId)) {
      newExpanded.delete(jobId);
    } else {
      newExpanded.add(jobId);
    }
    setExpandedJobs(newExpanded);
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
            <h3 className="text-lg font-medium">Audio Streaming & Conversations</h3>
            <div className="flex items-center space-x-2">
              <button
                onClick={cleanupOldSessions}
                className="flex items-center space-x-2 px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700 transition-colors text-sm"
                title="Remove old sessions (>1 hour old)"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Cleanup Old Sessions</span>
              </button>
              {streamingStatus?.stream_health && Object.keys(streamingStatus.stream_health).length > 0 && (
                <button
                  onClick={async () => {
                    const streamCount = Object.keys(streamingStatus.stream_health).length;
                    if (!streamingStatus || !confirm(`Remove ALL ${streamCount} active streams? This will force-delete all streams including actively streaming ones.`)) return;

                    try {
                      const response = await queueApi.cleanupOldSessions(0); // 0 seconds = all sessions
                      const data = response.data;
                      alert(`✅ Removed ${data.cleaned_count} stream(s)`);
                      fetchData();
                    } catch (error: any) {
                      console.error('❌ Error removing streams:', error);
                      alert(`Failed to remove streams: ${error.response?.data?.error || error.message}`);
                    }
                  }}
                  className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors text-sm"
                  title="Force remove ALL active streams"
                >
                  <Trash2 className="w-4 h-4" />
                  <span>Remove All Streams ({Object.keys(streamingStatus.stream_health).length})</span>
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
            {/* Stream Workers Section - Shows audio streams + listen jobs */}
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">Stream Workers (Client Sessions)</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {streamingStatus?.stream_health && Object.entries(streamingStatus.stream_health).map(([streamKey, health]) => {
                  // Extract client_id from stream key (format: audio:stream:{client_id})
                  const clientId = streamKey.replace('audio:stream:', '');

                  // Find all listen jobs for this client with deduplication
                  const allJobsRaw = Object.values(sessionJobs).flat().filter(job => job != null);

                  // Deduplicate by job_id
                  const jobMap = new Map();
                  allJobsRaw.forEach((job: any) => {
                    if (job && job.job_id) {
                      jobMap.set(job.job_id, job);
                    }
                  });
                  const allJobs = Array.from(jobMap.values());

                  // Debug logging for listen job filtering
                  console.log(`🔍 Stream ${streamKey}:`);
                  console.log(`  - clientId extracted: ${clientId}`);
                  console.log(`  - Total jobs available: ${allJobs.length}`);
                  const speechDetectionJobs = allJobs.filter((job: any) => job && job.job_type === 'stream_speech_detection_job');
                  console.log(`  - Speech detection jobs: ${speechDetectionJobs.length}`, speechDetectionJobs.map((j: any) => ({ job_id: j.job_id, meta_client_id: j.meta?.client_id })));

                  // Get all listen jobs for this client (only active/queued/processing, not completed)
                  const allListenJobs = allJobs.filter((job: any) =>
                    job && job.job_type === 'stream_speech_detection_job' &&
                    job.meta?.client_id === clientId &&
                    job.status !== 'completed' &&
                    job.status !== 'failed'
                  );

                  // Show only the LATEST active speech detection job (most recent created_at)
                  // Completed ones have already exited and shouldn't be shown here
                  const listenJobs = allListenJobs.length > 0
                    ? [allListenJobs.sort((a, b) =>
                        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
                      )[0]]
                    : [];

                  console.log(`  - All listen jobs (active): ${allListenJobs.length}, showing latest: ${listenJobs.length}`);

                  return (
                    <div key={streamKey} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">{streamKey}</span>
                        <span className="text-xs px-2 py-0.5 bg-green-100 text-green-700 rounded">Active</span>
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-600">Stream Length:</span>
                          <span className="font-medium">{health.stream_length}</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-600">Age:</span>
                          <span className="font-medium">{(health.stream_age_seconds || 0).toFixed(0)}s</span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-600">Pending:</span>
                          <span className={`font-medium ${health.total_pending && health.total_pending > 0 ? 'text-yellow-600' : 'text-green-600'}`}>
                            {health.total_pending}
                          </span>
                        </div>
                        {health.consumer_groups && health.consumer_groups.map((group) => (
                          <div key={group.name} className="mt-2 pt-2 border-t border-gray-200">
                            <div className="text-xs text-gray-600 mb-1">{group.name}:</div>
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

                        {/* Current Speech Detection Job */}
                        {listenJobs.length > 0 && (
                          <div className="mt-2 pt-2 border-t border-gray-200">
                            <div className="text-xs text-gray-600 mb-1">Current Speech Detection:</div>
                            {listenJobs.map((job) => {
                              const runtime = job.started_at
                                ? Math.floor((Date.now() - new Date(job.started_at).getTime()) / 1000)
                                : 0;
                              const minutes = Math.floor(runtime / 60);
                              const seconds = runtime % 60;

                              return (
                                <div key={job.job_id} className="bg-white rounded p-2 space-y-1">
                                  <div className="flex items-center justify-between">
                                    <div className="flex items-center space-x-1">
                                      {getStatusIcon(job.status)}
                                      <span className="text-gray-700 font-medium text-xs">{job.job_type}</span>
                                      <span className={`px-1 py-0.5 rounded text-xs ${getStatusColor(job.status)}`}>
                                        {job.status}
                                      </span>
                                    </div>
                                    <button
                                      onClick={() => viewJobDetails(job.job_id)}
                                      className="text-indigo-600 hover:text-indigo-900 flex-shrink-0"
                                    >
                                      <Eye className="w-3 h-3" />
                                    </button>
                                  </div>

                                  {/* Job metadata */}
                                  <div className="text-xs text-gray-600 space-y-0.5 pl-4">
                                    <div className="flex justify-between">
                                      <span>Job ID:</span>
                                      <span className="font-mono text-gray-800">{job.job_id.substring(0, 12)}...</span>
                                    </div>
                                    {job.started_at && (
                                      <div className="flex justify-between">
                                        <span>Runtime:</span>
                                        <span className="font-medium text-gray-800">{minutes}m {seconds}s</span>
                                      </div>
                                    )}
                                    {job.created_at && (
                                      <div className="flex justify-between">
                                        <span>Created:</span>
                                        <span className="text-gray-800">{new Date(job.created_at).toLocaleTimeString()}</span>
                                      </div>
                                    )}
                                    {job.meta?.speech_detected_at && (
                                      <div className="flex justify-between">
                                        <span>Speech Detected:</span>
                                        <span className="text-green-700 font-medium">{new Date(job.meta.speech_detected_at).toLocaleString()}</span>
                                      </div>
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Active and Completed Conversations Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Active Conversations - Grouped by conversation_id */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-3">Active Conversations</h4>
                {(() => {
                  // Group all jobs by conversation_id with deduplication
                  const allJobsRaw = Object.values(sessionJobs).flat().filter(job => job != null);

                  // Deduplicate by job_id
                  const jobMap = new Map();
                  allJobsRaw.forEach((job: any) => {
                    if (job && job.job_id) {
                      jobMap.set(job.job_id, job);
                    }
                  });
                  const allJobs = Array.from(jobMap.values());

                  // Group ALL jobs by conversation_id (regardless of status)
                  // Also link jobs by audio_uuid so persistence jobs get grouped with conversation
                  const allConversationJobs = new Map<string, any[]>();
                  const audioUuidToConversationId = new Map<string, string>();

                  // First pass: collect conversation_id to audio_uuid mappings
                  allJobs.forEach(job => {
                    if (!job) return;
                    const conversationId = job.meta?.conversation_id;
                    const audioUuid = job.meta?.audio_uuid;

                    if (conversationId && audioUuid) {
                      audioUuidToConversationId.set(audioUuid, conversationId);
                    }
                  });

                  // Second pass: group jobs by conversation_id or audio_uuid
                  // EXCLUDE session-level jobs (like audio persistence)
                  allJobs.forEach(job => {
                    if (!job) return;

                    // Skip session-level jobs (they run for entire session, not per conversation)
                    // Also skip audio persistence jobs by job_type (for backward compatibility with old jobs)
                    if (job.meta?.session_level === true || job.job_type === 'audio_streaming_persistence_job') {
                      return;
                    }

                    const conversationId = job.meta?.conversation_id;
                    const audioUuid = job.meta?.audio_uuid;

                    // Determine the grouping key
                    let groupKey = conversationId;
                    if (!groupKey && audioUuid) {
                      // Try to find conversation_id via audio_uuid mapping
                      groupKey = audioUuidToConversationId.get(audioUuid);
                    }

                    if (groupKey) {
                      if (!allConversationJobs.has(groupKey)) {
                        allConversationJobs.set(groupKey, []);
                      }
                      allConversationJobs.get(groupKey)!.push(job);
                    }
                  });

                  // Filter to only show conversations where at least one job is NOT completed
                  const conversationMap = new Map<string, any[]>();
                  allConversationJobs.forEach((jobs, conversationId) => {
                    const hasActiveJob = jobs.some(j => j.status !== 'completed' && j.status !== 'failed');
                    if (hasActiveJob) {
                      conversationMap.set(conversationId, jobs);
                    }
                  });

                  if (conversationMap.size === 0) {
                    return (
                      <div className="text-center py-8 text-gray-500 text-sm bg-gray-50 rounded-lg border border-gray-200">
                        No active conversations
                      </div>
                    );
                  }

                  return (
                    <div className="space-y-2">
                      {Array.from(conversationMap.entries()).map(([conversationId, jobs]) => {
                        const isExpanded = expandedSessions.has(conversationId);

                        // Find the open_conversation_job for metadata
                        const openConvJob = jobs.find(j => j.job_type === 'open_conversation_job');
                        const meta = openConvJob?.meta || {};

                        // Extract conversation info
                        const clientId = meta.client_id || 'Unknown';
                        const transcript = meta.transcript || '';
                        const speakers = meta.speakers || [];
                        const wordCount = meta.word_count || 0;
                        const lastUpdate = meta.last_update || '';
                        const createdAt = openConvJob?.created_at || null;

                        return (
                          <div key={conversationId} className="bg-cyan-50 rounded-lg border border-cyan-200 overflow-hidden">
                            <div
                              className="flex items-center justify-between p-3 cursor-pointer hover:bg-cyan-100 transition-colors"
                              onClick={() => toggleSessionExpansion(conversationId)}
                            >
                              <div className="flex-1">
                                <div className="flex items-center space-x-2">
                                  {isExpanded ? (
                                    <ChevronDown className="w-4 h-4 text-cyan-600" />
                                  ) : (
                                    <ChevronRight className="w-4 h-4 text-cyan-600" />
                                  )}
                                  <Brain className="w-4 h-4 text-cyan-600 animate-pulse" />
                                  <span className="text-sm font-medium text-gray-900">{clientId}</span>
                                  <span className="text-xs px-2 py-0.5 bg-cyan-100 text-cyan-700 rounded">Active</span>
                                  {speakers.length > 0 && (
                                    <span className="text-xs px-2 py-0.5 bg-purple-100 text-purple-700 rounded">
                                      {speakers.length} speaker{speakers.length > 1 ? 's' : ''}
                                    </span>
                                  )}
                                </div>
                                <div className="mt-1 text-xs text-gray-600">
                                  Conversation: {conversationId.substring(0, 8)}... •
                                  {createdAt && `Started: ${new Date(createdAt).toLocaleTimeString()} • `}
                                  Words: {wordCount}
                                  {lastUpdate && ` • Updated: ${new Date(lastUpdate).toLocaleTimeString()}`}
                                </div>
                                {transcript && (
                                  <div className="mt-1 text-xs text-gray-700 italic truncate">
                                    "{transcript.substring(0, 100)}{transcript.length > 100 ? '...' : ''}"
                                  </div>
                                )}
                              </div>
                            </div>

                          {/* Expanded Jobs Section */}
                          {isExpanded && (
                            <div className="border-t border-cyan-200 bg-white p-3">
                              {/* Pipeline Timeline */}
                              <div className="mb-4">
                                <h5 className="text-xs font-medium text-gray-700 mb-3">Pipeline Timeline:</h5>
                                {(() => {
                                  // Helper function to get display name from job type
                                  const getJobDisplayName = (jobType: string) => {
                                    const nameMap: { [key: string]: string } = {
                                      'stream_speech_detection_job': 'Speech',
                                      'open_conversation_job': 'Open',
                                      'transcribe_full_audio_job': 'Transcript',
                                      'recognise_speakers_job': 'Speakers',
                                      'process_memory_job': 'Memory'
                                    };
                                    return nameMap[jobType] || jobType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                  };

                                  // Helper function to get icon for job type
                                  const getJobIcon = (jobType: string) => {
                                    if (jobType.includes('speech') || jobType.includes('detect')) return Brain;
                                    if (jobType.includes('conversation') || jobType.includes('open')) return Brain;
                                    if (jobType.includes('transcribe')) return FileText;
                                    if (jobType.includes('speaker') || jobType.includes('recognise')) return Brain;
                                    if (jobType.includes('memory')) return Brain;
                                    return Brain; // Default icon
                                  };

                                  // Build dynamic pipeline from actual jobs with timing data
                                  // Sort by start time to show chronological order
                                  const jobsWithTiming = jobs
                                    .filter(j => j && j.started_at)
                                    .map(job => {
                                      const startTime = new Date(job.started_at!).getTime();
                                      const endTime = job.completed_at || job.ended_at
                                        ? new Date((job.completed_at || job.ended_at)!).getTime()
                                        : (job.status === 'processing' ? Date.now() : startTime);

                                      return {
                                        job,
                                        startTime,
                                        endTime,
                                        duration: (endTime - startTime) / 1000,
                                        name: getJobDisplayName(job.job_type),
                                        icon: getJobIcon(job.job_type)
                                      };
                                    })
                                    .sort((a, b) => a.startTime - b.startTime);

                                  const jobTimes = jobsWithTiming;

                                  // Find earliest start and latest end
                                  const validTimes = jobTimes.filter(t => t !== null);
                                  if (validTimes.length === 0) {
                                    return (
                                      <div className="text-xs text-gray-500 italic">No job timing data available</div>
                                    );
                                  }

                                  const earliestStart = Math.min(...validTimes.map(t => t!.startTime));
                                  const latestEnd = Math.max(...validTimes.map(t => t!.endTime));
                                  const totalDuration = (latestEnd - earliestStart) / 1000; // in seconds

                                  // Format duration for display
                                  const formatDuration = (seconds: number) => {
                                    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
                                    if (seconds < 60) return `${seconds.toFixed(1)}s`;
                                    const mins = Math.floor(seconds / 60);
                                    const secs = Math.floor(seconds % 60);
                                    return `${mins}m ${secs}s`;
                                  };

                                  // Generate time axis markers (0%, 25%, 50%, 75%, 100%)
                                  const timeMarkers = [0, 0.25, 0.5, 0.75, 1].map(pct => ({
                                    percent: pct * 100,
                                    time: formatDuration(totalDuration * pct)
                                  }));

                                  return (
                                    <div className="space-y-2">
                                      {/* Time axis */}
                                      <div className="relative h-4 border-b border-gray-300">
                                        {timeMarkers.map((marker, idx) => (
                                          <div
                                            key={idx}
                                            className="absolute"
                                            style={{ left: `${marker.percent}%`, transform: 'translateX(-50%)' }}
                                          >
                                            <div className="w-px h-2 bg-gray-400"></div>
                                            <div className="text-xs text-gray-500 mt-0.5 whitespace-nowrap">
                                              {marker.time}
                                            </div>
                                          </div>
                                        ))}
                                      </div>

                                      {/* Job timeline bars */}
                                      <div className="space-y-2 mt-6">
                                        {jobTimes.map((jobTime) => {
                                          const { job, startTime, endTime, duration, name, icon: Icon } = jobTime;

                                          // Calculate position and width as percentage of total timeline
                                          const startPercent = ((startTime - earliestStart) / (latestEnd - earliestStart)) * 100;
                                          const widthPercent = ((endTime - startTime) / (latestEnd - earliestStart)) * 100;

                                          // Use job type colors
                                          const jobColors = getJobTypeColor(job.job_type, job.status);
                                          const barColor = jobColors.bgColor;
                                          const borderColor = jobColors.borderColor;

                                          return (
                                            <div key={job.job_id} className="flex items-center space-x-2 h-8">
                                              {/* Stage Icon */}
                                              <div className={`w-8 h-8 rounded-full border-2 ${borderColor} ${barColor} flex items-center justify-center flex-shrink-0`}>
                                                <Icon className="w-4 h-4 text-white" />
                                              </div>

                                              {/* Stage Name */}
                                              <span className="text-xs text-gray-700 w-20 flex-shrink-0">{name}</span>

                                              {/* Timeline Container */}
                                              <div className="flex-1 relative h-6 bg-gray-100 rounded">
                                                {/* Job Bar */}
                                                <div
                                                  className={`absolute h-6 rounded ${barColor} ${job.status === 'processing' ? 'animate-pulse' : ''} flex items-center justify-center`}
                                                  style={{
                                                    left: `${startPercent}%`,
                                                    width: `${widthPercent}%`
                                                  }}
                                                  title={`Started: ${new Date(startTime).toLocaleTimeString()}\nDuration: ${formatDuration(duration)}`}
                                                >
                                                  <span className="text-xs text-white font-medium px-2 truncate">
                                                    {formatDuration(duration)}
                                                  </span>
                                                </div>
                                              </div>
                                            </div>
                                          );
                                        })}
                                      </div>

                                      {/* Total Duration */}
                                      <div className="text-xs text-gray-600 text-right mt-2">
                                        Total: {formatDuration(totalDuration)}
                                      </div>
                                    </div>
                                  );
                                })()}
                              </div>

                              <h5 className="text-xs font-medium text-gray-700 mb-2">Conversation Jobs:</h5>
                              {jobs.filter(j => j != null && j.job_id).length > 0 ? (
                                <div className="space-y-1">
                                  {jobs
                                    .filter(j => j != null && j.job_id)
                                    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
                                    .map((job, index) => (
                                    <div key={job.job_id} className={`p-2 bg-gray-50 rounded border ${getJobTypeColor(job.job_type, job.status).borderColor}`} style={{ borderLeftWidth: '12px' }}>
                                      <div
                                        className="flex items-center justify-between cursor-pointer hover:bg-gray-100 transition-colors rounded px-1 py-0.5"
                                        onClick={() => toggleJobExpansion(job.job_id)}
                                      >
                                        <div className="flex-1 min-w-0">
                                          <div className="flex items-center space-x-2">
                                            <span className="text-xs font-mono text-gray-500 flex-shrink-0">#{index + 1}</span>
                                            {expandedJobs.has(job.job_id) ? (
                                              <ChevronDown className="w-3 h-3 text-gray-500" />
                                            ) : (
                                              <ChevronRight className="w-3 h-3 text-gray-500" />
                                            )}
                                            <span className="flex-shrink-0">{getJobTypeIcon(job.job_type)}</span>
                                            <span className="flex-shrink-0">{getStatusIcon(job.status)}</span>
                                            <span className="text-xs font-medium text-gray-900 truncate">{job.job_type}</span>
                                            <span className={`text-xs px-1.5 py-0.5 rounded ${getStatusColor(job.status)}`}>
                                              {job.status}
                                            </span>
                                            <span className="text-xs text-gray-500">{job.queue}</span>
                                            {/* Show memory count badge on collapsed card */}
                                            {!expandedJobs.has(job.job_id) && job.job_type === 'process_memory_job' && job.result?.memories_created !== undefined && (
                                              <span className="text-xs px-1.5 py-0.5 bg-pink-100 text-pink-700 rounded">
                                                {job.result.memories_created} memories
                                              </span>
                                            )}
                                          </div>
                                        </div>
                                      </div>

                                      {/* Collapsible metadata section */}
                                      {expandedJobs.has(job.job_id) && (
                                        <div className="mt-1 text-xs text-gray-600 space-y-0.5">
                                          <div>
                                            {job.started_at && (
                                              <span>Started: {new Date(job.started_at).toLocaleTimeString()}</span>
                                            )}
                                            {job.started_at && (
                                              <span> • Duration: {formatDuration(job)}</span>
                                            )}
                                          </div>

                                          {/* Show job-specific metadata */}
                                          {job.meta && (
                                            <div className="space-y-0.5 pl-2 border-l-2 border-gray-300">
                                              {/* open_conversation_job metadata */}
                                              {job.job_type === 'open_conversation_job' && (
                                                <>
                                                  {job.meta.word_count !== undefined && (
                                                    <div>Words: <span className="font-medium">{job.meta.word_count}</span></div>
                                                  )}
                                                  {job.meta.speakers && job.meta.speakers.length > 0 && (
                                                    <div>Speakers: <span className="font-medium">{job.meta.speakers.join(', ')}</span></div>
                                                  )}
                                                  {job.meta.inactivity_seconds !== undefined && (
                                                    <div>Idle: <span className="font-medium">{Math.floor(job.meta.inactivity_seconds)}s</span></div>
                                                  )}
                                                  {job.meta.transcript && (
                                                    <div className="italic text-gray-500 truncate max-w-md">
                                                      "{job.meta.transcript.substring(0, 80)}..."
                                                    </div>
                                                  )}
                                                </>
                                              )}

                                              {/* transcribe_full_audio_job metadata */}
                                              {job.job_type === 'transcribe_full_audio_job' && job.result && (
                                                <>
                                                  {job.result.transcript && (
                                                    <div>Transcript: <span className="font-medium">{job.result.transcript.length} chars</span></div>
                                                  )}
                                                  {job.result.processing_time_seconds && (
                                                    <div>Processing: <span className="font-medium">{job.result.processing_time_seconds.toFixed(1)}s</span></div>
                                                  )}
                                                </>
                                              )}

                                              {/* recognise_speakers_job metadata */}
                                              {job.job_type === 'recognise_speakers_job' && job.result && (
                                                <>
                                                  {job.result.identified_speakers && job.result.identified_speakers.length > 0 && (
                                                    <div>Identified: <span className="font-medium">{job.result.identified_speakers.join(', ')}</span></div>
                                                  )}
                                                  {job.result.segment_count && (
                                                    <div>Segments: <span className="font-medium">{job.result.segment_count}</span></div>
                                                  )}
                                                </>
                                              )}

                                              {/* process_memory_job metadata */}
                                              {job.job_type === 'process_memory_job' && job.meta && (
                                                <>
                                                  {job.meta.memories_created !== undefined && (
                                                    <div>Memories: <span className="font-medium">{job.meta.memories_created} created</span></div>
                                                  )}
                                                  {job.meta.processing_time && (
                                                    <div>Processing: <span className="font-medium">{job.meta.processing_time.toFixed(1)}s</span></div>
                                                  )}
                                                  {job.meta.memory_details && job.meta.memory_details.length > 0 && (
                                                    <div className="mt-2">
                                                      <div className="text-xs font-medium text-gray-700 mb-1">Memories Created:</div>
                                                      {job.meta.memory_details.map((memory: any, idx: number) => (
                                                        <div key={idx} className="text-xs bg-pink-50 p-2 rounded mb-1">
                                                          "{memory.text}"
                                                        </div>
                                                      ))}
                                                    </div>
                                                  )}
                                                </>
                                              )}

                                              {/* Show conversation_id if present */}
                                              {job.meta.conversation_id && (
                                                <div className="font-mono text-gray-500">
                                                  Conv: {job.meta.conversation_id.substring(0, 8)}...
                                                </div>
                                              )}
                                            </div>
                                          )}
                                        </div>
                                      )}
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
                                <div className="text-xs text-gray-500 italic">No jobs found for this conversation</div>
                              )}
                            </div>
                          )}
                        </div>
                        );
                      })}
                    </div>
                  );
                })()}
              </div>

              {/* Completed Conversations - Grouped by conversation_id */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-3">Completed Conversations (Last Hour)</h4>
                {(() => {
                  // Group all jobs by conversation_id for completed conversations with deduplication
                  const allJobsRaw = Object.values(sessionJobs).flat().filter(job => job != null);

                  // Deduplicate by job_id
                  const jobMap = new Map();
                  allJobsRaw.forEach((job: any) => {
                    if (job && job.job_id) {
                      jobMap.set(job.job_id, job);
                    }
                  });
                  const allJobs = Array.from(jobMap.values());

                  // Group ALL jobs by conversation_id (regardless of status)
                  // Also link jobs by audio_uuid so persistence jobs get grouped with conversation
                  const allConversationJobs = new Map<string, any[]>();
                  const audioUuidToConversationId = new Map<string, string>();

                  // First pass: collect conversation_id to audio_uuid mappings
                  allJobs.forEach(job => {
                    if (!job) return;
                    const conversationId = job.meta?.conversation_id;
                    const audioUuid = job.meta?.audio_uuid;

                    if (conversationId && audioUuid) {
                      audioUuidToConversationId.set(audioUuid, conversationId);
                    }
                  });

                  // Second pass: group jobs by conversation_id or audio_uuid
                  // EXCLUDE session-level jobs (like audio persistence)
                  allJobs.forEach(job => {
                    if (!job) return;

                    // Skip session-level jobs (they run for entire session, not per conversation)
                    // Also skip audio persistence jobs by job_type (for backward compatibility with old jobs)
                    if (job.meta?.session_level === true || job.job_type === 'audio_streaming_persistence_job') {
                      return;
                    }

                    const conversationId = job.meta?.conversation_id;
                    const audioUuid = job.meta?.audio_uuid;

                    // Determine the grouping key
                    let groupKey = conversationId;
                    if (!groupKey && audioUuid) {
                      // Try to find conversation_id via audio_uuid mapping
                      groupKey = audioUuidToConversationId.get(audioUuid);
                    }

                    if (groupKey) {
                      if (!allConversationJobs.has(groupKey)) {
                        allConversationJobs.set(groupKey, []);
                      }
                      allConversationJobs.get(groupKey)!.push(job);
                    }
                  });

                  // Filter to only show conversations where ALL jobs are completed or failed
                  const conversationMap = new Map<string, any[]>();
                  allConversationJobs.forEach((jobs, conversationId) => {
                    const allJobsComplete = jobs.every(j => j.status === 'completed' || j.status === 'failed');
                    if (allJobsComplete) {
                      conversationMap.set(conversationId, jobs);
                    }
                  });

                  if (conversationMap.size === 0) {
                    return (
                      <div className="text-center py-8 text-gray-500 text-sm bg-gray-50 rounded-lg border border-gray-200">
                        No completed conversations
                      </div>
                    );
                  }

                  return (
                    <div className="space-y-2">
                      {Array.from(conversationMap.entries()).map(([conversationId, jobs]) => {
                        const isExpanded = expandedSessions.has(conversationId);

                        // Find the open_conversation_job for metadata
                        const openConvJob = jobs.find(j => j.job_type === 'open_conversation_job');
                        const meta = openConvJob?.meta || {};

                        // Find transcription job for title/summary
                        const transcriptionJob = jobs.find(j => j.job_type === 'transcribe_full_audio_job');
                        const transcriptionMeta = transcriptionJob?.meta || {};

                        // Extract conversation info from metadata
                        const clientId = meta.client_id || 'Unknown';
                        const transcript = meta.transcript || '';
                        const speakers = meta.speakers || [];
                        const wordCount = meta.word_count || 0;
                        const createdAt = openConvJob?.created_at || null;
                        const title = transcriptionMeta.title || null;
                        const summary = transcriptionMeta.summary || null;

                        // Check if all jobs are complete
                        const allComplete = jobs.every(j => j.status === 'completed');

                        return (
                          <div key={conversationId} className={`rounded-lg border overflow-hidden ${
                            allComplete
                              ? 'bg-green-50 border-green-200'
                              : 'bg-yellow-50 border-yellow-200'
                          }`}>
                            <div
                              className={`flex items-center justify-between p-3 cursor-pointer transition-colors ${
                                allComplete ? 'hover:bg-green-100' : 'hover:bg-yellow-100'
                              }`}
                              onClick={() => toggleSessionExpansion(conversationId)}
                            >
                              <div className="flex-1">
                                <div className="flex items-center space-x-2">
                                  {isExpanded ? (
                                    <ChevronDown className={`w-4 h-4 ${allComplete ? 'text-green-600' : 'text-yellow-600'}`} />
                                  ) : (
                                    <ChevronRight className={`w-4 h-4 ${allComplete ? 'text-green-600' : 'text-yellow-600'}`} />
                                  )}
                                  {allComplete ? (
                                    <CheckCircle className="w-4 h-4 text-green-600" />
                                  ) : (
                                    <Clock className="w-4 h-4 text-yellow-600" />
                                  )}
                                  <span className="text-sm font-medium text-gray-900">{clientId}</span>
                                  <span className={`text-xs px-2 py-0.5 rounded ${
                                    allComplete ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
                                  }`}>
                                    {allComplete ? 'Complete' : 'Processing'}
                                  </span>
                                  {speakers.length > 0 && (
                                    <span className="text-xs px-2 py-0.5 bg-purple-100 text-purple-700 rounded">
                                      {speakers.length} speaker{speakers.length > 1 ? 's' : ''}
                                    </span>
                                  )}
                                </div>
                                <div className="mt-1 text-xs text-gray-600">
                                  Conversation: {conversationId.substring(0, 8)}... •
                                  Words: {wordCount}
                                  {createdAt && (
                                    <> • Created: {new Date(createdAt).toLocaleString()}</>
                                  )}
                                </div>
                                {/* Show title/summary for completed, or transcript for in-progress or when no title exists */}
                                {allComplete ? (
                                  <>
                                    {title ? (
                                      <div className="mt-1 text-sm font-medium text-gray-900">
                                        {title}
                                      </div>
                                    ) : transcript ? (
                                      <div className="mt-1 text-xs text-gray-700 italic truncate">
                                        "{transcript.substring(0, 100)}{transcript.length > 100 ? '...' : ''}"
                                      </div>
                                    ) : null}
                                    {summary && (
                                      <div className="mt-1 text-xs text-gray-700 italic">
                                        {summary}
                                      </div>
                                    )}
                                  </>
                                ) : (
                                  transcript && (
                                    <div className="mt-1 text-xs text-gray-700 italic truncate">
                                      "{transcript.substring(0, 100)}{transcript.length > 100 ? '...' : ''}"
                                    </div>
                                  )
                                )}
                              </div>
                            </div>

                            {/* Expanded Jobs Section */}
                            {isExpanded && (
                              <div className={`border-t bg-white p-3 ${
                                allComplete ? 'border-green-200' : 'border-yellow-200'
                              }`}>
                                {/* Pipeline Timeline */}
                                <div className="mb-4">
                                  <h5 className="text-xs font-medium text-gray-700 mb-3">Pipeline Timeline:</h5>
                                  {(() => {
                                    // Helper function to get display name from job type
                                    const getJobDisplayName = (jobType: string) => {
                                      const nameMap: { [key: string]: string } = {
                                        'stream_speech_detection_job': 'Speech',
                                        'open_conversation_job': 'Open',
                                        'transcribe_full_audio_job': 'Transcript',
                                        'recognise_speakers_job': 'Speakers',
                                        'process_memory_job': 'Memory'
                                      };
                                      return nameMap[jobType] || jobType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                    };

                                    // Helper function to get icon for job type
                                    const getJobIcon = (jobType: string) => {
                                      if (jobType.includes('speech') || jobType.includes('detect')) return Brain;
                                      if (jobType.includes('conversation') || jobType.includes('open')) return Brain;
                                      if (jobType.includes('transcribe')) return FileText;
                                      if (jobType.includes('speaker') || jobType.includes('recognise')) return Brain;
                                      if (jobType.includes('memory')) return Brain;
                                      return Brain; // Default icon
                                    };

                                    // Build dynamic pipeline from actual jobs with timing data
                                    // Sort by start time to show chronological order
                                    const jobsWithTiming = jobs
                                      .filter(j => j && j.started_at)
                                      .map(job => {
                                        const startTime = new Date(job.started_at!).getTime();
                                        const endTime = job.completed_at || job.ended_at
                                          ? new Date((job.completed_at || job.ended_at)!).getTime()
                                          : (job.status === 'processing' ? Date.now() : startTime);

                                        return {
                                          job,
                                          startTime,
                                          endTime,
                                          duration: (endTime - startTime) / 1000,
                                          name: getJobDisplayName(job.job_type),
                                          icon: getJobIcon(job.job_type)
                                        };
                                      })
                                      .sort((a, b) => a.startTime - b.startTime);

                                    const jobTimes = jobsWithTiming;

                                    // Find earliest start and latest end
                                    const validTimes = jobTimes.filter(t => t !== null);
                                    if (validTimes.length === 0) {
                                      return (
                                        <div className="text-xs text-gray-500 italic">No job timing data available</div>
                                      );
                                    }

                                    const earliestStart = Math.min(...validTimes.map(t => t!.startTime));
                                    const latestEnd = Math.max(...validTimes.map(t => t!.endTime));
                                    const totalDuration = (latestEnd - earliestStart) / 1000; // in seconds

                                    // Format duration for display
                                    const formatDuration = (seconds: number) => {
                                      if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
                                      if (seconds < 60) return `${seconds.toFixed(1)}s`;
                                      const mins = Math.floor(seconds / 60);
                                      const secs = Math.floor(seconds % 60);
                                      return `${mins}m ${secs}s`;
                                    };

                                    // Generate time axis markers (0%, 25%, 50%, 75%, 100%)
                                    const timeMarkers = [0, 0.25, 0.5, 0.75, 1].map(pct => ({
                                      percent: pct * 100,
                                      time: formatDuration(totalDuration * pct)
                                    }));

                                    return (
                                      <div className="space-y-2">
                                        {/* Time axis */}
                                        <div className="relative h-4 border-b border-gray-300">
                                          {timeMarkers.map((marker, idx) => (
                                            <div
                                              key={idx}
                                              className="absolute"
                                              style={{ left: `${marker.percent}%`, transform: 'translateX(-50%)' }}
                                            >
                                              <div className="w-px h-2 bg-gray-400"></div>
                                              <div className="text-xs text-gray-500 mt-0.5 whitespace-nowrap">
                                                {marker.time}
                                              </div>
                                            </div>
                                          ))}
                                        </div>

                                        {/* Job timeline bars */}
                                        <div className="space-y-2 mt-6">
                                          {jobTimes.map((jobTime) => {
                                            const { job, startTime, endTime, duration, name, icon: Icon } = jobTime;

                                            // Calculate position and width as percentage of total timeline
                                            const startPercent = ((startTime - earliestStart) / (latestEnd - earliestStart)) * 100;
                                            const widthPercent = ((endTime - startTime) / (latestEnd - earliestStart)) * 100;

                                            // Use job type colors
                                            const jobColors = getJobTypeColor(job.job_type, job.status);
                                            const barColor = jobColors.bgColor;
                                            const borderColor = jobColors.borderColor;

                                            return (
                                              <div key={job.job_id} className="flex items-center space-x-2 h-8">
                                                {/* Stage Icon */}
                                                <div className={`w-8 h-8 rounded-full border-2 ${borderColor} ${barColor} flex items-center justify-center flex-shrink-0`}>
                                                  <Icon className="w-4 h-4 text-white" />
                                                </div>

                                                {/* Stage Name */}
                                                <span className="text-xs text-gray-700 w-20 flex-shrink-0">{name}</span>

                                                {/* Timeline Container */}
                                                <div className="flex-1 relative h-6 bg-gray-100 rounded">
                                                  {/* Job Bar */}
                                                  <div
                                                    className={`absolute h-6 rounded ${barColor} ${job.status === 'processing' ? 'animate-pulse' : ''} flex items-center justify-center`}
                                                    style={{
                                                      left: `${startPercent}%`,
                                                      width: `${widthPercent}%`
                                                    }}
                                                    title={`Started: ${new Date(startTime).toLocaleTimeString()}\nDuration: ${formatDuration(duration)}`}
                                                  >
                                                    <span className="text-xs text-white font-medium px-2 truncate">
                                                      {formatDuration(duration)}
                                                    </span>
                                                  </div>
                                                </div>
                                              </div>
                                            );
                                          })}
                                        </div>

                                        {/* Total Duration */}
                                        <div className="text-xs text-gray-600 text-right mt-2">
                                          Total: {formatDuration(totalDuration)}
                                        </div>
                                      </div>
                                    );
                                  })()}
                                </div>

                                <h5 className="text-xs font-medium text-gray-700 mb-2">Conversation Jobs:</h5>
                                {jobs.filter(j => j != null && j.job_id).length > 0 ? (
                                  <div className="space-y-1">
                                    {jobs
                                      .filter(j => j != null && j.job_id)
                                      .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
                                      .map((job, index) => (
                                      <div key={job.job_id} className={`p-2 bg-gray-50 rounded border ${getJobTypeColor(job.job_type, job.status).borderColor}`} style={{ borderLeftWidth: '12px' }}>
                                        <div className="flex items-center justify-between">
                                          <div
                                            className="flex-1 flex items-center space-x-2 cursor-pointer hover:bg-gray-100 transition-colors rounded px-1 py-0.5"
                                            onClick={() => toggleJobExpansion(job.job_id)}
                                          >
                                            {expandedJobs.has(job.job_id) ? (
                                              <ChevronDown className="w-3 h-3 text-gray-500 flex-shrink-0" />
                                            ) : (
                                              <ChevronRight className="w-3 h-3 text-gray-500 flex-shrink-0" />
                                            )}
                                            <span className="text-xs font-mono text-gray-500 flex-shrink-0">#{index + 1}</span>
                                            <span className="flex-shrink-0">{getJobTypeIcon(job.job_type)}</span>
                                            <span className="flex-shrink-0">{getStatusIcon(job.status)}</span>
                                            <span className="text-xs font-medium text-gray-900 truncate">{job.job_type}</span>
                                            <span className={`text-xs px-1.5 py-0.5 rounded ${getStatusColor(job.status)}`}>
                                              {job.status}
                                            </span>
                                            <span className="text-xs text-gray-500">{job.queue || job.data?.queue || 'unknown'}</span>
                                            {/* Show memory count badge on collapsed card */}
                                            {!expandedJobs.has(job.job_id) && job.job_type === 'process_memory_job' && job.result?.memories_created !== undefined && (
                                              <span className="text-xs px-1.5 py-0.5 bg-pink-100 text-pink-700 rounded">
                                                {job.result.memories_created} memories
                                              </span>
                                            )}
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

                                        {/* Collapsible metadata section */}
                                        {expandedJobs.has(job.job_id) && (
                                          <div className="mt-1 text-xs text-gray-600 space-y-0.5 pl-4">
                                            <div>
                                              {job.started_at && (
                                                <span>Started: {new Date(job.started_at).toLocaleTimeString()}</span>
                                              )}
                                              {job.started_at && (
                                                <span> • Duration: {formatDuration(job)}</span>
                                              )}
                                            </div>

                                            {/* Show job-specific metadata */}
                                            {job.meta && (
                                              <div className="space-y-0.5 pl-2 border-l-2 border-gray-300">
                                                {/* open_conversation_job metadata */}
                                                {job.job_type === 'open_conversation_job' && (
                                                  <>
                                                    {job.meta.word_count !== undefined && (
                                                      <div>Words: <span className="font-medium">{job.meta.word_count}</span></div>
                                                    )}
                                                    {job.meta.speakers && job.meta.speakers.length > 0 && (
                                                      <div>Speakers: <span className="font-medium">{job.meta.speakers.join(', ')}</span></div>
                                                    )}
                                                    {job.meta.inactivity_seconds !== undefined && (
                                                      <div>Idle: <span className="font-medium">{Math.floor(job.meta.inactivity_seconds)}s</span></div>
                                                    )}
                                                    {job.meta.transcript && (
                                                      <div className="italic text-gray-500 truncate max-w-md">
                                                        "{job.meta.transcript.substring(0, 80)}..."
                                                      </div>
                                                    )}
                                                  </>
                                                )}

                                                {/* transcribe_full_audio_job metadata */}
                                                {job.job_type === 'transcribe_full_audio_job' && job.result && (
                                                  <>
                                                    {job.result.transcript && (
                                                      <div>Transcript: <span className="font-medium">{job.result.transcript.length} chars</span></div>
                                                    )}
                                                    {job.result.processing_time_seconds && (
                                                      <div>Processing: <span className="font-medium">{job.result.processing_time_seconds.toFixed(1)}s</span></div>
                                                    )}
                                                  </>
                                                )}

                                                {/* recognise_speakers_job metadata */}
                                                {job.job_type === 'recognise_speakers_job' && job.result && (
                                                  <>
                                                    {job.result.identified_speakers && job.result.identified_speakers.length > 0 && (
                                                      <div>Identified: <span className="font-medium">{job.result.identified_speakers.join(', ')}</span></div>
                                                    )}
                                                    {job.result.segment_count && (
                                                      <div>Segments: <span className="font-medium">{job.result.segment_count}</span></div>
                                                    )}
                                                  </>
                                                )}

                                                {/* process_memory_job metadata */}
                                                {job.job_type === 'process_memory_job' && job.result && (
                                                  <>
                                                    {job.result.memories_created !== undefined && (
                                                      <div>Memories: <span className="font-medium">{job.result.memories_created} created</span></div>
                                                    )}
                                                    {job.result.processing_time_seconds && (
                                                      <div>Processing: <span className="font-medium">{job.result.processing_time_seconds.toFixed(1)}s</span></div>
                                                    )}
                                                  </>
                                                )}

                                                {/* Show conversation_id if present */}
                                                {job.meta.conversation_id && (
                                                  <div className="font-mono text-gray-500">
                                                    Conv: {job.meta.conversation_id.substring(0, 8)}...
                                                  </div>
                                                )}
                                              </div>
                                            )}
                                          </div>
                                        )}
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <div className="text-xs text-gray-500 italic">No jobs found for this conversation</div>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  );
                })()}
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

      {/* Old Jobs Table and Pagination - Removed in favor of session-based view above */}

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

                {/* Formatted Job Metadata - Job-specific displays */}
                {selectedJob.meta && Object.keys(selectedJob.meta).length > 0 && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Job Metadata</label>

                    {/* open_conversation_job formatted metadata */}
                    {selectedJob.func_name?.includes('open_conversation_job') && (
                      <div className="bg-blue-50 p-3 rounded mb-3 space-y-2">
                        {selectedJob.meta.word_count !== undefined && (
                          <div className="text-sm">
                            <span className="font-medium">Word Count:</span> {selectedJob.meta.word_count}
                          </div>
                        )}
                        {selectedJob.meta.speakers && selectedJob.meta.speakers.length > 0 && (
                          <div className="text-sm">
                            <span className="font-medium">Speakers:</span> {selectedJob.meta.speakers.join(', ')}
                          </div>
                        )}
                        {selectedJob.meta.transcript_length !== undefined && (
                          <div className="text-sm">
                            <span className="font-medium">Transcript Length:</span> {selectedJob.meta.transcript_length} chars
                          </div>
                        )}
                        {selectedJob.meta.duration_seconds !== undefined && (
                          <div className="text-sm">
                            <span className="font-medium">Duration:</span> {selectedJob.meta.duration_seconds.toFixed(1)}s
                          </div>
                        )}
                        {selectedJob.meta.inactivity_seconds !== undefined && (
                          <div className="text-sm">
                            <span className="font-medium">Idle Time:</span> {Math.floor(selectedJob.meta.inactivity_seconds)}s
                          </div>
                        )}
                        {selectedJob.meta.chunks_processed !== undefined && (
                          <div className="text-sm">
                            <span className="font-medium">Chunks Processed:</span> {selectedJob.meta.chunks_processed}
                          </div>
                        )}
                        {selectedJob.meta.transcript && (
                          <div className="mt-2">
                            <div className="text-sm font-medium mb-1">Transcript:</div>
                            <div className="text-sm italic text-gray-700 bg-white p-2 rounded border border-gray-200 max-h-32 overflow-y-auto">
                              "{selectedJob.meta.transcript}"
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* process_memory_job formatted metadata */}
                    {selectedJob.func_name?.includes('process_memory_job') && selectedJob.meta.memory_details && selectedJob.meta.memory_details.length > 0 && (
                      <div className="bg-pink-50 p-3 rounded mb-3 space-y-2">
                        <div className="text-sm">
                          <span className="font-medium">Memories Created:</span> {selectedJob.meta.memories_created || selectedJob.meta.memory_details.length}
                        </div>
                        {selectedJob.meta.processing_time !== undefined && (
                          <div className="text-sm">
                            <span className="font-medium">Processing Time:</span> {selectedJob.meta.processing_time.toFixed(1)}s
                          </div>
                        )}
                        <div className="mt-2">
                          <div className="text-sm font-medium mb-1">Memory Details:</div>
                          <div className="space-y-1">
                            {selectedJob.meta.memory_details.map((mem: any, idx: number) => (
                              <div key={idx} className="text-xs bg-pink-100 p-2 rounded border border-pink-200">
                                {mem.text}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* stream_speech_detection_job formatted metadata */}
                    {selectedJob.func_name?.includes('stream_speech_detection_job') && (
                      <div className="bg-yellow-50 p-3 rounded mb-3 space-y-2">
                        {selectedJob.meta.speech_detected_at && (
                          <div className="text-sm">
                            <span className="font-medium">Speech Detected At:</span> {new Date(selectedJob.meta.speech_detected_at).toLocaleString()}
                          </div>
                        )}
                        {selectedJob.meta.detected_speakers && selectedJob.meta.detected_speakers.length > 0 && (
                          <div className="text-sm">
                            <span className="font-medium">Detected Speakers:</span> {selectedJob.meta.detected_speakers.join(', ')}
                          </div>
                        )}
                        {selectedJob.meta.conversation_job_id && (
                          <div className="text-sm">
                            <span className="font-medium">Conversation Job:</span> {selectedJob.meta.conversation_job_id}
                          </div>
                        )}
                      </div>
                    )}

                    {/* transcribe_full_audio_job formatted metadata */}
                    {selectedJob.func_name?.includes('transcribe_full_audio_job') && (selectedJob.meta.title || selectedJob.meta.summary) && (
                      <div className="bg-purple-50 p-3 rounded mb-3 space-y-2">
                        {selectedJob.meta.title && (
                          <div className="text-sm">
                            <span className="font-medium">Title:</span> {selectedJob.meta.title}
                          </div>
                        )}
                        {selectedJob.meta.summary && (
                          <div className="text-sm">
                            <span className="font-medium">Summary:</span> {selectedJob.meta.summary}
                          </div>
                        )}
                        {selectedJob.meta.transcript_length !== undefined && (
                          <div className="text-sm">
                            <span className="font-medium">Transcript Length:</span> {selectedJob.meta.transcript_length} chars
                          </div>
                        )}
                        {selectedJob.meta.word_count !== undefined && (
                          <div className="text-sm">
                            <span className="font-medium">Word Count:</span> {selectedJob.meta.word_count}
                          </div>
                        )}
                        {selectedJob.meta.processing_time !== undefined && (
                          <div className="text-sm">
                            <span className="font-medium">Processing Time:</span> {selectedJob.meta.processing_time.toFixed(1)}s
                          </div>
                        )}
                      </div>
                    )}

                    {/* process_cropping_job formatted metadata */}
                    {selectedJob.func_name?.includes('process_cropping_job') && (
                      <div className="bg-green-50 p-3 rounded mb-3 space-y-2">
                        {selectedJob.meta.cropped_duration_seconds !== undefined && (
                          <div className="text-sm">
                            <span className="font-medium">Cropped Duration:</span> {selectedJob.meta.cropped_duration_seconds.toFixed(1)}s
                          </div>
                        )}
                        {selectedJob.meta.segments_cropped !== undefined && (
                          <div className="text-sm">
                            <span className="font-medium">Segments Cropped:</span> {selectedJob.meta.segments_cropped}
                          </div>
                        )}
                      </div>
                    )}

                    {/* Raw JSON metadata (collapsible) */}
                    <details className="mt-2">
                      <summary className="text-sm font-medium text-gray-700 cursor-pointer hover:text-gray-900">
                        Raw Metadata JSON
                      </summary>
                      <pre className="text-xs text-gray-900 bg-blue-50 p-2 rounded overflow-auto max-h-64 mt-2">
                        {JSON.stringify(selectedJob.meta, null, 2)}
                      </pre>
                    </details>
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