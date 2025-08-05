import React, { useRef, useState } from 'react'
import { BarChart3, RefreshCw, Settings, Info, AlertTriangle } from 'lucide-react'

// Import Plotly dynamically to avoid SSR issues
let Plotly: any = null;
if (typeof window !== 'undefined') {
  import('plotly.js-dist-min').then((module) => {
    Plotly = module.default;
  });
}

interface AnnotationSegment {
  start: number
  end: number
  speakerLabel?: string
}

interface AnalysisData {
  visualization: {
    speakers: string[]
    embeddings_2d: number[][]
    embeddings_3d: number[][]
    cluster_labels: number[]
    colors: string[]
  }
  clustering: {
    method: string
    n_clusters: number
    silhouette_score?: number | null
    n_noise?: number
  }
  similar_speakers: Array<{
    speaker1: string
    speaker2: string
    similarity: number
  }>
  quality_metrics: {
    n_speakers: number
    mean_similarity: number | null
    separation_quality: number | null
  }
  segment_info: {
    total_segments: number
    processed_segments: number
    unique_speakers: string[]
    total_duration: number
  }
  parameters: {
    reduction_method: string
    cluster_method: string
    similarity_threshold: number
  }
}

interface AnnotationEmbeddingAnalysisProps {
  segments: AnnotationSegment[]
  audioFile: File | null
  onAnalysisComplete?: (analysis: AnalysisData) => void
}

export default function AnnotationEmbeddingAnalysis({ 
  segments, 
  audioFile, 
  onAnalysisComplete 
}: AnnotationEmbeddingAnalysisProps) {
  const plotRef = useRef<HTMLDivElement>(null)
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isVisible, setIsVisible] = useState(false)
  const [view3D, setView3D] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings] = useState({
    method: 'umap',
    clusterMethod: 'dbscan',
    similarityThreshold: 0.8
  })

  const runAnalysis = async () => {
    if (!audioFile || segments.length === 0) {
      setError('No audio file or segments available for analysis')
      return
    }

    if (!Plotly) {
      setError('Plotly visualization library not loaded')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      // Prepare segments data - filter out segments without speaker labels
      const validSegments = segments.filter(seg => seg.speakerLabel && seg.speakerLabel.trim())
      
      if (validSegments.length === 0) {
        setError('No segments with speaker labels found')
        return
      }

      // Create FormData for the API request
      const formData = new FormData()
      formData.append('audio_file', audioFile)
      
      // Add segments data as JSON
      const segmentsData = validSegments.map(seg => ({
        start: seg.start,
        end: seg.end,
        speaker_label: seg.speakerLabel
      }))
      
      // Add analysis parameters
      formData.append('method', settings.method)
      formData.append('cluster_method', settings.clusterMethod)
      formData.append('similarity_threshold', settings.similarityThreshold.toString())
      
      // Add segments as JSON
      formData.append('segments', JSON.stringify(segmentsData))

      const response = await fetch('/api/annotations/analyze-segments', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Analysis failed: ${response.status} - ${errorText}`)
      }

      const data = await response.json()

      if (data.error || data.status === 'error') {
        throw new Error(data.error || data.message || 'Analysis failed')
      }

      setAnalysisData(data)
      setIsVisible(true)
      createPlot(data)
      
      if (onAnalysisComplete) {
        onAnalysisComplete(data)
      }

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed'
      setError(errorMessage)
      console.error('Segment analysis error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const createPlot = (data: AnalysisData) => {
    if (!plotRef.current || !Plotly || !data.visualization.speakers.length) return

    const { visualization } = data
    const embeddings = view3D ? visualization.embeddings_3d : visualization.embeddings_2d
    
    if (!embeddings.length) return

    // Create cluster color mapping
    const clusterColors: { [key: number]: string } = {}
    const uniqueClusters = [...new Set(visualization.cluster_labels)]
    uniqueClusters.forEach((cluster, index) => {
      if (cluster === -1) {
        clusterColors[cluster] = '#999999' // Gray for noise points
      } else {
        clusterColors[cluster] = `hsl(${(index * 360) / uniqueClusters.length}, 70%, 50%)`
      }
    })

    const trace = {
      x: embeddings.map(point => point[0]),
      y: embeddings.map(point => point[1]),
      z: view3D ? embeddings.map(point => point[2]) : undefined,
      mode: 'markers+text',
      type: view3D ? 'scatter3d' : 'scatter',
      text: visualization.speakers.map(speaker => {
        // Extract speaker name and segment info
        const parts = speaker.split('_')
        return parts[0] || speaker
      }),
      textposition: 'top center',
      textfont: { size: 9 },
      marker: {
        size: 8,
        color: visualization.cluster_labels.map(label => clusterColors[label]),
        line: { width: 1, color: '#000' },
        opacity: 0.8
      },
      hovertemplate: 
        '<b>%{text}</b><br>' +
        'X: %{x:.3f}<br>' +
        'Y: %{y:.3f}<br>' +
        (view3D ? 'Z: %{z:.3f}<br>' : '') +
        'Cluster: %{customdata}<br>' +
        '<extra></extra>',
      customdata: visualization.cluster_labels.map(label => 
        label === -1 ? 'Noise' : `Cluster ${label}`
      )
    }

    const layout = {
      title: {
        text: `Annotation Segments Analysis (${data.parameters.reduction_method.toUpperCase()})`,
        font: { size: 14 }
      },
      showlegend: false,
      hovermode: 'closest',
      xaxis: { title: 'Component 1' },
      yaxis: { title: 'Component 2' },
      zaxis: view3D ? { title: 'Component 3' } : undefined,
      margin: { l: 30, r: 30, t: 50, b: 30 },
      plot_bgcolor: '#fafafa',
      paper_bgcolor: '#ffffff',
      height: 400
    }

    const config = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
      displaylogo: false
    }

    Plotly.newPlot(plotRef.current, [trace], layout, config)
  }

  const formatMetric = (value: number | null | undefined, decimals: number = 3): string => {
    if (value === null || value === undefined || isNaN(value)) {
      return 'N/A'
    }
    return value.toFixed(decimals)
  }

  // Don't render anything if no segments or audio file
  if (!audioFile || segments.length === 0) {
    return null
  }

  const validSegments = segments.filter(seg => seg.speakerLabel && seg.speakerLabel.trim())
  if (validSegments.length === 0) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="h-4 w-4 text-yellow-600" />
          <p className="text-sm text-yellow-700">
            Add speaker labels to segments to enable embedding analysis
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Analysis Trigger Button */}
      <div className="flex items-center space-x-2">
        <button
          onClick={runAnalysis}
          disabled={isLoading}
          className="flex items-center space-x-2 px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
          title="Analyze speaker embedding clusters"
        >
          <BarChart3 className={`h-4 w-4 ${isLoading ? 'animate-pulse' : ''}`} />
          <span>{isLoading ? 'Analyzing...' : 'Analyze Segments'}</span>
        </button>
        
        {analysisData && (
          <button
            onClick={() => setIsVisible(!isVisible)}
            className="px-3 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
          >
            {isVisible ? 'Hide Analysis' : 'Show Analysis'}
          </button>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <div className="flex items-center space-x-2">
            <Info className="h-4 w-4 text-red-600" />
            <div>
              <h4 className="text-red-800 font-medium text-sm">Analysis Error</h4>
              <p className="text-red-600 text-sm">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Analysis Results */}
      {isVisible && analysisData && (
        <div className="space-y-4 border rounded-lg p-4 bg-gray-50">
          {/* Header with Controls */}
          <div className="flex items-center justify-between">
            <h4 className="text-md font-semibold text-gray-900">📊 Segment Clustering Analysis</h4>
            <div className="flex space-x-2">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-1 text-gray-600 hover:text-gray-800 border rounded"
                title="Settings"
              >
                <Settings className="h-3 w-3" />
              </button>
              <button
                onClick={runAnalysis}
                disabled={isLoading}
                className="p-1 text-blue-600 hover:text-blue-800 border rounded disabled:opacity-50"
                title="Refresh"
              >
                <RefreshCw className={`h-3 w-3 ${isLoading ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>

          {/* Settings Panel */}
          {showSettings && (
            <div className="bg-white border rounded p-3">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Method
                  </label>
                  <select
                    value={settings.method}
                    onChange={(e) => setSettings({ ...settings, method: e.target.value })}
                    className="w-full px-2 py-1 border border-gray-300 rounded text-xs"
                  >
                    <option value="umap">UMAP</option>
                    <option value="tsne">t-SNE</option>
                    <option value="pca">PCA</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Clustering
                  </label>
                  <select
                    value={settings.clusterMethod}
                    onChange={(e) => setSettings({ ...settings, clusterMethod: e.target.value })}
                    className="w-full px-2 py-1 border border-gray-300 rounded text-xs"
                  >
                    <option value="dbscan">DBSCAN</option>
                    <option value="kmeans">K-Means</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">
                    Threshold
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={settings.similarityThreshold}
                    onChange={(e) => setSettings({ ...settings, similarityThreshold: parseFloat(e.target.value) })}
                    className="w-full px-2 py-1 border border-gray-300 rounded text-xs"
                  />
                </div>
              </div>
            </div>
          )}

          {/* View Toggle */}
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setView3D(false)}
              className={`px-2 py-1 text-xs rounded ${
                !view3D 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              2D
            </button>
            <button
              onClick={() => setView3D(true)}
              className={`px-2 py-1 text-xs rounded ${
                view3D 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              3D
            </button>
          </div>

          {/* Plot */}
          <div className="bg-white border rounded">
            <div ref={plotRef} style={{ width: '100%', height: '400px' }} />
          </div>

          {/* Analysis Summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Segment Info */}
            <div className="bg-white border rounded p-3">
              <h5 className="text-sm font-semibold text-gray-900 mb-2">📋 Segments</h5>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Total:</span>
                  <span className="font-medium">{analysisData.segment_info.total_segments}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Processed:</span>
                  <span className="font-medium">{analysisData.segment_info.processed_segments}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Speakers:</span>
                  <span className="font-medium">{analysisData.segment_info.unique_speakers.length}</span>
                </div>
              </div>
            </div>

            {/* Clustering Info */}
            <div className="bg-white border rounded p-3">
              <h5 className="text-sm font-semibold text-gray-900 mb-2">🔍 Clusters</h5>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Found:</span>
                  <span className="font-medium">{analysisData.clustering.n_clusters}</span>
                </div>
                {analysisData.clustering.silhouette_score !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Silhouette:</span>
                    <span className="font-medium">{formatMetric(analysisData.clustering.silhouette_score)}</span>
                  </div>
                )}
                {analysisData.clustering.n_noise !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Noise:</span>
                    <span className="font-medium">{analysisData.clustering.n_noise}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Quality */}
            <div className="bg-white border rounded p-3">
              <h5 className="text-sm font-semibold text-gray-900 mb-2">📈 Quality</h5>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Separation:</span>
                  <span className="font-medium">{formatMetric(analysisData.quality_metrics.separation_quality)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg Similarity:</span>
                  <span className="font-medium">{formatMetric(analysisData.quality_metrics.mean_similarity)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Similar Speakers Warning */}
          {analysisData.similar_speakers.length > 0 && (
            <div className="bg-yellow-50 border border-yellow-200 rounded p-3">
              <h5 className="text-sm font-semibold text-yellow-800 mb-2">⚠️ Similar Segments Detected</h5>
              <div className="space-y-1">
                {analysisData.similar_speakers.slice(0, 3).map((pair, index) => (
                  <div key={index} className="flex justify-between items-center text-xs">
                    <span className="text-yellow-700">
                      {pair.speaker1.split('_')[0]} ↔ {pair.speaker2.split('_')[0]}
                    </span>
                    <span className="font-bold text-yellow-800">
                      {formatMetric(pair.similarity)}
                    </span>
                  </div>
                ))}
                {analysisData.similar_speakers.length > 3 && (
                  <p className="text-xs text-yellow-600 text-center">
                    ... and {analysisData.similar_speakers.length - 3} more
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}