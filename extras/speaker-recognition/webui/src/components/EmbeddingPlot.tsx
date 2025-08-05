import React, { useEffect, useRef, useState } from 'react'
import { RefreshCw, Download, Settings, Info } from 'lucide-react'

// Import Plotly dynamically to avoid SSR issues
let Plotly: any = null;
if (typeof window !== 'undefined') {
  import('plotly.js-dist-min').then((module) => {
    Plotly = module.default;
  });
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
    inertia?: number | null
  }
  similar_speakers: Array<{
    speaker1: string
    speaker2: string
    similarity: number | null
  }>
  quality_metrics: {
    n_speakers: number
    mean_similarity: number | null
    std_similarity: number | null
    separation_quality: number | null
  }
  parameters: {
    reduction_method: string
    cluster_method: string
    similarity_threshold: number
  }
}

interface EmbeddingPlotProps {
  userId?: number
  onRefresh?: () => void
}

export default function EmbeddingPlot({ userId, onRefresh }: EmbeddingPlotProps) {
  const plotRef = useRef<HTMLDivElement>(null)
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [view3D, setView3D] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings] = useState({
    method: 'umap',
    clusterMethod: 'dbscan',
    similarityThreshold: 0.8
  })

  const loadAnalysis = async () => {
    if (!Plotly) return // Plotly not loaded yet

    setIsLoading(true)
    setError(null)

    try {
      const params = new URLSearchParams({
        method: settings.method,
        cluster_method: settings.clusterMethod,
        similarity_threshold: settings.similarityThreshold.toString()
      })

      if (userId) {
        params.append('user_id', userId.toString())
      }

      const response = await fetch(`/api/speakers/analysis?${params}`)
      
      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`)
      }

      const data = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }

      setAnalysisData(data)
      createPlot(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load analysis')
      console.error('Analysis loading error:', err)
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
      text: visualization.speakers.map(speaker => speaker.split('_').pop() || speaker),
      textposition: 'top center',
      textfont: { size: 10 },
      marker: {
        size: 10,
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
        text: `Speaker Embeddings Analysis (${data.parameters.reduction_method.toUpperCase()})`,
        font: { size: 16 }
      },
      showlegend: false,
      hovermode: 'closest',
      xaxis: { title: 'Component 1' },
      yaxis: { title: 'Component 2' },
      zaxis: view3D ? { title: 'Component 3' } : undefined,
      margin: { l: 40, r: 40, t: 60, b: 40 },
      plot_bgcolor: '#fafafa',
      paper_bgcolor: '#ffffff'
    }

    const config = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
      displaylogo: false
    }

    Plotly.newPlot(plotRef.current, [trace], layout, config)
  }

  const downloadPlot = () => {
    if (!plotRef.current || !Plotly) return

    Plotly.downloadImage(plotRef.current, {
      format: 'png',
      width: 1200,
      height: 800,
      filename: `speaker_embeddings_${view3D ? '3d' : '2d'}`
    })
  }

  const formatMetric = (value: number | null | undefined, decimals: number = 3): string => {
    if (value === null || value === undefined || isNaN(value)) {
      return 'N/A'
    }
    return value.toFixed(decimals)
  }

  useEffect(() => {
    // Load Plotly and then load analysis
    const loadPlotlyAndAnalysis = async () => {
      if (!Plotly) {
        // Wait a bit for Plotly to load
        setTimeout(loadPlotlyAndAnalysis, 100)
        return
      }
      loadAnalysis()
    }
    
    loadPlotlyAndAnalysis()
  }, [userId, settings])

  useEffect(() => {
    if (analysisData) {
      createPlot(analysisData)
    }
  }, [view3D, analysisData])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">📊 Embedding Analysis</h3>
        <div className="flex space-x-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 text-gray-600 hover:text-gray-800 border rounded-md"
            title="Analysis Settings"
          >
            <Settings className="h-4 w-4" />
          </button>
          <button
            onClick={loadAnalysis}
            disabled={isLoading}
            className="p-2 text-blue-600 hover:text-blue-800 border rounded-md disabled:opacity-50"
            title="Refresh Analysis"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={downloadPlot}
            disabled={!analysisData}
            className="p-2 text-green-600 hover:text-green-800 border rounded-md disabled:opacity-50"
            title="Download Plot"
          >
            <Download className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="bg-gray-50 border rounded-lg p-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Reduction Method
              </label>
              <select
                value={settings.method}
                onChange={(e) => setSettings({ ...settings, method: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
              >
                <option value="umap">UMAP</option>
                <option value="tsne">t-SNE</option>
                <option value="pca">PCA</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Clustering Method
              </label>
              <select
                value={settings.clusterMethod}
                onChange={(e) => setSettings({ ...settings, clusterMethod: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
              >
                <option value="dbscan">DBSCAN</option>
                <option value="kmeans">K-Means</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Similarity Threshold
              </label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                value={settings.similarityThreshold}
                onChange={(e) => setSettings({ ...settings, similarityThreshold: parseFloat(e.target.value) })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
              />
            </div>
          </div>
        </div>
      )}

      {/* Plot Controls */}
      {analysisData && (
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setView3D(false)}
              className={`px-3 py-1 text-sm rounded-md ${
                !view3D 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              2D View
            </button>
            <button
              onClick={() => setView3D(true)}
              className={`px-3 py-1 text-sm rounded-md ${
                view3D 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              3D View
            </button>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Info className="h-5 w-5 text-red-600" />
            <div>
              <h4 className="text-red-800 font-medium">Analysis Error</h4>
              <p className="text-red-600 text-sm">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <p className="mt-2 text-gray-600">Analyzing speaker embeddings...</p>
        </div>
      )}

      {/* Main Plot */}
      <div className="bg-white border rounded-lg p-4">
        <div ref={plotRef} style={{ width: '100%', height: '500px' }} />
      </div>

      {/* Analysis Results */}
      {analysisData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Clustering Info */}
          <div className="bg-white border rounded-lg p-4">
            <h4 className="text-md font-semibold text-gray-900 mb-3">🔍 Clustering Results</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Method:</span>
                <span className="font-medium">{analysisData.clustering.method.toUpperCase()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Clusters Found:</span>
                <span className="font-medium">{analysisData.clustering.n_clusters}</span>
              </div>
              {analysisData.clustering.silhouette_score !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Silhouette Score:</span>
                  <span className="font-medium">{formatMetric(analysisData.clustering.silhouette_score)}</span>
                </div>
              )}
              {analysisData.clustering.n_noise !== undefined && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Noise Points:</span>
                  <span className="font-medium">{analysisData.clustering.n_noise}</span>
                </div>
              )}
            </div>
          </div>

          {/* Quality Metrics */}
          <div className="bg-white border rounded-lg p-4">
            <h4 className="text-md font-semibold text-gray-900 mb-3">📈 Quality Metrics</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Speakers:</span>
                <span className="font-medium">{analysisData.quality_metrics.n_speakers}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Avg. Similarity:</span>
                <span className="font-medium">{formatMetric(analysisData.quality_metrics.mean_similarity)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Separation Quality:</span>
                <span className="font-medium">{formatMetric(analysisData.quality_metrics.separation_quality)}</span>
              </div>
            </div>
          </div>

          {/* Similar Speakers */}
          {analysisData.similar_speakers.length > 0 && (
            <div className="bg-white border rounded-lg p-4 lg:col-span-2">
              <h4 className="text-md font-semibold text-gray-900 mb-3">⚠️ Similar Speakers</h4>
              <div className="space-y-2">
                {analysisData.similar_speakers.slice(0, 5).map((pair, index) => (
                  <div key={index} className="flex justify-between items-center p-2 bg-yellow-50 border border-yellow-200 rounded">
                    <div className="text-sm">
                      <span className="font-medium">{pair.speaker1.split('_').pop()}</span>
                      {' ↔ '}
                      <span className="font-medium">{pair.speaker2.split('_').pop()}</span>
                    </div>
                    <span className="text-sm font-bold text-yellow-700">
                      {formatMetric(pair.similarity)}
                    </span>
                  </div>
                ))}
                {analysisData.similar_speakers.length > 5 && (
                  <p className="text-xs text-gray-500 text-center">
                    ... and {analysisData.similar_speakers.length - 5} more similar pairs
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