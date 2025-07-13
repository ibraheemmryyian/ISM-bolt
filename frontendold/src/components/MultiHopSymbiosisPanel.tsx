import React, { useState, useEffect } from 'react';
import { Network, GitBranch, TrendingUp, AlertTriangle, Settings, Download, Eye } from 'lucide-react';

interface NetworkMetrics {
  total_nodes: number;
  total_edges: number;
  density: number;
  average_clustering: number;
  symbiosis_potential: number;
  sustainability_score: number;
  economic_value: number;
}

interface MultiHopPath {
  path_id: string;
  nodes: string[];
  edges: string[];
  total_cost: number;
  sustainability_impact: number;
  economic_value: number;
  complexity: string;
  confidence: number;
}

interface NetworkOptimization {
  objective: string;
  constraints: any;
  solution: any;
  improvement: number;
  execution_time: number;
}

interface NetworkResilience {
  overall_resilience: number;
  critical_nodes: string[];
  redundancy_paths: number;
  failure_scenarios: any[];
  recommendations: string[];
}

interface NetworkMonitoring {
  performance_metrics: any;
  alerts: any[];
  bottlenecks: string[];
  optimization_opportunities: string[];
}

const MultiHopSymbiosisPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'paths' | 'optimization' | 'resilience' | 'monitoring'>('overview');
  const [metrics, setMetrics] = useState<NetworkMetrics | null>(null);
  const [paths, setPaths] = useState<MultiHopPath[]>([]);
  const [optimization, setOptimization] = useState<NetworkOptimization | null>(null);
  const [resilience, setResilience] = useState<NetworkResilience | null>(null);
  const [monitoring, setMonitoring] = useState<NetworkMonitoring | null>(null);
  const [loading, setLoading] = useState(false);
  
  // Form states
  const [sourceNode, setSourceNode] = useState('');
  const [targetNode, setTargetNode] = useState('');
  const [maxHops, setMaxHops] = useState(5);
  const [minSustainability, setMinSustainability] = useState(0.5);
  const [maxCost, setMaxCost] = useState('');
  const [optimizationObjective, setOptimizationObjective] = useState('maximize_sustainability');
  const [optimizationConstraints, setOptimizationConstraints] = useState('');

  useEffect(() => {
    loadNetworkMetrics();
  }, []);

  const loadNetworkMetrics = async () => {
    try {
      const response = await fetch('/api/network/metrics');
      const data = await response.json();
      setMetrics(data.metrics);
    } catch (error) {
      console.error('Failed to load network metrics:', error);
    }
  };

  const findMultiHopPaths = async () => {
    if (!sourceNode || !targetNode) {
      alert('Please provide source and target nodes');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/network/paths', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source: sourceNode,
          target: targetNode,
          max_hops: maxHops,
          min_sustainability: minSustainability,
          max_cost: maxCost === '' ? 'inf' : parseFloat(maxCost)
        })
      });

      const data = await response.json();
      setPaths(data.paths || []);
    } catch (error) {
      console.error('Failed to find multi-hop paths:', error);
      alert('Failed to find multi-hop paths');
    } finally {
      setLoading(false);
    }
  };

  const optimizeNetwork = async () => {
    if (!optimizationObjective) {
      alert('Please select an optimization objective');
      return;
    }

    setLoading(true);
    try {
      const constraints = optimizationConstraints ? JSON.parse(optimizationConstraints) : {};
      
      const response = await fetch('/api/network/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          objective: optimizationObjective,
          constraints: constraints
        })
      });

      const data = await response.json();
      setOptimization(data.optimization);
    } catch (error) {
      console.error('Failed to optimize network:', error);
      alert('Failed to optimize network');
    } finally {
      setLoading(false);
    }
  };

  const analyzeResilience = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/network/resilience');
      const data = await response.json();
      setResilience(data.resilience);
    } catch (error) {
      console.error('Failed to analyze resilience:', error);
      alert('Failed to analyze network resilience');
    } finally {
      setLoading(false);
    }
  };

  const monitorPerformance = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/network/monitor');
      const data = await response.json();
      setMonitoring(data.monitoring);
    } catch (error) {
      console.error('Failed to monitor performance:', error);
      alert('Failed to monitor network performance');
    } finally {
      setLoading(false);
    }
  };

  const exportNetwork = async (format: string) => {
    try {
      const response = await fetch(`/api/network/export/${format}`);
      const data = await response.json();
      
      // Create download link
      const blob = new Blob([JSON.stringify(data.export)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `network_export.${format}`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export network:', error);
      alert('Failed to export network data');
    }
  };

  const createVisualization = async () => {
    try {
      const response = await fetch('/api/network/visualize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          output_path: 'network_visualization.png'
        })
      });

      const data = await response.json();
      if (data.success) {
        alert('Network visualization created successfully!');
      } else {
        alert('Failed to create visualization');
      }
    } catch (error) {
      console.error('Failed to create visualization:', error);
      alert('Failed to create network visualization');
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Multi-Hop Symbiosis Network</h1>
        <p className="text-gray-600">
          Advanced network analysis, optimization, and multi-hop path discovery
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', label: 'Network Overview', icon: <Network className="h-4 w-4" /> },
            { id: 'paths', label: 'Multi-Hop Paths', icon: <GitBranch className="h-4 w-4" /> },
            { id: 'optimization', label: 'Network Optimization', icon: <TrendingUp className="h-4 w-4" /> },
            { id: 'resilience', label: 'Resilience Analysis', icon: <AlertTriangle className="h-4 w-4" /> },
            { id: 'monitoring', label: 'Performance Monitoring', icon: <Settings className="h-4 w-4" /> }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                activeTab === tab.id
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Network Metrics */}
          <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Network Metrics</h2>
              <button
                onClick={loadNetworkMetrics}
                className="bg-indigo-600 text-white px-3 py-1 rounded text-sm hover:bg-indigo-700"
              >
                Refresh
              </button>
            </div>
            
            {metrics && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="border rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-600">Total Nodes</h3>
                  <p className="text-2xl font-bold text-blue-600">{metrics.total_nodes}</p>
                </div>
                <div className="border rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-600">Total Edges</h3>
                  <p className="text-2xl font-bold text-green-600">{metrics.total_edges}</p>
                </div>
                <div className="border rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-600">Network Density</h3>
                  <p className="text-2xl font-bold text-purple-600">{metrics.density.toFixed(3)}</p>
                </div>
                <div className="border rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-600">Symbiosis Potential</h3>
                  <p className="text-2xl font-bold text-emerald-600">{metrics.symbiosis_potential.toFixed(1)}%</p>
                </div>
                <div className="border rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-600">Sustainability Score</h3>
                  <p className="text-2xl font-bold text-green-600">{metrics.sustainability_score.toFixed(1)}%</p>
                </div>
                <div className="border rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-600">Economic Value</h3>
                  <p className="text-2xl font-bold text-orange-600">${metrics.economic_value.toLocaleString()}</p>
                </div>
                <div className="border rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-600">Clustering Coefficient</h3>
                  <p className="text-2xl font-bold text-indigo-600">{metrics.average_clustering.toFixed(3)}</p>
                </div>
              </div>
            )}
          </div>

          {/* Export and Visualization */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Network Export & Visualization</h2>
            <div className="flex space-x-4">
              <button
                onClick={() => exportNetwork('json')}
                className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
              >
                <Download className="h-4 w-4" />
                <span>Export JSON</span>
              </button>
              <button
                onClick={() => exportNetwork('csv')}
                className="flex items-center space-x-2 bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700"
              >
                <Download className="h-4 w-4" />
                <span>Export CSV</span>
              </button>
              <button
                onClick={() => exportNetwork('graphml')}
                className="flex items-center space-x-2 bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-700"
              >
                <Download className="h-4 w-4" />
                <span>Export GraphML</span>
              </button>
              <button
                onClick={createVisualization}
                className="flex items-center space-x-2 bg-orange-600 text-white px-4 py-2 rounded-md hover:bg-orange-700"
              >
                <Eye className="h-4 w-4" />
                <span>Create Visualization</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'paths' && (
        <div className="space-y-6">
          {/* Path Finding Form */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Find Multi-Hop Paths</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Source Node</label>
                <input
                  type="text"
                  value={sourceNode}
                  onChange={(e) => setSourceNode(e.target.value)}
                  placeholder="Enter source node ID"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Target Node</label>
                <input
                  type="text"
                  value={targetNode}
                  onChange={(e) => setTargetNode(e.target.value)}
                  placeholder="Enter target node ID"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Max Hops</label>
                <input
                  type="number"
                  value={maxHops}
                  onChange={(e) => setMaxHops(parseInt(e.target.value))}
                  min="1"
                  max="10"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Min Sustainability</label>
                <input
                  type="number"
                  value={minSustainability}
                  onChange={(e) => setMinSustainability(parseFloat(e.target.value))}
                  min="0"
                  max="1"
                  step="0.1"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Max Cost</label>
                <input
                  type="number"
                  value={maxCost}
                  onChange={(e) => setMaxCost(e.target.value)}
                  placeholder="Leave empty for unlimited"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>
        <button
          onClick={findMultiHopPaths}
              disabled={loading}
              className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 disabled:opacity-50"
        >
              {loading ? 'Finding Paths...' : 'Find Multi-Hop Paths'}
        </button>
      </div>

          {/* Path Results */}
          {paths.length > 0 && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Multi-Hop Paths Found</h2>
              <div className="space-y-4">
                {paths.map((path, index) => (
                  <div key={path.path_id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold">Path {index + 1}</h3>
                      <span className="text-sm text-gray-500">Confidence: {path.confidence.toFixed(2)}</span>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                      <div>
                        <p className="text-sm text-gray-600">Total Cost</p>
                        <p className="font-medium">${path.total_cost.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Sustainability Impact</p>
                        <p className="font-medium text-green-600">{path.sustainability_impact.toFixed(1)}%</p>
                      </div>
        <div>
                        <p className="text-sm text-gray-600">Economic Value</p>
                        <p className="font-medium text-blue-600">${path.economic_value.toLocaleString()}</p>
                      </div>
                    </div>
                    <div className="mb-3">
                      <p className="text-sm text-gray-600">Path Flow:</p>
                      <div className="flex items-center space-x-2 flex-wrap">
                        {path.nodes.map((node, nodeIndex) => (
                          <React.Fragment key={nodeIndex}>
                            <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm">
                              {node}
                  </span>
                            {nodeIndex < path.nodes.length - 1 && (
                              <span className="text-gray-400">→</span>
                            )}
                          </React.Fragment>
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-500">Complexity: {path.complexity}</span>
                      <button className="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700">
                        Analyze Path
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'optimization' && (
        <div className="space-y-6">
          {/* Network Optimization */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Network Optimization</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Optimization Objective</label>
                <select
                  value={optimizationObjective}
                  onChange={(e) => setOptimizationObjective(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="maximize_sustainability">Maximize Sustainability</option>
                  <option value="minimize_cost">Minimize Cost</option>
                  <option value="maximize_efficiency">Maximize Efficiency</option>
                  <option value="balance_all">Balance All Factors</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Constraints (JSON)</label>
                <textarea
                  value={optimizationConstraints}
                  onChange={(e) => setOptimizationConstraints(e.target.value)}
                  placeholder='{"max_cost": 1000000, "min_sustainability": 0.7}'
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>
            <button
              onClick={optimizeNetwork}
              disabled={loading}
              className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50"
            >
              {loading ? 'Optimizing...' : 'Optimize Network'}
            </button>
          </div>

          {/* Optimization Results */}
          {optimization && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Optimization Results</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Objective</p>
                  <p className="font-medium">{optimization.objective}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Improvement</p>
                  <p className="font-medium text-green-600">{optimization.improvement.toFixed(2)}%</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Execution Time</p>
                  <p className="font-medium">{optimization.execution_time.toFixed(2)}s</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'resilience' && (
        <div className="space-y-6">
          {/* Resilience Analysis */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Network Resilience Analysis</h2>
              <button
                onClick={analyzeResilience}
                disabled={loading}
                className="bg-orange-600 text-white px-4 py-2 rounded-md hover:bg-orange-700 disabled:opacity-50"
              >
                {loading ? 'Analyzing...' : 'Analyze Resilience'}
              </button>
            </div>
            
            {resilience && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="border rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-600">Overall Resilience</h3>
                    <p className="text-2xl font-bold text-green-600">{resilience.overall_resilience.toFixed(1)}%</p>
                  </div>
                  <div className="border rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-600">Critical Nodes</h3>
                    <p className="text-2xl font-bold text-red-600">{resilience.critical_nodes.length}</p>
                  </div>
                  <div className="border rounded-lg p-4">
                    <h3 className="text-sm font-medium text-gray-600">Redundancy Paths</h3>
                    <p className="text-2xl font-bold text-blue-600">{resilience.redundancy_paths}</p>
                  </div>
                </div>
                
                {resilience.critical_nodes.length > 0 && (
                  <div>
                    <h3 className="font-semibold mb-2">Critical Nodes</h3>
                    <div className="flex flex-wrap gap-2">
                      {resilience.critical_nodes.map((node, index) => (
                        <span key={index} className="px-2 py-1 bg-red-100 text-red-800 rounded text-sm">
                          {node}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                
                {resilience.recommendations.length > 0 && (
                  <div>
                    <h3 className="font-semibold mb-2">Recommendations</h3>
                    <ul className="list-disc list-inside space-y-1">
                      {resilience.recommendations.map((rec, index) => (
                        <li key={index} className="text-sm text-gray-700">{rec}</li>
            ))}
          </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'monitoring' && (
        <div className="space-y-6">
          {/* Performance Monitoring */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Performance Monitoring</h2>
              <button
                onClick={monitorPerformance}
                disabled={loading}
                className="bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-700 disabled:opacity-50"
              >
                {loading ? 'Monitoring...' : 'Monitor Performance'}
              </button>
            </div>
            
            {monitoring && (
              <div className="space-y-4">
                {monitoring.alerts.length > 0 && (
                  <div>
                    <h3 className="font-semibold mb-2 text-red-600">Active Alerts</h3>
                    <div className="space-y-2">
                      {monitoring.alerts.map((alert, index) => (
                        <div key={index} className="p-3 bg-red-50 border border-red-200 rounded-md">
                          <p className="text-sm text-red-800">{alert.message}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {monitoring.bottlenecks.length > 0 && (
                  <div>
                    <h3 className="font-semibold mb-2 text-orange-600">Bottlenecks</h3>
                    <div className="flex flex-wrap gap-2">
                      {monitoring.bottlenecks.map((bottleneck, index) => (
                        <span key={index} className="px-2 py-1 bg-orange-100 text-orange-800 rounded text-sm">
                          {bottleneck}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                
                {monitoring.optimization_opportunities.length > 0 && (
                  <div>
                    <h3 className="font-semibold mb-2 text-green-600">Optimization Opportunities</h3>
                    <ul className="list-disc list-inside space-y-1">
                      {monitoring.optimization_opportunities.map((opportunity, index) => (
                        <li key={index} className="text-sm text-gray-700">{opportunity}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default MultiHopSymbiosisPanel; 