import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  TrendingUp,
  Network, 
  GitBranch,
  Lightbulb,
  Filter,
  RefreshCw,
  Route,
  ArrowRight,
  BarChart3,
  ScatterChart
} from 'lucide-react';

interface AIAdvancedDashboardProps {
  userId: string;
  subscriptionTier: string;
}

interface GNNPrediction {
  source: string;
  target: string;
  confidence: number;
  sustainability_impact: number;
  economic_value: number;
  reasoning: string;
}

interface MultiHopPath {
  entities: string[];
  relationships: string[][];
  total_value: number;
  sustainability_score: number;
  complexity: string;
}

interface AIInsight {
  id: string;
  type: 'opportunity' | 'risk' | 'optimization' | 'trend';
  title: string;
  description: string;
  confidence: number;
  impact: 'high' | 'medium' | 'low';
  actionable: boolean;
  timestamp: string;
  action_url?: string;
}

interface AnalyticsResult {
  timestamp: string;
  metrics: {
    [key: string]: number;
  };
  confidence_score: number;
  insights: string[];
}

interface SimulationResult {
  scenario_name: string;
  iterations: number;
  results: {
    mean: number;
    std: number;
    percentiles: {
      [key: string]: number;
    };
  };
  risk_metrics: {
    [key: string]: number;
  };
}

interface Scenario {
  name: string;
  base_case: any;
  variations: any[];
  created_at: string;
}

export function AIAdvancedDashboard({ userId, subscriptionTier }: AIAdvancedDashboardProps) {
  const [activeTab, setActiveTab] = useState<'analytics' | 'simulation' | 'scenarios' | 'optimization'>('analytics');
  const [gnnPredictions, setGnnPredictions] = useState<GNNPrediction[]>([]);
  const [multiHopPaths, setMultiHopPaths] = useState<MultiHopPath[]>([]);
  const [aiInsights, setAiInsights] = useState<AIInsight[]>([]);
  const [symbiosisNetwork, setSymbiosisNetwork] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({
    confidence: 0.7,
    sustainability: 0.5,
    economic: 0.5,
    showOnlyActionable: false
  });
  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'detailed'>('grid');
  const [realTimeData, setRealTimeData] = useState<AnalyticsResult[]>([]);
  const [simulationResults, setSimulationResults] = useState<SimulationResult[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<string>('');
  
  // Form states
  const [historicalData, setHistoricalData] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [forecastPeriods, setForecastPeriods] = useState(12);
  const [simulationIterations, setSimulationIterations] = useState(10000);
  const [demandDistribution, setDemandDistribution] = useState('');
  const [supplyDistribution, setSupplyDistribution] = useState('');
  const [costDistribution, setCostDistribution] = useState('');

  useEffect(() => {
    loadAIData();
    loadScenarios();
  }, [userId]);

  const loadAIData = async () => {
    setLoading(true);
    try {
      // Load GNN predictions
      const gnnResponse = await fetch('/api/ai/gnn-predictions');
      const gnnData = await gnnResponse.json();
      setGnnPredictions(gnnData.predictions || []);

      // Load multi-hop paths
      const pathsResponse = await fetch('/api/ai/multi-hop-paths');
      const pathsData = await pathsResponse.json();
      setMultiHopPaths(pathsData.paths || []);

      // Load AI insights
      const insightsResponse = await fetch('/api/ai/insights');
      const insightsData = await insightsResponse.json();
      setAiInsights(insightsData.insights || []);

      // Load symbiosis network
      const networkResponse = await fetch('/api/ai/symbiosis-network');
      const networkData = await networkResponse.json();
      setSymbiosisNetwork(networkData.network);
    } catch (error) {
      console.error('Failed to load AI data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadScenarios = async () => {
    try {
      const response = await fetch('/api/scenarios');
      const data = await response.json();
      setScenarios(data.scenarios || []);
    } catch (error) {
      console.error('Failed to load scenarios:', error);
    }
  };

  const trainPredictiveModel = async () => {
    if (!historicalData || !targetColumn) {
      alert('Please provide historical data and target column');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/analytics/predictive/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          historical_data: JSON.parse(historicalData),
          target_column: targetColumn
        })
      });

      const data = await response.json();
      if (data.success) {
        alert('Model trained successfully!');
      } else {
        alert(`Training failed: ${data.error}`);
      }
    } catch (error) {
      console.error('Failed to train model:', error);
      alert('Failed to train model');
    } finally {
      setLoading(false);
    }
  };

  const runMonteCarloSimulation = async () => {
    if (!demandDistribution || !supplyDistribution || !costDistribution) {
      alert('Please provide all distribution parameters');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/simulation/monte-carlo/supply-chain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          demand_distribution: JSON.parse(demandDistribution),
          supply_distribution: JSON.parse(supplyDistribution),
          cost_distribution: JSON.parse(costDistribution),
          iterations: simulationIterations
        })
      });

      const data = await response.json();
      if (data.success) {
        setSimulationResults(prev => [...prev, data.simulation]);
        alert('Monte Carlo simulation completed!');
      } else {
        alert(`Simulation failed: ${data.error}`);
      }
    } catch (error) {
      console.error('Failed to run simulation:', error);
      alert('Failed to run simulation');
    } finally {
      setLoading(false);
    }
  };

  const createScenario = async () => {
    const scenarioName = prompt('Enter scenario name:');
    if (!scenarioName) return;

    const baseCase = prompt('Enter base case (JSON):');
    if (!baseCase) return;

    try {
      const response = await fetch('/api/scenarios/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_name: scenarioName,
          base_case: JSON.parse(baseCase),
          variations: []
        })
      });

      const data = await response.json();
      if (data.success) {
        await loadScenarios();
        alert('Scenario created successfully!');
      } else {
        alert(`Scenario creation failed: ${data.error}`);
      }
    } catch (error) {
      console.error('Failed to create scenario:', error);
      alert('Failed to create scenario');
    }
  };

  const analyzeScenario = async (scenarioName: string) => {
    try {
      const response = await fetch(`/api/scenarios/analyze/${scenarioName}`, {
        method: 'POST'
      });

      const data = await response.json();
      if (data.success) {
        alert(`Scenario analysis completed! Check console for details.`);
        console.log('Scenario analysis:', data.analysis);
    } else {
        alert(`Analysis failed: ${data.error}`);
      }
    } catch (error) {
      console.error('Failed to analyze scenario:', error);
      alert('Failed to analyze scenario');
    }
  };

  const createDashboard = async () => {
    try {
      const response = await fetch('/api/analytics/dashboard', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analytics_results: realTimeData,
          simulation_results: simulationResults
        })
      });

      const data = await response.json();
      if (data.success) {
        alert('Dashboard created successfully!');
        console.log('Dashboard data:', data.dashboard);
      } else {
        alert(`Dashboard creation failed: ${data.error}`);
      }
    } catch (error) {
      console.error('Failed to create dashboard:', error);
      alert('Failed to create dashboard');
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'opportunity':
        return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'risk':
        return <ScatterChart className="h-4 w-4 text-red-500" />;
      case 'optimization':
        return <BarChart3 className="h-4 w-4 text-blue-500" />;
      case 'trend':
        return <TrendingUp className="h-4 w-4 text-purple-500" />;
      default:
        return <Lightbulb className="h-4 w-4 text-yellow-500" />;
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high':
        return 'border-red-200 bg-red-50 text-red-800';
      case 'medium':
        return 'border-yellow-200 bg-yellow-50 text-yellow-800';
      case 'low':
        return 'border-green-200 bg-green-50 text-green-800';
      default:
        return 'border-gray-200 bg-gray-50 text-gray-800';
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Advanced Analytics & Simulation Dashboard</h1>
        <p className="text-gray-600">
          Real-time analytics, predictive modeling, Monte Carlo simulations, and scenario planning
        </p>
        </div>

        {/* Tab Navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'analytics', label: 'Real-time Analytics' },
            { id: 'simulation', label: 'Monte Carlo Simulation' },
            { id: 'scenarios', label: 'Scenario Planning' },
            { id: 'optimization', label: 'Optimization' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {activeTab === 'analytics' && (
        <div className="space-y-6">
          {/* Predictive Modeling */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Predictive Modeling</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Historical Data (JSON)
                </label>
                <textarea
                  value={historicalData}
                  onChange={(e) => setHistoricalData(e.target.value)}
                  placeholder='[{"feature1": 1, "feature2": 2, "target": 3}, ...]'
                  rows={4}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Target Column
                </label>
                <input
                  type="text"
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  placeholder="target"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <label className="block text-sm font-medium text-gray-700 mb-2 mt-4">
                  Forecast Periods
                </label>
                <input
                  type="number"
                  value={forecastPeriods}
                  onChange={(e) => setForecastPeriods(parseInt(e.target.value))}
                  min="1"
                  max="60"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>
            <button
              onClick={trainPredictiveModel}
              disabled={loading}
              className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 disabled:opacity-50"
            >
              {loading ? 'Training...' : 'Train Model'}
            </button>
          </div>

          {/* Analytics Results */}
          {realTimeData.length > 0 && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Real-time Analytics</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {realTimeData.slice(-3).map((data, index) => (
                  <div key={index} className="border rounded-lg p-4">
                    <h3 className="font-semibold mb-2">Analytics {index + 1}</h3>
                    <p className="text-sm text-gray-600">Confidence: {data.confidence_score.toFixed(2)}</p>
                    <p className="text-sm text-gray-600">Timestamp: {new Date(data.timestamp).toLocaleTimeString()}</p>
                </div>
              ))}
            </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'simulation' && (
        <div className="space-y-6">
          {/* Monte Carlo Simulation */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Monte Carlo Simulation</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Demand Distribution (JSON)
                </label>
                <textarea
                  value={demandDistribution}
                  onChange={(e) => setDemandDistribution(e.target.value)}
                  placeholder='{"type": "normal", "mean": 100, "std": 20}'
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Supply Distribution (JSON)
                </label>
                <textarea
                  value={supplyDistribution}
                  onChange={(e) => setSupplyDistribution(e.target.value)}
                  placeholder='{"type": "normal", "mean": 95, "std": 15}'
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Cost Distribution (JSON)
                </label>
                <textarea
                  value={costDistribution}
                  onChange={(e) => setCostDistribution(e.target.value)}
                  placeholder='{"type": "uniform", "min": 10, "max": 50}'
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Iterations
                </label>
                <input
                  type="number"
                  value={simulationIterations}
                  onChange={(e) => setSimulationIterations(parseInt(e.target.value))}
                  min="1000"
                  max="100000"
                  step="1000"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>
            <button
              onClick={runMonteCarloSimulation}
              disabled={loading}
              className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50"
            >
              {loading ? 'Running Simulation...' : 'Run Monte Carlo Simulation'}
            </button>
          </div>

          {/* Simulation Results */}
          {simulationResults.length > 0 && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Simulation Results</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {simulationResults.map((result, index) => (
                  <div key={index} className="border rounded-lg p-4">
                    <h3 className="font-semibold mb-2">{result.scenario_name}</h3>
                    <p className="text-sm text-gray-600">Mean: {result.results.mean.toFixed(2)}</p>
                    <p className="text-sm text-gray-600">Std: {result.results.std.toFixed(2)}</p>
                    <p className="text-sm text-gray-600">Iterations: {result.iterations}</p>
                  </div>
                ))}
                      </div>
                    </div>
          )}
        </div>
      )}

      {activeTab === 'scenarios' && (
        <div className="space-y-6">
          {/* Scenario Management */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Scenario Planning</h2>
              <button
                onClick={createScenario}
                className="bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-700"
              >
                Create New Scenario
              </button>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {scenarios.map((scenario) => (
                <div key={scenario.name} className="border rounded-lg p-4">
                  <h3 className="font-semibold mb-2">{scenario.name}</h3>
                  <p className="text-sm text-gray-600 mb-2">
                    Created: {new Date(scenario.created_at).toLocaleDateString()}
                  </p>
                  <p className="text-sm text-gray-600 mb-2">
                    Variations: {scenario.variations.length}
                  </p>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => analyzeScenario(scenario.name)}
                      className="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700"
                    >
                      Analyze
                    </button>
                    <button
                      onClick={() => setSelectedScenario(scenario.name)}
                      className="bg-green-600 text-white px-3 py-1 rounded text-sm hover:bg-green-700"
                    >
                      View Details
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {scenarios.length === 0 && (
              <div className="text-center py-8">
                <p className="text-gray-500">No scenarios created yet</p>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'optimization' && (
        <div className="space-y-6">
          {/* Supply Chain Optimization */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Supply Chain Optimization</h2>
            <p className="text-gray-600 mb-4">
              Optimize supply chain networks for cost, efficiency, and resilience
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold mb-2">Network Nodes</h3>
                <textarea
                  placeholder='[{"id": "node1", "type": "supplier", "capacity": 1000}, ...]'
                  rows={4}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div>
                <h3 className="font-semibold mb-2">Network Edges</h3>
                <textarea
                  placeholder='[{"from": "node1", "to": "node2", "cost": 10, "capacity": 500}, ...]'
                  rows={4}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
            </div>
            
            <button className="bg-orange-600 text-white px-4 py-2 rounded-md hover:bg-orange-700 mt-4">
              Optimize Supply Chain
            </button>
          </div>

          {/* Network Resilience */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Network Resilience Analysis</h2>
            <p className="text-gray-600 mb-4">
              Calculate network resilience and identify critical nodes
            </p>
            
            <textarea
              placeholder='{"nodes": [...], "edges": [...], "failure_scenarios": [...]}'
              rows={4}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 mb-4"
            />
            
            <button className="bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700">
              Calculate Resilience
                    </button>
          </div>
        </div>
      )}

      {/* Dashboard Creation */}
      <div className="fixed bottom-6 right-6">
        <button
          onClick={createDashboard}
          className="bg-indigo-600 text-white px-6 py-3 rounded-full shadow-lg hover:bg-indigo-700"
        >
          Create Dashboard
        </button>
      </div>
    </div>
  );
} 