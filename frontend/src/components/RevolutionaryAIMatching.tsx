import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  Search, 
  Network, 
  Zap, 
  Target, 
  TrendingUp, 
  BarChart3, 
  MessageSquare,
  Loader2,
  AlertCircle,
  CheckCircle,
  Info,
  ArrowRight,
  Users,
  Globe,
  Leaf,
  Euro,
  Clock,
  Star,
  Filter,
  Settings,
  RefreshCw
} from 'lucide-react';

interface Material {
  id: string;
  name: string;
  description: string;
  category: string;
  properties: Record<string, any>;
  company_id: string;
  quantity: number;
  unit: string;
  location: { lat: number; lng: number };
  availability: string;
  price?: number;
}

interface Match {
  id: string;
  material_id: string;
  matched_material_id: string;
  company_id: string;
  matched_company_id: string;
  match_score: number;
  match_type: 'direct' | 'multi_hop' | 'symbiotic' | 'circular';
  confidence: number;
  reasoning: string;
  carbon_savings: number;
  economic_benefit: number;
  route_optimization?: any;
  created_at: string;
}

interface LLMAnalysis {
  score: number;
  reasoning: string;
  applications: string[];
  technical_considerations: string[];
  environmental_benefits: string[];
  economic_feasibility: string;
  risks: string[];
}

interface MatchInsights {
  match_id: string;
  material1: any;
  material2: any;
  match_score: number;
  confidence: number;
  reasoning: string;
  carbon_savings: number;
  economic_benefit: number;
  match_type: string;
  created_at: string;
  recommendations: string[];
}

interface RevolutionaryAIMatchingProps {
  userId?: string;
}

export function RevolutionaryAIMatching({ userId }: RevolutionaryAIMatchingProps) {
  const [selectedMaterial, setSelectedMaterial] = useState<string>('');
  const [materials, setMaterials] = useState<Material[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [selectedMatch, setSelectedMatch] = useState<Match | null>(null);
  const [matchInsights, setMatchInsights] = useState<MatchInsights | null>(null);
  const [llmAnalysis, setLlmAnalysis] = useState<LLMAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [matchType, setMatchType] = useState<'direct' | 'multi_hop' | 'comprehensive'>('comprehensive');
  const [preferences, setPreferences] = useState({
    max_distance: 500,
    preferred_industries: [] as string[],
    min_carbon_savings: 0,
    min_economic_benefit: 0
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [semanticResults, setSemanticResults] = useState<any[]>([]);
  const [gnnScores, setGnnScores] = useState<Record<string, number>>({});

  // Load materials on component mount
  useEffect(() => {
    loadMaterials();
  }, []);

  const loadMaterials = async () => {
    try {
      const response = await fetch('/api/materials');
      if (response.ok) {
        const data = await response.json();
        setMaterials(data.materials || []);
      }
    } catch (err) {
      console.error('Failed to load materials:', err);
    }
  };

  const performSemanticSearch = async () => {
    if (!selectedMaterial) return;

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/ai-matching/semantic-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          material_id: selectedMaterial,
          top_k: 10
        })
      });

      if (!response.ok) throw new Error('Semantic search failed');

      const data = await response.json();
      setSemanticResults(data.matches || []);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const performLLMAnalysis = async (material1Id: string, material2Id: string) => {
    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/ai-matching/llm-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          material1_id: material1Id,
          material2_id: material2Id
        })
      });

      if (!response.ok) throw new Error('LLM analysis failed');

      const data = await response.json();
      setLlmAnalysis(data.analysis);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const performGNNScoring = async (material1Id: string, material2Id: string) => {
    try {
      const response = await fetch('/api/ai-matching/gnn-scoring', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          material1_id: material1Id,
          material2_id: material2Id
        })
      });

      if (!response.ok) throw new Error('GNN scoring failed');

      const data = await response.json();
      setGnnScores(prev => ({
        ...prev,
        [`${material1Id}_${material2Id}`]: data.score
      }));
    } catch (err: any) {
      console.error('GNN scoring failed:', err);
    }
  };

  const findMatches = async () => {
    if (!selectedMaterial) {
      setError('Please select a material');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/ai-matching/comprehensive', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          material_id: selectedMaterial,
          match_type: matchType,
          preferences
        })
      });

      if (!response.ok) throw new Error('Matching failed');

      const data = await response.json();
      setMatches(data.matches || []);
      
      if (data.matches && data.matches.length > 0) {
        setSelectedMatch(data.matches[0]);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const findMultiHopMatches = async () => {
    if (!selectedMaterial) {
      setError('Please select a material');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/ai-matching/multi-hop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          material_id: selectedMaterial,
          max_hops: 3
        })
      });

      if (!response.ok) throw new Error('Multi-hop analysis failed');

      const data = await response.json();
      setMatches(data.matches || []);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getMatchInsights = async (matchId: string) => {
    try {
      const response = await fetch(`/api/ai-matching/insights/${matchId}`);
      if (response.ok) {
        const data = await response.json();
        setMatchInsights(data.insights);
      }
    } catch (err) {
      console.error('Failed to get match insights:', err);
    }
  };

  const submitFeedback = async (matchId: string, rating: number, feedback: string) => {
    try {
      const response = await fetch('/api/ai-matching/learn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          match_id: matchId,
          rating,
          feedback,
          user_id: userId
        })
      });

      if (response.ok) {
        // Show success message
        console.log('Feedback submitted successfully');
      }
    } catch (err) {
      console.error('Failed to submit feedback:', err);
    }
  };

  const getMatchTypeIcon = (type: string) => {
    switch (type) {
      case 'direct': return <ArrowRight className="h-4 w-4" />;
      case 'multi_hop': return <Network className="h-4 w-4" />;
      case 'symbiotic': return <Users className="h-4 w-4" />;
      case 'circular': return <RefreshCw className="h-4 w-4" />;
      default: return <Target className="h-4 w-4" />;
    }
  };

  const getMatchTypeColor = (type: string) => {
    switch (type) {
      case 'direct': return 'text-blue-600';
      case 'multi_hop': return 'text-purple-600';
      case 'symbiotic': return 'text-green-600';
      case 'circular': return 'text-orange-600';
      default: return 'text-gray-600';
    }
  };

  const formatScore = (score: number) => {
    return score.toFixed(1);
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('de-DE', {
      style: 'currency',
      currency: 'EUR'
    }).format(amount);
  };

  return (
    <div className="bg-white rounded-xl shadow-sm p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Brain className="h-6 w-6 text-purple-500" />
          <h2 className="text-xl font-bold text-gray-900">Revolutionary AI Matching Engine</h2>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-500">Powered by LLM + GNN + Semantic Search</span>
        </div>
      </div>

      {/* Material Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Material for AI Analysis
        </label>
        <select
          value={selectedMaterial}
          onChange={(e) => setSelectedMaterial(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
        >
          <option value="">Choose a material...</option>
          {materials.map((material) => (
            <option key={material.id} value={material.id}>
              {material.name} - {material.category}
            </option>
          ))}
        </select>
      </div>

      {/* AI Analysis Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <button
          onClick={performSemanticSearch}
          disabled={!selectedMaterial || loading}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition disabled:opacity-50 flex items-center justify-center space-x-2"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
          <span>Semantic Search</span>
        </button>

        <button
          onClick={findMatches}
          disabled={!selectedMaterial || loading}
          className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition disabled:opacity-50 flex items-center justify-center space-x-2"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Brain className="h-4 w-4" />}
          <span>AI Matching</span>
        </button>

        <button
          onClick={findMultiHopMatches}
          disabled={!selectedMaterial || loading}
          className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition disabled:opacity-50 flex items-center justify-center space-x-2"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Network className="h-4 w-4" />}
          <span>Multi-Hop Analysis</span>
        </button>
      </div>

      {/* Advanced Options */}
      <div className="mb-6">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-800"
        >
          <Settings className="h-4 w-4" />
          <span>Advanced Options</span>
        </button>

        {showAdvanced && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Match Type
                </label>
                <select
                  value={matchType}
                  onChange={(e) => setMatchType(e.target.value as any)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="comprehensive">Comprehensive AI</option>
                  <option value="direct">Direct Matching</option>
                  <option value="multi_hop">Multi-Hop</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Distance (km)
                </label>
                <input
                  type="number"
                  value={preferences.max_distance}
                  onChange={(e) => setPreferences({...preferences, max_distance: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Min Carbon Savings (kg CO2)
                </label>
                <input
                  type="number"
                  value={preferences.min_carbon_savings}
                  onChange={(e) => setPreferences({...preferences, min_carbon_savings: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Min Economic Benefit (EUR)
                </label>
                <input
                  type="number"
                  value={preferences.min_economic_benefit}
                  onChange={(e) => setPreferences({...preferences, min_economic_benefit: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-500" />
            <span className="text-red-700">{error}</span>
          </div>
        </div>
      )}

      {/* Semantic Search Results */}
      {semanticResults.length > 0 && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
            <Search className="h-5 w-5 text-blue-500" />
            <span>Semantic Search Results</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {semanticResults.map((result, index) => (
              <div key={index} className="p-4 border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-600">Similarity Score</span>
                  <span className="text-lg font-bold text-blue-600">{(result.score * 100).toFixed(1)}%</span>
                </div>
                <div className="text-sm text-gray-600">
                  Material ID: {result.material_id}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* AI Matches */}
      {matches.length > 0 && (
        <div className="space-y-6">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
            <Brain className="h-5 w-5 text-purple-500" />
            <span>AI-Powered Matches</span>
            <span className="text-sm text-gray-500">({matches.length} found)</span>
          </h3>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Matches List */}
            <div className="space-y-4">
              {matches.map((match) => (
                <div
                  key={match.id}
                  className={`p-4 border rounded-lg cursor-pointer transition ${
                    selectedMatch?.id === match.id
                      ? 'border-purple-500 bg-purple-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => {
                    setSelectedMatch(match);
                    getMatchInsights(match.id);
                  }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <div className={`${getMatchTypeColor(match.match_type)}`}>
                        {getMatchTypeIcon(match.match_type)}
                      </div>
                      <span className="text-sm font-medium capitalize">{match.match_type.replace('_', ' ')}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-lg font-bold text-purple-600">{formatScore(match.match_score)}%</span>
                      <div className="flex items-center space-x-1">
                        <Star className="h-3 w-3 text-yellow-500" />
                        <span className="text-xs text-gray-600">{(match.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="flex items-center space-x-1">
                      <Leaf className="h-3 w-3 text-green-500" />
                      <span>{match.carbon_savings.toFixed(1)} kg CO2</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Euro className="h-3 w-3 text-green-500" />
                      <span>{formatCurrency(match.economic_benefit)}</span>
                    </div>
                  </div>

                  <div className="mt-2 text-xs text-gray-600 line-clamp-2">
                    {match.reasoning}
                  </div>
                </div>
              ))}
            </div>

            {/* Selected Match Details */}
            {selectedMatch && (
              <div className="p-6 bg-gray-50 rounded-lg">
                <h4 className="text-lg font-semibold text-gray-900 mb-4">Match Details</h4>
                
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-white rounded-lg">
                      <div className="text-2xl font-bold text-purple-600">{formatScore(selectedMatch.match_score)}%</div>
                      <div className="text-sm text-gray-600">Match Score</div>
                    </div>
                    <div className="text-center p-3 bg-white rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">{(selectedMatch.confidence * 100).toFixed(0)}%</div>
                      <div className="text-sm text-gray-600">Confidence</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-white rounded-lg">
                      <div className="text-lg font-bold text-green-600">{selectedMatch.carbon_savings.toFixed(1)}</div>
                      <div className="text-sm text-gray-600">kg CO2 Saved</div>
                    </div>
                    <div className="text-center p-3 bg-white rounded-lg">
                      <div className="text-lg font-bold text-green-600">{formatCurrency(selectedMatch.economic_benefit)}</div>
                      <div className="text-sm text-gray-600">Economic Benefit</div>
                    </div>
                  </div>

                  <div className="p-3 bg-white rounded-lg">
                    <h5 className="font-medium text-gray-900 mb-2">AI Reasoning</h5>
                    <p className="text-sm text-gray-700">{selectedMatch.reasoning}</p>
                  </div>

                  {matchInsights && (
                    <div className="p-3 bg-white rounded-lg">
                      <h5 className="font-medium text-gray-900 mb-2">Recommendations</h5>
                      <ul className="space-y-1">
                        {matchInsights.recommendations.map((rec, index) => (
                          <li key={index} className="text-sm text-gray-700 flex items-start space-x-2">
                            <Info className="h-3 w-3 text-blue-500 mt-0.5 flex-shrink-0" />
                            <span>{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Feedback Section */}
                  <div className="p-3 bg-white rounded-lg">
                    <h5 className="font-medium text-gray-900 mb-2">Rate this Match</h5>
                    <div className="flex items-center space-x-2">
                      {[1, 2, 3, 4, 5].map((rating) => (
                        <button
                          key={rating}
                          onClick={() => submitFeedback(selectedMatch.id, rating, 'User feedback')}
                          className="p-1 hover:bg-gray-100 rounded"
                        >
                          <Star className="h-4 w-4 text-yellow-400" />
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* LLM Analysis */}
      {llmAnalysis && (
        <div className="mt-6 p-6 bg-blue-50 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
            <MessageSquare className="h-5 w-5 text-blue-500" />
            <span>LLM Analysis</span>
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Compatibility Score</h4>
              <div className="text-3xl font-bold text-blue-600 mb-2">{llmAnalysis.score}%</div>
              <p className="text-sm text-gray-600">{llmAnalysis.reasoning}</p>
            </div>

            <div>
              <h4 className="font-medium text-gray-900 mb-2">Economic Feasibility</h4>
              <p className="text-sm text-gray-700">{llmAnalysis.economic_feasibility}</p>
            </div>

            <div>
              <h4 className="font-medium text-gray-900 mb-2">Applications</h4>
              <ul className="space-y-1">
                {llmAnalysis.applications.map((app, index) => (
                  <li key={index} className="text-sm text-gray-700 flex items-start space-x-2">
                    <CheckCircle className="h-3 w-3 text-green-500 mt-0.5 flex-shrink-0" />
                    <span>{app}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h4 className="font-medium text-gray-900 mb-2">Environmental Benefits</h4>
              <ul className="space-y-1">
                {llmAnalysis.environmental_benefits.map((benefit, index) => (
                  <li key={index} className="text-sm text-gray-700 flex items-start space-x-2">
                    <Leaf className="h-3 w-3 text-green-500 mt-0.5 flex-shrink-0" />
                    <span>{benefit}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* No Results State */}
      {!loading && matches.length === 0 && selectedMaterial && (
        <div className="text-center py-12">
          <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No AI Matches Found</h3>
          <p className="text-gray-600">Try adjusting your search criteria or preferences.</p>
        </div>
      )}
    </div>
  );
} 