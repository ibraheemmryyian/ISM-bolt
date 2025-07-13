import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Filter, 
  TrendingUp, 
  DollarSign, 
  Leaf, 
  Settings, 
  Users, 
  Target,
  BarChart3,
  Star,
  Award,
  Zap,
  Shield,
  Eye,
  EyeOff,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  CheckCircle,
  AlertTriangle,
  Clock,
  MapPin,
  Package,
  Calculator,
  Truck,
  Info,
  Loader2
} from 'lucide-react';
import ComprehensiveMatchAnalysis from './ComprehensiveMatchAnalysis';

interface Match {
  id: string;
  company_name: string;
  industry: string;
  match_score: number;
  potential_savings: number;
  description: string;
  comprehensive_analysis?: any;
}

interface EnhancedMatchingInterfaceProps {
  userId?: string;
}

const EnhancedMatchingInterface: React.FC<EnhancedMatchingInterfaceProps> = ({ userId = 'demo-user' }) => {
  const [matches, setMatches] = useState<Match[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMatch, setSelectedMatch] = useState<Match | null>(null);
  const [showComprehensiveAnalysis, setShowComprehensiveAnalysis] = useState(false);
  const [filters, setFilters] = useState({
    minScore: 0,
    minSavings: 0,
    minCarbonSavings: 0,
    quality: 'all',
    riskLevel: 'all'
  });
  const [sortBy, setSortBy] = useState('score');
  const [expandedMatch, setExpandedMatch] = useState<string | null>(null);
  const [showDetailedMetrics, setShowDetailedMetrics] = useState(false);

  useEffect(() => {
    loadMatches();
  }, []);

  const loadMatches = async () => {
    try {
      setLoading(true);
      setError(null);

      // Get user's company data
      const userResponse = await fetch(`/api/companies/${userId}`);
      const userData = await userResponse.json();

      // Get enhanced matches with comprehensive analysis
      const response = await fetch('/api/enhanced-matching', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          buyer_data: userData,
          seller_data: null, // Will get all potential sellers
          preferences: {
            min_score: filters.minScore,
            min_savings: filters.minSavings,
            min_carbon_savings: filters.minCarbonSavings
          }
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to load matches');
      }

      const result = await response.json();
      setMatches(result.matches || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load matches');
    } finally {
      setLoading(false);
    }
  };

  const toggleMatchExpansion = (matchId: string) => {
    setExpandedMatch(expandedMatch === matchId ? null : matchId);
  };

  const openComprehensiveAnalysis = (match: Match) => {
    setSelectedMatch(match);
    setShowComprehensiveAnalysis(true);
  };

  const getQualityColor = (quality: string) => {
    switch (quality.toLowerCase()) {
      case 'excellent': return 'text-green-600 bg-green-50';
      case 'good': return 'text-blue-600 bg-blue-50';
      case 'moderate': return 'text-yellow-600 bg-yellow-50';
      case 'poor': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'low': return 'text-green-600 bg-green-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'high': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'EUR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatNumber = (num: number, decimals = 1) => {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: 0,
      maximumFractionDigits: decimals,
    }).format(num);
  };

  const filteredAndSortedMatches = matches
    .filter(match => {
      const analysis = match.comprehensive_analysis;
      if (!analysis || analysis.error) return false;
      
      return (
        (analysis.overall_score * 100) >= filters.minScore &&
        (analysis.economic_summary?.total_economic_value || 0) >= filters.minSavings &&
        (analysis.environmental_summary?.net_carbon_savings_kg || 0) >= filters.minCarbonSavings &&
        (filters.quality === 'all' || analysis.match_quality.toLowerCase() === filters.quality) &&
        (filters.riskLevel === 'all' || analysis.risk_level.toLowerCase() === filters.riskLevel)
      );
    })
    .sort((a, b) => {
      const analysisA = a.comprehensive_analysis;
      const analysisB = b.comprehensive_analysis;
      
      switch (sortBy) {
        case 'score':
          return (analysisB?.overall_score || 0) - (analysisA?.overall_score || 0);
        case 'savings':
          return (analysisB?.economic_summary?.total_economic_value || 0) - (analysisA?.economic_summary?.total_economic_value || 0);
        case 'carbon':
          return (analysisB?.environmental_summary?.net_carbon_savings_kg || 0) - (analysisA?.environmental_summary?.net_carbon_savings_kg || 0);
        case 'quality':
          const qualityOrder = { 'excellent': 4, 'good': 3, 'moderate': 2, 'poor': 1 };
          return (qualityOrder[analysisB?.match_quality?.toLowerCase()] || 0) - (qualityOrder[analysisA?.match_quality?.toLowerCase()] || 0);
        default:
          return 0;
      }
    });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Target className="h-8 w-8 text-blue-600" />
                <h1 className="text-2xl font-bold text-gray-900">Enhanced Matching</h1>
              </div>
              <div className="text-sm text-gray-500">
                {filteredAndSortedMatches.length} matches found
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowDetailedMetrics(!showDetailedMetrics)}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                {showDetailedMetrics ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                <span>{showDetailedMetrics ? 'Hide' : 'Show'} Details</span>
              </button>
              <button
                onClick={loadMatches}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                Refresh Matches
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Filters and Controls */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Min Score (%)
              </label>
              <input
                type="number"
                value={filters.minScore}
                onChange={(e) => setFilters({...filters, minScore: parseInt(e.target.value) || 0})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Min Savings (€)
              </label>
              <input
                type="number"
                value={filters.minSavings}
                onChange={(e) => setFilters({...filters, minSavings: parseInt(e.target.value) || 0})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Min Carbon Savings (kg)
              </label>
              <input
                type="number"
                value={filters.minCarbonSavings}
                onChange={(e) => setFilters({...filters, minCarbonSavings: parseInt(e.target.value) || 0})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Quality
              </label>
              <select
                value={filters.quality}
                onChange={(e) => setFilters({...filters, quality: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Qualities</option>
                <option value="excellent">Excellent</option>
                <option value="good">Good</option>
                <option value="moderate">Moderate</option>
                <option value="poor">Poor</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Risk Level
              </label>
              <select
                value={filters.riskLevel}
                onChange={(e) => setFilters({...filters, riskLevel: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Risks</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Sort By
              </label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="score">Score</option>
                <option value="savings">Savings</option>
                <option value="carbon">Carbon</option>
                <option value="quality">Quality</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <Loader2 className="h-6 w-6 animate-spin mx-auto my-4" />
      )}

      {/* Error State */}
      {error && (
        <div className="bg-red-100 text-red-700 p-2 rounded mb-4 text-center">{error}</div>
      )}

      {/* Matches Grid */}
      {!loading && !error && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {filteredAndSortedMatches.map((match) => {
              const analysis = match.comprehensive_analysis;
              if (!analysis || analysis.error) return null;

              return (
                <div key={match.id} className="bg-white rounded-lg shadow-sm border hover:shadow-md transition-shadow">
                  {/* Match Header */}
                  <div className="p-6 border-b">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">{match.company_name}</h3>
                        <p className="text-sm text-gray-600">{match.industry}</p>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-blue-600">
                          {formatNumber(analysis.overall_score * 100)}%
                        </div>
                        <div className="text-sm text-gray-500">Overall Score</div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2 mb-3">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getQualityColor(analysis.match_quality)}`}>
                        {analysis.match_quality}
                      </span>
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(analysis.risk_level)}`}>
                        {analysis.risk_level} Risk
                      </span>
                    </div>
                    
                    <p className="text-sm text-gray-600">{match.description}</p>
                  </div>

                  {/* Key Metrics */}
                  <div className="p-6">
                    <div className="grid grid-cols-3 gap-4 mb-4">
                      <div className="text-center">
                        <div className="text-lg font-bold text-green-600">
                          {formatCurrency(analysis.economic_summary?.total_economic_value || 0)}
                        </div>
                        <div className="text-xs text-gray-500">Economic Value</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-green-600">
                          {formatNumber(analysis.environmental_summary?.net_carbon_savings_kg || 0)}
                        </div>
                        <div className="text-xs text-gray-500">CO2 Saved</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-blue-600">
                          {formatNumber(analysis.economic_summary?.payback_period_months || 0)}
                        </div>
                        <div className="text-xs text-gray-500">Payback (mo)</div>
                      </div>
                    </div>

                    {/* Detailed Metrics (Collapsible) */}
                    {showDetailedMetrics && (
                      <div className="border-t pt-4">
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Buyer Savings:</span>
                              <span className="font-semibold text-green-600">
                                {formatCurrency(analysis.economic_summary?.buyer_savings || 0)}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Seller Profit:</span>
                              <span className="font-semibold text-blue-600">
                                {formatCurrency(analysis.economic_summary?.seller_profit || 0)}
                              </span>
                            </div>
                          </div>
                          <div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">ROI:</span>
                              <span className="font-semibold">
                                {formatNumber(analysis.economic_summary?.roi_percentage || 0)}%
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Confidence:</span>
                              <span className="font-semibold">
                                {formatNumber(analysis.confidence_level * 100)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex space-x-2 mt-4">
                      <button
                        onClick={() => toggleMatchExpansion(match.id)}
                        className="flex-1 px-3 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 text-sm"
                      >
                        {expandedMatch === match.id ? (
                          <>
                            <ChevronUp className="h-4 w-4 inline mr-1" />
                            Hide Details
                          </>
                        ) : (
                          <>
                            <ChevronDown className="h-4 w-4 inline mr-1" />
                            Show Details
                          </>
                        )}
                      </button>
                      <button
                        onClick={() => openComprehensiveAnalysis(match)}
                        className="flex-1 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
                      >
                        <BarChart3 className="h-4 w-4 inline mr-1" />
                        Full Analysis
                      </button>
                    </div>
                  </div>

                  {/* Expanded Details */}
                  {expandedMatch === match.id && (
                    <div className="border-t p-6 bg-gray-50">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <div>
                          <h4 className="font-semibold mb-2">Financial Summary</h4>
                          <div className="space-y-1">
                            <div className="flex justify-between">
                              <span>Net Savings:</span>
                              <span className="font-semibold text-green-600">
                                {formatCurrency(analysis.economic_summary?.net_savings || 0)}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span>Savings %:</span>
                              <span className="font-semibold">
                                {formatNumber(analysis.economic_summary?.savings_percentage || 0)}%
                              </span>
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold mb-2">Environmental Summary</h4>
                          <div className="space-y-1">
                            <div className="flex justify-between">
                              <span>Carbon Savings:</span>
                              <span className="font-semibold text-green-600">
                                {formatNumber(analysis.environmental_summary?.net_carbon_savings_kg || 0)} kg
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span>Waste Efficiency:</span>
                              <span className="font-semibold">
                                {formatNumber(analysis.environmental_summary?.waste_management_efficiency * 100 || 0)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Top Recommendations */}
                      <div className="mt-4">
                        <h4 className="font-semibold mb-2">Key Recommendations</h4>
                        <div className="space-y-1">
                          {analysis.recommendations?.slice(0, 3).map((rec: string, index: number) => (
                            <div key={index} className="flex items-start space-x-2 text-sm">
                              <ArrowRight className="h-3 w-3 text-blue-600 mt-0.5 flex-shrink-0" />
                              <span>{rec}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* No Matches */}
          {filteredAndSortedMatches.length === 0 && (
            <div className="text-center py-12">
              <Target className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No matches found</h3>
              <p className="text-gray-600">Try adjusting your filters or criteria to find more matches.</p>
            </div>
          )}
        </div>
      )}

      {/* Comprehensive Analysis Modal */}
      {showComprehensiveAnalysis && selectedMatch && (
        <ComprehensiveMatchAnalysis
          buyerData={{ id: userId }}
          sellerData={{ id: selectedMatch.id, name: selectedMatch.company_name }}
          matchData={selectedMatch}
          onClose={() => setShowComprehensiveAnalysis(false)}
        />
      )}
    </div>
  );
};

export default EnhancedMatchingInterface; 