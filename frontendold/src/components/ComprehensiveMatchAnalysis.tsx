import React, { useState, useEffect } from 'react';
import { 
  Calculator, 
  Leaf, 
  Truck, 
  Settings, 
  DollarSign, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  Clock,
  MapPin,
  Package,
  Users,
  Target,
  BarChart3,
  Info,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Star,
  Award,
  Zap,
  Shield,
  Eye,
  EyeOff
} from 'lucide-react';

interface ComprehensiveAnalysis {
  match_id: string;
  buyer_id: string;
  seller_id: string;
  material_type: string;
  quantity_ton: number;
  readiness_assessment: any;
  financial_analysis: any;
  logistics_analysis: any;
  carbon_analysis: any;
  waste_analysis: any;
  overall_score: number;
  match_quality: string;
  confidence_level: number;
  risk_level: string;
  explanations: any;
  recommendations: string[];
  economic_summary: any;
  environmental_summary: any;
  risk_summary: any;
  analysis_date: string;
  methodology: string;
}

interface ComprehensiveMatchAnalysisProps {
  buyerData: any;
  sellerData: any;
  matchData: any;
  onClose?: () => void;
}

const ComprehensiveMatchAnalysis: React.FC<ComprehensiveMatchAnalysisProps> = ({
  buyerData,
  sellerData,
  matchData,
  onClose
}) => {
  const [analysis, setAnalysis] = useState<ComprehensiveAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['overview']));
  const [showDetailedCosts, setShowDetailedCosts] = useState(false);

  useEffect(() => {
    performComprehensiveAnalysis();
  }, [buyerData, sellerData, matchData]);

  const performComprehensiveAnalysis = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/comprehensive-match-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          buyer_data: buyerData,
          seller_data: sellerData,
          match_data: matchData
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to perform comprehensive analysis');
      }

      const result = await response.json();
      setAnalysis(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
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

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-8 max-w-2xl w-full mx-4">
          <div className="flex items-center justify-center space-x-3">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="text-lg font-semibold">Performing Comprehensive Analysis...</span>
          </div>
          <p className="text-gray-600 text-center mt-4">
            Analyzing financial, environmental, and operational aspects...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-8 max-w-2xl w-full mx-4">
          <div className="flex items-center space-x-3 text-red-600 mb-4">
            <AlertTriangle className="h-6 w-6" />
            <span className="text-lg font-semibold">Analysis Failed</span>
          </div>
          <p className="text-gray-600 mb-6">{error}</p>
          <div className="flex space-x-3">
            <button
              onClick={performComprehensiveAnalysis}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Retry Analysis
            </button>
            {onClose && (
              <button
                onClick={onClose}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
              >
                Close
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 overflow-y-auto">
      <div className="bg-white rounded-lg p-6 max-w-6xl w-full mx-4 my-8 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Comprehensive Match Analysis</h2>
            <p className="text-gray-600">
              {buyerData?.name} ↔ {sellerData?.name} • {analysis.material_type}
            </p>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="h-6 w-6" />
            </button>
          )}
        </div>

        {/* Overview Section */}
        <div className="mb-6">
          <div
            className="flex items-center justify-between p-4 bg-gray-50 rounded-lg cursor-pointer"
            onClick={() => toggleSection('overview')}
          >
            <div className="flex items-center space-x-3">
              <BarChart3 className="h-5 w-5 text-blue-600" />
              <h3 className="text-lg font-semibold">Match Overview</h3>
            </div>
            {expandedSections.has('overview') ? (
              <ChevronUp className="h-5 w-5" />
            ) : (
              <ChevronDown className="h-5 w-5" />
            )}
          </div>
          
          {expandedSections.has('overview') && (
            <div className="mt-4 p-4 border rounded-lg">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {formatNumber(analysis.overall_score * 100)}%
                  </div>
                  <div className="text-sm text-gray-600">Overall Score</div>
                </div>
                <div className="text-center">
                  <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getQualityColor(analysis.match_quality)}`}>
                    {analysis.match_quality}
                  </div>
                  <div className="text-sm text-gray-600 mt-1">Match Quality</div>
                </div>
                <div className="text-center">
                  <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getRiskColor(analysis.risk_level)}`}>
                    {analysis.risk_level} Risk
                  </div>
                  <div className="text-sm text-gray-600 mt-1">Risk Level</div>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold mb-2">Material Details</h4>
                  <div className="space-y-1 text-sm">
                    <div>Type: {analysis.material_type}</div>
                    <div>Quantity: {formatNumber(analysis.quantity_ton)} tons</div>
                    <div>Confidence: {formatNumber(analysis.confidence_level * 100)}%</div>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Key Metrics</h4>
                  <div className="space-y-1 text-sm">
                    <div>Economic Value: {formatCurrency(analysis.economic_summary?.total_economic_value || 0)}</div>
                    <div>Carbon Savings: {formatNumber(analysis.environmental_summary?.net_carbon_savings_kg || 0)} kg CO2</div>
                    <div>Payback Period: {formatNumber(analysis.economic_summary?.payback_period_months || 0)} months</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Financial Analysis Section */}
        <div className="mb-6">
          <div
            className="flex items-center justify-between p-4 bg-green-50 rounded-lg cursor-pointer"
            onClick={() => toggleSection('financial')}
          >
            <div className="flex items-center space-x-3">
              <DollarSign className="h-5 w-5 text-green-600" />
              <h3 className="text-lg font-semibold">Financial Analysis</h3>
            </div>
            {expandedSections.has('financial') ? (
              <ChevronUp className="h-5 w-5" />
            ) : (
              <ChevronDown className="h-5 w-5" />
            )}
          </div>
          
          {expandedSections.has('financial') && (
            <div className="mt-4 p-4 border rounded-lg">
              {analysis.financial_analysis && !analysis.financial_analysis.error ? (
                <div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Scenario Comparison */}
                    <div>
                      <h4 className="font-semibold mb-3">Scenario Comparison</h4>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center p-3 bg-blue-50 rounded">
                          <span className="font-medium">Waste Scenario</span>
                          <span className="font-bold text-blue-600">
                            {formatCurrency(analysis.financial_analysis.scenario_comparison?.waste_scenario?.total_cost || 0)}
                          </span>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-green-50 rounded">
                          <span className="font-medium">Fresh Scenario</span>
                          <span className="font-bold text-green-600">
                            {formatCurrency(analysis.financial_analysis.scenario_comparison?.fresh_scenario?.total_cost || 0)}
                          </span>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-yellow-50 rounded border-2 border-yellow-300">
                          <span className="font-medium">Net Savings</span>
                          <span className="font-bold text-yellow-600">
                            {formatCurrency(analysis.financial_analysis.scenario_comparison?.net_savings || 0)}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    {/* Financial Metrics */}
                    <div>
                      <h4 className="font-semibold mb-3">Financial Metrics</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span>Savings Percentage:</span>
                          <span className="font-semibold">
                            {formatNumber(analysis.financial_analysis.scenario_comparison?.savings_percentage || 0)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>ROI:</span>
                          <span className="font-semibold">
                            {formatNumber(analysis.financial_analysis.scenario_comparison?.roi_percentage || 0)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Payback Period:</span>
                          <span className="font-semibold">
                            {formatNumber(analysis.financial_analysis.scenario_comparison?.payback_period_months || 0)} months
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Buyer Savings:</span>
                          <span className="font-semibold text-green-600">
                            {formatCurrency(analysis.financial_analysis.buyer_savings || 0)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Seller Profit:</span>
                          <span className="font-semibold text-blue-600">
                            {formatCurrency(analysis.financial_analysis.seller_profit || 0)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Detailed Cost Breakdown */}
                  <div className="mt-6">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold">Detailed Cost Breakdown</h4>
                      <button
                        onClick={() => setShowDetailedCosts(!showDetailedCosts)}
                        className="flex items-center space-x-2 text-blue-600 hover:text-blue-800"
                      >
                        {showDetailedCosts ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                        <span>{showDetailedCosts ? 'Hide' : 'Show'} Details</span>
                      </button>
                    </div>
                    
                    {showDetailedCosts && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <h5 className="font-medium mb-2">Waste Scenario Costs</h5>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span>Material Cost:</span>
                              <span>{formatCurrency(analysis.financial_analysis.scenario_comparison?.waste_scenario?.material_cost || 0)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Transport:</span>
                              <span>{formatCurrency(analysis.financial_analysis.scenario_comparison?.waste_scenario?.transport_cost || 0)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Processing:</span>
                              <span>{formatCurrency(analysis.financial_analysis.scenario_comparison?.waste_scenario?.processing_cost || 0)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Equipment:</span>
                              <span>{formatCurrency(analysis.financial_analysis.scenario_comparison?.waste_scenario?.equipment_cost || 0)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Carbon Tax:</span>
                              <span>{formatCurrency(analysis.financial_analysis.scenario_comparison?.waste_scenario?.carbon_tax || 0)}</span>
                            </div>
                          </div>
                        </div>
                        <div>
                          <h5 className="font-medium mb-2">Fresh Scenario Costs</h5>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span>Material Cost:</span>
                              <span>{formatCurrency(analysis.financial_analysis.scenario_comparison?.fresh_scenario?.material_cost || 0)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Transport:</span>
                              <span>{formatCurrency(analysis.financial_analysis.scenario_comparison?.fresh_scenario?.transport_cost || 0)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Handling:</span>
                              <span>{formatCurrency(analysis.financial_analysis.scenario_comparison?.fresh_scenario?.handling_cost || 0)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Insurance:</span>
                              <span>{formatCurrency(analysis.financial_analysis.scenario_comparison?.fresh_scenario?.insurance_cost || 0)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Carbon Tax:</span>
                              <span>{formatCurrency(analysis.financial_analysis.scenario_comparison?.fresh_scenario?.carbon_tax || 0)}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-red-600">Financial analysis failed</div>
              )}
            </div>
          )}
        </div>

        {/* Readiness & Refinement Section */}
        <div className="mb-6">
          <div
            className="flex items-center justify-between p-4 bg-purple-50 rounded-lg cursor-pointer"
            onClick={() => toggleSection('readiness')}
          >
            <div className="flex items-center space-x-3">
              <Settings className="h-5 w-5 text-purple-600" />
              <h3 className="text-lg font-semibold">Readiness & Refinement</h3>
            </div>
            {expandedSections.has('readiness') ? (
              <ChevronUp className="h-5 w-5" />
            ) : (
              <ChevronDown className="h-5 w-5" />
            )}
          </div>
          
          {expandedSections.has('readiness') && (
            <div className="mt-4 p-4 border rounded-lg">
              {analysis.readiness_assessment && !analysis.readiness_assessment.error ? (
                <div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold mb-3">Material Readiness</h4>
                      <div className="space-y-3">
                        <div className="flex items-center space-x-3">
                          {analysis.readiness_assessment.is_ready_for_use ? (
                            <CheckCircle className="h-5 w-5 text-green-600" />
                          ) : (
                            <AlertTriangle className="h-5 w-5 text-yellow-600" />
                          )}
                          <span className="font-medium">
                            {analysis.readiness_assessment.is_ready_for_use ? 'Ready for Use' : 'Refinement Required'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Readiness Score:</span>
                          <span className="font-semibold">
                            {formatNumber(analysis.readiness_assessment.readiness_score * 100)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Equipment Available:</span>
                          <span className="font-semibold">
                            {analysis.readiness_assessment.buyer_equipment_available ? 'Yes' : 'No'}
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold mb-3">Refinement Requirements</h4>
                      {analysis.readiness_assessment.refinement_requirements ? (
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span>Processing Cost:</span>
                            <span className="font-semibold">
                              {formatCurrency(analysis.readiness_assessment.refinement_requirements.estimated_cost_per_ton)}/ton
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Processing Time:</span>
                            <span className="font-semibold">
                              {analysis.readiness_assessment.refinement_requirements.processing_time_days} days
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Equipment Cost:</span>
                            <span className="font-semibold">
                              {formatCurrency(analysis.readiness_assessment.total_equipment_cost)}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Payback Period:</span>
                            <span className="font-semibold">
                              {formatNumber(analysis.readiness_assessment.payback_period_months)} months
                            </span>
                          </div>
                        </div>
                      ) : (
                        <p className="text-gray-600">No refinement required</p>
                      )}
                    </div>
                  </div>
                  
                  {/* Equipment Recommendations */}
                  {analysis.readiness_assessment.equipment_recommendations?.length > 0 && (
                    <div className="mt-6">
                      <h4 className="font-semibold mb-3">Equipment Recommendations</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {analysis.readiness_assessment.equipment_recommendations.map((equipment: any, index: number) => (
                          <div key={index} className="p-3 border rounded-lg">
                            <div className="font-medium">{equipment.equipment_type}</div>
                            <div className="text-sm text-gray-600">{equipment.manufacturer} {equipment.model}</div>
                            <div className="flex justify-between mt-2 text-sm">
                              <span>Cost:</span>
                              <span className="font-semibold">{formatCurrency(equipment.purchase_cost)}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span>ROI:</span>
                              <span className="font-semibold">{formatNumber(equipment.roi_analysis?.roi_percentage || 0)}%</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-red-600">Readiness analysis failed</div>
              )}
            </div>
          )}
        </div>

        {/* Logistics Analysis Section */}
        <div className="mb-6">
          <div
            className="flex items-center justify-between p-4 bg-orange-50 rounded-lg cursor-pointer"
            onClick={() => toggleSection('logistics')}
          >
            <div className="flex items-center space-x-3">
              <Truck className="h-5 w-5 text-orange-600" />
              <h3 className="text-lg font-semibold">Logistics Analysis</h3>
            </div>
            {expandedSections.has('logistics') ? (
              <ChevronUp className="h-5 w-5" />
            ) : (
              <ChevronDown className="h-5 w-5" />
            )}
          </div>
          
          {expandedSections.has('logistics') && (
            <div className="mt-4 p-4 border rounded-lg">
              {analysis.logistics_analysis && !analysis.logistics_analysis.error ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-3">Transport Details</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Distance:</span>
                        <span className="font-semibold">{formatNumber(analysis.logistics_analysis.distance_km)} km</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Transit Time:</span>
                        <span className="font-semibold">{formatNumber(analysis.logistics_analysis.transit_days)} days</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Transport Cost:</span>
                        <span className="font-semibold">{formatCurrency(analysis.logistics_analysis.transport_cost)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Carbon Impact:</span>
                        <span className="font-semibold">{formatNumber(analysis.logistics_analysis.carbon_kg)} kg CO2</span>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-semibold mb-3">Route Options</h4>
                    {analysis.logistics_analysis.route_options?.slice(0, 3).map((route: any, index: number) => (
                      <div key={index} className="mb-3 p-3 border rounded">
                        <div className="font-medium">Route {index + 1}</div>
                        <div className="text-sm space-y-1">
                          <div>Cost: {formatCurrency(route.total_cost)}</div>
                          <div>Duration: {formatNumber(route.total_duration)} hours</div>
                          <div>Carbon: {formatNumber(route.total_carbon)} kg CO2</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-red-600">Logistics analysis failed</div>
              )}
            </div>
          )}
        </div>

        {/* Environmental Analysis Section */}
        <div className="mb-6">
          <div
            className="flex items-center justify-between p-4 bg-green-50 rounded-lg cursor-pointer"
            onClick={() => toggleSection('environmental')}
          >
            <div className="flex items-center space-x-3">
              <Leaf className="h-5 w-5 text-green-600" />
              <h3 className="text-lg font-semibold">Environmental Impact</h3>
            </div>
            {expandedSections.has('environmental') ? (
              <ChevronUp className="h-5 w-5" />
            ) : (
              <ChevronDown className="h-5 w-5" />
            )}
          </div>
          
          {expandedSections.has('environmental') && (
            <div className="mt-4 p-4 border rounded-lg">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold mb-3">Carbon Analysis</h4>
                  {analysis.carbon_analysis && !analysis.carbon_analysis.error ? (
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Net Carbon Savings:</span>
                        <span className="font-semibold text-green-600">
                          {formatNumber(analysis.carbon_analysis.net_carbon_savings_kg)} kg CO2
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Material Savings:</span>
                        <span className="font-semibold">
                          {formatNumber(analysis.carbon_analysis.material_carbon_savings_kg)} kg CO2
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Transport Impact:</span>
                        <span className="font-semibold">
                          {formatNumber(analysis.carbon_analysis.transport_carbon_kg)} kg CO2
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Savings %:</span>
                        <span className="font-semibold">
                          {formatNumber(analysis.carbon_analysis.carbon_savings_percentage)}%
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-red-600">Carbon analysis failed</div>
                  )}
                </div>
                
                <div>
                  <h4 className="font-semibold mb-3">Waste Management</h4>
                  {analysis.waste_analysis && !analysis.waste_analysis.error ? (
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Management Efficiency:</span>
                        <span className="font-semibold">
                          {formatNumber(analysis.waste_analysis.waste_management_efficiency * 100)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Recycling Potential:</span>
                        <span className="font-semibold">
                          {formatNumber(analysis.waste_analysis.recycling_potential * 100)}%
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-red-600">Waste analysis failed</div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Recommendations Section */}
        <div className="mb-6">
          <div
            className="flex items-center justify-between p-4 bg-blue-50 rounded-lg cursor-pointer"
            onClick={() => toggleSection('recommendations')}
          >
            <div className="flex items-center space-x-3">
              <Target className="h-5 w-5 text-blue-600" />
              <h3 className="text-lg font-semibold">Recommendations</h3>
            </div>
            {expandedSections.has('recommendations') ? (
              <ChevronUp className="h-5 w-5" />
            ) : (
              <ChevronDown className="h-5 w-5" />
            )}
          </div>
          
          {expandedSections.has('recommendations') && (
            <div className="mt-4 p-4 border rounded-lg">
              <div className="space-y-3">
                {analysis.recommendations?.map((recommendation, index) => (
                  <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                    <ArrowRight className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <span className="text-sm">{recommendation}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Analysis Metadata */}
        <div className="text-xs text-gray-500 border-t pt-4">
          <div className="flex justify-between">
            <span>Analysis Date: {new Date(analysis.analysis_date).toLocaleString()}</span>
            <span>Methodology: {analysis.methodology}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComprehensiveMatchAnalysis; 