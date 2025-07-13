import React, { useState, useEffect } from 'react';
import { 
  Calculator, 
  DollarSign, 
  TrendingUp, 
  TrendingDown, 
  ArrowRight,
  CheckCircle,
  AlertTriangle,
  Info,
  Eye,
  EyeOff,
  BarChart3,
  PieChart,
  Target,
  Clock,
  MapPin,
  Truck,
  Package,
  Shield,
  Zap
} from 'lucide-react';

interface CostBreakdownProps {
  buyerData: any;
  sellerData: any;
  matchData: any;
  onClose?: () => void;
}

const DetailedCostBreakdown: React.FC<CostBreakdownProps> = ({
  buyerData,
  sellerData,
  matchData,
  onClose
}) => {
  const [costData, setCostData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showDetailedCosts, setShowDetailedCosts] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadCostBreakdown();
  }, [buyerData, sellerData, matchData]);

  const loadCostBreakdown = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/cost-breakdown', {
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
        throw new Error('Failed to load cost breakdown');
      }

      const result = await response.json();
      setCostData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cost breakdown');
    } finally {
      setLoading(false);
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

  const getSavingsColor = (savings: number) => {
    return savings > 0 ? 'text-green-600' : 'text-red-600';
  };

  const getSavingsIcon = (savings: number) => {
    return savings > 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />;
  };

  if (loading) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-8 max-w-2xl w-full mx-4">
          <div className="flex items-center justify-center space-x-3">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="text-lg font-semibold">Calculating Cost Breakdown...</span>
          </div>
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
            <span className="text-lg font-semibold">Cost Analysis Failed</span>
          </div>
          <p className="text-gray-600 mb-6">{error}</p>
          <div className="flex space-x-3">
            <button
              onClick={loadCostBreakdown}
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

  if (!costData) {
    return null;
  }

  const financialBreakdown = costData.financial_breakdown;
  const logisticsBreakdown = costData.logistics_breakdown;
  const refinementBreakdown = costData.refinement_breakdown;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 overflow-y-auto">
      <div className="bg-white rounded-lg p-6 max-w-6xl w-full mx-4 my-8 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Detailed Cost Breakdown</h2>
            <p className="text-gray-600">
              {buyerData?.name} ↔ {sellerData?.name} • {matchData?.material_type}
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

        {/* Tabs */}
        <div className="border-b border-gray-200 mb-6">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'overview', label: 'Overview', icon: BarChart3 },
              { id: 'scenarios', label: 'Scenario Comparison', icon: Calculator },
              { id: 'logistics', label: 'Logistics', icon: Truck },
              { id: 'refinement', label: 'Refinement', icon: Settings },
              { id: 'roi', label: 'ROI Analysis', icon: TrendingUp }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="h-4 w-4" />
                <span>{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white p-4 rounded-lg border">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Total Cost</p>
                    <p className="text-2xl font-bold">
                      {formatCurrency(costData.summary?.total_cost || 0)}
                    </p>
                  </div>
                  <DollarSign className="h-8 w-8 text-blue-600" />
                </div>
              </div>
              
              <div className="bg-white p-4 rounded-lg border">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Net Savings</p>
                    <p className={`text-2xl font-bold ${getSavingsColor(costData.summary?.net_savings || 0)}`}>
                      {formatCurrency(costData.summary?.net_savings || 0)}
                    </p>
                  </div>
                  <div className={getSavingsColor(costData.summary?.net_savings || 0)}>
                    {getSavingsIcon(costData.summary?.net_savings || 0)}
                  </div>
                </div>
              </div>
              
              <div className="bg-white p-4 rounded-lg border">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Payback Period</p>
                    <p className="text-2xl font-bold">
                      {formatNumber(costData.summary?.payback_period || 0)} mo
                    </p>
                  </div>
                  <Clock className="h-8 w-8 text-green-600" />
                </div>
              </div>
              
              <div className="bg-white p-4 rounded-lg border">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">ROI</p>
                    <p className="text-2xl font-bold">
                      {formatNumber(costData.summary?.roi_percentage || 0)}%
                    </p>
                  </div>
                  <TrendingUp className="h-8 w-8 text-purple-600" />
                </div>
              </div>
            </div>

            {/* Cost Breakdown Chart */}
            <div className="bg-white p-6 rounded-lg border">
              <h3 className="text-lg font-semibold mb-4">Cost Distribution</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-medium mb-3">Waste Scenario</h4>
                  <div className="space-y-2">
                    {financialBreakdown?.scenario_comparison?.waste_scenario && (
                      Object.entries(financialBreakdown.scenario_comparison.waste_scenario)
                        .filter(([key, value]) => key.includes('cost') && typeof value === 'number' && value > 0)
                        .map(([key, value]) => (
                          <div key={key} className="flex justify-between items-center">
                            <span className="text-sm text-gray-600 capitalize">
                              {key.replace('_cost', '').replace(/_/g, ' ')}:
                            </span>
                            <span className="font-semibold">{formatCurrency(value as number)}</span>
                          </div>
                        ))
                    )}
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-3">Fresh Scenario</h4>
                  <div className="space-y-2">
                    {financialBreakdown?.scenario_comparison?.fresh_scenario && (
                      Object.entries(financialBreakdown.scenario_comparison.fresh_scenario)
                        .filter(([key, value]) => key.includes('cost') && typeof value === 'number' && value > 0)
                        .map(([key, value]) => (
                          <div key={key} className="flex justify-between items-center">
                            <span className="text-sm text-gray-600 capitalize">
                              {key.replace('_cost', '').replace(/_/g, ' ')}:
                            </span>
                            <span className="font-semibold">{formatCurrency(value as number)}</span>
                          </div>
                        ))
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Scenario Comparison Tab */}
        {activeTab === 'scenarios' && (
          <div className="space-y-6">
            {financialBreakdown?.scenario_comparison ? (
              <div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Waste Scenario */}
                  <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
                    <div className="flex items-center space-x-2 mb-4">
                      <Package className="h-5 w-5 text-blue-600" />
                      <h3 className="text-lg font-semibold text-blue-900">Waste Scenario</h3>
                    </div>
                    
                    <div className="space-y-3">
                      <div className="flex justify-between items-center p-3 bg-white rounded border">
                        <span className="font-medium">Material Cost</span>
                        <span className="font-bold text-blue-600">
                          {formatCurrency(financialBreakdown.scenario_comparison.waste_scenario.material_cost)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-white rounded border">
                        <span className="font-medium">Transport Cost</span>
                        <span className="font-bold text-blue-600">
                          {formatCurrency(financialBreakdown.scenario_comparison.waste_scenario.transport_cost)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-white rounded border">
                        <span className="font-medium">Processing Cost</span>
                        <span className="font-bold text-blue-600">
                          {formatCurrency(financialBreakdown.scenario_comparison.waste_scenario.processing_cost)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-white rounded border">
                        <span className="font-medium">Equipment Cost</span>
                        <span className="font-bold text-blue-600">
                          {formatCurrency(financialBreakdown.scenario_comparison.waste_scenario.equipment_cost)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-blue-100 rounded border-2 border-blue-300">
                        <span className="font-bold">Total Cost</span>
                        <span className="font-bold text-blue-800">
                          {formatCurrency(financialBreakdown.scenario_comparison.waste_scenario.total_cost)}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Fresh Scenario */}
                  <div className="bg-green-50 p-6 rounded-lg border border-green-200">
                    <div className="flex items-center space-x-2 mb-4">
                      <Target className="h-5 w-5 text-green-600" />
                      <h3 className="text-lg font-semibold text-green-900">Fresh Scenario</h3>
                    </div>
                    
                    <div className="space-y-3">
                      <div className="flex justify-between items-center p-3 bg-white rounded border">
                        <span className="font-medium">Material Cost</span>
                        <span className="font-bold text-green-600">
                          {formatCurrency(financialBreakdown.scenario_comparison.fresh_scenario.material_cost)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-white rounded border">
                        <span className="font-medium">Transport Cost</span>
                        <span className="font-bold text-green-600">
                          {formatCurrency(financialBreakdown.scenario_comparison.fresh_scenario.transport_cost)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-white rounded border">
                        <span className="font-medium">Handling Cost</span>
                        <span className="font-bold text-green-600">
                          {formatCurrency(financialBreakdown.scenario_comparison.fresh_scenario.handling_cost)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-white rounded border">
                        <span className="font-medium">Insurance Cost</span>
                        <span className="font-bold text-green-600">
                          {formatCurrency(financialBreakdown.scenario_comparison.fresh_scenario.insurance_cost)}
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center p-3 bg-green-100 rounded border-2 border-green-300">
                        <span className="font-bold">Total Cost</span>
                        <span className="font-bold text-green-800">
                          {formatCurrency(financialBreakdown.scenario_comparison.fresh_scenario.total_cost)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Comparison Summary */}
                <div className="mt-6 bg-yellow-50 p-6 rounded-lg border border-yellow-200">
                  <div className="flex items-center space-x-2 mb-4">
                    <ArrowRight className="h-5 w-5 text-yellow-600" />
                    <h3 className="text-lg font-semibold text-yellow-900">Comparison Summary</h3>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className={`text-2xl font-bold ${getSavingsColor(financialBreakdown.scenario_comparison.net_savings)}`}>
                        {formatCurrency(financialBreakdown.scenario_comparison.net_savings)}
                      </div>
                      <div className="text-sm text-gray-600">Net Savings</div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {formatNumber(financialBreakdown.scenario_comparison.savings_percentage)}%
                      </div>
                      <div className="text-sm text-gray-600">Savings Percentage</div>
                    </div>
                    
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {formatNumber(financialBreakdown.scenario_comparison.roi_percentage)}%
                      </div>
                      <div className="text-sm text-gray-600">ROI</div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-red-600">Financial breakdown not available</div>
            )}
          </div>
        )}

        {/* Logistics Tab */}
        {activeTab === 'logistics' && (
          <div className="space-y-6">
            {logisticsBreakdown ? (
              <div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Transport Details</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Distance:</span>
                        <span className="font-semibold">{formatNumber(logisticsBreakdown.distance_km)} km</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Transit Time:</span>
                        <span className="font-semibold">{formatNumber(logisticsBreakdown.transit_days)} days</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Transport Cost:</span>
                        <span className="font-semibold">{formatCurrency(logisticsBreakdown.transport_cost)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Carbon Impact:</span>
                        <span className="font-semibold">{formatNumber(logisticsBreakdown.carbon_kg)} kg CO2</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Route Options</h3>
                    <div className="space-y-3">
                      {logisticsBreakdown.route_options?.slice(0, 3).map((route: any, index: number) => (
                        <div key={index} className="p-3 border rounded">
                          <div className="font-medium mb-2">Route {index + 1}</div>
                          <div className="grid grid-cols-2 gap-2 text-sm">
                            <div>Cost: {formatCurrency(route.total_cost)}</div>
                            <div>Duration: {formatNumber(route.total_duration)}h</div>
                            <div>Carbon: {formatNumber(route.total_carbon)} kg</div>
                            <div>Reliability: {formatNumber(route.reliability_score * 100)}%</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-red-600">Logistics breakdown not available</div>
            )}
          </div>
        )}

        {/* Refinement Tab */}
        {activeTab === 'refinement' && (
          <div className="space-y-6">
            {refinementBreakdown ? (
              <div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Refinement Requirements</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Refinement Required:</span>
                        <span className="font-semibold">
                          {refinementBreakdown.refinement_required ? 'Yes' : 'No'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Processing Cost:</span>
                        <span className="font-semibold">
                          {formatCurrency(refinementBreakdown.refinement_requirements?.estimated_cost_per_ton || 0)}/ton
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Processing Time:</span>
                        <span className="font-semibold">
                          {refinementBreakdown.refinement_requirements?.processing_time_days || 0} days
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Equipment Cost:</span>
                        <span className="font-semibold">
                          {formatCurrency(refinementBreakdown.total_equipment_cost || 0)}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Equipment Recommendations</h3>
                    <div className="space-y-3">
                      {refinementBreakdown.equipment_recommendations?.map((equipment: any, index: number) => (
                        <div key={index} className="p-3 border rounded">
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
                </div>
              </div>
            ) : (
              <div className="text-red-600">Refinement breakdown not available</div>
            )}
          </div>
        )}

        {/* ROI Analysis Tab */}
        {activeTab === 'roi' && (
          <div className="space-y-6">
            {financialBreakdown?.scenario_comparison ? (
              <div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">ROI Metrics</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Payback Period:</span>
                        <span className="font-semibold">
                          {formatNumber(financialBreakdown.scenario_comparison.payback_period_months)} months
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">ROI Percentage:</span>
                        <span className="font-semibold text-green-600">
                          {formatNumber(financialBreakdown.scenario_comparison.roi_percentage)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Break-even Quantity:</span>
                        <span className="font-semibold">
                          {formatNumber(financialBreakdown.scenario_comparison.break_even_quantity)} tons
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-white p-6 rounded-lg border">
                    <h3 className="text-lg font-semibold mb-4">Risk Assessment</h3>
                    <div className="space-y-3">
                      {financialBreakdown.scenario_comparison.risk_assessment?.risk_factors?.map((risk: string, index: number) => (
                        <div key={index} className="flex items-center space-x-2">
                          <AlertTriangle className="h-4 w-4 text-yellow-600" />
                          <span className="text-sm">{risk}</span>
                        </div>
                      ))}
                      <div className="mt-4 p-3 bg-gray-50 rounded">
                        <div className="text-sm font-medium">Overall Risk Level:</div>
                        <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium mt-1 ${
                          financialBreakdown.scenario_comparison.risk_assessment?.overall_risk_level === 'low' ? 'text-green-600 bg-green-50' :
                          financialBreakdown.scenario_comparison.risk_assessment?.overall_risk_level === 'medium' ? 'text-yellow-600 bg-yellow-50' :
                          'text-red-600 bg-red-50'
                        }`}>
                          {financialBreakdown.scenario_comparison.risk_assessment?.overall_risk_level} Risk
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-red-600">ROI analysis not available</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DetailedCostBreakdown; 