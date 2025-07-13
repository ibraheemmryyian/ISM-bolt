import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import AuthenticatedLayout from './AuthenticatedLayout';
import { 
  Brain, 
  Users, 
  Package, 
  Truck, 
  Leaf, 
  TrendingUp,
  Target,
  ArrowRight,
  ArrowLeft,
  CheckCircle,
  Clock,
  DollarSign,
  Zap,
  AlertCircle,
  Loader2,
  Search,
  Filter,
  MapPin,
  Calculator,
  BarChart3,
  Award,
  Globe,
  Factory,
  Recycle,
  Star
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface AIMaterial {
  id: string;
  name: string;
  category: string;
  description: string;
  quantity: string;
  frequency: string;
  potential_value: string;
  quality_grade: string;
  potential_uses: string[];
  symbiosis_opportunities: string[];
  ai_generated: boolean;
  sustainability_score: number;
}

interface AIMatch {
  id: string;
  company_name: string;
  company_type: string;
  location: string;
  match_score: number;
  material_match: string;
  potential_savings: string;
  carbon_reduction: string;
  implementation_time: string;
  contact_info: string;
  ai_generated: boolean;
  sustainability_impact: string;
  risk_level: 'low' | 'medium' | 'high';
  partnership_type: string;
}

interface LogisticsPreview {
  origin: string;
  destination: string;
  material: string;
  weight_kg: number;
  transport_modes: {
    mode: string;
    cost: number;
    transit_time: number;
    carbon_emissions: number;
    reliability: number;
    sustainability_score: number;
  }[];
  total_cost: number;
  total_carbon: number;
  cost_breakdown: {
    transport: number;
    handling: number;
    customs: number;
    insurance: number;
  };
  recommendations: string[];
  is_feasible: boolean;
  roi_percentage: number;
  payback_period: string;
}

interface AIAnalysis {
  symbiosis_score: number;
  estimated_savings: string;
  carbon_reduction: string;
  top_opportunities: string[];
  recommended_partners: string[];
  implementation_roadmap: string[];
  market_analysis: {
    demand_trend: string;
    price_trend: string;
    competition_level: string;
  };
}

const DemoAIMatching: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<'inference' | 'matching' | 'logistics'>('inference');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [companyProfile, setCompanyProfile] = useState<any>(null);
  const [aiMaterials, setAiMaterials] = useState<AIMaterial[]>([]);
  const [aiMatches, setAiMatches] = useState<AIMatch[]>([]);
  const [selectedMatch, setSelectedMatch] = useState<AIMatch | null>(null);
  const [logisticsPreview, setLogisticsPreview] = useState<LogisticsPreview | null>(null);
  const [aiAnalysis, setAiAnalysis] = useState<AIAnalysis | null>(null);
  const [activeTab, setActiveTab] = useState('materials');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterCategory, setFilterCategory] = useState('all');
  
  const navigate = useNavigate();

  useEffect(() => {
    loadDemoData();
  }, []);

  const loadDemoData = async () => {
    try {
      setIsLoading(true);
      
      // Real AI analysis required - no demo data allowed
      throw new Error('❌ Demo data not allowed. Real AI analysis required for planetary moat.');
      
    } catch (error) {
      console.error('Error loading data:', error);
      setError('Real AI analysis required. Please ensure all APIs are configured.');
      setIsLoading(false);
    }
  };

  const generateLogisticsPreview = async (match: AIMatch) => {
    try {
      setIsLoading(true);
      setSelectedMatch(match);

      setTimeout(() => {
        setLogisticsPreview({
          origin: 'Dubai, UAE',
          destination: match.location,
          material: match.material_match,
          weight_kg: 50000,
          transport_modes: [
            {
              mode: 'Truck',
              cost: 8500,
              transit_time: 2,
              carbon_emissions: 750,
              reliability: 0.95,
              sustainability_score: 65
            },
            {
              mode: 'Sea',
              cost: 6200,
              transit_time: 5,
              carbon_emissions: 400,
              reliability: 0.90,
              sustainability_score: 85
            },
            {
              mode: 'Rail',
              cost: 7200,
              transit_time: 3,
              carbon_emissions: 300,
              reliability: 0.92,
              sustainability_score: 90
            }
          ],
          total_cost: 8500,
          total_carbon: 750,
          cost_breakdown: {
            transport: 6800,
            handling: 1020,
            customs: 510,
            insurance: 170
          },
          recommendations: [
            'Truck transport offers best cost-benefit ratio for this distance',
            'Consider bulk shipping for quantities over 100 tons',
            'Negotiate long-term contracts for better rates',
            'Explore carbon offset options to improve sustainability score'
          ],
          is_feasible: true,
          roi_percentage: 156,
          payback_period: '2.3 months'
        });
        setCurrentStep('logistics');
        setIsLoading(false);
      }, 1500);

    } catch (error) {
      console.error('Logistics preview error:', error);
      setIsLoading(false);
    }
  };

  const getMatchScoreColor = (score: number) => {
    if (score >= 90) return 'bg-green-100 text-green-800';
    if (score >= 80) return 'bg-blue-100 text-blue-800';
    if (score >= 70) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getRiskLevelColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getTransportModeIcon = (mode: string) => {
    switch (mode.toLowerCase()) {
      case 'truck': return <Truck className="w-4 h-4" />;
      case 'sea': return <Package className="w-4 h-4" />;
      case 'air': return <Zap className="w-4 h-4" />;
      case 'rail': return <BarChart3 className="w-4 h-4" />;
      default: return <Truck className="w-4 h-4" />;
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const filteredMaterials = aiMaterials.filter(material => 
    material.name.toLowerCase().includes(searchTerm.toLowerCase()) &&
    (filterCategory === 'all' || material.category === filterCategory)
  );

  const filteredMatches = aiMatches.filter(match => 
    match.company_name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <AuthenticatedLayout title="AI-Powered Industrial Symbiosis">
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2">AI-Powered Industrial Symbiosis</h1>
              <p className="text-blue-100">
                Discover materials, find partners, and optimize logistics with advanced AI
              </p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold">{aiAnalysis?.symbiosis_score || 0}%</div>
              <div className="text-blue-100">Symbiosis Score</div>
            </div>
          </div>
        </div>

        {/* Progress Steps */}
        <div className="flex items-center justify-center space-x-4">
          <div className={`flex items-center space-x-2 ${currentStep === 'inference' ? 'text-blue-600' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep === 'inference' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}>
              <Brain className="w-4 h-4" />
            </div>
            <span className="font-medium">AI Inference</span>
          </div>
          <ArrowRight className="w-5 h-5 text-gray-400" />
          <div className={`flex items-center space-x-2 ${currentStep === 'matching' ? 'text-blue-600' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep === 'matching' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}>
              <Users className="w-4 h-4" />
            </div>
            <span className="font-medium">AI Matching</span>
          </div>
          <ArrowRight className="w-5 h-5 text-gray-400" />
          <div className={`flex items-center space-x-2 ${currentStep === 'logistics' ? 'text-blue-600' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep === 'logistics' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}>
              <Truck className="w-4 h-4" />
            </div>
            <span className="font-medium">Logistics Preview</span>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-5 h-5 text-red-600" />
              <span className="text-red-800">{error}</span>
            </div>
          </div>
        )}

        {/* Step 1: AI Inference */}
        {currentStep === 'inference' && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="w-6 h-6 text-blue-600" />
                  <span>AI Material Inference</span>
                </CardTitle>
                <p className="text-gray-600">
                  Our advanced AI analyzes your company profile to identify potential materials and symbiosis opportunities.
                </p>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="text-center">
                      <Loader2 className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
                      <p className="text-gray-600">AI is analyzing your company profile...</p>
                      <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="text-center p-4 bg-blue-50 rounded-lg">
                        <Target className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                        <h3 className="font-semibold">Company Analysis</h3>
                        <p className="text-sm text-gray-600">Industry: {companyProfile?.industry}</p>
                      </div>
                      <div className="text-center p-4 bg-green-50 rounded-lg">
                        <Leaf className="w-8 h-8 text-green-600 mx-auto mb-2" />
                        <h3 className="font-semibold">Sustainability Focus</h3>
                        <p className="text-sm text-gray-600">Score: {companyProfile?.sustainability_score}%</p>
                      </div>
                      <div className="text-center p-4 bg-purple-50 rounded-lg">
                        <TrendingUp className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                        <h3 className="font-semibold">Growth Potential</h3>
                        <p className="text-sm text-gray-600">High Symbiosis Score</p>
                      </div>
                    </div>
                    
                    <Button 
                      onClick={loadDemoData}
                      disabled={isLoading}
                      className="w-full bg-blue-600 hover:bg-blue-700"
                    >
                      {isLoading ? (
                        <Loader2 className="w-4 h-4 animate-spin mr-2" />
                      ) : (
                        <Brain className="w-4 h-4 mr-2" />
                      )}
                      Start AI Analysis
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {/* Step 2: AI Matching */}
        {currentStep === 'matching' && (
          <div className="space-y-6">
            {/* AI Analysis Summary */}
            {aiAnalysis && (
              <Card className="bg-gradient-to-r from-green-50 to-blue-50 border-green-200">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2 text-green-800">
                    <Target className="w-6 h-6" />
                    <span>AI Analysis Results</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-green-600">{aiAnalysis.symbiosis_score}%</p>
                      <p className="text-sm text-gray-600">Symbiosis Score</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-blue-600">{aiAnalysis.estimated_savings}</p>
                      <p className="text-sm text-gray-600">Annual Savings</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-purple-600">{aiAnalysis.carbon_reduction}</p>
                      <p className="text-sm text-gray-600">Carbon Reduction</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-orange-600">{aiMatches.length}</p>
                      <p className="text-sm text-gray-600">Potential Matches</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Search and Filter */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Search className="w-5 h-5 text-blue-600" />
                  <span>Search & Filter</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex space-x-4">
                  <div className="flex-1">
                    <input
                      type="text"
                      placeholder="Search materials or companies..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <select
                    value={filterCategory}
                    onChange={(e) => setFilterCategory(e.target.value)}
                    className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="all">All Categories</option>
                    <option value="Metal Waste">Metal Waste</option>
                    <option value="Chemical Waste">Chemical Waste</option>
                    <option value="Energy">Energy</option>
                  </select>
                </div>
              </CardContent>
            </Card>

            {/* Materials and Matches Tabs */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Users className="w-6 h-6 text-blue-600" />
                  <span>AI-Generated Materials & Matches</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Tabs value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="materials">
                      <Package className="w-4 h-4 mr-2" />
                      Materials ({filteredMaterials.length})
                    </TabsTrigger>
                    <TabsTrigger value="matches">
                      <Users className="w-4 h-4 mr-2" />
                      Matches ({filteredMatches.length})
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="materials" className="space-y-4 mt-6">
                    {filteredMaterials.map((material) => (
                      <Card key={material.id} className="border-l-4 border-l-blue-500 hover:shadow-lg transition-shadow">
                        <CardContent className="p-6">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <h3 className="font-semibold text-lg">{material.name}</h3>
                                <Badge className="bg-blue-100 text-blue-800">
                                  <Brain className="w-3 h-3 mr-1" />
                                  AI Generated
                                </Badge>
                                <Badge className="bg-green-100 text-green-800">
                                  {material.sustainability_score}% Sustainable
                                </Badge>
                              </div>
                              <p className="text-gray-600 text-sm mb-3">{material.description}</p>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                                <div>
                                  <p className="text-gray-500">Quantity</p>
                                  <p className="font-medium">{material.quantity}</p>
                                </div>
                                <div>
                                  <p className="text-gray-500">Potential Value</p>
                                  <p className="font-medium text-green-600">{material.potential_value}</p>
                                </div>
                                <div>
                                  <p className="text-gray-500">Quality Grade</p>
                                  <p className="font-medium">{material.quality_grade}</p>
                                </div>
                                <div>
                                  <p className="text-gray-500">Category</p>
                                  <p className="font-medium">{material.category}</p>
                                </div>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </TabsContent>

                  <TabsContent value="matches" className="space-y-4 mt-6">
                    {filteredMatches.map((match) => (
                      <Card key={match.id} className="border-l-4 border-l-green-500 hover:shadow-lg transition-shadow cursor-pointer"
                            onClick={() => generateLogisticsPreview(match)}>
                        <CardContent className="p-6">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <h3 className="font-semibold text-lg">{match.company_name}</h3>
                                <Badge className={getMatchScoreColor(match.match_score)}>
                                  <Star className="w-3 h-3 mr-1" />
                                  {match.match_score}% match
                                </Badge>
                                <Badge className={getRiskLevelColor(match.risk_level)}>
                                  {match.risk_level} risk
                                </Badge>
                              </div>
                              <p className="text-gray-600 text-sm mb-2">{match.company_type} • {match.location}</p>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-3">
                                <div>
                                  <p className="text-gray-500">Material Match</p>
                                  <p className="font-medium">{match.material_match}</p>
                                </div>
                                <div>
                                  <p className="text-gray-500">Potential Savings</p>
                                  <p className="font-medium text-green-600">{match.potential_savings}</p>
                                </div>
                                <div>
                                  <p className="text-gray-500">Carbon Reduction</p>
                                  <p className="font-medium text-blue-600">{match.carbon_reduction}</p>
                                </div>
                                <div>
                                  <p className="text-gray-500">Implementation</p>
                                  <p className="font-medium">{match.implementation_time}</p>
                                </div>
                              </div>
                              <div className="flex items-center space-x-4 text-sm">
                                <span className="text-gray-500">Partnership Type: <span className="font-medium">{match.partnership_type}</span></span>
                                <span className="text-gray-500">Impact: <span className="font-medium">{match.sustainability_impact}</span></span>
                              </div>
                            </div>
                            <div className="flex flex-col items-end space-y-2">
                              <Badge className="bg-green-100 text-green-800">
                                <Brain className="w-3 h-3 mr-1" />
                                AI Matched
                              </Badge>
                              <Button size="sm" variant="outline">
                                <Truck className="w-4 h-4 mr-1" />
                                View Logistics
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Step 3: Logistics Preview */}
        {currentStep === 'logistics' && logisticsPreview && selectedMatch && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Truck className="w-6 h-6 text-blue-600" />
                  <span>Logistics Preview</span>
                </CardTitle>
                <p className="text-gray-600">
                  Shipping {logisticsPreview.material} from {logisticsPreview.origin} to {logisticsPreview.destination}
                </p>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  {/* Transport Modes Comparison */}
                  <div>
                    <h3 className="font-semibold mb-4">Transport Options</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {logisticsPreview.transport_modes.map((mode, index) => (
                        <Card key={index} className={`border-2 ${mode.mode === 'Truck' ? 'border-blue-500 bg-blue-50' : 'border-gray-200'}`}>
                          <CardContent className="p-4">
                            <div className="flex items-center space-x-2 mb-3">
                              {getTransportModeIcon(mode.mode)}
                              <h4 className="font-semibold">{mode.mode}</h4>
                            </div>
                            <div className="space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span>Cost:</span>
                                <span className="font-medium">{formatCurrency(mode.cost)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Time:</span>
                                <span className="font-medium">{mode.transit_time} days</span>
                              </div>
                              <div className="flex justify-between">
                                <span>CO2:</span>
                                <span className="font-medium">{mode.carbon_emissions} kg</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Reliability:</span>
                                <span className="font-medium">{(mode.reliability * 100).toFixed(0)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span>Sustainability:</span>
                                <span className="font-medium">{mode.sustainability_score}%</span>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </div>

                  {/* Cost Breakdown */}
                  <div>
                    <h3 className="font-semibold mb-4">Cost Breakdown</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <p className="text-2xl font-bold text-blue-600">{formatCurrency(logisticsPreview.cost_breakdown.transport)}</p>
                        <p className="text-sm text-gray-600">Transport</p>
                      </div>
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <p className="text-2xl font-bold text-green-600">{formatCurrency(logisticsPreview.cost_breakdown.handling)}</p>
                        <p className="text-sm text-gray-600">Handling</p>
                      </div>
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <p className="text-2xl font-bold text-orange-600">{formatCurrency(logisticsPreview.cost_breakdown.customs)}</p>
                        <p className="text-sm text-gray-600">Customs</p>
                      </div>
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <p className="text-2xl font-bold text-purple-600">{formatCurrency(logisticsPreview.cost_breakdown.insurance)}</p>
                        <p className="text-sm text-gray-600">Insurance</p>
                      </div>
                    </div>
                  </div>

                  {/* Feasibility Analysis */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card className={logisticsPreview.is_feasible ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50'}>
                      <CardContent className="p-4">
                        <div className="flex items-center space-x-2 mb-2">
                          {logisticsPreview.is_feasible ? (
                            <CheckCircle className="w-5 h-5 text-green-600" />
                          ) : (
                            <AlertCircle className="w-5 h-5 text-red-600" />
                          )}
                          <h3 className="font-semibold">Feasibility Analysis</h3>
                        </div>
                        <p className="text-sm text-gray-600 mb-3">
                          {logisticsPreview.is_feasible 
                            ? 'This partnership is economically viable with excellent ROI potential.'
                            : 'This partnership may not be economically viable at current costs.'
                          }
                        </p>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span>Total Cost:</span>
                            <span className="font-medium">{formatCurrency(logisticsPreview.total_cost)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Carbon Footprint:</span>
                            <span className="font-medium">{logisticsPreview.total_carbon} kg CO2</span>
                          </div>
                          <div className="flex justify-between">
                            <span>ROI Potential:</span>
                            <span className="font-medium text-green-600">{logisticsPreview.roi_percentage}%</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Payback Period:</span>
                            <span className="font-medium">{logisticsPreview.payback_period}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardContent className="p-4">
                        <h3 className="font-semibold mb-3">Recommendations</h3>
                        <ul className="space-y-2 text-sm">
                          {logisticsPreview.recommendations.map((rec, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                              <span>{rec}</span>
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex space-x-4">
                    <Button 
                      variant="outline" 
                      onClick={() => setCurrentStep('matching')}
                      className="flex items-center space-x-2"
                    >
                      <ArrowLeft className="w-4 h-4" />
                      Back to Matches
                    </Button>
                    <Button 
                      className="flex items-center space-x-2 bg-green-600 hover:bg-green-700"
                    >
                      <Users className="w-4 h-4" />
                      Contact Partner
                    </Button>
                    <Button 
                      variant="outline"
                      className="flex items-center space-x-2"
                    >
                      <Calculator className="w-4 h-4" />
                      Detailed Quote
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </AuthenticatedLayout>
  );
};

export default DemoAIMatching; 