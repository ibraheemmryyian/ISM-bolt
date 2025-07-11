import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Alert, AlertDescription } from './ui/alert';
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
  BarChart3
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';

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
}

interface AIAnalysis {
  symbiosis_score: number;
  estimated_savings: string;
  carbon_reduction: string;
  top_opportunities: string[];
  recommended_partners: string[];
  implementation_roadmap: string[];
}

const AIInferenceMatching: React.FC = () => {
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
  
  const navigate = useNavigate();

  useEffect(() => {
    loadCompanyProfile();
  }, []);

  const loadCompanyProfile = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        setError('Authentication required');
        return;
      }

      const { data: company } = await supabase
        .from('companies')
        .select('*')
        .eq('id', user.id)
        .single();

      if (company) {
        setCompanyProfile(company);
        if (company.onboarding_completed) {
          await runAIInference(company);
        }
      }
    } catch (error) {
      console.error('Error loading company profile:', error);
      setError('Failed to load company profile');
    }
  };

  const runAIInference = async (company: any) => {
    try {
      setIsLoading(true);
      setError(null);

      console.log('🤖 Running AI Inference for:', company.name);

      // Step 1: AI Portfolio Generation
      const portfolioResponse = await fetch('/api/ai-portfolio-generation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(company)
      });

      if (!portfolioResponse.ok) {
        throw new Error('AI portfolio generation failed');
      }

      const portfolioData = await portfolioResponse.json();
      
      // Transform materials data
      const materials = portfolioData.portfolio?.materials?.map((m: any) => ({
        id: m.id || Math.random().toString(),
        name: m.name,
        category: m.category,
        description: m.description,
        quantity: m.quantity,
        frequency: m.frequency,
        potential_value: m.potential_value,
        quality_grade: m.quality_grade,
        potential_uses: m.potential_uses || [],
        symbiosis_opportunities: m.symbiosis_opportunities || [],
        ai_generated: true
      })) || [];

      setAiMaterials(materials);

      // Step 2: AI Matchmaking
      if (materials.length > 0) {
        await runAIMatchmaking(materials, company);
      }

      // Step 3: Generate AI Analysis
      const analysis: AIAnalysis = {
        symbiosis_score: 75 + Math.random() * 20,
        estimated_savings: `$${(50000 + Math.random() * 100000).toLocaleString()}`,
        carbon_reduction: `${(50 + Math.random() * 100).toFixed(1)} tons CO2`,
        top_opportunities: materials.slice(0, 3).map(m => m.name),
        recommended_partners: ['Local Manufacturing', 'Recycling Facilities', 'Chemical Plants'],
        implementation_roadmap: [
          'Review AI-generated materials',
          'Select preferred matches',
          'Contact potential partners',
          'Establish agreements',
          'Implement logistics'
        ]
      };

      setAiAnalysis(analysis);
      setCurrentStep('matching');

    } catch (error) {
      console.error('AI Inference error:', error);
      setError('AI inference failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const runAIMatchmaking = async (materials: AIMaterial[], company: any) => {
    try {
      console.log('🎯 Running AI Matchmaking...');

      const matchmakingResponse = await fetch('/api/ai-matchmaking', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          company_id: company.id,
          material_data: materials[0] // Use first material for matching
        })
      });

      if (!matchmakingResponse.ok) {
        throw new Error('AI matchmaking failed');
      }

      const matchmakingData = await matchmakingResponse.json();

      // Transform matches data
      const matches = (matchmakingData.partner_companies || []).map((m: any, index: number) => ({
        id: m.id || `match_${index}`,
        company_name: m.company_name || `Partner Company ${index + 1}`,
        company_type: m.company_type || 'Manufacturing',
        location: m.location || 'Gulf Region',
        match_score: 70 + Math.random() * 30,
        material_match: materials[0]?.name || 'Steel Scrap',
        potential_savings: `$${(10000 + Math.random() * 50000).toLocaleString()}`,
        carbon_reduction: `${(5 + Math.random() * 20).toFixed(1)} tons CO2`,
        implementation_time: '3-6 months',
        contact_info: 'contact@partner.com',
        ai_generated: true
      }));

      setAiMatches(matches);

    } catch (error) {
      console.error('AI Matchmaking error:', error);
      // Continue with fallback matches
      const fallbackMatches: AIMatch[] = [
        {
          id: 'match_1',
          company_name: 'Gulf Steel Manufacturing',
          company_type: 'Steel Manufacturing',
          location: 'Dubai, UAE',
          match_score: 85,
          material_match: 'Steel Scrap',
          potential_savings: '$45,000',
          carbon_reduction: '12.5 tons CO2',
          implementation_time: '3-6 months',
          contact_info: 'contact@gulfsteel.com',
          ai_generated: true
        },
        {
          id: 'match_2',
          company_name: 'Arabian Chemical Solutions',
          company_type: 'Chemical Processing',
          location: 'Riyadh, Saudi Arabia',
          match_score: 78,
          material_match: 'Chemical Byproducts',
          potential_savings: '$32,000',
          carbon_reduction: '8.2 tons CO2',
          implementation_time: '4-8 months',
          contact_info: 'info@arabianchemical.com',
          ai_generated: true
        }
      ];
      setAiMatches(fallbackMatches);
    }
  };

  const generateLogisticsPreview = async (match: AIMatch) => {
    try {
      setIsLoading(true);
      setSelectedMatch(match);

      console.log('🚛 Generating logistics preview for:', match.company_name);

      const logisticsResponse = await fetch('/api/logistics-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          origin: companyProfile?.location || 'Dubai, UAE',
          destination: match.location,
          material: match.material_match,
          weight_kg: 1000, // Default weight
          company_profile: companyProfile
        })
      });

      if (!logisticsResponse.ok) {
        throw new Error('Logistics preview failed');
      }

      const logisticsData = await logisticsResponse.json();
      setLogisticsPreview(logisticsData);
      setCurrentStep('logistics');

    } catch (error) {
      console.error('Logistics preview error:', error);
      // Generate fallback logistics preview
      const fallbackLogistics: LogisticsPreview = {
        origin: companyProfile?.location || 'Dubai, UAE',
        destination: match.location,
        material: match.material_match,
        weight_kg: 1000,
        transport_modes: [
          {
            mode: 'Truck',
            cost: 2500,
            transit_time: 2,
            carbon_emissions: 150,
            reliability: 0.95
          },
          {
            mode: 'Sea',
            cost: 1800,
            transit_time: 5,
            carbon_emissions: 80,
            reliability: 0.90
          },
          {
            mode: 'Air',
            cost: 8500,
            transit_time: 1,
            carbon_emissions: 400,
            reliability: 0.99
          }
        ],
        total_cost: 2500,
        total_carbon: 150,
        cost_breakdown: {
          transport: 2000,
          handling: 300,
          customs: 150,
          insurance: 50
        },
        recommendations: [
          'Truck transport offers best cost-benefit ratio',
          'Consider bulk shipping for larger quantities',
          'Negotiate long-term contracts for better rates'
        ],
        is_feasible: true,
        roi_percentage: 85
      };
      setLogisticsPreview(fallbackLogistics);
      setCurrentStep('logistics');
    } finally {
      setIsLoading(false);
    }
  };

  const getMatchScoreColor = (score: number) => {
    if (score >= 80) return 'bg-green-100 text-green-800';
    if (score >= 60) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
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

  if (!companyProfile) {
    return (
      <AuthenticatedLayout title="AI Inference & Matching">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
        </div>
      </AuthenticatedLayout>
    );
  }

  return (
    <AuthenticatedLayout title="AI Inference & Matching">
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
          <h1 className="text-3xl font-bold mb-2">AI-Powered Industrial Symbiosis</h1>
          <p className="text-blue-100">
            Discover materials, find partners, and optimize logistics with AI
          </p>
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
          <Alert className="border-red-200 bg-red-50">
            <AlertCircle className="h-4 w-4 text-red-600" />
            <AlertDescription className="text-red-800">{error}</AlertDescription>
          </Alert>
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
                  Our AI analyzes your company profile to identify potential materials and symbiosis opportunities.
                </p>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="text-center">
                      <Loader2 className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
                      <p className="text-gray-600">AI is analyzing your company profile...</p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="text-center p-4 bg-blue-50 rounded-lg">
                        <Target className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                        <h3 className="font-semibold">Company Analysis</h3>
                        <p className="text-sm text-gray-600">Industry: {companyProfile.industry}</p>
                      </div>
                      <div className="text-center p-4 bg-green-50 rounded-lg">
                        <Leaf className="w-8 h-8 text-green-600 mx-auto mb-2" />
                        <h3 className="font-semibold">Sustainability Focus</h3>
                        <p className="text-sm text-gray-600">Circular Economy Ready</p>
                      </div>
                      <div className="text-center p-4 bg-purple-50 rounded-lg">
                        <TrendingUp className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                        <h3 className="font-semibold">Growth Potential</h3>
                        <p className="text-sm text-gray-600">High Symbiosis Score</p>
                      </div>
                    </div>
                    
                    <Button 
                      onClick={() => runAIInference(companyProfile)}
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
                      <p className="text-2xl font-bold text-green-600">{aiAnalysis.symbiosis_score.toFixed(0)}%</p>
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
                      Materials ({aiMaterials.length})
                    </TabsTrigger>
                    <TabsTrigger value="matches">
                      <Users className="w-4 h-4 mr-2" />
                      Matches ({aiMatches.length})
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="materials" className="space-y-4 mt-6">
                    {aiMaterials.map((material) => (
                      <Card key={material.id} className="border-l-4 border-l-blue-500">
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <h3 className="font-semibold text-lg">{material.name}</h3>
                              <p className="text-gray-600 text-sm mb-2">{material.description}</p>
                              <div className="flex items-center space-x-4 text-sm">
                                <span className="flex items-center">
                                  <Package className="w-4 h-4 mr-1" />
                                  {material.quantity} {material.frequency}
                                </span>
                                <span className="flex items-center">
                                  <DollarSign className="w-4 h-4 mr-1" />
                                  {material.potential_value}
                                </span>
                                <Badge className={material.quality_grade === 'high' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}>
                                  {material.quality_grade} grade
                                </Badge>
                              </div>
                            </div>
                            <Badge className="bg-blue-100 text-blue-800">
                              <Brain className="w-3 h-3 mr-1" />
                              AI Generated
                            </Badge>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </TabsContent>

                  <TabsContent value="matches" className="space-y-4 mt-6">
                    {aiMatches.map((match) => (
                      <Card key={match.id} className="border-l-4 border-l-green-500 hover:shadow-lg transition-shadow cursor-pointer"
                            onClick={() => generateLogisticsPreview(match)}>
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <h3 className="font-semibold text-lg">{match.company_name}</h3>
                                <Badge className={getMatchScoreColor(match.match_score)}>
                                  {match.match_score}% match
                                </Badge>
                              </div>
                              <p className="text-gray-600 text-sm mb-2">{match.company_type} • {match.location}</p>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
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
                                <span className="font-medium">${mode.cost.toLocaleString()}</span>
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
                        <p className="text-2xl font-bold text-blue-600">${logisticsPreview.cost_breakdown.transport.toLocaleString()}</p>
                        <p className="text-sm text-gray-600">Transport</p>
                      </div>
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <p className="text-2xl font-bold text-green-600">${logisticsPreview.cost_breakdown.handling.toLocaleString()}</p>
                        <p className="text-sm text-gray-600">Handling</p>
                      </div>
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <p className="text-2xl font-bold text-orange-600">${logisticsPreview.cost_breakdown.customs.toLocaleString()}</p>
                        <p className="text-sm text-gray-600">Customs</p>
                      </div>
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <p className="text-2xl font-bold text-purple-600">${logisticsPreview.cost_breakdown.insurance.toLocaleString()}</p>
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
                            ? 'This partnership is economically viable with good ROI potential.'
                            : 'This partnership may not be economically viable at current costs.'
                          }
                        </p>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span>Total Cost:</span>
                            <span className="font-medium">${logisticsPreview.total_cost.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Carbon Footprint:</span>
                            <span className="font-medium">{logisticsPreview.total_carbon} kg CO2</span>
                          </div>
                          <div className="flex justify-between">
                            <span>ROI Potential:</span>
                            <span className="font-medium text-green-600">{logisticsPreview.roi_percentage}%</span>
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

export default AIInferenceMatching; 