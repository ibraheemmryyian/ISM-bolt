import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { 
  CheckCircle, 
  XCircle, 
  Edit, 
  Eye, 
  Save, 
  RotateCcw, 
  AlertTriangle,
  TrendingUp,
  Users,
  Package,
  ShoppingCart,
  Trash2,
  Lightbulb,
  Target,
  Zap,
  Globe,
  DollarSign,
  Clock,
  MapPin,
  Scale,
  Activity,
  Sparkles,
  Leaf,
  Building2,
  ArrowRight,
  Check,
  X,
  ThumbsUp,
  ThumbsDown
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { useNavigate } from 'react-router-dom';

interface MaterialListing {
  id?: string;
  name: string;
  description: string;
  category: string;
  quantity: string;
  frequency: string;
  notes: string;
  potential_value?: string;
  quality_grade?: string;
  potential_uses?: string[];
  symbiosis_opportunities?: string[];
  sustainability_impact?: string;
  cost_savings?: string;
  ai_generated: boolean;
  approved?: boolean;
  rejected?: boolean;
}

interface CompanySuggestion {
  company_type: string;
  location: string;
  waste_they_can_use: string[];
  resources_they_can_provide: string[];
  estimated_partnership_value: string;
  carbon_reduction: string;
  implementation_time: string;
  ai_generated: boolean;
  approved?: boolean;
  rejected?: boolean;
}

interface GreenInitiative {
  initiative_name: string;
  description: string;
  current_practice: string;
  greener_alternative: string;
  cost_savings_per_month: string;
  carbon_reduction: string;
  implementation_cost: string;
  payback_period: string;
  difficulty: string;
  priority: string;
  ai_generated: boolean;
  approved?: boolean;
  rejected?: boolean;
}

interface AIInsights {
  symbiosis_score: string;
  estimated_savings: string;
  carbon_reduction: string;
  top_opportunities: string[];
  recommended_partners: string[];
  implementation_roadmap: string[];
}

interface PortfolioData {
  materials: MaterialListing[];
  requirements: MaterialListing[];
  company_suggestions: CompanySuggestion[];
  green_initiatives: GreenInitiative[];
  ai_insights: AIInsights;
}

const EnhancedPortfolioReview: React.FC = () => {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState('materials');
  const [showApproved, setShowApproved] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    loadPortfolioData();
  }, []);

  const loadPortfolioData = async () => {
    try {
      setLoading(true);
      
      // Try to load from localStorage first (from AI onboarding)
      const storedPortfolio = localStorage.getItem('symbioflows-portfolio');
      const storedRecommendations = localStorage.getItem('symbioflows-recommendations');
      
      if (storedPortfolio) {
        const portfolio = JSON.parse(storedPortfolio);
        
        // Transform the data to match our interface
        const transformedData: PortfolioData = {
          materials: portfolio.material_listings?.map((item: any) => ({
            name: item.material || item.name,
            description: item.description || 'AI-generated material listing',
            category: item.category || 'general',
            quantity: item.quantity || 'Variable',
            frequency: item.frequency || 'monthly',
            notes: item.notes || '',
            potential_value: item.value || 'Unknown',
            quality_grade: 'medium',
            potential_uses: item.potential_exchanges || [],
            symbiosis_opportunities: item.potential_exchanges || [],
            sustainability_impact: 'Reduces waste and promotes circular economy',
            cost_savings: 'Variable based on market conditions',
            ai_generated: true,
            approved: false,
            rejected: false
          })) || [],
          requirements: portfolio.resource_needs?.map((item: any) => ({
            name: item.name,
            description: item.description || 'Resource requirement',
            category: 'requirement',
            quantity: item.quantity || 'Variable',
            frequency: 'monthly',
            notes: item.notes || '',
            current_cost: item.current_cost || 'Unknown',
            priority: 'medium',
            potential_sources: item.potential_sources || [],
            symbiosis_opportunities: item.potential_sources || [],
            ai_generated: true,
            approved: false,
            rejected: false
          })) || [],
          company_suggestions: portfolio.potential_partners?.map((item: any) => ({
            company_type: item.company_type,
            location: item.location,
            waste_they_can_use: item.waste_they_can_use || [],
            resources_they_can_provide: item.resources_they_can_provide || [],
            estimated_partnership_value: item.estimated_partnership_value || 'Unknown',
            carbon_reduction: 'Variable',
            implementation_time: '3-6 months',
            ai_generated: true,
            approved: false,
            rejected: false
          })) || [],
          green_initiatives: [
            {
              initiative_name: 'Waste Exchange Program',
              description: 'Connect with local companies to exchange waste materials',
              current_practice: 'Disposing waste in landfills',
              greener_alternative: 'Exchange waste with partner companies',
              cost_savings_per_month: '$2000',
              carbon_reduction: '5 tons CO2',
              implementation_cost: '$5000',
              payback_period: '2.5 months',
              difficulty: 'medium',
              priority: 'high',
              ai_generated: true,
              approved: false,
              rejected: false
            },
            {
              initiative_name: 'Energy Efficiency Audit',
              description: 'Identify and implement energy-saving measures',
              current_practice: 'Standard energy usage',
              greener_alternative: 'Optimized energy consumption',
              cost_savings_per_month: '$1500',
              carbon_reduction: '3 tons CO2',
              implementation_cost: '$10000',
              payback_period: '6.7 months',
              difficulty: 'easy',
              priority: 'medium',
              ai_generated: true,
              approved: false,
              rejected: false
            }
          ],
          ai_insights: {
            symbiosis_score: portfolio.estimated_savings ? '85%' : '75%',
            estimated_savings: portfolio.estimated_savings || '$15K-75K annually',
            carbon_reduction: portfolio.environmental_impact || '10-40 tons CO2 reduction',
            top_opportunities: portfolio.waste_streams?.map((ws: any) => ws.name) || ['Waste Exchange', 'Material Recovery'],
            recommended_partners: portfolio.potential_partners?.map((pp: any) => pp.company_type) || ['Local Manufacturers'],
            implementation_roadmap: portfolio.roadmap?.map((r: any) => r.phase) || ['Initial Assessment', 'Partner Selection', 'Implementation']
          }
        };
        
        setPortfolioData(transformedData);
      } else {
        // Load from database if available
        const { data: { user } } = await supabase.auth.getUser();
        if (user) {
          const { data: company } = await supabase
            .from('companies')
            .select('*')
            .eq('user_id', user.id)
            .single();
            
          if (company) {
            // Trigger AI generation if not already done
            await triggerAIGeneration(company.id);
            
            // Load generated materials from database
            const { data: materials, error: materialsError } = await supabase
              .from('materials')
              .select('*')
              .eq('company_id', company.id);
            const { data: aiInsights, error: aiInsightsError } = await supabase
              .from('ai_insights')
              .select('impact, description, metadata, confidence_score, created_at')
              .eq('company_id', company.id)
              .order('created_at', { ascending: false })
              .limit(1);
            
            if (materials || aiInsights) {
              const transformedData: PortfolioData = {
                materials: materials?.map(m => ({
                  id: m.id,
                  name: m.name,
                  description: m.description,
                  category: m.category,
                  quantity: m.quantity,
                  frequency: m.frequency,
                  notes: m.notes,
                  potential_value: m.potential_value,
                  quality_grade: m.quality_grade,
                  potential_uses: m.potential_uses || [],
                  symbiosis_opportunities: m.symbiosis_opportunities || [],
                  ai_generated: m.ai_generated,
                  approved: false,
                  rejected: false
                })) || [],
                requirements: [],
                company_suggestions: [],
                green_initiatives: [],
                ai_insights: {
                  symbiosis_score: aiInsights?.confidence_score || '75%',
                  estimated_savings: aiInsights?.estimated_savings || '$25K annually',
                  carbon_reduction: aiInsights?.impact || '15 tons CO2',
                  top_opportunities: aiInsights?.top_opportunities || ['Material Exchange', 'Waste Reduction'],
                  recommended_partners: aiInsights?.recommended_partners || ['Local Manufacturers'],
                  implementation_roadmap: aiInsights?.implementation_roadmap || ['Review Materials', 'Select Partners', 'Implement']
                }
              };
              
              setPortfolioData(transformedData);
            }
          }
        }
      }
      
      // If still no data, create fallback
      if (!portfolioData) {
        setPortfolioData({
          materials: [],
          requirements: [],
          company_suggestions: [],
          green_initiatives: [],
          ai_insights: {
            symbiosis_score: '0%',
            estimated_savings: '$0',
            carbon_reduction: '0 tons CO2',
            top_opportunities: [],
            recommended_partners: [],
            implementation_roadmap: []
          }
        });
      }
      
    } catch (error) {
      console.error('Error loading portfolio data:', error);
    } finally {
      setLoading(false);
    }
  };

  const triggerAIGeneration = async (companyId: string) => {
    try {
      const response = await fetch(`/api/v1/companies/${companyId}/generate-listings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        console.log('AI generation triggered successfully');
      }
    } catch (error) {
      console.error('Error triggering AI generation:', error);
    }
  };

  const handleApprove = (type: string, index: number) => {
    if (!portfolioData) return;
    
    setPortfolioData(prev => {
      if (!prev) return prev;
      
      const newData = { ...prev };
      
      switch (type) {
        case 'materials':
          newData.materials[index].approved = true;
          newData.materials[index].rejected = false;
          break;
        case 'requirements':
          newData.requirements[index].approved = true;
          newData.requirements[index].rejected = false;
          break;
        case 'suggestions':
          newData.company_suggestions[index].approved = true;
          newData.company_suggestions[index].rejected = false;
          break;
        case 'initiatives':
          newData.green_initiatives[index].approved = true;
          newData.green_initiatives[index].rejected = false;
          break;
      }
      
      return newData;
    });
  };

  const handleReject = (type: string, index: number) => {
    if (!portfolioData) return;
    
    setPortfolioData(prev => {
      if (!prev) return prev;
      
      const newData = { ...prev };
      
      switch (type) {
        case 'materials':
          newData.materials[index].approved = false;
          newData.materials[index].rejected = true;
          break;
        case 'requirements':
          newData.requirements[index].approved = false;
          newData.requirements[index].rejected = true;
          break;
        case 'suggestions':
          newData.company_suggestions[index].approved = false;
          newData.company_suggestions[index].rejected = true;
          break;
        case 'initiatives':
          newData.green_initiatives[index].approved = false;
          newData.green_initiatives[index].rejected = true;
          break;
      }
      
      return newData;
    });
  };

  const handleSaveAndContinue = async () => {
    try {
      setSaving(true);
      
      // Save approved items to database
      if (portfolioData) {
        const { data: { user } } = await supabase.auth.getUser();
        if (user) {
          const { data: company } = await supabase
            .from('companies')
            .select('*')
            .eq('user_id', user.id)
            .single();
            
          if (company) {
            // Save approved materials
            const approvedMaterials = portfolioData.materials.filter(m => m.approved);
            for (const material of approvedMaterials) {
              await supabase
                .from('materials')
                .upsert({
                  company_id: company.id,
                  name: material.name,
                  description: material.description,
                  category: material.category,
                  quantity: material.quantity,
                  frequency: material.frequency,
                  notes: material.notes,
                  potential_value: material.potential_value,
                  quality_grade: material.quality_grade,
                  potential_uses: material.potential_uses,
                  symbiosis_opportunities: material.symbiosis_opportunities,
                  ai_generated: true,
                  approved: true
                });
            }
            
            // Save approved requirements
            const approvedRequirements = portfolioData.requirements.filter(r => r.approved);
            for (const requirement of approvedRequirements) {
              await supabase
                .from('requirements')
                .upsert({
                  company_id: company.id,
                  name: requirement.name,
                  description: requirement.description,
                  category: requirement.category,
                  quantity: requirement.quantity,
                  frequency: requirement.frequency,
                  notes: requirement.notes,
                  current_cost: requirement.current_cost,
                  priority: requirement.priority,
                  potential_sources: requirement.potential_sources,
                  symbiosis_opportunities: requirement.symbiosis_opportunities,
                  ai_generated: true,
                  approved: true
                });
            }
          }
        }
      }
      
      // Navigate to dashboard
      navigate('/dashboard');
      
    } catch (error) {
      console.error('Error saving portfolio:', error);
    } finally {
      setSaving(false);
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category.toLowerCase()) {
      case 'textile': return <Package className="w-5 h-5" />;
      case 'chemical': return <Zap className="w-5 h-5" />;
      case 'metal': return <Scale className="w-5 h-5" />;
      case 'plastic': return <Trash2 className="w-5 h-5" />;
      case 'organic': return <Leaf className="w-5 h-5" />;
      case 'electronic': return <Activity className="w-5 h-5" />;
      case 'energy': return <Zap className="w-5 h-5" />;
      case 'water': return <Globe className="w-5 h-5" />;
      default: return <Package className="w-5 h-5" />;
    }
  };

  const getQualityColor = (grade: string) => {
    switch (grade) {
      case 'high': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'hard': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
        <div className="container mx-auto p-6">
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-400 mx-auto mb-4"></div>
              <p className="text-gray-300">Loading your AI-generated portfolio...</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!portfolioData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
        <div className="container mx-auto p-6">
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              No portfolio data found. Please complete the AI onboarding first.
            </AlertDescription>
          </Alert>
          <Button onClick={() => navigate('/onboarding')} className="mt-4">
            Start AI Onboarding
          </Button>
        </div>
      </div>
    );
  }

  const approvedCount = 
    portfolioData.materials.filter(m => m.approved).length +
    portfolioData.requirements.filter(r => r.approved).length +
    portfolioData.company_suggestions.filter(s => s.approved).length +
    portfolioData.green_initiatives.filter(i => i.approved).length;

  const totalCount = 
    portfolioData.materials.length +
    portfolioData.requirements.length +
    portfolioData.company_suggestions.length +
    portfolioData.green_initiatives.length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      <div className="container mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => navigate('/')}
              className="flex items-center space-x-2 hover:opacity-80 transition-opacity"
            >
              <Sparkles className="h-8 w-8 text-emerald-400" />
              <span className="text-2xl font-bold text-white">SymbioFlows</span>
            </button>
            
            <div className="hidden md:flex items-center space-x-2 text-gray-300">
              <span>/</span>
              <span className="text-emerald-400 font-medium">AI Portfolio Review</span>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <Badge variant="outline" className="text-emerald-400 border-emerald-400">
              {approvedCount}/{totalCount} Approved
            </Badge>
            <Button 
              onClick={handleSaveAndContinue}
              disabled={saving}
              className="bg-emerald-500 hover:bg-emerald-600"
            >
              {saving ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Saving...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4 mr-2" />
                  Save & Continue
                </>
              )}
            </Button>
          </div>
        </div>

        {/* AI Insights Summary */}
        <Card className="bg-white/10 backdrop-blur-sm border-white/20">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-white">
              <Sparkles className="w-6 h-6 text-emerald-400" />
              <span>AI Analysis Summary</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-emerald-400 mb-2">
                  {portfolioData.ai_insights.symbiosis_score}
                </div>
                <div className="text-gray-300 text-sm">Symbiosis Score</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-400 mb-2">
                  {portfolioData.ai_insights.estimated_savings}
                </div>
                <div className="text-gray-300 text-sm">Estimated Savings</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-400 mb-2">
                  {portfolioData.ai_insights.carbon_reduction}
                </div>
                <div className="text-gray-300 text-sm">Carbon Reduction</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Main Content */}
        <Card className="bg-white/10 backdrop-blur-sm border-white/20">
          <CardHeader>
            <CardTitle className="text-white">Review AI-Generated Portfolio</CardTitle>
            <p className="text-gray-300">
              Review and approve the AI-generated materials, partnerships, and green initiatives for your company.
            </p>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-4 bg-white/10">
                <TabsTrigger value="materials" className="text-white">
                  <Package className="w-4 h-4 mr-2" />
                  Materials ({portfolioData.materials.length})
                </TabsTrigger>
                <TabsTrigger value="suggestions" className="text-white">
                  <Users className="w-4 h-4 mr-2" />
                  Partners ({portfolioData.company_suggestions.length})
                </TabsTrigger>
                <TabsTrigger value="initiatives" className="text-white">
                  <Leaf className="w-4 h-4 mr-2" />
                  Green Initiatives ({portfolioData.green_initiatives.length})
                </TabsTrigger>
                <TabsTrigger value="requirements" className="text-white">
                  <ShoppingCart className="w-4 h-4 mr-2" />
                  Requirements ({portfolioData.requirements.length})
                </TabsTrigger>
              </TabsList>

              {/* Materials Tab */}
              <TabsContent value="materials" className="space-y-4 mt-6">
                {portfolioData.materials.length === 0 ? (
                  <div className="text-center py-12">
                    <Package className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-300 mb-2">No Materials Generated</h3>
                    <p className="text-gray-400">Complete AI onboarding to generate material listings.</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {portfolioData.materials.map((material, index) => (
                      <Card key={index} className={`relative ${material.approved ? 'border-green-500 bg-green-50' : material.rejected ? 'border-red-500 bg-red-50' : 'bg-white'}`}>
                        <CardHeader>
                          <div className="flex items-center justify-between">
                            <CardTitle className="flex items-center gap-2">
                              {getCategoryIcon(material.category)}
                              {material.name}
                            </CardTitle>
                            <div className="flex gap-2">
                              <Badge className={getQualityColor(material.quality_grade || 'medium')}>
                                {material.quality_grade || 'medium'} quality
                              </Badge>
                              <Badge variant="outline" className="text-xs">
                                AI Generated
                              </Badge>
                            </div>
                          </div>
                        </CardHeader>
                        
                        <CardContent className="space-y-4">
                          <div>
                            <p className="text-sm text-gray-600 mb-2">{material.description}</p>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="font-medium">Quantity:</span> {material.quantity}
                              </div>
                              <div>
                                <span className="font-medium">Frequency:</span> {material.frequency}
                              </div>
                              <div>
                                <span className="font-medium">Value:</span> {material.potential_value}
                              </div>
                              <div>
                                <span className="font-medium">Savings:</span> {material.cost_savings}
                              </div>
                            </div>
                          </div>

                          {material.potential_uses && material.potential_uses.length > 0 && (
                            <div>
                              <h4 className="font-medium text-sm mb-2">Potential Uses:</h4>
                              <div className="flex flex-wrap gap-1">
                                {material.potential_uses.map((use, i) => (
                                  <Badge key={i} variant="outline" className="text-xs">
                                    {use}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}

                          {material.sustainability_impact && (
                            <div className="bg-green-50 p-3 rounded-lg">
                              <h4 className="font-medium text-sm text-green-800 mb-1">Environmental Impact:</h4>
                              <p className="text-sm text-green-700">{material.sustainability_impact}</p>
                            </div>
                          )}

                          <div className="flex gap-2 pt-4">
                            <Button
                              size="sm"
                              onClick={() => handleApprove('materials', index)}
                              disabled={material.approved}
                              className={`flex-1 ${material.approved ? 'bg-green-500' : 'bg-green-600 hover:bg-green-700'}`}
                            >
                              <Check className="w-4 h-4 mr-1" />
                              {material.approved ? 'Approved' : 'Approve'}
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleReject('materials', index)}
                              disabled={material.rejected}
                              className={`flex-1 ${material.rejected ? 'border-red-500 text-red-500' : ''}`}
                            >
                              <X className="w-4 h-4 mr-1" />
                              {material.rejected ? 'Rejected' : 'Reject'}
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </TabsContent>

              {/* Company Suggestions Tab */}
              <TabsContent value="suggestions" className="space-y-4 mt-6">
                {portfolioData.company_suggestions.length === 0 ? (
                  <div className="text-center py-12">
                    <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-300 mb-2">No Partner Suggestions</h3>
                    <p className="text-gray-400">Complete AI onboarding to generate partner suggestions.</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {portfolioData.company_suggestions.map((suggestion, index) => (
                      <Card key={index} className={`relative ${suggestion.approved ? 'border-green-500 bg-green-50' : suggestion.rejected ? 'border-red-500 bg-red-50' : 'bg-white'}`}>
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <Building2 className="w-5 h-5" />
                            {suggestion.company_type}
                          </CardTitle>
                        </CardHeader>
                        
                        <CardContent className="space-y-4">
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <span className="font-medium">Location:</span> {suggestion.location}
                            </div>
                            <div>
                              <span className="font-medium">Value:</span> {suggestion.estimated_partnership_value}
                            </div>
                            <div>
                              <span className="font-medium">Timeline:</span> {suggestion.implementation_time}
                            </div>
                            <div>
                              <span className="font-medium">Carbon Reduction:</span> {suggestion.carbon_reduction}
                            </div>
                          </div>

                          <div>
                            <h4 className="font-medium text-sm mb-2">Waste They Can Use:</h4>
                            <div className="flex flex-wrap gap-1">
                              {suggestion.waste_they_can_use.map((waste, i) => (
                                <Badge key={i} variant="outline" className="text-xs">
                                  {waste}
                                </Badge>
                              ))}
                            </div>
                          </div>

                          <div>
                            <h4 className="font-medium text-sm mb-2">Resources They Can Provide:</h4>
                            <div className="flex flex-wrap gap-1">
                              {suggestion.resources_they_can_provide.map((resource, i) => (
                                <Badge key={i} variant="outline" className="text-xs">
                                  {resource}
                                </Badge>
                              ))}
                            </div>
                          </div>

                          <div className="flex gap-2 pt-4">
                            <Button
                              size="sm"
                              onClick={() => handleApprove('suggestions', index)}
                              disabled={suggestion.approved}
                              className={`flex-1 ${suggestion.approved ? 'bg-green-500' : 'bg-green-600 hover:bg-green-700'}`}
                            >
                              <Check className="w-4 h-4 mr-1" />
                              {suggestion.approved ? 'Approved' : 'Approve'}
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleReject('suggestions', index)}
                              disabled={suggestion.rejected}
                              className={`flex-1 ${suggestion.rejected ? 'border-red-500 text-red-500' : ''}`}
                            >
                              <X className="w-4 h-4 mr-1" />
                              {suggestion.rejected ? 'Rejected' : 'Reject'}
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </TabsContent>

              {/* Green Initiatives Tab */}
              <TabsContent value="initiatives" className="space-y-4 mt-6">
                {portfolioData.green_initiatives.length === 0 ? (
                  <div className="text-center py-12">
                    <Leaf className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-300 mb-2">No Green Initiatives</h3>
                    <p className="text-gray-400">Complete AI onboarding to generate green initiatives.</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {portfolioData.green_initiatives.map((initiative, index) => (
                      <Card key={index} className={`relative ${initiative.approved ? 'border-green-500 bg-green-50' : initiative.rejected ? 'border-red-500 bg-red-50' : 'bg-white'}`}>
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <Leaf className="w-5 h-5" />
                            {initiative.initiative_name}
                          </CardTitle>
                        </CardHeader>
                        
                        <CardContent className="space-y-4">
                          <div>
                            <p className="text-sm text-gray-600 mb-3">{initiative.description}</p>
                            
                            <div className="bg-blue-50 p-3 rounded-lg mb-3">
                              <div className="grid grid-cols-2 gap-2 text-sm">
                                <div>
                                  <span className="font-medium text-blue-800">Current:</span> {initiative.current_practice}
                                </div>
                                <div>
                                  <span className="font-medium text-green-800">Alternative:</span> {initiative.greener_alternative}
                                </div>
                              </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="font-medium">Monthly Savings:</span> {initiative.cost_savings_per_month}
                              </div>
                              <div>
                                <span className="font-medium">Carbon Reduction:</span> {initiative.carbon_reduction}
                              </div>
                              <div>
                                <span className="font-medium">Implementation Cost:</span> {initiative.implementation_cost}
                              </div>
                              <div>
                                <span className="font-medium">Payback Period:</span> {initiative.payback_period}
                              </div>
                            </div>
                          </div>

                          <div className="flex gap-2">
                            <Badge className={getDifficultyColor(initiative.difficulty)}>
                              {initiative.difficulty} difficulty
                            </Badge>
                            <Badge className={getPriorityColor(initiative.priority)}>
                              {initiative.priority} priority
                            </Badge>
                          </div>

                          <div className="flex gap-2 pt-4">
                            <Button
                              size="sm"
                              onClick={() => handleApprove('initiatives', index)}
                              disabled={initiative.approved}
                              className={`flex-1 ${initiative.approved ? 'bg-green-500' : 'bg-green-600 hover:bg-green-700'}`}
                            >
                              <Check className="w-4 h-4 mr-1" />
                              {initiative.approved ? 'Approved' : 'Approve'}
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleReject('initiatives', index)}
                              disabled={initiative.rejected}
                              className={`flex-1 ${initiative.rejected ? 'border-red-500 text-red-500' : ''}`}
                            >
                              <X className="w-4 h-4 mr-1" />
                              {initiative.rejected ? 'Rejected' : 'Reject'}
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </TabsContent>

              {/* Requirements Tab */}
              <TabsContent value="requirements" className="space-y-4 mt-6">
                {portfolioData.requirements.length === 0 ? (
                  <div className="text-center py-12">
                    <ShoppingCart className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-300 mb-2">No Requirements Generated</h3>
                    <p className="text-gray-400">Complete AI onboarding to generate requirement listings.</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {portfolioData.requirements.map((requirement, index) => (
                      <Card key={index} className={`relative ${requirement.approved ? 'border-green-500 bg-green-50' : requirement.rejected ? 'border-red-500 bg-red-50' : 'bg-white'}`}>
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <ShoppingCart className="w-5 h-5" />
                            {requirement.name}
                          </CardTitle>
                        </CardHeader>
                        
                        <CardContent className="space-y-4">
                          <div>
                            <p className="text-sm text-gray-600 mb-2">{requirement.description}</p>
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <span className="font-medium">Quantity:</span> {requirement.quantity}
                              </div>
                              <div>
                                <span className="font-medium">Frequency:</span> {requirement.frequency}
                              </div>
                              <div>
                                <span className="font-medium">Current Cost:</span> {requirement.current_cost}
                              </div>
                              <div>
                                <span className="font-medium">Priority:</span> {requirement.priority}
                              </div>
                            </div>
                          </div>

                          {requirement.potential_sources && requirement.potential_sources.length > 0 && (
                            <div>
                              <h4 className="font-medium text-sm mb-2">Potential Sources:</h4>
                              <div className="flex flex-wrap gap-1">
                                {requirement.potential_sources.map((source, i) => (
                                  <Badge key={i} variant="outline" className="text-xs">
                                    {source}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}

                          <div className="flex gap-2 pt-4">
                            <Button
                              size="sm"
                              onClick={() => handleApprove('requirements', index)}
                              disabled={requirement.approved}
                              className={`flex-1 ${requirement.approved ? 'bg-green-500' : 'bg-green-600 hover:bg-green-700'}`}
                            >
                              <Check className="w-4 h-4 mr-1" />
                              {requirement.approved ? 'Approved' : 'Approve'}
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleReject('requirements', index)}
                              disabled={requirement.rejected}
                              className={`flex-1 ${requirement.rejected ? 'border-red-500 text-red-500' : ''}`}
                            >
                              <X className="w-4 h-4 mr-1" />
                              {requirement.rejected ? 'Rejected' : 'Reject'}
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default EnhancedPortfolioReview; 