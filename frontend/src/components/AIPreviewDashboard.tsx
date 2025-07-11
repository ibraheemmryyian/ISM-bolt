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
  Activity
} from 'lucide-react';
import { aiPreviewService } from '../lib/aiPreviewService';
import { supabase } from '../lib/supabase';

interface Material {
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
  ai_generated: boolean;
}

interface Requirement {
  id?: string;
  name: string;
  description: string;
  category: string;
  quantity: string;
  frequency: string;
  notes: string;
  current_cost?: string;
  priority?: string;
  potential_sources?: string[];
  symbiosis_opportunities?: string[];
  ai_generated: boolean;
}

interface PotentialMatch {
  id?: string;
  company_id: string;
  partner_company_id: string;
  company_name: string;
  partner_company_name: string;
  industry: string;
  match_reason: string;
  match_score: number;
  materials_involved: string[];
  potential_savings?: number;
  carbon_reduction?: number;
  status: 'pending' | 'accepted' | 'rejected';
  ai_generated: boolean;
}

interface AIInsights {
  symbiosis_score: string;
  estimated_savings: string;
  carbon_reduction: string;
  top_opportunities: string[];
  recommended_partners: string[];
  implementation_roadmap: string[];
}

interface AIPreviewData {
  company_profile: {
    id: string;
    name: string;
    industry: string;
    location: string;
    employee_count: number;
    products: string;
    main_materials: string;
    production_volume: string;
    process_description: string;
  };
  materials: Material[];
  requirements: Requirement[];
  potential_matches: PotentialMatch[];
  ai_insights: AIInsights;
  generation_status: 'generating' | 'completed' | 'error';
  generation_progress: number;
}

const AIPreviewDashboard: React.FC = () => {
  const [previewData, setPreviewData] = useState<AIPreviewData | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [editingMaterial, setEditingMaterial] = useState<string | null>(null);
  const [editingRequirement, setEditingRequirement] = useState<string | null>(null);
  const [selectedMatches, setSelectedMatches] = useState<Set<string>>(new Set());

  // Load AI preview data from service
  useEffect(() => {
    const loadPreviewData = async () => {
      setLoading(true);
      try {
        // Get current user
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) {
          throw new Error('User not authenticated');
        }

        // Get company ID from user
        const { data: company, error: companyError } = await supabase
          .from('companies')
          .select('*')
          .eq('user_id', user.id)
          .single();

        if (companyError || !company) {
          throw new Error('Company not found. Please complete AI onboarding first.');
        }

        // Check if company has enough data for AI analysis
        if (!company.industry || !company.products || !company.process_description) {
          throw new Error('Incomplete company profile. Please complete AI onboarding with detailed information.');
        }

        // Always generate fresh AI preview data
        console.log('Generating AI preview for company:', company.name);
        const generatedData = await aiPreviewService.generateAIPreview(company);
        setPreviewData(generatedData);
        
      } catch (error) {
        console.error('Error loading preview data:', error);
        // Show error instead of fake data
        setPreviewData({
          company_profile: {
            id: "error",
            name: "Error Loading Data",
            industry: "Unknown",
            location: "Unknown",
            employee_count: 0,
            products: "Unknown",
            main_materials: "Unknown",
            production_volume: "Unknown",
            process_description: "Unknown"
          },
          materials: [],
          requirements: [],
          potential_matches: [],
          ai_insights: {
            symbiosis_score: "0%",
            estimated_savings: "$0/year",
            carbon_reduction: "0 tons CO2/year",
            top_opportunities: [],
            recommended_partners: [],
            implementation_roadmap: []
          },
          generation_status: 'error',
          generation_progress: 0
        });
      } finally {
        setLoading(false);
      }
    };

    loadPreviewData();
  }, []);



  const handleSaveAll = async () => {
    if (!previewData) return;
    
    setLoading(true);
    try {
      // Get current user and company
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        throw new Error('User not authenticated');
      }

      const { data: company, error: companyError } = await supabase
        .from('companies')
        .select('id')
        .eq('user_id', user.id)
        .single();

      if (companyError || !company) {
        throw new Error('Company not found');
      }

      // Save AI content to database
      const success = await aiPreviewService.saveAIContent(company.id, previewData);
      
      if (success) {
        alert('All AI-generated content has been saved successfully!');
      } else {
        throw new Error('Failed to save content');
      }
    } catch (error) {
      console.error('Error saving content:', error);
      alert('Error saving content. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleRegenerate = async () => {
    setLoading(true);
    try {
      // Get current user and company
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        throw new Error('User not authenticated');
      }

      const { data: company, error: companyError } = await supabase
        .from('companies')
        .select('id')
        .eq('user_id', user.id)
        .single();

      if (companyError || !company) {
        throw new Error('Company not found');
      }

      // Regenerate AI content
      const newData = await aiPreviewService.regenerateAIContent(company.id);
      setPreviewData(newData);
      
      alert('AI content has been regenerated successfully!');
    } catch (error) {
      console.error('Error regenerating content:', error);
      alert('Error regenerating content. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleMatchSelection = (matchId: string) => {
    const newSelection = new Set(selectedMatches);
    if (newSelection.has(matchId)) {
      newSelection.delete(matchId);
    } else {
      newSelection.add(matchId);
    }
    setSelectedMatches(newSelection);
  };

  const getCategoryIcon = (category: string) => {
    const icons: { [key: string]: React.ReactNode } = {
      'agricultural_waste': <Trash2 className="w-4 h-4" />,
      'biological_waste': <Activity className="w-4 h-4" />,
      'water_waste': <Globe className="w-4 h-4" />,
      'raw_material': <Package className="w-4 h-4" />,
      'packaging': <ShoppingCart className="w-4 h-4" />,
      'energy': <Zap className="w-4 h-4" />,
      'chemical': <AlertTriangle className="w-4 h-4" />,
      'metal': <Scale className="w-4 h-4" />,
      'textile': <Target className="w-4 h-4" />
    };
    return icons[category] || <Package className="w-4 h-4" />;
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getQualityColor = (quality: string) => {
    switch (quality) {
      case 'high': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading && !previewData) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-lg">AI is analyzing your company profile...</p>
          <p className="mt-2 text-gray-600">This may take a few moments as our AI generates your personalized portfolio</p>
          <Progress value={previewData?.generation_progress || 0} className="mt-4 w-64 mx-auto" />
        </div>
      </div>
    );
  }

  if (!previewData) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Alert className="w-96">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Unable to load AI preview data. Please complete AI onboarding first or try again.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (previewData.generation_status === 'error') {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Alert className="w-96">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Error generating AI preview. Please ensure you have completed AI onboarding with detailed company information.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">AI Preview Dashboard</h1>
          <p className="text-gray-600 mt-2">
            Review and approve AI-generated content before publishing
          </p>
        </div>
        <div className="flex gap-3">
          <Button 
            variant="outline" 
            onClick={handleRegenerate}
            disabled={loading}
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Regenerate
          </Button>
          <Button 
            onClick={handleSaveAll}
            disabled={loading}
            className="bg-green-600 hover:bg-green-700"
          >
            <Save className="w-4 h-4 mr-2" />
            Save All Content
          </Button>
        </div>
      </div>

      {/* Company Profile Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="w-5 h-5" />
            Company Profile
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <h3 className="font-semibold text-lg">{previewData.company_profile.name}</h3>
              <p className="text-gray-600">{previewData.company_profile.industry}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Location</p>
              <p className="font-medium">{previewData.company_profile.location}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Employees</p>
              <p className="font-medium">{previewData.company_profile.employee_count}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="materials">Materials ({previewData.materials.length})</TabsTrigger>
          <TabsTrigger value="requirements">Requirements ({previewData.requirements.length})</TabsTrigger>
          <TabsTrigger value="matches">Matches ({previewData.potential_matches.length})</TabsTrigger>
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Materials Generated</p>
                    <p className="text-2xl font-bold">{previewData.materials.length}</p>
                  </div>
                  <Package className="w-8 h-8 text-blue-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Requirements Identified</p>
                    <p className="text-2xl font-bold">{previewData.requirements.length}</p>
                  </div>
                  <ShoppingCart className="w-8 h-8 text-green-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Potential Matches</p>
                    <p className="text-2xl font-bold">{previewData.potential_matches.length}</p>
                  </div>
                  <Users className="w-8 h-8 text-purple-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Symbiosis Score</p>
                    <p className="text-2xl font-bold">{previewData.ai_insights.symbiosis_score}</p>
                  </div>
                  <TrendingUp className="w-8 h-8 text-orange-600" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* AI Insights Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lightbulb className="w-5 h-5" />
                AI Insights Summary
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <DollarSign className="w-8 h-8 text-green-600 mx-auto mb-2" />
                  <p className="text-sm text-gray-600">Estimated Savings</p>
                  <p className="text-xl font-bold text-green-600">{previewData.ai_insights.estimated_savings}</p>
                </div>
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <Globe className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                  <p className="text-sm text-gray-600">Carbon Reduction</p>
                  <p className="text-xl font-bold text-blue-600">{previewData.ai_insights.carbon_reduction}</p>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <Target className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                  <p className="text-sm text-gray-600">Top Opportunities</p>
                  <p className="text-xl font-bold text-purple-600">{previewData.ai_insights.top_opportunities.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Materials Tab */}
        <TabsContent value="materials" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {previewData.materials.map((material, index) => (
              <Card key={index} className="relative">
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
                  <p className="text-gray-700">{material.description}</p>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500">Quantity</p>
                      <p className="font-medium">{material.quantity}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Frequency</p>
                      <p className="font-medium">{material.frequency}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Potential Value</p>
                      <p className="font-medium text-green-600">{material.potential_value}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Category</p>
                      <p className="font-medium capitalize">{material.category.replace('_', ' ')}</p>
                    </div>
                  </div>

                  <div>
                    <p className="text-gray-500 text-sm mb-2">Notes</p>
                    <p className="text-sm bg-gray-50 p-3 rounded">{material.notes}</p>
                  </div>

                  {material.potential_uses && material.potential_uses.length > 0 && (
                    <div>
                      <p className="text-gray-500 text-sm mb-2">Potential Uses</p>
                      <div className="flex flex-wrap gap-1">
                        {material.potential_uses.map((use, i) => (
                          <Badge key={i} variant="secondary" className="text-xs">
                            {use}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {material.symbiosis_opportunities && material.symbiosis_opportunities.length > 0 && (
                    <div>
                      <p className="text-gray-500 text-sm mb-2">Symbiosis Opportunities</p>
                      <div className="flex flex-wrap gap-1">
                        {material.symbiosis_opportunities.map((opportunity, i) => (
                          <Badge key={i} variant="outline" className="text-xs">
                            {opportunity}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Requirements Tab */}
        <TabsContent value="requirements" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {previewData.requirements.map((requirement, index) => (
              <Card key={index} className="relative">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      {getCategoryIcon(requirement.category)}
                      {requirement.name}
                    </CardTitle>
                    <div className="flex gap-2">
                      <Badge className={getPriorityColor(requirement.priority || 'medium')}>
                        {requirement.priority || 'medium'} priority
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        AI Generated
                      </Badge>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <p className="text-gray-700">{requirement.description}</p>
                  
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500">Quantity Needed</p>
                      <p className="font-medium">{requirement.quantity}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Frequency</p>
                      <p className="font-medium">{requirement.frequency}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Current Cost</p>
                      <p className="font-medium text-red-600">{requirement.current_cost}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Category</p>
                      <p className="font-medium capitalize">{requirement.category.replace('_', ' ')}</p>
                    </div>
                  </div>

                  <div>
                    <p className="text-gray-500 text-sm mb-2">Notes</p>
                    <p className="text-sm bg-gray-50 p-3 rounded">{requirement.notes}</p>
                  </div>

                  {requirement.potential_sources && requirement.potential_sources.length > 0 && (
                    <div>
                      <p className="text-gray-500 text-sm mb-2">Potential Sources</p>
                      <div className="flex flex-wrap gap-1">
                        {requirement.potential_sources.map((source, i) => (
                          <Badge key={i} variant="secondary" className="text-xs">
                            {source}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {requirement.symbiosis_opportunities && requirement.symbiosis_opportunities.length > 0 && (
                    <div>
                      <p className="text-gray-500 text-sm mb-2">Symbiosis Opportunities</p>
                      <div className="flex flex-wrap gap-1">
                        {requirement.symbiosis_opportunities.map((opportunity, i) => (
                          <Badge key={i} variant="outline" className="text-xs">
                            {opportunity}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Matches Tab */}
        <TabsContent value="matches" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Potential Symbiotic Matches</h3>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSelectedMatches(new Set(previewData.potential_matches.map(m => m.partner_company_id)))}
              >
                Select All
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSelectedMatches(new Set())}
              >
                Clear All
              </Button>
            </div>
          </div>

          <div className="space-y-4">
            {previewData.potential_matches.map((match, index) => (
              <Card key={index} className={`relative ${selectedMatches.has(match.partner_company_id) ? 'ring-2 ring-blue-500' : ''}`}>
                <CardContent className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-3">
                        <input
                          type="checkbox"
                          checked={selectedMatches.has(match.partner_company_id)}
                          onChange={() => handleMatchSelection(match.partner_company_id)}
                          className="w-4 h-4 text-blue-600 rounded"
                        />
                        <h4 className="text-lg font-semibold">{match.partner_company_name}</h4>
                        <Badge variant="outline">{match.industry}</Badge>
                        <Badge className="bg-green-100 text-green-800">
                          {Math.round(match.match_score * 100)}% Match
                        </Badge>
                      </div>
                      
                      <p className="text-gray-700 mb-4">{match.match_reason}</p>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div>
                          <p className="text-gray-500">Materials Involved</p>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {match.materials_involved.map((material, i) => (
                              <Badge key={i} variant="secondary" className="text-xs">
                                {material}
                              </Badge>
                            ))}
                          </div>
                        </div>
                        <div>
                          <p className="text-gray-500">Potential Savings</p>
                          <p className="font-medium text-green-600">${match.potential_savings?.toLocaleString()}/month</p>
                        </div>
                        <div>
                          <p className="text-gray-500">Carbon Reduction</p>
                          <p className="font-medium text-blue-600">{match.carbon_reduction} tons CO2/year</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* AI Insights Tab */}
        <TabsContent value="insights" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Top Opportunities */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  Top Opportunities
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {previewData.ai_insights.top_opportunities.map((opportunity, index) => (
                    <div key={index} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                        <span className="text-blue-600 font-semibold">{index + 1}</span>
                      </div>
                      <p className="font-medium">{opportunity}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Recommended Partners */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="w-5 h-5" />
                  Recommended Partners
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {previewData.ai_insights.recommended_partners.map((partner, index) => (
                    <div key={index} className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
                      <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                        <Users className="w-4 h-4 text-green-600" />
                      </div>
                      <p className="font-medium">{partner}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Implementation Roadmap */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="w-5 h-5" />
                  Implementation Roadmap
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {previewData.ai_insights.implementation_roadmap.map((step, index) => (
                    <div key={index} className="flex items-start gap-4">
                      <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0">
                        <span className="text-orange-600 font-semibold">{index + 1}</span>
                      </div>
                      <div className="flex-1">
                        <p className="font-medium">{step}</p>
                        <p className="text-sm text-gray-500 mt-1">
                          Estimated timeline: {index + 1}-{index + 3} months
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AIPreviewDashboard; 