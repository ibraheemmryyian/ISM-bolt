import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { 
  DollarSign, 
  Leaf, 
  Users, 
  Factory, 
  Target, 
  Award, 
  Calendar, 
  MapPin, 
  Brain, 
  Plus, 
  Lightbulb, 
  FileText, 
  ArrowRight, 
  Zap, 
  Recycle, 
  TrendingUp,
  CheckCircle,
  XCircle,
  Clock
} from 'lucide-react';
import { AISuggestionFeedback } from './AISuggestionFeedback';

interface CompanyProfile {
  id: string;
  name: string;
  industry: string;
  location: string;
  employee_count: number;
  joined_date: string;
  onboarding_completed: boolean;
}

interface AIRecommendation {
  id: string;
  title: string;
  description: string;
  category: 'energy' | 'waste' | 'efficiency' | 'partnership';
  potential_savings: number;
  implementation_difficulty: 'easy' | 'medium' | 'hard';
  priority: 'high' | 'medium' | 'low';
  ai_reasoning: string;
  confidence_score?: number;
}

interface MaterialListing {
  id: string;
  name: string;
  category: string;
  quantity: string;
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
}

interface DashboardStats {
  total_savings: number;
  carbon_reduced: number;
  partnerships_formed: number;
  materials_listed: number;
  matches_completed: number;
  sustainability_score: number;
}

const PerfectDashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [companyProfile, setCompanyProfile] = useState<CompanyProfile | null>(null);
  const [aiRecommendations, setAiRecommendations] = useState<AIRecommendation[]>([]);
  const [materialListings, setMaterialListings] = useState<MaterialListing[]>([]);
  const [stats, setStats] = useState<DashboardStats>({
    total_savings: 0,
    carbon_reduced: 0,
    partnerships_formed: 0,
    materials_listed: 0,
    matches_completed: 0,
    sustainability_score: 0
  });
  const navigate = useNavigate();

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError || !user) {
        navigate('/');
        return;
      }

      // Load company profile
      const { data: company, error: companyError } = await supabase
        .from('companies')
        .select('*')
        .eq('user_id', user.id)
        .maybeSingle();

      if (companyError) {
        console.error('Error loading company:', companyError);
        // Don't throw error, just set company to null and show onboarding prompt
        setCompanyProfile(null);
        return;
      }

      setCompanyProfile(company);

      // Load AI recommendations from database instead of hardcoded
      const { data: aiSuggestions, error: suggestionsError } = await supabase
        .from('ai_suggestions')
        .select('*')
        .eq('company_id', user.id)
        .order('created_at', { ascending: false })
        .limit(5);

      if (!suggestionsError && aiSuggestions) {
        setAiRecommendations(aiSuggestions.map(s => ({
          id: s.id,
          title: s.title,
          description: s.description,
          category: s.category as any,
          potential_savings: s.potential_savings,
          implementation_difficulty: s.implementation_difficulty,
          priority: s.priority,
          ai_reasoning: s.ai_reasoning,
          confidence_score: s.confidence_score
        })));
      }

      // Load material listings
      const { data: materials, error: materialsError } = await supabase
        .from('materials')
        .select('*')
        .eq('company_id', user.id)
        .order('created_at', { ascending: false });

      if (!materialsError && materials) {
        setMaterialListings(materials.map(m => ({
          id: m.id,
          name: m.name,
          category: m.category,
          quantity: m.quantity,
          status: m.status || 'pending',
          created_at: m.created_at
        })));
      }

      // Load dashboard stats
      const { data: statsData, error: statsError } = await supabase
        .from('company_stats')
        .select('*')
        .eq('company_id', user.id)
        .single();

      if (!statsError && statsData) {
        setStats({
          total_savings: statsData.total_savings || 0,
          carbon_reduced: statsData.carbon_reduced || 0,
          partnerships_formed: statsData.partnerships_formed || 0,
          materials_listed: statsData.materials_listed || 0,
          matches_completed: statsData.matches_completed || 0,
          sustainability_score: statsData.sustainability_score || 0
        });
      }

    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionFeedback = () => {
    // Refresh suggestions after feedback
    loadDashboardData();
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    navigate('/');
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      case 'medium': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'text-green-600 bg-green-50 border-green-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'hard': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'energy': return <Zap className="w-4 h-4" />;
      case 'waste': return <Recycle className="w-4 h-4" />;
      case 'efficiency': return <TrendingUp className="w-4 h-4" />;
      case 'partnership': return <Users className="w-4 h-4" />;
      default: return <Lightbulb className="w-4 h-4" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      {/* Welcome Section */}
      <div className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white">
                Welcome, {companyProfile?.name || 'Company'}
              </h1>
              <p className="text-gray-300 mt-1 flex items-center space-x-2">
                <Calendar className="w-4 h-4" />
                <span>Member since {companyProfile?.joined_date ? new Date(companyProfile.joined_date).toLocaleDateString() : 'Recently'}</span>
                <span>•</span>
                <MapPin className="w-4 h-4" />
                <span>{companyProfile?.location || 'Location'}</span>
              </p>
            </div>
            
            <div className="flex space-x-3">
              <button
                onClick={() => navigate('/onboarding')}
                className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition flex items-center space-x-2"
              >
                <Brain className="w-4 h-4" />
                <span>AI Onboarding</span>
              </button>
              <button
                onClick={() => navigate('/marketplace')}
                className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition flex items-center space-x-2"
              >
                <Plus className="w-4 h-4" />
                <span>Add Material</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-6">
        <div className="space-y-6">
          {/* Show onboarding prompt if no company profile */}
          {!companyProfile && (
            <div className="flex justify-center">
              <div className="w-full max-w-2xl">
                <div className="bg-slate-800 rounded-lg shadow-lg border border-slate-700 p-8 text-center">
                  <Brain className="w-16 h-16 text-emerald-400 mx-auto mb-4" />
                  <h2 className="text-2xl font-bold text-white mb-4">Welcome to Industrial Symbiosis Management</h2>
                  <p className="text-gray-400 mb-6">
                    To get started with personalized AI recommendations and connect with potential partners, 
                    please complete the AI onboarding process.
                  </p>
                  <button
                    onClick={() => navigate('/onboarding')}
                    className="bg-emerald-500 text-white px-6 py-3 rounded-lg hover:bg-emerald-600 transition flex items-center space-x-2 mx-auto"
                  >
                    <Brain className="w-4 h-4" />
                    <span>Start AI Onboarding</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Main Content - Only show if company profile exists */}
          {companyProfile && (
            <div className="space-y-6">
            {/* Stats Grid - Centered to match AI sections */}
            <div className="flex justify-center">
              <div className="w-full max-w-4xl">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="bg-slate-800 rounded-lg p-4 shadow-lg border border-slate-700">
                    <div className="flex items-center space-x-2">
                      <DollarSign className="w-8 h-8 text-emerald-400" />
                      <div>
                        <p className="text-sm text-gray-400">Total Savings</p>
                        <p className="text-2xl font-bold text-white">${stats.total_savings.toLocaleString()}</p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-800 rounded-lg p-4 shadow-lg border border-slate-700">
                    <div className="flex items-center space-x-2">
                      <Leaf className="w-8 h-8 text-green-400" />
                      <div>
                        <p className="text-sm text-gray-400">Carbon Reduced</p>
                        <p className="text-2xl font-bold text-white">{stats.carbon_reduced} tons</p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-800 rounded-lg p-4 shadow-lg border border-slate-700">
                    <div className="flex items-center space-x-2">
                      <Users className="w-8 h-8 text-blue-400" />
                      <div>
                        <p className="text-sm text-gray-400">Partnerships</p>
                        <p className="text-2xl font-bold text-white">{stats.partnerships_formed}</p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-800 rounded-lg p-4 shadow-lg border border-slate-700">
                    <div className="flex items-center space-x-2">
                      <Factory className="w-8 h-8 text-purple-400" />
                      <div>
                        <p className="text-sm text-gray-400">Materials Listed</p>
                        <p className="text-2xl font-bold text-white">{stats.materials_listed}</p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-800 rounded-lg p-4 shadow-lg border border-slate-700">
                    <div className="flex items-center space-x-2">
                      <Target className="w-8 h-8 text-orange-400" />
                      <div>
                        <p className="text-sm text-gray-400">Matches Completed</p>
                        <p className="text-2xl font-bold text-white">{stats.matches_completed}</p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-800 rounded-lg p-4 shadow-lg border border-slate-700">
                    <div className="flex items-center space-x-2">
                      <Award className="w-8 h-8 text-purple-400" />
                      <div>
                        <p className="text-sm text-gray-400">Sustainability Score</p>
                        <p className="text-2xl font-bold text-white">{stats.sustainability_score}%</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* AI-Powered Suggestions and Material Listings Grid - Centered and Skinnier */}
            <div className="flex justify-center">
              <div className="w-full max-w-4xl">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* AI-Powered Suggestions */}
                  <div className="bg-slate-800 rounded-lg shadow-lg border border-slate-700">
                    <div className="p-5 border-b border-slate-700">
                      <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                        <Lightbulb className="w-4 h-4 text-emerald-400" />
                        <span>AI-Powered Suggestions</span>
                      </h2>
                      <p className="text-gray-400 mt-1 text-sm">Personalized recommendations to improve your sustainability and efficiency</p>
                    </div>
                    <div className="p-5 space-y-4">
                      {aiRecommendations.length > 0 ? (
                        aiRecommendations.slice(0, 2).map((rec) => (
                          <div key={rec.id} className="border border-slate-600 rounded-lg p-4 hover:shadow-md transition-shadow bg-slate-700/50">
                            <div className="flex items-start justify-between mb-3">
                              <div className="flex items-center space-x-2">
                                {getCategoryIcon(rec.category)}
                                <h3 className="font-semibold text-base text-white">{rec.title}</h3>
                              </div>
                              <div className="flex space-x-1">
                                <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getPriorityColor(rec.priority)}`}>
                                  {rec.priority}
                                </span>
                              </div>
                            </div>
                            <p className="text-gray-300 mb-3 text-sm">{rec.description}</p>
                            <div className="flex items-center justify-between mb-3">
                              <div className="flex items-center space-x-4 text-xs">
                                <span className="text-emerald-400 font-semibold">
                                  Potential Savings: ${rec.potential_savings.toLocaleString()}
                                </span>
                              </div>
                              <button className="bg-slate-600 text-white px-3 py-1 rounded text-xs hover:bg-slate-500 transition">
                                Learn More <ArrowRight className="w-3 h-3 ml-1" />
                              </button>
                            </div>
                            
                            {/* AI Suggestion Feedback */}
                            <AISuggestionFeedback
                              suggestion={rec}
                              onFeedbackSubmitted={handleSuggestionFeedback}
                            />
                          </div>
                        ))
                      ) : (
                        <div className="text-center py-8">
                          <Lightbulb className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                          <p className="text-gray-400 mb-4">Complete AI onboarding to get personalized recommendations</p>
                          <button
                            onClick={() => navigate('/onboarding')}
                            className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition flex items-center space-x-2 mx-auto"
                          >
                            <Brain className="w-4 h-4" />
                            <span>Start AI Onboarding</span>
                          </button>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Material Listings Approval */}
                  <div className="bg-slate-800 rounded-lg shadow-lg border border-slate-700">
                    <div className="p-5 border-b border-slate-700">
                      <h2 className="text-lg font-bold text-white flex items-center space-x-2">
                        <FileText className="w-4 h-4 text-blue-400" />
                        <span>AI-Generated Material Listings</span>
                      </h2>
                      <p className="text-gray-400 mt-1 text-sm">Review and approve materials before they go live</p>
                    </div>
                    <div className="p-5 space-y-4">
                      {materialListings.length > 0 ? (
                        materialListings.slice(0, 2).map((material) => (
                          <div key={material.id} className="border border-slate-600 rounded-lg p-4 hover:shadow-md transition-shadow bg-slate-700/50">
                            <div className="flex items-center justify-between mb-3">
                              <div className="flex items-center space-x-2">
                                <Factory className="w-4 h-4 text-blue-400" />
                                <h3 className="font-semibold text-base text-white">{material.name}</h3>
                              </div>
                              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                                material.status === 'approved' ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
                                material.status === 'rejected' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                                'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                              }`}>
                                {material.status}
                              </span>
                            </div>
                            <p className="text-gray-300 mb-2 text-sm">Category: {material.category}</p>
                            <p className="text-gray-300 mb-3 text-sm">Quantity: {material.quantity}</p>
                            <div className="flex items-center justify-between">
                              <span className="text-xs text-gray-400">
                                {new Date(material.created_at).toLocaleDateString()}
                              </span>
                              <button className="bg-emerald-500 text-white px-3 py-1 rounded text-xs hover:bg-emerald-600 transition">
                                Review <ArrowRight className="w-3 h-3 ml-1" />
                              </button>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-center py-8">
                          <FileText className="w-12 h-12 text-gray-500 mx-auto mb-4" />
                          <p className="text-gray-400">No material listings to review</p>
                          <button
                            onClick={() => navigate('/marketplace')}
                            className="mt-4 bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition"
                          >
                            Add Your First Material
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PerfectDashboard; 