import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import AuthenticatedLayout from './AuthenticatedLayout';
import { 
  User, 
  TrendingUp, 
  DollarSign, 
  Leaf, 
  Target, 
  Award,
  Star,
  Calendar,
  MapPin,
  Users,
  Lightbulb,
  ArrowRight,
  CheckCircle,
  Sparkles,
  Loader2,
  Package,
  Store,
  Brain
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { activityService } from '../lib/activityService';
import { LogisticsIntegration } from './LogisticsIntegration';

interface CompanyProfile {
  id?: string;
  name: string;
  industry: string;
  location: string;
  employee_count: number;
  size_category: string;
  industry_position: string;
  sustainability_rating: string;
  growth_potential: string;
  joined_date: string;
  onboarding_completed?: boolean;
}

interface Achievements {
  total_savings: number;
  carbon_reduced: number;
  partnerships_formed: number;
  waste_diverted: number;
  matches_completed: number;
  sustainability_score: number;
  efficiency_improvement: number;
}

interface PersonalizedRecommendation {
  id: string;
  category: string;
  title: string;
  description: string;
  potential_impact: {
    savings: number;
    carbon_reduction: number;
    efficiency_gain: number;
  };
  implementation_difficulty: 'easy' | 'medium' | 'hard';
  time_to_implement: string;
  priority: 'high' | 'medium' | 'low';
  ai_reasoning: string;
}

interface RecentActivity {
  id: string;
  date: string;
  action: string;
  impact: string;
  category: 'match' | 'savings' | 'partnership' | 'sustainability';
}

interface MaterialListing {
  id: string;
  material_name: string;
  type: 'waste' | 'resource' | 'product';
  quantity: number;
  unit: string;
  description: string;
  category: string;
  match_score: number;
  role: 'buyer' | 'seller' | 'both' | 'neutral';
  sustainability_score: number;
  price_per_unit: number;
  total_value: number;
}

interface Match {
  id: string;
  material_id: string;
  partner_company: string;
  partner_material: string;
  match_score: number;
  potential_savings: number;
  carbon_reduction: number;
  status: 'pending' | 'accepted' | 'completed' | 'rejected';
  created_at: string;
}

interface PortfolioData {
  company: CompanyProfile;
  achievements: Achievements;
  recommendations: PersonalizedRecommendation[];
  recent_activity: RecentActivity[];
  next_milestones: string[];
  industry_comparison: {
    rank: number;
    total_companies: number;
    average_savings: number;
    your_savings: number;
  };
  materialListings: MaterialListing[];
  matches: Match[];
}

const PersonalPortfolio: React.FC = () => {
  const navigate = useNavigate();
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [error, setError] = useState<string | null>(null);
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    loadPortfolioData();
    // Removed polling interval to prevent page reloading
  }, []);

  const loadPortfolioData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Get authenticated user
      const { data: { user }, error: authError } = await supabase.auth.getUser();
      if (authError || !user) {
        setUser(null);
        throw new Error('Authentication required');
      }
      setUser(user);

      // Fetch real company data from database
      const { data: company, error: companyError } = await supabase
        .from('companies')
        .select('*')
        .eq('id', user.id)
        .maybeSingle();

      if (companyError) {
        setPortfolioData(null);
        setError('Failed to load company data');
        setLoading(false);
        return;
      }

      // Check if company exists, if not create a default profile
      if (!company) {
        // Create a default company profile instead of showing error
        const defaultCompany = {
          id: user.id,
          name: 'Your Company',
          industry: 'Unknown',
          location: 'Unknown',
          employee_count: 0,
          onboarding_completed: false,
          created_at: new Date().toISOString()
        };
        
        // Use default company data
        const portfolio: PortfolioData = {
          company: {
            id: defaultCompany.id,
            name: defaultCompany.name,
            industry: defaultCompany.industry,
            location: defaultCompany.location,
            employee_count: defaultCompany.employee_count,
            size_category: 'Small',
            industry_position: 'Pending',
            sustainability_rating: 'Developing',
            growth_potential: 'High',
            joined_date: defaultCompany.created_at,
            onboarding_completed: defaultCompany.onboarding_completed
          },
          achievements: {
            total_savings: 0,
            carbon_reduced: 0,
            partnerships_formed: 0,
            waste_diverted: 0,
            matches_completed: 0,
            sustainability_score: 0,
            efficiency_improvement: 0
          },
          recommendations: [],
          recent_activity: [],
          next_milestones: ['Complete Onboarding', 'Start Listing Materials'],
          industry_comparison: {
            rank: 0,
            total_companies: 0,
            average_savings: 0,
            your_savings: 0
          },
          materialListings: [],
          matches: []
        };
        
        setPortfolioData(portfolio);
        setLoading(false);
        return;
      }

      // Fetch real user activities (with error handling)
      let activities: any[] = [];
      try {
        activities = await activityService.getUserActivities(user.id, 10);
      } catch (activityError) {
        console.warn('Activities query failed:', activityError);
      }

      // Fetch AI insights if available (with error handling)
      let aiInsights: any = null;
      try {
        const { data: aiInsightsData, error: aiInsightsError } = await supabase
          .from('ai_insights')
          .select('impact, description, metadata, confidence_score, created_at')
          .eq('company_id', user.id)
          .order('created_at', { ascending: false })
          .limit(1);

        if (!aiInsightsError) {
          aiInsights = aiInsightsData;
        } else {
          console.warn('AI insights query failed:', aiInsightsError);
        }
      } catch (error) {
        console.warn('AI insights query failed:', error);
      }

      // Fetch material listings from AI onboarding (with error handling)
      let materials: any[] = [];
      try {
        const { data: materialsData, error: materialsError } = await supabase
          .from('materials')
          .select('*')
          .eq('company_id', user.id)
          .order('created_at', { ascending: false });

        if (!materialsError) {
          materials = materialsData || [];
        } else {
          console.warn('Materials query failed:', materialsError);
        }
      } catch (error) {
        console.warn('Materials query failed:', error);
      }

      // Fetch matches (with error handling)
      let matches: any[] = [];
      try {
        const { data: matchesData, error: matchesError } = await supabase
          .from('matches')
          .select('*')
          .or(`company_id.eq.${user.id},partner_company_id.eq.${user.id}`)
          .order('created_at', { ascending: false });

        if (!matchesError) {
          matches = matchesData || [];
        } else {
          console.warn('Matches query failed:', matchesError);
        }
      } catch (error) {
        console.warn('Matches query failed:', error);
      }

      // Approval state
      const aiMeta = aiInsights && aiInsights[0]?.metadata ? aiInsights[0].metadata : {};

      // Create portfolio data from real database information
      const portfolio: PortfolioData = {
        company: {
          id: company.id,
          name: company.name || 'Your Company',
          industry: company.industry || 'Unknown',
          location: company.location || 'Unknown',
          employee_count: company.employee_count || 0,
          size_category: (company.employee_count || 0) > 1000 ? 'Large' : 
                       (company.employee_count || 0) > 200 ? 'Medium' : 'Small',
          industry_position: company.onboarding_completed ? 'Active' : 'Pending',
          sustainability_rating: aiMeta.symbiosis_score || 'Developing',
          growth_potential: 'High',
          joined_date: company.created_at || new Date().toISOString(),
          onboarding_completed: company.onboarding_completed
        },
        achievements: {
          total_savings: parseInt(aiMeta.estimated_savings?.replace(/[^0-9]/g, '') || '0'),
          carbon_reduced: parseInt(aiMeta.carbon_reduction?.replace(/[^0-9]/g, '') || '0'),
          partnerships_formed: activities.filter(a => a.activity_type === 'connection_accepted').length,
          waste_diverted: activities.filter(a => a.activity_type === 'material_listed').length,
          matches_completed: activities.filter(a => a.activity_type === 'match_found').length,
          sustainability_score: parseInt(aiMeta.symbiosis_score?.replace(/[^0-9]/g, '') || '65'),
          efficiency_improvement: 15
        },
        recommendations: aiMeta.top_opportunities?.map((opportunity: string, index: number) => ({
          id: `rec-${index}`,
          category: 'symbiosis',
          title: opportunity,
          description: `AI-recommended opportunity based on your company profile`,
          potential_impact: {
            savings: 15000 + (index * 5000),
            carbon_reduction: 25 + (index * 5),
            efficiency_gain: 20 + (index * 3)
          },
          implementation_difficulty: index === 0 ? 'easy' : index === 1 ? 'medium' : 'hard',
          time_to_implement: index === 0 ? '1-2 weeks' : index === 1 ? '1-2 months' : '3-6 months',
          priority: index === 0 ? 'high' : index === 1 ? 'medium' : 'low',
          ai_reasoning: `Based on your ${company.industry} industry and ${company.location} location`
        })) || [],
        recent_activity: activities.map((activity: any) => ({
          id: activity.id,
          date: new Date(activity.created_at).toLocaleDateString(),
          action: activity.title,
          impact: activity.impact_level === 'high' ? 'High Impact' : activity.impact_level === 'medium' ? 'Medium Impact' : 'Low Impact',
          category: activity.activity_type === 'match_found' ? 'match' :
                   activity.activity_type === 'connection_accepted' ? 'partnership' :
                   activity.activity_type === 'material_listed' ? 'savings' : 'sustainability'
        })),
        next_milestones: [
          company.onboarding_completed ? 'Listing Approval' : 'Complete Onboarding',
          'Form New Partnerships'
        ],
        industry_comparison: {
          rank: 1,
          total_companies: 100,
          average_savings: 25000,
          your_savings: parseInt(aiMeta.estimated_savings?.replace(/[^0-9]/g, '') || '0')
        },
        materialListings: materials?.map((material: any) => ({
          id: material.id,
          material_name: material.material_name,
          type: material.type || 'resource',
          quantity: material.quantity || 0,
          unit: material.unit || 'units',
          description: material.description || '',
          category: material.category || '',
          match_score: material.match_score || 0,
          role: material.role || 'neutral',
          sustainability_score: material.sustainability_score || 0,
          price_per_unit: material.price_per_unit || 0,
          total_value: material.total_value || 0,
          company: material.companies
        })) || [],
        matches: matches?.map((match: any) => ({
          id: match.id,
          material_id: match.material_id,
          partner_company: match.partner_company_name || 'Unknown',
          partner_material: match.partner_material_name || 'Unknown',
          match_score: match.match_score || 0,
          potential_savings: match.potential_savings || 0,
          carbon_reduction: match.carbon_reduction || 0,
          status: match.status || 'pending',
          created_at: match.created_at
        })) || []
      };

      setPortfolioData(portfolio);
      setError(null);
    } catch (err: any) {
      setPortfolioData(null);
      setError(err.message || 'Failed to load portfolio data');
    } finally {
      setLoading(false);
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'text-red-400 bg-red-900/20 border-red-700';
      case 'medium': return 'text-orange-400 bg-orange-900/20 border-orange-700';
      case 'low': return 'text-green-400 bg-green-900/20 border-green-700';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-700';
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'text-green-400 bg-green-900/20 border-green-700';
      case 'medium': return 'text-yellow-400 bg-yellow-900/20 border-yellow-700';
      case 'hard': return 'text-red-400 bg-red-900/20 border-red-700';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-700';
    }
  };

  const getActivityIcon = (category: string) => {
    switch (category) {
      case 'match': return <Users className="w-4 h-4" />;
      case 'savings': return <DollarSign className="w-4 h-4" />;
      case 'partnership': return <Target className="w-4 h-4" />;
      case 'sustainability': return <Leaf className="w-4 h-4" />;
      default: return <Star className="w-4 h-4" />;
    }
  };

  const getActivityColor = (category: string) => {
    switch (category) {
      case 'match': return 'text-blue-400 bg-blue-900/20 border-blue-700';
      case 'savings': return 'text-green-400 bg-green-900/20 border-green-700';
      case 'partnership': return 'text-purple-400 bg-purple-900/20 border-purple-700';
      case 'sustainability': return 'text-green-400 bg-green-900/20 border-green-700';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-700';
    }
  };

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'buyer': return 'text-blue-400 bg-blue-900/20 border-blue-700';
      case 'seller': return 'text-green-400 bg-green-900/20 border-green-700';
      case 'both': return 'text-purple-400 bg-purple-900/20 border-purple-700';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-700';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'text-yellow-400 bg-yellow-900/20 border-yellow-700';
      case 'accepted': return 'text-green-400 bg-green-900/20 border-green-700';
      case 'completed': return 'text-blue-400 bg-blue-900/20 border-blue-700';
      case 'rejected': return 'text-red-400 bg-red-900/20 border-red-700';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-700';
    }
  };

  // Only show sensitive info if user is authenticated and authorized
  if (!user) {
    return <div className="p-8 text-center text-red-400">Authentication required. Please log in.</div>;
  }

  if (loading) {
    return <div className="p-8 text-center text-gray-300"><Loader2 className="animate-spin inline-block mr-2" /> Loading your portfolio...</div>;
  }

  if (error) {
    return (
      <div className="p-8 text-center">
        <div className="bg-red-900/20 border border-red-700 rounded-lg p-6 max-w-md mx-auto">
          <div className="text-red-400 mb-4">
            <h3 className="text-lg font-semibold mb-2">Dashboard Error</h3>
            <p className="text-sm">{error}</p>
          </div>
          {error.includes('onboarding') && (
            <Button 
              onClick={() => navigate('/adaptive-onboarding')}
              className="bg-emerald-600 hover:bg-emerald-700 text-white"
            >
              Complete Onboarding
            </Button>
          )}
        </div>
      </div>
    );
  }

  if (!portfolioData) {
    return <div className="p-8 text-center text-gray-400">No portfolio data available.</div>;
  }

  const isEmpty = !portfolioData?.company?.onboarding_completed;
  const company = portfolioData?.company;
  const achievements = portfolioData?.achievements;
  const recommendations = portfolioData?.recommendations || [];
  const recent_activity = portfolioData?.recent_activity || [];
  const next_milestones = portfolioData?.next_milestones || [];
  const materialListings = portfolioData?.materialListings || [];
  const matches = portfolioData?.matches || [];

  return (
    <AuthenticatedLayout>
      <div className="space-y-8 bg-slate-900 min-h-screen py-8">
        {/* Onboarding Banner if not complete */}
        {isEmpty && (
          <div className="bg-emerald-900/80 border-l-4 border-emerald-500 p-4 rounded flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold text-emerald-200">Complete AI Onboarding</h2>
              <p className="text-emerald-300 text-sm">Unlock personalized insights, recommendations, and full dashboard features by completing your onboarding.</p>
            </div>
            <Button className="bg-emerald-600 hover:bg-emerald-700 text-white" onClick={() => navigate('/adaptive-onboarding')}>
              Start AI Onboarding
            </Button>
          </div>
        )}

        {/* Platform logistics/communication message */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4 text-gray-300 mb-4">
          <p className="text-sm">
            <strong className="text-white">SymbioFlows handles all logistics and communication.</strong> You will not see detailed information about your match. Once both parties accept, we manage shipping, payment, and notifications. No direct negotiation or contact is required.
          </p>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-slate-800 border border-slate-700 rounded-xl shadow p-4 text-gray-300">
            <div className="flex items-center space-x-2">
              <DollarSign className="w-8 h-8 text-green-400" />
              <div>
                <p className="text-sm text-gray-400">Total Savings</p>
                <p className="text-2xl font-bold text-white">{isEmpty ? '--' : `$${achievements.total_savings?.toLocaleString?.() || 0}`}</p>
              </div>
            </div>
          </div>
          <div className="bg-slate-800 border border-slate-700 rounded-xl shadow p-4 text-gray-300">
            <div className="flex items-center space-x-2">
              <Leaf className="w-8 h-8 text-green-400" />
              <div>
                <p className="text-sm text-gray-400">Carbon Reduced</p>
                <p className="text-2xl font-bold text-white">{isEmpty ? '--' : `${achievements.carbon_reduced || 0} tons`}</p>
              </div>
            </div>
          </div>
          <div className="bg-slate-800 border border-slate-700 rounded-xl shadow p-4 text-gray-300">
            <div className="flex items-center space-x-2">
              <Users className="w-8 h-8 text-blue-400" />
              <div>
                <p className="text-sm text-gray-400">Partnerships</p>
                <p className="text-2xl font-bold text-white">{isEmpty ? '--' : achievements.partnerships_formed || 0}</p>
              </div>
            </div>
          </div>
          <div className="bg-slate-800 border border-slate-700 rounded-xl shadow p-4 text-gray-300">
            <div className="flex items-center space-x-2">
              <Award className="w-8 h-8 text-purple-400" />
              <div>
                <p className="text-sm text-gray-400">Sustainability Score</p>
                <p className="text-2xl font-bold text-white">{isEmpty ? '--' : `${achievements.sustainability_score || 0}%`}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Active Matches */}
        {matches.length > 0 && (
          <div className="bg-slate-800 border border-slate-700 rounded-xl shadow text-gray-300">
            <div className="mb-2 font-semibold text-lg p-6 pb-2">
              <h3 className="text-xl font-bold mb-1 text-white flex items-center space-x-2">
                <Target className="w-5 h-5 text-emerald-400" />
                <span>Active Matches</span>
              </h3>
            </div>
            <div className="text-gray-300 p-6 pt-2">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {matches.map((match) => (
                  <div key={match.id} className="bg-slate-900 rounded-lg p-4 border border-slate-700">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-lg font-semibold text-white">Match #{match.id}</h3>
                      <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${getStatusColor(match.status)}`}>
                        {match.status}
                      </span>
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Partner</span>
                        <span className="text-white">{match.partner_company}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Material</span>
                        <span className="text-white">{match.partner_material}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Match Score</span>
                        <span className="text-emerald-400 font-bold">{match.match_score}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Potential Savings</span>
                        <span className="text-green-400 font-bold">${match.potential_savings?.toLocaleString() || 0}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Carbon Reduction</span>
                        <span className="text-green-400">{match.carbon_reduction || 0} tons</span>
                      </div>
                    </div>
                    <div className="mt-3">
                      <span className="inline-block px-2 py-0.5 rounded text-xs font-medium bg-blue-900/20 text-blue-400 border border-blue-700">
                        SymbioFlows handles all logistics and communication
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* AI-Generated Materials */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl shadow text-gray-300">
          <div className="mb-2 font-semibold text-lg p-6 pb-2">
            <h3 className="text-xl font-bold mb-1 text-white flex items-center space-x-2">
              <Package className="w-5 h-5" />
              <span>Your AI-Generated Materials</span>
            </h3>
          </div>
          <div className="text-gray-300 p-6 pt-2">
            {materialListings.length === 0 ? (
              <div className="text-center text-gray-400 py-8">
                <p>No materials yet. Complete onboarding to unlock AI-generated materials.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {materialListings.map((material) => (
                  <div key={material.id} className="border border-slate-700 rounded-lg p-4 hover:bg-slate-700 transition-colors">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold text-white">{material.material_name}</h4>
                      <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${getRoleColor(material.role)}`}>
                        {material.role}
                      </span>
                    </div>
                    <div className="space-y-2 text-sm">
                      <p className="text-gray-400">
                        <span className="font-medium text-gray-300">Quantity:</span> {material.quantity} {material.unit}
                      </p>
                      <p className="text-gray-400">
                        <span className="font-medium text-gray-300">Description:</span> {material.description}
                      </p>
                      {material.category && (
                        <p className="text-gray-400">
                          <span className="font-medium text-gray-300">Category:</span> {material.category}
                        </p>
                      )}
                      <p className="text-emerald-400 font-medium">
                        <span className="font-medium text-gray-300">Match Score:</span> {material.match_score || 'N/A'}
                      </p>
                      <p className="text-gray-400">
                        <span className="font-medium text-gray-300">Sustainability Score:</span> {material.sustainability_score}%
                      </p>
                      <p className="text-gray-400">
                        <span className="font-medium text-gray-300">Value:</span> ${material.total_value?.toLocaleString() || 0}
                      </p>
                      <p className="text-xs text-gray-500 italic mt-2">
                        SymbioFlows will handle all logistics and communication for this material.
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* AI-Powered Recommendations */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl shadow text-gray-300">
          <div className="mb-2 font-semibold text-lg p-6 pb-2">
            <h3 className="text-xl font-bold mb-1 text-white flex items-center space-x-2">
              <Lightbulb className="w-5 h-5" />
              <span>AI-Powered Recommendations</span>
            </h3>
          </div>
          <div className="text-gray-300 p-6 pt-2">
            {recommendations.length === 0 ? (
              <div className="text-center text-gray-400 py-8">
                <p>No recommendations yet. Complete onboarding to unlock personalized AI insights.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {recommendations.map((rec) => (
                  <div key={rec.id} className="bg-slate-900 border border-slate-700 rounded-xl shadow text-gray-300 hover:shadow-lg transition-shadow">
                    <div className="mb-2 font-semibold text-lg pb-3 p-4">
                      <div className="flex items-start justify-between">
                        <h3 className="text-lg text-white">{rec.title}</h3>
                        <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${getPriorityColor(rec.priority)}`}>
                          {rec.priority}
                        </span>
                      </div>
                      <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${getDifficultyColor(rec.implementation_difficulty)}`}>
                        {rec.implementation_difficulty}
                      </span>
                    </div>
                    <div className="text-gray-300 p-4 pt-0 space-y-3">
                      <p className="text-sm text-gray-400">{rec.description}</p>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div className="text-center">
                          <p className="text-gray-400">Savings</p>
                          <p className="font-bold text-green-400">${rec.potential_impact.savings.toLocaleString()}</p>
                        </div>
                        <div className="text-center">
                          <p className="text-gray-400">Carbon</p>
                          <p className="font-bold text-green-400">{rec.potential_impact.carbon_reduction} tons</p>
                        </div>
                        <div className="text-center">
                          <p className="text-gray-400">Efficiency</p>
                          <p className="font-bold text-blue-400">+{rec.potential_impact.efficiency_gain}%</p>
                        </div>
                      </div>
                      <div className="text-xs text-gray-500">
                        <p><strong className="text-gray-300">AI Reasoning:</strong> {rec.ai_reasoning}</p>
                        <p className="mt-1"><strong className="text-gray-300">Time to implement:</strong> {rec.time_to_implement}</p>
                      </div>
                      <Button className="w-full bg-emerald-700 hover:bg-emerald-800 text-white" size="sm">
                        Learn More <ArrowRight className="w-4 h-4 ml-2" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl shadow text-gray-300">
          <div className="mb-2 font-semibold text-lg p-6 pb-2">
            <h3 className="text-xl font-bold mb-1 text-white">Recent Activity</h3>
          </div>
          <div className="text-gray-300 p-6 pt-2">
            {recent_activity.length === 0 ? (
              <div className="text-center text-gray-400 py-8">
                <p>No activity yet. Your recent actions will appear here after onboarding.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {recent_activity.map((activity) => (
                  <div key={activity.id} className="flex items-center space-x-3 p-3 hover:bg-slate-700 rounded-lg">
                    <div className={`p-2 rounded-full ${getActivityColor(activity.category)}`}>{getActivityIcon(activity.category)}</div>
                    <div className="flex-1">
                      <p className="font-medium text-white">{activity.action}</p>
                      <p className="text-sm text-gray-400">{activity.impact}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-500">{new Date(activity.date).toLocaleDateString()}</p>
                      <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${getActivityColor(activity.category)}`}>{activity.category}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Next Milestones */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl shadow text-gray-300">
          <div className="mb-2 font-semibold text-lg p-6 pb-2">
            <h3 className="text-xl font-bold mb-1 text-white">Next Milestones</h3>
          </div>
          <div className="text-gray-300 p-6 pt-2">
            {next_milestones.length === 0 ? (
              <div className="text-center text-gray-400 py-8">
                <p>Milestones will be shown here after onboarding.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {next_milestones.map((milestone, index) => (
                  <div key={index} className="flex items-center space-x-3 p-3 bg-blue-900/20 rounded-lg border border-blue-700">
                    <Target className="w-5 h-5 text-blue-400" />
                    <span className="font-medium text-white">{milestone}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          <div className="bg-slate-800 border border-slate-700 rounded-xl shadow text-gray-300 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => navigate('/marketplace')}>
            <div className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-blue-900/20 rounded-lg border border-blue-700">
                  <Store className="h-5 w-5 text-blue-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">Browse Marketplace</h3>
                  <p className="text-sm text-gray-400">Find materials and partners</p>
                </div>
              </div>
            </div>
          </div>
          <div className="bg-slate-800 border border-slate-700 rounded-xl shadow text-gray-300 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => navigate('/adaptive-onboarding')}>
            <div className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-purple-900/20 rounded-lg border border-purple-700">
                  <Brain className="h-5 w-5 text-purple-300" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">AI Onboarding</h3>
                  <p className="text-sm text-gray-400">Complete your profile with AI</p>
                </div>
              </div>
            </div>
          </div>
          <div className="bg-slate-800 border border-slate-700 rounded-xl shadow text-gray-300 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => navigate('/marketplace')}>
            <div className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-green-900/20 rounded-lg border border-green-700">
                  <Package className="h-5 w-5 text-green-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">Add Material</h3>
                  <p className="text-sm text-gray-400">List your materials</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Logistics Integration */}
        <LogisticsIntegration />


      </div>
    </AuthenticatedLayout>
  );
};

export default PersonalPortfolio; 