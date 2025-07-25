import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
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
import { AIServicesIntegration } from './AIServicesIntegration';
import { OrchestrationDashboard } from './OrchestrationDashboard';

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
  materialListings: {
    id: string;
    material_name: string;
    type: 'waste' | 'resource';
    quantity: number;
    unit: string;
    description: string;
    category: string;
    match_score: number;
  }[];
}

const POLL_INTERVAL = 10000; // 10 seconds for real-time updates

const PersonalPortfolio: React.FC = () => {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [error, setError] = useState<string | null>(null);
  const [activeIntegrationTab, setActiveIntegrationTab] = useState<'logistics' | 'ai' | 'orchestration'>('logistics');
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    let poller: NodeJS.Timeout;
    const fetchAndPoll = async () => {
      await loadPortfolioData();
      poller = setInterval(loadPortfolioData, POLL_INTERVAL);
    };
    fetchAndPoll();
    return () => clearInterval(poller);
  }, []);

  const loadPortfolioData = async () => {
    try {
      setLoading(true);
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
        .single();
      if (companyError) {
        setPortfolioData(null);
        setError('Failed to load company data');
        setLoading(false);
        return;
      }
      // Fetch real user activities
      const activities = await activityService.getUserActivities(user.id, 10);
      // Fetch AI insights if available
      const { data: aiInsights, error: aiInsightsError } = await supabase
        .from('ai_insights')
        .select('impact, description, metadata, confidence_score, created_at, approval_status')
        .eq('company_id', user.id)
        .order('created_at', { ascending: false })
        .limit(1);
      // Approval state
      const approvalStatus = aiInsights && aiInsights[0]?.approval_status;
      // Create portfolio data from real database information
      const aiMeta = aiInsights && aiInsights[0]?.metadata ? aiInsights[0].metadata : {};
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
        })) || [
          {
            id: 'rec-1',
            category: 'onboarding',
            title: 'Complete AI Onboarding',
            description: 'Finish the AI onboarding process to get personalized recommendations.',
            potential_impact: {
              savings: 15000,
              carbon_reduction: 25,
              efficiency_gain: 20
            },
            implementation_difficulty: 'easy',
            time_to_implement: '1-2 weeks',
            priority: 'high',
            ai_reasoning: 'Onboarding not completed'
          }
        ],
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
          approvalStatus === 'approved' ? 'Matching' : 'Awaiting Approval',
          'Form New Partnerships'
        ],
        industry_comparison: {
          rank: 1,
          total_companies: 100,
          average_savings: 25000,
          your_savings: parseInt(aiMeta.estimated_savings?.replace(/[^0-9]/g, '') || '0')
        },
        materialListings: [] // TODO: fetch and display real listings
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
      case 'high': return 'text-red-600 bg-red-50';
      case 'medium': return 'text-orange-600 bg-orange-50';
      case 'low': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'text-green-600 bg-green-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'hard': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
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
      case 'match': return 'text-blue-600 bg-blue-50';
      case 'savings': return 'text-green-600 bg-green-50';
      case 'partnership': return 'text-purple-600 bg-purple-50';
      case 'sustainability': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  // Only show sensitive info if user is authenticated and authorized
  if (!user) {
    return <div className="p-8 text-center text-red-600">Authentication required. Please log in.</div>;
  }

  if (loading) {
    return <div className="p-8 text-center"><Loader2 className="animate-spin inline-block mr-2" /> Loading your portfolio...</div>;
  }

  if (error) {
    return <div className="p-8 text-center text-red-600">{error}</div>;
  }

  if (!portfolioData) {
    return <div className="p-8 text-center text-gray-500">No portfolio data available.</div>;
  }

  // Always render the full dashboard layout, with empty/zeroed fields and placeholders if onboarding is not complete
  const isEmpty = !portfolioData?.company?.onboarding_completed;
  const company = portfolioData?.company || {
    name: '--',
    industry: '--',
    location: '--',
    employee_count: 0,
    size_category: '--',
    industry_position: '--',
    sustainability_rating: '--',
    growth_potential: '--',
    joined_date: '--',
    onboarding_completed: false
  };
  const achievements = portfolioData?.achievements || {
    total_savings: 0,
    carbon_reduced: 0,
    partnerships_formed: 0,
    waste_diverted: 0,
    matches_completed: 0,
    sustainability_score: 0,
    efficiency_improvement: 0
  };
  const recommendations = isEmpty ? [] : (portfolioData?.recommendations || []);
  const recent_activity = isEmpty ? [] : (portfolioData?.recent_activity || []);
  const next_milestones = isEmpty ? [] : (portfolioData?.next_milestones || []);
  const industry_comparison = portfolioData?.industry_comparison || {
    rank: '--',
    total_companies: '--',
    average_savings: '--',
    your_savings: '--'
  };
  const materialListings = isEmpty ? [] : (portfolioData?.materialListings || []);

  return (
    <AuthenticatedLayout>
      <div className="space-y-8 bg-slate-900 min-h-screen py-8">
        {/* Subtle Onboarding Banner if not complete */}
        {isEmpty && (
          <div className="bg-emerald-900/80 border-l-4 border-emerald-500 p-4 rounded flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold text-emerald-200">Complete AI Onboarding</h2>
              <p className="text-emerald-300 text-sm">Unlock personalized insights, recommendations, and full dashboard features by completing your onboarding.</p>
            </div>
            <Button className="bg-emerald-600 hover:bg-emerald-700 text-white" onClick={() => window.location.href = '/adaptive-onboarding'}>
              Start AI Onboarding
            </Button>
          </div>
        )}

        {/* Platform logistics/communication message */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4 text-white mb-4">
          <p className="text-sm">
            <strong>SymbioFlows handles all logistics and communication.</strong> You will not see detailed information about your match. Once both parties accept, we manage shipping, payment, and notifications. No direct negotiation or contact is required.
          </p>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="bg-slate-800 border-slate-700 text-white">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <DollarSign className="w-8 h-8 text-green-400" />
                <div>
                  <p className="text-sm text-gray-300">Total Savings</p>
                  <p className="text-2xl font-bold">{isEmpty ? '--' : `$${achievements.total_savings?.toLocaleString?.() || 0}`}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-slate-800 border-slate-700 text-white">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Leaf className="w-8 h-8 text-green-400" />
                <div>
                  <p className="text-sm text-gray-300">Carbon Reduced</p>
                  <p className="text-2xl font-bold">{isEmpty ? '--' : `${achievements.carbon_reduced || 0} tons`}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-slate-800 border-slate-700 text-white">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Users className="w-8 h-8 text-blue-400" />
                <div>
                  <p className="text-sm text-gray-300">Partnerships</p>
                  <p className="text-2xl font-bold">{isEmpty ? '--' : achievements.partnerships_formed || 0}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-slate-800 border-slate-700 text-white">
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Award className="w-8 h-8 text-purple-400" />
                <div>
                  <p className="text-sm text-gray-300">Sustainability Score</p>
                  <p className="text-2xl font-bold">{isEmpty ? '--' : `${achievements.sustainability_score || 0}%`}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Match Details Card (Buyer/Seller) */}
        <Card className="bg-slate-800 border-slate-700 text-white">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <DollarSign className="w-5 h-5 text-emerald-400" />
              <span>Match Details</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {/* Example: Show for Buyer (if user is buyer, otherwise show seller view) */}
            {/* In a real app, you would check user role; here we show both for demo */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Buyer View */}
              <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
                <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                  <span>Buyer View</span>
                  <Badge className="bg-emerald-900 text-emerald-300 border-emerald-700">You save 20% vs. virgin material</Badge>
                </h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Material</span>
                    <span className="text-white">Recycled Material #{1}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Total Cost (incl. shipping)</span>
                    <span className="text-emerald-400 font-bold">$8,000</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Virgin Material Price</span>
                    <span className="text-gray-400 line-through">$10,000</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Platform Fee</span>
                    <span className="text-gray-400">$400</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Shipping</span>
                    <span className="text-gray-400">$600</span>
                  </div>
                </div>
                <div className="mt-4">
                  <Badge className="bg-blue-900 text-blue-300 border-blue-700">SymbioFlows handles all logistics and communication</Badge>
                </div>
              </div>
              {/* Seller View */}
              <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
                <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                  <span>Seller View</span>
                  <Badge className="bg-emerald-900 text-emerald-300 border-emerald-700">Platform handles everything</Badge>
                </h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Material</span>
                    <span className="text-white">Recycled Material #{1}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Net Revenue</span>
                    <span className="text-emerald-400 font-bold">$7,600</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">Platform Fee</span>
                    <span className="text-gray-400">$400</span>
                  </div>
                </div>
                <div className="mt-4">
                  <Badge className="bg-blue-900 text-blue-300 border-blue-700">SymbioFlows manages payment, shipping, and notifications</Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* AI-Generated Materials */}
        <Card className="bg-slate-800 border-slate-700 text-white">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Package className="w-5 h-5" />
              <span>Your AI-Generated Materials</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isEmpty ? (
              <div className="text-center text-gray-400 py-8">
                <p>No materials yet. Complete onboarding to unlock AI-generated materials.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {materialListings.map((material, index) => (
                  <div key={index} className="border border-slate-700 rounded-lg p-4 hover:bg-slate-700 transition-colors">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold text-white">Material #{index + 1}</h4>
                      <Badge className="bg-emerald-900 text-emerald-300 border-emerald-700">
                        Matched Partner
                      </Badge>
                    </div>
                    <div className="space-y-2 text-sm">
                      <p className="text-gray-300">
                        <span className="font-medium">Quantity:</span> {material.quantity} {material.unit}
                      </p>
                      <p className="text-gray-300">
                        <span className="font-medium">Description:</span> {material.description}
                      </p>
                      {material.category && (
                        <p className="text-gray-300">
                          <span className="font-medium">Category:</span> {material.category}
                        </p>
                      )}
                      <p className="text-emerald-300 font-medium">
                        <span className="font-medium">Match Score:</span> {material.match_score || 'N/A'}
                      </p>
                      <p className="text-xs text-gray-400 italic mt-2">
                        SymbioFlows will handle all logistics and communication for this match.
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* AI-Powered Recommendations */}
        <Card className="bg-slate-800 border-slate-700 text-white">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Lightbulb className="w-5 h-5" />
              <span>AI-Powered Recommendations</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isEmpty ? (
              <div className="text-center text-gray-400 py-8">
                <p>No recommendations yet. Complete onboarding to unlock personalized AI insights.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {recommendations.map((rec) => (
                  <Card key={rec.id} className="bg-slate-900 border-slate-700 text-white hover:shadow-lg transition-shadow">
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <CardTitle className="text-lg">{rec.title}</CardTitle>
                        <Badge className={getPriorityColor(rec.priority)}>
                          {rec.priority}
                        </Badge>
                      </div>
                      <Badge className={getDifficultyColor(rec.implementation_difficulty)}>
                        {rec.implementation_difficulty}
                      </Badge>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <p className="text-sm text-gray-300">{rec.description}</p>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div className="text-center">
                          <p className="text-gray-300">Savings</p>
                          <p className="font-bold text-green-400">${rec.potential_impact.savings.toLocaleString()}</p>
                        </div>
                        <div className="text-center">
                          <p className="text-gray-300">Carbon</p>
                          <p className="font-bold text-green-400">{rec.potential_impact.carbon_reduction} tons</p>
                        </div>
                        <div className="text-center">
                          <p className="text-gray-300">Efficiency</p>
                          <p className="font-bold text-blue-400">+{rec.potential_impact.efficiency_gain}%</p>
                        </div>
                      </div>
                      <div className="text-xs text-gray-400">
                        <p><strong>AI Reasoning:</strong> {rec.ai_reasoning}</p>
                        <p className="mt-1"><strong>Time to implement:</strong> {rec.time_to_implement}</p>
                      </div>
                      <Button className="w-full bg-emerald-700 hover:bg-emerald-800 text-white" size="sm">
                        Learn More <ArrowRight className="w-4 h-4 ml-2" />
                      </Button>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Activity */}
        <Card className="bg-slate-800 border-slate-700 text-white">
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            {isEmpty ? (
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
                      <p className="text-sm text-gray-300">{activity.impact}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-400">{new Date(activity.date).toLocaleDateString()}</p>
                      <Badge className={getActivityColor(activity.category)}>{activity.category}</Badge>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Next Milestones */}
        <Card className="bg-slate-800 border-slate-700 text-white">
          <CardHeader>
            <CardTitle>Next Milestones</CardTitle>
          </CardHeader>
          <CardContent>
            {isEmpty ? (
              <div className="text-center text-gray-400 py-8">
                <p>Milestones will be shown here after onboarding.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {next_milestones.map((milestone, index) => (
                  <div key={index} className="flex items-center space-x-3 p-3 bg-blue-900/60 rounded-lg">
                    <Target className="w-5 h-5 text-blue-400" />
                    <span className="font-medium text-white">{milestone}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          <Card className="bg-slate-800 border-slate-700 text-white hover:shadow-lg transition-shadow cursor-pointer" onClick={() => window.location.href = '/marketplace'}>
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-blue-900 rounded-lg">
                  <Store className="h-5 w-5 text-blue-400" />
                </div>
                <div>
                  <h3 className="font-semibold">Browse Marketplace</h3>
                  <p className="text-sm text-gray-300">Find materials and partners</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-slate-800 border-slate-700 text-white hover:shadow-lg transition-shadow cursor-pointer" onClick={() => window.location.href = '/adaptive-onboarding'}>
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-purple-900 rounded-lg">
                  <Brain className="h-5 w-5 text-purple-300" />
                </div>
                <div>
                  <h3 className="font-semibold">AI Onboarding</h3>
                  <p className="text-sm text-gray-300">Complete your profile with AI</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-slate-800 border-slate-700 text-white hover:shadow-lg transition-shadow cursor-pointer" onClick={() => window.location.href = '/marketplace'}>
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-green-900 rounded-lg">
                  <Package className="h-5 w-5 text-green-400" />
                </div>
                <div>
                  <h3 className="font-semibold">Add Material</h3>
                  <p className="text-sm text-gray-300">List your materials</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Backend Integration Tabs */}
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-white">Backend Services Integration</h2>
            <div className="flex space-x-2">
              <Button variant="outline" size="sm" className="border-slate-600 text-white">
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh All
              </Button>
            </div>
          </div>
          {/* Integration Tabs */}
          <div className="border-b border-slate-700">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveIntegrationTab('logistics')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeIntegrationTab === 'logistics'
                    ? 'border-emerald-500 text-emerald-400'
                    : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-slate-600'
                }`}
              >
                <Truck className="h-4 w-4 inline mr-2" />
                Logistics
              </button>
              <button
                onClick={() => setActiveIntegrationTab('ai')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeIntegrationTab === 'ai'
                    ? 'border-emerald-500 text-emerald-400'
                    : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-slate-600'
                }`}
              >
                <Brain className="h-4 w-4 inline mr-2" />
                AI Services
              </button>
              <button
                onClick={() => setActiveIntegrationTab('orchestration')}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeIntegrationTab === 'orchestration'
                    ? 'border-emerald-500 text-emerald-400'
                    : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-slate-600'
                }`}
              >
                <Workflow className="h-4 w-4 inline mr-2" />
                Orchestration
              </button>
            </nav>
          </div>
          {/* Integration Content */}
          <div className="bg-slate-800 rounded-lg border border-slate-700 text-white">
            {activeIntegrationTab === 'logistics' && <LogisticsIntegration />}
            {activeIntegrationTab === 'ai' && <AIServicesIntegration />}
            {activeIntegrationTab === 'orchestration' && <OrchestrationDashboard />}
          </div>
        </div>
      </div>
    </AuthenticatedLayout>
  );
};

export default PersonalPortfolio; 