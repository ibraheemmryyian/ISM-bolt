import React, { useState, useEffect } from 'react';
import { 
  BarChart3, 
  Bell, 
  Brain, 
  Building2, 
  Calendar, 
  Crown, 
  Factory, 
  Globe, 
  Home, 
  MessageSquare, 
  Plus, 
  Recycle, 
  Settings, 
  ShoppingCart, 
  Star, 
  TrendingUp, 
  Users, 
  Workflow,
  Zap
} from 'lucide-react';
import { supabase } from '../lib/supabase';

interface DashboardProps {
  onNavigate: (view: 'home' | 'dashboard' | 'marketplace' | 'admin') => void;
  onSignOut: () => void;
}

interface UserProfile {
  name: string;
  email: string;
  role: string;
  subscription_tier: string;
  company_profile?: {
    role: string;
    location: string;
    organization_type: string;
    materials_of_interest: string;
    sustainability_goals: string;
  };
}

interface AIRecommendation {
  id: string;
  type: 'connection' | 'material' | 'opportunity';
  title: string;
  description: string;
  confidence: number;
  action: string;
}

interface Activity {
  id: string;
  type: 'match' | 'message' | 'listing' | 'connection';
  title: string;
  description: string;
  timestamp: string;
  status: 'pending' | 'completed' | 'active';
}

export function Dashboard({ onNavigate, onSignOut }: DashboardProps) {
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [aiRecommendations, setAiRecommendations] = useState<AIRecommendation[]>([]);
  const [recentActivity, setRecentActivity] = useState<Activity[]>([]);
  const [stats, setStats] = useState({
    connections: 0,
    materials_listed: 0,
    matches_found: 0,
    sustainability_score: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  async function loadDashboardData() {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      // Load user profile
      const { data: company } = await supabase
        .from('companies')
        .select('*')
        .eq('id', user.id)
        .single();

      const { data: profile } = await supabase
        .from('company_profiles')
        .select('*')
        .eq('company_id', user.id)
        .single();

      if (company) {
        setUserProfile({
          name: company.name,
          email: company.email,
          role: company.role,
          subscription_tier: 'Pro', // This would come from a subscription table
          company_profile: profile
        });
      }

      // Load user materials count
      const { count: materialsCount } = await supabase
        .from('materials')
        .select('*', { count: 'exact', head: true })
        .eq('company_id', user.id);

      setStats({
        connections: 12,
        materials_listed: materialsCount || 0,
        matches_found: 8,
        sustainability_score: 85
      });

      // Generate AI recommendations
      generateAIRecommendations(profile);
      
      // Generate recent activity
      generateRecentActivity();

    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  }

  function generateAIRecommendations(profile: any) {
    const recommendations: AIRecommendation[] = [
      {
        id: '1',
        type: 'connection',
        title: 'Connect with GreenTech Solutions',
        description: 'Based on your interest in sustainable materials, this company has complementary waste streams.',
        confidence: 92,
        action: 'Send Connection Request'
      },
      {
        id: '2',
        type: 'material',
        title: 'New Plastic Waste Available',
        description: 'High-quality PET bottles (500kg) available 15km from your location.',
        confidence: 88,
        action: 'View Details'
      },
      {
        id: '3',
        type: 'opportunity',
        title: 'Research Grant Opportunity',
        description: 'EU Circular Economy Grant matches your sustainability goals - deadline in 2 weeks.',
        confidence: 76,
        action: 'Learn More'
      }
    ];
    setAiRecommendations(recommendations);
  }

  function generateRecentActivity() {
    const activities: Activity[] = [
      {
        id: '1',
        type: 'match',
        title: 'New Material Match Found',
        description: 'Your plastic waste requirement matched with EcoRecycle Corp',
        timestamp: '2 hours ago',
        status: 'pending'
      },
      {
        id: '2',
        type: 'message',
        title: 'Message from Sustainable Materials Inc',
        description: 'Interested in your bio-plastic research collaboration',
        timestamp: '5 hours ago',
        status: 'active'
      },
      {
        id: '3',
        type: 'listing',
        title: 'Material Listing Updated',
        description: 'Your organic waste listing received 3 new inquiries',
        timestamp: '1 day ago',
        status: 'completed'
      }
    ];
    setRecentActivity(activities);
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <div className="flex items-center space-x-2">
                <Workflow className="h-8 w-8 text-emerald-500" />
                <span className="text-2xl font-bold text-gray-900">SymbioFlow</span>
              </div>
              <div className="hidden md:ml-10 md:flex md:space-x-8">
                <button
                  onClick={() => onNavigate('dashboard')}
                  className="text-emerald-600 border-b-2 border-emerald-600 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Dashboard
                </button>
                <button
                  onClick={() => onNavigate('marketplace')}
                  className="text-gray-500 hover:text-gray-700 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Marketplace
                </button>
                <button
                  onClick={() => onNavigate('home')}
                  className="text-gray-500 hover:text-gray-700 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Home
                </button>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button className="text-gray-400 hover:text-gray-500">
                <Bell className="h-6 w-6" />
              </button>
              <button className="text-gray-400 hover:text-gray-500">
                <MessageSquare className="h-6 w-6" />
              </button>
              <button
                onClick={onSignOut}
                className="text-gray-500 hover:text-gray-700 text-sm font-medium"
              >
                Sign Out
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Welcome back, {userProfile?.name}
              </h1>
              <p className="text-gray-600 mt-1">
                Here's what's happening with your circular economy network
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <Crown className="h-5 w-5 text-yellow-500" />
              <span className="text-sm font-medium text-gray-700">
                {userProfile?.subscription_tier} Plan
              </span>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Users className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Connections</p>
                <p className="text-2xl font-bold text-gray-900">{stats.connections}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <Factory className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Materials Listed</p>
                <p className="text-2xl font-bold text-gray-900">{stats.materials_listed}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Zap className="h-6 w-6 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Matches Found</p>
                <p className="text-2xl font-bold text-gray-900">{stats.matches_found}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="flex items-center">
              <div className="p-2 bg-emerald-100 rounded-lg">
                <TrendingUp className="h-6 w-6 text-emerald-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Sustainability Score</p>
                <p className="text-2xl font-bold text-gray-900">{stats.sustainability_score}%</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* AI Recommendations */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-2">
                  <Brain className="h-6 w-6 text-emerald-500" />
                  <h2 className="text-xl font-bold text-gray-900">AI Recommendations</h2>
                </div>
                <button className="text-emerald-600 hover:text-emerald-700 text-sm font-medium">
                  View All
                </button>
              </div>
              <div className="space-y-4">
                {aiRecommendations.map((rec) => (
                  <div key={rec.id} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          {rec.type === 'connection' && <Users className="h-4 w-4 text-blue-500" />}
                          {rec.type === 'material' && <Factory className="h-4 w-4 text-green-500" />}
                          {rec.type === 'opportunity' && <Star className="h-4 w-4 text-yellow-500" />}
                          <span className="text-sm font-medium text-gray-600 capitalize">
                            {rec.type}
                          </span>
                          <span className="text-xs bg-emerald-100 text-emerald-800 px-2 py-1 rounded-full">
                            {rec.confidence}% match
                          </span>
                        </div>
                        <h3 className="font-semibold text-gray-900 mb-1">{rec.title}</h3>
                        <p className="text-sm text-gray-600 mb-3">{rec.description}</p>
                        <button className="text-emerald-600 hover:text-emerald-700 text-sm font-medium">
                          {rec.action}
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Quick Actions & Recent Activity */}
          <div className="space-y-6">
            {/* Quick Actions */}
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Quick Actions</h2>
              <div className="space-y-3">
                <button 
                  onClick={() => onNavigate('marketplace')}
                  className="w-full flex items-center space-x-3 p-3 text-left hover:bg-gray-50 rounded-lg transition"
                >
                  <Plus className="h-5 w-5 text-emerald-500" />
                  <span className="font-medium">List New Material</span>
                </button>
                <button className="w-full flex items-center space-x-3 p-3 text-left hover:bg-gray-50 rounded-lg transition">
                  <Globe className="h-5 w-5 text-blue-500" />
                  <span className="font-medium">Find Partners</span>
                </button>
                <button className="w-full flex items-center space-x-3 p-3 text-left hover:bg-gray-50 rounded-lg transition">
                  <BarChart3 className="h-5 w-5 text-purple-500" />
                  <span className="font-medium">View Analytics</span>
                </button>
                <button className="w-full flex items-center space-x-3 p-3 text-left hover:bg-gray-50 rounded-lg transition">
                  <Settings className="h-5 w-5 text-gray-500" />
                  <span className="font-medium">Account Settings</span>
                </button>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-4">Recent Activity</h2>
              <div className="space-y-4">
                {recentActivity.map((activity) => (
                  <div key={activity.id} className="flex items-start space-x-3">
                    <div className={`p-2 rounded-full ${
                      activity.type === 'match' ? 'bg-green-100' :
                      activity.type === 'message' ? 'bg-blue-100' :
                      activity.type === 'listing' ? 'bg-purple-100' :
                      'bg-gray-100'
                    }`}>
                      {activity.type === 'match' && <Zap className="h-4 w-4 text-green-600" />}
                      {activity.type === 'message' && <MessageSquare className="h-4 w-4 text-blue-600" />}
                      {activity.type === 'listing' && <Factory className="h-4 w-4 text-purple-600" />}
                      {activity.type === 'connection' && <Users className="h-4 w-4 text-gray-600" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900">{activity.title}</p>
                      <p className="text-sm text-gray-600">{activity.description}</p>
                      <p className="text-xs text-gray-500 mt-1">{activity.timestamp}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
