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
  Zap,
  Eye,
  CheckCircle,
  AlertCircle,
  Clock,
  X,
  Info,
  Lightbulb,
  Target,
  ArrowRight,
  Sparkles,
  Moon,
  Sun,
  RotateCcw
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { useNavigate } from 'react-router-dom';
import { AISuggestionsService, AISuggestion } from '../lib/aiSuggestionsService';
import { FederatedLearningService } from '../lib/federatedLearningService';
import { AISuggestionFeedback } from './AISuggestionFeedback';
import { useTheme } from '../contexts/ThemeContext';

interface UserProfile {
  id: string;
  name: string;
  email: string;
  role: string;
  level: number;
  xp: number;
  streak_days: number;
  subscription?: {
    tier: string;
    status: string;
    expires_at?: string;
  };
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
  action_url?: string;
  status: string;
  created_at: string;
}

interface Activity {
  id: string;
  type: 'match' | 'message' | 'listing' | 'connection';
  title: string;
  description: string;
  timestamp: string;
  status: 'pending' | 'completed' | 'active';
}

interface MaterialMatch {
  id: string;
  material_name: string;
  company_name: string;
  match_score: number;
  distance: string;
  type: 'waste' | 'requirement';
  quantity: number;
  unit: string;
  description: string;
}

const Dashboard: React.FC = () => {
  const { theme, toggleTheme, isDark } = useTheme();
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [aiSuggestions, setAiSuggestions] = useState<AISuggestion[]>([]);
  const [recentActivity, setRecentActivity] = useState<Activity[]>([]);
  const [materialMatches, setMaterialMatches] = useState<MaterialMatch[]>([]);
  const [stats, setStats] = useState({
    connections: 0,
    materials_listed: 0,
    matches_found: 0,
    sustainability_score: 0,
    xp: 0,
    streak_days: 0
  });
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [notifications, setNotifications] = useState<any[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [feedbackMetrics, setFeedbackMetrics] = useState<any>(null);
  const navigate = useNavigate();

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setLoadError(null);

      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        // Create demo data for unauthenticated users
        setUserProfile({
          id: 'demo-user',
          name: 'Demo User',
          email: 'demo@example.com',
          role: 'user',
          level: 5,
          xp: 1250,
          streak_days: 7,
          subscription: {
            tier: 'free',
            status: 'active'
          },
          company_profile: {
            role: 'Manufacturing Manager',
            location: 'Dubai, UAE',
            organization_type: 'Manufacturing',
            materials_of_interest: 'Steel, Plastic, Electronics',
            sustainability_goals: 'Reduce waste by 30%'
          }
        });

        setStats({
          connections: 12,
          materials_listed: 8,
          matches_found: 15,
          sustainability_score: 85.5,
          xp: 1250,
          streak_days: 7
        });

        setRecentActivity([
          {
            id: 'act-1',
            type: 'match',
            title: 'New AI Match',
            description: 'Found 95% match with Emirates Steel',
            timestamp: new Date().toISOString(),
            status: 'active'
          }
        ]);

        setLoading(false);
        return;
      }

      // Load real user data
      const { data: profile, error: profileError } = await supabase
        .from('companies')
        .select('*')
        .eq('id', user.id)
        .single();

      if (profileError) {
        console.error('Error loading profile:', profileError);
        setLoadError('Failed to load user profile');
        setLoading(false);
        return;
      }

      setUserProfile({
        id: user.id,
        name: profile.name || user.email || 'Unknown User',
        email: user.email || 'unknown@example.com',
        role: profile.role || 'user',
        level: 1,
        xp: 0,
        streak_days: 0,
        company_profile: {
          role: profile.industry || 'Unknown',
          location: profile.location || 'Unknown',
          organization_type: profile.industry || 'Unknown',
          materials_of_interest: profile.process_description || 'Not specified',
          sustainability_goals: profile.sustainability_goals || 'Not specified'
        }
      });

      // Load AI suggestions
      await loadAISuggestions(user.id);

      // Load stats and other data
      await loadStats(user.id);
      await loadRecentActivity(user.id);
      await loadFeedbackMetrics(user.id);

      setLoading(false);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      setLoadError('Failed to load dashboard data');
      setLoading(false);
    }
  };

  const loadAISuggestions = async (userId: string) => {
    try {
      setSuggestionsLoading(true);
      
      // First try to get stored suggestions
      let suggestions = await AISuggestionsService.getStoredSuggestions(userId);
      
      // If no stored suggestions, generate new ones
      if (suggestions.length === 0) {
        suggestions = await AISuggestionsService.generateSuggestions(userId);
      }
      
      setAiSuggestions(suggestions);
    } catch (error) {
      console.error('Error loading AI suggestions:', error);
      // Don't set any fallback suggestions - let the UI show the onboarding prompt
      setAiSuggestions([]);
    } finally {
      setSuggestionsLoading(false);
    }
  };

  const loadStats = async (userId: string) => {
    try {
      // Load materials count
      const { data: materials, error: materialsError } = await supabase
        .from('materials')
        .select('id')
        .eq('company_id', userId);

      // Load connections count (placeholder)
      const connections = 0;

      // Load matches count (placeholder)
      const matches = 0;

      // Calculate sustainability score based on AI insights
      const { data: aiInsights } = await supabase
        .from('ai_insights')
        .select('sustainability_score, symbiosis_score')
        .eq('company_id', userId)
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      const sustainabilityScore = aiInsights?.sustainability_score || 
                                 aiInsights?.symbiosis_score || 
                                 65;

      setStats({
        connections,
        materials_listed: materials?.length || 0,
        matches_found: matches,
        sustainability_score: typeof sustainabilityScore === 'string' ? 
          parseInt(sustainabilityScore.replace(/[^0-9]/g, '')) : sustainabilityScore,
        xp: 0,
        streak_days: 0
      });
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const loadRecentActivity = async (userId: string) => {
    try {
      // Load recent materials
      const { data: materials } = await supabase
        .from('materials')
        .select('*')
        .eq('company_id', userId)
        .order('created_at', { ascending: false })
        .limit(5);

      const activities: Activity[] = [];
      
      if (materials && materials.length > 0) {
        materials.forEach(material => {
          activities.push({
            id: `material-${material.id}`,
            type: 'listing',
            title: 'Material Listed',
            description: `Added ${material.name} to marketplace`,
            timestamp: material.created_at,
            status: 'completed'
          });
        });
      }

      setRecentActivity(activities);
    } catch (error) {
      console.error('Error loading recent activity:', error);
    }
  };

  const loadFeedbackMetrics = async (userId: string) => {
    try {
      const metrics = await FederatedLearningService.getFeedbackMetrics(userId);
      setFeedbackMetrics(metrics);
    } catch (error) {
      console.error('Error loading feedback metrics:', error);
    }
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    navigate('/');
  };

  const handleNotifications = () => {
    // Handle notifications
  };

  const handleChats = () => {
    navigate('/chats');
  };

  const handleOnboarding = () => {
    navigate('/ai-onboarding');
  };

  const handleMarketplace = () => {
    navigate('/marketplace');
  };

  const handleSuggestionFeedback = async () => {
    // Refresh suggestions after feedback
    if (userProfile?.id) {
      await loadAISuggestions(userProfile.id);
      await loadFeedbackMetrics(userProfile.id);
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'energy':
        return <Zap className="h-4 w-4" />;
      case 'efficiency':
        return <TrendingUp className="h-4 w-4" />;
      case 'partnership':
        return <Users className="h-4 w-4" />;
      case 'waste':
        return <Recycle className="h-4 w-4" />;
      case 'materials':
        return <Factory className="h-4 w-4" />;
      case 'technology':
        return <Brain className="h-4 w-4" />;
      default:
        return <Lightbulb className="h-4 w-4" />;
    }
  };

  const getCategoryColor = (category: string) => {
    if (isDark) {
      switch (category) {
        case 'energy':
          return 'bg-yellow-900/30 text-yellow-300 border-yellow-700/50';
        case 'efficiency':
          return 'bg-blue-900/30 text-blue-300 border-blue-700/50';
        case 'partnership':
          return 'bg-purple-900/30 text-purple-300 border-purple-700/50';
        case 'waste':
          return 'bg-green-900/30 text-green-300 border-green-700/50';
        case 'materials':
          return 'bg-orange-900/30 text-orange-300 border-orange-700/50';
        case 'technology':
          return 'bg-indigo-900/30 text-indigo-300 border-indigo-700/50';
        default:
          return 'bg-gray-900/30 text-gray-300 border-gray-700/50';
      }
    } else {
      switch (category) {
        case 'energy':
          return 'bg-yellow-100 text-yellow-800 border-yellow-200';
        case 'efficiency':
          return 'bg-blue-100 text-blue-800 border-blue-200';
        case 'partnership':
          return 'bg-purple-100 text-purple-800 border-purple-200';
        case 'waste':
          return 'bg-green-100 text-green-800 border-green-200';
        case 'materials':
          return 'bg-orange-100 text-orange-800 border-orange-200';
        case 'technology':
          return 'bg-indigo-100 text-indigo-800 border-indigo-200';
        default:
          return 'bg-gray-100 text-gray-800 border-gray-200';
      }
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    if (isDark) {
      switch (difficulty) {
        case 'easy':
          return 'bg-green-900/30 text-green-300';
        case 'medium':
          return 'bg-yellow-900/30 text-yellow-300';
        case 'hard':
          return 'bg-red-900/30 text-red-300';
        default:
          return 'bg-gray-900/30 text-gray-300';
      }
    } else {
      switch (difficulty) {
        case 'easy':
          return 'bg-green-100 text-green-800';
        case 'medium':
          return 'bg-yellow-100 text-yellow-800';
        case 'hard':
          return 'bg-red-100 text-red-800';
        default:
          return 'bg-gray-100 text-gray-800';
      }
    }
  };

  const getPriorityColor = (priority: string) => {
    if (isDark) {
      switch (priority) {
        case 'high':
          return 'bg-red-900/30 text-red-300';
        case 'medium':
          return 'bg-yellow-900/30 text-yellow-300';
        case 'low':
          return 'bg-green-900/30 text-green-300';
        default:
          return 'bg-gray-900/30 text-gray-300';
      }
    } else {
      switch (priority) {
        case 'high':
          return 'bg-red-100 text-red-800';
        case 'medium':
          return 'bg-yellow-100 text-yellow-800';
        case 'low':
          return 'bg-green-100 text-green-800';
        default:
          return 'bg-gray-100 text-gray-800';
      }
    }
  };

  const isProOrAdmin = userProfile?.subscription?.tier === 'pro' || userProfile?.role === 'admin';

  if (loading) {
    return (
      <div className={`min-h-screen ${isDark ? 'bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900' : 'bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50'}`}>
        <div className="flex items-center justify-center h-screen">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-500 mx-auto mb-4"></div>
            <p className={`text-slate-300`}>Loading your dashboard...</p>
          </div>
        </div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className={`min-h-screen ${isDark ? 'bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900' : 'bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50'} flex items-center justify-center`}>
        <div className={`${isDark ? 'bg-slate-800/80' : 'bg-white/80'} backdrop-blur-sm rounded-xl shadow-lg p-8 max-w-md text-center border ${isDark ? 'border-slate-700' : 'border-slate-200'}`}>
          <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h3 className={`text-lg font-semibold ${isDark ? 'text-slate-100' : 'text-slate-900'} mb-2`}>Error Loading Dashboard</h3>
          <p className={`${isDark ? 'text-slate-300' : 'text-slate-600'} mb-4`}>{loadError}</p>
          <button
            onClick={loadDashboardData}
            className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen ${isDark ? 'bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900' : 'bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50'}`}>
      {/* Navigation */}
      <nav className={`${isDark ? 'bg-slate-800/80' : 'bg-white/80'} backdrop-blur-sm shadow-sm border-b ${isDark ? 'border-slate-700' : 'border-slate-200'}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <div className="flex items-center space-x-2">
                <Workflow className="h-8 w-8 text-emerald-500" />
                <span className={`text-2xl font-bold ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>SymbioFlow</span>
              </div>
              <div className="hidden md:ml-10 md:flex md:space-x-8">
                <button
                  onClick={() => navigate('/dashboard')}
                  className="text-emerald-600 border-b-2 border-emerald-600 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Dashboard
                </button>
                <button
                  onClick={handleMarketplace}
                  className={`${isDark ? 'text-slate-300 hover:text-slate-100' : 'text-slate-500 hover:text-slate-700'} px-1 pt-1 pb-4 text-sm font-medium`}
                >
                  Marketplace
                </button>
                <button
                  onClick={() => navigate('/')}
                  className={`${isDark ? 'text-slate-300 hover:text-slate-100' : 'text-slate-500 hover:text-slate-700'} px-1 pt-1 pb-4 text-sm font-medium`}
                >
                  Home
                </button>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={toggleTheme}
                className={`p-2 rounded-lg ${isDark ? 'bg-slate-700 hover:bg-slate-600' : 'bg-slate-100 hover:bg-slate-200'} transition`}
              >
                {isDark ? <Sun className="h-5 w-5 text-yellow-400" /> : <Moon className="h-5 w-5 text-slate-600" />}
              </button>
              <button
                onClick={handleMarketplace}
                className={`${isDark ? 'text-slate-300 hover:text-slate-100' : 'text-slate-600 hover:text-slate-900'} transition`}
              >
                Marketplace
              </button>
              <button
                onClick={handleNotifications}
                className={`${isDark ? 'text-slate-300 hover:text-slate-100' : 'text-slate-600 hover:text-slate-900'} transition relative`}
              >
                <Bell className="h-5 w-5" />
                {unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center">
                    {unreadCount}
                  </span>
                )}
              </button>
              <button
                onClick={handleChats}
                className={`${isDark ? 'text-slate-300 hover:text-slate-100' : 'text-slate-600 hover:text-slate-900'} transition`}
              >
                <MessageSquare className="h-5 w-5" />
              </button>
              <button
                onClick={handleSignOut}
                className={`${isDark ? 'text-slate-300 hover:text-slate-100' : 'text-slate-600 hover:text-slate-900'} transition`}
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
              <h1 className={`text-3xl font-bold ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>
                Welcome back, {userProfile?.name || ''}
              </h1>
              <p className={`${isDark ? 'text-slate-300' : 'text-slate-600'} mt-1`}>
                Level {userProfile?.level || 1} • {userProfile?.xp || 0} XP • {userProfile?.streak_days || 0} day streak
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Crown className={`h-5 w-5 ${isProOrAdmin ? 'text-yellow-500' : isDark ? 'text-slate-500' : 'text-slate-400'}`} />
                <span className={`text-sm font-medium ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                  {userProfile?.subscription?.tier?.toUpperCase() || 'FREE'} Plan
                </span>
              </div>
              {!isProOrAdmin && (
                <button
                  className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition text-sm font-medium"
                >
                  Upgrade to Pro
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="mb-6">
          <div className="flex space-x-2">
            <button
              onClick={handleOnboarding}
              className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition font-medium"
            >
              Complete AI Onboarding
            </button>
            <button
              onClick={handleMarketplace}
              className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition font-medium"
            >
              Add New Material
            </button>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className={`${isDark ? 'bg-slate-800/50' : 'bg-white/70'} backdrop-blur-sm rounded-xl shadow-sm border ${isDark ? 'border-slate-700' : 'border-slate-200'} p-6`}>
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Users className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className={`text-sm font-medium ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>Connections</p>
                <p className={`text-2xl font-bold ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>{stats.connections || 0}</p>
              </div>
            </div>
          </div>
          <div className={`${isDark ? 'bg-slate-800/50' : 'bg-white/70'} backdrop-blur-sm rounded-xl shadow-sm border ${isDark ? 'border-slate-700' : 'border-slate-200'} p-6`}>
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <Factory className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className={`text-sm font-medium ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>Materials Listed</p>
                <p className={`text-2xl font-bold ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>{stats.materials_listed || 0}</p>
              </div>
            </div>
          </div>
          <div className={`${isDark ? 'bg-slate-800/50' : 'bg-white/70'} backdrop-blur-sm rounded-xl shadow-sm border ${isDark ? 'border-slate-700' : 'border-slate-200'} p-6`}>
            <div className="flex items-center">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Zap className="h-6 w-6 text-purple-600" />
              </div>
              <div className="ml-4">
                <p className={`text-sm font-medium ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>AI Matches</p>
                <p className={`text-2xl font-bold ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>{stats.matches_found || 0}</p>
              </div>
            </div>
          </div>
          <div className={`${isDark ? 'bg-slate-800/50' : 'bg-white/70'} backdrop-blur-sm rounded-xl shadow-sm border ${isDark ? 'border-slate-700' : 'border-slate-200'} p-6`}>
            <div className="flex items-center">
              <div className="p-2 bg-emerald-100 rounded-lg">
                <TrendingUp className="h-6 w-6 text-emerald-600" />
              </div>
              <div className="ml-4">
                <p className={`text-sm font-medium ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>Sustainability Score</p>
                <p className={`text-2xl font-bold ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>{stats.sustainability_score?.toFixed(2) || '0.00'}%</p>
              </div>
            </div>
          </div>
        </div>

        {/* AI-Powered Suggestions */}
        <div className={`${isDark ? 'bg-slate-800/50' : 'bg-white/70'} backdrop-blur-sm rounded-xl shadow-sm border ${isDark ? 'border-slate-700' : 'border-slate-200'} p-6 mb-8`}>
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-emerald-100 rounded-lg">
                <Sparkles className="h-6 w-6 text-emerald-600" />
              </div>
              <div>
                <h2 className={`text-xl font-bold ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>AI-Powered Suggestions</h2>
                <p className={`${isDark ? 'text-slate-400' : 'text-slate-600'} text-sm`}>Personalized recommendations based on your company profile</p>
              </div>
            </div>
            <button
              onClick={() => loadAISuggestions(userProfile?.id || '')}
              disabled={suggestionsLoading}
              className="bg-emerald-500 text-white px-3 py-2 rounded-lg hover:bg-emerald-600 transition text-sm font-medium disabled:opacity-50 flex items-center space-x-2"
            >
              <RotateCcw className={`h-4 w-4 ${suggestionsLoading ? 'animate-spin' : ''}`} />
              <span>{suggestionsLoading ? 'Refreshing...' : 'Refresh'}</span>
            </button>
          </div>
          
          {suggestionsLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-500"></div>
              <span className={`ml-3 ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>Generating AI suggestions...</span>
            </div>
          ) : aiSuggestions.length > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {aiSuggestions.slice(0, 6).map((suggestion) => (
                <div key={suggestion.id} className={`${isDark ? 'bg-slate-700/50' : 'bg-white/50'} backdrop-blur-sm rounded-lg border ${isDark ? 'border-slate-600' : 'border-slate-200'} p-6 hover:shadow-md transition-shadow`}>
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-lg border ${getCategoryColor(suggestion.category)}`}>
                        {getCategoryIcon(suggestion.category)}
                      </div>
                      <div>
                        <h3 className={`font-semibold ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>{suggestion.title}</h3>
                        <div className="flex items-center space-x-2 mt-1">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPriorityColor(suggestion.priority)}`}>
                            {suggestion.priority}
                          </span>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(suggestion.implementation_difficulty)}`}>
                            {suggestion.implementation_difficulty}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-emerald-600">
                        ${suggestion.potential_savings.toLocaleString()}
                      </div>
                      <div className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>Potential Savings</div>
                    </div>
                  </div>
                  
                  <p className={`${isDark ? 'text-slate-300' : 'text-slate-600'} text-sm mb-4`}>{suggestion.description}</p>
                  
                  <div className="space-y-2 mb-4">
                    <div className="flex items-center justify-between text-xs">
                      <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>Implementation Time:</span>
                      <span className="font-medium">{suggestion.implementation_time}</span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>Carbon Reduction:</span>
                      <span className="font-medium">{suggestion.carbon_reduction} tons CO2</span>
                    </div>
                    <div className="flex items-center justify-between text-xs">
                      <span className={isDark ? 'text-slate-400' : 'text-slate-500'}>AI Confidence:</span>
                      <span className="font-medium">{suggestion.confidence_score}%</span>
                    </div>
                  </div>
                  
                  <div className={`${isDark ? 'bg-slate-600/50' : 'bg-slate-50'} rounded-lg p-3 mb-4`}>
                    <p className={`text-xs ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>
                      <span className="font-medium">AI Reasoning:</span> {suggestion.ai_reasoning}
                    </p>
                  </div>
                  
                  {/* Federated Learning Feedback */}
                  <AISuggestionFeedback
                    suggestion={suggestion}
                    onFeedbackSubmitted={handleSuggestionFeedback}
                  />
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Lightbulb className={`h-12 w-12 ${isDark ? 'text-slate-500' : 'text-slate-400'} mx-auto mb-4`} />
              <p className={`${isDark ? 'text-slate-400' : 'text-slate-500'} mb-4`}>No AI suggestions available yet.</p>
              <button
                onClick={handleOnboarding}
                className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition"
              >
                Complete AI Onboarding
              </button>
            </div>
          )}
        </div>

        {/* Recent Activity */}
        <div className={`${isDark ? 'bg-slate-800/50' : 'bg-white/70'} backdrop-blur-sm rounded-xl shadow-sm border ${isDark ? 'border-slate-700' : 'border-slate-200'} p-6`}>
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Clock className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <h2 className={`text-xl font-bold ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>Recent Activity</h2>
              <p className={`${isDark ? 'text-slate-400' : 'text-slate-600'} text-sm`}>Your latest marketplace activity</p>
            </div>
          </div>
          
          {recentActivity.length > 0 ? (
            <div className="space-y-4">
              {recentActivity.map((activity) => (
                <div key={activity.id} className={`flex items-center space-x-3 p-4 ${isDark ? 'bg-slate-700/50' : 'bg-white/50'} backdrop-blur-sm rounded-lg border ${isDark ? 'border-slate-600' : 'border-slate-200'}`}>
                  <div className="p-2 bg-emerald-100 rounded-lg">
                    <CheckCircle className="h-4 w-4 text-emerald-600" />
                  </div>
                  <div className="flex-1">
                    <p className={`font-medium ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>{activity.title}</p>
                    <p className={`text-sm ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>{activity.description}</p>
                    <p className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>{new Date(activity.timestamp).toLocaleDateString()}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Clock className={`h-12 w-12 ${isDark ? 'text-slate-500' : 'text-slate-400'} mx-auto mb-4`} />
              <p className={isDark ? 'text-slate-400' : 'text-slate-500'}>No recent activity.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;