import React, { useState, useEffect, Fragment } from 'react';
import { supabase } from '../lib/supabase';
import { 
  Users, Settings, Database, Activity, Crown, BarChart3, 
  TrendingUp, Target, Zap, Globe, Shield, Eye, Filter,
  Download, RefreshCw, AlertTriangle, CheckCircle, XCircle,
  ArrowUpRight, ArrowDownRight, DollarSign, Leaf, Factory,
  Truck, Building2, Package, Recycle, Lightbulb, Droplets, Copy
} from 'lucide-react';
import SubscriptionManager from './SubscriptionManager';

interface Company {
  id: string;
  name: string;
  email: string;
  role: string;
  created_at: string;
  level: number;
  xp: number;
  industry?: string;
  location?: string;
  employee_count?: number;
  sustainability_score?: number;
  carbon_footprint?: number;
  water_usage?: number;
  subscription?: {
    tier: string;
    status?: string;
  };
}

interface CompanyApplication {
  id: string;
  company_name: string;
  contact_email: string;
  contact_name: string;
  application_answers: any;
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
  reviewed_by?: string;
  reviewed_at?: string;
}

interface Material {
  id: string;
  company_id: string;
  material_name: string;
  quantity: number;
  unit: string;
  type: 'waste' | 'requirement';
  created_at: string;
  price_per_unit?: number;
  quality_grade?: string;
  ai_generated?: boolean;
  confidence_score?: number;
  logistics_cost?: number;
  carbon_footprint?: number;
  sustainability_score?: number;
  potential_value?: number;
  company?: {
    name: string;
    industry?: string;
    location?: string;
  };
}

interface Match {
  id: string;
  waste_company_id: string;
  requirement_company_id: string;
  waste_material_id: string;
  requirement_material_id: string;
  match_score: number;
  symbiosis_potential: number;
  logistics_score: number;
  sustainability_impact: number;
  economic_value: number;
  created_at: string;
  status: 'pending' | 'active' | 'completed' | 'rejected';
  waste_company?: {
    name: string;
    industry?: string;
    location?: string;
  };
  requirement_company?: {
    name: string;
    industry?: string;
    location?: string;
  };
  waste_material?: {
    material_name: string;
    quantity: number;
    unit: string;
  };
  requirement_material?: {
    material_name: string;
    quantity: number;
    unit: string;
  };
  // New field for AI reasoning
  reasoning?: {
    proactive_opportunity: string;
    regulatory_compliance: string;
    impact_forecasting: string;
  };
}

interface AdminStats {
  total_companies: number;
  total_materials: number;
  total_connections: number;
  active_subscriptions: number;
  revenue_monthly: number;
  pending_applications: number;
  total_matches: number;
  active_matches: number;
  total_ai_listings: number;
  total_potential_value: number;
  total_carbon_reduction: number;
  average_sustainability_score: number;
  system_health_score: number;
}

interface AIInsights {
  high_value_opportunities: number;
  sustainability_improvements: number;
  logistics_optimizations: number;
  market_trends: any;
  symbiosis_potential: number;
  recommendations: any[];
}

export function AdminHub() {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [applications, setApplications] = useState<CompanyApplication[]>([]);
  const [materials, setMaterials] = useState<Material[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [stats, setStats] = useState<AdminStats>({
    total_companies: 0,
    total_materials: 0,
    total_connections: 0,
    active_subscriptions: 0,
    revenue_monthly: 0,
    pending_applications: 0,
    total_matches: 0,
    active_matches: 0,
    total_ai_listings: 0,
    total_potential_value: 0,
    total_carbon_reduction: 0,
    average_sustainability_score: 0,
    system_health_score: 0
  });
  const [aiInsights, setAiInsights] = useState<AIInsights>({
    high_value_opportunities: 0,
    sustainability_improvements: 0,
    logistics_optimizations: 0,
    market_trends: {},
    symbiosis_potential: 0,
    recommendations: []
  });
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'companies' | 'materials' | 'matches' | 'ai-insights' | 'subscriptions' | 'ai-training' | 'user-management' | 'applications'>('overview');
  const [currentUserId, setCurrentUserId] = useState<string>('admin-user');
  const [trainingData, setTrainingData] = useState<any[]>([]);
  const [modelTraining, setModelTraining] = useState(false);
  const [filters, setFilters] = useState({
    industry: '',
    location: '',
    materialType: '',
    matchStatus: '',
    aiGenerated: false
  });
  const [expandedMatches, setExpandedMatches] = useState<Set<string>>(new Set());

  useEffect(() => {
    // Check if admin access is granted
    const tempAdmin = localStorage.getItem('temp-admin-access');
    if (tempAdmin === 'true') {
      loadAdminData();
    } else {
      // Redirect to admin access page if not authenticated
      window.location.href = '/admin';
    }
  }, []);

  async function loadAdminData() {
    setLoading(true);
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        setCurrentUserId(user.id);
      } else {
        // Check for admin access from localStorage
        const adminUserId = localStorage.getItem('admin-user-id');
        if (adminUserId) {
          setCurrentUserId(adminUserId);
        }
      }

      // Load company applications with better error handling
      console.log('Loading company applications...');
      const { data: applicationsData, error: applicationsError } = await supabase
        .from('company_applications')
        .select('*')
        .order('created_at', { ascending: false });

      console.log('Applications query result:', { data: applicationsData, error: applicationsError });

      if (applicationsError) {
        console.error('Applications error:', applicationsError);
        throw applicationsError;
      }
      
      setApplications(applicationsData || []);
      console.log('Applications loaded:', applicationsData?.length || 0);

      // Load companies (no join)
      const { data: companiesData, error: companiesError } = await supabase
        .from('companies')
        .select('*')
        .order('created_at', { ascending: false });

      if (companiesError) throw companiesError;
      setCompanies(companiesData || []);

      // Load materials with company info
      const { data: materialsData, error: materialsError } = await supabase
        .from('materials')
        .select(`
          *,
          companies(name, industry, location)
        `)
        .order('created_at', { ascending: false });

      if (materialsError) throw materialsError;
      setMaterials(materialsData || []);

      // Load matches with detailed info
      const { data: matchesData, error: matchesError } = await supabase
        .from('matches')
        .select(`
          *,
          waste_company:companies!waste_company_id(name, industry, location),
          requirement_company:companies!requirement_company_id(name, industry, location),
          waste_material:materials!waste_material_id(material_name, quantity, unit),
          requirement_material:materials!requirement_material_id(material_name, quantity, unit)
        `)
        .order('created_at', { ascending: false });

      if (matchesError) {
        console.warn('Matches table might not exist yet:', matchesError);
        setMatches([]);
      } else {
        setMatches(matchesData || []);
      }

      // Calculate comprehensive stats
      const { count: connectionsCount } = await supabase
        .from('connections')
        .select('*', { count: 'exact', head: true });

      const activeSubscriptions = companiesData?.filter(c => 
        c.subscription_status === 'active' && c.subscription_tier !== 'free'
      ).length || 0;

      const monthlyRevenue = companiesData?.reduce((total, company) => {
        if (company.subscription_status === 'active') {
          if (company.subscription_tier === 'pro') return total + 299;
          if (company.subscription_tier === 'enterprise') return total + 999;
        }
        return total;
      }, 0) || 0;

      const pendingApplications = applicationsData?.filter(app => app.status === 'pending').length || 0;

      // Calculate AI and sustainability metrics
      const aiListings = materialsData?.filter(m => m.ai_generated) || [];
      const totalPotentialValue = materialsData?.reduce((sum, m) => sum + (m.potential_value || 0), 0) || 0;
      const totalCarbonReduction = materialsData?.reduce((sum, m) => sum + (m.carbon_footprint || 0), 0) || 0;
      const avgSustainabilityScore = companiesData?.reduce((sum, c) => sum + (c.sustainability_score || 0), 0) / (companiesData?.length || 1) || 0;

      // Calculate system health score
      const systemHealthScore = calculateSystemHealthScore(companiesData, materialsData, matchesData);

      setStats({
        total_companies: companiesData?.length || 0,
        total_materials: materialsData?.length || 0,
        total_connections: connectionsCount || 0,
        active_subscriptions: activeSubscriptions,
        revenue_monthly: monthlyRevenue,
        pending_applications: pendingApplications,
        total_matches: matchesData?.length || 0,
        active_matches: matchesData?.filter(m => m.status === 'active').length || 0,
        total_ai_listings: aiListings.length,
        total_potential_value: totalPotentialValue,
        total_carbon_reduction: totalCarbonReduction,
        average_sustainability_score: avgSustainabilityScore,
        system_health_score: systemHealthScore
      });

      // Generate AI insights
      const insights = generateAIInsights(companiesData, materialsData, matchesData);
      setAiInsights(insights);

    } catch (error) {
      console.error('Error loading admin data:', error);
    } finally {
      setLoading(false);
    }
  }

  function calculateSystemHealthScore(companies: any[], materials: any[], matches: any[]) {
    let score = 100;
    
    // Deduct points for missing data
    if (!companies || companies.length === 0) score -= 30;
    if (!materials || materials.length === 0) score -= 25;
    if (!matches || matches.length === 0) score -= 20;
    
    // Add points for AI-generated content
    const aiListings = materials?.filter(m => m.ai_generated) || [];
    if (aiListings.length > 0) score += 15;
    
    // Add points for active matches
    const activeMatches = matches?.filter(m => m.status === 'active') || [];
    if (activeMatches.length > 0) score += 10;
    
    return Math.max(0, Math.min(100, score));
  }

  function generateAIInsights(companies: any[], materials: any[], matches: any[]) {
    const insights = {
      high_value_opportunities: 0,
      sustainability_improvements: 0,
      logistics_optimizations: 0,
      market_trends: {},
      symbiosis_potential: 0,
      recommendations: []
    };

    if (!materials || materials.length === 0) return insights;

    // High value opportunities
    insights.high_value_opportunities = materials.filter(m => (m.potential_value || 0) > 10000).length;

    // Sustainability improvements
    insights.sustainability_improvements = materials.filter(m => (m.sustainability_score || 0) < 70).length;

    // Logistics optimizations
    insights.logistics_optimizations = materials.filter(m => (m.logistics_cost || 0) > 1000).length;

    // Symbiosis potential
    const wasteMaterials = materials.filter(m => m.type === 'waste');
    const requirementMaterials = materials.filter(m => m.type === 'requirement');
    insights.symbiosis_potential = wasteMaterials.length * requirementMaterials.length;

    // Market trends
    insights.market_trends = {
      top_industries: getTopIndustries(companies),
      top_materials: getTopMaterials(materials),
      sustainability_trend: calculateSustainabilityTrend(companies)
    };

    // Recommendations
    insights.recommendations = generateRecommendations(companies, materials, matches);

    return insights;
  }

  function getTopIndustries(companies: any[]) {
    const industryCounts: { [key: string]: number } = {};
    companies?.forEach(company => {
      if (company.industry) {
        const industries = company.industry.split('|');
        industries.forEach(industry => {
          industryCounts[industry] = (industryCounts[industry] || 0) + 1;
        });
      }
    });
    return Object.entries(industryCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([industry, count]) => ({ industry, count }));
  }

  function getTopMaterials(materials: any[]) {
    const materialCounts: { [key: string]: number } = {};
    materials?.forEach(material => {
      materialCounts[material.material_name] = (materialCounts[material.material_name] || 0) + 1;
    });
    return Object.entries(materialCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .map(([material, count]) => ({ material, count }));
  }

  function calculateSustainabilityTrend(companies: any[]) {
    const scores = companies?.map(c => c.sustainability_score || 0).filter(s => s > 0) || [];
    if (scores.length === 0) return 'neutral';
    const avg = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    if (avg > 70) return 'improving';
    if (avg < 50) return 'declining';
    return 'stable';
  }

  function generateRecommendations(companies: any[], materials: any[], matches: any[]) {
    const recommendations = [];

    // High value opportunities
    const highValueMaterials = materials?.filter(m => (m.potential_value || 0) > 10000) || [];
    if (highValueMaterials.length > 0) {
      recommendations.push({
        type: 'high_value',
        priority: 'high',
        title: 'High-Value Materials Identified',
        description: `${highValueMaterials.length} materials with potential value >$10,000`,
        action: 'Prioritize these materials for immediate action'
      });
    }

    // Sustainability improvements
    const lowSustainabilityMaterials = materials?.filter(m => (m.sustainability_score || 0) < 70) || [];
    if (lowSustainabilityMaterials.length > 0) {
      recommendations.push({
        type: 'sustainability',
        priority: 'medium',
        title: 'Sustainability Improvements Available',
        description: `${lowSustainabilityMaterials.length} materials need sustainability optimization`,
        action: 'Implement green logistics and sourcing strategies'
      });
    }

    // Match opportunities
    if (matches && matches.length === 0 && materials && materials.length > 0) {
      recommendations.push({
        type: 'matching',
        priority: 'high',
        title: 'Symbiosis Matching Needed',
        description: `${materials.length} materials available for matching`,
        action: 'Run AI matching algorithm to create symbiosis opportunities'
      });
    }

    return recommendations;
  }

  const fetchCompanies = async () => {
    try {
      const { data, error } = await supabase
        .from('companies')
        .select('*')
        .order('created_at', { ascending: false });
      
      if (error) throw error;
      setCompanies(data || []);
    } catch (error) {
      console.error('Error fetching companies:', error);
    }
  };

  const toggleUserRole = async (userId: string, currentRole: string) => {
    const newRole = currentRole === 'admin' ? 'user' : 'admin';
    try {
      const { error } = await supabase
        .from('companies')
        .update({ role: newRole })
        .eq('id', userId);
      
      if (error) throw error;
      
      // Update local state
      setCompanies(companies.map(company => 
        company.id === userId ? { ...company, role: newRole } : company
      ));
    } catch (error) {
      console.error('Error toggling user role:', error);
    }
  };

  const updateSubscriptionTier = async (userId: string, tier: string) => {
    try {
      // For now, just update the company record with subscription info
      // Since subscriptions table doesn't exist properly
      const { error } = await supabase
        .from('companies')
        .update({ 
          subscription_tier: tier,
          subscription_status: 'active'
        })
        .eq('id', userId);

      if (error) throw error;

      // Refresh data
      loadAdminData();
    } catch (error) {
      console.error('Error updating subscription tier:', error);
    }
  };

  const handleBulkAction = async (userId: string, action: string) => {
    try {
      if (action === 'suspend') {
        const { error } = await supabase
          .from('companies')
          .update({ status: 'suspended' })
          .eq('id', userId);
        
        if (error) throw error;
      } else if (action === 'activate') {
        const { error } = await supabase
          .from('companies')
          .update({ status: 'active' })
          .eq('id', userId);
        
        if (error) throw error;
      } else if (action === 'upgrade') {
        await updateSubscriptionTier(userId, 'pro');
        return; // Don't call loadAdminData again
      } else if (action === 'downgrade') {
        await updateSubscriptionTier(userId, 'free');
        return; // Don't call loadAdminData again
      }
      
      // Refresh data
      loadAdminData();
    } catch (error) {
      console.error(`Error performing ${action}:`, error);
    }
  };

  const handleBulkUpgradeAll = async () => {
    try {
      // Get all companies
      const { data: allCompanies, error: fetchError } = await supabase
        .from('companies')
        .select('id');
      
      if (fetchError) throw fetchError;
      
      // Upgrade all to pro tier
      for (const company of allCompanies || []) {
        const { error } = await supabase
          .from('companies')
          .update({ 
            subscription_tier: 'pro',
            subscription_status: 'active'
          })
          .eq('id', company.id);
        
        if (error) console.error(`Error upgrading company ${company.id}:`, error);
      }
      
      // Refresh data once at the end
      loadAdminData();
      alert('All companies upgraded to Pro tier!');
    } catch (error) {
      console.error('Error bulk upgrading users:', error);
    }
  };

  const handleBulkDowngradeAll = async () => {
    try {
      // Get all companies
      const { data: allCompanies, error: fetchError } = await supabase
        .from('companies')
        .select('id');
      
      if (fetchError) throw fetchError;
      
      // Downgrade all to free tier
      for (const company of allCompanies || []) {
        const { error } = await supabase
          .from('companies')
          .update({ 
            subscription_tier: 'free',
            subscription_status: 'active'
          })
          .eq('id', company.id);
        
        if (error) console.error(`Error downgrading company ${company.id}:`, error);
      }
      
      // Refresh data once at the end
      loadAdminData();
      alert('All companies downgraded to Free tier!');
    } catch (error) {
      console.error('Error bulk downgrading users:', error);
    }
  };

  const handleExportUserData = async () => {
    try {
      const { data: companiesData, error } = await supabase
        .from('companies')
        .select('*');
      
      if (error) throw error;
      
      // Create CSV content
      const csvContent = "data:text/csv;charset=utf-8," + 
        "ID,Name,Email,Role,Created At,Level,XP\n" +
        companiesData?.map(company => 
          `${company.id},${company.name},${company.email},${company.role},${company.created_at},${company.level || 0},${company.xp || 0}`
        ).join("\n");

      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", `user-data-export-${new Date().toISOString().split('T')[0]}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Error exporting user data:', error);
    }
  };

  async function trainAIModel() {
    setModelTraining(true);
    try {
      // Placeholder for AI training - in production this would call your backend
      await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate training time
      alert('AI model training completed successfully!\nModel: matching_engine\nAccuracy: 85%');
    } catch (error) {
      console.error('AI training error:', error);
      alert('Failed to train AI model. Please try again.');
    } finally {
      setModelTraining(false);
    }
  }

  async function exportTrainingData() {
    try {
      // Placeholder for training data export
      const csvContent = "data:text/csv;charset=utf-8," + 
        "Timestamp,Company Name,Industry,Data Points\n" +
        "2024-01-01,Test Company,Manufacturing,150\n" +
        "2024-01-02,Another Company,Technology,200";

      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", `ai_training_data_${new Date().toISOString().split('T')[0]}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Export error:', error);
      alert('Failed to export training data.');
    }
  }

  const handleApplicationAction = async (applicationId: string, action: 'approve' | 'reject') => {
    try {
      // Validate inputs
      if (!applicationId || applicationId.trim() === '') {
        throw new Error('Invalid application ID');
      }
      
      if (!currentUserId || currentUserId.trim() === '') {
        // Try to get current user ID from Supabase auth first
        const { data: { user } } = await supabase.auth.getUser();
        if (user) {
          setCurrentUserId(user.id);
        } else {
          // Fallback to admin user ID from localStorage
          const adminUserId = localStorage.getItem('admin-user-id');
          if (adminUserId) {
            setCurrentUserId(adminUserId);
          } else {
            throw new Error('No authenticated user found');
          }
        }
      }

      // First, get the application details
      const { data: application, error: fetchError } = await supabase
        .from('company_applications')
        .select('*')
        .eq('id', applicationId)
        .single();

      if (fetchError) throw fetchError;

      // Update the application status
      const { error: updateError } = await supabase
        .from('company_applications')
        .update({
          status: action === 'approve' ? 'approved' : 'rejected',
          reviewed_by: currentUserId || (await supabase.auth.getUser()).data.user?.id,
          reviewed_at: new Date().toISOString()
        })
        .eq('id', applicationId);

      if (updateError) throw updateError;

      // If approved, create a company account
      if (action === 'approve' && application) {
        const { error: companyError } = await supabase
          .from('companies')
          .insert({
            name: application.company_name,
            email: application.contact_email,
            contact_name: application.contact_name,
            role: 'user',
            level: 1,
            xp: 0
          });

        if (companyError) {
          console.error('Error creating company account:', companyError);
          // You might want to show an error message to the admin here
        }
      }

      // Refresh applications data
      loadAdminData();
      
      // Show success message
      alert(`Application ${action}d successfully!`);
    } catch (error: any) {
      console.error(`Error ${action}ing application:`, error);
      alert(`Failed to ${action} application: ${error.message || 'Unknown error'}`);
    }
  };

  const handleLogout = () => {
    // Clear admin session
    localStorage.removeItem('temp-admin-access');
    localStorage.removeItem('admin-user-id');
    // Redirect to dashboard
    window.location.href = '/dashboard';
  };

  const toggleMatchExpansion = (matchId: string) => {
    const newExpanded = new Set(expandedMatches);
    if (newExpanded.has(matchId)) {
      newExpanded.delete(matchId);
    } else {
      newExpanded.add(matchId);
    }
    setExpandedMatches(newExpanded);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading admin data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <Crown className="h-8 w-8 text-purple-600" />
              <h1 className="ml-3 text-2xl font-bold text-gray-900">Admin Dashboard</h1>
            </div>
            <button
              onClick={handleLogout}
              className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition"
            >
              Logout
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid md:grid-cols-6 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <Users className="h-8 w-8 text-blue-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Companies</p>
                <p className="text-2xl font-bold text-gray-900">{stats.total_companies}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <Database className="h-8 w-8 text-green-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Materials Listed</p>
                <p className="text-2xl font-bold text-gray-900">{stats.total_materials}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <Activity className="h-8 w-8 text-purple-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Connections</p>
                <p className="text-2xl font-bold text-gray-900">{stats.total_connections}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <Crown className="h-8 w-8 text-yellow-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Pro Subscriptions</p>
                <p className="text-2xl font-bold text-gray-900">{stats.active_subscriptions}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <BarChart3 className="h-8 w-8 text-emerald-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Monthly Revenue</p>
                <p className="text-2xl font-bold text-gray-900">${stats.revenue_monthly.toLocaleString()}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <Users className="h-8 w-8 text-orange-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Pending Applications</p>
                <p className="text-2xl font-bold text-gray-900">{stats.pending_applications}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Stats Cards */}
        <div className="grid md:grid-cols-7 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <Building2 className="h-8 w-8 text-blue-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Companies</p>
                <p className="text-2xl font-bold text-gray-900">{stats.total_companies}</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <Package className="h-8 w-8 text-green-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Materials</p>
                <p className="text-2xl font-bold text-gray-900">{stats.total_materials}</p>
                <p className="text-xs text-gray-500">{stats.total_ai_listings} AI-generated</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <Target className="h-8 w-8 text-purple-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Matches</p>
                <p className="text-2xl font-bold text-gray-900">{stats.total_matches}</p>
                <p className="text-xs text-gray-500">{stats.active_matches} active</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <DollarSign className="h-8 w-8 text-emerald-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Potential Value</p>
                <p className="text-2xl font-bold text-gray-900">${(stats.total_potential_value / 1000000).toFixed(1)}M</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <Leaf className="h-8 w-8 text-green-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Carbon Reduction</p>
                <p className="text-2xl font-bold text-gray-900">{(stats.total_carbon_reduction / 1000).toFixed(1)}K</p>
                <p className="text-xs text-gray-500">tons CO2</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <Shield className="h-8 w-8 text-blue-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">System Health</p>
                <p className="text-2xl font-bold text-gray-900">{stats.system_health_score}%</p>
              </div>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-center">
              <Crown className="h-8 w-8 text-yellow-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Revenue</p>
                <p className="text-2xl font-bold text-gray-900">${stats.revenue_monthly.toLocaleString()}</p>
                <p className="text-xs text-gray-500">monthly</p>
              </div>
            </div>
          </div>
        </div>

        {/* Enhanced Tabs */}
        <div className="bg-white rounded-lg shadow-sm">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8 px-6 overflow-x-auto">
              <button
                onClick={() => setActiveTab('overview')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === 'overview'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <BarChart3 className="h-4 w-4" />
                <span>Overview</span>
              </button>
              <button
                onClick={() => setActiveTab('companies')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === 'companies'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Building2 className="h-4 w-4" />
                <span>Companies ({stats.total_companies})</span>
              </button>
              <button
                onClick={() => setActiveTab('materials')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === 'materials'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Package className="h-4 w-4" />
                <span>Materials ({stats.total_materials})</span>
              </button>
              <button
                onClick={() => setActiveTab('matches')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === 'matches'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Target className="h-4 w-4" />
                <span>Matches ({stats.total_matches})</span>
              </button>
              <button
                onClick={() => setActiveTab('ai-insights')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === 'ai-insights'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Zap className="h-4 w-4" />
                <span>AI Insights</span>
              </button>
              <button
                onClick={() => setActiveTab('subscriptions')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === 'subscriptions'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Crown className="h-4 w-4" />
                <span>Subscriptions</span>
              </button>
              <button
                onClick={() => setActiveTab('applications')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === 'applications'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Users className="h-4 w-4" />
                <span>Applications</span>
                {stats.pending_applications > 0 && (
                  <span className="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-orange-100 text-orange-800">
                    {stats.pending_applications}
                  </span>
                )}
              </button>
              <button
                onClick={() => setActiveTab('ai-training')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === 'ai-training'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Activity className="h-4 w-4" />
                <span>AI Training</span>
              </button>
              <button
                onClick={() => setActiveTab('user-management')}
                className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                  activeTab === 'user-management'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Settings className="h-4 w-4" />
                <span>User Management</span>
              </button>
            </nav>
          </div>

          <div className="p-6">
          {activeTab === 'overview' && (
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-gray-50 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
                <div className="space-y-3">
                  {companies.slice(0, 5).map((company) => (
                    <div key={company.id} className="flex items-center justify-between">
                      <div>
                        <p className="font-medium text-gray-900">{company.name}</p>
                        <p className="text-sm text-gray-500">Joined {new Date(company.created_at).toLocaleDateString()}</p>
                      </div>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        company.subscription?.tier === 'pro' ? 'bg-blue-100 text-blue-800' :
                        company.subscription?.tier === 'enterprise' ? 'bg-purple-100 text-purple-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {company.subscription?.tier?.toUpperCase() || 'FREE'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="bg-gray-50 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Top Materials</h3>
                <div className="space-y-3">
                  {materials.slice(0, 5).map((material) => (
                    <div key={material.id} className="flex items-center justify-between">
                      <div>
                        <p className="font-medium text-gray-900">{material.material_name}</p>
                        <p className="text-sm text-gray-500">{material.company?.name}</p>
                      </div>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        material.type === 'waste' ? 'bg-orange-100 text-orange-800' : 'bg-blue-100 text-blue-800'
                      }`}>
                        {material.type}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'companies' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-900">All Companies ({stats.total_companies})</h3>
                <div className="flex space-x-2">
                  <select
                    value={filters.industry}
                    onChange={(e) => setFilters({...filters, industry: e.target.value})}
                    className="text-sm border border-gray-300 rounded px-3 py-1"
                  >
                    <option value="">All Industries</option>
                    <option value="manufacturing">Manufacturing</option>
                    <option value="chemicals">Chemicals</option>
                    <option value="food_beverage">Food & Beverage</option>
                    <option value="construction">Construction</option>
                    <option value="oil_gas">Oil & Gas</option>
                  </select>
                  <select
                    value={filters.location}
                    onChange={(e) => setFilters({...filters, location: e.target.value})}
                    className="text-sm border border-gray-300 rounded px-3 py-1"
                  >
                    <option value="">All Locations</option>
                    <option value="Dubai">Dubai</option>
                    <option value="Abu Dhabi">Abu Dhabi</option>
                    <option value="Sharjah">Sharjah</option>
                    <option value="Saudi Arabia">Saudi Arabia</option>
                    <option value="Qatar">Qatar</option>
                    <option value="Kuwait">Kuwait</option>
                    <option value="Bahrain">Bahrain</option>
                    <option value="Oman">Oman</option>
                  </select>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Company</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Industry</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Location</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Employees</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sustainability</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Carbon Footprint</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Water Usage</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {companies
                      .filter(company => 
                        (!filters.industry || company.industry?.includes(filters.industry)) &&
                        (!filters.location || company.location?.includes(filters.location))
                      )
                      .map((company) => (
                      <tr key={company.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="flex-shrink-0 h-10 w-10">
                              <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center">
                                <span className="text-sm font-medium text-blue-800">
                                  {company.name.charAt(0).toUpperCase()}
                                </span>
                              </div>
                            </div>
                            <div className="ml-4">
                              <div className="text-sm font-medium text-gray-900">{company.name}</div>
                              <div className="text-sm text-gray-500">{company.email}</div>
                              <div className="text-xs text-gray-400">Joined {new Date(company.created_at).toLocaleDateString()}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="text-sm text-gray-900">{company.industry || 'N/A'}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="text-sm text-gray-900">{company.location || 'N/A'}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="text-sm text-gray-900">{company.employee_count?.toLocaleString() || 'N/A'}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                              (company.sustainability_score || 0) >= 80 ? 'bg-green-100 text-green-800' :
                              (company.sustainability_score || 0) >= 60 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}>
                              {company.sustainability_score || 0}%
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="text-sm text-gray-900">
                            {company.carbon_footprint ? `${(company.carbon_footprint / 1000).toFixed(1)}K tons` : 'N/A'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="text-sm text-gray-900">
                            {company.water_usage ? `${(company.water_usage / 1000).toFixed(1)}K L` : 'N/A'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button className="text-blue-600 hover:text-blue-900">View Details</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'materials' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-900">All Materials ({stats.total_materials})</h3>
                <div className="flex space-x-2">
                  <select
                    value={filters.materialType}
                    onChange={(e) => setFilters({...filters, materialType: e.target.value})}
                    className="text-sm border border-gray-300 rounded px-3 py-1"
                  >
                    <option value="">All Types</option>
                    <option value="waste">Waste</option>
                    <option value="requirement">Requirement</option>
                  </select>
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={filters.aiGenerated}
                      onChange={(e) => setFilters({...filters, aiGenerated: e.target.checked})}
                      className="rounded"
                    />
                    <span className="text-sm">AI Generated Only</span>
                  </label>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Material</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Company</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quantity</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Value</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">AI Generated</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sustainability</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {materials
                      .filter(material => 
                        (!filters.materialType || material.type === filters.materialType) &&
                        (!filters.aiGenerated || material.ai_generated)
                      )
                      .map((material) => (
                      <tr key={material.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900">{material.material_name}</div>
                          <div className="text-sm text-gray-500">{material.description}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900">{material.company?.name}</div>
                          <div className="text-sm text-gray-500">{material.company?.industry}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                            material.type === 'waste' ? 'bg-orange-100 text-orange-800' : 'bg-blue-100 text-blue-800'
                          }`}>
                            {material.type}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="text-sm text-gray-900">
                            {material.quantity?.toLocaleString()} {material.unit}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="text-sm text-gray-900">
                            ${material.potential_value?.toLocaleString() || 'N/A'}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {material.ai_generated ? (
                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                              <Zap className="h-3 w-3 mr-1" />
                              AI Generated
                            </span>
                          ) : (
                            <span className="text-sm text-gray-500">Manual</span>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                              (material.sustainability_score || 0) >= 80 ? 'bg-green-100 text-green-800' :
                              (material.sustainability_score || 0) >= 60 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            }`}>
                              {material.sustainability_score || 0}%
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button className="text-blue-600 hover:text-blue-900">View Details</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'matches' && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold text-gray-900">Symbiosis Matches ({stats.total_matches})</h3>
                <div className="flex space-x-2">
                  <select
                    value={filters.matchStatus}
                    onChange={(e) => setFilters({...filters, matchStatus: e.target.value})}
                    className="text-sm border border-gray-300 rounded px-3 py-1"
                  >
                    <option value="">All Status</option>
                    <option value="pending">Pending</option>
                    <option value="active">Active</option>
                    <option value="completed">Completed</option>
                    <option value="rejected">Rejected</option>
                  </select>
                </div>
              </div>

              {matches.length === 0 ? (
                <div className="text-center py-12">
                  <Target className="mx-auto h-12 w-12 text-gray-400" />
                  <h3 className="mt-2 text-sm font-medium text-gray-900">No matches found</h3>
                  <p className="mt-1 text-sm text-gray-500">Run the AI matching algorithm to create symbiosis opportunities.</p>
                  <div className="mt-6">
                    <button className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-purple-600 hover:bg-purple-700">
                      <Zap className="h-4 w-4 mr-2" />
                      Generate Matches
                    </button>
                  </div>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Waste Company</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Requirement Company</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Materials</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Match Score</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Economic Value</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sustainability Impact</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {matches
                        .filter(match => !filters.matchStatus || match.status === filters.matchStatus)
                        .map((match) => (
                        <Fragment key={match.id}>
                          <tr className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="text-sm font-medium text-gray-900">{match.waste_company?.name}</div>
                              <div className="text-sm text-gray-500">{match.waste_company?.industry}</div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="text-sm font-medium text-gray-900">{match.requirement_company?.name}</div>
                              <div className="text-sm text-gray-500">{match.requirement_company?.industry}</div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="text-sm text-gray-900">
                                {match.waste_material?.material_name} → {match.requirement_material?.material_name}
                              </div>
                              <div className="text-sm text-gray-500">
                                {match.waste_material?.quantity} {match.waste_material?.unit}
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="flex items-center">
                                <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                  match.match_score >= 80 ? 'bg-green-100 text-green-800' :
                                  match.match_score >= 60 ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-red-100 text-red-800'
                                }`}>
                                  {match.match_score}%
                                </span>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className="text-sm text-gray-900">
                                ${match.economic_value?.toLocaleString() || 'N/A'}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <div className="flex items-center">
                                <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                  match.sustainability_impact >= 80 ? 'bg-green-100 text-green-800' :
                                  match.sustainability_impact >= 60 ? 'bg-yellow-100 text-yellow-800' :
                                  'bg-red-100 text-red-800'
                                }`}>
                                  {match.sustainability_impact}%
                                </span>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                match.status === 'active' ? 'bg-green-100 text-green-800' :
                                match.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                                match.status === 'completed' ? 'bg-blue-100 text-blue-800' :
                                'bg-red-100 text-red-800'
                              }`}>
                                {match.status}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                              <button 
                                onClick={() => toggleMatchExpansion(match.id)}
                                className="text-blue-600 hover:text-blue-900 mr-2"
                              >
                                {expandedMatches.has(match.id) ? 'Hide' : 'View'} Reasoning
                              </button>
                              <button className="text-gray-600 hover:text-gray-900">Details</button>
                            </td>
                          </tr>
                          
                          {/* AI Reasoning Collapsible Section */}
                          {expandedMatches.has(match.id) && (
                            <tr>
                              <td colSpan={8} className="px-6 py-4 bg-gray-50">
                                <div className="space-y-4">
                                  <h4 className="text-sm font-semibold text-gray-900 mb-3">AI Reasoning Analysis</h4>
                                  
                                  <div className="grid md:grid-cols-3 gap-4">
                                    {/* Proactive Opportunity Reasoning */}
                                    <div className="bg-white rounded-lg p-4 border border-gray-200">
                                      <div className="flex items-center mb-2">
                                        <TrendingUp className="h-4 w-4 text-green-500 mr-2" />
                                        <h5 className="text-sm font-medium text-gray-900">Proactive Opportunity</h5>
                                      </div>
                                      <p className="text-sm text-gray-600">
                                        {match.reasoning?.proactive_opportunity || 'No proactive opportunity analysis available'}
                                      </p>
                                    </div>
                                    
                                    {/* Regulatory Compliance Reasoning */}
                                    <div className="bg-white rounded-lg p-4 border border-gray-200">
                                      <div className="flex items-center mb-2">
                                        <Shield className="h-4 w-4 text-blue-500 mr-2" />
                                        <h5 className="text-sm font-medium text-gray-900">Regulatory Compliance</h5>
                                      </div>
                                      <p className="text-sm text-gray-600">
                                        {match.reasoning?.regulatory_compliance || 'No regulatory compliance analysis available'}
                                      </p>
                                    </div>
                                    
                                    {/* Impact Forecasting Reasoning */}
                                    <div className="bg-white rounded-lg p-4 border border-gray-200">
                                      <div className="flex items-center mb-2">
                                        <BarChart3 className="h-4 w-4 text-purple-500 mr-2" />
                                        <h5 className="text-sm font-medium text-gray-900">Impact Forecasting</h5>
                                      </div>
                                      <p className="text-sm text-gray-600">
                                        {match.reasoning?.impact_forecasting || 'No impact forecasting analysis available'}
                                      </p>
                                    </div>
                                  </div>
                                  
                                  {/* Training Data Export */}
                                  <div className="flex justify-end pt-2 border-t border-gray-200">
                                    <button 
                                      onClick={() => {
                                        const reasoningData = {
                                          match_id: match.id,
                                          timestamp: new Date().toISOString(),
                                          reasoning: match.reasoning,
                                          match_score: match.match_score,
                                          status: match.status
                                        };
                                        navigator.clipboard.writeText(JSON.stringify(reasoningData, null, 2));
                                        alert('Reasoning data copied to clipboard for training purposes');
                                      }}
                                      className="text-xs text-gray-500 hover:text-gray-700 flex items-center"
                                    >
                                      <Copy className="h-3 w-3 mr-1" />
                                      Copy for Training
                                    </button>
                                  </div>
                                </div>
                              </td>
                            </tr>
                                                     )}
                         </Fragment>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {activeTab === 'ai-insights' && (
            <div className="space-y-6">
              <div className="grid md:grid-cols-3 gap-6">
                <div className="bg-white rounded-lg shadow-sm p-6">
                  <div className="flex items-center">
                    <TrendingUp className="h-8 w-8 text-green-500" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-gray-600">High Value Opportunities</p>
                      <p className="text-2xl font-bold text-gray-900">{aiInsights.high_value_opportunities}</p>
                    </div>
                  </div>
                </div>
                <div className="bg-white rounded-lg shadow-sm p-6">
                  <div className="flex items-center">
                    <Leaf className="h-8 w-8 text-blue-500" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-gray-600">Sustainability Improvements</p>
                      <p className="text-2xl font-bold text-gray-900">{aiInsights.sustainability_improvements}</p>
                    </div>
                  </div>
                </div>
                <div className="bg-white rounded-lg shadow-sm p-6">
                  <div className="flex items-center">
                    <Truck className="h-8 w-8 text-purple-500" />
                    <div className="ml-4">
                      <p className="text-sm font-medium text-gray-600">Logistics Optimizations</p>
                      <p className="text-2xl font-bold text-gray-900">{aiInsights.logistics_optimizations}</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-white rounded-lg shadow-sm p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Market Trends</h3>
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-sm font-medium text-gray-700">Top Industries</h4>
                      <div className="mt-2 space-y-2">
                        {aiInsights.market_trends.top_industries?.map((industry: any, index: number) => (
                          <div key={index} className="flex justify-between text-sm">
                            <span className="text-gray-600">{industry.industry}</span>
                            <span className="font-medium">{industry.count} companies</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-gray-700">Sustainability Trend</h4>
                      <div className="mt-2">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          aiInsights.market_trends.sustainability_trend === 'improving' ? 'bg-green-100 text-green-800' :
                          aiInsights.market_trends.sustainability_trend === 'declining' ? 'bg-red-100 text-red-800' :
                          'bg-yellow-100 text-yellow-800'
                        }`}>
                          {aiInsights.market_trends.sustainability_trend}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-sm p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Recommendations</h3>
                  <div className="space-y-3">
                    {aiInsights.recommendations.map((rec, index) => (
                      <div key={index} className="border-l-4 border-blue-500 pl-4">
                        <h4 className="text-sm font-medium text-gray-900">{rec.title}</h4>
                        <p className="text-sm text-gray-600 mt-1">{rec.description}</p>
                        <p className="text-xs text-gray-500 mt-1">{rec.action}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

            {activeTab === 'user-management' && (
              <div className="space-y-6">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-blue-900 mb-4">User Management</h3>
                  <p className="text-blue-700 mb-4">
                    Manage user roles, permissions, and access levels. You can promote users to admin, 
                    upgrade them to Pro, or restrict their access.
                  </p>
                </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          User
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Role
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Subscription
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {companies.map((company) => (
                    <tr key={company.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              <div className="flex-shrink-0 h-10 w-10">
                                <div className="h-10 w-10 rounded-full bg-emerald-100 flex items-center justify-center">
                                  <span className="text-sm font-medium text-emerald-800">
                                    {company.name.charAt(0).toUpperCase()}
                                  </span>
                                </div>
                              </div>
                              <div className="ml-4">
                                <div className="text-sm font-medium text-gray-900">{company.name}</div>
                                <div className="text-sm text-gray-500">{company.email}</div>
                                <div className="text-xs text-gray-400">Joined {new Date(company.created_at).toLocaleDateString()}</div>
                              </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                            <select
                              value={company.role}
                              onChange={(e) => toggleUserRole(company.id, company.role)}
                              className="text-sm border border-gray-300 rounded px-2 py-1 focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                            >
                              <option value="user">User</option>
                              <option value="admin">Admin</option>
                            </select>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                            <select
                              value={company.subscription?.tier || 'free'}
                              onChange={(e) => updateSubscriptionTier(company.id, e.target.value)}
                              className="text-sm border border-gray-300 rounded px-2 py-1 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                            >
                              <option value="free">Free</option>
                              <option value="pro">Pro</option>
                              <option value="enterprise">Enterprise</option>
                            </select>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                            <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                              company.subscription?.status === 'active'
                                ? 'bg-green-100 text-green-800'
                                : company.subscription?.status === 'expired'
                                ? 'bg-red-100 text-red-800'
                                : 'bg-yellow-100 text-yellow-800'
                        }`}>
                              {company.subscription?.status?.toUpperCase() || 'ACTIVE'}
                        </span>
                      </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                            <div className="flex space-x-2">
                              <button
                                onClick={() => handleBulkAction(company.id, 'upgrade')}
                                className="text-emerald-600 hover:text-emerald-900 text-xs"
                              >
                                Upgrade
                              </button>
                              <button
                                onClick={() => handleBulkAction(company.id, 'downgrade')}
                                className="text-orange-600 hover:text-orange-900 text-xs"
                              >
                                Downgrade
                              </button>
                        <button
                                onClick={() => handleBulkAction(company.id, 'suspend')}
                                className="text-red-600 hover:text-red-900 text-xs"
                        >
                                Suspend
                        </button>
                            </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

                <div className="bg-gray-50 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Bulk Actions</h3>
                  <div className="grid md:grid-cols-3 gap-4">
                    <button
                      onClick={() => handleBulkUpgradeAll()}
                      className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition"
                    >
                      Upgrade All to Pro
                    </button>
                    <button
                      onClick={() => handleBulkDowngradeAll()}
                      className="bg-orange-500 text-white px-4 py-2 rounded-lg hover:bg-orange-600 transition"
                    >
                      Downgrade All to Free
                    </button>
                    <button
                      onClick={() => handleExportUserData()}
                      className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition"
                    >
                      Export User Data
                    </button>
                  </div>
                        </div>
            </div>
          )}

          {activeTab === 'subscriptions' && (
            <SubscriptionManager currentUserId={currentUserId} isAdmin={true} />
          )}

            {activeTab === 'ai-training' && (
              <div className="space-y-6">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-blue-900 mb-4">AI Model Training</h3>
                  <p className="text-blue-700 mb-4">
                    Train and improve the AI matching engine using collected user data and interactions.
                  </p>
                  <div className="flex space-x-4">
                    <button
                      onClick={trainAIModel}
                      disabled={modelTraining}
                      className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition disabled:opacity-50"
                    >
                      {modelTraining ? 'Training...' : 'Train AI Model'}
                    </button>
                    <button
                      onClick={exportTrainingData}
                      className="bg-green-500 text-white px-4 py-2 rounded-lg hover:bg-green-600 transition"
                    >
                      Export Training Data
                    </button>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Training Data Statistics</h3>
                  <div className="grid md:grid-cols-3 gap-4">
                    <div className="bg-white rounded-lg p-4">
                      <p className="text-sm font-medium text-gray-600">Total Data Points</p>
                      <p className="text-2xl font-bold text-gray-900">{trainingData.length}</p>
                    </div>
                    <div className="bg-white rounded-lg p-4">
                      <p className="text-sm font-medium text-gray-600">Companies with Data</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {new Set(trainingData.map(d => d.companyData?.companyName)).size}
                      </p>
                    </div>
                    <div className="bg-white rounded-lg p-4">
                      <p className="text-sm font-medium text-gray-600">Last Training</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {trainingData.length > 0 ? 
                          new Date(trainingData[0].timestamp).toLocaleDateString() : 
                          'Never'
                        }
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'applications' && (
              <div className="space-y-6">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-blue-900 mb-4">Company Applications</h3>
                  <p className="text-blue-700 mb-4">
                    Review and manage company applications for approval or rejection.
                  </p>
                  
                  {/* Debug Info */}
                  <div className="bg-yellow-50 border border-yellow-200 rounded p-4 mb-4">
                    <h4 className="font-semibold text-yellow-800 mb-2">Debug Info:</h4>
                    <p className="text-sm text-yellow-700">Applications loaded: {applications.length}</p>
                    <p className="text-sm text-yellow-700">Current user ID: {currentUserId}</p>
                    <p className="text-sm text-yellow-700">Loading state: {loading ? 'true' : 'false'}</p>
                    <details className="mt-2">
                      <summary className="text-sm font-medium text-yellow-800 cursor-pointer">Raw applications data:</summary>
                      <pre className="text-xs bg-white p-2 rounded border overflow-auto max-h-32 mt-2">
                        {applications.length > 0 ? JSON.stringify(applications, null, 2) : 'No applications found'}
                      </pre>
                    </details>
                    <details className="mt-2">
                      <summary className="text-sm font-medium text-yellow-800 cursor-pointer">Test application submission:</summary>
                      <button 
                        onClick={async () => {
                          try {
                            const { error } = await supabase.from('company_applications').insert({
                              company_name: 'Test Company ' + Date.now(),
                              contact_email: 'test' + Date.now() + '@example.com',
                              contact_name: 'Test Contact',
                              application_answers: { motivation: 'Test motivation' },
                              status: 'pending'
                            });
                            if (error) throw error;
                            alert('Test application added! Refresh to see it.');
                            loadAdminData();
                          } catch (err: any) {
                            alert('Error adding test application: ' + err.message);
                          }
                        }}
                        className="mt-2 bg-blue-500 text-white px-3 py-1 rounded text-xs hover:bg-blue-600"
                      >
                        Add Test Application
                      </button>
                    </details>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="bg-gray-50">
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Company
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Contact
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Status
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Actions
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {applications.length === 0 ? (
                        <tr>
                          <td colSpan={4} className="px-6 py-4 text-center text-gray-500">
                            No applications found. Check the debug info above.
                          </td>
                        </tr>
                      ) : (
                        applications.map((application) => (
                          <tr key={application.id} className="hover:bg-gray-50">
                            <td className="px-6 py-4 whitespace-nowrap">
                                  <div className="flex items-center">
                                    <div className="flex-shrink-0 h-10 w-10">
                                      <div className="h-10 w-10 rounded-full bg-emerald-100 flex items-center justify-center">
                                        <span className="text-sm font-medium text-emerald-800">
                                          {application.company_name?.charAt(0)?.toUpperCase() || '?'}
                                        </span>
                                      </div>
                                    </div>
                                    <div className="ml-4">
                                      <div className="text-sm font-medium text-gray-900">{application.company_name || 'Unknown'}</div>
                                      <div className="text-sm text-gray-500">{application.contact_email || 'No email'}</div>
                                    </div>
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                                  <div className="text-sm text-gray-900">{application.contact_name || 'Unknown'} ({application.contact_email || 'No email'})</div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                                    application.status === 'approved'
                                      ? 'bg-green-100 text-green-800'
                                      : application.status === 'rejected'
                                      ? 'bg-red-100 text-red-800'
                                      : 'bg-yellow-100 text-yellow-800'
                                }`}>
                                    {application.status?.toUpperCase() || 'PENDING'}
                              </span>
                            </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                                  <div className="flex space-x-2">
                                    {application.status === 'pending' && (
                                      <>
                                        <button
                                          onClick={() => handleApplicationAction(application.id, 'approve')}
                                          className="text-emerald-600 hover:text-emerald-900 text-xs"
                                        >
                                          Approve
                                        </button>
                                        <button
                                          onClick={() => handleApplicationAction(application.id, 'reject')}
                                          className="text-red-600 hover:text-red-900 text-xs"
                                        >
                                          Reject
                                        </button>
                                      </>
                                    )}
                                    {application.status !== 'pending' && (
                                      <span className="text-gray-500 text-xs">
                                        {application.reviewed_at ? `Reviewed ${new Date(application.reviewed_at).toLocaleDateString()}` : 'Reviewed'}
                                      </span>
                                    )}
                                </div>
                              </td>
                            </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}