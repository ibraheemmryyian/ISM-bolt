import React, { useState, useEffect, Fragment, useCallback, useMemo } from 'react';
import { supabase } from '../lib/supabase';
import { 
  Users, Settings, Database, Activity, Crown, BarChart3, 
  TrendingUp, Target, Zap, Globe, Shield, Eye, Filter,
  Download, RefreshCw, AlertTriangle, CheckCircle, XCircle,
  ArrowUpRight, ArrowDownRight, DollarSign, Leaf, Factory,
  Truck, Building2, Package, Recycle, Lightbulb, Droplets, Copy
} from 'lucide-react';
import SubscriptionManager from './SubscriptionManager';
import { 
  Company, 
  Material, 
  Match, 
  CompanyApplication, 
  AdminStats, 
  AIInsights,
  ApiResponse 
} from '../types';

// Separate components for better performance
const AdminStatsCard = React.memo(({ stats }: { stats: AdminStats }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
    <div className="bg-white p-4 rounded-lg shadow">
      <div className="flex items-center">
        <Users className="h-8 w-8 text-blue-600" />
        <div className="ml-3">
          <p className="text-sm font-medium text-gray-600">Total Companies</p>
          <p className="text-2xl font-bold text-gray-900">{stats.total_companies}</p>
        </div>
      </div>
    </div>
    
    <div className="bg-white p-4 rounded-lg shadow">
      <div className="flex items-center">
        <Package className="h-8 w-8 text-green-600" />
        <div className="ml-3">
          <p className="text-sm font-medium text-gray-600">Total Materials</p>
          <p className="text-2xl font-bold text-gray-900">{stats.total_materials}</p>
        </div>
      </div>
    </div>
    
    <div className="bg-white p-4 rounded-lg shadow">
      <div className="flex items-center">
        <Target className="h-8 w-8 text-purple-600" />
        <div className="ml-3">
          <p className="text-sm font-medium text-gray-600">Total Matches</p>
          <p className="text-2xl font-bold text-gray-900">{stats.total_matches}</p>
        </div>
      </div>
    </div>
    
    <div className="bg-white p-4 rounded-lg shadow">
      <div className="flex items-center">
        <DollarSign className="h-8 w-8 text-yellow-600" />
        <div className="ml-3">
          <p className="text-sm font-medium text-gray-600">Monthly Revenue</p>
          <p className="text-2xl font-bold text-gray-900">${stats.revenue_monthly.toLocaleString()}</p>
        </div>
      </div>
    </div>
  </div>
));

const CompaniesTable = React.memo(({ 
  companies, 
  onToggleRole, 
  onUpdateSubscription, 
  onBulkAction 
}: {
  companies: Company[];
  onToggleRole: (userId: string, currentRole: string) => Promise<void>;
  onUpdateSubscription: (userId: string, tier: string) => Promise<void>;
  onBulkAction: (userId: string, action: string) => Promise<void>;
}) => {
  const [selectedCompanies, setSelectedCompanies] = useState<Set<string>>(new Set());

  const handleSelectAll = useCallback(() => {
    if (selectedCompanies.size === companies.length) {
      setSelectedCompanies(new Set());
    } else {
      setSelectedCompanies(new Set(companies.map(c => c.id)));
    }
  }, [companies, selectedCompanies.size]);

  const handleSelectCompany = useCallback((companyId: string) => {
    const newSelected = new Set(selectedCompanies);
    if (newSelected.has(companyId)) {
      newSelected.delete(companyId);
    } else {
      newSelected.add(companyId);
    }
    setSelectedCompanies(newSelected);
  }, [selectedCompanies]);

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900">Companies</h3>
          <div className="flex space-x-2">
            <button
              onClick={handleSelectAll}
              className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
            >
              {selectedCompanies.size === companies.length ? 'Deselect All' : 'Select All'}
            </button>
          </div>
        </div>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                <input
                  type="checkbox"
                  checked={selectedCompanies.size === companies.length}
                  onChange={handleSelectAll}
                  className="rounded border-gray-300"
                />
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Company
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Role
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Subscription
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
                  <input
                    type="checkbox"
                    checked={selectedCompanies.has(company.id)}
                    onChange={() => handleSelectCompany(company.id)}
                    className="rounded border-gray-300"
                  />
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div>
                    <div className="text-sm font-medium text-gray-900">{company.name}</div>
                    <div className="text-sm text-gray-500">{company.email}</div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                    company.role === 'admin' ? 'bg-red-100 text-red-800' :
                    company.role === 'moderator' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {company.role}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                    company.subscription?.tier === 'enterprise' ? 'bg-purple-100 text-purple-800' :
                    company.subscription?.tier === 'premium' ? 'bg-blue-100 text-blue-800' :
                    company.subscription?.tier === 'basic' ? 'bg-green-100 text-green-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {company.subscription?.tier || 'free'}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                  <div className="flex space-x-2">
                    <button
                      onClick={() => onToggleRole(company.id, company.role)}
                      className="text-indigo-600 hover:text-indigo-900"
                    >
                      Toggle Role
                    </button>
                    <button
                      onClick={() => onUpdateSubscription(company.id, company.subscription?.tier || 'free')}
                      className="text-green-600 hover:text-green-900"
                    >
                      Update Sub
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
});

const ApplicationsTable = React.memo(({ 
  applications, 
  onApplicationAction 
}: {
  applications: CompanyApplication[];
  onApplicationAction: (applicationId: string, action: 'approve' | 'reject') => Promise<void>;
}) => (
  <div className="bg-white rounded-lg shadow overflow-hidden">
    <div className="px-6 py-4 border-b border-gray-200">
      <h3 className="text-lg font-medium text-gray-900">Company Applications</h3>
    </div>
    
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
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
          {applications.map((application) => (
            <tr key={application.id} className="hover:bg-gray-50">
              <td className="px-6 py-4 whitespace-nowrap">
                <div>
                  <div className="text-sm font-medium text-gray-900">{application.company_name}</div>
                  <div className="text-sm text-gray-500">{application.contact_email}</div>
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-gray-900">{application.contact_name}</div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                  application.status === 'approved' ? 'bg-green-100 text-green-800' :
                  application.status === 'rejected' ? 'bg-red-100 text-red-800' :
                  'bg-yellow-100 text-yellow-800'
                }`}>
                  {application.status}
                </span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                {application.status === 'pending' && (
                  <div className="flex space-x-2">
                    <button
                      onClick={() => onApplicationAction(application.id, 'approve')}
                      className="text-green-600 hover:text-green-900"
                    >
                      Approve
                    </button>
                    <button
                      onClick={() => onApplicationAction(application.id, 'reject')}
                      className="text-red-600 hover:text-red-900"
                    >
                      Reject
                    </button>
                  </div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
));

export function AdminHub() {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [materials, setMaterials] = useState<Material[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [applications, setApplications] = useState<CompanyApplication[]>([]);
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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Memoized calculations
  const calculatedStats = useMemo(() => {
    const totalCompanies = companies.length;
    const totalMaterials = materials.length;
    const totalMatches = matches.length;
    const activeMatches = matches.filter(m => m.status === 'active').length;
    const pendingApplications = applications.filter(a => a.status === 'pending').length;
    
    const activeSubscriptions = companies.filter(c => 
      c.subscription?.status === 'active'
    ).length;
    
    const revenueMonthly = companies.reduce((total, company) => {
      const tier = company.subscription?.tier;
      const monthlyRevenue = tier === 'enterprise' ? 500 : 
                           tier === 'premium' ? 100 : 
                           tier === 'basic' ? 25 : 0;
      return total + monthlyRevenue;
    }, 0);

    const totalPotentialValue = matches.reduce((total, match) => 
      total + (match.economic_value || 0), 0
    );

    const totalCarbonReduction = matches.reduce((total, match) => 
      total + (match.sustainability_impact || 0), 0
    );

    const averageSustainabilityScore = companies.length > 0 ? 
      companies.reduce((total, company) => 
        total + (company.sustainability_score || 0), 0
      ) / companies.length : 0;

    return {
      total_companies: totalCompanies,
      total_materials: totalMaterials,
      total_connections: totalMatches,
      active_subscriptions: activeSubscriptions,
      revenue_monthly: revenueMonthly,
      pending_applications: pendingApplications,
      total_matches: totalMatches,
      active_matches: activeMatches,
      total_ai_listings: materials.filter(m => m.ai_generated).length,
      total_potential_value: totalPotentialValue,
      total_carbon_reduction: totalCarbonReduction,
      average_sustainability_score: averageSustainabilityScore,
      system_health_score: calculateSystemHealthScore(companies, materials, matches)
    };
  }, [companies, materials, matches, applications]);

  // Update stats when calculated stats change
  useEffect(() => {
    setStats(calculatedStats);
  }, [calculatedStats]);

  const loadAdminData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Load companies
      const { data: companiesData, error: companiesError } = await supabase
        .from('companies')
        .select('*')
        .order('created_at', { ascending: false });

      if (companiesError) throw companiesError;

      // Load materials with company info
      const { data: materialsData, error: materialsError } = await supabase
        .from('materials')
        .select(`
          *,
          company:companies(name, industry, location)
        `)
        .order('created_at', { ascending: false });

      if (materialsError) throw materialsError;

      // Load matches with company and material info
      const { data: matchesData, error: matchesError } = await supabase
        .from('matches')
        .select(`
          *,
          waste_company:companies!matches_waste_company_id_fkey(name, industry, location),
          requirement_company:companies!matches_requirement_company_id_fkey(name, industry, location),
          waste_material:materials!matches_waste_material_id_fkey(material_name, quantity, unit),
          requirement_material:materials!matches_requirement_material_id_fkey(material_name, quantity, unit)
        `)
        .order('created_at', { ascending: false });

      if (matchesError) throw matchesError;

      // Load company applications
      const { data: applicationsData, error: applicationsError } = await supabase
        .from('company_applications')
        .select('*')
        .order('created_at', { ascending: false });

      if (applicationsError) throw applicationsError;

      setCompanies(companiesData || []);
      setMaterials(materialsData || []);
      setMatches(matchesData || []);
      setApplications(applicationsData || []);

    } catch (err) {
      console.error('Error loading admin data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load admin data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAdminData();
  }, [loadAdminData]);

  const calculateSystemHealthScore = useCallback((companies: Company[], materials: Material[], matches: Match[]): number => {
    const totalCompanies = companies.length;
    const totalMaterials = materials.length;
    const totalMatches = matches.length;
    
    if (totalCompanies === 0) return 0;
    
    const materialPerCompany = totalMaterials / totalCompanies;
    const matchesPerCompany = totalMatches / totalCompanies;
    
    // Calculate health score based on various metrics
    let score = 0;
    
    // Company engagement (30%)
    score += Math.min(totalCompanies / 100, 1) * 30;
    
    // Material diversity (25%)
    score += Math.min(materialPerCompany / 10, 1) * 25;
    
    // Match success (25%)
    score += Math.min(matchesPerCompany / 5, 1) * 25;
    
    // System activity (20%)
    const activeMatches = matches.filter(m => m.status === 'active').length;
    score += Math.min(activeMatches / totalMatches, 1) * 20;
    
    return Math.round(score);
  }, []);

  const generateAIInsights = useCallback((companies: Company[], materials: Material[], matches: Match[]): AIInsights => {
    const totalCompanies = companies.length;
    const totalMaterials = materials.length;
    const totalMatches = matches.length;
    
    const highValueOpportunities = matches.filter(m => m.economic_value > 10000).length;
    const sustainabilityImprovements = matches.filter(m => m.sustainability_impact > 0).length;
    const logisticsOptimizations = matches.filter(m => m.logistics_score > 0.7).length;
    
    const symbiosisPotential = totalMatches > 0 ? 
      matches.reduce((total, match) => total + (match.symbiosis_potential || 0), 0) / totalMatches : 0;
    
    return {
      high_value_opportunities: highValueOpportunities,
      sustainability_improvements: sustainabilityImprovements,
      logistics_optimizations: logisticsOptimizations,
      market_trends: {},
      symbiosis_potential: symbiosisPotential,
      recommendations: []
    };
  }, []);

  const toggleUserRole = useCallback(async (userId: string, currentRole: string) => {
    try {
      const newRole = currentRole === 'user' ? 'admin' : 'user';
      
      const { error } = await supabase
        .from('companies')
        .update({ role: newRole })
        .eq('id', userId);
      
      if (error) throw error;
      
      // Update local state
      setCompanies(prev => prev.map(company => 
        company.id === userId ? { ...company, role: newRole } : company
      ));

    } catch (err) {
      console.error('Error toggling user role:', err);
      setError(err instanceof Error ? err.message : 'Failed to toggle user role');
    }
  }, []);

  const updateSubscriptionTier = useCallback(async (userId: string, tier: string) => {
    try {
      const newTier = tier === 'free' ? 'basic' : tier === 'basic' ? 'premium' : 'enterprise';
      
        const { error } = await supabase
          .from('companies')
          .update({ 
          subscription: { tier: newTier, status: 'active' }
        })
        .eq('id', userId);
      
      if (error) throw error;
      
      // Update local state
      setCompanies(prev => prev.map(company => 
        company.id === userId ? { 
          ...company, 
          subscription: { tier: newTier, status: 'active' }
        } : company
      ));

    } catch (err) {
      console.error('Error updating subscription:', err);
      setError(err instanceof Error ? err.message : 'Failed to update subscription');
    }
  }, []);

  const handleBulkAction = useCallback(async (userId: string, action: string) => {
    try {
      switch (action) {
        case 'upgrade':
          await updateSubscriptionTier(userId, 'free');
          break;
        case 'downgrade':
          await updateSubscriptionTier(userId, 'enterprise');
          break;
        default:
          console.warn('Unknown bulk action:', action);
      }
    } catch (err) {
      console.error('Error performing bulk action:', err);
      setError(err instanceof Error ? err.message : 'Failed to perform bulk action');
    }
  }, [updateSubscriptionTier]);

  const handleApplicationAction = useCallback(async (applicationId: string, action: 'approve' | 'reject') => {
    try {
      const { error } = await supabase
        .from('company_applications')
        .update({
          status: action === 'approve' ? 'approved' : 'rejected',
          reviewed_at: new Date().toISOString()
        })
        .eq('id', applicationId);

      if (error) throw error;

      // Update local state
      setApplications(prev => prev.map(app => 
        app.id === applicationId ? { 
          ...app, 
          status: action === 'approve' ? 'approved' : 'rejected',
          reviewed_at: new Date().toISOString()
        } : app
      ));

    } catch (err) {
      console.error('Error handling application action:', err);
      setError(err instanceof Error ? err.message : 'Failed to handle application action');
    }
  }, []);

  const handleLogout = useCallback(() => {
    supabase.auth.signOut();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-300 rounded w-1/4 mb-6"></div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="bg-white p-4 rounded-lg shadow">
                  <div className="h-16 bg-gray-300 rounded"></div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
  return (
      <div className="min-h-screen bg-gray-100 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex">
              <AlertTriangle className="h-5 w-5 text-red-400" />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <div className="mt-2 text-sm text-red-700">{error}</div>
            <button
                  onClick={loadAdminData}
                  className="mt-3 text-sm text-red-800 hover:text-red-900 underline"
            >
                  Try again
            </button>
          </div>
        </div>
      </div>
              </div>
            </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Admin Hub</h1>
              <p className="mt-2 text-gray-600">Manage the SymbioFlows platform</p>
              </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={loadAdminData}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </button>
              <button
                onClick={handleLogout}
                className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
              >
                Logout
              </button>
          </div>
                      </div>
                    </div>

        {/* Stats Cards */}
        <AdminStatsCard stats={stats} />

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Companies Table */}
          <div>
            <CompaniesTable
              companies={companies}
              onToggleRole={toggleUserRole}
              onUpdateSubscription={updateSubscriptionTier}
              onBulkAction={handleBulkAction}
            />
              </div>

          {/* Applications Table */}
                    <div>
            <ApplicationsTable
              applications={applications}
              onApplicationAction={handleApplicationAction}
            />
                  </div>
                </div>

        {/* AI Insights */}
        <div className="mt-8">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">AI Insights</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{aiInsights.high_value_opportunities}</div>
                <div className="text-sm text-gray-600">High Value Opportunities</div>
                      </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{aiInsights.sustainability_improvements}</div>
                <div className="text-sm text-gray-600">Sustainability Improvements</div>
                  </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">{aiInsights.logistics_optimizations}</div>
                <div className="text-sm text-gray-600">Logistics Optimizations</div>
                </div>
              </div>
            </div>
        </div>
      </div>
    </div>
  );
}