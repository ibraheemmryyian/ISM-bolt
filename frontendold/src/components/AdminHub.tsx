import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { Users, Settings, Database, Activity, Crown, BarChart3 } from 'lucide-react';
import { SubscriptionManager } from './SubscriptionManager';

interface Company {
  id: string;
  name: string;
  email: string;
  role: string;
  created_at: string;
  level: number;
  xp: number;
  subscription?: {
    tier: string;
    status?: string;
  };
}

interface Material {
  id: string;
  company_id: string;
  material_name: string;
  quantity: number;
  unit: string;
  type: 'waste' | 'requirement';
  created_at: string;
  company?: {
    name: string;
  };
}

interface AdminStats {
  total_companies: number;
  total_materials: number;
  total_connections: number;
  active_subscriptions: number;
  revenue_monthly: number;
}

export function AdminHub() {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [materials, setMaterials] = useState<Material[]>([]);
  const [stats, setStats] = useState<AdminStats>({
    total_companies: 0,
    total_materials: 0,
    total_connections: 0,
    active_subscriptions: 0,
    revenue_monthly: 0
  });
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'companies' | 'materials' | 'subscriptions' | 'ai-training' | 'user-management'>('overview');
  const [currentUserId, setCurrentUserId] = useState<string>('');
  const [trainingData, setTrainingData] = useState<any[]>([]);
  const [modelTraining, setModelTraining] = useState(false);

  useEffect(() => {
    loadAdminData();
  }, []);

  async function loadAdminData() {
    setLoading(true);
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        setCurrentUserId(user.id);
      }

      // Load companies with subscriptions
      const { data: companiesData, error: companiesError } = await supabase
        .from('companies')
        .select(`
          *,
          subscriptions(tier, status)
        `)
        .order('created_at', { ascending: false });

      if (companiesError) throw companiesError;
      
      const formattedCompanies = companiesData?.map(company => ({
        ...company,
        subscription: company.subscriptions?.[0] || null
      })) || [];
      
      setCompanies(formattedCompanies);

      // Load materials with company info
      const { data: materialsData, error: materialsError } = await supabase
        .from('materials')
        .select(`
          *,
          companies(name)
        `)
        .order('created_at', { ascending: false });

      if (materialsError) throw materialsError;
      setMaterials(materialsData || []);

      // Calculate stats
      const { count: connectionsCount } = await supabase
        .from('connections')
        .select('*', { count: 'exact', head: true });

      const activeSubscriptions = formattedCompanies.filter(c => 
        c.subscription?.status === 'active' && c.subscription?.tier !== 'free'
      ).length;

      const monthlyRevenue = formattedCompanies.reduce((total, company) => {
        if (company.subscription?.status === 'active') {
          if (company.subscription.tier === 'pro') return total + 299;
          if (company.subscription.tier === 'enterprise') return total + 999;
        }
        return total;
      }, 0);

      setStats({
        total_companies: formattedCompanies.length,
        total_materials: materialsData?.length || 0,
        total_connections: connectionsCount || 0,
        active_subscriptions: activeSubscriptions,
        revenue_monthly: monthlyRevenue
      });

    } catch (error) {
      console.error('Error loading admin data:', error);
    } finally {
      setLoading(false);
    }
  }

  const fetchCompanies = async () => {
    try {
      const response = await fetch('/api/admin/companies');
      if (response.ok) {
        const data = await response.json();
        setCompanies(data);
      }
    } catch (error) {
      console.error('Error fetching companies:', error);
    }
  };

  const toggleUserRole = async (userId: string, currentRole: string) => {
      const newRole = currentRole === 'admin' ? 'user' : 'admin';
    try {
      const response = await fetch(`/api/admin/users/${userId}/role`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: newRole })
      });
      if (response.ok) {
        setCompanies(companies.map(company => 
          company.id === userId ? { ...company, role: newRole } : company
        ));
      }
    } catch (error) {
      console.error('Error updating user role:', error);
    }
  };

  const updateSubscriptionTier = async (userId: string, tier: string) => {
    try {
      const response = await fetch(`/api/admin/users/${userId}/subscription`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tier })
      });
      if (response.ok) {
        setCompanies(companies.map(company => 
          company.id === userId ? { 
            ...company, 
            subscription: { ...company.subscription, tier } 
          } : company
        ));
      }
    } catch (error) {
      console.error('Error updating subscription tier:', error);
    }
  };

  const handleBulkAction = async (userId: string, action: string) => {
    try {
      const response = await fetch(`/api/admin/users/${userId}/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        // Refresh companies data
        fetchCompanies();
      }
    } catch (error) {
      console.error(`Error performing ${action}:`, error);
    }
  };

  const handleBulkUpgradeAll = async () => {
    try {
      const response = await fetch('/api/admin/users/bulk-upgrade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        fetchCompanies();
      }
    } catch (error) {
      console.error('Error bulk upgrading users:', error);
    }
  };

  const handleBulkDowngradeAll = async () => {
    try {
      const response = await fetch('/api/admin/users/bulk-downgrade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        fetchCompanies();
      }
    } catch (error) {
      console.error('Error bulk downgrading users:', error);
    }
  };

  const handleExportUserData = async () => {
    try {
      const response = await fetch('/api/admin/users/export', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'user-data-export.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Error exporting user data:', error);
    }
  };

  async function trainAIModel() {
    setModelTraining(true);
    try {
      const response = await fetch('/api/ai-train-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelType: 'matching_engine',
          parameters: {
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32
          }
        })
      });

      if (!response.ok) {
        throw new Error('Failed to train AI model');
      }

      const result = await response.json();
      alert(`AI model training completed successfully!\nModel: ${result.modelType}\nAccuracy: ${result.trainingResult.accuracy || 'N/A'}`);
    } catch (error) {
      console.error('AI training error:', error);
      alert('Failed to train AI model. Please try again.');
    } finally {
      setModelTraining(false);
    }
  }

  async function exportTrainingData() {
    try {
      const response = await fetch('/api/ai-training-data', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error('Failed to export training data');
      }

      const data = await response.json();
      
      // Create and download CSV file
      const csvContent = "data:text/csv;charset=utf-8," + 
        "Timestamp,Company Name,Industry,Data Points\n" +
        data.map((item: any) => 
          `${item.timestamp},${item.companyData.companyName || 'N/A'},${item.companyData.industry || 'N/A'},${item.dataPoints || 0}`
        ).join("\n");

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
                <Settings className="h-8 w-8 text-purple-500" />
                <span className="text-2xl font-bold text-gray-900">Admin Hub</span>
                <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs font-semibold rounded-full">
                  ADMIN ACCESS
                </span>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => window.location.href = '/dashboard'}
                className="text-gray-500 hover:text-gray-700 transition"
              >
                Back to Dashboard
              </button>
              <button
                onClick={() => {
                  localStorage.removeItem('temp-admin-access');
                  window.location.href = '/dashboard';
                }}
                className="text-red-600 hover:text-red-700 transition"
              >
                Exit Admin Mode
              </button>
            </div>
          </div>
                </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid md:grid-cols-5 gap-6 mb-8">
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
          </div>

          {/* Tabs */}
        <div className="bg-white rounded-lg shadow-sm">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8 px-6">
            <button
              onClick={() => setActiveTab('overview')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'overview'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
                Overview
            </button>
            <button
                onClick={() => setActiveTab('user-management')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'user-management'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
                User Management
            </button>
            <button
                onClick={() => setActiveTab('subscriptions')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'subscriptions'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
                Subscriptions
            </button>
            <button
                onClick={() => setActiveTab('ai-training')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'ai-training'
                    ? 'border-purple-500 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
                AI Training
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
          </div>
        </div>
      </div>
    </div>
  );
}