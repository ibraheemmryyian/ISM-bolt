import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { Crown, Check, X, Users, Settings } from 'lucide-react';

interface Subscription {
  id: string;
  company_id: string;
  tier: string;
  status: string;
  created_at: string;
  expires_at?: string;
}

interface Company {
  id: string;
  name: string;
  email: string;
  role: string;
  subscription?: Subscription;
}

interface SubscriptionManagerProps {
  currentUserId: string;
  isAdmin: boolean;
}

export function SubscriptionManager({ currentUserId, isAdmin }: SubscriptionManagerProps) {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState<string | null>(null);

  useEffect(() => {
    if (isAdmin) {
      loadCompaniesWithSubscriptions();
    }
  }, [isAdmin]);

  async function loadCompaniesWithSubscriptions() {
    try {
      const { data: companiesData, error } = await supabase
        .from('companies')
        .select(`
          *,
          subscriptions(*)
        `)
        .order('created_at', { ascending: false });

      if (error) throw error;

      const formattedCompanies = companiesData?.map(company => ({
        ...company,
        subscription: company.subscriptions?.[0] || null
      })) || [];

      setCompanies(formattedCompanies);
    } catch (error) {
      console.error('Error loading companies:', error);
    } finally {
      setLoading(false);
    }
  }

  async function updateSubscriptionTier(companyId: string, newTier: string) {
    setUpdating(companyId);
    try {
      // Check if subscription exists
      const { data: existingSubscription } = await supabase
        .from('subscriptions')
        .select('id')
        .eq('company_id', companyId)
        .single();

      if (existingSubscription) {
        // Update existing subscription
        const { error } = await supabase
          .from('subscriptions')
          .update({
            tier: newTier,
            status: 'active',
            expires_at: newTier === 'free' ? null : new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString()
          })
          .eq('company_id', companyId);

        if (error) throw error;
      } else {
        // Create new subscription
        const { error } = await supabase
          .from('subscriptions')
          .insert({
            company_id: companyId,
            tier: newTier,
            status: 'active',
            expires_at: newTier === 'free' ? null : new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString()
          });

        if (error) throw error;
      }

      // Reload data
      await loadCompaniesWithSubscriptions();
    } catch (error) {
      console.error('Error updating subscription:', error);
    } finally {
      setUpdating(null);
    }
  }

  if (!isAdmin) {
    return (
      <div className="bg-white rounded-xl shadow-sm p-6">
        <div className="text-center">
          <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Access Denied</h3>
          <p className="text-gray-600">You need admin privileges to manage subscriptions.</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-sm p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-16 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-sm p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Crown className="h-6 w-6 text-yellow-500" />
          <h2 className="text-xl font-bold text-gray-900">Subscription Management</h2>
        </div>
        <div className="text-sm text-gray-600">
          {companies.length} total companies
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
                Current Tier
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Expires
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
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                    company.subscription?.tier === 'enterprise' 
                      ? 'bg-purple-100 text-purple-800'
                      : company.subscription?.tier === 'pro'
                      ? 'bg-blue-100 text-blue-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {company.subscription?.tier?.toUpperCase() || 'FREE'}
                  </span>
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
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {company.subscription?.expires_at 
                    ? new Date(company.subscription.expires_at).toLocaleDateString()
                    : 'Never'
                  }
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                  <div className="flex space-x-2">
                    <select
                      value={company.subscription?.tier || 'free'}
                      onChange={(e) => updateSubscriptionTier(company.id, e.target.value)}
                      disabled={updating === company.id}
                      className="text-sm border border-gray-300 rounded px-2 py-1 focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                    >
                      <option value="free">Free</option>
                      <option value="pro">Pro</option>
                      <option value="enterprise">Enterprise</option>
                    </select>
                    {updating === company.id && (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-emerald-500"></div>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {companies.length === 0 && (
        <div className="text-center py-12">
          <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Companies Found</h3>
          <p className="text-gray-600">Companies will appear here once they register.</p>
        </div>
      )}

      {/* Subscription Tiers Info */}
      <div className="mt-8 grid md:grid-cols-3 gap-6">
        <div className="border border-gray-200 rounded-lg p-4">
          <h3 className="font-semibold text-gray-900 mb-2">Free Tier</h3>
          <ul className="text-sm text-gray-600 space-y-1">
            <li className="flex items-center"><Check className="h-4 w-4 text-green-500 mr-2" />Basic marketplace access</li>
            <li className="flex items-center"><Check className="h-4 w-4 text-green-500 mr-2" />Up to 5 material listings</li>
            <li className="flex items-center"><X className="h-4 w-4 text-red-500 mr-2" />No AI matching</li>
          </ul>
        </div>
        <div className="border border-blue-200 rounded-lg p-4 bg-blue-50">
          <h3 className="font-semibold text-blue-900 mb-2">Pro Tier</h3>
          <ul className="text-sm text-blue-700 space-y-1">
            <li className="flex items-center"><Check className="h-4 w-4 text-green-500 mr-2" />AI-powered matching</li>
            <li className="flex items-center"><Check className="h-4 w-4 text-green-500 mr-2" />Unlimited listings</li>
            <li className="flex items-center"><Check className="h-4 w-4 text-green-500 mr-2" />Advanced analytics</li>
            <li className="flex items-center"><Check className="h-4 w-4 text-green-500 mr-2" />Priority support</li>
          </ul>
        </div>
        <div className="border border-purple-200 rounded-lg p-4 bg-purple-50">
          <h3 className="font-semibold text-purple-900 mb-2">Enterprise Tier</h3>
          <ul className="text-sm text-purple-700 space-y-1">
            <li className="flex items-center"><Check className="h-4 w-4 text-green-500 mr-2" />Custom AI models</li>
            <li className="flex items-center"><Check className="h-4 w-4 text-green-500 mr-2" />API access</li>
            <li className="flex items-center"><Check className="h-4 w-4 text-green-500 mr-2" />White-label options</li>
            <li className="flex items-center"><Check className="h-4 w-4 text-green-500 mr-2" />Dedicated support</li>
          </ul>
        </div>
      </div>
    </div>
  );
}