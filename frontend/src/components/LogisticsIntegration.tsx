import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { 
  Truck, 
  Package, 
  DollarSign, 
  MapPin, 
  Calendar,
  CheckCircle,
  AlertTriangle,
  Clock,
  TrendingUp,
  Users,
  Factory,
  Recycle,
  ArrowRight,
  Loader2,
  Eye,
  Star,
  Zap
} from 'lucide-react';
import { supabase } from '../lib/supabase';
import { activityService } from '../lib/activityService';

interface LogisticsDeal {
  deal_id: string;
  material: string;
  quantity: number;
  unit_price: number;
  total_material_cost: number;
  shipping_cost: number;
  customs_cost: number;
  insurance_cost: number;
  handling_cost: number;
  platform_fee: number;
  total_cost_to_buyer: number;
  net_revenue_to_seller: number;
  status: string;
  expires_at: string;
  shipment_date?: string;
  delivery_date?: string;
}

interface LogisticsDashboard {
  total_deals: number;
  active_deals: number;
  total_spent: number;
  pending_deals: LogisticsDeal[];
}

export function LogisticsIntegration() {
  const [loading, setLoading] = useState(true);
  const [buyerDashboard, setBuyerDashboard] = useState<LogisticsDashboard | null>(null);
  const [sellerDashboard, setSellerDashboard] = useState<LogisticsDashboard | null>(null);
  const [currentUser, setCurrentUser] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<'buyer' | 'seller'>('buyer');

  useEffect(() => {
    loadLogisticsData();
  }, []);

  const loadLogisticsData = async () => {
    try {
      setLoading(true);
      
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      setCurrentUser(user);

      // Fetch real company data
      const { data: company } = await supabase
        .from('companies')
        .select('*')
        .eq('id', user.id)
        .single();

      // Fetch real material listings from AI onboarding
      const { data: materialListings } = await supabase
        .from('material_listings')
        .select('*')
        .eq('company_id', user.id)
        .order('created_at', { ascending: false });

      // Fetch real matches
      const { data: matches } = await supabase
        .from('matches')
        .select('*')
        .or(`company_id.eq.${user.id},partner_company_id.eq.${user.id}`)
        .order('created_at', { ascending: false });

      // Fetch real activities for deal tracking
      const activities = await activityService.getUserActivities(user.id, 50);

      // Determine if user is primarily a buyer or seller based on their materials
      const buyerMaterials = materialListings?.filter(m => m.role === 'buyer' || m.type === 'waste') || [];
      const sellerMaterials = materialListings?.filter(m => m.role === 'seller' || m.type === 'resource' || m.type === 'product') || [];
      
      const isPrimarilyBuyer = buyerMaterials.length > sellerMaterials.length;

      // Create real buyer dashboard data
      const buyerData: LogisticsDashboard = {
        total_deals: matches?.length || 0,
        active_deals: matches?.filter(m => m.status === 'pending' || m.status === 'accepted').length || 0,
        total_spent: matches?.reduce((sum, m) => sum + (m.potential_savings || 0), 0) || 0,
        pending_deals: matches?.filter(m => m.status === 'pending' || m.status === 'accepted').map(match => ({
          deal_id: match.id,
          material: match.partner_material_name || 'Unknown Material',
          quantity: match.quantity || 0,
          unit_price: match.unit_price || 0,
          total_material_cost: (match.quantity || 0) * (match.unit_price || 0),
          shipping_cost: match.shipping_cost || 0,
          customs_cost: match.customs_cost || 0,
          insurance_cost: match.insurance_cost || 0,
          handling_cost: match.handling_cost || 0,
          platform_fee: match.platform_fee || 0,
          total_cost_to_buyer: match.total_cost || 0,
          net_revenue_to_seller: match.net_revenue || 0,
          status: match.status || 'pending',
          expires_at: match.expires_at || new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
          shipment_date: match.shipment_date,
          delivery_date: match.delivery_date
        })) || []
      };

      // Create real seller dashboard data
      const sellerData: LogisticsDashboard = {
        total_deals: matches?.length || 0,
        active_deals: matches?.filter(m => m.status === 'pending' || m.status === 'accepted').length || 0,
        total_spent: matches?.reduce((sum, m) => sum + (m.net_revenue || 0), 0) || 0,
        pending_deals: matches?.filter(m => m.status === 'pending' || m.status === 'accepted').map(match => ({
          deal_id: match.id,
          material: match.material_name || 'Unknown Material',
          quantity: match.quantity || 0,
          unit_price: match.unit_price || 0,
          total_material_cost: (match.quantity || 0) * (match.unit_price || 0),
          shipping_cost: 0, // Seller doesn't pay shipping
          customs_cost: 0, // Seller doesn't pay customs
          insurance_cost: 0, // Seller doesn't pay insurance
          handling_cost: 0, // Seller doesn't pay handling
          platform_fee: match.platform_fee || 0,
          total_cost_to_buyer: match.total_cost || 0,
          net_revenue_to_seller: match.net_revenue || 0,
          status: match.status || 'pending',
          expires_at: match.expires_at || new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
          shipment_date: match.shipment_date,
          delivery_date: match.delivery_date
        })) || []
      };

      if (isPrimarilyBuyer || !company?.role) {
        setBuyerDashboard(buyerData);
        setActiveTab('buyer');
      } else {
        setSellerDashboard(sellerData);
        setActiveTab('seller');
      }

    } catch (error) {
      console.error('Error loading logistics data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchBuyerDashboard = async (userId: string): Promise<LogisticsDashboard> => {
    // This function is no longer needed as we load real data in loadLogisticsData
    return {
      total_deals: 0,
      active_deals: 0,
      total_spent: 0,
      pending_deals: []
    };
  };

  const fetchSellerDashboard = async (userId: string): Promise<LogisticsDashboard> => {
    // This function is no longer needed as we load real data in loadLogisticsData
    return {
      total_deals: 0,
      active_deals: 0,
      total_spent: 0,
      pending_deals: []
    };
  };

  const getAuthToken = async (): Promise<string> => {
    const { data: { session } } = await supabase.auth.getSession();
    return session?.access_token || '';
  };

  const handleAcceptDeal = async (dealId: string) => {
    try {
      const response = await fetch(`http://localhost:5026/deals/${dealId}/accept`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${await getAuthToken()}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          company_id: currentUser.id
        })
      });

      if (response.ok) {
        // Reload data
        await loadLogisticsData();
      }
    } catch (error) {
      console.error('Error accepting deal:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-yellow-900/20 text-yellow-400 border-yellow-700';
      case 'buyer_accepted': return 'bg-blue-900/20 text-blue-400 border-blue-700';
      case 'seller_accepted': return 'bg-blue-900/20 text-blue-400 border-blue-700';
      case 'accepted': return 'bg-green-900/20 text-green-400 border-green-700';
      case 'logistics_booked': return 'bg-purple-900/20 text-purple-400 border-purple-700';
      case 'completed': return 'bg-emerald-900/20 text-emerald-400 border-emerald-700';
      default: return 'bg-gray-900/20 text-gray-400 border-gray-700';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'pending': return 'Pending';
      case 'buyer_accepted': return 'Buyer Accepted';
      case 'seller_accepted': return 'Seller Accepted';
      case 'accepted': return 'Both Accepted';
      case 'logistics_booked': return 'Logistics Booked';
      case 'completed': return 'Completed';
      default: return 'Unknown';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-emerald-500" />
      </div>
    );
  }

  const dashboard = activeTab === 'buyer' ? buyerDashboard : sellerDashboard;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Truck className="h-6 w-6 text-emerald-400" />
            Logistics Dashboard
          </h2>
          <p className="text-gray-400 mt-1">
            {activeTab === 'buyer' ? 'Manage your purchases and shipping' : 'Track your sales and revenue'}
          </p>
        </div>
        
        <div className="flex gap-2">
          <Button
            variant={activeTab === 'buyer' ? 'default' : 'outline'}
            onClick={() => setActiveTab('buyer')}
            className="flex items-center gap-2"
          >
            <Package className="h-4 w-4" />
            Buyer View
          </Button>
          <Button
            variant={activeTab === 'seller' ? 'default' : 'outline'}
            onClick={() => setActiveTab('seller')}
            className="flex items-center gap-2"
          >
            <Factory className="h-4 w-4" />
            Seller View
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-800 border border-slate-700 rounded-xl shadow p-4 text-gray-300">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="text-sm font-medium text-gray-400">Total Deals</h3>
            <Package className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="p-0">
            <div className="text-2xl font-bold text-white">{dashboard?.total_deals || 0}</div>
            <p className="text-xs text-gray-400 mt-1">
              {dashboard?.active_deals || 0} currently active
            </p>
          </div>
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-xl shadow p-4 text-gray-300">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="text-sm font-medium text-gray-400">
              {activeTab === 'buyer' ? 'Total Spent' : 'Total Revenue'}
            </h3>
            <DollarSign className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="p-0">
            <div className="text-2xl font-bold text-white">
              ${dashboard?.total_spent?.toLocaleString() || '0'}
            </div>
            <p className="text-xs text-gray-400 mt-1">
              {activeTab === 'buyer' ? 'All time purchases' : 'All time earnings'}
            </p>
          </div>
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-xl shadow p-4 text-gray-300">
          <div className="flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="text-sm font-medium text-gray-400">Pending Deals</h3>
            <Clock className="h-4 w-4 text-emerald-400" />
          </div>
          <div className="p-0">
            <div className="text-2xl font-bold text-white">{dashboard?.pending_deals?.length || 0}</div>
            <p className="text-xs text-gray-400 mt-1">
              Awaiting action
            </p>
          </div>
        </div>
      </div>

      {/* Pending Deals */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl shadow text-gray-300">
        <div className="mb-2 font-semibold text-lg p-6 pb-2">
          <h3 className="text-xl font-bold mb-1 text-white flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-500" />
            Pending Deals
          </h3>
        </div>
        <div className="text-gray-300 p-6 pt-2">
          {dashboard?.pending_deals?.length === 0 ? (
            <div className="text-center py-8">
              <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
              <p className="text-gray-400">No pending deals</p>
            </div>
          ) : (
            <div className="space-y-4">
              {dashboard?.pending_deals?.map((deal) => (
                <div key={deal.deal_id} className="bg-slate-700 rounded-lg p-4 border border-slate-600">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className="bg-emerald-500/20 p-2 rounded-lg">
                        <Recycle className="h-5 w-5 text-emerald-400" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white">{deal.material}</h3>
                        <p className="text-sm text-gray-400">
                          {deal.quantity.toLocaleString()} units
                        </p>
                      </div>
                    </div>
                    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${getStatusColor(deal.status)}`}>
                      {getStatusText(deal.status)}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div>
                      <p className="text-xs text-gray-400">Unit Price</p>
                      <p className="text-white font-medium">${deal.unit_price}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-400">Material Cost</p>
                      <p className="text-white font-medium">${deal.total_material_cost.toLocaleString()}</p>
                    </div>
                    {activeTab === 'buyer' && (
                      <>
                        <div>
                          <p className="text-xs text-gray-400">Shipping</p>
                          <p className="text-white font-medium">${deal.shipping_cost.toLocaleString()}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Total Cost</p>
                          <p className="text-white font-medium">${deal.total_cost_to_buyer.toLocaleString()}</p>
                        </div>
                      </>
                    )}
                    {activeTab === 'seller' && (
                      <div>
                        <p className="text-xs text-gray-400">Net Revenue</p>
                        <p className="text-white font-medium">${deal.net_revenue_to_seller.toLocaleString()}</p>
                      </div>
                    )}
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-sm text-gray-400">
                      <Calendar className="h-4 w-4" />
                      Expires: {new Date(deal.expires_at).toLocaleDateString()}
                    </div>
                    
                    {deal.status === 'pending' && (
                      <Button
                        onClick={() => handleAcceptDeal(deal.deal_id)}
                        className="bg-emerald-500 hover:bg-emerald-600"
                      >
                        Accept Deal
                      </Button>
                    )}
                    
                    {deal.status === 'accepted' && (
                      <Button variant="outline" className="border-emerald-500 text-emerald-400">
                        <Eye className="h-4 w-4 mr-2" />
                        View Details
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      
    </div>
  );
} 