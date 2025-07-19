import React, { useState, useEffect } from 'react';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle 
} from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
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

      // Fetch user's company role to determine buyer/seller view
      const { data: company } = await supabase
        .from('companies')
        .select('role')
        .eq('id', user.id)
        .single();

      if (company?.role === 'buyer' || !company?.role) {
        // Load buyer dashboard
        const buyerData = await fetchBuyerDashboard(user.id);
        setBuyerDashboard(buyerData);
        setActiveTab('buyer');
      } else {
        // Load seller dashboard
        const sellerData = await fetchSellerDashboard(user.id);
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
    try {
      const response = await fetch(`http://localhost:5026/dashboard/buyer/${userId}`, {
        headers: {
          'Authorization': `Bearer ${await getAuthToken()}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        return await response.json();
      } else {
        // Fallback to mock data
        return generateMockBuyerDashboard();
      }
    } catch (error) {
      console.error('Error fetching buyer dashboard:', error);
      return generateMockBuyerDashboard();
    }
  };

  const fetchSellerDashboard = async (userId: string): Promise<LogisticsDashboard> => {
    try {
      const response = await fetch(`http://localhost:5026/dashboard/seller/${userId}`, {
        headers: {
          'Authorization': `Bearer ${await getAuthToken()}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        return await response.json();
      } else {
        // Fallback to mock data
        return generateMockSellerDashboard();
      }
    } catch (error) {
      console.error('Error fetching seller dashboard:', error);
      return generateMockSellerDashboard();
    }
  };

  const getAuthToken = async (): Promise<string> => {
    const { data: { session } } = await supabase.auth.getSession();
    return session?.access_token || '';
  };

  const generateMockBuyerDashboard = (): LogisticsDashboard => ({
    total_deals: 12,
    active_deals: 3,
    total_spent: 45000,
    pending_deals: [
      {
        deal_id: 'deal_001',
        material: 'Recycled Aluminum',
        quantity: 5000,
        unit_price: 2.50,
        total_material_cost: 12500,
        shipping_cost: 1200,
        customs_cost: 625,
        insurance_cost: 250,
        handling_cost: 300,
        platform_fee: 625,
        total_cost_to_buyer: 15400,
        net_revenue_to_seller: 11875,
        status: 'pending',
        expires_at: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
      },
      {
        deal_id: 'deal_002',
        material: 'Recycled Plastic',
        quantity: 2000,
        unit_price: 1.80,
        total_material_cost: 3600,
        shipping_cost: 800,
        customs_cost: 180,
        insurance_cost: 72,
        handling_cost: 150,
        platform_fee: 180,
        total_cost_to_buyer: 4802,
        net_revenue_to_seller: 3420,
        status: 'buyer_accepted',
        expires_at: new Date(Date.now() + 12 * 60 * 60 * 1000).toISOString()
      }
    ]
  });

  const generateMockSellerDashboard = (): LogisticsDashboard => ({
    total_deals: 8,
    active_deals: 2,
    total_spent: 28000,
    pending_deals: [
      {
        deal_id: 'deal_003',
        material: 'Recycled Steel',
        quantity: 3000,
        unit_price: 3.20,
        total_material_cost: 9600,
        shipping_cost: 0,
        customs_cost: 0,
        insurance_cost: 0,
        handling_cost: 0,
        platform_fee: 480,
        total_cost_to_buyer: 10080,
        net_revenue_to_seller: 9120,
        status: 'pending',
        expires_at: new Date(Date.now() + 18 * 60 * 60 * 1000).toISOString()
      }
    ]
  });

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
      case 'pending': return 'bg-yellow-500';
      case 'buyer_accepted': return 'bg-blue-500';
      case 'seller_accepted': return 'bg-blue-500';
      case 'accepted': return 'bg-green-500';
      case 'logistics_booked': return 'bg-purple-500';
      case 'completed': return 'bg-emerald-500';
      default: return 'bg-gray-500';
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
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">Total Deals</CardTitle>
            <Package className="h-4 w-4 text-emerald-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{dashboard?.total_deals || 0}</div>
            <p className="text-xs text-gray-400 mt-1">
              {dashboard?.active_deals || 0} currently active
            </p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">
              {activeTab === 'buyer' ? 'Total Spent' : 'Total Revenue'}
            </CardTitle>
            <DollarSign className="h-4 w-4 text-emerald-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              ${dashboard?.total_spent?.toLocaleString() || '0'}
            </div>
            <p className="text-xs text-gray-400 mt-1">
              {activeTab === 'buyer' ? 'All time purchases' : 'All time earnings'}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-gray-400">Pending Deals</CardTitle>
            <Clock className="h-4 w-4 text-emerald-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{dashboard?.pending_deals?.length || 0}</div>
            <p className="text-xs text-gray-400 mt-1">
              Awaiting action
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Pending Deals */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-500" />
            Pending Deals
          </CardTitle>
        </CardHeader>
        <CardContent>
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
                    <Badge className={getStatusColor(deal.status)}>
                      {getStatusText(deal.status)}
                    </Badge>
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
        </CardContent>
      </Card>

      {/* Logistics Features */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Zap className="h-5 w-5 text-emerald-400" />
            Logistics Features
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="bg-slate-700 rounded-lg p-4 border border-slate-600">
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-blue-500/20 p-2 rounded-lg">
                  <Truck className="h-5 w-5 text-blue-400" />
                </div>
                <h4 className="font-semibold text-white">Freightos Integration</h4>
              </div>
              <p className="text-sm text-gray-400">
                Real-time shipping quotes and booking through Freightos
              </p>
            </div>

            <div className="bg-slate-700 rounded-lg p-4 border border-slate-600">
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-green-500/20 p-2 rounded-lg">
                  <DollarSign className="h-5 w-5 text-green-400" />
                </div>
                <h4 className="font-semibold text-white">Complete Cost Breakdown</h4>
              </div>
              <p className="text-sm text-gray-400">
                Material cost, shipping, customs, insurance, and handling
              </p>
            </div>

            <div className="bg-slate-700 rounded-lg p-4 border border-slate-600">
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-purple-500/20 p-2 rounded-lg">
                  <Users className="h-5 w-5 text-purple-400" />
                </div>
                <h4 className="font-semibold text-white">No Direct Communication</h4>
              </div>
              <p className="text-sm text-gray-400">
                You handle everything - buyers and sellers never communicate directly
              </p>
            </div>

            <div className="bg-slate-700 rounded-lg p-4 border border-slate-600">
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-yellow-500/20 p-2 rounded-lg">
                  <Star className="h-5 w-5 text-yellow-400" />
                </div>
                <h4 className="font-semibold text-white">Platform Fee Management</h4>
              </div>
              <p className="text-sm text-gray-400">
                Automated 5% platform fee calculation and collection
              </p>
            </div>

            <div className="bg-slate-700 rounded-lg p-4 border border-slate-600">
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-red-500/20 p-2 rounded-lg">
                  <MapPin className="h-5 w-5 text-red-400" />
                </div>
                <h4 className="font-semibold text-white">Shipment Tracking</h4>
              </div>
              <p className="text-sm text-gray-400">
                Real-time tracking from pickup to delivery
              </p>
            </div>

            <div className="bg-slate-700 rounded-lg p-4 border border-slate-600">
              <div className="flex items-center gap-3 mb-3">
                <div className="bg-emerald-500/20 p-2 rounded-lg">
                  <CheckCircle className="h-5 w-5 text-emerald-400" />
                </div>
                <h4 className="font-semibold text-white">Payment Processing</h4>
              </div>
              <p className="text-sm text-gray-400">
                Secure payment handling and escrow services
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 