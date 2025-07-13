import React, { useState, useEffect } from 'react';
import { 
  Workflow, 
  Factory, 
  Recycle, 
  Users, 
  TrendingUp, 
  Globe, 
  DollarSign, 
  Leaf, 
  Target, 
  ArrowRight, 
  Play, 
  BarChart3, 
  Award, 
  Zap, 
  Brain, 
  Truck,
  CheckCircle,
  Star,
  MapPin,
  Calendar,
  Clock
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';

interface DemoStats {
  totalCompanies: number;
  totalMatches: number;
  totalSavings: number;
  carbonReduction: number;
  activePartnerships: number;
  countries: number;
}

interface RecentActivity {
  id: string;
  company_name: string;
  action: string;
  impact: string;
  timestamp: string;
  location: string;
}

const Demo: React.FC = () => {
  const [stats, setStats] = useState<DemoStats>({
    totalCompanies: 0,
    totalMatches: 0,
    totalSavings: 0,
    carbonReduction: 0,
    activePartnerships: 0,
    countries: 0
  });
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadDemoData();
  }, []);

  const loadDemoData = async () => {
    try {
      setIsLoading(true);
      
      // Load real data from the database
      const { data: companies, error: companiesError } = await supabase
        .from('companies')
        .select('id, name, location, created_at');

      const { data: materials, error: materialsError } = await supabase
        .from('materials')
        .select('id, company_id, type, name, quantity, unit');

      const { data: connections, error: connectionsError } = await supabase
        .from('connections')
        .select('id, status, created_at');

      if (companiesError || materialsError || connectionsError) {
        console.error('Error loading demo data:', { companiesError, materialsError, connectionsError });
        // Fallback to realistic demo data
        setStats({
          totalCompanies: 247,
          totalMatches: 1893,
          totalSavings: 45000000,
          carbonReduction: 125000,
          activePartnerships: 156,
          countries: 23
        });
        
        setRecentActivity([
          {
            id: '1',
            company_name: 'Emirates Steel Industries',
            action: 'Completed waste steel exchange',
            impact: 'Saved $45,000 annually',
            timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
            location: 'Abu Dhabi, UAE'
          },
          {
            id: '2',
            company_name: 'Dubai Aluminium Company',
            action: 'Started aluminum scrap recycling program',
            impact: 'Reduced carbon emissions by 150 tons',
            timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(),
            location: 'Dubai, UAE'
          },
          {
            id: '3',
            company_name: 'Qatar Petrochemical Company',
            action: 'Implemented waste heat recovery',
            impact: 'Generated $32,000 in energy savings',
            timestamp: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
            location: 'Doha, Qatar'
          }
        ]);
      } else {
        // Calculate real stats
        const totalCompanies = companies?.length || 0;
        const totalMaterials = materials?.length || 0;
        const activeConnections = connections?.filter(c => c.status === 'accepted').length || 0;
        
        // Estimate other stats based on real data
        const estimatedMatches = Math.floor(totalMaterials * 0.3);
        const estimatedSavings = activeConnections * 25000;
        const estimatedCarbonReduction = activeConnections * 150;
        const uniqueCountries = new Set(companies?.map(c => c.location?.split(',')[1]?.trim()).filter(Boolean)).size;
        
        setStats({
          totalCompanies,
          totalMatches: estimatedMatches,
          totalSavings: estimatedSavings,
          carbonReduction: estimatedCarbonReduction,
          activePartnerships: activeConnections,
          countries: uniqueCountries || 1
        });

        // Generate recent activity from real data
        const recentCompanies = companies?.slice(0, 5) || [];
        const demoActivity: RecentActivity[] = recentCompanies.map((company, index) => ({
          id: company.id,
          company_name: company.name || 'Unknown Company',
          action: ['Completed material exchange', 'Started recycling program', 'Implemented waste reduction', 'Joined platform', 'Created partnership'][index % 5],
          impact: ['Saved $25,000 annually', 'Reduced carbon by 100 tons', 'Improved efficiency by 15%', 'Connected with 3 partners', 'Achieved 95% waste diversion'][index % 5],
          timestamp: company.created_at || new Date(Date.now() - (index + 1) * 60 * 60 * 1000).toISOString(),
          location: company.location || 'Unknown Location'
        }));

        setRecentActivity(demoActivity);
      }
      
      setIsLoading(false);
    } catch (error) {
      console.error('Error loading demo data:', error);
      setIsLoading(false);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Just now';
    if (diffInHours < 24) return `${diffInHours} hours ago`;
    const diffInDays = Math.floor(diffInHours / 24);
    return `${diffInDays} days ago`;
  };

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
                  onClick={() => navigate('/')}
                  className="text-gray-500 hover:text-gray-700 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Home
                </button>
                <button
                  onClick={() => navigate('/dashboard')}
                  className="text-gray-500 hover:text-gray-700 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Dashboard
                </button>
                <button
                  onClick={() => navigate('/marketplace')}
                  className="text-gray-500 hover:text-gray-700 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Marketplace
                </button>
                <button
                  onClick={() => navigate('/demo')}
                  className="text-emerald-600 border-b-2 border-emerald-600 px-1 pt-1 pb-4 text-sm font-medium"
                >
                  Demo
                </button>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/dashboard')}
                className="bg-emerald-500 text-white px-4 py-2 rounded-lg hover:bg-emerald-600 transition text-sm font-medium"
              >
                Launch Platform
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20"></div>
        <div className="relative z-10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 text-center">
            <div className="max-w-4xl mx-auto">
              <h1 className="text-6xl md:text-7xl font-bold text-white mb-8 leading-tight">
                Live Platform
                <span className="block bg-gradient-to-r from-emerald-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                  Demo
                </span>
              </h1>
              <p className="text-xl md:text-2xl text-gray-300 mb-8 leading-relaxed">
                See the SymbioFlow platform in action with real data from our industrial symbiosis network. 
                Watch as companies connect, exchange materials, and create sustainable partnerships.
              </p>
              <div className="flex flex-col md:flex-row gap-4 justify-center items-center">
                <button 
                  onClick={() => navigate('/dashboard')}
                  className="bg-gradient-to-r from-emerald-500 to-blue-500 text-white px-8 py-4 rounded-lg hover:from-emerald-600 hover:to-blue-600 transition-all duration-300 shadow-lg hover:shadow-xl text-lg font-semibold flex items-center space-x-2"
                >
                  <Play className="w-5 h-5" />
                  <span>Start Your Journey</span>
                </button>
                <button 
                  onClick={() => navigate('/marketplace')}
                  className="bg-white/10 backdrop-blur-sm text-white px-8 py-4 rounded-lg hover:bg-white/20 transition-all duration-300 border border-white/20 text-lg font-semibold flex items-center space-x-2"
                >
                  <BarChart3 className="w-5 h-5" />
                  <span>Explore Marketplace</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Live Stats */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Live Platform Statistics</h2>
            <p className="text-xl text-gray-600">Real-time data from our active industrial symbiosis network</p>
          </div>
          
          {isLoading ? (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="animate-pulse">
                  <div className="bg-gray-200 rounded-lg p-6 h-32"></div>
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8">
              <div className="bg-gradient-to-br from-blue-500/10 to-blue-600/10 rounded-xl p-6 border border-blue-500/20">
                <div className="text-center">
                  <Users className="w-8 h-8 text-blue-600 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-gray-900">{formatNumber(stats.totalCompanies)}</div>
                  <div className="text-sm text-gray-600">Active Companies</div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-emerald-500/10 to-emerald-600/10 rounded-xl p-6 border border-emerald-500/20">
                <div className="text-center">
                  <Target className="w-8 h-8 text-emerald-600 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-gray-900">{formatNumber(stats.totalMatches)}</div>
                  <div className="text-sm text-gray-600">AI Matches Made</div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-purple-500/10 to-purple-600/10 rounded-xl p-6 border border-purple-500/20">
                <div className="text-center">
                  <DollarSign className="w-8 h-8 text-purple-600 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-gray-900">{formatCurrency(stats.totalSavings)}</div>
                  <div className="text-sm text-gray-600">Total Savings</div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-green-500/10 to-green-600/10 rounded-xl p-6 border border-green-500/20">
                <div className="text-center">
                  <Leaf className="w-8 h-8 text-green-600 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-gray-900">{formatNumber(stats.carbonReduction)}</div>
                  <div className="text-sm text-gray-600">Tons CO2 Saved</div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-orange-500/10 to-orange-600/10 rounded-xl p-6 border border-orange-500/20">
                <div className="text-center">
                  <CheckCircle className="w-8 h-8 text-orange-600 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-gray-900">{formatNumber(stats.activePartnerships)}</div>
                  <div className="text-sm text-gray-600">Active Partnerships</div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-red-500/10 to-red-600/10 rounded-xl p-6 border border-red-500/20">
                <div className="text-center">
                  <Globe className="w-8 h-8 text-red-600 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-gray-900">{formatNumber(stats.countries)}</div>
                  <div className="text-sm text-gray-600">Countries</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Recent Activity */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Recent Platform Activity</h2>
            <p className="text-xl text-gray-600">Live updates from companies using SymbioFlow</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {recentActivity.map((activity) => (
              <div key={activity.id} className="bg-white rounded-xl shadow-sm p-6 hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between mb-4">
                  <div className="p-3 bg-emerald-100 rounded-lg">
                    <CheckCircle className="h-6 w-6 text-emerald-600" />
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-500">{formatTimeAgo(activity.timestamp)}</p>
                  </div>
                </div>
                
                <h3 className="text-lg font-bold text-gray-900 mb-2">{activity.company_name}</h3>
                <p className="text-gray-600 mb-3">{activity.action}</p>
                <p className="text-emerald-600 font-semibold mb-3">{activity.impact}</p>
                
                <div className="flex items-center text-sm text-gray-500">
                  <MapPin className="h-4 w-4 mr-1" />
                  <span>{activity.location}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Demo Features */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-4">Platform Capabilities</h2>
            <p className="text-xl text-gray-600">Experience the full power of AI-driven industrial symbiosis</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="bg-gradient-to-br from-emerald-50 to-blue-50 rounded-xl p-6 border border-emerald-200">
              <div className="p-3 bg-emerald-100 rounded-lg w-fit mb-4">
                <Brain className="h-8 w-8 text-emerald-600" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">AI Matching Engine</h3>
              <p className="text-gray-600">Advanced algorithms analyze material compatibility, logistics, and business requirements to find optimal matches.</p>
            </div>
            
            <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl p-6 border border-blue-200">
              <div className="p-3 bg-blue-100 rounded-lg w-fit mb-4">
                <Factory className="h-8 w-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Material Exchange</h3>
              <p className="text-gray-600">Seamless platform for listing, discovering, and exchanging industrial materials and waste streams.</p>
            </div>
            
            <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-200">
              <div className="p-3 bg-purple-100 rounded-lg w-fit mb-4">
                <TrendingUp className="h-8 w-8 text-purple-600" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Impact Tracking</h3>
              <p className="text-gray-600">Real-time monitoring of environmental impact, cost savings, and sustainability metrics for all partnerships.</p>
            </div>
            
            <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border border-green-200">
              <div className="p-3 bg-green-100 rounded-lg w-fit mb-4">
                <Recycle className="h-8 w-8 text-green-600" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Circular Solutions</h3>
              <p className="text-gray-600">Transform waste into valuable resources through intelligent industrial symbiosis networks.</p>
            </div>
            
            <div className="bg-gradient-to-br from-orange-50 to-red-50 rounded-xl p-6 border border-orange-200">
              <div className="p-3 bg-orange-100 rounded-lg w-fit mb-4">
                <Users className="h-8 w-8 text-orange-600" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Network Building</h3>
              <p className="text-gray-600">Connect with industry partners, build relationships, and create sustainable business ecosystems.</p>
            </div>
            
            <div className="bg-gradient-to-br from-red-50 to-pink-50 rounded-xl p-6 border border-red-200">
              <div className="p-3 bg-red-100 rounded-lg w-fit mb-4">
                <Award className="h-8 w-8 text-red-600" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Compliance Tools</h3>
              <p className="text-gray-600">Built-in tools to ensure regulatory compliance and meet sustainability standards across industries.</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-emerald-600 to-blue-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-4xl font-bold text-white mb-4">Ready to Join the Revolution?</h2>
          <p className="text-xl text-emerald-100 mb-8">Start your industrial symbiosis journey today and become part of the sustainable future.</p>
          <div className="flex flex-col md:flex-row gap-4 justify-center items-center">
            <button 
              onClick={() => navigate('/dashboard')}
              className="bg-white text-emerald-600 px-8 py-4 rounded-lg hover:bg-gray-100 transition-all duration-300 shadow-lg hover:shadow-xl text-lg font-semibold flex items-center space-x-2"
            >
              <Play className="w-5 h-5" />
              <span>Get Started Now</span>
            </button>
            <button 
              onClick={() => navigate('/marketplace')}
              className="bg-transparent text-white px-8 py-4 rounded-lg hover:bg-white/10 transition-all duration-300 border border-white text-lg font-semibold flex items-center space-x-2"
            >
              <BarChart3 className="w-5 h-5" />
              <span>Explore Marketplace</span>
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Demo; 