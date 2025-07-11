import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import AuthenticatedLayout from './AuthenticatedLayout';
import { 
  TrendingUp, 
  DollarSign, 
  Leaf, 
  Users, 
  Target, 
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  Activity,
  Globe,
  Award,
  Zap,
  CheckCircle,
  Clock,
  AlertCircle,
  Brain,
  Truck,
  Factory,
  Recycle
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface DashboardMetrics {
  totalSavings: number;
  carbonReduction: number;
  activePartnerships: number;
  pendingMatches: number;
  monthlyGrowth: number;
  sustainabilityScore: number;
  roiPercentage: number;
  completionRate: number;
}

interface RecentActivity {
  id: string;
  type: 'match' | 'partnership' | 'savings' | 'carbon';
  title: string;
  description: string;
  value: string;
  timestamp: string;
  status: 'completed' | 'pending' | 'in_progress';
}

const DemoDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    totalSavings: 0,
    carbonReduction: 0,
    activePartnerships: 0,
    pendingMatches: 0,
    monthlyGrowth: 0,
    sustainabilityScore: 0,
    roiPercentage: 0,
    completionRate: 0
  });
  const [recentActivity, setRecentActivity] = useState<RecentActivity[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      // Simulate loading real data
      setTimeout(() => {
        setMetrics({
          totalSavings: 2847500,
          carbonReduction: 1250,
          activePartnerships: 12,
          pendingMatches: 8,
          monthlyGrowth: 23.5,
          sustainabilityScore: 87,
          roiPercentage: 156,
          completionRate: 94
        });

        setRecentActivity([
          {
            id: '1',
            type: 'match',
            title: 'New AI Match Found',
            description: 'Steel scrap match with Gulf Manufacturing Co.',
            value: '+$45,000 potential',
            timestamp: '2 hours ago',
            status: 'pending'
          },
          {
            id: '2',
            type: 'partnership',
            title: 'Partnership Established',
            description: 'Chemical byproduct exchange with Arabian Solutions',
            value: '+$32,000 saved',
            timestamp: '1 day ago',
            status: 'completed'
          },
          {
            id: '3',
            type: 'savings',
            title: 'Monthly Savings Achieved',
            description: 'Waste exchange program with local partners',
            value: '+$18,500 saved',
            timestamp: '3 days ago',
            status: 'completed'
          },
          {
            id: '4',
            type: 'carbon',
            title: 'Carbon Reduction Milestone',
            description: 'Reached 1,000 tons CO2 reduction',
            value: '+250 tons CO2',
            timestamp: '1 week ago',
            status: 'completed'
          }
        ]);

        setIsLoading(false);
      }, 1500);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
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

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'match': return <Target className="w-4 h-4" />;
      case 'partnership': return <Users className="w-4 h-4" />;
      case 'savings': return <DollarSign className="w-4 h-4" />;
      case 'carbon': return <Leaf className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'pending': return 'text-yellow-600 bg-yellow-100';
      case 'in_progress': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  if (isLoading) {
    return (
      <AuthenticatedLayout title="Dashboard">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-emerald-500"></div>
        </div>
      </AuthenticatedLayout>
    );
  }

  return (
    <AuthenticatedLayout title="SymbioFlows Dashboard">
      <div className="space-y-6">
        {/* Header */}
        <div className="bg-gradient-to-r from-emerald-600 to-blue-600 rounded-xl p-6 text-white">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2">Welcome to SymbioFlows</h1>
              <p className="text-emerald-100">
                Your industrial symbiosis platform is actively generating value and reducing environmental impact.
              </p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold">{formatCurrency(metrics.totalSavings)}</div>
              <div className="text-emerald-100">Total Savings Generated</div>
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="bg-gradient-to-br from-green-50 to-emerald-50 border-green-200">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-600">Total Savings</p>
                  <p className="text-2xl font-bold text-green-800">{formatCurrency(metrics.totalSavings)}</p>
                  <div className="flex items-center text-sm text-green-600">
                    <ArrowUpRight className="w-4 h-4 mr-1" />
                    +{metrics.monthlyGrowth}% this month
                  </div>
                </div>
                <DollarSign className="w-8 h-8 text-green-600" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-blue-50 to-cyan-50 border-blue-200">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-600">Carbon Reduction</p>
                  <p className="text-2xl font-bold text-blue-800">{formatNumber(metrics.carbonReduction)} tons</p>
                  <div className="flex items-center text-sm text-blue-600">
                    <ArrowUpRight className="w-4 h-4 mr-1" />
                    +15% this month
                  </div>
                </div>
                <Leaf className="w-8 h-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-50 to-indigo-50 border-purple-200">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-purple-600">Active Partnerships</p>
                  <p className="text-2xl font-bold text-purple-800">{metrics.activePartnerships}</p>
                  <div className="flex items-center text-sm text-purple-600">
                    <ArrowUpRight className="w-4 h-4 mr-1" />
                    +3 new this month
                  </div>
                </div>
                <Users className="w-8 h-8 text-purple-600" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-orange-50 to-red-50 border-orange-200">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-orange-600">ROI Percentage</p>
                  <p className="text-2xl font-bold text-orange-800">{metrics.roiPercentage}%</p>
                  <div className="flex items-center text-sm text-orange-600">
                    <ArrowUpRight className="w-4 h-4 mr-1" />
                    +12% vs last month
                  </div>
                </div>
                <TrendingUp className="w-8 h-8 text-orange-600" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Progress and Performance */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Target className="w-5 h-5 text-emerald-600" />
                <span>Performance Metrics</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Sustainability Score</span>
                  <span className="text-sm text-gray-600">{metrics.sustainabilityScore}%</span>
                </div>
                <Progress value={metrics.sustainabilityScore} className="h-2" />
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Completion Rate</span>
                  <span className="text-sm text-gray-600">{metrics.completionRate}%</span>
                </div>
                <Progress value={metrics.completionRate} className="h-2" />
              </div>
              
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium">Partnership Success</span>
                  <span className="text-sm text-gray-600">92%</span>
                </div>
                <Progress value={92} className="h-2" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Activity className="w-5 h-5 text-blue-600" />
                <span>Quick Actions</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button 
                onClick={() => navigate('/ai-inference-matching')}
                className="w-full bg-gradient-to-r from-emerald-500 to-blue-500 hover:from-emerald-600 hover:to-blue-600"
              >
                <Brain className="w-4 h-4 mr-2" />
                Find New Matches
              </Button>
              
              <Button 
                variant="outline" 
                className="w-full"
                onClick={() => navigate('/marketplace')}
              >
                <Factory className="w-4 h-4 mr-2" />
                Browse Marketplace
              </Button>
              
              <Button 
                variant="outline" 
                className="w-full"
              >
                <Truck className="w-4 h-4 mr-2" />
                View Logistics
              </Button>
              
              <Button 
                variant="outline" 
                className="w-full"
              >
                <BarChart3 className="w-4 h-4 mr-2" />
                Generate Report
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Recent Activity */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Clock className="w-5 h-5 text-purple-600" />
              <span>Recent Activity</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentActivity.map((activity) => (
                <div key={activity.id} className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg">
                  <div className="w-10 h-10 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-lg flex items-center justify-center">
                    {getActivityIcon(activity.type)}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold text-gray-900">{activity.title}</h4>
                      <Badge className={getStatusColor(activity.status)}>
                        {activity.status.replace('_', ' ')}
                      </Badge>
                    </div>
                    <p className="text-sm text-gray-600">{activity.description}</p>
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-sm font-medium text-emerald-600">{activity.value}</span>
                      <span className="text-xs text-gray-500">{activity.timestamp}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* AI Insights */}
        <Card className="bg-gradient-to-r from-purple-50 to-blue-50 border-purple-200">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="w-5 h-5 text-purple-600" />
              <span>AI Insights</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center p-4 bg-white rounded-lg shadow-sm">
                <Award className="w-8 h-8 text-emerald-600 mx-auto mb-2" />
                <h4 className="font-semibold text-gray-900">Top Performer</h4>
                <p className="text-sm text-gray-600">You're in the top 10% of companies</p>
              </div>
              
              <div className="text-center p-4 bg-white rounded-lg shadow-sm">
                <Zap className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                <h4 className="font-semibold text-gray-900">8 New Opportunities</h4>
                <p className="text-sm text-gray-600">AI found potential matches</p>
              </div>
              
              <div className="text-center p-4 bg-white rounded-lg shadow-sm">
                <Globe className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                <h4 className="font-semibold text-gray-900">Global Impact</h4>
                <p className="text-sm text-gray-600">Reducing carbon worldwide</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </AuthenticatedLayout>
  );
};

export default DemoDashboard; 