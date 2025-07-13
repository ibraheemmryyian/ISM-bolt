import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  TrendingUp, 
  Users, 
  DollarSign, 
  Leaf, 
  Target, 
  Lightbulb,
  ArrowRight,
  CheckCircle,
  Clock,
  Star,
  Home,
  Workflow
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface CompanyData {
  id: string;
  name: string;
  industry: string;
  location: string;
  employee_count: number;
  annual_revenue?: number;
  sustainability_score?: number;
  matches_count?: number;
  savings_achieved?: number;
  carbon_reduced?: number;
}

interface AIInsight {
  id: string;
  type: 'match' | 'opportunity' | 'suggestion' | 'savings';
  title: string;
  description: string;
  impact: 'high' | 'medium' | 'low';
  estimated_savings?: number;
  carbon_reduction?: number;
  action_required: boolean;
  priority: 'urgent' | 'high' | 'medium' | 'low';
}

interface PortfolioData {
  company_overview: {
    size_category: string;
    industry_position: string;
    sustainability_rating: string;
    growth_potential: string;
  };
  achievements: {
    total_savings: number;
    carbon_reduced: number;
    partnerships_formed: number;
    waste_diverted: number;
  };
  recommendations: Array<{
    category: string;
    suggestions: string[];
    priority: string;
  }>;
  recent_activity: Array<{
    date: string;
    action: string;
    impact: string;
  }>;
}

const Dashboard: React.FC = () => {
  const [companyData, setCompanyData] = useState<CompanyData | null>(null);
  const [aiInsights, setAiInsights] = useState<AIInsight[]>([]);
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // Load company data
      const companyResponse = await fetch('/api/companies/current');
      const company = await companyResponse.json();
      setCompanyData(company);

      // Load AI insights
      const insightsResponse = await fetch('/api/ai-insights');
      const insights = await insightsResponse.json();
      setAiInsights(insights.insights || []);

      // Load portfolio data
      const portfolioResponse = await fetch('/api/portfolio');
      const portfolio = await portfolioResponse.json();
      setPortfolioData(portfolio.portfolio);

    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent': return 'text-red-600 bg-red-50';
      case 'high': return 'text-orange-600 bg-orange-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'low': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getImpactIcon = (type: string) => {
    switch (type) {
      case 'match': return <Users className="w-5 h-5" />;
      case 'opportunity': return <Target className="w-5 h-5" />;
      case 'suggestion': return <Lightbulb className="w-5 h-5" />;
      case 'savings': return <DollarSign className="w-5 h-5" />;
      default: return <Star className="w-5 h-5" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header with Home Button */}
      <div className="flex items-center justify-between mb-6">
        <button 
          onClick={() => navigate('/')}
          className="flex items-center space-x-2 text-blue-600 hover:text-blue-800 transition"
        >
          <Workflow className="h-8 w-8" />
          <span className="text-xl font-bold">SymbioFlows</span>
        </button>
        <div className="flex space-x-4">
          <Button 
            variant="outline" 
            onClick={() => navigate('/green-initiatives')}
            className="flex items-center space-x-2"
          >
            <Leaf className="w-4 h-4" />
            <span>Green Initiatives</span>
          </Button>
          <Button 
            variant="outline" 
            onClick={() => navigate('/portfolio')}
            className="flex items-center space-x-2"
          >
            <Star className="w-4 h-4" />
            <span>My Portfolio</span>
          </Button>
        </div>
      </div>

      {/* Welcome Header */}
      <div className="bg-gradient-to-r from-blue-600 to-green-600 rounded-lg p-6 text-white">
        <h1 className="text-3xl font-bold mb-2">
          Welcome back, {companyData?.name}!
        </h1>
        <p className="text-blue-100">
          Here's what's happening with your industrial symbiosis journey
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <DollarSign className="w-8 h-8 text-green-600" />
              <div>
                <p className="text-sm text-gray-600">Total Savings</p>
                <p className="text-2xl font-bold">${portfolioData?.achievements.total_savings?.toLocaleString() || '0'}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Leaf className="w-8 h-8 text-green-600" />
              <div>
                <p className="text-sm text-gray-600">Carbon Reduced</p>
                <p className="text-2xl font-bold">{portfolioData?.achievements.carbon_reduced || 0} tons</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Users className="w-8 h-8 text-blue-600" />
              <div>
                <p className="text-sm text-gray-600">Partnerships</p>
                <p className="text-2xl font-bold">{portfolioData?.achievements.partnerships_formed || 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-8 h-8 text-purple-600" />
              <div>
                <p className="text-sm text-gray-600">Waste Diverted</p>
                <p className="text-2xl font-bold">{portfolioData?.achievements.waste_diverted || 0} tons</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* AI Insights & Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Priority Actions */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="w-5 h-5" />
              <span>Priority Actions</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {aiInsights
              .filter(insight => insight.action_required)
              .sort((a, b) => {
                const priorityOrder = { urgent: 0, high: 1, medium: 2, low: 3 };
                return priorityOrder[a.priority] - priorityOrder[b.priority];
              })
              .slice(0, 5)
              .map((insight) => (
                <div key={insight.id} className="border rounded-lg p-4 hover:bg-gray-50">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center space-x-3">
                      {getImpactIcon(insight.type)}
                      <div>
                        <h4 className="font-semibold">{insight.title}</h4>
                        <p className="text-sm text-gray-600">{insight.description}</p>
                        {insight.estimated_savings && (
                          <p className="text-sm text-green-600 font-medium">
                            Potential savings: ${insight.estimated_savings.toLocaleString()}
                          </p>
                        )}
                      </div>
                    </div>
                    <Badge className={getPriorityColor(insight.priority)}>
                      {insight.priority}
                    </Badge>
                  </div>
                  <Button className="mt-3 w-full" size="sm">
                    Take Action <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </div>
              ))}
          </CardContent>
        </Card>

        {/* Company Portfolio Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Star className="w-5 h-5" />
              <span>Your Company Profile</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {portfolioData?.company_overview && (
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Company Size:</span>
                  <Badge variant="outline">{portfolioData.company_overview.size_category}</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Industry Position:</span>
                  <Badge variant="outline">{portfolioData.company_overview.industry_position}</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Sustainability Rating:</span>
                  <Badge variant="outline" className="text-green-600">
                    {portfolioData.company_overview.sustainability_rating}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Growth Potential:</span>
                  <Badge variant="outline" className="text-blue-600">
                    {portfolioData.company_overview.growth_potential}
                  </Badge>
                </div>
              </div>
            )}

            {/* Recent Activity */}
            <div className="mt-6">
              <h4 className="font-semibold mb-3">Recent Activity</h4>
              <div className="space-y-2">
                {portfolioData?.recent_activity?.slice(0, 3).map((activity, index) => (
                  <div key={index} className="flex items-center space-x-3 text-sm">
                    <CheckCircle className="w-4 h-4 text-green-600" />
                    <div>
                      <p className="font-medium">{activity.action}</p>
                      <p className="text-gray-600">{activity.date} • {activity.impact}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* AI Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Lightbulb className="w-5 h-5" />
            <span>AI Recommendations for Growth</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {portfolioData?.recommendations?.map((rec, index) => (
              <div key={index} className="border rounded-lg p-4">
                <h4 className="font-semibold mb-2">{rec.category}</h4>
                <ul className="space-y-1">
                  {rec.suggestions.slice(0, 3).map((suggestion, idx) => (
                    <li key={idx} className="text-sm text-gray-600 flex items-start space-x-2">
                      <span className="text-green-600 mt-1">•</span>
                      <span>{suggestion}</span>
                    </li>
                  ))}
                </ul>
                <Badge className={`mt-3 ${getPriorityColor(rec.priority)}`}>
                  {rec.priority} priority
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Button 
              className="h-20 flex flex-col space-y-2" 
              variant="outline"
              onClick={() => navigate('/marketplace')}
            >
              <Users className="w-6 h-6" />
              <span className="text-sm">Find Matches</span>
            </Button>
            <Button 
              className="h-20 flex flex-col space-y-2" 
              variant="outline"
              onClick={() => navigate('/green-initiatives')}
            >
              <Leaf className="w-6 h-6" />
              <span className="text-sm">Green Initiatives</span>
            </Button>
            <Button 
              className="h-20 flex flex-col space-y-2" 
              variant="outline"
              onClick={() => navigate('/portfolio')}
            >
              <Target className="w-6 h-6" />
              <span className="text-sm">My Portfolio</span>
            </Button>
            <Button 
              className="h-20 flex flex-col space-y-2" 
              variant="outline"
              onClick={() => navigate('/onboarding')}
            >
              <TrendingUp className="w-6 h-6" />
              <span className="text-sm">AI Onboarding</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;