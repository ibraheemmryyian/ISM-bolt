import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  User, 
  TrendingUp, 
  DollarSign, 
  Leaf, 
  Target, 
  Award,
  Star,
  Calendar,
  MapPin,
  Users,
  Lightbulb,
  ArrowRight,
  CheckCircle
} from 'lucide-react';

interface CompanyProfile {
  id: string;
  name: string;
  industry: string;
  location: string;
  employee_count: number;
  size_category: string;
  industry_position: string;
  sustainability_rating: string;
  growth_potential: string;
  joined_date: string;
}

interface Achievements {
  total_savings: number;
  carbon_reduced: number;
  partnerships_formed: number;
  waste_diverted: number;
  matches_completed: number;
  sustainability_score: number;
  efficiency_improvement: number;
}

interface PersonalizedRecommendation {
  id: string;
  category: string;
  title: string;
  description: string;
  potential_impact: {
    savings: number;
    carbon_reduction: number;
    efficiency_gain: number;
  };
  implementation_difficulty: 'easy' | 'medium' | 'hard';
  time_to_implement: string;
  priority: 'high' | 'medium' | 'low';
  ai_reasoning: string;
}

interface RecentActivity {
  id: string;
  date: string;
  action: string;
  impact: string;
  category: 'match' | 'savings' | 'partnership' | 'sustainability';
}

interface PortfolioData {
  company: CompanyProfile;
  achievements: Achievements;
  recommendations: PersonalizedRecommendation[];
  recent_activity: RecentActivity[];
  next_milestones: string[];
  industry_comparison: {
    rank: number;
    total_companies: number;
    average_savings: number;
    your_savings: number;
  };
}

const PersonalPortfolio: React.FC = () => {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  useEffect(() => {
    loadPortfolioData();
  }, []);

  const loadPortfolioData = async () => {
    try {
      setLoading(true);
      
      const response = await fetch('/api/portfolio/personal');
      const data = await response.json();
      
      if (data.success) {
        setPortfolioData(data.portfolio);
      }
    } catch (error) {
      console.error('Error loading portfolio data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'text-red-600 bg-red-50';
      case 'medium': return 'text-orange-600 bg-orange-50';
      case 'low': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'text-green-600 bg-green-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'hard': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getActivityIcon = (category: string) => {
    switch (category) {
      case 'match': return <Users className="w-4 h-4" />;
      case 'savings': return <DollarSign className="w-4 h-4" />;
      case 'partnership': return <Target className="w-4 h-4" />;
      case 'sustainability': return <Leaf className="w-4 h-4" />;
      default: return <Star className="w-4 h-4" />;
    }
  };

  const getActivityColor = (category: string) => {
    switch (category) {
      case 'match': return 'text-blue-600 bg-blue-50';
      case 'savings': return 'text-green-600 bg-green-50';
      case 'partnership': return 'text-purple-600 bg-purple-50';
      case 'sustainability': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!portfolioData) {
    return (
      <div className="container mx-auto p-6">
        <Card>
          <CardContent className="p-6 text-center">
            <p className="text-gray-600">No portfolio data available</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Company Profile Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
        <div className="flex items-center space-x-4">
          <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center">
            <User className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">{portfolioData.company.name}</h1>
            <p className="text-blue-100 flex items-center space-x-2">
              <MapPin className="w-4 h-4" />
              <span>{portfolioData.company.location}</span>
              <span>•</span>
              <span>{portfolioData.company.industry}</span>
              <span>•</span>
              <span>{portfolioData.company.employee_count} employees</span>
            </p>
            <p className="text-blue-100 mt-1">
              Member since {new Date(portfolioData.company.joined_date).toLocaleDateString()}
            </p>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <DollarSign className="w-8 h-8 text-green-600" />
              <div>
                <p className="text-sm text-gray-600">Total Savings</p>
                <p className="text-2xl font-bold">${portfolioData.achievements.total_savings.toLocaleString()}</p>
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
                <p className="text-2xl font-bold">{portfolioData.achievements.carbon_reduced} tons</p>
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
                <p className="text-2xl font-bold">{portfolioData.achievements.partnerships_formed}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Award className="w-8 h-8 text-purple-600" />
              <div>
                <p className="text-sm text-gray-600">Sustainability Score</p>
                <p className="text-2xl font-bold">{portfolioData.achievements.sustainability_score}%</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Company Overview */}
        <Card>
          <CardHeader>
            <CardTitle>Company Profile</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Size Category</p>
                <p className="font-semibold">{portfolioData.company.size_category}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Industry Position</p>
                <p className="font-semibold">{portfolioData.company.industry_position}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Sustainability Rating</p>
                <Badge variant="outline" className="text-green-600">
                  {portfolioData.company.sustainability_rating}
                </Badge>
              </div>
              <div>
                <p className="text-sm text-gray-600">Growth Potential</p>
                <Badge variant="outline" className="text-blue-600">
                  {portfolioData.company.growth_potential}
                </Badge>
              </div>
            </div>
            
            <div className="border-t pt-4">
              <p className="text-sm text-gray-600 mb-2">Efficiency Improvement</p>
              <Progress value={portfolioData.achievements.efficiency_improvement} className="w-full" />
              <p className="text-sm text-gray-600 mt-1">
                {portfolioData.achievements.efficiency_improvement}% improvement in operational efficiency
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Industry Comparison */}
        <Card>
          <CardHeader>
            <CardTitle>Industry Standing</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-600">
                #{portfolioData.industry_comparison.rank}
              </p>
              <p className="text-sm text-gray-600">
                out of {portfolioData.industry_comparison.total_companies} companies
              </p>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Your Savings</span>
                <span className="font-semibold text-green-600">
                  ${portfolioData.industry_comparison.your_savings.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Industry Average</span>
                <span className="font-semibold text-gray-600">
                  ${portfolioData.industry_comparison.average_savings.toLocaleString()}
                </span>
              </div>
            </div>
            
            <div className="bg-blue-50 p-3 rounded-lg">
              <p className="text-sm text-blue-800">
                You're performing {portfolioData.industry_comparison.your_savings > portfolioData.industry_comparison.average_savings ? 'above' : 'below'} the industry average!
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Personalized Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Lightbulb className="w-5 h-5" />
            <span>AI-Powered Recommendations</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {portfolioData.recommendations.map((rec) => (
              <Card key={rec.id} className="hover:shadow-lg transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <CardTitle className="text-lg">{rec.title}</CardTitle>
                    <Badge className={getPriorityColor(rec.priority)}>
                      {rec.priority}
                    </Badge>
                  </div>
                  <Badge className={getDifficultyColor(rec.implementation_difficulty)}>
                    {rec.implementation_difficulty}
                  </Badge>
                </CardHeader>
                <CardContent className="space-y-3">
                  <p className="text-sm text-gray-600">{rec.description}</p>
                  
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="text-center">
                      <p className="text-gray-600">Savings</p>
                      <p className="font-bold text-green-600">${rec.potential_impact.savings.toLocaleString()}</p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-600">Carbon</p>
                      <p className="font-bold text-green-600">{rec.potential_impact.carbon_reduction} tons</p>
                    </div>
                    <div className="text-center">
                      <p className="text-gray-600">Efficiency</p>
                      <p className="font-bold text-blue-600">+{rec.potential_impact.efficiency_gain}%</p>
                    </div>
                  </div>
                  
                  <div className="text-xs text-gray-500">
                    <p><strong>AI Reasoning:</strong> {rec.ai_reasoning}</p>
                    <p className="mt-1"><strong>Time to implement:</strong> {rec.time_to_implement}</p>
                  </div>
                  
                  <Button className="w-full" size="sm">
                    Learn More <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {portfolioData.recent_activity.map((activity) => (
              <div key={activity.id} className="flex items-center space-x-3 p-3 hover:bg-gray-50 rounded-lg">
                <div className={`p-2 rounded-full ${getActivityColor(activity.category)}`}>
                  {getActivityIcon(activity.category)}
                </div>
                <div className="flex-1">
                  <p className="font-medium">{activity.action}</p>
                  <p className="text-sm text-gray-600">{activity.impact}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-500">{new Date(activity.date).toLocaleDateString()}</p>
                  <Badge className={getActivityColor(activity.category)}>
                    {activity.category}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Next Milestones */}
      <Card>
        <CardHeader>
          <CardTitle>Next Milestones</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {portfolioData.next_milestones.map((milestone, index) => (
              <div key={index} className="flex items-center space-x-3 p-3 bg-blue-50 rounded-lg">
                <Target className="w-5 h-5 text-blue-600" />
                <span className="font-medium">{milestone}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PersonalPortfolio; 