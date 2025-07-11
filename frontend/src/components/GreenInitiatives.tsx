import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import AuthenticatedLayout from './AuthenticatedLayout';
import { 
  Leaf, 
  CheckCircle, 
  XCircle, 
  TrendingUp,
  Lightbulb,
  Target,
  Award,
  ArrowRight,
  ArrowLeft,
  Home,
  Workflow
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface GreenInitiative {
  id: string;
  category: string;
  question: string;
  description: string;
  impact: 'high' | 'medium' | 'low';
  potential_savings: number;
  carbon_reduction: number;
  implementation_time: string;
  difficulty: 'easy' | 'medium' | 'hard';
  status: 'not_implemented' | 'implemented' | 'in_progress';
  company_response?: 'yes' | 'no' | 'planning';
}

interface CompanyGreenProfile {
  total_initiatives: number;
  implemented_count: number;
  in_progress_count: number;
  total_savings: number;
  total_carbon_reduction: number;
  sustainability_score: number;
  next_priorities: string[];
}

const GreenInitiatives: React.FC = () => {
  const [initiatives, setInitiatives] = useState<GreenInitiative[]>([]);
  const [companyProfile, setCompanyProfile] = useState<CompanyGreenProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const navigate = useNavigate();

  useEffect(() => {
    loadGreenInitiatives();
  }, []);

  const loadGreenInitiatives = async () => {
    try {
      setLoading(true);
      
      const response = await fetch('/api/green-initiatives');
      const data = await response.json();
      
      if (data.success) {
        setInitiatives(data.initiatives);
        setCompanyProfile(data.companyProfile);
      }
    } catch (error) {
      console.error('Error loading green initiatives:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleInitiativeResponse = async (initiativeId: string, response: 'yes' | 'no' | 'planning') => {
    try {
      const apiResponse = await fetch('/api/green-initiatives/respond', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          initiativeId,
          response
        }),
      });

      const data = await apiResponse.json();
      
      if (data.success) {
        // Update local state
        setInitiatives(prev => prev.map(initiative => 
          initiative.id === initiativeId 
            ? { ...initiative, company_response: response }
            : initiative
        ));
        
        // Reload company profile
        loadGreenInitiatives();
      }
    } catch (error) {
      console.error('Error updating initiative response:', error);
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
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

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'implemented': return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'in_progress': return <TrendingUp className="w-5 h-5 text-blue-600" />;
      default: return <Target className="w-5 h-5 text-gray-400" />;
    }
  };

  const getResponseButtonVariant = (initiativeId: string, response: string) => {
    const initiative = initiatives.find(i => i.id === initiativeId);
    return initiative?.company_response === response ? 'default' : 'outline';
  };

  const filteredInitiatives = selectedCategory === 'all' 
    ? initiatives 
    : initiatives.filter(i => i.category === selectedCategory);

  const categories = ['all', ...new Set(initiatives.map(i => i.category))];

  if (loading) {
    return (
      <AuthenticatedLayout title="Green Initiatives">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-emerald-500"></div>
        </div>
      </AuthenticatedLayout>
    );
  }

  return (
    <AuthenticatedLayout title="Green Initiatives">
      <div className="space-y-6">

        {/* Header */}
        <div className="bg-gradient-to-r from-emerald-600 to-blue-600 rounded-lg p-6 text-white">
          <h1 className="text-3xl font-bold mb-2">Green Initiatives Portfolio</h1>
          <p className="text-emerald-100">
            Discover sustainability opportunities tailored to your company
          </p>
        </div>

        {/* Company Green Profile */}
        {companyProfile && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="bg-white/10 backdrop-blur-sm border-gray-700">
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <Award className="w-8 h-8 text-emerald-400" />
                  <div>
                    <p className="text-sm text-gray-300">Sustainability Score</p>
                    <p className="text-2xl font-bold text-white">{companyProfile.sustainability_score}%</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/10 backdrop-blur-sm border-gray-700">
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="w-8 h-8 text-emerald-400" />
                  <div>
                    <p className="text-sm text-gray-300">Implemented</p>
                    <p className="text-2xl font-bold text-white">{companyProfile.implemented_count}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/10 backdrop-blur-sm border-gray-700">
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-8 h-8 text-blue-400" />
                  <div>
                    <p className="text-sm text-gray-300">In Progress</p>
                    <p className="text-2xl font-bold text-white">{companyProfile.in_progress_count}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-white/10 backdrop-blur-sm border-gray-700">
              <CardContent className="p-4">
                <div className="flex items-center space-x-2">
                  <Leaf className="w-8 h-8 text-emerald-400" />
                  <div>
                    <p className="text-sm text-gray-300">Carbon Reduced</p>
                    <p className="text-2xl font-bold text-white">{companyProfile.total_carbon_reduction} tons</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Progress Overview */}
        {companyProfile && (
          <Card className="bg-white/10 backdrop-blur-sm border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Your Green Journey</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Overall Progress</span>
                  <span className="font-semibold text-white">
                    {companyProfile.implemented_count} / {companyProfile.total_initiatives} initiatives
                  </span>
                </div>
                <Progress 
                  value={(companyProfile.implemented_count / companyProfile.total_initiatives) * 100} 
                  className="w-full" 
                />
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div className="text-center">
                    <p className="text-gray-300">Total Savings</p>
                    <p className="font-bold text-emerald-400">${companyProfile.total_savings.toLocaleString()}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-300">Carbon Reduction</p>
                    <p className="font-bold text-emerald-400">{companyProfile.total_carbon_reduction} tons CO2</p>
                  </div>
                  <div className="text-center">
                    <p className="text-gray-300">Next Priorities</p>
                    <p className="font-bold text-blue-400">{companyProfile.next_priorities.length} identified</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Category Filter */}
        <Card className="bg-white/10 backdrop-blur-sm border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">Filter by Category</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {categories.map(category => (
                <Button
                  key={category}
                  variant={selectedCategory === category ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setSelectedCategory(category)}
                  className={selectedCategory === category ? 'bg-emerald-600 hover:bg-emerald-700' : 'border-gray-600 text-gray-300 hover:bg-gray-700 hover:text-white'}
                >
                  {category === 'all' ? 'All Categories' : category}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Initiatives List */}
        <div className="space-y-4">
          {filteredInitiatives.map((initiative) => (
            <Card key={initiative.id} className="bg-white/10 backdrop-blur-sm border-gray-700 hover:shadow-lg transition-shadow hover:bg-white/15">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(initiative.status)}
                    <div>
                      <CardTitle className="text-lg text-white">{initiative.question}</CardTitle>
                      <p className="text-gray-300 mt-1">{initiative.description}</p>
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <Badge className={getImpactColor(initiative.impact)}>
                      {initiative.impact} impact
                    </Badge>
                    <Badge className={getDifficultyColor(initiative.difficulty)}>
                      {initiative.difficulty}
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                  <div className="text-center">
                    <p className="text-sm text-gray-300">Potential Savings</p>
                    <p className="font-bold text-emerald-400">${initiative.potential_savings.toLocaleString()}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-300">Carbon Reduction</p>
                    <p className="font-bold text-emerald-400">{initiative.carbon_reduction} tons CO2</p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-300">Implementation</p>
                    <p className="font-bold text-blue-400">{initiative.implementation_time}</p>
                  </div>
                  <div className="text-center">
                    <p className="text-sm text-gray-300">Category</p>
                    <p className="font-bold text-purple-400">{initiative.category}</p>
                  </div>
                </div>

                {/* Response Buttons */}
                <div className="border-t border-gray-600 pt-4">
                  <p className="text-sm font-medium mb-3 text-gray-300">Do you have this implemented?</p>
                  <div className="flex space-x-3">
                    <Button
                      variant={getResponseButtonVariant(initiative.id, 'yes')}
                      onClick={() => handleInitiativeResponse(initiative.id, 'yes')}
                      className="flex items-center space-x-2"
                    >
                      <CheckCircle className="w-4 h-4" />
                      <span>Yes, we do this</span>
                    </Button>
                    <Button
                      variant={getResponseButtonVariant(initiative.id, 'planning')}
                      onClick={() => handleInitiativeResponse(initiative.id, 'planning')}
                      className="flex items-center space-x-2"
                    >
                      <TrendingUp className="w-4 h-4" />
                      <span>Planning to implement</span>
                    </Button>
                    <Button
                      variant={getResponseButtonVariant(initiative.id, 'no')}
                      onClick={() => handleInitiativeResponse(initiative.id, 'no')}
                      className="flex items-center space-x-2"
                    >
                      <XCircle className="w-4 h-4" />
                      <span>Not yet</span>
                    </Button>
                  </div>
                </div>

                {/* Action Button */}
                {initiative.company_response === 'no' && (
                  <div className="mt-4 pt-4 border-t border-gray-600">
                    <Button className="w-full border-gray-600 text-gray-300 hover:bg-gray-700 hover:text-white" variant="outline">
                      <Lightbulb className="w-4 h-4 mr-2" />
                      Get Implementation Guide
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Next Steps */}
        {companyProfile && companyProfile.next_priorities.length > 0 && (
          <Card className="bg-white/10 backdrop-blur-sm border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Recommended Next Steps</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {companyProfile.next_priorities.map((priority, index) => (
                  <div key={index} className="flex items-center space-x-3 p-3 bg-blue-900/30 rounded-lg border border-blue-700">
                    <Target className="w-5 h-5 text-blue-400" />
                    <span className="font-medium text-gray-300">{priority}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </AuthenticatedLayout>
  );
};

export default GreenInitiatives; 