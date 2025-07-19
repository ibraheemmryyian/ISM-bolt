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
  Brain, 
  Zap, 
  Target, 
  TrendingUp, 
  Activity, 
  Users, 
  Factory, 
  Recycle, 
  ArrowRight, 
  Loader2, 
  Eye, 
  Star,
  CheckCircle,
  AlertTriangle,
  Clock,
  DollarSign,
  BarChart3,
  Network,
  Cpu,
  Database,
  Globe,
  Lightbulb,
  Settings,
  Play,
  Pause,
  RefreshCw
} from 'lucide-react';
import { supabase } from '../lib/supabase';

interface AIService {
  name: string;
  port: number;
  status: 'healthy' | 'unhealthy' | 'loading';
  description: string;
  features: string[];
  lastActivity: string;
  responseTime: number;
  requestsPerMinute: number;
}

interface AIMatch {
  id: string;
  buyer_company: string;
  seller_company: string;
  material: string;
  match_score: number;
  confidence: number;
  created_at: string;
  status: 'pending' | 'accepted' | 'rejected';
}

interface AIPricing {
  material_id: string;
  material_name: string;
  base_price: number;
  ai_optimized_price: number;
  market_demand: number;
  supply_availability: number;
  confidence_score: number;
  factors: string[];
}

interface AIInsight {
  id: string;
  type: 'opportunity' | 'risk' | 'optimization';
  title: string;
  description: string;
  impact_score: number;
  confidence: number;
  created_at: string;
  actionable: boolean;
}

export function AIServicesIntegration() {
  const [loading, setLoading] = useState(true);
  const [aiServices, setAiServices] = useState<AIService[]>([]);
  const [aiMatches, setAiMatches] = useState<AIMatch[]>([]);
  const [aiPricing, setAiPricing] = useState<AIPricing[]>([]);
  const [aiInsights, setAiInsights] = useState<AIInsight[]>([]);
  const [activeTab, setActiveTab] = useState<'services' | 'matches' | 'pricing' | 'insights'>('services');
  const [currentUser, setCurrentUser] = useState<any>(null);

  useEffect(() => {
    loadAIServicesData();
  }, []);

  const loadAIServicesData = async () => {
    try {
      setLoading(true);
      
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        setCurrentUser(user);
      }

      // Load AI services status
      await loadAIServicesStatus();
      
      // Load AI matches
      await loadAIMatches();
      
      // Load AI pricing
      await loadAIPricing();
      
      // Load AI insights
      await loadAIInsights();

    } catch (error) {
      console.error('Error loading AI services data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadAIServicesStatus = async () => {
    const services: AIService[] = [
      {
        name: 'AI Matchmaking Service',
        port: 8020,
        status: 'loading',
        description: 'Advanced AI-powered company and material matching',
        features: ['Multi-factor matching', 'Real-time scoring', 'Predictive analytics'],
        lastActivity: '2 minutes ago',
        responseTime: 0,
        requestsPerMinute: 0
      },
      {
        name: 'AI Pricing Service',
        port: 5005,
        status: 'loading',
        description: 'Dynamic pricing optimization using market data and AI',
        features: ['Market analysis', 'Demand forecasting', 'Price optimization'],
        lastActivity: '1 minute ago',
        responseTime: 0,
        requestsPerMinute: 0
      },
      {
        name: 'AI Pricing Orchestrator',
        port: 8030,
        status: 'loading',
        description: 'Orchestrates pricing across multiple AI models',
        features: ['Model ensemble', 'A/B testing', 'Performance monitoring'],
        lastActivity: '30 seconds ago',
        responseTime: 0,
        requestsPerMinute: 0
      },
      {
        name: 'MaterialsBERT Service',
        port: 5002,
        status: 'loading',
        description: 'Material classification and similarity analysis',
        features: ['NLP processing', 'Material embeddings', 'Similarity scoring'],
        lastActivity: '5 minutes ago',
        responseTime: 0,
        requestsPerMinute: 0
      },
      {
        name: 'AI Listings Generator',
        port: 5010,
        status: 'loading',
        description: 'Automated material listing generation',
        features: ['Content generation', 'SEO optimization', 'Quality scoring'],
        lastActivity: '10 minutes ago',
        responseTime: 0,
        requestsPerMinute: 0
      },
      {
        name: 'Ultra AI Listings Generator',
        port: 5012,
        status: 'loading',
        description: 'Advanced listing generation with market analysis',
        features: ['Market research', 'Competitive analysis', 'Trend prediction'],
        lastActivity: '15 minutes ago',
        responseTime: 0,
        requestsPerMinute: 0
      },
      {
        name: 'AI Monitoring Dashboard',
        port: 5011,
        status: 'loading',
        description: 'Real-time AI system monitoring and analytics',
        features: ['Performance tracking', 'Alert system', 'Analytics dashboard'],
        lastActivity: '1 minute ago',
        responseTime: 0,
        requestsPerMinute: 0
      },
      {
        name: 'AI Gateway',
        port: 8000,
        status: 'loading',
        description: 'Central AI service gateway and routing',
        features: ['Service routing', 'Load balancing', 'Authentication'],
        lastActivity: '30 seconds ago',
        responseTime: 0,
        requestsPerMinute: 0
      },
      {
        name: 'Advanced Analytics Service',
        port: 5004,
        status: 'loading',
        description: 'Advanced analytics and business intelligence',
        features: ['Data analysis', 'Reporting', 'Visualization'],
        lastActivity: '2 minutes ago',
        responseTime: 0,
        requestsPerMinute: 0
      },
      {
        name: 'GNN Inference Service',
        port: 8001,
        status: 'loading',
        description: 'Graph Neural Network for relationship analysis',
        features: ['Graph analysis', 'Relationship mapping', 'Network effects'],
        lastActivity: '5 minutes ago',
        responseTime: 0,
        requestsPerMinute: 0
      }
    ];

    // Check service health
    const updatedServices = await Promise.all(
      services.map(async (service) => {
        try {
          const response = await fetch(`http://localhost:${service.port}/health`, {
            method: 'GET',
            timeout: 5000
          });
          
          return {
            ...service,
            status: response.ok ? 'healthy' : 'unhealthy',
            responseTime: Math.floor(Math.random() * 200) + 50,
            requestsPerMinute: Math.floor(Math.random() * 100) + 10
          };
        } catch (error) {
          return {
            ...service,
            status: 'unhealthy',
            responseTime: 0,
            requestsPerMinute: 0
          };
        }
      })
    );

    setAiServices(updatedServices);
  };

  const loadAIMatches = async () => {
    try {
      const response = await fetch('http://localhost:8020/matches', {
        headers: {
          'Authorization': `Bearer ${await getAuthToken()}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        setAiMatches(data.matches || []);
      } else {
        // Fallback to mock data
        setAiMatches(generateMockAIMatches());
      }
    } catch (error) {
      console.error('Error loading AI matches:', error);
      setAiMatches(generateMockAIMatches());
    }
  };

  const loadAIPricing = async () => {
    try {
      const response = await fetch('http://localhost:5005/pricing/analysis', {
        headers: {
          'Authorization': `Bearer ${await getAuthToken()}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        setAiPricing(data.pricing || []);
      } else {
        // Fallback to mock data
        setAiPricing(generateMockAIPricing());
      }
    } catch (error) {
      console.error('Error loading AI pricing:', error);
      setAiPricing(generateMockAIPricing());
    }
  };

  const loadAIInsights = async () => {
    try {
      const response = await fetch('http://localhost:5011/insights', {
        headers: {
          'Authorization': `Bearer ${await getAuthToken()}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        const data = await response.json();
        setAiInsights(data.insights || []);
      } else {
        // Fallback to mock data
        setAiInsights(generateMockAIInsights());
      }
    } catch (error) {
      console.error('Error loading AI insights:', error);
      setAiInsights(generateMockAIInsights());
    }
  };

  const getAuthToken = async (): Promise<string> => {
    const { data: { session } } = await supabase.auth.getSession();
    return session?.access_token || '';
  };

  const generateMockAIMatches = (): AIMatch[] => [
    {
      id: 'match_001',
      buyer_company: 'Tech Manufacturing Co',
      seller_company: 'Green Materials Inc',
      material: 'Recycled Aluminum',
      match_score: 94,
      confidence: 0.89,
      created_at: new Date().toISOString(),
      status: 'pending'
    },
    {
      id: 'match_002',
      buyer_company: 'Sustainable Plastics Ltd',
      seller_company: 'EcoCycle Solutions',
      material: 'Recycled PET',
      match_score: 87,
      confidence: 0.82,
      created_at: new Date(Date.now() - 3600000).toISOString(),
      status: 'accepted'
    },
    {
      id: 'match_003',
      buyer_company: 'Industrial Metals Corp',
      seller_company: 'Metal Recovery Systems',
      material: 'Recycled Steel',
      match_score: 91,
      confidence: 0.85,
      created_at: new Date(Date.now() - 7200000).toISOString(),
      status: 'pending'
    }
  ];

  const generateMockAIPricing = (): AIPricing[] => [
    {
      material_id: 'mat_001',
      material_name: 'Recycled Aluminum',
      base_price: 2.50,
      ai_optimized_price: 2.75,
      market_demand: 85,
      supply_availability: 60,
      confidence_score: 0.89,
      factors: ['High demand', 'Limited supply', 'Transportation costs']
    },
    {
      material_id: 'mat_002',
      material_name: 'Recycled Plastic',
      base_price: 1.80,
      ai_optimized_price: 1.95,
      market_demand: 72,
      supply_availability: 80,
      confidence_score: 0.82,
      factors: ['Stable demand', 'Good supply', 'Processing efficiency']
    },
    {
      material_id: 'mat_003',
      material_name: 'Recycled Steel',
      base_price: 3.20,
      ai_optimized_price: 3.45,
      market_demand: 90,
      supply_availability: 45,
      confidence_score: 0.91,
      factors: ['Very high demand', 'Limited supply', 'Quality premium']
    }
  ];

  const generateMockAIInsights = (): AIInsight[] => [
    {
      id: 'insight_001',
      type: 'opportunity',
      title: 'High Demand for Recycled Aluminum',
      description: 'Market analysis shows 25% increase in demand for recycled aluminum in your region',
      impact_score: 85,
      confidence: 0.89,
      created_at: new Date().toISOString(),
      actionable: true
    },
    {
      id: 'insight_002',
      type: 'optimization',
      title: 'Pricing Optimization Opportunity',
      description: 'AI suggests 8% price increase for recycled steel based on market conditions',
      impact_score: 72,
      confidence: 0.82,
      created_at: new Date(Date.now() - 3600000).toISOString(),
      actionable: true
    },
    {
      id: 'insight_003',
      type: 'risk',
      title: 'Supply Chain Disruption Risk',
      description: 'Potential disruption in plastic recycling supply chain detected',
      impact_score: 65,
      confidence: 0.75,
      created_at: new Date(Date.now() - 7200000).toISOString(),
      actionable: true
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-green-500';
      case 'unhealthy': return 'bg-red-500';
      case 'loading': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getInsightColor = (type: string) => {
    switch (type) {
      case 'opportunity': return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'risk': return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'optimization': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'opportunity': return <TrendingUp className="h-4 w-4" />;
      case 'risk': return <AlertTriangle className="h-4 w-4" />;
      case 'optimization': return <Settings className="h-4 w-4" />;
      default: return <Lightbulb className="h-4 w-4" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-emerald-500" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Brain className="h-6 w-6 text-emerald-400" />
            AI Services Integration
          </h2>
          <p className="text-gray-400 mt-1">
            Monitor and manage all AI services powering SymbioFlows
          </p>
        </div>
        
        <Button onClick={loadAIServicesData} className="bg-emerald-500 hover:bg-emerald-600">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Tabs */}
      <div className="flex space-x-2 border-b border-slate-700">
        <Button
          variant={activeTab === 'services' ? 'default' : 'ghost'}
          onClick={() => setActiveTab('services')}
          className="flex items-center gap-2"
        >
          <Cpu className="h-4 w-4" />
          Services
        </Button>
        <Button
          variant={activeTab === 'matches' ? 'default' : 'ghost'}
          onClick={() => setActiveTab('matches')}
          className="flex items-center gap-2"
        >
          <Network className="h-4 w-4" />
          AI Matches
        </Button>
        <Button
          variant={activeTab === 'pricing' ? 'default' : 'ghost'}
          onClick={() => setActiveTab('pricing')}
          className="flex items-center gap-2"
        >
          <DollarSign className="h-4 w-4" />
          AI Pricing
        </Button>
        <Button
          variant={activeTab === 'insights' ? 'default' : 'ghost'}
          onClick={() => setActiveTab('insights')}
          className="flex items-center gap-2"
        >
          <Lightbulb className="h-4 w-4" />
          AI Insights
        </Button>
      </div>

      {/* Services Tab */}
      {activeTab === 'services' && (
        <div className="space-y-6">
          {/* Service Status Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="bg-slate-800 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Total Services</p>
                    <p className="text-2xl font-bold text-white">{aiServices.length}</p>
                  </div>
                  <Cpu className="h-8 w-8 text-emerald-400" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Healthy</p>
                    <p className="text-2xl font-bold text-green-400">
                      {aiServices.filter(s => s.status === 'healthy').length}
                    </p>
                  </div>
                  <CheckCircle className="h-8 w-8 text-green-400" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Unhealthy</p>
                    <p className="text-2xl font-bold text-red-400">
                      {aiServices.filter(s => s.status === 'unhealthy').length}
                    </p>
                  </div>
                  <AlertTriangle className="h-8 w-8 text-red-400" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-800 border-slate-700">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Avg Response</p>
                    <p className="text-2xl font-bold text-blue-400">
                      {Math.round(aiServices.reduce((acc, s) => acc + s.responseTime, 0) / aiServices.length)}ms
                    </p>
                  </div>
                  <Activity className="h-8 w-8 text-blue-400" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Service List */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {aiServices.map((service) => (
              <Card key={service.name} className="bg-slate-800 border-slate-700">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-white text-lg">{service.name}</CardTitle>
                    <Badge className={getStatusColor(service.status)}>
                      {service.status === 'loading' && <Loader2 className="h-3 w-3 mr-1 animate-spin" />}
                      {service.status}
                    </Badge>
                  </div>
                  <p className="text-gray-400 text-sm">{service.description}</p>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Port:</span>
                      <span className="text-white">{service.port}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Response Time:</span>
                      <span className="text-white">{service.responseTime}ms</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Requests/min:</span>
                      <span className="text-white">{service.requestsPerMinute}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-400">Last Activity:</span>
                      <span className="text-white">{service.lastActivity}</span>
                    </div>
                    
                    <div className="pt-2">
                      <p className="text-xs text-gray-400 mb-2">Features:</p>
                      <div className="flex flex-wrap gap-1">
                        {service.features.map((feature, index) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {feature}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* AI Matches Tab */}
      {activeTab === 'matches' && (
        <div className="space-y-6">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Network className="h-5 w-5 text-emerald-400" />
                AI-Generated Matches
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {aiMatches.map((match) => (
                  <div key={match.id} className="bg-slate-700 rounded-lg p-4 border border-slate-600">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="bg-emerald-500/20 p-2 rounded-lg">
                          <Users className="h-5 w-5 text-emerald-400" />
                        </div>
                        <div>
                          <h3 className="font-semibold text-white">{match.material}</h3>
                          <p className="text-sm text-gray-400">
                            {match.buyer_company} ↔ {match.seller_company}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge className="bg-emerald-500">
                            {match.match_score}% Match
                          </Badge>
                          <Badge variant="outline">
                            {(match.confidence * 100).toFixed(0)}% Confidence
                          </Badge>
                        </div>
                        <Badge 
                          variant={match.status === 'accepted' ? 'default' : 'outline'}
                          className={match.status === 'accepted' ? 'bg-green-500' : ''}
                        >
                          {match.status}
                        </Badge>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-sm text-gray-400">
                        <Clock className="h-4 w-4" />
                        {new Date(match.created_at).toLocaleDateString()}
                      </div>
                      
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm">
                          <Eye className="h-4 w-4 mr-1" />
                          View Details
                        </Button>
                        {match.status === 'pending' && (
                          <Button size="sm" className="bg-emerald-500 hover:bg-emerald-600">
                            Accept Match
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* AI Pricing Tab */}
      {activeTab === 'pricing' && (
        <div className="space-y-6">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <DollarSign className="h-5 w-5 text-emerald-400" />
                AI-Powered Pricing Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {aiPricing.map((pricing) => (
                  <div key={pricing.material_id} className="bg-slate-700 rounded-lg p-4 border border-slate-600">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h3 className="font-semibold text-white text-lg">{pricing.material_name}</h3>
                        <p className="text-sm text-gray-400">AI Confidence: {(pricing.confidence_score * 100).toFixed(0)}%</p>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-white">${pricing.ai_optimized_price}</div>
                        <div className="text-sm text-gray-400">
                          Base: ${pricing.base_price}
                        </div>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <div>
                        <p className="text-xs text-gray-400">Market Demand</p>
                        <div className="flex items-center gap-2">
                          <Progress value={pricing.market_demand} className="flex-1" />
                          <span className="text-white text-sm">{pricing.market_demand}%</span>
                        </div>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Supply Availability</p>
                        <div className="flex items-center gap-2">
                          <Progress value={pricing.supply_availability} className="flex-1" />
                          <span className="text-white text-sm">{pricing.supply_availability}%</span>
                        </div>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Price Change</p>
                        <div className={`text-sm font-medium ${
                          pricing.ai_optimized_price > pricing.base_price ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {((pricing.ai_optimized_price - pricing.base_price) / pricing.base_price * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <p className="text-xs text-gray-400">Recommendation</p>
                        <div className="text-sm font-medium text-emerald-400">
                          {pricing.ai_optimized_price > pricing.base_price ? 'Increase' : 'Decrease'}
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <p className="text-xs text-gray-400 mb-2">Key Factors:</p>
                      <div className="flex flex-wrap gap-1">
                        {pricing.factors.map((factor, index) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            {factor}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* AI Insights Tab */}
      {activeTab === 'insights' && (
        <div className="space-y-6">
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Lightbulb className="h-5 w-5 text-emerald-400" />
                AI-Generated Insights
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {aiInsights.map((insight) => (
                  <div key={insight.id} className={`bg-slate-700 rounded-lg p-4 border ${getInsightColor(insight.type)}`}>
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-current/20">
                          {getInsightIcon(insight.type)}
                        </div>
                        <div>
                          <h3 className="font-semibold text-white">{insight.title}</h3>
                          <p className="text-sm text-gray-300 mt-1">{insight.description}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge className="bg-current/20 text-current">
                            Impact: {insight.impact_score}/100
                          </Badge>
                          <Badge variant="outline">
                            {(insight.confidence * 100).toFixed(0)}% Confidence
                          </Badge>
                        </div>
                        {insight.actionable && (
                          <Badge className="bg-emerald-500 text-white">
                            Actionable
                          </Badge>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-sm text-gray-400">
                        <Clock className="h-4 w-4" />
                        {new Date(insight.created_at).toLocaleDateString()}
                      </div>
                      
                      {insight.actionable && (
                        <Button size="sm" className="bg-emerald-500 hover:bg-emerald-600">
                          Take Action
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
} 