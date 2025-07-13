import React, { useState, useEffect } from 'react';
import { Factory, Recycle, Users, Workflow, Home, CheckCircle, TrendingUp, Globe, DollarSign, Leaf, Target, ArrowRight, Play, BarChart3, Award, Zap, Brain, Truck } from 'lucide-react';
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

const DemoLandingPage: React.FC = () => {
  const [stats, setStats] = useState<DemoStats>({
    totalCompanies: 0,
    totalMatches: 0,
    totalSavings: 0,
    carbonReduction: 0,
    activePartnerships: 0,
    countries: 0
  });
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    loadDemoStats();
  }, []);

  const loadDemoStats = async () => {
    try {
      // Simulate loading real data
      setTimeout(() => {
        setStats({
          totalCompanies: 247,
          totalMatches: 1893,
          totalSavings: 45000000, // $45M
          carbonReduction: 125000, // 125K tons
          activePartnerships: 156,
          countries: 23
        });
        setIsLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Error loading demo stats:', error);
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Hero Section */}
      <header className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 to-purple-600/20"></div>
        <div className="relative z-10">
          <nav className="container mx-auto px-6 py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-gradient-to-r from-emerald-400 to-blue-500 rounded-lg">
                  <Workflow className="h-8 w-8 text-white" />
                </div>
                <div>
                  <span className="text-2xl font-bold text-white">SymbioFlows</span>
                  <div className="text-xs text-emerald-300">Industrial Symbiosis Platform</div>
                </div>
              </div>
              <div className="hidden md:flex items-center space-x-6">
                <a href="#impact" className="text-gray-300 hover:text-white transition">Impact</a>
                <a href="#technology" className="text-gray-300 hover:text-white transition">Technology</a>
                <a href="#demo" className="text-gray-300 hover:text-white transition">Live Demo</a>
                <button 
                  onClick={() => navigate('/dashboard')}
                  className="bg-gradient-to-r from-emerald-500 to-blue-500 text-white px-6 py-2 rounded-lg hover:from-emerald-600 hover:to-blue-600 transition-all duration-300 shadow-lg hover:shadow-xl"
                >
                  Launch Platform
                </button>
              </div>
            </div>
          </nav>

          <div className="container mx-auto px-6 py-24 text-center">
            <div className="max-w-4xl mx-auto">
              <h1 className="text-6xl md:text-7xl font-bold text-white mb-8 leading-tight">
                The Future of
                <span className="block bg-gradient-to-r from-emerald-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                  Industrial Symbiosis
                </span>
              </h1>
              <p className="text-xl md:text-2xl text-gray-300 mb-8 leading-relaxed">
                AI-powered platform connecting industries worldwide to create circular economies, 
                reduce waste, and generate millions in savings through intelligent resource matching.
              </p>
              <div className="flex flex-col md:flex-row gap-4 justify-center items-center">
                <button 
                  onClick={() => navigate('/dashboard')}
                  className="bg-gradient-to-r from-emerald-500 to-blue-500 text-white px-8 py-4 rounded-lg hover:from-emerald-600 hover:to-blue-600 transition-all duration-300 shadow-lg hover:shadow-xl text-lg font-semibold flex items-center space-x-2"
                >
                  <Play className="w-5 h-5" />
                  <span>Start Live Demo</span>
                </button>
                <button className="bg-white/10 backdrop-blur-sm text-white px-8 py-4 rounded-lg hover:bg-white/20 transition-all duration-300 border border-white/20 text-lg font-semibold flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5" />
                  <span>View Impact Report</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Impact Stats */}
      <section id="impact" className="py-20 bg-white/5 backdrop-blur-sm">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">Global Impact</h2>
            <p className="text-xl text-gray-300">Transforming industries across 23 countries</p>
          </div>
          
          {isLoading ? (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="animate-pulse">
                  <div className="bg-white/10 rounded-lg p-6 h-32"></div>
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-8">
              <div className="bg-gradient-to-br from-blue-500/20 to-blue-600/20 backdrop-blur-sm rounded-xl p-6 border border-blue-500/30">
                <div className="text-center">
                  <Users className="w-8 h-8 text-blue-400 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-white">{formatNumber(stats.totalCompanies)}</div>
                  <div className="text-sm text-gray-300">Companies</div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-emerald-500/20 to-emerald-600/20 backdrop-blur-sm rounded-xl p-6 border border-emerald-500/30">
                <div className="text-center">
                  <Target className="w-8 h-8 text-emerald-400 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-white">{formatNumber(stats.totalMatches)}</div>
                  <div className="text-sm text-gray-300">AI Matches</div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-purple-500/20 to-purple-600/20 backdrop-blur-sm rounded-xl p-6 border border-purple-500/30">
                <div className="text-center">
                  <DollarSign className="w-8 h-8 text-purple-400 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-white">{formatCurrency(stats.totalSavings)}</div>
                  <div className="text-sm text-gray-300">Total Savings</div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-green-500/20 to-green-600/20 backdrop-blur-sm rounded-xl p-6 border border-green-500/30">
                <div className="text-center">
                  <Leaf className="w-8 h-8 text-green-400 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-white">{formatNumber(stats.carbonReduction)}</div>
                  <div className="text-sm text-gray-300">Tons CO2 Saved</div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-orange-500/20 to-orange-600/20 backdrop-blur-sm rounded-xl p-6 border border-orange-500/30">
                <div className="text-center">
                  <CheckCircle className="w-8 h-8 text-orange-400 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-white">{formatNumber(stats.activePartnerships)}</div>
                  <div className="text-sm text-gray-300">Partnerships</div>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-red-500/20 to-red-600/20 backdrop-blur-sm rounded-xl p-6 border border-red-500/30">
                <div className="text-center">
                  <Globe className="w-8 h-8 text-red-400 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-white">{formatNumber(stats.countries)}</div>
                  <div className="text-sm text-gray-300">Countries</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Technology Section */}
      <section id="technology" className="py-20">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">Cutting-Edge Technology</h2>
            <p className="text-xl text-gray-300">AI-powered platform with real-time logistics and carbon tracking</p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 hover:border-emerald-500/50 transition-all duration-300">
              <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-lg flex items-center justify-center mb-6">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">AI-Powered Matching</h3>
              <p className="text-gray-300 mb-4">
                Advanced machine learning algorithms analyze company profiles and find optimal symbiosis opportunities with 95% accuracy.
              </p>
              <ul className="text-sm text-gray-400 space-y-2">
                <li>• Real-time compatibility scoring</li>
                <li>• Multi-factor analysis</li>
                <li>• Predictive modeling</li>
              </ul>
            </div>
            
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 hover:border-emerald-500/50 transition-all duration-300">
              <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-lg flex items-center justify-center mb-6">
                <Truck className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Real-Time Logistics</h3>
              <p className="text-gray-300 mb-4">
                Integrated with Freightos API for real-time shipping costs, carbon emissions, and multi-modal transport optimization.
              </p>
              <ul className="text-sm text-gray-400 space-y-2">
                <li>• Live freight rates</li>
                <li>• Carbon footprint tracking</li>
                <li>• Route optimization</li>
              </ul>
            </div>
            
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10 hover:border-emerald-500/50 transition-all duration-300">
              <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-lg flex items-center justify-center mb-6">
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Analytics & Insights</h3>
              <p className="text-gray-300 mb-4">
                Comprehensive dashboards with ROI calculations, sustainability metrics, and partnership performance tracking.
              </p>
              <ul className="text-sm text-gray-400 space-y-2">
                <li>• ROI projections</li>
                <li>• Sustainability scoring</li>
                <li>• Performance analytics</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" className="py-20 bg-gradient-to-r from-emerald-900/20 to-blue-900/20">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">Live Demo</h2>
            <p className="text-xl text-gray-300">Experience the platform in action</p>
          </div>
          
          <div className="max-w-4xl mx-auto">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10">
              <div className="grid md:grid-cols-2 gap-8 items-center">
                <div>
                  <h3 className="text-2xl font-semibold text-white mb-4">Ready to Transform Your Industry?</h3>
                  <p className="text-gray-300 mb-6">
                    Join hundreds of companies already saving millions through intelligent resource matching and circular economy partnerships.
                  </p>
                  <div className="space-y-4">
                    <div className="flex items-center space-x-3">
                      <CheckCircle className="w-5 h-5 text-emerald-400" />
                      <span className="text-gray-300">AI-powered onboarding in 5 minutes</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <CheckCircle className="w-5 h-5 text-emerald-400" />
                      <span className="text-gray-300">Real-time logistics and cost analysis</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <CheckCircle className="w-5 h-5 text-emerald-400" />
                      <span className="text-gray-300">Carbon footprint tracking</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <CheckCircle className="w-5 h-5 text-emerald-400" />
                      <span className="text-gray-300">ROI projections and feasibility analysis</span>
                    </div>
                  </div>
                </div>
                
                <div className="text-center">
                  <div className="bg-gradient-to-r from-emerald-500 to-blue-500 rounded-xl p-8 mb-6">
                    <Award className="w-16 h-16 text-white mx-auto mb-4" />
                    <h4 className="text-xl font-semibold text-white mb-2">Industry Leader</h4>
                    <p className="text-emerald-100">Trusted by Fortune 500 companies</p>
                  </div>
                  <button 
                    onClick={() => navigate('/dashboard')}
                    className="w-full bg-gradient-to-r from-emerald-500 to-blue-500 text-white px-8 py-4 rounded-lg hover:from-emerald-600 hover:to-blue-600 transition-all duration-300 shadow-lg hover:shadow-xl text-lg font-semibold flex items-center justify-center space-x-2"
                  >
                    <Zap className="w-5 h-5" />
                    <span>Launch Demo Now</span>
                    <ArrowRight className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-white/10">
        <div className="container mx-auto px-6 text-center">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <Workflow className="h-8 w-8 text-emerald-400" />
            <span className="text-2xl font-bold text-white">SymbioFlows</span>
          </div>
          <p className="text-gray-400 mb-4">
            The future of industrial symbiosis is here. Join the circular economy revolution.
          </p>
          <div className="flex justify-center space-x-6 text-sm text-gray-500">
            <span>© 2024 SymbioFlows. All rights reserved.</span>
            <span>•</span>
            <span>Privacy Policy</span>
            <span>•</span>
            <span>Terms of Service</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default DemoLandingPage; 