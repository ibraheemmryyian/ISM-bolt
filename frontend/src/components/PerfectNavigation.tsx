import React, { useState, useEffect } from 'react';
import { 
  Workflow,
  Target,
  Users,
  Brain,
  Sparkles,
  FileText,
  Bell,
  Settings,
  LogOut,
  Home,
  BarChart3,
  MessageSquare,
  Plus,
  Eye,
  TrendingUp,
  Leaf,
  Menu,
  X
} from 'lucide-react';
import { useNavigate, useLocation } from 'react-router-dom';
import { supabase } from '../lib/supabase';

interface PerfectNavigationProps {
  onSignOut?: () => void;
  showDuringOnboarding?: boolean;
}

const PerfectNavigation: React.FC<PerfectNavigationProps> = ({ 
  onSignOut,
  showDuringOnboarding = true 
}) => {
  const [companyProfile, setCompanyProfile] = useState<any>(null);
  const [hasCompletedOnboarding, setHasCompletedOnboarding] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    loadUserContext();
  }, []);

  const loadUserContext = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        const { data: company } = await supabase
          .from('companies')
          .select('*')
          .eq('id', user.id)
          .single();
        
        if (company) {
          setCompanyProfile(company);
          setHasCompletedOnboarding(company.onboarding_completed || false);
        }
      }
    } catch (error) {
      console.error('Error loading user context:', error);
    }
  };

  const handleSignOut = async () => {
    try {
      await supabase.auth.signOut();
      if (onSignOut) {
        onSignOut();
      }
      navigate('/');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  const navItems = [
    {
      path: '/dashboard',
      label: 'Dashboard',
      icon: Target,
      description: 'Your main dashboard'
    },
    {
      path: '/marketplace',
      label: 'Marketplace',
      icon: Users,
      description: 'Browse materials and partners'
    },
    {
      path: '/ai-inference-matching',
      label: 'AI Matching',
      icon: Brain,
      description: 'AI-powered material matching'
    },
    {
      path: '/material-listings',
      label: 'Material Listings',
      icon: FileText,
      description: 'Manage your materials'
    }
  ];

  return (
    <>
      {/* Top Navigation Bar (Green) */}
      <header className="bg-gradient-to-r from-emerald-600 to-emerald-700 shadow-lg border-b border-emerald-500">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/')}
                className="flex items-center space-x-2 hover:opacity-80 transition-opacity"
              >
                <Workflow className="h-8 w-8 text-white" />
                <span className="text-2xl font-bold text-white">SymbioFlows</span>
              </button>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-6">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.path}
                    onClick={() => navigate(item.path)}
                    className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-all duration-200 whitespace-nowrap ${
                      isActive(item.path)
                        ? 'bg-white/20 text-white shadow-lg'
                        : 'text-white/80 hover:text-white hover:bg-white/10'
                    }`}
                    title={item.description}
                  >
                    <Icon className="w-4 h-4" />
                    <span className="font-medium">{item.label}</span>
                  </button>
                );
              })}
            </div>

            <div className="flex items-center space-x-4">
              <button className="text-white hover:text-emerald-200 transition p-2 rounded-lg hover:bg-white/10">
                <Bell className="w-5 h-5" />
              </button>
              <button className="text-white hover:text-emerald-200 transition p-2 rounded-lg hover:bg-white/10">
                <Settings className="w-5 h-5" />
              </button>
              <button 
                onClick={handleSignOut}
                className="text-white hover:text-emerald-200 transition p-2 rounded-lg hover:bg-white/10"
              >
                <LogOut className="w-5 h-5" />
              </button>
              {/* Mobile menu button */}
              <button
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                className="md:hidden text-white hover:text-emerald-200 transition p-2 rounded-lg hover:bg-white/10"
              >
                {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Mobile Menu (for green bar only) */}
      {isMobileMenuOpen && (
        <div className="md:hidden bg-emerald-700 border-b border-emerald-600">
          <div className="container mx-auto px-6 py-4">
            <div className="space-y-2">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.path}
                    onClick={() => {
                      navigate(item.path);
                      setIsMobileMenuOpen(false);
                    }}
                    className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                      isActive(item.path)
                        ? 'bg-white/20 text-white'
                        : 'text-white/80 hover:text-white hover:bg-white/10'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span className="font-medium">{item.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default PerfectNavigation; 