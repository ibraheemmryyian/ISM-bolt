import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { 
  Workflow,
  Home,
  ArrowLeft,
  Bell,
  Settings,
  LogOut,
  Users,
  Target,
  Leaf,
  Sparkles,
  Brain
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';

interface AuthenticatedLayoutProps {
  children: React.ReactNode;
  title?: string;
  onSignOut?: () => void;
}

interface CompanyProfile {
  name: string;
  industry: string;
  location: string;
  employee_count: number;
}

const AuthenticatedLayout: React.FC<AuthenticatedLayoutProps> = ({ 
  children, 
  title = "Dashboard",
  onSignOut 
}) => {
  const [companyProfile, setCompanyProfile] = useState<CompanyProfile | null>(null);
  const [hasCompletedOnboarding, setHasCompletedOnboarding] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    loadUserContext();
  }, []);

  const loadUserContext = () => {
    // Load company profile from localStorage (from onboarding)
    const storedCompanyProfile = localStorage.getItem('symbioflows-company-profile');
    const storedPortfolio = localStorage.getItem('symbioflows-portfolio');
    
    if (storedCompanyProfile) {
      try {
        const profile = JSON.parse(storedCompanyProfile);
        setCompanyProfile(profile);
      } catch (error) {
        console.error('Error parsing company profile:', error);
      }
    }
    
    setHasCompletedOnboarding(!!(storedPortfolio && storedCompanyProfile));
  };

  const handleSignOut = async () => {
    try {
      await supabase.auth.signOut();
      if (onSignOut) {
        onSignOut();
      }
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const getCompanyName = () => {
    if (companyProfile?.name && companyProfile.name.trim() !== '') {
      return companyProfile.name.trim();
    }
    return 'Your Company';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      {/* Header with Navigation */}
      <header className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/')}
                className="flex items-center space-x-2 hover:opacity-80 transition-opacity"
              >
                <Workflow className="h-8 w-8 text-emerald-400" />
                <span className="text-2xl font-bold text-white">SymbioFlows</span>
              </button>
              
              <div className="hidden md:flex items-center space-x-2 text-gray-300">
                <span>/</span>
                <span className="text-emerald-400 font-medium">{title}</span>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <button className="text-gray-300 hover:text-white transition">
                <Bell className="w-5 h-5" />
              </button>
              <button className="text-gray-300 hover:text-white transition">
                <Settings className="w-5 h-5" />
              </button>
              <button 
                onClick={handleSignOut}
                className="text-gray-300 hover:text-white transition"
              >
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </header>



      {/* Quick Navigation */}
      <div className="bg-slate-800/30 border-b border-slate-700">
        <div className="container mx-auto px-6 py-3">
          <div className="flex items-center space-x-4 overflow-x-auto">
            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => navigate('/dashboard')}
              className="text-gray-300 hover:text-white hover:bg-slate-700"
            >
              <Target className="w-4 h-4 mr-2" />
              Dashboard
            </Button>
            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => navigate('/marketplace')}
              className="text-gray-300 hover:text-white hover:bg-slate-700"
            >
              <Users className="w-4 h-4 mr-2" />
              Marketplace
            </Button>

            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => navigate('/ai-inference-matching')}
              className="text-gray-300 hover:text-white hover:bg-slate-700"
            >
              <Brain className="w-4 h-4 mr-2" />
              AI Inference & Matching
            </Button>
            {hasCompletedOnboarding ? (
              <Button 
                variant="ghost" 
                size="sm"
                onClick={() => navigate('/onboarding')}
                className="text-gray-300 hover:text-white hover:bg-slate-700"
              >
                <Brain className="w-4 h-4 mr-2" />
                Manage AI Profile
              </Button>
            ) : (
              <Button 
                variant="ghost" 
                size="sm"
                onClick={() => navigate('/onboarding')}
                className="text-gray-300 hover:text-white hover:bg-slate-700"
              >
                <Sparkles className="w-4 h-4 mr-2" />
                AI Onboarding
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="container mx-auto p-6">
        {children}
      </main>
    </div>
  );
};

export default AuthenticatedLayout; 