import React, { useState, useEffect } from 'react';
import { Factory, Recycle, Search, TrendingUp, Users, Workflow } from 'lucide-react';
import { supabase } from './lib/supabase';
import { AuthModal } from './components/AuthModal';
import { MaterialForm } from './components/MaterialForm';
import { AdminHub } from './components/AdminHub';
import { OnboardingForm } from './components/OnboardingForm';
import { GlobalMap } from './components/GlobalMap';
import { RoleInfo } from './components/RoleInfo';
import { Dashboard } from './components/Dashboard';
import { Marketplace } from './components/Marketplace';
import { isUserAdmin } from './lib/supabase';

type CurrentView = 'home' | 'dashboard' | 'marketplace' | 'admin';

function App() {
  const [session, setSession] = useState(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showMaterialForm, setShowMaterialForm] = useState<'waste' | 'requirement' | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [currentView, setCurrentView] = useState<CurrentView>('home');

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      if (session?.user) {
        checkAdminStatus(session.user.id);
      }
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      if (session?.user) {
        checkAdminStatus(session.user.id);
      } else {
        setIsAdmin(false);
        setCurrentView('home');
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  const checkAdminStatus = async (userId: string) => {
    const adminStatus = await isUserAdmin(userId);
    setIsAdmin(adminStatus);
  };

  const handleAction = () => {
    if (!session) {
      setShowAuthModal(true);
    } else if (isAdmin) {
      setCurrentView('admin');
    } else {
      setCurrentView('dashboard');
    }
  };

  const handleListMaterials = () => {
    if (session) {
      setShowMaterialForm('waste');
    } else {
      setShowAuthModal(true);
    }
  };

  const handleFindResources = () => {
    if (session) {
      setShowMaterialForm('requirement');
    } else {
      setShowAuthModal(true);
    }
  };

  const handleNavigation = (view: CurrentView) => {
    if (view === 'dashboard' || view === 'marketplace') {
      if (!session) {
        setShowAuthModal(true);
        return;
      }
    }
    setCurrentView(view);
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    setCurrentView('home');
  };

  // Render different views based on current state
  if (currentView === 'admin' && isAdmin) {
    return (
      <div>
        <nav className="bg-slate-900 p-4">
          <div className="container mx-auto flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <Workflow className="h-8 w-8 text-emerald-400" />
              <span className="text-2xl font-bold text-white">SymbioFlow</span>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setCurrentView('home')}
                className="text-white hover:text-emerald-400 transition"
              >
                Back to Main Site
              </button>
              <button
                onClick={handleSignOut}
                className="text-white hover:text-emerald-400 transition"
              >
                Sign Out
              </button>
            </div>
          </div>
        </nav>
        <AdminHub />
      </div>
    );
  }

  if (currentView === 'dashboard' && session) {
    return <Dashboard onNavigate={handleNavigation} onSignOut={handleSignOut} />;
  }

  if (currentView === 'marketplace' && session) {
    return <Marketplace onNavigate={handleNavigation} onSignOut={handleSignOut} />;
  }

  // Home page
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800">
      {/* Hero Section */}
      <header className="relative">
        <div className="absolute inset-0 z-0">
          <img 
            src="https://images.unsplash.com/photo-1590502160462-58b41354f588?auto=format&fit=crop&q=80"
            alt="Industrial background"
            className="w-full h-[600px] object-cover opacity-20"
          />
        </div>
        
        <nav className="relative z-10 container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Workflow className="h-8 w-8 text-emerald-400" />
              <span className="text-2xl font-bold text-white">SymbioFlow</span>
            </div>
            <div className="hidden md:flex space-x-8">
              <a href="#how-it-works" className="text-gray-300 hover:text-white transition">How It Works</a>
              <a href="#map" className="text-gray-300 hover:text-white transition">Global Impact</a>
              <a href="#roles" className="text-gray-300 hover:text-white transition">How to Help</a>
              {session && (
                <>
                  <button
                    onClick={() => handleNavigation('dashboard')}
                    className="text-gray-300 hover:text-white transition"
                  >
                    Dashboard
                  </button>
                  <button
                    onClick={() => handleNavigation('marketplace')}
                    className="text-gray-300 hover:text-white transition"
                  >
                    Marketplace
                  </button>
                </>
              )}
            </div>
            <div className="flex items-center space-x-4">
              {session ? (
                <>
                  <button 
                    onClick={() => handleNavigation('dashboard')}
                    className="bg-emerald-500 text-white px-6 py-2 rounded-lg hover:bg-emerald-600 transition"
                  >
                    {isAdmin ? 'Admin Dashboard' : 'My Dashboard'}
                  </button>
                  <button
                    onClick={handleSignOut}
                    className="text-white hover:text-emerald-400 transition"
                  >
                    Sign Out
                  </button>
                </>
              ) : (
                <button 
                  onClick={handleAction}
                  className="bg-emerald-500 text-white px-6 py-2 rounded-lg hover:bg-emerald-600 transition"
                >
                  Get Started
                </button>
              )}
            </div>
          </div>
        </nav>

        <div className="relative z-10 container mx-auto px-6 py-24 text-center">
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
            Connecting the Circular Economy
          </h1>
          <p className="text-xl text-gray-300 mb-12 max-w-3xl mx-auto">
            Join our global network of researchers, industries, and innovators building a sustainable future through resource optimization and waste reduction.
          </p>
          <div className="flex flex-col md:flex-row gap-4 justify-center">
            <button 
              onClick={handleAction}
              className="bg-emerald-500 text-white px-8 py-4 rounded-lg hover:bg-emerald-600 transition flex items-center justify-center space-x-2"
            >
              <Users className="h-5 w-5" />
              <span>Get Started</span>
            </button>
            <button 
              onClick={() => handleNavigation('marketplace')}
              className="bg-slate-700 text-white px-8 py-4 rounded-lg hover:bg-slate-600 transition flex items-center justify-center space-x-2"
            >
              <Factory className="h-5 w-5" />
              <span>Browse Marketplace</span>
            </button>
          </div>
        </div>
      </header>

      {/* Features Section */}
      <section id="how-it-works" className="py-24 bg-slate-800">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-white text-center mb-16">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-12">
            <div className="bg-slate-700 p-8 rounded-xl">
              <div className="bg-emerald-500/10 p-3 rounded-lg w-fit mb-6">
                <Factory className="h-6 w-6 text-emerald-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">List Resources</h3>
              <p className="text-gray-300">
                Register your organization and share your resource needs or available materials.
              </p>
            </div>
            <div className="bg-slate-700 p-8 rounded-xl">
              <div className="bg-emerald-500/10 p-3 rounded-lg w-fit mb-6">
                <Recycle className="h-6 w-6 text-emerald-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">AI Matching</h3>
              <p className="text-gray-300">
                Our AI system connects you with compatible partners based on your needs and capabilities.
              </p>
            </div>
            <div className="bg-slate-700 p-8 rounded-xl">
              <div className="bg-emerald-500/10 p-3 rounded-lg w-fit mb-6">
                <TrendingUp className="h-6 w-6 text-emerald-400" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Create Value</h3>
              <p className="text-gray-300">
                Transform waste into resources and build sustainable partnerships.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Global Map Section */}
      <section id="map">
        <GlobalMap />
      </section>

      {/* Role Information Section */}
      <section id="roles">
        <RoleInfo />
      </section>

      {/* Stats Section */}
      <section className="py-24 bg-slate-900">
        <div className="container mx-auto px-6">
          <div className="grid md:grid-cols-3 gap-12 text-center">
            <div>
              <div className="text-4xl font-bold text-emerald-400 mb-2">500+</div>
              <div className="text-gray-300">Active Organizations</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-emerald-400 mb-2">1,200+</div>
              <div className="text-gray-300">Successful Matches</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-emerald-400 mb-2">50,000t</div>
              <div className="text-gray-300">Waste Reduced</div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-slate-800">
        <div className="container mx-auto px-6 text-center">
          <div className="bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-2xl p-12">
            <h2 className="text-3xl font-bold text-white mb-6">
              Ready to Join the Circular Economy?
            </h2>
            <p className="text-white/90 mb-8 max-w-2xl mx-auto">
              Start your journey towards sustainable business practices and resource efficiency today.
            </p>
            <button 
              onClick={handleAction}
              className="bg-white text-emerald-600 px-8 py-4 rounded-lg hover:bg-gray-100 transition flex items-center space-x-2 mx-auto"
            >
              <Users className="h-5 w-5" />
              <span>Join the Network</span>
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 py-12">
        <div className="container mx-auto px-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Workflow className="h-6 w-6 text-emerald-400" />
              <span className="text-xl font-bold text-white">SymbioFlow</span>
            </div>
            <div className="text-gray-400 text-sm">
              © 2025 SymbioFlow. All rights reserved.
            </div>
          </div>
        </div>
      </footer>

      {/* Modals */}
      {showAuthModal && (
        <AuthModal onClose={() => setShowAuthModal(false)} />
      )}
      {showMaterialForm && (
        <MaterialForm 
          type={showMaterialForm} 
          onClose={() => setShowMaterialForm(null)} 
        />
      )}
      {showOnboarding && (
        <OnboardingForm onClose={() => setShowOnboarding(false)} />
      )}
    </div>
  );
}

export default App;
