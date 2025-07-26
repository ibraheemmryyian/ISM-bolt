import React, { useState, useEffect } from 'react';
import { Factory, Recycle, Users, Workflow } from 'lucide-react';
import { supabase } from './lib/supabase';
import { AuthModal } from './components/AuthModal';
import { MaterialForm } from './components/MaterialForm';
import { AdminHub } from './components/AdminHub';
import { OnboardingForm } from './components/OnboardingForm';
import { GlobalMap } from './components/GlobalMap';
import { RoleInfo } from './components/RoleInfo';
import PersonalPortfolio from './components/PersonalPortfolio';
import { Marketplace } from './components/Marketplace';
import { isUserAdmin } from './lib/supabase';
import { TransactionPage } from './components/TransactionPage';
import { Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import AIOnboardingWizard from './components/AIOnboardingWizard';
import { NotificationsPanel } from './components/NotificationsPanel';
import { ChatsPanel } from './components/ChatsPanel';
import { AdminAccessPage } from './components/AdminAccessPage';
import { ReviewAIListings } from './components/ReviewAIListings';
import { AdaptiveAIOnboarding } from './components/AdaptiveAIOnboarding';

import ErrorBoundary from './components/ErrorBoundary';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { NotificationProvider } from './lib/notificationContext';

function LandingPage({ onGetStarted, onMarketplace, session, handleSignOut }: any) {
  const navigate = useNavigate();
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
              <span className="text-2xl font-bold text-white">SymbioFlows</span>
            </div>
            <div className="hidden md:flex space-x-8">
              <a href="#how-it-works" className="text-gray-300 hover:text-white transition">How It Works</a>
              <a href="#map" className="text-gray-300 hover:text-white transition">Global Impact</a>
              <a href="#roles" className="text-gray-300 hover:text-white transition">How to Help</a>
              {session && (
                <>
                  <button
                    onClick={() => navigate('/dashboard')}
                    className="text-gray-300 hover:text-white transition"
                  >
                    Dashboard
                  </button>
                  <button
                    onClick={() => navigate('/marketplace')}
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
                    onClick={() => navigate('/dashboard')}
                    className="bg-emerald-500 text-white px-6 py-2 rounded-lg hover:bg-emerald-600 transition"
                  >
                    My Dashboard
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
                  onClick={onGetStarted}
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
            {session ? (
              <button 
                onClick={() => navigate('/dashboard')}
                className="bg-emerald-500 text-white px-8 py-4 rounded-lg hover:bg-emerald-600 transition flex items-center justify-center space-x-2"
              >
                <Users className="h-5 w-5" />
                <span>Go to Dashboard</span>
              </button>
            ) : (
            <button 
                onClick={onGetStarted}
              className="bg-emerald-500 text-white px-8 py-4 rounded-lg hover:bg-emerald-600 transition flex items-center justify-center space-x-2"
            >
              <Users className="h-5 w-5" />
              <span>Get Started</span>
            </button>
            )}
            <button 
              onClick={onMarketplace}
              className="bg-slate-700 text-white px-8 py-4 rounded-lg hover:bg-slate-600 transition flex items-center justify-center space-x-2"
            >
              <Factory className="h-5 w-5" />
              <span>Browse Marketplace</span>
            </button>
          </div>
        </div>
      </header>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 bg-slate-800">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-white mb-8 text-center">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-slate-700 rounded-xl p-6 shadow-sm flex flex-col items-center">
              <Users className="h-10 w-10 text-emerald-400 mb-4" />
              <h3 className="font-semibold text-lg text-white mb-2">Join & Onboard</h3>
              <p className="text-slate-200 text-center">Sign up, answer a few questions about your company, and let our AI infer your waste, byproducts, and needs.</p>
            </div>
            <div className="bg-slate-700 rounded-xl p-6 shadow-sm flex flex-col items-center">
              <Factory className="h-10 w-10 text-emerald-400 mb-4" />
              <h3 className="font-semibold text-lg text-white mb-2">AI-Driven Matching</h3>
              <p className="text-slate-200 text-center">Our AI continuously matches your outputs and needs with the global marketplace—no manual listing required.</p>
            </div>
            <div className="bg-slate-700 rounded-xl p-6 shadow-sm flex flex-col items-center">
              <Recycle className="h-10 w-10 text-emerald-400 mb-4" />
              <h3 className="font-semibold text-lg text-white mb-2">Connect & Transact</h3>
              <p className="text-slate-200 text-center">Get notified of matches, chat with partners, and close the loop with seamless transactions and logistics.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Global Impact Section */}
      <section id="map" className="py-20 bg-slate-900">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-white mb-8 text-center">Global Impact</h2>
          <GlobalMap />
        </div>
      </section>

      {/* How to Help Section */}
      <section id="roles" className="py-20 bg-slate-800">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl font-bold text-white mb-8 text-center">How to Help</h2>
          <RoleInfo />
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 bg-slate-900 text-center text-gray-400 text-sm">
        &copy; {new Date().getFullYear()} SymbioFlows. All rights reserved.
      </footer>
    </div>
  );
}

function App() {
  const [session, setSession] = useState<any>(null);
  const [sessionChecked, setSessionChecked] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showMaterialForm, setShowMaterialForm] = useState<'waste' | 'requirement' | null>(null);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    // Mark that user has seen the landing page
    localStorage.setItem('symbioflows-landing-seen', 'true');

    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setSessionChecked(true);
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      setSessionChecked(true);
    });

    return () => subscription.unsubscribe();
  }, []);

  // Dynamic document title based on route
  useEffect(() => {
    const path = location.pathname;
    let title = 'SymbioFlows - Industrial Symbiosis Platform';
    if (path === '/' || path === '/home') title = 'Home | SymbioFlows';
    else if (path.startsWith('/dashboard') || path.startsWith('/portfolio')) title = 'Personal Portfolio | SymbioFlows';
    else if (path.startsWith('/marketplace')) title = 'Marketplace | SymbioFlows';
    else if (path.startsWith('/admin')) title = 'Admin Hub | SymbioFlows';
    else if (path.startsWith('/onboarding') || path.startsWith('/adaptive-onboarding')) title = 'AI Onboarding | SymbioFlows';
    else if (path.startsWith('/notifications')) title = 'Notifications | SymbioFlows';
    else if (path.startsWith('/chats')) title = 'Chats | SymbioFlows';
    else if (path.startsWith('/transaction')) title = 'Transaction | SymbioFlows';
    else if (path.startsWith('/review-ai-listings')) title = 'Review AI Listings | SymbioFlows';
    document.title = title;
  }, [location.pathname]);

  if (!sessionChecked) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  const handleSignOut = async () => {
    try {
      await supabase.auth.signOut();
      setSession(null);
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  return (
    <ErrorBoundary>
      <NotificationProvider>
        <div className="App">
          <Routes>
            <Route path="/" element={
              <LandingPage 
                onGetStarted={() => setShowAuthModal(true)}
                onMarketplace={() => navigate('/marketplace')}
                session={session}
                handleSignOut={handleSignOut}
              />
            } />
            <Route path="/dashboard" element={
              session ? (
                <PersonalPortfolio />
              ) : (
                <Navigate to="/" replace />
              )
            } />
            {/* Optionally keep /portfolio as an alias, or remove if not needed */}
            {/* <Route path="/portfolio" element={
              session ? (
                <PersonalPortfolio />
              ) : (
                <Navigate to="/" replace />
              )
            } /> */}
            <Route path="/marketplace" element={
              session ? (
                <Marketplace onSignOut={handleSignOut} />
              ) : (
                <Navigate to="/" replace />
              )
            } />
            <Route path="/admin" element={<AdminAccessPage />} />
            <Route path="/onboarding" element={
              session ? (
                <AdaptiveAIOnboarding
                  onClose={() => navigate('/dashboard')}
                  onComplete={() => navigate('/dashboard')}
                />
              ) : (
                <Navigate to="/" replace />
              )
            } />
            <Route path="/adaptive-onboarding" element={
              session ? (
                <AdaptiveAIOnboarding
                  onClose={() => navigate('/dashboard')}
                  onComplete={() => navigate('/dashboard')}
                />
              ) : (
                <Navigate to="/" replace />
              )
            } />
            <Route path="/notifications" element={
              session ? (
                <NotificationsPanel companyId={session.user?.id || ''} />
              ) : (
                <Navigate to="/" replace />
              )
            } />
            <Route path="/chats" element={
              session ? (
                <ChatsPanel />
              ) : (
                <Navigate to="/" replace />
              )
            } />
            <Route path="/transaction/:transactionId" element={
              session ? (
                <TransactionPage />
              ) : (
                <Navigate to="/" replace />
              )
            } />
            <Route path="/review-ai-listings" element={
              session ? (
                <ReviewAIListings onConfirm={() => {
                  navigate('/dashboard');
                }} />
              ) : (
                <Navigate to="/" replace />
              )
            } />
          </Routes>
          {/* Auth Modal */}
          {showAuthModal && (
            <AuthModal 
              onClose={() => setShowAuthModal(false)}
            />
          )}
          {/* Material Form Modal */}
          {showMaterialForm && (
            <MaterialForm 
              type={showMaterialForm}
              onClose={() => setShowMaterialForm(null)}
            />
          )}
          {/* Remove legacy onboarding modal */}
          {/* Global Toast Container */}
          <ToastContainer
            position="top-right"
            autoClose={5000}
            hideProgressBar={false}
            newestOnTop={false}
            closeOnClick
            rtl={false}
            pauseOnFocusLoss
            draggable
            pauseOnHover
            theme="light"
          />
        </div>
      </NotificationProvider>
    </ErrorBoundary>
  );
}

export default App;