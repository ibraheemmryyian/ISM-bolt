import React, { useState, useEffect } from 'react';
import { Factory, Recycle, Users, Workflow, Home } from 'lucide-react';
import { supabase } from './lib/supabase';
import { AuthModal } from './components/AuthModal';
import { MaterialForm } from './components/MaterialForm';
import { AdminHub } from './components/AdminHub';
import { OnboardingForm } from './components/OnboardingForm';
import { GlobalMap } from './components/GlobalMap';
import { RoleInfo } from './components/RoleInfo';
import Dashboard from './components/Dashboard';
import { Marketplace } from './components/Marketplace';
import { isUserAdmin } from './lib/supabase';
import { TransactionPage } from './components/TransactionPage';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import AIOnboardingWizard from './components/AIOnboardingWizard';
import GreenInitiatives from './components/GreenInitiatives';
import PersonalPortfolio from './components/PersonalPortfolio';
import PerfectDashboard from './components/PerfectDashboard';
import { NotificationsPanel } from './components/NotificationsPanel';
import { ChatsPanel } from './components/ChatsPanel';
import { AdminAccessPage } from './components/AdminAccessPage';
import { ReviewAIListings } from './components/ReviewAIListings';
import ErrorBoundary from './components/ErrorBoundary';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { NotificationProvider } from './lib/notificationContext';
import { ThemeProvider } from './contexts/ThemeContext';
import PerfectNavigation from './components/PerfectNavigation';

// Import new comprehensive analysis components
import EnhancedMatchingInterface from './components/EnhancedMatchingInterface';
import ComprehensiveMatchAnalysis from './components/ComprehensiveMatchAnalysis';
import DetailedCostBreakdown from './components/DetailedCostBreakdown';

function LandingPage({ onGetStarted, onMarketplace, session, isAdmin, handleSignOut }: any) {
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
                    {(() => {
                      console.log('isAdmin state:', isAdmin);
                      return isAdmin ? 'Admin Dashboard' : 'My Dashboard';
                    })()}
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
              <h3 className="font-semibold text-lg text-white mb-2">AI-Powered Onboarding</h3>
              <p className="text-slate-200 text-center">Our AI assistant guides you through a simple conversation to understand your company's needs and opportunities.</p>
            </div>
            <div className="bg-slate-700 rounded-xl p-6 shadow-sm flex flex-col items-center">
              <Factory className="h-10 w-10 text-emerald-400 mb-4" />
              <h3 className="font-semibold text-lg text-white mb-2">Smart Matching</h3>
              <p className="text-slate-200 text-center">Our AI continuously finds the best matches for your waste streams and resource needs with other companies.</p>
            </div>
            <div className="bg-slate-700 rounded-xl p-6 shadow-sm flex flex-col items-center">
              <Recycle className="h-10 w-10 text-emerald-400 mb-4" />
              <h3 className="font-semibold text-lg text-white mb-2">Grow & Save</h3>
              <p className="text-slate-200 text-center">Connect with partners, reduce costs, improve sustainability, and track your progress with personalized insights.</p>
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

// Create a component that uses useLocation inside Router context
function AppContent({ session, isAdmin, handleSignOut, setShowAuthModal }: any) {
  const location = useLocation();
  
  return (
    <>
      {/* Only show PerfectNavigation if not on the root (Home) page */}
      {location.pathname !== '/' && (
        <PerfectNavigation onSignOut={handleSignOut} />
      )}
      <div className="App">
        <Routes>
          <Route path="/" element={
            <LandingPage 
              onGetStarted={() => setShowAuthModal(true)}
              onMarketplace={() => window.location.href = '/marketplace'}
              session={session}
              isAdmin={isAdmin}
              handleSignOut={handleSignOut}
            />
          } />
          <Route path="/dashboard" element={
            session ? (
              <PerfectDashboard />
            ) : (
              <Navigate to="/" replace />
            )
          } />
          <Route path="/marketplace" element={
            session ? (
              <Marketplace onSignOut={handleSignOut} />
            ) : (
              <Navigate to="/" replace />
            )
          } />
          <Route path="/admin" element={
            isAdmin ? (
              <AdminHub />
            ) : (
              <AdminAccessPage />
            )
          } />
          <Route path="/onboarding" element={
            session ? (
              <AIOnboardingWizard 
                companyProfile={{
                  name: session.user?.user_metadata?.company_name || '',
                  industry: session.user?.user_metadata?.industry || '',
                  location: session.user?.user_metadata?.location || '',
                  products: '',
                  employee_count: 0
                }}
                onComplete={() => {}}
                onCancel={() => {}}
              />
            ) : (
              <Navigate to="/" replace />
            )
          } />
          <Route path="/green-initiatives" element={
            session ? (
              <GreenInitiatives />
            ) : (
              <Navigate to="/" replace />
            )
          } />
          <Route path="/portfolio" element={
            session ? (
              <PersonalPortfolio />
            ) : (
              <Navigate to="/" replace />
            )
          } />
          <Route path="/notifications" element={
            session ? (
              <NotificationsPanel companyId={session.user.id} />
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
                // Handle confirmation - navigate to dashboard
                window.location.href = '/dashboard';
              }} />
            ) : (
              <Navigate to="/" replace />
            )
          } />
          
          {/* New Comprehensive Analysis Routes */}
          <Route path="/enhanced-matching" element={
            session ? (
              <EnhancedMatchingInterface userId={session.user.id} />
            ) : (
              <Navigate to="/" replace />
            )
          } />
          
          <Route path="/comprehensive-analysis" element={
            session ? (
              <ComprehensiveMatchAnalysis 
                buyerData={{ id: session.user.id }}
                sellerData={{ id: 'demo-seller' }}
                matchData={{ id: 'demo-match' }}
              />
            ) : (
              <Navigate to="/" replace />
            )
          } />
          
          <Route path="/cost-breakdown" element={
            session ? (
              <DetailedCostBreakdown 
                buyerData={{ id: session.user.id }}
                sellerData={{ id: 'demo-seller' }}
                matchData={{ id: 'demo-match' }}
              />
            ) : (
              <Navigate to="/" replace />
            )
          } />
        </Routes>
      </div>
    </>
  );
}

function App() {
  const [session, setSession] = useState<any>(null);
  const [isAdmin, setIsAdmin] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showMaterialForm, setShowMaterialForm] = useState<'waste' | 'requirement' | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(false);

  useEffect(() => {
    // Check if user has seen the landing page
    const hasSeenLanding = localStorage.getItem('symbioflows-landing-seen');
    
    if (!hasSeenLanding) {
      // Redirect to investor landing page for first-time visitors
      window.location.href = '/investor.html';
      return;
    }

    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      if (session?.user) {
        checkAdminStatus(session.user.id);
      }
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      if (session?.user) {
        checkAdminStatus(session.user.id);
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  const checkAdminStatus = async (userId: string) => {
    try {
      const adminStatus = await isUserAdmin(userId);
      setIsAdmin(adminStatus);
    } catch (error) {
      console.error('Error checking admin status:', error);
    }
  };

  const handleSignOut = async () => {
    try {
      await supabase.auth.signOut();
      setSession(null);
      setIsAdmin(false);
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  return (
    <ErrorBoundary>
      <NotificationProvider>
        <ThemeProvider>
          <Router>
            <AppContent 
              session={session}
              isAdmin={isAdmin}
              handleSignOut={handleSignOut}
              setShowAuthModal={setShowAuthModal}
            />
            
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

            {/* Onboarding Modal */}
            {showOnboarding && (
              <OnboardingForm 
                onClose={() => setShowOnboarding(false)}
              />
            )}

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
          </Router>
        </ThemeProvider>
      </NotificationProvider>
    </ErrorBoundary>
  );
}

export default App;
