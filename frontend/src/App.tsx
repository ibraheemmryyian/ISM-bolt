import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { NotificationProvider } from './lib/notificationContext';
import ErrorBoundary from './components/ErrorBoundary';
import AuthModal from './components/AuthModal';
import AuthCallback from './components/AuthCallback';
import AuthenticatedLayout from './components/AuthenticatedLayout';
import Dashboard from './components/Dashboard';
import DemoDashboard from './components/DemoDashboard';
import DemoLandingPage from './components/DemoLandingPage';
import DemoAIMatching from './components/DemoAIMatching';
import AIInferenceMatching from './components/AIInferenceMatching';
import Marketplace from './components/Marketplace';
import AIOnboardingWizard from './components/AIOnboardingWizard';
import AdaptiveAIOnboarding from './components/AdaptiveAIOnboarding';
import AIAdvancedDashboard from './components/AIAdvancedDashboard';
import GnnPlayground from './components/GnnPlayground';
import GnnMatchesPanel from './components/GnnMatchesPanel';
import GnnSymbiosisPanel from './components/GnnSymbiosisPanel';
import MultiHopSymbiosisPanel from './components/MultiHopSymbiosisPanel';
import ProactiveOpportunitiesPanel from './components/ProactiveOpportunitiesPanel';
import GlobalImpactPanel from './components/GlobalImpactPanel';
import FinancialAnalysisPanel from './components/FinancialAnalysisPanel';
import LogisticsPanel from './components/LogisticsPanel';
import ComprehensiveMatchAnalysis from './components/ComprehensiveMatchAnalysis';
import RevolutionaryAIMatching from './components/RevolutionaryAIMatching';
import PersonalPortfolio from './components/PersonalPortfolio';
import AdminHub from './components/AdminHub';
import AdminAccessPage from './components/AdminAccessPage';
import RequestAccess from './components/RequestAccess';
import ResetPassword from './components/ResetPassword';
import SubscriptionManager from './components/SubscriptionManager';
import PaymentProcessor from './components/PaymentProcessor';
import TransactionPage from './components/TransactionPage';
import HeightProjectTracker from './components/HeightProjectTracker';
import GreenInitiatives from './components/GreenInitiatives';
import FuturePlans from './components/FuturePlans';
import GlobalMap from './components/GlobalMap';
import ChatInterface from './components/ChatInterface';
import NotificationToast from './components/NotificationToast';
import UserFeedbackModal from './components/UserFeedbackModal';
import AIExplanationModal from './components/AIExplanationModal';
import AIComprehensiveOnboarding from './components/AIComprehensiveOnboarding';
import AIPreviewDashboard from './components/AIPreviewDashboard';
import AIBackendInterface from './components/AIBackendInterface';
import ReviewAIListings from './components/ReviewAIListings';
import DebugAdmin from './components/DebugAdmin';
import PluginManager from './components/PluginManager';
import SubscriptionUpgradeModal from './components/SubscriptionUpgradeModal';
import DetailedCostBreakdown from './components/DetailedCostBreakdown';
import EnhancedMatchingInterface from './components/EnhancedMatchingInterface';
import EnhancedPortfolioReview from './components/EnhancedPortfolioReview';
import ScientificMaterialCard from './components/ScientificMaterialCard';
import SearchAndFilter from './components/SearchAndFilter';
import ShippingCalculator from './components/ShippingCalculator';
import SymbiosisNetworkGraph from './components/SymbiosisNetworkGraph';
import NotificationsPanel from './components/NotificationsPanel';
import ChatsPanel from './components/ChatsPanel';
import MaterialForm from './components/MaterialForm';
import OnboardingForm from './components/OnboardingForm';
import OnboardingWizard from './components/OnboardingWizard';
import ProgressIndicator from './components/ProgressIndicator';
import FormValidation from './components/FormValidation';
import LoadingSkeleton from './components/LoadingSkeleton';
import Tooltip from './components/Tooltip';
import Breadcrumbs from './components/Breadcrumbs';
import KeyboardShortcutsHelp from './components/KeyboardShortcutsHelp';
import RoleInfo from './components/RoleInfo';

function App() {
  return (
    <ErrorBoundary>
      <NotificationProvider>
        <Router>
          <div className="App">
            <Toaster 
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#363636',
                  color: '#fff',
                },
                success: {
                  duration: 3000,
                  iconTheme: {
                    primary: '#10b981',
                    secondary: '#fff',
                  },
                },
                error: {
                  duration: 5000,
                  iconTheme: {
                    primary: '#ef4444',
                    secondary: '#fff',
                  },
                },
              }}
            />
            
            <Routes>
              {/* Demo Landing Page */}
              <Route path="/" element={<DemoLandingPage />} />
              
              {/* Demo Routes */}
              <Route path="/demo" element={<DemoLandingPage />} />
              <Route path="/demo/dashboard" element={<DemoDashboard />} />
              <Route path="/demo/ai-matching" element={<DemoAIMatching />} />
              
              {/* Auth Routes */}
              <Route path="/auth/callback" element={<AuthCallback />} />
              <Route path="/auth/reset-password" element={<ResetPassword />} />
              <Route path="/request-access" element={<RequestAccess />} />
              
              {/* Main App Routes */}
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/ai-inference-matching" element={<AIInferenceMatching />} />
              <Route path="/marketplace" element={<Marketplace />} />
              <Route path="/onboarding" element={<AIOnboardingWizard />} />
              <Route path="/adaptive-onboarding" element={<AdaptiveAIOnboarding />} />
              <Route path="/ai-advanced" element={<AIAdvancedDashboard />} />
              <Route path="/gnn-playground" element={<GnnPlayground />} />
              <Route path="/gnn-matches" element={<GnnMatchesPanel />} />
              <Route path="/gnn-symbiosis" element={<GnnSymbiosisPanel />} />
              <Route path="/multi-hop-symbiosis" element={<MultiHopSymbiosisPanel />} />
              <Route path="/proactive-opportunities" element={<ProactiveOpportunitiesPanel />} />
              <Route path="/global-impact" element={<GlobalImpactPanel />} />
              <Route path="/financial-analysis" element={<FinancialAnalysisPanel />} />
              <Route path="/logistics" element={<LogisticsPanel />} />
              <Route path="/comprehensive-analysis" element={<ComprehensiveMatchAnalysis />} />
              <Route path="/revolutionary-ai" element={<RevolutionaryAIMatching />} />
              <Route path="/portfolio" element={<PersonalPortfolio />} />
              <Route path="/admin" element={<AdminHub />} />
              <Route path="/admin-access" element={<AdminAccessPage />} />
              <Route path="/subscription" element={<SubscriptionManager />} />
              <Route path="/payment" element={<PaymentProcessor />} />
              <Route path="/transactions" element={<TransactionPage />} />
              <Route path="/height-tracker" element={<HeightProjectTracker />} />
              <Route path="/green-initiatives" element={<GreenInitiatives />} />
              <Route path="/future-plans" element={<FuturePlans />} />
              <Route path="/global-map" element={<GlobalMap />} />
              <Route path="/chat" element={<ChatInterface />} />
              <Route path="/ai-comprehensive" element={<AIComprehensiveOnboarding />} />
              <Route path="/ai-preview" element={<AIPreviewDashboard />} />
              <Route path="/ai-backend" element={<AIBackendInterface />} />
              <Route path="/review-listings" element={<ReviewAIListings />} />
              <Route path="/debug-admin" element={<DebugAdmin />} />
              <Route path="/plugins" element={<PluginManager />} />
              <Route path="/cost-breakdown" element={<DetailedCostBreakdown />} />
              <Route path="/enhanced-matching" element={<EnhancedMatchingInterface />} />
              <Route path="/enhanced-portfolio" element={<EnhancedPortfolioReview />} />
              <Route path="/scientific-materials" element={<ScientificMaterialCard />} />
              <Route path="/search-filter" element={<SearchAndFilter />} />
              <Route path="/shipping-calculator" element={<ShippingCalculator />} />
              <Route path="/symbiosis-network" element={<SymbiosisNetworkGraph />} />
              <Route path="/notifications" element={<NotificationsPanel />} />
              <Route path="/chats" element={<ChatsPanel />} />
              <Route path="/material-form" element={<MaterialForm />} />
              <Route path="/onboarding-form" element={<OnboardingForm />} />
              <Route path="/onboarding-wizard" element={<OnboardingWizard />} />
              <Route path="/progress" element={<ProgressIndicator />} />
              <Route path="/form-validation" element={<FormValidation />} />
              <Route path="/loading" element={<LoadingSkeleton />} />
              <Route path="/tooltip" element={<Tooltip />} />
              <Route path="/breadcrumbs" element={<Breadcrumbs />} />
              <Route path="/keyboard-help" element={<KeyboardShortcutsHelp />} />
              <Route path="/role-info" element={<RoleInfo />} />
              
              {/* Fallback */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
            
            {/* Global Modals */}
            <AuthModal />
            <NotificationToast />
            <UserFeedbackModal />
            <AIExplanationModal />
            <SubscriptionUpgradeModal />
          </div>
        </Router>
      </NotificationProvider>
    </ErrorBoundary>
  );
}

export default App;
