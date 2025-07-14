import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, ArrowRight } from 'lucide-react';

export function SimpleLandingPage() {
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is already authenticated
    const adminAccess = localStorage.getItem('temp-admin-access');
    if (adminAccess === 'true') {
      navigate('/dashboard');
    }
  }, [navigate]);

  const goToDashboard = () => {
    navigate('/dashboard');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 flex items-center justify-center">
      <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md text-center">
        <div className="mb-8">
          <Shield className="h-16 w-16 text-purple-500 mx-auto mb-4" />
          <h1 className="text-3xl font-bold text-gray-900 mb-2">SymbioFlows</h1>
          <p className="text-gray-600">Advanced AI-Powered Industrial Symbiosis Platform</p>
        </div>

        <div className="space-y-6">
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-purple-800 mb-2">Welcome Back!</h2>
            <p className="text-purple-700 text-sm">
              Access your personalized dashboard with advanced AI matching, analytics, and symbiosis opportunities.
            </p>
          </div>

          <button
            onClick={goToDashboard}
            className="w-full bg-purple-500 text-white px-6 py-3 rounded-lg hover:bg-purple-600 transition flex items-center justify-center space-x-2"
          >
            <span>Access Dashboard</span>
            <ArrowRight className="h-4 w-4" />
          </button>

          <div className="text-sm text-gray-500">
            <p>Password: <code className="bg-gray-100 px-2 py-1 rounded">NA10EN</code></p>
            <p className="mt-2">Or use the Quick Access button for development</p>
          </div>
        </div>
      </div>
    </div>
  );
} 