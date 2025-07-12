import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Redirect to dashboard or login
    navigate('/dashboard');
  }, [navigate]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-r from-emerald-600 to-blue-600">
      <div className="text-center text-white">
        <h1 className="text-4xl font-bold mb-4">SymbioFlows</h1>
        <p className="text-xl">Industrial Symbiosis Management Platform</p>
        <div className="mt-8">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto"></div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;