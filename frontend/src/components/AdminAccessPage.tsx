import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, Key, Users, Settings, ArrowRight, CheckCircle } from 'lucide-react';

export function AdminAccessPage() {
  const [isAdmin, setIsAdmin] = useState(false);
  const [secretKey, setSecretKey] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const tempAdmin = localStorage.getItem('temp-admin-access');
    if (tempAdmin === 'true') {
      setIsAdmin(true);
      // Redirect immediately to admin hub
      navigate('/admin-hub');
    }
  }, [navigate]);

  const handleAccess = () => {
    if (secretKey === 'NA10EN') {
      localStorage.setItem('temp-admin-access', 'true');
      setIsAdmin(true);
      // Redirect directly to admin hub
      navigate('/admin-hub');
    } else {
      alert('Wrong password! Access denied.');
      setSecretKey('');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md">
        <div className="text-center mb-8">
          <Shield className="h-16 w-16 text-purple-500 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Admin Access</h1>
          <p className="text-gray-600">Enter admin password to access the admin panel</p>
        </div>

        {isAdmin ? (
          <div className="text-center">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
              <CheckCircle className="h-8 w-8 text-green-500 mx-auto mb-2" />
              <p className="text-green-800 font-medium">Admin Access Granted!</p>
              <p className="text-green-600 text-sm">Redirecting to admin panel...</p>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Admin Password
              </label>
              <div className="flex space-x-2">
                <input
                  type="password"
                  value={secretKey}
                  onChange={(e) => setSecretKey(e.target.value)}
                  placeholder="Enter admin password"
                  className="flex-1 border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      handleAccess();
                    }
                  }}
                />
                <button
                  onClick={handleAccess}
                  className="bg-purple-500 text-white px-4 py-2 rounded-lg hover:bg-purple-600 transition"
                >
                  <Key className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="text-center">
              <button
                onClick={() => navigate('/dashboard')}
                className="text-gray-500 hover:text-gray-700 transition"
              >
                Back to Dashboard
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 