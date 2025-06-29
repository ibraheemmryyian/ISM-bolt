import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, Key, Users, Settings, ArrowRight, CheckCircle } from 'lucide-react';

export function AdminAccessPage() {
  const [isAdmin, setIsAdmin] = useState(false);
  const [secretKey, setSecretKey] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const tempAdmin = localStorage.getItem('temp-admin-access');
    setIsAdmin(tempAdmin === 'true');
  }, []);

  const handleAccess = () => {
    if (secretKey === 'secret123') {
      localStorage.setItem('temp-admin-access', 'true');
      setIsAdmin(true);
      setTimeout(() => {
        navigate('/admin');
      }, 1000);
    } else {
      alert('Invalid secret key!');
    }
  };

  const handleDirectAccess = () => {
    localStorage.setItem('temp-admin-access', 'true');
    setIsAdmin(true);
    setTimeout(() => {
      navigate('/admin');
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md">
        <div className="text-center mb-8">
          <Shield className="h-16 w-16 text-purple-500 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Admin Access</h1>
          <p className="text-gray-600">Access the admin panel to manage users and subscriptions</p>
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
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="font-semibold text-blue-900 mb-2">Quick Access Methods:</h3>
              <div className="space-y-2 text-sm text-blue-800">
                <p>1. <strong>Dashboard Button:</strong> Go to dashboard and click "Admin Access"</p>
                <p>2. <strong>Direct URL:</strong> Add ?admin=secret123 to any page</p>
                <p>3. <strong>Secret Key:</strong> Enter the key below</p>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Enter Secret Key
              </label>
              <div className="flex space-x-2">
                <input
                  type="password"
                  value={secretKey}
                  onChange={(e) => setSecretKey(e.target.value)}
                  placeholder="Enter secret key"
                  className="flex-1 border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                />
                <button
                  onClick={handleAccess}
                  className="bg-purple-500 text-white px-4 py-2 rounded-lg hover:bg-purple-600 transition"
                >
                  <Key className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="border-t pt-6">
              <button
                onClick={handleDirectAccess}
                className="w-full bg-emerald-500 text-white py-3 px-4 rounded-lg hover:bg-emerald-600 transition flex items-center justify-center space-x-2"
              >
                <Settings className="h-5 w-5" />
                <span>Grant Admin Access</span>
                <ArrowRight className="h-5 w-5" />
              </button>
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