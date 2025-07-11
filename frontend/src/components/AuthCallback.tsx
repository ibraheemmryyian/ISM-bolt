import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { CheckCircle, XCircle, Loader2, ArrowRight } from 'lucide-react';

export function AuthCallback() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState('');

  useEffect(() => {
    const handleAuthCallback = async () => {
      try {
        const { data, error } = await supabase.auth.getSession();
        
        if (error) {
          console.error('Auth callback error:', error);
          setStatus('error');
          setMessage('Authentication failed. Please try again.');
          return;
        }

        if (data.session) {
          // User is authenticated
          setStatus('success');
          setMessage('Email verified successfully! Redirecting to dashboard...');
          
          // Redirect to dashboard after a short delay
          setTimeout(() => {
            navigate('/dashboard');
          }, 2000);
        } else {
          // Check for specific error types
          const errorParam = searchParams.get('error');
          const errorDescription = searchParams.get('error_description');
          
          if (errorParam) {
            setStatus('error');
            setMessage(errorDescription || 'Authentication failed. Please try again.');
          } else {
            setStatus('error');
            setMessage('No session found. Please try signing in again.');
          }
        }
      } catch (err) {
        console.error('Auth callback error:', err);
        setStatus('error');
        setMessage('An unexpected error occurred. Please try again.');
      }
    };

    handleAuthCallback();
  }, [navigate, searchParams]);

  const handleRetry = () => {
    navigate('/');
  };

  const handleGoToSignIn = () => {
    navigate('/');
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
      <div className="max-w-md w-full bg-white rounded-xl shadow-lg p-8">
        <div className="text-center">
          {status === 'loading' && (
            <>
              <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 mb-4">
                <Loader2 className="h-6 w-6 text-blue-600 animate-spin" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Verifying Your Email
              </h2>
              <p className="text-gray-600">
                Please wait while we verify your email address...
              </p>
            </>
          )}

          {status === 'success' && (
            <>
              <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-green-100 mb-4">
                <CheckCircle className="h-6 w-6 text-green-600" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Email Verified!
              </h2>
              <p className="text-gray-600 mb-6">
                {message}
              </p>
              <div className="flex items-center justify-center text-sm text-gray-500">
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Redirecting...
              </div>
            </>
          )}

          {status === 'error' && (
            <>
              <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-100 mb-4">
                <XCircle className="h-6 w-6 text-red-600" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Verification Failed
              </h2>
              <p className="text-gray-600 mb-6">
                {message}
              </p>
              <div className="space-y-3">
                <button
                  onClick={handleRetry}
                  className="w-full bg-emerald-500 text-white py-2 px-4 rounded-lg hover:bg-emerald-600 transition"
                >
                  Try Again
                </button>
                <button
                  onClick={handleGoToSignIn}
                  className="w-full text-sm text-gray-600 hover:text-gray-800 flex items-center justify-center"
                >
                  Go to Sign In
                  <ArrowRight className="h-4 w-4 ml-1" />
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
} 