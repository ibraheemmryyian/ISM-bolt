import React, { useState, useEffect } from 'react';
import { supabase, testSupabaseConnection } from '../lib/supabase';
import { X, Mail, CheckCircle, AlertCircle, Users, Building, Clock, Wifi, WifiOff } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useNotifications } from '../lib/notificationContext';

interface AuthModalProps {
  onClose: () => void;
}

export function AuthModal({ onClose }: AuthModalProps) {
  const navigate = useNavigate();
  const { showNotification } = useNotifications();
  const [isSignUp, setIsSignUp] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [emailSent, setEmailSent] = useState(false);
  const [applicationSubmitted, setApplicationSubmitted] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [formData, setFormData] = useState({
    password: '',
    companyName: '',
    email: '',
    industry: '',
    wasteStreams: '',
    sustainabilityGoals: '',
  });

  // Test connection on component mount
  useEffect(() => {
    const testConnection = async () => {
      setConnectionStatus('checking');
      const isConnected = await testSupabaseConnection();
      setConnectionStatus(isConnected ? 'connected' : 'disconnected');
    };
    
    testConnection();
  }, []);

  const retryConnection = async () => {
    setConnectionStatus('checking');
    const isConnected = await testSupabaseConnection();
    setConnectionStatus(isConnected ? 'connected' : 'disconnected');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (isSignUp) {
        // Check if email already exists
        const { data: existingEmail } = await supabase
          .from('companies')
          .select('email')
          .eq('email', formData.email)
          .single();

        if (existingEmail) {
          throw new Error('Email already registered. Please use a different email or sign in.');
        }

        // Create auth user with real email (Supabase will send verification email)
        const { error: signUpError, data: authData } = await supabase.auth.signUp({
          email: formData.email,
          password: formData.password,
          options: {
            data: {
              company_name: formData.companyName,
            },
            emailRedirectTo: `${window.location.origin}/auth/callback`
          }
        });

        if (signUpError) throw signUpError;

        if (authData.user && !authData.user.email_confirmed_at) {
          // Email verification required
          setEmailSent(true);
          return;
        }

        if (authData.user) {
          // Create company record with pending status
          const { error: companyError } = await supabase.from('companies').insert([
            {
              id: authData.user.id,
              name: formData.companyName,
              email: formData.email,
              role: 'pending', // Pending approval
              industry: formData.industry,
              process_description: `Waste Streams: ${formData.wasteStreams}\nSustainability Goals: ${formData.sustainabilityGoals}`,
              onboarding_completed: false,
            },
          ]);

          if (companyError) throw companyError;

          // Submit application for approval
          const { error: applicationError } = await supabase.from('company_applications').insert([
            {
              company_name: formData.companyName,
              contact_name: formData.companyName, // Use company name as contact name
              contact_email: formData.email,
              application_answers: {
                industry: formData.industry,
                waste_streams: formData.wasteStreams,
                sustainability_goals: formData.sustainabilityGoals,
              },
              status: 'pending'
            }
          ]);

          if (applicationError) throw applicationError;

          // Show application submitted message
          setApplicationSubmitted(true);
          showNotification({
            type: 'success',
            title: 'Application Submitted!',
            message: 'We\'ll review your application within 24-48 hours.',
            duration: 8000
          });
        }
      } else {
        // Sign in - use email only
        const { error: signInError } = await supabase.auth.signInWithPassword({
          email: formData.email,
          password: formData.password,
        });

        if (signInError) {
          if (signInError.message.includes('Invalid login credentials')) {
            throw new Error('Invalid email or password. If you just created an account, please check your email for verification.');
          }
          if (signInError.message.includes('Email not confirmed')) {
            throw new Error('Please verify your email address before signing in. Check your inbox for a verification link.');
          }
          throw signInError;
        }

        onClose();
        showNotification({
          type: 'success',
          title: 'Welcome back!',
          message: 'Successfully signed in to your account.',
          duration: 3000
        });
        navigate('/dashboard');
      }
    } catch (err: any) {
      console.error('Auth Error:', err);
      
      // Provide more specific error messages
      if (err.message?.includes('fetch')) {
        setError('Network error: Unable to connect to the server. Please check your internet connection and try again.');
      } else if (err.message?.includes('Invalid login credentials')) {
        setError('Invalid username/email or password. Please check your credentials and try again.');
      } else if (err.message?.includes('Email not confirmed')) {
        setError('Please verify your email address before signing in. Check your inbox for a verification link.');
      } else if (err.message?.includes('Username already taken')) {
        setError('This username is already taken. Please choose a different username.');
      } else if (err.message?.includes('Email already registered')) {
        setError('This email is already registered. Please use a different email or sign in with your existing account.');
      } else if (err.message?.includes('pending approval')) {
        setError('Your account is pending approval. Please wait for admin approval or contact support.');
      } else {
        setError(err.message || 'Authentication failed. Please try again. If the problem persists, please contact support.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleResendVerification = async () => {
    setLoading(true);
    setError('');
    
    try {
      const { error } = await supabase.auth.resend({
        type: 'signup',
        email: formData.email,
        options: {
          emailRedirectTo: `${window.location.origin}/auth/callback`
        }
      });
      
      if (error) throw error;
      
      setError('');
      // Show success message
      setTimeout(() => {
        setEmailSent(false);
        setFormData({ password: '', companyName: '', email: '', industry: '', wasteStreams: '', sustainabilityGoals: '' });
      }, 3000);
    } catch (err: any) {
      setError(err.message || 'Failed to resend verification email.');
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordReset = async () => {
    if (!formData.email) {
      setError('Please enter your email address to reset your password.');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const { error } = await supabase.auth.resetPasswordForEmail(formData.email, {
        redirectTo: `${window.location.origin}/auth/reset-password`
      });
      
      if (error) throw error;
      
      setError('');
      alert('Password reset email sent! Check your inbox.');
    } catch (err: any) {
      setError(err.message || 'Failed to send password reset email.');
    } finally {
      setLoading(false);
    }
  };

  if (applicationSubmitted) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
        <div className="bg-white rounded-xl p-6 w-full max-w-md relative">
          <button
            onClick={onClose}
            className="absolute right-4 top-4 text-gray-500 hover:text-gray-700"
          >
            <X className="h-5 w-5" />
          </button>

          <div className="text-center">
            <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-emerald-100 mb-4">
              <CheckCircle className="h-6 w-6 text-emerald-600" />
            </div>
            
            <h2 className="text-2xl font-bold mb-4">Application Submitted!</h2>
            
            <p className="text-gray-600 mb-6">
              Thank you for your interest in SymbioFlows. We've received your application and will review it within 24-48 hours.
            </p>
            
            <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4 mb-6">
              <div className="flex items-start">
                <Clock className="h-5 w-5 text-emerald-600 mt-0.5 mr-3 flex-shrink-0" />
                <div className="text-sm text-emerald-800">
                  <p className="font-medium">What happens next:</p>
                  <ol className="list-decimal list-inside mt-2 space-y-1">
                    <li>We'll review your company information</li>
                    <li>You'll receive an approval email</li>
                    <li>Once approved, you can sign in and access the platform</li>
                  </ol>
                </div>
              </div>
            </div>

            <button
              onClick={onClose}
              className="w-full bg-emerald-500 text-white py-2 px-4 rounded-lg hover:bg-emerald-600 transition"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (emailSent) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
        <div className="bg-white rounded-xl p-6 w-full max-w-md relative">
          <button
            onClick={onClose}
            className="absolute right-4 top-4 text-gray-500 hover:text-gray-700"
          >
            <X className="h-5 w-5" />
          </button>

          <div className="text-center">
            <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 mb-4">
              <Mail className="h-6 w-6 text-blue-600" />
            </div>
            
            <h2 className="text-2xl font-bold mb-4">Check Your Email</h2>
            
            <p className="text-gray-600 mb-6">
              We've sent a verification link to <strong>{formData.email}</strong>
            </p>
            
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
              <div className="flex items-start">
                <CheckCircle className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
                <div className="text-sm text-blue-800">
                  <p className="font-medium">Next steps:</p>
                  <ol className="list-decimal list-inside mt-2 space-y-1">
                    <li>Check your email inbox (and spam folder)</li>
                    <li>Click the verification link in the email</li>
                    <li>Return here to sign in</li>
                  </ol>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <button
                onClick={handleResendVerification}
                disabled={loading}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition disabled:opacity-50"
              >
                {loading ? 'Sending...' : 'Resend Verification Email'}
              </button>
              
              <button
                onClick={() => {
                  setEmailSent(false);
                  setFormData({ password: '', companyName: '', email: '', industry: '', wasteStreams: '', sustainabilityGoals: '' });
                }}
                className="w-full text-sm text-gray-600 hover:text-gray-800"
              >
                Use a different email
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-xl p-6 w-full max-w-md relative">
        <button
          onClick={onClose}
          className="absolute right-4 top-4 text-gray-500 hover:text-gray-700"
        >
          <X className="h-5 w-5" />
        </button>

        <div className="text-center mb-6">
          <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-emerald-100 mb-4">
            {isSignUp ? <Building className="h-6 w-6 text-emerald-600" /> : <Users className="h-6 w-6 text-emerald-600" />}
          </div>
          <h2 className="text-2xl font-bold">
            {isSignUp ? 'Join SymbioFlows' : 'Welcome Back'}
          </h2>
          <p className="text-gray-600 mt-2">
            {isSignUp ? 'Create your account to start connecting with industrial partners' : 'Sign in to access your dashboard'}
          </p>
          
          {/* Connection Status Indicator */}
          <div className="mt-4 flex items-center justify-center space-x-2">
            {connectionStatus === 'checking' && (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                <span className="text-sm text-blue-600">Checking connection...</span>
              </>
            )}
            {connectionStatus === 'connected' && (
              <>
                <Wifi className="h-4 w-4 text-green-500" />
                <span className="text-sm text-green-600">Connected</span>
              </>
            )}
            {connectionStatus === 'disconnected' && (
              <>
                <WifiOff className="h-4 w-4 text-red-500" />
                <span className="text-sm text-red-600">Connection failed</span>
                <button
                  onClick={retryConnection}
                  className="ml-2 text-xs text-blue-600 hover:text-blue-800 underline"
                >
                  Retry
                </button>
              </>
            )}
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {connectionStatus === 'disconnected' && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
              <div className="flex items-start">
                <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 mr-3 flex-shrink-0" />
                <div className="text-sm text-red-800">
                  <p className="font-medium">Connection Error</p>
                  <p className="mt-1">Unable to connect to the server. Please check your internet connection and try again.</p>
                </div>
              </div>
            </div>
          )}
          {isSignUp && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Company Name
              </label>
              <input
                type="text"
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
                value={formData.companyName}
                onChange={(e) => setFormData({ ...formData, companyName: e.target.value })}
              />
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Email Address
            </label>
            <input
              type="email"
              required
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              placeholder="your-email@company.com"
            />
            {isSignUp && (
              <p className="text-xs text-gray-500 mt-1">
                We'll send a verification link to this email
              </p>
            )}
            {!isSignUp && (
              <p className="text-xs text-gray-500 mt-1">
                Required for password reset
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Password
            </label>
            <input
              type="password"
              required
              minLength={6}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
              value={formData.password}
              onChange={(e) => setFormData({ ...formData, password: e.target.value })}
            />
            <p className="text-xs text-gray-500 mt-1">
              {isSignUp && 'Password must be at least 6 characters long'}
            </p>
          </div>

          {error && (
            <div className="text-red-500 text-sm bg-red-50 p-3 rounded-lg border border-red-200">
              <div className="flex items-start">
                <AlertCircle className="h-4 w-4 text-red-500 mt-0.5 mr-2 flex-shrink-0" />
                <span>{error}</span>
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={loading || connectionStatus === 'disconnected'}
            className="w-full bg-emerald-500 text-white py-2 px-4 rounded-lg hover:bg-emerald-600 transition disabled:opacity-50"
          >
            {loading ? 'Processing...' : (isSignUp ? 'Create Account & Continue' : 'Sign In')}
          </button>

          {!isSignUp && (
            <button
              type="button"
              onClick={handlePasswordReset}
              disabled={loading || !formData.email}
              className="w-full text-sm text-emerald-600 hover:text-emerald-700 disabled:opacity-50"
            >
              Forgot your password?
            </button>
          )}

          <div className="text-center">
            <button
              type="button"
              onClick={() => {
                setIsSignUp(!isSignUp);
                setError('');
                setFormData({ password: '', companyName: '', email: '', industry: '', wasteStreams: '', sustainabilityGoals: '' });
              }}
              className="text-sm text-emerald-600 hover:text-emerald-700"
            >
              {isSignUp ? 'Already have an account? Sign in' : "Don't have an account? Sign up"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}