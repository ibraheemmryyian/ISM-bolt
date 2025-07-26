import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import PaymentProcessor from './PaymentProcessor';

interface SubscriptionPlan {
  id: string;
  name: string;
  type: string;
  monthly_price: number;
  features: string[];
  description: string;
  popular?: boolean;
}

interface UserSubscription {
  id: string;
  plan_name: string;
  plan_type: string;
  monthly_price: number;
  status: string;
  activated_at: string;
  next_billing_date: string;
}

const SubscriptionManager: React.FC = () => {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [userSubscription, setUserSubscription] = useState<UserSubscription | null>(null);
  const [showPaymentModal, setShowPaymentModal] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState<SubscriptionPlan | null>(null);
  const [error, setError] = useState<string | null>(null);

  const subscriptionPlans: SubscriptionPlan[] = [
    {
      id: 'free',
      name: 'Free Plan',
      type: 'FREE',
      monthly_price: 0,
      features: [
        '5 material listings per month',
        'Basic matching algorithm',
        'Email support',
        'Standard shipping calculator'
      ],
      description: 'Perfect for small businesses getting started'
    },
    {
      id: 'pro',
      name: 'Pro Plan',
      type: 'PRO',
      monthly_price: 99,
      features: [
        'Unlimited material listings',
        'Advanced AI matching',
        'Priority support',
        'Advanced analytics',
        'Custom shipping rates',
        'Scientific material data',
        'GNN symbiosis network'
      ],
      description: 'Ideal for growing businesses',
      popular: true
    },
    {
      id: 'enterprise',
      name: 'Enterprise Plan',
      type: 'ENTERPRISE',
      monthly_price: 299,
      features: [
        'Everything in Pro',
        'Dedicated account manager',
        'Custom integrations',
        'White-label solutions',
        'Advanced reporting',
        'API access',
        'Multi-location support',
        'Custom AI training'
      ],
      description: 'For large enterprises and corporations'
    }
  ];

  useEffect(() => {
    // Get current user
    const getUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      setUser(user);
      if (user) {
        fetchUserSubscription();
      }
    };
    getUser();

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      setUser(session?.user || null);
      if (session?.user) {
        fetchUserSubscription();
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  const fetchUserSubscription = async () => {
    try {
      const { data: companies } = await supabase
        .from('companies')
        .select('id, company_name, subscription_plan, subscription_status, subscription_expires_at')
        .eq('user_id', user?.id)
        .maybeSingle();

      if (companies) {
        const { data: subscription } = await supabase
          .from('subscriptions')
          .select('*')
          .eq('company_id', companies.id)
          .eq('status', 'ACTIVE')
          .maybeSingle();

        if (subscription) {
          setUserSubscription({
            id: subscription.id,
            plan_name: subscription.plan_name,
            plan_type: subscription.plan_type,
            monthly_price: subscription.monthly_price,
            status: subscription.status,
            activated_at: subscription.activated_at,
            next_billing_date: subscription.next_billing_date
          });
        }
      }
    } catch (error) {
      console.error('Error fetching subscription:', error);
      setError('Failed to load subscription information');
    } finally {
      setLoading(false);
    }
  };

  const handleUpgrade = (plan: SubscriptionPlan) => {
    setSelectedPlan(plan);
    setShowPaymentModal(true);
  };

  const handlePaymentSuccess = async (result: any) => {
    setShowPaymentModal(false);
    setSelectedPlan(null);
    await fetchUserSubscription();
    // Show success message
  };

  const handlePaymentError = (error: string) => {
    setError(error);
    setShowPaymentModal(false);
  };

  const handleCancelSubscription = async () => {
    if (!userSubscription) return;

    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/api/payments/cancel-subscription`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user?.access_token}`
        },
        body: JSON.stringify({
          subscriptionId: userSubscription.id
        })
      });

      if (response.ok) {
        await fetchUserSubscription();
        setError(null);
      } else {
        throw new Error('Failed to cancel subscription');
      }
    } catch (error) {
      console.error('Error canceling subscription:', error);
      setError('Failed to cancel subscription');
    }
  };

  const getCurrentPlan = () => {
    return subscriptionPlans.find(plan => plan.type === userSubscription?.plan_type) || subscriptionPlans[0];
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[1, 2, 3].map((i) => (
              <div key={i} className="bg-white rounded-lg shadow-lg p-6">
                <div className="h-6 bg-gray-200 rounded w-3/4 mb-4"></div>
                <div className="h-4 bg-gray-200 rounded w-1/2 mb-6"></div>
                <div className="space-y-2">
                  {[1, 2, 3, 4].map((j) => (
                    <div key={j} className="h-3 bg-gray-200 rounded"></div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Subscription Management</h1>
        <p className="text-gray-600">Choose the perfect plan for your business needs</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <div className="flex">
            <svg className="w-5 h-5 text-red-400 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div className="ml-3">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          </div>
        </div>
      )}

      {userSubscription && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-8">
          <h2 className="text-xl font-semibold text-blue-900 mb-4">Current Subscription</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-blue-700">Plan</p>
              <p className="font-semibold text-blue-900">{userSubscription.plan_name}</p>
            </div>
            <div>
              <p className="text-sm text-blue-700">Status</p>
              <p className="font-semibold text-blue-900 capitalize">{userSubscription.status.toLowerCase()}</p>
            </div>
            <div>
              <p className="text-sm text-blue-700">Next Billing</p>
              <p className="font-semibold text-blue-900">{formatDate(userSubscription.next_billing_date)}</p>
            </div>
          </div>
          {userSubscription.plan_type !== 'FREE' && (
            <button
              onClick={handleCancelSubscription}
              className="mt-4 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
            >
              Cancel Subscription
            </button>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {subscriptionPlans.map((plan) => {
          const isCurrentPlan = plan.type === userSubscription?.plan_type;
          const isUpgrade = plan.monthly_price > (userSubscription?.monthly_price || 0);

          return (
            <div
              key={plan.id}
              className={`relative bg-white rounded-lg shadow-lg p-6 border-2 ${
                plan.popular ? 'border-blue-500' : 'border-gray-200'
              } ${isCurrentPlan ? 'ring-2 ring-green-500' : ''}`}
            >
              {plan.popular && (
                <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                  <span className="bg-blue-500 text-white px-3 py-1 rounded-full text-sm font-semibold">
                    Most Popular
                  </span>
                </div>
              )}

              {isCurrentPlan && (
                <div className="absolute -top-3 right-4">
                  <span className="bg-green-500 text-white px-3 py-1 rounded-full text-sm font-semibold">
                    Current Plan
                  </span>
                </div>
              )}

              <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-gray-900 mb-2">{plan.name}</h3>
                <div className="mb-4">
                  <span className="text-4xl font-bold text-gray-900">
                    {plan.monthly_price === 0 ? 'Free' : `$${plan.monthly_price}`}
                  </span>
                  {plan.monthly_price > 0 && (
                    <span className="text-gray-600">/month</span>
                  )}
                </div>
                <p className="text-gray-600 text-sm">{plan.description}</p>
              </div>

              <ul className="space-y-3 mb-6">
                {plan.features.map((feature, index) => (
                  <li key={index} className="flex items-start">
                    <svg className="w-5 h-5 text-green-500 mt-0.5 mr-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span className="text-gray-700">{feature}</span>
                  </li>
                ))}
              </ul>

              <div className="text-center">
                {isCurrentPlan ? (
                  <button
                    disabled
                    className="w-full bg-gray-300 text-gray-500 py-3 px-4 rounded-lg cursor-not-allowed"
                  >
                    Current Plan
                  </button>
                ) : (
                  <button
                    onClick={() => handleUpgrade(plan)}
                    className={`w-full py-3 px-4 rounded-lg font-semibold transition-colors ${
                      isUpgrade
                        ? 'bg-blue-600 text-white hover:bg-blue-700'
                        : 'bg-gray-600 text-white hover:bg-gray-700'
                    }`}
                  >
                    {isUpgrade ? 'Upgrade' : 'Downgrade'}
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {showPaymentModal && selectedPlan && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <PaymentProcessor
              subscriptionData={{
                plan_name: selectedPlan.name,
                plan_type: selectedPlan.type,
                monthly_price: selectedPlan.monthly_price.toString(),
                description: selectedPlan.description,
                customer_name: user?.user_metadata?.full_name || 'Customer',
                customer_email: user?.email || '',
                company_id: userSubscription?.id || ''
              }}
              onSuccess={handlePaymentSuccess}
              onError={handlePaymentError}
              onCancel={() => setShowPaymentModal(false)}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default SubscriptionManager;