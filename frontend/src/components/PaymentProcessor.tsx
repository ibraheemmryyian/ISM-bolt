import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import { loadStripe } from '@stripe/stripe-js';

// Load Stripe
const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY);

interface PaymentProcessorProps {
  exchangeData?: {
    exchange_id: string;
    material_name: string;
    quantity: number;
    unit_price: string;
    material_cost: string;
    shipping_cost: string;
    tax_amount?: string;
    platform_fee?: string;
    total_amount: string;
    shipping_address: {
      street1: string;
      street2?: string;
      city: string;
      state: string;
      zip: string;
      country: string;
    };
  };
  subscriptionData?: {
    plan_name: string;
    plan_type: string;
    monthly_price: string;
    description: string;
    customer_name: string;
    customer_email: string;
    company_id: string;
  };
  onSuccess?: (result: any) => void;
  onError?: (error: string) => void;
  onCancel?: () => void;
}

const PaymentProcessor: React.FC<PaymentProcessorProps> = ({
  exchangeData,
  subscriptionData,
  onSuccess,
  onError,
  onCancel
}) => {
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [paymentIntentId, setPaymentIntentId] = useState<string | null>(null);
  const [clientSecret, setClientSecret] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [paymentStatus, setPaymentStatus] = useState<'pending' | 'processing' | 'completed' | 'failed'>('pending');

  useEffect(() => {
    // Get current user
    const getUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      setUser(user);
    };
    getUser();

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      setUser(session?.user || null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const createPaymentIntent = async () => {
    if (!user) {
      setError('User not authenticated');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const endpoint = exchangeData ? '/api/payments/create-payment-intent' : '/api/payments/create-subscription';
      const data = exchangeData ? { 
        exchangeData,
        amount: parseFloat(exchangeData.total_amount)
      } : { 
        subscriptionData,
        amount: parseFloat(subscriptionData.monthly_price)
      };

      const response = await fetch(`${import.meta.env.VITE_API_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user.access_token}`
        },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error('Failed to create payment intent');
      }

      const result = await response.json();
      
      if (result.success) {
        setPaymentIntentId(result.payment_intent_id);
        setClientSecret(result.client_secret);
        setPaymentStatus('processing');
      } else {
        throw new Error(result.error || 'Failed to create payment intent');
      }
    } catch (error) {
      console.error('Payment intent creation error:', error);
      setError(error instanceof Error ? error.message : 'Payment creation failed');
      setPaymentStatus('failed');
    } finally {
      setLoading(false);
    }
  };

  const handleStripePayment = async () => {
    if (!clientSecret) {
      setError('No payment intent available');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const stripe = await stripePromise;
      if (!stripe) {
        throw new Error('Stripe failed to load');
      }

      const { error: stripeError, paymentIntent } = await stripe.confirmPayment({
        clientSecret,
        confirmParams: {
          return_url: `${window.location.origin}/payment/success`,
          payment_method_data: {
            billing_details: {
              name: subscriptionData?.customer_name || user?.user_metadata?.full_name || '',
              email: subscriptionData?.customer_email || user?.email || '',
            },
          },
        },
      });

      if (stripeError) {
        throw new Error(stripeError.message);
      }

      if (paymentIntent.status === 'succeeded') {
        setPaymentStatus('completed');
        onSuccess?.(paymentIntent);
      } else {
        setPaymentStatus('failed');
        setError('Payment failed');
      }
    } catch (error) {
      console.error('Stripe payment error:', error);
      setError(error instanceof Error ? error.message : 'Payment failed');
      setPaymentStatus('failed');
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount: string) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(parseFloat(amount));
  };

  const renderExchangeSummary = () => {
    if (!exchangeData) return null;

    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Material Exchange Summary</h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-600">Material:</span>
            <span className="font-medium">{exchangeData.material_name}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Quantity:</span>
            <span className="font-medium">{exchangeData.quantity}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Unit Price:</span>
            <span className="font-medium">{formatCurrency(exchangeData.unit_price)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Material Cost:</span>
            <span className="font-medium">{formatCurrency(exchangeData.material_cost)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Shipping:</span>
            <span className="font-medium">{formatCurrency(exchangeData.shipping_cost)}</span>
          </div>
          {exchangeData.tax_amount && (
            <div className="flex justify-between">
              <span className="text-gray-600">Tax:</span>
              <span className="font-medium">{formatCurrency(exchangeData.tax_amount)}</span>
            </div>
          )}
          {exchangeData.platform_fee && (
            <div className="flex justify-between">
              <span className="text-gray-600">Platform Fee:</span>
              <span className="font-medium">{formatCurrency(exchangeData.platform_fee)}</span>
            </div>
          )}
          <div className="border-t pt-3">
            <div className="flex justify-between text-lg font-semibold">
              <span>Total:</span>
              <span className="text-green-600">{formatCurrency(exchangeData.total_amount)}</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderSubscriptionSummary = () => {
    if (!subscriptionData) return null;

    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Subscription Summary</h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-600">Plan:</span>
            <span className="font-medium">{subscriptionData.plan_name}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Type:</span>
            <span className="font-medium">{subscriptionData.plan_type}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Monthly Price:</span>
            <span className="font-medium">{formatCurrency(subscriptionData.monthly_price)}</span>
          </div>
          <div className="border-t pt-3">
            <div className="flex justify-between text-lg font-semibold">
              <span>Total:</span>
              <span className="text-green-600">{formatCurrency(subscriptionData.monthly_price)}</span>
            </div>
          </div>
        </div>
        <p className="text-sm text-gray-500 mt-3">{subscriptionData.description}</p>
      </div>
    );
  };

  const renderPaymentStatus = () => {
    switch (paymentStatus) {
      case 'completed':
        return (
          <div className="text-center">
            <div className="text-green-600 text-6xl mb-4">✓</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Payment Successful!</h3>
            <p className="text-gray-600">Your payment has been processed successfully.</p>
          </div>
        );
      case 'failed':
        return (
          <div className="text-center">
            <div className="text-red-600 text-6xl mb-4">✗</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">Payment Failed</h3>
            <p className="text-gray-600">{error || 'There was an error processing your payment.'}</p>
            <button
              onClick={() => {
                setPaymentStatus('pending');
                setError(null);
                setPaymentIntentId(null);
                setClientSecret(null);
              }}
              className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            >
              Try Again
            </button>
          </div>
        );
      default:
        return null;
    }
  };

  if (paymentStatus === 'completed' || paymentStatus === 'failed') {
    return (
      <div className="max-w-md mx-auto bg-white rounded-lg shadow-lg p-6">
        {renderPaymentStatus()}
      </div>
    );
  }

  return (
    <div className="max-w-md mx-auto bg-white rounded-lg shadow-lg p-6">
      {exchangeData && renderExchangeSummary()}
      {subscriptionData && renderSubscriptionSummary()}

      <div className="space-y-4">
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}

        {!clientSecret ? (
          <button
            onClick={createPaymentIntent}
            disabled={loading}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Creating Payment...' : (exchangeData ? 'Pay with Stripe' : 'Start your subscription today')}
          </button>
        ) : (
          <button
            onClick={handleStripePayment}
            disabled={loading}
            className="w-full bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Processing Payment...' : 'Complete Payment with Stripe'}
          </button>
        )}

        <button
          onClick={onCancel}
          className="w-full bg-gray-300 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-400"
        >
          Cancel
        </button>
      </div>

      <div className="mt-6 text-center text-sm text-gray-500">
        <p>Your payment is secured by Stripe's industry-leading security standards.</p>
        <p className="mt-2">
          <span className="inline-flex items-center">
            <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z" clipRule="evenodd" />
            </svg>
            Secure Payment
          </span>
        </p>
      </div>
    </div>
  );
};

export default PaymentProcessor; 