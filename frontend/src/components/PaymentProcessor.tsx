import React, { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';

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
  const [orderId, setOrderId] = useState<string | null>(null);
  const [approvalUrl, setApprovalUrl] = useState<string | null>(null);
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

  const createPaymentOrder = async () => {
    if (!user) {
      setError('User not authenticated');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const endpoint = exchangeData ? '/api/payments/create-order' : '/api/payments/create-subscription';
      const data = exchangeData ? { exchangeData } : { subscriptionData };

      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user.access_token}`
        },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error('Failed to create payment order');
      }

      const result = await response.json();
      setOrderId(result.orderId || result.subscriptionId);
      setApprovalUrl(result.approvalUrl);
      setPaymentStatus('processing');

      // Redirect to PayPal
      if (result.approvalUrl) {
        window.location.href = result.approvalUrl;
      }
    } catch (err) {
      console.error('Payment creation error:', err);
      setError(err instanceof Error ? err.message : 'Payment creation failed');
      setPaymentStatus('failed');
    } finally {
      setLoading(false);
    }
  };

  const capturePayment = async (orderId: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/api/payments/capture/${orderId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user?.access_token}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to capture payment');
      }

      const result = await response.json();
      setPaymentStatus('completed');
      onSuccess?.(result);
    } catch (err) {
      console.error('Payment capture error:', err);
      setError(err instanceof Error ? err.message : 'Payment capture failed');
      setPaymentStatus('failed');
    } finally {
      setLoading(false);
    }
  };

  const handleReturnFromPayPal = () => {
    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get('token');
    const payerId = urlParams.get('PayerID');

    if (token && payerId) {
      capturePayment(token);
    } else {
      setError('Invalid PayPal return parameters');
      setPaymentStatus('failed');
    }
  };

  useEffect(() => {
    // Check if returning from PayPal
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('token') || urlParams.get('PayerID')) {
      handleReturnFromPayPal();
    }
  }, []);

  const formatCurrency = (amount: string) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(parseFloat(amount));
  };

  const renderExchangeSummary = () => {
    if (!exchangeData) return null;

    return (
      <div className="bg-gray-50 p-4 rounded-lg mb-6">
        <h3 className="text-lg font-semibold mb-4">Order Summary</h3>
        <div className="space-y-2">
          <div className="flex justify-between">
            <span>Material:</span>
            <span className="font-medium">{exchangeData.material_name}</span>
          </div>
          <div className="flex justify-between">
            <span>Quantity:</span>
            <span>{exchangeData.quantity}</span>
          </div>
          <div className="flex justify-between">
            <span>Unit Price:</span>
            <span>{formatCurrency(exchangeData.unit_price)}</span>
          </div>
          <div className="flex justify-between">
            <span>Material Cost:</span>
            <span>{formatCurrency(exchangeData.material_cost)}</span>
          </div>
          <div className="flex justify-between">
            <span>Shipping:</span>
            <span>{formatCurrency(exchangeData.shipping_cost)}</span>
          </div>
          {exchangeData.tax_amount && (
            <div className="flex justify-between">
              <span>Tax:</span>
              <span>{formatCurrency(exchangeData.tax_amount)}</span>
            </div>
          )}
          {exchangeData.platform_fee && (
            <div className="flex justify-between">
              <span>Platform Fee:</span>
              <span>{formatCurrency(exchangeData.platform_fee)}</span>
            </div>
          )}
          <div className="border-t pt-2 mt-2">
            <div className="flex justify-between font-semibold text-lg">
              <span>Total:</span>
              <span>{formatCurrency(exchangeData.total_amount)}</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderSubscriptionSummary = () => {
    if (!subscriptionData) return null;

    return (
      <div className="bg-gray-50 p-4 rounded-lg mb-6">
        <h3 className="text-lg font-semibold mb-4">Subscription Summary</h3>
        <div className="space-y-2">
          <div className="flex justify-between">
            <span>Plan:</span>
            <span className="font-medium">{subscriptionData.plan_name}</span>
          </div>
          <div className="flex justify-between">
            <span>Type:</span>
            <span>{subscriptionData.plan_type}</span>
          </div>
          <div className="flex justify-between">
            <span>Monthly Price:</span>
            <span>{formatCurrency(subscriptionData.monthly_price)}</span>
          </div>
          <div className="flex justify-between">
            <span>Customer:</span>
            <span>{subscriptionData.customer_name}</span>
          </div>
          <div className="flex justify-between">
            <span>Email:</span>
            <span>{subscriptionData.customer_email}</span>
          </div>
        </div>
      </div>
    );
  };

  const renderPaymentStatus = () => {
    switch (paymentStatus) {
      case 'pending':
        return (
          <div className="text-center">
            <div className="mb-4">
              <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto"></div>
            </div>
            <p className="text-gray-600">Preparing payment...</p>
          </div>
        );
      case 'processing':
        return (
          <div className="text-center">
            <div className="mb-4">
              <div className="w-16 h-16 border-4 border-yellow-200 border-t-yellow-600 rounded-full animate-spin mx-auto"></div>
            </div>
            <p className="text-gray-600">Processing payment...</p>
          </div>
        );
      case 'completed':
        return (
          <div className="text-center">
            <div className="mb-4">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
            </div>
            <p className="text-green-600 font-semibold">Payment Successful!</p>
            <p className="text-gray-600 text-sm mt-2">Your transaction has been completed.</p>
          </div>
        );
      case 'failed':
        return (
          <div className="text-center">
            <div className="mb-4">
              <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto">
                <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </div>
            </div>
            <p className="text-red-600 font-semibold">Payment Failed</p>
            <p className="text-gray-600 text-sm mt-2">Please try again or contact support.</p>
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
        <div className="mt-6 space-y-3">
          {paymentStatus === 'completed' && (
            <button
              onClick={() => onSuccess?.({ status: 'completed' })}
              className="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors"
            >
              Continue
            </button>
          )}
          {paymentStatus === 'failed' && (
            <div className="space-y-3">
              <button
                onClick={() => {
                  setPaymentStatus('pending');
                  setError(null);
                }}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Try Again
              </button>
              <button
                onClick={onCancel}
                className="w-full bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 transition-colors"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-md mx-auto bg-white rounded-lg shadow-lg p-6">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          {exchangeData ? 'Complete Purchase' : 'Subscribe to Plan'}
        </h2>
        <p className="text-gray-600">
          {exchangeData ? 'Secure payment via PayPal' : 'Start your subscription today'}
        </p>
      </div>

      {renderExchangeSummary()}
      {renderSubscriptionSummary()}

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

      <div className="space-y-4">
        <button
          onClick={createPaymentOrder}
          disabled={loading}
          className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
        >
          {loading ? (
            <>
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
              Processing...
            </>
          ) : (
            <>
              <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24" fill="currentColor">
                <path d="M20 4H4c-1.11 0-1.99.89-1.99 2L2 18c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V6c0-1.11-.89-2-2-2zm0 14H4v-6h16v6zm0-10H4V6h16v2z"/>
              </svg>
              Pay with PayPal
            </>
          )}
        </button>

        <button
          onClick={onCancel}
          disabled={loading}
          className="w-full bg-gray-200 text-gray-800 py-3 px-4 rounded-lg hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Cancel
        </button>
      </div>

      <div className="mt-6 text-center">
        <p className="text-xs text-gray-500">
          Your payment is secured by PayPal's industry-leading security standards.
        </p>
      </div>
    </div>
  );
};

export default PaymentProcessor; 