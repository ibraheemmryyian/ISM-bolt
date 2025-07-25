import React from 'react';
import { loadStripe } from '@stripe/stripe-js';

interface StripePaymentProcessorProps {
  priceId: string; // Stripe Price ID for subscription or one-time payment
  customerEmail?: string;
  metadata?: Record<string, string>;
  onError?: (error: string) => void;
}

const stripePromise = loadStripe(import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY as string);

const StripePaymentProcessor: React.FC<StripePaymentProcessorProps> = ({ priceId, customerEmail, metadata = {}, onError }) => {
  const handleCheckout = async () => {
    try {
      const stripe = await stripePromise;
      if (!stripe) throw new Error('Stripe failed to initialize');

      // Call backend to create checkout session
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/payments/stripe/create-checkout-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          customerEmail,
          lineItems: [
            {
              price: priceId,
              quantity: 1
            }
          ],
          mode: 'subscription',
          metadata
        })
      });

      const { id, url, error } = await response.json();
      if (error) throw new Error(error);

      // Redirect to Stripe Hosted Checkout
      if (url) {
        window.location.href = url;
      } else {
        await stripe.redirectToCheckout({ sessionId: id });
      }
    } catch (err) {
      console.error('Stripe checkout error:', err);
      onError?.(err instanceof Error ? err.message : 'Payment error');
    }
  };

  return (
    <button
      onClick={handleCheckout}
      className="w-full bg-indigo-600 text-white py-3 px-4 rounded-lg hover:bg-indigo-700 transition-colors"
    >
      Pay with Stripe
    </button>
  );
};

export default StripePaymentProcessor;