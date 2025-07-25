const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
const { createClient } = require('@supabase/supabase-js');

class StripePaymentService {
  constructor() {
    this.stripe = stripe;
    this.supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );
  }

  /**
   * Create a Stripe payment intent for material exchange
   */
  async createPaymentIntent(amount, currency = 'usd', metadata = {}) {
    try {
      const paymentIntent = await this.stripe.paymentIntents.create({
        amount: Math.round(amount * 100), // Convert to cents
        currency,
        metadata: {
          ...metadata,
          service: 'symbioflows_material_exchange'
        },
        automatic_payment_methods: {
          enabled: true,
        },
      });

      return {
        success: true,
        client_secret: paymentIntent.client_secret,
        payment_intent_id: paymentIntent.id,
        amount: paymentIntent.amount,
        currency: paymentIntent.currency,
        status: paymentIntent.status
      };
    } catch (error) {
      console.error('Error creating Stripe payment intent:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Create a Stripe subscription for premium features
   */
  async createSubscription(customerId, priceId, metadata = {}) {
    try {
      const subscription = await this.stripe.subscriptions.create({
        customer: customerId,
        items: [{ price: priceId }],
        metadata: {
          ...metadata,
          service: 'symbioflows_premium_subscription'
        },
        payment_behavior: 'default_incomplete',
        payment_settings: { save_default_payment_method: 'on_subscription' },
        expand: ['latest_invoice.payment_intent'],
      });

      return {
        success: true,
        subscription_id: subscription.id,
        client_secret: subscription.latest_invoice.payment_intent.client_secret,
        status: subscription.status
      };
    } catch (error) {
      console.error('Error creating Stripe subscription:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Create or retrieve a Stripe customer
   */
  async createCustomer(email, name, metadata = {}) {
    try {
      // Check if customer already exists
      const existingCustomers = await this.stripe.customers.list({
        email: email,
        limit: 1
      });

      if (existingCustomers.data.length > 0) {
        return {
          success: true,
          customer_id: existingCustomers.data[0].id,
          is_new: false
        };
      }

      // Create new customer
      const customer = await this.stripe.customers.create({
        email,
        name,
        metadata: {
          ...metadata,
          service: 'symbioflows'
        }
      });

      return {
        success: true,
        customer_id: customer.id,
        is_new: true
      };
    } catch (error) {
      console.error('Error creating Stripe customer:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Process webhook events from Stripe
   */
  async handleWebhook(event) {
    try {
      switch (event.type) {
        case 'payment_intent.succeeded':
          await this.handlePaymentSuccess(event.data.object);
          break;
        case 'payment_intent.payment_failed':
          await this.handlePaymentFailure(event.data.object);
          break;
        case 'customer.subscription.created':
          await this.handleSubscriptionCreated(event.data.object);
          break;
        case 'customer.subscription.updated':
          await this.handleSubscriptionUpdated(event.data.object);
          break;
        case 'customer.subscription.deleted':
          await this.handleSubscriptionCancelled(event.data.object);
          break;
        case 'invoice.payment_succeeded':
          await this.handleInvoicePaymentSucceeded(event.data.object);
          break;
        case 'invoice.payment_failed':
          await this.handleInvoicePaymentFailed(event.data.object);
          break;
        default:
          console.log(`Unhandled event type: ${event.type}`);
      }

      return { success: true };
    } catch (error) {
      console.error('Error handling Stripe webhook:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Handle successful payment
   */
  async handlePaymentSuccess(paymentIntent) {
    try {
      const { data, error } = await this.supabase
        .from('payments')
        .update({
          status: 'completed',
          stripe_payment_intent_id: paymentIntent.id,
          amount: paymentIntent.amount / 100,
          currency: paymentIntent.currency,
          updated_at: new Date().toISOString()
        })
        .eq('stripe_payment_intent_id', paymentIntent.id);

      if (error) {
        console.error('Error updating payment record:', error);
      }
    } catch (error) {
      console.error('Error handling payment success:', error);
    }
  }

  /**
   * Handle failed payment
   */
  async handlePaymentFailure(paymentIntent) {
    try {
      const { data, error } = await this.supabase
        .from('payments')
        .update({
          status: 'failed',
          stripe_payment_intent_id: paymentIntent.id,
          updated_at: new Date().toISOString()
        })
        .eq('stripe_payment_intent_id', paymentIntent.id);

      if (error) {
        console.error('Error updating payment record:', error);
      }
    } catch (error) {
      console.error('Error handling payment failure:', error);
    }
  }

  /**
   * Handle subscription creation
   */
  async handleSubscriptionCreated(subscription) {
    try {
      const { data, error } = await this.supabase
        .from('subscriptions')
        .insert({
          stripe_subscription_id: subscription.id,
          customer_id: subscription.customer,
          status: subscription.status,
          current_period_start: new Date(subscription.current_period_start * 1000).toISOString(),
          current_period_end: new Date(subscription.current_period_end * 1000).toISOString(),
          created_at: new Date().toISOString()
        });

      if (error) {
        console.error('Error creating subscription record:', error);
      }
    } catch (error) {
      console.error('Error handling subscription creation:', error);
    }
  }

  /**
   * Handle subscription updates
   */
  async handleSubscriptionUpdated(subscription) {
    try {
      const { data, error } = await this.supabase
        .from('subscriptions')
        .update({
          status: subscription.status,
          current_period_start: new Date(subscription.current_period_start * 1000).toISOString(),
          current_period_end: new Date(subscription.current_period_end * 1000).toISOString(),
          updated_at: new Date().toISOString()
        })
        .eq('stripe_subscription_id', subscription.id);

      if (error) {
        console.error('Error updating subscription record:', error);
      }
    } catch (error) {
      console.error('Error handling subscription update:', error);
    }
  }

  /**
   * Handle subscription cancellation
   */
  async handleSubscriptionCancelled(subscription) {
    try {
      const { data, error } = await this.supabase
        .from('subscriptions')
        .update({
          status: 'cancelled',
          cancelled_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
        .eq('stripe_subscription_id', subscription.id);

      if (error) {
        console.error('Error updating subscription record:', error);
      }
    } catch (error) {
      console.error('Error handling subscription cancellation:', error);
    }
  }

  /**
   * Handle successful invoice payment
   */
  async handleInvoicePaymentSucceeded(invoice) {
    try {
      // Update subscription status if needed
      if (invoice.subscription) {
        const { data, error } = await this.supabase
          .from('subscriptions')
          .update({
            status: 'active',
            updated_at: new Date().toISOString()
          })
          .eq('stripe_subscription_id', invoice.subscription);

        if (error) {
          console.error('Error updating subscription status:', error);
        }
      }
    } catch (error) {
      console.error('Error handling invoice payment success:', error);
    }
  }

  /**
   * Handle failed invoice payment
   */
  async handleInvoicePaymentFailed(invoice) {
    try {
      // Update subscription status if needed
      if (invoice.subscription) {
        const { data, error } = await this.supabase
          .from('subscriptions')
          .update({
            status: 'past_due',
            updated_at: new Date().toISOString()
          })
          .eq('stripe_subscription_id', invoice.subscription);

        if (error) {
          console.error('Error updating subscription status:', error);
        }
      }
    } catch (error) {
      console.error('Error handling invoice payment failure:', error);
    }
  }

  /**
   * Refund a payment
   */
  async refundPayment(paymentIntentId, amount = null) {
    try {
      const refundParams = {
        payment_intent: paymentIntentId
      };

      if (amount) {
        refundParams.amount = Math.round(amount * 100);
      }

      const refund = await this.stripe.refunds.create(refundParams);

      return {
        success: true,
        refund_id: refund.id,
        amount: refund.amount / 100,
        status: refund.status
      };
    } catch (error) {
      console.error('Error creating refund:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Cancel a subscription
   */
  async cancelSubscription(subscriptionId) {
    try {
      const subscription = await this.stripe.subscriptions.cancel(subscriptionId);

      return {
        success: true,
        subscription_id: subscription.id,
        status: subscription.status
      };
    } catch (error) {
      console.error('Error cancelling subscription:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }
}

module.exports = new StripePaymentService(); 