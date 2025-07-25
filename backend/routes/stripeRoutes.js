const express = require('express');
const router = express.Router();
const stripePaymentService = require('../services/stripePaymentService');
const { authenticate } = require('../middleware/auth');

/**
 * Create a payment intent for material exchange
 */
router.post('/create-payment-intent', authenticate, async (req, res) => {
  try {
    const { exchangeData, amount } = req.body;
    const userId = req.user.id;

    if (!amount || amount <= 0) {
      return res.status(400).json({
        success: false,
        error: 'Invalid amount'
      });
    }

    // Create or get customer
    const customerResult = await stripePaymentService.createCustomer(
      req.user.email,
      req.user.user_metadata?.full_name || req.user.email,
      { user_id: userId }
    );

    if (!customerResult.success) {
      return res.status(500).json({
        success: false,
        error: 'Failed to create customer'
      });
    }

    // Create payment intent
    const paymentIntentResult = await stripePaymentService.createPaymentIntent(
      amount,
      'usd',
      {
        user_id: userId,
        customer_id: customerResult.customer_id,
        exchange_id: exchangeData?.exchange_id,
        material_name: exchangeData?.material_name,
        quantity: exchangeData?.quantity
      }
    );

    if (!paymentIntentResult.success) {
      return res.status(500).json({
        success: false,
        error: paymentIntentResult.error
      });
    }

    // Save payment record to database
    const { createClient } = require('@supabase/supabase-js');
    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    const { error: dbError } = await supabase
      .from('payments')
      .insert({
        user_id: userId,
        stripe_payment_intent_id: paymentIntentResult.payment_intent_id,
        stripe_customer_id: customerResult.customer_id,
        amount: amount,
        currency: 'usd',
        status: 'pending',
        payment_type: 'material_exchange',
        metadata: {
          exchange_id: exchangeData?.exchange_id,
          material_name: exchangeData?.material_name,
          quantity: exchangeData?.quantity
        },
        created_at: new Date().toISOString()
      });

    if (dbError) {
      console.error('Error saving payment record:', dbError);
    }

    res.json({
      success: true,
      payment_intent_id: paymentIntentResult.payment_intent_id,
      client_secret: paymentIntentResult.client_secret,
      amount: paymentIntentResult.amount,
      currency: paymentIntentResult.currency
    });

  } catch (error) {
    console.error('Error creating payment intent:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

/**
 * Create a subscription for premium features
 */
router.post('/create-subscription', authenticate, async (req, res) => {
  try {
    const { subscriptionData, amount } = req.body;
    const userId = req.user.id;

    if (!amount || amount <= 0) {
      return res.status(400).json({
        success: false,
        error: 'Invalid amount'
      });
    }

    // Create or get customer
    const customerResult = await stripePaymentService.createCustomer(
      subscriptionData.customer_email || req.user.email,
      subscriptionData.customer_name || req.user.user_metadata?.full_name || req.user.email,
      { user_id: userId }
    );

    if (!customerResult.success) {
      return res.status(500).json({
        success: false,
        error: 'Failed to create customer'
      });
    }

    // For now, we'll create a payment intent for the first month
    // In production, you'd want to create a proper subscription with a price ID
    const paymentIntentResult = await stripePaymentService.createPaymentIntent(
      amount,
      'usd',
      {
        user_id: userId,
        customer_id: customerResult.customer_id,
        subscription_type: subscriptionData.plan_type,
        plan_name: subscriptionData.plan_name
      }
    );

    if (!paymentIntentResult.success) {
      return res.status(500).json({
        success: false,
        error: paymentIntentResult.error
      });
    }

    // Save subscription record to database
    const { createClient } = require('@supabase/supabase-js');
    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    const { error: dbError } = await supabase
      .from('subscriptions')
      .insert({
        user_id: userId,
        stripe_customer_id: customerResult.customer_id,
        plan_name: subscriptionData.plan_name,
        plan_type: subscriptionData.plan_type,
        monthly_price: amount,
        status: 'pending',
        created_at: new Date().toISOString()
      });

    if (dbError) {
      console.error('Error saving subscription record:', dbError);
    }

    res.json({
      success: true,
      payment_intent_id: paymentIntentResult.payment_intent_id,
      client_secret: paymentIntentResult.client_secret,
      amount: paymentIntentResult.amount,
      currency: paymentIntentResult.currency
    });

  } catch (error) {
    console.error('Error creating subscription:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

/**
 * Handle Stripe webhooks
 */
router.post('/webhook', express.raw({ type: 'application/json' }), async (req, res) => {
  const sig = req.headers['stripe-signature'];
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

  let event;

  try {
    const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
    event = stripe.webhooks.constructEvent(req.body, sig, webhookSecret);
  } catch (err) {
    console.error('Webhook signature verification failed:', err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  try {
    const result = await stripePaymentService.handleWebhook(event);
    
    if (result.success) {
      res.json({ received: true });
    } else {
      console.error('Webhook handling failed:', result.error);
      res.status(500).json({ error: 'Webhook handling failed' });
    }
  } catch (error) {
    console.error('Webhook processing error:', error);
    res.status(500).json({ error: 'Webhook processing failed' });
  }
});

/**
 * Get payment status
 */
router.get('/payment-status/:paymentIntentId', authenticate, async (req, res) => {
  try {
    const { paymentIntentId } = req.params;
    const userId = req.user.id;

    const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
    const paymentIntent = await stripe.paymentIntents.retrieve(paymentIntentId);

    // Verify the payment belongs to the user
    const { createClient } = require('@supabase/supabase-js');
    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    const { data: payment, error } = await supabase
      .from('payments')
      .select('*')
      .eq('stripe_payment_intent_id', paymentIntentId)
      .eq('user_id', userId)
      .single();

    if (error || !payment) {
      return res.status(404).json({
        success: false,
        error: 'Payment not found'
      });
    }

    res.json({
      success: true,
      status: paymentIntent.status,
      amount: paymentIntent.amount / 100,
      currency: paymentIntent.currency,
      created: paymentIntent.created
    });

  } catch (error) {
    console.error('Error retrieving payment status:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

/**
 * Cancel subscription
 */
router.post('/cancel-subscription/:subscriptionId', authenticate, async (req, res) => {
  try {
    const { subscriptionId } = req.params;
    const userId = req.user.id;

    // Verify the subscription belongs to the user
    const { createClient } = require('@supabase/supabase-js');
    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    const { data: subscription, error } = await supabase
      .from('subscriptions')
      .select('*')
      .eq('stripe_subscription_id', subscriptionId)
      .eq('user_id', userId)
      .single();

    if (error || !subscription) {
      return res.status(404).json({
        success: false,
        error: 'Subscription not found'
      });
    }

    const result = await stripePaymentService.cancelSubscription(subscriptionId);

    if (result.success) {
      res.json({
        success: true,
        message: 'Subscription cancelled successfully'
      });
    } else {
      res.status(500).json({
        success: false,
        error: result.error
      });
    }

  } catch (error) {
    console.error('Error cancelling subscription:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

/**
 * Refund payment
 */
router.post('/refund/:paymentIntentId', authenticate, async (req, res) => {
  try {
    const { paymentIntentId } = req.params;
    const { amount } = req.body;
    const userId = req.user.id;

    // Verify the payment belongs to the user
    const { createClient } = require('@supabase/supabase-js');
    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    const { data: payment, error } = await supabase
      .from('payments')
      .select('*')
      .eq('stripe_payment_intent_id', paymentIntentId)
      .eq('user_id', userId)
      .single();

    if (error || !payment) {
      return res.status(404).json({
        success: false,
        error: 'Payment not found'
      });
    }

    const result = await stripePaymentService.refundPayment(paymentIntentId, amount);

    if (result.success) {
      res.json({
        success: true,
        refund_id: result.refund_id,
        amount: result.amount,
        status: result.status
      });
    } else {
      res.status(500).json({
        success: false,
        error: result.error
      });
    }

  } catch (error) {
    console.error('Error creating refund:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
});

module.exports = router; 