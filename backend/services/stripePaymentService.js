/* eslint-disable camelcase */
/**
 * StripePaymentService
 * --------------------
 * Production-grade service wrapper around the Stripe SDK.
 * Responsibilities:
 * 1. Create checkout sessions for one-time payments or subscriptions.
 * 2. Verify and handle webhook events.
 * 3. Persist payment metadata to Supabase.
 *
 * Stripe best-practice notes:
 *   • Use API version locked via dashboard.
 *   • Verify all webhook signatures.
 *   • Keep idempotency keys for retries.
 *   • Store prices / product IDs in env or DB – never trust the client.
 */

const Stripe = require('stripe');
const { createClient } = require('@supabase/supabase-js');
const { v4: uuidv4 } = require('uuid');

class StripePaymentService {
  constructor () {
    const secretKey = process.env.STRIPE_SECRET_KEY;
    if (!secretKey) {
      throw new Error('STRIPE_SECRET_KEY is not configured');
    }
    this.stripe = Stripe(secretKey, {
      apiVersion: '2023-10-16'
    });

    // Supabase client used to persist payment data
    this.supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

    this.webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  }

  /**
   * Create a checkout session (one-time payment or subscription)
   * @param {Object} params – expects { customerEmail, lineItems, mode, metadata }
   * @returns {Promise<Stripe.Checkout.Session>}
   */
  async createCheckoutSession (params) {
    const {
      customerEmail,
      lineItems = [],
      mode = 'payment',
      metadata = {}
    } = params;

    if (!Array.isArray(lineItems) || lineItems.length === 0) {
      throw new Error('At least one line item is required');
    }

    const session = await this.stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      mode,
      line_items: lineItems,
      success_url: `${process.env.FRONTEND_URL}/payment/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${process.env.FRONTEND_URL}/payment/cancel`,
      customer_email: customerEmail,
      metadata
    });

    // Persist initial payment record
    await this.supabase.from('payments').insert({
      id: uuidv4(),
      stripe_session_id: session.id,
      status: 'pending',
      amount_total: session.amount_total,
      currency: session.currency,
      metadata
    });

    return session;
  }

  /**
   * Express handler for Stripe webhooks.
   * Ensure you pass the raw body to this function.
   */
  async handleWebhook (req, res) {
    if (!this.webhookSecret) {
      console.warn('Stripe webhook secret not configured – rejecting');
      return res.status(400).send('Webhook not configured');
    }

    let event;
    try {
      event = this.stripe.webhooks.constructEvent(
        req.rawBody,
        req.headers['stripe-signature'],
        this.webhookSecret
      );
    } catch (err) {
      console.error('⚠️  Webhook signature verification failed.', err.message);
      return res.status(400).send(`Webhook Error: ${err.message}`);
    }

    // Handle the event
    switch (event.type) {
      case 'checkout.session.completed':
        await this.#handleCheckoutCompleted(event.data.object);
        break;
      case 'invoice.payment_succeeded':
        await this.#handleInvoicePaid(event.data.object);
        break;
      case 'invoice.payment_failed':
        await this.#handleInvoiceFailed(event.data.object);
        break;
      default:
        console.log(`Unhandled event type ${event.type}`);
    }

    res.json({ received: true });
  }

  /**
   * Private helpers – update Supabase records accordingly
   */
  async #handleCheckoutCompleted (session) {
    await this.supabase
      .from('payments')
      .update({ status: 'succeeded', stripe_payment_intent: session.payment_intent })
      .eq('stripe_session_id', session.id);
  }

  async #handleInvoicePaid (invoice) {
    await this.supabase
      .from('subscriptions')
      .update({ status: 'active' })
      .eq('stripe_subscription_id', invoice.subscription);
  }

  async #handleInvoiceFailed (invoice) {
    await this.supabase
      .from('subscriptions')
      .update({ status: 'past_due' })
      .eq('stripe_subscription_id', invoice.subscription);
  }
}

module.exports = new StripePaymentService();