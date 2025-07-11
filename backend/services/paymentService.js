const paypal = require('@paypal/checkout-server-sdk');
const { supabase } = require('../supabase');

class PaymentService {
  constructor() {
    this.environment = process.env.NODE_ENV === 'production' 
      ? new paypal.core.LiveEnvironment(process.env.PAYPAL_CLIENT_ID, process.env.PAYPAL_CLIENT_SECRET)
      : new paypal.core.SandboxEnvironment(process.env.PAYPAL_CLIENT_ID, process.env.PAYPAL_CLIENT_SECRET);
    
    this.client = new paypal.core.PayPalHttpClient(this.environment);
  }

  /**
   * Create a PayPal order for material exchange
   */
  async createOrder(exchangeData) {
    try {
      const request = new paypal.orders.OrdersCreateRequest();
      request.prefer("return=representation");
      request.requestBody({
        intent: 'CAPTURE',
        purchase_units: [{
          amount: {
            currency_code: 'USD',
            value: exchangeData.total_amount,
            breakdown: {
              item_total: {
                currency_code: 'USD',
                value: exchangeData.material_cost
              },
              shipping: {
                currency_code: 'USD',
                value: exchangeData.shipping_cost
              },
              tax_total: {
                currency_code: 'USD',
                value: exchangeData.tax_amount || '0.00'
              },
              handling: {
                currency_code: 'USD',
                value: exchangeData.platform_fee || '0.00'
              }
            }
          },
          items: [{
            name: exchangeData.material_name,
            description: `Material exchange: ${exchangeData.material_name}`,
            quantity: exchangeData.quantity,
            unit_amount: {
              currency_code: 'USD',
              value: exchangeData.unit_price
            },
            category: 'PHYSICAL_GOODS'
          }],
          shipping: {
            address: {
              address_line_1: exchangeData.shipping_address.street1,
              address_line_2: exchangeData.shipping_address.street2,
              admin_area_2: exchangeData.shipping_address.city,
              admin_area_1: exchangeData.shipping_address.state,
              postal_code: exchangeData.shipping_address.zip,
              country_code: exchangeData.shipping_address.country
            }
          }
        }],
        application_context: {
          return_url: `${process.env.FRONTEND_URL}/payment/success`,
          cancel_url: `${process.env.FRONTEND_URL}/payment/cancel`,
          shipping_preference: 'SET_PROVIDED_ADDRESS'
        }
      });

      const order = await this.client.execute(request);
      
      // Save order to database
      const { error } = await supabase
        .from('payment_orders')
        .insert({
          paypal_order_id: order.result.id,
          exchange_id: exchangeData.exchange_id,
          amount: exchangeData.total_amount,
          currency: 'USD',
          status: 'CREATED',
          created_at: new Date().toISOString()
        });

      if (error) throw error;

      return {
        orderId: order.result.id,
        approvalUrl: order.result.links.find(link => link.rel === 'approve').href
      };
    } catch (error) {
      console.error('Error creating PayPal order:', error);
      throw error;
    }
  }

  /**
   * Capture a PayPal payment
   */
  async capturePayment(orderId) {
    try {
      const request = new paypal.orders.OrdersCaptureRequest(orderId);
      request.requestBody({});

      const capture = await this.client.execute(request);
      
      // Update order status in database
      const { error } = await supabase
        .from('payment_orders')
        .update({
          status: 'COMPLETED',
          capture_id: capture.result.purchase_units[0].payments.captures[0].id,
          updated_at: new Date().toISOString()
        })
        .eq('paypal_order_id', orderId);

      if (error) throw error;

      return {
        captureId: capture.result.purchase_units[0].payments.captures[0].id,
        status: capture.result.status,
        amount: capture.result.purchase_units[0].payments.captures[0].amount.value
      };
    } catch (error) {
      console.error('Error capturing PayPal payment:', error);
      throw error;
    }
  }

  /**
   * Create subscription for Pro/Enterprise plans
   */
  async createSubscription(subscriptionData) {
    try {
      const request = new paypal.catalogs.ProductsPostRequest();
      request.requestBody({
        name: subscriptionData.plan_name,
        description: subscriptionData.description,
        type: 'SERVICE',
        category: 'SOFTWARE',
        image_url: 'https://example.com/logo.png',
        home_url: process.env.FRONTEND_URL
      });

      const product = await this.client.execute(request);

      // Create billing plan
      const billingPlanRequest = new paypal.catalogs.PlansPostRequest();
      billingPlanRequest.requestBody({
        product_id: product.result.id,
        name: subscriptionData.plan_name,
        description: subscriptionData.description,
        status: 'ACTIVE',
        billing_cycles: [{
          frequency: {
            interval_unit: 'MONTH',
            interval_count: 1
          },
          tenure_type: 'REGULAR',
          sequence: 1,
          total_cycles: 0,
          pricing_scheme: {
            fixed_price: {
              value: subscriptionData.monthly_price,
              currency_code: 'USD'
            }
          }
        }],
        payment_preferences: {
          auto_bill_outstanding: true,
          setup_fee: {
            value: '0',
            currency_code: 'USD'
          },
          setup_fee_failure_action: 'CONTINUE',
          payment_failure_threshold: 3
        }
      });

      const billingPlan = await this.client.execute(billingPlanRequest);

      // Create subscription
      const subscriptionRequest = new paypal.billing.SubscriptionsPostRequest();
      subscriptionRequest.requestBody({
        plan_id: billingPlan.result.id,
        start_time: new Date(Date.now() + 60000).toISOString(), // Start in 1 minute
        subscriber: {
          name: {
            given_name: subscriptionData.customer_name
          },
          email_address: subscriptionData.customer_email
        },
        application_context: {
          brand_name: 'SymbioFlows',
          locale: 'en-US',
          shipping_preference: 'NO_SHIPPING',
          user_action: 'SUBSCRIBE_NOW',
          payment_method: {
            payer_selected: 'PAYPAL',
            payee_preferred: 'IMMEDIATE_PAYMENT_REQUIRED'
          },
          return_url: `${process.env.FRONTEND_URL}/subscription/success`,
          cancel_url: `${process.env.FRONTEND_URL}/subscription/cancel`
        }
      });

      const subscription = await this.client.execute(subscriptionRequest);

      // Save subscription to database
      const { error } = await supabase
        .from('subscriptions')
        .insert({
          paypal_subscription_id: subscription.result.id,
          company_id: subscriptionData.company_id,
          plan_name: subscriptionData.plan_name,
          monthly_price: subscriptionData.monthly_price,
          status: 'ACTIVE',
          created_at: new Date().toISOString()
        });

      if (error) throw error;

      return {
        subscriptionId: subscription.result.id,
        approvalUrl: subscription.result.links.find(link => link.rel === 'approve').href
      };
    } catch (error) {
      console.error('Error creating PayPal subscription:', error);
      throw error;
    }
  }

  /**
   * Process refund
   */
  async processRefund(captureId, amount, reason) {
    try {
      const request = new paypal.payments.CapturesRefundRequest(captureId);
      request.requestBody({
        amount: {
          value: amount,
          currency_code: 'USD'
        },
        note_to_payer: reason
      });

      const refund = await this.client.execute(request);

      // Save refund to database
      const { error } = await supabase
        .from('refunds')
        .insert({
          paypal_refund_id: refund.result.id,
          capture_id: captureId,
          amount: amount,
          reason: reason,
          status: 'COMPLETED',
          created_at: new Date().toISOString()
        });

      if (error) throw error;

      return {
        refundId: refund.result.id,
        status: refund.result.status,
        amount: refund.result.amount.value
      };
    } catch (error) {
      console.error('Error processing refund:', error);
      throw error;
    }
  }

  /**
   * Get payment analytics
   */
  async getPaymentAnalytics(companyId, dateRange = '30') {
    try {
      const { data: orders, error } = await supabase
        .from('payment_orders')
        .select('*')
        .eq('company_id', companyId)
        .gte('created_at', new Date(Date.now() - dateRange * 24 * 60 * 60 * 1000).toISOString())
        .order('created_at', { ascending: false });

      if (error) throw error;

      const analytics = {
        total_orders: orders.length,
        total_revenue: orders.reduce((sum, order) => sum + parseFloat(order.amount), 0),
        successful_orders: orders.filter(order => order.status === 'COMPLETED').length,
        failed_orders: orders.filter(order => order.status === 'FAILED').length,
        average_order_value: orders.length > 0 ? 
          orders.reduce((sum, order) => sum + parseFloat(order.amount), 0) / orders.length : 0
      };

      return analytics;
    } catch (error) {
      console.error('Error getting payment analytics:', error);
      throw error;
    }
  }

  /**
   * Webhook handler for PayPal events
   */
  async handleWebhook(event) {
    try {
      const { event_type, resource } = event;

      switch (event_type) {
        case 'PAYMENT.CAPTURE.COMPLETED':
          await this.handlePaymentCompleted(resource);
          break;
        case 'PAYMENT.CAPTURE.DENIED':
          await this.handlePaymentDenied(resource);
          break;
        case 'BILLING.SUBSCRIPTION.ACTIVATED':
          await this.handleSubscriptionActivated(resource);
          break;
        case 'BILLING.SUBSCRIPTION.CANCELLED':
          await this.handleSubscriptionCancelled(resource);
          break;
        default:
          console.log(`Unhandled webhook event: ${event_type}`);
      }
    } catch (error) {
      console.error('Error handling webhook:', error);
      throw error;
    }
  }

  async handlePaymentCompleted(resource) {
    // Update exchange status to paid
    const { error } = await supabase
      .from('material_exchanges')
      .update({ 
        payment_status: 'PAID',
        paid_at: new Date().toISOString()
      })
      .eq('paypal_capture_id', resource.id);

    if (error) throw error;
  }

  async handlePaymentDenied(resource) {
    // Update exchange status to payment failed
    const { error } = await supabase
      .from('material_exchanges')
      .update({ 
        payment_status: 'FAILED',
        payment_failed_at: new Date().toISOString()
      })
      .eq('paypal_capture_id', resource.id);

    if (error) throw error;
  }

  async handleSubscriptionActivated(resource) {
    // Update subscription status
    const { error } = await supabase
      .from('subscriptions')
      .update({ 
        status: 'ACTIVE',
        activated_at: new Date().toISOString()
      })
      .eq('paypal_subscription_id', resource.id);

    if (error) throw error;
  }

  async handleSubscriptionCancelled(resource) {
    // Update subscription status
    const { error } = await supabase
      .from('subscriptions')
      .update({ 
        status: 'CANCELLED',
        cancelled_at: new Date().toISOString()
      })
      .eq('paypal_subscription_id', resource.id);

    if (error) throw error;
  }
}

module.exports = new PaymentService(); 