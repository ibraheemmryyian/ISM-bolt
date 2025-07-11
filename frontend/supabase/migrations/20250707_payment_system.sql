-- Payment System Migration
-- Creates tables for PayPal integration and payment processing

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Payment Orders Table
CREATE TABLE IF NOT EXISTS payment_orders (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    paypal_order_id VARCHAR(255) UNIQUE NOT NULL,
    exchange_id UUID REFERENCES material_exchanges(id) ON DELETE CASCADE,
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    status VARCHAR(50) DEFAULT 'CREATED',
    capture_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Subscriptions Table
CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    paypal_subscription_id VARCHAR(255) UNIQUE NOT NULL,
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    plan_name VARCHAR(100) NOT NULL,
    plan_type VARCHAR(50) NOT NULL, -- 'PRO', 'ENTERPRISE', 'CUSTOM'
    monthly_price DECIMAL(10,2) NOT NULL,
    status VARCHAR(50) DEFAULT 'PENDING',
    activated_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    next_billing_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Refunds Table
CREATE TABLE IF NOT EXISTS refunds (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    paypal_refund_id VARCHAR(255) UNIQUE NOT NULL,
    capture_id VARCHAR(255) NOT NULL,
    order_id UUID REFERENCES payment_orders(id) ON DELETE CASCADE,
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    reason TEXT,
    status VARCHAR(50) DEFAULT 'PENDING',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Payment Analytics Table
CREATE TABLE IF NOT EXISTS payment_analytics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    total_orders INTEGER DEFAULT 0,
    total_revenue DECIMAL(10,2) DEFAULT 0,
    successful_orders INTEGER DEFAULT 0,
    failed_orders INTEGER DEFAULT 0,
    average_order_value DECIMAL(10,2) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(company_id, date)
);

-- Payment Methods Table (for future multi-provider support)
CREATE TABLE IF NOT EXISTS payment_methods (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL, -- 'PAYPAL', 'STRIPE', 'ADYEN', etc.
    method_type VARCHAR(50) NOT NULL, -- 'CARD', 'BANK', 'WALLET'
    is_default BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add payment-related columns to existing tables
ALTER TABLE material_exchanges 
ADD COLUMN IF NOT EXISTS payment_status VARCHAR(50) DEFAULT 'PENDING',
ADD COLUMN IF NOT EXISTS payment_method VARCHAR(50),
ADD COLUMN IF NOT EXISTS paypal_capture_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS paid_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS payment_failed_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE companies 
ADD COLUMN IF NOT EXISTS subscription_plan VARCHAR(50) DEFAULT 'FREE',
ADD COLUMN IF NOT EXISTS subscription_status VARCHAR(50) DEFAULT 'ACTIVE',
ADD COLUMN IF NOT EXISTS subscription_expires_at TIMESTAMP WITH TIME ZONE;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_payment_orders_company_id ON payment_orders(company_id);
CREATE INDEX IF NOT EXISTS idx_payment_orders_status ON payment_orders(status);
CREATE INDEX IF NOT EXISTS idx_payment_orders_created_at ON payment_orders(created_at);
CREATE INDEX IF NOT EXISTS idx_subscriptions_company_id ON subscriptions(company_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);
CREATE INDEX IF NOT EXISTS idx_refunds_order_id ON refunds(order_id);
CREATE INDEX IF NOT EXISTS idx_payment_analytics_company_date ON payment_analytics(company_id, date);
CREATE INDEX IF NOT EXISTS idx_material_exchanges_payment_status ON material_exchanges(payment_status);

-- Create updated_at triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_payment_orders_updated_at 
    BEFORE UPDATE ON payment_orders 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subscriptions_updated_at 
    BEFORE UPDATE ON subscriptions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_refunds_updated_at 
    BEFORE UPDATE ON refunds 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_payment_methods_updated_at 
    BEFORE UPDATE ON payment_methods 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) Policies
ALTER TABLE payment_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE refunds ENABLE ROW LEVEL SECURITY;
ALTER TABLE payment_analytics ENABLE ROW LEVEL SECURITY;
ALTER TABLE payment_methods ENABLE ROW LEVEL SECURITY;

-- Payment Orders RLS
CREATE POLICY "Users can view their own payment orders" ON payment_orders
    FOR SELECT USING (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can create payment orders" ON payment_orders
    FOR INSERT WITH CHECK (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can update their own payment orders" ON payment_orders
    FOR UPDATE USING (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

-- Subscriptions RLS
CREATE POLICY "Users can view their own subscriptions" ON subscriptions
    FOR SELECT USING (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can create subscriptions" ON subscriptions
    FOR INSERT WITH CHECK (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can update their own subscriptions" ON subscriptions
    FOR UPDATE USING (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

-- Refunds RLS
CREATE POLICY "Users can view their own refunds" ON refunds
    FOR SELECT USING (
        order_id IN (
            SELECT id FROM payment_orders WHERE company_id IN (
                SELECT id FROM companies WHERE user_id = auth.uid()
            )
        )
    );

CREATE POLICY "Users can create refunds" ON refunds
    FOR INSERT WITH CHECK (
        order_id IN (
            SELECT id FROM payment_orders WHERE company_id IN (
                SELECT id FROM companies WHERE user_id = auth.uid()
            )
        )
    );

-- Payment Analytics RLS
CREATE POLICY "Users can view their own payment analytics" ON payment_analytics
    FOR SELECT USING (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can create payment analytics" ON payment_analytics
    FOR INSERT WITH CHECK (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

-- Payment Methods RLS
CREATE POLICY "Users can view their own payment methods" ON payment_methods
    FOR SELECT USING (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can manage their own payment methods" ON payment_methods
    FOR ALL USING (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

-- Insert sample subscription plans
INSERT INTO subscriptions (paypal_subscription_id, company_id, plan_name, plan_type, monthly_price, status)
VALUES 
    ('sample-pro-plan', '00000000-0000-0000-0000-000000000000', 'Pro Plan', 'PRO', 99.00, 'ACTIVE'),
    ('sample-enterprise-plan', '00000000-0000-0000-0000-000000000000', 'Enterprise Plan', 'ENTERPRISE', 299.00, 'ACTIVE')
ON CONFLICT (paypal_subscription_id) DO NOTHING;

-- Create views for easier querying
CREATE OR REPLACE VIEW payment_summary AS
SELECT 
    po.id,
    po.paypal_order_id,
    po.amount,
    po.currency,
    po.status,
    po.created_at,
    c.company_name,
    c.user_id
FROM payment_orders po
JOIN companies c ON po.company_id = c.id;

CREATE OR REPLACE VIEW subscription_summary AS
SELECT 
    s.id,
    s.paypal_subscription_id,
    s.plan_name,
    s.plan_type,
    s.monthly_price,
    s.status,
    s.activated_at,
    s.next_billing_date,
    c.company_name,
    c.user_id
FROM subscriptions s
JOIN companies c ON s.company_id = c.id;

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE ON payment_orders TO authenticated;
GRANT SELECT, INSERT, UPDATE ON subscriptions TO authenticated;
GRANT SELECT, INSERT ON refunds TO authenticated;
GRANT SELECT, INSERT ON payment_analytics TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON payment_methods TO authenticated;
GRANT SELECT ON payment_summary TO authenticated;
GRANT SELECT ON subscription_summary TO authenticated; 