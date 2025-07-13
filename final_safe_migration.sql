-- =====================================================
-- FINAL SAFE MIGRATION FOR ISM AI PLATFORM
-- Handles Existing Objects Gracefully
-- =====================================================

-- This migration creates the essential database structure for the ISM AI platform
-- FINAL-SAFE: Handles existing triggers, functions, and tables gracefully

-- =====================================================
-- DROP EXISTING TRIGGERS FIRST
-- =====================================================

-- Drop existing triggers if they exist
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
DROP TRIGGER IF EXISTS update_companies_updated_at ON companies;
DROP TRIGGER IF EXISTS update_materials_updated_at ON materials;
DROP TRIGGER IF EXISTS update_products_updated_at ON products;
DROP TRIGGER IF EXISTS update_ai_matches_updated_at ON ai_matches;
DROP TRIGGER IF EXISTS update_ai_insights_updated_at ON ai_insights;
DROP TRIGGER IF EXISTS update_ai_recommendations_updated_at ON ai_recommendations;

-- =====================================================
-- CORE TABLES
-- =====================================================

-- Users table (simplified)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    company_name VARCHAR(255),
    subscription_tier VARCHAR(50) DEFAULT 'free',
    onboarding_completed BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Companies table
CREATE TABLE IF NOT EXISTS companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    location VARCHAR(255),
    employee_count INTEGER,
    products TEXT,
    main_materials TEXT,
    production_volume VARCHAR(100),
    process_description TEXT,
    sustainability_goals TEXT[],
    current_waste_management VARCHAR(100),
    onboarding_completed BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- MATERIALS AND PRODUCTS
-- =====================================================

-- Materials table
CREATE TABLE IF NOT EXISTS materials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    quantity DECIMAL(15,2),
    unit VARCHAR(50),
    cost_per_unit DECIMAL(10,2),
    supplier VARCHAR(255),
    sustainability_rating DECIMAL(3,2),
    carbon_footprint DECIMAL(10,2),
    recyclability_percentage DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Products table
CREATE TABLE IF NOT EXISTS products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    production_volume INTEGER,
    unit_cost DECIMAL(10,2),
    selling_price DECIMAL(10,2),
    materials_used JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- AI MATCHING SYSTEM
-- =====================================================

-- AI matches table
CREATE TABLE IF NOT EXISTS ai_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_score DECIMAL(3,2) CHECK (match_score >= 0 AND match_score <= 1),
    match_type VARCHAR(50),
    potential_savings DECIMAL(15,2),
    carbon_reduction DECIMAL(10,2),
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Symbiosis opportunities table
CREATE TABLE IF NOT EXISTS symbiosis_opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region VARCHAR(100),
    industry VARCHAR(100),
    opportunity_type VARCHAR(50),
    description TEXT,
    potential_savings DECIMAL(15,2),
    carbon_reduction_potential DECIMAL(10,2),
    companies_involved INTEGER,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- ANALYTICS AND METRICS
-- =====================================================

-- User analytics
CREATE TABLE IF NOT EXISTS user_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_duration INTEGER,
    features_used TEXT[],
    matches_viewed INTEGER,
    matches_accepted INTEGER,
    total_savings DECIMAL(15,2),
    carbon_reduction DECIMAL(10,2),
    date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Business metrics
CREATE TABLE IF NOT EXISTS business_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    revenue DECIMAL(15,2),
    cost_savings DECIMAL(15,2),
    carbon_reduction DECIMAL(10,2),
    waste_reduction_percentage DECIMAL(5,2),
    efficiency_improvement DECIMAL(5,2),
    period VARCHAR(20),
    metric_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_type VARCHAR(50) NOT NULL,
    value DECIMAL(10,2),
    unit VARCHAR(50),
    improvement_percentage DECIMAL(5,2),
    baseline_value DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- AI INSIGHTS AND RECOMMENDATIONS
-- =====================================================

-- AI insights table
CREATE TABLE IF NOT EXISTS ai_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    insight_type VARCHAR(50) NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    impact_level VARCHAR(20),
    estimated_savings DECIMAL(15,2),
    carbon_reduction DECIMAL(10,2),
    action_required BOOLEAN DEFAULT false,
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    status VARCHAR(50) DEFAULT 'new',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI recommendations table
CREATE TABLE IF NOT EXISTS ai_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    recommendation_type VARCHAR(50),
    title VARCHAR(200),
    description TEXT,
    implementation_steps JSONB,
    estimated_cost DECIMAL(15,2),
    expected_roi DECIMAL(5,2),
    priority VARCHAR(20),
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- COMPETITIVE INTELLIGENCE
-- =====================================================

-- Competitor data
CREATE TABLE IF NOT EXISTS competitors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    location VARCHAR(255),
    sustainability_score DECIMAL(3,2),
    market_share DECIMAL(5,2),
    recent_activities JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market analysis
CREATE TABLE IF NOT EXISTS market_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    market_segment VARCHAR(100),
    growth_rate DECIMAL(5,2),
    sustainability_trends JSONB,
    regulatory_changes TEXT[],
    opportunities TEXT[],
    analysis_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- LOGISTICS AND SHIPPING
-- =====================================================

-- Shipping routes
CREATE TABLE IF NOT EXISTS shipping_routes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    distance_km DECIMAL(10,2),
    transport_mode VARCHAR(50),
    cost_per_km DECIMAL(10,2),
    carbon_footprint DECIMAL(10,2),
    estimated_duration_hours INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Logistics optimization
CREATE TABLE IF NOT EXISTS logistics_optimization (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    optimization_type VARCHAR(50),
    savings_percentage DECIMAL(5,2),
    carbon_reduction DECIMAL(10,2),
    implementation_cost DECIMAL(15,2),
    payback_period_months INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- SUBSCRIPTION AND BILLING
-- =====================================================

-- Subscription plans
CREATE TABLE IF NOT EXISTS subscription_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2),
    features JSONB,
    ai_credits INTEGER,
    max_companies INTEGER,
    max_matches_per_month INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User subscriptions
CREATE TABLE IF NOT EXISTS user_subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(50) DEFAULT 'active',
    start_date DATE DEFAULT CURRENT_DATE,
    end_date DATE,
    payment_method VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- NOTIFICATIONS AND MESSAGING
-- =====================================================

-- Notifications table
CREATE TABLE IF NOT EXISTS notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(200),
    message TEXT,
    type VARCHAR(50),
    read BOOLEAN DEFAULT false,
    action_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject VARCHAR(200),
    content TEXT,
    read BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- SYSTEM MONITORING
-- =====================================================

-- System health
CREATE TABLE IF NOT EXISTS system_health (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    health_score DECIMAL(3,2) CHECK (health_score >= 0 AND health_score <= 1),
    cpu_usage DECIMAL(5,2),
    memory_usage DECIMAL(5,2),
    disk_usage DECIMAL(5,2),
    active_users INTEGER,
    api_requests_per_minute INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    endpoint VARCHAR(200),
    method VARCHAR(10),
    status_code INTEGER,
    response_time INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- User indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Company indexes
CREATE INDEX IF NOT EXISTS idx_companies_industry ON companies(industry);
CREATE INDEX IF NOT EXISTS idx_companies_location ON companies(location);

-- Material indexes
CREATE INDEX IF NOT EXISTS idx_materials_category ON materials(category);

-- AI match indexes
CREATE INDEX IF NOT EXISTS idx_ai_matches_score ON ai_matches(match_score);
CREATE INDEX IF NOT EXISTS idx_ai_matches_status ON ai_matches(status);

-- Analytics indexes
CREATE INDEX IF NOT EXISTS idx_user_analytics_date ON user_analytics(date);

-- Business metrics indexes
CREATE INDEX IF NOT EXISTS idx_business_metrics_period ON business_metrics(period);

-- AI insights indexes
CREATE INDEX IF NOT EXISTS idx_ai_insights_type ON ai_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_ai_insights_status ON ai_insights(status);

-- API usage indexes
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON api_usage(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_usage_created_at ON api_usage(created_at);

-- =====================================================
-- SAMPLE DATA INSERTION
-- =====================================================

-- Insert sample subscription plans
INSERT INTO subscription_plans (name, price, features, ai_credits, max_companies, max_matches_per_month) VALUES
    ('Free', 0.00, '["Basic matching", "Limited insights"]', 100, 1, 10),
    ('Professional', 99.00, '["Advanced matching", "Full insights", "Priority support"]', 1000, 5, 100),
    ('Enterprise', 299.00, '["Unlimited matching", "Custom insights", "Dedicated support"]', 10000, 50, 1000)
ON CONFLICT DO NOTHING;

-- Insert sample performance metrics
INSERT INTO performance_metrics (metric_type, value, unit, improvement_percentage, baseline_value) VALUES
    ('response_time', 150, 'ms', 25.0, 200),
    ('matching_accuracy', 0.92, 'percentage', 15.0, 0.80),
    ('user_satisfaction', 4.5, 'stars', 12.5, 4.0),
    ('carbon_reduction', 1250, 'kg CO2', 30.0, 962)
ON CONFLICT DO NOTHING;

-- Insert sample system health
INSERT INTO system_health (health_score, cpu_usage, memory_usage, disk_usage, active_users, api_requests_per_minute) VALUES
    (0.98, 15.5, 45.2, 30.1, 150, 1200)
ON CONFLICT DO NOTHING;

-- =====================================================
-- TRIGGERS FOR UPDATED_AT
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at (using IF NOT EXISTS pattern)
DO $$
BEGIN
    -- Users trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_users_updated_at') THEN
        CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
    
    -- Companies trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_companies_updated_at') THEN
        CREATE TRIGGER update_companies_updated_at BEFORE UPDATE ON companies
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
    
    -- Materials trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_materials_updated_at') THEN
        CREATE TRIGGER update_materials_updated_at BEFORE UPDATE ON materials
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
    
    -- Products trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_products_updated_at') THEN
        CREATE TRIGGER update_products_updated_at BEFORE UPDATE ON products
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
    
    -- AI matches trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_ai_matches_updated_at') THEN
        CREATE TRIGGER update_ai_matches_updated_at BEFORE UPDATE ON ai_matches
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
    
    -- AI insights trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_ai_insights_updated_at') THEN
        CREATE TRIGGER update_ai_insights_updated_at BEFORE UPDATE ON ai_insights
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
    
    -- AI recommendations trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_ai_recommendations_updated_at') THEN
        CREATE TRIGGER update_ai_recommendations_updated_at BEFORE UPDATE ON ai_recommendations
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- =====================================================
-- MIGRATION COMPLETION
-- =====================================================

-- Log migration completion
INSERT INTO system_health (health_score, cpu_usage, memory_usage, disk_usage, active_users, api_requests_per_minute)
VALUES (1.0, 0, 0, 0, 0, 0)
ON CONFLICT DO NOTHING;

-- Print completion message
DO $$
BEGIN
    RAISE NOTICE '🎉 FINAL SAFE MIGRATION SUCCESSFUL!';
    RAISE NOTICE '📊 Created/Updated 20+ essential tables';
    RAISE NOTICE '⚡ Created performance indexes';
    RAISE NOTICE '🔄 Added updated_at triggers safely';
    RAISE NOTICE '📈 Inserted sample data';
    RAISE NOTICE '🚀 ISM AI Platform foundation ready!';
    RAISE NOTICE '💡 Next: Add relationships and RLS policies';
END $$; 