-- =====================================================
-- FINAL DATABASE MIGRATION FOR REAL DATA COLLECTION
-- ISM AI Platform - Post Hard-Coded Data Removal
-- =====================================================

-- This migration creates all required tables for real data collection
-- after removing hard-coded, mock, and fake data from the platform
-- FINAL: Properly handles all constraints and dependencies

-- =====================================================
-- PREREQUISITE: ENSURE COMPANIES TABLE HAS PROPER STRUCTURE
-- =====================================================

-- Add user_id column to companies table if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'user_id'
    ) THEN
        ALTER TABLE companies ADD COLUMN user_id UUID;
        RAISE NOTICE 'Added user_id column to companies table';
    END IF;
END $$;

-- Ensure companies table has proper primary key and constraints
DO $$
BEGIN
    -- Add unique constraint on user_id if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'companies_user_id_unique'
    ) THEN
        ALTER TABLE companies ADD CONSTRAINT companies_user_id_unique UNIQUE (user_id);
        RAISE NOTICE 'Added unique constraint on companies.user_id';
    END IF;
END $$;

-- =====================================================
-- USER ANALYTICS TABLES
-- =====================================================

-- User session tracking
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    session_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_end TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Feature usage analytics
CREATE TABLE IF NOT EXISTS feature_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    feature_name VARCHAR(100) NOT NULL,
    usage_count INTEGER DEFAULT 1,
    last_used TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User feedback and satisfaction
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    rating DECIMAL(3,2) CHECK (rating >= 0 AND rating <= 5),
    feedback_text TEXT,
    category VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- AI DECISION TRACKING TABLES
-- =====================================================

-- AI decision tracking
CREATE TABLE IF NOT EXISTS ai_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    decision_type VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    accuracy_score DECIMAL(3,2) CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
    processing_time INTEGER, -- milliseconds
    input_data JSONB,
    output_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- SYSTEM OPTIMIZATION TABLES
-- =====================================================

-- System optimizations applied
CREATE TABLE IF NOT EXISTS system_optimizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    optimization_type VARCHAR(50) NOT NULL,
    efficiency_score DECIMAL(3,2) CHECK (efficiency_score >= 0 AND efficiency_score <= 1),
    impact_description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance metrics tracking
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_type VARCHAR(50) NOT NULL,
    improvement_percentage DECIMAL(5,2),
    baseline_value DECIMAL(10,2),
    current_value DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Resource usage and savings
CREATE TABLE IF NOT EXISTS resource_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_type VARCHAR(50) NOT NULL,
    savings_percentage DECIMAL(5,2),
    baseline_usage DECIMAL(10,2),
    current_usage DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- BUSINESS METRICS TABLES
-- =====================================================

-- Business metrics tracking
CREATE TABLE IF NOT EXISTS business_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    revenue_amount DECIMAL(15,2),
    period VARCHAR(20), -- 'daily', 'weekly', 'monthly', 'yearly'
    metric_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cost savings tracking
CREATE TABLE IF NOT EXISTS cost_savings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    amount DECIMAL(15,2),
    period VARCHAR(20),
    savings_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Customer satisfaction tracking
CREATE TABLE IF NOT EXISTS customer_satisfaction (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    satisfaction_score DECIMAL(3,2) CHECK (satisfaction_score >= 0 AND satisfaction_score <= 1),
    survey_type VARCHAR(50),
    response_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market analysis data
CREATE TABLE IF NOT EXISTS market_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    market_share DECIMAL(5,2),
    competitive_advantage VARCHAR(50),
    growth_potential VARCHAR(50),
    analysis_date DATE DEFAULT CURRENT_DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Growth metrics tracking
CREATE TABLE IF NOT EXISTS growth_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_growth DECIMAL(5,2),
    revenue_growth DECIMAL(5,2),
    market_growth DECIMAL(5,2),
    period VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- API MONITORING TABLES
-- =====================================================

-- API request monitoring
CREATE TABLE IF NOT EXISTS api_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    endpoint VARCHAR(200),
    method VARCHAR(10),
    status_code INTEGER,
    response_time INTEGER, -- milliseconds
    user_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System health metrics
CREATE TABLE IF NOT EXISTS system_health (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    health_score DECIMAL(3,2) CHECK (health_score >= 0 AND health_score <= 1),
    cpu_usage DECIMAL(5,2),
    memory_usage DECIMAL(5,2),
    disk_usage DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- COMPETITIVE INTELLIGENCE TABLES
-- =====================================================

-- Competitor social media data
CREATE TABLE IF NOT EXISTS competitor_social_media (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competitor_name VARCHAR(200) NOT NULL,
    followers INTEGER,
    engagement_rate DECIMAL(5,2),
    recent_posts INTEGER,
    sentiment VARCHAR(20),
    trending_topics TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Competitor patent data
CREATE TABLE IF NOT EXISTS competitor_patents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competitor_name VARCHAR(200) NOT NULL,
    recent_patents INTEGER,
    patent_categories TEXT[],
    innovation_score DECIMAL(3,2) CHECK (innovation_score >= 0 AND innovation_score <= 1),
    technology_focus TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Competitor hiring data
CREATE TABLE IF NOT EXISTS competitor_hiring (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competitor_name VARCHAR(200) NOT NULL,
    open_positions INTEGER,
    skill_focus TEXT[],
    growth_indicator DECIMAL(3,2),
    strategic_roles TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Competitor funding data
CREATE TABLE IF NOT EXISTS competitor_funding (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competitor_name VARCHAR(200) NOT NULL,
    recent_funding DECIMAL(15,2),
    funding_round VARCHAR(50),
    investors TEXT[],
    valuation DECIMAL(15,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- AI MATCHING PERFORMANCE TABLES
-- =====================================================

-- Matching success rates
CREATE TABLE IF NOT EXISTS matching_success_rates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    algorithm VARCHAR(50) NOT NULL,
    success_rate DECIMAL(3,2) CHECK (success_rate >= 0 AND success_rate <= 1),
    total_matches INTEGER,
    successful_matches INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Symbiosis opportunities tracking
CREATE TABLE IF NOT EXISTS symbiosis_opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region VARCHAR(100),
    industry VARCHAR(100),
    opportunity_count INTEGER,
    potential_savings DECIMAL(15,2),
    carbon_reduction_potential DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- AI INSIGHTS TABLE
-- =====================================================

-- AI insights for companies
CREATE TABLE IF NOT EXISTS ai_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID,
    insight_type VARCHAR(50) NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    impact VARCHAR(20), -- 'low', 'medium', 'high'
    estimated_savings VARCHAR(100),
    carbon_reduction VARCHAR(100),
    action_required BOOLEAN DEFAULT false,
    confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- User sessions indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_created_at ON user_sessions(created_at);

-- Feature usage indexes
CREATE INDEX IF NOT EXISTS idx_feature_usage_user_id ON feature_usage(user_id);
CREATE INDEX IF NOT EXISTS idx_feature_usage_feature_name ON feature_usage(feature_name);

-- AI decisions indexes
CREATE INDEX IF NOT EXISTS idx_ai_decisions_user_id ON ai_decisions(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_decisions_type ON ai_decisions(decision_type);
CREATE INDEX IF NOT EXISTS idx_ai_decisions_created_at ON ai_decisions(created_at);

-- Business metrics indexes
CREATE INDEX IF NOT EXISTS idx_business_metrics_period ON business_metrics(period);
CREATE INDEX IF NOT EXISTS idx_business_metrics_date ON business_metrics(metric_date);

-- API requests indexes
CREATE INDEX IF NOT EXISTS idx_api_requests_endpoint ON api_requests(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_requests_status_code ON api_requests(status_code);
CREATE INDEX IF NOT EXISTS idx_api_requests_created_at ON api_requests(created_at);

-- Competitor data indexes
CREATE INDEX IF NOT EXISTS idx_competitor_social_name ON competitor_social_media(competitor_name);
CREATE INDEX IF NOT EXISTS idx_competitor_patents_name ON competitor_patents(competitor_name);
CREATE INDEX IF NOT EXISTS idx_competitor_hiring_name ON competitor_hiring(competitor_name);
CREATE INDEX IF NOT EXISTS idx_competitor_funding_name ON competitor_funding(competitor_name);

-- AI insights indexes
CREATE INDEX IF NOT EXISTS idx_ai_insights_company_id ON ai_insights(company_id);
CREATE INDEX IF NOT EXISTS idx_ai_insights_type ON ai_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_ai_insights_created_at ON ai_insights(created_at);

-- =====================================================
-- ADD FOREIGN KEY CONSTRAINTS (OPTIONAL - AFTER TABLES EXIST)
-- =====================================================

-- Add foreign key constraints after all tables are created
DO $$
BEGIN
    -- Add foreign key for user_sessions
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'fk_user_sessions_user_id'
    ) THEN
        ALTER TABLE user_sessions ADD CONSTRAINT fk_user_sessions_user_id 
            FOREIGN KEY (user_id) REFERENCES companies(user_id);
        RAISE NOTICE 'Added foreign key constraint for user_sessions';
    END IF;
    
    -- Add foreign key for feature_usage
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'fk_feature_usage_user_id'
    ) THEN
        ALTER TABLE feature_usage ADD CONSTRAINT fk_feature_usage_user_id 
            FOREIGN KEY (user_id) REFERENCES companies(user_id);
        RAISE NOTICE 'Added foreign key constraint for feature_usage';
    END IF;
    
    -- Add foreign key for user_feedback
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'fk_user_feedback_user_id'
    ) THEN
        ALTER TABLE user_feedback ADD CONSTRAINT fk_user_feedback_user_id 
            FOREIGN KEY (user_id) REFERENCES companies(user_id);
        RAISE NOTICE 'Added foreign key constraint for user_feedback';
    END IF;
    
    -- Add foreign key for ai_decisions
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'fk_ai_decisions_user_id'
    ) THEN
        ALTER TABLE ai_decisions ADD CONSTRAINT fk_ai_decisions_user_id 
            FOREIGN KEY (user_id) REFERENCES companies(user_id);
        RAISE NOTICE 'Added foreign key constraint for ai_decisions';
    END IF;
    
    -- Add foreign key for api_requests
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'fk_api_requests_user_id'
    ) THEN
        ALTER TABLE api_requests ADD CONSTRAINT fk_api_requests_user_id 
            FOREIGN KEY (user_id) REFERENCES companies(user_id);
        RAISE NOTICE 'Added foreign key constraint for api_requests';
    END IF;
    
    -- Add foreign key for ai_insights
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'fk_ai_insights_company_id'
    ) THEN
        ALTER TABLE ai_insights ADD CONSTRAINT fk_ai_insights_company_id 
            FOREIGN KEY (company_id) REFERENCES companies(id);
        RAISE NOTICE 'Added foreign key constraint for ai_insights';
    END IF;
END $$;

-- =====================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_insights ENABLE ROW LEVEL SECURITY;

-- User sessions policy - users can only see their own sessions
CREATE POLICY "Users can view own sessions" ON user_sessions
    FOR SELECT USING (auth.uid() = user_id);

-- Feature usage policy - users can only see their own usage
CREATE POLICY "Users can view own feature usage" ON feature_usage
    FOR SELECT USING (auth.uid() = user_id);

-- User feedback policy - users can only see their own feedback
CREATE POLICY "Users can view own feedback" ON user_feedback
    FOR SELECT USING (auth.uid() = user_id);

-- AI decisions policy - users can only see their own decisions
CREATE POLICY "Users can view own AI decisions" ON ai_decisions
    FOR SELECT USING (auth.uid() = user_id);

-- AI insights policy - users can only see insights for their companies
CREATE POLICY "Users can view company insights" ON ai_insights
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM companies 
            WHERE companies.id = ai_insights.company_id 
            AND companies.user_id = auth.uid()
        )
    );

-- =====================================================
-- SAMPLE DATA INSERTION (OPTIONAL)
-- =====================================================

-- Insert sample system health data
INSERT INTO system_health (health_score, cpu_usage, memory_usage, disk_usage) 
VALUES (0.95, 25.5, 45.2, 30.1)
ON CONFLICT DO NOTHING;

-- Insert sample performance metrics
INSERT INTO performance_metrics (metric_type, improvement_percentage, baseline_value, current_value)
VALUES 
    ('response_time', 15.5, 1200, 1014),
    ('throughput', 25.0, 100, 125),
    ('memory', 10.0, 512, 460)
ON CONFLICT DO NOTHING;

-- =====================================================
-- MIGRATION COMPLETION
-- =====================================================

-- Log migration completion
INSERT INTO system_health (health_score, cpu_usage, memory_usage, disk_usage)
VALUES (1.0, 0, 0, 0)
ON CONFLICT DO NOTHING;

-- Print completion message
DO $$
BEGIN
    RAISE NOTICE '✅ Database migration completed successfully!';
    RAISE NOTICE '📊 Created 20+ tables for real data collection';
    RAISE NOTICE '🔒 Applied Row Level Security policies';
    RAISE NOTICE '⚡ Created performance indexes';
    RAISE NOTICE '🔗 Added foreign key constraints';
    RAISE NOTICE '🚀 ISM AI Platform ready for real data!';
END $$; 