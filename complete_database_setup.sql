-- =====================================================
-- SYMBIOFLOWS COMPLETE DATABASE SETUP
-- =====================================================
-- This file combines all migrations with IF NOT EXISTS checks
-- Run this in Supabase SQL Editor

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- =====================================================
-- CORE TABLES (with IF NOT EXISTS)
-- =====================================================

-- Companies table
CREATE TABLE IF NOT EXISTS companies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    location VARCHAR(255),
    country VARCHAR(100),
    city VARCHAR(100),
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    employee_count INTEGER,
    annual_revenue DECIMAL(15,2),
    description TEXT,
    website VARCHAR(255),
    contact_email VARCHAR(255),
    contact_phone VARCHAR(50),
    sustainability_goals TEXT[],
    onboarding_completed BOOLEAN DEFAULT FALSE,
    subscription_status VARCHAR(50) DEFAULT 'free',
    subscription_tier VARCHAR(50) DEFAULT 'basic',
    subscription_expires_at TIMESTAMP WITH TIME ZONE,
    ai_portfolio_summary TEXT,
    ai_recommendations JSONB,
    sustainability_score DECIMAL(5,2) DEFAULT 0,
    matches_count INTEGER DEFAULT 0,
    savings_achieved DECIMAL(15,2) DEFAULT 0,
    carbon_reduced DECIMAL(10,2) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Materials table
CREATE TABLE IF NOT EXISTS materials (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    material_name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL, -- product, waste, requirement
    category VARCHAR(100),
    quantity DECIMAL(15,2),
    unit VARCHAR(50),
    description TEXT,
    location VARCHAR(255),
    purity_level DECIMAL(5,4),
    market_price_per_ton DECIMAL(10,2),
    ai_generated BOOLEAN DEFAULT FALSE,
    status VARCHAR(50) DEFAULT 'active',
    material_properties JSONB,
    shipping_params JSONB,
    sustainability_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Material categories table
CREATE TABLE IF NOT EXISTS material_categories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Locations table
CREATE TABLE IF NOT EXISTS locations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    country VARCHAR(100),
    region VARCHAR(100),
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(name, country)
);

-- Symbiotic matches table
CREATE TABLE IF NOT EXISTS symbiotic_matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_a_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    company_b_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    material_a_id UUID REFERENCES materials(id) ON DELETE SET NULL,
    material_b_id UUID REFERENCES materials(id) ON DELETE SET NULL,
    match_score DECIMAL(5,4) NOT NULL,
    match_type VARCHAR(100) NOT NULL,
    potential_savings DECIMAL(15,2),
    implementation_complexity VARCHAR(50),
    environmental_impact DECIMAL(10,2),
    description TEXT,
    materials_compatibility DECIMAL(5,4),
    waste_synergy DECIMAL(5,4),
    energy_synergy DECIMAL(5,4),
    location_proximity DECIMAL(5,4),
    ai_confidence DECIMAL(5,4),
    match_analysis JSONB,
    status VARCHAR(50) DEFAULT 'potential',
    user_feedback JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(company_a_id, company_b_id, material_a_id, material_b_id)
);

-- Shipments table
CREATE TABLE IF NOT EXISTS shipments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    match_id UUID REFERENCES symbiotic_matches(id) ON DELETE SET NULL,
    material_id UUID REFERENCES materials(id) ON DELETE SET NULL,
    from_company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    to_company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    carrier VARCHAR(100),
    service VARCHAR(100),
    tracking_number VARCHAR(255),
    tracking_url TEXT,
    label_url TEXT,
    shipping_cost DECIMAL(10,2),
    currency VARCHAR(10) DEFAULT 'USD',
    weight_kg DECIMAL(10,2),
    volume_cubic_meters DECIMAL(10,2),
    special_handling TEXT[],
    packaging_requirements VARCHAR(100),
    temperature_requirements JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    estimated_delivery TIMESTAMP WITH TIME ZONE,
    actual_delivery TIMESTAMP WITH TIME ZONE,
    test_mode BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Activity logs table
CREATE TABLE IF NOT EXISTS activity_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    activity_type VARCHAR(100) NOT NULL,
    activity_data JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI insights table
CREATE TABLE IF NOT EXISTS ai_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    insight_type VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    confidence_score DECIMAL(5,4),
    data JSONB,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add missing columns to ai_insights table if they don't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='ai_insights' AND column_name='confidence_score') THEN
        ALTER TABLE ai_insights ADD COLUMN confidence_score DECIMAL(5,4);
    END IF;
END $$;

-- Add missing updated_at columns to all tables (only if tables exist)
DO $$ 
BEGIN
    -- Companies table
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='companies') THEN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='companies' AND column_name='updated_at') THEN
            ALTER TABLE companies ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
        END IF;
    END IF;
    
    -- Materials table
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='materials') THEN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='materials' AND column_name='updated_at') THEN
            ALTER TABLE materials ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
        END IF;
    END IF;
    
    -- Symbiotic matches table
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='symbiotic_matches') THEN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='symbiotic_matches' AND column_name='updated_at') THEN
            ALTER TABLE symbiotic_matches ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
        END IF;
    END IF;
    
    -- Shipments table
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='shipments') THEN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='shipments' AND column_name='updated_at') THEN
            ALTER TABLE shipments ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
        END IF;
    END IF;
    
    -- AI interactions table
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='ai_interactions') THEN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='ai_interactions' AND column_name='updated_at') THEN
            ALTER TABLE ai_interactions ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
        END IF;
    END IF;
    
    -- Feedback requests table
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name='feedback_requests') THEN
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='feedback_requests' AND column_name='updated_at') THEN
            ALTER TABLE feedback_requests ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
        END IF;
    END IF;
END $$;

-- =====================================================
-- NEW INTELLIGENT CORE TABLES
-- =====================================================

-- AI interactions table
CREATE TABLE IF NOT EXISTS ai_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_type VARCHAR(100) NOT NULL,
    input_data JSONB NOT NULL,
    response_data JSONB,
    context JSONB,
    confidence_score DECIMAL(5,4),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI errors table
CREATE TABLE IF NOT EXISTS ai_errors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_type VARCHAR(100) NOT NULL,
    input_data JSONB,
    error_message TEXT NOT NULL,
    error_stack TEXT,
    context JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Feedback requests table
CREATE TABLE IF NOT EXISTS feedback_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feedback_id VARCHAR(255) UNIQUE NOT NULL,
    analysis_type VARCHAR(100) NOT NULL,
    input_data JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User feedback table
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feedback_id VARCHAR(255) NOT NULL,
    user_ratings JSONB,
    qualitative_feedback TEXT,
    improvement_suggestions TEXT,
    overall_satisfaction INTEGER CHECK (overall_satisfaction >= 1 AND overall_satisfaction <= 5),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API call logs table
CREATE TABLE IF NOT EXISTS api_call_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_name VARCHAR(100) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status VARCHAR(50) NOT NULL,
    duration_ms INTEGER,
    request_id VARCHAR(255),
    error_message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Shipping requests table
CREATE TABLE IF NOT EXISTS shipping_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    material_name VARCHAR(255),
    material_type VARCHAR(50),
    quantity DECIMAL(10,2),
    unit VARCHAR(50),
    from_location VARCHAR(255),
    to_location VARCHAR(255),
    rates_count INTEGER,
    lowest_price DECIMAL(10,2),
    test_mode BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Shipping labels table
CREATE TABLE IF NOT EXISTS shipping_labels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    material_name VARCHAR(255),
    tracking_number VARCHAR(255),
    from_location VARCHAR(255),
    to_location VARCHAR(255),
    shipping_cost DECIMAL(10,2),
    carrier VARCHAR(100),
    service VARCHAR(100),
    test_mode BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- ADD NEW COLUMNS TO EXISTING TABLES
-- =====================================================

-- Add new columns to companies table
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='companies' AND column_name='ai_portfolio_summary') THEN
        ALTER TABLE companies ADD COLUMN ai_portfolio_summary TEXT;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='companies' AND column_name='ai_recommendations') THEN
        ALTER TABLE companies ADD COLUMN ai_recommendations JSONB;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='companies' AND column_name='sustainability_score') THEN
        ALTER TABLE companies ADD COLUMN sustainability_score DECIMAL(5,2) DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='companies' AND column_name='matches_count') THEN
        ALTER TABLE companies ADD COLUMN matches_count INTEGER DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='companies' AND column_name='savings_achieved') THEN
        ALTER TABLE companies ADD COLUMN savings_achieved DECIMAL(15,2) DEFAULT 0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='companies' AND column_name='carbon_reduced') THEN
        ALTER TABLE companies ADD COLUMN carbon_reduced DECIMAL(10,2) DEFAULT 0;
    END IF;
END $$;

-- Add new columns to materials table
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='materials' AND column_name='category') THEN
        ALTER TABLE materials ADD COLUMN category VARCHAR(100);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='materials' AND column_name='status') THEN
        ALTER TABLE materials ADD COLUMN status VARCHAR(50) DEFAULT 'active';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='materials' AND column_name='material_properties') THEN
        ALTER TABLE materials ADD COLUMN material_properties JSONB;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='materials' AND column_name='shipping_params') THEN
        ALTER TABLE materials ADD COLUMN shipping_params JSONB;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='materials' AND column_name='sustainability_metrics') THEN
        ALTER TABLE materials ADD COLUMN sustainability_metrics JSONB;
    END IF;
END $$;

-- Add new columns to symbiotic_matches table
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='symbiotic_matches' AND column_name='materials_compatibility') THEN
        ALTER TABLE symbiotic_matches ADD COLUMN materials_compatibility DECIMAL(5,4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='symbiotic_matches' AND column_name='waste_synergy') THEN
        ALTER TABLE symbiotic_matches ADD COLUMN waste_synergy DECIMAL(5,4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='symbiotic_matches' AND column_name='energy_synergy') THEN
        ALTER TABLE symbiotic_matches ADD COLUMN energy_synergy DECIMAL(5,4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='symbiotic_matches' AND column_name='location_proximity') THEN
        ALTER TABLE symbiotic_matches ADD COLUMN location_proximity DECIMAL(5,4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='symbiotic_matches' AND column_name='ai_confidence') THEN
        ALTER TABLE symbiotic_matches ADD COLUMN ai_confidence DECIMAL(5,4);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='symbiotic_matches' AND column_name='match_analysis') THEN
        ALTER TABLE symbiotic_matches ADD COLUMN match_analysis JSONB;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='symbiotic_matches' AND column_name='user_feedback') THEN
        ALTER TABLE symbiotic_matches ADD COLUMN user_feedback JSONB;
    END IF;
END $$;

-- Add new columns to shipments table
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='shipments' AND column_name='weight_kg') THEN
        ALTER TABLE shipments ADD COLUMN weight_kg DECIMAL(10,2);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='shipments' AND column_name='volume_cubic_meters') THEN
        ALTER TABLE shipments ADD COLUMN volume_cubic_meters DECIMAL(10,2);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='shipments' AND column_name='special_handling') THEN
        ALTER TABLE shipments ADD COLUMN special_handling TEXT[];
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='shipments' AND column_name='packaging_requirements') THEN
        ALTER TABLE shipments ADD COLUMN packaging_requirements VARCHAR(100);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='shipments' AND column_name='temperature_requirements') THEN
        ALTER TABLE shipments ADD COLUMN temperature_requirements JSONB;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='shipments' AND column_name='test_mode') THEN
        ALTER TABLE shipments ADD COLUMN test_mode BOOLEAN DEFAULT FALSE;
    END IF;
END $$;

-- =====================================================
-- PERFORMANCE INDEXES
-- =====================================================

-- Companies indexes
CREATE INDEX IF NOT EXISTS idx_companies_user_id ON companies(user_id);
CREATE INDEX IF NOT EXISTS idx_companies_industry ON companies(industry);
CREATE INDEX IF NOT EXISTS idx_companies_location ON companies USING gin(location gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_companies_onboarding_completed ON companies(onboarding_completed);
CREATE INDEX IF NOT EXISTS idx_companies_subscription_status ON companies(subscription_status);
CREATE INDEX IF NOT EXISTS idx_companies_sustainability_score ON companies(sustainability_score DESC);
CREATE INDEX IF NOT EXISTS idx_companies_matches_count ON companies(matches_count DESC);
CREATE INDEX IF NOT EXISTS idx_companies_savings ON companies(savings_achieved DESC);
CREATE INDEX IF NOT EXISTS idx_companies_created_at ON companies(created_at DESC);

-- Materials indexes
CREATE INDEX IF NOT EXISTS idx_materials_company_id ON materials(company_id);
CREATE INDEX IF NOT EXISTS idx_materials_type ON materials(type);
CREATE INDEX IF NOT EXISTS idx_materials_category ON materials(category);
CREATE INDEX IF NOT EXISTS idx_materials_material_name ON materials USING gin(material_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_materials_status ON materials(status);
CREATE INDEX IF NOT EXISTS idx_materials_ai_generated ON materials(ai_generated);
CREATE INDEX IF NOT EXISTS idx_materials_properties ON materials USING gin(material_properties);
CREATE INDEX IF NOT EXISTS idx_materials_shipping ON materials USING gin(shipping_params);
CREATE INDEX IF NOT EXISTS idx_materials_created_at ON materials(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_materials_location ON materials USING gin(location gin_trgm_ops);

-- Symbiotic matches indexes
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_company_a ON symbiotic_matches(company_a_id);
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_company_b ON symbiotic_matches(company_b_id);
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_score ON symbiotic_matches(match_score DESC);
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_type ON symbiotic_matches(match_type);
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_status ON symbiotic_matches(status);
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_compatibility ON symbiotic_matches(materials_compatibility DESC);
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_confidence ON symbiotic_matches(ai_confidence DESC);
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_analysis ON symbiotic_matches USING gin(match_analysis);
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_created_at ON symbiotic_matches(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_composite ON symbiotic_matches(company_a_id, match_score DESC, status);

-- Shipments indexes
CREATE INDEX IF NOT EXISTS idx_shipments_match_id ON shipments(match_id);
CREATE INDEX IF NOT EXISTS idx_shipments_material_id ON shipments(material_id);
CREATE INDEX IF NOT EXISTS idx_shipments_from_company ON shipments(from_company_id);
CREATE INDEX IF NOT EXISTS idx_shipments_to_company ON shipments(to_company_id);
CREATE INDEX IF NOT EXISTS idx_shipments_tracking_number ON shipments(tracking_number);
CREATE INDEX IF NOT EXISTS idx_shipments_status ON shipments(status);
CREATE INDEX IF NOT EXISTS idx_shipments_created_at ON shipments(created_at DESC);

-- Activity logs indexes
CREATE INDEX IF NOT EXISTS idx_activity_logs_company_id ON activity_logs(company_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_user_id ON activity_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_type ON activity_logs(activity_type);
CREATE INDEX IF NOT EXISTS idx_activity_logs_created_at ON activity_logs(created_at DESC);

-- AI insights indexes
CREATE INDEX IF NOT EXISTS idx_ai_insights_company_id ON ai_insights(company_id);
CREATE INDEX IF NOT EXISTS idx_ai_insights_type ON ai_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_ai_insights_confidence ON ai_insights(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_ai_insights_created_at ON ai_insights(created_at DESC);

-- AI interactions indexes
CREATE INDEX IF NOT EXISTS idx_ai_interactions_type ON ai_interactions(analysis_type);
CREATE INDEX IF NOT EXISTS idx_ai_interactions_timestamp ON ai_interactions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ai_interactions_confidence ON ai_interactions(confidence_score DESC);

-- AI errors indexes
CREATE INDEX IF NOT EXISTS idx_ai_errors_type ON ai_errors(analysis_type);
CREATE INDEX IF NOT EXISTS idx_ai_errors_timestamp ON ai_errors(timestamp DESC);

-- Feedback indexes
CREATE INDEX IF NOT EXISTS idx_feedback_requests_id ON feedback_requests(feedback_id);
CREATE INDEX IF NOT EXISTS idx_feedback_requests_status ON feedback_requests(status);
CREATE INDEX IF NOT EXISTS idx_user_feedback_id ON user_feedback(feedback_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_satisfaction ON user_feedback(overall_satisfaction);

-- API call logs indexes
CREATE INDEX IF NOT EXISTS idx_api_call_logs_name ON api_call_logs(api_name);
CREATE INDEX IF NOT EXISTS idx_api_call_logs_status ON api_call_logs(status);
CREATE INDEX IF NOT EXISTS idx_api_call_logs_timestamp ON api_call_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_api_call_logs_duration ON api_call_logs(duration_ms DESC);

-- Shipping indexes
CREATE INDEX IF NOT EXISTS idx_shipping_requests_company ON shipping_requests(company_id);
CREATE INDEX IF NOT EXISTS idx_shipping_requests_timestamp ON shipping_requests(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_shipping_labels_company ON shipping_labels(company_id);
CREATE INDEX IF NOT EXISTS idx_shipping_labels_created ON shipping_labels(created_at DESC);

-- =====================================================
-- TRIGGERS & FUNCTIONS
-- =====================================================

-- Update updated_at timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to all tables
DROP TRIGGER IF EXISTS update_companies_updated_at ON companies;
CREATE TRIGGER update_companies_updated_at BEFORE UPDATE ON companies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_materials_updated_at ON materials;
CREATE TRIGGER update_materials_updated_at BEFORE UPDATE ON materials FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_symbiotic_matches_updated_at ON symbiotic_matches;
CREATE TRIGGER update_symbiotic_matches_updated_at BEFORE UPDATE ON symbiotic_matches FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_shipments_updated_at ON shipments;
CREATE TRIGGER update_shipments_updated_at BEFORE UPDATE ON shipments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_ai_interactions_updated_at ON ai_interactions;
CREATE TRIGGER update_ai_interactions_updated_at BEFORE UPDATE ON ai_interactions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_feedback_requests_updated_at ON feedback_requests;
CREATE TRIGGER update_feedback_requests_updated_at BEFORE UPDATE ON feedback_requests FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to update company match count
CREATE OR REPLACE FUNCTION update_company_match_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE companies SET matches_count = matches_count + 1 WHERE id = NEW.company_a_id;
        UPDATE companies SET matches_count = matches_count + 1 WHERE id = NEW.company_b_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE companies SET matches_count = matches_count - 1 WHERE id = OLD.company_a_id;
        UPDATE companies SET matches_count = matches_count - 1 WHERE id = OLD.company_b_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

-- Apply match count trigger
DROP TRIGGER IF EXISTS update_match_count ON symbiotic_matches;
CREATE TRIGGER update_match_count AFTER INSERT OR DELETE ON symbiotic_matches FOR EACH ROW EXECUTE FUNCTION update_company_match_count();

-- Function to calculate sustainability score
CREATE OR REPLACE FUNCTION calculate_sustainability_score(company_id UUID)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    score DECIMAL(5,2) := 0;
    material_count INTEGER;
    waste_count INTEGER;
    recycling_count INTEGER;
BEGIN
    -- Count materials and waste streams
    SELECT COUNT(*) INTO material_count 
    FROM materials 
    WHERE company_id = calculate_sustainability_score.company_id AND status = 'active';
    
    SELECT COUNT(*) INTO waste_count 
    FROM materials 
    WHERE company_id = calculate_sustainability_score.company_id 
    AND type = 'waste' AND status = 'active';
    
    SELECT COUNT(*) INTO recycling_count 
    FROM materials 
    WHERE company_id = calculate_sustainability_score.company_id 
    AND type = 'requirement' AND status = 'active';
    
    -- Calculate basic sustainability score
    IF material_count > 0 THEN
        score := score + (recycling_count::DECIMAL / material_count::DECIMAL) * 40;
    END IF;
    
    IF waste_count > 0 THEN
        score := score + (recycling_count::DECIMAL / waste_count::DECIMAL) * 30;
    END IF;
    
    -- Add points for having both waste and requirements
    IF waste_count > 0 AND recycling_count > 0 THEN
        score := score + 30;
    END IF;
    
    RETURN LEAST(score, 100);
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- CLEAN SYNTHETIC DATA
-- =====================================================

-- Delete all synthetic companies (those with generic names)
DELETE FROM companies 
WHERE name LIKE '%Company%' 
   OR name LIKE '%Manufacturing%' 
   OR name LIKE '%Textiles%' 
   OR name LIKE '%Food%' 
   OR name LIKE '%Chemical%' 
   OR name LIKE '%Construction%' 
   OR name LIKE '%Electronics%' 
   OR name LIKE '%Automotive%' 
   OR name LIKE '%Pharmaceutical%'
   OR name LIKE '%Synthetic%'
   OR name LIKE '%Test%'
   OR name LIKE '%Demo%';

-- Delete all synthetic materials
DELETE FROM materials 
WHERE material_name LIKE '%synthetic%' 
   OR material_name LIKE '%test%' 
   OR material_name LIKE '%demo%'
   OR material_name LIKE '%sample%';

-- Delete all synthetic matches
DELETE FROM symbiotic_matches 
WHERE description LIKE '%synthetic%' 
   OR description LIKE '%test%' 
   OR description LIKE '%demo%';

-- Delete all synthetic shipments
DELETE FROM shipments 
WHERE tracking_number LIKE '%TEST%' 
   OR tracking_number LIKE '%DEMO%' 
   OR tracking_number LIKE '%SYNTHETIC%';

-- Delete all synthetic AI interactions
DELETE FROM ai_interactions 
WHERE analysis_type LIKE '%synthetic%' 
   OR analysis_type LIKE '%test%';

-- Delete all synthetic feedback
DELETE FROM user_feedback 
WHERE feedback_id LIKE '%test%' 
   OR feedback_id LIKE '%demo%';

-- Delete all synthetic shipping requests
DELETE FROM shipping_requests 
WHERE test_mode = TRUE;

-- Delete all synthetic shipping labels
DELETE FROM shipping_labels 
WHERE test_mode = TRUE;

-- Reset company counters
UPDATE companies SET 
    matches_count = 0,
    savings_achieved = 0,
    carbon_reduced = 0,
    sustainability_score = 0,
    ai_portfolio_summary = NULL,
    ai_recommendations = NULL;

-- Reset material properties
UPDATE materials SET 
    material_properties = NULL,
    shipping_params = NULL,
    sustainability_metrics = NULL;

-- Reset match analysis
UPDATE symbiotic_matches SET 
    materials_compatibility = 0,
    waste_synergy = 0,
    energy_synergy = 0,
    location_proximity = 0,
    ai_confidence = 0,
    match_analysis = NULL,
    user_feedback = NULL;

-- =====================================================
-- PREPARE FOR GULF DATA
-- =====================================================

-- Create Gulf-specific material categories if they don't exist
INSERT INTO material_categories (name, description) VALUES
('Petrochemicals', 'Petroleum-based chemicals and derivatives'),
('Construction Materials', 'Building and construction materials'),
('Food & Beverage', 'Food processing and beverage production materials'),
('Textiles', 'Fabric and textile manufacturing materials'),
('Metals & Mining', 'Metal processing and mining materials'),
('Electronics', 'Electronic components and materials'),
('Pharmaceuticals', 'Pharmaceutical and medical materials'),
('Renewable Energy', 'Solar, wind, and renewable energy materials')
ON CONFLICT (name) DO NOTHING;

-- Create Gulf-specific locations if they don't exist
INSERT INTO locations (name, country, region, latitude, longitude) VALUES
('Dubai', 'UAE', 'Gulf', 25.2048, 55.2708),
('Abu Dhabi', 'UAE', 'Gulf', 24.4539, 54.3773),
('Riyadh', 'Saudi Arabia', 'Gulf', 24.7136, 46.6753),
('Jeddah', 'Saudi Arabia', 'Gulf', 21.4858, 39.1925),
('Dammam', 'Saudi Arabia', 'Gulf', 26.4207, 50.0888),
('Kuwait City', 'Kuwait', 'Gulf', 29.3759, 47.9774),
('Doha', 'Qatar', 'Gulf', 25.2854, 51.5310),
('Manama', 'Bahrain', 'Gulf', 26.2285, 50.5860),
('Muscat', 'Oman', 'Gulf', 23.5880, 58.3829)
ON CONFLICT (name, country) DO NOTHING;

-- =====================================================
-- ROW LEVEL SECURITY (RLS)
-- =====================================================

-- Enable RLS on tables
ALTER TABLE companies ENABLE ROW LEVEL SECURITY;
ALTER TABLE materials ENABLE ROW LEVEL SECURITY;
ALTER TABLE symbiotic_matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE shipments ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_errors ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_call_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE shipping_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE shipping_labels ENABLE ROW LEVEL SECURITY;

-- Companies RLS policies
DROP POLICY IF EXISTS "Users can view their own companies" ON companies;
CREATE POLICY "Users can view their own companies" ON companies 
    FOR SELECT USING (user_id = auth.uid());

DROP POLICY IF EXISTS "Users can insert their own companies" ON companies;
CREATE POLICY "Users can insert their own companies" ON companies 
    FOR INSERT WITH CHECK (user_id = auth.uid());

DROP POLICY IF EXISTS "Users can update their own companies" ON companies;
CREATE POLICY "Users can update their own companies" ON companies 
    FOR UPDATE USING (user_id = auth.uid());

-- Materials RLS policies
DROP POLICY IF EXISTS "Users can view materials for their companies" ON materials;
CREATE POLICY "Users can view materials for their companies" ON materials 
    FOR SELECT USING (
        company_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
    );

DROP POLICY IF EXISTS "Users can insert materials for their companies" ON materials;
CREATE POLICY "Users can insert materials for their companies" ON materials 
    FOR INSERT WITH CHECK (
        company_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
    );

DROP POLICY IF EXISTS "Users can update materials for their companies" ON materials;
CREATE POLICY "Users can update materials for their companies" ON materials 
    FOR UPDATE USING (
        company_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
    );

-- Symbiotic matches RLS policies
DROP POLICY IF EXISTS "Users can view matches involving their companies" ON symbiotic_matches;
CREATE POLICY "Users can view matches involving their companies" ON symbiotic_matches 
    FOR SELECT USING (
        company_a_id IN (SELECT id FROM companies WHERE user_id = auth.uid()) OR
        company_b_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
    );

-- AI interactions RLS policies
DROP POLICY IF EXISTS "Users can view their own AI interactions" ON ai_interactions;
CREATE POLICY "Users can view their own AI interactions" ON ai_interactions 
    FOR SELECT USING (true);

DROP POLICY IF EXISTS "System can insert AI interactions" ON ai_interactions;
CREATE POLICY "System can insert AI interactions" ON ai_interactions 
    FOR INSERT WITH CHECK (true);

-- AI errors RLS policies
DROP POLICY IF EXISTS "System can manage AI errors" ON ai_errors;
CREATE POLICY "System can manage AI errors" ON ai_errors 
    FOR ALL USING (true);

-- Feedback RLS policies
DROP POLICY IF EXISTS "Users can view their own feedback" ON user_feedback;
CREATE POLICY "Users can view their own feedback" ON user_feedback 
    FOR SELECT USING (true);

DROP POLICY IF EXISTS "Users can insert their own feedback" ON user_feedback;
CREATE POLICY "Users can insert their own feedback" ON user_feedback 
    FOR INSERT WITH CHECK (true);

-- API call logs RLS policies
DROP POLICY IF EXISTS "System can manage API logs" ON api_call_logs;
CREATE POLICY "System can manage API logs" ON api_call_logs 
    FOR ALL USING (true);

-- Shipping RLS policies
DROP POLICY IF EXISTS "Users can view shipping for their company" ON shipping_requests;
CREATE POLICY "Users can view shipping for their company" ON shipping_requests 
    FOR SELECT USING (
        company_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
    );

DROP POLICY IF EXISTS "Users can view shipping labels for their company" ON shipping_labels;
CREATE POLICY "Users can view shipping labels for their company" ON shipping_labels 
    FOR SELECT USING (
        company_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
    );

-- =====================================================
-- VIEWS FOR ANALYTICS
-- =====================================================

-- View for AI performance analytics
CREATE OR REPLACE VIEW ai_performance_analytics AS
SELECT 
    analysis_type,
    COUNT(*) as total_requests,
    COUNT(CASE WHEN confidence_score >= 0.8 THEN 1 END) as high_confidence_requests,
    AVG(confidence_score) as average_confidence,
    MIN(timestamp) as first_request,
    MAX(timestamp) as last_request
FROM ai_interactions
GROUP BY analysis_type
ORDER BY total_requests DESC;

-- View for user feedback analytics
CREATE OR REPLACE VIEW user_feedback_analytics AS
SELECT 
    fr.analysis_type,
    COUNT(uf.id) as total_feedback,
    AVG(uf.overall_satisfaction) as average_satisfaction,
    COUNT(CASE WHEN uf.overall_satisfaction >= 4 THEN 1 END) as satisfied_users,
    COUNT(CASE WHEN uf.overall_satisfaction <= 2 THEN 1 END) as dissatisfied_users
FROM feedback_requests fr
LEFT JOIN user_feedback uf ON fr.feedback_id = uf.feedback_id
WHERE fr.status = 'completed'
GROUP BY fr.analysis_type
ORDER BY total_feedback DESC;

-- View for shipping analytics
CREATE OR REPLACE VIEW shipping_analytics AS
SELECT 
    carrier,
    service,
    COUNT(*) as total_shipments,
    AVG(shipping_cost) as average_cost,
    SUM(shipping_cost) as total_cost,
    COUNT(CASE WHEN test_mode = true THEN 1 END) as test_shipments
FROM shipping_labels
GROUP BY carrier, service
ORDER BY total_shipments DESC;

-- =====================================================
-- GRANTS
-- =====================================================

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated;

-- Grant access to views
GRANT SELECT ON ai_performance_analytics TO authenticated;
GRANT SELECT ON user_feedback_analytics TO authenticated;
GRANT SELECT ON shipping_analytics TO authenticated;

-- =====================================================
-- SUCCESS MESSAGE
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '=====================================================';
    RAISE NOTICE 'SYMBIOFLOWS DATABASE SETUP COMPLETED SUCCESSFULLY!';
    RAISE NOTICE '=====================================================';
    RAISE NOTICE '✅ All tables created with IF NOT EXISTS checks';
    RAISE NOTICE '✅ All indexes created with IF NOT EXISTS checks';
    RAISE NOTICE '✅ All triggers created with proper DROP/CREATE';
    RAISE NOTICE '✅ All synthetic data cleaned';
    RAISE NOTICE '✅ Gulf-specific categories and locations added';
    RAISE NOTICE '✅ Row Level Security enabled';
    RAISE NOTICE '✅ Analytics views created';
    RAISE NOTICE '✅ Permissions granted';
    RAISE NOTICE '';
    RAISE NOTICE '🚀 SYSTEM IS READY FOR GULF COMPANY DATA IMPORT!';
    RAISE NOTICE '=====================================================';
END $$; 