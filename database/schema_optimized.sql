-- =====================================================
-- SYMBIOFLOWS OPTIMIZED DATABASE SCHEMA
-- =====================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- =====================================================
-- CORE TABLES
-- =====================================================

-- Companies table - Core company profiles
CREATE TABLE companies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100) NOT NULL,
    location VARCHAR(255) NOT NULL,
    employee_count INTEGER DEFAULT 0,
    annual_revenue DECIMAL(15,2) DEFAULT 0,
    products TEXT,
    main_materials TEXT,
    production_volume VARCHAR(100),
    process_description TEXT,
    sustainability_goals TEXT[],
    current_waste_management TEXT,
    waste_quantity DECIMAL(10,2),
    waste_unit VARCHAR(50),
    waste_frequency VARCHAR(50),
    resource_needs TEXT,
    energy_consumption VARCHAR(100),
    environmental_certifications TEXT,
    current_recycling_practices TEXT,
    partnership_interests TEXT[],
    geographic_preferences VARCHAR(255),
    technology_interests TEXT,
    onboarding_completed BOOLEAN DEFAULT FALSE,
    subscription_tier VARCHAR(50) DEFAULT 'free',
    subscription_status VARCHAR(50) DEFAULT 'active',
    ai_portfolio_summary TEXT,
    ai_recommendations JSONB,
    sustainability_score DECIMAL(5,2) DEFAULT 0,
    matches_count INTEGER DEFAULT 0,
    savings_achieved DECIMAL(15,2) DEFAULT 0,
    carbon_reduced DECIMAL(10,2) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Materials table - Material listings (waste/requirements)
CREATE TABLE materials (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    material_name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('waste', 'requirement')),
    category VARCHAR(100),
    state VARCHAR(50), -- solid, liquid, gas
    quantity DECIMAL(10,2) NOT NULL,
    unit VARCHAR(50) NOT NULL,
    frequency VARCHAR(50),
    description TEXT,
    quality_grade VARCHAR(50) CHECK (quality_grade IN ('high', 'medium', 'low')),
    location VARCHAR(255),
    price_per_unit DECIMAL(10,2),
    currency VARCHAR(10) DEFAULT 'USD',
    ai_generated BOOLEAN DEFAULT FALSE,
    material_properties JSONB, -- Density, melting point, etc.
    shipping_params JSONB, -- Calculated shipping parameters
    sustainability_metrics JSONB,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Symbiotic matches table - AI-generated potential partnerships
CREATE TABLE symbiotic_matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_a_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    company_b_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    material_a_id UUID REFERENCES materials(id) ON DELETE SET NULL,
    material_b_id UUID REFERENCES materials(id) ON DELETE SET NULL,
    match_score DECIMAL(5,4) NOT NULL,
    match_type VARCHAR(100) NOT NULL, -- material_exchange, waste_recycling, energy_sharing
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
    status VARCHAR(50) DEFAULT 'potential', -- potential, contacted, in_negotiation, completed, rejected
    user_feedback JSONB, -- Store user ratings and feedback
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(company_a_id, company_b_id, material_a_id, material_b_id)
);

-- Shipments table - Shipping and logistics data
CREATE TABLE shipments (
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
    status VARCHAR(50) DEFAULT 'pending', -- pending, shipped, delivered, failed
    estimated_delivery TIMESTAMP WITH TIME ZONE,
    actual_delivery TIMESTAMP WITH TIME ZONE,
    test_mode BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Activity logs table - User dashboard and AI feedback loops
CREATE TABLE activity_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    activity_type VARCHAR(100) NOT NULL, -- material_added, match_found, shipment_created, etc.
    activity_data JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- AI & ANALYTICS TABLES
-- =====================================================

-- AI insights table - AI-generated insights and recommendations
CREATE TABLE ai_insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    insight_type VARCHAR(100) NOT NULL, -- portfolio_analysis, match_recommendation, sustainability_tip
    title VARCHAR(255) NOT NULL,
    description TEXT,
    confidence_score DECIMAL(5,4),
    data JSONB,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API call logs table - Monitor external API usage
CREATE TABLE api_call_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_name VARCHAR(100) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status VARCHAR(50) NOT NULL, -- success, error
    duration_ms INTEGER,
    request_id VARCHAR(255),
    error_message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Shipping requests table - Track shipping rate requests
CREATE TABLE shipping_requests (
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

-- Shipping labels table - Track created shipping labels
CREATE TABLE shipping_labels (
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
-- PERFORMANCE INDEXES
-- =====================================================

-- Companies indexes
CREATE INDEX idx_companies_user_id ON companies(user_id);
CREATE INDEX idx_companies_industry ON companies(industry);
CREATE INDEX idx_companies_location ON companies USING gin(location gin_trgm_ops);
CREATE INDEX idx_companies_onboarding_completed ON companies(onboarding_completed);
CREATE INDEX idx_companies_subscription_status ON companies(subscription_status);
CREATE INDEX idx_companies_sustainability_score ON companies(sustainability_score DESC);
CREATE INDEX idx_companies_created_at ON companies(created_at DESC);

-- Materials indexes
CREATE INDEX idx_materials_company_id ON materials(company_id);
CREATE INDEX idx_materials_type ON materials(type);
CREATE INDEX idx_materials_category ON materials(category);
CREATE INDEX idx_materials_material_name ON materials USING gin(material_name gin_trgm_ops);
CREATE INDEX idx_materials_status ON materials(status);
CREATE INDEX idx_materials_ai_generated ON materials(ai_generated);
CREATE INDEX idx_materials_created_at ON materials(created_at DESC);
CREATE INDEX idx_materials_location ON materials USING gin(location gin_trgm_ops);

-- Symbiotic matches indexes
CREATE INDEX idx_symbiotic_matches_company_a ON symbiotic_matches(company_a_id);
CREATE INDEX idx_symbiotic_matches_company_b ON symbiotic_matches(company_b_id);
CREATE INDEX idx_symbiotic_matches_score ON symbiotic_matches(match_score DESC);
CREATE INDEX idx_symbiotic_matches_type ON symbiotic_matches(match_type);
CREATE INDEX idx_symbiotic_matches_status ON symbiotic_matches(status);
CREATE INDEX idx_symbiotic_matches_created_at ON symbiotic_matches(created_at DESC);
CREATE INDEX idx_symbiotic_matches_composite ON symbiotic_matches(company_a_id, match_score DESC, status);

-- Shipments indexes
CREATE INDEX idx_shipments_match_id ON shipments(match_id);
CREATE INDEX idx_shipments_material_id ON shipments(material_id);
CREATE INDEX idx_shipments_from_company ON shipments(from_company_id);
CREATE INDEX idx_shipments_to_company ON shipments(to_company_id);
CREATE INDEX idx_shipments_tracking_number ON shipments(tracking_number);
CREATE INDEX idx_shipments_status ON shipments(status);
CREATE INDEX idx_shipments_created_at ON shipments(created_at DESC);

-- Activity logs indexes
CREATE INDEX idx_activity_logs_company_id ON activity_logs(company_id);
CREATE INDEX idx_activity_logs_user_id ON activity_logs(user_id);
CREATE INDEX idx_activity_logs_type ON activity_logs(activity_type);
CREATE INDEX idx_activity_logs_created_at ON activity_logs(created_at DESC);

-- AI insights indexes
CREATE INDEX idx_ai_insights_company_id ON ai_insights(company_id);
CREATE INDEX idx_ai_insights_type ON ai_insights(insight_type);
CREATE INDEX idx_ai_insights_confidence ON ai_insights(confidence_score DESC);
CREATE INDEX idx_ai_insights_created_at ON ai_insights(created_at DESC);

-- API call logs indexes
CREATE INDEX idx_api_call_logs_api_name ON api_call_logs(api_name);
CREATE INDEX idx_api_call_logs_status ON api_call_logs(status);
CREATE INDEX idx_api_call_logs_timestamp ON api_call_logs(timestamp DESC);
CREATE INDEX idx_api_call_logs_duration ON api_call_logs(duration_ms DESC);

-- =====================================================
-- TRIGGERS & FUNCTIONS
-- =====================================================

-- Update updated_at timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to all tables
CREATE TRIGGER update_companies_updated_at BEFORE UPDATE ON companies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_materials_updated_at BEFORE UPDATE ON materials FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_symbiotic_matches_updated_at BEFORE UPDATE ON symbiotic_matches FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_shipments_updated_at BEFORE UPDATE ON shipments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

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

CREATE TRIGGER update_match_count AFTER INSERT OR DELETE ON symbiotic_matches FOR EACH ROW EXECUTE FUNCTION update_company_match_count();

-- =====================================================
-- ROW LEVEL SECURITY (RLS)
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE companies ENABLE ROW LEVEL SECURITY;
ALTER TABLE materials ENABLE ROW LEVEL SECURITY;
ALTER TABLE symbiotic_matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE shipments ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_insights ENABLE ROW LEVEL SECURITY;

-- Companies RLS policies
CREATE POLICY "Users can view their own company" ON companies FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can update their own company" ON companies FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own company" ON companies FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Materials RLS policies
CREATE POLICY "Users can view materials from their company" ON materials FOR SELECT USING (
    company_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
);
CREATE POLICY "Users can update materials from their company" ON materials FOR UPDATE USING (
    company_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
);
CREATE POLICY "Users can insert materials for their company" ON materials FOR INSERT WITH CHECK (
    company_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
);

-- Symbiotic matches RLS policies
CREATE POLICY "Users can view matches involving their company" ON symbiotic_matches FOR SELECT USING (
    company_a_id IN (SELECT id FROM companies WHERE user_id = auth.uid()) OR
    company_b_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
);
CREATE POLICY "Users can update matches involving their company" ON symbiotic_matches FOR UPDATE USING (
    company_a_id IN (SELECT id FROM companies WHERE user_id = auth.uid()) OR
    company_b_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
);

-- Shipments RLS policies
CREATE POLICY "Users can view shipments involving their company" ON shipments FOR SELECT USING (
    from_company_id IN (SELECT id FROM companies WHERE user_id = auth.uid()) OR
    to_company_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
);

-- Activity logs RLS policies
CREATE POLICY "Users can view their own activity logs" ON activity_logs FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own activity logs" ON activity_logs FOR INSERT WITH CHECK (auth.uid() = user_id);

-- AI insights RLS policies
CREATE POLICY "Users can view insights for their company" ON ai_insights FOR SELECT USING (
    company_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- View for company dashboard data
CREATE VIEW company_dashboard AS
SELECT 
    c.id,
    c.name,
    c.industry,
    c.location,
    c.sustainability_score,
    c.matches_count,
    c.savings_achieved,
    c.carbon_reduced,
    COUNT(m.id) as materials_count,
    COUNT(sm.id) as active_matches_count,
    COUNT(s.id) as shipments_count
FROM companies c
LEFT JOIN materials m ON c.id = m.company_id AND m.status = 'active'
LEFT JOIN symbiotic_matches sm ON (c.id = sm.company_a_id OR c.id = sm.company_b_id) AND sm.status = 'potential'
LEFT JOIN shipments s ON (c.id = s.from_company_id OR c.id = s.to_company_id) AND s.status = 'shipped'
GROUP BY c.id, c.name, c.industry, c.location, c.sustainability_score, c.matches_count, c.savings_achieved, c.carbon_reduced;

-- View for material marketplace
CREATE VIEW material_marketplace AS
SELECT 
    m.id,
    m.material_name,
    m.type,
    m.category,
    m.quantity,
    m.unit,
    m.quality_grade,
    m.location,
    m.price_per_unit,
    m.currency,
    c.name as company_name,
    c.industry as company_industry,
    c.sustainability_score as company_sustainability_score,
    m.created_at
FROM materials m
JOIN companies c ON m.company_id = c.id
WHERE m.status = 'active'
ORDER BY m.created_at DESC;

-- View for high-potential matches
CREATE VIEW high_potential_matches AS
SELECT 
    sm.id,
    sm.match_score,
    sm.match_type,
    sm.potential_savings,
    sm.environmental_impact,
    c1.name as company_a_name,
    c1.industry as company_a_industry,
    c2.name as company_b_name,
    c2.industry as company_b_industry,
    m1.material_name as material_a_name,
    m2.material_name as material_b_name,
    sm.created_at
FROM symbiotic_matches sm
JOIN companies c1 ON sm.company_a_id = c1.id
JOIN companies c2 ON sm.company_b_id = c2.id
LEFT JOIN materials m1 ON sm.material_a_id = m1.id
LEFT JOIN materials m2 ON sm.material_b_id = m2.id
WHERE sm.match_score >= 0.7 AND sm.status = 'potential'
ORDER BY sm.match_score DESC;

-- =====================================================
-- PERFORMANCE MONITORING
-- =====================================================

-- Create a function to analyze query performance
CREATE OR REPLACE FUNCTION analyze_query_performance()
RETURNS TABLE (
    query_text TEXT,
    calls BIGINT,
    total_time DOUBLE PRECISION,
    mean_time DOUBLE PRECISION,
    rows BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        query::TEXT,
        calls,
        total_time,
        mean_time,
        rows
    FROM pg_stat_statements
    WHERE query LIKE '%companies%' OR query LIKE '%materials%' OR query LIKE '%symbiotic_matches%'
    ORDER BY total_time DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO authenticated; 