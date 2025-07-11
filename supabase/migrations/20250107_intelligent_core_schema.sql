-- =====================================================
-- SYMBIOFLOWS INTELLIGENT CORE SCHEMA MIGRATION
-- =====================================================

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- =====================================================
-- NEW TABLES FOR INTELLIGENT CORE
-- =====================================================

-- AI interactions table - Track AI analysis requests
CREATE TABLE IF NOT EXISTS ai_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_type VARCHAR(100) NOT NULL,
    input_data JSONB NOT NULL,
    response_data JSONB,
    context JSONB,
    confidence_score DECIMAL(5,4),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI errors table - Track AI analysis errors
CREATE TABLE IF NOT EXISTS ai_errors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_type VARCHAR(100) NOT NULL,
    input_data JSONB,
    error_message TEXT NOT NULL,
    error_stack TEXT,
    context JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Feedback requests table - Track feedback collection requests
CREATE TABLE IF NOT EXISTS feedback_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feedback_id VARCHAR(255) UNIQUE NOT NULL,
    analysis_type VARCHAR(100) NOT NULL,
    input_data JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User feedback table - Store user feedback for AI improvement
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feedback_id VARCHAR(255) NOT NULL,
    user_ratings JSONB,
    qualitative_feedback TEXT,
    improvement_suggestions TEXT,
    overall_satisfaction INTEGER CHECK (overall_satisfaction >= 1 AND overall_satisfaction <= 5),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API call logs table - Monitor external API usage
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

-- Shipping requests table - Track shipping rate requests
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

-- Shipping labels table - Track created shipping labels
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
-- ENHANCE EXISTING TABLES
-- =====================================================

-- Add new columns to companies table
ALTER TABLE companies 
ADD COLUMN IF NOT EXISTS ai_portfolio_summary TEXT,
ADD COLUMN IF NOT EXISTS ai_recommendations JSONB,
ADD COLUMN IF NOT EXISTS sustainability_score DECIMAL(5,2) DEFAULT 0,
ADD COLUMN IF NOT EXISTS matches_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS savings_achieved DECIMAL(15,2) DEFAULT 0,
ADD COLUMN IF NOT EXISTS carbon_reduced DECIMAL(10,2) DEFAULT 0;

-- Add new columns to materials table
ALTER TABLE materials 
ADD COLUMN IF NOT EXISTS material_properties JSONB,
ADD COLUMN IF NOT EXISTS shipping_params JSONB,
ADD COLUMN IF NOT EXISTS sustainability_metrics JSONB;

-- Add new columns to symbiotic_matches table
ALTER TABLE symbiotic_matches 
ADD COLUMN IF NOT EXISTS materials_compatibility DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS waste_synergy DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS energy_synergy DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS location_proximity DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS ai_confidence DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS match_analysis JSONB,
ADD COLUMN IF NOT EXISTS user_feedback JSONB;

-- Add new columns to shipments table
ALTER TABLE shipments 
ADD COLUMN IF NOT EXISTS weight_kg DECIMAL(10,2),
ADD COLUMN IF NOT EXISTS volume_cubic_meters DECIMAL(10,2),
ADD COLUMN IF NOT EXISTS special_handling TEXT[],
ADD COLUMN IF NOT EXISTS packaging_requirements VARCHAR(100),
ADD COLUMN IF NOT EXISTS temperature_requirements JSONB,
ADD COLUMN IF NOT EXISTS test_mode BOOLEAN DEFAULT FALSE;

-- =====================================================
-- PERFORMANCE INDEXES
-- =====================================================

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

-- Enhanced company indexes
CREATE INDEX IF NOT EXISTS idx_companies_sustainability_score ON companies(sustainability_score DESC);
CREATE INDEX IF NOT EXISTS idx_companies_matches_count ON companies(matches_count DESC);
CREATE INDEX IF NOT EXISTS idx_companies_savings ON companies(savings_achieved DESC);

-- Enhanced material indexes
CREATE INDEX IF NOT EXISTS idx_materials_properties ON materials USING gin(material_properties);
CREATE INDEX IF NOT EXISTS idx_materials_shipping ON materials USING gin(shipping_params);

-- Enhanced match indexes
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_compatibility ON symbiotic_matches(materials_compatibility DESC);
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_confidence ON symbiotic_matches(ai_confidence DESC);
CREATE INDEX IF NOT EXISTS idx_symbiotic_matches_analysis ON symbiotic_matches USING gin(match_analysis);

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

-- Apply updated_at trigger to new tables
DROP TRIGGER IF EXISTS update_ai_interactions_updated_at ON ai_interactions;
CREATE TRIGGER update_ai_interactions_updated_at 
    BEFORE UPDATE ON ai_interactions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_feedback_requests_updated_at ON feedback_requests;
CREATE TRIGGER update_feedback_requests_updated_at 
    BEFORE UPDATE ON feedback_requests 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

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
CREATE TRIGGER update_match_count 
    AFTER INSERT OR DELETE ON symbiotic_matches 
    FOR EACH ROW EXECUTE FUNCTION update_company_match_count();

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
-- ROW LEVEL SECURITY (RLS)
-- =====================================================

-- Enable RLS on new tables
ALTER TABLE ai_interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_errors ENABLE ROW LEVEL SECURITY;
ALTER TABLE feedback_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_call_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE shipping_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE shipping_labels ENABLE ROW LEVEL SECURITY;

-- AI interactions RLS policies
CREATE POLICY "Users can view their own AI interactions" ON ai_interactions 
    FOR SELECT USING (true); -- Allow read access for analytics

CREATE POLICY "System can insert AI interactions" ON ai_interactions 
    FOR INSERT WITH CHECK (true);

-- AI errors RLS policies
CREATE POLICY "System can manage AI errors" ON ai_errors 
    FOR ALL USING (true);

-- Feedback RLS policies
CREATE POLICY "Users can view their own feedback" ON user_feedback 
    FOR SELECT USING (true);

CREATE POLICY "Users can insert their own feedback" ON user_feedback 
    FOR INSERT WITH CHECK (true);

-- API call logs RLS policies
CREATE POLICY "System can manage API logs" ON api_call_logs 
    FOR ALL USING (true);

-- Shipping RLS policies
CREATE POLICY "Users can view shipping for their company" ON shipping_requests 
    FOR SELECT USING (
        company_id IN (SELECT id FROM companies WHERE user_id = auth.uid())
    );

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
    AVG(EXTRACT(EPOCH FROM (timestamp - LAG(timestamp) OVER (PARTITION BY analysis_type ORDER BY timestamp)))) as avg_response_time_seconds
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