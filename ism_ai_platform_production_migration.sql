-- =====================================================
-- ISM AI PLATFORM: PRODUCTION MIGRATION SCRIPT
-- Creates 20+ Essential Tables for Real Data Integration
-- Safe for Supabase/Postgres
-- =====================================================

-- =========================
-- COMPANIES & PROFILES
-- =========================
CREATE TABLE IF NOT EXISTS companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    role VARCHAR(50),
    level INTEGER,
    xp INTEGER,
    streak_days INTEGER,
    last_activity TIMESTAMP WITH TIME ZONE,
    has_completed_onboarding BOOLEAN,
    username VARCHAR(255),
    contact_name VARCHAR(255),
    description TEXT,
    industry VARCHAR(100),
    location VARCHAR(255),
    processes TEXT,
    products TEXT,
    volume VARCHAR(100),
    subscription_tier VARCHAR(50),
    subscription_status VARCHAR(50),
    subscription_expires_at TIMESTAMP WITH TIME ZONE,
    current_waste_management VARCHAR(100),
    waste_quantity DECIMAL,
    waste_unit VARCHAR(50),
    waste_frequency VARCHAR(50),
    user_type VARCHAR(50),
    onboarding_completed BOOLEAN,
    user_id UUID,
    process_description TEXT,
    application_status VARCHAR(50),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ai_portfolio_summary TEXT,
    ai_recommendations TEXT,
    sustainability_score DECIMAL,
    matches_count INTEGER,
    savings_achieved DECIMAL,
    carbon_reduced DECIMAL,
    employee_count INTEGER,
    main_materials TEXT,
    production_volume VARCHAR(100),
    sustainability_goals TEXT[],
    annual_revenue DECIMAL,
    operating_hours VARCHAR(100),
    waste_quantities TEXT,
    waste_frequencies TEXT,
    resource_needs TEXT,
    energy_consumption DECIMAL,
    environmental_certifications TEXT,
    current_recycling_practices TEXT,
    partnership_interests TEXT,
    geographic_preferences TEXT,
    technology_interests TEXT,
    carbon_footprint DECIMAL,
    water_usage DECIMAL
);

CREATE TABLE IF NOT EXISTS company_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    industry VARCHAR(100),
    location VARCHAR(255),
    description TEXT,
    website VARCHAR(255),
    contact_email VARCHAR(255),
    contact_phone VARCHAR(50),
    founded_year INTEGER,
    employee_count INTEGER,
    annual_revenue DECIMAL,
    sustainability_rating DECIMAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- USERS & ROLES
-- =========================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT,
    company_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255),
    email VARCHAR(255),
    company_name VARCHAR(255),
    role VARCHAR(50),
    email_verified_at TIMESTAMP WITH TIME ZONE,
    email_verification_sent_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS user_roles (
    user_id UUID,
    role_id UUID
);

-- =========================
-- MATERIALS & MATCHING
-- =========================
CREATE TABLE IF NOT EXISTS materials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    material_name VARCHAR(255),
    quantity DECIMAL,
    unit VARCHAR(50),
    description TEXT,
    type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ai_tags TEXT,
    estimated_value DECIMAL,
    priority_score DECIMAL,
    is_sponsored BOOLEAN,
    embeddings JSONB,
    ai_generated BOOLEAN,
    availability VARCHAR(50),
    location VARCHAR(255),
    price_per_unit DECIMAL,
    current_cost DECIMAL,
    potential_sources TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    category VARCHAR(100),
    status VARCHAR(50),
    material_properties JSONB,
    shipping_params JSONB,
    sustainability_metrics JSONB
);

CREATE TABLE IF NOT EXISTS material_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS material_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    material_id UUID REFERENCES materials(id) ON DELETE CASCADE,
    matched_material_id UUID,
    match_score DECIMAL,
    status VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- AI & ANALYTICS
-- =========================
CREATE TABLE IF NOT EXISTS ai_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    insight_type VARCHAR(100),
    title VARCHAR(255),
    description TEXT,
    confidence DECIMAL,
    impact VARCHAR(50),
    action_required BOOLEAN,
    action_url TEXT,
    metadata JSONB,
    status VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    confidence_score DECIMAL
);

CREATE TABLE IF NOT EXISTS ai_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    type VARCHAR(100),
    title VARCHAR(255),
    description TEXT,
    confidence DECIMAL,
    action_url TEXT,
    status VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ai_portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    portfolio_data JSONB,
    status VARCHAR(50),
    approved_at TIMESTAMP WITH TIME ZONE,
    approved_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- MATCHES & SYMBIOSIS
-- =========================
CREATE TABLE IF NOT EXISTS matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    partner_company_id UUID,
    match_score DECIMAL,
    potential_savings DECIMAL,
    carbon_reduction DECIMAL,
    materials_involved TEXT,
    status VARCHAR(50),
    accepted_at TIMESTAMP WITH TIME ZONE,
    rejected_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS symbiosis_opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    title VARCHAR(255),
    description TEXT,
    type VARCHAR(50),
    potential_partners TEXT,
    estimated_savings DECIMAL,
    environmental_impact TEXT,
    implementation_timeline VARCHAR(100),
    difficulty_level VARCHAR(50),
    status VARCHAR(50),
    ai_generated BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS symbiotic_matches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_a_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    company_b_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    material_a_id UUID,
    material_b_id UUID,
    match_score DECIMAL,
    match_type VARCHAR(50),
    potential_savings DECIMAL,
    implementation_complexity VARCHAR(50),
    environmental_impact TEXT,
    description TEXT,
    materials_compatibility TEXT,
    waste_synergy TEXT,
    energy_synergy TEXT,
    location_proximity TEXT,
    ai_confidence DECIMAL,
    match_analysis TEXT,
    status VARCHAR(50),
    user_feedback TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- LOGISTICS & SHIPPING
-- =========================
CREATE TABLE IF NOT EXISTS shipments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id UUID,
    material_id UUID,
    from_company_id UUID,
    to_company_id UUID,
    carrier VARCHAR(100),
    service VARCHAR(100),
    tracking_number VARCHAR(100),
    tracking_url TEXT,
    label_url TEXT,
    shipping_cost DECIMAL,
    currency VARCHAR(10),
    weight_kg DECIMAL,
    volume_cubic_meters DECIMAL,
    special_handling TEXT,
    packaging_requirements TEXT,
    temperature_requirements TEXT,
    status VARCHAR(50),
    estimated_delivery TIMESTAMP WITH TIME ZONE,
    actual_delivery TIMESTAMP WITH TIME ZONE,
    test_mode BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS shipping_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    material_name VARCHAR(255),
    material_type VARCHAR(100),
    quantity DECIMAL,
    unit VARCHAR(50),
    from_location VARCHAR(255),
    to_location VARCHAR(255),
    rates_count INTEGER,
    lowest_price DECIMAL,
    test_mode BOOLEAN,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- SUBSCRIPTIONS & BILLING
-- =========================
CREATE TABLE IF NOT EXISTS subscription_tiers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100),
    price_monthly DECIMAL,
    features JSONB,
    limits JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    tier VARCHAR(100),
    status VARCHAR(50),
    starts_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    features JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- NOTIFICATIONS & MESSAGES
-- =========================
CREATE TABLE IF NOT EXISTS notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    type VARCHAR(50),
    title VARCHAR(255),
    message TEXT,
    data JSONB,
    read_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sender_id UUID,
    recipient_id UUID,
    subject VARCHAR(255),
    content TEXT,
    message_type VARCHAR(50),
    read_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- LOGS & ACTIVITY
-- =========================
CREATE TABLE IF NOT EXISTS activity_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    activity_type VARCHAR(100),
    description TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    user_id UUID,
    activity_type VARCHAR(100),
    activity_data JSONB,
    ip_address VARCHAR(50),
    user_agent VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- FEEDBACK & GDPR
-- =========================
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feedback_id UUID,
    user_ratings JSONB,
    qualitative_feedback TEXT,
    improvement_suggestions TEXT,
    overall_satisfaction DECIMAL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gdpr_consents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    consent_given BOOLEAN,
    consent_type VARCHAR(100),
    consent_text TEXT,
    given_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- LOCATIONS
-- =========================
CREATE TABLE IF NOT EXISTS locations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255),
    country VARCHAR(100),
    region VARCHAR(100),
    latitude DECIMAL,
    longitude DECIMAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- CONNECTIONS & PARTNERSHIPS
-- =========================
CREATE TABLE IF NOT EXISTS connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    requester_id UUID,
    recipient_id UUID,
    status VARCHAR(50),
    message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS partnership_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    requester_id UUID,
    recipient_id UUID,
    material_id UUID,
    request_type VARCHAR(100),
    message TEXT,
    status VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    responded_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- =========================
-- TRANSACTIONS
-- =========================
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connection_id UUID,
    buyer_id UUID,
    seller_id UUID,
    material_id UUID,
    amount DECIMAL,
    fee DECIMAL,
    fee_percentage DECIMAL,
    status VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- API LOGS
-- =========================
CREATE TABLE IF NOT EXISTS api_call_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_name VARCHAR(100),
    endpoint VARCHAR(255),
    method VARCHAR(10),
    status VARCHAR(50),
    duration_ms INTEGER,
    request_id UUID,
    error_message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =========================
-- INDEXES (EXAMPLES)
-- =========================
CREATE INDEX IF NOT EXISTS idx_companies_name ON companies(name);
CREATE INDEX IF NOT EXISTS idx_materials_company_id ON materials(company_id);
CREATE INDEX IF NOT EXISTS idx_matches_company_id ON matches(company_id);
CREATE INDEX IF NOT EXISTS idx_notifications_company_id ON notifications(company_id);
CREATE INDEX IF NOT EXISTS idx_activity_logs_company_id ON activity_logs(company_id);

-- =========================
-- MIGRATION COMPLETION
-- =========================
DO $$
BEGIN
    RAISE NOTICE '🎉 ISM AI Platform migration completed: 20+ tables created!';
END $$; 