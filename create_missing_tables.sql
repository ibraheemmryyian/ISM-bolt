-- Clean slate: Drop existing tables if they exist to avoid conflicts
DROP TABLE IF EXISTS activity_log CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS subscriptions CASCADE;
DROP TABLE IF EXISTS company_profiles CASCADE;

-- Create missing tables for SymbioFlows platform

-- Company Profiles table
CREATE TABLE IF NOT EXISTS company_profiles (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    company_id UUID,
    industry VARCHAR(100),
    location VARCHAR(200),
    description TEXT,
    website VARCHAR(255),
    contact_email VARCHAR(255),
    contact_phone VARCHAR(50),
    founded_year INTEGER,
    employee_count INTEGER,
    annual_revenue DECIMAL(15,2),
    sustainability_rating DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Subscriptions table
CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    company_id UUID,
    tier VARCHAR(50) DEFAULT 'free',
    status VARCHAR(20) DEFAULT 'active',
    starts_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    features JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Messages table (completely standalone)
CREATE TABLE IF NOT EXISTS messages (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    sender_id UUID,
    recipient_id UUID,
    subject VARCHAR(255),
    content TEXT,
    message_type VARCHAR(50) DEFAULT 'general',
    read_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Activity Log table
CREATE TABLE IF NOT EXISTS activity_log (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    company_id UUID,
    activity_type VARCHAR(100),
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Green Initiatives table
CREATE TABLE IF NOT EXISTS green_initiatives (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    category VARCHAR(100) NOT NULL,
    question TEXT NOT NULL,
    description TEXT,
    impact VARCHAR(20) CHECK (impact IN ('high', 'medium', 'low')),
    potential_savings DECIMAL(10,2),
    carbon_reduction DECIMAL(10,2),
    implementation_time VARCHAR(50),
    difficulty VARCHAR(20) CHECK (difficulty IN ('easy', 'medium', 'hard')),
    industry_filters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Green Initiative Responses table
CREATE TABLE IF NOT EXISTS green_initiative_responses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    initiative_id UUID REFERENCES green_initiatives(id) ON DELETE CASCADE,
    response VARCHAR(20) CHECK (response IN ('yes', 'no', 'planning')),
    notes TEXT,
    responded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(company_id, initiative_id)
);

-- Company Achievements table
CREATE TABLE IF NOT EXISTS company_achievements (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    total_savings DECIMAL(12,2) DEFAULT 0,
    carbon_reduced DECIMAL(10,2) DEFAULT 0,
    partnerships_formed INTEGER DEFAULT 0,
    waste_diverted DECIMAL(10,2) DEFAULT 0,
    matches_completed INTEGER DEFAULT 0,
    sustainability_score INTEGER DEFAULT 0,
    efficiency_improvement DECIMAL(5,2) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(company_id)
);

-- Company Activities table
CREATE TABLE IF NOT EXISTS company_activities (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    action VARCHAR(200) NOT NULL,
    impact TEXT,
    category VARCHAR(50) CHECK (category IN ('match', 'savings', 'partnership', 'sustainability')),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI Insights table
CREATE TABLE IF NOT EXISTS ai_insights (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    type VARCHAR(50) CHECK (type IN ('match', 'opportunity', 'suggestion', 'savings')),
    title VARCHAR(200) NOT NULL,
    description TEXT,
    impact VARCHAR(20) CHECK (impact IN ('high', 'medium', 'low')),
    estimated_savings DECIMAL(10,2),
    carbon_reduction DECIMAL(10,2),
    action_required BOOLEAN DEFAULT false,
    priority VARCHAR(20) CHECK (priority IN ('urgent', 'high', 'medium', 'low')),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'dismissed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Partnerships table to track successful matches
CREATE TABLE IF NOT EXISTS partnerships (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    company_a_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    company_b_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    match_type VARCHAR(100),
    savings_achieved DECIMAL(10,2),
    carbon_reduced DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'terminated')),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    notes TEXT
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_company_profiles_company_id ON company_profiles(company_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_company_id ON subscriptions(company_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);
CREATE INDEX IF NOT EXISTS idx_messages_sender_id ON messages(sender_id);
CREATE INDEX IF NOT EXISTS idx_messages_recipient_id ON messages(recipient_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_log_company_id ON activity_log(company_id);
CREATE INDEX IF NOT EXISTS idx_activity_log_created_at ON activity_log(created_at DESC);

-- Enable Row Level Security (RLS)
ALTER TABLE company_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE green_initiatives ENABLE ROW LEVEL SECURITY;
ALTER TABLE green_initiative_responses ENABLE ROW LEVEL SECURITY;
ALTER TABLE company_achievements ENABLE ROW LEVEL SECURITY;
ALTER TABLE company_activities ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE partnerships ENABLE ROW LEVEL SECURITY;

-- Create basic RLS policies (will be refined later)
CREATE POLICY "Allow all operations for now" ON company_profiles FOR ALL USING (true);
CREATE POLICY "Allow all operations for now" ON subscriptions FOR ALL USING (true);
CREATE POLICY "Allow all operations for now" ON messages FOR ALL USING (true);
CREATE POLICY "Allow all operations for now" ON activity_log FOR ALL USING (true);
CREATE POLICY "Companies can view their own green initiative responses" ON green_initiative_responses
    FOR SELECT USING (company_id = auth.uid()::text::uuid);
CREATE POLICY "Companies can insert their own green initiative responses" ON green_initiative_responses
    FOR INSERT WITH CHECK (company_id = auth.uid()::text::uuid);
CREATE POLICY "Companies can view their own achievements" ON company_achievements
    FOR SELECT USING (company_id = auth.uid()::text::uuid);
CREATE POLICY "Companies can view their own activities" ON company_activities
    FOR SELECT USING (company_id = auth.uid()::text::uuid);
CREATE POLICY "Companies can view their own insights" ON ai_insights
    FOR SELECT USING (company_id = auth.uid()::text::uuid);
CREATE POLICY "Companies can view their own partnerships" ON partnerships
    FOR SELECT USING (company_a_id = auth.uid()::text::uuid OR company_b_id = auth.uid()::text::uuid);

-- Function to calculate sustainability score
CREATE OR REPLACE FUNCTION calculate_sustainability_score(company_id UUID)
RETURNS INTEGER AS $$
DECLARE
    total_initiatives INTEGER;
    implemented_count INTEGER;
    score INTEGER;
BEGIN
    -- Count total initiatives
    SELECT COUNT(*) INTO total_initiatives FROM green_initiatives;
    
    -- Count implemented initiatives
    SELECT COUNT(*) INTO implemented_count 
    FROM green_initiative_responses 
    WHERE company_id = $1 AND response = 'yes';
    
    -- Calculate score (0-100)
    IF total_initiatives > 0 THEN
        score := (implemented_count * 100) / total_initiatives;
    ELSE
        score := 0;
    END IF;
    
    RETURN score;
END;
$$ LANGUAGE plpgsql; 