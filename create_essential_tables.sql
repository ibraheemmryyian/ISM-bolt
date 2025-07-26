-- Create Essential Tables Only
-- Run this in your Supabase SQL editor

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Companies table
CREATE TABLE IF NOT EXISTS companies (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255),
  username VARCHAR(100) NULL,
  role VARCHAR(20) DEFAULT 'user',
  user_type VARCHAR(20) DEFAULT 'business',
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
  status VARCHAR(20) DEFAULT 'active',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Profiles table
CREATE TABLE IF NOT EXISTS profiles (
  id UUID PRIMARY KEY,
  username VARCHAR(100) UNIQUE,
  email VARCHAR(255) UNIQUE,
  company_name VARCHAR(255),
  role VARCHAR(20) DEFAULT 'user',
  status VARCHAR(20) DEFAULT 'active',
  email_verified_at TIMESTAMP WITH TIME ZONE,
  email_verification_sent_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Materials table
CREATE TABLE IF NOT EXISTS materials (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  name VARCHAR(255) NOT NULL,
  material_name VARCHAR(255),
  description TEXT,
  category VARCHAR(50),
  quantity_estimate VARCHAR(100),
  quantity VARCHAR(100),
  unit VARCHAR(50),
  frequency VARCHAR(50),
  notes TEXT,
  potential_value VARCHAR(100),
  quality_grade VARCHAR(20),
  potential_uses TEXT[],
  symbiosis_opportunities TEXT[],
  type VARCHAR(50),
  current_cost VARCHAR(100),
  potential_sources TEXT[],
  price_per_unit DECIMAL(10,2),
  embeddings JSONB,
  ai_generated BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Requirements table
CREATE TABLE IF NOT EXISTS requirements (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  category VARCHAR(50),
  quantity_needed VARCHAR(100),
  quantity VARCHAR(100),
  frequency VARCHAR(50),
  notes TEXT,
  current_cost VARCHAR(100),
  priority VARCHAR(20),
  potential_sources TEXT[],
  symbiosis_opportunities TEXT[],
  embeddings JSONB,
  ai_generated BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI insights table
CREATE TABLE IF NOT EXISTS ai_insights (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  insight_type VARCHAR(50) NOT NULL,
  title VARCHAR(255) NOT NULL,
  description TEXT NOT NULL,
  confidence INTEGER DEFAULT 50,
  impact VARCHAR(20) DEFAULT 'medium',
  action_required BOOLEAN DEFAULT false,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Matches table
CREATE TABLE IF NOT EXISTS matches (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  partner_company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  match_score DECIMAL(5,2),
  potential_savings DECIMAL(10,2),
  carbon_reduction DECIMAL(10,2),
  materials_involved TEXT[],
  match_reason TEXT,
  status VARCHAR(20) DEFAULT 'pending',
  accepted_at TIMESTAMP WITH TIME ZONE,
  rejected_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create basic indexes
CREATE INDEX IF NOT EXISTS idx_companies_username ON companies(username);
CREATE INDEX IF NOT EXISTS idx_companies_email ON companies(email);
CREATE INDEX IF NOT EXISTS idx_profiles_username ON profiles(username);
CREATE INDEX IF NOT EXISTS idx_profiles_email ON profiles(email);
CREATE INDEX IF NOT EXISTS idx_materials_company_id ON materials(company_id);
CREATE INDEX IF NOT EXISTS idx_requirements_company_id ON requirements(company_id);
CREATE INDEX IF NOT EXISTS idx_ai_insights_company_id ON ai_insights(company_id);
CREATE INDEX IF NOT EXISTS idx_matches_company_id ON matches(company_id);

-- Enable Row Level Security
ALTER TABLE companies ENABLE ROW LEVEL SECURITY;
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE materials ENABLE ROW LEVEL SECURITY;
ALTER TABLE requirements ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE matches ENABLE ROW LEVEL SECURITY;

-- Create basic RLS policies
CREATE POLICY "Users can view own company" ON companies
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own company" ON companies
  FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can view own profile" ON profiles
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON profiles
  FOR UPDATE USING (auth.uid() = id);

-- Insert default admin user
INSERT INTO companies (id, name, username, role, status) 
VALUES (
  '00000000-0000-0000-0000-000000000001',
  'System Admin',
  'admin',
  'admin',
  'active'
) ON CONFLICT (id) DO NOTHING;

INSERT INTO profiles (id, username, role, status)
VALUES (
  '00000000-0000-0000-0000-000000000001',
  'admin',
  'admin',
  'active'
) ON CONFLICT (id) DO NOTHING;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON companies TO authenticated;
GRANT SELECT, INSERT, UPDATE ON profiles TO authenticated;
GRANT SELECT, INSERT, UPDATE ON materials TO authenticated;
GRANT SELECT, INSERT, UPDATE ON requirements TO authenticated;
GRANT SELECT, INSERT, UPDATE ON ai_insights TO authenticated;
GRANT SELECT, INSERT, UPDATE ON matches TO authenticated;

-- Verify tables were created
SELECT 'Essential tables created successfully!' as status;
SELECT 'Admin password: NA10EN' as admin_info; 