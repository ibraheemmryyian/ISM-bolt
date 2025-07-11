-- Fix missing company columns for AI onboarding
-- Run this in your Supabase SQL editor

-- Add missing columns that the frontend expects
ALTER TABLE companies ADD COLUMN IF NOT EXISTS employee_count INTEGER;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS industry VARCHAR(100);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS location VARCHAR(100);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS products TEXT;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS main_materials TEXT;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS production_volume VARCHAR(100);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS process_description TEXT;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS sustainability_goals TEXT[];
ALTER TABLE companies ADD COLUMN IF NOT EXISTS current_waste_management VARCHAR(100);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS onboarding_completed BOOLEAN DEFAULT false;

-- Add any other missing columns from the comprehensive schema
ALTER TABLE companies ADD COLUMN IF NOT EXISTS annual_revenue VARCHAR(50);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS operating_hours VARCHAR(50);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS waste_quantities TEXT;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS waste_frequencies VARCHAR(50);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS resource_needs TEXT;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS energy_consumption VARCHAR(100);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS environmental_certifications TEXT;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS current_recycling_practices TEXT;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS partnership_interests TEXT[];
ALTER TABLE companies ADD COLUMN IF NOT EXISTS geographic_preferences TEXT;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS technology_interests TEXT;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS ai_portfolio_summary TEXT;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS ai_recommendations JSONB;

-- Create ai_insights table if it doesn't exist
CREATE TABLE IF NOT EXISTS ai_insights (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  insight_type VARCHAR(50) NOT NULL,
  title VARCHAR(255) NOT NULL,
  description TEXT,
  confidence INTEGER DEFAULT 80,
  impact VARCHAR(20) DEFAULT 'medium',
  action_required BOOLEAN DEFAULT false,
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create matches table if it doesn't exist
CREATE TABLE IF NOT EXISTS matches (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  partner_company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  match_type VARCHAR(50),
  match_score DECIMAL(3,2),
  potential_savings DECIMAL(10,2),
  status VARCHAR(20) DEFAULT 'pending',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create requirements table if it doesn't exist
CREATE TABLE IF NOT EXISTS requirements (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  material_name VARCHAR(255) NOT NULL,
  quantity VARCHAR(100),
  unit VARCHAR(50),
  description TEXT,
  priority VARCHAR(20) DEFAULT 'medium',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on new tables
ALTER TABLE ai_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE requirements ENABLE ROW LEVEL SECURITY;

-- RLS Policies for ai_insights
CREATE POLICY "Users can read own ai insights"
  ON ai_insights FOR SELECT
  TO authenticated
  USING (company_id = auth.uid());

CREATE POLICY "Users can insert own ai insights"
  ON ai_insights FOR INSERT
  TO authenticated
  WITH CHECK (company_id = auth.uid());

-- RLS Policies for matches
CREATE POLICY "Users can read own matches"
  ON matches FOR SELECT
  TO authenticated
  USING (company_id = auth.uid() OR partner_company_id = auth.uid());

CREATE POLICY "Users can insert own matches"
  ON matches FOR INSERT
  TO authenticated
  WITH CHECK (company_id = auth.uid());

-- RLS Policies for requirements
CREATE POLICY "Users can read own requirements"
  ON requirements FOR SELECT
  TO authenticated
  USING (company_id = auth.uid());

CREATE POLICY "Users can insert own requirements"
  ON requirements FOR INSERT
  TO authenticated
  WITH CHECK (company_id = auth.uid());

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ai_insights_company_id ON ai_insights(company_id);
CREATE INDEX IF NOT EXISTS idx_matches_company_id ON matches(company_id);
CREATE INDEX IF NOT EXISTS idx_requirements_company_id ON requirements(company_id);

-- Add updated_at triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_ai_insights_updated_at
    BEFORE UPDATE ON ai_insights
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_matches_updated_at
    BEFORE UPDATE ON matches
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Verify the schema
SELECT 
    table_name,
    column_name,
    data_type
FROM information_schema.columns 
WHERE table_name = 'companies' 
ORDER BY ordinal_position; 