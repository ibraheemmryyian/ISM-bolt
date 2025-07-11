-- Comprehensive Database Schema for AI Onboarding
-- Run this in your Supabase SQL editor

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Update companies table with all AI onboarding fields
ALTER TABLE companies ADD COLUMN IF NOT EXISTS onboarding_completed BOOLEAN DEFAULT false;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS current_waste_management VARCHAR(100);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS waste_quantity VARCHAR(100);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS waste_unit VARCHAR(50);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS waste_frequency VARCHAR(50);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS user_type VARCHAR(20) DEFAULT 'business';

-- Add comprehensive AI onboarding fields
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

-- Create symbiosis_opportunities table
CREATE TABLE IF NOT EXISTS symbiosis_opportunities (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  title VARCHAR(255) NOT NULL,
  description TEXT,
  type VARCHAR(100),
  potential_partners TEXT[],
  estimated_savings VARCHAR(100),
  environmental_impact VARCHAR(100),
  implementation_timeline VARCHAR(100),
  difficulty_level VARCHAR(50),
  status VARCHAR(50) DEFAULT 'pending',
  ai_generated BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create ai_portfolios table for storing generated portfolios
CREATE TABLE IF NOT EXISTS ai_portfolios (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  portfolio_data JSONB NOT NULL,
  status VARCHAR(50) DEFAULT 'pending_approval',
  approved_at TIMESTAMP WITH TIME ZONE,
  approved_by UUID,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create portfolio_materials table for AI-generated materials
CREATE TABLE IF NOT EXISTS portfolio_materials (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  portfolio_id UUID REFERENCES ai_portfolios(id) ON DELETE CASCADE,
  material_name VARCHAR(255) NOT NULL,
  quantity VARCHAR(100),
  unit VARCHAR(50),
  type VARCHAR(50),
  description TEXT,
  availability VARCHAR(50),
  price_per_unit DECIMAL(10,2),
  quality_grade VARCHAR(20),
  potential_uses TEXT[],
  potential_sources TEXT[],
  ai_generated BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on all tables
ALTER TABLE companies ENABLE ROW LEVEL SECURITY;
ALTER TABLE symbiosis_opportunities ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_portfolios ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_materials ENABLE ROW LEVEL SECURITY;

-- RLS Policies for companies
DROP POLICY IF EXISTS "Users can read own company data" ON companies;
DROP POLICY IF EXISTS "Users can update own company data" ON companies;
DROP POLICY IF EXISTS "Users can insert own company data" ON companies;

CREATE POLICY "Users can read own company data"
  ON companies FOR SELECT
  TO authenticated
  USING (auth.uid() = id);

CREATE POLICY "Users can update own company data"
  ON companies FOR UPDATE
  TO authenticated
  USING (auth.uid() = id);

CREATE POLICY "Users can insert own company data"
  ON companies FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = id);

-- RLS Policies for symbiosis_opportunities
CREATE POLICY "Users can read own opportunities"
  ON symbiosis_opportunities FOR SELECT
  TO authenticated
  USING (company_id = auth.uid());

CREATE POLICY "Users can insert own opportunities"
  ON symbiosis_opportunities FOR INSERT
  TO authenticated
  WITH CHECK (company_id = auth.uid());

CREATE POLICY "Users can update own opportunities"
  ON symbiosis_opportunities FOR UPDATE
  TO authenticated
  USING (company_id = auth.uid());

-- RLS Policies for ai_portfolios
CREATE POLICY "Users can read own portfolios"
  ON ai_portfolios FOR SELECT
  TO authenticated
  USING (company_id = auth.uid());

CREATE POLICY "Users can insert own portfolios"
  ON ai_portfolios FOR INSERT
  TO authenticated
  WITH CHECK (company_id = auth.uid());

CREATE POLICY "Users can update own portfolios"
  ON ai_portfolios FOR UPDATE
  TO authenticated
  USING (company_id = auth.uid());

-- RLS Policies for portfolio_materials
CREATE POLICY "Users can read own portfolio materials"
  ON portfolio_materials FOR SELECT
  TO authenticated
  USING (portfolio_id IN (
    SELECT id FROM ai_portfolios WHERE company_id = auth.uid()
  ));

CREATE POLICY "Users can insert own portfolio materials"
  ON portfolio_materials FOR INSERT
  TO authenticated
  WITH CHECK (portfolio_id IN (
    SELECT id FROM ai_portfolios WHERE company_id = auth.uid()
  ));

CREATE POLICY "Users can update own portfolio materials"
  ON portfolio_materials FOR UPDATE
  TO authenticated
  USING (portfolio_id IN (
    SELECT id FROM ai_portfolios WHERE company_id = auth.uid()
  ));

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_companies_user_type ON companies(user_type);
CREATE INDEX IF NOT EXISTS idx_companies_industry ON companies(industry);
CREATE INDEX IF NOT EXISTS idx_companies_location ON companies(location);
CREATE INDEX IF NOT EXISTS idx_symbiosis_opportunities_company_id ON symbiosis_opportunities(company_id);
CREATE INDEX IF NOT EXISTS idx_symbiosis_opportunities_type ON symbiosis_opportunities(type);
CREATE INDEX IF NOT EXISTS idx_ai_portfolios_company_id ON ai_portfolios(company_id);
CREATE INDEX IF NOT EXISTS idx_ai_portfolios_status ON ai_portfolios(status);
CREATE INDEX IF NOT EXISTS idx_portfolio_materials_portfolio_id ON portfolio_materials(portfolio_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
DROP TRIGGER IF EXISTS update_companies_updated_at ON companies;
DROP TRIGGER IF EXISTS update_symbiosis_opportunities_updated_at ON symbiosis_opportunities;
DROP TRIGGER IF EXISTS update_ai_portfolios_updated_at ON ai_portfolios;

CREATE TRIGGER update_companies_updated_at
    BEFORE UPDATE ON companies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_symbiosis_opportunities_updated_at
    BEFORE UPDATE ON symbiosis_opportunities
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_portfolios_updated_at
    BEFORE UPDATE ON ai_portfolios
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Verify schema
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_name IN ('companies', 'symbiosis_opportunities', 'ai_portfolios', 'portfolio_materials')
ORDER BY table_name, ordinal_position; 