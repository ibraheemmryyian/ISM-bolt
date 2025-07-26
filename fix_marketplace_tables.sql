-- Fix Marketplace Loading Issues
-- Run this in your Supabase SQL editor to create missing tables and fix schema issues

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Ensure materials table exists with correct schema
CREATE TABLE IF NOT EXISTS materials (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  material_name VARCHAR(255) NOT NULL,
  type VARCHAR(50) NOT NULL CHECK (type IN ('waste', 'requirement', 'product')),
  category VARCHAR(100),
  quantity DECIMAL(15,2),
  unit VARCHAR(50),
  description TEXT,
  location VARCHAR(255),
  price_per_unit DECIMAL(10,2),
  current_cost VARCHAR(100),
  potential_sources TEXT[],
  quality_grade VARCHAR(20),
  potential_uses TEXT[],
  symbiosis_opportunities TEXT[],
  embeddings JSONB,
  ai_generated BOOLEAN DEFAULT false,
  status VARCHAR(50) DEFAULT 'active',
  material_properties JSONB,
  shipping_params JSONB,
  sustainability_metrics JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Ensure ai_insights table exists with correct schema
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

-- 3. Ensure matches table exists
CREATE TABLE IF NOT EXISTS matches (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  partner_company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  material_id UUID REFERENCES materials(id) ON DELETE CASCADE,
  partner_material_id UUID REFERENCES materials(id) ON DELETE CASCADE,
  match_score DECIMAL(5,4),
  potential_savings DECIMAL(15,2),
  carbon_reduction DECIMAL(10,2),
  status VARCHAR(50) DEFAULT 'pending',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Ensure requirements table exists
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

-- 5. Add missing columns to companies table
ALTER TABLE companies 
ADD COLUMN IF NOT EXISTS ai_portfolio_summary TEXT,
ADD COLUMN IF NOT EXISTS ai_recommendations JSONB,
ADD COLUMN IF NOT EXISTS sustainability_score DECIMAL(5,2) DEFAULT 0,
ADD COLUMN IF NOT EXISTS matches_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS savings_achieved DECIMAL(15,2) DEFAULT 0,
ADD COLUMN IF NOT EXISTS carbon_reduced DECIMAL(10,2) DEFAULT 0;

-- 6. Add missing columns to materials table
ALTER TABLE materials 
ADD COLUMN IF NOT EXISTS material_properties JSONB,
ADD COLUMN IF NOT EXISTS shipping_params JSONB,
ADD COLUMN IF NOT EXISTS sustainability_metrics JSONB;

-- 7. Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_materials_company_id ON materials(company_id);
CREATE INDEX IF NOT EXISTS idx_materials_type ON materials(type);
CREATE INDEX IF NOT EXISTS idx_materials_created_at ON materials(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_insights_company_id ON ai_insights(company_id);
CREATE INDEX IF NOT EXISTS idx_ai_insights_created_at ON ai_insights(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_matches_company_id ON matches(company_id);
CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);

-- 8. Enable Row Level Security
ALTER TABLE materials ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE requirements ENABLE ROW LEVEL SECURITY;

-- 9. Create RLS policies
-- Materials policies
CREATE POLICY "Users can view all materials" ON materials FOR SELECT TO authenticated USING (true);
CREATE POLICY "Users can insert their own materials" ON materials FOR INSERT TO authenticated WITH CHECK (company_id = auth.uid());
CREATE POLICY "Users can update their own materials" ON materials FOR UPDATE TO authenticated USING (company_id = auth.uid());

-- AI Insights policies
CREATE POLICY "Users can view all ai_insights" ON ai_insights FOR SELECT TO authenticated USING (true);
CREATE POLICY "Users can insert their own ai_insights" ON ai_insights FOR INSERT TO authenticated WITH CHECK (company_id = auth.uid());

-- Matches policies
CREATE POLICY "Users can view matches" ON matches FOR SELECT TO authenticated USING (company_id = auth.uid() OR partner_company_id = auth.uid());
CREATE POLICY "Users can insert matches" ON matches FOR INSERT TO authenticated WITH CHECK (company_id = auth.uid());

-- Requirements policies
CREATE POLICY "Users can view all requirements" ON requirements FOR SELECT TO authenticated USING (true);
CREATE POLICY "Users can insert their own requirements" ON requirements FOR INSERT TO authenticated WITH CHECK (company_id = auth.uid());

-- 10. Insert sample data for testing (optional)
-- This will help ensure the marketplace loads properly
INSERT INTO materials (company_id, material_name, type, category, quantity, unit, description, ai_generated)
SELECT 
  c.id,
  'Sample Material',
  'waste',
  'Industrial Waste',
  100.0,
  'tons',
  'Sample material for testing marketplace functionality',
  true
FROM companies c
WHERE c.id = (SELECT id FROM companies LIMIT 1)
ON CONFLICT DO NOTHING;

-- 11. Verify tables were created successfully
SELECT 'Marketplace tables created successfully!' as status;
SELECT 
  table_name, 
  column_count 
FROM (
  SELECT 'materials' as table_name, COUNT(*) as column_count FROM information_schema.columns WHERE table_name = 'materials'
  UNION ALL
  SELECT 'ai_insights', COUNT(*) FROM information_schema.columns WHERE table_name = 'ai_insights'
  UNION ALL
  SELECT 'matches', COUNT(*) FROM information_schema.columns WHERE table_name = 'matches'
  UNION ALL
  SELECT 'requirements', COUNT(*) FROM information_schema.columns WHERE table_name = 'requirements'
) as table_info; 