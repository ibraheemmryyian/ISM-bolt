-- Safe migration script for Supabase
-- This will only add missing tables and columns without dropping existing data

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create roles table if it doesn't exist
CREATE TABLE IF NOT EXISTS roles (
  id SERIAL PRIMARY KEY,
  name VARCHAR(50) UNIQUE NOT NULL
);

-- Create companies table if it doesn't exist
CREATE TABLE IF NOT EXISTS companies (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
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
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create materials table if it doesn't exist
CREATE TABLE IF NOT EXISTS materials (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  category VARCHAR(50),
  quantity_estimate VARCHAR(100),
  potential_value VARCHAR(100),
  quality_grade VARCHAR(20),
  potential_uses TEXT[],
  symbiosis_opportunities TEXT[],
  embeddings JSONB, -- Store embeddings as JSONB instead of vector
  ai_generated BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create requirements table if it doesn't exist
CREATE TABLE IF NOT EXISTS requirements (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  name VARCHAR(255) NOT NULL,
  description TEXT,
  category VARCHAR(50),
  quantity_needed VARCHAR(100),
  current_cost VARCHAR(100),
  priority VARCHAR(20),
  potential_sources TEXT[],
  symbiosis_opportunities TEXT[],
  embeddings JSONB, -- Store embeddings as JSONB instead of vector
  ai_generated BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create ai_insights table if it doesn't exist
CREATE TABLE IF NOT EXISTS ai_insights (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  symbiosis_score VARCHAR(10),
  estimated_savings VARCHAR(100),
  carbon_reduction VARCHAR(100),
  top_opportunities TEXT[],
  recommended_partners TEXT[],
  implementation_roadmap TEXT[],
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create matches table if it doesn't exist
CREATE TABLE IF NOT EXISTS matches (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  partner_company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  match_score DECIMAL(5,2),
  potential_savings DECIMAL(10,2),
  carbon_reduction DECIMAL(10,2),
  materials_involved TEXT[],
  status VARCHAR(20) DEFAULT 'pending',
  accepted_at TIMESTAMP WITH TIME ZONE,
  rejected_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create notifications table if it doesn't exist
CREATE TABLE IF NOT EXISTS notifications (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  type VARCHAR(50),
  title VARCHAR(255),
  message TEXT,
  data JSONB,
  read_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create users table if it doesn't exist
CREATE TABLE IF NOT EXISTS users (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255),
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create user_roles table if it doesn't exist
CREATE TABLE IF NOT EXISTS user_roles (
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  role_id INTEGER REFERENCES roles(id) ON DELETE CASCADE,
  PRIMARY KEY (user_id, role_id)
);

-- Create gdpr_consents table if it doesn't exist
CREATE TABLE IF NOT EXISTS gdpr_consents (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  consent_given BOOLEAN DEFAULT false,
  consent_type VARCHAR(100),
  consent_text TEXT,
  given_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add missing columns to existing tables (if they don't exist)
DO $$
BEGIN
  -- Add embeddings column to materials if it doesn't exist
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'materials' AND column_name = 'embeddings') THEN
    ALTER TABLE materials ADD COLUMN embeddings JSONB;
  END IF;
  
  -- Add embeddings column to requirements if it doesn't exist
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'requirements' AND column_name = 'embeddings') THEN
    ALTER TABLE requirements ADD COLUMN embeddings JSONB;
  END IF;
  
  -- Add ai_generated column to materials if it doesn't exist
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'materials' AND column_name = 'ai_generated') THEN
    ALTER TABLE materials ADD COLUMN ai_generated BOOLEAN DEFAULT false;
  END IF;
  
  -- Add ai_generated column to requirements if it doesn't exist
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'requirements' AND column_name = 'ai_generated') THEN
    ALTER TABLE requirements ADD COLUMN ai_generated BOOLEAN DEFAULT false;
  END IF;
END $$;

-- Create indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_materials_company_id ON materials(company_id);
CREATE INDEX IF NOT EXISTS idx_requirements_company_id ON requirements(company_id);
CREATE INDEX IF NOT EXISTS idx_matches_company_id ON matches(company_id);
CREATE INDEX IF NOT EXISTS idx_notifications_company_id ON notifications(company_id);

-- Create GIN indexes for JSONB embeddings columns (only if columns exist)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'materials' AND column_name = 'embeddings') THEN
    CREATE INDEX IF NOT EXISTS idx_materials_embeddings ON materials USING GIN (embeddings);
  END IF;
  
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'requirements' AND column_name = 'embeddings') THEN
    CREATE INDEX IF NOT EXISTS idx_requirements_embeddings ON requirements USING GIN (embeddings);
  END IF;
END $$;

-- Create update trigger for companies table if it doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_companies_updated_at ON companies;
CREATE TRIGGER update_companies_updated_at 
    BEFORE UPDATE ON companies 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert default roles if they don't exist
INSERT INTO roles (name) VALUES ('admin') ON CONFLICT DO NOTHING;
INSERT INTO roles (name) VALUES ('user') ON CONFLICT DO NOTHING;

-- Success message
SELECT 'Database schema updated successfully without losing data!' as status; 