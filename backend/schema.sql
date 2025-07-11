-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Companies table (migrate from localStorage)
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

-- AI-generated materials (outputs)
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
  embeddings JSONB, -- Store embeddings as JSONB instead of vector
  ai_generated BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI-generated requirements (inputs)
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
  embeddings JSONB, -- Store embeddings as JSONB instead of vector
  ai_generated BOOLEAN DEFAULT false,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI insights and recommendations
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

-- Matches between companies
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

-- Notifications
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

-- Users table
CREATE TABLE IF NOT EXISTS users (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255),
  company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Roles table
CREATE TABLE IF NOT EXISTS roles (
  id SERIAL PRIMARY KEY,
  name VARCHAR(50) UNIQUE NOT NULL
);

-- User roles (many-to-many)
CREATE TABLE IF NOT EXISTS user_roles (
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  role_id INTEGER REFERENCES roles(id) ON DELETE CASCADE,
  PRIMARY KEY (user_id, role_id)
);

-- GDPR consents
CREATE TABLE IF NOT EXISTS gdpr_consents (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  consent_given BOOLEAN DEFAULT false,
  consent_type VARCHAR(100),
  consent_text TEXT,
  given_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Default roles
INSERT INTO roles (name) VALUES ('admin') ON CONFLICT DO NOTHING;
INSERT INTO roles (name) VALUES ('user') ON CONFLICT DO NOTHING;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_materials_company_id ON materials(company_id);
CREATE INDEX IF NOT EXISTS idx_requirements_company_id ON requirements(company_id);
CREATE INDEX IF NOT EXISTS idx_matches_company_id ON matches(company_id);
CREATE INDEX IF NOT EXISTS idx_notifications_company_id ON notifications(company_id);
CREATE INDEX IF NOT EXISTS idx_materials_embeddings ON materials USING GIN (embeddings);
CREATE INDEX IF NOT EXISTS idx_requirements_embeddings ON requirements USING GIN (embeddings);

-- Update trigger for companies table
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_companies_updated_at 
    BEFORE UPDATE ON companies 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column(); 