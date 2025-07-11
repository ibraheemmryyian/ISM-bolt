-- AI Schema Fix Migration
-- Add missing tables and columns for AI functionality

-- Requirements table for AI-generated input needs
CREATE TABLE IF NOT EXISTS requirements (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  company_id uuid REFERENCES companies(id) ON DELETE CASCADE,
  name text NOT NULL,
  description text,
  category text,
  quantity_needed text,
  current_cost text,
  priority text DEFAULT 'medium',
  potential_sources text[],
  symbiosis_opportunities text[],
  embeddings jsonb,
  ai_generated boolean DEFAULT false,
  created_at timestamptz DEFAULT now()
);

-- AI Insights table for storing AI-generated insights
CREATE TABLE IF NOT EXISTS ai_insights (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  company_id uuid REFERENCES companies(id) ON DELETE CASCADE,
  insight_type text NOT NULL,
  title text NOT NULL,
  description text,
  confidence integer DEFAULT 85,
  impact text DEFAULT 'medium',
  action_required boolean DEFAULT false,
  priority text DEFAULT 'medium',
  metadata jsonb,
  created_at timestamptz DEFAULT now()
);

-- Notifications table for real-time updates
CREATE TABLE IF NOT EXISTS notifications (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  company_id uuid REFERENCES companies(id) ON DELETE CASCADE,
  type text NOT NULL,
  title text NOT NULL,
  message text,
  data jsonb,
  read_at timestamptz,
  created_at timestamptz DEFAULT now()
);

-- Add missing columns to materials table
ALTER TABLE materials ADD COLUMN IF NOT EXISTS category text;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS quantity_estimate text;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS potential_value text;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS quality_grade text;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS potential_uses text[];
ALTER TABLE materials ADD COLUMN IF NOT EXISTS symbiosis_opportunities text[];
ALTER TABLE materials ADD COLUMN IF NOT EXISTS embeddings jsonb;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS ai_generated boolean DEFAULT false;

-- Add missing columns to companies table
ALTER TABLE companies ADD COLUMN IF NOT EXISTS products text;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS main_materials text;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS production_volume text;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS process_description text;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS sustainability_goals text[];
ALTER TABLE companies ADD COLUMN IF NOT EXISTS current_waste_management text;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS onboarding_completed boolean DEFAULT false;

-- Enable RLS on new tables
ALTER TABLE requirements ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;

-- RLS Policies for requirements
CREATE POLICY "Users can read all requirements" ON requirements
  FOR SELECT TO authenticated USING (true);

CREATE POLICY "Users can create requirements for their company" ON requirements
  FOR INSERT TO authenticated WITH CHECK (company_id = auth.uid());

CREATE POLICY "Users can update their own requirements" ON requirements
  FOR UPDATE TO authenticated USING (company_id = auth.uid());

-- RLS Policies for ai_insights
CREATE POLICY "Users can read their own AI insights" ON ai_insights
  FOR SELECT TO authenticated USING (company_id = auth.uid());

CREATE POLICY "Users can create AI insights for their company" ON ai_insights
  FOR INSERT TO authenticated WITH CHECK (company_id = auth.uid());

-- RLS Policies for notifications
CREATE POLICY "Users can read their own notifications" ON notifications
  FOR SELECT TO authenticated USING (company_id = auth.uid());

CREATE POLICY "Users can update their own notifications" ON notifications
  FOR UPDATE TO authenticated USING (company_id = auth.uid());

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_requirements_company_id ON requirements(company_id);
CREATE INDEX IF NOT EXISTS idx_ai_insights_company_id ON ai_insights(company_id);
CREATE INDEX IF NOT EXISTS idx_notifications_company_id ON notifications(company_id);
CREATE INDEX IF NOT EXISTS idx_materials_company_id ON materials(company_id); 