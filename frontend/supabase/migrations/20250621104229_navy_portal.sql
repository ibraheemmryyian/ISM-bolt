/*
  # Enhanced SymbioFlow Database Schema

  1. New Tables
    - `subscription_tiers` - Define available subscription tiers
    - `user_achievements` - Track user achievements and gamification
    - `ai_insights` - Store AI-generated insights and recommendations
    - `material_analytics` - Track material performance and metrics
    - `partnership_requests` - Manage connection requests between companies

  2. Enhanced Tables
    - Add tier-based features to existing tables
    - Add gamification elements
    - Add AI-powered fields

  3. Security
    - Enable RLS on all new tables
    - Add appropriate policies for each table
*/

-- Subscription tiers definition
CREATE TABLE IF NOT EXISTS subscription_tiers (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL UNIQUE,
  price_monthly numeric NOT NULL DEFAULT 0,
  features jsonb NOT NULL DEFAULT '{}',
  limits jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz DEFAULT now()
);

-- Insert default tiers
INSERT INTO subscription_tiers (name, price_monthly, features, limits) VALUES
('free', 0, 
 '{"manual_listings": true, "basic_messaging": true, "marketplace_access": true}',
 '{"max_listings": 5, "max_connections": 10, "ai_features": false}'),
('pro', 299, 
 '{"ai_matching": true, "sponsored_listings": true, "advanced_analytics": true, "priority_support": true, "api_access": true}',
 '{"max_listings": -1, "max_connections": -1, "ai_features": true}'),
('enterprise', 999, 
 '{"custom_ai": true, "dedicated_support": true, "white_label": true, "advanced_integrations": true}',
 '{"max_listings": -1, "max_connections": -1, "ai_features": true, "custom_features": true}')
ON CONFLICT (name) DO NOTHING;

-- User achievements for gamification
CREATE TABLE IF NOT EXISTS user_achievements (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  company_id uuid REFERENCES companies(id),
  achievement_type text NOT NULL,
  title text NOT NULL,
  description text NOT NULL,
  xp_reward integer NOT NULL DEFAULT 0,
  unlocked_at timestamptz DEFAULT now(),
  progress integer DEFAULT 100,
  metadata jsonb DEFAULT '{}'
);

ALTER TABLE user_achievements ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own achievements"
  ON user_achievements
  FOR SELECT
  TO authenticated
  USING (company_id = auth.uid());

-- AI insights and recommendations
CREATE TABLE IF NOT EXISTS ai_insights (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  company_id uuid REFERENCES companies(id),
  insight_type text NOT NULL CHECK (insight_type IN ('opportunity', 'warning', 'trend', 'optimization')),
  title text NOT NULL,
  description text NOT NULL,
  confidence integer NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
  impact text NOT NULL CHECK (impact IN ('low', 'medium', 'high')),
  action_required boolean DEFAULT false,
  action_url text,
  metadata jsonb DEFAULT '{}',
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'dismissed', 'acted')),
  created_at timestamptz DEFAULT now(),
  expires_at timestamptz
);

ALTER TABLE ai_insights ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own insights"
  ON ai_insights
  FOR SELECT
  TO authenticated
  USING (company_id = auth.uid());

CREATE POLICY "Users can update own insights"
  ON ai_insights
  FOR UPDATE
  TO authenticated
  USING (company_id = auth.uid());

-- Material analytics and performance tracking
CREATE TABLE IF NOT EXISTS material_analytics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  material_id uuid REFERENCES materials(id),
  views integer DEFAULT 0,
  inquiries integer DEFAULT 0,
  connections_made integer DEFAULT 0,
  estimated_value numeric DEFAULT 0,
  performance_score integer DEFAULT 0,
  last_updated timestamptz DEFAULT now()
);

ALTER TABLE material_analytics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read analytics for their materials"
  ON material_analytics
  FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM materials 
      WHERE id = material_analytics.material_id 
      AND company_id = auth.uid()
    )
  );

-- Partnership requests and connection management
CREATE TABLE IF NOT EXISTS partnership_requests (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  requester_id uuid REFERENCES companies(id),
  recipient_id uuid REFERENCES companies(id),
  material_id uuid REFERENCES materials(id),
  request_type text NOT NULL CHECK (request_type IN ('connection', 'partnership', 'inquiry')),
  message text,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'declined', 'expired')),
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now(),
  responded_at timestamptz,
  expires_at timestamptz DEFAULT (now() + interval '30 days')
);

ALTER TABLE partnership_requests ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read their partnership requests"
  ON partnership_requests
  FOR SELECT
  TO authenticated
  USING (requester_id = auth.uid() OR recipient_id = auth.uid());

CREATE POLICY "Users can create partnership requests"
  ON partnership_requests
  FOR INSERT
  TO authenticated
  WITH CHECK (requester_id = auth.uid());

CREATE POLICY "Recipients can update requests"
  ON partnership_requests
  FOR UPDATE
  TO authenticated
  USING (recipient_id = auth.uid());

-- Add gamification fields to companies table
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'level') THEN
    ALTER TABLE companies ADD COLUMN level integer DEFAULT 1;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'xp') THEN
    ALTER TABLE companies ADD COLUMN xp integer DEFAULT 0;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'streak_days') THEN
    ALTER TABLE companies ADD COLUMN streak_days integer DEFAULT 0;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'last_activity') THEN
    ALTER TABLE companies ADD COLUMN last_activity timestamptz DEFAULT now();
  END IF;
END $$;

-- Add AI-enhanced fields to materials table
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'materials' AND column_name = 'ai_tags') THEN
    ALTER TABLE materials ADD COLUMN ai_tags text[];
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'materials' AND column_name = 'estimated_value') THEN
    ALTER TABLE materials ADD COLUMN estimated_value numeric;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'materials' AND column_name = 'priority_score') THEN
    ALTER TABLE materials ADD COLUMN priority_score integer DEFAULT 50;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'materials' AND column_name = 'is_sponsored') THEN
    ALTER TABLE materials ADD COLUMN is_sponsored boolean DEFAULT false;
  END IF;
END $$;

-- Update subscriptions table to reference tiers
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'subscriptions' AND column_name = 'tier_id') THEN
    ALTER TABLE subscriptions ADD COLUMN tier_id uuid REFERENCES subscription_tiers(id);
  END IF;
END $$;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ai_insights_company_id ON ai_insights(company_id);
CREATE INDEX IF NOT EXISTS idx_ai_insights_created_at ON ai_insights(created_at);
CREATE INDEX IF NOT EXISTS idx_material_analytics_material_id ON material_analytics(material_id);
CREATE INDEX IF NOT EXISTS idx_partnership_requests_recipient ON partnership_requests(recipient_id);
CREATE INDEX IF NOT EXISTS idx_partnership_requests_status ON partnership_requests(status);
CREATE INDEX IF NOT EXISTS idx_materials_sponsored ON materials(is_sponsored) WHERE is_sponsored = true;

-- Insert sample achievements for existing users
INSERT INTO user_achievements (company_id, achievement_type, title, description, xp_reward)
SELECT 
  id,
  'welcome',
  'Welcome to SymbioFlow',
  'Successfully joined the industrial symbiosis revolution',
  100
FROM companies
WHERE id NOT IN (SELECT company_id FROM user_achievements WHERE achievement_type = 'welcome')
ON CONFLICT DO NOTHING;

-- Insert sample AI insights for pro users
INSERT INTO ai_insights (company_id, insight_type, title, description, confidence, impact)
SELECT 
  c.id,
  'opportunity',
  'Market Opportunity Detected',
  'High demand for your material type in nearby industrial zones',
  85 + floor(random() * 15),
  'high'
FROM companies c
JOIN subscriptions s ON c.id = s.company_id
WHERE s.tier = 'pro'
AND c.id NOT IN (SELECT company_id FROM ai_insights WHERE insight_type = 'opportunity')
LIMIT 50
ON CONFLICT DO NOTHING;