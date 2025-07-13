/*
  # Create subscription and AI system tables

  1. New Tables
    - `subscriptions`
      - `id` (uuid, primary key)
      - `company_id` (uuid, foreign key)
      - `tier` (text) - 'free', 'pro', 'enterprise'
      - `status` (text) - 'active', 'cancelled', 'expired'
      - `created_at` (timestamp)
      - `expires_at` (timestamp)
    
    - `ai_recommendations`
      - `id` (uuid, primary key)
      - `company_id` (uuid, foreign key)
      - `type` (text) - 'connection', 'material', 'opportunity'
      - `title` (text)
      - `description` (text)
      - `confidence` (numeric)
      - `action_url` (text)
      - `status` (text) - 'pending', 'viewed', 'acted'
      - `created_at` (timestamp)
    
    - `connections`
      - `id` (uuid, primary key)
      - `requester_id` (uuid, foreign key)
      - `recipient_id` (uuid, foreign key)
      - `status` (text) - 'pending', 'accepted', 'declined'
      - `message` (text)
      - `created_at` (timestamp)
    
    - `material_matches`
      - `id` (uuid, primary key)
      - `material_id` (uuid, foreign key)
      - `matched_material_id` (uuid, foreign key)
      - `match_score` (numeric)
      - `status` (text) - 'pending', 'contacted', 'completed'
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on all tables
    - Add appropriate policies for each table
*/

-- Subscriptions table
CREATE TABLE IF NOT EXISTS subscriptions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  company_id uuid REFERENCES companies(id),
  tier text NOT NULL DEFAULT 'free' CHECK (tier IN ('free', 'pro', 'enterprise')),
  status text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'cancelled', 'expired')),
  created_at timestamptz DEFAULT now(),
  expires_at timestamptz
);

ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own subscription"
  ON subscriptions
  FOR SELECT
  TO authenticated
  USING (company_id = auth.uid());

-- AI Recommendations table
CREATE TABLE IF NOT EXISTS ai_recommendations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  company_id uuid REFERENCES companies(id),
  type text NOT NULL CHECK (type IN ('connection', 'material', 'opportunity')),
  title text NOT NULL,
  description text NOT NULL,
  confidence numeric NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
  action_url text,
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'viewed', 'acted')),
  created_at timestamptz DEFAULT now()
);

ALTER TABLE ai_recommendations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own recommendations"
  ON ai_recommendations
  FOR SELECT
  TO authenticated
  USING (company_id = auth.uid());

CREATE POLICY "Users can update own recommendations"
  ON ai_recommendations
  FOR UPDATE
  TO authenticated
  USING (company_id = auth.uid());

-- Connections table
CREATE TABLE IF NOT EXISTS connections (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  requester_id uuid REFERENCES companies(id),
  recipient_id uuid REFERENCES companies(id),
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'declined')),
  message text,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE connections ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read their connections"
  ON connections
  FOR SELECT
  TO authenticated
  USING (requester_id = auth.uid() OR recipient_id = auth.uid());

CREATE POLICY "Users can create connections"
  ON connections
  FOR INSERT
  TO authenticated
  WITH CHECK (requester_id = auth.uid());

CREATE POLICY "Users can update received connections"
  ON connections
  FOR UPDATE
  TO authenticated
  USING (recipient_id = auth.uid());

-- Material matches table
CREATE TABLE IF NOT EXISTS material_matches (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  material_id uuid REFERENCES materials(id),
  matched_material_id uuid REFERENCES materials(id),
  match_score numeric NOT NULL CHECK (match_score >= 0 AND match_score <= 100),
  status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'contacted', 'completed')),
  created_at timestamptz DEFAULT now()
);

ALTER TABLE material_matches ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read matches for their materials"
  ON material_matches
  FOR SELECT
  TO authenticated
  USING (
    EXISTS (
      SELECT 1 FROM materials 
      WHERE id = material_matches.material_id 
      AND company_id = auth.uid()
    ) OR
    EXISTS (
      SELECT 1 FROM materials 
      WHERE id = material_matches.matched_material_id 
      AND company_id = auth.uid()
    )
  );

-- Insert default subscriptions for existing companies
INSERT INTO subscriptions (company_id, tier, status)
SELECT id, 'pro', 'active'
FROM companies
WHERE id NOT IN (SELECT company_id FROM subscriptions WHERE company_id IS NOT NULL)
ON CONFLICT DO NOTHING;

-- Insert sample AI recommendations
INSERT INTO ai_recommendations (company_id, type, title, description, confidence)
SELECT 
  c.id,
  (ARRAY['connection', 'material', 'opportunity'])[floor(random() * 3 + 1)],
  'Sample Recommendation',
  'This is a sample AI-generated recommendation for testing purposes.',
  floor(random() * 30 + 70)
FROM companies c
LIMIT 10
ON CONFLICT DO NOTHING;