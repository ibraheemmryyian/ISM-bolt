-- Create activities table for tracking real user activities
-- This replaces the hardcoded recent activity feeds

CREATE TABLE IF NOT EXISTS user_activities (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  company_id uuid REFERENCES companies(id) ON DELETE CASCADE,
  activity_type text NOT NULL CHECK (activity_type IN (
    'onboarding_completed',
    'material_listed',
    'material_updated',
    'match_found',
    'connection_requested',
    'connection_accepted',
    'transaction_completed',
    'ai_recommendation_generated',
    'profile_updated',
    'subscription_upgraded',
    'waste_audit_completed',
    'sustainability_goal_set'
  )),
  title text NOT NULL,
  description text,
  impact_level text CHECK (impact_level IN ('high', 'medium', 'low')),
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now()
);

-- Enable RLS
ALTER TABLE user_activities ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view their own activities"
  ON user_activities
  FOR SELECT
  TO authenticated
  USING (company_id = auth.uid());

CREATE POLICY "Users can insert their own activities"
  ON user_activities
  FOR INSERT
  TO authenticated
  WITH CHECK (company_id = auth.uid());

-- Index for performance
CREATE INDEX idx_user_activities_company_id_created_at 
  ON user_activities(company_id, created_at DESC);

-- Function to log activities
CREATE OR REPLACE FUNCTION log_user_activity(
  p_company_id uuid,
  p_activity_type text,
  p_title text,
  p_description text DEFAULT NULL,
  p_impact_level text DEFAULT 'medium',
  p_metadata jsonb DEFAULT '{}'
) RETURNS uuid AS $$
DECLARE
  activity_id uuid;
BEGIN
  INSERT INTO user_activities (
    company_id,
    activity_type,
    title,
    description,
    impact_level,
    metadata
  ) VALUES (
    p_company_id,
    p_activity_type,
    p_title,
    p_description,
    p_impact_level,
    p_metadata
  ) RETURNING id INTO activity_id;
  
  RETURN activity_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION log_user_activity TO authenticated;

-- Insert some sample activities for existing companies (optional)
INSERT INTO user_activities (company_id, activity_type, title, description, impact_level)
SELECT 
  c.id,
  'onboarding_completed',
  'Completed AI Onboarding',
  'Successfully completed the AI-powered onboarding process',
  'high'
FROM companies c
WHERE c.id IN (
  SELECT DISTINCT company_id 
  FROM subscriptions 
  WHERE tier IN ('pro', 'enterprise')
)
ON CONFLICT DO NOTHING; 