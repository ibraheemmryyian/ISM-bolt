-- Add username column to companies table
ALTER TABLE companies ADD COLUMN IF NOT EXISTS username TEXT UNIQUE;

-- Create index for faster username lookups
CREATE INDEX IF NOT EXISTS idx_companies_username ON companies(username);

-- Update RLS policies to allow username-based access
DROP POLICY IF EXISTS "Users can read own company data" ON companies;
CREATE POLICY "Users can read own company data"
  ON companies
  FOR SELECT
  TO authenticated
  USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update own company data" ON companies;
CREATE POLICY "Users can update own company data"
  ON companies
  FOR UPDATE
  TO authenticated
  USING (auth.uid() = id);

-- Function to get company by username
CREATE OR REPLACE FUNCTION get_company_by_username(username_param TEXT)
RETURNS TABLE (
    id UUID,
    username TEXT,
    email TEXT,
    name TEXT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT c.id, c.username, c.email, c.name, c.created_at
    FROM companies c
    WHERE c.username = username_param;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER; 