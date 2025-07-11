-- Company Applications Table for Access Requests
CREATE TABLE IF NOT EXISTS company_applications (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  company_name text NOT NULL,
  website text,
  industry text,
  contact_name text NOT NULL,
  contact_email text NOT NULL,
  application_answers jsonb,
  status text NOT NULL DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
  reviewed_by uuid,
  reviewed_at timestamptz,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Index for fast lookup
CREATE INDEX IF NOT EXISTS idx_company_applications_status ON company_applications(status);

-- Enable RLS
ALTER TABLE company_applications ENABLE ROW LEVEL SECURITY;

-- Allow admin to read all applications
CREATE POLICY "Admin can read applications"
  ON company_applications
  FOR SELECT
  USING (EXISTS (SELECT 1 FROM companies WHERE companies.id = auth.uid() AND companies.role = 'admin'));

-- Allow anyone to insert (request access)
CREATE POLICY "Anyone can request access"
  ON company_applications
  FOR INSERT
  WITH CHECK (true); 