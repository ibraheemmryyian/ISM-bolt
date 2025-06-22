/*
  # Add company profiles table

  1. New Tables
    - `company_profiles`
      - `id` (uuid, primary key)
      - `company_id` (uuid, foreign key to companies)
      - `role` (text)
      - `location` (text)
      - `interests` (text)
      - `organization_type` (text)
      - `materials_of_interest` (text)
      - `sustainability_goals` (text)
      - `created_at` (timestamp)

  2. Security
    - Enable RLS
    - Add policies for authenticated users to:
      - Read their own profile
      - Create/update their own profile
*/

CREATE TABLE company_profiles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  company_id uuid REFERENCES companies(id),
  role text NOT NULL,
  location text NOT NULL,
  interests text,
  organization_type text NOT NULL,
  materials_of_interest text,
  sustainability_goals text,
  created_at timestamptz DEFAULT now(),
  CONSTRAINT valid_role CHECK (
    role IN ('researcher', 'buyer', 'seller', 'investor', 'consultant')
  ),
  CONSTRAINT valid_org_type CHECK (
    organization_type IN (
      'manufacturing',
      'research',
      'waste_management',
      'recycling',
      'consulting',
      'investment'
    )
  )
);

ALTER TABLE company_profiles ENABLE ROW LEVEL SECURITY;

-- Allow users to read their own profile
CREATE POLICY "Users can read own profile"
  ON company_profiles
  FOR SELECT
  TO authenticated
  USING (company_id = auth.uid());

-- Allow users to create their own profile
CREATE POLICY "Users can create own profile"
  ON company_profiles
  FOR INSERT
  TO authenticated
  WITH CHECK (company_id = auth.uid());

-- Allow users to update their own profile
CREATE POLICY "Users can update own profile"
  ON company_profiles
  FOR UPDATE
  TO authenticated
  USING (company_id = auth.uid());