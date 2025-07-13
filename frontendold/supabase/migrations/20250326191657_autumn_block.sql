/*
  # Create materials and companies tables

  1. New Tables
    - `companies`
      - `id` (uuid, primary key)
      - `name` (text)
      - `email` (text)
      - `created_at` (timestamp)
    
    - `materials`
      - `id` (uuid, primary key)
      - `company_id` (uuid, foreign key)
      - `material_name` (text)
      - `quantity` (numeric)
      - `unit` (text)
      - `description` (text)
      - `type` (text) - either 'waste' or 'requirement'
      - `created_at` (timestamp)

  2. Security
    - Enable RLS on both tables
    - Add policies for authenticated users to:
      - Read their own company data
      - Read all materials
      - Create/update their own materials
*/

-- Companies table
CREATE TABLE companies (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  email text NOT NULL UNIQUE,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE companies ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own company data"
  ON companies
  FOR SELECT
  TO authenticated
  USING (auth.uid() = id);

CREATE POLICY "Users can update own company data"
  ON companies
  FOR UPDATE
  TO authenticated
  USING (auth.uid() = id);

-- Materials table
CREATE TABLE materials (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  company_id uuid REFERENCES companies(id),
  material_name text NOT NULL,
  quantity numeric NOT NULL,
  unit text NOT NULL,
  description text,
  type text NOT NULL CHECK (type IN ('waste', 'requirement')),
  created_at timestamptz DEFAULT now()
);

ALTER TABLE materials ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read all materials"
  ON materials
  FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Users can create materials for their company"
  ON materials
  FOR INSERT
  TO authenticated
  WITH CHECK (company_id = auth.uid());

CREATE POLICY "Users can update their own materials"
  ON materials
  FOR UPDATE
  TO authenticated
  USING (company_id = auth.uid());