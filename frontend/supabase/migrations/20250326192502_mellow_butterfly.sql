/*
  # Add role column to companies table

  1. Changes
    - Add role column to companies table with type 'user' or 'admin'
    - Set default value to 'user'
    - Add check constraint to ensure valid roles
  
  2. Security
    - Only admins can update roles
    - Users can read their own role
*/

ALTER TABLE companies 
ADD COLUMN IF NOT EXISTS role text NOT NULL DEFAULT 'user'
CHECK (role IN ('user', 'admin'));

-- Policy to allow users to read their own role
CREATE POLICY "Users can read their own role"
ON companies
FOR SELECT
TO authenticated
USING (auth.uid() = id);

-- Policy to allow admins to update roles
CREATE POLICY "Admins can update roles"
ON companies
FOR UPDATE
TO authenticated
USING (
  EXISTS (
    SELECT 1 FROM companies 
    WHERE id = auth.uid() 
    AND role = 'admin'
  )
)
WITH CHECK (
  EXISTS (
    SELECT 1 FROM companies 
    WHERE id = auth.uid() 
    AND role = 'admin'
  )
);