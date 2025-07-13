/*
  # Add read-all policy for company locations
  
  1. Changes
    - Add policy to allow authenticated users to read all company locations
    - This enables the global map feature to work correctly
*/

-- Add policy to allow reading all company locations
CREATE POLICY "Users can read all company locations"
  ON company_profiles
  FOR SELECT
  TO authenticated
  USING (true);