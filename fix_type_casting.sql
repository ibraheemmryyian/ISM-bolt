-- Fix Type Casting Issues
-- Run this in your Supabase SQL editor to fix UUID/text comparison errors

-- Drop existing policies that might have type casting issues
DROP POLICY IF EXISTS "Users can view own company" ON companies;
DROP POLICY IF EXISTS "Users can update own company" ON companies;
DROP POLICY IF EXISTS "Users can view own profile" ON profiles;
DROP POLICY IF EXISTS "Users can update own profile" ON profiles;

-- Recreate policies with correct type casting
CREATE POLICY "Users can view own company" ON companies
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own company" ON companies
  FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can view own profile" ON profiles
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON profiles
  FOR UPDATE USING (auth.uid() = id);

-- Fix any other tables that might have similar issues
-- Materials table policies
DROP POLICY IF EXISTS "Users can read own materials" ON materials;
DROP POLICY IF EXISTS "Users can create own materials" ON materials;
DROP POLICY IF EXISTS "Users can update own materials" ON materials;

CREATE POLICY "Users can read own materials" ON materials
  FOR SELECT USING (company_id = auth.uid());

CREATE POLICY "Users can create own materials" ON materials
  FOR INSERT WITH CHECK (company_id = auth.uid());

CREATE POLICY "Users can update own materials" ON materials
  FOR UPDATE USING (company_id = auth.uid());

-- Requirements table policies
DROP POLICY IF EXISTS "Users can read own requirements" ON requirements;
DROP POLICY IF EXISTS "Users can create own requirements" ON requirements;
DROP POLICY IF EXISTS "Users can update own requirements" ON requirements;

CREATE POLICY "Users can read own requirements" ON requirements
  FOR SELECT USING (company_id = auth.uid());

CREATE POLICY "Users can create own requirements" ON requirements
  FOR INSERT WITH CHECK (company_id = auth.uid());

CREATE POLICY "Users can update own requirements" ON requirements
  FOR UPDATE USING (company_id = auth.uid());

-- AI insights table policies
DROP POLICY IF EXISTS "Users can read own insights" ON ai_insights;
DROP POLICY IF EXISTS "Users can create own insights" ON ai_insights;
DROP POLICY IF EXISTS "Users can update own insights" ON ai_insights;

CREATE POLICY "Users can read own insights" ON ai_insights
  FOR SELECT USING (company_id = auth.uid());

CREATE POLICY "Users can create own insights" ON ai_insights
  FOR INSERT WITH CHECK (company_id = auth.uid());

CREATE POLICY "Users can update own insights" ON ai_insights
  FOR UPDATE USING (company_id = auth.uid());

-- Matches table policies
DROP POLICY IF EXISTS "Users can read own matches" ON matches;
DROP POLICY IF EXISTS "Users can create own matches" ON matches;
DROP POLICY IF EXISTS "Users can update own matches" ON matches;

CREATE POLICY "Users can read own matches" ON matches
  FOR SELECT USING (company_id = auth.uid() OR partner_company_id = auth.uid());

CREATE POLICY "Users can create own matches" ON matches
  FOR INSERT WITH CHECK (company_id = auth.uid());

CREATE POLICY "Users can update own matches" ON matches
  FOR UPDATE USING (company_id = auth.uid() OR partner_company_id = auth.uid());

-- Verify the fix
SELECT 'Type casting issues fixed successfully' as status; 