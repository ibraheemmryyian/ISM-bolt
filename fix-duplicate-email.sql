-- Fix for duplicate email constraint violation
-- This script provides multiple options to handle the "duplicate key value violates unique constraint companies_email_key" error

-- Option 1: Check if the email already exists and show the existing record
-- Replace 'menaxox544@boxmach.com' with the actual email causing the issue
SELECT id, name, email, role, created_at 
FROM companies 
WHERE email = 'menaxox544@boxmach.com';

-- Option 2: If you want to update the existing company record instead of creating a new one
-- Replace 'menaxox544@boxmach.com' with the actual email and 'NovaChem Solutions' with the new company name
UPDATE companies 
SET name = 'NovaChem Solutions'
WHERE email = 'menaxox544@boxmach.com';

-- Option 3: If you want to delete the existing record and allow a new one to be created
-- WARNING: This will delete all related data (materials, messages, etc.)
-- Replace 'menaxox544@boxmach.com' with the actual email
-- DELETE FROM companies WHERE email = 'menaxox544@boxmach.com';

-- Option 4: If you want to temporarily disable the unique constraint (NOT RECOMMENDED for production)
-- ALTER TABLE companies DROP CONSTRAINT companies_email_key;
-- After fixing the duplicates, re-add the constraint:
-- ALTER TABLE companies ADD CONSTRAINT companies_email_key UNIQUE (email);

-- Option 5: Find all duplicate emails in the database
SELECT email, COUNT(*) as count
FROM companies 
GROUP BY email 
HAVING COUNT(*) > 1
ORDER BY count DESC;

-- Option 6: Clean up duplicate emails by keeping only the most recent record
-- This creates a temporary table with the records to keep
WITH duplicates_to_keep AS (
  SELECT DISTINCT ON (email) id, email, name, role, created_at
  FROM companies
  ORDER BY email, created_at DESC
)
DELETE FROM companies 
WHERE id NOT IN (SELECT id FROM duplicates_to_keep);

-- Option 7: If the user wants to use a different email, update the auth.users table as well
-- First, find the auth user ID:
-- SELECT id, email FROM auth.users WHERE email = 'menaxox544@boxmach.com';

-- Then update both tables (replace with actual values):
-- UPDATE auth.users SET email = 'new-email@example.com' WHERE email = 'menaxox544@boxmach.com';
-- UPDATE companies SET email = 'new-email@example.com' WHERE email = 'menaxox544@boxmach.com';

-- Option 8: Create a function to handle email conflicts gracefully
CREATE OR REPLACE FUNCTION handle_company_signup(
  p_email TEXT,
  p_name TEXT,
  p_user_id UUID
) RETURNS JSON AS $$
DECLARE
  existing_company RECORD;
  result JSON;
BEGIN
  -- Check if company already exists
  SELECT * INTO existing_company 
  FROM companies 
  WHERE email = p_email;
  
  IF existing_company IS NOT NULL THEN
    -- Company exists, return error
    result := json_build_object(
      'success', false,
      'error', 'Email already registered',
      'existing_company', json_build_object(
        'id', existing_company.id,
        'name', existing_company.name,
        'email', existing_company.email,
        'created_at', existing_company.created_at
      )
    );
  ELSE
    -- Company doesn't exist, create new one
    INSERT INTO companies (id, name, email, role)
    VALUES (p_user_id, p_name, p_email, 'user');
    
    result := json_build_object(
      'success', true,
      'message', 'Company created successfully'
    );
  END IF;
  
  RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Usage example:
-- SELECT handle_company_signup('menaxox544@boxmach.com', 'NovaChem Solutions', 'user-uuid-here'); 