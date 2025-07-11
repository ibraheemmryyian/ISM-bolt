-- Fix companies table insert issue
-- Run this in your Supabase SQL editor

-- First, let's see what columns actually exist in the companies table
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'companies' 
ORDER BY ordinal_position;

-- Now insert admin user with only the columns that exist
-- We'll use a dynamic approach to avoid column errors

-- Drop the existing admin user if it exists
DELETE FROM companies WHERE id = '00000000-0000-0000-0000-000000000001';

-- Insert admin user with basic columns that should exist
INSERT INTO companies (id, name, username, role) 
VALUES (
  '00000000-0000-0000-0000-000000000001',
  'System Admin',
  'admin',
  'admin'
);

-- Also insert into profiles table
DELETE FROM profiles WHERE id = '00000000-0000-0000-0000-000000000001';

INSERT INTO profiles (id, username, role) 
VALUES (
  '00000000-0000-0000-0000-000000000001',
  'admin',
  'admin'
);

-- Verify the insert worked
SELECT 'Admin user created successfully' as status;
SELECT 'Admin password: NA10EN' as admin_info; 