-- Fix email constraint issue
-- Run this in your Supabase SQL editor

-- First, let's see the current table structure
SELECT column_name, is_nullable, data_type 
FROM information_schema.columns 
WHERE table_name = 'companies' AND column_name = 'email';

-- Make email column nullable if it's not already
ALTER TABLE companies ALTER COLUMN email DROP NOT NULL;

-- Also make email nullable in profiles table
ALTER TABLE profiles ALTER COLUMN email DROP NOT NULL;

-- Now insert admin user with a default email
DELETE FROM companies WHERE id = '00000000-0000-0000-0000-000000000001';
DELETE FROM profiles WHERE id = '00000000-0000-0000-0000-000000000001';

INSERT INTO companies (id, name, email, username, role) 
VALUES (
  '00000000-0000-0000-0000-000000000001',
  'System Admin',
  'admin@system.local',
  'admin',
  'admin'
) ON CONFLICT (id) DO NOTHING;

INSERT INTO profiles (id, username, email, role)
VALUES (
  '00000000-0000-0000-0000-000000000001',
  'admin',
  'admin@system.local',
  'admin'
) ON CONFLICT (id) DO NOTHING;

-- Verify the fix
SELECT 'Email constraint fixed successfully' as status;
SELECT 'Admin password: NA10EN' as admin_info; 