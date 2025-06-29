-- Admin Setup Script
-- Run this in your Supabase SQL editor to make yourself an admin

-- First, find your user ID (replace 'your-email@example.com' with your actual email)
-- SELECT id, name, email, role FROM companies WHERE email = 'your-email@example.com';

-- Then update your role to admin (replace 'your-user-id' with your actual user ID)
-- UPDATE companies SET role = 'admin' WHERE id = 'your-user-id';

-- Or update by email directly:
-- UPDATE companies SET role = 'admin' WHERE email = 'your-email@example.com';

-- Verify the change:
-- SELECT id, name, email, role FROM companies WHERE role = 'admin'; 