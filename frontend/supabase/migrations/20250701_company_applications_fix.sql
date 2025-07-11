-- Fix company_applications table to make contact_name optional
-- Since we simplified the request access form

-- Make contact_name optional
ALTER TABLE company_applications ALTER COLUMN contact_name DROP NOT NULL;

-- Also make website and industry optional since we removed them from the form
ALTER TABLE company_applications ALTER COLUMN website DROP NOT NULL;
ALTER TABLE company_applications ALTER COLUMN industry DROP NOT NULL; 