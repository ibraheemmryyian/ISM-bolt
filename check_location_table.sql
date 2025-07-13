-- =====================================================
-- CHECK EXISTING LOCATION TABLE AND COMPANY DATA
-- =====================================================

-- First, let's see what tables exist
SELECT 
    table_name,
    table_type
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;

-- Check the structure of the location table
SELECT 
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'location' 
AND table_schema = 'public'
ORDER BY ordinal_position;

-- Show all data in the location table
SELECT * FROM location;

-- Count total records in location table
SELECT COUNT(*) as total_records FROM location;

-- Show unique locations
SELECT DISTINCT location FROM location ORDER BY location;

-- Show company count by location
SELECT 
    location,
    company_count,
    companies
FROM location
ORDER BY company_count DESC;

-- Check if there are any other tables with company data
SELECT 
    table_name,
    table_type
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND (
    table_name LIKE '%company%' 
    OR table_name LIKE '%business%' 
    OR table_name LIKE '%enterprise%'
    OR table_name LIKE '%firm%'
    OR table_name LIKE '%corp%'
)
ORDER BY table_name;

-- Check all tables to see what we have
SELECT 
    table_name,
    table_type
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name NOT LIKE 'pg_%'
AND table_name NOT LIKE 'sql_%'
ORDER BY table_name; 