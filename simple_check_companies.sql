-- =====================================================
-- SIMPLE CHECK FOR EXISTING COMPANIES
-- =====================================================

-- Check what tables exist with 'company' in the name
SELECT 
    table_name,
    table_type
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name LIKE '%company%'
ORDER BY table_name;

-- Check if companies table exists
SELECT 
    EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'companies'
    ) as companies_table_exists;

-- If companies table exists, show its structure
SELECT 
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'companies' 
AND table_schema = 'public'
ORDER BY ordinal_position;

-- Count total companies (if table exists)
SELECT 
    COUNT(*) as total_companies
FROM companies;

-- Show all existing companies
SELECT 
    id,
    name,
    industry,
    location,
    employee_count,
    products,
    main_materials,
    production_volume,
    onboarding_completed,
    created_at
FROM companies
ORDER BY name;

-- Show companies by industry
SELECT 
    industry,
    COUNT(*) as company_count,
    STRING_AGG(name, ', ') as companies
FROM companies
GROUP BY industry
ORDER BY company_count DESC;

-- Show companies by location
SELECT 
    location,
    COUNT(*) as company_count,
    STRING_AGG(name, ', ') as companies
FROM companies
GROUP BY location
ORDER BY company_count DESC;

-- Show companies with sustainability goals
SELECT 
    name,
    industry,
    sustainability_goals,
    current_waste_management
FROM companies
WHERE sustainability_goals IS NOT NULL 
AND array_length(sustainability_goals, 1) > 0
ORDER BY name; 