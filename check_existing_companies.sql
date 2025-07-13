-- =====================================================
-- CHECK EXISTING COMPANIES IN DATABASE
-- =====================================================

-- First, let's see what tables exist
SELECT 
    table_name,
    table_type
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name LIKE '%company%'
ORDER BY table_name;

-- Check if companies table exists and what's in it
DO $$
BEGIN
    -- Check if companies table exists
    IF EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'companies'
    ) THEN
        RAISE NOTICE '✅ Companies table exists!';
        
        -- Show table structure
        RAISE NOTICE '📋 Companies table structure:';
        FOR r IN (
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = 'companies' 
            AND table_schema = 'public'
            ORDER BY ordinal_position
        ) LOOP
            RAISE NOTICE '   %: % (%)', r.column_name, r.data_type, r.is_nullable;
        END LOOP;
        
        -- Count companies
        EXECUTE 'SELECT COUNT(*) FROM companies' INTO r;
        RAISE NOTICE '📊 Total companies in database: %', r;
        
    ELSE
        RAISE NOTICE '❌ Companies table does not exist';
    END IF;
END $$;

-- If companies table exists, show all companies
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