-- Fix Onboarding Issues
-- Run this in your Supabase SQL editor to resolve the database schema and form issues

-- 1. Fix Database Schema Cache Issue
DO $$ 
BEGIN
    -- Check if current_waste_management column exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'current_waste_management'
    ) THEN
        -- Add the column if it doesn't exist
        ALTER TABLE companies ADD COLUMN current_waste_management VARCHAR(100);
        RAISE NOTICE 'Added current_waste_management column to companies table';
    ELSE
        RAISE NOTICE 'current_waste_management column already exists';
    END IF;
END $$;

-- 2. Add missing columns for proper form handling
DO $$ 
BEGIN
    -- Add waste_quantity column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'waste_quantity'
    ) THEN
        ALTER TABLE companies ADD COLUMN waste_quantity VARCHAR(100);
        RAISE NOTICE 'Added waste_quantity column to companies table';
    END IF;
    
    -- Add waste_unit column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'waste_unit'
    ) THEN
        ALTER TABLE companies ADD COLUMN waste_unit VARCHAR(50);
        RAISE NOTICE 'Added waste_unit column to companies table';
    END IF;
    
    -- Add waste_frequency column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'waste_frequency'
    ) THEN
        ALTER TABLE companies ADD COLUMN waste_frequency VARCHAR(50);
        RAISE NOTICE 'Added waste_frequency column to companies table';
    END IF;
    
    -- Add user_type column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'user_type'
    ) THEN
        ALTER TABLE companies ADD COLUMN user_type VARCHAR(20) DEFAULT 'business';
        RAISE NOTICE 'Added user_type column to companies table';
    END IF;
END $$;

-- 3. Refresh schema cache by running a query
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'companies' 
AND column_name IN ('current_waste_management', 'waste_quantity', 'waste_unit', 'waste_frequency', 'user_type')
ORDER BY column_name;

-- 4. Update RLS policies to include new columns
DO $$
BEGIN
    -- Drop existing policies if they exist
    DROP POLICY IF EXISTS "Users can read own company data" ON companies;
    DROP POLICY IF EXISTS "Users can update own company data" ON companies;
    DROP POLICY IF EXISTS "Users can insert own company data" ON companies;
    
    -- Create new policies
    CREATE POLICY "Users can read own company data"
        ON companies
        FOR SELECT
        TO authenticated
        USING (auth.uid() = id);
    
    CREATE POLICY "Users can update own company data"
        ON companies
        FOR UPDATE
        TO authenticated
        USING (auth.uid() = id);
    
    CREATE POLICY "Users can insert own company data"
        ON companies
        FOR INSERT
        TO authenticated
        WITH CHECK (auth.uid() = id);
    
    RAISE NOTICE 'Updated RLS policies for companies table';
END $$;

-- 5. Verify the fix worked
SELECT 
    'Schema Check Complete' as status,
    COUNT(*) as total_columns,
    COUNT(CASE WHEN column_name IN ('current_waste_management', 'waste_quantity', 'waste_unit', 'waste_frequency', 'user_type') THEN 1 END) as required_columns_found
FROM information_schema.columns 
WHERE table_name = 'companies'; 