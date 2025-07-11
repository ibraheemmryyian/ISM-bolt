-- Fix Business Profile Schema
-- Add missing columns for the business profile form

-- Add process_description column if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'process_description'
    ) THEN
        ALTER TABLE companies ADD COLUMN process_description TEXT;
    END IF;
END $$;

-- Add other potentially missing columns for business profile
DO $$ 
BEGIN
    -- Add waste_quantity if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'waste_quantity'
    ) THEN
        ALTER TABLE companies ADD COLUMN waste_quantity NUMERIC;
    END IF;
    
    -- Add waste_unit if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'waste_unit'
    ) THEN
        ALTER TABLE companies ADD COLUMN waste_unit VARCHAR(50);
    END IF;
    
    -- Add waste_frequency if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'waste_frequency'
    ) THEN
        ALTER TABLE companies ADD COLUMN waste_frequency VARCHAR(50);
    END IF;
    
    -- Add current_waste_management if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'current_waste_management'
    ) THEN
        ALTER TABLE companies ADD COLUMN current_waste_management VARCHAR(100);
    END IF;
    
    -- Add user_type if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'user_type'
    ) THEN
        ALTER TABLE companies ADD COLUMN user_type VARCHAR(50) DEFAULT 'business';
    END IF;
    
    -- Add onboarding_completed if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'onboarding_completed'
    ) THEN
        ALTER TABLE companies ADD COLUMN onboarding_completed BOOLEAN DEFAULT FALSE;
    END IF;
    
    -- Add application_status if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'application_status'
    ) THEN
        ALTER TABLE companies ADD COLUMN application_status VARCHAR(50) DEFAULT 'pending';
    END IF;
END $$;

-- Update RLS policies to include new columns
DROP POLICY IF EXISTS "Users can view their own company" ON companies;
CREATE POLICY "Users can view their own company" ON companies
    FOR SELECT USING (auth.uid()::text = user_id::text);

DROP POLICY IF EXISTS "Users can update their own company" ON companies;
CREATE POLICY "Users can update their own company" ON companies
    FOR UPDATE USING (auth.uid()::text = user_id::text);

DROP POLICY IF EXISTS "Users can insert their own company" ON companies;
CREATE POLICY "Users can insert their own company" ON companies
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

-- Grant necessary permissions
GRANT ALL ON companies TO authenticated;
GRANT ALL ON companies TO service_role;

-- Refresh schema cache
NOTIFY pgrst, 'reload schema';

-- Show the updated table structure
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_name = 'companies' 
ORDER BY ordinal_position; 