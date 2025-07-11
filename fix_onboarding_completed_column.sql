-- Fix Missing onboarding_completed Column
-- Run this in your Supabase SQL editor

-- Add onboarding_completed column if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'onboarding_completed'
    ) THEN
        ALTER TABLE companies ADD COLUMN onboarding_completed BOOLEAN DEFAULT false;
        RAISE NOTICE 'Added onboarding_completed column to companies table';
    ELSE
        RAISE NOTICE 'onboarding_completed column already exists';
    END IF;
END $$;

-- Verify the column exists
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_name = 'companies' 
AND column_name = 'onboarding_completed'; 