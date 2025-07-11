-- Fix Database Schema Cache Issue
-- Run this in your Supabase SQL editor to ensure all columns exist

-- First, let's check if the column exists and add it if it doesn't
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

-- Refresh the schema cache by running a simple query
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'companies' 
AND column_name = 'current_waste_management';

-- Also ensure other important columns exist
DO $$ 
BEGIN
    -- Add any missing columns that might be needed
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'waste_quantity'
    ) THEN
        ALTER TABLE companies ADD COLUMN waste_quantity VARCHAR(100);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'waste_frequency'
    ) THEN
        ALTER TABLE companies ADD COLUMN waste_frequency VARCHAR(50);
    END IF;
    
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' 
        AND column_name = 'user_type'
    ) THEN
        ALTER TABLE companies ADD COLUMN user_type VARCHAR(20) DEFAULT 'business';
    END IF;
END $$;

-- Verify all columns exist
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'companies' 
ORDER BY ordinal_position; 