-- Migration to add user_type column to companies table
-- Run this in your Supabase SQL editor

-- Add user_type column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' AND column_name = 'user_type'
    ) THEN
        ALTER TABLE companies ADD COLUMN user_type VARCHAR(20) CHECK (user_type IN ('business', 'researcher'));
    END IF;
END $$;

-- Update existing records to have a default user_type
UPDATE companies 
SET user_type = 'business' 
WHERE user_type IS NULL;

-- Make user_type NOT NULL after setting defaults
ALTER TABLE companies ALTER COLUMN user_type SET NOT NULL; 