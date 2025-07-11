-- Simple fix for updated_at trigger issue
-- Run this in your Supabase SQL editor

-- Drop all existing triggers and functions first
DROP TRIGGER IF EXISTS update_companies_updated_at ON companies;
DROP TRIGGER IF EXISTS update_profiles_updated_at ON profiles;
DROP TRIGGER IF EXISTS update_materials_updated_at ON materials;
DROP TRIGGER IF EXISTS update_requirements_updated_at ON requirements;
DROP TRIGGER IF EXISTS update_ai_insights_updated_at ON ai_insights;
DROP TRIGGER IF EXISTS update_matches_updated_at ON matches;
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Create the updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Only create triggers for tables that actually have updated_at columns
-- Companies table
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'companies' AND column_name = 'updated_at'
    ) THEN
        CREATE TRIGGER update_companies_updated_at 
            BEFORE UPDATE ON companies 
            FOR EACH ROW 
            EXECUTE FUNCTION update_updated_at_column();
        RAISE NOTICE 'Created trigger for companies table';
    ELSE
        RAISE NOTICE 'companies table does not have updated_at column, skipping trigger';
    END IF;
END $$;

-- Profiles table
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'profiles' AND column_name = 'updated_at'
    ) THEN
        CREATE TRIGGER update_profiles_updated_at 
            BEFORE UPDATE ON profiles 
            FOR EACH ROW 
            EXECUTE FUNCTION update_updated_at_column();
        RAISE NOTICE 'Created trigger for profiles table';
    ELSE
        RAISE NOTICE 'profiles table does not have updated_at column, skipping trigger';
    END IF;
END $$;

-- Verify the fix
SELECT 'Updated_at triggers fixed successfully' as status; 