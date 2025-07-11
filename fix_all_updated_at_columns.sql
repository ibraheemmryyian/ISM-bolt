-- Comprehensive fix for all updated_at column issues
-- Run this in your Supabase SQL editor

-- First, ensure all tables have updated_at columns
DO $$
DECLARE
    table_name text;
    column_exists boolean;
BEGIN
    -- List of tables that should have updated_at
    FOR table_name IN SELECT unnest(ARRAY['companies', 'profiles', 'materials', 'requirements', 'ai_insights', 'matches']) LOOP
        -- Check if updated_at column exists
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = table_name AND column_name = 'updated_at'
        ) INTO column_exists;
        
        -- Add updated_at column if it doesn't exist
        IF NOT column_exists THEN
            EXECUTE format('ALTER TABLE %I ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()', table_name);
            RAISE NOTICE 'Added updated_at column to table: %', table_name;
        ELSE
            RAISE NOTICE 'updated_at column already exists in table: %', table_name;
        END IF;
    END LOOP;
END $$;

-- Drop all existing triggers and functions
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
    -- Only update if the column exists
    IF TG_OP = 'UPDATE' THEN
        NEW.updated_at = NOW();
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for all tables that have updated_at columns
CREATE TRIGGER update_companies_updated_at 
    BEFORE UPDATE ON companies 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_profiles_updated_at 
    BEFORE UPDATE ON profiles 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Only create triggers for tables that actually have updated_at columns
DO $$
DECLARE
    table_name text;
    column_exists boolean;
BEGIN
    FOR table_name IN SELECT unnest(ARRAY['materials', 'requirements', 'ai_insights', 'matches']) LOOP
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = table_name AND column_name = 'updated_at'
        ) INTO column_exists;
        
        IF column_exists THEN
            EXECUTE format('CREATE TRIGGER update_%I_updated_at BEFORE UPDATE ON %I FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()', table_name, table_name);
            RAISE NOTICE 'Created trigger for table: %', table_name;
        END IF;
    END LOOP;
END $$;

-- Verify the fix
SELECT 'All updated_at columns and triggers fixed successfully' as status; 