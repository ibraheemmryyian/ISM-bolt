-- Fix materials table - add missing columns for AI generator
-- Run this in your Supabase SQL editor

-- Add missing columns to materials table
ALTER TABLE materials 
ADD COLUMN IF NOT EXISTS material_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS unit VARCHAR(50),
ADD COLUMN IF NOT EXISTS type VARCHAR(50),
ADD COLUMN IF NOT EXISTS current_cost VARCHAR(100),
ADD COLUMN IF NOT EXISTS potential_sources TEXT[],
ADD COLUMN IF NOT EXISTS price_per_unit DECIMAL(10,2);

-- Check what columns exist in the materials table
-- This will help us understand the current structure
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'materials' 
ORDER BY ordinal_position;

-- Update existing records to have material_name = name if material_name is null
-- First, let's check if 'name' column exists
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'materials' AND column_name = 'name'
    ) THEN
        UPDATE materials 
        SET material_name = name 
        WHERE material_name IS NULL AND name IS NOT NULL;
    ELSIF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'materials' AND column_name = 'material_name'
    ) THEN
        -- material_name column already exists, no need to update
        RAISE NOTICE 'material_name column already exists';
    ELSE
        -- Neither name nor material_name exists, set a default value
        UPDATE materials 
        SET material_name = 'Unknown Material' 
        WHERE material_name IS NULL;
    END IF;
END $$;

-- Add indexes for new columns
CREATE INDEX IF NOT EXISTS idx_materials_type ON materials(type);
CREATE INDEX IF NOT EXISTS idx_materials_material_name ON materials(material_name);

-- Verify the table structure
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'materials' 
ORDER BY ordinal_position; 