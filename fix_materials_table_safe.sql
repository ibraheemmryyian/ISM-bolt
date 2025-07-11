-- Safe migration to fix materials table - add missing columns for AI generator
-- Run this in your Supabase SQL editor

-- Step 1: Add missing columns to materials table
ALTER TABLE materials 
ADD COLUMN IF NOT EXISTS material_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS unit VARCHAR(50),
ADD COLUMN IF NOT EXISTS type VARCHAR(50),
ADD COLUMN IF NOT EXISTS current_cost VARCHAR(100),
ADD COLUMN IF NOT EXISTS potential_sources TEXT[],
ADD COLUMN IF NOT EXISTS price_per_unit DECIMAL(10,2);

-- Step 2: Set default values for material_name where it's null
UPDATE materials 
SET material_name = 'AI Generated Material' 
WHERE material_name IS NULL;

-- Step 3: Add indexes for new columns
CREATE INDEX IF NOT EXISTS idx_materials_type ON materials(type);
CREATE INDEX IF NOT EXISTS idx_materials_material_name ON materials(material_name);

-- Step 4: Verify the table structure
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'materials' 
ORDER BY ordinal_position; 