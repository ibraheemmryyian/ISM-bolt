-- =====================================================
-- ISM AI PLATFORM - REAL DATA PREPARATION SCRIPT (FINAL)
-- Prepares database for 50 real company profiles
-- Handles existing objects gracefully
-- =====================================================

-- Drop existing views/tables if they exist
DROP VIEW IF EXISTS high_value_targets CASCADE;
DROP VIEW IF EXISTS symbiosis_opportunities CASCADE;
DROP TABLE IF EXISTS bulk_imports CASCADE;

-- Create bulk imports table for tracking real company data imports
-- This table stores the results of bulk imports of real company profiles

CREATE TABLE IF NOT EXISTS bulk_imports (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    import_id VARCHAR(255) NOT NULL UNIQUE,
    summary JSONB NOT NULL,
    results JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID REFERENCES auth.users(id),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create index for fast lookups
CREATE INDEX IF NOT EXISTS idx_bulk_imports_import_id ON bulk_imports(import_id);
CREATE INDEX IF NOT EXISTS idx_bulk_imports_created_at ON bulk_imports(created_at);
CREATE INDEX IF NOT EXISTS idx_bulk_imports_status ON bulk_imports(status);

-- Add RLS policies
ALTER TABLE bulk_imports ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Users can view their own bulk imports" ON bulk_imports;
DROP POLICY IF EXISTS "Users can create bulk imports" ON bulk_imports;
DROP POLICY IF EXISTS "Users can update their own bulk imports" ON bulk_imports;

-- Allow authenticated users to read their own imports
CREATE POLICY "Users can view their own bulk imports" ON bulk_imports
    FOR SELECT USING (auth.uid() = created_by);

-- Allow authenticated users to create imports
CREATE POLICY "Users can create bulk imports" ON bulk_imports
    FOR INSERT WITH CHECK (auth.uid() = created_by);

-- Allow authenticated users to update their own imports
CREATE POLICY "Users can update their own bulk imports" ON bulk_imports
    FOR UPDATE USING (auth.uid() = created_by);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_bulk_imports_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if it exists
DROP TRIGGER IF EXISTS update_bulk_imports_updated_at ON bulk_imports;

-- Create trigger for updated_at
CREATE TRIGGER update_bulk_imports_updated_at
    BEFORE UPDATE ON bulk_imports
    FOR EACH ROW
    EXECUTE FUNCTION update_bulk_imports_updated_at();

-- Add columns to companies table for real data processing
-- Check if columns exist before adding them
DO $$
BEGIN
    -- Add data_quality_score column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'data_quality_score') THEN
        ALTER TABLE companies ADD COLUMN data_quality_score INTEGER DEFAULT 0;
    END IF;
    
    -- Add processing_status column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'processing_status') THEN
        ALTER TABLE companies ADD COLUMN processing_status VARCHAR(50) DEFAULT 'pending';
    END IF;
    
    -- Add enriched_data column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'enriched_data') THEN
        ALTER TABLE companies ADD COLUMN enriched_data JSONB;
    END IF;
    
    -- Add business_metrics column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'business_metrics') THEN
        ALTER TABLE companies ADD COLUMN business_metrics JSONB;
    END IF;
    
    -- Add ai_insights column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'ai_insights') THEN
        ALTER TABLE companies ADD COLUMN ai_insights JSONB;
    END IF;
    
    -- Add recommendations column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'recommendations') THEN
        ALTER TABLE companies ADD COLUMN recommendations JSONB;
    END IF;
    
    -- Add last_processed_at column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'last_processed_at') THEN
        ALTER TABLE companies ADD COLUMN last_processed_at TIMESTAMP WITH TIME ZONE;
    END IF;
    
    -- Add processing_errors column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'processing_errors') THEN
        ALTER TABLE companies ADD COLUMN processing_errors JSONB;
    END IF;
    
    -- Add contact_info column if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'contact_info') THEN
        ALTER TABLE companies ADD COLUMN contact_info JSONB;
    END IF;
    
    -- Add email column if it doesn't exist (for backward compatibility)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'email') THEN
        ALTER TABLE companies ADD COLUMN email VARCHAR(255);
    END IF;
    
    -- Add phone column if it doesn't exist (for backward compatibility)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'phone') THEN
        ALTER TABLE companies ADD COLUMN phone VARCHAR(50);
    END IF;
    
    -- Add website column if it doesn't exist (for backward compatibility)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'website') THEN
        ALTER TABLE companies ADD COLUMN website VARCHAR(255);
    END IF;
END $$;

-- Drop existing indexes if they exist
DROP INDEX IF EXISTS idx_companies_data_quality_score;
DROP INDEX IF EXISTS idx_companies_processing_status;
DROP INDEX IF EXISTS idx_companies_business_metrics;
DROP INDEX IF EXISTS idx_companies_ai_insights;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_companies_data_quality_score ON companies(data_quality_score);
CREATE INDEX IF NOT EXISTS idx_companies_processing_status ON companies(processing_status);
CREATE INDEX IF NOT EXISTS idx_companies_business_metrics ON companies USING GIN(business_metrics);
CREATE INDEX IF NOT EXISTS idx_companies_ai_insights ON companies USING GIN(ai_insights);

-- Create function to calculate business metrics
CREATE OR REPLACE FUNCTION calculate_company_business_metrics(company_id UUID)
RETURNS JSONB AS $$
DECLARE
    result JSONB;
    waste_value NUMERIC := 0;
    potential_savings NUMERIC := 0;
    carbon_reduction NUMERIC := 0;
    symbiosis_score NUMERIC := 0;
BEGIN
    -- Calculate waste value from materials
    SELECT COALESCE(SUM(
        CASE 
            WHEN m.quantity IS NOT NULL AND m.unit_price IS NOT NULL 
            THEN m.quantity * m.unit_price
            ELSE 0
        END
    ), 0) INTO waste_value
    FROM materials m
    WHERE m.company_id = company_id AND m.type = 'waste';

    -- Calculate potential savings (placeholder logic)
    potential_savings := waste_value * 0.3; -- 30% potential savings
    
    -- Calculate carbon reduction (placeholder logic)
    carbon_reduction := waste_value * 0.1; -- 10% of waste value as carbon reduction
    
    -- Calculate symbiosis score (placeholder logic)
    symbiosis_score := LEAST(waste_value / 10000, 1.0); -- Normalized score
    
    result := jsonb_build_object(
        'total_waste_value', waste_value,
        'potential_savings', potential_savings,
        'carbon_reduction_potential', carbon_reduction,
        'symbiosis_score', symbiosis_score,
        'calculated_at', NOW()
    );
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Create function to update company processing status
CREATE OR REPLACE FUNCTION update_company_processing_status(
    p_company_id UUID,
    p_status VARCHAR(50),
    p_data_quality_score INTEGER DEFAULT NULL,
    p_enriched_data JSONB DEFAULT NULL,
    p_ai_insights JSONB DEFAULT NULL,
    p_recommendations JSONB DEFAULT NULL,
    p_errors JSONB DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    UPDATE companies 
    SET 
        processing_status = p_status,
        data_quality_score = COALESCE(p_data_quality_score, data_quality_score),
        enriched_data = COALESCE(p_enriched_data, enriched_data),
        ai_insights = COALESCE(p_ai_insights, ai_insights),
        recommendations = COALESCE(p_recommendations, recommendations),
        processing_errors = COALESCE(p_errors, processing_errors),
        last_processed_at = CASE WHEN p_status = 'completed' THEN NOW() ELSE last_processed_at END,
        updated_at = NOW()
    WHERE id = p_company_id;
END;
$$ LANGUAGE plpgsql;

-- Create view for high-value targets (with flexible column handling)
CREATE OR REPLACE VIEW high_value_targets AS
SELECT 
    c.id,
    c.name,
    c.industry,
    c.location,
    -- Handle contact_info column if it exists, otherwise use individual columns
    CASE 
        WHEN c.contact_info IS NOT NULL THEN c.contact_info
        ELSE jsonb_build_object(
            'email', c.email,
            'phone', c.phone,
            'website', c.website
        )
    END as contact_info,
    c.business_metrics->>'total_waste_value' as waste_value,
    c.business_metrics->>'potential_savings' as potential_savings,
    c.business_metrics->>'symbiosis_score' as symbiosis_score,
    c.business_metrics->>'carbon_reduction_potential' as carbon_reduction,
    c.data_quality_score,
    c.processing_status,
    c.created_at
FROM companies c
WHERE 
    c.processing_status = 'completed' 
    AND c.business_metrics IS NOT NULL
    AND (c.business_metrics->>'potential_savings')::NUMERIC > 50000
ORDER BY (c.business_metrics->>'potential_savings')::NUMERIC DESC;

-- Create view for symbiosis opportunities
CREATE OR REPLACE VIEW symbiosis_opportunities AS
SELECT 
    c1.id as company_a_id,
    c1.name as company_a_name,
    c1.industry as company_a_industry,
    c2.id as company_b_id,
    c2.name as company_b_name,
    c2.industry as company_b_industry,
    c1.location,
    (c1.business_metrics->>'potential_savings')::NUMERIC + (c2.business_metrics->>'potential_savings')::NUMERIC as combined_potential,
    c1.business_metrics->>'symbiosis_score' as company_a_score,
    c2.business_metrics->>'symbiosis_score' as company_b_score
FROM companies c1
CROSS JOIN companies c2
WHERE 
    c1.id < c2.id 
    AND c1.processing_status = 'completed'
    AND c2.processing_status = 'completed'
    AND c1.location = c2.location
    AND c1.industry != c2.industry
ORDER BY combined_potential DESC;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON bulk_imports TO authenticated;
GRANT SELECT ON high_value_targets TO authenticated;
GRANT SELECT ON symbiosis_opportunities TO authenticated;
GRANT EXECUTE ON FUNCTION calculate_company_business_metrics(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION update_company_processing_status(UUID, VARCHAR, INTEGER, JSONB, JSONB, JSONB, JSONB) TO authenticated;

-- =====================================================
-- VERIFICATION QUERIES - Run these to confirm setup
-- =====================================================

-- Check if tables were created
SELECT table_name FROM information_schema.tables 
WHERE table_name IN ('bulk_imports', 'companies') 
AND table_schema = 'public';

-- Check if new columns were added to companies table
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'companies' 
AND column_name IN ('data_quality_score', 'processing_status', 'enriched_data', 'business_metrics', 'ai_insights', 'recommendations', 'contact_info', 'email', 'phone', 'website');

-- Check if views were created
SELECT viewname FROM pg_views 
WHERE viewname IN ('high_value_targets', 'symbiosis_opportunities');

-- Check if functions were created
SELECT routine_name FROM information_schema.routines 
WHERE routine_name IN ('calculate_company_business_metrics', 'update_company_processing_status', 'update_bulk_imports_updated_at');

-- Show current companies table structure
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_name = 'companies' 
ORDER BY ordinal_position;

-- =====================================================
-- SUCCESS MESSAGE
-- =====================================================
-- If all queries above return results, your database is ready for real data!
-- You can now import your 50 company profiles and start extracting value. 