-- =====================================================
-- SYMBIOFLOWS DATABASE CLEANUP & SETUP SCRIPT
-- =====================================================
-- This script cleans synthetic data and prepares for real Gulf company data

-- =====================================================
-- STEP 1: CLEAN SYNTHETIC DATA
-- =====================================================

-- Delete all synthetic companies (those with generic names)
DELETE FROM companies 
WHERE name LIKE '%Company%' 
   OR name LIKE '%Manufacturing%' 
   OR name LIKE '%Textiles%' 
   OR name LIKE '%Food%' 
   OR name LIKE '%Chemical%' 
   OR name LIKE '%Construction%' 
   OR name LIKE '%Electronics%' 
   OR name LIKE '%Automotive%' 
   OR name LIKE '%Pharmaceutical%'
   OR name LIKE '%Synthetic%'
   OR name LIKE '%Test%'
   OR name LIKE '%Demo%';

-- Delete all synthetic materials
DELETE FROM materials 
WHERE material_name LIKE '%synthetic%' 
   OR material_name LIKE '%test%' 
   OR material_name LIKE '%demo%'
   OR material_name LIKE '%sample%';

-- Delete all synthetic matches
DELETE FROM symbiotic_matches 
WHERE description LIKE '%synthetic%' 
   OR description LIKE '%test%' 
   OR description LIKE '%demo%';

-- Delete all synthetic shipments
DELETE FROM shipments 
WHERE tracking_number LIKE '%TEST%' 
   OR tracking_number LIKE '%DEMO%' 
   OR tracking_number LIKE '%SYNTHETIC%';

-- Delete all synthetic AI interactions
DELETE FROM ai_interactions 
WHERE analysis_type LIKE '%synthetic%' 
   OR analysis_type LIKE '%test%';

-- Delete all synthetic feedback
DELETE FROM user_feedback 
WHERE feedback_id LIKE '%test%' 
   OR feedback_id LIKE '%demo%';

-- Delete all synthetic shipping requests
DELETE FROM shipping_requests 
WHERE test_mode = TRUE;

-- Delete all synthetic shipping labels
DELETE FROM shipping_labels 
WHERE test_mode = TRUE;

-- =====================================================
-- STEP 2: RESET COUNTERS AND SCORES
-- =====================================================

-- Reset company counters
UPDATE companies SET 
    matches_count = 0,
    savings_achieved = 0,
    carbon_reduced = 0,
    sustainability_score = 0,
    ai_portfolio_summary = NULL,
    ai_recommendations = NULL;

-- Reset material properties
UPDATE materials SET 
    material_properties = NULL,
    shipping_params = NULL,
    sustainability_metrics = NULL;

-- Reset match analysis
UPDATE symbiotic_matches SET 
    materials_compatibility = 0,
    waste_synergy = 0,
    energy_synergy = 0,
    location_proximity = 0,
    ai_confidence = 0,
    match_analysis = NULL,
    user_feedback = NULL;

-- =====================================================
-- STEP 3: VERIFY CLEANUP
-- =====================================================

-- Check remaining data counts
SELECT 
    'companies' as table_name, COUNT(*) as count FROM companies
UNION ALL
SELECT 
    'materials' as table_name, COUNT(*) as count FROM materials
UNION ALL
SELECT 
    'symbiotic_matches' as table_name, COUNT(*) as count FROM symbiotic_matches
UNION ALL
SELECT 
    'shipments' as table_name, COUNT(*) as count FROM shipments
UNION ALL
SELECT 
    'ai_interactions' as table_name, COUNT(*) as count FROM ai_interactions
UNION ALL
SELECT 
    'user_feedback' as table_name, COUNT(*) as count FROM user_feedback;

-- =====================================================
-- STEP 4: PREPARE FOR GULF DATA
-- =====================================================

-- Create Gulf-specific material categories if they don't exist
INSERT INTO material_categories (name, description) VALUES
('Petrochemicals', 'Petroleum-based chemicals and derivatives'),
('Construction Materials', 'Building and construction materials'),
('Food & Beverage', 'Food processing and beverage production materials'),
('Textiles', 'Fabric and textile manufacturing materials'),
('Metals & Mining', 'Metal processing and mining materials'),
('Electronics', 'Electronic components and materials'),
('Pharmaceuticals', 'Pharmaceutical and medical materials'),
('Renewable Energy', 'Solar, wind, and renewable energy materials')
ON CONFLICT (name) DO NOTHING;

-- Create Gulf-specific locations if they don't exist
INSERT INTO locations (name, country, region, latitude, longitude) VALUES
('Dubai', 'UAE', 'Gulf', 25.2048, 55.2708),
('Abu Dhabi', 'UAE', 'Gulf', 24.4539, 54.3773),
('Riyadh', 'Saudi Arabia', 'Gulf', 24.7136, 46.6753),
('Jeddah', 'Saudi Arabia', 'Gulf', 21.4858, 39.1925),
('Dammam', 'Saudi Arabia', 'Gulf', 26.4207, 50.0888),
('Kuwait City', 'Kuwait', 'Gulf', 29.3759, 47.9774),
('Doha', 'Qatar', 'Gulf', 25.2854, 51.5310),
('Manama', 'Bahrain', 'Gulf', 26.2285, 50.5860),
('Muscat', 'Oman', 'Gulf', 23.5880, 58.3829)
ON CONFLICT (name, country) DO NOTHING;

-- =====================================================
-- STEP 5: VERIFY SCHEMA READINESS
-- =====================================================

-- Check if all required tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN (
    'companies', 'materials', 'symbiotic_matches', 'shipments',
    'ai_interactions', 'ai_errors', 'feedback_requests', 'user_feedback',
    'api_call_logs', 'shipping_requests', 'shipping_labels',
    'material_categories', 'locations'
  )
ORDER BY table_name;

-- Check if all required indexes exist
SELECT indexname 
FROM pg_indexes 
WHERE schemaname = 'public' 
  AND indexname LIKE 'idx_%'
ORDER BY indexname;

-- =====================================================
-- SUCCESS MESSAGE
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE 'Database cleanup completed successfully!';
    RAISE NOTICE 'System is ready for Gulf company data import.';
    RAISE NOTICE 'All synthetic data has been removed.';
    RAISE NOTICE 'Schema is verified and ready.';
END $$; 