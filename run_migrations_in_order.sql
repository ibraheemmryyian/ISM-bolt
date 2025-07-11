-- =====================================================
-- SYMBIOFLOWS MIGRATION EXECUTION ORDER
-- =====================================================
-- This script runs migrations in the correct order to avoid dependency issues

-- Step 1: Run the base schema first (creates core tables)
\i database/schema_optimized.sql

-- Step 2: Run the intelligent core migration (adds new features)
\i supabase/migrations/20250107_intelligent_core_schema.sql

-- Step 3: Clean synthetic data and prepare for Gulf data
\i clean_and_setup_database.sql

-- =====================================================
-- VERIFICATION QUERIES
-- =====================================================

-- Check if all tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN (
    'companies', 'materials', 'symbiotic_matches', 'shipments',
    'ai_interactions', 'ai_errors', 'feedback_requests', 'user_feedback',
    'api_call_logs', 'shipping_requests', 'shipping_labels'
  )
ORDER BY table_name;

-- Check if all indexes exist
SELECT indexname 
FROM pg_indexes 
WHERE schemaname = 'public' 
  AND indexname LIKE 'idx_%'
ORDER BY indexname;

-- Check if all triggers exist
SELECT trigger_name, event_object_table
FROM information_schema.triggers
WHERE trigger_schema = 'public'
ORDER BY event_object_table, trigger_name;

-- =====================================================
-- SUCCESS MESSAGE
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE 'All migrations completed successfully!';
    RAISE NOTICE 'Database is ready for Gulf company data import.';
END $$; 