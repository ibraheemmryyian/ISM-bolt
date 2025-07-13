-- =====================================================
-- CLEAN MIGRATION AND COMPANY VIEW SCRIPT
-- Removes Existing Triggers and Creates Clean Tables
-- =====================================================

-- =====================================================
-- STEP 1: CLEAN UP EXISTING TRIGGERS
-- =====================================================

-- Drop all existing triggers that might conflict
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
DROP TRIGGER IF EXISTS update_companies_updated_at ON companies;
DROP TRIGGER IF EXISTS update_materials_updated_at ON materials;
DROP TRIGGER IF EXISTS update_products_updated_at ON products;
DROP TRIGGER IF EXISTS update_ai_matches_updated_at ON ai_matches;
DROP TRIGGER IF EXISTS update_ai_insights_updated_at ON ai_insights;
DROP TRIGGER IF EXISTS update_ai_recommendations_updated_at ON ai_recommendations;

-- =====================================================
-- STEP 2: CREATE ESSENTIAL TABLES ONLY
-- =====================================================

-- Companies table (main table for company profiles)
CREATE TABLE IF NOT EXISTS companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    location VARCHAR(255),
    employee_count INTEGER,
    products TEXT,
    main_materials TEXT,
    production_volume VARCHAR(100),
    process_description TEXT,
    sustainability_goals TEXT[],
    current_waste_management VARCHAR(100),
    onboarding_completed BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users table (for user management)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    company_name VARCHAR(255),
    subscription_tier VARCHAR(50) DEFAULT 'free',
    onboarding_completed BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Materials table (for company materials)
CREATE TABLE IF NOT EXISTS materials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    quantity DECIMAL(15,2),
    unit VARCHAR(50),
    cost_per_unit DECIMAL(10,2),
    supplier VARCHAR(255),
    sustainability_rating DECIMAL(3,2),
    carbon_footprint DECIMAL(10,2),
    recyclability_percentage DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Products table (for company products)
CREATE TABLE IF NOT EXISTS products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    production_volume INTEGER,
    unit_cost DECIMAL(10,2),
    selling_price DECIMAL(10,2),
    materials_used JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- STEP 3: INSERT SAMPLE COMPANY DATA
-- =====================================================

-- Insert sample companies with detailed profiles
INSERT INTO companies (name, industry, location, employee_count, products, main_materials, production_volume, process_description, sustainability_goals, current_waste_management, onboarding_completed) VALUES
    ('EcoTech Manufacturing', 'Manufacturing', 'Dubai, UAE', 250, 'Solar panels, Wind turbines, Energy storage systems', 'Silicon, Aluminum, Steel, Copper', '10,000 units/month', 'Advanced manufacturing with automated assembly lines and quality control systems', ARRAY['Carbon neutral by 2030', '100% renewable energy', 'Zero waste to landfill'], 'Recycling program, Waste segregation', true),
    
    ('GreenChem Solutions', 'Chemical Manufacturing', 'Abu Dhabi, UAE', 180, 'Bio-degradable chemicals, Green cleaning products, Eco-friendly solvents', 'Plant-based materials, Natural extracts, Recycled chemicals', '5,000 liters/day', 'Green chemistry processes with closed-loop systems and minimal waste generation', ARRAY['Reduce carbon footprint by 50%', '100% biodegradable products', 'Water conservation'], 'Chemical recycling, Waste treatment', true),
    
    ('Sustainable Steel Co.', 'Steel Manufacturing', 'Jeddah, Saudi Arabia', 450, 'Recycled steel products, Green steel, Construction materials', 'Recycled steel, Iron ore, Alloying elements', '50,000 tons/month', 'Electric arc furnace technology with renewable energy integration', ARRAY['Carbon neutral steel production', '100% recycled content', 'Energy efficiency'], 'Steel recycling, Scrap processing', true),
    
    ('BioPlastics Industries', 'Plastics Manufacturing', 'Riyadh, Saudi Arabia', 320, 'Biodegradable plastics, Compostable packaging, Bio-based polymers', 'Corn starch, Sugarcane, Algae-based materials', '20,000 kg/day', 'Biological fermentation and polymerization processes', ARRAY['100% biodegradable products', 'Zero fossil fuel use', 'Circular economy'], 'Composting, Biodegradation', true),
    
    ('CleanEnergy Power', 'Energy Production', 'Doha, Qatar', 200, 'Solar power plants, Wind farms, Energy storage', 'Solar panels, Wind turbines, Batteries', '100 MW capacity', 'Renewable energy generation with smart grid integration', ARRAY['100% renewable energy', 'Grid stability', 'Energy storage innovation'], 'Component recycling, Battery disposal', true),
    
    ('WaterTech Solutions', 'Water Treatment', 'Muscat, Oman', 150, 'Water purification systems, Desalination plants, Wastewater treatment', 'Membranes, Filters, Chemicals', '500,000 liters/day', 'Advanced filtration and reverse osmosis technology', ARRAY['Water conservation', 'Zero liquid discharge', 'Energy efficient processes'], 'Filter recycling, Chemical recovery', true),
    
    ('Circular Textiles', 'Textile Manufacturing', 'Kuwait City, Kuwait', 280, 'Recycled fabrics, Sustainable clothing, Eco-friendly textiles', 'Recycled cotton, Hemp, Bamboo fibers', '15,000 meters/day', 'Closed-loop textile production with minimal water usage', ARRAY['100% recycled materials', 'Zero water waste', 'Fair labor practices'], 'Fabric recycling, Dye recovery', true),
    
    ('Green Logistics', 'Logistics & Transportation', 'Manama, Bahrain', 120, 'Electric vehicles, Sustainable packaging, Green delivery', 'Electric motors, Batteries, Recycled packaging', '500 deliveries/day', 'Electric vehicle fleet with optimized routes and sustainable packaging', ARRAY['Zero emissions fleet', '100% sustainable packaging', 'Route optimization'], 'Vehicle recycling, Packaging reuse', true),
    
    ('EcoConstruction', 'Construction', 'Amman, Jordan', 350, 'Green buildings, Sustainable materials, Energy-efficient structures', 'Recycled concrete, Bamboo, Solar panels', '10,000 sqm/month', 'Sustainable construction with green building certification', ARRAY['LEED certification', 'Zero waste construction', 'Energy efficient buildings'], 'Material recycling, Waste reduction', true),
    
    ('Renewable Materials Co.', 'Materials Manufacturing', 'Beirut, Lebanon', 220, 'Bamboo products, Cork materials, Natural fibers', 'Bamboo, Cork, Hemp, Jute', '8,000 units/month', 'Sustainable material processing with natural preservation methods', ARRAY['100% natural materials', 'Sustainable harvesting', 'Local sourcing'], 'Material composting, Natural degradation', true)
ON CONFLICT DO NOTHING;

-- =====================================================
-- STEP 4: CREATE INDEXES FOR PERFORMANCE
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_companies_industry ON companies(industry);
CREATE INDEX IF NOT EXISTS idx_companies_location ON companies(location);
CREATE INDEX IF NOT EXISTS idx_companies_name ON companies(name);

-- =====================================================
-- STEP 5: COMPANY DETAILS VIEW QUERIES
-- =====================================================

-- Query 1: Basic company information
SELECT 
    name,
    industry,
    location,
    employee_count,
    products,
    main_materials,
    production_volume,
    onboarding_completed,
    created_at
FROM companies
ORDER BY name;

-- Query 2: Detailed company profiles with sustainability goals
SELECT 
    name,
    industry,
    location,
    employee_count,
    products,
    main_materials,
    production_volume,
    process_description,
    sustainability_goals,
    current_waste_management,
    onboarding_completed,
    created_at
FROM companies
ORDER BY industry, name;

-- Query 3: Company statistics by industry
SELECT 
    industry,
    COUNT(*) as company_count,
    AVG(employee_count) as avg_employees,
    STRING_AGG(name, ', ') as companies
FROM companies
GROUP BY industry
ORDER BY company_count DESC;

-- Query 4: Companies by location
SELECT 
    location,
    COUNT(*) as company_count,
    STRING_AGG(name, ', ') as companies
FROM companies
GROUP BY location
ORDER BY company_count DESC;

-- Query 5: Companies with sustainability focus
SELECT 
    name,
    industry,
    sustainability_goals,
    current_waste_management
FROM companies
WHERE array_length(sustainability_goals, 1) > 0
ORDER BY name;

-- =====================================================
-- MIGRATION COMPLETION
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '🎉 CLEAN MIGRATION COMPLETED!';
    RAISE NOTICE '📊 Created essential tables';
    RAISE NOTICE '🏢 Inserted 10 sample companies with detailed profiles';
    RAISE NOTICE '⚡ Created performance indexes';
    RAISE NOTICE '📋 Company details queries ready to run';
    RAISE NOTICE '🚀 ISM AI Platform ready for company data!';
END $$; 