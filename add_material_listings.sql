-- SQL script to add material listings directly to the database
-- Replace 'YOUR_USER_ID' with your actual user ID

-- Insert waste material listings
INSERT INTO materials (name, type, description, quantity, unit, frequency, specifications, sustainability_impact, market_value, logistics_notes, user_id, created_at, updated_at)
VALUES 
('Chemical Waste', 'waste', 'High-quality chemical waste from your production processes. Can be recycled or repurposed for various applications.', 500, 'tons', 'monthly', 'pH 6-8, low heavy metal content', 'Reduces landfill waste by 30%', '$200-300 per ton', 'Available for pickup at our facility', 'YOUR_USER_ID', NOW(), NOW()),
('Organic Waste', 'waste', 'Organic waste byproducts from your manufacturing. Suitable for composting or biogas production.', 300, 'tons', 'monthly', 'High carbon content, biodegradable', 'Can be converted to renewable energy', '$150-200 per ton', 'Available in bulk quantities', 'YOUR_USER_ID', NOW(), NOW()),
('Aqueous Waste', 'waste', 'Aqueous waste streams from your processes. Contains recoverable minerals and compounds.', 1000, 'liters', 'weekly', 'Low toxicity, treatable', 'Reduces water pollution', '$0.50-1.00 per liter', 'Available in IBC containers', 'YOUR_USER_ID', NOW(), NOW());

-- Insert requirement material listings
INSERT INTO materials (name, type, description, quantity, unit, frequency, specifications, sustainability_impact, market_value, logistics_notes, user_id, created_at, updated_at)
VALUES 
('Raw Chemicals', 'requirement', 'Seeking high-quality raw chemicals for your production processes.', 1000, 'kg', 'monthly', 'USP grade, 99% purity', 'Prefer suppliers with sustainable practices', '$5-10 per kg', 'Need delivery to our facility', 'YOUR_USER_ID', NOW(), NOW()),
('Catalysts', 'requirement', 'Looking for industrial catalysts for your chemical processes.', 200, 'kg', 'quarterly', 'High activity, low metal content', 'Reduces energy consumption in processes', '$20-30 per kg', 'Need specialized handling', 'YOUR_USER_ID', NOW(), NOW()),
('Solvents', 'requirement', 'Require industrial solvents for your manufacturing.', 2000, 'liters', 'monthly', 'High purity, low water content', 'Prefer recycled or bio-based options', '$2-5 per liter', 'Bulk delivery preferred', 'YOUR_USER_ID', NOW(), NOW());

-- Update company profile to mark onboarding as completed
UPDATE companies
SET onboarding_completed = true, updated_at = NOW()
WHERE id = 'YOUR_USER_ID';