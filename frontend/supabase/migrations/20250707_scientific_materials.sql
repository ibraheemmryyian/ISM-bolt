-- Add scientific material data columns for Next-Gen Materials API integration
-- This enhances materials with scientific properties and sustainability metrics

-- Add scientific data columns to materials table
ALTER TABLE materials ADD COLUMN IF NOT EXISTS scientific_properties jsonb;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS sustainability_metrics jsonb;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS environmental_impact jsonb;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS circular_opportunities jsonb;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS material_classification varchar(100);
ALTER TABLE materials ADD COLUMN IF NOT EXISTS recyclability_score decimal(3,2);
ALTER TABLE materials ADD COLUMN IF NOT EXISTS carbon_footprint decimal(10,2);
ALTER TABLE materials ADD COLUMN IF NOT EXISTS renewable_content decimal(5,2);
ALTER TABLE materials ADD COLUMN IF NOT EXISTS biodegradability_score decimal(3,2);
ALTER TABLE materials ADD COLUMN IF NOT EXISTS toxicity_level varchar(20);
ALTER TABLE materials ADD COLUMN IF NOT EXISTS energy_intensity decimal(10,2);
ALTER TABLE materials ADD COLUMN IF NOT EXISTS water_intensity decimal(10,2);

-- Add scientific data columns to ai_insights table
ALTER TABLE ai_insights ADD COLUMN IF NOT EXISTS scientific_data jsonb;
ALTER TABLE ai_insights ADD COLUMN IF NOT EXISTS sustainability_score decimal(5,2);
ALTER TABLE ai_insights ADD COLUMN IF NOT EXISTS material_compatibility_score decimal(5,2);
ALTER TABLE ai_insights ADD COLUMN IF NOT EXISTS circular_economy_potential decimal(5,2);
ALTER TABLE ai_insights ADD COLUMN IF NOT EXISTS environmental_risk_assessment jsonb;
ALTER TABLE ai_insights ADD COLUMN IF NOT EXISTS regulatory_compliance_status jsonb;

-- Create material classifications table
CREATE TABLE IF NOT EXISTS material_classifications (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name varchar(100) NOT NULL,
  category varchar(50) NOT NULL,
  subcategory varchar(50),
  properties jsonb,
  sustainability_profile jsonb,
  created_at timestamptz DEFAULT now()
);

-- Create sustainable alternatives table
CREATE TABLE IF NOT EXISTS sustainable_alternatives (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  original_material_id uuid REFERENCES materials(id) ON DELETE CASCADE,
  alternative_material_id uuid REFERENCES materials(id) ON DELETE CASCADE,
  sustainability_improvement decimal(5,2),
  cost_comparison jsonb,
  performance_comparison jsonb,
  implementation_difficulty varchar(20),
  created_at timestamptz DEFAULT now()
);

-- Create circular economy opportunities table
CREATE TABLE IF NOT EXISTS circular_opportunities (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  material_id uuid REFERENCES materials(id) ON DELETE CASCADE,
  opportunity_type varchar(50) NOT NULL,
  description text,
  potential_impact jsonb,
  implementation_steps text[],
  success_metrics jsonb,
  created_at timestamptz DEFAULT now()
);

-- Enable RLS on new tables
ALTER TABLE material_classifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE sustainable_alternatives ENABLE ROW LEVEL SECURITY;
ALTER TABLE circular_opportunities ENABLE ROW LEVEL SECURITY;

-- RLS Policies for material_classifications
CREATE POLICY "Anyone can view material classifications"
  ON material_classifications
  FOR SELECT
  TO authenticated
  USING (true);

-- RLS Policies for sustainable_alternatives
CREATE POLICY "Users can view sustainable alternatives"
  ON sustainable_alternatives
  FOR SELECT
  TO authenticated
  USING (true);

-- RLS Policies for circular_opportunities
CREATE POLICY "Users can view circular opportunities"
  ON circular_opportunities
  FOR SELECT
  TO authenticated
  USING (true);

-- Create indexes for performance
CREATE INDEX idx_materials_scientific_properties ON materials USING gin(scientific_properties);
CREATE INDEX idx_materials_sustainability_metrics ON materials USING gin(sustainability_metrics);
CREATE INDEX idx_materials_recyclability_score ON materials(recyclability_score DESC);
CREATE INDEX idx_materials_carbon_footprint ON materials(carbon_footprint ASC);
CREATE INDEX idx_sustainable_alternatives_original ON sustainable_alternatives(original_material_id);
CREATE INDEX idx_circular_opportunities_material ON circular_opportunities(material_id);

-- Create function to calculate material sustainability score
CREATE OR REPLACE FUNCTION calculate_material_sustainability_score(
  p_recyclability decimal DEFAULT 0,
  p_carbon_footprint decimal DEFAULT 0,
  p_renewable_content decimal DEFAULT 0,
  p_biodegradability decimal DEFAULT 0
) RETURNS decimal AS $$
DECLARE
  score decimal;
BEGIN
  score := 0;
  
  -- Recyclability contribution (25%)
  score := score + (COALESCE(p_recyclability, 0) * 0.25);
  
  -- Carbon footprint contribution (25%) - inverse relationship
  score := score + (GREATEST(0, 100 - (COALESCE(p_carbon_footprint, 0) * 10)) * 0.25);
  
  -- Renewable content contribution (25%)
  score := score + (COALESCE(p_renewable_content, 0) * 0.25);
  
  -- Biodegradability contribution (25%)
  score := score + (COALESCE(p_biodegradability, 0) * 0.25);
  
  RETURN LEAST(100, score);
END;
$$ LANGUAGE plpgsql;

-- Create function to find sustainable alternatives
CREATE OR REPLACE FUNCTION find_sustainable_alternatives(
  p_material_id uuid,
  p_min_improvement decimal DEFAULT 10
) RETURNS TABLE (
  alternative_id uuid,
  alternative_name text,
  sustainability_improvement decimal,
  cost_impact text,
  implementation_difficulty text
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    sa.alternative_material_id,
    m.name,
    sa.sustainability_improvement,
    sa.cost_comparison->>'impact' as cost_impact,
    sa.implementation_difficulty
  FROM sustainable_alternatives sa
  JOIN materials m ON m.id = sa.alternative_material_id
  WHERE sa.original_material_id = p_material_id
    AND sa.sustainability_improvement >= p_min_improvement
  ORDER BY sa.sustainability_improvement DESC;
END;
$$ LANGUAGE plpgsql;

-- Create function to get circular economy opportunities
CREATE OR REPLACE FUNCTION get_circular_opportunities(
  p_material_id uuid
) RETURNS TABLE (
  opportunity_id uuid,
  opportunity_type text,
  description text,
  potential_impact jsonb
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    co.id,
    co.opportunity_type,
    co.description,
    co.potential_impact
  FROM circular_opportunities co
  WHERE co.material_id = p_material_id
  ORDER BY co.potential_impact->>'impact_score' DESC;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT EXECUTE ON FUNCTION calculate_material_sustainability_score TO authenticated;
GRANT EXECUTE ON FUNCTION find_sustainable_alternatives TO authenticated;
GRANT EXECUTE ON FUNCTION get_circular_opportunities TO authenticated;

-- Create view for material sustainability insights
CREATE OR REPLACE VIEW material_sustainability_insights AS
SELECT 
  m.id,
  m.name,
  m.category,
  m.recyclability_score,
  m.carbon_footprint,
  m.renewable_content,
  m.biodegradability_score,
  calculate_material_sustainability_score(
    m.recyclability_score,
    m.carbon_footprint,
    m.renewable_content,
    m.biodegradability_score
  ) as sustainability_score,
  m.scientific_properties,
  m.sustainability_metrics,
  m.circular_opportunities
FROM materials m
WHERE m.scientific_properties IS NOT NULL;

-- Grant select on view
GRANT SELECT ON material_sustainability_insights TO authenticated;

-- Insert some sample material classifications
INSERT INTO material_classifications (name, category, subcategory, properties, sustainability_profile) VALUES
('Polyethylene (PE)', 'Polymers', 'Thermoplastics', 
 '{"density": 0.92, "melting_point": 130, "tensile_strength": 30}', 
 '{"recyclability": 85, "carbon_footprint": 2.1, "renewable_content": 0}'),
('Polylactic Acid (PLA)', 'Polymers', 'Bioplastics',
 '{"density": 1.25, "melting_point": 180, "tensile_strength": 50}',
 '{"recyclability": 70, "carbon_footprint": 0.8, "renewable_content": 100}'),
('Aluminum', 'Metals', 'Light Metals',
 '{"density": 2.7, "melting_point": 660, "tensile_strength": 310}',
 '{"recyclability": 95, "carbon_footprint": 8.1, "renewable_content": 0}'),
('Steel', 'Metals', 'Ferrous Metals',
 '{"density": 7.85, "melting_point": 1370, "tensile_strength": 400}',
 '{"recyclability": 90, "carbon_footprint": 1.8, "renewable_content": 0}')
ON CONFLICT DO NOTHING; 