-- Create material_exchanges table for tracking shipping and transactions
-- This keeps companies on-platform by handling logistics

CREATE TABLE IF NOT EXISTS material_exchanges (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  from_company_id uuid REFERENCES companies(id) ON DELETE CASCADE,
  to_company_id uuid REFERENCES companies(id) ON DELETE CASCADE,
  material_id uuid REFERENCES materials(id) ON DELETE CASCADE,
  quantity numeric NOT NULL,
  quantity_unit varchar(20) DEFAULT 'tons',
  
  -- Shipping details
  shipping_rate_id varchar(255),
  shipping_cost decimal(10,2),
  tracking_number varchar(100),
  label_url text,
  transaction_id varchar(255),
  
  -- Exchange details
  exchange_type varchar(20) DEFAULT 'sale' CHECK (exchange_type IN ('sale', 'trade', 'donation')),
  status varchar(20) DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'shipped', 'delivered', 'completed', 'cancelled')),
  
  -- Platform fees
  platform_fee decimal(10,2) DEFAULT 0,
  platform_fee_percentage decimal(5,2) DEFAULT 5.00,
  
  -- Timestamps
  confirmed_at timestamptz,
  shipped_at timestamptz,
  delivered_at timestamptz,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Enable RLS
ALTER TABLE material_exchanges ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view exchanges they're involved in"
  ON material_exchanges
  FOR SELECT
  TO authenticated
  USING (from_company_id = auth.uid() OR to_company_id = auth.uid());

CREATE POLICY "Users can create exchanges from their company"
  ON material_exchanges
  FOR INSERT
  TO authenticated
  WITH CHECK (from_company_id = auth.uid());

CREATE POLICY "Users can update exchanges they're involved in"
  ON material_exchanges
  FOR UPDATE
  TO authenticated
  USING (from_company_id = auth.uid() OR to_company_id = auth.uid());

-- Indexes for performance
CREATE INDEX idx_material_exchanges_from_company ON material_exchanges(from_company_id);
CREATE INDEX idx_material_exchanges_to_company ON material_exchanges(to_company_id);
CREATE INDEX idx_material_exchanges_material ON material_exchanges(material_id);
CREATE INDEX idx_material_exchanges_status ON material_exchanges(status);
CREATE INDEX idx_material_exchanges_created_at ON material_exchanges(created_at DESC);

-- Add shipping-related columns to companies table
ALTER TABLE companies ADD COLUMN IF NOT EXISTS address text;
ALTER TABLE companies ADD COLUMN IF NOT EXISTS city varchar(100);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS state varchar(50);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS zip varchar(20);
ALTER TABLE companies ADD COLUMN IF NOT EXISTS country varchar(50) DEFAULT 'US';
ALTER TABLE companies ADD COLUMN IF NOT EXISTS phone varchar(20);

-- Add shipping-related columns to materials table
ALTER TABLE materials ADD COLUMN IF NOT EXISTS shipping_weight numeric;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS shipping_dimensions jsonb;
ALTER TABLE materials ADD COLUMN IF NOT EXISTS shipping_category varchar(50);

-- Create function to calculate platform fees
CREATE OR REPLACE FUNCTION calculate_platform_fee(
  p_exchange_amount decimal,
  p_fee_percentage decimal DEFAULT 5.00
) RETURNS decimal AS $$
BEGIN
  RETURN (p_exchange_amount * p_fee_percentage / 100);
END;
$$ LANGUAGE plpgsql;

-- Create function to update exchange status
CREATE OR REPLACE FUNCTION update_exchange_status(
  p_exchange_id uuid,
  p_status varchar(20)
) RETURNS void AS $$
BEGIN
  UPDATE material_exchanges 
  SET 
    status = p_status,
    updated_at = now(),
    confirmed_at = CASE WHEN p_status = 'confirmed' THEN now() ELSE confirmed_at END,
    shipped_at = CASE WHEN p_status = 'shipped' THEN now() ELSE shipped_at END,
    delivered_at = CASE WHEN p_status = 'delivered' THEN now() ELSE delivered_at END
  WHERE id = p_exchange_id;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT EXECUTE ON FUNCTION calculate_platform_fee TO authenticated;
GRANT EXECUTE ON FUNCTION update_exchange_status TO authenticated;

-- Create view for exchange analytics
CREATE OR REPLACE VIEW exchange_analytics AS
SELECT 
  c.name as company_name,
  c.industry,
  COUNT(*) as total_exchanges,
  SUM(me.shipping_cost) as total_shipping_costs,
  SUM(me.platform_fee) as total_platform_fees,
  AVG(me.shipping_cost) as avg_shipping_cost,
  COUNT(CASE WHEN me.status = 'completed' THEN 1 END) as completed_exchanges
FROM material_exchanges me
JOIN companies c ON (me.from_company_id = c.id OR me.to_company_id = c.id)
GROUP BY c.id, c.name, c.industry;

-- Grant select on view
GRANT SELECT ON exchange_analytics TO authenticated; 