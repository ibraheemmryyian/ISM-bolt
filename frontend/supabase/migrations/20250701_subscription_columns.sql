-- Add subscription columns to companies table
-- Run this in your Supabase SQL editor

-- Add subscription-related columns to companies table
ALTER TABLE companies 
ADD COLUMN IF NOT EXISTS subscription_tier text DEFAULT 'free' CHECK (subscription_tier IN ('free', 'pro', 'enterprise')),
ADD COLUMN IF NOT EXISTS subscription_status text DEFAULT 'active' CHECK (subscription_status IN ('active', 'suspended', 'cancelled')),
ADD COLUMN IF NOT EXISTS subscription_expires_at timestamptz,
ADD COLUMN IF NOT EXISTS contact_name text,
ADD COLUMN IF NOT EXISTS email text;

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_companies_subscription_tier ON companies(subscription_tier);
CREATE INDEX IF NOT EXISTS idx_companies_subscription_status ON companies(subscription_status);

-- Update existing companies to have default subscription values
UPDATE companies 
SET subscription_tier = 'free', 
    subscription_status = 'active' 
WHERE subscription_tier IS NULL OR subscription_status IS NULL;

-- Success message
SELECT 'Subscription columns added to companies table successfully!' as status; 