-- 1. Companies table
create table if not exists public.companies (
  id uuid primary key,
  name text not null,
  email text,
  role text default 'user',
  created_at timestamp with time zone default now()
);

-- 2. Company profiles
create table if not exists public.company_profiles (
  id uuid default uuid_generate_v4() primary key,
  company_id uuid references public.companies(id) on delete cascade,
  role text,
  location text,
  organization_type text,
  materials_of_interest text,
  sustainability_goals text,
  interests text
);

-- 3. Materials (listings)
create table if not exists public.materials (
  id uuid default uuid_generate_v4() primary key,
  company_id uuid references public.companies(id) on delete cascade,
  material_name text not null,
  quantity numeric,
  unit text,
  description text,
  type text check (type in ('waste', 'requirement')),
  created_at timestamp with time zone default now()
);

-- 4. Messages (async chat)
create table if not exists public.messages (
  id uuid default uuid_generate_v4() primary key,
  sender_id uuid references public.companies(id) on delete cascade,
  receiver_id uuid references public.companies(id) on delete cascade,
  content text not null,
  created_at timestamp with time zone default now()
);

-- 5. AI Recommendations
create table if not exists public.ai_recommendations (
  id uuid default uuid_generate_v4() primary key,
  company_id uuid references public.companies(id) on delete cascade,
  type text,
  title text,
  description text,
  confidence numeric,
  action_url text,
  status text,
  created_at timestamp with time zone default now()
);

-- 6. Connections (business requests)
create table if not exists public.connections (
  id uuid default uuid_generate_v4() primary key,
  requester_id uuid references public.companies(id) on delete cascade,
  recipient_id uuid references public.companies(id) on delete cascade,
  status text,
  created_at timestamp with time zone default now()
);

-- 7. Subscriptions (tiers)
create table if not exists public.subscriptions (
  id uuid default uuid_generate_v4() primary key,
  company_id uuid references public.companies(id) on delete cascade,
  tier text,
  status text,
  created_at timestamp with time zone default now(),
  expires_at timestamp with time zone
);

-- 8. Material matches (optional, for analytics/history)
create table if not exists public.material_matches (
  id uuid default uuid_generate_v4() primary key,
  material_id uuid references public.materials(id) on delete cascade,
  matched_material_id uuid references public.materials(id) on delete cascade,
  match_score numeric,
  status text,
  created_at timestamp with time zone default now()
);

-- 9. Transactions (records each transaction and fee)
create table if not exists public.transactions (
  id uuid default uuid_generate_v4() primary key,
  connection_id uuid references public.connections(id) on delete cascade,
  buyer_id uuid references public.companies(id),
  seller_id uuid references public.companies(id),
  material_id uuid references public.materials(id),
  amount numeric,
  fee numeric,
  fee_percentage numeric,
  status text,
  created_at timestamp with time zone default now()
);

-- AI Matches table for storing continuous matching results
CREATE TABLE IF NOT EXISTS ai_matches (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  material_id UUID REFERENCES materials(id) ON DELETE CASCADE,
  counterpart_id UUID REFERENCES materials(id) ON DELETE CASCADE,
  match_score DECIMAL(5,4) NOT NULL DEFAULT 0,
  match_quality VARCHAR(50) DEFAULT 'unknown',
  sustainability_score DECIMAL(5,4) DEFAULT 0,
  blockchain_status VARCHAR(50) DEFAULT 'pending',
  trigger_type VARCHAR(50) DEFAULT 'manual', -- 'new_listing', 'periodic', 'manual'
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_ai_matches_material_id ON ai_matches(material_id);
CREATE INDEX IF NOT EXISTS idx_ai_matches_score ON ai_matches(match_score DESC);
CREATE INDEX IF NOT EXISTS idx_ai_matches_created_at ON ai_matches(created_at DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_ai_matches_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at
CREATE TRIGGER trigger_update_ai_matches_updated_at
  BEFORE UPDATE ON ai_matches
  FOR EACH ROW
  EXECUTE FUNCTION update_ai_matches_updated_at(); 