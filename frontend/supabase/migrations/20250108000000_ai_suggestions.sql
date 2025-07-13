-- Create ai_suggestions table
CREATE TABLE IF NOT EXISTS ai_suggestions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('energy', 'efficiency', 'partnership', 'waste', 'materials', 'technology')),
    potential_savings INTEGER NOT NULL DEFAULT 0,
    implementation_difficulty TEXT NOT NULL CHECK (implementation_difficulty IN ('easy', 'medium', 'hard')),
    priority TEXT NOT NULL CHECK (priority IN ('high', 'medium', 'low')),
    ai_reasoning TEXT NOT NULL,
    estimated_impact TEXT,
    implementation_time TEXT,
    carbon_reduction INTEGER NOT NULL DEFAULT 0,
    confidence_score INTEGER NOT NULL DEFAULT 0 CHECK (confidence_score >= 0 AND confidence_score <= 100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_ai_suggestions_company_id ON ai_suggestions(company_id);
CREATE INDEX IF NOT EXISTS idx_ai_suggestions_created_at ON ai_suggestions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ai_suggestions_category ON ai_suggestions(category);
CREATE INDEX IF NOT EXISTS idx_ai_suggestions_priority ON ai_suggestions(priority);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_ai_suggestions_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_ai_suggestions_updated_at
    BEFORE UPDATE ON ai_suggestions
    FOR EACH ROW
    EXECUTE FUNCTION update_ai_suggestions_updated_at();

-- Enable RLS
ALTER TABLE ai_suggestions ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY "Users can view their own AI suggestions" ON ai_suggestions
    FOR SELECT USING (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert their own AI suggestions" ON ai_suggestions
    FOR INSERT WITH CHECK (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can update their own AI suggestions" ON ai_suggestions
    FOR UPDATE USING (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    );

CREATE POLICY "Users can delete their own AI suggestions" ON ai_suggestions
    FOR DELETE USING (
        company_id IN (
            SELECT id FROM companies WHERE user_id = auth.uid()
        )
    ); 