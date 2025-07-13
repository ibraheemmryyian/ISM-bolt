-- Create federated_learning_feedback table
CREATE TABLE IF NOT EXISTS federated_learning_feedback (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    type TEXT NOT NULL CHECK (type IN ('suggestion', 'material', 'opportunity')),
    action TEXT NOT NULL CHECK (action IN ('approve', 'reject', 'already_using', 'possible', 'implemented', 'not_relevant')),
    reason TEXT,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    item_id TEXT NOT NULL,
    item_data JSONB NOT NULL,
    confidence_score INTEGER CHECK (confidence_score >= 0 AND confidence_score <= 100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_federated_learning_user_id ON federated_learning_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_federated_learning_company_id ON federated_learning_feedback(company_id);
CREATE INDEX IF NOT EXISTS idx_federated_learning_type ON federated_learning_feedback(type);
CREATE INDEX IF NOT EXISTS idx_federated_learning_action ON federated_learning_feedback(action);
CREATE INDEX IF NOT EXISTS idx_federated_learning_created_at ON federated_learning_feedback(created_at DESC);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_federated_learning_feedback_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_federated_learning_feedback_updated_at
    BEFORE UPDATE ON federated_learning_feedback
    FOR EACH ROW
    EXECUTE FUNCTION update_federated_learning_feedback_updated_at();

-- Enable RLS
ALTER TABLE federated_learning_feedback ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY "Users can view their own federated learning feedback" ON federated_learning_feedback
    FOR SELECT USING (user_id = auth.uid());

CREATE POLICY "Users can insert their own federated learning feedback" ON federated_learning_feedback
    FOR INSERT WITH CHECK (user_id = auth.uid());

CREATE POLICY "Users can update their own federated learning feedback" ON federated_learning_feedback
    FOR UPDATE USING (user_id = auth.uid());

CREATE POLICY "Users can delete their own federated learning feedback" ON federated_learning_feedback
    FOR DELETE USING (user_id = auth.uid()); 