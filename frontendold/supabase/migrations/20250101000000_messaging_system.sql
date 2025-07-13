-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  participant_count INTEGER NOT NULL DEFAULT 2,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create conversation participants table
CREATE TABLE IF NOT EXISTS conversation_participants (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
  user_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(conversation_id, user_id)
);

-- Create messages table
CREATE TABLE IF NOT EXISTS messages (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
  sender_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create connections table (for following/connecting with companies)
CREATE TABLE IF NOT EXISTS connections (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  follower_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  following_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(follower_id, following_id)
);

-- Create favorites table (for bookmarking materials)
CREATE TABLE IF NOT EXISTS favorites (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES companies(id) ON DELETE CASCADE,
  material_id UUID REFERENCES materials(id) ON DELETE CASCADE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, material_id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_conversation_participants_conversation_id ON conversation_participants(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_participants_user_id ON conversation_participants(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_sender_id ON messages(sender_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_connections_follower_id ON connections(follower_id);
CREATE INDEX IF NOT EXISTS idx_connections_following_id ON connections(following_id);
CREATE INDEX IF NOT EXISTS idx_favorites_user_id ON favorites(user_id);
CREATE INDEX IF NOT EXISTS idx_favorites_material_id ON favorites(material_id);

-- Create function to update conversation updated_at timestamp
CREATE OR REPLACE FUNCTION update_conversation_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  UPDATE conversations 
  SET updated_at = NOW() 
  WHERE id = NEW.conversation_id;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update conversation timestamp when new message is added
CREATE TRIGGER update_conversation_timestamp_trigger
  AFTER INSERT ON messages
  FOR EACH ROW
  EXECUTE FUNCTION update_conversation_timestamp();

-- Enable Row Level Security (RLS)
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_participants ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE favorites ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for conversations
CREATE POLICY "Users can view conversations they participate in" ON conversations
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM conversation_participants 
      WHERE conversation_id = conversations.id 
      AND user_id = auth.uid()::text
    )
  );

CREATE POLICY "Users can create conversations" ON conversations
  FOR INSERT WITH CHECK (true);

CREATE POLICY "Users can update conversations they participate in" ON conversations
  FOR UPDATE USING (
    EXISTS (
      SELECT 1 FROM conversation_participants 
      WHERE conversation_id = conversations.id 
      AND user_id = auth.uid()::text
    )
  );

-- Create RLS policies for conversation_participants
CREATE POLICY "Users can view conversation participants for their conversations" ON conversation_participants
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM conversation_participants cp2
      WHERE cp2.conversation_id = conversation_participants.conversation_id 
      AND cp2.user_id = auth.uid()::text
    )
  );

CREATE POLICY "Users can add participants to conversations they're in" ON conversation_participants
  FOR INSERT WITH CHECK (
    EXISTS (
      SELECT 1 FROM conversation_participants 
      WHERE conversation_id = conversation_participants.conversation_id 
      AND user_id = auth.uid()::text
    )
  );

-- Create RLS policies for messages
CREATE POLICY "Users can view messages in their conversations" ON messages
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM conversation_participants 
      WHERE conversation_id = messages.conversation_id 
      AND user_id = auth.uid()::text
    )
  );

CREATE POLICY "Users can send messages to conversations they participate in" ON messages
  FOR INSERT WITH CHECK (
    sender_id = auth.uid()::text AND
    EXISTS (
      SELECT 1 FROM conversation_participants 
      WHERE conversation_id = messages.conversation_id 
      AND user_id = auth.uid()::text
    )
  );

-- Create RLS policies for connections
CREATE POLICY "Users can view their own connections" ON connections
  FOR SELECT USING (follower_id = auth.uid()::text);

CREATE POLICY "Users can create their own connections" ON connections
  FOR INSERT WITH CHECK (follower_id = auth.uid()::text);

CREATE POLICY "Users can delete their own connections" ON connections
  FOR DELETE USING (follower_id = auth.uid()::text);

-- Create RLS policies for favorites
CREATE POLICY "Users can view their own favorites" ON favorites
  FOR SELECT USING (user_id = auth.uid()::text);

CREATE POLICY "Users can create their own favorites" ON favorites
  FOR INSERT WITH CHECK (user_id = auth.uid()::text);

CREATE POLICY "Users can delete their own favorites" ON favorites
  FOR DELETE USING (user_id = auth.uid()::text);

-- Insert some sample data for testing
INSERT INTO conversations (id, participant_count) VALUES 
  ('550e8400-e29b-41d4-a716-446655440001', 2),
  ('550e8400-e29b-41d4-a716-446655440002', 2)
ON CONFLICT DO NOTHING;

-- Insert sample conversation participants (assuming some companies exist)
INSERT INTO conversation_participants (conversation_id, user_id) VALUES 
  ('550e8400-e29b-41d4-a716-446655440001', (SELECT id FROM companies LIMIT 1)),
  ('550e8400-e29b-41d4-a716-446655440001', (SELECT id FROM companies LIMIT 1 OFFSET 1)),
  ('550e8400-e29b-41d4-a716-446655440002', (SELECT id FROM companies LIMIT 1)),
  ('550e8400-e29b-41d4-a716-446655440002', (SELECT id FROM companies LIMIT 1 OFFSET 2))
ON CONFLICT DO NOTHING;

-- Insert sample messages
INSERT INTO messages (conversation_id, sender_id, content) VALUES 
  ('550e8400-e29b-41d4-a716-446655440001', (SELECT id FROM companies LIMIT 1), 'Hi! I saw your material listing and I\'m interested.'),
  ('550e8400-e29b-41d4-a716-446655440001', (SELECT id FROM companies LIMIT 1 OFFSET 1), 'Great! What would you like to know?'),
  ('550e8400-e29b-41d4-a716-446655440002', (SELECT id FROM companies LIMIT 1), 'Let\'s discuss a potential partnership.')
ON CONFLICT DO NOTHING; 