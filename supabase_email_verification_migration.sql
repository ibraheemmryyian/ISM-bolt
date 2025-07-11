-- Email Verification Migration Script
-- Run this in your Supabase SQL editor to enable email verification

-- 1. Update profiles table for email verification
ALTER TABLE profiles 
ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS email_verification_sent_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'suspended'));

-- 2. Update companies table for email verification
ALTER TABLE companies 
ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'suspended'));

-- 3. Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_profiles_email ON profiles(email);
CREATE INDEX IF NOT EXISTS idx_profiles_username ON profiles(username);
CREATE INDEX IF NOT EXISTS idx_profiles_status ON profiles(status);
CREATE INDEX IF NOT EXISTS idx_companies_email ON companies(email);
CREATE INDEX IF NOT EXISTS idx_companies_status ON companies(status);

-- 4. Add unique constraints (if they don't exist)
DO $$ 
BEGIN
    -- Add unique email constraint to profiles if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'unique_email' 
        AND conrelid = 'profiles'::regclass
    ) THEN
        ALTER TABLE profiles ADD CONSTRAINT unique_email UNIQUE (email);
    END IF;
    
    -- Add unique username constraint to profiles if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'unique_username' 
        AND conrelid = 'profiles'::regclass
    ) THEN
        ALTER TABLE profiles ADD CONSTRAINT unique_username UNIQUE (username);
    END IF;
END $$;

-- 5. Create function to update email verification status
CREATE OR REPLACE FUNCTION update_email_verification_status()
RETURNS TRIGGER AS $$
BEGIN
    -- Update profiles table when auth.users email is confirmed
    IF NEW.email_confirmed_at IS NOT NULL AND OLD.email_confirmed_at IS NULL THEN
        UPDATE profiles 
        SET email_verified_at = NEW.email_confirmed_at,
            status = 'active'
        WHERE id = NEW.id;
        
        UPDATE companies 
        SET email_verified_at = NEW.email_confirmed_at,
            status = 'active'
        WHERE id = NEW.id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 6. Create trigger to automatically update verification status
DROP TRIGGER IF EXISTS email_verification_trigger ON auth.users;
CREATE TRIGGER email_verification_trigger
    AFTER UPDATE ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION update_email_verification_status();

-- 7. Create function to handle user registration
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    -- Insert into profiles table
    INSERT INTO profiles (
        id,
        username,
        email,
        company_name,
        status,
        email_verification_sent_at
    ) VALUES (
        NEW.id,
        NEW.raw_user_meta_data->>'username',
        NEW.email,
        NEW.raw_user_meta_data->>'company_name',
        'pending',
        NOW()
    );
    
    -- Insert into companies table
    INSERT INTO companies (
        id,
        name,
        email,
        username,
        role,
        status
    ) VALUES (
        NEW.id,
        NEW.raw_user_meta_data->>'company_name',
        NEW.email,
        NEW.raw_user_meta_data->>'username',
        'user',
        'pending'
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 8. Create trigger for new user registration
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION handle_new_user();

-- 9. Create RLS policies for email verification
-- Allow users to view their own profile
CREATE POLICY "Users can view own profile" ON profiles
    FOR SELECT USING (auth.uid()::text = id);

-- Allow users to update their own profile
CREATE POLICY "Users can update own profile" ON profiles
    FOR UPDATE USING (auth.uid()::text = id);

-- Allow users to view their own company
CREATE POLICY "Users can view own company" ON companies
    FOR SELECT USING (auth.uid()::text = id);

-- Allow users to update their own company
CREATE POLICY "Users can update own company" ON companies
    FOR UPDATE USING (auth.uid()::text = id);

-- 10. Create view for user verification status
CREATE OR REPLACE VIEW user_verification_status AS
SELECT 
    p.id,
    p.username,
    p.email,
    p.company_name,
    p.status as profile_status,
    p.email_verified_at,
    c.status as company_status,
    c.role,
    CASE 
        WHEN p.email_verified_at IS NOT NULL THEN 'verified'
        ELSE 'unverified'
    END as verification_status
FROM profiles p
LEFT JOIN companies c ON p.id = c.id;

-- 11. Grant permissions
GRANT SELECT ON user_verification_status TO authenticated;
GRANT SELECT ON profiles TO authenticated;
GRANT SELECT ON companies TO authenticated;

-- 12. Create function to resend verification email
CREATE OR REPLACE FUNCTION resend_verification_email(user_email TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    -- This would typically call Supabase's auth.resend() function
    -- For now, we'll just update the sent timestamp
    UPDATE profiles 
    SET email_verification_sent_at = NOW()
    WHERE email = user_email;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 13. Add comments for documentation
COMMENT ON TABLE profiles IS 'User profiles with email verification support';
COMMENT ON TABLE companies IS 'Company information with email verification support';
COMMENT ON FUNCTION update_email_verification_status() IS 'Updates verification status when email is confirmed';
COMMENT ON FUNCTION handle_new_user() IS 'Handles new user registration and profile creation';
COMMENT ON VIEW user_verification_status IS 'View showing user verification status';

-- 14. Verify the migration
SELECT 
    'Migration completed successfully' as status,
    COUNT(*) as total_profiles,
    COUNT(CASE WHEN email_verified_at IS NOT NULL THEN 1 END) as verified_profiles
FROM profiles; 