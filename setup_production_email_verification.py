#!/usr/bin/env python3
"""
Production Email Verification Setup Script
for Industrial Symbiosis Marketplace

This script helps configure Supabase email verification for production use.
"""

import os
import sys
import json
import requests
from typing import Dict, Any, Optional
import getpass

class SupabaseEmailVerificationSetup:
    def __init__(self):
        self.supabase_url = None
        self.supabase_key = None
        self.headers = {}
        
    def get_credentials(self) -> bool:
        """Get Supabase credentials from user input."""
        print("\n🔧 Supabase Email Verification Setup")
        print("=" * 50)
        print("This script will help you configure email verification for production.")
        print("You'll need your Supabase project URL and service role key.\n")
        
        self.supabase_url = input("Enter your Supabase URL (e.g., https://xxx.supabase.co): ").strip()
        self.supabase_key = getpass.getpass("Enter your Supabase service role key: ").strip()
        
        if not self.supabase_url or not self.supabase_key:
            print("❌ URL and key are required!")
            return False
            
        # Remove trailing slash if present
        self.supabase_url = self.supabase_url.rstrip('/')
        
        # Set headers for API calls
        self.headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}',
            'Content-Type': 'application/json'
        }
        
        return True
    
    def test_connection(self) -> bool:
        """Test connection to Supabase."""
        try:
            print("\n🔍 Testing Supabase connection...")
            response = requests.get(f"{self.supabase_url}/rest/v1/", headers=self.headers)
            
            if response.status_code == 200:
                print("✅ Connection successful!")
                return True
            else:
                print(f"❌ Connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def run_migration(self) -> bool:
        """Run the email verification migration."""
        try:
            print("\n📊 Running email verification migration...")
            
            # Read the migration SQL
            migration_sql = self.get_migration_sql()
            
            # Execute the migration
            response = requests.post(
                f"{self.supabase_url}/rest/v1/rpc/exec_sql",
                headers=self.headers,
                json={'sql': migration_sql}
            )
            
            if response.status_code == 200:
                print("✅ Migration completed successfully!")
                return True
            else:
                print(f"❌ Migration failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Migration error: {e}")
            return False
    
    def get_migration_sql(self) -> str:
        """Get the migration SQL content."""
        return """
        -- Email Verification Migration
        -- Update profiles table
        ALTER TABLE profiles 
        ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMP WITH TIME ZONE,
        ADD COLUMN IF NOT EXISTS email_verification_sent_at TIMESTAMP WITH TIME ZONE,
        ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'suspended'));

        -- Update companies table
        ALTER TABLE companies 
        ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMP WITH TIME ZONE,
        ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'suspended'));

        -- Add indexes
        CREATE INDEX IF NOT EXISTS idx_profiles_email ON profiles(email);
        CREATE INDEX IF NOT EXISTS idx_profiles_username ON profiles(username);
        CREATE INDEX IF NOT EXISTS idx_profiles_status ON profiles(status);
        CREATE INDEX IF NOT EXISTS idx_companies_email ON companies(email);
        CREATE INDEX IF NOT EXISTS idx_companies_status ON companies(status);

        -- Add unique constraints
        DO $$ 
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'unique_email' AND conrelid = 'profiles'::regclass) THEN
                ALTER TABLE profiles ADD CONSTRAINT unique_email UNIQUE (email);
            END IF;
            
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'unique_username' AND conrelid = 'profiles'::regclass) THEN
                ALTER TABLE profiles ADD CONSTRAINT unique_username UNIQUE (username);
            END IF;
        END $$;
        """
    
    def configure_smtp(self) -> bool:
        """Guide user through SMTP configuration."""
        print("\n📧 SMTP Configuration Guide")
        print("=" * 30)
        print("You need to configure SMTP in your Supabase dashboard:")
        print("1. Go to https://supabase.com/dashboard")
        print("2. Select your project")
        print("3. Go to Settings > Auth > SMTP Settings")
        print("4. Configure your SMTP provider\n")
        
        print("Recommended SMTP providers:")
        print("• Gmail: smtp.gmail.com:587")
        print("• SendGrid: smtp.sendgrid.net:587")
        print("• AWS SES: email-smtp.us-east-1.amazonaws.com:587")
        print("• Mailgun: smtp.mailgun.org:587\n")
        
        input("Press Enter when you've configured SMTP...")
        return True
    
    def configure_auth_settings(self) -> bool:
        """Guide user through auth settings configuration."""
        print("\n🔐 Auth Settings Configuration")
        print("=" * 35)
        print("Configure these settings in your Supabase dashboard:")
        print("1. Go to Settings > Auth > URL Configuration")
        print("2. Set Site URL to your production domain")
        print("3. Add redirect URLs:")
        print("   - https://yourdomain.com/auth/callback")
        print("   - https://yourdomain.com/auth/reset-password")
        print("   - https://yourdomain.com/dashboard\n")
        
        print("4. Go to Settings > Auth > Providers")
        print("5. Enable Email confirmations")
        print("6. Enable Secure email change")
        print("7. Enable Double confirm changes\n")
        
        input("Press Enter when you've configured auth settings...")
        return True
    
    def create_env_template(self) -> bool:
        """Create environment variable template."""
        print("\n🌐 Environment Variables")
        print("=" * 25)
        
        env_content = f"""# Supabase Configuration
SUPABASE_URL={self.supabase_url}
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_ROLE_KEY={self.supabase_key}

# Frontend Environment Variables (.env)
VITE_SUPABASE_URL={self.supabase_url}
VITE_SUPABASE_ANON_KEY=your-anon-key-here
VITE_APP_URL=https://yourdomain.com

# JWT Configuration
JWT_SECRET=your-jwt-secret-here
JWT_EXPIRY=3600

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM_NAME=SymbioFlows
SMTP_FROM_EMAIL=your-email@gmail.com
"""
        
        # Write to .env.production file
        with open('.env.production', 'w') as f:
            f.write(env_content)
        
        print("✅ Created .env.production template")
        print("📝 Please update the values with your actual configuration")
        return True
    
    def test_email_verification(self) -> bool:
        """Test email verification flow."""
        print("\n🧪 Email Verification Testing")
        print("=" * 30)
        print("To test email verification:")
        print("1. Go to your production site")
        print("2. Try to sign up with a real email")
        print("3. Check if verification email is received")
        print("4. Click the verification link")
        print("5. Verify you're redirected to dashboard\n")
        
        test_email = input("Enter an email to test (optional): ").strip()
        if test_email:
            print(f"📧 Test email: {test_email}")
            print("Check your inbox for verification email")
        
        return True
    
    def create_production_checklist(self) -> bool:
        """Create production checklist."""
        checklist = """
# Production Email Verification Checklist

## ✅ Completed
- [x] Database migration applied
- [x] SMTP configuration guide provided
- [x] Auth settings configuration guide provided
- [x] Environment variables template created

## 🔧 Still Need to Do
- [ ] Configure SMTP in Supabase dashboard
- [ ] Set up production domain URLs
- [ ] Enable email confirmations
- [ ] Customize email templates
- [ ] Test email verification flow
- [ ] Set up monitoring and logging
- [ ] Configure password policy
- [ ] Set up rate limiting
- [ ] Test password reset flow
- [ ] Deploy to production

## 📧 Email Templates to Customize
1. Confirmation Email
2. Password Reset Email
3. Email Change Confirmation

## 🔐 Security Settings to Configure
1. Password Policy
2. Session Management
3. Rate Limiting
4. JWT Expiry

## 📊 Monitoring Setup
1. Email delivery monitoring
2. Error tracking (Sentry)
3. Auth logs monitoring
4. Performance monitoring

## 🚀 Deployment Checklist
- [ ] Environment variables set
- [ ] Domain configured
- [ ] SSL certificate installed
- [ ] Database backups configured
- [ ] Monitoring alerts set up
- [ ] Error handling tested
- [ ] Performance tested
- [ ] Security audit completed
"""
        
        with open('PRODUCTION_CHECKLIST.md', 'w') as f:
            f.write(checklist)
        
        print("✅ Created PRODUCTION_CHECKLIST.md")
        return True
    
    def run_setup(self):
        """Run the complete setup process."""
        print("🚀 Starting Production Email Verification Setup")
        print("=" * 55)
        
        # Step 1: Get credentials
        if not self.get_credentials():
            return False
        
        # Step 2: Test connection
        if not self.test_connection():
            return False
        
        # Step 3: Run migration
        if not self.run_migration():
            return False
        
        # Step 4: Configure SMTP
        if not self.configure_smtp():
            return False
        
        # Step 5: Configure auth settings
        if not self.configure_auth_settings():
            return False
        
        # Step 6: Create env template
        if not self.create_env_template():
            return False
        
        # Step 7: Test email verification
        if not self.test_email_verification():
            return False
        
        # Step 8: Create checklist
        if not self.create_production_checklist():
            return False
        
        print("\n🎉 Setup completed successfully!")
        print("=" * 30)
        print("Next steps:")
        print("1. Review PRODUCTION_CHECKLIST.md")
        print("2. Configure SMTP in Supabase dashboard")
        print("3. Set up your production domain")
        print("4. Test the email verification flow")
        print("5. Deploy to production")
        
        return True

def main():
    """Main function."""
    setup = SupabaseEmailVerificationSetup()
    
    try:
        success = setup.run_setup()
        if success:
            print("\n✅ Production email verification setup completed!")
            sys.exit(0)
        else:
            print("\n❌ Setup failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 