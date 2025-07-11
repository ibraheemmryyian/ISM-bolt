#!/usr/bin/env python3
"""
Database Setup and Import Script for ISM [AI] Platform
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file for frontend with Supabase configuration."""
    env_content = """# Supabase Configuration
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key-here

# Backend API URL
VITE_API_URL=http://localhost:5000
"""
    
    env_path = Path("frontend/.env")
    if env_path.exists():
        print(f"⚠️  .env file already exists at {env_path}")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            return
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created {env_path}")
    print("📝 Please edit this file with your actual Supabase credentials")

def get_supabase_credentials():
    """Get Supabase credentials from user input."""
    print("\n🔧 Supabase Configuration")
    print("=" * 50)
    print("You can find these in your Supabase project settings:")
    print("1. Go to https://supabase.com/dashboard")
    print("2. Select your project")
    print("3. Go to Settings > API")
    print("4. Copy the URL and anon key\n")
    
    url = input("Enter your Supabase URL: ").strip()
    key = input("Enter your Supabase anon key: ").strip()
    
    if not url or not key:
        print("❌ URL and key are required!")
        return None, None
    
    return url, key

def update_import_script(url, key):
    """Update the import script with actual credentials."""
    script_path = Path("import_companies.py")
    
    if not script_path.exists():
        print("❌ import_companies.py not found!")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace placeholder values
    content = content.replace(
        'SUPABASE_URL = "https://your-project.supabase.co"',
        f'SUPABASE_URL = "{url}"'
    )
    content = content.replace(
        'SUPABASE_KEY = "your-anon-key"',
        f'SUPABASE_KEY = "{key}"'
    )
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print("✅ Updated import_companies.py with your credentials")
    return True

def update_env_file(url, key):
    """Update the .env file with actual credentials."""
    env_path = Path("frontend/.env")
    
    if not env_path.exists():
        print("❌ .env file not found! Run setup first.")
        return False
    
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Replace placeholder values
    content = content.replace(
        'VITE_SUPABASE_URL=https://your-project.supabase.co',
        f'VITE_SUPABASE_URL={url}'
    )
    content = content.replace(
        'VITE_SUPABASE_ANON_KEY=your-anon-key-here',
        f'VITE_SUPABASE_ANON_KEY={key}'
    )
    
    with open(env_path, 'w') as f:
        f.write(content)
    
    print("✅ Updated .env file with your credentials")
    return True

def check_requirements():
    """Check if required packages are installed."""
    required_packages = ['supabase']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install supabase")
        return False
    
    print("✅ All required packages are installed")
    return True

def main():
    """Main setup function."""
    print("🚀 ISM [AI] Database Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Create .env file if it doesn't exist
    env_path = Path("frontend/.env")
    if not env_path.exists():
        create_env_file()
        print("\n📋 Next steps:")
        print("1. Edit frontend/.env with your Supabase credentials")
        print("2. Run this script again to import data")
        return
    
    # Get credentials and update files
    url, key = get_supabase_credentials()
    if not url or not key:
        return
    
    if update_env_file(url, key) and update_import_script(url, key):
        print("\n✅ Configuration complete!")
        print("\n📋 Next steps:")
        print("1. Run: python import_companies.py")
        print("   This will import 250 Gulf companies from gulf_company_data.txt")
        print("   and generate realistic material listings for each company")
        print("2. Start your frontend: cd frontend && npm run dev")
        print("3. Start your backend: cd backend && npm start")
        print("4. Access admin at: http://localhost:5173/admin")
        print("   Password: NA10EN")
        print("\n📊 Expected import results:")
        print("   - 250 Gulf companies (Construction, Oil & Gas, Healthcare, etc.)")
        print("   - 1000+ material listings (waste and requirements)")
        print("   - 1 test application for admin review")

if __name__ == "__main__":
    main() 