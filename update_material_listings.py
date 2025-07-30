#!/usr/bin/env python3
"""
Script to directly update the database with material listings
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import requests
import sys

# Load environment variables from .env file
env_file = Path('backend/.env')
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Loaded environment from: {env_file}")
else:
    print(f"⚠️ .env file not found: {env_file}")
    sys.exit(1)

# Get Supabase credentials from environment
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    print("❌ Missing Supabase credentials in .env file")
    sys.exit(1)

print("🔄 Connecting to Supabase database...")

# Function to generate material listings
def generate_material_listings(company_profile):
    """Generate material listings for a company profile"""
    print(f"🏭 Generating material listings for: {company_profile.get('name', 'Unknown')}")
    
    industry_category = company_profile.get('industry', 'chemical_manufacturing')
    
    # Generate waste material listings
    waste_listings = [
        {
            'name': 'Chemical Waste',
            'type': 'waste',
            'description': f"High-quality chemical waste from {company_profile.get('name')} production processes. Can be recycled or repurposed for various applications.",
            'quantity': 500,
            'unit': 'tons',
            'frequency': 'monthly',
            'specifications': 'pH 6-8, low heavy metal content',
            'sustainability_impact': 'Reduces landfill waste by 30%',
            'market_value': '$200-300 per ton',
            'logistics_notes': 'Available for pickup at our facility',
            'user_id': company_profile.get('user_id')
        },
        {
            'name': 'Organic Waste',
            'type': 'waste',
            'description': f"Organic waste byproducts from {company_profile.get('name')} manufacturing. Suitable for composting or biogas production.",
            'quantity': 300,
            'unit': 'tons',
            'frequency': 'monthly',
            'specifications': 'High carbon content, biodegradable',
            'sustainability_impact': 'Can be converted to renewable energy',
            'market_value': '$150-200 per ton',
            'logistics_notes': 'Available in bulk quantities',
            'user_id': company_profile.get('user_id')
        },
        {
            'name': 'Aqueous Waste',
            'type': 'waste',
            'description': f"Aqueous waste streams from {company_profile.get('name')} processes. Contains recoverable minerals and compounds.",
            'quantity': 1000,
            'unit': 'liters',
            'frequency': 'weekly',
            'specifications': 'Low toxicity, treatable',
            'sustainability_impact': 'Reduces water pollution',
            'market_value': '$0.50-1.00 per liter',
            'logistics_notes': 'Available in IBC containers',
            'user_id': company_profile.get('user_id')
        }
    ]
    
    # Generate requirement material listings
    requirement_listings = [
        {
            'name': 'Raw Chemicals',
            'type': 'requirement',
            'description': f"Seeking high-quality raw chemicals for {company_profile.get('name')} production processes.",
            'quantity': 1000,
            'unit': 'kg',
            'frequency': 'monthly',
            'specifications': 'USP grade, 99% purity',
            'sustainability_impact': 'Prefer suppliers with sustainable practices',
            'market_value': '$5-10 per kg',
            'logistics_notes': 'Need delivery to our facility',
            'user_id': company_profile.get('user_id')
        },
        {
            'name': 'Catalysts',
            'type': 'requirement',
            'description': f"Looking for industrial catalysts for {company_profile.get('name')} chemical processes.",
            'quantity': 200,
            'unit': 'kg',
            'frequency': 'quarterly',
            'specifications': 'High activity, low metal content',
            'sustainability_impact': 'Reduces energy consumption in processes',
            'market_value': '$20-30 per kg',
            'logistics_notes': 'Need specialized handling',
            'user_id': company_profile.get('user_id')
        },
        {
            'name': 'Solvents',
            'type': 'requirement',
            'description': f"Require industrial solvents for {company_profile.get('name')} manufacturing.",
            'quantity': 2000,
            'unit': 'liters',
            'frequency': 'monthly',
            'specifications': 'High purity, low water content',
            'sustainability_impact': 'Prefer recycled or bio-based options',
            'market_value': '$2-5 per liter',
            'logistics_notes': 'Bulk delivery preferred',
            'user_id': company_profile.get('user_id')
        }
    ]
    
    return waste_listings + requirement_listings

# Function to insert material listings into Supabase
def insert_material_listings(listings):
    """Insert material listings into Supabase"""
    print(f"📥 Inserting {len(listings)} material listings into database...")
    
    headers = {
        'apikey': SUPABASE_ANON_KEY,
        'Authorization': f'Bearer {SUPABASE_ANON_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=minimal'
    }
    
    url = f"{SUPABASE_URL}/rest/v1/materials"
    
    try:
        response = requests.post(url, headers=headers, json=listings)
        if response.status_code in [200, 201, 204]:
            print(f"✅ Successfully inserted {len(listings)} material listings")
            return True
        else:
            print(f"❌ Failed to insert material listings: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error inserting material listings: {str(e)}")
        return False

# Function to update company profile
def update_company_profile(company_id, update_data):
    """Update company profile in Supabase"""
    print(f"📝 Updating company profile for ID: {company_id}")
    
    headers = {
        'apikey': SUPABASE_ANON_KEY,
        'Authorization': f'Bearer {SUPABASE_ANON_KEY}',
        'Content-Type': 'application/json',
        'Prefer': 'return=minimal'
    }
    
    url = f"{SUPABASE_URL}/rest/v1/companies?id=eq.{company_id}"
    
    try:
        response = requests.patch(url, headers=headers, json=update_data)
        if response.status_code in [200, 201, 204]:
            print(f"✅ Successfully updated company profile")
            return True
        else:
            print(f"❌ Failed to update company profile: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error updating company profile: {str(e)}")
        return False

def main():
    """Main function"""
    print("🚀 SymbioFlows Material Listings Updater")
    print("=" * 50)
    
    # Get user ID from command line
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        user_id = input("Enter your user ID: ")
    
    if not user_id:
        print("❌ No user ID provided")
        sys.exit(1)
    
    # Create company profile
    company_profile = {
        'name': 'Your Company',
        'industry': 'chemical_manufacturing',
        'products': 'Industrial chemicals, solvents, cleaning agents',
        'production_volume': '5000 tons per year',
        'processes': 'Chemical synthesis, distillation, purification',
        'location': 'Global',
        'employee_count': '100-200',
        'onboarding_completed': True,
        'user_id': user_id
    }
    
    # Generate material listings
    material_listings = generate_material_listings(company_profile)
    
    # Insert material listings into database
    success = insert_material_listings(material_listings)
    
    if success:
        # Update company profile
        update_success = update_company_profile(user_id, {
            'onboarding_completed': True,
            'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ')
        })
        
        if update_success:
            print("\n🎉 Material listings have been added to your account!")
            print("🔄 Please refresh your dashboard to see the new listings")
        else:
            print("\n⚠️ Material listings were added but company profile update failed")
    else:
        print("\n❌ Failed to add material listings")
    
    print("=" * 50)

if __name__ == "__main__":
    main()