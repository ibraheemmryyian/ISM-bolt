#!/usr/bin/env python3
"""
Check Database State
Simple script to check the current state of companies and materials in the database.
"""

import os
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
)

def check_database_state():
    """Check the current state of the database"""
    print("🔍 Checking Database State...")
    print("=" * 50)
    
    try:
        # Check companies
        print("📋 Fetching companies...")
        companies_response = supabase.table('companies').select('*').execute()
        companies = companies_response.data
        print(f"✅ Found {len(companies)} companies")
        
        # Check materials
        print("📦 Fetching materials...")
        materials_response = supabase.table('materials').select('*').execute()
        materials = materials_response.data
        print(f"✅ Found {len(materials)} materials")
        
        # Check requirements
        print("📋 Fetching requirements...")
        requirements_response = supabase.table('requirements').select('*').execute()
        requirements = requirements_response.data
        print(f"✅ Found {len(requirements)} requirements")
        
        print("\n" + "=" * 50)
        print("📊 DATABASE SUMMARY:")
        print(f"🏭 Companies: {len(companies)}")
        print(f"📦 Materials (outputs): {len(materials)}")
        print(f"📋 Requirements (inputs): {len(requirements)}")
        print(f"📈 Total listings: {len(materials) + len(requirements)}")
        
        # Check materials with company_id
        materials_with_company = [m for m in materials if m.get('company_id')]
        materials_without_company = [m for m in materials if not m.get('company_id')]
        
        print(f"\n🔗 Materials with company_id: {len(materials_with_company)}")
        print(f"❌ Materials without company_id: {len(materials_without_company)}")
        
        # Check requirements with company_id
        requirements_with_company = [r for r in requirements if r.get('company_id')]
        requirements_without_company = [r for r in requirements if not r.get('company_id')]
        
        print(f"🔗 Requirements with company_id: {len(requirements_with_company)}")
        print(f"❌ Requirements without company_id: {len(requirements_without_company)}")
        
        # Show sample companies
        if companies:
            print(f"\n🏭 Sample Companies (first 5):")
            for i, company in enumerate(companies[:5]):
                print(f"  {i+1}. {company.get('name', 'Unknown')} (ID: {company.get('id', 'No ID')})")
        
        # Show sample materials
        if materials:
            print(f"\n📦 Sample Materials (first 5):")
            for i, material in enumerate(materials[:5]):
                company_id = material.get('company_id', 'No company_id')
                print(f"  {i+1}. {material.get('name', 'Unknown')} (Company ID: {company_id})")
        
        # Check for orphaned materials (materials without valid company_id)
        valid_company_ids = {c['id'] for c in companies}
        orphaned_materials = [m for m in materials if m.get('company_id') not in valid_company_ids]
        
        if orphaned_materials:
            print(f"\n⚠️  WARNING: Found {len(orphaned_materials)} orphaned materials!")
            print("   These materials have company_id values that don't exist in the companies table.")
        
        # Check materials per company
        if companies and materials_with_company:
            materials_per_company = {}
            for material in materials_with_company:
                company_id = material['company_id']
                if company_id not in materials_per_company:
                    materials_per_company[company_id] = 0
                materials_per_company[company_id] += 1
            
            avg_materials = sum(materials_per_company.values()) / len(materials_per_company)
            print(f"\n📊 Average materials per company: {avg_materials:.1f}")
            
            companies_with_materials = len(materials_per_company)
            companies_without_materials = len(companies) - companies_with_materials
            print(f"🏭 Companies with materials: {companies_with_materials}")
            print(f"🏭 Companies without materials: {companies_without_materials}")
        
        print("\n" + "=" * 50)
        
    except Exception as e:
        print(f"❌ Error checking database: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database_state() 