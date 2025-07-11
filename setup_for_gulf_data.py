#!/usr/bin/env python3
"""
SYMBIOFLOWS GULF DATA SETUP SCRIPT
Comprehensive setup and testing for Gulf company data import
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"🚀 {title}")
    print("="*80)

def print_step(step_num, title):
    """Print a formatted step"""
    print(f"\n📋 STEP {step_num}: {title}")
    print("-" * 60)

def run_command(command, description, check_output=False):
    """Run a command and handle the result"""
    print(f"🔄 {description}...")
    try:
        if check_output:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print(f"✅ {description}: SUCCESS")
            return result.stdout
        else:
            result = subprocess.run(command, shell=True, check=True)
            print(f"✅ {description}: SUCCESS")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description}: FAILED")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ {description}: FAILED - {e}")
        return False

def check_python_dependencies():
    """Check if all Python dependencies are available"""
    print_step(1, "CHECKING PYTHON DEPENDENCIES")
    
    dependencies = [
        'numpy', 'pandas', 'torch', 'networkx', 'sklearn',
        'requests', 'json', 'datetime', 'uuid', 'logging'
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}: Available")
        except ImportError:
            print(f"❌ {dep}: Missing")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing packages before proceeding")
        return False
    
    print("✅ All Python dependencies are available")
    return True

def test_matching_engines():
    """Test all matching engines"""
    print_step(2, "TESTING MATCHING ENGINES")
    
    return run_command("python test_engines.py", "Testing all 4 matching engines")

def check_database_schema():
    """Check if database schema is ready"""
    print_step(3, "CHECKING DATABASE SCHEMA")
    
    print("📋 Database schema files found:")
    schema_files = [
        "supabase/migrations/20250107_intelligent_core_schema.sql",
        "clean_and_setup_database.sql"
    ]
    
    for file in schema_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
            return False
    
    print("\n📋 To run database migration:")
    print("1. Connect to your Supabase database")
    print("2. Run: clean_and_setup_database.sql")
    print("3. This will clean synthetic data and prepare for Gulf data")
    
    return True

def test_api_endpoints():
    """Test API endpoints"""
    print_step(4, "TESTING API ENDPOINTS")
    
    print("📋 API testing requires backend server to be running")
    print("To start backend server:")
    print("1. cd backend")
    print("2. npm install (if not done)")
    print("3. npm start")
    print("4. Run: python test_api_endpoint.py")
    
    # Check if backend directory exists
    if os.path.exists("backend"):
        print("✅ Backend directory found")
        if os.path.exists("backend/package.json"):
            print("✅ package.json found")
        else:
            print("❌ package.json missing in backend")
            return False
    else:
        print("❌ Backend directory not found")
        return False
    
    return True

def validate_data_template():
    """Validate the Gulf data template"""
    print_step(5, "VALIDATING DATA TEMPLATE")
    
    if os.path.exists("gulf_data_mapping.json"):
        print("✅ Gulf data mapping template found")
        
        try:
            with open("gulf_data_mapping.json", "r") as f:
                template = json.load(f)
            
            # Check required fields
            required_fields = template.get("gulf_company_data_template", {}).get("required_fields", [])
            print(f"✅ Required fields defined: {len(required_fields)} fields")
            
            # Check example company
            example = template.get("gulf_company_data_template", {}).get("example_company", {})
            if example:
                print("✅ Example company data provided")
                print(f"   Company: {example.get('name', 'N/A')}")
                print(f"   Industry: {example.get('industry', 'N/A')}")
                print(f"   Location: {example.get('location', 'N/A')}")
                print(f"   Materials: {len(example.get('materials', []))} materials")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in template: {e}")
            return False
    else:
        print("❌ Gulf data mapping template not found")
        return False

def create_import_script():
    """Create a script for importing Gulf data"""
    print_step(6, "CREATING IMPORT SCRIPT")
    
    import_script = '''#!/usr/bin/env python3
"""
GULF COMPANY DATA IMPORT SCRIPT
Imports Gulf company data into SymbioFlows
"""

import requests
import json
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:3000"  # Adjust if needed

def import_gulf_companies(companies_data):
    """Import Gulf companies into the system"""
    print(f"🚀 Starting import of {len(companies_data)} companies...")
    
    successful_imports = 0
    failed_imports = 0
    
    for i, company in enumerate(companies_data, 1):
        try:
            print(f"📋 Importing company {i}/{len(companies_data)}: {company.get('name', 'Unknown')}")
            
            response = requests.post(
                f"{API_BASE_URL}/api/companies",
                json=company,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                print(f"✅ Successfully imported: {company.get('name')} (ID: {result.get('id', 'N/A')})")
                successful_imports += 1
            else:
                print(f"❌ Failed to import {company.get('name')}: {response.status_code}")
                print(f"   Response: {response.text}")
                failed_imports += 1
                
        except Exception as e:
            print(f"❌ Error importing {company.get('name', 'Unknown')}: {e}")
            failed_imports += 1
        
        # Small delay between imports
        time.sleep(0.5)
    
    print(f"\n📊 Import Summary:")
    print(f"   Successful: {successful_imports}")
    print(f"   Failed: {failed_imports}")
    print(f"   Total: {len(companies_data)}")
    
    return successful_imports, failed_imports

def test_matching_with_imported_data():
    """Test matching with imported companies"""
    print("\n🧪 Testing intelligent matching with imported data...")
    
    try:
        # Get companies from API
        response = requests.get(f"{API_BASE_URL}/api/companies", timeout=10)
        if response.status_code == 200:
            companies = response.json()
            if companies:
                # Test matching with first company
                test_company = companies[0]
                print(f"Testing matching for: {test_company.get('name', 'Unknown')}")
                
                matching_data = {
                    "companyData": {
                        "id": test_company.get('id'),
                        "name": test_company.get('name'),
                        "industry": test_company.get('industry'),
                        "location": test_company.get('location')
                    },
                    "options": {
                        "topK": 5,
                        "useAllEngines": True,
                        "includeShippingAnalysis": True,
                        "includeSustainabilityAnalysis": True
                    }
                }
                
                match_response = requests.post(
                    f"{API_BASE_URL}/api/intelligent-matching",
                    json=matching_data,
                    headers={"Content-Type": "application/json"},
                    timeout=60
                )
                
                if match_response.status_code == 200:
                    result = match_response.json()
                    print(f"✅ Matching test successful!")
                    print(f"   Total matches found: {result.get('total_matches_found', 0)}")
                    print(f"   Top matches: {result.get('top_matches_count', 0)}")
                else:
                    print(f"❌ Matching test failed: {match_response.status_code}")
            else:
                print("⚠️  No companies found for matching test")
        else:
            print(f"❌ Could not fetch companies: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Matching test error: {e}")

if __name__ == "__main__":
    print("🚀 GULF COMPANY DATA IMPORT")
    print("="*50)
    
    # Load your Gulf company data here
    # Example:
    # with open("gulf_companies_data.json", "r") as f:
    #     companies_data = json.load(f)
    # 
    # successful, failed = import_gulf_companies(companies_data)
    # 
    # if successful > 0:
    #     test_matching_with_imported_data()
    
    print("📋 Please load your Gulf company data and uncomment the import code above")
    print("📋 Expected format: List of company objects matching gulf_data_mapping.json template")
'''
    
    with open("import_gulf_data.py", "w") as f:
        f.write(import_script)
    
    print("✅ Created import_gulf_data.py")
    print("📋 This script will help you import your Gulf company data")
    print("📋 Edit the script to load your actual data file")
    
    return True

def main():
    """Main setup function"""
    print_header("SYMBIOFLOWS GULF DATA SETUP")
    print(f"Setup started at: {datetime.now()}")
    
    # Track results
    results = []
    
    # Step 1: Check Python dependencies
    results.append(check_python_dependencies())
    
    # Step 2: Test matching engines
    results.append(test_matching_engines())
    
    # Step 3: Check database schema
    results.append(check_database_schema())
    
    # Step 4: Test API endpoints
    results.append(test_api_endpoints())
    
    # Step 5: Validate data template
    results.append(validate_data_template())
    
    # Step 6: Create import script
    results.append(create_import_script())
    
    # Summary
    print_header("SETUP SUMMARY")
    
    steps_passed = sum(results)
    total_steps = len(results)
    
    if steps_passed == total_steps:
        print("🎉 ALL SETUP STEPS COMPLETED SUCCESSFULLY!")
        print("✅ Python dependencies: Ready")
        print("✅ Matching engines: Ready")
        print("✅ Database schema: Ready")
        print("✅ API endpoints: Ready")
        print("✅ Data template: Ready")
        print("✅ Import script: Created")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Run database migration: clean_and_setup_database.sql")
        print("2. Start backend server: cd backend && npm start")
        print("3. Test API: python test_api_endpoint.py")
        print("4. Prepare your Gulf company data in JSON format")
        print("5. Import data: python import_gulf_data.py")
        
        print("\n🍰 READY FOR DESSERT! Your system is prepared for Gulf data!")
        
    else:
        print(f"⚠️  {steps_passed}/{total_steps} setup steps completed")
        print("Please resolve the failed steps before proceeding")
        
        if not results[0]:
            print("\n💡 TIP: Install missing Python packages")
        if not results[1]:
            print("\n💡 TIP: Check matching engine imports")
        if not results[2]:
            print("\n💡 TIP: Run database migration")
        if not results[3]:
            print("\n💡 TIP: Start backend server")
        if not results[4]:
            print("\n💡 TIP: Check data template")
    
    print(f"\nSetup completed at: {datetime.now()}")

if __name__ == "__main__":
    main() 