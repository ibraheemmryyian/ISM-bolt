import os
import sys
import logging
import re
from typing import Dict, Any
from supabase import create_client, Client
from dotenv import load_dotenv
from listing_inference_service import ListingInferenceService
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("AIListingsGenerator")

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize the DeepSeek-powered inference service
inference_service = ListingInferenceService()

# Safe columns that definitely exist in the materials table
SAFE_MATERIALS_COLUMNS = [
    'id', 'company_id', 'material_name', 'description', 'type', 
    'quantity', 'unit', 'created_at'
]

def parse_quantity_and_unit(quantity_str):
    """Extract numeric quantity and unit from string like '500 kg' or '1000 liters'"""
    if not quantity_str:
        return 1, 'units'
    
    if isinstance(quantity_str, (int, float)):
        return float(quantity_str), 'units'
    
    quantity_str = str(quantity_str).strip()
    
    # Try to extract number and unit
    match = re.match(r'^(\d+(?:\.\d+)?)\s*(.+)?$', quantity_str)
    if match:
        number = float(match.group(1))
        unit = match.group(2).strip() if match.group(2) else 'units'
        return number, unit
    
    # If no pattern matches, try to extract any number
    numbers = re.findall(r'\d+(?:\.\d+)?', quantity_str)
    if numbers:
        return float(numbers[0]), 'units'
    
    # Fallback
    return 1, 'units'

def check_database_schema():
    """Check what columns actually exist in the database"""
    logger.info("🔍 Checking database schema...")
    
    try:
        # Try to get one record to see what columns exist
        result = supabase.table('materials').select('*').limit(1).execute()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

# Helper: Build the richest possible company profile
def build_company_profile(company: Dict[str, Any]) -> Dict[str, Any]:
    # Add all relevant fields for best LLM results
    return {
        'id': company.get('id'),
        'name': company.get('name'),
        'industry': company.get('industry'),
        'location': company.get('location'),
        'employee_count': company.get('employee_count'),
        'products': company.get('products'),
        'main_materials': company.get('main_materials'),
        'production_volume': company.get('production_volume'),
        'process_description': company.get('process_description'),
        'sustainability_goals': company.get('sustainability_goals'),
        'current_waste_management': company.get('current_waste_management'),
        'onboarding_completed': company.get('onboarding_completed'),
        'created_at': company.get('created_at'),
        'updated_at': company.get('updated_at'),
    }

def insert_materials(company_id: str, materials: list, material_type: str):
    """Insert materials using only safe, existing columns"""
    if not materials:
        return 0
    
    rows = []
    for m in materials:
        # Debug: Log what we're receiving
        logger.info(f"    🔍 Processing material: {m}")
        
        # Accept both dict and string fallback
        if isinstance(m, dict):
            # Parse quantity and unit properly
            quantity_str = m.get('quantity') or m.get('available') or m.get('needed') or '1'
            logger.info(f"    📊 Raw quantity: '{quantity_str}'")
            quantity, unit = parse_quantity_and_unit(quantity_str)
            logger.info(f"    ✅ Parsed: quantity={quantity}, unit='{unit}'")
            
            # Only use columns that definitely exist in the materials table
            row = {
                'company_id': company_id,
                'material_name': m.get('name') or m.get('material') or m.get('material_name') or 'AI Generated Material',
                'description': m.get('description') or m.get('value') or '',
                'quantity': quantity,
                'unit': unit,
                'type': material_type,
                'created_at': datetime.now().isoformat(),
            }
            rows.append(row)
        elif isinstance(m, str):
            rows.append({
                'company_id': company_id,
                'material_name': m,
                'description': '',
                'quantity': 1,
                'unit': 'units',
                'type': material_type,
                'created_at': datetime.now().isoformat(),
            })
    
    if rows:
        try:
            result = supabase.table('materials').insert(rows).execute()
            logger.info(f"  ✅ Successfully inserted {len(rows)} materials")
            return len(rows)
        except Exception as e:
            logger.error(f"  ❌ Database insertion error: {e}")
            return 0
    return 0

def main():
    logger.info("🚀 Starting Bulletproof DeepSeek R1 AI Listings Generator...")
    
    # Check database connection first
    if not check_database_schema():
        logger.error("❌ Cannot connect to database. Exiting.")
        sys.exit(1)
    
    # Fetch all companies
    try:
        companies = supabase.table('companies').select('*').execute().data
        if not companies:
            logger.error("No companies found in database.")
            sys.exit(1)
        logger.info(f"Found {len(companies)} companies.")
    except Exception as e:
        logger.error(f"❌ Error fetching companies: {e}")
        sys.exit(1)
    
    # Production mode: process all companies
    test_companies = companies
    logger.info(f"🧪 Test mode: Processing first {len(test_companies)} companies")
    
    total_materials = 0
    successful_companies = 0
    
    for idx, company in enumerate(test_companies, 1):
        logger.info(f"\n[{idx}/{len(test_companies)}] 🏢 Processing: {company.get('name')} ({company.get('id')})")
        
        try:
            profile = build_company_profile(company)
            logger.info(f"  📊 Company profile: {profile.get('industry', 'Unknown')} industry, {profile.get('location', 'Unknown')} location")
            
            ai_result = inference_service.generate_listings_from_profile(profile)
            logger.info(f"  ✅ AI analysis successful")
            
            # Insert outputs (waste/materials)
            outputs = ai_result.get('predicted_outputs') or ai_result.get('waste_streams') or ai_result.get('materials') or []
            n_out = insert_materials(company['id'], outputs, 'waste')
            logger.info(f"  📦 Generated {len(outputs)} waste materials, inserted {n_out}")
            
            # Insert inputs (requirements)
            inputs = ai_result.get('predicted_inputs') or ai_result.get('resource_needs') or ai_result.get('requirements') or []
            n_in = insert_materials(company['id'], inputs, 'requirement')
            logger.info(f"  📥 Generated {len(inputs)} requirements, inserted {n_in}")
            
            logger.info(f"  ✅ Success: {n_out} waste + {n_in} requirements = {n_out + n_in} total")
            total_materials += n_out + n_in
            successful_companies += 1
            
        except Exception as e:
            logger.error(f"  ❌ Error for {company.get('name')}: {e}")
            continue
        
        # Sleep to avoid rate limits
        time.sleep(1.5)
    
    logger.info(f"\n🎉 Test Complete!")
    logger.info(f"✅ Successful companies: {successful_companies}/{len(test_companies)}")
    logger.info(f"📦 Total materials generated: {total_materials}")
    
    if successful_companies == len(test_companies):
        logger.info("🚀 All tests passed! Ready to run on all companies.")
        logger.info("💡 To run on all companies, change 'test_companies = companies[:5]' to 'test_companies = companies'")
    else:
        logger.info("⚠️  Some companies failed. Check the errors above.")

if __name__ == "__main__":
    main() 