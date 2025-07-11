#!/usr/bin/env python3
"""
Fixed AI Listings Generator - Only uses existing database columns
"""

import os
import sys
import logging
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
logger = logging.getLogger("FixedAIListingsGenerator")

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize the DeepSeek-powered inference service
inference_service = ListingInferenceService()

# Helper: Build the richest possible company profile
def build_company_profile(company: Dict[str, Any]) -> Dict[str, Any]:
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
    """Insert materials using only existing database columns"""
    if not materials:
        return 0
    
    rows = []
    for m in materials:
        if isinstance(m, dict):
            # Only include columns that exist in the database
            row = {
                'company_id': company_id,
                'material_name': m.get('name') or m.get('material') or m.get('material_name') or 'AI Generated Material',
                'description': m.get('description') or m.get('value') or '',
                'quantity': m.get('quantity') or m.get('available') or m.get('needed') or '',
                'unit': m.get('unit') or '',
                'type': material_type,
                'ai_generated': True,
                'created_at': datetime.now().isoformat(),
                'quality_grade': m.get('quality_grade') or 'A',
                'current_cost': m.get('current_cost') or '',
                'potential_sources': m.get('potential_sources') or [],
                'price_per_unit': m.get('price_per_unit') or 0.0,
            }
            rows.append(row)
        elif isinstance(m, str):
            rows.append({
                'company_id': company_id,
                'material_name': m,
                'description': '',
                'quantity': '',
                'unit': '',
                'type': material_type,
                'ai_generated': True,
                'created_at': datetime.now().isoformat(),
                'quality_grade': 'A',
                'current_cost': '',
                'potential_sources': [],
                'price_per_unit': 0.0,
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
    logger.info("🚀 Starting Fixed AI Listings Generator...")
    
    # Fetch all companies
    try:
        companies = supabase.table('companies').select('*').execute().data
        if not companies:
            logger.error("❌ No companies found in database.")
            sys.exit(1)
        logger.info(f"📊 Found {len(companies)} companies.")
    except Exception as e:
        logger.error(f"❌ Error fetching companies: {e}")
        sys.exit(1)
    
    total_materials = 0
    successful_companies = 0
    
    # Limit to first 10 companies for testing
    test_companies = companies[:10]
    logger.info(f"🧪 Testing with first {len(test_companies)} companies...")
    
    for idx, company in enumerate(test_companies, 1):
        logger.info(f"\n[{idx}/{len(test_companies)}] Processing: {company.get('name')}")
        
        try:
            profile = build_company_profile(company)
            ai_result = inference_service.generate_listings_from_profile(profile)
            
            # Insert outputs (waste/materials)
            outputs = ai_result.get('predicted_outputs') or ai_result.get('waste_streams') or ai_result.get('materials') or []
            n_out = insert_materials(company['id'], outputs, 'waste')
            
            # Insert inputs (requirements)
            inputs = ai_result.get('predicted_inputs') or ai_result.get('resource_needs') or ai_result.get('requirements') or []
            n_in = insert_materials(company['id'], inputs, 'requirement')
            
            logger.info(f"  📦 Generated: {n_out} waste + {n_in} requirements = {n_out + n_in} total")
            total_materials += n_out + n_in
            successful_companies += 1
            
        except Exception as e:
            logger.error(f"  ❌ Error for {company.get('name')}: {e}")
            continue
        
        # Sleep to avoid rate limits
        time.sleep(1)
    
    logger.info(f"\n🎉 Test Complete!")
    logger.info(f"✅ Successful companies: {successful_companies}/{len(test_companies)}")
    logger.info(f"📦 Total materials generated: {total_materials}")
    
    if successful_companies == len(test_companies):
        logger.info("🚀 All tests passed! Ready to run on all companies.")
        logger.info("💡 To run on all companies, change 'test_companies = companies[:10]' to 'test_companies = companies'")
    else:
        logger.info("⚠️  Some companies failed. Check the errors above.")

if __name__ == "__main__":
    main() 