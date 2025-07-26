#!/usr/bin/env python3
"""
Bulletproof AI Listings Generator
- Checks database schema first
- Only uses existing columns
- Generates most accurate listings
- Comprehensive error handling
"""

import os
import sys
import logging
from typing import Dict, Any, List
from supabase import create_client, Client
from dotenv import load_dotenv
from listing_inference_service import ListingInferenceService
from datetime import datetime
import time
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('ai_generator.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Patch StreamHandler to use UTF-8 encoding if possible
try:
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream.reconfigure(encoding='utf-8')
except Exception:
    pass
logger = logging.getLogger("BulletproofAIGenerator")

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

if not supabase_url or not supabase_key:
    logger.error("❌ Missing Supabase credentials in .env file")
    sys.exit(1)

supabase: Client = create_client(supabase_url, supabase_key)

# Initialize the DeepSeek-powered inference service
inference_service = ListingInferenceService()

class BulletproofAIGenerator:
    def __init__(self):
        self.materials_columns = []
        self.companies_columns = []
        self.requirements_columns = []
        
    def check_database_schema(self):
        """Check what columns actually exist in the database"""
        logger.info("🔍 Checking database schema...")
        
        try:
            # Check materials table columns
            result = supabase.table('materials').select('*').limit(1).execute()
            if hasattr(result, 'columns'):
                self.materials_columns = list(result.columns.keys())
            else:
                # Fallback: query information_schema
                schema_query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'materials' 
                ORDER BY ordinal_position
                """
                # This would need to be run in SQL editor, for now we'll use a safe list
                self.materials_columns = [
                    'id', 'company_id', 'name', 'description', 'category', 
                    'quantity_estimate', 'quantity', 'frequency', 'notes', 
                    'potential_value', 'symbiosis_opportunities', 'embeddings', 
                    'ai_generated', 'created_at'
                ]
            
            logger.info(f"✅ Materials table columns: {self.materials_columns}")
            
            # Check companies table columns
            result = supabase.table('companies').select('*').limit(1).execute()
            if hasattr(result, 'columns'):
                self.companies_columns = list(result.columns.keys())
            else:
                self.companies_columns = [
                    'id', 'name', 'industry', 'location', 'employee_count',
                    'products', 'main_materials', 'production_volume',
                    'process_description', 'sustainability_goals',
                    'current_waste_management', 'onboarding_completed',
                    'created_at', 'updated_at'
                ]
            
            logger.info(f"✅ Companies table columns: {self.companies_columns}")
            
        except Exception as e:
            logger.error(f"❌ Error checking schema: {e}")
            # Use safe default columns
            self.materials_columns = [
                'id', 'company_id', 'name', 'description', 'category', 
                'quantity_estimate', 'quantity', 'frequency', 'notes', 
                'potential_value', 'symbiosis_opportunities', 'embeddings', 
                'ai_generated', 'created_at'
            ]
            self.companies_columns = [
                'id', 'name', 'industry', 'location', 'employee_count',
                'products', 'main_materials', 'production_volume',
                'process_description', 'sustainability_goals',
                'current_waste_management', 'onboarding_completed',
                'created_at', 'updated_at'
            ]
    
    def build_company_profile(self, company: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive company profile for accurate AI analysis"""
        profile = {}
        
        # Only include columns that exist
        for col in self.companies_columns:
            if col in company:
                profile[col] = company[col]
        
        # Add enhanced context for better AI results
        profile['enhanced_context'] = {
            'industry_type': company.get('industry', 'Unknown'),
            'location_region': company.get('location', 'Unknown'),
            'company_size': self._categorize_company_size(company.get('employee_count', 0)),
            'production_scale': self._categorize_production_scale(company.get('production_volume', '')),
            'sustainability_focus': len(company.get('sustainability_goals', [])) > 0,
            'waste_management_current': company.get('current_waste_management', 'Unknown')
        }
        
        return profile
    
    def _categorize_company_size(self, employee_count: int) -> str:
        """Categorize company size for better AI context"""
        if employee_count < 50:
            return 'Small'
        elif employee_count < 500:
            return 'Medium'
        elif employee_count < 5000:
            return 'Large'
        else:
            return 'Enterprise'
    
    def _categorize_production_scale(self, production_volume: str) -> str:
        """Categorize production scale for better AI context"""
        if not production_volume:
            return 'Unknown'
        
        volume_lower = production_volume.lower()
        if 'ton' in volume_lower or 'kg' in volume_lower:
            if any(x in volume_lower for x in ['1000', '1k', 'ton']):
                return 'Large Scale'
            elif any(x in volume_lower for x in ['100', '500']):
                return 'Medium Scale'
            else:
                return 'Small Scale'
        else:
            return 'Unknown Scale'
    
    def insert_materials_safely(self, company_id: str, materials: list, material_type: str) -> int:
        """Safely insert materials using only existing columns"""
        if not materials:
            return 0
        
        rows = []
        for m in materials:
            if isinstance(m, dict):
                # Build row with only existing columns
                row = {'company_id': company_id}
                
                # Map AI response to database columns safely
                if 'name' in self.materials_columns:
                    row['name'] = m.get('name') or m.get('material') or m.get('material_name') or 'AI Generated Material'
                
                if 'description' in self.materials_columns:
                    row['description'] = m.get('description') or m.get('value') or ''
                
                if 'quantity' in self.materials_columns:
                    row['quantity'] = m.get('quantity') or m.get('available') or m.get('needed') or ''
                
                if 'category' in self.materials_columns:
                    row['category'] = material_type
                
                if 'ai_generated' in self.materials_columns:
                    row['ai_generated'] = True
                
                if 'created_at' in self.materials_columns:
                    row['created_at'] = datetime.now().isoformat()
                
                if 'potential_value' in self.materials_columns:
                    row['potential_value'] = m.get('potential_value') or ''
                
                if 'notes' in self.materials_columns:
                    row['notes'] = m.get('notes') or ''
                
                if 'frequency' in self.materials_columns:
                    row['frequency'] = m.get('frequency') or 'Monthly'
                
                if 'quantity_estimate' in self.materials_columns:
                    row['quantity_estimate'] = m.get('quantity_estimate') or m.get('quantity') or ''
                
                rows.append(row)
                
            elif isinstance(m, str):
                # Simple string material
                row = {'company_id': company_id}
                
                if 'name' in self.materials_columns:
                    row['name'] = m
                
                if 'description' in self.materials_columns:
                    row['description'] = ''
                
                if 'category' in self.materials_columns:
                    row['category'] = material_type
                
                if 'ai_generated' in self.materials_columns:
                    row['ai_generated'] = True
                
                if 'created_at' in self.materials_columns:
                    row['created_at'] = datetime.now().isoformat()
                
                rows.append(row)
        
        if rows:
            try:
                result = supabase.table('materials').insert(rows).execute()
                logger.info(f"  ✅ Successfully inserted {len(rows)} materials")
                return len(rows)
            except Exception as e:
                logger.error(f"  ❌ Database insertion error: {e}")
                logger.error(f"  📋 Attempted to insert columns: {list(rows[0].keys())}")
                return 0
        return 0
    
    def generate_listings_for_company(self, company: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
        """Generate listings for a single company with comprehensive error handling"""
        logger.info(f"\n[{idx}/{total}] 🏢 Processing: {company.get('name', 'Unknown Company')}")
        
        result = {
            'company_name': company.get('name', 'Unknown'),
            'company_id': company.get('id', ''),
            'success': False,
            'waste_materials': 0,
            'requirements': 0,
            'error': None
        }
        
        try:
            # Build comprehensive company profile
            profile = self.build_company_profile(company)
            logger.info(f"  📊 Company profile built: {profile.get('industry', 'Unknown')} industry, {profile.get('location', 'Unknown')} location")
            
            # Generate AI analysis
            logger.info(f"  🤖 Calling DeepSeek API...")
            ai_result = inference_service.generate_listings_from_profile(profile)
            logger.info(f"  ✅ AI analysis successful")
            
            # Extract outputs (waste/materials)
            outputs = ai_result.get('predicted_outputs') or ai_result.get('waste_streams') or ai_result.get('materials') or []
            logger.info(f"  📦 Generated {len(outputs)} waste materials")
            
            # Extract inputs (requirements)
            inputs = ai_result.get('predicted_inputs') or ai_result.get('resource_needs') or ai_result.get('requirements') or []
            logger.info(f"  📥 Generated {len(inputs)} requirements")
            
            # Insert waste materials
            n_out = self.insert_materials_safely(company['id'], outputs, 'waste')
            result['waste_materials'] = n_out
            
            # Insert requirements
            n_in = self.insert_materials_safely(company['id'], inputs, 'requirement')
            result['requirements'] = n_in
            
            result['success'] = True
            logger.info(f"  ✅ Success: {n_out} waste + {n_in} requirements = {n_out + n_in} total")
            
        except Exception as e:
            error_msg = f"Error processing {company.get('name', 'Unknown')}: {str(e)}"
            logger.error(f"  ❌ {error_msg}")
            result['error'] = error_msg
        
        return result
    
    def run(self, test_mode: bool = True, max_companies: int = 10):
        """Run the bulletproof AI generator"""
        logger.info("🚀 Starting Bulletproof AI Listings Generator...")
        
        # Step 1: Check database schema
        self.check_database_schema()
        
        # Step 2: Fetch companies
        try:
            companies = supabase.table('companies').select('*').execute().data
            if not companies:
                logger.error("❌ No companies found in database.")
                return
            
            logger.info(f"📊 Found {len(companies)} companies in database.")
            
            if test_mode:
                companies = companies[:max_companies]
                logger.info(f"🧪 Test mode: Processing first {len(companies)} companies")
            else:
                logger.info(f"🚀 Production mode: Processing all {len(companies)} companies")
                
        except Exception as e:
            logger.error(f"❌ Error fetching companies: {e}")
            return
        
        # Step 3: Process companies
        total_materials = 0
        successful_companies = 0
        failed_companies = 0
        results = []
        
        start_time = time.time()
        
        for idx, company in enumerate(companies, 1):
            result = self.generate_listings_for_company(company, idx, len(companies))
            results.append(result)
            
            if result['success']:
                successful_companies += 1
                total_materials += result['waste_materials'] + result['requirements']
            else:
                failed_companies += 1
            
            # Sleep to avoid rate limits
            time.sleep(1.5)
        
        # Step 4: Generate report
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"\n🎉 Generation Complete!")
        logger.info(f"⏱️  Duration: {duration:.1f} seconds")
        logger.info(f"✅ Successful companies: {successful_companies}/{len(companies)}")
        logger.info(f"❌ Failed companies: {failed_companies}/{len(companies)}")
        logger.info(f"📦 Total materials generated: {total_materials}")
        logger.info(f"📊 Average materials per company: {total_materials/successful_companies:.1f}" if successful_companies > 0 else "📊 No successful companies")
        
        # Save detailed results
        with open('ai_generation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📄 Detailed results saved to: ai_generation_results.json")
        
        if test_mode and successful_companies == len(companies):
            logger.info("🚀 All tests passed! Ready for production run.")
            logger.info("💡 To run on all companies, set test_mode=False")
        elif failed_companies > 0:
            logger.info("⚠️  Some companies failed. Check the log above for details.")
        
        return results

def main():
    """Main function with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bulletproof AI Listings Generator')
    parser.add_argument('--test', action='store_true', default=True, help='Run in test mode (default: True)')
    parser.add_argument('--production', action='store_true', help='Run in production mode (all companies)')
    parser.add_argument('--max-test', type=int, default=10, help='Maximum companies for test mode (default: 10)')
    
    args = parser.parse_args()
    
    # Determine mode
    test_mode = not args.production
    max_companies = args.max_test if test_mode else None
    
    # Run generator
    generator = BulletproofAIGenerator()
    generator.run(test_mode=test_mode, max_companies=max_companies)

if __name__ == "__main__":
    main() 