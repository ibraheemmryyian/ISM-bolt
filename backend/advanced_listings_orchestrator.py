import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from supabase import create_client, Client
# from .dynamic_materials_integration_service import DynamicMaterialsIntegrationService
# from .advanced_ai_prompts_service import AdvancedAIPromptsService

# Fallback implementations to prevent import errors
class DynamicMaterialsIntegrationService:
    def __init__(self, *args, **kwargs):
        pass

class AdvancedAIPromptsService:
    def __init__(self, *args, **kwargs):
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvancedListingsOrchestrator")

MATERIALSBERT_ENDPOINT = os.environ.get('MATERIALSBERT_ENDPOINT', 'http://localhost:8001')

class AdvancedListingsOrchestrator:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError("Missing Supabase credentials")
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.materials_service = DynamicMaterialsIntegrationService()
        self.advanced_ai_service = AdvancedAIPromptsService()

    async def enrich_material(self, company: Dict[str, Any], material_name: str, material_type: str) -> Optional[Dict[str, Any]]:
        context = {
            'industry': company.get('industry'),
            'location': company.get('location'),
            'company_size': company.get('employee_count', 0),
            'type': material_type
        }
        try:
            # 1. Gather core material data from all sources
            material_data = await self.materials_service.get_comprehensive_material_data(material_name, context)

            # 2. Call MaterialsBERT for semantic/embedding analysis
            bert_payload = {
                'material': material_name,
                'context': context
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{MATERIALSBERT_ENDPOINT}/analyze", json=bert_payload, timeout=30) as resp:
                    bert_result = await resp.json() if resp.status == 200 else {}
            bert_analysis = bert_result.get('analysis', {})

            # 3. Call AdvancedAIPromptsService for DeepSeek-powered reasoning/description
            strategic_result = self.advanced_ai_service.strategic_material_analysis(company)
            # Use the first predicted output that matches material_name, if available
            ds_listing = None
            for out in strategic_result.get('predicted_outputs', []):
                if material_name.lower() in out.get('name', '').lower():
                    ds_listing = out
                    break
            # Fallback: just use the first output
            if not ds_listing and strategic_result.get('predicted_outputs'):
                ds_listing = strategic_result['predicted_outputs'][0]

            # 4. Aggregate all results into Supabase schema
            now = datetime.utcnow().isoformat()
            listing = {
                'id': None,  # Let Supabase autogenerate
                'company_id': company.get('id'),
                'material_name': material_name,
                'quantity': ds_listing.get('quantity', {}).get('value') if ds_listing else None,
                'unit': ds_listing.get('quantity', {}).get('unit') if ds_listing else None,
                'description': ds_listing.get('description') if ds_listing else material_data.description if hasattr(material_data, 'description') else None,
                'type': material_type,
                'created_at': now,
                'ai_tags': [material_type, company.get('industry', ''), 'ai_generated'],
                'estimated_value': ds_listing.get('value_breakdown', {}).get('potential_market_value') if ds_listing else None,
                'priority_score': None,
                'is_sponsored': False,
                'embeddings': bert_analysis.get('embeddings'),
                'ai_generated': True,
                'availability': 'Available' if material_type == 'waste' else 'Needed',
                'location': company.get('location'),
                'price_per_unit': material_data.market_price if hasattr(material_data, 'market_price') else None,
                'current_cost': None,
                'potential_sources': bert_analysis.get('sources'),
                'updated_at': now,
                'category': material_data.category if hasattr(material_data, 'category') else None,
                'status': 'active',
                'material_properties': bert_analysis.get('material_properties'),
                'shipping_params': None,  # Could be filled from Freightos if needed
                'sustainability_metrics': bert_analysis.get('sustainability_metrics'),
            }
            return listing
        except Exception as e:
            logger.error(f"Error enriching material {material_name}: {e}")
            return None

    async def process_company(self, company: Dict[str, Any]) -> List[Dict[str, Any]]:
        listings = []
        materials = company.get('materials', [])
        waste_streams = company.get('waste_streams', [])
        tasks = []
        for material in materials:
            if material and material.strip():
                tasks.append(self.enrich_material(company, material, 'raw_material'))
        for waste in waste_streams:
            if waste and waste.strip():
                tasks.append(self.enrich_material(company, waste, 'waste'))
        results = await asyncio.gather(*tasks)
        for listing in results:
            if listing:
                listings.append(listing)
        return listings

    def save_listings_to_database(self, listings: List[Dict[str, Any]]):
        for listing in listings:
            try:
                # Remove None fields for Supabase
                clean_listing = {k: v for k, v in listing.items() if v is not None}
                self.supabase.table('materials').insert(clean_listing).execute()
                logger.info(f"Inserted listing for {listing['company_id']} - {listing['material_name']}")
            except Exception as e:
                logger.error(f"Error inserting listing: {e}")

    async def generate_all_listings(self):
        companies = self.supabase.table('companies').select('*').execute().data
        if not companies:
            logger.error("No companies found in Supabase.")
            return
        all_listings = []
        for company in companies:
            logger.info(f"Processing company: {company.get('name', 'Unknown')}")
            listings = await self.process_company(company)
            self.save_listings_to_database(listings)
            all_listings.extend(listings)
        logger.info(f"Generated {len(all_listings)} listings in total.")
        return all_listings

# CLI entry point
if __name__ == "__main__":
    orchestrator = AdvancedListingsOrchestrator()
    asyncio.run(orchestrator.generate_all_listings()) 