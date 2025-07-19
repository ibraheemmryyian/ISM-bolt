# Only the following microservices are required for material generation, listings, and matches:
# - ListingInferenceService: Generates material listings from company data
# - AIListingsGenerator: Generates AI-powered listings
# - RevolutionaryAIMatching: Runs advanced AI matching between companies/materials
# - GNNReasoningEngine: (if used for matching)
# - MultiHopSymbiosisNetwork: (if used for matching)
# - DynamicMaterialsIntegrationService: (if used for listings/matching)
#
# All other services (pricing, production orchestrator, monitoring, retraining, meta-learning, opportunity engine, impact forecasting, etc.) are NOT required and are removed.

import json
import csv
import os
import sys
import asyncio
import aiohttp
import requests
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

DATA_FILE = Path(__file__).parent.parent / "fixed_realworlddata.json"
LISTINGS_CSV = "material_listings.csv"
MATCHES_CSV = "material_matches.csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import only the required microservices for material generation, listings, and matches
try:
    from listing_inference_service import ListingInferenceService
    from ai_listings_generator import AIListingsGenerator
    from revolutionary_ai_matching import RevolutionaryAIMatching
    from gnn_reasoning_engine import GNNReasoningEngine
    from multi_hop_symbiosis_network import MultiHopSymbiosisNetwork
    from dynamic_materials_integration_service import DynamicMaterialsIntegrationService
    
    REQUIRED_SERVICES_AVAILABLE = True
    logger.info("✅ All required microservices imported successfully")
except ImportError as e:
    logger.error(f"❌ CRITICAL ERROR: Failed to import required microservices: {e}")
    logger.error("🔧 REQUIRED SETUP:")
    logger.error("   1. Ensure these services are available:")
    logger.error("      - listing_inference_service.py")
    logger.error("      - ai_listings_generator.py")
    logger.error("      - revolutionary_ai_matching.py")
    logger.error("      - gnn_reasoning_engine.py")
    logger.error("      - multi_hop_symbiosis_network.py")
    logger.error("      - dynamic_materials_integration_service.py")
    logger.error("   2. Install required dependencies")
    logger.error("   3. Check that all service files exist")
    sys.exit(1)

class CleanMaterialDataGenerator:
    """Clean, production-ready material data generator using only essential services"""
    
    def __init__(self):
        self.logger = logger
        self.session = aiohttp.ClientSession()
        
        # Initialize only the required microservices
        self.services = {}
        self._initialize_services()
        
        # Configuration
        self.config = {
            'max_concurrent_requests': 10,
            'timeout': 30,
            'retry_attempts': 3
        }
    
    def _initialize_services(self):
        """Initialize only the required microservices for material generation, listings, and matches"""
        required_services = [
            ('listing_inference', 'ListingInferenceService'),
            ('ai_listings', 'AIListingsGenerator'),
            ('revolutionary_matching', 'RevolutionaryAIMatching'),
            ('gnn_reasoning', 'GNNReasoningEngine'),
            ('multi_hop', 'MultiHopSymbiosisNetwork'),
            ('materials_integration', 'DynamicMaterialsIntegrationService')
        ]
        
        missing_services = []
        
        for service_key, service_class in required_services:
            if service_class in globals():
                try:
                    self.services[service_key] = globals()[service_class]()
                    self.logger.info(f"✅ Initialized {service_key}")
                except Exception as e:
                    missing_services.append(f"{service_key} (initialization failed: {e})")
            else:
                missing_services.append(service_key)
        
        if missing_services:
            self.logger.error(f"❌ CRITICAL ERROR: Missing or failed services: {missing_services}")
            raise RuntimeError(f"Missing or failed services: {missing_services}")
        
        self.logger.info(f"✅ Successfully initialized {len(self.services)} required microservices")
    
    async def generate_material_listings(self, company: dict) -> List[dict]:
        """Generate material listings using only essential services"""
        company_id = company.get('id')
        company_name = company.get('name', 'Unknown Company')
        
        self.logger.info(f"🚀 Generating material listings for {company_name}")
        
        listings = []
        
        # 1. Use listing inference service
        try:
            self.logger.info(f"  📦 Running listing inference service...")
            inference_result = await self.services['listing_inference'].generate_listings_from_profile(company)
            if inference_result and inference_result.get('predicted_outputs'):
                listings.extend(inference_result['predicted_outputs'])
                self.logger.info(f"  ✅ Listing inference generated {len(inference_result['predicted_outputs'])} listings")
        except Exception as e:
            self.logger.warning(f"  ⚠️ Listing inference failed: {e}")
        
        # 2. Use AI listings generator
        try:
            self.logger.info(f"  🤖 Running AI listings generator...")
            ai_result = await self.services['ai_listings'].generate_ai_listings(company)
            if ai_result:
                listings.extend(ai_result)
                self.logger.info(f"  ✅ AI listings generated {len(ai_result)} listings")
        except Exception as e:
            self.logger.warning(f"  ⚠️ AI listings generation failed: {e}")
        
        # 3. Use materials integration service if available
        try:
            self.logger.info(f"  🔬 Running materials integration analysis...")
            materials_analysis = await self.services['materials_integration'].get_comprehensive_material_analysis(
                company.get('industry', 'manufacturing'),
                {'company': company_name, 'location': company.get('location', 'Global')}
            )
            if materials_analysis and materials_analysis.get('materials'):
                listings.extend(materials_analysis['materials'])
                self.logger.info(f"  ✅ Materials integration generated {len(materials_analysis['materials'])} listings")
        except Exception as e:
            self.logger.warning(f"  ⚠️ Materials integration failed: {e}")
        
        # 4. Enhance and standardize listings
        enhanced_listings = []
        for listing in listings:
            enhanced_listing = {
                'company_id': company_id,
                'company_name': company_name,
                'material_name': listing.get('name') or listing.get('material_name') or 'Unknown Material',
                'material_type': listing.get('type') or listing.get('category', 'unknown'),
                'quantity': listing.get('quantity') or listing.get('quantity_estimate', 100),
                'unit': listing.get('unit', 'tons'),
                'description': listing.get('description', ''),
                'quality_grade': listing.get('quality_grade', 'B'),
                'potential_value': listing.get('potential_value', 0),
                'ai_generated': True,
                'generated_at': datetime.now().isoformat()
            }
            enhanced_listings.append(enhanced_listing)
        
        self.logger.info(f"✅ Generated {len(enhanced_listings)} material listings for {company_name}")
        return enhanced_listings
    
    async def generate_material_matches(self, company_id: str, material: dict) -> List[dict]:
        """Generate material matches using only essential services"""
        material_name = material.get('material_name', 'Unknown Material')
        
        self.logger.info(f"🔗 Generating matches for material: {material_name}")
        
        matches = []
        
        # 1. Use revolutionary AI matching
        try:
            self.logger.info(f"  🚀 Running revolutionary AI matching...")
            revolutionary_matches = await self.services['revolutionary_matching'].find_matches(material)
            if revolutionary_matches:
                matches.extend(revolutionary_matches)
                self.logger.info(f"  ✅ Revolutionary matching found {len(revolutionary_matches)} matches")
        except Exception as e:
            self.logger.warning(f"  ⚠️ Revolutionary matching failed: {e}")
        
        # 2. Use GNN reasoning if available
        try:
            self.logger.info(f"  🧠 Running GNN reasoning...")
            gnn_matches = await self.services['gnn_reasoning'].find_gnn_matches(material)
            if gnn_matches:
                matches.extend(gnn_matches)
                self.logger.info(f"  ✅ GNN reasoning found {len(gnn_matches)} matches")
        except Exception as e:
            self.logger.warning(f"  ⚠️ GNN reasoning failed: {e}")
        
        # 3. Use multi-hop symbiosis if available
        try:
            self.logger.info(f"  🌐 Running multi-hop symbiosis...")
            multi_hop_matches = await self.services['multi_hop'].find_symbiosis_matches(material)
            if multi_hop_matches:
                matches.extend(multi_hop_matches)
                self.logger.info(f"  ✅ Multi-hop symbiosis found {len(multi_hop_matches)} matches")
        except Exception as e:
            self.logger.warning(f"  ⚠️ Multi-hop symbiosis failed: {e}")
        
        # 4. Deduplicate and standardize matches
        unique_matches = self._deduplicate_matches(matches)
        
        # 5. Enhance matches with metadata
        enhanced_matches = []
        for match in unique_matches:
            enhanced_match = {
                'source_company_id': company_id,
                'source_material_name': material_name,
                'target_company_id': match.get('company_id'),
                'target_company_name': match.get('company_name', 'Unknown Company'),
                'target_material_name': match.get('material_name'),
                'match_score': match.get('score', 0.5),
                'match_type': match.get('type', 'direct'),
                'potential_value': match.get('potential_value', 0),
                'ai_generated': True,
                'generated_at': datetime.now().isoformat()
            }
            enhanced_matches.append(enhanced_match)
        
        self.logger.info(f"✅ Generated {len(enhanced_matches)} material matches for {material_name}")
        return enhanced_matches
    
    def _deduplicate_matches(self, matches: List[dict]) -> List[dict]:
        """Remove duplicate matches based on company and material combinations"""
        seen = set()
        unique_matches = []
        
        for match in matches:
            key = (match.get('company_id'), match.get('material_name'))
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        return unique_matches
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()

async def main():
    """Main execution function"""
    logger.info("🚀 Starting Clean Material Data Generator")
    
    # Check if data file exists
    if not DATA_FILE.exists():
        logger.error(f"❌ Data file not found: {DATA_FILE}")
        logger.error("Please ensure fixed_realworlddata.json exists in the project root")
        return
    
    # Load company data
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            companies = json.load(f)
        logger.info(f"✅ Loaded {len(companies)} companies from {DATA_FILE}")
    except Exception as e:
        logger.error(f"❌ Failed to load company data: {e}")
        return
    
    # Initialize generator
    generator = CleanMaterialDataGenerator()
    
    try:
        all_listings = []
        all_matches = []
        
        # Process each company
        for i, company in enumerate(companies, 1):
            logger.info(f"📊 Processing company {i}/{len(companies)}: {company.get('name', 'Unknown')}")
            
            try:
                # Generate listings
                listings = await generator.generate_material_listings(company)
                all_listings.extend(listings)
                
                # Generate matches for each listing
                for listing in listings:
                    matches = await generator.generate_material_matches(company.get('id'), listing)
                    all_matches.extend(matches)
                
                logger.info(f"✅ Completed company {i}/{len(companies)}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process company {company.get('name', 'Unknown')}: {e}")
                continue
        
        # Save results to CSV
        logger.info("💾 Saving results to CSV files...")
        
        # Save listings
        with open(LISTINGS_CSV, 'w', newline='', encoding='utf-8') as f:
            if all_listings:
                writer = csv.DictWriter(f, fieldnames=all_listings[0].keys())
                writer.writeheader()
                writer.writerows(all_listings)
        
        # Save matches
        with open(MATCHES_CSV, 'w', newline='', encoding='utf-8') as f:
            if all_matches:
                writer = csv.DictWriter(f, fieldnames=all_matches[0].keys())
                writer.writeheader()
                writer.writerows(all_matches)
        
        logger.info(f"✅ Successfully generated:")
        logger.info(f"   📦 {len(all_listings)} material listings -> {LISTINGS_CSV}")
        logger.info(f"   🔗 {len(all_matches)} material matches -> {MATCHES_CSV}")
        
    finally:
        await generator.close()
    
    logger.info("🎉 Clean Material Data Generator completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 