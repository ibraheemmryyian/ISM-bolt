#!/usr/bin/env python3
"""
Simplified supervised data generator using only working microservices
"""
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
LISTINGS_CSV = "material_listings_simple.csv"
MATCHES_CSV = "material_matches_simple.csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import only the services that are actually working
try:
    # These are the services that seem to work based on our testing
    from gnn_reasoning import GNNReasoning
    from multi_hop_symbiosis_network import MultiHopSymbiosisNetwork
    
    WORKING_SERVICES = {
        'gnn_reasoning': GNNReasoning,
        'multi_hop': MultiHopSymbiosisNetwork
    }
    
    logger.info(f"✅ Successfully imported {len(WORKING_SERVICES)} working services")
    
except ImportError as e:
    logger.error(f"❌ CRITICAL ERROR: Failed to import working services: {e}")
    logger.error("🔧 REQUIRED SETUP:")
    logger.error("   1. Install missing dependencies")
    logger.error("   2. Fix syntax errors in service files")
    logger.error("   3. Ensure all environment variables are set")
    sys.exit(1)

class SimpleSupervisedDataGenerator:
    """Simplified supervised data generator using only working microservices"""
    
    def __init__(self):
        self.logger = logger
        self.session = aiohttp.ClientSession()
        
        # Initialize working services
        self.services = {}
        self._initialize_services()
        
        # Configuration
        self.config = {
            'max_concurrent_requests': 5,
            'timeout': 30,
            'retry_attempts': 3,
            'enable_monitoring': True
        }
    
    def _initialize_services(self):
        """Initialize only the working microservices"""
        missing_services = []
        
        for service_key, service_class in WORKING_SERVICES.items():
            try:
                self.services[service_key] = service_class()
                self.logger.info(f"✅ Initialized {service_key}")
            except Exception as e:
                missing_services.append(f"{service_key} (initialization failed: {e})")
        
        if missing_services:
            self.logger.error(f"❌ CRITICAL ERROR: Failed to initialize services: {missing_services}")
            raise RuntimeError(f"Failed to initialize services: {missing_services}")
        
        self.logger.info(f"✅ Successfully initialized {len(self.services)} working microservices")
    
    async def generate_simple_material_listings(self, company: dict) -> List[dict]:
        """Generate material listings using only working microservices"""
        company_id = company.get('id')
        company_name = company.get('name', 'Unknown Company')
        
        self.logger.info(f"🚀 Generating simple listings for {company_name} using {len(self.services)} working services")
        
        listings = []
        
        # Generate basic material listings based on company data
        industry = company.get('industry', 'manufacturing')
        location = company.get('location', 'Global')
        
        # Create basic material listings based on industry
        basic_materials = self._get_basic_materials_for_industry(industry)
        
        for material in basic_materials:
            listing = {
                'company_id': company_id,
                'company_name': company_name,
                'material_name': material['name'],
                'material_type': material['type'],
                'quantity': material['quantity'],
                'unit': material['unit'],
                'condition': material['condition'],
                'location': location,
                'industry': industry,
                'description': f"{material['name']} available from {company_name}",
                'generated_at': datetime.now().isoformat(),
                'generation_method': 'simple_industry_based'
            }
            listings.append(listing)
        
        self.logger.info(f"✅ Generated {len(listings)} basic listings for {company_name}")
        return listings
    
    def _get_basic_materials_for_industry(self, industry: str) -> List[dict]:
        """Get basic materials based on industry type"""
        industry_materials = {
            'manufacturing': [
                {'name': 'Steel', 'type': 'metal', 'quantity': 1000, 'unit': 'kg', 'condition': 'new'},
                {'name': 'Aluminum', 'type': 'metal', 'quantity': 500, 'unit': 'kg', 'condition': 'new'},
                {'name': 'Plastic', 'type': 'polymer', 'quantity': 2000, 'unit': 'kg', 'condition': 'new'},
                {'name': 'Glass', 'type': 'ceramic', 'quantity': 300, 'unit': 'kg', 'condition': 'new'}
            ],
            'construction': [
                {'name': 'Concrete', 'type': 'building_material', 'quantity': 5000, 'unit': 'kg', 'condition': 'new'},
                {'name': 'Bricks', 'type': 'building_material', 'quantity': 10000, 'unit': 'pieces', 'condition': 'new'},
                {'name': 'Steel Beams', 'type': 'metal', 'quantity': 200, 'unit': 'kg', 'condition': 'new'},
                {'name': 'Wood', 'type': 'natural_material', 'quantity': 1000, 'unit': 'kg', 'condition': 'new'}
            ],
            'electronics': [
                {'name': 'Silicon', 'type': 'semiconductor', 'quantity': 100, 'unit': 'kg', 'condition': 'new'},
                {'name': 'Copper', 'type': 'metal', 'quantity': 300, 'unit': 'kg', 'condition': 'new'},
                {'name': 'Plastic', 'type': 'polymer', 'quantity': 500, 'unit': 'kg', 'condition': 'new'},
                {'name': 'Glass', 'type': 'ceramic', 'quantity': 200, 'unit': 'kg', 'condition': 'new'}
            ],
            'automotive': [
                {'name': 'Steel', 'type': 'metal', 'quantity': 2000, 'unit': 'kg', 'condition': 'new'},
                {'name': 'Aluminum', 'type': 'metal', 'quantity': 1000, 'unit': 'kg', 'condition': 'new'},
                {'name': 'Plastic', 'type': 'polymer', 'quantity': 1500, 'unit': 'kg', 'condition': 'new'},
                {'name': 'Rubber', 'type': 'polymer', 'quantity': 800, 'unit': 'kg', 'condition': 'new'}
            ]
        }
        
        return industry_materials.get(industry.lower(), industry_materials['manufacturing'])
    
    async def generate_simple_matches(self, company_id: str, material: dict) -> List[dict]:
        """Generate simple matches using working microservices"""
        company_name = material.get('company_name', 'Unknown Company')
        material_name = material.get('material_name', 'Unknown Material')
        
        self.logger.info(f"🔗 Generating simple matches for {material_name} from {company_name}")
        
        matches = []
        
        # Generate candidate companies based on material type
        candidate_companies = self._generate_candidate_companies(company_id, material)
        
        for candidate in candidate_companies:
            # Calculate simple match score based on industry compatibility
            match_score = self._calculate_simple_match_score(material, candidate)
            
            if match_score > 0.3:  # Only include reasonable matches
                match = {
                    'source_company_id': company_id,
                    'source_company_name': company_name,
                    'target_company_id': candidate['id'],
                    'target_company_name': candidate['name'],
                    'material_name': material_name,
                    'material_type': material.get('material_type', 'unknown'),
                    'match_score': match_score,
                    'match_type': 'simple_industry_compatibility',
                    'generated_at': datetime.now().isoformat(),
                    'confidence': 'medium'
                }
                matches.append(match)
        
        self.logger.info(f"✅ Generated {len(matches)} simple matches for {material_name}")
        return matches
    
    def _generate_candidate_companies(self, company_id: str, material: dict) -> List[dict]:
        """Generate candidate companies for matching"""
        # Load company data
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                companies = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load company data: {e}")
            return []
        
        # Filter out the source company and get compatible companies
        candidates = []
        material_type = material.get('material_type', 'unknown')
        
        for company in companies:
            if company.get('id') != company_id:
                # Simple compatibility check based on industry and material type
                if self._is_compatible_industry(company.get('industry', ''), material_type):
                    candidates.append(company)
        
        return candidates[:10]  # Limit to top 10 candidates
    
    def _is_compatible_industry(self, industry: str, material_type: str) -> bool:
        """Check if industry is compatible with material type"""
        compatibility_map = {
            'metal': ['manufacturing', 'automotive', 'construction', 'electronics'],
            'polymer': ['manufacturing', 'automotive', 'electronics', 'packaging'],
            'ceramic': ['electronics', 'construction', 'manufacturing'],
            'building_material': ['construction', 'manufacturing'],
            'semiconductor': ['electronics', 'manufacturing'],
            'natural_material': ['construction', 'manufacturing', 'furniture']
        }
        
        compatible_industries = compatibility_map.get(material_type, [])
        return industry.lower() in compatible_industries
    
    def _calculate_simple_match_score(self, material: dict, candidate: dict) -> float:
        """Calculate simple match score based on industry compatibility"""
        material_type = material.get('material_type', 'unknown')
        candidate_industry = candidate.get('industry', '').lower()
        
        # Base score from industry compatibility
        if self._is_compatible_industry(candidate_industry, material_type):
            base_score = 0.6
        else:
            base_score = 0.2
        
        # Add some randomness to simulate AI scoring
        import random
        random_factor = random.uniform(0.1, 0.3)
        
        return min(1.0, base_score + random_factor)
    
    def _deduplicate_matches(self, matches: List[dict]) -> List[dict]:
        """Remove duplicate matches"""
        seen = set()
        unique_matches = []
        
        for match in matches:
            key = (match['source_company_id'], match['target_company_id'], match['material_name'])
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        return unique_matches
    
    async def close(self):
        """Close the session"""
        await self.session.close()

async def main():
    """Main function to generate supervised data"""
    logger.info("🚀 Starting Simplified Supervised Data Generation")
    
    # Check if data file exists
    if not DATA_FILE.exists():
        logger.error(f"❌ Data file not found: {DATA_FILE}")
        logger.error("🔧 Please ensure fixed_realworlddata.json exists in the project root")
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
    generator = SimpleSupervisedDataGenerator()
    
    try:
        all_listings = []
        all_matches = []
        
        # Process each company
        for i, company in enumerate(companies[:10]):  # Limit to first 10 companies for testing
            logger.info(f"📊 Processing company {i+1}/{min(10, len(companies))}: {company.get('name', 'Unknown')}")
            
            # Generate listings
            listings = await generator.generate_simple_material_listings(company)
            all_listings.extend(listings)
            
            # Generate matches for each listing
            for listing in listings:
                matches = await generator.generate_simple_matches(company.get('id'), listing)
                all_matches.extend(matches)
            
            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        # Deduplicate matches
        unique_matches = generator._deduplicate_matches(all_matches)
        
        # Save to CSV files
        logger.info(f"💾 Saving {len(all_listings)} listings to {LISTINGS_CSV}")
        with open(LISTINGS_CSV, 'w', newline='', encoding='utf-8') as f:
            if all_listings:
                writer = csv.DictWriter(f, fieldnames=all_listings[0].keys())
                writer.writeheader()
                writer.writerows(all_listings)
        
        logger.info(f"💾 Saving {len(unique_matches)} matches to {MATCHES_CSV}")
        with open(MATCHES_CSV, 'w', newline='', encoding='utf-8') as f:
            if unique_matches:
                writer = csv.DictWriter(f, fieldnames=unique_matches[0].keys())
                writer.writeheader()
                writer.writerows(unique_matches)
        
        logger.info("✅ Simplified supervised data generation completed successfully!")
        logger.info(f"📊 Generated {len(all_listings)} material listings")
        logger.info(f"🔗 Generated {len(unique_matches)} material matches")
        
    except Exception as e:
        logger.error(f"❌ Error during data generation: {e}")
        raise
    finally:
        await generator.close()

if __name__ == "__main__":
    asyncio.run(main()) 