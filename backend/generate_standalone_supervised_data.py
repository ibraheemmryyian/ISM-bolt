#!/usr/bin/env python3
"""
Standalone supervised data generator - NO EXTERNAL DEPENDENCIES
Generates material listings and matches for all companies
"""
import json
import csv
import os
import sys
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
DATA_FILE = Path(__file__).parent.parent / "fixed_realworlddata.json"
LISTINGS_CSV = "material_listings_standalone.csv"
MATCHES_CSV = "material_matches_standalone.csv"

class StandaloneDataGenerator:
    """Standalone data generator with no external dependencies"""
    
    def __init__(self):
        self.logger = logger
        
        # Material database
        self.materials_database = {
            'metals': [
                'Steel', 'Aluminum', 'Copper', 'Iron', 'Zinc', 'Nickel', 'Titanium', 'Chromium', 'Manganese', 'Cobalt',
                'Tungsten', 'Molybdenum', 'Vanadium', 'Niobium', 'Tantalum', 'Beryllium', 'Magnesium', 'Lead', 'Tin'
            ],
            'polymers': [
                'Polyethylene', 'Polypropylene', 'PVC', 'PET', 'Polystyrene', 'ABS', 'Nylon', 'Polycarbonate',
                'Polyurethane', 'Silicone', 'Teflon', 'Acrylic', 'Polyester', 'Epoxy', 'Phenolic', 'Urea-formaldehyde'
            ],
            'ceramics': [
                'Silicon Carbide', 'Aluminum Oxide', 'Zirconia', 'Silica', 'Boron Nitride', 'Titanium Dioxide',
                'Calcium Carbonate', 'Barium Titanate', 'Lead Zirconate Titanate', 'Ferrite'
            ],
            'composites': [
                'Carbon Fiber', 'Glass Fiber', 'Kevlar', 'Boron Fiber', 'Aramid Fiber', 'Basalt Fiber',
                'Natural Fiber', 'Metal Matrix Composite', 'Ceramic Matrix Composite', 'Polymer Matrix Composite'
            ],
            'semiconductors': [
                'Silicon', 'Germanium', 'Gallium Arsenide', 'Indium Phosphide', 'Cadmium Telluride',
                'Copper Indium Gallium Selenide', 'Perovskite', 'Graphene', 'Carbon Nanotubes', 'Quantum Dots'
            ],
            'building_materials': [
                'Concrete', 'Cement', 'Bricks', 'Steel Beams', 'Wood', 'Glass', 'Insulation', 'Drywall',
                'Roofing Materials', 'Flooring Materials', 'Paint', 'Adhesives', 'Sealants'
            ],
            'natural_materials': [
                'Wood', 'Bamboo', 'Cork', 'Natural Rubber', 'Cotton', 'Wool', 'Silk', 'Leather',
                'Stone', 'Clay', 'Sand', 'Gravel', 'Limestone', 'Marble', 'Granite'
            ]
        }
        
        # Industry-material compatibility
        self.industry_materials = {
            'manufacturing': ['metals', 'polymers', 'ceramics', 'composites'],
            'construction': ['metals', 'building_materials', 'natural_materials', 'ceramics'],
            'electronics': ['semiconductors', 'metals', 'polymers', 'ceramics'],
            'automotive': ['metals', 'polymers', 'composites', 'natural_materials'],
            'aerospace': ['metals', 'composites', 'ceramics', 'polymers'],
            'energy': ['metals', 'ceramics', 'semiconductors', 'composites'],
            'medical': ['metals', 'polymers', 'ceramics', 'composites'],
            'textile': ['natural_materials', 'polymers', 'metals'],
            'food': ['natural_materials', 'polymers', 'metals'],
            'chemical': ['polymers', 'metals', 'ceramics', 'natural_materials']
        }
        
        # Material properties for realistic generation
        self.material_properties = {
            'Steel': {'density': 7850, 'unit': 'kg', 'conditions': ['new', 'scrap', 'recycled']},
            'Aluminum': {'density': 2700, 'unit': 'kg', 'conditions': ['new', 'scrap', 'recycled']},
            'Copper': {'density': 8960, 'unit': 'kg', 'conditions': ['new', 'scrap', 'recycled']},
            'Plastic': {'density': 1000, 'unit': 'kg', 'conditions': ['new', 'recycled']},
            'Glass': {'density': 2500, 'unit': 'kg', 'conditions': ['new', 'recycled']},
            'Wood': {'density': 600, 'unit': 'kg', 'conditions': ['new', 'recycled']},
            'Concrete': {'density': 2400, 'unit': 'kg', 'conditions': ['new', 'demolition']},
            'Silicon': {'density': 2330, 'unit': 'kg', 'conditions': ['new', 'recycled']}
        }
    
    def generate_material_listings(self, company: dict) -> List[dict]:
        """Generate material listings for a company"""
        company_id = company.get('id')
        company_name = company.get('name', 'Unknown Company')
        industry = company.get('industry', 'manufacturing').lower()
        location = company.get('location', 'Global')
        
        self.logger.info(f"🚀 Generating material listings for {company_name} ({industry})")
        
        listings = []
        
        # Get compatible material types for this industry
        compatible_types = self.industry_materials.get(industry, ['metals', 'polymers'])
        
        # Generate 3-8 listings per company
        num_listings = random.randint(3, 8)
        
        for _ in range(num_listings):
            # Select random material type and material
            material_type = random.choice(compatible_types)
            material_name = random.choice(self.materials_database[material_type])
            
            # Generate realistic quantities
            if material_type in ['metals', 'polymers']:
                quantity = random.randint(100, 5000)
                unit = 'kg'
            elif material_type in ['building_materials']:
                quantity = random.randint(1000, 10000)
                unit = 'kg'
            elif material_type in ['semiconductors']:
                quantity = random.randint(10, 500)
                unit = 'kg'
            else:
                quantity = random.randint(100, 2000)
                unit = 'kg'
            
            # Select condition
            conditions = self.material_properties.get(material_name, {}).get('conditions', ['new'])
            condition = random.choice(conditions)
            
            # Generate description
            descriptions = [
                f"High-quality {material_name.lower()} available for industrial use",
                f"Premium {material_name.lower()} suitable for {industry} applications",
                f"Reliable {material_name.lower()} supply from established manufacturer",
                f"Certified {material_name.lower()} meeting industry standards",
                f"Bulk {material_name.lower()} supply for large-scale operations"
            ]
            description = random.choice(descriptions)
            
            listing = {
                'company_id': company_id,
                'company_name': company_name,
                'material_name': material_name,
                'material_type': material_type,
                'quantity': quantity,
                'unit': unit,
                'condition': condition,
                'location': location,
                'industry': industry,
                'description': description,
                'generated_at': datetime.now().isoformat(),
                'generation_method': 'standalone_ai',
                'confidence_score': round(random.uniform(0.7, 0.95), 3)
            }
            
            listings.append(listing)
        
        self.logger.info(f"✅ Generated {len(listings)} listings for {company_name}")
        return listings
    
    def generate_matches(self, company_id: str, material: dict, all_companies: List[dict]) -> List[dict]:
        """Generate matches for a material"""
        company_name = material.get('company_name', 'Unknown Company')
        material_name = material.get('material_name', 'Unknown Material')
        material_type = material.get('material_type', 'unknown')
        
        self.logger.info(f"🔗 Generating matches for {material_name} from {company_name}")
        
        matches = []
        
        # Generate candidate companies
        candidate_companies = self._get_candidate_companies(company_id, material_type, all_companies)
        self.logger.info(f"Found {len(candidate_companies)} candidate companies for {material_name} ({material_type})")
        for cand in candidate_companies:
            self.logger.info(f"Candidate: {cand.get('name')} | Industry: {cand.get('industry')}")
        
        for candidate in candidate_companies:
            # Calculate match score using multiple factors
            match_score = self._calculate_match_score(material, candidate)
            self.logger.info(f"Match score for {candidate.get('name')}: {match_score}")
            
            if match_score > 0.1:  # Lowered threshold for debugging
                match = {
                    'source_company_id': company_id,
                    'source_company_name': company_name,
                    'target_company_id': candidate['id'],
                    'target_company_name': candidate['name'],
                    'material_name': material_name,
                    'material_type': material_type,
                    'match_score': round(match_score, 3),
                    'match_type': 'standalone_ai_compatibility',
                    'generated_at': datetime.now().isoformat(),
                    'confidence': self._get_confidence_level(match_score),
                    'compatibility_factors': self._get_compatibility_factors(material, candidate)
                }
                matches.append(match)
        
        self.logger.info(f"✅ Generated {len(matches)} matches for {material_name}")
        return matches
    
    def _get_candidate_companies(self, company_id: str, material_type: str, all_companies: List[dict]) -> List[dict]:
        """Get candidate companies for matching"""
        candidates = []
        
        for company in all_companies:
            if company.get('id') != company_id:
                # Check if company industry is compatible with material type
                if self._is_compatible_industry(company.get('industry', ''), material_type):
                    candidates.append(company)
        
        # Return top 15 candidates
        return candidates[:15]
    
    def _is_compatible_industry(self, industry: str, material_type: str) -> bool:
        """Check if industry is compatible with material type (TEMP: always True for debugging)"""
        # DEBUG: Loosen logic to always return True
        return True
    
    def _calculate_match_score(self, material: dict, candidate: dict) -> float:
        """Calculate match score using multiple factors"""
        score = 0.0
        
        # Industry compatibility (40% weight)
        material_type = material.get('material_type', 'unknown')
        if self._is_compatible_industry(candidate.get('industry', ''), material_type):
            score += 0.4
        
        # Location proximity (20% weight)
        source_location = material.get('location', 'Global')
        target_location = candidate.get('location', 'Global')
        if source_location == target_location:
            score += 0.2
        elif source_location in target_location or target_location in source_location:
            score += 0.1
        
        # Company size compatibility (15% weight)
        source_size = material.get('company_size', 'medium')
        target_size = candidate.get('company_size', 'medium')
        if source_size == target_size:
            score += 0.15
        
        # Material condition preference (15% weight)
        material_condition = material.get('condition', 'new')
        if material_condition == 'new':
            score += 0.15
        elif material_condition == 'recycled':
            score += 0.1
        else:
            score += 0.05
        
        # Random factor to simulate AI uncertainty (10% weight)
        random_factor = random.uniform(0, 0.1)
        score += random_factor
        
        return min(1.0, score)
    
    def _get_confidence_level(self, match_score: float) -> str:
        """Get confidence level based on match score"""
        if match_score >= 0.8:
            return 'high'
        elif match_score >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _get_compatibility_factors(self, material: dict, candidate: dict) -> List[str]:
        """Get list of compatibility factors"""
        factors = []
        
        material_type = material.get('material_type', 'unknown')
        candidate_industry = candidate.get('industry', '').lower()
        
        if self._is_compatible_industry(candidate_industry, material_type):
            factors.append('industry_compatibility')
        
        if material.get('location') == candidate.get('location'):
            factors.append('location_proximity')
        
        if material.get('company_size') == candidate.get('company_size'):
            factors.append('company_size_match')
        
        if material.get('condition') == 'new':
            factors.append('quality_preference')
        
        return factors
    
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

def main():
    """Main function to generate supervised data"""
    logger.info("🚀 Starting Standalone Supervised Data Generation")
    
    # Check if data file exists
    if not DATA_FILE.exists():
        logger.error(f"❌ Data file not found: {DATA_FILE}")
        logger.error("🔧 Creating sample data for demonstration...")
        
        # Create sample data
        sample_companies = [
            {'id': '1', 'name': 'SteelCorp Manufacturing', 'industry': 'manufacturing', 'location': 'Texas, USA', 'company_size': 'large'},
            {'id': '2', 'name': 'GreenTech Electronics', 'industry': 'electronics', 'location': 'California, USA', 'company_size': 'medium'},
            {'id': '3', 'name': 'BuildRight Construction', 'industry': 'construction', 'location': 'New York, USA', 'company_size': 'large'},
            {'id': '4', 'name': 'AutoParts Inc', 'industry': 'automotive', 'location': 'Michigan, USA', 'company_size': 'medium'},
            {'id': '5', 'name': 'ChemSolutions', 'industry': 'chemical', 'location': 'Texas, USA', 'company_size': 'large'},
            {'id': '6', 'name': 'AeroSpace Dynamics', 'industry': 'aerospace', 'location': 'Washington, USA', 'company_size': 'large'},
            {'id': '7', 'name': 'EnergyFlow Power', 'industry': 'energy', 'location': 'California, USA', 'company_size': 'medium'},
            {'id': '8', 'name': 'MedTech Innovations', 'industry': 'medical', 'location': 'Massachusetts, USA', 'company_size': 'medium'},
            {'id': '9', 'name': 'TextileWorld', 'industry': 'textile', 'location': 'North Carolina, USA', 'company_size': 'medium'},
            {'id': '10', 'name': 'FoodQuality Corp', 'industry': 'food', 'location': 'Illinois, USA', 'company_size': 'large'}
        ]
        
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(sample_companies, f, indent=2)
        
        logger.info(f"✅ Created sample data with {len(sample_companies)} companies")
    
    # Load company data
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            companies = json.load(f)
        logger.info(f"✅ Loaded {len(companies)} companies from {DATA_FILE}")
    except Exception as e:
        logger.error(f"❌ Failed to load company data: {e}")
        return
    
    # Initialize generator
    generator = StandaloneDataGenerator()
    
    try:
        all_listings = []
        all_matches = []
        
        # Process each company
        for i, company in enumerate(companies):
            logger.info(f"📊 Processing company {i+1}/{len(companies)}: {company.get('name', 'Unknown')}")
            
            # Generate listings
            listings = generator.generate_material_listings(company)
            all_listings.extend(listings)
            
            # Generate matches for each listing
            for listing in listings:
                matches = generator.generate_matches(company.get('id'), listing, companies)
                all_matches.extend(matches)
        
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
        
        logger.info("✅ Standalone supervised data generation completed successfully!")
        logger.info(f"📊 Generated {len(all_listings)} material listings")
        logger.info(f"🔗 Generated {len(unique_matches)} material matches")
        logger.info(f"🏢 Processed {len(companies)} companies")
        
        # Show sample of generated data
        if all_listings:
            logger.info("📋 Sample listing:")
            logger.info(f"   Company: {all_listings[0]['company_name']}")
            logger.info(f"   Material: {all_listings[0]['material_name']}")
            logger.info(f"   Quantity: {all_listings[0]['quantity']} {all_listings[0]['unit']}")
        
        if unique_matches:
            logger.info("🔗 Sample match:")
            logger.info(f"   From: {unique_matches[0]['source_company_name']}")
            logger.info(f"   To: {unique_matches[0]['target_company_name']}")
            logger.info(f"   Material: {unique_matches[0]['material_name']}")
            logger.info(f"   Score: {unique_matches[0]['match_score']}")
        
    except Exception as e:
        logger.error(f"❌ Error during data generation: {e}")
        raise

if __name__ == "__main__":
    main() 