#!/usr/bin/env python3
"""
Company Data Enrichment Script

Adds estimated quantitative data to waste streams based on company size and industry.
Run this script after importing companies to ensure the AI onboarding process has
realistic waste stream quantities for demonstration purposes.
"""

import json
import os
import sys
import logging
import random
from typing import Dict, List, Any
import aiohttp
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_enrichment')

# Industry-specific waste generation factors (tons per employee per year)
WASTE_GENERATION_FACTORS = {
    'Manufacturing': {
        'base': 4.5,
        'variance': 1.5,
        'streams': {
            'Plastic Scrap': {'min': 0.5, 'max': 2.0},
            'Metal Shavings': {'min': 0.8, 'max': 3.0},
            'Scrap Metal': {'min': 1.0, 'max': 4.0},
            'Cutting Fluids': {'min': 0.2, 'max': 0.8},
            'Dust': {'min': 0.1, 'max': 0.5},
            'Packaging Waste': {'min': 0.3, 'max': 1.2},
            'Cardboard Trimmings': {'min': 0.2, 'max': 0.9},
            'default': {'min': 0.3, 'max': 1.0}
        }
    },
    'Construction': {
        'base': 8.0,
        'variance': 3.0,
        'streams': {
            'Concrete Debris': {'min': 3.0, 'max': 10.0},
            'Wood Scraps': {'min': 1.0, 'max': 4.0},
            'Metal Offcuts': {'min': 0.5, 'max': 2.0},
            'Construction Waste': {'min': 2.0, 'max': 8.0},
            'default': {'min': 1.0, 'max': 3.0}
        }
    },
    'Food & Beverage': {
        'base': 5.0,
        'variance': 2.0,
        'streams': {
            'Food Waste': {'min': 2.0, 'max': 6.0},
            'Organic Matter': {'min': 1.5, 'max': 5.0},
            'Wastewater': {'min': 5.0, 'max': 20.0, 'unit': 'kiloliters'},
            'Packaging Scraps': {'min': 0.5, 'max': 2.0},
            'default': {'min': 0.5, 'max': 2.0}
        }
    },
    'Textiles': {
        'base': 3.0,
        'variance': 1.0,
        'streams': {
            'Fabric Scraps': {'min': 1.0, 'max': 3.0},
            'Thread Waste': {'min': 0.2, 'max': 0.8},
            'Dye Wastewater': {'min': 2.0, 'max': 8.0, 'unit': 'kiloliters'},
            'default': {'min': 0.3, 'max': 1.0}
        }
    },
    'default': {
        'base': 2.0,
        'variance': 0.8,
        'streams': {
            'default': {'min': 0.3, 'max': 1.0}
        }
    }
}

# Units by waste type
DEFAULT_UNITS = {
    'Wastewater': 'kiloliters',
    'Dye Wastewater': 'kiloliters',
    'Process Water': 'kiloliters',
    'default': 'tons'
}

class CompanyDataEnricher:
    """Enriches company data with quantitative waste stream information"""
    
    def __init__(self):
        self.backend_url = os.environ.get('BACKEND_URL', 'http://localhost:3000')
        self.companies = []
    
    async def load_companies(self):
        """Load companies from the API or file"""
        try:
            # Try to load from API first
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.backend_url}/api/companies") as response:
                    if response.status == 200:
                        self.companies = await response.json()
                        logger.info(f"Loaded {len(self.companies)} companies from API")
                        return True
            
            # Fallback to file
            possible_files = [
                "/workspace/data/fixed_realworlddata.json",
                "/workspace/fixed_realworlddata.json",
                "fixed_realworlddata.json",
                "../data/fixed_realworlddata.json"
            ]
            
            for file_path in possible_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.companies = json.load(f)
                        logger.info(f"Loaded {len(self.companies)} companies from {file_path}")
                        return True
            
            logger.error("No companies found via API or file")
            return False
        except Exception as e:
            logger.error(f"Error loading companies: {e}")
            return False
    
    def _calculate_waste_quantity(self, company: Dict[str, Any], waste_stream: str) -> Dict[str, Any]:
        """Calculate estimated waste quantity based on industry and company size"""
        industry = company.get('industry', 'default')
        employee_count = company.get('employee_count', 100)
        
        # Get industry factors or default
        industry_factors = WASTE_GENERATION_FACTORS.get(industry, WASTE_GENERATION_FACTORS['default'])
        
        # Get stream-specific factors or default for this industry
        stream_factors = industry_factors['streams'].get(waste_stream, industry_factors['streams']['default'])
        
        # Calculate base quantity: employees × base factor × random variance
        base_factor = industry_factors['base']
        variance = industry_factors['variance']
        base_waste = employee_count * (base_factor + random.uniform(-variance, variance)) / 10  # Yearly to monthly
        
        # Apply stream-specific adjustment
        min_factor = stream_factors['min']
        max_factor = stream_factors['max']
        stream_factor = random.uniform(min_factor, max_factor)
        quantity = base_waste * stream_factor
        
        # Round to reasonable precision
        if quantity >= 100:
            quantity = round(quantity)
        elif quantity >= 10:
            quantity = round(quantity, 1)
        else:
            quantity = round(quantity, 2)
            
        # Determine unit
        unit = stream_factors.get('unit', DEFAULT_UNITS.get(waste_stream, DEFAULT_UNITS['default']))
        
        return {
            'quantity': quantity,
            'unit': unit
        }
    
    def enrich_company_data(self, company: Dict[str, Any]) -> Dict[str, Any]:
        """Add quantitative data to company waste streams"""
        enriched = dict(company)
        enriched_waste_streams = []
        
        # Process each waste stream
        for waste_stream in company.get('waste_streams', []):
            waste_data = {
                'name': waste_stream,
                'quantity': None,
                'unit': None
            }
            
            # Calculate quantity and unit
            quantity_data = self._calculate_waste_quantity(company, waste_stream)
            waste_data['quantity'] = quantity_data['quantity']
            waste_data['unit'] = quantity_data['unit']
            
            enriched_waste_streams.append(waste_data)
        
        # Replace simple waste stream list with enriched data
        enriched['waste_streams'] = enriched_waste_streams
        return enriched
    
    async def update_company(self, company_id: str, updated_data: Dict[str, Any]):
        """Update company data via API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    f"{self.backend_url}/api/companies/{company_id}",
                    json=updated_data
                ) as response:
                    if response.status == 200:
                        logger.info(f"Updated company {company_id}")
                        return True
                    else:
                        logger.error(f"Failed to update company {company_id}: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error updating company {company_id}: {e}")
            return False
    
    async def process_all_companies(self):
        """Process and update all companies"""
        if not self.companies:
            logger.error("No companies loaded")
            return False
        
        success_count = 0
        for company in self.companies:
            company_id = company.get('id')
            company_name = company.get('name', 'Unknown')
            
            try:
                # Enrich company data
                enriched_data = self.enrich_company_data(company)
                
                # Save to disk for demonstration/backup
                os.makedirs('/workspace/data/enriched', exist_ok=True)
                with open(f'/workspace/data/enriched/{company_name.replace(" ", "_")}.json', 'w', encoding='utf-8') as f:
                    json.dump(enriched_data, f, indent=2)
                
                # Update in system if API available
                if company_id:
                    if await self.update_company(company_id, enriched_data):
                        success_count += 1
                else:
                    logger.warning(f"No ID for company {company_name}, saved to disk only")
                    success_count += 1
            except Exception as e:
                logger.error(f"Error processing company {company_name}: {e}")
        
        logger.info(f"Successfully processed {success_count}/{len(self.companies)} companies")
        return success_count > 0
    
    async def save_example_enriched_data(self):
        """Save example enriched data for demonstration"""
        if not self.companies:
            logger.warning("No companies loaded, creating example data only")
            # Create example company
            example_company = {
                "name": "Example Manufacturing",
                "industry": "Manufacturing",
                "location": "Example City",
                "employee_count": 120,
                "waste_streams": ["Plastic Scrap", "Metal Shavings", "Packaging Waste"],
                "products": ["Example Products"],
                "main_materials": ["Raw Materials"]
            }
            self.companies = [example_company]
        
        # Create enriched directory
        os.makedirs('/workspace/data/enriched', exist_ok=True)
        
        # Process each company
        for company in self.companies:
            enriched_data = self.enrich_company_data(company)
            company_name = company.get('name', 'Unknown').replace(" ", "_")
            
            # Save individual company file
            with open(f'/workspace/data/enriched/{company_name}.json', 'w', encoding='utf-8') as f:
                json.dump(enriched_data, f, indent=2)
            
            # Add material listings with quantitative data
            material_listings = self._generate_material_listings(enriched_data)
            with open(f'/workspace/data/enriched/{company_name}_listings.json', 'w', encoding='utf-8') as f:
                json.dump(material_listings, f, indent=2)
        
        # Save all enriched data in one file
        all_enriched = [self.enrich_company_data(company) for company in self.companies]
        with open('/workspace/data/enriched/all_companies_enriched.json', 'w', encoding='utf-8') as f:
            json.dump(all_enriched, f, indent=2)
        
        logger.info(f"✅ Created enriched data examples for {len(self.companies)} companies")
        return True
    
    def _generate_material_listings(self, company: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate material listings for demonstration"""
        listings = []
        
        # Convert waste streams to waste listings
        for waste in company.get('waste_streams', []):
            if isinstance(waste, dict):
                listings.append({
                    "name": waste['name'],
                    "type": "waste",
                    "quantity": waste['quantity'],
                    "unit": waste['unit'],
                    "description": f"{waste['name']} generated from {company['name']}'s operations",
                    "ai_generated": True,
                    "confidence_score": random.uniform(0.7, 0.95)
                })
            else:
                # Fallback for string waste streams
                quantity_data = self._calculate_waste_quantity(company, waste)
                listings.append({
                    "name": waste,
                    "type": "waste",
                    "quantity": quantity_data['quantity'],
                    "unit": quantity_data['unit'],
                    "description": f"{waste} generated from {company['name']}'s operations",
                    "ai_generated": True,
                    "confidence_score": random.uniform(0.7, 0.95)
                })
        
        # Generate requirement listings based on main materials
        for material in company.get('main_materials', []):
            quantity = random.randint(5, 500) * 10  # Some reasonable quantity
            listings.append({
                "name": material,
                "type": "requirement",
                "quantity": quantity,
                "unit": "tons",
                "description": f"{material} needed for {company['name']}'s production processes",
                "ai_generated": True,
                "confidence_score": random.uniform(0.75, 0.98)
            })
        
        return listings

async def main():
    """Main function"""
    enricher = CompanyDataEnricher()
    
    # Load companies
    if not await enricher.load_companies():
        logger.error("Failed to load companies. Exiting.")
        return 1
    
    # Process all companies
    if not await enricher.process_all_companies():
        logger.warning("Failed to process all companies. Creating example data.")
    
    # Save example data for demonstration
    if not await enricher.save_example_enriched_data():
        logger.error("Failed to save example data.")
        return 1
    
    logger.info("✅ Enrichment process completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))