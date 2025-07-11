#!/usr/bin/env python3
"""
AI-Powered Material Listings Generator
Generates intelligent material listings for companies based on their profiles
"""

import os
import json
import requests
import logging
from typing import List, Dict, Any
from datetime import datetime
import openai
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIListingsGenerator:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError("Missing Supabase credentials")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.use_openai = True
        else:
            logger.warning("OpenAI API key not found, using fallback generation")
            self.use_openai = False
    
    def get_all_companies(self) -> List[Dict[str, Any]]:
        """Fetch all companies from the database"""
        try:
            response = self.supabase.table('companies').select('*').execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching companies: {e}")
            return []
    
    def generate_listing_description(self, company: Dict[str, Any], material: str) -> str:
        """Generate a compelling listing description using AI"""
        
        if self.use_openai:
            try:
                prompt = f"""
                Generate a professional material exchange listing for a company with the following profile:
                
                Company: {company.get('name', 'Unknown')}
                Industry: {company.get('industry', 'Unknown')}
                Location: {company.get('location', 'Unknown')}
                Materials: {company.get('materials', [])}
                Products: {company.get('products', [])}
                Waste Streams: {company.get('waste_streams', [])}
                Sustainability Score: {company.get('sustainability_score', 0)}
                
                Material Available: {material}
                
                Create a compelling listing description that includes:
                1. Material quality and specifications
                2. Available quantity and frequency
                3. Environmental benefits of the exchange
                4. Logistics and delivery options
                5. Contact information and terms
                
                Make it professional, detailed, and attractive to potential partners.
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert in industrial symbiosis and circular economy. Create compelling material exchange listings."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return self._generate_fallback_description(company, material)
        else:
            return self._generate_fallback_description(company, material)
    
    def _generate_fallback_description(self, company: Dict[str, Any], material: str) -> str:
        """Fallback description generation without AI"""
        company_name = company.get('name', 'Company')
        industry = company.get('industry', 'Manufacturing')
        location = company.get('location', 'Gulf Region')
        sustainability_score = company.get('sustainability_score', 50)
        
        description = f"""
        {company_name} is offering {material} as part of our commitment to circular economy practices.
        
        **Material Details:**
        - High-quality {material} from {industry} operations
        - Consistent supply with regular availability
        - Compliant with industry standards and regulations
        
        **Environmental Impact:**
        - Reduces waste to landfill by 100%
        - Supports circular economy initiatives
        - Contributes to sustainability goals (Score: {sustainability_score}/100)
        
        **Logistics:**
        - Available for pickup or delivery from {location}
        - Flexible scheduling options
        - Professional packaging and handling
        
        **Contact:**
        Please reach out to discuss quantities, pricing, and delivery arrangements.
        """
        
        return description.strip()
    
    def calculate_material_metrics(self, company: Dict[str, Any], material: str) -> Dict[str, Any]:
        """Calculate material-specific metrics"""
        base_quantity = 1000  # Base metric tons
        frequency = "Monthly"
        
        # Adjust based on company size
        employee_count = company.get('employee_count', 100)
        if employee_count > 10000:
            base_quantity = 5000
            frequency = "Weekly"
        elif employee_count > 1000:
            base_quantity = 2000
            frequency = "Bi-weekly"
        
        # Adjust based on material type
        if "waste" in material.lower():
            base_quantity *= 0.8
        elif "chemical" in material.lower():
            base_quantity *= 1.2
        
        return {
            "quantity": round(base_quantity, 2),
            "unit": "metric_tons",
            "frequency": frequency,
            "price_range": "Negotiable",
            "quality_grade": "A",
            "certification": "ISO 14001 compliant"
        }

    def generate_listings_for_company(self, company: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate listings for all materials and waste streams of a company"""
        listings = []
        
        # Generate listings for materials
        materials = company.get('materials', [])
        for material in materials:
            if material and material.strip():
                listing = {
                    "company_id": company.get('id'),
                    "company_name": company.get('name'),
                    "material_name": material,
                    "material_type": "raw_material",
                    "description": self.generate_listing_description(company, material),
                    "metrics": self.calculate_material_metrics(company, material),
                    "status": "active",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
                listings.append(listing)
        
        # Generate listings for waste streams
        waste_streams = company.get('waste_streams', [])
        for waste in waste_streams:
            if waste and waste.strip():
                listing = {
                    "company_id": company.get('id'),
                    "company_name": company.get('name'),
                    "material_name": waste,
                    "material_type": "waste_stream",
                    "description": self.generate_listing_description(company, waste),
                    "metrics": self.calculate_material_metrics(company, waste),
                    "status": "active",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
                listings.append(listing)
        
        return listings

    def save_listings_to_database(self, listings: List[Dict[str, Any]]) -> bool:
        """Save generated listings to the database"""
        try:
            for listing in listings:
                # Check if listing already exists
                existing = self.supabase.table('material_listings').select('id').eq('company_id', listing['company_id']).eq('material_name', listing['material_name']).execute()
                
                if not existing.data:
                    # Insert new listing
                    self.supabase.table('material_listings').insert(listing).execute()
                    logger.info(f"Created listing for {listing['company_name']} - {listing['material_name']}")
                else:
                    # Update existing listing
                    listing_id = existing.data[0]['id']
                    self.supabase.table('material_listings').update(listing).eq('id', listing_id).execute()
                    logger.info(f"Updated listing for {listing['company_name']} - {listing['material_name']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving listings to database: {e}")
            return False
    
    def generate_all_listings(self) -> Dict[str, Any]:
        """Generate listings for all companies"""
        logger.info("Starting AI listings generation...")
        
        companies = self.get_all_companies()
        if not companies:
            return {"success": False, "message": "No companies found", "listings_generated": 0}
        
        total_listings = 0
        successful_companies = 0
        
        for company in companies:
            try:
                listings = self.generate_listings_for_company(company)
                if listings:
                    if self.save_listings_to_database(listings):
                        total_listings += len(listings)
                        successful_companies += 1
                        logger.info(f"Generated {len(listings)} listings for {company.get('name', 'Unknown')}")
                    else:
                        logger.error(f"Failed to save listings for {company.get('name', 'Unknown')}")
                else:
                    logger.warning(f"No listings generated for {company.get('name', 'Unknown')}")
            
        except Exception as e:
                logger.error(f"Error generating listings for {company.get('name', 'Unknown')}: {e}")
        
        result = {
            "success": True,
            "message": f"Generated {total_listings} listings for {successful_companies} companies",
            "listings_generated": total_listings,
            "companies_processed": len(companies),
            "successful_companies": successful_companies
        }
        
        logger.info(f"AI listings generation completed: {result}")
        return result

def main():
    """Main function to run the AI listings generator"""
    try:
    generator = AIListingsGenerator()
        result = generator.generate_all_listings()
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    main() 