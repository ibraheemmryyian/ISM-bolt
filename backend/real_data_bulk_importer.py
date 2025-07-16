#!/usr/bin/env python3
"""
Real Data Bulk Importer for ISM AI Platform
Imports 50 Gulf companies, generates AI listings, and creates matches
"""

import json
import asyncio
import aiohttp
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDataBulkImporter:
    def __init__(self):
        self.backend_url = os.environ.get('BACKEND_URL', 'http://localhost:3000')
        self.ai_gateway_url = os.environ.get('AI_GATEWAY_URL', 'http://localhost:3000')
        self.companies_data = []
        self.imported_companies = []
        self.generated_listings = []
        self.created_matches = []
        
    async def load_company_data(self):
        """Load the 50 Gulf companies data"""
        try:
            # Try both possible locations
            possible_files = [
                "data/50_gulf_companies_fixed.json",
                "../data/50_gulf_companies_fixed.json",
                "data/50_real_gulf_companies_cleaned.json",
                "../data/50_real_gulf_companies_cleaned.json"
            ]
            data_file = None
            for pf in possible_files:
                if os.path.exists(pf):
                    data_file = pf
                    break
            if not data_file:
                logger.error(f"Company data file not found in any known location: {possible_files}")
                return False
                
            with open(data_file, 'r', encoding='utf-8') as f:
                self.companies_data = json.load(f)
                
            logger.info(f"Loaded {len(self.companies_data)} companies from data file: {data_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading company data: {e}")
            return False
    
    async def import_companies(self):
        """Import companies into the system"""
        logger.info("Starting company import...")
        
        for i, company in enumerate(self.companies_data, 1):
            try:
                # Prepare company data with all required fields and sensible defaults
                name = company.get("name", f"Company {i}")
                industry = company.get("industry", "manufacturing")
                location = company.get("location", "Unknown")
                # Generate a unique email if not present
                base_email = company.get("email") or f"{name.lower().replace(' ', '').replace(',', '').replace('.', '')}{i}@example.com"
                email = base_email
                # Ensure email is unique by appending index if needed
                if any(c['data'].get('email') == email for c in self.imported_companies):
                    email = f"{name.lower().replace(' ', '').replace(',', '').replace('.', '')}{i}@example.com"
                
                company_data = {
                    "name": name,
                    "industry": industry,
                    "location": location,
                    "email": email,
                    "employee_count": int(company.get("employee_count", 0)),
                    "sustainability_score": float(company.get("sustainability_score", 0)),
                    "carbon_footprint": float(company.get("carbon_footprint", 0)),
                    "water_usage": float(company.get("water_usage", 0)),
                    "subscription_tier": company.get("subscription_tier", "pro"),
                    "subscription_status": company.get("subscription_status", "active"),
                    "role": company.get("role", "user"),
                    # Optional fields with defaults
                    "products": ", ".join(company.get("products", [])),
                    "main_materials": ", ".join(company.get("materials", [])),
                    "process_description": company.get("process_description", ""),
                    "sustainability_goals": company.get("sustainability_goals", []),
                    "current_waste_management": company.get("current_waste_management", ""),
                    "waste_quantity": float(company.get("waste_quantity", 0)),
                    "waste_unit": company.get("waste_unit", ""),
                    "waste_frequency": company.get("waste_frequency", ""),
                    "resource_needs": company.get("resource_needs", ""),
                    "energy_consumption": company.get("energy_consumption", ""),
                    "environmental_certifications": company.get("environmental_certifications", ""),
                    "current_recycling_practices": company.get("current_recycling_practices", ""),
                    "partnership_interests": company.get("partnership_interests", []),
                    "geographic_preferences": company.get("geographic_preferences", ""),
                    "technology_interests": company.get("technology_interests", ""),
                    "onboarding_completed": company.get("onboarding_completed", False),
                    "ai_portfolio_summary": company.get("ai_portfolio_summary", ""),
                    "ai_recommendations": company.get("ai_recommendations", {}),
                    "matches_count": int(company.get("matches_count", 0)),
                    "savings_achieved": float(company.get("savings_achieved", 0)),
                    "carbon_reduced": float(company.get("carbon_reduced", 0)),
                }
                
                # Import via backend API
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.backend_url}/api/companies",
                        json=company_data
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            company_id = result.get("id")
                            if company_id:
                                self.imported_companies.append({
                                    "id": company_id,
                                    "data": company_data
                                })
                                logger.info(f"Imported company {i}/{len(self.companies_data)}: {company_data['name']}")
                            else:
                                logger.warning(f"No company ID returned for {company_data['name']}")
                        else:
                            logger.error(f"Failed to import {company_data['name']}: {response.status}")
                            
            except Exception as e:
                logger.error(f"Error importing company {company.get('name', f'Company {i}')}: {e}")
                
        logger.info(f"Company import completed. Successfully imported {len(self.imported_companies)} companies")
        return len(self.imported_companies) > 0
    
    async def generate_ai_listings(self):
        """Generate AI listings for all companies"""
        logger.info("Starting AI listings generation...")
        
        for i, company in enumerate(self.imported_companies, 1):
            try:
                company_id = company["id"]
                company_data = company["data"]
                
                # Generate AI listings via backend API
                listing_request = {
                    "company_id": company_id,
                    "industry": company_data["industry"],
                    "location": company_data["location"],
                    "employee_count": company_data["employee_count"],
                    "sustainability_score": company_data["sustainability_score"]
                }
                
                async with aiohttp.ClientSession() as session:
                    # Generate AI listings for the company
                    async with session.post(
                        f"{self.backend_url}/api/ai/generate-listings/{company_id}",
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            if result.get('success') and result.get('data'):
                                listings = result['data'].get('listings', [])
                                self.generated_listings.extend(listings)
                                logger.info(f"Generated {len(listings)} AI listings for {company_data['name']}")
                            else:
                                logger.warning(f"No listings generated for {company_data['name']}")
                        else:
                            logger.error(f"Failed to generate listings for {company_data['name']}: {response.status}")
                            
            except Exception as e:
                logger.error(f"Error generating listings for {company['data']['name']}: {e}")
                
        logger.info(f"AI listings generation completed. Generated {len(self.generated_listings)} listings")
        return len(self.generated_listings) > 0
    
    async def create_matches(self):
        """Create AI matches between companies"""
        logger.info("Starting AI matching process...")
        
        try:
            # Run AI matching via backend API with retry logic
            async with aiohttp.ClientSession() as session:
                for attempt in range(3):  # Try 3 times
                    try:
                        async with session.post(
                            f"{self.backend_url}/api/ai/matching/run",
                            timeout=60
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                self.created_matches = result.get("matches", [])
                                logger.info(f"AI matching completed. Created {len(self.created_matches)} matches")
                                return True
                            elif response.status == 429:  # Rate limit
                                if attempt < 2:  # Not the last attempt
                                    wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                                    logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                                    await asyncio.sleep(wait_time)
                                    continue
                                else:
                                    logger.error("AI matching failed due to rate limiting after retries")
                                    return False
                            else:
                                logger.error(f"AI matching failed: {response.status}")
                                return False
                    except asyncio.TimeoutError:
                        if attempt < 2:
                            logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                            continue
                        else:
                            logger.error("AI matching failed due to timeout after retries")
                            return False
                        
        except Exception as e:
            logger.error(f"Error in AI matching: {e}")
            return False
    
    async def run_complete_import(self):
        """Run the complete import process"""
        logger.info("🚀 Starting Real Data Bulk Import Process")
        logger.info("=" * 50)
        
        # Step 1: Load company data
        if not await self.load_company_data():
            logger.error("Failed to load company data. Exiting.")
            return False
            
        # Step 2: Import companies
        if not await self.import_companies():
            logger.error("Failed to import companies. Exiting.")
            return False
            
        # Step 3: Generate AI listings
        if not await self.generate_ai_listings():
            logger.warning("Failed to generate AI listings, but continuing...")
            
        # Step 4: Create matches
        if not await self.create_matches():
            logger.warning("Failed to create matches, but continuing...")
            
        # Summary
        logger.info("=" * 50)
        logger.info("📊 IMPORT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"✅ Companies imported: {len(self.imported_companies)}")
        logger.info(f"✅ AI listings generated: {len(self.generated_listings)}")
        logger.info(f"✅ Matches created: {len(self.created_matches)}")
        logger.info("=" * 50)
        logger.info("🎉 Real data bulk import completed!")
        
        return True

async def main():
    """Main function"""
    importer = RealDataBulkImporter()
    success = await importer.run_complete_import()
    
    if success:
        print("\n🎉 SUCCESS! Your ISM AI platform is now populated with real data!")
        print("🌐 Access your admin dashboard at: http://localhost:5173/admin")
        print("📊 Check out all the companies, listings, and matches!")
    else:
        print("\n❌ Import failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 