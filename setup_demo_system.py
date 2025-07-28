#!/usr/bin/env python3
"""
SymbioFlows Demo System Setup

This script prepares the system for a video-ready demonstration:
1. Imports real company data
2. Enriches it with quantitative waste stream information
3. Prepares material listings and matches
4. Sets up AI onboarding flow

Run this script before recording your demo video.
"""

import os
import sys
import json
import logging
import subprocess
import asyncio
import aiohttp
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('demo_setup.log')
    ]
)

logger = logging.getLogger('demo_setup')

class DemoSetup:
    """Prepares the system for a video-ready demonstration"""
    
    def __init__(self):
        self.backend_url = os.environ.get('BACKEND_URL', 'http://localhost:3000')
        self.ai_service_url = os.environ.get('AI_SERVICE_URL', 'http://localhost:5001')
        self.data_dir = Path('/workspace/data')
        self.enriched_dir = self.data_dir / 'enriched'
        self.backend_dir = Path('/workspace/backend')
        
    def ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.enriched_dir, exist_ok=True)
        
    async def check_services(self):
        """Check if required services are running"""
        services = [
            {'name': 'Backend Service', 'url': f"{self.backend_url}/api/health"},
            {'name': 'AI Service', 'url': f"{self.ai_service_url}/health"}
        ]
        
        for service in services:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(service['url'], timeout=5) as response:
                        if response.status == 200:
                            logger.info(f"✅ {service['name']} is running")
                        else:
                            logger.warning(f"⚠️ {service['name']} returned status {response.status}")
            except Exception as e:
                logger.warning(f"⚠️ {service['name']} is not available: {e}")
                logger.info(f"   You may need to start {service['name']} manually")
    
    async def import_companies(self):
        """Import companies using real_data_bulk_importer.py"""
        logger.info("Importing companies...")
        
        # Check if the import script exists
        importer_path = self.backend_dir / 'real_data_bulk_importer.py'
        if not importer_path.exists():
            logger.error(f"❌ Company importer not found at {importer_path}")
            return False
        
        # Run the importer
        try:
            # Try using the script directly first
            data_path = self.data_dir / 'fixed_realworlddata.json'
            if not data_path.exists():
                logger.warning(f"⚠️ Data file not found at {data_path}")
                data_path = Path('/workspace/fixed_realworlddata.json')
            
            # Prepare the command
            cmd = [
                sys.executable,
                str(importer_path),
                '--file', str(data_path),
                '--mode', 'prod'
            ]
            
            # Run the command
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"❌ Company import failed: {result.stderr}")
                
                # Try an alternative approach by calling the API directly
                logger.info("Attempting direct API import...")
                
                # Load the company data
                with open(data_path, 'r', encoding='utf-8') as f:
                    companies = json.load(f)
                
                # Import each company via API
                success_count = 0
                async with aiohttp.ClientSession() as session:
                    for company in companies:
                        try:
                            async with session.post(
                                f"{self.backend_url}/api/companies",
                                json=company
                            ) as response:
                                if response.status == 200:
                                    success_count += 1
                                    logger.info(f"✅ Imported {company.get('name', 'Unknown')}")
                                else:
                                    logger.warning(f"⚠️ Failed to import {company.get('name', 'Unknown')}: {response.status}")
                        except Exception as e:
                            logger.error(f"❌ Error importing {company.get('name', 'Unknown')}: {e}")
                
                if success_count > 0:
                    logger.info(f"✅ Successfully imported {success_count}/{len(companies)} companies via API")
                    return True
                else:
                    logger.error("❌ Failed to import any companies")
                    return False
            else:
                logger.info("✅ Successfully imported companies")
                logger.info(result.stdout)
                return True
                
        except Exception as e:
            logger.error(f"❌ Error running company importer: {e}")
            return False
    
    async def enrich_company_data(self):
        """Enrich company data with quantitative information"""
        logger.info("Enriching company data...")
        
        # Check if the enrichment script exists
        enricher_path = self.backend_dir / 'enrich_company_data.py'
        if not enricher_path.exists():
            logger.error(f"❌ Data enrichment script not found at {enricher_path}")
            return False
        
        # Run the enricher
        try:
            cmd = [sys.executable, str(enricher_path)]
            
            # Run the command
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"❌ Data enrichment failed: {result.stderr}")
                return False
            else:
                logger.info("✅ Successfully enriched company data")
                logger.info(result.stdout)
                return True
                
        except Exception as e:
            logger.error(f"❌ Error running data enrichment: {e}")
            return False
    
    async def generate_material_listings(self):
        """Generate material listings for companies"""
        logger.info("Generating material listings...")
        
        try:
            # Call the listing generation API endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.backend_url}/api/ai/generate-all-listings",
                    timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Generated {result.get('count', 0)} material listings")
                        return True
                    else:
                        logger.warning(f"⚠️ Listing generation API returned status {response.status}")
                        
                        # Fallback to the enriched listings
                        logger.info("Using pre-generated enriched listings as fallback...")
                        
                        # Check for enriched listing files
                        listing_files = list(self.enriched_dir.glob('*_listings.json'))
                        if not listing_files:
                            logger.error("❌ No enriched listing files found")
                            return False
                        
                        # Import the listings via API
                        success_count = 0
                        for file_path in listing_files:
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    listings = json.load(f)
                                
                                # Extract company name from filename
                                company_name = file_path.stem.replace('_listings', '')
                                
                                # Send listings to API
                                async with session.post(
                                    f"{self.backend_url}/api/materials/bulk-import",
                                    json={
                                        'company_name': company_name.replace('_', ' '),
                                        'listings': listings
                                    }
                                ) as listing_response:
                                    if listing_response.status == 200:
                                        success_count += 1
                                        logger.info(f"✅ Imported listings for {company_name}")
                                    else:
                                        logger.warning(f"⚠️ Failed to import listings for {company_name}: {listing_response.status}")
                            except Exception as e:
                                logger.error(f"❌ Error importing listings from {file_path}: {e}")
                        
                        if success_count > 0:
                            logger.info(f"✅ Successfully imported listings for {success_count} companies")
                            return True
                        else:
                            logger.error("❌ Failed to import any listings")
                            return False
                        
        except Exception as e:
            logger.error(f"❌ Error generating material listings: {e}")
            
            # Try another approach
            logger.info("Trying alternative listing generation approach...")
            
            try:
                # Use the listing generator directly if available
                generator_path = self.backend_dir / 'ai_listings_generator.py'
                if generator_path.exists():
                    cmd = [sys.executable, str(generator_path)]
                    
                    # Run the command
                    logger.info(f"Running: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False
                    )
                    
                    if result.returncode == 0:
                        logger.info("✅ Successfully generated listings")
                        logger.info(result.stdout)
                        return True
                    else:
                        logger.warning(f"⚠️ Listing generator failed: {result.stderr}")
                        return False
                else:
                    logger.warning(f"⚠️ Listing generator not found at {generator_path}")
                    return False
                    
            except Exception as e2:
                logger.error(f"❌ Error running listing generator: {e2}")
                return False
    
    async def generate_matches(self):
        """Generate matches between material listings"""
        logger.info("Generating matches...")
        
        try:
            # Call the matching API endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.backend_url}/api/ai/matching/run",
                    timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Generated {len(result.get('matches', []))} matches")
                        return True
                    else:
                        logger.warning(f"⚠️ Match generation API returned status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Error generating matches: {e}")
            
            # Try another approach
            logger.info("Trying alternative match generation approach...")
            
            try:
                # Use the matching engine directly if available
                matcher_path = self.backend_dir / 'improved_ai_matching_engine.py'
                if matcher_path.exists():
                    cmd = [sys.executable, str(matcher_path)]
                    
                    # Run the command
                    logger.info(f"Running: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False
                    )
                    
                    if result.returncode == 0:
                        logger.info("✅ Successfully generated matches")
                        logger.info(result.stdout)
                        return True
                    else:
                        logger.warning(f"⚠️ Match generator failed: {result.stderr}")
                        return False
                else:
                    logger.warning(f"⚠️ Match generator not found at {matcher_path}")
                    return False
                    
            except Exception as e2:
                logger.error(f"❌ Error running match generator: {e2}")
                return False
    
    async def setup_ai_onboarding(self):
        """Setup AI onboarding flow"""
        logger.info("Setting up AI onboarding flow...")
        
        try:
            # Check if adaptive onboarding is available
            onboarding_path = self.backend_dir / 'adaptive_ai_onboarding.py'
            if not onboarding_path.exists():
                logger.warning(f"⚠️ AI onboarding script not found at {onboarding_path}")
                logger.info("Skipping AI onboarding setup")
                return True
            
            # Initialize onboarding
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.backend_url}/api/adaptive-onboarding/initialize",
                    json={"reset": True}
                ) as response:
                    if response.status == 200:
                        logger.info("✅ Successfully initialized AI onboarding")
                        return True
                    else:
                        logger.warning(f"⚠️ AI onboarding initialization API returned status {response.status}")
                        logger.info("Continuing without explicit onboarding initialization")
                        return True
                        
        except Exception as e:
            logger.error(f"❌ Error setting up AI onboarding: {e}")
            logger.info("Continuing without AI onboarding setup")
            return True
    
    async def run_setup(self):
        """Run the complete setup process"""
        logger.info("🚀 Starting SymbioFlows Demo Setup")
        logger.info("=" * 60)
        
        # Create required directories
        self.ensure_directories()
        
        # Check services
        await self.check_services()
        
        # Step 1: Import companies
        if not await self.import_companies():
            logger.warning("⚠️ Company import had issues, but continuing...")
        
        # Step 2: Enrich company data
        if not await self.enrich_company_data():
            logger.warning("⚠️ Data enrichment had issues, but continuing...")
        
        # Step 3: Generate material listings
        if not await self.generate_material_listings():
            logger.warning("⚠️ Listing generation had issues, but continuing...")
        
        # Step 4: Generate matches
        if not await self.generate_matches():
            logger.warning("⚠️ Match generation had issues, but continuing...")
        
        # Step 5: Setup AI onboarding
        if not await self.setup_ai_onboarding():
            logger.warning("⚠️ AI onboarding setup had issues, but continuing...")
        
        # Setup complete
        logger.info("=" * 60)
        logger.info("🎉 Demo Setup Complete!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("To record your demo video, follow these steps:")
        logger.info("1. Create a new account in the system")
        logger.info("2. Go through the AI onboarding process")
        logger.info("3. Review and modify the generated material listings")
        logger.info("4. View potential matches")
        logger.info("=" * 60)
        
        return True

async def main():
    """Main function"""
    setup = DemoSetup()
    await setup.run_setup()
    return 0

if __name__ == "__main__":
    asyncio.run(main())