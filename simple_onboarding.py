#!/usr/bin/env python3
"""
Simple Onboarding Script - Directly calls the listing inference service
to generate material listings without requiring the backend server
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path

# Simple dotenv implementation
def load_dotenv(dotenv_path):
    """Load environment variables from .env file"""
    try:
        with open(dotenv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")
        return True
    except Exception as e:
        print(f"Error loading .env file: {e}")
        return False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('onboarding.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add backend directory to path
backend_path = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_path.absolute()))

# Import the listing inference service
try:
    from listing_inference_service import ListingInferenceService
except ImportError as e:
    logger.error(f"Failed to import ListingInferenceService: {e}")
    sys.exit(1)

async def run_onboarding():
    """Run the onboarding process"""
    logger.info("Starting onboarding process...")
    
    # Load environment variables from backend/.env
    env_file = Path(__file__).parent / 'backend' / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment from: {env_file}")
    else:
        logger.warning(f"Environment file not found: {env_file}")
    
    # Initialize the listing inference service
    try:
        service = ListingInferenceService()
        logger.info("Initialized ListingInferenceService")
    except Exception as e:
        logger.error(f"Failed to initialize ListingInferenceService: {e}")
        return
    
    # Get company profile from user
    company_profile = await get_company_profile()
    
    # Generate listings
    try:
        logger.info(f"Generating listings for company: {company_profile.get('name')}")
        result = await service.generate_listings_from_profile(company_profile)
        
        # Save results to file
        output_file = Path('onboarding_results.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Display results
        logger.info(f"Generated {len(result.get('predicted_outputs', []))} material listings")
        logger.info(f"Results saved to: {output_file}")
        
        # Print material listings
        print("\n===== GENERATED MATERIAL LISTINGS =====")
        for i, material in enumerate(result.get('predicted_outputs', [])):
            print(f"\n[{i+1}] {material.get('material_name', 'Unknown Material')}")
            print(f"    Category: {material.get('category', 'Unknown')}")
            print(f"    Quantity: {material.get('quantity', 0)} {material.get('unit', 'units')}")
            print(f"    Quality: {material.get('quality_grade', 'Unknown')}")
            print(f"    Value: ${material.get('potential_value', 0):,.2f}")
            print(f"    Description: {material.get('description', 'No description')}")
        
    except Exception as e:
        logger.error(f"Error generating listings: {e}")

async def get_company_profile():
    """Get company profile from user input"""
    print("\n===== COMPANY PROFILE =====")
    print("Please provide information about your company:")
    
    company_profile = {}
    
    # Basic information
    company_profile['name'] = input("Company Name: ").strip() or "Demo Company"
    company_profile['industry'] = input("Industry (e.g., manufacturing, chemical, automotive): ").strip() or "manufacturing"
    company_profile['location'] = input("Location: ").strip() or "Global"
    company_profile['employee_count'] = int(input("Number of Employees: ").strip() or "1000")
    
    # Products and materials
    products_input = input("Main Products (comma-separated): ").strip()
    company_profile['products'] = [p.strip() for p in products_input.split(',')] if products_input else ["product1", "product2"]
    
    materials_input = input("Main Materials Used (comma-separated): ").strip()
    company_profile['materials'] = [m.strip() for m in materials_input.split(',')] if materials_input else ["steel", "aluminum", "plastic"]
    
    waste_input = input("Waste Streams (comma-separated): ").strip()
    company_profile['waste_streams'] = [w.strip() for w in waste_input.split(',')] if waste_input else ["scrap", "waste"]
    
    # Additional information
    company_profile['production_volume'] = input("Annual Production Volume (e.g., 10000 tons): ").strip() or "10000 tons"
    company_profile['sustainability_goals'] = input("Sustainability Goals: ").strip() or "Reduce waste and carbon footprint"
    
    return company_profile

def main():
    """Main entry point"""
    print("🚀 SymbioFlows Simple Onboarding")
    print("📝 This script will generate material listings for your company")
    print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 60)
    
    try:
        asyncio.run(run_onboarding())
        print("\n✅ Onboarding completed successfully!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Onboarding failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import time
    main()