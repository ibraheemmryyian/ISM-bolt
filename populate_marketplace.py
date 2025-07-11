#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marketplace Populator
Generates realistic waste and requirement materials for all companies
"""

import os
import random
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize Supabase
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# Material definitions
WASTE_MATERIALS = [
    'Metal Scrap', 'Plastic Waste', 'Wood Waste', 'Paper Waste', 'Chemical Waste',
    'Electronic Waste', 'Fabric Scraps', 'Rubber Waste', 'Glass Waste', 'Oil Waste',
    'Organic Waste', 'Food Scraps', 'Packaging Waste', 'Concrete Waste', 'Brick Waste',
    'Tile Waste', 'Paint Waste', 'Steel Scrap', 'Aluminum Waste', 'Copper Scrap',
    'Textile Waste', 'Yarn Waste', 'Dye Waste', 'Thread Waste', 'Fiber Waste',
    'Cement Waste', 'Sand Waste', 'Gravel Waste', 'Construction Debris'
]

REQUIRED_MATERIALS = [
    'Raw Steel', 'Aluminum Sheets', 'Copper Wire', 'Plastic Pellets', 'Wood Boards',
    'Paper Rolls', 'Industrial Chemicals', 'Electronic Components', 'Fabric Rolls',
    'Rubber Sheets', 'Glass Panels', 'Industrial Lubricants', 'Fresh Vegetables',
    'Fresh Fruits', 'Dairy Products', 'Meat Products', 'Grains', 'Cooking Oils',
    'Packaging Materials', 'Preservatives', 'Spices', 'Sugar', 'Water',
    'Cleaning Supplies', 'Raw Cotton', 'Synthetic Fibers', 'Textile Dyes',
    'Cement', 'Steel Beams', 'Plastic Pipes', 'Bricks', 'Tiles', 'Paint',
    'Sand', 'Gravel', 'Construction Tools', 'Raw Chemicals', 'Industrial Solvents'
]

def generate_listing(company_id, company_name, material_type):
    """Generate a single material listing"""
    if material_type == 'waste':
        material_name = random.choice(WASTE_MATERIALS)
        quantity = random.randint(50, 2000)
        price_per_unit = round(random.uniform(0.5, 15.0), 2)
        descriptions = [
            f"High-quality {material_name} available for recycling",
            f"Clean {material_name} from industrial production",
            f"Regular supply of {material_name}, suitable for various applications",
            f"Industrial {material_name}, properly sorted and cleaned"
        ]
    else:
        material_name = random.choice(REQUIRED_MATERIALS)
        quantity = random.randint(100, 5000)
        price_per_unit = round(random.uniform(10.0, 200.0), 2)
        descriptions = [
            f"Seeking reliable supplier for {material_name}",
            f"Looking for high-quality {material_name} for production",
            f"Need regular supply of {material_name} for manufacturing",
            f"Searching for {material_name} suppliers in the region"
        ]
    
    return {
        'company_id': company_id,
        'material_name': material_name,
        'quantity': quantity,
        'unit': random.choice(['kg', 'tons', 'liters', 'pieces', 'cubic meters']),
        'description': random.choice(descriptions),
        'type': material_type,
        'price_per_unit': price_per_unit,
        'frequency': random.choice(['daily', 'weekly', 'monthly', 'quarterly']),
        'quality_grade': random.choice(['A', 'B', 'C']),
        'ai_generated': True,
        'created_at': datetime.now().isoformat()
    }

def generate_company_listings(company):
    """Generate listings for a single company"""
    company_id = company['id']
    company_name = company['name']
    
    listings = []
    
    # Generate 3-6 waste listings
    num_waste = random.randint(3, 6)
    for _ in range(num_waste):
        waste_listing = generate_listing(company_id, company_name, 'waste')
        listings.append(waste_listing)
    
    # Generate 2-4 requirement listings
    num_requirements = random.randint(2, 4)
    for _ in range(num_requirements):
        requirement_listing = generate_listing(company_id, company_name, 'requirement')
        listings.append(requirement_listing)
    
    return listings

def main():
    """Main function to populate marketplace"""
    print("Starting Marketplace Population...")
    print("=" * 50)
    
    try:
        # Get all companies
        print("Fetching companies...")
        response = supabase.table('companies').select('id, name').execute()
        companies = response.data
        
        if not companies:
            print("No companies found")
            return
        
        print(f"Found {len(companies)} companies")
        print("Generating listings...")
        print("-" * 50)
        
        total_listings = 0
        
        for i, company in enumerate(companies, 1):
            print(f"Processing {i}/{len(companies)}: {company['name']}")
            
            try:
                company_listings = generate_company_listings(company)
                
                # Insert listings
                if company_listings:
                    result = supabase.table('materials').insert(company_listings).execute()
                    total_listings += len(company_listings)
                    print(f"  Added {len(company_listings)} listings")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                continue
        
        print("-" * 50)
        print("MARKETPLACE POPULATION COMPLETE!")
        print(f"Companies: {len(companies)}")
        print(f"Total listings: {total_listings}")
        print("Your marketplace is now fully populated!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main() 