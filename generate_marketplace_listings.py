#!/usr/bin/env python3
"""
AI Marketplace Listings Generator
Generates realistic waste and requirement materials for all companies in the database
"""

import os
import json
import random
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

class MarketplaceListingsGenerator:
    def __init__(self):
        # Define realistic materials by industry
        self.materials_data = {
            'manufacturing': {
                'waste': [
                    'Metal Scrap', 'Plastic Waste', 'Wood Waste', 'Paper Waste', 'Chemical Waste',
                    'Electronic Waste', 'Fabric Scraps', 'Rubber Waste', 'Glass Waste', 'Oil Waste',
                    'Steel Scrap', 'Aluminum Waste', 'Copper Scrap', 'Plastic Pellets Waste'
                ],
                'requirements': [
                    'Raw Steel', 'Aluminum Sheets', 'Copper Wire', 'Plastic Pellets', 'Wood Boards',
                    'Paper Rolls', 'Industrial Chemicals', 'Electronic Components', 'Fabric Rolls',
                    'Rubber Sheets', 'Glass Panels', 'Industrial Lubricants', 'Steel Beams'
                ]
            },
            'food_processing': {
                'waste': [
                    'Organic Waste', 'Food Scraps', 'Packaging Waste', 'Vegetable Waste', 'Fruit Waste',
                    'Dairy Waste', 'Meat Waste', 'Grain Waste', 'Cooking Oil Waste', 'Water Waste',
                    'Bread Waste', 'Sugar Waste', 'Spice Waste', 'Beverage Waste'
                ],
                'requirements': [
                    'Fresh Vegetables', 'Fresh Fruits', 'Dairy Products', 'Meat Products', 'Grains',
                    'Cooking Oils', 'Packaging Materials', 'Preservatives', 'Spices', 'Sugar',
                    'Water', 'Cleaning Supplies', 'Beverage Ingredients'
                ]
            },
            'textile': {
                'waste': [
                    'Fabric Scraps', 'Yarn Waste', 'Dye Waste', 'Chemical Waste', 'Thread Waste',
                    'Fiber Waste', 'Packaging Waste', 'Water Waste', 'Heat Waste', 'Noise Waste',
                    'Cotton Waste', 'Synthetic Fiber Waste', 'Textile Dye Waste'
                ],
                'requirements': [
                    'Raw Cotton', 'Synthetic Fibers', 'Textile Dyes', 'Industrial Chemicals',
                    'Threads', 'Packaging Materials', 'Water', 'Energy', 'Machinery Parts',
                    'Maintenance Supplies', 'Fabric Rolls', 'Yarn Spools'
                ]
            },
            'construction': {
                'waste': [
                    'Concrete Waste', 'Wood Waste', 'Metal Waste', 'Plastic Waste', 'Glass Waste',
                    'Brick Waste', 'Tile Waste', 'Paint Waste', 'Chemical Waste', 'Packaging Waste',
                    'Steel Waste', 'Aluminum Waste', 'Cement Waste', 'Sand Waste'
                ],
                'requirements': [
                    'Cement', 'Wood Boards', 'Steel Beams', 'Plastic Pipes', 'Glass Panels',
                    'Bricks', 'Tiles', 'Paint', 'Industrial Chemicals', 'Packaging Materials',
                    'Sand', 'Gravel', 'Reinforcement Steel', 'Construction Tools'
                ]
            },
            'chemical': {
                'waste': [
                    'Chemical Waste', 'Solvent Waste', 'Acid Waste', 'Base Waste', 'Toxic Waste',
                    'Packaging Waste', 'Water Waste', 'Heat Waste', 'Gas Waste', 'Solid Waste',
                    'Reactive Chemical Waste', 'Organic Solvent Waste', 'Inorganic Waste'
                ],
                'requirements': [
                    'Raw Chemicals', 'Industrial Solvents', 'Acids', 'Bases', 'Catalysts',
                    'Packaging Materials', 'Water', 'Energy', 'Safety Equipment', 'Lab Supplies',
                    'Chemical Reagents', 'Industrial Gases', 'Chemical Containers'
                ]
            }
        }

    def detect_industry(self, company_name):
        """Detect industry based on company name"""
        name_lower = company_name.lower()
        
        if any(word in name_lower for word in ['food', 'beverage', 'dairy', 'meat', 'grain', 'agriculture']):
            return 'food_processing'
        elif any(word in name_lower for word in ['textile', 'fabric', 'clothing', 'garment', 'fashion']):
            return 'textile'
        elif any(word in name_lower for word in ['construction', 'building', 'contractor', 'infrastructure']):
            return 'construction'
        elif any(word in name_lower for word in ['chemical', 'pharma', 'lab', 'petrochemical']):
            return 'chemical'
        else:
            return 'manufacturing'

    def generate_listing(self, company_id, company_name, material_type):
        """Generate a single material listing"""
        industry = self.detect_industry(company_name)
        
        if material_type == 'waste':
            materials = self.materials_data[industry]['waste']
            quantity = random.randint(50, 2000)
            price_range = (0.5, 15.0)
            descriptions = [
                f"High-quality {material_name} available for recycling",
                f"Clean {material_name} from {industry} production",
                f"Regular supply of {material_name}, suitable for various applications",
                f"Industrial {material_name}, properly sorted and cleaned"
            ]
        else:
            materials = self.materials_data[industry]['requirements']
            quantity = random.randint(100, 5000)
            price_range = (10.0, 200.0)
            descriptions = [
                f"Seeking reliable supplier for {material_name}",
                f"Looking for high-quality {material_name} for production",
                f"Need regular supply of {material_name} for manufacturing",
                f"Searching for {material_name} suppliers in the region"
            ]
        
        material_name = random.choice(materials)
        description = random.choice(descriptions)
        price_per_unit = round(random.uniform(*price_range), 2)
        
        return {
            'company_id': company_id,
            'material_name': material_name,
            'quantity': quantity,
            'unit': random.choice(['kg', 'tons', 'liters', 'pieces', 'cubic meters']),
            'description': description,
            'type': material_type,
            'price_per_unit': price_per_unit,
            'frequency': random.choice(['daily', 'weekly', 'monthly', 'quarterly']),
            'quality_grade': random.choice(['A', 'B', 'C']),
            'ai_generated': True,
            'created_at': datetime.now().isoformat()
        }

    def generate_company_listings(self, company):
        """Generate multiple listings for a single company"""
        company_id = company['id']
        company_name = company['name']
        
        listings = []
        
        # Generate 3-6 waste listings
        num_waste = random.randint(3, 6)
        for _ in range(num_waste):
            waste_listing = self.generate_listing(company_id, company_name, 'waste')
            listings.append(waste_listing)
        
        # Generate 2-4 requirement listings
        num_requirements = random.randint(2, 4)
        for _ in range(num_requirements):
            requirement_listing = self.generate_listing(company_id, company_name, 'requirement')
            listings.append(requirement_listing)
        
        return listings

    def generate_all_listings(self):
        """Generate listings for all companies"""
        print("🚀 Starting AI Marketplace Listings Generator...")
        print("=" * 60)
        
        try:
            # Get all companies
            print("📋 Fetching companies from database...")
            response = supabase.table('companies').select('id, name').execute()
            companies = response.data
            
            if not companies:
                print("❌ No companies found in database")
                return
            
            print(f"✅ Found {len(companies)} companies")
            print("🏭 Generating listings for each company...")
            print("-" * 60)
            
            total_listings = 0
            all_listings = []
            
            for i, company in enumerate(companies, 1):
                print(f"Processing {i}/{len(companies)}: {company['name']}")
                
                try:
                    company_listings = self.generate_company_listings(company)
                    all_listings.extend(company_listings)
                    
                    # Insert listings for this company
                    if company_listings:
                        result = supabase.table('materials').insert(company_listings).execute()
                        total_listings += len(company_listings)
                        print(f"  ✅ Added {len(company_listings)} listings")
                    
                except Exception as e:
                    print(f"  ❌ Error: {str(e)}")
                    continue
            
            print("-" * 60)
            print("📊 GENERATION COMPLETE!")
            print(f"🏭 Companies processed: {len(companies)}")
            print(f"📦 Total listings created: {total_listings}")
            
            # Calculate breakdown
            waste_count = len([l for l in all_listings if l['type'] == 'waste'])
            requirement_count = len([l for l in all_listings if l['type'] == 'requirement'])
            
            print(f"🗑️  Waste listings: {waste_count}")
            print(f"📋 Requirement listings: {requirement_count}")
            print(f"💰 Average price range: $0.50 - $200.00 per unit")
            
            print("\n🎉 Your marketplace is now fully populated!")
            print("🌐 Visit your marketplace to see all the generated materials.")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Fatal error: {str(e)}")
            raise

def main():
    """Main function"""
    generator = MarketplaceListingsGenerator()
    generator.generate_all_listings()

if __name__ == "__main__":
    main() 