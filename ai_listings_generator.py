import os
import json
import random
import asyncio
from typing import List, Dict, Any
from supabase import create_client, Client
from datetime import datetime, timedelta
import uuid
from decimal import Decimal

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

class AIListingsGenerator:
    def __init__(self):
        self.industries = {
            'manufacturing': {
                'waste_materials': [
                    'metal scrap', 'plastic waste', 'wood waste', 'paper waste', 'chemical waste',
                    'electronic waste', 'fabric scraps', 'rubber waste', 'glass waste', 'oil waste'
                ],
                'required_materials': [
                    'raw metals', 'plastic pellets', 'wood boards', 'paper rolls', 'chemicals',
                    'electronic components', 'fabric rolls', 'rubber sheets', 'glass panels', 'lubricants'
                ]
            },
            'food_processing': {
                'waste_materials': [
                    'organic waste', 'food scraps', 'packaging waste', 'vegetable waste', 'fruit waste',
                    'dairy waste', 'meat waste', 'grain waste', 'oil waste', 'water waste'
                ],
                'required_materials': [
                    'fresh produce', 'packaging materials', 'preservatives', 'spices', 'grains',
                    'dairy products', 'meat products', 'cooking oils', 'water', 'cleaning supplies'
                ]
            },
            'textile': {
                'waste_materials': [
                    'fabric scraps', 'yarn waste', 'dye waste', 'chemical waste', 'thread waste',
                    'fiber waste', 'packaging waste', 'water waste', 'heat waste', 'noise waste'
                ],
                'required_materials': [
                    'raw cotton', 'synthetic fibers', 'dyes', 'chemicals', 'threads',
                    'packaging materials', 'water', 'energy', 'machinery parts', 'maintenance supplies'
                ]
            },
            'construction': {
                'waste_materials': [
                    'concrete waste', 'wood waste', 'metal waste', 'plastic waste', 'glass waste',
                    'brick waste', 'tile waste', 'paint waste', 'chemical waste', 'packaging waste'
                ],
                'required_materials': [
                    'cement', 'wood boards', 'steel beams', 'plastic pipes', 'glass panels',
                    'bricks', 'tiles', 'paint', 'chemicals', 'packaging materials'
                ]
            },
            'chemical': {
                'waste_materials': [
                    'chemical waste', 'solvent waste', 'acid waste', 'base waste', 'toxic waste',
                    'packaging waste', 'water waste', 'heat waste', 'gas waste', 'solid waste'
                ],
                'required_materials': [
                    'raw chemicals', 'solvents', 'acids', 'bases', 'catalysts',
                    'packaging materials', 'water', 'energy', 'safety equipment', 'lab supplies'
                ]
            }
        }

    def generate_company_profile(self, company_name: str) -> Dict[str, Any]:
        """Generate a realistic company profile based on the name"""
        # Simple industry detection from company name
        name_lower = company_name.lower()
        
        if any(word in name_lower for word in ['food', 'beverage', 'dairy', 'meat', 'grain']):
            industry = 'food_processing'
        elif any(word in name_lower for word in ['textile', 'fabric', 'clothing', 'garment']):
            industry = 'textile'
        elif any(word in name_lower for word in ['construction', 'building', 'contractor']):
            industry = 'construction'
        elif any(word in name_lower for word in ['chemical', 'pharma', 'lab']):
            industry = 'chemical'
        else:
            industry = 'manufacturing'
        
        return {
            'industry': industry,
            'size': random.choice(['small', 'medium', 'large']),
            'location': random.choice(['Dubai', 'Abu Dhabi', 'Sharjah', 'Ajman', 'Ras Al Khaimah', 'Fujairah', 'Umm Al Quwain']),
            'production_capacity': random.randint(100, 10000)
        }

    def generate_material_listing(self, company_id: str, company_name: str, material_type: str) -> Dict[str, Any]:
        """Generate a realistic material listing, matching the actual materials table schema"""
        profile = self.generate_company_profile(company_name)
        industry = profile['industry']
        if material_type == 'waste':
            materials = self.industries[industry]['waste_materials']
        else:
            materials = self.industries[industry]['required_materials']
        material_name = random.choice(materials)
        # Generate realistic quantities and costs
        if material_type == 'waste':
            quantity = random.randint(10, 1000)
            unit = random.choice(['kg', 'tons', 'liters', 'pieces', 'cubic meters'])
            current_cost = round(random.uniform(10, 100), 2)
        else:
            quantity = random.randint(100, 5000)
            unit = random.choice(['kg', 'tons', 'liters', 'pieces', 'cubic meters'])
            current_cost = round(random.uniform(100, 1000), 2)
        # Generate realistic pricing
        price_per_unit = round(random.uniform(0.1, 50.0), 2)
        # Generate a robust, production-grade material listing
        now = datetime.now().astimezone()
        # Generate a UUID for the primary key
        material_id = str(uuid.uuid4())
        # Compose a rich description
        description = f"{material_type.title()} - {material_name} for {company_name} ({industry})"
        # AI tags and advanced fields
        ai_tags = [material_type, industry, 'ai_generated']
        estimated_value = round(price_per_unit * quantity, 2)
        priority_score = random.randint(1, 100)
        is_sponsored = False
        ai_generated = True
        availability = 'Available' if material_type == 'waste' else 'Needed'
        location = profile['location']
        # Compose the insert dict
        return {
            'id': material_id,
            'company_id': company_id,
            'material_name': material_name,
            'quantity': float(quantity),
            'unit': unit,
            'description': description,
            'type': material_type,
            'created_at': now.isoformat(),
            'ai_tags': ai_tags,
            'estimated_value': float(estimated_value),
            'priority_score': priority_score,
            'is_sponsored': is_sponsored,
            'embeddings': None,  # Placeholder for future AI embeddings
            'ai_generated': ai_generated,
            'availability': availability,
            'location': location,
            'price_per_unit': float(price_per_unit),
            'current_cost': str(round(price_per_unit * quantity, 2)),
            'potential_sources': [f"{material_name} supplier", f"{industry} recycler"],
            'updated_at': now.isoformat(),
            'category': material_type,
            'status': 'active',
            'material_properties': {
                'material_name': material_name,
                'quantity': float(quantity),
                'unit': unit,
                'description': description,
                'industry': industry
            },
            'shipping_params': {
                'preferred_method': random.choice(['truck', 'ship', 'rail']),
                'lead_time_days': random.randint(1, 14)
            },
            'sustainability_metrics': {
                'carbon_footprint': round(random.uniform(0.1, 10.0), 2),
                'recyclability': random.choice(['high', 'medium', 'low'])
            }
        }

    def generate_listings_for_company(self, company: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple listings for a single company"""
        listings = []
        company_id = company['id']
        company_name = company['name']
        
        # Generate 2-5 waste listings
        num_waste = random.randint(2, 5)
        for _ in range(num_waste):
            waste_listing = self.generate_material_listing(company_id, company_name, 'waste')
            listings.append(waste_listing)
        
        # Generate 1-3 requirement listings
        num_requirements = random.randint(1, 3)
        for _ in range(num_requirements):
            requirement_listing = self.generate_material_listing(company_id, company_name, 'requirement')
            listings.append(requirement_listing)
        
        return listings

    async def generate_all_listings(self):
        """Generate listings for all companies in the database"""
        print("🚀 Starting AI listings generation...")
        
        # Get all companies
        try:
            response = supabase.table('companies').select('id, name').execute()
            companies = response.data
            
            if not companies:
                print("❌ No companies found in database")
                return
            
            print(f"📋 Found {len(companies)} companies. Generating listings...")
            
            all_listings = []
            
            for i, company in enumerate(companies, 1):
                print(f"🏭 Processing company {i}/{len(companies)}: {company['name']}")
                
                try:
                    company_listings = self.generate_listings_for_company(company)
                    all_listings.extend(company_listings)
                    
                    # Insert listings for this company
                    if company_listings:
                        result = supabase.table('materials').insert(company_listings).execute()
                        print(f"  ✅ Added {len(company_listings)} listings for {company['name']}")
                    
                except Exception as e:
                    print(f"  ❌ Error processing {company['name']}: {str(e)}")
                    continue
            
            print(f"\n🎉 Successfully generated {len(all_listings)} total listings!")
            print(f"📊 Breakdown:")
            print(f"   - Companies processed: {len(companies)}")
            print(f"   - Total listings created: {len(all_listings)}")
            
            # Calculate statistics
            waste_count = len([l for l in all_listings if l['category'] == 'waste'])
            requirement_count = len([l for l in all_listings if l['category'] == 'requirement'])
            
            print(f"   - Waste listings: {waste_count}")
            print(f"   - Requirement listings: {requirement_count}")
            
            print(f"\n🌐 Your marketplace is now populated with {len(all_listings)} materials!")
            print(f"   Visit your marketplace to see all the generated listings.")
            
        except Exception as e:
            print(f"❌ Error generating listings: {str(e)}")
            raise

def main():
    """Main function to run the listings generator"""
    generator = AIListingsGenerator()
    
    # Run the generator
    asyncio.run(generator.generate_all_listings())

if __name__ == "__main__":
    main() 