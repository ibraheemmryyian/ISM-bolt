import os
import json
import random
import asyncio
from typing import List, Dict, Any
from supabase import create_client, Client
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize OpenAI client (optional - will use fallback if not available)
try:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False
    print("OpenAI not available, using fallback generation")

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
        """Generate a realistic material listing"""
        profile = self.generate_company_profile(company_name)
        industry = profile['industry']
        
        if material_type == 'waste':
            materials = self.industries[industry]['waste_materials']
        else:
            materials = self.industries[industry]['required_materials']
        
        material_name = random.choice(materials)
        
        # Generate realistic quantities
        if material_type == 'waste':
            quantity = random.randint(10, 1000)
            unit = random.choice(['kg', 'tons', 'liters', 'pieces', 'cubic meters'])
        else:
            quantity = random.randint(100, 5000)
            unit = random.choice(['kg', 'tons', 'liters', 'pieces', 'cubic meters'])
        
        # Generate realistic descriptions
        descriptions = {
            'waste': [
                f"High-quality {material_name} available for recycling or repurposing",
                f"Clean {material_name} waste from {profile['industry']} production",
                f"Regular supply of {material_name} waste, suitable for various applications",
                f"Industrial {material_name} waste, properly sorted and cleaned"
            ],
            'requirement': [
                f"Seeking reliable supplier for {material_name}",
                f"Looking for high-quality {material_name} for production",
                f"Need regular supply of {material_name} for manufacturing",
                f"Searching for {material_name} suppliers in the region"
            ]
        }
        
        description = random.choice(descriptions[material_type])
        
        # Generate realistic pricing
        if material_type == 'waste':
            price_per_unit = random.uniform(0.1, 5.0)  # Usually cheaper for waste
        else:
            price_per_unit = random.uniform(5.0, 50.0)  # More expensive for requirements
        
        return {
            'company_id': company_id,
            'material_name': material_name,
            'quantity': quantity,
            'unit': unit,
            'description': description,
            'type': material_type,
            'price_per_unit': round(price_per_unit, 2),
            'frequency': random.choice(['daily', 'weekly', 'monthly', 'quarterly']),
            'quality_grade': random.choice(['A', 'B', 'C']),
            'ai_generated': True,
            'created_at': datetime.now().isoformat()
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
            waste_count = len([l for l in all_listings if l['type'] == 'waste'])
            requirement_count = len([l for l in all_listings if l['type'] == 'requirement'])
            
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