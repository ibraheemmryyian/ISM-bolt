import os
import re
import json
from supabase import create_client, Client
from typing import Dict, List, Any
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def parse_gulf_company_data(file_path: str) -> List[Dict[str, Any]]:
    """Parse the Gulf company data file and extract structured data."""
    companies = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split by company sections (each company starts with "Company X")
    company_sections = re.split(r'\nCompany \d+\n', content)
    
    for i, section in enumerate(company_sections):
        if not section.strip():
            continue
            
        company_data = {}
        lines = section.strip().split('\n')
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'Name':
                    company_data['name'] = value
                elif key == 'Industry':
                    company_data['industry'] = value
                elif key == 'Products':
                    company_data['products'] = value
                elif key == 'Location':
                    company_data['location'] = value
                elif key == 'Volume':
                    company_data['volume'] = value
                elif key == 'Materials':
                    company_data['materials'] = value
                elif key == 'Processes':
                    company_data['processes'] = value
                elif key == 'Waste Materials':
                    company_data['waste_materials'] = value
        
        if company_data.get('name'):
            company_data['company_number'] = i
            companies.append(company_data)
    
    return companies

def generate_materials_for_gulf_company(company: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate material listings based on Gulf company profile."""
    materials = []
    
    # Extract waste materials from company data
    waste_materials_str = company.get('waste_materials', '')
    if waste_materials_str:
        waste_list = [m.strip() for m in waste_materials_str.split(',')]
        
        for waste in waste_list:
            if waste:
                # Parse volume information
                volume_info = company.get('volume', '')
                quantity = parse_volume_to_quantity(volume_info)
                
                waste_material = {
                    'material_name': waste,
                    'quantity': quantity,
                    'unit': get_unit_from_volume(volume_info),
                    'type': 'waste',
                    'description': f"{waste} from {company['name']} {company['industry']} operations",
                    'availability': 'Available',
                    'price_per_unit': round(random.uniform(0.1, 5.0), 2),
                    'location': company.get('location', 'Unknown'),
                    'created_at': datetime.now().isoformat()
                }
                materials.append(waste_material)
    
    # Generate requirement materials based on industry
    requirement_materials = get_requirement_materials_for_industry(company.get('industry', ''))
    
    for req_material in requirement_materials:
        requirement = {
            'material_name': req_material,
            'quantity': random.randint(500, 50000),
            'unit': 'kg' if req_material != 'Energy' else 'kWh',
            'type': 'requirement',
            'description': f"{req_material} needed for {company['name']} operations",
            'availability': 'Needed',
            'price_per_unit': round(random.uniform(0.5, 10.0), 2),
            'location': company.get('location', 'Unknown'),
            'created_at': datetime.now().isoformat()
        }
        materials.append(requirement)
    
    return materials

def parse_volume_to_quantity(volume_str: str) -> int:
    """Parse volume string to quantity number."""
    if not volume_str:
        return random.randint(100, 10000)
    
    # Extract numbers from volume string
    numbers = re.findall(r'\d+', volume_str)
    if numbers:
        base_quantity = int(numbers[0])
        # Add some randomness to the quantity
        return int(base_quantity * random.uniform(0.8, 1.2))
    
    return random.randint(100, 10000)

def get_unit_from_volume(volume_str: str) -> str:
    """Extract unit from volume string."""
    if 'cubic meters' in volume_str.lower():
        return 'm³'
    elif 'metric tons' in volume_str.lower():
        return 'kg'
    elif 'liters' in volume_str.lower():
        return 'L'
    else:
        return 'kg'

def get_requirement_materials_for_industry(industry: str) -> List[str]:
    """Get requirement materials based on industry type."""
    industry_lower = industry.lower()
    
    base_requirements = ['Energy', 'Water', 'Packaging Materials']
    
    if 'construction' in industry_lower or 'real estate' in industry_lower:
        return base_requirements + ['Steel', 'Concrete', 'Glass', 'Wood', 'Aluminum']
    elif 'manufacturing' in industry_lower:
        return base_requirements + ['Raw Materials', 'Polymers', 'Fabrics', 'Agricultural Produce']
    elif 'oil' in industry_lower or 'gas' in industry_lower:
        return base_requirements + ['Drilling Fluids', 'Catalysts', 'Hydrocarbons', 'Pipelines']
    elif 'healthcare' in industry_lower:
        return base_requirements + ['Pharmaceuticals', 'Medical Supplies', 'Lab Reagents', 'Sterile Packaging']
    elif 'tourism' in industry_lower or 'hospitality' in industry_lower:
        return base_requirements + ['Food & Beverages', 'Linens', 'Cleaning Supplies', 'Paper Products']
    elif 'water treatment' in industry_lower:
        return base_requirements + ['Raw Water', 'Chemical Coagulants', 'Disinfectants', 'Filtration Media']
    elif 'logistics' in industry_lower or 'transportation' in industry_lower:
        return base_requirements + ['Fuel', 'Vehicle Parts', 'Packaging Materials']
    else:
        return base_requirements + ['Raw Materials', 'Cleaning Supplies']

def create_gulf_company_record(company: Dict[str, Any]) -> Dict[str, Any]:
    """Create a company record for the database from Gulf company data."""
    # Generate realistic email based on company name
    email_name = company['name'].lower().replace(' ', '').replace('&', '').replace('.', '')
    email = f"{email_name}@gulfcompany.com"
    
    # Generate contact name
    contact_name = f"Contact at {company['name']}"
    
    return {
        'name': company['name'],
        'email': email,
        'contact_name': contact_name,
        'role': 'user',
        'level': random.randint(1, 5),
        'xp': random.randint(0, 1000),
        'industry': company.get('industry', 'Unknown'),
        'location': company.get('location', 'Unknown'),
        'description': f"{company['name']} - {company.get('industry', 'Industrial')} company based in {company.get('location', 'Gulf Region')}",
        'products': company.get('products', ''),
        'processes': company.get('processes', ''),
        'volume': company.get('volume', ''),
        'created_at': datetime.now().isoformat()
    }

def import_gulf_companies_and_materials():
    """Main function to import Gulf companies and generate materials."""
    print("🚀 Starting Gulf Company and Material Import...")
    print("=" * 60)
    
    # Parse company data
    companies = parse_gulf_company_data('gulf_company_data.txt')
    print(f"📊 Found {len(companies)} Gulf companies to import")
    
    imported_companies = 0
    imported_materials = 0
    
    for i, company in enumerate(companies):
        try:
            print(f"🔄 Processing company {i+1}/{len(companies)}: {company['name']}")
            
            # Create company record
            company_record = create_gulf_company_record(company)
            
            # Insert company into database
            result = supabase.table('companies').insert(company_record).execute()
            
            if result.data:
                company_id = result.data[0]['id']
                imported_companies += 1
                
                # Generate and insert materials
                materials = generate_materials_for_gulf_company(company)
                
                for material in materials:
                    material['company_id'] = company_id
                    
                    # Insert material
                    material_result = supabase.table('materials').insert(material).execute()
                    if material_result.data:
                        imported_materials += 1
                
                print(f"  ✅ Imported {len(materials)} materials for {company['name']}")
            else:
                print(f"  ❌ Failed to import company {company['name']}")
                
        except Exception as e:
            print(f"  ❌ Error importing {company['name']}: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("🎉 Import completed!")
    print(f"📈 Companies imported: {imported_companies}")
    print(f"📦 Materials imported: {imported_materials}")
    print(f"📊 Success rate: {(imported_companies/len(companies)*100):.1f}%")

def create_test_application():
    """Create a test application to verify the admin interface."""
    try:
        test_app = {
            'company_name': 'Test Company LLC',
            'contact_email': 'test@example.com',
            'contact_name': 'John Doe',
            'application_answers': {
                'industry': 'Manufacturing',
                'waste_volume': '1000 metric tons annually',
                'sustainability_goals': 'Reduce waste by 50% in 2 years'
            },
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        }
        
        result = supabase.table('company_applications').insert(test_app).execute()
        if result.data:
            print("✅ Test application created successfully")
        else:
            print("❌ Failed to create test application")
            
    except Exception as e:
        print(f"❌ Error creating test application: {str(e)}")

if __name__ == "__main__":
    # Check if Supabase credentials are set
    if SUPABASE_URL == "https://your-project.supabase.co":
        print("❌ ERROR: Please update the Supabase URL and key in this script")
        print("📝 You can find these in your Supabase project settings")
        print("🔧 Or run: python setup_database.py")
        exit(1)
    
    # Import companies and materials
    import_gulf_companies_and_materials()
    
    # Create a test application
    print("\n🧪 Creating test application...")
    create_test_application() 