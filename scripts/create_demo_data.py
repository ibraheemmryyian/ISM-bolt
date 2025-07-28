#!/usr/bin/env python3
"""
Simple Demo Data Creator - No Dependencies Required
Creates demo configuration and marketplace data for video capture
"""

import json
import os
from datetime import datetime

def create_demo_directories():
    """Create necessary directories"""
    os.makedirs("frontend/src/config", exist_ok=True)
    os.makedirs("frontend/src/data", exist_ok=True)
    os.makedirs("data", exist_ok=True)

def create_demo_config():
    """Create demo configuration file"""
    config = {
        "demo_mode": True,
        "fast_ai_responses": True,
        "skip_email_verification": True,
        "auto_generate_portfolio": True,
        "demo_data_prefill": {
            "industry": "Manufacturing",
            "products": "Industrial components and machinery parts for automotive and construction industries",
            "production_volume": "500 tonnes per month",
            "processes": "CNC machining, metal forming, assembly, quality control, surface treatment, packaging"
        }
    }
    
    with open("frontend/src/config/demo-config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created demo configuration")

def create_sample_companies():
    """Create sample company data"""
    companies = [
        {
            "name": "Gulf Advanced Manufacturing",
            "industry": "Manufacturing",
            "location": "Dubai, UAE",
            "employee_count": 250,
            "waste_streams": ["metal shavings", "plastic offcuts", "packaging waste", "defective products"],
            "sustainability_score": 87
        },
        {
            "name": "Emirates Textile Industries",
            "industry": "Textiles",
            "location": "Abu Dhabi, UAE",
            "employee_count": 180,
            "waste_streams": ["fabric scraps", "yarn waste", "dyeing chemicals", "packaging materials"],
            "sustainability_score": 82
        },
        {
            "name": "Qatar Food Processing Co.",
            "industry": "Food & Beverage",
            "location": "Doha, Qatar",
            "employee_count": 120,
            "waste_streams": ["organic waste", "packaging materials", "expired products", "wastewater"],
            "sustainability_score": 79
        },
        {
            "name": "Saudi Chemical Solutions",
            "industry": "Chemicals",
            "location": "Riyadh, Saudi Arabia",
            "employee_count": 320,
            "waste_streams": ["chemical byproducts", "contaminated materials", "solvents", "catalyst waste"],
            "sustainability_score": 91
        },
        {
            "name": "Emirates Construction Materials",
            "industry": "Construction",
            "location": "Sharjah, UAE",
            "employee_count": 450,
            "waste_streams": ["concrete waste", "metal scraps", "wood waste", "insulation materials"],
            "sustainability_score": 74
        }
    ]
    
    with open("data/demo_company_data.json", 'w') as f:
        json.dump(companies, f, indent=2)
    
    print("✅ Created sample company data")
    return companies

def estimate_waste_tonnage(employee_count, industry):
    """Simple waste tonnage estimation"""
    waste_per_employee = {
        'Manufacturing': 2.5,
        'Textiles': 1.8,
        'Food & Beverage': 3.2,
        'Chemicals': 4.1,
        'Construction': 5.8
    }
    
    base_rate = waste_per_employee.get(industry, 2.0)
    annual_tonnes = employee_count * base_rate
    
    return {
        'annual_tonnes': round(annual_tonnes, 1),
        'monthly_tonnes': round(annual_tonnes / 12, 1),
        'weekly_tonnes': round(annual_tonnes / 52, 1)
    }

def create_marketplace_data(companies):
    """Create marketplace data from companies"""
    materials = []
    matches = []
    
    # Material categories and pricing
    material_values = {
        'metal': (200, 800),
        'plastic': (50, 300),
        'organic': (10, 100),
        'chemical': (100, 500),
        'concrete': (30, 80),
        'textile': (20, 200),
        'paper': (40, 120)
    }
    
    material_id = 1
    
    for company in companies:
        waste_estimates = estimate_waste_tonnage(company['employee_count'], company['industry'])
        
        # Create waste materials
        for i, stream in enumerate(company['waste_streams']):
            # Determine category
            category = 'default'
            if any(word in stream.lower() for word in ['metal', 'steel', 'aluminum']):
                category = 'metal'
            elif any(word in stream.lower() for word in ['plastic', 'polymer']):
                category = 'plastic'
            elif any(word in stream.lower() for word in ['organic', 'food', 'wood']):
                category = 'organic'
            elif any(word in stream.lower() for word in ['chemical', 'solvent']):
                category = 'chemical'
            elif any(word in stream.lower() for word in ['concrete', 'cement']):
                category = 'concrete'
            elif any(word in stream.lower() for word in ['fabric', 'textile', 'yarn']):
                category = 'textile'
            elif any(word in stream.lower() for word in ['paper', 'cardboard']):
                category = 'paper'
            
            # Calculate stream quantity (distribute total waste)
            stream_quantity = waste_estimates['monthly_tonnes'] / (len(company['waste_streams']) * (i + 1) * 0.5)
            stream_quantity = max(1.0, round(stream_quantity, 1))
            
            # Get price range and calculate value
            min_price, max_price = material_values.get(category, (25, 150))
            price = round(min_price + (max_price - min_price) * 0.6, 2)
            
            material = {
                "id": f"material_{material_id}",
                "company_id": f"company_{companies.index(company) + 1}",
                "company_name": company['name'],
                "material_name": stream,
                "type": "waste",
                "category": category,
                "quantity": stream_quantity,
                "unit": "tonnes",
                "description": f"High-quality {stream} from {company['industry']} operations in {company['location']}",
                "price_per_unit": price,
                "location": company['location'],
                "availability": "continuous",
                "sustainability_score": 75 + (i * 5),
                "created_at": datetime.now().isoformat()
            }
            materials.append(material)
            material_id += 1
        
        # Create requirements
        industry_requirements = {
            'Manufacturing': [('Steel scrap', 'metal', 50, 400), ('Recycled plastic', 'plastic', 20, 200)],
            'Textiles': [('Cotton waste', 'textile', 15, 150), ('Fabric scraps', 'textile', 25, 100)],
            'Food & Beverage': [('Organic fertilizer', 'organic', 30, 80), ('Recycled packaging', 'paper', 10, 120)],
            'Chemicals': [('Catalyst waste', 'chemical', 5, 600), ('Metal containers', 'metal', 15, 300)],
            'Construction': [('Concrete aggregate', 'concrete', 100, 60), ('Steel reinforcement', 'metal', 30, 500)]
        }
        
        requirements = industry_requirements.get(company['industry'], [('Recycled materials', 'default', 10, 100)])
        for req_name, req_category, req_quantity, max_price in requirements:
            material = {
                "id": f"material_{material_id}",
                "company_id": f"company_{companies.index(company) + 1}",
                "company_name": company['name'],
                "material_name": req_name,
                "type": "requirement",
                "category": req_category,
                "quantity": req_quantity,
                "unit": "tonnes",
                "description": f"Seeking {req_name} for {company['industry']} operations",
                "max_price_per_unit": max_price,
                "location": company['location'],
                "urgency": "medium",
                "created_at": datetime.now().isoformat()
            }
            materials.append(material)
            material_id += 1
    
    # Create matches
    waste_materials = [m for m in materials if m['type'] == 'waste']
    requirement_materials = [m for m in materials if m['type'] == 'requirement']
    
    match_id = 1
    for requirement in requirement_materials[:10]:  # Limit matches for demo
        for waste in waste_materials:
            if (waste['category'] == requirement['category'] and 
                waste['company_id'] != requirement['company_id']):
                
                match_score = 75 + (match_id % 20)  # 75-95 range
                matched_quantity = min(waste['quantity'], requirement['quantity'])
                potential_savings = matched_quantity * waste['price_per_unit'] * 0.7
                
                match = {
                    "id": f"match_{match_id}",
                    "waste_material_id": waste['id'],
                    "requirement_material_id": requirement['id'],
                    "waste_company": waste['company_name'],
                    "requirement_company": requirement['company_name'],
                    "material_name": waste['material_name'],
                    "match_score": match_score,
                    "potential_savings": round(potential_savings, 2),
                    "carbon_reduction": round(potential_savings * 0.2, 2),
                    "matched_quantity": matched_quantity,
                    "status": "pending",
                    "created_at": datetime.now().isoformat()
                }
                matches.append(match)
                match_id += 1
                
                if len(matches) >= 15:  # Limit for demo
                    break
        
        if len(matches) >= 15:
            break
    
    # Create marketplace data structure
    marketplace_data = {
        "materials": materials,
        "matches": matches,
        "companies": [
            {
                "id": f"company_{i+1}",
                "name": company['name'],
                "industry": company['industry'],
                "location": company['location'],
                "sustainability_score": company['sustainability_score'],
                "employee_count": company['employee_count']
            }
            for i, company in enumerate(companies)
        ]
    }
    
    with open("frontend/src/data/demo-marketplace.json", 'w') as f:
        json.dump(marketplace_data, f, indent=2)
    
    print(f"✅ Created marketplace data: {len(materials)} materials, {len(matches)} matches")
    return marketplace_data

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    🎬 DEMO DATA CREATOR                      ║
║              Generate Demo Files for Video Capture          ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    print("🏗️  Creating demo directories...")
    create_demo_directories()
    
    print("⚙️  Creating demo configuration...")
    create_demo_config()
    
    print("🏭 Creating sample companies...")
    companies = create_sample_companies()
    
    print("📦 Generating marketplace data...")
    marketplace_data = create_marketplace_data(companies)
    
    print("\n" + "="*60)
    print("🎬 DEMO SETUP COMPLETE!")
    print("="*60)
    print(f"📊 Companies: {len(companies)}")
    print(f"📦 Materials: {len(marketplace_data['materials'])}")
    print(f"🔗 Matches: {len(marketplace_data['matches'])}")
    print("\n📋 Next Steps:")
    print("1. cd frontend && npm run dev")
    print("2. Navigate to http://localhost:5173")
    print("3. Click 'Get Started'")
    print("4. Complete onboarding (fields auto-filled)")
    print("5. View portfolio and marketplace")
    print("\n🎬 Ready for video capture!")
    print("="*60)

if __name__ == "__main__":
    main()