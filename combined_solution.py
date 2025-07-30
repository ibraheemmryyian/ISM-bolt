#!/usr/bin/env python3
"""
Combined Solution - Starts services and generates material listings
without requiring any external dependencies
"""

import os
import sys
import json
import time
import subprocess
import logging
from datetime import datetime
import random
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('combined_solution.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    print("🚀 SymbioFlows Combined Solution")
    print("📝 This script will start services and generate material listings")
    print("⏰ Started at:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 60)
    
    try:
        # Start services
        print("\n===== STARTING SERVICES =====")
        services = start_services()
        
        # Generate listings
        print("\n===== GENERATING MATERIAL LISTINGS =====")
        company_profile = get_company_profile()
        result = generate_listings(company_profile)
        
        # Save results to file
        output_file = 'onboarding_results.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Display results
        print(f"\nGenerated {len(result.get('predicted_outputs', []))} material listings")
        print(f"Results saved to: {output_file}")
        
        # Print material listings
        print("\n===== GENERATED MATERIAL LISTINGS =====")
        for i, material in enumerate(result.get('predicted_outputs', [])):
            print(f"\n[{i+1}] {material.get('name', 'Unknown Material')}")
            print(f"    Category: {material.get('category', 'Unknown')}")
            print(f"    Quantity: {material.get('quantity', 0)} {material.get('unit', 'units')}")
            print(f"    Quality Grade: {material.get('quality_grade', 'Unknown')}")
            print(f"    Potential Value: ${material.get('potential_value', 0):,.2f}")
            print(f"    Description: {material.get('description', 'No description')}")
        
        # Keep services running
        print("\n===== SERVICES ARE RUNNING =====")
        print("Press Ctrl+C to stop all services")
        
        try:
            while True:
                time.sleep(10)
                check_services_health(services)
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            stop_services(services)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Combined solution failed: {e}")
        sys.exit(1)

def start_services():
    """Start services"""
    logger.info("Starting services...")
    
    services = []
    
    # Start AI service
    try:
        ai_service = subprocess.Popen(
            ['python3', 'ai_service_flask/ai_gateway.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        services.append(('ai_service', ai_service))
        logger.info(f"Started AI service with PID {ai_service.pid}")
        print(f"✅ Started AI service with PID {ai_service.pid}")
    except Exception as e:
        logger.error(f"Failed to start AI service: {e}")
        print(f"❌ Failed to start AI service: {e}")
    
    # Wait for services to start
    time.sleep(5)
    
    return services

def check_services_health(services):
    """Check if services are healthy"""
    for name, process in services:
        if process.poll() is not None:
            logger.error(f"{name} has stopped unexpectedly!")
            print(f"❌ {name} has stopped unexpectedly!")
        else:
            logger.info(f"{name} is running (PID: {process.pid})")

def stop_services(services):
    """Stop services"""
    for name, process in services:
        try:
            logger.info(f"Stopping {name}...")
            process.terminate()
            process.wait(timeout=5)
            logger.info(f"Stopped {name}")
            print(f"✅ Stopped {name}")
        except Exception as e:
            logger.error(f"Failed to stop {name}: {e}")
            print(f"❌ Failed to stop {name}: {e}")
            try:
                process.kill()
                logger.info(f"Killed {name}")
                print(f"✅ Killed {name}")
            except Exception as e:
                logger.error(f"Failed to kill {name}: {e}")
                print(f"❌ Failed to kill {name}: {e}")

def get_company_profile():
    """Get company profile with default values (no user input)"""
    print("\n===== COMPANY PROFILE =====")
    print("Using default company profile...")
    
    company_profile = {
        'name': "Demo Manufacturing Company",
        'industry': "manufacturing",
        'location': "Global",
        'employee_count': 1000,
        'products': ["metal parts", "plastic components", "electronic assemblies"],
        'materials': ["steel", "aluminum", "plastic", "copper"],
        'waste_streams': ["metal scrap", "plastic waste", "industrial dust"],
        'production_volume': "10000 tons",
        'sustainability_goals': "Reduce waste and carbon footprint"
    }
    
    # Print the company profile
    print(f"Company Name: {company_profile['name']}")
    print(f"Industry: {company_profile['industry']}")
    print(f"Location: {company_profile['location']}")
    print(f"Employees: {company_profile['employee_count']}")
    print(f"Products: {', '.join(company_profile['products'])}")
    print(f"Materials: {', '.join(company_profile['materials'])}")
    print(f"Waste Streams: {', '.join(company_profile['waste_streams'])}")
    
    return company_profile

def generate_listings(company_profile):
    """Generate material listings based on company profile"""
    result = {
        'company_name': company_profile.get('name', 'Unknown Company'),
        'industry': company_profile.get('industry', 'Unknown'),
        'location': company_profile.get('location', 'Unknown'),
        'predicted_outputs': [],
        'predicted_inputs': [],
        'generation_metadata': {
            'generated_at': datetime.now().isoformat(),
            'ai_confidence_score': 0.85
        }
    }
    
    # Generate material listings from materials
    for material in company_profile.get('materials', []):
        listing = generate_material_listing(material, company_profile, 'primary')
        result['predicted_outputs'].append(listing)
    
    # Generate material listings from waste streams
    for waste in company_profile.get('waste_streams', []):
        listing = generate_material_listing(waste, company_profile, 'waste')
        result['predicted_outputs'].append(listing)
    
    # Generate some byproduct materials
    byproducts = generate_byproducts(company_profile)
    for byproduct in byproducts:
        listing = generate_material_listing(byproduct, company_profile, 'byproduct')
        result['predicted_outputs'].append(listing)
    
    # Generate some input requirements
    requirements = generate_requirements(company_profile)
    for requirement in requirements:
        result['predicted_inputs'].append(requirement)
    
    # Update metadata
    result['generation_metadata']['total_listings'] = len(result['predicted_outputs'])
    result['generation_metadata']['total_requirements'] = len(result['predicted_inputs'])
    
    return result

def generate_material_listing(material_name, company_profile, material_type):
    """Generate a material listing"""
    industry = company_profile.get('industry', 'manufacturing')
    company_name = company_profile.get('name', 'Unknown Company')
    company_size = company_profile.get('employee_count', 1000)
    
    # Material properties based on type
    material_properties = {
        'primary': {
            'quality_grades': ['A', 'B+', 'B'],
            'value_range': (5000, 15000),
            'quantity_range': (500, 2000),
            'units': ['tons', 'kg'],
            'categories': ['raw_material', 'processed_material', 'specialty_material']
        },
        'waste': {
            'quality_grades': ['B', 'C', 'D'],
            'value_range': (1000, 5000),
            'quantity_range': (200, 1000),
            'units': ['tons', 'kg'],
            'categories': ['waste', 'recyclable', 'byproduct']
        },
        'byproduct': {
            'quality_grades': ['B-', 'C+', 'C'],
            'value_range': (2000, 8000),
            'quantity_range': (100, 500),
            'units': ['tons', 'kg'],
            'categories': ['byproduct', 'secondary_material', 'industrial_output']
        }
    }
    
    properties = material_properties.get(material_type, material_properties['primary'])
    
    # Select random properties
    quality_grade = random.choice(properties['quality_grades'])
    category = random.choice(properties['categories'])
    unit = random.choice(properties['units'])
    
    # Calculate quantity based on company size
    base_quantity = random.uniform(properties['quantity_range'][0], properties['quantity_range'][1])
    size_multiplier = company_size / 1000
    quantity = round(base_quantity * size_multiplier, 2)
    
    # Calculate potential value
    base_value = random.uniform(properties['value_range'][0], properties['value_range'][1])
    value_multiplier = 1.0
    if quality_grade.startswith('A'):
        value_multiplier = 1.5
    elif quality_grade.startswith('B'):
        value_multiplier = 1.2
    potential_value = round(base_value * quantity * value_multiplier, 2)
    
    # Generate description
    description = generate_description(material_name, company_name, industry, material_type, quality_grade)
    
    # Create listing
    listing = {
        'name': material_name.replace('_', ' ').title(),
        'category': category,
        'quantity': quantity,
        'unit': unit,
        'quality_grade': quality_grade,
        'potential_value': potential_value,
        'description': description,
        'potential_uses': generate_potential_uses(material_name, industry),
        'symbiosis_opportunities': generate_symbiosis_opportunities(material_name, industry),
        'notes': f"Generated from {company_name}'s {material_type} materials analysis"
    }
    
    return listing

def generate_description(material_name, company_name, industry, material_type, quality_grade):
    """Generate a description for a material"""
    material_name = material_name.replace('_', ' ').title()
    
    descriptions = {
        'primary': [
            f"High-quality {material_name} from {company_name}'s production process. Meets industry standards for {industry} applications.",
            f"Premium {material_name} produced by {company_name}. Suitable for various {industry} applications requiring {quality_grade}-grade materials.",
            f"Industrial-grade {material_name} with excellent properties for {industry} use. Produced by {company_name} under strict quality control."
        ],
        'waste': [
            f"Recyclable {material_name} from {company_name}'s manufacturing process. Can be repurposed for various applications.",
            f"{material_name} waste stream from {company_name}'s operations. Has potential for recycling and reuse in compatible industries.",
            f"Industrial {material_name} waste from {company_name}. Can be processed for secondary use in compatible applications."
        ],
        'byproduct': [
            f"{material_name} byproduct from {company_name}'s {industry} operations. Consistent quality and regular availability.",
            f"Secondary {material_name} generated during {company_name}'s production process. Suitable for various industrial applications.",
            f"{material_name} byproduct with {quality_grade}-grade quality. Generated consistently from {company_name}'s manufacturing process."
        ]
    }
    
    return random.choice(descriptions.get(material_type, descriptions['primary']))

def generate_potential_uses(material_name, industry):
    """Generate potential uses for a material"""
    general_uses = [
        "Manufacturing input",
        "Construction material",
        "Industrial component",
        "Raw material processing",
        "Secondary manufacturing"
    ]
    
    industry_specific_uses = {
        'manufacturing': [
            "Component fabrication",
            "Assembly line input",
            "Product manufacturing",
            "Industrial tooling"
        ],
        'chemical': [
            "Chemical processing",
            "Reagent production",
            "Catalyst manufacturing",
            "Industrial synthesis"
        ],
        'automotive': [
            "Vehicle components",
            "Automotive manufacturing",
            "Parts production",
            "Assembly materials"
        ],
        'construction': [
            "Building materials",
            "Structural components",
            "Infrastructure development",
            "Construction supplies"
        ]
    }
    
    # Get industry-specific uses
    specific_uses = industry_specific_uses.get(industry.lower(), [])
    
    # Combine general and specific uses
    all_uses = general_uses + specific_uses
    
    # Select random uses
    num_uses = random.randint(2, 4)
    return random.sample(all_uses, min(num_uses, len(all_uses)))

def generate_symbiosis_opportunities(material_name, industry):
    """Generate symbiosis opportunities for a material"""
    opportunities = [
        f"Partnership with {random.choice(['local', 'regional', 'national'])} {random.choice(['manufacturing', 'chemical', 'processing', 'recycling'])} companies",
        f"Material exchange with {random.choice(['complementary', 'adjacent', 'related'])} industries",
        f"Waste-to-resource conversion through {random.choice(['innovative', 'established', 'emerging'])} technologies",
        f"Circular economy integration with {random.choice(['local', 'regional', 'specialized'])} partners",
        f"Resource optimization through {random.choice(['collaborative', 'strategic', 'innovative'])} partnerships"
    ]
    
    # Select random opportunities
    num_opportunities = random.randint(1, 3)
    return random.sample(opportunities, num_opportunities)

def generate_byproducts(company_profile):
    """Generate byproducts based on company profile"""
    industry = company_profile.get('industry', '').lower()
    products = company_profile.get('products', [])
    
    byproduct_mapping = {
        'manufacturing': ['metal_shavings', 'industrial_dust', 'excess_material'],
        'chemical': ['chemical_residue', 'reaction_byproducts', 'process_water'],
        'automotive': ['metal_scraps', 'paint_residue', 'plastic_trim'],
        'construction': ['wood_scraps', 'concrete_debris', 'metal_offcuts'],
        'food': ['organic_waste', 'processing_water', 'packaging_waste'],
        'textile': ['fabric_scraps', 'thread_waste', 'dye_residue']
    }
    
    # Get industry-specific byproducts
    byproducts = byproduct_mapping.get(industry, ['excess_material', 'process_waste'])
    
    # Add product-specific byproducts
    for product in products:
        product_lower = product.lower()
        if 'metal' in product_lower or 'steel' in product_lower:
            byproducts.append('metal_shavings')
        elif 'plastic' in product_lower:
            byproducts.append('plastic_waste')
        elif 'chemical' in product_lower:
            byproducts.append('chemical_residue')
        elif 'wood' in product_lower:
            byproducts.append('sawdust')
    
    # Remove duplicates and limit to 3 byproducts
    unique_byproducts = list(set(byproducts))
    return unique_byproducts[:3]

def generate_requirements(company_profile):
    """Generate input requirements based on company profile"""
    industry = company_profile.get('industry', '').lower()
    materials = company_profile.get('materials', [])
    
    # Base requirements by industry
    requirement_mapping = {
        'manufacturing': ['raw_materials', 'industrial_supplies', 'equipment_parts'],
        'chemical': ['chemical_precursors', 'catalysts', 'laboratory_supplies'],
        'automotive': ['metal_parts', 'electronic_components', 'assembly_materials'],
        'construction': ['building_materials', 'fasteners', 'tools_and_equipment'],
        'food': ['ingredients', 'packaging_materials', 'processing_aids'],
        'textile': ['fabrics', 'dyes', 'thread_and_yarn']
    }
    
    # Get industry-specific requirements
    requirements = []
    industry_reqs = requirement_mapping.get(industry, ['raw_materials', 'supplies'])
    
    for req_name in industry_reqs:
        requirement = {
            'name': req_name.replace('_', ' ').title(),
            'description': f"Essential {req_name.replace('_', ' ')} needed for {industry} operations",
            'category': 'essential_input',
            'quantity_needed': random.randint(100, 1000),
            'unit': random.choice(['tons', 'kg', 'units']),
            'current_cost': random.randint(5000, 20000),
            'priority': random.choice(['high', 'medium', 'low']),
            'potential_sources': [
                f"Local {industry} suppliers",
                "Industrial partners",
                "Specialized vendors"
            ],
            'symbiosis_opportunities': [
                "Material exchange programs",
                "Collaborative procurement",
                "Waste-to-resource partnerships"
            ]
        }
        requirements.append(requirement)
    
    # Add material-specific requirements
    for material in materials:
        material_lower = material.lower()
        if 'metal' in material_lower or 'steel' in material_lower:
            requirements.append({
                'name': 'Metal Processing Equipment',
                'description': f"Specialized equipment for processing {material}",
                'category': 'equipment',
                'quantity_needed': random.randint(1, 10),
                'unit': 'units',
                'current_cost': random.randint(10000, 50000),
                'priority': 'medium',
                'potential_sources': [
                    "Equipment manufacturers",
                    "Industrial suppliers",
                    "Specialized vendors"
                ],
                'symbiosis_opportunities': [
                    "Equipment sharing programs",
                    "Joint procurement initiatives",
                    "Technology partnerships"
                ]
            })
    
    return requirements[:3]  # Limit to 3 requirements

if __name__ == "__main__":
    main()