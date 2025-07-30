#!/usr/bin/env python3
"""
Simple Demo Service for SymbioFlows
Bypasses complex ML dependencies and provides working demo functionality
"""

import json
import csv
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDemoService:
    """Simplified demo service that works without complex ML dependencies"""
    
    def __init__(self):
        self.logger = logger
        self.demo_data = self._load_demo_data()
        
    def _load_demo_data(self) -> Dict[str, Any]:
        """Load demo data for the service"""
        return {
            'materials': [
                {
                    'id': 'mat_001',
                    'name': 'Wood Scraps',
                    'type': 'Organic Waste',
                    'quantity': '500 kg/day',
                    'company': 'Furniture Factory A',
                    'location': 'Germany'
                },
                {
                    'id': 'mat_002', 
                    'name': 'Metal Waste',
                    'type': 'Industrial Waste',
                    'quantity': '200 kg/day',
                    'company': 'Automotive Factory B',
                    'location': 'France'
                },
                {
                    'id': 'mat_003',
                    'name': 'Plastic Waste',
                    'type': 'Packaging Waste',
                    'quantity': '300 kg/day',
                    'company': 'Food Processing C',
                    'location': 'Netherlands'
                }
            ],
            'companies': [
                {
                    'id': 'comp_001',
                    'name': 'Paper Mill X',
                    'industry': 'Paper Manufacturing',
                    'needs': ['Wood Scraps', 'Paper Waste'],
                    'location': 'Germany'
                },
                {
                    'id': 'comp_002',
                    'name': 'Metal Recycler Y',
                    'industry': 'Metal Recycling',
                    'needs': ['Metal Waste', 'Scrap Metal'],
                    'location': 'France'
                },
                {
                    'id': 'comp_003',
                    'name': 'Plastic Recycler Z',
                    'industry': 'Plastic Recycling',
                    'needs': ['Plastic Waste', 'PET Bottles'],
                    'location': 'Netherlands'
                }
            ]
        }
    
    def generate_matches(self, material_id: str) -> List[Dict[str, Any]]:
        """Generate matches for a given material"""
        self.logger.info(f"Generating matches for material {material_id}")
        
        # Find the material
        material = next((m for m in self.demo_data['materials'] if m['id'] == material_id), None)
        if not material:
            return []
        
        matches = []
        for company in self.demo_data['companies']:
            # Simple matching logic
            if material['name'] in company['needs'] or material['type'] in company['needs']:
                match_score = random.uniform(0.7, 0.95)
                matches.append({
                    'match_id': f"match_{material_id}_{company['id']}",
                    'material_id': material_id,
                    'material_name': material['name'],
                    'company_id': company['id'],
                    'company_name': company['name'],
                    'match_score': round(match_score, 3),
                    'match_reason': f"Company {company['name']} needs {material['name']} for {company['industry']}",
                    'potential_revenue': f"€{random.randint(5000, 50000)}/year",
                    'carbon_reduction': f"{random.randint(10, 100)} tons CO2/year",
                    'created_at': datetime.now().isoformat()
                })
        
        return sorted(matches, key=lambda x: x['match_score'], reverse=True)
    
    def get_all_materials(self) -> List[Dict[str, Any]]:
        """Get all available materials"""
        return self.demo_data['materials']
    
    def get_all_companies(self) -> List[Dict[str, Any]]:
        """Get all available companies"""
        return self.demo_data['companies']
    
    def get_match_statistics(self) -> Dict[str, Any]:
        """Get match statistics"""
        total_matches = 0
        total_revenue = 0
        total_carbon_reduction = 0
        
        for material in self.demo_data['materials']:
            matches = self.generate_matches(material['id'])
            total_matches += len(matches)
            for match in matches:
                # Extract numbers from strings
                revenue_str = match['potential_revenue']
                revenue = int(revenue_str.replace('€', '').replace('/year', ''))
                total_revenue += revenue
                
                carbon_str = match['carbon_reduction']
                carbon = int(carbon_str.replace(' tons CO2/year', ''))
                total_carbon_reduction += carbon
        
        return {
            'total_matches': total_matches,
            'total_potential_revenue': f"€{total_revenue:,}/year",
            'total_carbon_reduction': f"{total_carbon_reduction} tons CO2/year",
            'average_match_score': 0.85,
            'materials_processed': len(self.demo_data['materials']),
            'companies_involved': len(self.demo_data['companies'])
        }
    
    def add_material(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new material"""
        material_id = f"mat_{len(self.demo_data['materials']) + 1:03d}"
        new_material = {
            'id': material_id,
            'name': material_data.get('name', 'Unknown Material'),
            'type': material_data.get('type', 'General Waste'),
            'quantity': material_data.get('quantity', '100 kg/day'),
            'company': material_data.get('company', 'Unknown Company'),
            'location': material_data.get('location', 'Unknown Location')
        }
        
        self.demo_data['materials'].append(new_material)
        self.logger.info(f"Added new material: {new_material['name']}")
        
        return {
            'success': True,
            'material_id': material_id,
            'message': f"Material {new_material['name']} added successfully"
        }
    
    def export_data(self, format: str = 'json') -> str:
        """Export demo data"""
        if format == 'csv':
            # Export to CSV
            csv_file = 'demo_data.csv'
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Type', 'ID', 'Name', 'Details'])
                
                for material in self.demo_data['materials']:
                    writer.writerow(['Material', material['id'], material['name'], 
                                   f"{material['type']} - {material['quantity']} - {material['company']}"])
                
                for company in self.demo_data['companies']:
                    writer.writerow(['Company', company['id'], company['name'], 
                                   f"{company['industry']} - {', '.join(company['needs'])}"])
            
            return csv_file
        else:
            # Export to JSON
            json_file = 'demo_data.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(self.demo_data, f, indent=2)
            
            return json_file

def main():
    """Main demo function"""
    print("🚀 SymbioFlows Simple Demo Service")
    print("=" * 50)
    
    service = SimpleDemoService()
    
    # Demo the service
    print("\n📊 Available Materials:")
    materials = service.get_all_materials()
    for material in materials:
        print(f"  • {material['name']} ({material['type']}) - {material['quantity']}")
    
    print("\n🏢 Available Companies:")
    companies = service.get_all_companies()
    for company in companies:
        print(f"  • {company['name']} ({company['industry']})")
    
    print("\n🔗 Generating Matches:")
    for material in materials:
        matches = service.generate_matches(material['id'])
        print(f"\n  {material['name']} matches:")
        for match in matches[:3]:  # Show top 3 matches
            print(f"    → {match['company_name']} (Score: {match['match_score']})")
    
    print("\n📈 Statistics:")
    stats = service.get_match_statistics()
    for key, value in stats.items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 To use this service in your application:")
    print("   from demo_simple_service import SimpleDemoService")
    print("   service = SimpleDemoService()")
    print("   matches = service.generate_matches('mat_001')")

if __name__ == "__main__":
    main()