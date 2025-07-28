#!/usr/bin/env python3
"""
Demo Data Import Service - Production Grade
Intelligently imports real company data and estimates missing quantitative factors
"""

import json
import asyncio
import aiohttp
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IndustryBenchmarks:
    """Industry benchmarks for estimating tonnage from employee count"""
    # Waste generation per employee per year (tonnes)
    waste_per_employee_per_year: Dict[str, float]
    # Production volume per employee (various units)
    production_per_employee: Dict[str, Dict[str, float]]
    # Common waste streams by industry
    typical_waste_streams: Dict[str, List[str]]
    # Conversion factors for different material types
    material_density_factors: Dict[str, float]

class DemoDataImportService:
    """
    Advanced service for importing real company data with intelligent quantitative estimation
    """
    
    def __init__(self):
        self.backend_url = os.environ.get('BACKEND_URL', 'http://localhost:3000')
        self.industry_benchmarks = self._initialize_industry_benchmarks()
        self.imported_companies = []
        self.generated_materials = []
        self.created_matches = []
        
    def _initialize_industry_benchmarks(self) -> IndustryBenchmarks:
        """Initialize comprehensive industry benchmarks for accurate estimation"""
        return IndustryBenchmarks(
            waste_per_employee_per_year={
                'manufacturing': 2.5,
                'textiles': 1.8,
                'food_beverage': 3.2,
                'chemicals': 4.1,
                'construction': 5.8,
                'electronics': 1.2,
                'automotive': 3.5,
                'pharmaceuticals': 0.9,
                'mining': 12.5,
                'energy': 2.1,
                'agriculture': 1.5,
                'logistics': 0.8,
                'services': 0.3,
                'retail': 0.5,
                'default': 2.0
            },
            production_per_employee={
                'manufacturing': {'units_per_month': 500, 'tonnes_per_month': 2.0},
                'textiles': {'metres_per_month': 1000, 'tonnes_per_month': 1.5},
                'food_beverage': {'tonnes_per_month': 3.0, 'units_per_month': 10000},
                'chemicals': {'litres_per_month': 2000, 'tonnes_per_month': 2.5},
                'construction': {'projects_per_month': 2, 'tonnes_per_month': 15.0},
                'electronics': {'units_per_month': 200, 'tonnes_per_month': 0.5},
                'automotive': {'units_per_month': 50, 'tonnes_per_month': 8.0},
                'pharmaceuticals': {'units_per_month': 50000, 'tonnes_per_month': 0.3},
                'default': {'units_per_month': 100, 'tonnes_per_month': 1.0}
            },
            typical_waste_streams={
                'manufacturing': ['metal shavings', 'plastic offcuts', 'packaging waste', 'defective products'],
                'textiles': ['fabric scraps', 'yarn waste', 'chemical wastewater', 'dyeing chemicals'],
                'food_beverage': ['organic waste', 'packaging materials', 'wastewater', 'expired products'],
                'chemicals': ['chemical byproducts', 'contaminated materials', 'solvents', 'catalyst waste'],
                'construction': ['concrete waste', 'metal scraps', 'wood waste', 'insulation materials'],
                'electronics': ['electronic components', 'precious metals', 'plastic casings', 'circuit boards'],
                'automotive': ['metal scrap', 'rubber waste', 'oil waste', 'paint waste', 'glass'],
                'pharmaceuticals': ['expired medicines', 'packaging waste', 'chemical waste', 'laboratory waste'],
                'mining': ['ore tailings', 'rock waste', 'processing chemicals', 'equipment waste'],
                'energy': ['ash waste', 'metal components', 'insulation materials', 'chemical waste'],
                'agriculture': ['organic waste', 'packaging materials', 'chemical containers', 'equipment waste'],
                'default': ['general waste', 'packaging materials', 'office waste', 'equipment waste']
            },
            material_density_factors={
                'metal': 2.5,
                'plastic': 0.9,
                'organic': 0.6,
                'chemical': 1.2,
                'glass': 2.0,
                'wood': 0.7,
                'paper': 0.8,
                'textile': 0.5,
                'electronic': 1.8,
                'rubber': 1.3,
                'concrete': 2.4,
                'default': 1.0
            }
        )

    async def import_real_company_data(self, company_data_file: str, demo_mode: bool = True) -> Dict[str, Any]:
        """
        Import real company data and enhance with intelligent estimations
        """
        logger.info(f"Starting demo data import from {company_data_file}")
        
        try:
            # Load raw company data
            companies = await self._load_company_data(company_data_file)
            if not companies:
                raise ValueError("No valid company data found")
            
            logger.info(f"Loaded {len(companies)} companies for processing")
            
            # Process each company with intelligent enhancement
            enhanced_companies = []
            for i, company in enumerate(companies):
                try:
                    enhanced_company = await self._enhance_company_data(company, i)
                    enhanced_companies.append(enhanced_company)
                    logger.info(f"Enhanced company {i+1}/{len(companies)}: {company.get('name', 'Unknown')}")
                except Exception as e:
                    logger.error(f"Failed to enhance company {company.get('name', 'Unknown')}: {e}")
                    continue
            
            # Import enhanced companies
            import_results = await self._import_enhanced_companies(enhanced_companies)
            
            # Generate material listings for demo
            if demo_mode:
                await self._generate_demo_materials()
                await self._generate_demo_matches()
            
            # Create demo summary
            demo_summary = {
                'success': True,
                'companies_imported': len(self.imported_companies),
                'materials_generated': len(self.generated_materials),
                'matches_created': len(self.created_matches),
                'demo_ready': True,
                'next_steps': [
                    'Navigate to /onboarding to test AI onboarding flow',
                    'Check dashboard for material listings',
                    'Browse marketplace for matches',
                    'Test messaging and transaction flow'
                ]
            }
            
            logger.info("Demo data import completed successfully")
            return demo_summary
            
        except Exception as e:
            logger.error(f"Demo data import failed: {e}")
            raise

    async def _load_company_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and validate company data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'companies' in data:
                companies = data['companies']
            elif isinstance(data, list):
                companies = data
            else:
                raise ValueError("Invalid data format - expected list of companies")
            
            # Basic validation
            valid_companies = []
            for company in companies:
                if self._validate_company_data(company):
                    valid_companies.append(company)
            
            return valid_companies
            
        except Exception as e:
            logger.error(f"Error loading company data: {e}")
            return []

    def _validate_company_data(self, company: Dict[str, Any]) -> bool:
        """Validate basic company data requirements"""
        required_fields = ['name']
        optional_but_useful = ['industry', 'employee_count', 'waste_streams', 'location']
        
        # Check required fields
        for field in required_fields:
            if not company.get(field):
                logger.warning(f"Company missing required field '{field}': {company}")
                return False
        
        return True

    async def _enhance_company_data(self, company: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Enhance company data with intelligent estimations"""
        enhanced = company.copy()
        
        # Normalize industry
        industry = self._normalize_industry(company.get('industry', 'manufacturing'))
        enhanced['industry'] = industry
        
        # Estimate employee count if missing
        if not enhanced.get('employee_count'):
            enhanced['employee_count'] = self._estimate_employee_count(company)
        
        # Estimate waste tonnage based on employees and industry
        waste_estimates = self._estimate_waste_tonnage(
            enhanced['employee_count'], 
            industry, 
            company.get('waste_streams', [])
        )
        enhanced['estimated_waste_tonnage'] = waste_estimates
        
        # Estimate production volume
        production_estimates = self._estimate_production_volume(
            enhanced['employee_count'], 
            industry
        )
        enhanced['estimated_production'] = production_estimates
        
        # Enhance waste streams with realistic quantities
        enhanced_waste_streams = self._enhance_waste_streams(
            company.get('waste_streams', []), 
            industry, 
            waste_estimates
        )
        enhanced['enhanced_waste_streams'] = enhanced_waste_streams
        
        # Add demo-friendly attributes
        enhanced.update({
            'demo_ready': True,
            'onboarding_completed': False,
            'sustainability_score': np.random.randint(65, 95),
            'location': company.get('location', 'Gulf Region'),
            'created_at': datetime.now().isoformat(),
            'demo_index': index
        })
        
        return enhanced

    def _normalize_industry(self, industry: str) -> str:
        """Normalize industry names to match our benchmarks"""
        industry_lower = industry.lower()
        
        industry_mapping = {
            'manufacturing': ['manufacturing', 'production', 'factory'],
            'textiles': ['textile', 'fabric', 'clothing', 'garment'],
            'food_beverage': ['food', 'beverage', 'agriculture', 'farming'],
            'chemicals': ['chemical', 'pharmaceutical', 'cosmetic'],
            'construction': ['construction', 'building', 'engineering'],
            'electronics': ['electronics', 'technology', 'tech', 'computer'],
            'automotive': ['automotive', 'car', 'vehicle', 'transport'],
            'energy': ['energy', 'oil', 'gas', 'power', 'utility'],
            'mining': ['mining', 'extraction', 'mineral'],
            'logistics': ['logistics', 'shipping', 'transport', 'delivery'],
            'services': ['services', 'consulting', 'finance', 'banking'],
            'retail': ['retail', 'trade', 'commerce', 'sales']
        }
        
        for normalized, keywords in industry_mapping.items():
            if any(keyword in industry_lower for keyword in keywords):
                return normalized
        
        return 'manufacturing'  # Default fallback

    def _estimate_employee_count(self, company: Dict[str, Any]) -> int:
        """Estimate employee count based on available company information"""
        # If we have production or revenue indicators, use those
        # Otherwise, use reasonable defaults based on company type
        
        if company.get('revenue'):
            # Rough estimate: $100k revenue per employee
            revenue_str = str(company['revenue']).lower()
            if 'million' in revenue_str:
                revenue_millions = float(revenue_str.split('million')[0].strip())
                return max(10, int(revenue_millions * 10))
            elif 'billion' in revenue_str:
                revenue_billions = float(revenue_str.split('billion')[0].strip())
                return max(100, int(revenue_billions * 1000))
        
        # Default estimates by industry
        industry_defaults = {
            'manufacturing': np.random.randint(50, 300),
            'textiles': np.random.randint(30, 200),
            'food_beverage': np.random.randint(25, 150),
            'chemicals': np.random.randint(40, 250),
            'construction': np.random.randint(20, 400),
            'electronics': np.random.randint(15, 100),
            'automotive': np.random.randint(100, 500),
            'services': np.random.randint(10, 50)
        }
        
        industry = self._normalize_industry(company.get('industry', 'manufacturing'))
        return industry_defaults.get(industry, np.random.randint(25, 150))

    def _estimate_waste_tonnage(self, employee_count: int, industry: str, waste_streams: List[str]) -> Dict[str, float]:
        """Estimate waste tonnage based on employees, industry, and waste streams"""
        
        base_waste_per_employee = self.industry_benchmarks.waste_per_employee_per_year.get(
            industry, self.industry_benchmarks.waste_per_employee_per_year['default']
        )
        
        # Calculate base annual waste
        annual_waste_tonnes = employee_count * base_waste_per_employee
        
        # Adjust based on number of waste streams (more streams = more waste diversity, not necessarily volume)
        stream_count = len(waste_streams) if waste_streams else 3
        stream_multiplier = 1.0 + (stream_count - 3) * 0.1  # 10% per additional stream
        
        annual_waste_tonnes *= max(0.5, min(2.0, stream_multiplier))
        
        # Break down by time periods
        return {
            'annual_tonnes': round(annual_waste_tonnes, 1),
            'monthly_tonnes': round(annual_waste_tonnes / 12, 1),
            'weekly_tonnes': round(annual_waste_tonnes / 52, 1),
            'daily_tonnes': round(annual_waste_tonnes / 365, 2)
        }

    def _estimate_production_volume(self, employee_count: int, industry: str) -> Dict[str, float]:
        """Estimate production volume based on employees and industry"""
        
        benchmarks = self.industry_benchmarks.production_per_employee.get(
            industry, self.industry_benchmarks.production_per_employee['default']
        )
        
        estimated_production = {}
        for unit, per_employee in benchmarks.items():
            monthly_total = employee_count * per_employee
            estimated_production.update({
                f'monthly_{unit}': round(monthly_total, 1),
                f'annual_{unit}': round(monthly_total * 12, 1),
                f'daily_{unit}': round(monthly_total / 30, 2)
            })
        
        return estimated_production

    def _enhance_waste_streams(self, waste_streams: List[str], industry: str, waste_estimates: Dict[str, float]) -> List[Dict[str, Any]]:
        """Enhance waste streams with realistic quantities and categories"""
        
        if not waste_streams:
            waste_streams = self.industry_benchmarks.typical_waste_streams.get(
                industry, self.industry_benchmarks.typical_waste_streams['default']
            )[:4]  # Take top 4 typical streams
        
        total_monthly_waste = waste_estimates['monthly_tonnes']
        enhanced_streams = []
        
        for i, stream in enumerate(waste_streams):
            # Distribute waste across streams (first stream gets more)
            weight = 1.0 / (i + 1)  # First gets 1.0, second gets 0.5, etc.
            
            # Normalize weights
            total_weight = sum(1.0 / (j + 1) for j in range(len(waste_streams)))
            stream_percentage = weight / total_weight
            
            stream_tonnes = total_monthly_waste * stream_percentage
            
            # Determine material category and density
            category = self._categorize_waste_stream(stream)
            density_factor = self.industry_benchmarks.material_density_factors.get(category, 1.0)
            
            enhanced_streams.append({
                'name': stream,
                'category': category,
                'monthly_tonnes': round(stream_tonnes, 2),
                'monthly_volume_m3': round(stream_tonnes / density_factor, 2),
                'annual_tonnes': round(stream_tonnes * 12, 1),
                'estimated_value_per_tonne': self._estimate_material_value(stream, category),
                'recyclability_score': np.random.randint(60, 95),
                'availability': 'continuous'
            })
        
        return enhanced_streams

    def _categorize_waste_stream(self, stream_name: str) -> str:
        """Categorize waste stream into material type"""
        stream_lower = stream_name.lower()
        
        categories = {
            'metal': ['metal', 'steel', 'aluminum', 'copper', 'iron', 'scrap'],
            'plastic': ['plastic', 'polymer', 'polyethylene', 'pvc', 'packaging'],
            'organic': ['organic', 'food', 'biological', 'agricultural', 'wood'],
            'chemical': ['chemical', 'solvent', 'acid', 'catalyst', 'pharmaceutical'],
            'glass': ['glass', 'silica'],
            'textile': ['fabric', 'yarn', 'cotton', 'textile', 'fiber'],
            'electronic': ['electronic', 'circuit', 'component', 'semiconductor'],
            'paper': ['paper', 'cardboard', 'packaging'],
            'rubber': ['rubber', 'tire', 'elastic']
        }
        
        for category, keywords in categories.items():
            if any(keyword in stream_lower for keyword in keywords):
                return category
        
        return 'default'

    def _estimate_material_value(self, stream_name: str, category: str) -> float:
        """Estimate material value per tonne"""
        
        value_ranges = {
            'metal': (200, 800),
            'plastic': (50, 300),
            'organic': (10, 100),
            'chemical': (100, 500),
            'glass': (30, 150),
            'textile': (20, 200),
            'electronic': (500, 2000),
            'paper': (40, 120),
            'rubber': (80, 250),
            'default': (25, 150)
        }
        
        min_val, max_val = value_ranges.get(category, value_ranges['default'])
        return round(np.random.uniform(min_val, max_val), 2)

    async def _import_enhanced_companies(self, companies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import enhanced companies into the system"""
        logger.info(f"Importing {len(companies)} enhanced companies")
        
        # For demo purposes, we'll store them locally and also try to import via API
        self.imported_companies = []
        
        for i, company in enumerate(companies):
            try:
                # Try API import first
                success = await self._import_company_via_api(company)
                if success:
                    company['import_status'] = 'success'
                    company['import_method'] = 'api'
                else:
                    # Store locally for demo
                    company['import_status'] = 'local'
                    company['import_method'] = 'local'
                
                company['import_timestamp'] = datetime.now().isoformat()
                self.imported_companies.append(company)
                
            except Exception as e:
                logger.error(f"Failed to import company {company.get('name')}: {e}")
                continue
        
        logger.info(f"Successfully imported {len(self.imported_companies)} companies")
        return {'imported_count': len(self.imported_companies)}

    async def _import_company_via_api(self, company: Dict[str, Any]) -> bool:
        """Try to import company via API"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(
                    f"{self.backend_url}/api/companies",
                    json=company,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"API import failed for {company.get('name')}: {e}")
            return False

    async def _generate_demo_materials(self):
        """Generate realistic material listings for demo"""
        logger.info("Generating demo material listings")
        
        self.generated_materials = []
        
        for company in self.imported_companies[:10]:  # Demo with first 10 companies
            try:
                # Generate waste materials
                for stream in company.get('enhanced_waste_streams', []):
                    material = {
                        'company_id': company.get('id', f"demo_{company.get('demo_index')}"),
                        'company_name': company['name'],
                        'material_name': stream['name'],
                        'type': 'waste',
                        'category': stream['category'],
                        'quantity': stream['monthly_tonnes'],
                        'unit': 'tonnes',
                        'description': f"High-quality {stream['name']} from {company['industry']} operations",
                        'price_per_unit': stream['estimated_value_per_tonne'],
                        'location': company['location'],
                        'availability': 'continuous',
                        'sustainability_score': stream['recyclability_score'],
                        'created_at': datetime.now().isoformat()
                    }
                    self.generated_materials.append(material)
                
                # Generate some requirements (what they need)
                industry_needs = self._get_industry_requirements(company['industry'])
                for need in industry_needs[:2]:  # Add 2 requirements per company
                    material = {
                        'company_id': company.get('id', f"demo_{company.get('demo_index')}"),
                        'company_name': company['name'],
                        'material_name': need['name'],
                        'type': 'requirement',
                        'category': need['category'],
                        'quantity': need['quantity'],
                        'unit': need['unit'],
                        'description': f"Seeking {need['name']} for {company['industry']} operations",
                        'max_price_per_unit': need['max_price'],
                        'location': company['location'],
                        'urgency': need.get('urgency', 'medium'),
                        'created_at': datetime.now().isoformat()
                    }
                    self.generated_materials.append(material)
                    
            except Exception as e:
                logger.error(f"Failed to generate materials for {company.get('name')}: {e}")
        
        logger.info(f"Generated {len(self.generated_materials)} material listings")

    def _get_industry_requirements(self, industry: str) -> List[Dict[str, Any]]:
        """Get typical material requirements for an industry"""
        
        requirements_by_industry = {
            'manufacturing': [
                {'name': 'Steel scrap', 'category': 'metal', 'quantity': 50, 'unit': 'tonnes', 'max_price': 400},
                {'name': 'Recycled plastic', 'category': 'plastic', 'quantity': 20, 'unit': 'tonnes', 'max_price': 200}
            ],
            'textiles': [
                {'name': 'Cotton waste', 'category': 'textile', 'quantity': 15, 'unit': 'tonnes', 'max_price': 150},
                {'name': 'Fabric scraps', 'category': 'textile', 'quantity': 25, 'unit': 'tonnes', 'max_price': 100}
            ],
            'construction': [
                {'name': 'Concrete aggregate', 'category': 'concrete', 'quantity': 100, 'unit': 'tonnes', 'max_price': 80},
                {'name': 'Steel reinforcement', 'category': 'metal', 'quantity': 30, 'unit': 'tonnes', 'max_price': 500}
            ],
            'default': [
                {'name': 'Recycled materials', 'category': 'default', 'quantity': 10, 'unit': 'tonnes', 'max_price': 100},
                {'name': 'Industrial equipment', 'category': 'equipment', 'quantity': 5, 'unit': 'units', 'max_price': 1000}
            ]
        }
        
        return requirements_by_industry.get(industry, requirements_by_industry['default'])

    async def _generate_demo_matches(self):
        """Generate realistic matches between materials for demo"""
        logger.info("Generating demo matches")
        
        self.created_matches = []
        
        # Simple matching logic for demo
        waste_materials = [m for m in self.generated_materials if m['type'] == 'waste']
        requirement_materials = [m for m in self.generated_materials if m['type'] == 'requirement']
        
        for requirement in requirement_materials:
            for waste in waste_materials:
                # Simple category matching
                if (waste['category'] == requirement['category'] and 
                    waste['company_id'] != requirement['company_id']):
                    
                    match_score = np.random.randint(75, 95)
                    potential_savings = min(waste['quantity'], requirement['quantity']) * waste['price_per_unit'] * 0.7
                    
                    match = {
                        'waste_material_id': waste.get('id', f"waste_{len(self.created_matches)}"),
                        'requirement_material_id': requirement.get('id', f"req_{len(self.created_matches)}"),
                        'waste_company': waste['company_name'],
                        'requirement_company': requirement['company_name'],
                        'material_name': waste['material_name'],
                        'match_score': match_score,
                        'potential_savings': round(potential_savings, 2),
                        'carbon_reduction': round(potential_savings * 0.2, 2),  # Estimate
                        'matched_quantity': min(waste['quantity'], requirement['quantity']),
                        'status': 'pending',
                        'created_at': datetime.now().isoformat()
                    }
                    self.created_matches.append(match)
                    
                    # Limit matches for demo
                    if len(self.created_matches) >= 20:
                        break
            
            if len(self.created_matches) >= 20:
                break
        
        logger.info(f"Generated {len(self.created_matches)} demo matches")

    def get_demo_summary(self) -> Dict[str, Any]:
        """Get comprehensive demo summary"""
        return {
            'demo_ready': True,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'companies_imported': len(self.imported_companies),
                'materials_generated': len(self.generated_materials),
                'matches_created': len(self.created_matches),
                'waste_materials': len([m for m in self.generated_materials if m['type'] == 'waste']),
                'requirement_materials': len([m for m in self.generated_materials if m['type'] == 'requirement'])
            },
            'sample_companies': [
                {
                    'name': company['name'],
                    'industry': company['industry'],
                    'employees': company['employee_count'],
                    'estimated_waste_tonnes_annual': company['estimated_waste_tonnage']['annual_tonnes'],
                    'waste_streams_count': len(company.get('enhanced_waste_streams', []))
                }
                for company in self.imported_companies[:5]
            ],
            'demo_flow_ready': {
                'account_creation': True,
                'ai_onboarding': True,
                'material_listings': True,
                'matching_system': True,
                'messaging': True
            }
        }

# Example usage function
async def main():
    """Example usage for demo setup"""
    service = DemoDataImportService()
    
    # Replace with your actual data file path
    data_file = "data/your_company_data.json"
    
    try:
        result = await service.import_real_company_data(data_file, demo_mode=True)
        print("Demo Import Complete!")
        print(json.dumps(result, indent=2))
        
        # Get detailed summary
        summary = service.get_demo_summary()
        print("\nDemo Summary:")
        print(json.dumps(summary, indent=2))
        
    except Exception as e:
        print(f"Demo setup failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())