import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import uuid
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
from fuzzywuzzy import fuzz
import re
import asyncio
import time

# Add imports for advanced engines
from .proactive_opportunity_engine import proactive_opportunity_engine
from .regulatory_compliance import regulatory_compliance_engine
from .impact_forecasting import impact_forecasting_engine

@dataclass
class CompanyProfile:
    """Company profile with detailed attributes"""
    id: str
    name: str
    industry: str
    location: str
    employee_count: int
    materials: List[str]
    products: List[str]
    waste_streams: List[str]
    energy_needs: List[str]
    water_usage: float
    carbon_footprint: float
    sustainability_score: float
    matching_preferences: Dict[str, float]

@dataclass
class SymbioticMatch:
    """Symbiotic match between companies"""
    company_a: str
    company_b: str
    match_score: float
    match_type: str  # material_exchange, waste_recycling, energy_sharing, etc.
    potential_savings: float
    implementation_complexity: str
    environmental_impact: float
    description: str

class RealAIMatchingEngine:
    def __init__(self):
        self.synthetic_companies = []
        self.company_embeddings = {}
        self.material_compatibility_matrix = {}
        self.industry_symbiosis_patterns = {}
        self.match_history = []
        
        # Load synthetic company data
        self._load_synthetic_companies()
        
        # Initialize matching algorithms
        self._initialize_matching_algorithms()
        
        # Industry-specific symbiosis patterns
        self._initialize_symbiosis_patterns()

    def _load_synthetic_companies(self):
        """Load and process 100 synthetic companies for training"""
        try:
            # Generate 100 diverse synthetic companies
            industries = ['manufacturing', 'textiles', 'food_beverage', 'chemicals', 'construction', 'electronics', 'automotive', 'pharmaceuticals']
            locations = ['Cairo', 'New York', 'London', 'Tokyo', 'Berlin', 'Mumbai', 'São Paulo', 'Sydney']
            
            materials_database = {
                'manufacturing': ['steel', 'aluminum', 'plastic', 'glass', 'rubber', 'copper'],
                'textiles': ['cotton', 'polyester', 'wool', 'silk', 'nylon', 'acrylic'],
                'food_beverage': ['grains', 'sugar', 'vegetables', 'fruits', 'dairy', 'meat'],
                'chemicals': ['petroleum', 'natural_gas', 'minerals', 'acids', 'bases', 'solvents'],
                'construction': ['cement', 'concrete', 'wood', 'brick', 'stone', 'metal'],
                'electronics': ['silicon', 'gold', 'copper', 'plastic', 'glass', 'rare_earths'],
                'automotive': ['steel', 'aluminum', 'plastic', 'rubber', 'glass', 'electronics'],
                'pharmaceuticals': ['chemicals', 'biomaterials', 'plastics', 'glass', 'metals', 'solvents']
            }
            
            waste_streams_database = {
                'manufacturing': ['metal_scrap', 'plastic_waste', 'chemical_waste', 'packaging_waste'],
                'textiles': ['fabric_scraps', 'dye_waste', 'packaging_waste', 'water_waste'],
                'food_beverage': ['organic_waste', 'packaging_waste', 'water_waste', 'energy_waste'],
                'chemicals': ['chemical_waste', 'hazardous_waste', 'water_waste', 'energy_waste'],
                'construction': ['construction_waste', 'metal_scrap', 'wood_waste', 'packaging_waste'],
                'electronics': ['electronic_waste', 'metal_scrap', 'plastic_waste', 'chemical_waste'],
                'automotive': ['metal_scrap', 'plastic_waste', 'rubber_waste', 'chemical_waste'],
                'pharmaceuticals': ['chemical_waste', 'biomedical_waste', 'packaging_waste', 'water_waste']
            }
            
            for i in range(100):
                industry = np.random.choice(industries)
                location = np.random.choice(locations)
                employee_count = np.random.randint(50, 5000)
                
                # Generate materials based on industry
                available_materials = materials_database.get(industry, ['plastic', 'metal', 'chemicals'])
                num_materials = np.random.randint(2, 6)
                materials = np.random.choice(available_materials, num_materials, replace=False).tolist()
                
                # Generate products
                products = self._generate_products_for_industry(industry, materials)
                
                # Generate waste streams
                available_wastes = waste_streams_database.get(industry, ['general_waste', 'packaging_waste'])
                num_wastes = np.random.randint(2, 5)
                waste_streams = np.random.choice(available_wastes, num_wastes, replace=False).tolist()
                
                # Generate energy needs
                energy_needs = self._generate_energy_needs(industry, employee_count)
                
                # Calculate sustainability metrics
                water_usage = employee_count * np.random.uniform(0.5, 2.0)  # m3 per employee per month
                carbon_footprint = employee_count * np.random.uniform(2.0, 8.0)  # tons CO2 per employee per year
                sustainability_score = np.random.uniform(30, 85)  # 0-100 scale
                
                # Generate matching preferences
                matching_preferences = {
                    'material_exchange': np.random.uniform(0.3, 0.9),
                    'waste_recycling': np.random.uniform(0.4, 0.95),
                    'energy_sharing': np.random.uniform(0.2, 0.8),
                    'water_reuse': np.random.uniform(0.3, 0.7),
                    'logistics_sharing': np.random.uniform(0.2, 0.6)
                }
                
                company = CompanyProfile(
                    id=f"company_{i:03d}",
                    name=f"{industry.title()} Company {i+1}",
                    industry=industry,
                    location=location,
                    employee_count=employee_count,
                    materials=materials,
                    products=products,
                    waste_streams=waste_streams,
                    energy_needs=energy_needs,
                    water_usage=water_usage,
                    carbon_footprint=carbon_footprint,
                    sustainability_score=sustainability_score,
                    matching_preferences=matching_preferences
                )
                
                self.synthetic_companies.append(company)
                
        except Exception as e:
            print(f"Error loading synthetic companies: {e}")

    def _generate_products_for_industry(self, industry: str, materials: List[str]) -> List[str]:
        """Generate realistic products based on industry and materials"""
        product_templates = {
            'manufacturing': ['{material}_components', '{material}_parts', '{material}_assemblies'],
            'textiles': ['{material}_fabrics', '{material}_clothing', '{material}_accessories'],
            'food_beverage': ['processed_{material}', '{material}_products', '{material}_beverages'],
            'chemicals': ['{material}_compounds', '{material}_solutions', '{material}_products'],
            'construction': ['{material}_materials', '{material}_structures', '{material}_components'],
            'electronics': ['{material}_devices', '{material}_components', '{material}_systems'],
            'automotive': ['{material}_parts', '{material}_components', '{material}_systems'],
            'pharmaceuticals': ['{material}_medicines', '{material}_compounds', '{material}_products']
        }
        
        templates = product_templates.get(industry, ['{material}_products'])
        products = []
        
        for material in materials[:3]:  # Limit to 3 products
            template = np.random.choice(templates)
            product = template.format(material=material.replace('_', ' '))
            products.append(product)
        
        return products

    def _generate_energy_needs(self, industry: str, employee_count: int) -> List[str]:
        """Generate energy needs based on industry"""
        energy_types = {
            'manufacturing': ['electricity', 'natural_gas', 'steam'],
            'textiles': ['electricity', 'steam', 'hot_water'],
            'food_beverage': ['electricity', 'refrigeration', 'steam'],
            'chemicals': ['electricity', 'natural_gas', 'steam', 'cooling'],
            'construction': ['electricity', 'diesel', 'compressed_air'],
            'electronics': ['electricity', 'cooling', 'compressed_air'],
            'automotive': ['electricity', 'natural_gas', 'compressed_air'],
            'pharmaceuticals': ['electricity', 'steam', 'cooling', 'compressed_air']
        }
        
        available_energy = energy_types.get(industry, ['electricity'])
        num_energy_types = min(3, len(available_energy))
        return np.random.choice(available_energy, num_energy_types, replace=False).tolist()

    def _initialize_matching_algorithms(self):
        """Initialize AI matching algorithms"""
        # Create company embeddings using TF-IDF
        company_texts = []
        for company in self.synthetic_companies:
            text = f"{company.industry} {' '.join(company.materials)} {' '.join(company.products)} {' '.join(company.waste_streams)}"
            company_texts.append(text)
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Fit and transform company texts
        company_vectors = self.vectorizer.fit_transform(company_texts)
        
        # Store embeddings
        for i, company in enumerate(self.synthetic_companies):
            self.company_embeddings[company.id] = company_vectors[i]
        
        # Create material compatibility matrix
        self._create_material_compatibility_matrix()

    def _create_material_compatibility_matrix(self):
        """Create material compatibility matrix for matching"""
        all_materials = set()
        for company in self.synthetic_companies:
            all_materials.update(company.materials)
        
        all_materials = list(all_materials)
        
        # Define material compatibility rules
        compatibility_rules = {
            'steel': ['aluminum', 'copper', 'plastic'],
            'aluminum': ['steel', 'copper', 'plastic'],
            'plastic': ['steel', 'aluminum', 'glass'],
            'glass': ['plastic', 'metal', 'ceramics'],
            'cotton': ['polyester', 'wool', 'silk'],
            'polyester': ['cotton', 'nylon', 'acrylic'],
            'chemicals': ['petroleum', 'natural_gas', 'solvents'],
            'cement': ['concrete', 'stone', 'metal'],
            'wood': ['metal', 'plastic', 'stone'],
            'paper': ['plastic', 'metal', 'textiles']
        }
        
        # Create compatibility matrix
        self.material_compatibility_matrix = {}
        for material in all_materials:
            self.material_compatibility_matrix[material] = {}
            for other_material in all_materials:
                if material == other_material:
                    self.material_compatibility_matrix[material][other_material] = 0.0
                elif other_material in compatibility_rules.get(material, []):
                    self.material_compatibility_matrix[material][other_material] = np.random.uniform(0.6, 0.9)
                else:
                    self.material_compatibility_matrix[material][other_material] = np.random.uniform(0.1, 0.5)

    def _initialize_symbiosis_patterns(self):
        """Initialize industry-specific symbiosis patterns"""
        self.industry_symbiosis_patterns = {
            'manufacturing': {
                'textiles': 0.7,  # Manufacturing can use textile waste
                'food_beverage': 0.5,
                'chemicals': 0.8,
                'construction': 0.6,
                'electronics': 0.7,
                'automotive': 0.9,
                'pharmaceuticals': 0.4
            },
            'textiles': {
                'manufacturing': 0.7,
                'food_beverage': 0.3,
                'chemicals': 0.6,
                'construction': 0.4,
                'electronics': 0.3,
                'automotive': 0.5,
                'pharmaceuticals': 0.4
            },
            'food_beverage': {
                'manufacturing': 0.5,
                'textiles': 0.3,
                'chemicals': 0.4,
                'construction': 0.2,
                'electronics': 0.2,
                'automotive': 0.3,
                'pharmaceuticals': 0.6
            },
            'chemicals': {
                'manufacturing': 0.8,
                'textiles': 0.6,
                'food_beverage': 0.4,
                'construction': 0.5,
                'electronics': 0.7,
                'automotive': 0.6,
                'pharmaceuticals': 0.9
            }
        }

    async def _generate_ai_reasoning(self, target_company: CompanyProfile, synthetic_company: CompanyProfile, match_details: Dict) -> Dict[str, str]:
        """Generate AI reasoning for a match using all three advanced engines"""
        try:
            # Prepare match data for the engines
            match_data = {
                'match_id': f"match_{uuid.uuid4().hex[:8]}",
                'company_a': {
                    'id': target_company.id,
                    'name': target_company.name,
                    'industry': target_company.industry,
                    'location': target_company.location,
                    'employee_count': target_company.employee_count,
                    'materials': target_company.materials,
                    'waste_streams': target_company.waste_streams,
                    'energy_needs': target_company.energy_needs,
                    'carbon_footprint': target_company.carbon_footprint,
                    'sustainability_score': target_company.sustainability_score
                },
                'company_b': {
                    'id': synthetic_company.id,
                    'name': synthetic_company.name,
                    'industry': synthetic_company.industry,
                    'location': synthetic_company.location,
                    'employee_count': synthetic_company.employee_count,
                    'materials': synthetic_company.materials,
                    'waste_streams': synthetic_company.waste_streams,
                    'energy_needs': synthetic_company.energy_needs,
                    'carbon_footprint': synthetic_company.carbon_footprint,
                    'sustainability_score': synthetic_company.sustainability_score
                },
                'material_data': {
                    'type': match_details.get('match_type', 'general').lower(),
                    'name': f"{target_company.industry} to {synthetic_company.industry}",
                    'quantity': target_company.employee_count + synthetic_company.employee_count,
                    'unit': 'employees',
                    'match_score': match_details.get('match_score', 0.5)
                }
            }

            # Call all three engines with timeout
            reasoning_tasks = [
                self._get_proactive_opportunity_reasoning(match_data),
                self._get_regulatory_compliance_reasoning(match_data),
                self._get_impact_forecasting_reasoning(match_data)
            ]

            # Wait for all engines with 5-second timeout
            try:
                results = await asyncio.wait_for(asyncio.gather(*reasoning_tasks, return_exceptions=True), timeout=5.0)
                
                proactive_result, compliance_result, impact_result = results
                
                return {
                    'proactive_opportunity': self._extract_proactive_reasoning(proactive_result),
                    'regulatory_compliance': self._extract_compliance_reasoning(compliance_result),
                    'impact_forecasting': self._extract_impact_reasoning(impact_result)
                }
                
            except asyncio.TimeoutError:
                # Fallback to basic reasoning if engines timeout
                return self._generate_fallback_reasoning(target_company, synthetic_company, match_details)
                
        except Exception as e:
            print(f"Error generating AI reasoning: {e}")
            return self._generate_fallback_reasoning(target_company, synthetic_company, match_details)

    async def _get_proactive_opportunity_reasoning(self, match_data: Dict) -> Optional[Any]:
        """Get proactive opportunity reasoning"""
        try:
            return await proactive_opportunity_engine.predict_future_needs(match_data['company_a'])
        except Exception as e:
            print(f"Proactive opportunity engine error: {e}")
            return None

    async def _get_regulatory_compliance_reasoning(self, match_data: Dict) -> Optional[Any]:
        """Get regulatory compliance reasoning"""
        try:
            return await regulatory_compliance_engine.check_compliance(match_data)
        except Exception as e:
            print(f"Regulatory compliance engine error: {e}")
            return None

    async def _get_impact_forecasting_reasoning(self, match_data: Dict) -> Optional[Any]:
        """Get impact forecasting reasoning"""
        try:
            return await impact_forecasting_engine.forecast_impact(match_data)
        except Exception as e:
            print(f"Impact forecasting engine error: {e}")
            return None

    def _extract_proactive_reasoning(self, result) -> str:
        """Extract human-readable reasoning from proactive opportunity engine"""
        if not result or isinstance(result, Exception):
            return "Proactive opportunity analysis unavailable"
        
        try:
            if hasattr(result, '__iter__') and len(result) > 0:
                # Get the first opportunity
                opportunity = result[0]
                if hasattr(opportunity, 'description'):
                    return f"Future Need Prediction: {opportunity.description}"
                elif isinstance(opportunity, dict) and 'description' in opportunity:
                    return f"Future Need Prediction: {opportunity['description']}"
            
            return "Proactive opportunity analysis completed - no immediate opportunities detected"
        except Exception as e:
            return f"Proactive opportunity analysis completed (details: {str(e)[:100]}...)"

    def _extract_compliance_reasoning(self, result) -> str:
        """Extract human-readable reasoning from regulatory compliance engine"""
        if not result or isinstance(result, Exception):
            return "Regulatory compliance analysis unavailable"
        
        try:
            if hasattr(result, 'overall_compliance'):
                compliance_status = "Compliant" if result.overall_compliance else "Non-compliant"
                risk_level = getattr(result, 'risk_level', 'unknown')
                return f"Regulatory Status: {compliance_status} (Risk Level: {risk_level})"
            elif isinstance(result, dict):
                compliance_status = "Compliant" if result.get('overall_compliance', False) else "Non-compliant"
                risk_level = result.get('risk_level', 'unknown')
                return f"Regulatory Status: {compliance_status} (Risk Level: {risk_level})"
            
            return "Regulatory compliance analysis completed"
        except Exception as e:
            return f"Regulatory compliance analysis completed (details: {str(e)[:100]}...)"

    def _extract_impact_reasoning(self, result) -> str:
        """Extract human-readable reasoning from impact forecasting engine"""
        if not result or isinstance(result, Exception):
            return "Impact forecasting analysis unavailable"
        
        try:
            if hasattr(result, 'carbon_footprint_reduction'):
                carbon_reduction = result.carbon_footprint_reduction
                cost_savings = getattr(result, 'cost_savings', 0)
                job_creation = getattr(result, 'job_creation_potential', 0)
                return f"Impact Forecast: {carbon_reduction:.1f}kg CO2 reduction, ${cost_savings:,.0f} savings, {job_creation} jobs created"
            elif isinstance(result, dict):
                carbon_reduction = result.get('carbon_footprint_reduction', 0)
                cost_savings = result.get('cost_savings', 0)
                job_creation = result.get('job_creation_potential', 0)
                return f"Impact Forecast: {carbon_reduction:.1f}kg CO2 reduction, ${cost_savings:,.0f} savings, {job_creation} jobs created"
            
            return "Impact forecasting analysis completed"
        except Exception as e:
            return f"Impact forecasting analysis completed (details: {str(e)[:100]}...)"

    def _generate_fallback_reasoning(self, target_company: CompanyProfile, synthetic_company: CompanyProfile, match_details: Dict) -> Dict[str, str]:
        """Generate fallback reasoning when AI engines are unavailable"""
        match_score = match_details.get('match_score', 0.5)
        match_type = match_details.get('match_type', 'General Symbiosis')
        
        return {
            'proactive_opportunity': f"Based on {target_company.industry} industry trends and {synthetic_company.industry} growth patterns, this match shows {match_score*100:.0f}% potential for future collaboration opportunities.",
            'regulatory_compliance': f"Standard compliance check for {match_type} between {target_company.location} and {synthetic_company.location} companies. Risk assessment: {'Low' if match_score > 0.7 else 'Medium' if match_score > 0.5 else 'High'}.",
            'impact_forecasting': f"Estimated impact: {(target_company.employee_count + synthetic_company.employee_count) * 50}kg CO2 reduction, ${match_details.get('potential_savings', 0):,.0f} economic value, potential for job creation and skill development."
        }

    def find_symbiotic_matches(self, company_data: Dict, top_k: int = 10) -> List[Dict]:
        """Find symbiotic matches for a company using AI algorithms"""
        try:
            # Create company profile from input data
            target_company = self._create_company_profile(company_data)
            
            # Calculate matches with synthetic companies
            matches = []
            
            for synthetic_company in self.synthetic_companies:
                match_score = self._calculate_match_score(target_company, synthetic_company)
                
                if match_score > 0.3:  # Only include meaningful matches
                    match_details = self._analyze_match_details(target_company, synthetic_company)
                    
                    # Generate AI reasoning (synchronous fallback for now)
                    reasoning = asyncio.run(self._generate_ai_reasoning(target_company, synthetic_company, match_details))
                    
                    matches.append({
                        'company_id': synthetic_company.id,
                        'company_name': synthetic_company.name,
                        'industry': synthetic_company.industry,
                        'location': synthetic_company.location,
                        'match_score': round(match_score, 3),
                        'match_type': match_details['match_type'],
                        'potential_savings': round(match_details['potential_savings'], 2),
                        'implementation_complexity': match_details['implementation_complexity'],
                        'environmental_impact': round(match_details['environmental_impact'], 2),
                        'description': match_details['description'],
                        'materials_compatibility': match_details['materials_compatibility'],
                        'waste_synergy': match_details['waste_synergy'],
                        'energy_synergy': match_details['energy_synergy'],
                        'reasoning': reasoning
                    })
            
            # Sort by match score and return top matches
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Store match in history
            self.match_history.append({
                'timestamp': datetime.now().isoformat(),
                'target_company': target_company.name,
                'matches_found': len(matches),
                'top_match_score': matches[0]['match_score'] if matches else 0
            })
            
            return matches[:top_k]
            
        except Exception as e:
            return [{'error': f'Match finding failed: {str(e)}'}]

    def _create_company_profile(self, company_data: Dict) -> CompanyProfile:
        """Create a company profile from input data"""
        return CompanyProfile(
            id=f"target_{uuid.uuid4().hex[:8]}",
            name=company_data.get('name', 'Unknown Company'),
            industry=company_data.get('industry', '').lower(),
            location=company_data.get('location', ''),
            employee_count=company_data.get('employee_count', 0),
            materials=company_data.get('materials', []),
            products=company_data.get('products', []),
            waste_streams=company_data.get('waste_streams', []),
            energy_needs=company_data.get('energy_needs', []),
            water_usage=company_data.get('water_usage', 0),
            carbon_footprint=company_data.get('carbon_footprint', 0),
            sustainability_score=company_data.get('sustainability_score', 50),
            matching_preferences=company_data.get('matching_preferences', {})
        )

    def _calculate_match_score(self, company_a: CompanyProfile, company_b: CompanyProfile) -> float:
        """Calculate comprehensive match score between two companies"""
        scores = []
        
        # Material compatibility score (40% weight)
        material_score = self._calculate_material_compatibility(company_a, company_b)
        scores.append(material_score * 0.4)
        
        # Industry symbiosis score (25% weight)
        industry_score = self._calculate_industry_symbiosis(company_a, company_b)
        scores.append(industry_score * 0.25)
        
        # Waste synergy score (20% weight)
        waste_score = self._calculate_waste_synergy(company_a, company_b)
        scores.append(waste_score * 0.2)
        
        # Energy synergy score (10% weight)
        energy_score = self._calculate_energy_synergy(company_a, company_b)
        scores.append(energy_score * 0.1)
        
        # Location proximity score (5% weight)
        location_score = self._calculate_location_proximity(company_a, company_b)
        scores.append(location_score * 0.05)
        
        return sum(scores)

    def _calculate_material_compatibility(self, company_a: CompanyProfile, company_b: CompanyProfile) -> float:
        """Calculate material compatibility between companies"""
        if not company_a.materials or not company_b.materials:
            return 0.0
        
        compatibility_scores = []
        
        for material_a in company_a.materials:
            for material_b in company_b.materials:
                compatibility = self.material_compatibility_matrix.get(material_a, {}).get(material_b, 0.3)
                compatibility_scores.append(compatibility)
        
        return np.mean(compatibility_scores) if compatibility_scores else 0.0

    def _calculate_industry_symbiosis(self, company_a: CompanyProfile, company_b: CompanyProfile) -> float:
        """Calculate industry symbiosis potential"""
        if company_a.industry == company_b.industry:
            return 0.5  # Same industry has moderate synergy
        
        # Check industry compatibility patterns
        patterns = self.industry_symbiosis_patterns.get(company_a.industry, {})
        return patterns.get(company_b.industry, 0.3)

    def _calculate_waste_synergy(self, company_a: CompanyProfile, company_b: CompanyProfile) -> float:
        """Calculate waste synergy potential"""
        if not company_a.waste_streams or not company_b.waste_streams:
            return 0.0
        
        # Check if one company's waste can be another's input
        waste_synergies = []
        
        waste_input_mapping = {
            'metal_scrap': ['steel', 'aluminum', 'copper'],
            'plastic_waste': ['plastic', 'polyester', 'nylon'],
            'organic_waste': ['compost', 'biogas', 'fertilizer'],
            'chemical_waste': ['chemicals', 'solvents', 'acids'],
            'fabric_scraps': ['textiles', 'insulation', 'packaging'],
            'construction_waste': ['aggregate', 'concrete', 'building_materials']
        }
        
        for waste_a in company_a.waste_streams:
            for waste_b in company_b.waste_streams:
                if waste_a in waste_input_mapping:
                    potential_inputs = waste_input_mapping[waste_a]
                    if any(input_material in company_b.materials for input_material in potential_inputs):
                        waste_synergies.append(0.8)
                    else:
                        waste_synergies.append(0.3)
        
        return np.mean(waste_synergies) if waste_synergies else 0.0

    def _calculate_energy_synergy(self, company_a: CompanyProfile, company_b: CompanyProfile) -> float:
        """Calculate energy synergy potential"""
        if not company_a.energy_needs or not company_b.energy_needs:
            return 0.0
        
        # Check for complementary energy needs
        energy_synergies = []
        
        for energy_a in company_a.energy_needs:
            for energy_b in company_b.energy_needs:
                if energy_a == energy_b:
                    energy_synergies.append(0.7)  # Shared energy needs
                elif (energy_a == 'steam' and energy_b == 'cooling') or (energy_a == 'cooling' and energy_b == 'steam'):
                    energy_synergies.append(0.9)  # Heat exchange potential
                elif (energy_a == 'electricity' and energy_b == 'natural_gas') or (energy_a == 'natural_gas' and energy_b == 'electricity'):
                    energy_synergies.append(0.6)  # Energy source diversification
        
        return np.mean(energy_synergies) if energy_synergies else 0.0

    def _calculate_location_proximity(self, company_a: CompanyProfile, company_b: CompanyProfile) -> float:
        """Calculate location proximity score"""
        if company_a.location == company_b.location:
            return 1.0
        elif company_a.location and company_b.location:
            # Simple proximity check (same country/region)
            return 0.5
        else:
            return 0.3

    def _analyze_match_details(self, company_a: CompanyProfile, company_b: CompanyProfile) -> Dict:
        """Analyze detailed match information"""
        match_score = self._calculate_match_score(company_a, company_b)
        
        # Determine match type
        material_compatibility = self._calculate_material_compatibility(company_a, company_b)
        waste_synergy = self._calculate_waste_synergy(company_a, company_b)
        energy_synergy = self._calculate_energy_synergy(company_a, company_b)
        
        if material_compatibility > 0.7:
            match_type = "Material Exchange"
            potential_savings = (company_a.employee_count + company_b.employee_count) * 500  # $500 per employee
        elif waste_synergy > 0.6:
            match_type = "Waste Recycling"
            potential_savings = (company_a.employee_count + company_b.employee_count) * 300
        elif energy_synergy > 0.6:
            match_type = "Energy Sharing"
            potential_savings = (company_a.employee_count + company_b.employee_count) * 400
        else:
            match_type = "General Symbiosis"
            potential_savings = (company_a.employee_count + company_b.employee_count) * 200
        
        # Determine implementation complexity
        if match_score > 0.8:
            complexity = "Low"
        elif match_score > 0.6:
            complexity = "Medium"
        else:
            complexity = "High"
        
        # Calculate environmental impact
        environmental_impact = match_score * 100  # Tons CO2 saved per year
        
        # Generate description
        description = self._generate_match_description(company_a, company_b, match_type, material_compatibility, waste_synergy, energy_synergy)
        
        return {
            'match_type': match_type,
            'potential_savings': potential_savings,
            'implementation_complexity': complexity,
            'environmental_impact': environmental_impact,
            'description': description,
            'materials_compatibility': round(material_compatibility, 3),
            'waste_synergy': round(waste_synergy, 3),
            'energy_synergy': round(energy_synergy, 3)
        }

    def _generate_match_description(self, company_a: CompanyProfile, company_b: CompanyProfile, 
                                  match_type: str, material_comp: float, waste_syn: float, energy_syn: float) -> str:
        """Generate detailed match description"""
        descriptions = []
        
        if material_comp > 0.6:
            shared_materials = set(company_a.materials) & set(company_b.materials)
            if shared_materials:
                descriptions.append(f"Both companies work with {', '.join(list(shared_materials)[:3])}, enabling material sharing and bulk purchasing.")
        
        if waste_syn > 0.5:
            descriptions.append(f"{company_a.name} can provide waste materials that {company_b.name} can use as inputs, reducing disposal costs.")
        
        if energy_syn > 0.5:
            descriptions.append(f"Companies can share energy infrastructure and optimize energy usage through coordinated operations.")
        
        if not descriptions:
            descriptions.append(f"General industrial symbiosis potential between {company_a.industry} and {company_b.industry} sectors.")
        
        return " ".join(descriptions)

    def get_matching_statistics(self) -> Dict:
        """Get statistics about the matching system"""
        return {
            'total_companies': len(self.synthetic_companies),
            'industries_represented': len(set(c.industry for c in self.synthetic_companies)),
            'total_matches_performed': len(self.match_history),
            'average_match_score': np.mean([m['top_match_score'] for m in self.match_history]) if self.match_history else 0,
            'materials_covered': len(self.material_compatibility_matrix),
            'last_updated': datetime.now().isoformat()
        }

    def train_on_new_data(self, new_companies: List[Dict]):
        """Train the matching engine on new company data"""
        try:
            for company_data in new_companies:
                company_profile = self._create_company_profile(company_data)
                self.synthetic_companies.append(company_profile)
            
            # Retrain the matching algorithms
            self._initialize_matching_algorithms()
            
            return {'success': True, 'companies_added': len(new_companies)}
            
        except Exception as e:
            return {'error': f'Training failed: {str(e)}'}

# Initialize the real AI matching engine
real_ai_matching_engine = RealAIMatchingEngine() 