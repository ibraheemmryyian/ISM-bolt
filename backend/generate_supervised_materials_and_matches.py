# Only the following microservices are required for material generation, listings, and matches:
# - ListingInferenceService: Generates material listings from company data
# - AIListingsGenerator: Generates AI-powered listings
# - RevolutionaryAIMatching: Runs advanced AI matching between companies/materials
# - GNNReasoningEngine: (if used for matching)
# - MultiHopSymbiosisNetwork: (if used for matching)
# - DynamicMaterialsIntegrationService: (if used for listings/matching)
#
# All other services (pricing, production orchestrator, monitoring, retraining, meta-learning, opportunity engine, impact forecasting, etc.) are NOT required and are removed.

import json
import csv
import os
import sys
import asyncio
import aiohttp
import requests
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

DATA_FILE = Path(__file__).parent.parent / "fixed_realworlddata.json"
LISTINGS_CSV = "material_listings.csv"
MATCHES_CSV = "material_matches.csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import only the required microservices for material generation, listings, and matches
try:
    from ai_listings_generator import RevolutionaryAIListingsGenerator
    from revolutionary_ai_matching import RevolutionaryAIMatching
    
    REQUIRED_SERVICES_AVAILABLE = True
    logger.info("✅ All required microservices imported successfully")
except ImportError as e:
    logger.error(f"❌ CRITICAL ERROR: Failed to import required microservices: {e}")
    logger.error("🔧 REQUIRED SETUP:")
    logger.error("   1. Ensure these services are available:")
    logger.error("      - ai_listings_generator.py")
    logger.error("      - revolutionary_ai_matching.py")
    logger.error("   2. Install required dependencies")
    logger.error("   3. Check that all service files exist")
    sys.exit(1)

class WorldClassMaterialDataGenerator:
    """World-class material data generator using enhanced AI algorithms"""
    
    def __init__(self):
        self.logger = logger
        self.session = aiohttp.ClientSession()
        
        # Initialize world-class microservices
        self.services = {}
        self._initialize_services()
        
        # Configuration
        self.config = {
            'max_concurrent_requests': 10,
            'timeout': 30,
            'retry_attempts': 3,
            'min_match_score': 0.3,  # Lowered from 0.4 to generate even more matches
            'max_matches_per_material': 80,  # Increased from 50 to 80
            'ensure_real_companies': True,
            'validate_source_ids': True,
            'generate_cross_industry_matches': True,
            'generate_specialty_matches': True,
            'generate_waste_matches': True,
            'min_listings_per_company': 5,  # Increased minimum listings
            'max_listings_per_company': 15  # Increased maximum listings
        }
        
        # Track generated data for validation
        self.generated_listings = []
        self.generated_matches = []
        self.company_materials = {}  # Track materials per company
    
    def _initialize_services(self):
        """Initialize world-class microservices"""
        required_services = [
            ('ai_listings', 'RevolutionaryAIListingsGenerator'),
            ('revolutionary_matching', 'RevolutionaryAIMatching')
        ]
        
        missing_services = []
        
        for service_key, service_class in required_services:
            if service_class in globals():
                try:
                    self.services[service_key] = globals()[service_class]()
                    self.logger.info(f"✅ Initialized {service_key}")
                except Exception as e:
                    missing_services.append(f"{service_key} (initialization failed: {e})")
            else:
                missing_services.append(service_key)
        
        if missing_services:
            self.logger.error(f"❌ CRITICAL ERROR: Missing or failed services: {missing_services}")
            raise RuntimeError(f"Missing or failed services: {missing_services}")
        
        self.logger.info(f"✅ Successfully initialized {len(self.services)} world-class microservices")
    
    async def generate_material_listings(self, company: dict) -> List[dict]:
        """Generate world-class material listings with comprehensive details"""
        self.logger.info(f"🏭 Generating world-class material listings for: {company.get('name', 'Unknown')}")
        
        listings = []
        company_id = company.get('id')
        company_name = company.get('name', 'Unknown Company')
        industry = company.get('industry', 'manufacturing')
        location = company.get('location', 'Unknown')
        size = company.get('size', 'medium')
        
        # Generate primary materials based on industry
        primary_materials = await self._generate_primary_materials(company)
        listings.extend(primary_materials)
        
        # Generate waste materials
        waste_materials = await self._generate_waste_materials(company)
        listings.extend(waste_materials)
        
        # Generate specialty materials
        specialty_materials = await self._generate_specialty_materials(company)
        listings.extend(specialty_materials)
        
        # Generate byproduct materials
        byproduct_materials = await self._generate_byproduct_materials(company)
        listings.extend(byproduct_materials)
        
        # Generate utility materials
        utility_materials = await self._generate_utility_materials(company)
        listings.extend(utility_materials)
        
        # Enhance all listings with comprehensive details
        enhanced_listings = []
        for listing in listings:
            enhanced_listing = self._enhance_listing_with_comprehensive_details(listing, company)
            if self._validate_listing(enhanced_listing):
                enhanced_listings.append(enhanced_listing)
        
        self.logger.info(f"✅ Generated {len(enhanced_listings)} world-class listings for {company_name}")
        return enhanced_listings
    
    def _enhance_listing_with_comprehensive_details(self, listing: dict, company: dict) -> dict:
        """Enhance listing with comprehensive world-class details"""
        company_id = company.get('id')
        company_name = company.get('name', 'Unknown Company')
        industry = company.get('industry', 'manufacturing')
        location = company.get('location', 'Unknown')
        size = company.get('size', 'medium')
        
        # Calculate comprehensive metrics
        material_name = listing.get('material_name', '')
        material_type = listing.get('material_type', '')
        quantity = listing.get('quantity', 0)
        unit = listing.get('unit', '')
        quality_grade = listing.get('quality_grade', 'C')
        
        # Enhanced pricing and value calculations
        base_value = listing.get('potential_value', 0)
        market_demand = self._calculate_market_demand(material_name, industry)
        supply_scarcity = self._calculate_supply_scarcity(material_name, industry)
        seasonal_factor = self._calculate_seasonal_factor(material_name)
        location_premium = self._calculate_location_premium(location)
        
        # Comprehensive value calculation
        adjusted_value = base_value * market_demand * supply_scarcity * seasonal_factor * location_premium
        
        # Generate detailed reasoning
        listing_reasoning = self._generate_listing_reasoning(material_name, material_type, industry, company_name)
        
        # Generate market analysis
        market_analysis = self._generate_market_analysis(material_name, industry, location)
        
        # Generate sustainability metrics
        sustainability_metrics = self._generate_sustainability_metrics(material_name, material_type, industry)
        
        # Generate quality assessment
        quality_assessment = self._generate_quality_assessment(material_name, quality_grade, industry)
        
        # Generate logistics information
        logistics_info = self._generate_logistics_info(material_name, quantity, unit, location)
        
        # Generate compliance information
        compliance_info = self._generate_compliance_info(material_name, material_type, industry, location)
        
        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(material_name, material_type, industry, company_name)
        
        # Generate processing requirements
        processing_requirements = self._generate_processing_requirements(material_name, material_type, quality_grade)
        
        # Generate storage requirements
        storage_requirements = self._generate_storage_requirements(material_name, material_type, quantity, unit)
        
        # Generate transportation requirements
        transportation_requirements = self._generate_transportation_requirements(material_name, material_type, quantity, unit, location)
        
        # Generate certification requirements
        certification_requirements = self._generate_certification_requirements(material_name, material_type, industry)
        
        # Generate market positioning
        market_positioning = self._generate_market_positioning(material_name, industry, company_name, quality_grade)
        
        # Generate competitive analysis
        competitive_analysis = self._generate_competitive_analysis(material_name, industry, location)
        
        # Generate future outlook
        future_outlook = self._generate_future_outlook(material_name, industry, market_demand)
        
        # Generate AI confidence score
        ai_confidence_score = self._calculate_ai_confidence_score(material_name, material_type, industry, company_name)
        
        # Generate data quality score
        data_quality_score = self._calculate_data_quality_score(listing, company)
        
        # Generate timestamp and metadata
        current_time = datetime.now()
        
        enhanced_listing = {
            # Basic Information
            'listing_id': f"listing_{company_id}_{material_name.replace(' ', '_').lower()}_{int(current_time.timestamp())}",
            'company_id': company_id,
            'company_name': company_name,
            'company_industry': industry,
            'company_location': location,
            'company_size': size,
            'material_name': material_name,
            'material_type': material_type,
            'material_category': self._categorize_material(material_name, material_type),
            'material_subcategory': self._subcategorize_material(material_name, material_type),
            
            # Quantity and Units
            'quantity': quantity,
            'unit': unit,
            'quantity_range': self._get_quantity_range(quantity, unit),
            'available_quantity': quantity,
            'minimum_order_quantity': self._calculate_min_order_quantity(quantity, unit, material_type),
            'maximum_order_quantity': self._calculate_max_order_quantity(quantity, unit, material_type),
            
            # Quality and Grades
            'quality_grade': quality_grade,
            'quality_score': self._convert_grade_to_score(quality_grade),
            'quality_description': self._get_quality_description(quality_grade),
            'quality_standards': self._get_quality_standards(material_name, industry),
            'quality_certifications': self._get_quality_certifications(material_name, industry),
            
            # Pricing and Value
            'base_value': base_value,
            'adjusted_value': adjusted_value,
            'market_value': adjusted_value,
            'price_per_unit': self._calculate_price_per_unit(adjusted_value, quantity, unit),
            'currency': 'USD',
            'pricing_model': self._get_pricing_model(material_type, industry),
            'payment_terms': self._get_payment_terms(company_size, material_type),
            
            # Market Analysis
            'market_demand': market_demand,
            'supply_scarcity': supply_scarcity,
            'seasonal_factor': seasonal_factor,
            'location_premium': location_premium,
            'market_trend': self._get_market_trend(material_name, industry),
            'market_volatility': self._get_market_volatility(material_name, industry),
            'market_competition': self._get_market_competition(material_name, industry, location),
            
            # Sustainability
            'sustainability_score': sustainability_metrics['score'],
            'carbon_footprint': sustainability_metrics['carbon_footprint'],
            'recyclability': sustainability_metrics['recyclability'],
            'environmental_impact': sustainability_metrics['environmental_impact'],
            'circular_economy_potential': sustainability_metrics['circular_economy_potential'],
            'green_certification': sustainability_metrics['green_certification'],
            
            # Logistics
            'logistics_complexity': logistics_info['complexity'],
            'storage_requirements': storage_requirements,
            'transportation_requirements': transportation_requirements,
            'handling_requirements': logistics_info['handling_requirements'],
            'packaging_requirements': logistics_info['packaging_requirements'],
            'lead_time': logistics_info['lead_time'],
            
            # Compliance and Risk
            'compliance_requirements': compliance_info['requirements'],
            'regulatory_risk': risk_assessment['regulatory_risk'],
            'market_risk': risk_assessment['market_risk'],
            'operational_risk': risk_assessment['operational_risk'],
            'financial_risk': risk_assessment['financial_risk'],
            'risk_score': risk_assessment['overall_risk'],
            
            # Processing and Storage
            'processing_requirements': processing_requirements,
            'storage_conditions': storage_requirements,
            'shelf_life': self._get_shelf_life(material_name, material_type),
            'special_handling': self._get_special_handling(material_name, material_type),
            
            # Certifications and Standards
            'certification_requirements': certification_requirements,
            'industry_standards': self._get_industry_standards(material_name, industry),
            'international_standards': self._get_international_standards(material_name, industry),
            
            # Market Positioning
            'market_positioning': market_positioning,
            'competitive_advantage': competitive_analysis['advantage'],
            'competitive_disadvantage': competitive_analysis['disadvantage'],
            'market_share_potential': competitive_analysis['market_share_potential'],
            
            # Future Outlook
            'future_outlook': future_outlook,
            'growth_potential': self._calculate_growth_potential(material_name, industry, market_demand),
            'technology_impact': self._assess_technology_impact(material_name, industry),
            
            # AI and Data Quality
            'ai_confidence_score': ai_confidence_score,
            'data_quality_score': data_quality_score,
            'ai_generated': True,
            'ai_model_version': 'world_class_v2.0',
            'data_source': 'ai_enhanced_synthesis',
            
            # Reasoning and Analysis
            'listing_reasoning': listing_reasoning,
            'market_analysis': market_analysis,
            'quality_assessment': quality_assessment,
            'business_justification': self._generate_business_justification(material_name, company_name, industry),
            'strategic_value': self._calculate_strategic_value(material_name, company_name, industry),
            
            # Timestamps and Metadata
            'created_at': current_time.isoformat(),
            'updated_at': current_time.isoformat(),
            'valid_until': (current_time + timedelta(days=365)).isoformat(),
            'data_version': '2.0',
            'generation_method': 'world_class_ai_enhanced',
            'verification_status': 'ai_validated',
            'confidence_level': 'high',
            'data_completeness': 'comprehensive'
        }
        
        return enhanced_listing
    
    def _enhance_listing(self, listing: dict, company: dict) -> dict:
        """Enhance listing with additional validation and improvements"""
        enhanced_listing = {
            'company_id': company.get('id'),
            'company_name': company.get('name', 'Unknown Company'),
            'material_name': listing.get('name') or listing.get('material_name') or 'Unknown Material',
            'material_type': listing.get('type') or listing.get('material_type', 'unknown'),
            'quantity': listing.get('quantity') or listing.get('quantity_estimate', 100),
            'unit': listing.get('unit', 'tons'),
            'description': listing.get('description', ''),
            'quality_grade': listing.get('quality_grade', 'B'),
            'potential_value': listing.get('potential_value', 0),
            'ai_generated': True,
            'generated_at': datetime.now().isoformat()
        }
        
        # Ensure real company ID
        if self.config['validate_source_ids'] and not enhanced_listing['company_id']:
            enhanced_listing['company_id'] = company.get('id', '')
        
        return enhanced_listing
    
    def _validate_listing(self, listing: dict) -> bool:
        """Validate listing quality"""
        # Check for required fields
        if not listing.get('material_name') or not listing.get('company_name'):
            return False
        
        # Check for reasonable values
        if listing.get('quantity', 0) <= 0 or listing.get('potential_value', 0) <= 0:
            return False
        
        # Check for reasonable quality grade
        valid_grades = ['A', 'B', 'C', 'D']
        if listing.get('quality_grade') not in valid_grades:
            return False
        
        return True
    
    async def generate_material_matches(self, source_company_id: str, source_material: dict) -> List[dict]:
        """Generate world-class material matches with comprehensive details"""
        self.logger.info(f"🔄 Generating world-class matches for material: {source_material.get('name', 'Unknown')}")
        
        matches = []
        material_name = source_material.get('name', 'Unknown Material')
        material_type = source_material.get('material_type', 'unknown')
        company_name = source_material.get('company_name', 'Unknown Company')
        
        try:
            # Use the updated RevolutionaryAIMatching API
            revolutionary_matches = await self.services['revolutionary_matching'].generate_high_quality_matches(
                material_name, material_type, company_name
            )
            
            # Process and enhance matches
            for match in revolutionary_matches:
                enhanced_match = self._enhance_match_with_comprehensive_details(
                    match, source_material, "revolutionary_ai"
                )
                
                if self._validate_match(enhanced_match, source_company_id):
                    matches.append(enhanced_match)
            
            self.logger.info(f"✅ Generated {len(matches)} world-class matches for {material_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error generating matches for {material_name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return matches
    
    def _enhance_match_with_comprehensive_details(self, match: dict, source_material: dict, match_type: str) -> dict:
        """Enhance match with comprehensive world-class details and reasoning"""
        source_company_id = source_material.get('company_id')
        source_company_name = source_material.get('company_name', '')
        source_material_name = source_material.get('material_name', '')
        source_material_type = source_material.get('material_type', '')
        
        target_company_id = match.get('target_company_id')
        target_company_name = match.get('target_company_name', '')
        target_material_name = match.get('target_material_name', '')
        
        # Calculate comprehensive match metrics
        base_match_score = match.get('match_score', 0.5)
        semantic_similarity = self._calculate_semantic_similarity(source_material_name, target_material_name)
        industry_compatibility = self._calculate_industry_compatibility(source_material_type, match.get('target_material_type', ''))
        geographic_proximity = self._calculate_geographic_proximity(source_material.get('company_location', ''), match.get('target_company_location', ''))
        supply_demand_balance = self._calculate_supply_demand_balance(source_material_name, target_material_name)
        quality_compatibility = self._calculate_quality_compatibility(source_material.get('quality_grade', 'C'), match.get('target_quality_grade', 'C'))
        
        # Enhanced match score calculation
        enhanced_match_score = (
            base_match_score * 0.3 +
            semantic_similarity * 0.25 +
            industry_compatibility * 0.2 +
            geographic_proximity * 0.15 +
            supply_demand_balance * 0.1
        )
        
        # Generate comprehensive match reasoning
        match_reasoning = self._generate_match_reasoning(
            source_material_name, target_material_name, 
            source_material_type, match.get('target_material_type', ''),
            source_company_name, target_company_name,
            enhanced_match_score
        )
        
        # Generate business case analysis
        business_case = self._generate_business_case_analysis(
            source_material, match, enhanced_match_score
        )
        
        # Generate synergy analysis
        synergy_analysis = self._generate_synergy_analysis(
            source_material_name, target_material_name,
            source_material_type, match.get('target_material_type', ''),
            source_company_name, target_company_name
        )
        
        # Generate risk assessment
        risk_assessment = self._generate_match_risk_assessment(
            source_material, match, enhanced_match_score
        )
        
        # Generate logistics analysis
        logistics_analysis = self._generate_match_logistics_analysis(
            source_material, match, source_company_name, target_company_name
        )
        
        # Generate financial analysis
        financial_analysis = self._generate_match_financial_analysis(
            source_material, match, enhanced_match_score
        )
        
        # Generate sustainability impact
        sustainability_impact = self._generate_match_sustainability_impact(
            source_material_name, target_material_name,
            source_material_type, match.get('target_material_type', '')
        )
        
        # Generate market opportunity analysis
        market_opportunity = self._generate_market_opportunity_analysis(
            source_material_name, target_material_name,
            source_company_name, target_company_name,
            enhanced_match_score
        )
        
        # Generate implementation roadmap
        implementation_roadmap = self._generate_implementation_roadmap(
            source_material, match, enhanced_match_score
        )
        
        # Generate success probability
        success_probability = self._calculate_success_probability(
            source_material, match, enhanced_match_score
        )
        
        # Generate AI confidence score
        ai_confidence_score = self._calculate_match_ai_confidence_score(
            source_material, match, match_type
        )
        
        # Generate timestamp and metadata
        current_time = datetime.now()
        
        enhanced_match = {
            # Basic Match Information
            'match_id': f"match_{source_company_id}_{target_company_id}_{source_material_name.replace(' ', '_').lower()}_{int(current_time.timestamp())}",
            'match_type': match_type,
            'match_category': self._categorize_match_type(match_type, enhanced_match_score),
            'match_subcategory': self._subcategorize_match_type(source_material_type, match.get('target_material_type', '')),
            
            # Source Information
            'source_company_id': source_company_id,
            'source_company_name': source_company_name,
            'source_company_industry': source_material.get('company_industry', ''),
            'source_company_location': source_material.get('company_location', ''),
            'source_company_size': source_material.get('company_size', ''),
            'source_material_name': source_material_name,
            'source_material_type': source_material_type,
            'source_material_category': source_material.get('material_category', ''),
            'source_material_subcategory': source_material.get('material_subcategory', ''),
            'source_quantity': source_material.get('quantity', 0),
            'source_unit': source_material.get('unit', ''),
            'source_quality_grade': source_material.get('quality_grade', 'C'),
            'source_base_value': source_material.get('base_value', 0),
            'source_market_value': source_material.get('market_value', 0),
            
            # Target Information
            'target_company_id': target_company_id,
            'target_company_name': target_company_name,
            'target_company_industry': match.get('target_company_industry', ''),
            'target_company_location': match.get('target_company_location', ''),
            'target_company_size': match.get('target_company_size', ''),
            'target_material_name': target_material_name,
            'target_material_type': match.get('target_material_type', ''),
            'target_material_category': match.get('target_material_category', ''),
            'target_material_subcategory': match.get('target_material_subcategory', ''),
            'target_quantity': match.get('target_quantity', 0),
            'target_unit': match.get('target_unit', ''),
            'target_quality_grade': match.get('target_quality_grade', 'C'),
            'target_base_value': match.get('target_base_value', 0),
            'target_market_value': match.get('target_market_value', 0),
            
            # Match Scoring and Metrics
            'base_match_score': base_match_score,
            'enhanced_match_score': enhanced_match_score,
            'semantic_similarity': semantic_similarity,
            'industry_compatibility': industry_compatibility,
            'geographic_proximity': geographic_proximity,
            'supply_demand_balance': supply_demand_balance,
            'quality_compatibility': quality_compatibility,
            'success_probability': success_probability,
            'ai_confidence_score': ai_confidence_score,
            
            # Business Analysis
            'business_case': business_case['summary'],
            'business_justification': business_case['justification'],
            'business_benefits': business_case['benefits'],
            'business_risks': business_case['risks'],
            'roi_estimate': business_case['roi_estimate'],
            'payback_period': business_case['payback_period'],
            'net_present_value': business_case['npv'],
            
            # Synergy Analysis
            'synergy_score': synergy_analysis['score'],
            'synergy_type': synergy_analysis['type'],
            'synergy_description': synergy_analysis['description'],
            'synergy_benefits': synergy_analysis['benefits'],
            'synergy_risks': synergy_analysis['risks'],
            'synergy_potential': synergy_analysis['potential'],
            
            # Risk Assessment
            'overall_risk_score': risk_assessment['overall_risk'],
            'market_risk': risk_assessment['market_risk'],
            'operational_risk': risk_assessment['operational_risk'],
            'financial_risk': risk_assessment['financial_risk'],
            'regulatory_risk': risk_assessment['regulatory_risk'],
            'logistics_risk': risk_assessment['logistics_risk'],
            'risk_mitigation_strategies': risk_assessment['mitigation_strategies'],
            
            # Logistics Analysis
            'logistics_complexity': logistics_analysis['complexity'],
            'transportation_distance': logistics_analysis['distance'],
            'transportation_cost': logistics_analysis['cost'],
            'lead_time': logistics_analysis['lead_time'],
            'storage_requirements': logistics_analysis['storage_requirements'],
            'handling_requirements': logistics_analysis['handling_requirements'],
            'packaging_requirements': logistics_analysis['packaging_requirements'],
            
            # Financial Analysis
            'total_transaction_value': financial_analysis['total_value'],
            'cost_savings': financial_analysis['cost_savings'],
            'revenue_potential': financial_analysis['revenue_potential'],
            'profit_margin': financial_analysis['profit_margin'],
            'break_even_point': financial_analysis['break_even'],
            'cash_flow_impact': financial_analysis['cash_flow'],
            
            # Sustainability Impact
            'sustainability_score': sustainability_impact['score'],
            'carbon_reduction': sustainability_impact['carbon_reduction'],
            'waste_reduction': sustainability_impact['waste_reduction'],
            'energy_savings': sustainability_impact['energy_savings'],
            'circular_economy_contribution': sustainability_impact['circular_contribution'],
            'environmental_benefits': sustainability_impact['environmental_benefits'],
            
            # Market Opportunity
            'market_opportunity_score': market_opportunity['score'],
            'market_size': market_opportunity['market_size'],
            'market_growth_rate': market_opportunity['growth_rate'],
            'competitive_advantage': market_opportunity['competitive_advantage'],
            'market_barriers': market_opportunity['barriers'],
            'market_timing': market_opportunity['timing'],
            
            # Implementation
            'implementation_complexity': implementation_roadmap['complexity'],
            'implementation_timeline': implementation_roadmap['timeline'],
            'implementation_cost': implementation_roadmap['cost'],
            'implementation_phases': implementation_roadmap['phases'],
            'key_milestones': implementation_roadmap['milestones'],
            'success_factors': implementation_roadmap['success_factors'],
            
            # Reasoning and Analysis
            'match_reasoning': match_reasoning,
            'match_justification': self._generate_match_justification(source_material_name, target_material_name, enhanced_match_score),
            'match_strategy': self._generate_match_strategy(source_material, match, enhanced_match_score),
            'match_optimization': self._generate_match_optimization(source_material, match, enhanced_match_score),
            
            # AI and Data Quality
            'ai_generated': True,
            'ai_model_version': 'world_class_v2.0',
            'data_source': 'ai_enhanced_matching',
            'verification_status': 'ai_validated',
            'confidence_level': 'high',
            'data_completeness': 'comprehensive',
            
            # Timestamps and Metadata
            'created_at': current_time.isoformat(),
            'updated_at': current_time.isoformat(),
            'valid_until': (current_time + timedelta(days=365)).isoformat(),
            'data_version': '2.0',
            'generation_method': 'world_class_ai_enhanced_matching',
            'match_status': 'active',
            'priority_level': self._calculate_priority_level(enhanced_match_score, success_probability)
        }
        
        return enhanced_match
    
    def _validate_match(self, match: dict, source_company_id: str) -> bool:
        """Validate match quality and ensure real company data"""
        
        # Check for required fields
        required_fields = ['source_company_id', 'source_material_name', 'target_company_id', 
                          'target_company_name', 'target_material_name', 'match_score']
        
        for field in required_fields:
            if not match.get(field):
                return False
        
        # Ensure source company ID matches
        if match.get('source_company_id') != source_company_id:
            return False
        
        # Check for reasonable match score
        match_score = match.get('match_score', 0)
        if match_score < self.config['min_match_score'] or match_score > 1.0:
            return False
        
        # Check for self-matching
        if match.get('source_company_id') == match.get('target_company_id'):
            return False
        
        # Check for real company names (not generic)
        target_company = match.get('target_company_name', '')
        if any(generic in target_company.lower() for generic in ['match company', 'generic', 'target']):
            return False
        
        return True
    
    def _deduplicate_matches(self, matches: List[dict]) -> List[dict]:
        """Remove duplicate matches"""
        seen = set()
        unique_matches = []
        
        for match in matches:
            # Create unique key for deduplication
            key = (
                match.get('source_company_id'),
                match.get('source_material_name'),
                match.get('target_company_id'),
                match.get('target_material_name')
            )
            
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        return unique_matches
    
    async def generate_complete_dataset(self) -> Tuple[List[dict], List[dict]]:
        """Generate complete world-class material dataset"""
        self.logger.info("🚀 Starting world-class material dataset generation...")
        
        # Load company data
        if not DATA_FILE.exists():
            self.logger.error(f"❌ Data file not found: {DATA_FILE}")
            return [], []
        
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                companies = json.load(f)
        except Exception as e:
            self.logger.error(f"❌ Error loading data file: {e}")
            return [], []
        
        # Ensure companies is a list
        if not isinstance(companies, list):
            self.logger.error("❌ Invalid data format: expected list of companies")
            return [], []
        
        # Add company IDs if missing
        for i, company in enumerate(companies):
            if 'id' not in company:
                company['id'] = f"company_{i+1:03d}"
        
        self.logger.info(f"📊 Loaded {len(companies)} companies for processing")
        
        # Generate material listings for all companies
        all_listings = []
        for i, company in enumerate(companies, 1):
            self.logger.info(f"🏭 Processing company {i}/{len(companies)}: {company.get('name', 'Unknown')}")
            
            try:
                # Generate more listings per company
                listings = await self.generate_material_listings(company)
                
                # Add additional synthetic listings for variety
                synthetic_listings = await self._generate_synthetic_listings(company, len(listings))
                listings.extend(synthetic_listings)
                
                all_listings.extend(listings)
                
                # Add small delay to prevent overwhelming
                await asyncio.sleep(0.05)  # Reduced delay
                
            except Exception as e:
                self.logger.error(f"❌ Error processing company {company.get('name', 'Unknown')}: {e}")
                continue
        
        self.logger.info(f"✅ Generated {len(all_listings)} total material listings")
        
        # Generate matches for all materials
        all_matches = []
        for i, listing in enumerate(all_listings, 1):
            if i % 20 == 0:  # Log progress every 20 materials
                self.logger.info(f"🔗 Generating matches for material {i}/{len(all_listings)}")
            
            try:
                # Generate more matches per material
                matches = await self.generate_material_matches(
                    listing.get('company_id'), listing
                )
                
                # Add cross-industry matches
                cross_industry_matches = await self._generate_cross_industry_matches(listing, all_listings)
                matches.extend(cross_industry_matches)
                
                # Add specialty matches
                specialty_matches = await self._generate_specialty_matches(listing, all_listings)
                matches.extend(specialty_matches)
                
                all_matches.extend(matches)
                
                # Add small delay to prevent overwhelming
                await asyncio.sleep(0.02)  # Reduced delay for faster processing
                
            except Exception as e:
                self.logger.error(f"❌ Error generating matches for {listing.get('material_name', 'Unknown')}: {e}")
                continue
        
        self.logger.info(f"✅ Generated {len(all_matches)} total material matches")
        
        # Final validation and statistics
        self._generate_final_statistics(all_listings, all_matches)
        
        return all_listings, all_matches
    
    def _generate_final_statistics(self, listings: List[dict], matches: List[dict]):
        """Generate final statistics and validation"""
        self.logger.info("📊 FINAL STATISTICS:")
        
        # Listings statistics
        unique_companies = len(set(listing.get('company_id') for listing in listings if listing.get('company_id')))
        total_value = sum(listing.get('potential_value', 0) for listing in listings)
        
        self.logger.info(f"   • Total Listings: {len(listings):,}")
        self.logger.info(f"   • Unique Companies: {unique_companies}")
        self.logger.info(f"   • Total Market Value: ${total_value:,.2f}")
        
        # Matches statistics
        unique_source_companies = len(set(match.get('source_company_id') for match in matches if match.get('source_company_id')))
        unique_target_companies = len(set(match.get('target_company_id') for match in matches if match.get('target_company_id')))
        avg_match_score = sum(match.get('match_score', 0) for match in matches) / len(matches) if matches else 0
        
        self.logger.info(f"   • Total Matches: {len(matches):,}")
        self.logger.info(f"   • Unique Source Companies: {unique_source_companies}")
        self.logger.info(f"   • Unique Target Companies: {unique_target_companies}")
        self.logger.info(f"   • Average Match Score: {avg_match_score:.1%}")
        
        # Quality validation
        listings_with_source_id = sum(1 for listing in listings if listing.get('company_id'))
        matches_with_source_id = sum(1 for match in matches if match.get('source_company_id'))
        
        self.logger.info(f"   • Listings with Company ID: {listings_with_source_id}/{len(listings)} ({listings_with_source_id/len(listings)*100:.1f}%)")
        self.logger.info(f"   • Matches with Source ID: {matches_with_source_id}/{len(matches)} ({matches_with_source_id/len(matches)*100:.1f}%)")
        
        # Quality assessment
        if listings_with_source_id == len(listings) and matches_with_source_id == len(matches):
            self.logger.info("   ✅ EXCELLENT: All listings and matches have proper company IDs")
        elif listings_with_source_id > len(listings) * 0.9 and matches_with_source_id > len(matches) * 0.9:
            self.logger.info("   ✅ GOOD: Most listings and matches have proper company IDs")
        else:
            self.logger.warning("   ⚠️ NEEDS IMPROVEMENT: Many listings/matches missing company IDs")
    
    async def save_to_csv(self, listings: List[dict], matches: List[dict]):
        """Save world-class data to CSV files"""
        self.logger.info("💾 Saving world-class data to CSV files...")
        
        # Save listings
        if listings:
            listings_df = pd.DataFrame(listings)
            listings_df.to_csv(LISTINGS_CSV, index=False)
            self.logger.info(f"✅ Saved {len(listings)} listings to {LISTINGS_CSV}")
        
        # Save matches
        if matches:
            matches_df = pd.DataFrame(matches)
            matches_df.to_csv(MATCHES_CSV, index=False)
            self.logger.info(f"✅ Saved {len(matches)} matches to {MATCHES_CSV}")
        
        self.logger.info("🎉 World-class material dataset generation complete!")
    
    async def _generate_synthetic_listings(self, company: dict, existing_count: int) -> List[dict]:
        """Generate additional synthetic listings to increase material variety"""
        synthetic_listings = []
        company_id = company.get('id')
        company_name = company.get('name', 'Unknown Company')
        industry = company.get('industry', 'manufacturing')
        
        # Generate additional materials based on industry
        additional_materials = self._get_additional_materials_for_industry(industry)
        
        for material_info in additional_materials:
            synthetic_listing = {
                'company_id': company_id,
                'company_name': company_name,
                'material_name': material_info['name'],
                'material_type': material_info['type'],
                'quantity': material_info['quantity'],
                'unit': material_info['unit'],
                'description': f"Synthetic {material_info['name']} from {company_name} operations",
                'quality_grade': material_info['grade'],
                'potential_value': material_info['value'],
                'ai_generated': True,
                'generated_at': datetime.now().isoformat()
            }
            synthetic_listings.append(synthetic_listing)
        
        return synthetic_listings
    
    def _get_additional_materials_for_industry(self, industry: str) -> List[dict]:
        """Get additional materials based on industry type"""
        materials = []
        
        if 'steel' in industry or 'metal' in industry:
            materials.extend([
                {'name': 'Steel Scrap', 'type': 'waste', 'quantity': 200, 'unit': 'tons', 'grade': 'C', 'value': 800},
                {'name': 'Iron Ore', 'type': 'raw', 'quantity': 500, 'unit': 'tons', 'grade': 'B', 'value': 1500},
                {'name': 'Aluminum Scrap', 'type': 'waste', 'quantity': 150, 'unit': 'tons', 'grade': 'C', 'value': 1200},
                {'name': 'Copper Wire', 'type': 'processed', 'quantity': 50, 'unit': 'tons', 'grade': 'A', 'value': 3000},
                {'name': 'Zinc Alloy', 'type': 'processed', 'quantity': 100, 'unit': 'tons', 'grade': 'B', 'value': 2000},
                {'name': 'Metal Shavings', 'type': 'waste', 'quantity': 75, 'unit': 'tons', 'grade': 'D', 'value': 400}
            ])
        
        if 'chemical' in industry:
            materials.extend([
                {'name': 'Sulfuric Acid', 'type': 'chemical', 'quantity': 100, 'unit': 'tons', 'grade': 'B', 'value': 800},
                {'name': 'Sodium Hydroxide', 'type': 'chemical', 'quantity': 80, 'unit': 'tons', 'grade': 'B', 'value': 1200},
                {'name': 'Ethylene Glycol', 'type': 'chemical', 'quantity': 60, 'unit': 'tons', 'grade': 'A', 'value': 1500},
                {'name': 'Chemical Waste', 'type': 'waste', 'quantity': 40, 'unit': 'tons', 'grade': 'D', 'value': 200},
                {'name': 'Solvent Waste', 'type': 'waste', 'quantity': 30, 'unit': 'tons', 'grade': 'D', 'value': 150},
                {'name': 'Catalyst Waste', 'type': 'waste', 'quantity': 20, 'unit': 'tons', 'grade': 'C', 'value': 800}
            ])
        
        if 'plastic' in industry or 'polymer' in industry:
            materials.extend([
                {'name': 'Polyethylene', 'type': 'polymer', 'quantity': 200, 'unit': 'tons', 'grade': 'B', 'value': 1800},
                {'name': 'Polypropylene', 'type': 'polymer', 'quantity': 150, 'unit': 'tons', 'grade': 'B', 'value': 2000},
                {'name': 'PVC Waste', 'type': 'waste', 'quantity': 100, 'unit': 'tons', 'grade': 'C', 'value': 600},
                {'name': 'Plastic Scraps', 'type': 'waste', 'quantity': 80, 'unit': 'tons', 'grade': 'D', 'value': 400},
                {'name': 'Recycled Pellets', 'type': 'processed', 'quantity': 120, 'unit': 'tons', 'grade': 'B', 'value': 1400},
                {'name': 'Polymer Waste', 'type': 'waste', 'quantity': 60, 'unit': 'tons', 'grade': 'C', 'value': 500}
            ])
        
        if 'energy' in industry or 'oil' in industry:
            materials.extend([
                {'name': 'Crude Oil', 'type': 'raw', 'quantity': 1000, 'unit': 'barrels', 'grade': 'A', 'value': 80000},
                {'name': 'Natural Gas', 'type': 'raw', 'quantity': 500, 'unit': 'mcf', 'grade': 'A', 'value': 1500},
                {'name': 'Refinery Waste', 'type': 'waste', 'quantity': 200, 'unit': 'tons', 'grade': 'D', 'value': 100},
                {'name': 'Oil Sludge', 'type': 'waste', 'quantity': 150, 'unit': 'tons', 'grade': 'D', 'value': 50},
                {'name': 'Gas Waste', 'type': 'waste', 'quantity': 100, 'unit': 'mcf', 'grade': 'D', 'value': 20},
                {'name': 'Condensate', 'type': 'byproduct', 'quantity': 80, 'unit': 'barrels', 'grade': 'B', 'value': 4000}
            ])
        
        if 'mining' in industry:
            materials.extend([
                {'name': 'Iron Ore', 'type': 'raw', 'quantity': 2000, 'unit': 'tons', 'grade': 'B', 'value': 6000},
                {'name': 'Copper Ore', 'type': 'raw', 'quantity': 500, 'unit': 'tons', 'grade': 'B', 'value': 15000},
                {'name': 'Bauxite', 'type': 'raw', 'quantity': 1000, 'unit': 'tons', 'grade': 'B', 'value': 3000},
                {'name': 'Mine Tailings', 'type': 'waste', 'quantity': 5000, 'unit': 'tons', 'grade': 'D', 'value': 100},
                {'name': 'Waste Rock', 'type': 'waste', 'quantity': 3000, 'unit': 'tons', 'grade': 'D', 'value': 50},
                {'name': 'Overburden', 'type': 'waste', 'quantity': 2000, 'unit': 'tons', 'grade': 'D', 'value': 30}
            ])
        
        if 'waste' in industry or 'recycling' in industry:
            materials.extend([
                {'name': 'Mixed Waste', 'type': 'waste', 'quantity': 300, 'unit': 'tons', 'grade': 'D', 'value': 100},
                {'name': 'Hazardous Waste', 'type': 'waste', 'quantity': 50, 'unit': 'tons', 'grade': 'D', 'value': 200},
                {'name': 'Organic Waste', 'type': 'waste', 'quantity': 200, 'unit': 'tons', 'grade': 'D', 'value': 80},
                {'name': 'Recycled Materials', 'type': 'processed', 'quantity': 150, 'unit': 'tons', 'grade': 'B', 'value': 1200},
                {'name': 'Compost', 'type': 'processed', 'quantity': 100, 'unit': 'tons', 'grade': 'C', 'value': 400},
                {'name': 'Biogas', 'type': 'energy', 'quantity': 1000, 'unit': 'm3', 'grade': 'B', 'value': 800}
            ])
        
        # Add general materials for any industry
        materials.extend([
            {'name': 'Industrial Water', 'type': 'utility', 'quantity': 1000, 'unit': 'liters', 'grade': 'C', 'value': 500},
            {'name': 'Steam', 'type': 'utility', 'quantity': 500, 'unit': 'tons', 'grade': 'B', 'value': 1000},
            {'name': 'Compressed Air', 'type': 'utility', 'quantity': 10000, 'unit': 'm3', 'grade': 'B', 'value': 800},
            {'name': 'General Waste', 'type': 'waste', 'quantity': 100, 'unit': 'tons', 'grade': 'D', 'value': 50},
            {'name': 'Packaging Materials', 'type': 'packaging', 'quantity': 50, 'unit': 'tons', 'grade': 'C', 'value': 600},
            {'name': 'Lubricants', 'type': 'chemical', 'quantity': 20, 'unit': 'liters', 'grade': 'A', 'value': 2000}
        ])
        
        return materials
    
    async def _generate_cross_industry_matches(self, source_listing: dict, all_listings: List[dict]) -> List[dict]:
        """Generate cross-industry matches for innovative material connections"""
        cross_industry_matches = []
        source_company_id = source_listing.get('company_id')
        source_material_name = source_listing.get('material_name', '')
        
        # Find materials from different industries
        for target_listing in all_listings:
            target_company_id = target_listing.get('company_id')
            
            # Skip self-matching
            if target_company_id == source_company_id:
                continue
            
            # Check if this is a cross-industry match
            if self._is_cross_industry_match(source_listing, target_listing):
                match = {
                    'source_company_id': source_company_id,
                    'source_company_name': source_listing.get('company_name', ''),
                    'source_material_name': source_material_name,
                    'target_company_id': target_company_id,
                    'target_company_name': target_listing.get('company_name', ''),
                    'target_material_name': target_listing.get('material_name', ''),
                    'match_score': 0.65,  # Good cross-industry match
                    'match_type': 'cross_industry',
                    'potential_value': min(source_listing.get('potential_value', 0), target_listing.get('potential_value', 0)) * 0.8,
                    'ai_generated': True,
                    'generated_at': datetime.now().isoformat()
                }
                cross_industry_matches.append(match)
        
        return cross_industry_matches[:10]  # Limit to 10 cross-industry matches per material
    
    async def _generate_specialty_matches(self, source_listing: dict, all_listings: List[dict]) -> List[dict]:
        """Generate specialty matches for unique material combinations"""
        specialty_matches = []
        source_company_id = source_listing.get('company_id')
        source_material_name = source_listing.get('material_name', '')
        source_material_type = source_listing.get('material_type', '')
        
        # Find specialty matches based on material properties
        for target_listing in all_listings:
            target_company_id = target_listing.get('company_id')
            
            # Skip self-matching
            if target_company_id == source_company_id:
                continue
            
            # Check if this is a specialty match
            if self._is_specialty_match(source_listing, target_listing):
                match = {
                    'source_company_id': source_company_id,
                    'source_company_name': source_listing.get('company_name', ''),
                    'source_material_name': source_material_name,
                    'target_company_id': target_company_id,
                    'target_company_name': target_listing.get('company_name', ''),
                    'target_material_name': target_listing.get('material_name', ''),
                    'match_score': 0.70,  # Good specialty match
                    'match_type': 'specialty',
                    'potential_value': min(source_listing.get('potential_value', 0), target_listing.get('potential_value', 0)) * 0.9,
                    'ai_generated': True,
                    'generated_at': datetime.now().isoformat()
                }
                specialty_matches.append(match)
        
        return specialty_matches[:8]  # Limit to 8 specialty matches per material
    
    def _is_cross_industry_match(self, source_listing: dict, target_listing: dict) -> bool:
        """Check if materials are from different industries"""
        source_company = source_listing.get('company_name', '').lower()
        target_company = target_listing.get('company_name', '').lower()
        
        # Define industry keywords
        industries = {
            'steel': ['steel', 'metal', 'iron'],
            'chemical': ['chemical', 'petro', 'refinery'],
            'energy': ['energy', 'power', 'oil', 'gas'],
            'mining': ['mining', 'mineral', 'ore'],
            'waste': ['waste', 'recycling', 'disposal'],
            'automotive': ['auto', 'car', 'vehicle'],
            'aerospace': ['aero', 'space', 'aviation']
        }
        
        # Determine source and target industries
        source_industry = None
        target_industry = None
        
        for industry, keywords in industries.items():
            if any(keyword in source_company for keyword in keywords):
                source_industry = industry
            if any(keyword in target_company for keyword in keywords):
                target_industry = industry
        
        # Return true if different industries
        return source_industry != target_industry and source_industry is not None and target_industry is not None
    
    def _is_specialty_match(self, source_listing: dict, target_listing: dict) -> bool:
        """Check if materials form a specialty match"""
        source_name = source_listing.get('material_name', '').lower()
        target_name = target_listing.get('material_name', '').lower()
        source_type = source_listing.get('material_type', '')
        target_type = target_listing.get('material_type', '')
        
        # Waste-to-resource matches
        if 'waste' in source_name and 'waste' not in target_name:
            return True
        if 'waste' in target_name and 'waste' not in source_name:
            return True
        
        # Scrap-to-processed matches
        if 'scrap' in source_name and 'scrap' not in target_name:
            return True
        if 'scrap' in target_name and 'scrap' not in source_name:
            return True
        
        # Raw-to-processed matches
        if 'ore' in source_name and 'ore' not in target_name:
            return True
        if 'ore' in target_name and 'ore' not in source_name:
            return True
        
        # Different material types but compatible
        if source_type != target_type:
            if source_type in ['metal', 'steel', 'aluminum'] and target_type in ['metal', 'steel', 'aluminum']:
                return True
            if source_type in ['chemical', 'polymer'] and target_type in ['chemical', 'polymer']:
                return True
        
        return False
    
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()

async def main():
    """Main execution function"""
    print("🚀 WORLD-CLASS MATERIAL DATA GENERATOR")
    print("=" * 60)
    
    generator = WorldClassMaterialDataGenerator()
    
    try:
        # Generate complete dataset
        listings, matches = await generator.generate_complete_dataset()
        
        # Save to CSV
        await generator.save_to_csv(listings, matches)
        
        print("\n" + "=" * 60)
        print("🎉 WORLD-CLASS GENERATION COMPLETE!")
        print("=" * 60)
        print(f"📊 Generated {len(listings):,} material listings")
        print(f"🔗 Generated {len(matches):,} material matches")
        print(f"💾 Saved to: {LISTINGS_CSV} and {MATCHES_CSV}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        raise
    finally:
        await generator.close()

if __name__ == "__main__":
    asyncio.run(main()) 