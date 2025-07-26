#!/usr/bin/env python3
"""
Enhanced AI Generator with Advanced Prompts
- Integrates the four strategic AI prompts
- Generates BOTH waste outputs AND resource inputs (requirements)
- Ultra-accurate material listings with proper locations
- Extended timeouts and retry logic
- Comprehensive error handling
"""

import os
import sys
import json
import logging
import time
import random
from typing import Dict, List, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
import requests
from backend.advanced_ai_prompts_service import AdvancedAIPromptsService
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('enhanced_ai_generator.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Patch StreamHandler to use UTF-8 encoding if possible
try:
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream.reconfigure(encoding='utf-8')
except Exception:
    pass
logger = logging.getLogger("EnhancedAIGenerator")

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize Advanced AI Service
advanced_ai_service = AdvancedAIPromptsService()

class EnhancedAIGenerator:
    def __init__(self):
        self.supabase = supabase
        self.ai_service = advanced_ai_service
        self.max_retries = 3
        self.base_timeout = 180  # 3 minutes base timeout
        self.retry_delay = 5  # 5 seconds between retries
        
    def generate_ultra_accurate_listings(self, company: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ultra-accurate listings using the Strategic Material & Synergy Analysis prompt
        """
        logger.info(f"🎯 Generating ultra-accurate listings for: {company.get('name', 'Unknown')}")
        
        try:
            # Use the advanced strategic analysis
            strategic_result = self.ai_service.strategic_material_analysis(company)
            
            if not strategic_result:
                logger.error(f"❌ Failed to get strategic analysis for {company.get('name')}")
                return self._get_fallback_listings(company)
            
            # Debug: Log the AI response structure
            logger.info(f"🔍 AI Response keys: {list(strategic_result.keys())}")
            logger.info(f"🔍 Predicted outputs count: {len(strategic_result.get('predicted_outputs', []))}")
            logger.info(f"🔍 Predicted inputs count: {len(strategic_result.get('predicted_inputs', []))}")
            
            if strategic_result.get('predicted_outputs'):
                logger.info(f"🔍 Sample output: {strategic_result['predicted_outputs'][0] if strategic_result['predicted_outputs'] else 'None'}")
            if strategic_result.get('predicted_inputs'):
                logger.info(f"🔍 Sample input: {strategic_result['predicted_inputs'][0] if strategic_result['predicted_inputs'] else 'None'}")
            
            # Check if AI returned empty results and try adaptive prompting
            if (
                len(strategic_result.get('predicted_outputs', [])) == 0 and
                len(strategic_result.get('predicted_inputs', [])) == 0
            ):
                logger.warning(f"⚠️ AI returned no outputs/inputs for {company.get('name')}, trying adaptive prompting...")
                
                # Try with a less strict prompt
                adaptive_result = self._try_adaptive_prompting(company)
                if adaptive_result:
                    logger.info(f"✅ Adaptive prompting successful for {company.get('name')}")
                    strategic_result = adaptive_result
                else:
                    logger.warning(f"⚠️ Adaptive prompting also failed for {company.get('name')}, using fallback listings.")
                    return self._get_fallback_listings(company)
            
            # Transform strategic analysis into database format
            transformed_listings = self._transform_strategic_to_listings(strategic_result, company['id'])
            
            logger.info(f"✅ Generated {len(transformed_listings['waste_materials'])} waste materials and {len(transformed_listings['requirements'])} requirements")
            
            return transformed_listings
            
        except Exception as e:
            logger.error(f"❌ Error in ultra-accurate generation: {str(e)}")
            return self._get_fallback_listings(company)
    
    def _transform_strategic_to_listings(self, strategic_result: Dict[str, Any], company_id: str) -> Dict[str, Any]:
        """
        Transform strategic analysis results into database-ready listings
        """
        waste_materials = []
        requirements = []
        
        # Transform outputs (waste materials) from predicted_outputs
        outputs = strategic_result.get('predicted_outputs', [])
        
        for output in outputs:
            # Handle quantity object structure: {"value": 1000, "unit": "kg"}
            quantity_obj = output.get('quantity', {})
            if isinstance(quantity_obj, dict):
                quantity_value = quantity_obj.get('value', 1)
                unit_value = quantity_obj.get('unit', 'units')
            else:
                # Fallback for string quantity
                quantity_str = str(quantity_obj)
                numeric_match = re.search(r'[\d,]+', quantity_str)
                if numeric_match:
                    quantity_value = int(numeric_match.group().replace(',', ''))
                else:
                    quantity_value = 1
                
                unit_match = re.search(r'([a-zA-Z]+)', quantity_str)
                unit_value = unit_match.group(1) if unit_match else 'units'
            
            # Sanity check for unrealistic quantities
            if quantity_value > 1000000:  # Flag quantities over 1 million
                logger.warning(f"  ⚠️ Large quantity detected: {quantity_value} for {output.get('name', 'Unknown')}")
            
            # Validate units
            valid_units = ['kg', 'liters', 'units', 'pieces', 'tons', 'meters', 'gallons', 'pounds', 'm³', 'kwh']
            if unit_value.lower() not in [u.lower() for u in valid_units]:
                logger.warning(f"  ⚠️ Unusual unit detected: {unit_value} for {output.get('name', 'Unknown')}")
            
            # Get material name from the 'name' field
            material_name = output.get('name', 'Unknown Material')
            
            # Skip materials with invalid names
            if not material_name or material_name.lower() in ['unknown', 'unknown material', 'none', 'material']:
                logger.warning(f"  ⚠️ Skipping invalid material name: {material_name}")
                continue
            
            material = {
                'company_id': company_id,
                'material_name': material_name,
                'description': f"Waste material from {strategic_result.get('company_profile', {}).get('name', 'company')} operations",
                'quantity': quantity_value,  # Use numeric value only
                'unit': unit_value,
                'type': 'waste',
                'ai_generated': True
            }
            waste_materials.append(material)
            logger.info(f"  ✅ Transformed waste material: {material_name} ({quantity_value} {unit_value})")
        
        # Transform inputs (requirements) from predicted_inputs
        inputs = strategic_result.get('predicted_inputs', [])
        
        for input_item in inputs:
            # Handle quantity object structure: {"value": 500, "unit": "kg"}
            quantity_obj = input_item.get('quantity', {})
            if isinstance(quantity_obj, dict):
                quantity_value = quantity_obj.get('value', 1)
                unit_value = quantity_obj.get('unit', 'units')
            else:
                # Fallback for string quantity
                quantity_str = str(quantity_obj)
                numeric_match = re.search(r'[\d,]+', quantity_str)
                if numeric_match:
                    quantity_value = int(numeric_match.group().replace(',', ''))
                else:
                    quantity_value = 1
                
                unit_match = re.search(r'([a-zA-Z]+)', quantity_str)
                unit_value = unit_match.group(1) if unit_match else 'units'
            
            # Get requirement name from the 'name' field
            requirement_name = input_item.get('name', 'Unknown Requirement')
            
            # Skip requirements with invalid names
            if not requirement_name or requirement_name.lower() in ['unknown', 'unknown requirement', 'none', 'requirement']:
                logger.warning(f"  ⚠️ Skipping invalid requirement name: {requirement_name}")
                continue
            
            requirement = {
                'company_id': company_id,
                'name': requirement_name,
                'material_name': requirement_name,
                'description': f"Required material for {strategic_result.get('company_profile', {}).get('name', 'company')} operations",
                'category': 'general',
                'quantity_needed': str(quantity_value),  # Convert to string for requirements table
                'current_cost': 'Variable',
                'priority': 'medium',
                'ai_generated': True
            }
            requirements.append(requirement)
            logger.info(f"  ✅ Transformed requirement: {requirement_name} ({quantity_value} {unit_value})")
        
        logger.info(f"  📊 Transformed {len(waste_materials)} waste materials and {len(requirements)} requirements")
        
        return {
            'waste_materials': waste_materials,
            'requirements': requirements,
            'strategic_analysis': strategic_result
        }
    
    def _try_adaptive_prompting(self, company: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Try adaptive prompting with a less strict approach when the first attempt fails
        """
        try:
            logger.info(f"🔄 Attempting adaptive prompting for {company.get('name')}")
            
            # Create a less strict prompt that focuses on generating at least some results
            adaptive_prompt = f"""
You are analyzing a company for industrial symbiosis opportunities. The previous analysis returned no results, so we need a more flexible approach.

Company: {company.get('name', 'Unknown')}
Industry: {company.get('industry', 'Unknown')}
Location: {company.get('location', 'Unknown')}

TASK: Generate at least 3-5 waste outputs and 3-5 resource inputs for this company. Focus on the most common and realistic materials for this industry.

IMPORTANT: 
- Generate at least 3 outputs and 3 inputs, even if you need to be more general
- Use industry-typical materials and quantities
- If you're unsure about specific names, use descriptive terms like "Manufacturing Waste", "Process Water", etc.
- Focus on quantity and realistic units rather than perfect specificity

Provide the response as a JSON object with 'predicted_outputs' and 'predicted_inputs' arrays.
Each item should have: name, quantity (with value and unit), description.
"""
            
            # Call the AI service with the adaptive prompt
            adaptive_result = self.ai_service._call_deepseek_api_directly(adaptive_prompt)
            
            if adaptive_result and (
                len(adaptive_result.get('predicted_outputs', [])) > 0 or
                len(adaptive_result.get('predicted_inputs', [])) > 0
            ):
                logger.info(f"✅ Adaptive prompting generated {len(adaptive_result.get('predicted_outputs', []))} outputs and {len(adaptive_result.get('predicted_inputs', []))} inputs")
                return adaptive_result
            else:
                logger.warning(f"❌ Adaptive prompting also returned empty results")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error in adaptive prompting: {str(e)}")
            return None
    
    def _get_fallback_listings(self, company: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback listings if advanced AI fails
        """
        logger.warning(f"⚠️ Using fallback listings for {company.get('name')}")
        
        # Generate basic fallback based on industry
        industry = company.get('industry', 'manufacturing').lower()
        location = company.get('location', 'Unknown Location')
        
        fallback_waste = self._generate_fallback_waste(industry, location)
        fallback_requirements = self._generate_fallback_requirements(industry, location)
        
        return {
            'waste_materials': fallback_waste,
            'requirements': fallback_requirements,
            'strategic_analysis': None
        }
    
    def _generate_fallback_waste(self, industry: str, location: str) -> List[Dict[str, Any]]:
        """Generate fallback waste materials based on industry"""
        waste_templates = {
            'manufacturing': [
                {'name': 'Metal Scrap', 'quantity': '500', 'unit': 'kg', 'description': 'Various metal scraps from manufacturing processes'},
                {'name': 'Packaging Waste', 'quantity': '200', 'unit': 'kg', 'description': 'Cardboard, plastic, and other packaging materials'},
                {'name': 'Process Waste', 'quantity': '100', 'unit': 'kg', 'description': 'General process waste and byproducts'},
                {'name': 'Wood Waste', 'quantity': '150', 'unit': 'kg', 'description': 'Wood scraps and sawdust from manufacturing'},
                {'name': 'Electronic Waste', 'quantity': '50', 'unit': 'units', 'description': 'Electronic components and circuit boards'}
            ],
            'chemical': [
                {'name': 'Chemical Byproducts', 'quantity': '300', 'unit': 'liters', 'description': 'Chemical process byproducts and waste solutions'},
                {'name': 'Contaminated Water', 'quantity': '1000', 'unit': 'liters', 'description': 'Process water with chemical contaminants'},
                {'name': 'Filter Media', 'quantity': '50', 'unit': 'kg', 'description': 'Used filter media and adsorbents'},
                {'name': 'Chemical Sludge', 'quantity': '200', 'unit': 'kg', 'description': 'Chemical sludge and residues'},
                {'name': 'Solvent Waste', 'quantity': '100', 'unit': 'liters', 'description': 'Used solvents and cleaning agents'}
            ],
            'food': [
                {'name': 'Organic Waste', 'quantity': '800', 'unit': 'kg', 'description': 'Food processing waste and organic byproducts'},
                {'name': 'Cooking Oil', 'quantity': '200', 'unit': 'liters', 'description': 'Used cooking oil and fats'},
                {'name': 'Packaging Waste', 'quantity': '300', 'unit': 'kg', 'description': 'Food packaging materials'},
                {'name': 'Food Waste', 'quantity': '500', 'unit': 'kg', 'description': 'Expired and spoiled food products'},
                {'name': 'Animal Byproducts', 'quantity': '150', 'unit': 'kg', 'description': 'Animal waste and byproducts'}
            ],
            'textile': [
                {'name': 'Fabric Scraps', 'quantity': '400', 'unit': 'kg', 'description': 'Textile scraps and fabric waste'},
                {'name': 'Dye Waste', 'quantity': '150', 'unit': 'liters', 'description': 'Waste dye solutions and colorants'},
                {'name': 'Fiber Waste', 'quantity': '200', 'unit': 'kg', 'description': 'Fiber waste and textile byproducts'},
                {'name': 'Yarn Waste', 'quantity': '100', 'unit': 'kg', 'description': 'Waste yarn and thread materials'},
                {'name': 'Textile Waste', 'quantity': '300', 'unit': 'kg', 'description': 'General textile waste materials'}
            ],
            'healthcare': [
                {'name': 'Medical Waste', 'quantity': '100', 'unit': 'kg', 'description': 'General medical waste and disposables'},
                {'name': 'Contaminated Linens', 'quantity': '200', 'unit': 'pieces', 'description': 'Contaminated medical linens and textiles'},
                {'name': 'Pharmaceutical Waste', 'quantity': '50', 'unit': 'kg', 'description': 'Expired medications and pharmaceutical waste'},
                {'name': 'Medical Sharps', 'quantity': '20', 'unit': 'units', 'description': 'Used syringes and medical sharps'},
                {'name': 'Sterilization Byproducts', 'quantity': '80', 'unit': 'kg', 'description': 'Sterilization process byproducts'}
            ],
            'tourism': [
                {'name': 'Food Waste', 'quantity': '300', 'unit': 'kg', 'description': 'Food waste from hospitality operations'},
                {'name': 'Used Cooking Oil', 'quantity': '100', 'unit': 'liters', 'description': 'Used cooking oil from kitchens'},
                {'name': 'Wastewater', 'quantity': '2000', 'unit': 'liters', 'description': 'Wastewater from hospitality operations'},
                {'name': 'Soiled Linens', 'quantity': '150', 'unit': 'pieces', 'description': 'Soiled linens and towels'},
                {'name': 'Packaging Waste', 'quantity': '250', 'unit': 'kg', 'description': 'Packaging waste from hospitality operations'}
            ],
            'hospitality': [
                {'name': 'Food Waste', 'quantity': '400', 'unit': 'kg', 'description': 'Food waste from hospitality operations'},
                {'name': 'Used Cooking Oil', 'quantity': '120', 'unit': 'liters', 'description': 'Used cooking oil from kitchens'},
                {'name': 'Wastewater', 'quantity': '2500', 'unit': 'liters', 'description': 'Wastewater from hospitality operations'},
                {'name': 'Soiled Linens', 'quantity': '200', 'unit': 'pieces', 'description': 'Soiled linens and towels'},
                {'name': 'Packaging Waste', 'quantity': '300', 'unit': 'kg', 'description': 'Packaging waste from hospitality operations'}
            ]
        }
        
        # Map industry to template key
        industry_mapping = {
            'healthcare': 'healthcare',
            'tourism & hospitality': 'tourism',
            'hospitality': 'hospitality',
            'manufacturing (general)': 'manufacturing',
            'manufacturing': 'manufacturing',
            'chemical': 'chemical',
            'food': 'food',
            'textile': 'textile',
            'food processing': 'food',
            'medical': 'healthcare',
            'hotel': 'hospitality',
            'restaurant': 'food'
        }
        
        template_key = industry_mapping.get(industry.lower(), 'manufacturing')
        waste_list = waste_templates.get(template_key, waste_templates['manufacturing'])
        
        fallback_waste = []
        for i, waste in enumerate(waste_list):
            # Parse quantity for fallback materials
            quantity_str = f"{waste['quantity']} {waste['unit']}"
            numeric_match = re.search(r'[\d,]+', quantity_str)
            if numeric_match:
                quantity_value = int(numeric_match.group().replace(',', ''))
            else:
                quantity_value = 1
            
            material = {
                'material_name': waste['name'],
                'description': f"{waste['description']} from {location}",
                'quantity': quantity_value,
                'unit': waste['unit'],
                'type': 'waste',
                'ai_generated': True
            }
            fallback_waste.append(material)
        
        return fallback_waste
    
    def _generate_fallback_requirements(self, industry: str, location: str) -> List[Dict[str, Any]]:
        """Generate fallback requirements based on industry"""
        requirement_templates = {
            'manufacturing': [
                {'name': 'Raw Materials', 'quantity': '1000', 'unit': 'kg', 'description': 'Primary raw materials for manufacturing'},
                {'name': 'Energy', 'quantity': '5000', 'unit': 'kWh', 'description': 'Electrical energy for production processes'},
                {'name': 'Water', 'quantity': '2000', 'unit': 'liters', 'description': 'Process water for manufacturing operations'},
                {'name': 'Packaging Materials', 'quantity': '300', 'unit': 'kg', 'description': 'Packaging materials for finished products'},
                {'name': 'Lubricants', 'quantity': '100', 'unit': 'liters', 'description': 'Industrial lubricants and oils'}
            ],
            'chemical': [
                {'name': 'Chemical Reagents', 'quantity': '500', 'unit': 'kg', 'description': 'Chemical reagents and solvents'},
                {'name': 'Catalysts', 'quantity': '100', 'unit': 'kg', 'description': 'Catalytic materials for chemical processes'},
                {'name': 'Purified Water', 'quantity': '3000', 'unit': 'liters', 'description': 'High-purity water for chemical processes'},
                {'name': 'Raw Chemicals', 'quantity': '800', 'unit': 'kg', 'description': 'Primary chemical raw materials'},
                {'name': 'Energy', 'quantity': '4000', 'unit': 'kWh', 'description': 'Energy for chemical processes'}
            ],
            'food': [
                {'name': 'Fresh Ingredients', 'quantity': '2000', 'unit': 'kg', 'description': 'Fresh food ingredients and raw materials'},
                {'name': 'Packaging Materials', 'quantity': '800', 'unit': 'kg', 'description': 'Food-grade packaging materials'},
                {'name': 'Refrigeration', 'quantity': '1000', 'unit': 'kWh', 'description': 'Refrigeration and cooling energy'},
                {'name': 'Cooking Oil', 'quantity': '300', 'unit': 'liters', 'description': 'Cooking oils and fats'},
                {'name': 'Spices and Seasonings', 'quantity': '100', 'unit': 'kg', 'description': 'Spices and food seasonings'}
            ],
            'textile': [
                {'name': 'Raw Fibers', 'quantity': '1500', 'unit': 'kg', 'description': 'Natural and synthetic fibers'},
                {'name': 'Dyes and Chemicals', 'quantity': '300', 'unit': 'kg', 'description': 'Textile dyes and processing chemicals'},
                {'name': 'Water', 'quantity': '5000', 'unit': 'liters', 'description': 'Process water for textile operations'},
                {'name': 'Energy', 'quantity': '3000', 'unit': 'kWh', 'description': 'Energy for textile processing'},
                {'name': 'Packaging Materials', 'quantity': '400', 'unit': 'kg', 'description': 'Packaging for finished textiles'}
            ],
            'healthcare': [
                {'name': 'Medical Supplies', 'quantity': '500', 'unit': 'units', 'description': 'Medical supplies and disposables'},
                {'name': 'Pharmaceuticals', 'quantity': '200', 'unit': 'units', 'description': 'Pharmaceutical products and medications'},
                {'name': 'Medical Equipment', 'quantity': '50', 'unit': 'units', 'description': 'Medical equipment and devices'},
                {'name': 'Sterilization Materials', 'quantity': '100', 'unit': 'kg', 'description': 'Sterilization materials and chemicals'},
                {'name': 'Energy', 'quantity': '2000', 'unit': 'kWh', 'description': 'Energy for medical operations'}
            ],
            'tourism': [
                {'name': 'Fresh Ingredients', 'quantity': '1000', 'unit': 'kg', 'description': 'Fresh food ingredients for hospitality'},
                {'name': 'Cleaning Supplies', 'quantity': '200', 'unit': 'kg', 'description': 'Cleaning supplies and chemicals'},
                {'name': 'Linens and Towels', 'quantity': '300', 'unit': 'pieces', 'description': 'Linens and towels for guests'},
                {'name': 'Energy', 'quantity': '1500', 'unit': 'kWh', 'description': 'Energy for hospitality operations'},
                {'name': 'Water', 'quantity': '3000', 'unit': 'liters', 'description': 'Water for hospitality operations'}
            ],
            'hospitality': [
                {'name': 'Fresh Ingredients', 'quantity': '1200', 'unit': 'kg', 'description': 'Fresh food ingredients for hospitality'},
                {'name': 'Cleaning Supplies', 'quantity': '250', 'unit': 'kg', 'description': 'Cleaning supplies and chemicals'},
                {'name': 'Linens and Towels', 'quantity': '400', 'unit': 'pieces', 'description': 'Linens and towels for guests'},
                {'name': 'Energy', 'quantity': '1800', 'unit': 'kWh', 'description': 'Energy for hospitality operations'},
                {'name': 'Water', 'quantity': '3500', 'unit': 'liters', 'description': 'Water for hospitality operations'}
            ]
        }
        
        # Map industry to template key (same mapping as waste)
        industry_mapping = {
            'healthcare': 'healthcare',
            'tourism & hospitality': 'tourism',
            'hospitality': 'hospitality',
            'manufacturing (general)': 'manufacturing',
            'manufacturing': 'manufacturing',
            'chemical': 'chemical',
            'food': 'food',
            'textile': 'textile',
            'food processing': 'food',
            'medical': 'healthcare',
            'hotel': 'hospitality',
            'restaurant': 'food'
        }
        
        template_key = industry_mapping.get(industry.lower(), 'manufacturing')
        req_list = requirement_templates.get(template_key, requirement_templates['manufacturing'])
        
        fallback_requirements = []
        for i, req in enumerate(req_list):
            requirement = {
                'name': req['name'],
                'material_name': req['name'],
                'description': f"{req['description']} for {location} operations",
                'category': 'general',
                'quantity': f"{req['quantity']} {req['unit']}",
                'unit': req['unit'],
                'frequency': 'monthly',
                'notes': f'Fallback generation for {industry} industry',
                'potential_value': 'Variable',
                'quality_grade': 'high',
                'type': 'requirement',
                'ai_generated': True
            }
            fallback_requirements.append(requirement)
        
        return fallback_requirements
    
    def insert_listings_with_retry(self, company_id: str, listings: Dict[str, Any]) -> Dict[str, int]:
        """
        Insert listings with retry logic and comprehensive error handling
        """
        results = {'waste_inserted': 0, 'requirements_inserted': 0, 'errors': []}
        
        # Insert waste materials into materials table
        if listings.get('waste_materials'):
            try:
                for material in listings['waste_materials']:
                    # Prepare material data for materials table - minimal structure
                    material_data = {
                        'company_id': company_id,
                        'material_name': material.get('material_name', material.get('name', 'Unknown Material')),
                        'description': material.get('description', ''),
                        'quantity': material.get('quantity', ''),
                        'unit': material.get('unit', 'units'),
                        'type': 'waste',
                        'ai_generated': True
                        # Only include fields that definitely exist
                    }
                    
                    # Retry logic for each material
                    for attempt in range(self.max_retries):
                        try:
                            logger.info(f"  🔍 Attempting to insert waste material: {material_data.get('name')}")
                            logger.info(f"  📋 Material data: {material_data}")
                            result = self.supabase.table('materials').insert(material_data).execute()
                            if result.data:
                                results['waste_inserted'] += 1
                                logger.info(f"  ✅ Successfully inserted waste material: {material_data.get('name')}")
                                break
                            else:
                                raise Exception("No data returned from insert")
                        except Exception as e:
                            logger.error(f"  ❌ Insert error for {material_data.get('name')}: {str(e)}")
                            if attempt == self.max_retries - 1:
                                results['errors'].append(f"Failed to insert waste material {material.get('name')}: {str(e)}")
                            else:
                                time.sleep(self.retry_delay)
                                continue
            except Exception as e:
                results['errors'].append(f"Error inserting waste materials: {str(e)}")
        
        # Insert requirements into requirements table
        if listings.get('requirements'):
            try:
                for requirement in listings['requirements']:
                    # Prepare requirement data for requirements table
                    requirement_data = {
                        'company_id': company_id,
                        'name': requirement.get('name', 'Unknown Requirement'),
                        'description': requirement.get('description', ''),
                        'category': requirement.get('category', 'general'),
                        'quantity_needed': requirement.get('quantity_needed', ''),
                        'current_cost': requirement.get('current_cost', ''),
                        'priority': 'medium',
                        'ai_generated': True
                        # Remove created_at - let database set it automatically
                    }
                    
                    # Retry logic for each requirement
                    for attempt in range(self.max_retries):
                        try:
                            logger.info(f"  🔍 Attempting to insert requirement: {requirement_data.get('name')}")
                            logger.info(f"  📋 Requirement data: {requirement_data}")
                            result = self.supabase.table('requirements').insert(requirement_data).execute()
                            if result.data:
                                results['requirements_inserted'] += 1
                                logger.info(f"  ✅ Successfully inserted requirement: {requirement_data.get('name')}")
                                break
                            else:
                                raise Exception("No data returned from insert")
                        except Exception as e:
                            logger.error(f"  ❌ Insert error for {requirement_data.get('name')}: {str(e)}")
                            if attempt == self.max_retries - 1:
                                results['errors'].append(f"Failed to insert requirement {requirement.get('name')}: {str(e)}")
                            else:
                                time.sleep(self.retry_delay)
                                continue
            except Exception as e:
                results['errors'].append(f"Error inserting requirements: {str(e)}")
        
        return results
    
    def process_company_with_advanced_ai(self, company: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
        """
        Process a single company with advanced AI and comprehensive error handling
        """
        logger.info(f"\n[{idx}/{total}] 🏢 Processing: {company.get('name', 'Unknown Company')}")
        
        result = {
            'company_name': company.get('name', 'Unknown'),
            'company_id': company.get('id', ''),
            'success': False,
            'waste_materials': 0,
            'requirements': 0,
            'errors': [],
            'strategic_analysis': None
        }
        
        try:
            # Build comprehensive company profile
            profile = self._build_comprehensive_profile(company)
            logger.info(f"  📊 Company profile: {profile.get('industry', 'Unknown')} industry, {profile.get('location', 'Unknown')} location")
            
            # Generate ultra-accurate listings with retry logic
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"  🤖 Calling Advanced AI (attempt {attempt + 1}/{self.max_retries})...")
                    listings = self.generate_ultra_accurate_listings(profile)
                    logger.info(f"  ✅ Advanced AI analysis successful")
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"  ❌ All AI attempts failed: {str(e)}")
                        result['errors'].append(f"AI generation failed: {str(e)}")
                        listings = self._get_fallback_listings(profile)
                    else:
                        logger.warning(f"  ⚠️ AI attempt {attempt + 1} failed, retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
            
            # Insert listings with retry logic
            logger.info(f"  💾 Inserting listings to database...")
            insert_results = self.insert_listings_with_retry(company['id'], listings)
            
            result['waste_materials'] = insert_results['waste_inserted']
            result['requirements'] = insert_results['requirements_inserted']
            result['errors'].extend(insert_results['errors'])
            result['strategic_analysis'] = listings.get('strategic_analysis')
            
            if not result['errors']:
                result['success'] = True
                logger.info(f"  ✅ Success: {result['waste_materials']} waste + {result['requirements']} requirements = {result['waste_materials'] + result['requirements']} total")
            else:
                logger.warning(f"  ⚠️ Completed with errors: {len(result['errors'])} errors")
            
        except Exception as e:
            error_msg = f"Unexpected error for {company.get('name')}: {str(e)}"
            logger.error(f"  ❌ {error_msg}")
            result['errors'].append(error_msg)
        
        return result
    
    def _build_comprehensive_profile(self, company: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a comprehensive company profile for the advanced AI
        """
        # Clean and validate location data
        location = company.get('location', 'Unknown Location')
        if isinstance(location, str):
            # Remove any distance measurements (like "24km")
            location = location.replace('km', '').replace('miles', '').strip()
            # Ensure it's a proper location format
            if location and location != 'Unknown Location':
                # Add country if not present
                if ',' not in location and len(location.split()) <= 2:
                    location = f"{location}, United Arab Emirates"  # Default to UAE for Gulf companies
        
        return {
            'id': company.get('id'),
            'name': company.get('name', 'Unknown Company'),
            'industry': company.get('industry', 'Manufacturing'),
            'location': location,
            'employee_count': company.get('employee_count', 100),
            'products': company.get('products', 'Various products'),
            'main_materials': company.get('main_materials', 'Various materials'),
            'production_volume': company.get('production_volume', 'Standard production volume'),
            'process_description': company.get('process_description', 'Standard manufacturing processes'),
            'sustainability_goals': company.get('sustainability_goals', ['Reduce waste', 'Improve efficiency']),
            'current_waste_management': company.get('current_waste_management', 'Standard waste management'),
            'onboarding_completed': company.get('onboarding_completed', True),
            'created_at': company.get('created_at'),
            'updated_at': company.get('updated_at'),
        }
    
    def generate_all_listings(self, test_mode: bool = False):
        """
        Generate listings for all companies with advanced AI
        """
        start_time = time.time()
        logger.info("🚀 Starting Enhanced AI Listings Generator with Advanced Prompts...")
        logger.info("=" * 80)
        
        try:
            # Fetch all companies
            logger.info("📋 Fetching companies from database...")
            companies_response = self.supabase.table('companies').select('*').execute()
            companies = companies_response.data
            
            if not companies:
                logger.error("❌ No companies found in database")
                return
            
            logger.info(f"✅ Found {len(companies)} companies")
            
            # Test mode: process only first 5 companies
            if test_mode:
                companies = companies[:5]
                logger.info(f"🧪 Test mode: Processing first {len(companies)} companies")
            
            logger.info("🏭 Generating ultra-accurate listings for each company...")
            logger.info("-" * 80)
            
            results = []
            successful_companies = 0
            failed_companies = 0
            total_materials = 0
            
            for idx, company in enumerate(companies, 1):
                result = self.process_company_with_advanced_ai(company, idx, len(companies))
                results.append(result)
                
                if result['success']:
                    successful_companies += 1
                    total_materials += result['waste_materials'] + result['requirements']
                else:
                    failed_companies += 1
                
                # Sleep to avoid rate limits
                time.sleep(2)
            
            # Generate comprehensive report
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("-" * 80)
            logger.info("🎉 ENHANCED AI GENERATION COMPLETE!")
            logger.info(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            logger.info(f"✅ Successful companies: {successful_companies}/{len(companies)}")
            logger.info(f"❌ Failed companies: {failed_companies}/{len(companies)}")
            logger.info(f"📦 Total materials generated: {total_materials}")
            logger.info(f"📊 Average materials per successful company: {total_materials/successful_companies:.1f}" if successful_companies > 0 else "📊 No successful companies")
            
            # Save detailed results
            with open('enhanced_ai_generation_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"📄 Detailed results saved to: enhanced_ai_generation_results.json")
            
            if test_mode and successful_companies == len(companies):
                logger.info("🚀 All tests passed! Ready for production run.")
                logger.info("💡 To run on all companies, set test_mode=False")
            elif failed_companies > 0:
                logger.info("⚠️ Some companies failed. Check the log above for details.")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Fatal error in generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

def main():
    """Main function to run the enhanced AI generator"""
    generator = EnhancedAIGenerator()
    
    # Check if test mode is requested
    test_mode = '--test' in sys.argv
    
    if test_mode:
        logger.info("🧪 Running in TEST MODE (first 5 companies only)")
    else:
        logger.info("🚀 Running in PRODUCTION MODE (all companies)")
    
    # Run the enhanced generator
    results = generator.generate_all_listings(test_mode=test_mode)
    
    if results:
        logger.info("✅ Enhanced AI generation completed successfully!")
    else:
        logger.error("❌ Enhanced AI generation failed!")

if __name__ == "__main__":
    main() 