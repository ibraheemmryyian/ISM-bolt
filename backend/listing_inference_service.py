import os
import json
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
import traceback

# Use the provided DeepSeek API key
DEEPSEEK_API_KEY = 'sk-7ce79f30332d45d5b3acb8968b052132'
DEEPSEEK_BASE_URL = 'https://api.deepseek.com/v1/chat/completions'
DEEPSEEK_MODEL = 'deepseek-code'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ListingInferenceService:
    def __init__(self):
        self.deepseek_api_key = DEEPSEEK_API_KEY
        self.deepseek_base_url = DEEPSEEK_BASE_URL
        self.deepseek_model = DEEPSEEK_MODEL
        
    def generate_listings_from_profile(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced Phase 1: Generate comprehensive AI-powered material listings, requirements, 
        company suggestions, and green initiatives from company profile.
        
        Args:
            company_profile: Dictionary containing company information
            
        Returns:
            Dictionary with comprehensive analysis including materials, requirements, 
            company suggestions, and green initiatives
        """
        try:
            logger.info(f"Starting Enhanced AI inference for company: {company_profile.get('name', 'Unknown')}")
            
            # Generate comprehensive analysis
            analysis_result = self._generate_comprehensive_analysis(company_profile)
            
            logger.info(f"Successfully generated comprehensive analysis with {len(analysis_result.get('predicted_outputs', []))} outputs and {len(analysis_result.get('predicted_inputs', []))} inputs")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in generate_listings_from_profile: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_enhanced_fallback_listings(company_profile)
    
    def _generate_comprehensive_analysis(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis including materials, requirements, company suggestions, and green initiatives.
        """
        
        # Extract key information
        industry = company_profile.get('industry', 'Unknown')
        products = company_profile.get('products', 'Unknown')
        description = company_profile.get('process_description', '')
        location = company_profile.get('location', 'Unknown')
        employee_count = company_profile.get('employee_count', 0)
        main_materials = company_profile.get('main_materials', 'Unknown')
        production_volume = company_profile.get('production_volume', 'Unknown')
        
        # Construct comprehensive prompt
        prompt = self._construct_comprehensive_prompt(company_profile)
        
        # Call DeepSeek API
        response = self._call_deepseek_api(prompt)
        
        # Parse and validate response
        parsed_response = self._parse_comprehensive_response(response)
        
        # Add AI insights and recommendations
        parsed_response['ai_insights'] = self._generate_ai_insights(company_profile, parsed_response)
        parsed_response['company_suggestions'] = self._generate_company_suggestions(company_profile, parsed_response)
        parsed_response['green_initiatives'] = self._generate_green_initiatives(company_profile, parsed_response)
        
        return parsed_response
    
    def _construct_comprehensive_prompt(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construct comprehensive prompt for material listings, requirements, and analysis.
        """
        
        industry = company_profile.get('industry', 'Unknown')
        products = company_profile.get('products', 'Unknown')
        description = company_profile.get('process_description', '')
        location = company_profile.get('location', 'Unknown')
        employee_count = company_profile.get('employee_count', 0)
        main_materials = company_profile.get('main_materials', 'Unknown')
        production_volume = company_profile.get('production_volume', 'Unknown')
        
        system_prompt = """You are an expert in industrial symbiosis, circular economies, and sustainable business practices. Your task is to analyze a company's profile and provide a comprehensive analysis including:

1. PREDICTED OUTPUTS (8-15 waste streams, byproducts, excess materials) - Generate a realistic number of waste materials that this company would produce based on their industry, size, and operations.

2. PREDICTED INPUTS (10-20 operational needs, raw materials, resources) - Generate a comprehensive list of materials this company needs for their operations, including raw materials, consumables, and resources.

3. COMPANY SUGGESTIONS (5-8 potential partners and collaboration opportunities)
4. GREEN INITIATIVES (6-10 sustainability improvements with cost savings)

For each material listing, include:
- name: Material name
- category: (textile, chemical, metal, plastic, organic, electronic, energy, water, etc.)
- description: Detailed explanation
- quantity: Estimated amount with units
- frequency: (daily, weekly, monthly, quarterly, annually, batch)
- notes: Special considerations
- potential_value: Estimated market value
- quality_grade: (high, medium, low)
- potential_uses: Array of possible applications
- symbiosis_opportunities: Array of partnership opportunities
- sustainability_impact: Environmental benefits
- cost_savings: Potential financial benefits

For company suggestions, include:
- company_type: Type of business
- location: Geographic area
- waste_they_can_use: Materials they can utilize
- resources_they_can_provide: What they can offer
- estimated_partnership_value: Financial benefit
- carbon_reduction: Environmental impact
- implementation_time: Timeline for partnership

For green initiatives, include:
- initiative_name: Name of the improvement
- description: Detailed explanation
- current_practice: What they're doing now
- greener_alternative: What they could do instead
- cost_savings_per_month: Monthly financial benefit
- carbon_reduction: Environmental impact
- implementation_cost: Upfront investment
- payback_period: Time to recoup investment
- difficulty: (easy, medium, hard)
- priority: (high, medium, low)

IMPORTANT: Generate 8-15 waste streams and 10-20 requirements per company to create a realistic industrial symbiosis marketplace. Be comprehensive and industry-specific.

Provide response as JSON with keys: predicted_outputs, predicted_inputs, company_suggestions, green_initiatives"""

        user_content = f"""Analyze this company profile for industrial symbiosis opportunities:

Company: {company_profile.get('name', 'Unknown')}
Industry: {industry}
Products: {products}
Description: {description}
Location: {location}
Employees: {employee_count}
Main Materials: {main_materials}
Production Volume: {production_volume}

Generate comprehensive material listings, requirements, company suggestions, and green initiatives with detailed cost savings and environmental impact analysis."""

        return {
            "model": self.deepseek_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "response_format": { "type": "json_object" },
            "temperature": 0.7,
            "max_tokens": 4000
        }
    
    def _call_deepseek_api(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the DeepSeek API with the comprehensive prompt structure."""
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.deepseek_api_key}'
        }
        
        try:
            logger.info(f"Calling DeepSeek API with model {self.deepseek_model}...")
            response = requests.post(
                self.deepseek_base_url,
                headers=headers,
                json=prompt_data,
                timeout=120  # Increased timeout to 2 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("DeepSeek API call successful")
                return result
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                raise Exception(f"DeepSeek API returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error calling DeepSeek API: {str(e)}")
            raise Exception(f"Failed to call DeepSeek API: {str(e)}")
    
    def _parse_comprehensive_response(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the comprehensive DeepSeek API response."""
        
        try:
            # Extract the content from the API response
            if 'choices' in api_response and len(api_response['choices']) > 0:
                content = api_response['choices'][0]['message']['content']
                
                # Parse the JSON content
                if isinstance(content, str):
                    parsed = json.loads(content)
                else:
                    parsed = content
                
                # Validate the structure
                if not isinstance(parsed, dict):
                    raise ValueError("Response is not a dictionary")
                
                # Ensure all required keys exist
                required_keys = ['predicted_outputs', 'predicted_inputs', 'company_suggestions', 'green_initiatives']
                for key in required_keys:
                    if key not in parsed:
                        parsed[key] = []
                
                # Validate and clean each section
                parsed['predicted_outputs'] = [
                    self._validate_and_clean_material_item(item, 'output') 
                    for item in parsed['predicted_outputs'] 
                    if self._validate_material_item(item)
                ]
                
                parsed['predicted_inputs'] = [
                    self._validate_and_clean_material_item(item, 'input') 
                    for item in parsed['predicted_inputs'] 
                    if self._validate_material_item(item)
                ]
                
                parsed['company_suggestions'] = [
                    self._validate_and_clean_company_suggestion(item)
                    for item in parsed['company_suggestions']
                    if self._validate_company_suggestion(item)
                ]
                
                parsed['green_initiatives'] = [
                    self._validate_and_clean_green_initiative(item)
                    for item in parsed['green_initiatives']
                    if self._validate_green_initiative(item)
                ]
                
                return parsed
            else:
                raise ValueError("Invalid API response structure")
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise ValueError(f"Failed to parse JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            raise
    
    def _validate_material_item(self, item: Dict[str, Any]) -> bool:
        """Validate that a material item has the required fields."""
        required_fields = ['name', 'category', 'description', 'quantity', 'frequency', 'notes']
        return all(field in item and item[field] for field in required_fields)
    
    def _validate_company_suggestion(self, item: Dict[str, Any]) -> bool:
        """Validate that a company suggestion has the required fields."""
        required_fields = ['company_type', 'location', 'waste_they_can_use', 'resources_they_can_provide']
        return all(field in item and item[field] for field in required_fields)
    
    def _validate_green_initiative(self, item: Dict[str, Any]) -> bool:
        """Validate that a green initiative has the required fields."""
        required_fields = ['initiative_name', 'description', 'current_practice', 'greener_alternative']
        return all(field in item and item[field] for field in required_fields)
    
    def _validate_and_clean_material_item(self, item: Dict[str, Any], item_type: str) -> Dict[str, Any]:
        """Validate and clean a material item, ensuring all required fields are present."""
        
        # Ensure all required fields exist with defaults if missing
        cleaned_item = {
            'name': item.get('name', 'Unknown'),
            'category': item.get('category', 'general'),
            'description': item.get('description', ''),
            'quantity': item.get('quantity', 'Unknown'),
            'frequency': item.get('frequency', 'monthly'),
            'notes': item.get('notes', ''),
            'potential_value': item.get('potential_value', 'Unknown'),
            'quality_grade': item.get('quality_grade', 'medium'),
            'potential_uses': item.get('potential_uses', []),
            'symbiosis_opportunities': item.get('symbiosis_opportunities', []),
            'sustainability_impact': item.get('sustainability_impact', ''),
            'cost_savings': item.get('cost_savings', ''),
            'ai_generated': True
        }
        
        # Add type-specific fields
        if item_type == 'output':
            cleaned_item.update({
                'quantity_estimate': item.get('quantity', 'Unknown'),
                'type': 'output'
            })
        else:  # input
            cleaned_item.update({
                'quantity_needed': item.get('quantity', 'Unknown'),
                'current_cost': item.get('current_cost', 'Unknown'),
                'priority': item.get('priority', 'medium'),
                'potential_sources': item.get('potential_sources', []),
                'type': 'input'
            })
        
        return cleaned_item
    
    def _validate_and_clean_company_suggestion(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean a company suggestion."""
        return {
            'company_type': item.get('company_type', 'Unknown'),
            'location': item.get('location', 'Unknown'),
            'waste_they_can_use': item.get('waste_they_can_use', []),
            'resources_they_can_provide': item.get('resources_they_can_provide', []),
            'estimated_partnership_value': item.get('estimated_partnership_value', 'Unknown'),
            'carbon_reduction': item.get('carbon_reduction', 'Unknown'),
            'implementation_time': item.get('implementation_time', 'Unknown'),
            'ai_generated': True
        }
    
    def _validate_and_clean_green_initiative(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean a green initiative."""
        return {
            'initiative_name': item.get('initiative_name', 'Unknown'),
            'description': item.get('description', ''),
            'current_practice': item.get('current_practice', ''),
            'greener_alternative': item.get('greener_alternative', ''),
            'cost_savings_per_month': item.get('cost_savings_per_month', 'Unknown'),
            'carbon_reduction': item.get('carbon_reduction', 'Unknown'),
            'implementation_cost': item.get('implementation_cost', 'Unknown'),
            'payback_period': item.get('payback_period', 'Unknown'),
            'difficulty': item.get('difficulty', 'medium'),
            'priority': item.get('priority', 'medium'),
            'ai_generated': True
        }
    
    def _generate_ai_insights(self, company_profile: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI insights from the analysis."""
        
        outputs = analysis.get('predicted_outputs', [])
        inputs = analysis.get('predicted_inputs', [])
        suggestions = analysis.get('company_suggestions', [])
        initiatives = analysis.get('green_initiatives', [])
        
        # Calculate insights
        total_savings = 0
        total_carbon_reduction = 0
        
        # Calculate from green initiatives
        for initiative in initiatives:
            savings_str = initiative.get('cost_savings_per_month', '0')
            try:
                savings = float(savings_str.replace('$', '').replace(',', '').split()[0])
                total_savings += savings * 12  # Annual savings
            except:
                pass
            
            carbon_str = initiative.get('carbon_reduction', '0')
            try:
                carbon = float(carbon_str.replace('tons', '').replace('CO2', '').strip())
                total_carbon_reduction += carbon
            except:
                pass
        
        return {
            'symbiosis_score': f"{min(95, 50 + len(outputs) * 5 + len(suggestions) * 3)}%",
            'estimated_savings': f"${total_savings:,.0f} annually",
            'carbon_reduction': f"{total_carbon_reduction:.1f} tons CO2 annually",
            'top_opportunities': [output.get('name', '') for output in outputs[:3]],
            'recommended_partners': [suggestion.get('company_type', '') for suggestion in suggestions[:3]],
            'implementation_roadmap': [
                "Review and approve AI-generated materials",
                "Select preferred partner matches", 
                "Contact potential partners",
                "Establish supply agreements",
                "Implement green initiatives"
            ]
        }
    
    def _generate_company_suggestions(self, company_profile: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate additional company suggestions based on the analysis."""
        # This is already handled in the main analysis, but we can add more here if needed
        return analysis.get('company_suggestions', [])
    
    def _generate_green_initiatives(self, company_profile: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate additional green initiatives based on the analysis."""
        # This is already handled in the main analysis, but we can add more here if needed
        return analysis.get('green_initiatives', [])
    
    def _get_enhanced_fallback_listings(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced fallback listings when AI service is unavailable."""
        
        industry = company_profile.get('industry', 'Unknown').lower()
        location = company_profile.get('location', 'Unknown')
        
        # Industry-specific fallback materials
        fallback_outputs = []
        fallback_inputs = []
        
        if 'manufacturing' in industry or 'production' in industry:
            fallback_outputs = [
                {
                    'name': 'Production Waste',
                    'category': 'general',
                    'description': 'Various waste materials from manufacturing processes',
                    'quantity': 'Variable based on production volume',
                    'frequency': 'daily',
                    'notes': 'Requires sorting and processing',
                    'potential_value': '$500-2000/month',
                        'quality_grade': 'medium',
                    'potential_uses': ['Recycling', 'Energy recovery', 'Material recovery'],
                    'symbiosis_opportunities': ['Local recyclers', 'Energy companies', 'Material processors'],
                    'sustainability_impact': 'Reduces landfill waste',
                    'cost_savings': '$1000-5000/month in disposal costs',
                    'quantity_estimate': 'Variable',
                    'type': 'output',
                    'ai_generated': True
                }
            ]
            
            fallback_inputs = [
                    {
                        'name': 'Raw Materials',
                    'category': 'general',
                    'description': 'Primary materials for production',
                    'quantity': 'Based on production volume',
                    'frequency': 'weekly',
                    'notes': 'Quality and consistency important',
                    'current_cost': 'Variable',
                    'priority': 'high',
                    'potential_sources': ['Local suppliers', 'Recycled materials', 'Waste exchanges'],
                    'symbiosis_opportunities': ['Material exchanges', 'Bulk purchasing'],
                        'quantity_needed': 'Variable',
                    'type': 'input',
                    'ai_generated': True
                }
            ]
        
        # Enhanced fallback company suggestions
        fallback_suggestions = [
            {
                'company_type': 'Local Manufacturing Companies',
                'location': location,
                'waste_they_can_use': ['Production waste', 'Packaging materials'],
                'resources_they_can_provide': ['Raw materials', 'Technical expertise'],
                'estimated_partnership_value': '$25K annually',
                'carbon_reduction': '10-20 tons CO2',
                'implementation_time': '3-6 months',
                'ai_generated': True
            },
            {
                'company_type': 'Recycling Facilities',
                'location': location,
                'waste_they_can_use': ['All waste streams'],
                'resources_they_can_provide': ['Recycled materials', 'Waste processing'],
                'estimated_partnership_value': '$15K annually',
                'carbon_reduction': '5-15 tons CO2',
                'implementation_time': '1-3 months',
                'ai_generated': True
            }
        ]
        
        # Enhanced fallback green initiatives
        fallback_initiatives = [
            {
                'initiative_name': 'Waste Exchange Program',
                'description': 'Connect with local companies to exchange waste materials',
                'current_practice': 'Disposing waste in landfills',
                'greener_alternative': 'Exchange waste with partner companies',
                'cost_savings_per_month': '$2000',
                'carbon_reduction': '5 tons CO2',
                'implementation_cost': '$5000',
                'payback_period': '2.5 months',
                'difficulty': 'medium',
                'priority': 'high',
                'ai_generated': True
            },
            {
                'initiative_name': 'Energy Efficiency Audit',
                'description': 'Identify and implement energy-saving measures',
                'current_practice': 'Standard energy usage',
                'greener_alternative': 'Optimized energy consumption',
                'cost_savings_per_month': '$1500',
                'carbon_reduction': '3 tons CO2',
                'implementation_cost': '$10000',
                'payback_period': '6.7 months',
                'difficulty': 'easy',
                'priority': 'medium',
                'ai_generated': True
            }
        ]
        
        return {
            'predicted_outputs': fallback_outputs,
            'predicted_inputs': fallback_inputs,
            'company_suggestions': fallback_suggestions,
            'green_initiatives': fallback_initiatives,
            'ai_insights': {
                'symbiosis_score': '75%',
                'estimated_savings': '$42,000 annually',
                'carbon_reduction': '8.0 tons CO2 annually',
                'top_opportunities': ['Production Waste', 'Raw Materials'],
                'recommended_partners': ['Local Manufacturing Companies', 'Recycling Facilities'],
                'implementation_roadmap': [
                    'Review and approve AI-generated materials',
                    'Select preferred partner matches',
                    'Contact potential partners',
                    'Establish supply agreements',
                    'Implement green initiatives'
                ]
            }
        }

# Legacy function for backward compatibility
def generate_listings_from_profile(company_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    """
    service = ListingInferenceService()
    return service.generate_listings_from_profile(company_profile)

# Main execution for testing
if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            company_profile = json.loads(sys.argv[1])
            service = ListingInferenceService()
            result = service.generate_listings_from_profile(company_profile)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print('Error:', e)
    else:
        print(json.dumps({'error': 'No company profile provided'}, indent=2)) 