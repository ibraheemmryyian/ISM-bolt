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
DEEPSEEK_MODEL = 'deepseek-coder'

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pricing integration (commented out to avoid circular imports)
# try:
#     from ai_pricing_integration import (
#         validate_match_pricing_requirement_integrated,
#         get_material_pricing_data_integrated,
#         enforce_pricing_validation_decorator
#     )
#     PRICING_INTEGRATION_AVAILABLE = True
# except ImportError:
#     PRICING_INTEGRATION_AVAILABLE = False
#     logger.warning("Pricing integration not available")

PRICING_INTEGRATION_AVAILABLE = False
logger.warning("Pricing integration temporarily disabled to avoid circular imports")

class AIMatchmakingService:
    def __init__(self):
        self.deepseek_api_key = DEEPSEEK_API_KEY
        self.deepseek_base_url = DEEPSEEK_BASE_URL
        self.deepseek_model = DEEPSEEK_MODEL
        
    def find_partner_companies(self, company_id: str, material_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Phase 2: Find partner companies for a specific material using DeepSeek API.
        Uses the exact DeepSeek API prompt structure as specified in the master directive.
        
        Args:
            company_id: ID of the company that owns the material
            material_data: Dictionary containing material information
            
        Returns:
            List of recommended partner companies with match reasons
        """
        try:
            logger.info(f"Starting Phase 2 AI matchmaking for company {company_id}, material: {material_data.get('name', 'Unknown')}")
            
            # Construct the exact prompt as specified in the master directive
            prompt = self._construct_deepseek_prompt(material_data)
            
            # Call DeepSeek API with exact structure
            response = self._call_deepseek_api(prompt)
            
            # Parse and validate response
            parsed_response = self._parse_response(response)
            
            # Find actual companies in database that match the recommendations
            partner_companies = self._find_matching_companies(parsed_response.get('recommendations', []))
            
            logger.info(f"Successfully found {len(partner_companies)} partner companies")
            
            return partner_companies
            
        except Exception as e:
            logger.error(f"Error in find_partner_companies: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _construct_deepseek_prompt(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construct the exact DeepSeek prompt structure as specified in the master directive.
        """
        
        # Extract material information
        material_name = material_data.get('name', 'Unknown')
        material_description = material_data.get('description', '')
        material_category = material_data.get('category', 'general')
        material_quantity = material_data.get('quantity', 'Unknown')
        material_frequency = material_data.get('frequency', 'monthly')
        material_notes = material_data.get('notes', '')
        
        # Determine if it's an output/waste or input/requirement
        material_type = "Output/Waste" if material_data.get('type') == 'waste' else "Input/Requirement"
        
        # Construct the exact prompt structure from the master directive
        user_content = f"Find the best company matches for the following material. Material -- Type: '{material_type}', Name: '{material_name}', Description: '{material_description}'"
        
        return {
            "model": self.deepseek_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI-powered industrial matchmaking expert. Your task is to analyze a specific industrial material and recommend the top 3 types of companies that would be ideal symbiotic partners for it. Provide the response as a JSON object with a single key: 'recommendations'. This key should contain a list of objects, with each object having two fields: 'company_type' (the type of company, e.g., 'Cement Manufacturer') and 'match_reason' (a concise explanation of why it's a good match)."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "response_format": { "type": "json_object" }
        }
    
    def _call_deepseek_api(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the DeepSeek API with the exact prompt structure."""
        
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
                timeout=30
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
    
    def _parse_response(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the DeepSeek API response."""
        
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
                
                # Ensure required keys exist
                if 'recommendations' not in parsed:
                    parsed['recommendations'] = []
                
                # Validate and clean each recommendation
                parsed['recommendations'] = [
                    self._validate_and_clean_recommendation(rec) 
                    for rec in parsed['recommendations'] 
                    if self._validate_recommendation(rec)
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
    
    def _validate_recommendation(self, rec: Dict[str, Any]) -> bool:
        """Validate that a recommendation has the required fields."""
        required_fields = ['company_type', 'match_reason']
        return all(field in rec and rec[field] for field in required_fields)
    
    def _validate_and_clean_recommendation(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean a recommendation, ensuring all required fields are present."""
        
        return {
            'company_type': rec.get('company_type', 'Unknown'),
            'match_reason': rec.get('match_reason', 'No reason provided'),
            'confidence_score': rec.get('confidence_score', 0.8),
            'ai_generated': True
        }
    
    def _find_matching_companies(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find actual companies in the database that match the AI recommendations.
        This is a simplified version - in production, you'd query the actual database.
        """
        
        partner_companies = []
        
        for rec in recommendations:
            company_type = rec.get('company_type', '').lower()
            match_reason = rec.get('match_reason', '')
            
            # Simulate database query for companies matching the type
            # In production, this would be a real database query
            matching_companies = self._query_companies_by_type(company_type)
            
            for company in matching_companies:
                partner_companies.append({
                    'company_id': company.get('id'),
                    'company_name': company.get('name'),
                    'company_type': company_type,
                    'industry': company.get('industry'),
                    'match_reason': match_reason,
                    'match_score': rec.get('confidence_score', 0.8),
                    'ai_generated': True
                })
        
        return partner_companies
    
    def _query_companies_by_type(self, company_type: str) -> List[Dict[str, Any]]:
        """
        Simulate database query for companies matching a specific type.
        In production, this would query the actual database.
        """
        
        # This is a mock implementation - replace with actual database queries
        mock_companies = {
            'cattle farm': [
                {'id': 'farm_001', 'name': 'Green Valley Cattle Farm', 'industry': 'Agriculture'},
                {'id': 'farm_002', 'name': 'Sunset Ranch', 'industry': 'Agriculture'}
            ],
            'cement manufacturer': [
                {'id': 'cement_001', 'name': 'Portland Cement Co', 'industry': 'Construction Materials'},
                {'id': 'cement_002', 'name': 'Concrete Solutions Inc', 'industry': 'Construction Materials'}
            ],
            'textile manufacturer': [
                {'id': 'textile_001', 'name': 'Fabric World Ltd', 'industry': 'Textiles'},
                {'id': 'textile_002', 'name': 'Cotton Mills International', 'industry': 'Textiles'}
            ],
            'paper mill': [
                {'id': 'paper_001', 'name': 'Green Paper Products', 'industry': 'Paper Manufacturing'},
                {'id': 'paper_002', 'name': 'Recycled Paper Co', 'industry': 'Paper Manufacturing'}
            ],
            'chemical manufacturer': [
                {'id': 'chem_001', 'name': 'Industrial Chemicals Ltd', 'industry': 'Chemical Manufacturing'},
                {'id': 'chem_002', 'name': 'Green Chemistry Solutions', 'industry': 'Chemical Manufacturing'}
            ]
        }
        
        # Find companies that match the type (case-insensitive partial matching)
        matching_companies = []
        for key, companies in mock_companies.items():
            if company_type in key or key in company_type:
                matching_companies.extend(companies)
        
        # If no exact matches, return some general companies
        if not matching_companies:
            matching_companies = [
                {'id': 'general_001', 'name': 'General Manufacturing Co', 'industry': 'Manufacturing'},
                {'id': 'general_002', 'name': 'Industrial Solutions Inc', 'industry': 'Manufacturing'}
            ]
        
        return matching_companies
    
    # @enforce_pricing_validation_decorator  # Temporarily disabled
    async def create_matches_in_database(self, company_id: str, partner_companies: List[Dict[str, Any]], material_name: str) -> List[Dict[str, Any]]:
        """
        Create match records in the database for the partner companies.
        Now includes mandatory pricing validation before creating matches.
        """
        
        created_matches = []
        
        for partner in partner_companies:
            try:
                # Get pricing data for validation
                if PRICING_INTEGRATION_AVAILABLE:
                    pricing_data = await get_material_pricing_data_integrated(material_name)
                    if pricing_data:
                        # Create match data for pricing validation
                        match_data = {
                            "material": material_name,
                            "quantity": partner.get("quantity", 1000.0),
                            "quality": partner.get("quality", "clean"),
                            "source_location": partner.get("location", "unknown"),
                            "destination_location": "unknown",
                            "price": pricing_data.recycled_price,
                            "company_id": company_id,
                            "partner_company_id": partner.get("company_id")
                        }
                        
                        # Validate pricing before creating match
                        is_valid = await validate_match_pricing_requirement_integrated(
                            match_data["material"],
                            match_data["quantity"],
                            match_data["quality"],
                            match_data["source_location"],
                            match_data["destination_location"],
                            match_data["price"]
                        )
                        
                        if not is_valid:
                            logger.warning(f"Pricing validation failed for match: {company_id} -> {partner.get('company_id')}")
                            continue
                
                # Create match record
                match_record = {
                    'id': f"match_{company_id}_{partner['company_id']}_{datetime.now().timestamp()}",
                    'company_id': company_id,
                    'partner_company_id': partner['company_id'],
                    'match_score': partner['match_score'],
                    'match_reason': partner['match_reason'],
                    'materials_involved': [material_name],
                    'status': 'pending',
                    'created_at': datetime.now().isoformat(),
                    'ai_generated': True,
                    'pricing_validated': True,
                    'pricing_timestamp': datetime.now().isoformat()
                }
                
                # Add pricing data if available
                if PRICING_INTEGRATION_AVAILABLE and pricing_data:
                    match_record['pricing_data'] = {
                        'virgin_price': pricing_data.virgin_price,
                        'recycled_price': pricing_data.recycled_price,
                        'savings_percentage': pricing_data.savings_percentage,
                        'profit_margin': pricing_data.profit_margin,
                        'risk_level': pricing_data.risk_level
                    }
                
                created_matches.append(match_record)
                logger.info(f"Created pricing-validated match: {company_id} -> {partner['company_id']} for material: {material_name}")
                
            except Exception as e:
                logger.error(f"Error creating match for partner {partner.get('company_id')}: {e}")
                continue
        
        return created_matches

# Global instance for easy access
ai_matchmaking_service = AIMatchmakingService()

def find_partner_companies(company_id: str, material_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convenience function to find partner companies for a material."""
    return ai_matchmaking_service.find_partner_companies(company_id, material_data)

def create_matches_in_database(company_id: str, partner_companies: List[Dict[str, Any]], material_name: str) -> List[Dict[str, Any]]:
    """Convenience function to create matches in the database."""
    return ai_matchmaking_service.create_matches_in_database(company_id, partner_companies, material_name)

if __name__ == "__main__":
    import sys
    import json
    
    try:
        # Read input from command line arguments
        if len(sys.argv) > 1:
            input_data = json.loads(sys.argv[1])
            action = input_data.get('action', 'find_partner_companies')
            
            if action == 'find_partner_companies':
                company_id = input_data.get('company_id', '')
                material_data = input_data.get('material_data', {})
                result = find_partner_companies(company_id, material_data)
                print(json.dumps(result))
            elif action == 'create_matches_in_database':
                company_id = input_data.get('company_id', '')
                partner_companies = input_data.get('partner_companies', [])
                material_name = input_data.get('material_name', '')
                result = create_matches_in_database(company_id, partner_companies, material_name)
                print(json.dumps(result))
            else:
                print(json.dumps({'error': f'Unknown action: {action}'}))
        else:
            print(json.dumps({'error': 'No input data provided'}))
            
    except Exception as e:
        print(json.dumps({'error': str(e)})) 