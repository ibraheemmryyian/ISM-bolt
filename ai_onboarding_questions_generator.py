import logging
import json
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIOnboardingQuestionsGenerator:
    """
    Advanced AI-Driven Onboarding Questions Generator
    Uses DeepSeek R1 to generate personalized questions for each company
    following the 80/20 rule for maximum impact with minimal effort
    """
    def __init__(self):
        self.deepseek_api_key = 'sk-7ce79f30332d45d5b3acb8968b052132'
        self.deepseek_base_url = 'https://api.deepseek.com/v1/chat/completions'
        self.deepseek_model = 'deepseek-r1'  # Using R1 for advanced reasoning
        self.industry_knowledge_templates = self._initialize_industry_templates()

    def _initialize_industry_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize industry-specific knowledge assessment templates"""
        return {
            "chemical_manufacturing": {
                "critical_knowledge_areas": [
                    "chemical_processes",
                    "hazardous_materials",
                    "waste_streams",
                    "safety_compliance",
                    "production_volumes"
                ],
                "material_categories": [
                    "raw_chemicals",
                    "solvents",
                    "catalysts",
                    "byproducts",
                    "waste_streams"
                ],
                "waste_types": [
                    "hazardous_waste",
                    "organic_waste",
                    "aqueous_waste",
                    "solid_waste",
                    "gaseous_waste"
                ]
            },
            "food_processing": {
                "critical_knowledge_areas": [
                    "food_safety",
                    "organic_waste",
                    "packaging_materials",
                    "production_cycles",
                    "seasonal_variations"
                ],
                "material_categories": [
                    "raw_ingredients",
                    "packaging_materials",
                    "preservatives",
                    "cleaning_supplies",
                    "organic_waste"
                ],
                "waste_types": [
                    "organic_waste",
                    "packaging_waste",
                    "water_waste",
                    "food_scraps",
                    "processing_waste"
                ]
            },
            "steel_manufacturing": {
                "critical_knowledge_areas": [
                    "steel_types",
                    "production_processes",
                    "slag_management",
                    "energy_consumption",
                    "quality_standards"
                ],
                "material_categories": [
                    "raw_materials",
                    "alloys",
                    "refractory_materials",
                    "slag_byproducts",
                    "energy_resources"
                ],
                "waste_types": [
                    "slag_waste",
                    "metal_scrap",
                    "dust_particulates",
                    "heat_waste",
                    "process_waste"
                ]
            },
            "textile_manufacturing": {
                "critical_knowledge_areas": [
                    "fiber_types",
                    "dyeing_processes",
                    "water_consumption",
                    "chemical_usage",
                    "waste_management"
                ],
                "material_categories": [
                    "raw_fibers",
                    "dyes_chemicals",
                    "water_resources",
                    "packaging_materials",
                    "textile_waste"
                ],
                "waste_types": [
                    "fabric_scraps",
                    "dye_waste",
                    "water_waste",
                    "chemical_waste",
                    "packaging_waste"
                ]
            },
            "automotive_manufacturing": {
                "critical_knowledge_areas": [
                    "production_volumes",
                    "material_specifications",
                    "quality_standards",
                    "supply_chain",
                    "waste_streams"
                ],
                "material_categories": [
                    "metals_alloys",
                    "plastics_polymers",
                    "electronic_components",
                    "lubricants_fluids",
                    "packaging_materials"
                ],
                "waste_types": [
                    "metal_scrap",
                    "plastic_waste",
                    "electronic_waste",
                    "hazardous_waste",
                    "packaging_waste"
                ]
            },
            "general_manufacturing": {
                "critical_knowledge_areas": [
                    "production_processes",
                    "material_requirements",
                    "waste_management",
                    "quality_standards",
                    "supply_chain"
                ],
                "material_categories": [
                    "raw_materials",
                    "packaging_materials",
                    "energy_resources",
                    "byproducts",
                    "waste_streams"
                ],
                "waste_types": [
                    "solid_waste",
                    "liquid_waste",
                    "packaging_waste",
                    "byproduct_waste",
                    "general_waste"
                ]
            }
        }

    def assess_company_knowledge_gaps(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"🔍 Assessing knowledge gaps for: {company_profile.get('name', 'Unknown')}")
            industry_category = self._categorize_industry(company_profile.get('industry', ''))
            knowledge_assessment = self._analyze_existing_knowledge(company_profile, industry_category)
            questions_data = self._generate_gap_filling_questions(company_profile, knowledge_assessment, industry_category)
            return {
                'knowledge_assessment': knowledge_assessment,
                'questions_data': questions_data,
                'industry_category': industry_category,
                'estimated_completion_time': questions_data.get('estimated_completion_time', '5-8 minutes'),
                'confidence_score': knowledge_assessment.get('confidence_score', 0.0)
            }
        except Exception as e:
            logger.error(f"❌ Error assessing knowledge gaps: {str(e)}")
            raise RuntimeError(f"Failed to assess company knowledge gaps: {str(e)}")

    def _categorize_industry(self, industry: str) -> str:
        industry_lower = industry.lower()
        if any(word in industry_lower for word in ['chemical', 'pharma', 'petrochemical', 'polymer']):
            return 'chemical_manufacturing'
        elif any(word in industry_lower for word in ['food', 'beverage', 'dairy', 'meat', 'grain', 'agriculture']):
            return 'food_processing'
        elif any(word in industry_lower for word in ['steel', 'metal', 'iron', 'aluminum', 'foundry']):
            return 'steel_manufacturing'
        elif any(word in industry_lower for word in ['textile', 'fabric', 'clothing', 'garment', 'apparel']):
            return 'textile_manufacturing'
        elif any(word in industry_lower for word in ['automotive', 'vehicle', 'car', 'truck', 'transportation']):
            return 'automotive_manufacturing'
        else:
            return 'general_manufacturing'

    def _analyze_existing_knowledge(self, company_profile: Dict[str, Any], industry_category: str) -> Dict[str, Any]:
        knowledge_areas = self.industry_knowledge_templates.get(industry_category, {}).get('critical_knowledge_areas', [])
        existing_data = {
            'company_name': bool(company_profile.get('name')),
            'industry': bool(company_profile.get('industry')),
            'location': bool(company_profile.get('location')),
            'employee_count': bool(company_profile.get('employee_count')),
            'products': bool(company_profile.get('products')),
            'main_materials': bool(company_profile.get('main_materials')),
            'production_volume': bool(company_profile.get('production_volume')),
            'process_description': bool(company_profile.get('process_description')),
            'waste_streams': bool(company_profile.get('waste_streams')),
            'sustainability_goals': bool(company_profile.get('sustainability_goals'))
        }
        total_fields = len(existing_data)
        filled_fields = sum(existing_data.values())
        confidence_score = filled_fields / total_fields if total_fields > 0 else 0.0
        critical_gaps = []
        if not existing_data['products']:
            critical_gaps.append('production_processes')
        if not existing_data['main_materials']:
            critical_gaps.append('material_requirements')
        if not existing_data['production_volume']:
            critical_gaps.append('production_scale')
        if not existing_data['waste_streams']:
            critical_gaps.append('waste_management')
        return {
            'confidence_score': confidence_score,
            'existing_data': existing_data,
            'critical_gaps': critical_gaps,
            'knowledge_areas': knowledge_areas,
            'data_completeness': f"{filled_fields}/{total_fields} fields completed"
        }

    def _generate_gap_filling_questions(self, company_profile: Dict[str, Any], knowledge_assessment: Dict[str, Any], industry_category: str) -> Dict[str, Any]:
        try:
            logger.info(f"🎯 Generating gap-filling questions for {company_profile.get('name')}")
            prompt = self._create_gap_filling_prompt(company_profile, knowledge_assessment, industry_category)
            response = self._call_deepseek_r1(prompt)
            if response:
                questions_data = self._parse_questions_response(response)
                logger.info(f"✅ Generated {len(questions_data.get('questions', []))} targeted questions")
                return questions_data
            else:
                logger.error(f"❌ Failed to generate gap-filling questions: DeepSeek returned no response")
                raise RuntimeError("DeepSeek API failed to generate questions and no fallback is allowed in production.")
        except Exception as e:
            logger.error(f"❌ Error generating gap-filling questions: {str(e)}")
            raise RuntimeError(f"Failed to generate gap-filling questions: {str(e)}")

    def _create_gap_filling_prompt(self, company_profile: Dict[str, Any], knowledge_assessment: Dict[str, Any], industry_category: str) -> str:
        company_name = company_profile.get('name', 'Unknown')
        industry = company_profile.get('industry', 'Unknown')
        location = company_profile.get('location', 'Unknown')
        products = company_profile.get('products', 'Unknown')
        employee_count = company_profile.get('employee_count', 0)
        critical_gaps = knowledge_assessment.get('critical_gaps', [])
        confidence_score = knowledge_assessment.get('confidence_score', 0.0)
        industry_template = self.industry_knowledge_templates.get(industry_category, {})
        material_categories = industry_template.get('material_categories', [])
        waste_types = industry_template.get('waste_types', [])
        prompt = f"""
You are an expert industrial symbiosis consultant conducting a targeted onboarding session for a company. Your goal is to gather the most critical missing information needed for accurate waste and resource matching.

COMPANY PROFILE:
- Name: {company_name}
- Industry: {industry}
- Location: {location}
- Products: {products}
- Employee Count: {employee_count}
- Industry Category: {industry_category}

KNOWLEDGE ASSESSMENT:
- Confidence Score: {confidence_score:.2f}
- Critical Gaps: {', '.join(critical_gaps)}
- Data Completeness: {knowledge_assessment.get('data_completeness', 'Unknown')}

INDUSTRY-SPECIFIC CONTEXT:
- Material Categories: {', '.join(material_categories)}
- Waste Types: {', '.join(waste_types)}

TASK: Generate 5-8 targeted questions that will fill the critical knowledge gaps and provide the most valuable information for industrial symbiosis matching. Focus on questions that will reveal:

1. **Production Processes & Materials** (2-3 questions)
   - Specific processes, material types, quantities
   - This directly impacts waste generation and resource needs

2. **Waste Streams & Management** (2-3 questions)
   - Current waste types, quantities, disposal methods
   - This reveals immediate symbiosis opportunities

3. **Resource Requirements** (1-2 questions)
   - What resources are needed, current suppliers
   - This identifies potential supply chain opportunities

4. **Operational Details** (1 question)
   - Production schedules, quality requirements
   - This affects logistics and matching feasibility

CRITICAL REQUIREMENTS:
- Questions must be specific to {industry_category} industry
- Focus on quantifiable information (volumes, frequencies, costs)
- Questions should be answerable in 1-2 sentences
- Prioritize questions that will have the biggest impact on matching accuracy
- Include follow-up questions for deeper insights

Provide the response as a JSON object with this structure:
{{
    "questions": [
        {{
            "id": "q1",
            "category": "production_processes",
            "question": "What specific {industry_category} processes do you use?",
            "importance": "high",
            "expected_answer_type": "multiselect",
            "options": ["process1", "process2", "process3"],
            "follow_up_question": "Which process generates the most waste?",
            "reasoning": "Essential for identifying waste streams and resource needs"
        }}
    ],
    "estimated_completion_time": "5-8 minutes",
    "key_insights_expected": [
        "Production capacity and waste generation potential",
        "Primary resource requirements and waste streams",
        "Current waste management practices and opportunities"
    ],
    "material_listings_focus": [
        "waste_materials",
        "required_materials",
        "byproducts"
    ]
}}
"""
        return prompt

    def _call_deepseek_r1(self, prompt: str) -> Optional[Dict[str, Any]]:
        headers = {
            'Authorization': f'Bearer {self.deepseek_api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': self.deepseek_model,
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            'temperature': 0.2,
            'max_tokens': 1024
        }
        try:
            response = requests.post(self.deepseek_base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            # DeepSeek returns the content in choices[0]['message']['content']
            content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            return json.loads(content)
        except Exception as e:
            logger.error(f"DeepSeek R1 API call failed: {str(e)}")
            return None

    def _parse_questions_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if 'questions' in response:
                return response
            # If response is a string, try to parse as JSON
            if isinstance(response, str):
                return json.loads(response)
            raise ValueError('No questions found in DeepSeek response')
        except Exception as e:
            logger.error(f"Error parsing questions response: {str(e)}")
            raise RuntimeError(f"Failed to parse questions response: {str(e)}")

    def generate_onboarding_questions(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        return self.assess_company_knowledge_gaps(company_profile)

    def generate_material_listings_from_answers(self, company_profile: Dict[str, Any], answers: Dict[str, str]) -> Dict[str, Any]:
        try:
            logger.info(f"🏭 Generating material listings for: {company_profile.get('name', 'Unknown')}")
            processed_data = self._process_answers_for_listings(company_profile, answers)
            listings = self._generate_industry_specific_listings(processed_data)
            waste_requirements = self._generate_waste_requirements(processed_data)
            return {
                'material_listings': listings,
                'waste_requirements': waste_requirements,
                'generated_at': datetime.now().isoformat(),
                'ai_model': self.deepseek_model,
                'confidence_score': processed_data.get('confidence_score', 0.0)
            }
        except Exception as e:
            logger.error(f"❌ Error generating material listings: {str(e)}")
            raise RuntimeError(f"Failed to generate material listings: {str(e)}")

    def _process_answers_for_listings(self, company_profile: Dict[str, Any], answers: Dict[str, str]) -> Dict[str, Any]:
        industry_category = self._categorize_industry(company_profile.get('industry', ''))
        processed_data = {
            'company_name': company_profile.get('name', 'Unknown'),
            'industry_category': industry_category,
            'production_processes': [],
            'main_materials': [],
            'waste_streams': [],
            'production_volume': '',
            'location': company_profile.get('location', ''),
            'employee_count': company_profile.get('employee_count', 0)
        }
        for question_id, answer in answers.items():
            if 'process' in question_id.lower() or 'production' in question_id.lower():
                if isinstance(answer, list):
                    processed_data['production_processes'].extend(answer)
                else:
                    processed_data['production_processes'].append(answer)
            elif 'material' in question_id.lower() or 'raw' in question_id.lower():
                if isinstance(answer, list):
                    processed_data['main_materials'].extend(answer)
                else:
                    processed_data['main_materials'].append(answer)
            elif 'waste' in question_id.lower():
                if isinstance(answer, list):
                    processed_data['waste_streams'].extend(answer)
                else:
                    processed_data['waste_streams'].append(answer)
            elif 'volume' in question_id.lower() or 'quantity' in question_id.lower():
                processed_data['production_volume'] = answer
        total_fields = 4  # processes, materials, waste, volume
        filled_fields = sum([
            len(processed_data['production_processes']) > 0,
            len(processed_data['main_materials']) > 0,
            len(processed_data['waste_streams']) > 0,
            bool(processed_data['production_volume'])
        ])
        processed_data['confidence_score'] = filled_fields / total_fields if total_fields > 0 else 0.0
        return processed_data

    def _generate_industry_specific_listings(self, processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        industry_category = processed_data.get('industry_category', 'general_manufacturing')
        industry_template = self.industry_knowledge_templates.get(industry_category, {})
        listings = []
        waste_types = industry_template.get('waste_types', [])
        for waste_type in waste_types[:3]:
            listing = {
                'material_name': waste_type.replace('_', ' ').title(),
                'type': 'waste',
                'quantity': self._estimate_quantity(processed_data, 'waste'),
                'unit': self._get_appropriate_unit(waste_type),
                'description': f"High-quality {waste_type.replace('_', ' ')} from {processed_data['company_name']} production processes",
                'frequency': 'monthly',
                'quality_grade': 'A',
                'ai_generated': True,
                'industry_specific': True
            }
            listings.append(listing)
        material_categories = industry_template.get('material_categories', [])
        for material_category in material_categories[:3]:
            listing = {
                'material_name': material_category.replace('_', ' ').title(),
                'type': 'requirement',
                'quantity': self._estimate_quantity(processed_data, 'requirement'),
                'unit': self._get_appropriate_unit(material_category),
                'description': f"Seeking reliable supplier for {material_category.replace('_', ' ')} for {processed_data['company_name']} production",
                'frequency': 'monthly',
                'quality_grade': 'A',
                'ai_generated': True,
                'industry_specific': True
            }
            listings.append(listing)
        return listings

    def _generate_waste_requirements(self, processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        industry_category = processed_data.get('industry_category', 'general_manufacturing')
        requirements = []
        if industry_category == 'chemical_manufacturing':
            requirements.extend([
                {
                    'requirement_type': 'hazardous_waste_disposal',
                    'description': 'Safe disposal of hazardous chemical waste',
                    'priority': 'high',
                    'compliance_required': True
                },
                {
                    'requirement_type': 'waste_treatment',
                    'description': 'On-site or off-site waste treatment facilities',
                    'priority': 'medium',
                    'compliance_required': True
                }
            ])
        elif industry_category == 'food_processing':
            requirements.extend([
                {
                    'requirement_type': 'organic_waste_management',
                    'description': 'Composting or anaerobic digestion for organic waste',
                    'priority': 'high',
                    'compliance_required': False
                },
                {
                    'requirement_type': 'packaging_recycling',
                    'description': 'Recycling program for packaging materials',
                    'priority': 'medium',
                    'compliance_required': False
                }
            ])
        elif industry_category == 'steel_manufacturing':
            requirements.extend([
                {
                    'requirement_type': 'slag_utilization',
                    'description': 'Slag processing and utilization for construction materials',
                    'priority': 'high',
                    'compliance_required': False
                },
                {
                    'requirement_type': 'heat_recovery',
                    'description': 'Waste heat recovery systems',
                    'priority': 'medium',
                    'compliance_required': False
                }
            ])
        return requirements

    def _estimate_quantity(self, processed_data: Dict[str, Any], listing_type: str) -> int:
        employee_count = processed_data.get('employee_count', 100)
        if employee_count <= 50:
            base_quantity = 100 if listing_type == 'waste' else 500
        elif employee_count <= 200:
            base_quantity = 500 if listing_type == 'waste' else 1000
        elif employee_count <= 500:
            base_quantity = 1000 if listing_type == 'waste' else 2000
        else:
            base_quantity = 2000 if listing_type == 'waste' else 5000
        production_volume = processed_data.get('production_volume', '')
        if production_volume:
            numbers = re.findall(r'[\d,]+', production_volume)
            if numbers:
                volume_number = int(numbers[0].replace(',', ''))
                if volume_number > 10000:
                    base_quantity *= 2
                elif volume_number > 5000:
                    base_quantity *= 1.5
        return int(base_quantity)

    def _get_appropriate_unit(self, material_type: str) -> str:
        if any(word in material_type.lower() for word in ['waste', 'scrap', 'solid']):
            return 'tons'
        elif any(word in material_type.lower() for word in ['liquid', 'aqueous', 'solvent']):
            return 'liters'
        elif any(word in material_type.lower() for word in ['gas', 'gaseous']):
            return 'cubic meters'
        else:
            return 'kg'

    def process_onboarding_answers(self, questions: List[Dict[str, Any]], answers: Dict[str, str]) -> Dict[str, Any]:
        # This method can be expanded for more advanced answer processing if needed
        return self.generate_material_listings_from_answers({}, answers)

def process_company_onboarding_answers(questions: List[Dict[str, Any]], answers: Dict[str, Any]) -> Dict[str, Any]:
    generator = AIOnboardingQuestionsGenerator()
    return generator.process_onboarding_answers(questions, answers)

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ai_onboarding_questions_generator.py <command> [data]")
        sys.exit(1)
    command = sys.argv[1]
    generator = AIOnboardingQuestionsGenerator()
    try:
        if command == 'assess-knowledge':
            if len(sys.argv) < 3:
                print("Error: Company profile data required")
                sys.exit(1)
            company_profile = json.loads(sys.argv[2])
            result = generator.assess_company_knowledge_gaps(company_profile)
            print(json.dumps(result))
        elif command == 'generate-questions':
            if len(sys.argv) < 3:
                print("Error: Company profile and knowledge assessment data required")
                sys.exit(1)
            data = json.loads(sys.argv[2])
            company_profile = data.get('companyProfile', {})
            knowledge_assessment = data.get('knowledgeAssessment', {})
            result = generator._generate_gap_filling_questions(company_profile, knowledge_assessment, generator._categorize_industry(company_profile.get('industry', '')))
            print(json.dumps(result))
        elif command == 'generate-listings':
            if len(sys.argv) < 3:
                print("Error: Company profile and answers data required")
                sys.exit(1)
            data = json.loads(sys.argv[2])
            company_profile = data.get('companyProfile', {})
            answers = data.get('answers', {})
            result = generator.generate_material_listings_from_answers(company_profile, answers)
            print(json.dumps(result))
        elif command == 'generate-onboarding-questions':
            if len(sys.argv) < 3:
                print("Error: Company profile data required")
                sys.exit(1)
            company_profile = json.loads(sys.argv[2])
            result = generator.generate_onboarding_questions(company_profile)
            print(json.dumps(result))
        else:
            print(f"Unknown command: {command}")
            print("Available commands: assess-knowledge, generate-questions, generate-listings, generate-onboarding-questions")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(json.dumps({
            'error': str(e),
            'success': False
        }))
        sys.exit(1)

if __name__ == "__main__":
    main() 