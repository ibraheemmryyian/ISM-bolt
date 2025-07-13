import json
import logging
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

        # Industry-specific knowledge assessment templates
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
            }
        }

    def assess_company_knowledge_gaps(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess what knowledge gaps exist for a company and determine what questions to ask
        """
        try:
            logger.info(f"🔍 Assessing knowledge gaps for: {company_profile.get('name', 'Unknown')}")

            # Determine industry category
            industry_category = self._categorize_industry(company_profile.get('industry', ''))

            # Analyze existing profile data
            knowledge_assessment = self._analyze_existing_knowledge(company_profile, industry_category)

            # Generate targeted questions based on gaps
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
            return self._get_fallback_assessment(company_profile)

    def _categorize_industry(self, industry: str) -> str:
        """Categorize industry into specific manufacturing types"""
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
        """Analyze what information is already available and what's missing"""
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

        # Calculate confidence score based on available data
        total_fields = len(existing_data)
        filled_fields = sum(existing_data.values())
        confidence_score = filled_fields / total_fields if total_fields > 0 else 0.0

        # Identify critical gaps
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
        """Generate targeted questions to fill knowledge gaps"""
        try:
            logger.info(f"🎯 Generating gap-filling questions for {company_profile.get('name')}")

            # Create the prompt for targeted question generation
            prompt = self._create_gap_filling_prompt(company_profile, knowledge_assessment, industry_category)

            # Call DeepSeek R1
            response = self._call_deepseek_r1(prompt)

            if response:
                questions_data = self._parse_questions_response(response)
                logger.info(f"✅ Generated {len(questions_data.get('questions', []))} targeted questions")
                return questions_data
            else:
                logger.error(f"❌ Failed to generate gap-filling questions")
                return self._get_fallback_questions(company_profile, industry_category)

        except Exception as e:
            logger.error(f"❌ Error generating gap-filling questions: {str(e)}")
            return self._get_fallback_questions(company_profile, industry_category)

    def _create_gap_filling_prompt(self, company_profile: Dict[str, Any], knowledge_assessment: Dict[str, Any], industry_category: str) -> str:
        """Create a prompt for generating gap-filling questions"""

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
        """
        Call DeepSeek R1 API for advanced reasoning
        """
        try:
            prompt_data = {
                "model": self.deepseek_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert industrial symbiosis consultant with deep knowledge of manufacturing processes, waste streams, and resource optimization. Your questions are precise, industry-specific, and designed to extract maximum value with minimal effort."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,  # Low temperature for consistent, focused questions
                "max_tokens": 2000
            }

            headers = {
                'Authorization': f'Bearer {self.deepseek_api_key}',
                'Content-Type': 'application/json'
            }

            response = requests.post(
                self.deepseek_base_url,
                headers=headers,
                json=prompt_data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    if isinstance(content, str):
                        return json.loads(content)
                    else:
                        return content
                else:
                    logger.error("Invalid response structure from DeepSeek R1")
                    return None
            else:
                logger.error(f"DeepSeek R1 API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error calling DeepSeek R1: {str(e)}")
            return None

    def _parse_questions_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate the questions response
        """
        try:
            # Validate structure
            if 'questions' not in response:
                logger.error("Response missing 'questions' key")
                return self._get_fallback_questions({}, 'general_manufacturing')

            questions = response.get('questions', [])
            if not questions:
                logger.error("No questions generated")
                return self._get_fallback_questions({}, 'general_manufacturing')

            # Validate each question
            validated_questions = []
            for i, question in enumerate(questions):
                if isinstance(question, dict) and 'question' in question:
                    validated_question = {
                        'id': question.get('id', f'q{i+1}'),
                        'category': question.get('category', 'general'),
                        'question': question.get('question', ''),
                        'importance': question.get('importance', 'medium'),
                        'expected_answer_type': question.get('expected_answer_type', 'text'),
                        'options': question.get('options', []),
                        'follow_up_question': question.get('follow_up_question', ''),
                        'reasoning': question.get('reasoning', '')
                    }
                    validated_questions.append(validated_question)

            return {
                'questions': validated_questions,
                'estimated_completion_time': response.get('estimated_completion_time', '5-8 minutes'),
                'key_insights_expected': response.get('key_insights_expected', []),
                'material_listings_focus': response.get('material_listings_focus', [])
            }

        except Exception as e:
            logger.error(f"Error parsing questions response: {str(e)}")
            return self._get_fallback_questions({}, 'general_manufacturing')

    def _get_fallback_questions(self, company_profile: Dict[str, Any], industry_category: str) -> Dict[str, Any]:
        """
        Fallback questions if AI generation fails
        """
        industry_template = self.industry_knowledge_templates.get(industry_category, {})
        material_categories = industry_template.get('material_categories', [])
        waste_types = industry_template.get('waste_types', [])

        # Industry-specific fallback questions
        fallback_questions = {
            "chemical_manufacturing": [
                {
                    "id": "q1",
                    "category": "production_processes",
                    "question": "What chemical processes do you use in production?",
                    "importance": "high",
                    "expected_answer_type": "multiselect",
                    "options": ["synthesis", "distillation", "crystallization", "filtration", "drying"],
                    "follow_up_question": "Which process generates the most waste?",
                    "reasoning": "Essential for identifying waste streams and resource needs"
                },
                {
                    "id": "q2",
                    "category": "waste_management",
                    "question": "What types of hazardous waste do you generate?",
                    "importance": "high",
                    "expected_answer_type": "multiselect",
                    "options": ["solvents", "acids", "bases", "heavy_metals", "organic_waste"],
                    "follow_up_question": "How do you currently dispose of hazardous waste?",
                    "reasoning": "Critical for compliance and symbiosis opportunities"
                }
            ],
            "food_processing": [
                {
                    "id": "q1",
                    "category": "production_processes",
                    "question": "What food processing methods do you use?",
                    "importance": "high",
                    "expected_answer_type": "multiselect",
                    "options": ["cooking", "freezing", "canning", "drying", "fermentation"],
                    "follow_up_question": "What is your daily production volume?",
                    "reasoning": "Determines waste generation and resource needs"
                },
                {
                    "id": "q2",
                    "category": "waste_management",
                    "question": "What organic waste streams do you generate?",
                    "importance": "high",
                    "expected_answer_type": "multiselect",
                    "options": ["food_scraps", "peels", "cores", "packaging", "water_waste"],
                    "follow_up_question": "Do you have any waste treatment facilities?",
                    "reasoning": "Identifies immediate symbiosis opportunities"
                }
            ],
            "steel_manufacturing": [
                {
                    "id": "q1",
                    "category": "production_processes",
                    "question": "What steel production processes do you use?",
                    "importance": "high",
                    "expected_answer_type": "multiselect",
                    "options": ["blast_furnace", "electric_arc", "basic_oxygen", "rolling", "casting"],
                    "follow_up_question": "What is your monthly steel production capacity?",
                    "reasoning": "Determines waste generation and resource requirements"
                },
                {
                    "id": "q2",
                    "category": "waste_management",
                    "question": "How do you handle slag and metal scrap?",
                    "importance": "high",
                    "expected_answer_type": "text",
                    "follow_up_question": "What percentage of slag is currently recycled?",
                    "reasoning": "Slag utilization is a major symbiosis opportunity"
                }
            ],
            "textile_manufacturing": [
                {
                    "id": "q1",
                    "category": "production_processes",
                    "question": "What textile manufacturing processes do you use?",
                    "importance": "high",
                    "expected_answer_type": "multiselect",
                    "options": ["spinning", "weaving", "knitting", "dyeing", "finishing"],
                    "follow_up_question": "What is your daily fabric production?",
                    "reasoning": "Determines waste generation and resource needs"
                },
                {
                    "id": "q2",
                    "category": "waste_management",
                    "question": "What types of textile waste do you generate?",
                    "importance": "high",
                    "expected_answer_type": "multiselect",
                    "options": ["fabric_scraps", "yarn_waste", "dye_waste", "water_waste", "packaging"],
                    "follow_up_question": "How do you currently handle textile waste?",
                    "reasoning": "Identifies recycling and reuse opportunities"
                }
            ],
            "automotive_manufacturing": [
                {
                    "id": "q1",
                    "category": "production_processes",
                    "question": "What automotive manufacturing processes do you use?",
                    "importance": "high",
                    "expected_answer_type": "multiselect",
                    "options": ["stamping", "welding", "painting", "assembly", "testing"],
                    "follow_up_question": "What is your monthly production capacity?",
                    "reasoning": "Determines waste generation and resource requirements"
                },
                {
                    "id": "q2",
                    "category": "waste_management",
                    "question": "What types of manufacturing waste do you generate?",
                    "importance": "high",
                    "expected_answer_type": "multiselect",
                    "options": ["metal_scrap", "plastic_waste", "paint_waste", "packaging", "electronic_waste"],
                    "follow_up_question": "How do you currently handle metal scrap?",
                    "reasoning": "Metal recycling is a major symbiosis opportunity"
                }
            ],
            "general_manufacturing": [
                {
                    "id": "q1",
                    "category": "production_processes",
                    "question": "What are your main production processes?",
                    "importance": "high",
                    "expected_answer_type": "text",
                    "follow_up_question": "What is your daily production volume?",
                    "reasoning": "Essential for understanding waste generation"
                },
                {
                    "id": "q2",
                    "category": "waste_management",
                    "question": "What types of waste do you generate?",
                    "importance": "high",
                    "expected_answer_type": "text",
                    "follow_up_question": "How do you currently handle waste disposal?",
                    "reasoning": "Identifies symbiosis opportunities"
                }
            ]
        }

        # Get industry-specific questions or default to manufacturing
        questions = fallback_questions.get(industry_category, fallback_questions['general_manufacturing'])

        return {
            'questions': questions,
            'estimated_completion_time': '5-8 minutes',
            'key_insights_expected': [
                'Production capacity and waste generation potential',
                'Primary resource requirements and waste streams',
                'Current waste management practices and opportunities'
            ],
            'material_listings_focus': [
                'waste_materials',
                'required_materials',
                'byproducts'
            ],
            'note': 'Using fallback questions due to AI generation failure'
        }

    def _get_fallback_assessment(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback assessment if AI generation fails"""
        industry_category = self._categorize_industry(company_profile.get('industry', ''))
        knowledge_assessment = self._analyze_existing_knowledge(company_profile, industry_category)
        questions_data = self._get_fallback_questions(company_profile, industry_category)

        return {
            'knowledge_assessment': knowledge_assessment,
            'questions_data': questions_data,
            'industry_category': industry_category,
            'estimated_completion_time': '5-8 minutes',
            'confidence_score': knowledge_assessment.get('confidence_score', 0.0)
        }

    def generate_onboarding_questions(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personalized onboarding questions for a company
        (Legacy method for backward compatibility)
        """
        return self.assess_company_knowledge_gaps(company_profile)

    def generate_material_listings_from_answers(self, company_profile: Dict[str, Any], answers: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate material listings and waste requirements based on onboarding answers
        """
        try:
            logger.info(f"🏭 Generating material listings for: {company_profile.get('name', 'Unknown')}")

            # Process the answers to extract key information
            processed_data = self._process_answers_for_listings(company_profile, answers)

            # Generate industry-specific material listings
            listings = self._generate_industry_specific_listings(processed_data)

            # Generate waste requirements
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
            return self._get_fallback_listings(company_profile, answers)

    def _process_answers_for_listings(self, company_profile: Dict[str, Any], answers: Dict[str, str]) -> Dict[str, Any]:
        """Process onboarding answers to extract key information for listings"""
        industry_category = self._categorize_industry(company_profile.get('industry', ''))

        # Extract key information from answers
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

        # Process each answer to extract relevant information
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

        # Calculate confidence score based on data completeness
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
        """Generate industry-specific material listings"""
        industry_category = processed_data.get('industry_category', 'general_manufacturing')
        industry_template = self.industry_knowledge_templates.get(industry_category, {})

        listings = []

        # Generate waste material listings
        waste_types = industry_template.get('waste_types', [])
        for waste_type in waste_types[:3]:  # Limit to top 3 waste types
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

        # Generate required material listings
        material_categories = industry_template.get('material_categories', [])
        for material_category in material_categories[:3]:  # Limit to top 3 material categories
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
        """Generate waste management requirements"""
        industry_category = processed_data.get('industry_category', 'general_manufacturing')
        industry_template = self.industry_knowledge_templates.get(industry_category, {})

        requirements = []

        # Generate waste management requirements based on industry
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
        """Estimate quantity based on company size and industry"""
        employee_count = processed_data.get('employee_count', 100)

        # Base quantities by company size
        if employee_count <= 50:
            base_quantity = 100 if listing_type == 'waste' else 500
        elif employee_count <= 200:
            base_quantity = 500 if listing_type == 'waste' else 1000
        elif employee_count <= 500:
            base_quantity = 1000 if listing_type == 'waste' else 2000
        else:
            base_quantity = 2000 if listing_type == 'waste' else 5000

        # Adjust based on production volume if available
        production_volume = processed_data.get('production_volume', '')
        if production_volume:
            # Extract numbers from production volume string
            numbers = re.findall(r'[\d,]+', production_volume)
            if numbers:
                volume_number = int(numbers[0].replace(',', ''))
                # Scale quantity based on production volume
                if volume_number > 10000:
                    base_quantity *= 2
                elif volume_number > 5000:
                    base_quantity *= 1.5

        return base_quantity

    def _get_appropriate_unit(self, material_type: str) -> str:
        """Get appropriate unit for material type"""
        if any(word in material_type.lower() for word in ['waste', 'scrap', 'solid']):
            return 'tons'
        elif any(word in material_type.lower() for word in ['liquid', 'aqueous', 'solvent']):
            return 'liters'
        elif any(word in material_type.lower() for word in ['gas', 'gaseous']):
            return 'cubic meters'
        else:
            return 'kg'

    def _get_fallback_listings(self, company_profile: Dict[str, Any], answers: Dict[str, str]) -> Dict[str, Any]:
        """Fallback listings if AI generation fails"""
        processed_data = self._process_answers_for_listings(company_profile, answers)
        listings = self._generate_industry_specific_listings(processed_data)
        waste_requirements = self._generate_waste_requirements(processed_data)

        return {
            'material_listings': listings,
            'waste_requirements': waste_requirements,
            'generated_at': datetime.now().isoformat(),
            'ai_model': 'fallback',
            'confidence_score': processed_data.get('confidence_score', 0.0)
        }

    def process_onboarding_answers(self, questions: List[Dict[str, Any]], 
                                 answers: Dict[str, str]) -> Dict[str, Any]:
        """
        Process onboarding answers and generate insights
        """
        try:
            logger.info(f"📊 Processing onboarding answers")

            # Process each answer
            processed_answers = {}
            for question in questions:
                question_id = question.get('id')
                if question_id in answers:
                    processed_answers[question_id] = self._process_answer(answers[question_id], question)

            # Generate insights
            insights = self._generate_insights_from_answers(processed_answers)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(processed_answers)

            return {
                'processed_answers': processed_answers,
                'insights': insights,
                'confidence_score': confidence_score,
                'processed_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error processing onboarding answers: {str(e)}")
            return {
                'error': str(e),
                'processed_answers': {},
                'insights': [],
                'confidence_score': 0.0
            }

    def _process_answer(self, answer: str, question: Dict[str, Any]) -> Any:
        """Process individual answer based on expected type"""
        answer_type = question.get('expected_answer_type', 'text')

        if answer_type == 'numeric':
            # Extract numeric value from answer
            numbers = re.findall(r'[\d,]+', answer)
            if numbers:
                return int(numbers[0].replace(',', ''))
            return 0

        elif answer_type == 'multiselect':
            # Handle multiselect answers
            if isinstance(answer, list):
                return answer
            elif isinstance(answer, str):
                # Split by common delimiters
                return [item.strip() for item in answer.replace(',', ';').split(';') if item.strip()]
            return []

        else:
            return answer

    def _generate_insights_from_answers(self, processed_answers: Dict[str, Any]) -> List[str]:
        """Generate insights from processed answers"""
        insights = []

        # Analyze production processes
        if any('process' in key for key in processed_answers.keys()):
            insights.append("Production processes identified - waste streams can be mapped")

        # Analyze waste streams
        if any('waste' in key for key in processed_answers.keys()):
            insights.append("Waste streams identified - symbiosis opportunities available")

        # Analyze material requirements
        if any('material' in key for key in processed_answers.keys()):
            insights.append("Material requirements identified - supply chain optimization possible")

        return insights

    def _calculate_confidence_score(self, processed_answers: Dict[str, Any]) -> float:
        """Calculate confidence score based on answer quality"""
        if not processed_answers:
            return 0.0

        # Count non-empty answers
        non_empty_answers = sum(1 for answer in processed_answers.values() 
                              if answer and (isinstance(answer, str) and answer.strip() or 
                                           isinstance(answer, (list, int)) and answer))

        return min(1.0, non_empty_answers / len(processed_answers))


# Legacy functions for backward compatibility
def generate_company_onboarding_questions(company_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function for backward compatibility"""
    generator = AIOnboardingQuestionsGenerator()
    return generator.generate_onboarding_questions(company_profile)


def process_company_onboarding_answers(questions: List[Dict[str, Any]], 
                                     answers: Dict[str, str]) -> Dict[str, Any]:
    """Legacy function for backward compatibility"""
    generator = AIOnboardingQuestionsGenerator()
    return generator.process_onboarding_answers(questions, answers)


def main():
    """Main function to handle command line arguments"""
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

            # Generate targeted questions based on knowledge gaps
            result = generator._generate_gap_filling_questions(company_profile, knowledge_assessment, 
                                                             generator._categorize_industry(company_profile.get('industry', '')))
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