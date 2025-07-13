import json
import logging
import os
from typing import Dict, List, Any, Optional, Enum
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionType(Enum):
    """Enhanced question types for advanced AI onboarding"""
    TEXT = "text"
    SELECT = "select"
    MULTISELECT = "multiselect"
    NUMERIC = "numeric"
    MATERIAL_DETAILS = "material_details"  # New type for chemical composition
    WASTE_STREAM = "waste_stream"  # New type for waste characterization
    CHEMICAL_STRUCTURE = "chemical_structure"  # New type for molecular details
    PROCESS_DETAILS = "process_details"  # New type for production processes

class AIOnboardingQuestionsGenerator:
    """
    Enhanced AI-Driven Onboarding Questions Generator
    Uses DeepSeek R1 to generate personalized questions for each company
    with Next-Gen Materials Project integration and chemical structure analysis
    """
    
    def __init__(self):
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.materials_project_api_key = os.getenv('MATERIALS_PROJECT_API_KEY')
        if not self.deepseek_api_key:
            raise ValueError("❌ DEEPSEEK_API_KEY environment variable is required for real AI analysis")
        self.deepseek_base_url = 'https://api.deepseek.com/v1/chat/completions'
        self.deepseek_model = 'deepseek-reasoner'
        self.materials_project_url = 'https://api.materialsproject.org/v2'
        
    def generate_onboarding_questions(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personalized onboarding questions with enhanced material analysis
        """
        try:
            logger.info(f"🎯 Generating enhanced onboarding questions for: {company_profile.get('name', 'Unknown')}")
            
            # Create the enhanced prompt for question generation
            prompt = self._create_enhanced_question_generation_prompt(company_profile)
            
            # Call DeepSeek R1
            response = self._call_deepseek_r1(prompt)
            
            if response:
                questions_data = self._parse_enhanced_questions_response(response)
                logger.info(f"✅ Generated {len(questions_data.get('questions', []))} enhanced questions")
                return questions_data
            else:
                logger.error(f"❌ Failed to generate questions for {company_profile.get('name')}")
                return self._get_enhanced_fallback_questions(company_profile)
                
        except Exception as e:
            logger.error(f"❌ Error generating onboarding questions: {str(e)}")
            return self._get_enhanced_fallback_questions(company_profile)
    
    def _create_enhanced_question_generation_prompt(self, company_profile: Dict[str, Any]) -> str:
        """
        Create an enhanced prompt for generating personalized onboarding questions
        with chemical structure and Next-Gen Materials integration
        """
        company_name = company_profile.get('name', 'Unknown')
        industry = company_profile.get('industry', 'Unknown')
        location = company_profile.get('location', 'Unknown')
        products = company_profile.get('products', 'Unknown')
        employee_count = company_profile.get('employee_count', 0)
        
        prompt = f"""
You are an expert industrial symbiosis consultant with deep knowledge of materials science, chemical engineering, and circular economy principles. You're conducting an advanced onboarding session for a company to enable next-generation industrial symbiosis matching.

COMPANY PROFILE:
- Name: {company_name}
- Industry: {industry}
- Location: {location}
- Products: {products}
- Employee Count: {employee_count}

TASK: Generate 8-12 targeted questions that will provide comprehensive information for advanced industrial symbiosis matching, including chemical structure analysis and Next-Gen Materials Project integration.

QUESTION CATEGORIES:

1. **Production Scale & Capacity** (2 questions)
   - Production volumes, batch sizes, operational frequency
   - Seasonal variations, peak production periods

2. **Key Processes & Materials** (3-4 questions)
   - Main production processes with technical details
   - Primary raw materials with chemical specifications
   - Process temperatures, pressures, catalysts used

3. **Waste Stream Characterization** (2-3 questions) - NEW ENHANCED CATEGORY
   - Detailed waste stream identification
   - Chemical composition of waste streams
   - Physical properties (phase, temperature, pH, etc.)
   - Quantification of waste volumes

4. **Chemical Structure Analysis** (1-2 questions) - NEW CATEGORY
   - Molecular composition of key materials
   - Crystal structure information where applicable
   - Material properties (density, melting point, etc.)

5. **Resource Constraints & Opportunities** (1-2 questions)
   - Most expensive or difficult-to-obtain resources
   - Potential for material substitution
   - Energy requirements and efficiency opportunities

ENHANCED REQUIREMENTS:
- Include questions that can integrate with Next-Gen Materials Project API
- Focus on chemical composition and molecular structure where relevant
- Questions should enable GNN reasoning engine analysis
- Prioritize questions that reveal high-value symbiosis opportunities
- Include questions for waste stream characterization
- Questions should be answerable in 1-3 sentences
- Provide follow-up questions for complex topics

Provide the response as a JSON object with this enhanced structure:
{{
    "questions": [
        {{
            "id": "q1",
            "type": "material_details",
            "category": "chemical_analysis",
            "question": "What are the primary chemical components of your main raw materials?",
            "importance": "high",
            "expected_answer_type": "chemical_structure",
            "follow_up_question": "Do you have access to material safety data sheets (MSDS) for these materials?",
            "materials_project_integration": true,
            "gnn_analysis_enabled": true,
            "reasoning": "Required for advanced material matching and Next-Gen Materials Project integration"
        }}
    ],
    "estimated_completion_time": "5-8 minutes",
    "key_insights_expected": [
        "Chemical composition for advanced material matching",
        "Waste stream characterization for GNN analysis",
        "Process details for symbiosis optimization",
        "Resource constraints for opportunity identification"
    ],
    "next_gen_materials_integration": true,
    "gnn_reasoning_enabled": true
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
                    {"role": "system", "content": "You are an expert industrial symbiosis consultant with deep knowledge of materials science, chemical engineering, and circular economy principles. Your questions are precise, industry-specific, and designed to extract maximum value with minimal effort, enabling advanced material matching and Next-Gen Materials Project integration."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 3000
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
                    message = result['choices'][0]['message']
                    return {
                        'reasoning_content': message.get('reasoning_content'),
                        'content': message.get('content')
                    }
                else:
                    logger.error("Invalid response structure from DeepSeek Reasoner")
                    return None
            else:
                logger.error(f"DeepSeek Reasoner API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error calling DeepSeek Reasoner: {str(e)}")
            return None
    
    def _parse_enhanced_questions_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate the enhanced questions response
        """
        try:
            # Validate structure
            if 'questions' not in response:
                logger.error("Response missing 'questions' key")
                return self._get_enhanced_fallback_questions({})
            
            questions = response.get('questions', [])
            if not questions:
                logger.error("No questions generated")
                return self._get_enhanced_fallback_questions({})
            
            # Validate each question with enhanced fields
            validated_questions = []
            for i, question in enumerate(questions):
                if isinstance(question, dict) and 'question' in question:
                    validated_question = {
                        'id': question.get('id', f'q{i+1}'),
                        'type': question.get('type', 'text'),
                        'category': question.get('category', 'general'),
                        'question': question.get('question', ''),
                        'importance': question.get('importance', 'medium'),
                        'expected_answer_type': question.get('expected_answer_type', 'text'),
                        'follow_up_question': question.get('follow_up_question', ''),
                        'materials_project_integration': question.get('materials_project_integration', False),
                        'gnn_analysis_enabled': question.get('gnn_analysis_enabled', False),
                        'reasoning': question.get('reasoning', '')
                    }
                    validated_questions.append(validated_question)
            
            return {
                'questions': validated_questions,
                'estimated_completion_time': response.get('estimated_completion_time', '5-8 minutes'),
                'key_insights_expected': response.get('key_insights_expected', []),
                'next_gen_materials_integration': response.get('next_gen_materials_integration', True),
                'gnn_reasoning_enabled': response.get('gnn_reasoning_enabled', True),
                'generated_at': datetime.now().isoformat(),
                'ai_model': self.deepseek_model
            }
            
        except Exception as e:
            logger.error(f"Error parsing enhanced questions response: {str(e)}")
            return self._get_enhanced_fallback_questions({})
    
    def _get_enhanced_fallback_questions(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced fallback questions with chemical structure and materials analysis
        """
        industry = company_profile.get('industry', 'manufacturing').lower()
        
        # Industry-specific enhanced questions
        if 'chemical' in industry or 'pharmaceutical' in industry:
            return {
                'questions': [
                    {
                        'id': 'q1',
                        'type': 'material_details',
                        'category': 'chemical_analysis',
                        'question': 'What are the primary chemical compounds used in your production processes?',
                        'importance': 'high',
                        'expected_answer_type': 'chemical_structure',
                        'follow_up_question': 'Do you have molecular formulas or CAS numbers for these compounds?',
                        'materials_project_integration': True,
                        'gnn_analysis_enabled': True,
                        'reasoning': 'Chemical composition is critical for advanced material matching'
                    },
                    {
                        'id': 'q2',
                        'type': 'waste_stream',
                        'category': 'waste_characterization',
                        'question': 'What chemical waste streams do you generate? Include pH, temperature, and chemical composition.',
                        'importance': 'high',
                        'expected_answer_type': 'chemical_structure',
                        'follow_up_question': 'Are these waste streams hazardous or non-hazardous?',
                        'materials_project_integration': True,
                        'gnn_analysis_enabled': True,
                        'reasoning': 'Waste stream characterization enables precise symbiosis matching'
                    },
                    {
                        'id': 'q3',
                        'type': 'process_details',
                        'category': 'production_process',
                        'question': 'What are the key process parameters (temperature, pressure, catalysts) in your main production steps?',
                        'importance': 'medium',
                        'expected_answer_type': 'text',
                        'follow_up_question': 'Do you use any specialized equipment or reactors?',
                        'materials_project_integration': False,
                        'gnn_analysis_enabled': True,
                        'reasoning': 'Process details help identify energy and resource exchange opportunities'
                    }
                ],
                'estimated_completion_time': '5-8 minutes',
                'key_insights_expected': [
                    'Chemical composition for advanced material matching',
                    'Waste stream characterization for GNN analysis',
                    'Process details for symbiosis optimization'
                ],
                'next_gen_materials_integration': True,
                'gnn_reasoning_enabled': True
            }
        else:
            # General manufacturing enhanced questions
            return {
                'questions': [
                    {
                        'id': 'q1',
                        'type': 'material_details',
                        'category': 'material_analysis',
                        'question': 'What are the main materials used in your production? Include material types and properties.',
                        'importance': 'high',
                        'expected_answer_type': 'text',
                        'follow_up_question': 'Do you have specifications for material density, melting points, or other properties?',
                        'materials_project_integration': True,
                        'gnn_analysis_enabled': True,
                        'reasoning': 'Material properties enable advanced matching algorithms'
                    },
                    {
                        'id': 'q2',
                        'type': 'waste_stream',
                        'category': 'waste_characterization',
                        'question': 'What types of waste do you generate? Include quantities, physical state, and composition.',
                        'importance': 'high',
                        'expected_answer_type': 'text',
                        'follow_up_question': 'How do you currently dispose of or handle these waste streams?',
                        'materials_project_integration': True,
                        'gnn_analysis_enabled': True,
                        'reasoning': 'Waste characterization is essential for symbiosis opportunities'
                    },
                    {
                        'id': 'q3',
                        'type': 'process_details',
                        'category': 'production_process',
                        'question': 'Describe your main production processes and any by-products generated.',
                        'importance': 'medium',
                        'expected_answer_type': 'text',
                        'follow_up_question': 'Are there any process inefficiencies or resource constraints?',
                        'materials_project_integration': False,
                        'gnn_analysis_enabled': True,
                        'reasoning': 'Process understanding reveals optimization opportunities'
                    }
                ],
                'estimated_completion_time': '5-8 minutes',
                'key_insights_expected': [
                    'Material properties for advanced matching',
                    'Waste stream characterization',
                    'Process optimization opportunities'
                ],
                'next_gen_materials_integration': True,
                'gnn_reasoning_enabled': True
            }

    def process_enhanced_onboarding_answers(self, questions: List[Dict[str, Any]], 
                                          answers: Dict[str, str]) -> Dict[str, Any]:
        """
        Process enhanced onboarding answers with Next-Gen Materials Project integration
        """
        try:
            logger.info("🔬 Processing enhanced onboarding answers with materials analysis")
            
            # Enrich answers with materials project data
            enriched_answers = self._enrich_answers_with_materials_data(answers)
            
            # Generate enhanced insights
            insights = self._generate_enhanced_insights_from_answers(enriched_answers)
            
            # Calculate confidence score
            confidence_score = self._calculate_enhanced_confidence_score(enriched_answers)
            
            # Generate Next-Gen Materials recommendations
            materials_recommendations = self._generate_materials_recommendations(enriched_answers)
            
            return {
                'enriched_answers': enriched_answers,
                'insights': insights,
                'confidence_score': confidence_score,
                'materials_recommendations': materials_recommendations,
                'next_gen_materials_integration': True,
                'gnn_analysis_ready': True,
                'processed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing enhanced onboarding answers: {str(e)}")
            return {
                'enriched_answers': answers,
                'insights': ['Basic analysis completed'],
                'confidence_score': 0.5,
                'materials_recommendations': [],
                'next_gen_materials_integration': False,
                'gnn_analysis_ready': False,
                'processed_at': datetime.now().isoformat()
            }
    
    def _enrich_answers_with_materials_data(self, answers: Dict[str, str]) -> Dict[str, Any]:
        """
        Enrich answers with Next-Gen Materials Project data
        """
        enriched_answers = answers.copy()
        
        try:
            # Look for chemical compounds in answers
            chemical_compounds = self._extract_chemical_compounds(answers)
            
            for compound in chemical_compounds:
                materials_data = self._query_materials_project(compound)
                if materials_data:
                    enriched_answers[f'materials_data_{compound}'] = materials_data
            
            return enriched_answers
            
        except Exception as e:
            logger.error(f"Error enriching answers with materials data: {str(e)}")
            return enriched_answers
    
    def _extract_chemical_compounds(self, answers: Dict[str, str]) -> List[str]:
        """
        Extract chemical compounds from answers using simple pattern matching
        """
        compounds = []
        
        # Simple pattern matching for common chemical formulas
        import re
        chemical_patterns = [
            r'\b[A-Z][a-z]?\d*\b',  # Simple chemical formulas like H2O, CO2
            r'\b[A-Z]{2,}\d*\b',    # Compounds like NaCl, Fe2O3
        ]
        
        for answer in answers.values():
            for pattern in chemical_patterns:
                matches = re.findall(pattern, answer)
                compounds.extend(matches)
        
        return list(set(compounds))  # Remove duplicates
    
    def _query_materials_project(self, compound: str) -> Optional[Dict[str, Any]]:
        """
        Query Next-Gen Materials Project API for compound data
        """
        if not self.materials_project_api_key:
            return None
            
        try:
            headers = {
                'X-API-KEY': self.materials_project_api_key,
                'Content-Type': 'application/json'
            }
            
            # Query for material properties
            response = requests.get(
                f"{self.materials_project_url}/materials/{compound}/properties",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Materials Project API error for {compound}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying Materials Project for {compound}: {str(e)}")
            return None
    
    def _generate_enhanced_insights_from_answers(self, enriched_answers: Dict[str, Any]) -> List[str]:
        """
        Generate enhanced insights from enriched answers
        """
        insights = []
        
        # Analyze chemical compounds
        chemical_compounds = [k for k in enriched_answers.keys() if k.startswith('materials_data_')]
        if chemical_compounds:
            insights.append(f"Identified {len(chemical_compounds)} chemical compounds for advanced analysis")
            insights.append("Materials Project integration enabled for property analysis")
        
        # Analyze waste streams
        waste_answers = [v for k, v in enriched_answers.items() if 'waste' in k.lower()]
        if waste_answers:
            insights.append("Waste stream characterization completed")
            insights.append("GNN analysis ready for waste matching")
        
        # Analyze processes
        process_answers = [v for k, v in enriched_answers.items() if 'process' in k.lower()]
        if process_answers:
            insights.append("Production process analysis completed")
            insights.append("Energy and resource optimization opportunities identified")
        
        return insights
    
    def _calculate_enhanced_confidence_score(self, enriched_answers: Dict[str, Any]) -> float:
        """
        Calculate enhanced confidence score based on enriched data
        """
        score = 0.0
        total_factors = 0
        
        # Chemical data availability
        chemical_data = [k for k in enriched_answers.keys() if k.startswith('materials_data_')]
        if chemical_data:
            score += 0.3
        total_factors += 1
        
        # Waste stream details
        waste_details = [v for k, v in enriched_answers.items() if 'waste' in k.lower() and len(str(v)) > 50]
        if waste_details:
            score += 0.3
        total_factors += 1
        
        # Process details
        process_details = [v for k, v in enriched_answers.items() if 'process' in k.lower() and len(str(v)) > 50]
        if process_details:
            score += 0.2
        total_factors += 1
        
        # Material properties
        material_properties = [v for k, v in enriched_answers.items() if 'material' in k.lower() and len(str(v)) > 30]
        if material_properties:
            score += 0.2
        total_factors += 1
        
        return score / total_factors if total_factors > 0 else 0.5
    
    def _generate_materials_recommendations(self, enriched_answers: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate Next-Gen Materials recommendations
        """
        recommendations = []
        
        try:
            # Generate recommendations based on enriched data
            chemical_data = [k for k in enriched_answers.keys() if k.startswith('materials_data_')]
            
            if chemical_data:
                recommendations.append({
                    'type': 'material_substitution',
                    'title': 'Advanced Material Substitution',
                    'description': 'Consider alternative materials with similar properties but better sustainability profiles',
                    'confidence': 0.8,
                    'materials_project_integration': True
                })
            
            waste_streams = [v for k, v in enriched_answers.items() if 'waste' in k.lower()]
            if waste_streams:
                recommendations.append({
                    'type': 'waste_valorization',
                    'title': 'Waste Valorization Opportunities',
                    'description': 'Identify high-value applications for waste streams through advanced material analysis',
                    'confidence': 0.7,
                    'gnn_analysis_enabled': True
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating materials recommendations: {str(e)}")
            return []

# Convenience functions
def generate_company_onboarding_questions(company_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for generating onboarding questions"""
    generator = AIOnboardingQuestionsGenerator()
    return generator.generate_onboarding_questions(company_profile)

def process_company_onboarding_answers(questions: List[Dict[str, Any]], 
                                     answers: Dict[str, str]) -> Dict[str, Any]:
    """Convenience function for processing onboarding answers"""
    generator = AIOnboardingQuestionsGenerator()
    return generator.process_enhanced_onboarding_answers(questions, answers)

if __name__ == "__main__":
    # Example usage
    generator = AIOnboardingQuestionsGenerator()
    
    # Example company profile
    company_profile = {
        'name': 'TissuePro Manufacturing',
        'industry': 'Manufacturing',
        'location': 'Dubai, UAE',
        'products': 'Tissue paper products',
        'employee_count': 150
    }
    
    # Generate questions
    questions_data = generator.generate_onboarding_questions(company_profile)
    print(json.dumps(questions_data, indent=2))
    
    # Example answers
    example_answers = {
        'q1': 'We produce 50 tons of tissue paper daily',
        'q2': 'Our main processes are pulping, pressing, and packaging',
        'q3': 'We use wood pulp, water, and chemicals as primary materials',
        'q4': 'We currently send waste to landfill',
        'q5': 'Water and energy are our most expensive resources'
    }
    
    # Process answers
    processed_data = generator.process_enhanced_onboarding_answers(questions_data['questions'], example_answers)
    print(json.dumps(processed_data, indent=2)) 