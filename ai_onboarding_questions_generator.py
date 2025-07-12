import json
import logging
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIOnboardingQuestionsGenerator:
    """
    AI-Driven Onboarding Questions Generator
    Uses DeepSeek R1 to generate personalized questions for each company
    following the 80/20 rule for maximum impact with minimal effort
    """
    
    def __init__(self):
        self.deepseek_api_key = 'sk-7ce79f30332d45d5b3acb8968b052132'
        self.deepseek_base_url = 'https://api.deepseek.com/v1/chat/completions'
        self.deepseek_model = 'deepseek-r1'  # Using R1 for advanced reasoning
    
    def generate_onboarding_questions(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personalized onboarding questions for a company
        """
        try:
            logger.info(f"🎯 Generating onboarding questions for: {company_profile.get('name', 'Unknown')}")
            
            # Create the prompt for question generation
            prompt = self._create_question_generation_prompt(company_profile)
            
            # Call DeepSeek R1
            response = self._call_deepseek_r1(prompt)
            
            if response:
                questions_data = self._parse_questions_response(response)
                logger.info(f"✅ Generated {len(questions_data.get('questions', []))} questions")
                return questions_data
            else:
                logger.error(f"❌ Failed to generate questions for {company_profile.get('name')}")
                raise Exception("AI question generation failed - no fallback available. Please ensure AI service is operational.")
                
        except Exception as e:
            logger.error(f"❌ Error generating onboarding questions: {str(e)}")
            raise Exception("AI question generation failed - no fallback available. Please ensure AI service is operational.")
    
    def _create_question_generation_prompt(self, company_profile: Dict[str, Any]) -> str:
        """
        Create a prompt for generating personalized onboarding questions
        """
        company_name = company_profile.get('name', 'Unknown')
        industry = company_profile.get('industry', 'Unknown')
        location = company_profile.get('location', 'Unknown')
        products = company_profile.get('products', 'Unknown')
        employee_count = company_profile.get('employee_count', 0)
        
        prompt = f"""
You are an expert industrial symbiosis consultant conducting an onboarding session for a company. Your goal is to gather the most critical information needed for accurate waste and resource matching using the 80/20 rule - 80% of the results from 20% of the effort.

COMPANY PROFILE:
- Name: {company_name}
- Industry: {industry}
- Location: {location}
- Products: {products}
- Employee Count: {employee_count}

TASK: Generate 5-8 targeted questions that will provide the most valuable information for industrial symbiosis matching. Focus on questions that will reveal:

1. **Production Scale & Capacity** (1-2 questions)
   - Production volumes, batch sizes, operational frequency
   - This directly impacts waste quantities and resource needs

2. **Key Processes & Materials** (2-3 questions)
   - Main production processes, primary raw materials
   - This determines waste types and resource requirements

3. **Current Waste Management** (1-2 questions)
   - How they currently handle waste, disposal methods
   - This reveals immediate symbiosis opportunities

4. **Resource Constraints** (1 question)
   - What resources are most expensive or difficult to obtain
   - This identifies high-value symbiosis opportunities

CRITICAL REQUIREMENTS:
- Questions must be specific to their industry and scale
- Focus on quantifiable information (volumes, frequencies, costs)
- Avoid generic questions that apply to all companies
- Questions should be answerable in 1-2 sentences
- Prioritize questions that will have the biggest impact on matching accuracy

Provide the response as a JSON object with this structure:
{{
    "questions": [
        {{
            "id": "q1",
            "category": "production_scale",
            "question": "What is your daily production volume in [relevant units]?",
            "importance": "high",
            "expected_answer_type": "numeric",
            "follow_up_question": "Is this consistent year-round or seasonal?"
        }}
    ],
    "estimated_completion_time": "3-5 minutes",
    "key_insights_expected": [
        "Production capacity and waste generation potential",
        "Primary resource requirements",
        "Current waste management practices"
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
                raise Exception("AI question generation failed - no fallback available. Please ensure AI service is operational.")
            
            questions = response.get('questions', [])
            if not questions:
                logger.error("No questions generated")
                raise Exception("AI question generation failed - no fallback available. Please ensure AI service is operational.")
            
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
                        'follow_up_question': question.get('follow_up_question', '')
                    }
                    validated_questions.append(validated_question)
            
            return {
                'questions': validated_questions,
                'estimated_completion_time': response.get('estimated_completion_time', '3-5 minutes'),
                'key_insights_expected': response.get('key_insights_expected', []),
                'generated_at': datetime.now().isoformat(),
                'ai_model': self.deepseek_model
            }
            
        except Exception as e:
            logger.error(f"Error parsing questions response: {str(e)}")
            raise Exception("AI question generation failed - no fallback available. Please ensure AI service is operational.")
    
    def process_onboarding_answers(self, questions: List[Dict[str, Any]], 
                                 answers: Dict[str, str]) -> Dict[str, Any]:
        """
        Process onboarding answers and enrich company profile
        """
        try:
            enriched_profile = {}
            
            for question in questions:
                question_id = question.get('id')
                if question_id in answers:
                    answer = answers[question_id]
                    category = question.get('category', 'general')
                    
                    # Process answer based on category and expected type
                    processed_answer = self._process_answer(answer, question)
                    
                    # Add to enriched profile
                    if category not in enriched_profile:
                        enriched_profile[category] = {}
                    
                    enriched_profile[category][question_id] = {
                        'question': question.get('question'),
                        'answer': answer,
                        'processed_value': processed_answer,
                        'importance': question.get('importance', 'medium')
                    }
            
            # Generate insights from the answers
            insights = self._generate_insights_from_answers(enriched_profile)
            
            return {
                'enriched_profile': enriched_profile,
                'insights': insights,
                'processed_at': datetime.now().isoformat(),
                'confidence_score': self._calculate_confidence_score(enriched_profile)
            }
            
        except Exception as e:
            logger.error(f"Error processing onboarding answers: {str(e)}")
            return {'error': str(e)}
    
    def _process_answer(self, answer: str, question: Dict[str, Any]) -> Any:
        """
        Process answer based on expected type
        """
        answer_type = question.get('expected_answer_type', 'text')
        
        if answer_type == 'numeric':
            # Extract numeric value from answer
            import re
            numbers = re.findall(r'[\d,]+', answer)
            if numbers:
                return int(numbers[0].replace(',', ''))
            return None
        elif answer_type == 'boolean':
            # Convert to boolean
            return answer.lower() in ['yes', 'true', '1', 'y']
        else:
            # Return as text
            return answer.strip()
    
    def _generate_insights_from_answers(self, enriched_profile: Dict[str, Any]) -> List[str]:
        """
        Generate insights from the onboarding answers
        """
        insights = []
        
        # Analyze production scale
        if 'production_scale' in enriched_profile:
            insights.append("Production scale analysis completed")
        
        # Analyze key processes
        if 'key_processes' in enriched_profile:
            insights.append("Key production processes identified")
        
        # Analyze materials
        if 'materials' in enriched_profile:
            insights.append("Primary materials and quantities documented")
        
        # Analyze waste management
        if 'waste_management' in enriched_profile:
            insights.append("Current waste management practices assessed")
        
        return insights
    
    def _calculate_confidence_score(self, enriched_profile: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on completeness and quality of answers
        """
        total_questions = 0
        answered_questions = 0
        high_importance_answered = 0
        high_importance_total = 0
        
        for category, questions in enriched_profile.items():
            for question_id, data in questions.items():
                total_questions += 1
                if data.get('answer'):
                    answered_questions += 1
                    if data.get('importance') == 'high':
                        high_importance_answered += 1
                if data.get('importance') == 'high':
                    high_importance_total += 1
        
        if total_questions == 0:
            return 0.0
        
        # Calculate score based on completion and high-importance questions
        completion_score = answered_questions / total_questions
        importance_score = high_importance_answered / high_importance_total if high_importance_total > 0 else 1.0
        
        # Weighted average (importance questions count more)
        confidence_score = (completion_score * 0.4) + (importance_score * 0.6)
        
        return round(confidence_score, 2)

# Convenience functions
def generate_company_onboarding_questions(company_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for generating onboarding questions"""
    generator = AIOnboardingQuestionsGenerator()
    return generator.generate_onboarding_questions(company_profile)

def process_company_onboarding_answers(questions: List[Dict[str, Any]], 
                                     answers: Dict[str, str]) -> Dict[str, Any]:
    """Convenience function for processing onboarding answers"""
    generator = AIOnboardingQuestionsGenerator()
    return generator.process_onboarding_answers(questions, answers)

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
    processed_data = generator.process_onboarding_answers(questions_data['questions'], example_answers)
    print(json.dumps(processed_data, indent=2)) 