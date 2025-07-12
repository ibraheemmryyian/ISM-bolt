import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIService:
    """
    Advanced AI Service using DeepSeek R1 for industrial symbiosis analysis
    No fallbacks - only real AI analysis
    """
    
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-r1"
        
        if not self.api_key:
            raise ValueError("❌ DEEPSEEK_API_KEY environment variable is required for real AI analysis")
        
        logger.info("✅ DeepSeek R1 API initialized for real AI analysis")
    
    def _call_deepseek_api_directly(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> Optional[Dict[str, Any]]:
        """
        Make direct call to DeepSeek R1 API
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                try:
                    # Try to parse as JSON
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If not JSON, return as text
                    return {"analysis": content, "raw_response": content}
            else:
                logger.error(f"❌ DeepSeek R1 API error: {response.status_code} - {response.text}")
                raise Exception(f"DeepSeek R1 API failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ DeepSeek R1 API call failed: {str(e)}")
            raise Exception(f"Real AI analysis failed: {str(e)}")
    
    def analyze_company_for_symbiosis(self, company_data: Dict) -> Dict[str, Any]:
        """
        Analyze company for industrial symbiosis opportunities using DeepSeek R1
        """
        try:
            prompt = f"""
            You are an expert industrial symbiosis analyst. Analyze this company for industrial symbiosis opportunities:
            
            Company Data: {json.dumps(company_data, indent=2)}
            
            Provide a comprehensive analysis in JSON format with the following structure:
            {{
                "waste_materials": [
                    {{
                        "name": "material name",
                        "description": "detailed description",
                        "quantity": "estimated quantity",
                        "frequency": "daily/weekly/monthly",
                        "potential_value": "estimated value",
                        "quality_grade": "high/medium/low",
                        "potential_uses": ["use1", "use2"],
                        "symbiosis_opportunities": ["opportunity1", "opportunity2"]
                    }}
                ],
                "requirements": [
                    {{
                        "name": "material name",
                        "description": "detailed description", 
                        "quantity": "estimated quantity",
                        "frequency": "daily/weekly/monthly",
                        "current_cost": "estimated cost",
                        "priority": "high/medium/low",
                        "potential_sources": ["source1", "source2"],
                        "symbiosis_opportunities": ["opportunity1", "opportunity2"]
                    }}
                ],
                "strategic_analysis": {{
                    "symbiosis_potential": "high/medium/low",
                    "key_opportunities": ["opportunity1", "opportunity2"],
                    "estimated_savings": "estimated annual savings",
                    "implementation_timeline": "estimated timeline",
                    "risk_assessment": "risk level and considerations"
                }}
            }}
            
            Focus on real, actionable industrial symbiosis opportunities. Be specific about materials, quantities, and potential matches.
            """
            
            result = self._call_deepseek_api_directly(prompt)
            
            if not result:
                raise Exception("DeepSeek R1 returned empty response")
            
            logger.info("✅ Company symbiosis analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Company symbiosis analysis failed: {str(e)}")
            raise Exception(f"Real AI analysis failed: {str(e)}")
    
    def generate_onboarding_questions(self, company_data: Dict) -> List[Dict[str, Any]]:
        """
        Generate intelligent onboarding questions using DeepSeek R1
        """
        try:
            prompt = f"""
            You are an expert industrial symbiosis consultant. Generate intelligent onboarding questions for this company:
            
            Company Data: {json.dumps(company_data, indent=2)}
            
            Generate 5-8 targeted questions in JSON format:
            [
                {{
                    "id": "unique_id",
                    "category": "production_scale|key_processes|materials|waste_management|energy|location|goals",
                    "question": "specific question text",
                    "importance": "high|medium|low",
                    "expected_answer_type": "numeric|text|multiple_choice|boolean",
                    "follow_up_question": "follow-up question if needed"
                }}
            ]
            
            Questions should be specific to their industry and designed to identify symbiosis opportunities.
            """
            
            result = self._call_deepseek_api_directly(prompt)
            
            if not result or not isinstance(result, list):
                raise Exception("DeepSeek R1 returned invalid question format")
            
            logger.info(f"✅ Generated {len(result)} intelligent onboarding questions")
            return result
            
        except Exception as e:
            logger.error(f"❌ Question generation failed: {str(e)}")
            raise Exception(f"Real AI question generation failed: {str(e)}")
    
    def generate_material_listings(self, company_data: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive material listings using DeepSeek R1
        """
        try:
            prompt = f"""
            You are an expert industrial materials analyst. Generate comprehensive material listings for this company:
            
            Company Data: {json.dumps(company_data, indent=2)}
            
            Generate detailed material listings in JSON format:
            {{
                "waste_materials": [
                    {{
                        "name": "material name",
                        "description": "detailed description",
                        "category": "chemical|metal|plastic|organic|electronic|construction|other",
                        "quantity": "estimated quantity with units",
                        "frequency": "daily|weekly|monthly|quarterly|annually",
                        "notes": "additional details",
                        "potential_value": "estimated value",
                        "quality_grade": "high|medium|low",
                        "potential_uses": ["use1", "use2", "use3"],
                        "symbiosis_opportunities": ["opportunity1", "opportunity2"]
                    }}
                ],
                "requirements": [
                    {{
                        "name": "material name",
                        "description": "detailed description",
                        "category": "chemical|metal|plastic|organic|electronic|construction|other",
                        "quantity": "estimated quantity with units",
                        "frequency": "daily|weekly|monthly|quarterly|annually",
                        "notes": "additional details",
                        "current_cost": "estimated current cost",
                        "priority": "high|medium|low",
                        "potential_sources": ["source1", "source2"],
                        "symbiosis_opportunities": ["opportunity1", "opportunity2"]
                    }}
                ]
            }}
            
            Be specific about materials, quantities, and potential industrial symbiosis opportunities.
            """
            
            result = self._call_deepseek_api_directly(prompt)
            
            if not result:
                raise Exception("DeepSeek R1 returned empty material listings")
            
            logger.info("✅ Material listings generation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Material listings generation failed: {str(e)}")
            raise Exception(f"Real AI material generation failed: {str(e)}")
    
    def generate_sustainability_insights(self, company_data: Dict) -> Dict[str, Any]:
        """
        Generate sustainability insights using DeepSeek R1
        """
        try:
            prompt = f"""
            You are an expert sustainability analyst. Generate comprehensive sustainability insights for this company:
            
            Company Data: {json.dumps(company_data, indent=2)}
            
            Provide sustainability analysis in JSON format:
            {{
                "carbon_footprint_analysis": {{
                    "current_emissions": "estimated current emissions",
                    "emission_sources": ["source1", "source2"],
                    "reduction_potential": "estimated reduction potential",
                    "recommendations": ["rec1", "rec2"]
                }},
                "waste_management_analysis": {{
                    "current_waste": "current waste generation",
                    "waste_types": ["type1", "type2"],
                    "reduction_potential": "estimated reduction potential",
                    "recommendations": ["rec1", "rec2"]
                }},
                "circular_economy_opportunities": {{
                    "reuse_opportunities": ["opp1", "opp2"],
                    "recycling_opportunities": ["opp1", "opp2"],
                    "symbiosis_opportunities": ["opp1", "opp2"],
                    "implementation_priority": "high|medium|low"
                }},
                "sustainability_score": {{
                    "overall_score": "0-100",
                    "environmental_score": "0-100",
                    "economic_score": "0-100",
                    "social_score": "0-100",
                    "improvement_areas": ["area1", "area2"]
                }}
            }}
            
            Focus on actionable sustainability improvements and industrial symbiosis opportunities.
            """
            
            result = self._call_deepseek_api_directly(prompt)
            
            if not result:
                raise Exception("DeepSeek R1 returned empty sustainability insights")
            
            logger.info("✅ Sustainability insights generation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Sustainability insights generation failed: {str(e)}")
            raise Exception(f"Real AI sustainability analysis failed: {str(e)}")
    
    def analyze_conversational_input(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """
        Analyze conversational input using DeepSeek R1
        """
        try:
            context_str = f"Context: {json.dumps(context, indent=2)}" if context else ""
            
            prompt = f"""
            You are an expert industrial symbiosis conversational AI. Analyze this user input:
            
            User Input: {user_input}
            {context_str}
            
            Provide analysis in JSON format:
            {{
                "intent": "question|request|feedback|complaint|other",
                "confidence": "0.0-1.0",
                "entities": ["entity1", "entity2"],
                "sentiment": "positive|neutral|negative",
                "response_type": "informational|actionable|clarification|redirect",
                "key_topics": ["topic1", "topic2"],
                "suggested_response": "suggested response text",
                "follow_up_questions": ["question1", "question2"],
                "action_items": ["action1", "action2"]
            }}
            
            Focus on industrial symbiosis, materials, sustainability, and business opportunities.
            """
            
            result = self._call_deepseek_api_directly(prompt)
            
            if not result:
                raise Exception("DeepSeek R1 returned empty conversational analysis")
            
            logger.info("✅ Conversational analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Conversational analysis failed: {str(e)}")
            raise Exception(f"Real AI conversational analysis failed: {str(e)}")
