from .matching_engine import match_buyers_sellers
import os
import json
import requests
from typing import Dict, List, Optional
import time

class SecureAIService:
    """Secure AI service using DeepSeek R1 for advanced reasoning and analysis"""
    
    def __init__(self):
        # Store API key securely - in production, use environment variables
        self.api_key = "sk-7ce79f30332d45d5b3acb8968b052132"
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-r1"  # Updated to use DeepSeek R1
        
    def _make_request(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Make secure API request to DeepSeek R1"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"DeepSeek R1 API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"DeepSeek R1 request failed: {str(e)}")
            return None
    
    def analyze_company_data(self, company_data: Dict) -> Dict:
        """Analyze company data to identify waste opportunities and sustainability insights"""
        prompt = f"""
        Analyze this company data for industrial symbiosis opportunities:
        
        Company Data: {json.dumps(company_data, indent=2)}
        
        Provide a comprehensive analysis in JSON format:
        {{
            "waste_opportunities": ["specific waste streams and their potential"],
            "recycling_potential": ["recycling opportunities with estimated value"],
            "sustainability_goals": ["recommended sustainability objectives"],
            "market_opportunities": ["potential partnerships and markets"],
            "risk_factors": ["identified risks and mitigation strategies"],
            "improvement_areas": ["specific areas for optimization"],
            "similar_companies": ["companies with similar waste streams"],
            "regulatory_considerations": ["relevant environmental regulations"]
        }}
        
        Focus on actionable insights for circular economy implementation.
        """
        
        messages = [
            {"role": "system", "content": "You are DeepSeek R1, an expert in industrial symbiosis and circular economy analysis. Provide detailed, actionable insights based on company data."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages, temperature=0.3)
        if response:
            try:
                return json.loads(response)
            except:
                raise Exception("AI analysis failed - no fallback available. Please ensure AI service is operational.")
        raise Exception("AI service unavailable - no fallback available. Please check AI service connectivity.")

    def generate_intelligent_questions(self, company_data: Dict, context: str = "") -> List[Dict]:
        """Generate intelligent onboarding questions based on company data"""
        prompt = f"""
        Generate intelligent onboarding questions for this company:
        
        Company Data: {json.dumps(company_data, indent=2)}
        Context: {context}
        
        Create 5-7 intelligent questions in JSON format:
        [
            {{
                "question": "specific question text",
                "type": "text|textarea|select|number",
                "key": "unique_identifier",
                "required": true/false,
                "reasoning": "why this question is important",
                "category": "waste|sustainability|operations|logistics"
            }}
        ]
        
        Questions should be:
        1. Industry-specific and relevant
        2. Progressive (build on each other)
        3. Focused on symbiosis opportunities
        4. Actionable for circular economy
        """
        
        messages = [
            {"role": "system", "content": "You are DeepSeek R1, an expert in industrial onboarding. Generate intelligent, progressive questions that uncover symbiosis opportunities."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages, temperature=0.4)
        if response:
            try:
                return json.loads(response)
            except:
                raise Exception("AI question generation failed - no fallback available. Please ensure AI service is operational.")
        raise Exception("AI service unavailable - no fallback available. Please check AI service connectivity.")

    def generate_material_listings(self, company_data: Dict) -> List[Dict]:
        """Generate intelligent material listings based on company data"""
        prompt = f"""
        Generate intelligent material listings for this company:
        
        Company Data: {json.dumps(company_data, indent=2)}
        
        Create material listings in JSON format:
        [
            {{
                "name": "material name",
                "type": "waste|byproduct|resource",
                "description": "detailed description",
                "quantity": "estimated quantity",
                "frequency": "daily|weekly|monthly",
                "specifications": "technical specifications",
                "sustainability_impact": "environmental impact",
                "market_value": "estimated market value",
                "logistics_notes": "transportation and handling requirements"
            }}
        ]
        
        Focus on:
        1. Real waste streams from the industry
        2. Valuable byproducts
        3. Marketable resources
        4. Sustainability benefits
        """
        
        messages = [
            {"role": "system", "content": "You are DeepSeek R1, an expert in industrial materials and waste streams. Generate realistic, valuable material listings."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages, temperature=0.3)
        if response:
            try:
                return json.loads(response)
            except:
                raise Exception("AI material listing generation failed - no fallback available. Please ensure AI service is operational.")
        raise Exception("AI service unavailable - no fallback available. Please check AI service connectivity.")

    def generate_sustainability_insights(self, company_data: Dict) -> Dict:
        """Generate comprehensive sustainability insights"""
        prompt = f"""
        Generate comprehensive sustainability insights for this company:
        
        Company Data: {json.dumps(company_data, indent=2)}
        
        Provide detailed insights in JSON format:
        {{
            "carbon_footprint_analysis": {{
                "estimated_emissions": "detailed emission analysis",
                "reduction_opportunities": ["specific reduction strategies"],
                "carbon_credits_potential": "carbon credit opportunities"
            }},
            "waste_reduction_strategies": {{
                "current_waste_streams": ["identified waste streams"],
                "reduction_targets": ["specific reduction targets"],
                "implementation_steps": ["detailed implementation plan"]
            }},
            "circular_economy_opportunities": {{
                "resource_recovery": ["resource recovery opportunities"],
                "byproduct_utilization": ["byproduct utilization strategies"],
                "closed_loop_systems": ["closed loop system opportunities"]
            }},
            "regulatory_compliance": {{
                "current_requirements": ["current regulatory requirements"],
                "upcoming_changes": ["anticipated regulatory changes"],
                "compliance_strategies": ["compliance strategies"]
            }},
            "financial_benefits": {{
                "cost_savings": ["specific cost savings opportunities"],
                "revenue_opportunities": ["revenue generation opportunities"],
                "investment_requirements": ["required investments"]
            }},
            "transformation_roadmap": [
                {{
                    "phase": "implementation phase",
                    "timeline": "estimated timeline",
                    "actions": ["specific actions to take"],
                    "resources_required": ["resources needed"],
                    "success_metrics": ["how to measure success"]
                }}
            ],
            "roi_analysis": {{
                "total_investment": "estimated total investment",
                "annual_savings": "estimated annual savings",
                "payback_period": "estimated payback period",
                "long_term_benefits": ["long-term benefits"],
                "risk_factors": ["potential risks"],
                "reasoning": "detailed ROI reasoning"
            }}
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are DeepSeek R1, an expert sustainability analyst. Use your advanced reasoning capabilities to provide comprehensive, actionable sustainability insights with quantifiable benefits and realistic implementation plans. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages, temperature=0.2)  # Lower temperature for precise analysis
        if response:
            try:
                return json.loads(response)
            except:
                raise Exception("AI sustainability analysis failed - no fallback available. Please ensure AI service is operational.")
        raise Exception("AI service unavailable - no fallback available. Please check AI service connectivity.")
    
    def analyze_conversational_input(self, user_input: str, conversation_context: Dict) -> Dict:
        """Analyze conversational input to extract intent, entities, and sentiment"""
        prompt = f"""
        Analyze this user input from an onboarding conversation:
        
        User Input: "{user_input}"
        
        Conversation Context: {conversation_context}
        
        Extract the following information in JSON format:
        {{
            "intent": "provide_company_info|ask_question|clarify_information|express_concern|request_help|confirm_information|general",
            "entities": {{
                "company_name": "extracted company name if mentioned",
                "industry": "extracted industry if mentioned",
                "location": "extracted location if mentioned",
                "materials": "extracted materials if mentioned",
                "processes": "extracted processes if mentioned",
                "size": "extracted company size if mentioned",
                "question_topic": "what the user is asking about",
                "unclear_topic": "what the user finds unclear",
                "concern_type": "type of concern expressed",
                "help_topic": "what help the user is requesting",
                "confirmed_info": "what information the user is confirming"
            }},
            "sentiment": "positive|negative|neutral",
            "confidence": 0.0-1.0
        }}
        
        Focus on extracting relevant information for industrial symbiosis and circular economy onboarding.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert in natural language processing for industrial onboarding conversations. Extract intent, entities, and sentiment accurately."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages, temperature=0.2)
        if response:
            try:
                return json.loads(response)
            except:
                raise Exception("AI conversational analysis failed - no fallback available. Please ensure AI service is operational.")
        raise Exception("AI service unavailable - no fallback available. Please check AI service connectivity.")

# Global instance
ai_service = SecureAIService()
