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
        """Intelligent analysis of company data using DeepSeek R1's advanced reasoning"""
        prompt = f"""
        You are an expert industrial sustainability analyst with deep knowledge of circular economy and industrial symbiosis. Analyze this company data using advanced reasoning:

        COMPANY DATA:
        - Name: {company_data.get('name', 'Unknown')}
        - Industry: {company_data.get('industry', 'Unknown')}
        - Location: {company_data.get('location', 'Unknown')}
        - Processes: {company_data.get('processes', 'Unknown')}
        - Materials: {company_data.get('materials', [])}
        - Size: {company_data.get('employee_count', 'Unknown')} employees

        TASK: Provide a comprehensive analysis using DeepSeek R1's reasoning capabilities. Consider:
        1. Industry-specific waste streams and byproducts
        2. Resource efficiency opportunities based on processes
        3. Sustainability goals aligned with their operations
        4. Market opportunities in their region and industry
        5. Risk factors specific to their business model
        6. Areas for improvement in their current operations
        7. Types of companies they should connect with for symbiosis
        8. Regulatory considerations for their industry and location

        REQUIREMENTS:
        - Use logical reasoning to connect their processes to potential opportunities
        - Consider geographical and industry-specific factors
        - Provide actionable, practical insights
        - Focus on industrial symbiosis and circular economy principles

        Return ONLY valid JSON with this exact structure:
        {{
            "waste_opportunities": ["specific waste streams with reasoning"],
            "recycling_potential": ["recycling opportunities with business case"],
            "sustainability_goals": ["realistic sustainability objectives"],
            "market_opportunities": ["specific market opportunities with reasoning"],
            "risk_factors": ["identified risks with impact assessment"],
            "improvement_areas": ["specific areas for improvement with rationale"],
            "similar_companies": ["types of companies to connect with and why"],
            "regulatory_considerations": ["regulatory requirements with compliance notes"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are DeepSeek R1, an expert industrial sustainability analyst. Use your advanced reasoning capabilities to provide precise, actionable insights for industrial symbiosis. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages, temperature=0.2)  # Lower temperature for more precise reasoning
        if response:
            try:
                return json.loads(response)
            except:
                return self._fallback_analysis(company_data)
        return self._fallback_analysis(company_data)
    
    def generate_intelligent_questions(self, company_data: Dict, context: str = "") -> List[Dict]:
        """Generate intelligent, contextual questions using DeepSeek R1's reasoning"""
        prompt = f"""
        You are an expert sustainability consultant with deep knowledge of industrial processes and circular economy. Generate intelligent, targeted questions for this company using advanced reasoning:

        COMPANY PROFILE:
        - Name: {company_data.get('name', 'Unknown')}
        - Industry: {company_data.get('industry', 'Unknown')}
        - Location: {company_data.get('location', 'Unknown')}
        - Current Processes: {company_data.get('processes', 'Unknown')}
        - Materials: {company_data.get('materials', [])}
        - Company Size: {company_data.get('employee_count', 'Unknown')} employees
        
        CONTEXT: {context}

        TASK: Generate 5-8 intelligent questions that will reveal their:
        1. Waste streams and byproducts (industry-specific)
        2. Resource efficiency opportunities (process-based)
        3. Sustainability goals and challenges (business-focused)
        4. Potential for industrial symbiosis (geographical and industry factors)
        5. Market and regulatory considerations (location and industry-specific)

        REASONING REQUIREMENTS:
        - Use logical reasoning to connect their industry/processes to relevant questions
        - Consider geographical factors and local regulations
        - Focus on questions that reveal actionable opportunities
        - Ensure questions are specific to their business model

        Return ONLY valid JSON array with this exact structure for each question:
        {{
            "question": "specific, targeted question text",
            "type": "text|textarea|number|checkbox|select",
            "key": "unique_identifier",
            "required": true|false,
            "reasoning": "detailed explanation of why this question is important and what insights it will reveal",
            "category": "waste|efficiency|sustainability|symbiosis|regulatory",
            "expected_insight": "what valuable information this question will uncover"
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are DeepSeek R1, an expert sustainability consultant. Use your advanced reasoning to generate intelligent, targeted questions that reveal valuable insights for industrial symbiosis. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages, temperature=0.3)  # Lower temperature for more focused questions
        if response:
            try:
                return json.loads(response)
            except:
                return self._fallback_questions(company_data)
        return self._fallback_questions(company_data)
    
    def generate_material_listings(self, company_data: Dict) -> List[Dict]:
        """Generate intelligent material listings using DeepSeek R1's reasoning"""
        prompt = f"""
        You are an expert in industrial materials and circular economy with deep knowledge of waste-to-resource opportunities. Generate realistic material listings for this company using advanced reasoning:

        COMPANY DATA:
        - Name: {company_data.get('name', 'Unknown')}
        - Industry: {company_data.get('industry', 'Unknown')}
        - Location: {company_data.get('location', 'Unknown')}
        - Processes: {company_data.get('processes', 'Unknown')}
        - Materials: {company_data.get('materials', [])}
        - Size: {company_data.get('employee_count', 'Unknown')} employees

        TASK: Generate realistic material listings using logical reasoning:

        FOR WASTE MATERIALS (things they want to get rid of):
        - Analyze their industry and processes to identify realistic waste streams
        - Consider production volumes and typical waste generation rates
        - Focus on materials that other companies might actually want
        - Include realistic quantities based on their company size

        FOR REQUIREMENT MATERIALS (things they need):
        - Analyze their processes to identify realistic material needs
        - Consider what other companies might have as waste
        - Include realistic specifications and quantities
        - Focus on materials that support their operations

        REASONING REQUIREMENTS:
        - Use logical reasoning to connect their processes to material flows
        - Consider industry standards and typical material quantities
        - Ensure materials are realistic for their size and location
        - Focus on materials with actual symbiosis potential

        Return ONLY valid JSON array with this exact structure for each listing:
        {{
            "name": "specific material name",
            "type": "waste|requirement",
            "description": "detailed description with reasoning",
            "quantity": "realistic quantity with units (based on company size)",
            "frequency": "how often available/needed (realistic for their operations)",
            "specifications": "quality, purity, or other specs (industry-appropriate)",
            "sustainability_impact": "environmental benefit with quantification",
            "market_value": "estimated value with reasoning",
            "logistics_notes": "transportation considerations specific to their location",
            "reasoning": "why this material is realistic for their operations"
        }}

        Generate 8-12 realistic listings total (mix of waste and requirements).
        """
        
        messages = [
            {"role": "system", "content": "You are DeepSeek R1, an expert in industrial materials and circular economy. Use your advanced reasoning to generate realistic, actionable material listings that create genuine symbiosis opportunities. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._make_request(messages, temperature=0.4)  # Balanced temperature for creativity and accuracy
        if response:
            try:
                return json.loads(response)
            except:
                return self._fallback_material_listings(company_data)
        return self._fallback_material_listings(company_data)
    
    def generate_sustainability_insights(self, company_data: Dict) -> Dict:
        """Generate comprehensive sustainability insights using DeepSeek R1's reasoning"""
        prompt = f"""
        You are an expert sustainability analyst with deep knowledge of industrial processes, circular economy, and environmental impact assessment. Provide comprehensive sustainability insights for this company using advanced reasoning:

        COMPANY DATA:
        - Name: {company_data.get('name', 'Unknown')}
        - Industry: {company_data.get('industry', 'Unknown')}
        - Location: {company_data.get('location', 'Unknown')}
        - Processes: {company_data.get('processes', 'Unknown')}
        - Materials: {company_data.get('materials', [])}
        - Size: {company_data.get('employee_count', 'Unknown')} employees

        TASK: Provide comprehensive sustainability analysis using logical reasoning:

        ANALYSIS AREAS:
        1. Current Environmental Impact: Analyze their processes and materials to estimate current environmental footprint
        2. Carbon Reduction Opportunities: Identify specific ways to reduce carbon emissions
        3. Waste Minimization Strategies: Propose strategies to minimize waste generation
        4. Resource Efficiency Improvements: Suggest ways to improve resource efficiency
        5. Circular Economy Integration: Identify opportunities to integrate circular economy principles
        6. Sustainability Goals: Propose realistic, measurable sustainability objectives
        7. Implementation Roadmap: Provide a step-by-step implementation plan
        8. ROI Analysis: Estimate the return on investment for sustainability initiatives

        REASONING REQUIREMENTS:
        - Use logical reasoning to connect their operations to environmental impacts
        - Consider industry benchmarks and best practices
        - Provide quantifiable estimates where possible
        - Focus on practical, implementable solutions
        - Consider their size and resources in recommendations

        Return ONLY valid JSON with this exact structure:
        {{
            "current_environmental_impact": {{
                "carbon_footprint": "estimated CO2 emissions with reasoning",
                "waste_generation": "estimated waste quantities with types",
                "resource_consumption": "key resources consumed with quantities",
                "environmental_risks": ["specific environmental risks with impact assessment"]
            }},
            "carbon_reduction_opportunities": [
                {{
                    "opportunity": "specific opportunity description",
                    "potential_reduction": "estimated CO2 reduction",
                    "implementation_cost": "estimated cost",
                    "payback_period": "estimated payback time",
                    "reasoning": "why this opportunity is suitable for them"
                }}
            ],
            "waste_minimization_strategies": [
                {{
                    "strategy": "specific strategy description",
                    "waste_reduction": "estimated waste reduction",
                    "cost_savings": "estimated cost savings",
                    "implementation_steps": ["step-by-step implementation"],
                    "reasoning": "why this strategy is appropriate"
                }}
            ],
            "resource_efficiency_improvements": [
                {{
                    "improvement": "specific improvement description",
                    "efficiency_gain": "estimated efficiency improvement",
                    "cost_benefit": "cost-benefit analysis",
                    "implementation_timeline": "estimated timeline",
                    "reasoning": "why this improvement is feasible"
                }}
            ],
            "circular_economy_integration": [
                {{
                    "integration": "specific circular economy integration",
                    "environmental_benefit": "quantified environmental benefit",
                    "economic_benefit": "quantified economic benefit",
                    "implementation_requirements": ["requirements for implementation"],
                    "reasoning": "why this integration makes sense"
                }}
            ],
            "sustainability_goals": [
                {{
                    "goal": "specific, measurable sustainability goal",
                    "target_year": "target year for achievement",
                    "baseline": "current baseline measurement",
                    "target": "specific target value",
                    "reasoning": "why this goal is appropriate and achievable"
                }}
            ],
            "implementation_roadmap": [
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
                return self._fallback_sustainability_insights(company_data)
        return self._fallback_sustainability_insights(company_data)
    
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
                return self._fallback_conversational_analysis(user_input)
        return self._fallback_conversational_analysis(user_input)
    
    def _fallback_analysis(self, company_data: Dict) -> Dict:
        """Fallback analysis when AI service is unavailable"""
        return {
            "waste_opportunities": ["General waste streams based on industry"],
            "recycling_potential": ["Standard recycling opportunities"],
            "sustainability_goals": ["Reduce waste, improve efficiency"],
            "market_opportunities": ["Circular economy partnerships"],
            "risk_factors": ["Regulatory compliance, market changes"],
            "improvement_areas": ["Process optimization, waste reduction"],
            "similar_companies": ["Companies in same industry"],
            "regulatory_considerations": ["Local environmental regulations"]
        }
    
    def _fallback_questions(self, company_data: Dict) -> List[Dict]:
        """Fallback questions when AI service is unavailable"""
        return [
            {
                "question": "What are your main waste streams?",
                "type": "textarea",
                "key": "waste_streams",
                "required": False,
                "reasoning": "Understanding waste streams helps identify recycling opportunities",
                "category": "waste"
            },
            {
                "question": "What sustainability goals do you have?",
                "type": "textarea",
                "key": "sustainability_goals",
                "required": False,
                "reasoning": "Sustainability goals help align with circular economy opportunities",
                "category": "sustainability"
            }
        ]
    
    def _fallback_material_listings(self, company_data: Dict) -> List[Dict]:
        """Fallback material listings when AI service is unavailable"""
        return [
            {
                "name": "General Waste Materials",
                "type": "waste",
                "description": "Various waste materials from production",
                "quantity": "Variable",
                "frequency": "Regular",
                "specifications": "Mixed",
                "sustainability_impact": "Reduces landfill waste",
                "market_value": "Variable",
                "logistics_notes": "Local pickup preferred"
            }
        ]
    
    def _fallback_sustainability_insights(self, company_data: Dict) -> Dict:
        """Fallback sustainability insights when AI service is unavailable"""
        return {
            "carbon_footprint_analysis": {
                "estimated_emissions": "Variable based on operations",
                "reduction_opportunities": ["Energy efficiency", "Waste reduction"],
                "carbon_credits_potential": "Moderate"
            },
            "waste_reduction_strategies": {
                "current_waste_streams": ["Production waste", "Packaging waste"],
                "reduction_targets": ["10-20% waste reduction"],
                "implementation_steps": ["Audit current waste", "Identify opportunities"]
            },
            "circular_economy_opportunities": {
                "resource_recovery": ["Material recycling", "Energy recovery"],
                "byproduct_utilization": ["Waste-to-resource conversion"],
                "closed_loop_systems": ["Internal recycling loops"]
            },
            "regulatory_compliance": {
                "current_requirements": ["Environmental regulations"],
                "upcoming_changes": ["Stricter waste management"],
                "compliance_strategies": ["Regular audits", "Proactive compliance"]
            },
            "financial_benefits": {
                "cost_savings": ["Reduced waste disposal costs"],
                "revenue_opportunities": ["Waste-to-resource sales"],
                "investment_requirements": ["Process optimization investments"]
            }
        }

    def _fallback_conversational_analysis(self, user_input: str) -> Dict:
        """Fallback conversational analysis when AI service is unavailable"""
        user_input_lower = user_input.lower()
        
        # Simple keyword-based intent detection
        intent = "general"
        if any(word in user_input_lower for word in ["company", "name", "industry", "location", "materials", "processes"]):
            intent = "provide_company_info"
        elif any(word in user_input_lower for word in ["what", "how", "why", "when", "where", "?"]):
            intent = "ask_question"
        elif any(word in user_input_lower for word in ["unclear", "confused", "don't understand", "not sure"]):
            intent = "clarify_information"
        elif any(word in user_input_lower for word in ["concern", "worried", "problem", "issue"]):
            intent = "express_concern"
        elif any(word in user_input_lower for word in ["help", "assist", "support"]):
            intent = "request_help"
        
        # Simple entity extraction
        entities = {}
        if "company" in user_input_lower and "name" in user_input_lower:
            entities["company_name"] = "extracted"
        if "industry" in user_input_lower:
            entities["industry"] = "extracted"
        if "location" in user_input_lower or "city" in user_input_lower or "country" in user_input_lower:
            entities["location"] = "extracted"
        
        # Simple sentiment analysis
        sentiment = "neutral"
        if any(word in user_input_lower for word in ["good", "great", "excellent", "happy", "satisfied"]):
            sentiment = "positive"
        elif any(word in user_input_lower for word in ["bad", "terrible", "worried", "concerned", "frustrated"]):
            sentiment = "negative"
        
        return {
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "confidence": 0.6
        }

# Global instance
ai_service = SecureAIService()
