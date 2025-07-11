import os
import json
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import sys
import traceback

# DeepSeek API Configuration
DEEPSEEK_API_KEY = 'sk-7ce79f30332d45d5b3acb8968b052132'
DEEPSEEK_BASE_URL = 'https://api.deepseek.com/v1/chat/completions'
DEEPSEEK_MODEL = 'deepseek-coder'  # Can be changed to deepseek-r1 for advanced reasoning

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAIPromptsService:
    """
    Advanced AI Service implementing the four strategic prompts for industrial symbiosis:
    1. Strategic Material & Synergy Analysis
    2. Precision AI Matchmaking
    3. Advanced Conversational AI & Intent Analysis
    4. Strategic Company Transformation Analysis
    """
    
    def __init__(self):
        self.deepseek_api_key = DEEPSEEK_API_KEY
        self.deepseek_base_url = DEEPSEEK_BASE_URL
        self.deepseek_model = DEEPSEEK_MODEL
        
    def strategic_material_analysis(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prompt 1: Strategic Material & Synergy Analysis
        Use this for: Analyzing a company to generate a complete, strategic report on its potential 
        waste outputs, resource inputs, and detailed circular economy opportunities.
        """
        try:
            logger.info(f"Starting Strategic Material Analysis for company: {company_profile.get('name', 'Unknown')}")
            
            system_prompt = """You are a world-renowned authority on industrial symbiosis and circular economy transformation, with 20 years of experience consulting for Fortune 500 companies. Your task is to conduct a forensic analysis of a given company's operations to map its complete resource lifecycle. Based on the company profile provided, you will generate a highly detailed, realistic, and actionable synergy report. Your analysis must be deeply rooted in the company's specific industry, scale of operations, and geographic context.

TASK: Generate a comprehensive industrial synergy report. The number of inputs, outputs, and initiatives should be realistically proportional to the size and complexity of the company described. For a standard manufacturing facility, aim for approximately 5-10 outputs and 5-10 inputs.

CRITICAL INSTRUCTIONS:
1. **Be Realistic:** Ground every prediction in the realities of the specified industry. Avoid generic suggestions.
2. **Quantify Everything:** Use realistic estimates for all quantities, values, and impacts. Clearly state your units.
3. **Think Strategically:** The report must flow from granular data to high-level strategic recommendations.
4. **MANDATORY SPECIFIC MATERIAL NAMES:** You MUST provide specific, industry-relevant material names. NEVER use generic terms like "Unknown Material", "Material", "Waste", "Byproduct", or "Output". Use specific names like:
   - For healthcare: "Medical sharps", "Contaminated linens", "Pharmaceutical waste", "Sterilization byproducts", "Medical waste", "Used syringes", "Expired medications"
   - For hospitality: "Food waste", "Used cooking oil", "Wastewater", "Soiled linens", "Packaging waste", "Organic waste", "Plastic waste"
   - For manufacturing: "Metal scrap", "Plastic waste", "Chemical byproducts", "Process water", "Packaging materials", "Wood waste", "Electronic waste"
   - For textiles: "Fabric scraps", "Dye waste", "Fiber waste", "Chemical sludge", "Wastewater", "Textile waste", "Yarn waste"
   - For food processing: "Food waste", "Organic waste", "Packaging waste", "Wastewater", "Used cooking oil", "Animal byproducts"
5. **MANDATORY SPECIFIC REQUIREMENT NAMES:** For inputs/requirements, use specific names like "Raw cotton", "Chemical dyes", "Process water", "Energy", "Packaging materials", "Fresh ingredients", "Medical supplies"
6. **VALIDATE QUANTITIES:** Ensure all quantities are realistic for the company size and industry. Use reasonable units like kg, liters, units, pieces, tons.
7. **QUALITY CONTROL:** Every material and requirement must have a specific, descriptive name that clearly identifies what it is.

Provide the final response as a single, valid JSON object following this exact structure:"""

            response_structure = {
                "executive_summary": {
                    "key_findings": "A concise summary of the most significant waste streams and resource needs.",
                    "primary_opportunities": "A high-level overview of the top 2-3 symbiosis and sustainability opportunities.",
                    "estimated_total_impact": "An aggregate estimate of potential annual cost savings and revenue."
                },
                "predicted_outputs": [
                    {
                        "name": "Material Name",
                        "category": "e.g., metal, plastic, organic, chemical, energy",
                        "description": "Detailed description of the material, its source within the process, and its physical/chemical state.",
                        "quantity": {"value": 0.0, "unit": "e.g., tons/month, m³/week"},
                        "frequency": "e.g., daily, weekly, batch, continuous",
                        "quality_grade": "high, medium, or low, with a brief justification.",
                        "logistical_considerations": "Notes on storage requirements, handling, and potential transport challenges.",
                        "value_breakdown": {
                            "disposal_cost_savings": "Estimated savings from avoiding landfill or disposal fees.",
                            "potential_market_value": "Estimated revenue if sold to a symbiotic partner.",
                            "currency": "USD, EUR, etc."
                        },
                        "potential_symbiotic_partners": [
                            {
                                "industry_type": "e.g., Construction, Agriculture, Energy Production",
                                "use_case": "A specific application for this material in that industry."
                            }
                        ],
                        "sustainability_impact": "Environmental benefits, such as CO2 reduction, water saved, or landfill diversion."
                    }
                ],
                "predicted_inputs": [
                    {
                        "name": "Resource Name",
                        "category": "e.g., raw material, consumable, chemical, water, energy",
                        "description": "Detailed description of the resource and its function in the company's operations.",
                        "quantity": {"value": 0.0, "unit": "e.g., tons/year, kWh/month"},
                        "frequency": "e.g., daily, weekly, per-project",
                        "quality_specification": "Required purity, grade, or technical specifications.",
                        "cost_breakdown": {
                            "procurement_cost": "Estimated cost of acquiring the resource.",
                            "logistics_cost": "Estimated cost of transportation and storage.",
                            "currency": "USD, EUR, etc."
                        },
                        "potential_symbiotic_sourcing": [
                            {
                                "source_industry_type": "The industry that produces this as a byproduct.",
                                "synergy_description": "How sourcing this byproduct could reduce costs and improve sustainability."
                            }
                        ]
                    }
                ],
                "strategic_recommendations": {
                    "company_partnerships": [
                        {
                            "partner_name_or_type": "Specific company name or type of business.",
                            "collaboration_rationale": "A compelling, data-driven reason for the partnership.",
                            "action_plan": "A 3-step plan to initiate and develop the partnership.",
                            "estimated_value_creation": "Quantified potential financial and environmental benefits."
                        }
                    ],
                    "green_initiatives": [
                        {
                            "initiative_name": "e.g., Closed-Loop Water Recycling System",
                            "description": "A detailed explanation of the initiative and its operational impact.",
                            "implementation_plan": "Key steps for implementation, including required technology and personnel.",
                            "estimated_roi": "Return on Investment projection over a 1-3 year period.",
                            "sustainability_kpis": "Key Performance Indicators to track success (e.g., % water reduction, CO2 offset)."
                        }
                    ]
                }
            }

            user_content = f"""Analyze this company profile for comprehensive industrial symbiosis opportunities:

Company: {company_profile.get('name', 'Unknown')}
Industry: {company_profile.get('industry', 'Unknown')}
Products: {company_profile.get('products', 'Unknown')}
Description: {company_profile.get('process_description', '')}
Location: {company_profile.get('location', 'Unknown')}
Employees: {company_profile.get('employee_count', 0)}
Main Materials: {company_profile.get('main_materials', 'Unknown')}
Production Volume: {company_profile.get('production_volume', 'Unknown')}

Generate a comprehensive strategic analysis following the exact JSON structure provided."""

            prompt_data = {
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
                "response_format": {"type": "json_object"},
                "temperature": 0.3,  # Lower temperature for more precise analysis
                "max_tokens": 4000
            }

            response = self._call_deepseek_api(prompt_data)
            return self._parse_strategic_analysis_response(response)

        except Exception as e:
            logger.error(f"Error in strategic_material_analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_fallback_strategic_analysis(company_profile)

    def precision_ai_matchmaking(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prompt 2: Precision AI Matchmaking
        Use this for: Taking a single industrial material and finding the top 3 most viable 
        symbiotic company partners, complete with a detailed business case for each match.
        """
        try:
            logger.info(f"Starting Precision AI Matchmaking for material: {material_data.get('name', 'Unknown')}")
            
            system_prompt = """You are a highly specialized AI-powered industrial matchmaking expert. Your core function is to analyze a specific industrial byproduct or waste stream and identify the most valuable and viable symbiotic partnerships. Your analysis must be practical, insightful, and financially grounded.

TASK: For the provided industrial material, identify the top 3 most promising types of symbiotic partner companies. For each recommendation, provide a detailed business case.

Provide the response as a single, valid JSON object with a single key: 'symbiotic_matches'."""

            response_structure = {
                "symbiotic_matches": [
                    {
                        "rank": 1,
                        "company_type": "e.g., Cement Manufacturer",
                        "match_strength": 95,
                        "match_rationale": {
                            "specific_application": "How exactly the target company would use the material. e.g., 'As a supplementary cementitious material (SCM) to replace a percentage of clinker in concrete production.'",
                            "synergy_value": "A description of the mutual benefits. e.g., 'The cement plant reduces its energy consumption, CO2 emissions, and raw material costs. The source company eliminates disposal fees and creates a new revenue stream.'",
                            "potential_challenges": "Realistic obstacles to consider. e.g., 'Requires consistent quality control of the material. May require investment in drying or grinding equipment.'",
                            "integration_steps": "A brief, actionable plan. e.g., '1. Lab testing of material. 2. Pilot batch production. 3. Long-term supply agreement.'"
                        }
                    },
                    {
                        "rank": 2,
                        "company_type": "e.g., Asphalt Producer",
                        "match_strength": 80,
                        "match_rationale": {
                            "specific_application": "e.g., 'Used as a mineral filler in hot mix asphalt, improving the mechanical properties of the pavement.'",
                            "synergy_value": "e.g., 'Reduces the need for virgin mineral fillers for the asphalt producer, lowering costs. Creates value from a waste stream for the source company.'",
                            "potential_challenges": "e.g., 'Potential for airborne particulates during mixing requires specialized handling equipment.'",
                            "integration_steps": "e.g., '1. Material characterization. 2. Mix design testing. 3. Regulatory approval for road construction.'"
                        }
                    },
                    {
                        "rank": 3,
                        "company_type": "e.g., Agricultural Soil Amendment Supplier",
                        "match_strength": 65,
                        "match_rationale": {
                            "specific_application": "e.g., 'As a soil conditioner to improve pH balance and provide essential micronutrients, if chemical composition is appropriate.'",
                            "synergy_value": "e.g., 'Offers a low-cost, sustainable alternative to traditional lime or fertilizers. Provides an environmentally friendly disposal route for the source company.'",
                            "potential_challenges": "e.g., 'Risk of heavy metal contamination must be rigorously tested and managed. Requires regional and seasonal market alignment.'",
                            "integration_steps": "e.g., '1. Full chemical analysis for agricultural safety. 2. Field trials to prove efficacy. 3. Certification from agricultural bodies.'"
                        }
                    }
                ]
            }

            user_content = f"""Analyze this industrial material for symbiotic partnership opportunities:

Material Name: {material_data.get('name', 'Unknown')}
Category: {material_data.get('category', 'Unknown')}
Description: {material_data.get('description', '')}
Quantity: {material_data.get('quantity', 'Unknown')}
Frequency: {material_data.get('frequency', 'Unknown')}
Quality Grade: {material_data.get('quality_grade', 'Unknown')}
Current Status: {material_data.get('type', 'Unknown')}  # waste or requirement

Find the top 3 most viable symbiotic partner companies with detailed business cases."""

            prompt_data = {
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
                "response_format": {"type": "json_object"},
                "temperature": 0.4,
                "max_tokens": 3000
            }

            response = self._call_deepseek_api(prompt_data)
            return self._parse_matchmaking_response(response)

        except Exception as e:
            logger.error(f"Error in precision_ai_matchmaking: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_fallback_matchmaking(material_data)

    def conversational_intent_analysis(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prompt 3: Advanced Conversational AI & Intent Analysis
        Use this for: Powering a chatbot or NLP system to precisely understand user queries 
        related to industrial symbiosis, extract key information, and intelligently suggest next actions.
        """
        try:
            logger.info(f"Starting Conversational Intent Analysis for input: {user_input[:50]}...")
            
            system_prompt = """You are a sophisticated Linguistic and Intent Analysis Engine designed for an industrial symbiosis platform. Your purpose is to analyze user input with extreme precision to understand their goal and extract critical information. You must reason about the user's request and classify it according to the defined schema.

TASK: Analyze the user's intent, extract all relevant entities, infer the user's probable role, and determine the most logical next step. Your entire response must be a single, valid JSON object.

INTENT LIBRARY:
- `greeting`: User is starting the conversation.
- `goodbye`: User is ending the conversation.
- `find_buyer`: User wants to find a partner to take their output/waste material.
- `find_supplier`: User is looking to source an input material, preferably a byproduct.
- `get_company_profile`: User is asking for information about a specific company.
- `get_material_data`: User is asking for details about a specific material.
- `sustainability_advice`: User is asking for general sustainability or circular economy advice.
- `logistics_query`: User has questions about transportation or storage.
- `cost_query`: User is asking about pricing, value, or costs.
- `help_request`: User is explicitly asking for help or is confused.
- `feedback`: User is providing feedback on the system.
- `unclassified`: The intent cannot be reliably determined from the list.

Return ONLY a valid JSON object with this exact structure:"""

            response_structure = {
                "analysis": {
                    "intent": {
                        "type": "intent_from_library",
                        "confidence": 0.0,
                        "reasoning": "A detailed, step-by-step explanation of why this intent was chosen over others based on keywords and sentence structure."
                    },
                    "entities": [
                        {
                            "entity_type": "e.g., material_name, company_name, location, quantity, category",
                            "value": "The extracted text from the user input.",
                            "confidence": 0.0
                        }
                    ],
                    "user_persona": {
                        "inferred_role": "e.g., Facility Manager, Sustainability Officer, CEO, Logistics Coordinator, Unknown",
                        "confidence": 0.0,
                        "reasoning": "Explanation for the inferred role based on the user's language and query."
                    }
                },
                "next_action": {
                    "clarification_question": "A question to ask the user if confidence is low or information is missing (e.g., 'What quantity of wood chips are you producing per month?'). Should be null if not needed.",
                    "suggested_system_action": "The recommended next API call or function for the application to execute (e.g., 'trigger_material_matchmaking', 'fetch_company_data'). Should be null if a clarification question is posed."
                }
            }

            context_info = ""
            if context:
                context_info = f"Conversation Context: {json.dumps(context, indent=2)}"

            user_content = f"""USER INPUT: "{user_input}"

{context_info}

Analyze the user's intent, extract entities, and determine the next action."""

            prompt_data = {
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
                "response_format": {"type": "json_object"},
                "temperature": 0.2,  # Very low temperature for precise intent analysis
                "max_tokens": 2000
            }

            response = self._call_deepseek_api(prompt_data)
            return self._parse_intent_analysis_response(response)

        except Exception as e:
            logger.error(f"Error in conversational_intent_analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_fallback_intent_analysis(user_input)

    def strategic_company_transformation(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prompt 4: Strategic Company Transformation Analysis
        Use this for: Generating a C-suite level strategic report that analyzes a company's 
        current sustainability posture and provides a phased, actionable roadmap for circular transformation.
        """
        try:
            logger.info(f"Starting Strategic Company Transformation Analysis for company: {company_data.get('name', 'Unknown')}")
            
            system_prompt = """You are a Principal Strategy Consultant specializing in Circular Economy Transformation. You are tasked with analyzing a company's profile to create a high-level strategic report for its executive leadership. Your analysis must be sharp, financially sound, and actionable, converting operational data into a clear roadmap for enhanced profitability and sustainability.

TASK: Generate a comprehensive strategic analysis in the format of a C-suite briefing. Your insights should be based on deep industry knowledge and circular economy principles. Focus on transforming risks into opportunities and analysis into a clear action plan.

Return ONLY a valid JSON object with this exact structure:"""

            response_structure = {
                "executive_summary": {
                    "current_state_assessment": "A brief, honest assessment of the company's current circularity maturity.",
                    "core_recommendation": "The single most impactful strategic shift the company should make.",
                    "summary_of_potential_gains": "Aggregated top-line estimate of potential revenue, cost savings, and key sustainability wins (e.g., CO2 reduction)."
                },
                "strategic_roadmap": {
                    "phase_1_quick_wins": "Initiatives with low cost and high ROI to be implemented within 6 months.",
                    "phase_2_strategic_integration": "Larger projects for process and business model integration (6-18 months).",
                    "phase_3_market_leadership": "Long-term initiatives to establish the company as a leader in sustainability (18+ months)."
                },
                "detailed_analysis": {
                    "waste_stream_valorization": [
                        {
                            "material": "Specific waste stream.",
                            "quantity_estimate": "e.g., 50 tons/month",
                            "current_cost": "Estimated current disposal cost.",
                            "recommended_action": "e.g., 'Sell as raw material to local brick manufacturers.'",
                            "potential_revenue_or_savings": "e.g., '$5,000/month (revenue + cost avoidance)'"
                        }
                    ],
                    "resource_efficiency_opportunities": [
                        {
                            "area": "e.g., 'Water usage in cleaning processes'",
                            "problem": "Brief description of the inefficiency.",
                            "solution": "A specific technological or process-based solution.",
                            "cost_savings_category": "low, medium, or high.",
                            "co2_reduction_potential": "e.g., 'Approx. 50 tonnes CO2e/year'"
                        }
                    ],
                    "symbiotic_partnership_targets": [
                        {
                            "partner_industry": "e.g., 'Greenhouses'",
                            "rationale": "Why this partnership is strategic (e.g., 'Utilize waste heat and CO2 to boost crop yields').",
                            "first_step": "A concrete first action (e.g., 'Contact the regional agricultural association')."
                        }
                    ],
                    "risk_factors": [
                        {
                            "risk_type": "e.g., 'Regulatory', 'Market', 'Operational'",
                            "description": "Specific risk (e.g., 'Upcoming ban on landfilling organic waste').",
                            "mitigation_strategy": "A proactive strategy to address the risk."
                        }
                    ],
                    "innovation_opportunities": [
                        {
                            "opportunity": "A forward-thinking idea (e.g., 'Develop a new product line from recycled material').",
                            "strategic_advantage": "How this innovation could create a competitive moat."
                        }
                    ],
                    "key_regulatory_considerations": [
                        {
                            "regulation": "e.g., 'Extended Producer Responsibility (EPR) laws'",
                            "implication": "How this regulation impacts the company's operations and strategy."
                        }
                    ]
                }
            }

            user_content = f"""COMPANY DATA:
- Name: {company_data.get('name', 'Unknown')}
- Industry: {company_data.get('industry', 'Unknown')}
- Location: {company_data.get('location', 'Unknown')}
- Key Processes: {company_data.get('processes', 'Unknown')}
- Known Materials: {company_data.get('materials', [])}
- Size: {company_data.get('employee_count', 'Unknown')} employees

Generate a comprehensive strategic analysis following the exact JSON structure provided."""

            prompt_data = {
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
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
                "max_tokens": 4000
            }

            response = self._call_deepseek_api(prompt_data)
            return self._parse_transformation_response(response)

        except Exception as e:
            logger.error(f"Error in strategic_company_transformation: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_fallback_transformation(company_data)

    def _call_deepseek_api(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the DeepSeek API with the provided prompt structure."""
        
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
                timeout=60  # 1 minute timeout to prevent long waits
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

    def _call_deepseek_api_directly(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Direct API call method for adaptive prompting
        """
        try:
            prompt_data = {
                "model": self.deepseek_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in industrial symbiosis and circular economy analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.4,  # Slightly higher for more creative responses
                "max_tokens": 3000
            }
            
            response = self._call_deepseek_api(prompt_data)
            return self._parse_strategic_analysis_response(response)
            
        except Exception as e:
            logger.error(f"Error in direct API call: {str(e)}")
            return None

    def _parse_strategic_analysis_response(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the strategic material analysis response."""
        try:
            if 'choices' in api_response and len(api_response['choices']) > 0:
                content = api_response['choices'][0]['message']['content']
                
                if isinstance(content, str):
                    parsed = json.loads(content)
                else:
                    parsed = content
                
                # Validate structure
                required_keys = ['executive_summary', 'predicted_outputs', 'predicted_inputs', 'strategic_recommendations']
                for key in required_keys:
                    if key not in parsed:
                        parsed[key] = {}
                
                return parsed
            else:
                raise ValueError("Invalid API response structure")
                
        except Exception as e:
            logger.error(f"Error parsing strategic analysis response: {str(e)}")
            raise

    def _parse_matchmaking_response(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the precision matchmaking response."""
        try:
            if 'choices' in api_response and len(api_response['choices']) > 0:
                content = api_response['choices'][0]['message']['content']
                
                if isinstance(content, str):
                    parsed = json.loads(content)
                else:
                    parsed = content
                
                # Validate structure
                if 'symbiotic_matches' not in parsed:
                    parsed['symbiotic_matches'] = []
                
                return parsed
            else:
                raise ValueError("Invalid API response structure")
                
        except Exception as e:
            logger.error(f"Error parsing matchmaking response: {str(e)}")
            raise

    def _parse_intent_analysis_response(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the conversational intent analysis response."""
        try:
            if 'choices' in api_response and len(api_response['choices']) > 0:
                content = api_response['choices'][0]['message']['content']
                
                if isinstance(content, str):
                    parsed = json.loads(content)
                else:
                    parsed = content
                
                # Validate structure
                if 'analysis' not in parsed:
                    parsed['analysis'] = {}
                if 'next_action' not in parsed:
                    parsed['next_action'] = {}
                
                return parsed
            else:
                raise ValueError("Invalid API response structure")
                
        except Exception as e:
            logger.error(f"Error parsing intent analysis response: {str(e)}")
            raise

    def _parse_transformation_response(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the strategic transformation response."""
        try:
            if 'choices' in api_response and len(api_response['choices']) > 0:
                content = api_response['choices'][0]['message']['content']
                
                if isinstance(content, str):
                    parsed = json.loads(content)
                else:
                    parsed = content
                
                # Validate structure
                required_keys = ['executive_summary', 'strategic_roadmap', 'detailed_analysis']
                for key in required_keys:
                    if key not in parsed:
                        parsed[key] = {}
                
                return parsed
            else:
                raise ValueError("Invalid API response structure")
                
        except Exception as e:
            logger.error(f"Error parsing transformation response: {str(e)}")
            raise

    # Fallback methods for error handling
    def _get_fallback_strategic_analysis(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback response for strategic material analysis."""
        return {
            "executive_summary": {
                "key_findings": "Analysis temporarily unavailable. Please try again.",
                "primary_opportunities": "Multiple symbiosis opportunities identified.",
                "estimated_total_impact": "Significant cost savings and environmental benefits possible."
            },
            "predicted_outputs": [],
            "predicted_inputs": [],
            "strategic_recommendations": {
                "company_partnerships": [],
                "green_initiatives": []
            }
        }

    def _get_fallback_matchmaking(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback response for precision matchmaking."""
        return {
            "symbiotic_matches": [
                {
                    "rank": 1,
                    "company_type": "Local Manufacturers",
                    "match_strength": 75,
                    "match_rationale": {
                        "specific_application": "General industrial applications",
                        "synergy_value": "Mutual benefit through waste exchange",
                        "potential_challenges": "Requires quality assessment and logistics planning",
                        "integration_steps": "Contact local business associations for introductions"
                    }
                }
            ]
        }

    def _get_fallback_intent_analysis(self, user_input: str) -> Dict[str, Any]:
        """Fallback response for conversational intent analysis."""
        return {
            "analysis": {
                "intent": {
                    "type": "help_request",
                    "confidence": 0.5,
                    "reasoning": "Unable to determine specific intent"
                },
                "entities": [],
                "user_persona": {
                    "inferred_role": "Unknown",
                    "confidence": 0.0,
                    "reasoning": "Insufficient information to determine role"
                }
            },
            "next_action": {
                "clarification_question": "Could you please provide more details about what you're looking for?",
                "suggested_system_action": None
            }
        }

    def _get_fallback_transformation(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback response for strategic transformation analysis."""
        return {
            "executive_summary": {
                "current_state_assessment": "Analysis temporarily unavailable",
                "core_recommendation": "Begin with waste stream assessment",
                "summary_of_potential_gains": "Significant opportunities for cost savings and sustainability improvements"
            },
            "strategic_roadmap": {
                "phase_1_quick_wins": "Start with waste audit and identify immediate opportunities",
                "phase_2_strategic_integration": "Develop partnerships and implement process improvements",
                "phase_3_market_leadership": "Establish circular economy leadership position"
            },
            "detailed_analysis": {
                "waste_stream_valorization": [],
                "resource_efficiency_opportunities": [],
                "symbiotic_partnership_targets": [],
                "risk_factors": [],
                "innovation_opportunities": [],
                "key_regulatory_considerations": []
            }
        }

# Convenience functions for easy integration
def analyze_company_strategically(company_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for strategic material analysis."""
    service = AdvancedAIPromptsService()
    return service.strategic_material_analysis(company_profile)

def find_precision_matches(material_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for precision AI matchmaking."""
    service = AdvancedAIPromptsService()
    return service.precision_ai_matchmaking(material_data)

def analyze_conversational_intent(user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function for conversational intent analysis."""
    service = AdvancedAIPromptsService()
    return service.conversational_intent_analysis(user_input, context)

def analyze_company_transformation(company_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for strategic company transformation analysis."""
    service = AdvancedAIPromptsService()
    return service.strategic_company_transformation(company_data) 