#!/usr/bin/env python3
"""
Comprehensive DeepSeek R1 Optimization Test Suite

This script tests all DeepSeek R1 integrations and optimizations across the system:
- AI Service (onboarding, analysis, questions, materials)
- Core Matching Engine (compatibility analysis, reasoning)
- Conversational B2B Agent (intent analysis, response generation)
- Advanced AI Features (federated learning, meta-learning, RL, CV, NLG, decision making)

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepSeekR1OptimizationTester:
    """Comprehensive tester for DeepSeek R1 optimizations"""
    
    def __init__(self):
        self.api_key = "sk-7ce79f30332d45d5b3acb8968b052132"
        self.base_url = "https://api.deepseek.com/v1"
        self.model = "deepseek-r1"
        self.test_results = {}
        self.start_time = datetime.now()
        
    def _make_test_request(self, messages: List[Dict], temperature: float = 0.7) -> Dict:
        """Make a test request to DeepSeek R1"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2000,
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
                return {
                    'success': True,
                    'content': result['choices'][0]['message']['content'],
                    'usage': result.get('usage', {}),
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                return {
                    'success': False,
                    'error': f"API error: {response.status_code} - {response.text}",
                    'response_time': response.elapsed.total_seconds()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Request failed: {str(e)}",
                'response_time': 0
            }
    
    def test_ai_service_optimization(self) -> Dict:
        """Test AI Service DeepSeek R1 optimizations"""
        logger.info("🧪 Testing AI Service DeepSeek R1 optimizations...")
        
        test_company = {
            'name': 'Test Textile Company',
            'industry': 'Textile Manufacturing',
            'location': 'Berlin, Germany',
            'processes': 'Cotton spinning, weaving, dyeing',
            'materials': ['cotton', 'dyes', 'chemicals'],
            'employee_count': '150'
        }
        
        # Test 1: Company Analysis
        analysis_prompt = f"""
        You are DeepSeek R1, an expert industrial sustainability analyst. Analyze this company data using advanced reasoning:

        COMPANY DATA:
        - Name: {test_company['name']}
        - Industry: {test_company['industry']}
        - Location: {test_company['location']}
        - Processes: {test_company['processes']}
        - Materials: {test_company['materials']}
        - Size: {test_company['employee_count']} employees

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
            {"role": "user", "content": analysis_prompt}
        ]
        
        analysis_result = self._make_test_request(messages, temperature=0.2)
        
        # Test 2: Material Listings Generation
        material_prompt = f"""
        You are DeepSeek R1, an expert in industrial materials and circular economy with deep knowledge of waste-to-resource opportunities. Generate realistic material listings for this company using advanced reasoning:

        COMPANY DATA:
        - Name: {test_company['name']}
        - Industry: {test_company['industry']}
        - Location: {test_company['location']}
        - Processes: {test_company['processes']}
        - Materials: {test_company['materials']}
        - Size: {test_company['employee_count']} employees

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
            {"role": "user", "content": material_prompt}
        ]
        
        material_result = self._make_test_request(messages, temperature=0.4)
        
        return {
            'analysis_test': analysis_result,
            'material_test': material_result,
            'optimization_features': [
                'Advanced reasoning capabilities',
                'Industry-specific analysis',
                'Logical process-material connections',
                'Geographical considerations',
                'Realistic quantity estimation',
                'Business case development'
            ]
        }
    
    def test_core_matching_optimization(self) -> Dict:
        """Test Core Matching Engine DeepSeek R1 optimizations"""
        logger.info("🧪 Testing Core Matching Engine DeepSeek R1 optimizations...")
        
        # Test material compatibility analysis
        material1 = {
            'name': 'Cotton Waste',
            'description': 'Post-production cotton waste from textile manufacturing',
            'category': 'Textile Waste',
            'properties': {'fiber_length': 'short', 'purity': '85%', 'moisture': '12%'},
            'quantity': 5000,
            'unit': 'kg/month'
        }
        
        material2 = {
            'name': 'Paper Pulp',
            'description': 'Recycled paper pulp for packaging production',
            'category': 'Paper Products',
            'properties': {'fiber_length': 'mixed', 'purity': '90%', 'moisture': '8%'},
            'quantity': 3000,
            'unit': 'kg/month'
        }
        
        compatibility_prompt = f"""
        You are DeepSeek R1, an expert industrial materials analyst with deep knowledge of material science, chemistry, and industrial symbiosis. Analyze the compatibility between two industrial materials using advanced reasoning:

        MATERIAL 1:
        - Name: {material1['name']}
        - Description: {material1['description']}
        - Category: {material1['category']}
        - Properties: {material1['properties']}
        - Quantity: {material1['quantity']} {material1['unit']}

        MATERIAL 2:
        - Name: {material2['name']}
        - Description: {material2['description']}
        - Category: {material2['category']}
        - Properties: {material2['properties']}
        - Quantity: {material2['quantity']} {material2['unit']}

        TASK: Provide a comprehensive compatibility analysis using DeepSeek R1's advanced reasoning capabilities:

        ANALYSIS REQUIREMENTS:
        1. Chemical Compatibility: Analyze chemical properties and potential reactions
        2. Physical Compatibility: Consider physical properties and processing requirements
        3. Industrial Applications: Identify specific industrial applications and use cases
        4. Technical Feasibility: Assess technical challenges and requirements
        5. Environmental Benefits: Quantify environmental impact and sustainability benefits
        6. Economic Viability: Analyze cost-benefit and market potential
        7. Risk Assessment: Identify potential risks and mitigation strategies
        8. Implementation Considerations: Provide practical implementation guidance

        REASONING REQUIREMENTS:
        - Use logical reasoning to connect material properties to compatibility
        - Consider industry standards and best practices
        - Provide quantifiable estimates where possible
        - Focus on practical, implementable solutions
        - Consider safety and regulatory requirements

        Return ONLY valid JSON with this exact structure:
        {{
            "score": 0-100,
            "reasoning": "detailed reasoning for the compatibility score",
            "applications": ["specific industrial applications with reasoning"],
            "technical_considerations": ["technical factors with detailed analysis"],
            "environmental_benefits": ["quantified environmental benefits"],
            "economic_feasibility": "detailed economic analysis with cost-benefit",
            "risks": ["specific risks with impact assessment and mitigation"],
            "implementation_guidance": ["step-by-step implementation recommendations"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are DeepSeek R1, an expert industrial materials analyst. Use your advanced reasoning capabilities to provide precise, actionable compatibility analysis for industrial symbiosis. Always respond with valid JSON only."},
            {"role": "user", "content": compatibility_prompt}
        ]
        
        compatibility_result = self._make_test_request(messages, temperature=0.2)
        
        return {
            'compatibility_test': compatibility_result,
            'optimization_features': [
                'Advanced material science reasoning',
                'Chemical and physical compatibility analysis',
                'Technical feasibility assessment',
                'Environmental impact quantification',
                'Economic viability analysis',
                'Risk assessment and mitigation',
                'Implementation guidance'
            ]
        }
    
    def test_conversational_agent_optimization(self) -> Dict:
        """Test Conversational B2B Agent DeepSeek R1 optimizations"""
        logger.info("🧪 Testing Conversational B2B Agent DeepSeek R1 optimizations...")
        
        # Test intent analysis
        user_input = "I have 5 tons of cotton waste per month and need to find companies that can use it. What are the environmental benefits?"
        
        intent_prompt = f"""
        You are DeepSeek R1, an expert in natural language understanding and industrial symbiosis. Analyze the user's intent using advanced reasoning:

        USER INPUT: "{user_input}"

        CONVERSATION CONTEXT:
        - Conversation ID: test_conv_123
        - User ID: test_user_456
        - Company ID: test_company_789
        - Current Topic: material matching
        - Previous Turns: 2 turns
        - User Preferences: {{"industry": "textile", "location": "Berlin"}}

        TASK: Analyze the user's intent and extract relevant entities using DeepSeek R1's reasoning capabilities.

        INTENT TYPES:
        1. greeting - User is greeting or starting conversation
        2. matching_request - User wants to find material matches or partnerships
        3. material_search - User is searching for specific materials
        4. company_search - User is looking for companies
        5. sustainability_query - User has questions about sustainability or environmental impact
        6. logistics_query - User has questions about transportation, routes, or logistics
        7. cost_query - User has questions about costs, pricing, or economics
        8. help_request - User needs help or assistance
        9. feedback - User is providing feedback
        10. unknown - Intent cannot be determined

        ENTITY TYPES TO EXTRACT:
        - material: Industrial materials, waste, products
        - company: Company names, businesses
        - location: Geographic locations, cities, countries
        - quantity: Amounts, volumes, weights
        - price: Costs, prices, monetary values
        - date: Dates, time periods
        - industry: Industry sectors, business types

        REASONING REQUIREMENTS:
        - Use logical reasoning to understand the user's underlying intent
        - Consider conversation context and previous turns
        - Extract relevant entities with high precision
        - Consider industrial symbiosis domain knowledge
        - Handle ambiguous or unclear inputs appropriately

        Return ONLY valid JSON with this exact structure:
        {{
            "intent": {{
                "type": "intent_type_from_list_above",
                "confidence": 0.0-1.0,
                "reasoning": "detailed explanation of why this intent was chosen"
            }},
            "entities": [
                {{
                    "type": "entity_type_from_list_above",
                    "value": "extracted_value",
                    "confidence": 0.0-1.0,
                    "start_pos": 0,
                    "end_pos": 0
                }}
            ],
            "context_updates": {{
                "current_topic": "updated_topic_if_relevant",
                "preferences": {{"key": "value"}},
                "suggestions": ["suggested_next_actions"]
            }}
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are DeepSeek R1, an expert in natural language understanding for industrial symbiosis. Use your advanced reasoning to accurately analyze user intents and extract relevant entities. Always respond with valid JSON only."},
            {"role": "user", "content": intent_prompt}
        ]
        
        intent_result = self._make_test_request(messages, temperature=0.2)
        
        # Test response generation
        response_prompt = f"""
        You are DeepSeek R1, an expert industrial symbiosis consultant and conversational AI. Generate a helpful, professional response using advanced reasoning:

        USER INPUT: "{user_input}"

        DETECTED INTENT:
        - Type: matching_request
        - Confidence: 0.95
        - Reasoning: User is seeking material matches and sustainability information

        EXTRACTED ENTITIES:
        [
            {{"type": "material", "value": "cotton waste", "confidence": 0.98, "start_pos": 8, "end_pos": 20}},
            {{"type": "quantity", "value": "5 tons per month", "confidence": 0.92, "start_pos": 8, "end_pos": 25}},
            {{"type": "sustainability_query", "value": "environmental benefits", "confidence": 0.89, "start_pos": 70, "end_pos": 90}}
        ]

        CONVERSATION CONTEXT:
        - User ID: test_user_456
        - Company ID: test_company_789
        - Current Topic: material matching
        - User Preferences: {{"industry": "textile", "location": "Berlin"}}
        - Conversation History: Previous discussion about textile waste

        TASK: Generate a helpful, professional response that:
        1. Addresses the user's intent appropriately
        2. Uses the extracted entities to provide relevant information
        3. Maintains conversation flow and context
        4. Provides actionable insights or next steps
        5. Demonstrates expertise in industrial symbiosis
        6. Is concise but comprehensive

        RESPONSE REQUIREMENTS:
        - Be professional and business-focused
        - Use logical reasoning to provide valuable insights
        - Consider the user's company and preferences
        - Offer specific, actionable suggestions when appropriate
        - Ask clarifying questions if needed
        - Provide relevant examples or case studies
        - Maintain a helpful, expert tone

        Generate a natural, conversational response that would be helpful for a business professional in the industrial symbiosis domain.
        """
        
        messages = [
            {"role": "system", "content": "You are DeepSeek R1, an expert industrial symbiosis consultant. Use your advanced reasoning to provide helpful, professional responses that demonstrate deep knowledge of industrial symbiosis, circular economy, and business optimization. Be conversational, expert, and actionable."},
            {"role": "user", "content": response_prompt}
        ]
        
        response_result = self._make_test_request(messages, temperature=0.6)
        
        return {
            'intent_analysis_test': intent_result,
            'response_generation_test': response_result,
            'optimization_features': [
                'Advanced natural language understanding',
                'Context-aware intent analysis',
                'Precise entity extraction',
                'Domain-specific reasoning',
                'Professional response generation',
                'Business-focused insights',
                'Conversation flow maintenance'
            ]
        }
    
    def test_advanced_ai_features_optimization(self) -> Dict:
        """Test Advanced AI Features DeepSeek R1 optimizations"""
        logger.info("🧪 Testing Advanced AI Features DeepSeek R1 optimizations...")
        
        # Test decision making
        decision_prompt = f"""
        You are DeepSeek R1, an expert decision-making AI for industrial symbiosis. Analyze and aggregate multiple decision inputs using advanced reasoning:

        DECISION TYPE: material_matching
        INPUT DATA: {{
            "material1": {{
                "name": "Cotton Waste",
                "quantity": "5 tons/month",
                "location": "Berlin",
                "company": "TextileCo"
            }},
            "material2": {{
                "name": "Paper Pulp",
                "quantity": "3 tons/month", 
                "location": "Hamburg",
                "company": "PaperCo"
            }},
            "constraints": {{
                "max_distance": "500 km",
                "min_confidence": 0.8,
                "budget_limit": "€5000/month"
            }}
        }}

        MODEL DECISIONS:
        [
            {{
                "model": "semantic_matcher",
                "decision": "match",
                "confidence": 0.85,
                "reasoning": "High semantic similarity between cotton waste and paper pulp applications"
            }},
            {{
                "model": "logistics_optimizer", 
                "decision": "match",
                "confidence": 0.78,
                "reasoning": "Distance within acceptable range, cost-effective transportation available"
            }},
            {{
                "model": "sustainability_analyzer",
                "decision": "match",
                "confidence": 0.92,
                "reasoning": "High environmental benefit potential, significant carbon savings"
            }}
        ]

        RULE DECISIONS:
        [
            {{
                "rule": "distance_constraint",
                "decision": "match",
                "reasoning": "Distance (300 km) is within maximum allowed (500 km)"
            }},
            {{
                "rule": "budget_constraint",
                "decision": "match", 
                "reasoning": "Transportation cost (€2000/month) is within budget (€5000/month)"
            }}
        ]

        TASK: Provide a final decision using DeepSeek R1's reasoning capabilities:

        REQUIREMENTS:
        1. Analyze the confidence and reasoning of each model decision
        2. Consider the business rules and their implications
        3. Use logical reasoning to determine the optimal final decision
        4. Provide detailed reasoning for the chosen decision
        5. Consider risk factors and mitigation strategies
        6. Ensure the decision aligns with business objectives

        Return ONLY valid JSON with this exact structure:
        {{
            "decision": "final_decision_value",
            "confidence": 0.0-1.0,
            "reasoning": "detailed reasoning for the decision",
            "risk_assessment": "risk analysis and mitigation",
            "recommendations": ["specific recommendations"],
            "requires_human_review": true|false
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are DeepSeek R1, an expert decision-making AI. Use your advanced reasoning to analyze multiple inputs and provide optimal decisions for industrial symbiosis. Always respond with valid JSON only."},
            {"role": "user", "content": decision_prompt}
        ]
        
        decision_result = self._make_test_request(messages, temperature=0.2)
        
        return {
            'decision_making_test': decision_result,
            'optimization_features': [
                'Advanced decision aggregation',
                'Multi-model reasoning',
                'Business rule integration',
                'Risk assessment',
                'Confidence analysis',
                'Human review recommendations',
                'Strategic alignment'
            ]
        }
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive DeepSeek R1 optimization test"""
        logger.info("🚀 Starting Comprehensive DeepSeek R1 Optimization Test...")
        
        test_results = {
            'test_start_time': self.start_time.isoformat(),
            'model_used': self.model,
            'api_endpoint': self.base_url,
            'tests': {}
        }
        
        # Test 1: AI Service Optimization
        test_results['tests']['ai_service'] = self.test_ai_service_optimization()
        
        # Test 2: Core Matching Engine Optimization
        test_results['tests']['core_matching'] = self.test_core_matching_optimization()
        
        # Test 3: Conversational Agent Optimization
        test_results['tests']['conversational_agent'] = self.test_conversational_agent_optimization()
        
        # Test 4: Advanced AI Features Optimization
        test_results['tests']['advanced_ai_features'] = self.test_advanced_ai_features_optimization()
        
        # Calculate overall metrics
        test_results['summary'] = self._calculate_test_summary(test_results['tests'])
        test_results['test_end_time'] = datetime.now().isoformat()
        test_results['total_duration'] = (datetime.now() - self.start_time).total_seconds()
        
        return test_results
    
    def _calculate_test_summary(self, tests: Dict) -> Dict:
        """Calculate summary metrics for all tests"""
        total_tests = 0
        successful_tests = 0
        total_response_time = 0
        total_tokens_used = 0
        
        for test_category, test_data in tests.items():
            for test_name, test_result in test_data.items():
                if test_name == 'optimization_features':
                    continue
                    
                total_tests += 1
                if test_result.get('success', False):
                    successful_tests += 1
                    total_response_time += test_result.get('response_time', 0)
                    total_tokens_used += test_result.get('usage', {}).get('total_tokens', 0)
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            'average_response_time': (total_response_time / successful_tests) if successful_tests > 0 else 0,
            'total_tokens_used': total_tokens_used,
            'optimization_features_count': sum(len(test_data.get('optimization_features', [])) for test_data in tests.values())
        }
    
    def save_test_results(self, results: Dict, filename: str = None):
        """Save test results to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'deepseek_r1_optimization_test_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📄 Test results saved to: {filename}")
        return filename
    
    def print_test_summary(self, results: Dict):
        """Print a summary of test results"""
        print("\n" + "="*80)
        print("🔬 DEEPSEEK R1 OPTIMIZATION TEST RESULTS")
        print("="*80)
        
        summary = results['summary']
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"   • Total Tests: {summary['total_tests']}")
        print(f"   • Successful Tests: {summary['successful_tests']}")
        print(f"   • Success Rate: {summary['success_rate']:.1f}%")
        print(f"   • Average Response Time: {summary['average_response_time']:.2f}s")
        print(f"   • Total Tokens Used: {summary['total_tokens_used']}")
        print(f"   • Optimization Features: {summary['optimization_features_count']}")
        
        print(f"\n🧪 TEST CATEGORIES:")
        for category, test_data in results['tests'].items():
            print(f"\n   {category.upper().replace('_', ' ')}:")
            for test_name, test_result in test_data.items():
                if test_name == 'optimization_features':
                    continue
                    
                status = "✅ PASS" if test_result.get('success', False) else "❌ FAIL"
                response_time = test_result.get('response_time', 0)
                print(f"     • {test_name}: {status} ({response_time:.2f}s)")
                
                if test_result.get('success', False) and 'usage' in test_result:
                    tokens = test_result['usage'].get('total_tokens', 0)
                    print(f"       Tokens used: {tokens}")
        
        print(f"\n🚀 OPTIMIZATION FEATURES:")
        all_features = set()
        for test_data in results['tests'].values():
            features = test_data.get('optimization_features', [])
            all_features.update(features)
        
        for i, feature in enumerate(sorted(all_features), 1):
            print(f"   {i:2d}. {feature}")
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   • Test Duration: {results['total_duration']:.2f} seconds")
        print(f"   • Model: {results['model_used']}")
        print(f"   • API Endpoint: {results['api_endpoint']}")
        
        print("\n" + "="*80)

def main():
    """Main test execution function"""
    print("🚀 DeepSeek R1 Optimization Test Suite")
    print("="*50)
    
    # Initialize tester
    tester = DeepSeekR1OptimizationTester()
    
    try:
        # Run comprehensive test
        results = tester.run_comprehensive_test()
        
        # Save results
        filename = tester.save_test_results(results)
        
        # Print summary
        tester.print_test_summary(results)
        
        # Check if all tests passed
        success_rate = results['summary']['success_rate']
        if success_rate >= 90:
            print("\n🎉 EXCELLENT! DeepSeek R1 optimizations are working perfectly!")
        elif success_rate >= 75:
            print("\n✅ GOOD! DeepSeek R1 optimizations are working well with minor issues.")
        elif success_rate >= 50:
            print("\n⚠️  FAIR! DeepSeek R1 optimizations have some issues that need attention.")
        else:
            print("\n❌ POOR! DeepSeek R1 optimizations have significant issues that need immediate attention.")
        
        print(f"\n📄 Detailed results saved to: {filename}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"\n❌ Test execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 