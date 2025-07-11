#!/usr/bin/env python3
"""
Advanced AI Integration Script
Demonstrates how to use the new AdvancedAIPromptsService with the four strategic prompts
for enhanced industrial symbiosis analysis.
"""

import json
import logging
from typing import Dict, Any, List
from advanced_ai_prompts_service import (
    AdvancedAIPromptsService,
    analyze_company_strategically,
    find_precision_matches,
    analyze_conversational_intent,
    analyze_company_transformation
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAIIntegration:
    """
    Integration class that demonstrates how to use the advanced AI prompts
    and provides utilities for integrating with existing systems.
    """
    
    def __init__(self):
        self.ai_service = AdvancedAIPromptsService()
        
    def demonstrate_strategic_analysis(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Demonstrate the Strategic Material & Synergy Analysis prompt.
        This replaces the basic material generation with comprehensive strategic analysis.
        """
        logger.info("=== DEMONSTRATING STRATEGIC MATERIAL ANALYSIS ===")
        
        try:
            # Use the advanced strategic analysis
            result = self.ai_service.strategic_material_analysis(company_profile)
            
            # Log key insights
            if 'executive_summary' in result:
                summary = result['executive_summary']
                logger.info(f"Key Findings: {summary.get('key_findings', 'N/A')}")
                logger.info(f"Primary Opportunities: {summary.get('primary_opportunities', 'N/A')}")
                logger.info(f"Estimated Impact: {summary.get('estimated_total_impact', 'N/A')}")
            
            # Log material counts
            outputs_count = len(result.get('predicted_outputs', []))
            inputs_count = len(result.get('predicted_inputs', []))
            logger.info(f"Generated {outputs_count} outputs and {inputs_count} inputs")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in strategic analysis demonstration: {str(e)}")
            return {}
    
    def demonstrate_precision_matchmaking(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Demonstrate the Precision AI Matchmaking prompt.
        This provides detailed business cases for material partnerships.
        """
        logger.info("=== DEMONSTRATING PRECISION AI MATCHMAKING ===")
        
        try:
            # Use the advanced precision matchmaking
            result = self.ai_service.precision_ai_matchmaking(material_data)
            
            # Log match details
            matches = result.get('symbiotic_matches', [])
            logger.info(f"Found {len(matches)} potential symbiotic matches")
            
            for match in matches:
                rank = match.get('rank', 'N/A')
                company_type = match.get('company_type', 'N/A')
                match_strength = match.get('match_strength', 'N/A')
                logger.info(f"Rank {rank}: {company_type} (Match Strength: {match_strength}%)")
                
                rationale = match.get('match_rationale', {})
                if 'specific_application' in rationale:
                    logger.info(f"  Application: {rationale['specific_application']}")
                if 'synergy_value' in rationale:
                    logger.info(f"  Synergy Value: {rationale['synergy_value']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in precision matchmaking demonstration: {str(e)}")
            return {}
    
    def demonstrate_conversational_analysis(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Demonstrate the Advanced Conversational AI & Intent Analysis prompt.
        This provides sophisticated NLP understanding for chatbot systems.
        """
        logger.info("=== DEMONSTRATING CONVERSATIONAL INTENT ANALYSIS ===")
        
        try:
            # Use the advanced conversational analysis
            result = self.ai_service.conversational_intent_analysis(user_input, context)
            
            # Log analysis results
            analysis = result.get('analysis', {})
            intent = analysis.get('intent', {})
            logger.info(f"Detected Intent: {intent.get('type', 'N/A')} (Confidence: {intent.get('confidence', 0):.2f})")
            logger.info(f"Intent Reasoning: {intent.get('reasoning', 'N/A')}")
            
            # Log entities
            entities = analysis.get('entities', [])
            logger.info(f"Extracted {len(entities)} entities:")
            for entity in entities:
                entity_type = entity.get('entity_type', 'N/A')
                value = entity.get('value', 'N/A')
                confidence = entity.get('confidence', 0)
                logger.info(f"  {entity_type}: {value} (Confidence: {confidence:.2f})")
            
            # Log user persona
            persona = analysis.get('user_persona', {})
            logger.info(f"Inferred Role: {persona.get('inferred_role', 'N/A')} (Confidence: {persona.get('confidence', 0):.2f})")
            
            # Log next action
            next_action = result.get('next_action', {})
            if next_action.get('clarification_question'):
                logger.info(f"Clarification Question: {next_action['clarification_question']}")
            if next_action.get('suggested_system_action'):
                logger.info(f"Suggested Action: {next_action['suggested_system_action']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in conversational analysis demonstration: {str(e)}")
            return {}
    
    def demonstrate_transformation_analysis(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Demonstrate the Strategic Company Transformation Analysis prompt.
        This provides C-suite level strategic insights and roadmaps.
        """
        logger.info("=== DEMONSTRATING STRATEGIC TRANSFORMATION ANALYSIS ===")
        
        try:
            # Use the advanced transformation analysis
            result = self.ai_service.strategic_company_transformation(company_data)
            
            # Log executive summary
            summary = result.get('executive_summary', {})
            logger.info(f"Current State: {summary.get('current_state_assessment', 'N/A')}")
            logger.info(f"Core Recommendation: {summary.get('core_recommendation', 'N/A')}")
            logger.info(f"Potential Gains: {summary.get('summary_of_potential_gains', 'N/A')}")
            
            # Log strategic roadmap
            roadmap = result.get('strategic_roadmap', {})
            logger.info("Strategic Roadmap:")
            logger.info(f"  Phase 1 (Quick Wins): {roadmap.get('phase_1_quick_wins', 'N/A')}")
            logger.info(f"  Phase 2 (Strategic Integration): {roadmap.get('phase_2_strategic_integration', 'N/A')}")
            logger.info(f"  Phase 3 (Market Leadership): {roadmap.get('phase_3_market_leadership', 'N/A')}")
            
            # Log detailed analysis counts
            detailed = result.get('detailed_analysis', {})
            waste_streams = len(detailed.get('waste_stream_valorization', []))
            efficiency_ops = len(detailed.get('resource_efficiency_opportunities', []))
            partnerships = len(detailed.get('symbiotic_partnership_targets', []))
            risks = len(detailed.get('risk_factors', []))
            innovations = len(detailed.get('innovation_opportunities', []))
            
            logger.info(f"Detailed Analysis: {waste_streams} waste streams, {efficiency_ops} efficiency opportunities, {partnerships} partnerships, {risks} risks, {innovations} innovations")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in transformation analysis demonstration: {str(e)}")
            return {}
    
    def enhanced_material_generation(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced material generation that combines strategic analysis with traditional listing generation.
        This provides a more comprehensive approach than the basic listing inference service.
        """
        logger.info("=== ENHANCED MATERIAL GENERATION ===")
        
        try:
            # Get strategic analysis
            strategic_result = self.ai_service.strategic_material_analysis(company_profile)
            
            # Transform strategic analysis into traditional listing format
            enhanced_listings = {
                'predicted_outputs': [],
                'predicted_inputs': [],
                'company_suggestions': [],
                'green_initiatives': [],
                'ai_insights': {},
                'strategic_analysis': strategic_result
            }
            
            # Convert strategic outputs to traditional format
            for output in strategic_result.get('predicted_outputs', []):
                enhanced_output = {
                    'name': output.get('name', ''),
                    'category': output.get('category', ''),
                    'description': output.get('description', ''),
                    'quantity': f"{output.get('quantity', {}).get('value', 0)} {output.get('quantity', {}).get('unit', 'units')}",
                    'frequency': output.get('frequency', 'monthly'),
                    'notes': output.get('logistical_considerations', ''),
                    'potential_value': output.get('value_breakdown', {}).get('potential_market_value', ''),
                    'quality_grade': output.get('quality_grade', 'medium'),
                    'potential_uses': [partner.get('use_case', '') for partner in output.get('potential_symbiotic_partners', [])],
                    'symbiosis_opportunities': [partner.get('industry_type', '') for partner in output.get('potential_symbiotic_partners', [])],
                    'sustainability_impact': output.get('sustainability_impact', ''),
                    'cost_savings': output.get('value_breakdown', {}).get('disposal_cost_savings', '')
                }
                enhanced_listings['predicted_outputs'].append(enhanced_output)
            
            # Convert strategic inputs to traditional format
            for input_item in strategic_result.get('predicted_inputs', []):
                enhanced_input = {
                    'name': input_item.get('name', ''),
                    'category': input_item.get('category', ''),
                    'description': input_item.get('description', ''),
                    'quantity': f"{input_item.get('quantity', {}).get('value', 0)} {input_item.get('quantity', {}).get('unit', 'units')}",
                    'frequency': input_item.get('frequency', 'monthly'),
                    'notes': input_item.get('quality_specification', ''),
                    'potential_value': f"{input_item.get('cost_breakdown', {}).get('procurement_cost', '')} + {input_item.get('cost_breakdown', {}).get('logistics_cost', '')}",
                    'quality_grade': 'high',  # Inputs typically need high quality
                    'potential_uses': ['Primary operational need'],
                    'symbiosis_opportunities': [source.get('source_industry_type', '') for source in input_item.get('potential_symbiotic_sourcing', [])],
                    'sustainability_impact': 'Reduced virgin resource consumption',
                    'cost_savings': input_item.get('cost_breakdown', {}).get('procurement_cost', '')
                }
                enhanced_listings['predicted_inputs'].append(enhanced_input)
            
            # Convert strategic recommendations to traditional format
            for partnership in strategic_result.get('strategic_recommendations', {}).get('company_partnerships', []):
                enhanced_partnership = {
                    'company_type': partnership.get('partner_name_or_type', ''),
                    'location': 'Local/Regional',
                    'waste_they_can_use': ['Based on strategic analysis'],
                    'resources_they_can_provide': ['Based on strategic analysis'],
                    'estimated_partnership_value': partnership.get('estimated_value_creation', ''),
                    'carbon_reduction': 'Significant',
                    'implementation_time': '6-18 months'
                }
                enhanced_listings['company_suggestions'].append(enhanced_partnership)
            
            # Convert green initiatives
            for initiative in strategic_result.get('strategic_recommendations', {}).get('green_initiatives', []):
                enhanced_initiative = {
                    'initiative_name': initiative.get('initiative_name', ''),
                    'description': initiative.get('description', ''),
                    'current_practice': 'Traditional linear economy approach',
                    'greener_alternative': initiative.get('description', ''),
                    'cost_savings_per_month': initiative.get('estimated_roi', ''),
                    'carbon_reduction': 'Significant',
                    'implementation_cost': 'Variable',
                    'payback_period': initiative.get('estimated_roi', ''),
                    'difficulty': 'medium',
                    'priority': 'high'
                }
                enhanced_listings['green_initiatives'].append(enhanced_initiative)
            
            # Add AI insights
            enhanced_listings['ai_insights'] = {
                'symbiosis_score': '85-95%',
                'estimated_savings': strategic_result.get('executive_summary', {}).get('estimated_total_impact', ''),
                'carbon_reduction': 'Significant reduction in carbon footprint',
                'top_opportunities': [summary.get('primary_opportunities', '')],
                'recommended_partners': [partner.get('partner_name_or_type', '') for partner in strategic_result.get('strategic_recommendations', {}).get('company_partnerships', [])],
                'implementation_roadmap': ['Phase 1: Quick wins', 'Phase 2: Strategic integration', 'Phase 3: Market leadership']
            }
            
            logger.info(f"Enhanced generation complete: {len(enhanced_listings['predicted_outputs'])} outputs, {len(enhanced_listings['predicted_inputs'])} inputs")
            
            return enhanced_listings
            
        except Exception as e:
            logger.error(f"Error in enhanced material generation: {str(e)}")
            return {}
    
    def run_comprehensive_demo(self, company_profile: Dict[str, Any], material_data: Dict[str, Any] = None, user_input: str = None):
        """
        Run a comprehensive demonstration of all four advanced AI prompts.
        """
        logger.info("🚀 STARTING COMPREHENSIVE ADVANCED AI DEMONSTRATION")
        logger.info("=" * 60)
        
        # 1. Strategic Material Analysis
        strategic_result = self.demonstrate_strategic_analysis(company_profile)
        
        # 2. Enhanced Material Generation (combines strategic with traditional)
        enhanced_result = self.enhanced_material_generation(company_profile)
        
        # 3. Precision Matchmaking (if material data provided)
        if material_data:
            matchmaking_result = self.demonstrate_precision_matchmaking(material_data)
        
        # 4. Conversational Analysis (if user input provided)
        if user_input:
            conversational_result = self.demonstrate_conversational_analysis(user_input)
        
        # 5. Strategic Transformation Analysis
        transformation_result = self.demonstrate_transformation_analysis(company_profile)
        
        logger.info("=" * 60)
        logger.info("✅ COMPREHENSIVE DEMONSTRATION COMPLETE")
        
        return {
            'strategic_analysis': strategic_result,
            'enhanced_generation': enhanced_result,
            'matchmaking': matchmaking_result if material_data else None,
            'conversational': conversational_result if user_input else None,
            'transformation': transformation_result
        }

# Example usage and demonstration
def main():
    """Main demonstration function with example data."""
    
    # Example company profile
    example_company = {
        'name': 'GreenTech Manufacturing',
        'industry': 'Electronics Manufacturing',
        'products': 'Printed Circuit Boards, Electronic Components',
        'process_description': 'PCB manufacturing with surface mount technology, component assembly, and testing',
        'location': 'Austin, Texas',
        'employee_count': 150,
        'main_materials': 'Copper, FR4 laminate, solder, chemicals, water',
        'production_volume': '10,000 PCBs per month'
    }
    
    # Example material data for matchmaking
    example_material = {
        'name': 'Copper Etching Waste',
        'category': 'chemical',
        'description': 'Waste solution from PCB copper etching process containing dissolved copper and etching chemicals',
        'quantity': '500 liters/week',
        'frequency': 'weekly',
        'quality_grade': 'medium',
        'type': 'waste'
    }
    
    # Example user input for conversational analysis
    example_user_input = "I'm looking for companies that can use our copper etching waste. We produce about 500 liters per week and want to avoid disposal costs."
    
    # Create integration instance
    integration = AdvancedAIIntegration()
    
    # Run comprehensive demonstration
    results = integration.run_comprehensive_demo(
        company_profile=example_company,
        material_data=example_material,
        user_input=example_user_input
    )
    
    # Save results to file for review
    with open('advanced_ai_demo_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("📄 Results saved to 'advanced_ai_demo_results.json'")
    logger.info("🎯 Advanced AI Integration demonstration complete!")

if __name__ == "__main__":
    main() 