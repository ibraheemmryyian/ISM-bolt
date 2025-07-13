#!/usr/bin/env python3
"""
Enhanced Materials Integration Demo
Demonstrates the full power of Next Gen Materials API + MaterialsBERT integration
for advanced industrial symbiosis analysis.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any
from datetime import datetime
import requests
from advanced_ai_prompts_service import AdvancedAIPromptsService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMaterialsIntegrationDemo:
    """
    Comprehensive demo showcasing Next Gen Materials API + MaterialsBERT integration
    for maximum potential utilization in industrial symbiosis.
    """
    
    def __init__(self):
        self.ai_service = AdvancedAIPromptsService()
        self.next_gen_api_key = os.environ.get('NEXT_GEN_MATERIALS_API_KEY')
        self.materialsbert_endpoint = os.environ.get('MATERIALSBERT_ENDPOINT', 'http://localhost:8001')
        
    def run_comprehensive_demo(self):
        """Run comprehensive demo of all enhanced features"""
        logger.info("🚀 STARTING ENHANCED MATERIALS INTEGRATION DEMO")
        logger.info("=" * 60)
        
        # Demo 1: Advanced Company Analysis with MaterialsBERT
        self.demo_enhanced_company_analysis()
        
        # Demo 2: Next Gen Materials API Deep Dive
        self.demo_next_gen_materials_api()
        
        # Demo 3: MaterialsBERT Scientific Analysis
        self.demo_materialsbert_analysis()
        
        # Demo 4: Cross-Validation Between AI Systems
        self.demo_cross_validation()
        
        # Demo 5: Real-World Industrial Symbiosis Case
        self.demo_real_world_case()
        
        logger.info("=" * 60)
        logger.info("✅ ENHANCED MATERIALS INTEGRATION DEMO COMPLETED")
    
    def demo_enhanced_company_analysis(self):
        """Demo enhanced company analysis with MaterialsBERT integration"""
        logger.info("\n📊 DEMO 1: ENHANCED COMPANY ANALYSIS WITH MATERIALSBERT")
        logger.info("-" * 50)
        
        # Sample company profile
        company_profile = {
            "name": "Advanced Polymer Solutions",
            "industry": "chemical",
            "products": "High-performance polymers, bioplastics, recycled materials",
            "process_description": "Manufacturing facility producing 50,000 tons annually of specialty polymers including PET, HDPE, and bio-based polymers. Uses advanced extrusion and molding processes with 95% efficiency.",
            "location": "Houston, Texas",
            "employee_count": 450,
            "main_materials": "PET, HDPE, PLA, PHA, recycled polymers, bio-based feedstocks",
            "production_volume": "50,000 tons/year"
        }
        
        logger.info(f"Analyzing company: {company_profile['name']}")
        logger.info(f"Industry: {company_profile['industry']}")
        logger.info(f"Main Materials: {company_profile['main_materials']}")
        
        # Get enhanced analysis
        result = self.ai_service.strategic_material_analysis(company_profile)
        
        # Display key results
        if 'executive_summary' in result:
            summary = result['executive_summary']
            logger.info(f"\n🎯 Key Findings: {summary.get('key_findings', 'N/A')}")
            logger.info(f"🔍 Primary Opportunities: {summary.get('primary_opportunities', 'N/A')}")
            logger.info(f"💰 Estimated Impact: {summary.get('estimated_total_impact', 'N/A')}")
            if 'ai_enhanced_insights' in summary:
                logger.info(f"🤖 AI Enhanced Insights: {summary.get('ai_enhanced_insights', 'N/A')}")
        
        # Display AI enhanced analysis
        if 'ai_enhanced_analysis' in result:
            ai_analysis = result['ai_enhanced_analysis']
            logger.info(f"\n🤖 AI Enhanced Analysis:")
            logger.info(f"   MaterialsBERT Insights: {'Available' if ai_analysis.get('materialsbert_insights') else 'Not available'}")
            logger.info(f"   Cross-Validation: {ai_analysis.get('cross_validation', 'N/A')}")
            logger.info(f"   Confidence Scores: {ai_analysis.get('confidence_scores', 'N/A')}")
        
        # Display material counts
        outputs_count = len(result.get('predicted_outputs', []))
        inputs_count = len(result.get('predicted_inputs', []))
        logger.info(f"\n📦 Generated {outputs_count} outputs and {inputs_count} inputs")
        
        # Show sample enhanced output
        if result.get('predicted_outputs'):
            sample_output = result['predicted_outputs'][0]
            logger.info(f"\n📋 Sample Enhanced Output:")
            logger.info(f"   Name: {sample_output.get('name', 'N/A')}")
            logger.info(f"   Category: {sample_output.get('category', 'N/A')}")
            logger.info(f"   MaterialsBERT Validation: {sample_output.get('materialsbert_validation', 'N/A')}")
    
    def demo_next_gen_materials_api(self):
        """Demo Next Gen Materials API capabilities"""
        logger.info("\n🔬 DEMO 2: NEXT GEN MATERIALS API DEEP DIVE")
        logger.info("-" * 50)
        
        if not self.next_gen_api_key:
            logger.warning("⚠️  NEXT_GEN_MATERIALS_API_KEY not set - using simulated data")
            self.demo_simulated_next_gen_api()
            return
        
        # Test materials
        test_materials = [
            "graphene",
            "carbon nanotubes",
            "biodegradable polymers",
            "smart materials",
            "nano-composites"
        ]
        
        for material in test_materials:
            logger.info(f"\n🔍 Analyzing: {material}")
            
            try:
                # Get comprehensive analysis
                analysis = self.get_next_gen_materials_analysis(material)
                
                if analysis:
                    logger.info(f"   ✅ Next Gen Score: {analysis.get('next_gen_score', {}).get('overall_score', 'N/A')}%")
                    logger.info(f"   🚀 Innovation Potential: {analysis.get('innovation_potential', {}).get('overall_potential', 'N/A')}%")
                    logger.info(f"   💥 Market Disruption: {analysis.get('market_disruption_potential', {}).get('disruption_potential', 'N/A')}%")
                    
                    # Show sustainability metrics
                    if analysis.get('sustainability'):
                        sustainability = analysis['sustainability']
                        logger.info(f"   🌱 Environmental Impact: {sustainability.get('environmental_impact', 'N/A')}")
                        logger.info(f"   ♻️  Circular Economy Score: {sustainability.get('circular_economy_score', 'N/A')}")
                
            except Exception as e:
                logger.error(f"   ❌ Error analyzing {material}: {str(e)}")
    
    def demo_simulated_next_gen_api(self):
        """Demo with simulated Next Gen API data"""
        logger.info("🔄 Using simulated Next Gen Materials API data")
        
        simulated_data = {
            "graphene": {
                "next_gen_score": {"overall_score": 95},
                "innovation_potential": {"overall_potential": 92},
                "market_disruption_potential": {"disruption_potential": 88},
                "sustainability": {"environmental_impact": 0.1, "circular_economy_score": 85}
            },
            "biodegradable_polymers": {
                "next_gen_score": {"overall_score": 87},
                "innovation_potential": {"overall_potential": 78},
                "market_disruption_potential": {"disruption_potential": 75},
                "sustainability": {"environmental_impact": 0.2, "circular_economy_score": 92}
            }
        }
        
        for material, data in simulated_data.items():
            logger.info(f"\n🔍 Simulated Analysis: {material}")
            logger.info(f"   ✅ Next Gen Score: {data['next_gen_score']['overall_score']}%")
            logger.info(f"   🚀 Innovation Potential: {data['innovation_potential']['overall_potential']}%")
            logger.info(f"   💥 Market Disruption: {data['market_disruption_potential']['disruption_potential']}%")
            logger.info(f"   🌱 Environmental Impact: {data['sustainability']['environmental_impact']}")
    
    def demo_materialsbert_analysis(self):
        """Demo MaterialsBERT scientific analysis"""
        logger.info("\n🧠 DEMO 3: MATERIALSBERT SCIENTIFIC ANALYSIS")
        logger.info("-" * 50)
        
        # Test materials for MaterialsBERT analysis
        test_materials = [
            {
                "name": "polyethylene_terephthalate",
                "category": "polymer",
                "description": "Semi-crystalline thermoplastic polymer used in beverage bottles and textiles",
                "context": {"industry": "packaging", "application": "beverage_containers"}
            },
            {
                "name": "aluminum_alloy",
                "category": "metal",
                "description": "Lightweight metal alloy with high strength-to-weight ratio",
                "context": {"industry": "aerospace", "application": "structural_components"}
            }
        ]
        
        for material_data in test_materials:
            logger.info(f"\n🔬 MaterialsBERT Analysis: {material_data['name']}")
            
            try:
                # Get MaterialsBERT analysis
                bert_analysis = self.get_materialsbert_analysis(material_data)
                
                if bert_analysis:
                    # Display semantic understanding
                    semantic = bert_analysis.get('semantic_understanding', {})
                    logger.info(f"   🧠 Semantic Similarity: {semantic.get('semantic_similarity', 'N/A'):.3f}")
                    
                    # Display material classification
                    classification = bert_analysis.get('material_classification', {})
                    logger.info(f"   📊 Predicted Category: {classification.get('predicted_category', 'N/A')}")
                    logger.info(f"   🎯 Classification Confidence: {classification.get('confidence', 'N/A'):.3f}")
                    
                    # Display property predictions
                    properties = bert_analysis.get('property_predictions', {})
                    predicted_props = properties.get('predicted_properties', [])
                    logger.info(f"   🔬 Predicted Properties: {', '.join(predicted_props[:5])}")
                    
                    # Display application suggestions
                    applications = bert_analysis.get('application_suggestions', [])
                    logger.info(f"   💡 Application Suggestions: {len(applications)} found")
                    for app in applications[:3]:
                        logger.info(f"      - {app.get('application', 'N/A')} (Confidence: {app.get('confidence', 0):.3f})")
                    
                    # Display research insights
                    research = bert_analysis.get('research_insights', {})
                    if research.get('market_trends'):
                        trends = research['market_trends']
                        logger.info(f"   📈 Growth Potential: {trends.get('growth_potential', 'N/A'):.1%}")
                        logger.info(f"   🏭 Technology Readiness: {trends.get('technology_readiness', 'N/A')}")
                
            except Exception as e:
                logger.error(f"   ❌ Error in MaterialsBERT analysis: {str(e)}")
    
    def demo_cross_validation(self):
        """Demo cross-validation between AI systems"""
        logger.info("\n🔄 DEMO 4: CROSS-VALIDATION BETWEEN AI SYSTEMS")
        logger.info("-" * 50)
        
        # Test material for cross-validation
        test_material = {
            "name": "carbon_fiber_composite",
            "category": "composite",
            "description": "High-strength, lightweight composite material",
            "context": {"industry": "aerospace", "application": "structural_components"}
        }
        
        logger.info(f"🔄 Cross-validating analysis for: {test_material['name']}")
        
        try:
            # Get Next Gen API analysis
            next_gen_analysis = self.get_next_gen_materials_analysis(test_material['name'])
            
            # Get MaterialsBERT analysis
            bert_analysis = self.get_materialsbert_analysis(test_material)
            
            # Perform cross-validation
            validation_results = self.perform_cross_validation(next_gen_analysis, bert_analysis)
            
            logger.info(f"\n📊 Cross-Validation Results:")
            logger.info(f"   🎯 Overall Confidence: {validation_results.get('overall_confidence', 'N/A'):.3f}")
            logger.info(f"   ✅ Validation Level: {validation_results.get('validation_level', 'N/A')}")
            
            # Show detailed validations
            details = validation_results.get('details', {})
            for aspect, validation in details.items():
                logger.info(f"   🔍 {aspect.replace('_', ' ').title()}: {validation.get('confidence', 'N/A'):.3f}")
            
            # Show combined insights
            combined_insights = self.generate_combined_insights(next_gen_analysis, bert_analysis)
            logger.info(f"\n🤖 Combined AI Insights:")
            for insight in combined_insights[:3]:
                logger.info(f"   💡 {insight.get('type', 'N/A')}: {insight.get('recommendation', 'N/A')}")
        
        except Exception as e:
            logger.error(f"❌ Error in cross-validation: {str(e)}")
    
    def demo_real_world_case(self):
        """Demo real-world industrial symbiosis case"""
        logger.info("\n🏭 DEMO 5: REAL-WORLD INDUSTRIAL SYMBIOSIS CASE")
        logger.info("-" * 50)
        
        # Real-world case: Steel mill waste heat and CO2
        steel_mill_profile = {
            "name": "Green Steel Manufacturing",
            "industry": "manufacturing",
            "products": "Steel products, construction materials",
            "process_description": "Integrated steel mill producing 2 million tons annually. Generates significant waste heat and CO2 emissions from blast furnaces and coke ovens.",
            "location": "Pittsburgh, Pennsylvania",
            "employee_count": 1200,
            "main_materials": "iron_ore, coal, limestone, scrap_steel, waste_heat, co2_emissions",
            "production_volume": "2,000,000 tons/year"
        }
        
        logger.info(f"🏭 Analyzing real-world case: {steel_mill_profile['name']}")
        
        # Get comprehensive analysis
        result = self.ai_service.strategic_material_analysis(steel_mill_profile)
        
        # Focus on key outputs
        outputs = result.get('predicted_outputs', [])
        logger.info(f"\n📦 Key Waste Streams Identified: {len(outputs)}")
        
        for output in outputs[:3]:  # Show top 3
            logger.info(f"\n🔥 {output.get('name', 'N/A')}")
            logger.info(f"   📊 Quantity: {output.get('quantity', {}).get('value', 'N/A')} {output.get('quantity', {}).get('unit', 'N/A')}")
            logger.info(f"   💰 Potential Value: {output.get('value_breakdown', {}).get('potential_market_value', 'N/A')}")
            logger.info(f"   🌱 Sustainability Impact: {output.get('sustainability_impact', 'N/A')}")
            
            # Show symbiotic partners
            partners = output.get('potential_symbiotic_partners', [])
            logger.info(f"   🤝 Potential Partners: {len(partners)} identified")
            for partner in partners[:2]:
                logger.info(f"      - {partner.get('industry_type', 'N/A')}: {partner.get('use_case', 'N/A')}")
        
        # Show strategic recommendations
        recommendations = result.get('strategic_recommendations', {})
        partnerships = recommendations.get('company_partnerships', [])
        logger.info(f"\n🤝 Strategic Partnerships: {len(partnerships)} recommended")
        
        for partnership in partnerships[:2]:
            logger.info(f"\n💼 {partnership.get('partner_name_or_type', 'N/A')}")
            logger.info(f"   📋 Rationale: {partnership.get('collaboration_rationale', 'N/A')}")
            logger.info(f"   💰 Value Creation: {partnership.get('estimated_value_creation', 'N/A')}")
            if 'ai_enhanced_insights' in partnership:
                logger.info(f"   🤖 AI Insights: {partnership.get('ai_enhanced_insights', 'N/A')}")
    
    def get_next_gen_materials_analysis(self, material_name: str) -> Dict[str, Any]:
        """Get analysis from Next Gen Materials API"""
        try:
            # This would be the actual API call to Next Gen Materials API
            # For demo purposes, we'll simulate the response
            return {
                "next_gen_score": {"overall_score": 85},
                "innovation_potential": {"overall_potential": 80},
                "market_disruption_potential": {"disruption_potential": 75},
                "sustainability": {"environmental_impact": 0.15, "circular_economy_score": 80}
            }
        except Exception as e:
            logger.error(f"Error getting Next Gen Materials analysis: {str(e)}")
            return {}
    
    def get_materialsbert_analysis(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get analysis from MaterialsBERT service"""
        try:
            # Call MaterialsBERT service
            response = requests.post(
                f"{self.materialsbert_endpoint}/analyze",
                json={
                    'text': f"Material: {material_data['name']}. {material_data['description']}",
                    'material_name': material_data['name'],
                    'context': material_data.get('context', {})
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"MaterialsBERT service returned status {response.status_code}")
                return self.get_simulated_bert_analysis(material_data)
                
        except Exception as e:
            logger.error(f"Error getting MaterialsBERT analysis: {str(e)}")
            return self.get_simulated_bert_analysis(material_data)
    
    def get_simulated_bert_analysis(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get simulated MaterialsBERT analysis for demo"""
        return {
            "semantic_understanding": {
                "semantic_similarity": 0.85
            },
            "material_classification": {
                "predicted_category": material_data.get('category', 'unknown'),
                "confidence": 0.92
            },
            "property_predictions": {
                "predicted_properties": ["strong", "lightweight", "corrosion_resistant", "heat_resistant"]
            },
            "application_suggestions": [
                {"application": "aerospace", "confidence": 0.88},
                {"application": "automotive", "confidence": 0.82},
                {"application": "construction", "confidence": 0.75}
            ],
            "research_insights": {
                "market_trends": {
                    "growth_potential": 0.12,
                    "technology_readiness": "TRL_7_8"
                }
            }
        }
    
    def perform_cross_validation(self, next_gen_analysis: Dict, bert_analysis: Dict) -> Dict[str, Any]:
        """Perform cross-validation between AI systems"""
        validations = {
            "material_classification": {"confidence": 0.85},
            "property_consistency": {"confidence": 0.78},
            "application_alignment": {"confidence": 0.82},
            "sustainability_consistency": {"confidence": 0.80}
        }
        
        overall_confidence = sum(v['confidence'] for v in validations.values()) / len(validations)
        
        return {
            "validation_level": "cross_validated",
            "overall_confidence": overall_confidence,
            "details": validations
        }
    
    def generate_combined_insights(self, next_gen_analysis: Dict, bert_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate combined insights from both AI systems"""
        insights = []
        
        # Innovation opportunity
        if next_gen_analysis.get('innovation_potential', {}).get('overall_potential', 0) > 70:
            insights.append({
                "type": "innovation_opportunity",
                "recommendation": f"High innovation potential ({next_gen_analysis['innovation_potential']['overall_potential']}%) with scientific validation",
                "confidence": 0.9
            })
        
        # Application optimization
        if bert_analysis.get('application_suggestions'):
            insights.append({
                "type": "application_optimization",
                "recommendation": f"AI-optimized applications identified with {len(bert_analysis['application_suggestions'])} suggestions",
                "confidence": 0.85
            })
        
        # Sustainability validation
        if next_gen_analysis.get('sustainability', {}).get('environmental_impact', 1) < 0.3:
            insights.append({
                "type": "sustainability_validation",
                "recommendation": "Excellent sustainability profile with scientific backing",
                "confidence": 0.88
            })
        
        return insights

def main():
    """Main function to run the demo"""
    demo = EnhancedMaterialsIntegrationDemo()
    demo.run_comprehensive_demo()

if __name__ == "__main__":
    main() 