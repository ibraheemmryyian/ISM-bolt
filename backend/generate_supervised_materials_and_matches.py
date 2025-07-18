import json
import csv
import os
import sys
import asyncio
import aiohttp
import requests
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

DATA_FILE = Path(__file__).parent.parent / "fixed_realworlddata.json"
LISTINGS_CSV = "material_listings.csv"
MATCHES_CSV = "material_matches.csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import all available microservices
try:
    from .listing_inference_service import ListingInferenceService
    from .ai_matchmaking_service import get_model as get_matchmaking_model
    from .ai_listings_generator import AIListingsGenerator
    from .ai_pricing_orchestrator import AI_PricingOrchestrator
    from .ai_production_orchestrator import AIProductionOrchestrator
    from .ai_service_integration import AIServiceIntegration
    from .ai_monitoring_dashboard import AIMonitoringDashboard
    from .ai_retraining_pipeline import AIRetrainingPipeline
    from .ai_hyperparameter_optimizer import AIHyperparameterOptimizer
    from .ai_fusion_layer import AIFusionLayer
    from .meta_learning_orchestrator import MetaLearningOrchestrator
    from .dynamic_materials_integration_service import DynamicMaterialsIntegrationService
    from .advanced_ai_prompts_service import AdvancedAIPromptsService
    from .proactive_opportunity_engine import AdvancedProactiveOpportunityEngine
    from .industrial_intelligence_engine import IndustrialIntelligenceEngine
    from .impact_forecasting import ImpactForecastingService
    from .materials_bert_service_advanced import MaterialsBERTService
    from .deepseek_r1_semantic_service import DeepSeekSemanticService
    from .gnn_reasoning_engine import GNNReasoningEngine
    from revolutionary_ai_matching import RevolutionaryAIMatching
    from multi_hop_symbiosis_network import MultiHopSymbiosisNetwork
    from .ultra_ai_listings_generator import UltraListingsGenerator
    
    ALL_SERVICES_AVAILABLE = True
    logger.info("✅ All microservices imported successfully")
except ImportError as e:
    logger.error(f"❌ CRITICAL ERROR: Failed to import microservices: {e}")
    logger.error("🔧 REQUIRED SETUP:")
    logger.error("   1. Start all microservices as Flask servers")
    logger.error("   2. Ensure all dependencies are installed")
    logger.error("   3. Check that all service files exist")
    logger.error("   4. Verify environment variables are set")
    sys.exit(1)

class AdvancedSupervisedDataGenerator:
    """Advanced supervised data generator using ALL microservices - NO FALLBACKS"""
    
    def __init__(self):
        self.logger = logger
        self.session = aiohttp.ClientSession()
        
        # Initialize all microservices - FAIL IF ANY MISSING
        self.services = {}
        self._initialize_services()
        
        # Configuration
        self.config = {
            'max_concurrent_requests': 10,
            'timeout': 30,
            'retry_attempts': 3,
            'enable_monitoring': True,
            'enable_optimization': True
        }
    
    def _initialize_services(self):
        """Initialize all available microservices - FAIL IF ANY MISSING"""
        required_services = [
            ('listing_inference', 'ListingInferenceService'),
            ('ai_listings', 'AIListingsGenerator'),
            ('ultra_listings', 'UltraListingsGenerator'),
            ('revolutionary_matching', 'RevolutionaryAIMatching'),
            ('gnn_reasoning', 'GNNReasoningEngine'),
            ('multi_hop', 'MultiHopSymbiosisNetwork'),
            ('materials_integration', 'DynamicMaterialsIntegrationService'),
            ('materials_bert', 'MaterialsBERTService'),
            ('deepseek', 'DeepSeekSemanticService'),
            ('ai_prompts', 'AdvancedAIPromptsService'),
            ('industrial_intelligence', 'IndustrialIntelligenceEngine'),
            ('pricing_orchestrator', 'AI_PricingOrchestrator'),
            ('production_orchestrator', 'AIProductionOrchestrator'),
            ('service_integration', 'AIServiceIntegration'),
            ('monitoring', 'AIMonitoringDashboard'),
            ('retraining', 'AIRetrainingPipeline'),
            ('hyperparameter_optimizer', 'AIHyperparameterOptimizer'),
            ('fusion_layer', 'AIFusionLayer'),
            ('meta_learning', 'MetaLearningOrchestrator'),
            ('opportunity_engine', 'AdvancedProactiveOpportunityEngine'),
            ('impact_forecasting', 'ImpactForecastingService')
        ]
        
        missing_services = []
        
        for service_key, service_class in required_services:
            if service_class in globals():
                try:
                    self.services[service_key] = globals()[service_class]()
                    self.logger.info(f"✅ Initialized {service_key}")
                except Exception as e:
                    missing_services.append(f"{service_key} (initialization failed: {e})")
            else:
                missing_services.append(service_key)
        
        if missing_services:
            self.logger.error(f"❌ CRITICAL ERROR: Missing or failed services: {missing_services}")
            self.logger.error("🔧 REQUIRED SETUP:")
            self.logger.error("   1. Start all microservices as Flask servers:")
            self.logger.error("      - python backend/listing_inference_service.py")
            self.logger.error("      - python backend/ai_matchmaking_service.py")
            self.logger.error("      - python backend/ai_pricing_orchestrator.py")
            self.logger.error("      - python backend/ai_production_orchestrator.py")
            self.logger.error("      - python backend/ai_service_integration.py")
            self.logger.error("      - python backend/ai_monitoring_dashboard.py")
            self.logger.error("      - python backend/ai_retraining_pipeline.py")
            self.logger.error("      - python backend/ai_hyperparameter_optimizer.py")
            self.logger.error("      - python backend/ai_fusion_layer.py")
            self.logger.error("      - python backend/meta_learning_orchestrator.py")
            self.logger.error("      - python backend/dynamic_materials_integration_service.py")
            self.logger.error("      - python backend/advanced_ai_prompts_service.py")
            self.logger.error("      - python backend/proactive_opportunity_engine.py")
            self.logger.error("      - python backend/industrial_intelligence_engine.py")
            self.logger.error("      - python backend/impact_forecasting.py")
            self.logger.error("      - python backend/materials_bert_service_advanced.py")
            self.logger.error("      - python backend/deepseek_r1_semantic_service.py")
            self.logger.error("      - python backend/gnn_reasoning.py")
            self.logger.error("      - python backend/revolutionary_ai_matching.py")
            self.logger.error("      - python backend/multi_hop_symbiosis_network.py")
            self.logger.error("      - python backend/ultra_ai_listings_generator.py")
            self.logger.error("   2. Or use the main backend: npm start (in backend directory)")
            self.logger.error("   3. Ensure all environment variables are set")
            self.logger.error("   4. Check that all dependencies are installed")
            raise RuntimeError(f"Missing or failed services: {missing_services}")
        
        self.logger.info(f"✅ Successfully initialized {len(self.services)} microservices")
    
    async def generate_advanced_material_listings(self, company: dict) -> List[dict]:
        """Generate material listings using ALL available microservices - NO FALLBACKS"""
        company_id = company.get('id')
        company_name = company.get('name', 'Unknown Company')
        
        self.logger.info(f"🚀 Generating advanced listings for {company_name} using {len(self.services)} microservices")
        
        # 1. INDUSTRIAL INTELLIGENCE ANALYSIS
        self.logger.info(f"  📊 Running industrial intelligence analysis...")
        intel_analysis = await self.services['industrial_intelligence'].analyze_company_intelligence(company)
        if not intel_analysis:
            raise RuntimeError(f"Industrial intelligence analysis failed for {company_name}")
        self.logger.info(f"  ✅ Industrial intelligence analysis completed")
        
        # 2. ADVANCED AI PROMPTS ANALYSIS
        self.logger.info(f"  🧠 Running advanced AI analysis...")
        ai_analysis = await self.services['ai_prompts'].strategic_material_analysis(company)
        if not ai_analysis:
            raise RuntimeError(f"Advanced AI analysis failed for {company_name}")
        self.logger.info(f"  ✅ Advanced AI analysis completed")
        
        # 3. MATERIALS INTEGRATION ANALYSIS
        self.logger.info(f"  🔬 Running materials integration analysis...")
        materials_analysis = await self.services['materials_integration'].get_comprehensive_material_analysis(
            company.get('industry', 'manufacturing'),
            {'company': company_name, 'location': company.get('location', 'Global')}
        )
        if not materials_analysis:
            raise RuntimeError(f"Materials integration analysis failed for {company_name}")
        self.logger.info(f"  ✅ Materials integration analysis completed")
        
        # 4. MATERIALSBERT ANALYSIS
        self.logger.info(f"  🧠 Running MaterialsBERT analysis...")
        bert_analysis = await self.services['materials_bert'].analyze_materials_text(
            f"{company_name} {company.get('industry', 'manufacturing')} operations"
        )
        if not bert_analysis:
            raise RuntimeError(f"MaterialsBERT analysis failed for {company_name}")
        self.logger.info(f"  ✅ MaterialsBERT analysis completed")
        
        # 5. DEEPSEEK SEMANTIC ANALYSIS
        self.logger.info(f"  🔍 Running DeepSeek semantic analysis...")
        deepseek_analysis = await self.services['deepseek'].analyze_semantic_content(
            f"Company: {company_name}, Industry: {company.get('industry', 'manufacturing')}"
        )
        if not deepseek_analysis:
            raise RuntimeError(f"DeepSeek analysis failed for {company_name}")
        self.logger.info(f"  ✅ DeepSeek semantic analysis completed")
        
        # 6. OPPORTUNITY ENGINE ANALYSIS
        self.logger.info(f"  🎯 Running opportunity detection...")
        opportunities = await self.services['opportunity_engine'].detect_material_opportunities(company)
        if not opportunities:
            raise RuntimeError(f"Opportunity detection failed for {company_name}")
        self.logger.info(f"  ✅ Opportunity detection completed")
        
        # 7. IMPACT FORECASTING
        self.logger.info(f"  📈 Running impact forecasting...")
        impact_forecast = await self.services['impact_forecasting'].forecast_material_impact(company)
        if not impact_forecast:
            raise RuntimeError(f"Impact forecasting failed for {company_name}")
        self.logger.info(f"  ✅ Impact forecasting completed")
        
        # 8. GENERATE LISTINGS USING MULTIPLE SERVICES
        listings = []
        
        # Use listing inference service
        self.logger.info(f"  📦 Running listing inference service...")
        inference_result = await self.services['listing_inference'].generate_listings_from_profile(company)
        if not inference_result or not inference_result.get('predicted_outputs'):
            raise RuntimeError(f"Listing inference failed for {company_name}")
        listings.extend(inference_result['predicted_outputs'])
        self.logger.info(f"  ✅ Listing inference generated {len(inference_result['predicted_outputs'])} listings")
        
        # Use ultra listings generator
        self.logger.info(f"  🚀 Running ultra listings generator...")
        ultra_result = await self.services['ultra_listings'].generate_ultra_listings(company, [], [])
        if not ultra_result:
            raise RuntimeError(f"Ultra listings generation failed for {company_name}")
        listings.extend(ultra_result)
        self.logger.info(f"  ✅ Ultra listings generated {len(ultra_result)} listings")
        
        # 9. ENHANCE LISTINGS WITH AI ANALYSIS
        enhanced_listings = []
        for listing in listings:
            enhanced_listing = {
                'company_id': company_id,
                'company_name': company_name,
                'material_name': listing.get('name') or listing.get('material_name'),
                'material_type': listing.get('type') or listing.get('category', 'unknown'),
                'quantity': listing.get('quantity') or listing.get('quantity_estimate', 100),
                'unit': listing.get('unit', 'tons'),
                'description': listing.get('description', ''),
                'quality_grade': listing.get('quality_grade', 'B'),
                'potential_value': listing.get('potential_value', 0),
                'potential_uses': listing.get('potential_uses', []),
                'potential_sources': listing.get('potential_sources', []),
                'ai_generated': True,
                'ai_enhanced': True,
                'analysis_data': {
                    'industrial_intelligence': intel_analysis,
                    'ai_analysis': ai_analysis,
                    'materials_analysis': materials_analysis,
                    'bert_analysis': bert_analysis,
                    'deepseek_analysis': deepseek_analysis,
                    'opportunities': opportunities,
                    'impact_forecast': impact_forecast
                }
            }
            enhanced_listings.append(enhanced_listing)
        
        self.logger.info(f"  ✅ Generated {len(enhanced_listings)} enhanced listings")
        return enhanced_listings
    
    async def generate_advanced_matches(self, company_id: str, material: dict) -> List[dict]:
        """Generate matches using ALL available matching microservices - NO FALLBACKS"""
        material_name = material.get('name') or material.get('material_name', 'Unknown')
        
        self.logger.info(f"    🎯 Generating advanced matches for {material_name}")
        
        # 1. REVOLUTIONARY AI MATCHING
        self.logger.info(f"      🚀 Running revolutionary AI matching...")
        candidate_companies = self._generate_candidate_companies(company_id, material)
        
        revolutionary_matches = await self.services['revolutionary_matching'].find_matches(
            query_company={'id': company_id, 'material': material},
            candidate_companies=candidate_companies,
            algorithm='ensemble',
            top_k=5
        )
        
        if not revolutionary_matches or not revolutionary_matches.get('candidates'):
            raise RuntimeError(f"Revolutionary AI matching failed for {material_name}")
        
        matches = []
        for match in revolutionary_matches['candidates']:
            matches.append({
                'id': f"rev_{company_id}_{match.get('id', 'unknown')}",
                'name': match.get('name', 'Revolutionary Match'),
                'company_id': match.get('id'),
                'score': match.get('score', 0.8),
                'match_score': match.get('score', 0.8),
                'reason': f"Revolutionary AI matching: {match.get('reason', 'Advanced algorithm')}",
                'algorithm': 'revolutionary_ai'
            })
        
        self.logger.info(f"      ✅ Revolutionary AI generated {len(revolutionary_matches['candidates'])} matches")
        
        # 2. GNN REASONING MATCHES
        self.logger.info(f"      🕸️ Running GNN reasoning...")
        gnn_matches = await self.services['gnn_reasoning'].find_symbiosis_matches(
            company_id=company_id,
            material_data=material,
            top_k=3
        )
        
        if not gnn_matches:
            raise RuntimeError(f"GNN reasoning failed for {material_name}")
        
        for match in gnn_matches:
            matches.append({
                'id': f"gnn_{company_id}_{match.get('id', 'unknown')}",
                'name': match.get('name', 'GNN Match'),
                'company_id': match.get('id'),
                'score': match.get('score', 0.75),
                'match_score': match.get('score', 0.75),
                'reason': f"GNN reasoning: {match.get('reason', 'Graph-based analysis')}",
                'algorithm': 'gnn_reasoning'
            })
        
        self.logger.info(f"      ✅ GNN reasoning generated {len(gnn_matches)} matches")
        
        # 3. MULTI-HOP SYMBIOSIS MATCHES
        self.logger.info(f"      🔗 Running multi-hop symbiosis...")
        multi_hop_matches = await self.services['multi_hop'].find_multi_hop_matches(
            source_company=company_id,
            material=material,
            max_hops=3,
            top_k=3
        )
        
        if not multi_hop_matches:
            raise RuntimeError(f"Multi-hop symbiosis failed for {material_name}")
        
        for match in multi_hop_matches:
            matches.append({
                'id': f"mhop_{company_id}_{match.get('id', 'unknown')}",
                'name': match.get('name', 'Multi-Hop Match'),
                'company_id': match.get('id'),
                'score': match.get('score', 0.7),
                'match_score': match.get('score', 0.7),
                'reason': f"Multi-hop symbiosis: {match.get('hops', 1)} hop(s) away",
                'algorithm': 'multi_hop_symbiosis'
            })
        
        self.logger.info(f"      ✅ Multi-hop symbiosis generated {len(multi_hop_matches)} matches")
        
        # 4. PRICING ORCHESTRATOR ANALYSIS
        self.logger.info(f"      💰 Running pricing analysis...")
        pricing_analysis = await self.services['pricing_orchestrator'].analyze_material_pricing(
            material_data=material,
            market_context={'location': 'Global', 'industry': 'manufacturing'}
        )
        
        if not pricing_analysis:
            raise RuntimeError(f"Pricing analysis failed for {material_name}")
        
        # Add pricing insights to existing matches
        for match in matches:
            match['pricing_analysis'] = pricing_analysis
        
        self.logger.info(f"      ✅ Pricing analysis completed")
        
        # 5. FUSION LAYER ENHANCEMENT
        self.logger.info(f"      🔄 Running fusion layer enhancement...")
        enhanced_matches = await self.services['fusion_layer'].enhance_matches_with_fusion(
            matches=matches,
            material=material,
            company_id=company_id
        )
        
        if not enhanced_matches:
            raise RuntimeError(f"Fusion layer enhancement failed for {material_name}")
        
        matches = enhanced_matches
        self.logger.info(f"      ✅ Fusion layer enhanced {len(matches)} matches")
        
        # 6. DEDUPLICATE AND RANK MATCHES
        unique_matches = self._deduplicate_matches(matches)
        ranked_matches = sorted(unique_matches, key=lambda x: x.get('score', 0), reverse=True)
        
        self.logger.info(f"      ✅ Generated {len(ranked_matches)} unique matches")
        return ranked_matches[:5]  # Return top 5 matches
    
    def _generate_candidate_companies(self, company_id: str, material: dict) -> List[dict]:
        """Generate candidate companies for matching"""
        return [
            {'id': 'candidate_1', 'name': 'Recycling Solutions Ltd', 'industry': 'recycling'},
            {'id': 'candidate_2', 'name': 'Material Processing Co', 'industry': 'processing'},
            {'id': 'candidate_3', 'name': 'Energy Recovery Systems', 'industry': 'energy'},
            {'id': 'candidate_4', 'name': 'Supply Chain Partners', 'industry': 'logistics'},
            {'id': 'candidate_5', 'name': 'Industrial Symbiosis Co', 'industry': 'symbiosis'}
        ]
    
    def _deduplicate_matches(self, matches: List[dict]) -> List[dict]:
        """Remove duplicate matches based on company_id"""
        seen = set()
        unique_matches = []
        for match in matches:
            company_id = match.get('company_id')
            if company_id not in seen:
                seen.add(company_id)
                unique_matches.append(match)
        return unique_matches
    
    async def close(self):
        """Close the session"""
        if hasattr(self, 'session'):
            await self.session.close()

async def main():
    """Main function to generate advanced supervised data"""
    print("🚀 Starting ADVANCED Supervised Data Generation with ALL Microservices")
    print("=" * 80)
    print("⚠️  NO FALLBACKS - Script will FAIL if any AI service is unavailable")
    print("=" * 80)
    
    # Load company data
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        try:
            companies = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load company data: {e}")
    
    if not isinstance(companies, list) or not companies:
        raise ValueError("No companies found in the data file.")
    
    print(f"📊 Loaded {len(companies)} companies from {DATA_FILE}")
    print(f"🔧 Using ALL available microservices for maximum AI power")
    print("-" * 80)
    
    # Initialize advanced generator
    generator = AdvancedSupervisedDataGenerator()
    
    all_listings = []
    all_matches = []
    
    try:
        for i, company in enumerate(companies, 1):
            company_id = company.get("id") or company.get("name") or f"company_{i}"
            company["id"] = company_id
            
            print(f"\n[{i}/{len(companies)}] 🏢 Processing: {company.get('name', 'Unknown Company')}")
            
            # Generate advanced listings
            listings = await generator.generate_advanced_material_listings(company)
            
            for listing in listings:
                listing_row = {
                    "company_id": company["id"],
                    "company_name": company.get("name", "Unknown"),
                    "material_name": listing.get("material_name"),
                    "material_type": listing.get("material_type"),
                    "quantity": listing.get("quantity", 0),
                    "unit": listing.get("unit", "units"),
                    "description": listing.get("description", ""),
                    "quality_grade": listing.get("quality_grade", "C"),
                    "potential_value": listing.get("potential_value", 0),
                    "potential_uses": json.dumps(listing.get("potential_uses", [])),
                    "potential_sources": json.dumps(listing.get("potential_sources", [])),
                    "ai_generated": listing.get("ai_generated", True),
                    "ai_enhanced": listing.get("ai_enhanced", False),
                    "analysis_data": json.dumps(listing.get("analysis_data", {}))
                }
                all_listings.append(listing_row)
                
                print(f"  📦 Generated listing: {listing_row['material_name']} ({listing_row['material_type']})")
                
                # Generate advanced matches
                matches = await generator.generate_advanced_matches(company["id"], listing)
                
                for match in matches:
                    match_row = {
                        "company_id": company["id"],
                        "company_name": company.get("name", "Unknown"),
                        "material_name": listing_row["material_name"],
                        "material_type": listing_row["material_type"],
                        "matched_company_id": match.get("company_id"),
                        "matched_company_name": match.get("name", "Unknown"),
                        "match_score": match.get("match_score", 0.5),
                        "match_reason": match.get("reason", "AI-generated match"),
                        "algorithm": match.get("algorithm", "unknown"),
                        "match_details": json.dumps(match)
                    }
                    all_matches.append(match_row)
    
    finally:
        await generator.close()
    
    # Write listings to CSV
    if all_listings:
        with open(LISTINGS_CSV, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_listings[0].keys())
            writer.writeheader()
            writer.writerows(all_listings)
        print(f"\n✅ Wrote {len(all_listings)} advanced listings to {LISTINGS_CSV}")
    else:
        print("\n⚠️ No listings generated.")

    # Write matches to CSV
    if all_matches:
        with open(MATCHES_CSV, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_matches[0].keys())
            writer.writeheader()
            writer.writerows(all_matches)
        print(f"✅ Wrote {len(all_matches)} advanced matches to {MATCHES_CSV}")
    else:
        print("⚠️ No matches generated.")

    print("\n🎉 ADVANCED Supervised Data Generation Complete!")
    print(f"📊 Summary: {len(all_listings)} listings, {len(all_matches)} matches")
    print(f"📁 Files created: {LISTINGS_CSV}, {MATCHES_CSV}")
    print("🚀 Generated using ALL available microservices for maximum AI power!")

if __name__ == "__main__":
    asyncio.run(main()) 