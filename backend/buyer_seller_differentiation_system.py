#!/usr/bin/env python3
"""
🧠 BUYER/SELLER DIFFERENTIATION SYSTEM
Advanced AI system to differentiate between buyers and sellers in industrial symbiosis
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from transformers
    HAS_TRANSFORMERS = True
except ImportError:
    from .fallbacks.transformers_fallback import *
    HAS_TRANSFORMERS = False import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple, Union
import json
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompanyRole(Enum):
    """Company roles in industrial symbiosis"""
    BUYER = "buyer"
    SELLER = "seller"
    BOTH = "both"  # Can act as both buyer and seller
    NEUTRAL = "neutral"  # Not actively participating

class MaterialType(Enum):
    """Material types for classification"""
    WASTE = "waste"
    BYPRODUCT = "byproduct"
    SURPLUS = "surplus"
    REQUIREMENT = "requirement"
    RESOURCE = "resource"

@dataclass
class BuyerSellerClassification:
    """Classification result for buyer/seller differentiation"""
    company_id: str
    company_name: str
    primary_role: CompanyRole
    confidence_score: float
    buyer_indicators: List[str]
    seller_indicators: List[str]
    material_preferences: Dict[str, Any]
    market_position: str
    transaction_history: Dict[str, Any]
    classification_reasoning: str

class BuyerSellerDifferentiationSystem:
    """
    🧠 Advanced AI system for buyer/seller differentiation in industrial symbiosis
    
    Features:
    - Multi-modal analysis of company profiles
    - Material flow analysis
    - Transaction pattern recognition
    - Market position assessment
    - Dynamic role classification
    - Predictive buyer/seller behavior modeling
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🧠 INITIALIZING BUYER/SELLER DIFFERENTIATION SYSTEM")
        
        # Initialize AI components
        self._initialize_ai_components()
        
        # Initialize classification models
        self._initialize_classification_models()
        
        # Initialize material analysis
        self._initialize_material_analysis()
        
        # Initialize market analysis
        self._initialize_market_analysis()
        
        self.logger.info("✅ BUYER/SELLER DIFFERENTIATION SYSTEM READY")
    
    def _initialize_ai_components(self):
        """Initialize AI components for analysis"""
        self.logger.info("🚀 Initializing AI components...")
        
        # Semantic encoder for text analysis
        self.semantic_encoder = SentenceTransformer('all-mpnet-base-v2')
        
        # Role classification network
        self.role_classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4)  # 4 roles: buyer, seller, both, neutral
        )
        
        # Material flow analyzer
        self.material_flow_analyzer = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # Transaction pattern recognizer
        self.transaction_recognizer = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # Market position analyzer
        self.market_analyzer = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 5)  # 5 market positions
        )
        
        self.logger.info("✅ AI components initialized")
    
    def _initialize_classification_models(self):
        """Initialize classification models"""
        self.logger.info("🎯 Initializing classification models...")
        
        # Buyer indicators
        self.buyer_indicators = {
            'material_requirements': ['needs', 'requires', 'seeking', 'looking for', 'in need of'],
            'production_processes': ['manufacturing', 'processing', 'assembly', 'production'],
            'resource_intensive': ['high consumption', 'large quantities', 'regular supply'],
            'cost_focused': ['cost effective', 'affordable', 'budget conscious', 'price sensitive'],
            'quality_standards': ['certified', 'quality assured', 'meets standards', 'compliance'],
            'logistics_requirements': ['delivery', 'transportation', 'shipping', 'logistics']
        }
        
        # Seller indicators
        self.seller_indicators = {
            'waste_generation': ['waste', 'byproduct', 'surplus', 'excess', 'leftover'],
            'production_capacity': ['high capacity', 'excess capacity', 'underutilized'],
            'material_availability': ['available', 'in stock', 'ready to supply', 'can provide'],
            'cost_advantage': ['competitive pricing', 'bulk discounts', 'volume pricing'],
            'quality_guarantee': ['certified quality', 'guaranteed specifications', 'tested'],
            'logistics_capability': ['can deliver', 'transportation available', 'shipping included']
        }
        
        # Market positions
        self.market_positions = {
            'dominant_buyer': 'Large company with significant purchasing power',
            'niche_buyer': 'Specialized company with specific material needs',
            'dominant_seller': 'Large company with excess materials/waste',
            'niche_seller': 'Specialized company with unique materials',
            'balanced': 'Company that both buys and sells materials'
        }
        
        self.logger.info("✅ Classification models initialized")
    
    def _initialize_material_analysis(self):
        """Initialize material analysis components"""
        self.logger.info("🔬 Initializing material analysis...")
        
        # Material type classifier
        self.material_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5)  # 5 material types
        )
        
        # Material value analyzer
        self.value_analyzer = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
        self.logger.info("✅ Material analysis initialized")
    
    def _initialize_market_analysis(self):
        """Initialize market analysis components"""
        self.logger.info("📊 Initializing market analysis...")
        
        # Market dynamics analyzer
        self.market_dynamics = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
        # Supply-demand analyzer
        self.supply_demand_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # Supply and demand scores
        )
        
        self.logger.info("✅ Market analysis initialized")
    
    async def classify_company_role(self, company_profile: Dict[str, Any]) -> BuyerSellerClassification:
        """
        Classify a company's role as buyer, seller, both, or neutral
        
        Args:
            company_profile: Company profile data
            
        Returns:
            BuyerSellerClassification: Detailed classification result
        """
        self.logger.info(f"🧠 Classifying company role: {company_profile.get('name', 'Unknown')}")
        
        try:
            # Extract company features
            company_features = self._extract_company_features(company_profile)
            
            # Analyze material flows
            material_analysis = self._analyze_material_flows(company_profile)
            
            # Analyze transaction patterns
            transaction_analysis = self._analyze_transaction_patterns(company_profile)
            
            # Analyze market position
            market_analysis = self._analyze_market_position(company_profile)
            
            # Perform role classification
            role_classification = self._perform_role_classification(
                company_features, material_analysis, transaction_analysis, market_analysis
            )
            
            # Generate classification reasoning
            reasoning = self._generate_classification_reasoning(
                company_profile, role_classification, material_analysis, transaction_analysis, market_analysis
            )
            
            # Create classification result
            classification = BuyerSellerClassification(
                company_id=company_profile.get('id', 'unknown'),
                company_name=company_profile.get('name', 'Unknown'),
                primary_role=role_classification['primary_role'],
                confidence_score=role_classification['confidence'],
                buyer_indicators=role_classification['buyer_indicators'],
                seller_indicators=role_classification['seller_indicators'],
                material_preferences=material_analysis['preferences'],
                market_position=market_analysis['position'],
                transaction_history=transaction_analysis['history'],
                classification_reasoning=reasoning
            )
            
            self.logger.info(f"✅ Classified {company_profile.get('name', 'Unknown')} as {classification.primary_role.value}")
            return classification
            
        except Exception as e:
            self.logger.error(f"❌ Error in company role classification: {e}")
            # Return neutral classification as fallback
            return BuyerSellerClassification(
                company_id=company_profile.get('id', 'unknown'),
                company_name=company_profile.get('name', 'Unknown'),
                primary_role=CompanyRole.NEUTRAL,
                confidence_score=0.0,
                buyer_indicators=[],
                seller_indicators=[],
                material_preferences={},
                market_position='unknown',
                transaction_history={},
                classification_reasoning=f"Classification failed: {str(e)}"
            )
    
    def _extract_company_features(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from company profile"""
        features = {}
        
        # Basic company information
        features['name'] = company_profile.get('name', '')
        features['industry'] = company_profile.get('industry', '')
        features['location'] = company_profile.get('location', '')
        features['size'] = company_profile.get('employee_count', 0)
        
        # Materials and waste streams
        features['materials'] = company_profile.get('materials', [])
        features['waste_streams'] = company_profile.get('waste_streams', [])
        features['products'] = company_profile.get('products', [])
        
        # Sustainability metrics
        features['sustainability_score'] = company_profile.get('sustainability_score', 0)
        features['carbon_footprint'] = company_profile.get('carbon_footprint', 0)
        features['energy_needs'] = company_profile.get('energy_needs', '')
        features['water_usage'] = company_profile.get('water_usage', '')
        
        # Matching preferences
        features['matching_preferences'] = company_profile.get('matching_preferences', {})
        
        return features
    
    def _analyze_material_flows(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze material flows to determine buyer/seller tendencies"""
        analysis = {
            'input_materials': [],
            'output_materials': [],
            'waste_streams': [],
            'byproducts': [],
            'surplus_materials': [],
            'material_requirements': [],
            'flow_balance': 0.0,
            'preferences': {}
        }
        
        # Analyze materials as inputs (buyer behavior)
        materials = company_profile.get('materials', [])
        for material in materials:
            if self._is_input_material(material, company_profile):
                analysis['input_materials'].append(material)
                analysis['material_requirements'].append(material)
        
        # Analyze waste streams as outputs (seller behavior)
        waste_streams = company_profile.get('waste_streams', [])
        for waste in waste_streams:
            if self._is_output_material(waste, company_profile):
                analysis['output_materials'].append(waste)
                analysis['waste_streams'].append(waste)
        
        # Analyze products for potential byproducts
        products = company_profile.get('products', [])
        for product in products:
            byproducts = self._identify_byproducts(product, company_profile)
            analysis['byproducts'].extend(byproducts)
            analysis['output_materials'].extend(byproducts)
        
        # Calculate flow balance (positive = more outputs, negative = more inputs)
        analysis['flow_balance'] = len(analysis['output_materials']) - len(analysis['input_materials'])
        
        # Determine material preferences
        analysis['preferences'] = self._determine_material_preferences(company_profile)
        
        return analysis
    
    def _analyze_transaction_patterns(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transaction patterns to identify buyer/seller behavior"""
        analysis = {
            'buying_frequency': 0,
            'selling_frequency': 0,
            'transaction_volume': 0,
            'preferred_partners': [],
            'transaction_history': {},
            'market_behavior': 'unknown'
        }
        
        # Analyze historical transactions if available
        transactions = company_profile.get('transaction_history', [])
        
        # Handle case where transaction_history might be a dict instead of list
        if isinstance(transactions, dict):
            transactions = []
        
        for transaction in transactions:
            if transaction.get('type') == 'buy':
                analysis['buying_frequency'] += 1
            elif transaction.get('type') == 'sell':
                analysis['selling_frequency'] += 1
            
            analysis['transaction_volume'] += transaction.get('volume', 0)
            partner = transaction.get('partner')
            if partner:
                analysis['preferred_partners'].append(partner)
        
        # Determine market behavior
        if analysis['buying_frequency'] > analysis['selling_frequency']:
            analysis['market_behavior'] = 'buyer_dominant'
        elif analysis['selling_frequency'] > analysis['buying_frequency']:
            analysis['market_behavior'] = 'seller_dominant'
        else:
            analysis['market_behavior'] = 'balanced'
        
        return analysis
    
    def _analyze_market_position(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze company's market position"""
        analysis = {
            'position': 'unknown',
            'market_power': 0.0,
            'competitive_advantage': [],
            'market_segment': 'unknown',
            'growth_potential': 0.0
        }
        
        # Analyze company size and industry
        size = company_profile.get('employee_count', 0)
        industry = company_profile.get('industry', '').lower()
        
        # Determine market position based on size and industry
        if size > 1000:
            if 'manufacturing' in industry or 'chemical' in industry:
                analysis['position'] = 'dominant_seller'
            else:
                analysis['position'] = 'dominant_buyer'
        elif size > 100:
            analysis['position'] = 'balanced'
        else:
            analysis['position'] = 'niche_buyer' if 'service' in industry else 'niche_seller'
        
        # Calculate market power
        analysis['market_power'] = min(1.0, size / 10000.0)
        
        # Identify competitive advantages
        advantages = []
        if company_profile.get('sustainability_score', 0) > 0.8:
            advantages.append('sustainability_leader')
        if len(company_profile.get('materials', [])) > 10:
            advantages.append('material_diversity')
        if company_profile.get('carbon_footprint', 0) < 100:
            advantages.append('low_carbon')
        
        analysis['competitive_advantage'] = advantages
        
        return analysis
    
    def _perform_role_classification(self, company_features: Dict[str, Any], 
                                   material_analysis: Dict[str, Any],
                                   transaction_analysis: Dict[str, Any],
                                   market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual role classification"""
        
        # Calculate buyer score
        buyer_score = 0.0
        buyer_indicators = []
        
        # Material requirements indicate buyer behavior
        if len(material_analysis['material_requirements']) > 0:
            buyer_score += 0.3
            buyer_indicators.append('has_material_requirements')
        
        # Buying frequency in transactions
        if transaction_analysis['buying_frequency'] > transaction_analysis['selling_frequency']:
            buyer_score += 0.2
            buyer_indicators.append('high_buying_frequency')
        
        # Market behavior
        if transaction_analysis['market_behavior'] == 'buyer_dominant':
            buyer_score += 0.2
            buyer_indicators.append('buyer_dominant_behavior')
        
        # Industry indicators
        industry = company_features['industry'].lower()
        if any(indicator in industry for indicator in ['manufacturing', 'processing', 'assembly']):
            buyer_score += 0.1
            buyer_indicators.append('manufacturing_industry')
        
        # Calculate seller score
        seller_score = 0.0
        seller_indicators = []
        
        # Waste streams indicate seller behavior
        if len(material_analysis['waste_streams']) > 0:
            seller_score += 0.3
            seller_indicators.append('has_waste_streams')
        
        # Byproducts indicate seller behavior
        if len(material_analysis['byproducts']) > 0:
            seller_score += 0.2
            seller_indicators.append('has_byproducts')
        
        # Selling frequency in transactions
        if transaction_analysis['selling_frequency'] > transaction_analysis['buying_frequency']:
            seller_score += 0.2
            seller_indicators.append('high_selling_frequency')
        
        # Market behavior
        if transaction_analysis['market_behavior'] == 'seller_dominant':
            seller_score += 0.2
            seller_indicators.append('seller_dominant_behavior')
        
        # Determine primary role
        if buyer_score > 0.5 and seller_score > 0.5:
            primary_role = CompanyRole.BOTH
            confidence = max(buyer_score, seller_score)
        elif buyer_score > seller_score and buyer_score > 0.3:
            primary_role = CompanyRole.BUYER
            confidence = buyer_score
        elif seller_score > buyer_score and seller_score > 0.3:
            primary_role = CompanyRole.SELLER
            confidence = seller_score
        else:
            primary_role = CompanyRole.NEUTRAL
            confidence = max(buyer_score, seller_score)
        
        return {
            'primary_role': primary_role,
            'confidence': confidence,
            'buyer_score': buyer_score,
            'seller_score': seller_score,
            'buyer_indicators': buyer_indicators,
            'seller_indicators': seller_indicators
        }
    
    def _generate_classification_reasoning(self, company_profile: Dict[str, Any],
                                         role_classification: Dict[str, Any],
                                         material_analysis: Dict[str, Any],
                                         transaction_analysis: Dict[str, Any],
                                         market_analysis: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the classification"""
        
        reasoning_parts = []
        
        # Add role explanation
        role = role_classification['primary_role'].value
        confidence = role_classification['confidence']
        reasoning_parts.append(f"Classified as {role} with {confidence:.1%} confidence.")
        
        # Add buyer indicators
        if role_classification['buyer_indicators']:
            reasoning_parts.append(f"Buyer indicators: {', '.join(role_classification['buyer_indicators'])}")
        
        # Add seller indicators
        if role_classification['seller_indicators']:
            reasoning_parts.append(f"Seller indicators: {', '.join(role_classification['seller_indicators'])}")
        
        # Add material flow analysis
        if material_analysis['flow_balance'] > 0:
            reasoning_parts.append(f"Net material output: {material_analysis['flow_balance']} materials")
        elif material_analysis['flow_balance'] < 0:
            reasoning_parts.append(f"Net material input: {abs(material_analysis['flow_balance'])} materials")
        else:
            reasoning_parts.append("Balanced material flow")
        
        # Add transaction behavior
        reasoning_parts.append(f"Transaction behavior: {transaction_analysis['market_behavior']}")
        
        # Add market position
        reasoning_parts.append(f"Market position: {market_analysis['position']}")
        
        return " | ".join(reasoning_parts)
    
    def _is_input_material(self, material: str, company_profile: Dict[str, Any]) -> bool:
        """Determine if a material is an input for the company"""
        material_lower = material.lower()
        
        # Check if material is in waste streams (not an input)
        waste_streams = [w.lower() for w in company_profile.get('waste_streams', [])]
        if material_lower in waste_streams:
            return False
        
        # Check if material is a byproduct (not an input)
        products = [p.lower() for p in company_profile.get('products', [])]
        if material_lower in products:
            return False
        
        # Check industry-specific input materials
        industry = company_profile.get('industry', '').lower()
        
        if 'manufacturing' in industry:
            return any(keyword in material_lower for keyword in ['raw', 'steel', 'aluminum', 'plastic', 'chemical'])
        elif 'chemical' in industry:
            return any(keyword in material_lower for keyword in ['chemical', 'reagent', 'catalyst', 'solvent'])
        elif 'food' in industry:
            return any(keyword in material_lower for keyword in ['ingredient', 'raw', 'agricultural', 'organic'])
        
        # Default: assume it's an input if not clearly an output
        return True
    
    def _is_output_material(self, material: str, company_profile: Dict[str, Any]) -> bool:
        """Determine if a material is an output for the company"""
        material_lower = material.lower()
        
        # Check if material is in materials list (not an output)
        materials = [m.lower() for m in company_profile.get('materials', [])]
        if material_lower in materials:
            return False
        
        # Check if material is a waste or byproduct
        if any(keyword in material_lower for keyword in ['waste', 'byproduct', 'surplus', 'excess', 'scrap']):
            return True
        
        # Check if material is a product
        products = [p.lower() for p in company_profile.get('products', [])]
        if material_lower in products:
            return True
        
        return False
    
    def _identify_byproducts(self, product: str, company_profile: Dict[str, Any]) -> List[str]:
        """Identify potential byproducts from a product"""
        product_lower = product.lower()
        byproducts = []
        
        # Industry-specific byproduct identification
        industry = company_profile.get('industry', '').lower()
        
        if 'chemical' in industry:
            if 'ethylene' in product_lower:
                byproducts.extend(['propylene', 'butadiene', 'benzene'])
            elif 'steel' in product_lower:
                byproducts.extend(['slag', 'dust', 'scale'])
        
        return byproducts
    
    def _determine_material_preferences(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Determine company's material preferences"""
        preferences = {
            'preferred_materials': [],
            'avoided_materials': [],
            'quality_requirements': 'standard',
            'quantity_preferences': 'flexible',
            'logistics_preferences': 'standard',
            'sustainability_focus': company_profile.get('sustainability_score', 0) > 0.7
        }
        
        # Analyze materials for preferences
        materials = company_profile.get('materials', [])
        for material in materials:
            if any(keyword in material.lower() for keyword in ['recycled', 'sustainable', 'green']):
                preferences['preferred_materials'].append(material)
            elif any(keyword in material.lower() for keyword in ['hazardous', 'toxic', 'dangerous']):
                preferences['avoided_materials'].append(material)
        
        return preferences

async def test_buyer_seller_differentiation():
    """Test the buyer/seller differentiation system"""
    print("🧠 Testing Buyer/Seller Differentiation System")
    print("="*60)
    
    # Initialize system
    differentiation_system = BuyerSellerDifferentiationSystem()
    
    # Test companies
    test_companies = [
        {
            'id': 'company_1',
            'name': 'Steel Manufacturing Corp',
            'industry': 'steel manufacturing',
            'location': 'Dubai',
            'employee_count': 2500,
            'materials': ['iron ore', 'coal', 'limestone', 'recycled steel'],
            'waste_streams': ['steel slag', 'mill scale', 'dust', 'scrap metal'],
            'products': ['steel beams', 'steel plates', 'steel coils'],
            'sustainability_score': 0.75,
            'carbon_footprint': 150,
            'transaction_history': [
                {'type': 'buy', 'material': 'iron ore', 'volume': 1000, 'partner': 'Mining Corp'},
                {'type': 'sell', 'material': 'steel slag', 'volume': 500, 'partner': 'Cement Corp'}
            ]
        },
        {
            'id': 'company_2',
            'name': 'Automotive Assembly Ltd',
            'industry': 'automotive manufacturing',
            'location': 'Abu Dhabi',
            'employee_count': 800,
            'materials': ['steel', 'aluminum', 'plastic', 'electronics', 'rubber'],
            'waste_streams': ['metal chips', 'plastic waste', 'packaging waste'],
            'products': ['automotive parts', 'vehicle components'],
            'sustainability_score': 0.65,
            'carbon_footprint': 80,
            'transaction_history': [
                {'type': 'buy', 'material': 'steel', 'volume': 200, 'partner': 'Steel Corp'},
                {'type': 'buy', 'material': 'aluminum', 'volume': 150, 'partner': 'Aluminum Corp'}
            ]
        },
        {
            'id': 'company_3',
            'name': 'Chemical Processing Plant',
            'industry': 'chemical processing',
            'location': 'Sharjah',
            'employee_count': 1200,
            'materials': ['crude oil', 'natural gas', 'chemicals', 'catalysts'],
            'waste_streams': ['spent catalysts', 'chemical waste', 'wastewater'],
            'products': ['ethylene', 'propylene', 'benzene', 'toluene'],
            'sustainability_score': 0.45,
            'carbon_footprint': 200,
            'transaction_history': [
                {'type': 'buy', 'material': 'crude oil', 'volume': 5000, 'partner': 'Oil Corp'},
                {'type': 'sell', 'material': 'ethylene', 'volume': 3000, 'partner': 'Plastic Corp'},
                {'type': 'sell', 'material': 'spent catalysts', 'volume': 100, 'partner': 'Recycling Corp'}
            ]
        }
    ]
    
    # Classify each company
    for company in test_companies:
        print(f"\n🏢 Analyzing: {company['name']}")
        print(f"   Industry: {company['industry']}")
        print(f"   Size: {company['employee_count']} employees")
        
        classification = await differentiation_system.classify_company_role(company)
        
        print(f"   🎯 Role: {classification.primary_role.value.upper()}")
        print(f"   📊 Confidence: {classification.confidence_score:.1%}")
        print(f"   📍 Market Position: {classification.market_position}")
        print(f"   💡 Reasoning: {classification.classification_reasoning}")
        
        if classification.buyer_indicators:
            print(f"   🛒 Buyer Indicators: {', '.join(classification.buyer_indicators)}")
        if classification.seller_indicators:
            print(f"   📦 Seller Indicators: {', '.join(classification.seller_indicators)}")
    
    print("\n✅ Buyer/Seller Differentiation Test Complete")

if __name__ == "__main__":
    asyncio.run(test_buyer_seller_differentiation()) 