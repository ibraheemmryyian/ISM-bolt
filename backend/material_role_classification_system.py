#!/usr/bin/env python3
"""
🧠 MATERIAL ROLE CLASSIFICATION SYSTEM
Advanced AI system to classify each material as buyer or seller role
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple, Union
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MaterialRole(Enum):
    """Material roles in industrial symbiosis"""
    BUYER = "buyer"  # Company wants/needs this material
    SELLER = "seller"  # Company has this material to sell
    BOTH = "both"  # Company both needs and has this material
    NEUTRAL = "neutral"  # Not clearly buyer or seller

@dataclass
class MaterialClassification:
    """Classification result for individual material"""
    material_name: str
    material_type: str
    company_name: str
    company_industry: str
    material_role: MaterialRole
    confidence_score: float
    classification_reasoning: str
    buyer_indicators: List[str]
    seller_indicators: List[str]
    material_properties: Dict[str, Any]

class MaterialRoleClassificationSystem:
    """
    🧠 Advanced AI system for material-level buyer/seller classification
    
    Features:
    - Individual material analysis
    - Context-aware classification
    - Industry-specific reasoning
    - Material property analysis
    - Company context integration
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🧠 INITIALIZING MATERIAL ROLE CLASSIFICATION SYSTEM")
        
        # Initialize AI components
        self._initialize_ai_components()
        
        # Initialize classification rules
        self._initialize_classification_rules()
        
        # Initialize material databases
        self._initialize_material_databases()
        
        self.logger.info("✅ MATERIAL ROLE CLASSIFICATION SYSTEM READY")
    
    def _initialize_ai_components(self):
        """Initialize AI components for analysis"""
        self.logger.info("🚀 Initializing AI components...")
        
        # Semantic encoder for material analysis
        self.semantic_encoder = SentenceTransformer('all-mpnet-base-v2')
        
        # Material role classifier
        self.material_role_classifier = nn.Sequential(
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
        
        # Material property analyzer
        self.property_analyzer = nn.Sequential(
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
        
        self.logger.info("✅ AI components initialized")
    
    def _initialize_classification_rules(self):
        """Initialize classification rules"""
        self.logger.info("🎯 Initializing classification rules...")
        
        # Buyer indicators (materials companies want/need)
        self.buyer_indicators = {
            'raw_materials': ['ore', 'crude', 'raw', 'primary', 'base'],
            'production_inputs': ['steel', 'aluminum', 'plastic', 'chemical', 'reagent'],
            'energy_materials': ['fuel', 'gas', 'electricity', 'power', 'energy'],
            'processing_materials': ['catalyst', 'solvent', 'lubricant', 'coolant'],
            'packaging_materials': ['packaging', 'container', 'box', 'bag', 'wrapper'],
            'maintenance_materials': ['spare parts', 'tools', 'equipment', 'maintenance'],
            'quality_materials': ['certified', 'grade', 'standard', 'specification']
        }
        
        # Seller indicators (materials companies have to sell)
        self.seller_indicators = {
            'waste_materials': ['waste', 'scrap', 'residue', 'sludge', 'effluent'],
            'byproducts': ['byproduct', 'co-product', 'secondary', 'derivative'],
            'surplus_materials': ['surplus', 'excess', 'leftover', 'unused', 'spare'],
            'recyclable_materials': ['recyclable', 'reusable', 'recoverable', 'reclaimed'],
            'processed_materials': ['processed', 'refined', 'treated', 'upgraded'],
            'finished_products': ['product', 'finished', 'final', 'completed'],
            'intermediate_materials': ['intermediate', 'semi-finished', 'partially processed']
        }
        
        # Industry-specific material patterns
        self.industry_patterns = {
            'steel_manufacturing': {
                'buyer_materials': ['iron ore', 'coal', 'limestone', 'recycled steel'],
                'seller_materials': ['steel slag', 'mill scale', 'dust', 'scrap metal', 'steel products']
            },
            'automotive_manufacturing': {
                'buyer_materials': ['steel', 'aluminum', 'plastic', 'electronics', 'rubber'],
                'seller_materials': ['metal chips', 'plastic waste', 'packaging waste', 'automotive parts']
            },
            'chemical_processing': {
                'buyer_materials': ['crude oil', 'natural gas', 'chemicals', 'catalysts'],
                'seller_materials': ['spent catalysts', 'chemical waste', 'wastewater', 'chemical products']
            },
            'food_processing': {
                'buyer_materials': ['raw ingredients', 'agricultural products', 'packaging'],
                'seller_materials': ['food waste', 'byproducts', 'packaging waste', 'processed food']
            },
            'construction': {
                'buyer_materials': ['cement', 'steel', 'aggregates', 'wood', 'materials'],
                'seller_materials': ['construction waste', 'demolition waste', 'excess materials']
            }
        }
        
        self.logger.info("✅ Classification rules initialized")
    
    def _initialize_material_databases(self):
        """Initialize material databases"""
        self.logger.info("📚 Initializing material databases...")
        
        # Material properties database
        self.material_properties = {
            'steel': {
                'type': 'metal',
                'density': 7.85,
                'recyclability': 'high',
                'value': 'high',
                'common_uses': ['construction', 'automotive', 'manufacturing']
            },
            'aluminum': {
                'type': 'metal',
                'density': 2.7,
                'recyclability': 'excellent',
                'value': 'high',
                'common_uses': ['automotive', 'aerospace', 'packaging']
            },
            'plastic': {
                'type': 'polymer',
                'density': 0.9,
                'recyclability': 'medium',
                'value': 'medium',
                'common_uses': ['packaging', 'automotive', 'electronics']
            },
            'waste': {
                'type': 'waste',
                'density': 1.0,
                'recyclability': 'variable',
                'value': 'low',
                'common_uses': ['recycling', 'disposal', 'energy recovery']
            }
        }
        
        self.logger.info("✅ Material databases initialized")
    
    async def classify_material_role(self, material_name: str, material_type: str, 
                                   company_profile: Dict[str, Any]) -> MaterialClassification:
        """
        Classify a specific material's role for a company
        
        Args:
            material_name: Name of the material
            material_type: Type of material (waste, requirement, product, etc.)
            company_profile: Company profile data
            
        Returns:
            MaterialClassification: Detailed classification result
        """
        self.logger.info(f"🧠 Classifying material role: {material_name} for {company_profile.get('name', 'Unknown')}")
        
        try:
            # Analyze material properties
            material_properties = self._analyze_material_properties(material_name)
            
            # Analyze material context
            material_context = self._analyze_material_context(material_name, material_type, company_profile)
            
            # Perform role classification
            role_classification = self._perform_material_role_classification(
                material_name, material_type, material_properties, material_context, company_profile
            )
            
            # Generate classification reasoning
            reasoning = self._generate_material_classification_reasoning(
                material_name, material_type, role_classification, material_properties, material_context, company_profile
            )
            
            # Create classification result
            classification = MaterialClassification(
                material_name=material_name,
                material_type=material_type,
                company_name=company_profile.get('name', 'Unknown'),
                company_industry=company_profile.get('industry', 'Unknown'),
                material_role=role_classification['primary_role'],
                confidence_score=role_classification['confidence'],
                classification_reasoning=reasoning,
                buyer_indicators=role_classification['buyer_indicators'],
                seller_indicators=role_classification['seller_indicators'],
                material_properties=material_properties
            )
            
            self.logger.info(f"✅ Classified {material_name} as {classification.material_role.value}")
            return classification
            
        except Exception as e:
            self.logger.error(f"❌ Error in material role classification: {e}")
            # Return neutral classification as fallback
            return MaterialClassification(
                material_name=material_name,
                material_type=material_type,
                company_name=company_profile.get('name', 'Unknown'),
                company_industry=company_profile.get('industry', 'Unknown'),
                material_role=MaterialRole.NEUTRAL,
                confidence_score=0.0,
                classification_reasoning=f"Classification failed: {str(e)}",
                buyer_indicators=[],
                seller_indicators=[],
                material_properties={}
            )
    
    def _analyze_material_properties(self, material_name: str) -> Dict[str, Any]:
        """Analyze material properties"""
        material_lower = material_name.lower()
        properties = {
            'type': 'unknown',
            'density': 1.0,
            'recyclability': 'unknown',
            'value': 'unknown',
            'common_uses': [],
            'is_waste': False,
            'is_raw_material': False,
            'is_processed': False
        }
        
        # Check if it's waste
        if any(keyword in material_lower for keyword in ['waste', 'scrap', 'residue', 'sludge', 'effluent']):
            properties['is_waste'] = True
            properties['type'] = 'waste'
            properties['value'] = 'low'
        
        # Check if it's a raw material
        elif any(keyword in material_lower for keyword in ['ore', 'crude', 'raw', 'primary']):
            properties['is_raw_material'] = True
            properties['type'] = 'raw_material'
            properties['value'] = 'medium'
        
        # Check if it's processed
        elif any(keyword in material_lower for keyword in ['processed', 'refined', 'treated', 'finished']):
            properties['is_processed'] = True
            properties['type'] = 'processed'
            properties['value'] = 'high'
        
        # Check specific material types
        if 'steel' in material_lower:
            properties.update(self.material_properties.get('steel', {}))
        elif 'aluminum' in material_lower:
            properties.update(self.material_properties.get('aluminum', {}))
        elif 'plastic' in material_lower:
            properties.update(self.material_properties.get('plastic', {}))
        
        return properties
    
    def _analyze_material_context(self, material_name: str, material_type: str, 
                                company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze material context within company"""
        context = {
            'is_input_material': False,
            'is_output_material': False,
            'is_waste_stream': False,
            'is_product': False,
            'industry_relevance': 0.0,
            'company_specific_patterns': []
        }
        
        material_lower = material_name.lower()
        industry = company_profile.get('industry', '').lower()
        
        # Check if it's in materials list (input)
        materials = [m.lower() for m in company_profile.get('materials', [])]
        if material_lower in materials:
            context['is_input_material'] = True
        
        # Check if it's in waste streams (output)
        waste_streams = [w.lower() for w in company_profile.get('waste_streams', [])]
        if material_lower in waste_streams:
            context['is_output_material'] = True
            context['is_waste_stream'] = True
        
        # Check if it's in products (output)
        products = [p.lower() for p in company_profile.get('products', [])]
        if material_lower in products:
            context['is_output_material'] = True
            context['is_product'] = True
        
        # Check industry-specific patterns
        if industry in self.industry_patterns:
            patterns = self.industry_patterns[industry]
            if material_lower in [m.lower() for m in patterns.get('buyer_materials', [])]:
                context['industry_relevance'] += 0.5
                context['company_specific_patterns'].append('industry_buyer_material')
            if material_lower in [m.lower() for m in patterns.get('seller_materials', [])]:
                context['industry_relevance'] += 0.5
                context['company_specific_patterns'].append('industry_seller_material')
        
        return context
    
    def _perform_material_role_classification(self, material_name: str, material_type: str,
                                            material_properties: Dict[str, Any],
                                            material_context: Dict[str, Any],
                                            company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual material role classification"""
        
        # Calculate buyer score
        buyer_score = 0.0
        buyer_indicators = []
        
        # Material type indicates buyer behavior
        if material_type.lower() in ['requirement', 'need', 'input']:
            buyer_score += 0.4
            buyer_indicators.append('material_requirement')
        
        # Context indicates buyer behavior
        if material_context['is_input_material']:
            buyer_score += 0.3
            buyer_indicators.append('input_material')
        
        # Industry patterns indicate buyer behavior
        if 'industry_buyer_material' in material_context['company_specific_patterns']:
            buyer_score += 0.2
            buyer_indicators.append('industry_buyer_pattern')
        
        # Material properties indicate buyer behavior
        if material_properties['is_raw_material']:
            buyer_score += 0.1
            buyer_indicators.append('raw_material')
        
        # Calculate seller score
        seller_score = 0.0
        seller_indicators = []
        
        # Material type indicates seller behavior
        if material_type.lower() in ['waste', 'byproduct', 'surplus', 'output']:
            seller_score += 0.4
            seller_indicators.append('material_output')
        
        # Context indicates seller behavior
        if material_context['is_output_material']:
            seller_score += 0.3
            seller_indicators.append('output_material')
        
        # Waste streams indicate seller behavior
        if material_context['is_waste_stream']:
            seller_score += 0.2
            seller_indicators.append('waste_stream')
        
        # Industry patterns indicate seller behavior
        if 'industry_seller_material' in material_context['company_specific_patterns']:
            seller_score += 0.2
            seller_indicators.append('industry_seller_pattern')
        
        # Material properties indicate seller behavior
        if material_properties['is_waste']:
            seller_score += 0.1
            seller_indicators.append('waste_material')
        
        # Determine primary role
        if buyer_score > 0.3 and seller_score > 0.3:
            primary_role = MaterialRole.BOTH
            confidence = max(buyer_score, seller_score)
        elif buyer_score > seller_score and buyer_score > 0.3:
            primary_role = MaterialRole.BUYER
            confidence = buyer_score
        elif seller_score > buyer_score and seller_score > 0.3:
            primary_role = MaterialRole.SELLER
            confidence = seller_score
        else:
            primary_role = MaterialRole.NEUTRAL
            confidence = max(buyer_score, seller_score)
        
        return {
            'primary_role': primary_role,
            'confidence': confidence,
            'buyer_score': buyer_score,
            'seller_score': seller_score,
            'buyer_indicators': buyer_indicators,
            'seller_indicators': seller_indicators
        }
    
    def _generate_material_classification_reasoning(self, material_name: str, material_type: str,
                                                  role_classification: Dict[str, Any],
                                                  material_properties: Dict[str, Any],
                                                  material_context: Dict[str, Any],
                                                  company_profile: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the material classification"""
        
        reasoning_parts = []
        
        # Add role explanation
        role = role_classification['primary_role'].value
        confidence = role_classification['confidence']
        reasoning_parts.append(f"Material '{material_name}' classified as {role} with {confidence:.1%} confidence.")
        
        # Add material type explanation
        reasoning_parts.append(f"Material type: {material_type}")
        
        # Add buyer indicators
        if role_classification['buyer_indicators']:
            reasoning_parts.append(f"Buyer indicators: {', '.join(role_classification['buyer_indicators'])}")
        
        # Add seller indicators
        if role_classification['seller_indicators']:
            reasoning_parts.append(f"Seller indicators: {', '.join(role_classification['seller_indicators'])}")
        
        # Add context information
        if material_context['is_input_material']:
            reasoning_parts.append("Company lists this as input material")
        if material_context['is_output_material']:
            reasoning_parts.append("Company lists this as output material")
        if material_context['is_waste_stream']:
            reasoning_parts.append("Company lists this as waste stream")
        if material_context['is_product']:
            reasoning_parts.append("Company lists this as product")
        
        # Add industry relevance
        if material_context['industry_relevance'] > 0:
            reasoning_parts.append(f"Industry relevance: {material_context['industry_relevance']:.1%}")
        
        return " | ".join(reasoning_parts)

async def test_material_role_classification():
    """Test the material role classification system"""
    print("🧠 Testing Material Role Classification System")
    print("="*60)
    
    # Initialize system
    classification_system = MaterialRoleClassificationSystem()
    
    # Test company
    test_company = {
        'id': 'company_1',
        'name': 'Steel Manufacturing Corp',
        'industry': 'steel manufacturing',
        'location': 'Dubai',
        'employee_count': 2500,
        'materials': ['iron ore', 'coal', 'limestone', 'recycled steel'],
        'waste_streams': ['steel slag', 'mill scale', 'dust', 'scrap metal'],
        'products': ['steel beams', 'steel plates', 'steel coils'],
        'sustainability_score': 0.75,
        'carbon_footprint': 150
    }
    
    # Test materials
    test_materials = [
        ('iron ore', 'requirement'),
        ('steel slag', 'waste'),
        ('steel beams', 'product'),
        ('coal', 'requirement'),
        ('scrap metal', 'waste'),
        ('limestone', 'requirement')
    ]
    
    print(f"\n🏢 Company: {test_company['name']}")
    print(f"   Industry: {test_company['industry']}")
    print(f"   Size: {test_company['employee_count']} employees")
    
    # Classify each material
    for material_name, material_type in test_materials:
        print(f"\n🔬 Analyzing Material: {material_name} ({material_type})")
        
        classification = await classification_system.classify_material_role(
            material_name, material_type, test_company
        )
        
        print(f"   🎯 Role: {classification.material_role.value.upper()}")
        print(f"   📊 Confidence: {classification.confidence_score:.1%}")
        print(f"   💡 Reasoning: {classification.classification_reasoning}")
        
        if classification.buyer_indicators:
            print(f"   🛒 Buyer Indicators: {', '.join(classification.buyer_indicators)}")
        if classification.seller_indicators:
            print(f"   📦 Seller Indicators: {', '.join(classification.seller_indicators)}")
    
    print("\n✅ Material Role Classification Test Complete")

if __name__ == "__main__":
    asyncio.run(test_material_role_classification()) 