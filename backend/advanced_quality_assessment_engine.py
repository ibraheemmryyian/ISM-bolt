"""
Advanced Quality Assessment Engine for Material Listings
Implements multi-factor quality evaluation with dynamic scoring
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime
import logging
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
try:
    from transformers
    HAS_TRANSFORMERS = True
except ImportError:
    from .fallbacks.transformers_fallback import *
    HAS_TRANSFORMERS = False import AutoTokenizer, AutoModel
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    from .fallbacks.spacy_fallback import *
    HAS_SPACY = False
from collections import defaultdict
import re

class QualityGrade(Enum):
    """Dynamic quality grades based on assessment scores"""
    PREMIUM = "PREMIUM"  # 85-100% score
    HIGH = "HIGH"       # 70-84% score
    STANDARD = "STANDARD"  # 55-69% score
    BASIC = "BASIC"     # 40-54% score
    LOW = "LOW"         # Below 40% score

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for material listings"""
    completeness_score: float
    technical_accuracy: float
    market_alignment: float
    sustainability_score: float
    contextual_relevance: float
    innovation_factor: float
    data_freshness: float
    linguistic_quality: float
    uniqueness_score: float
    verification_status: float
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'completeness': 0.15,
            'technical': 0.20,
            'market': 0.15,
            'sustainability': 0.10,
            'contextual': 0.15,
            'innovation': 0.05,
            'freshness': 0.05,
            'linguistic': 0.05,
            'uniqueness': 0.05,
            'verification': 0.05
        }
        
        score = (
            self.completeness_score * weights['completeness'] +
            self.technical_accuracy * weights['technical'] +
            self.market_alignment * weights['market'] +
            self.sustainability_score * weights['sustainability'] +
            self.contextual_relevance * weights['contextual'] +
            self.innovation_factor * weights['innovation'] +
            self.data_freshness * weights['freshness'] +
            self.linguistic_quality * weights['linguistic'] +
            self.uniqueness_score * weights['uniqueness'] +
            self.verification_status * weights['verification']
        )
        
        return min(max(score, 0), 1.0)  # Clamp between 0 and 1

class AdvancedQualityAssessmentEngine:
    """
    Advanced quality assessment system using multiple evaluation criteria
    and machine learning models for sophisticated quality determination
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
        
        # Initialize material knowledge base
        self.material_knowledge = self._load_material_knowledge()
        
        # Initialize industry-specific validators
        self.industry_validators = self._initialize_industry_validators()
        
        # Track historical quality patterns
        self.quality_history = defaultdict(list)
        
        # Neural quality predictor
        self.quality_predictor = self._build_quality_predictor()
        
    def _load_material_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive material knowledge base"""
        return {
            'metals': {
                'properties': ['tensile_strength', 'density', 'melting_point', 'corrosion_resistance'],
                'quality_indicators': ['purity', 'grade', 'certification', 'source'],
                'market_factors': ['LME_price', 'demand_trend', 'supply_chain']
            },
            'polymers': {
                'properties': ['molecular_weight', 'glass_transition', 'melt_flow_index'],
                'quality_indicators': ['virgin_content', 'contamination_level', 'color_consistency'],
                'market_factors': ['oil_price_correlation', 'recycled_content_premium']
            },
            'chemicals': {
                'properties': ['purity', 'pH', 'concentration', 'stability'],
                'quality_indicators': ['certification', 'hazard_class', 'shelf_life'],
                'market_factors': ['regulatory_compliance', 'transportation_cost']
            },
            'waste': {
                'properties': ['composition', 'moisture_content', 'calorific_value'],
                'quality_indicators': ['sorting_level', 'contamination', 'consistency'],
                'market_factors': ['disposal_cost_avoidance', 'circular_economy_value']
            }
        }
    
    def _initialize_industry_validators(self) -> Dict[str, Any]:
        """Initialize industry-specific validation rules"""
        return {
            'petrochemical': {
                'required_specs': ['API_gravity', 'sulfur_content', 'viscosity'],
                'certifications': ['ISO_9001', 'API_certification'],
                'min_description_length': 150
            },
            'manufacturing': {
                'required_specs': ['dimensions', 'tolerance', 'material_grade'],
                'certifications': ['ISO_14001', 'quality_certification'],
                'min_description_length': 100
            },
            'food_processing': {
                'required_specs': ['organic_certification', 'nutritional_content', 'shelf_life'],
                'certifications': ['HACCP', 'organic_certification'],
                'min_description_length': 120
            }
        }
    
    def _build_quality_predictor(self) -> nn.Module:
        """Build neural network for quality prediction"""
        class QualityPredictor(nn.Module):
            def __init__(self, input_dim=50, hidden_dim=128):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.dropout1 = nn.Dropout(0.3)
                
                self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
                self.dropout2 = nn.Dropout(0.2)
                
                self.fc3 = nn.Linear(hidden_dim // 2, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = torch.relu(self.bn1(self.fc1(x)))
                x = self.dropout1(x)
                x = torch.relu(self.bn2(self.fc2(x)))
                x = self.dropout2(x)
                x = self.sigmoid(self.fc3(x))
                return x
        
        return QualityPredictor()
    
    async def assess_listing_quality(self, listing: Dict[str, Any], company_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment of a material listing
        """
        try:
            # Extract listing components
            material_name = listing.get('material_name', '')
            description = listing.get('description', '')
            material_type = listing.get('material_type', 'unknown')
            quantity = listing.get('quantity', 0)
            unit = listing.get('unit', '')
            metadata = listing.get('metadata', {})
            
            # Compute individual quality metrics
            completeness = await self._assess_completeness(listing)
            technical_accuracy = await self._assess_technical_accuracy(listing, material_type)
            market_alignment = await self._assess_market_alignment(listing, company_context)
            sustainability = await self._assess_sustainability(listing, company_context)
            contextual_relevance = await self._assess_contextual_relevance(listing, company_context)
            innovation = await self._assess_innovation_factor(listing, metadata)
            freshness = await self._assess_data_freshness(listing)
            linguistic = await self._assess_linguistic_quality(description)
            uniqueness = await self._assess_uniqueness(listing, company_context)
            verification = await self._assess_verification_status(listing, metadata)
            
            # Create quality metrics object
            metrics = QualityMetrics(
                completeness_score=completeness,
                technical_accuracy=technical_accuracy,
                market_alignment=market_alignment,
                sustainability_score=sustainability,
                contextual_relevance=contextual_relevance,
                innovation_factor=innovation,
                data_freshness=freshness,
                linguistic_quality=linguistic,
                uniqueness_score=uniqueness,
                verification_status=verification
            )
            
            # Calculate overall score
            overall_score = metrics.overall_score
            
            # Use neural predictor for additional insights
            neural_features = self._extract_neural_features(listing, company_context)
            neural_score = self._predict_quality_neural(neural_features)
            
            # Combine algorithmic and neural scores
            final_score = 0.7 * overall_score + 0.3 * neural_score
            
            # Determine quality grade
            quality_grade = self._determine_quality_grade(final_score)
            
            # Generate detailed feedback
            feedback = self._generate_quality_feedback(metrics, listing)
            
            # Store in history for pattern learning
            self.quality_history[company_context.get('industry', 'unknown')].append({
                'score': final_score,
                'timestamp': datetime.now(),
                'material_type': material_type
            })
            
            return {
                'quality_grade': quality_grade,
                'overall_score': final_score,
                'metrics': {
                    'completeness': completeness,
                    'technical_accuracy': technical_accuracy,
                    'market_alignment': market_alignment,
                    'sustainability': sustainability,
                    'contextual_relevance': contextual_relevance,
                    'innovation': innovation,
                    'data_freshness': freshness,
                    'linguistic_quality': linguistic,
                    'uniqueness': uniqueness,
                    'verification': verification
                },
                'neural_score': neural_score,
                'feedback': feedback,
                'improvement_suggestions': self._generate_improvement_suggestions(metrics, listing)
            }
            
        except Exception as e:
            self.logger.error(f"Error in quality assessment: {e}")
            return self._get_default_assessment()
    
    async def _assess_completeness(self, listing: Dict[str, Any]) -> float:
        """Assess data completeness of the listing"""
        required_fields = [
            'material_name', 'description', 'quantity', 'unit', 
            'material_type', 'potential_value', 'quality_grade'
        ]
        
        optional_fields = [
            'certifications', 'specifications', 'images', 'samples_available',
            'minimum_order_quantity', 'lead_time', 'payment_terms'
        ]
        
        # Check required fields
        required_score = sum(1 for field in required_fields if listing.get(field)) / len(required_fields)
        
        # Check optional fields (bonus points)
        optional_score = sum(0.5 for field in optional_fields if listing.get(field)) / len(optional_fields)
        
        # Check description length
        description = listing.get('description', '')
        desc_length_score = min(len(description) / 300, 1.0)  # Optimal at 300+ chars
        
        # Check metadata richness
        metadata = listing.get('metadata', {})
        metadata_score = min(len(metadata) / 10, 1.0)  # Optimal at 10+ metadata fields
        
        # Weighted combination
        return (
            0.4 * required_score +
            0.2 * optional_score +
            0.2 * desc_length_score +
            0.2 * metadata_score
        )
    
    async def _assess_technical_accuracy(self, listing: Dict[str, Any], material_type: str) -> float:
        """Assess technical accuracy based on material type"""
        material_name = listing.get('material_name', '').lower()
        description = listing.get('description', '').lower()
        
        # Get material category
        material_category = self._categorize_material(material_name, material_type)
        
        if material_category not in self.material_knowledge:
            return 0.5  # Default score for unknown categories
        
        knowledge = self.material_knowledge[material_category]
        
        # Check for technical properties mentioned
        properties_mentioned = sum(
            1 for prop in knowledge['properties'] 
            if prop.lower() in description
        )
        property_score = properties_mentioned / len(knowledge['properties'])
        
        # Check for quality indicators
        quality_mentioned = sum(
            1 for indicator in knowledge['quality_indicators']
            if indicator.lower() in description or indicator.lower() in str(listing.get('metadata', {}))
        )
        quality_score = quality_mentioned / len(knowledge['quality_indicators'])
        
        # Check unit appropriateness
        unit = listing.get('unit', '').lower()
        unit_score = self._validate_unit_for_material(unit, material_category)
        
        # Validate quantity range
        quantity = listing.get('quantity', 0)
        quantity_score = self._validate_quantity_range(quantity, material_category, unit)
        
        return (
            0.3 * property_score +
            0.3 * quality_score +
            0.2 * unit_score +
            0.2 * quantity_score
        )
    
    async def _assess_market_alignment(self, listing: Dict[str, Any], company_context: Dict[str, Any]) -> float:
        """Assess alignment with market conditions and demand"""
        material_name = listing.get('material_name', '')
        potential_value = listing.get('potential_value', 0)
        industry = company_context.get('industry', '')
        
        # Check if pricing is realistic
        price_score = self._validate_pricing(potential_value, material_name, industry)
        
        # Check market demand indicators
        demand_keywords = ['high-demand', 'scarce', 'sought-after', 'popular', 'trending']
        description = listing.get('description', '').lower()
        demand_score = sum(1 for kw in demand_keywords if kw in description) / len(demand_keywords)
        
        # Industry-material alignment
        alignment_score = self._check_industry_material_alignment(material_name, industry)
        
        # Geographic relevance
        location = company_context.get('location', '')
        geo_score = self._assess_geographic_relevance(material_name, location)
        
        return (
            0.3 * price_score +
            0.2 * demand_score +
            0.3 * alignment_score +
            0.2 * geo_score
        )
    
    async def _assess_sustainability(self, listing: Dict[str, Any], company_context: Dict[str, Any]) -> float:
        """Assess sustainability aspects of the listing"""
        description = listing.get('description', '').lower()
        metadata = listing.get('metadata', {})
        sustainability_metrics = listing.get('sustainability_metrics', {})
        
        # Environmental keywords
        eco_keywords = [
            'recycled', 'renewable', 'biodegradable', 'sustainable', 
            'eco-friendly', 'green', 'circular', 'zero-waste', 'carbon-neutral'
        ]
        eco_score = sum(1 for kw in eco_keywords if kw in description) / len(eco_keywords)
        
        # Check for certifications
        eco_certs = ['ISO14001', 'FSC', 'PEFC', 'Cradle2Cradle', 'EcoLabel']
        cert_text = str(listing.get('certifications', [])) + str(metadata)
        cert_score = sum(1 for cert in eco_certs if cert.lower() in cert_text.lower()) / len(eco_certs)
        
        # Company sustainability score influence
        company_sustainability = company_context.get('sustainability_score', 0) / 100
        
        # Waste material bonus
        waste_bonus = 0.2 if listing.get('material_type', '').lower() == 'waste' else 0
        
        return min(
            0.3 * eco_score +
            0.3 * cert_score +
            0.2 * company_sustainability +
            waste_bonus,
            1.0
        )
    
    async def _assess_contextual_relevance(self, listing: Dict[str, Any], company_context: Dict[str, Any]) -> float:
        """Assess how well the listing fits the company context"""
        material_name = listing.get('material_name', '').lower()
        description = listing.get('description', '').lower()
        
        # Check if material aligns with company products
        products = [p.lower() for p in company_context.get('products', [])]
        product_alignment = any(product in description for product in products)
        
        # Check if material aligns with company materials
        materials = [m.lower() for m in company_context.get('materials', [])]
        material_alignment = any(mat in material_name for mat in materials)
        
        # Industry-specific relevance
        industry = company_context.get('industry', '').lower()
        industry_score = 1.0 if industry in description else 0.5
        
        # Company size appropriateness
        employee_count = company_context.get('employee_count', 0)
        quantity = listing.get('quantity', 0)
        size_score = self._assess_quantity_company_size_fit(quantity, employee_count)
        
        return (
            0.3 * (1.0 if product_alignment else 0.3) +
            0.3 * (1.0 if material_alignment else 0.3) +
            0.2 * industry_score +
            0.2 * size_score
        )
    
    async def _assess_innovation_factor(self, listing: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        """Assess innovation and uniqueness factors"""
        description = listing.get('description', '').lower()
        
        # Innovation keywords
        innovation_keywords = [
            'innovative', 'novel', 'cutting-edge', 'advanced', 'next-generation',
            'breakthrough', 'revolutionary', 'patented', 'proprietary'
        ]
        
        innovation_score = sum(1 for kw in innovation_keywords if kw in description) / len(innovation_keywords)
        
        # Technical advancement indicators
        tech_indicators = metadata.get('technology_level', 0)
        
        # R&D mentions
        rd_score = 1.0 if 'r&d' in description or 'research' in description else 0.5
        
        return (
            0.5 * innovation_score +
            0.3 * (tech_indicators / 10 if isinstance(tech_indicators, (int, float)) else 0.5) +
            0.2 * rd_score
        )
    
    async def _assess_data_freshness(self, listing: Dict[str, Any]) -> float:
        """Assess how recent and up-to-date the data is"""
        generated_at = listing.get('generated_at', '')
        updated_at = listing.get('updated_at', '')
        
        try:
            # Parse timestamps
            if generated_at:
                gen_time = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
                age_days = (datetime.now() - gen_time).days
                
                # Scoring: 1.0 for today, decreasing over time
                if age_days == 0:
                    return 1.0
                elif age_days <= 7:
                    return 0.9
                elif age_days <= 30:
                    return 0.7
                elif age_days <= 90:
                    return 0.5
                else:
                    return 0.3
            
            return 0.5  # Default if no timestamp
            
        except Exception:
            return 0.5
    
    async def _assess_linguistic_quality(self, description: str) -> float:
        """Assess the linguistic quality of the description"""
        if not description:
            return 0.0
        
        # Parse with spaCy
        doc = self.nlp(description)
        
        # Grammar and spelling (simplified - would use language tool in production)
        grammar_score = 0.9  # Placeholder - would use proper grammar checker
        
        # Sentence variety
        sentences = list(doc.sents)
        if len(sentences) > 1:
            # Check for sentence length variety
            lengths = [len(sent.text.split()) for sent in sentences]
            variety_score = 1.0 - (np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0)
        else:
            variety_score = 0.5
        
        # Vocabulary richness
        words = [token.text.lower() for token in doc if token.is_alpha]
        unique_ratio = len(set(words)) / len(words) if words else 0
        vocab_score = min(unique_ratio * 2, 1.0)  # Scale up as unique words are good
        
        # Professional tone indicators
        professional_words = [
            'specifications', 'compliance', 'certified', 'quality', 
            'standards', 'performance', 'efficiency', 'reliability'
        ]
        prof_score = sum(1 for word in professional_words if word in description.lower()) / len(professional_words)
        
        return (
            0.3 * grammar_score +
            0.2 * variety_score +
            0.3 * vocab_score +
            0.2 * prof_score
        )
    
    async def _assess_uniqueness(self, listing: Dict[str, Any], company_context: Dict[str, Any]) -> float:
        """Assess uniqueness compared to other listings"""
        # In production, this would compare against a database of existing listings
        # For now, we'll use heuristics
        
        description = listing.get('description', '')
        
        # Check for template indicators
        template_phrases = [
            "Key materials:", "Main products:", "Waste streams:",
            "Located in", "employees. Sustainability:"
        ]
        template_score = sum(1 for phrase in template_phrases if phrase in description) / len(template_phrases)
        uniqueness_from_template = 1.0 - template_score
        
        # Check for specific details vs generic content
        specific_numbers = len(re.findall(r'\d+\.?\d*', description))
        specific_score = min(specific_numbers / 5, 1.0)  # Optimal at 5+ specific numbers
        
        # Industry-specific terminology
        industry = company_context.get('industry', '')
        industry_terms = self._get_industry_specific_terms(industry)
        industry_term_score = sum(1 for term in industry_terms if term in description.lower()) / max(len(industry_terms), 1)
        
        return (
            0.4 * uniqueness_from_template +
            0.3 * specific_score +
            0.3 * industry_term_score
        )
    
    async def _assess_verification_status(self, listing: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        """Assess verification and validation status"""
        # Check for verification indicators
        verified_fields = ['verified', 'validated', 'confirmed', 'certified']
        verification_score = 0.0
        
        for field in verified_fields:
            if listing.get(field) or metadata.get(field):
                verification_score += 0.25
        
        # Check for third-party validation
        certifications = listing.get('certifications', [])
        if certifications:
            verification_score = min(verification_score + 0.2 * len(certifications), 1.0)
        
        # Data source reliability (if available)
        source_reliability = metadata.get('source_reliability', 0.5)
        
        return 0.7 * verification_score + 0.3 * source_reliability
    
    def _categorize_material(self, material_name: str, material_type: str) -> str:
        """Categorize material into knowledge base categories"""
        material_lower = material_name.lower()
        
        if any(metal in material_lower for metal in ['steel', 'iron', 'aluminum', 'copper', 'metal']):
            return 'metals'
        elif any(polymer in material_lower for polymer in ['plastic', 'polymer', 'resin', 'polyethylene', 'pvc']):
            return 'polymers'
        elif any(chem in material_lower for chem in ['acid', 'base', 'solvent', 'chemical', 'compound']):
            return 'chemicals'
        elif material_type.lower() == 'waste' or 'waste' in material_lower:
            return 'waste'
        else:
            return 'other'
    
    def _validate_unit_for_material(self, unit: str, material_category: str) -> float:
        """Validate if unit is appropriate for material category"""
        unit_mapping = {
            'metals': ['tons', 'kg', 'pounds', 'mt'],
            'polymers': ['tons', 'kg', 'pounds', 'pellets'],
            'chemicals': ['liters', 'gallons', 'kg', 'drums'],
            'waste': ['tons', 'cubic meters', 'm3', 'containers']
        }
        
        valid_units = unit_mapping.get(material_category, ['tons', 'kg'])
        return 1.0 if any(valid_unit in unit.lower() for valid_unit in valid_units) else 0.5
    
    def _validate_quantity_range(self, quantity: float, material_category: str, unit: str) -> float:
        """Validate if quantity is within reasonable range"""
        # Simplified validation - in production would use historical data
        if quantity <= 0:
            return 0.0
        
        # Convert to approximate tons for comparison
        if 'kg' in unit.lower():
            quantity_tons = quantity / 1000
        elif 'pounds' in unit.lower():
            quantity_tons = quantity / 2204.62
        else:
            quantity_tons = quantity
        
        # Reasonable ranges by category
        ranges = {
            'metals': (0.1, 10000),
            'polymers': (0.05, 5000),
            'chemicals': (0.01, 1000),
            'waste': (1, 50000)
        }
        
        min_qty, max_qty = ranges.get(material_category, (0.1, 10000))
        
        if min_qty <= quantity_tons <= max_qty:
            return 1.0
        elif quantity_tons < min_qty:
            return max(0.3, quantity_tons / min_qty)
        else:
            return max(0.3, max_qty / quantity_tons)
    
    def _validate_pricing(self, potential_value: float, material_name: str, industry: str) -> float:
        """Validate if pricing is realistic"""
        # Simplified - in production would use market data APIs
        if potential_value <= 0:
            return 0.0
        
        # Basic sanity checks
        if potential_value < 10:  # Too low for industrial materials
            return 0.3
        elif potential_value > 10000000:  # Suspiciously high
            return 0.3
        else:
            # Log-based scoring for reasonable range
            log_value = np.log10(potential_value)
            if 2 <= log_value <= 6:  # $100 to $1M range
                return 1.0
            else:
                return 0.7
    
    def _check_industry_material_alignment(self, material_name: str, industry: str) -> float:
        """Check if material aligns with industry"""
        # Industry-material mapping (simplified)
        industry_materials = {
            'petrochemical': ['polymer', 'plastic', 'resin', 'chemical', 'oil'],
            'steel': ['iron', 'steel', 'metal', 'slag', 'scrap'],
            'manufacturing': ['metal', 'plastic', 'component', 'material'],
            'food': ['organic', 'waste', 'packaging', 'ingredient'],
            'construction': ['cement', 'sand', 'aggregate', 'steel', 'glass']
        }
        
        industry_lower = industry.lower()
        material_lower = material_name.lower()
        
        for ind, materials in industry_materials.items():
            if ind in industry_lower:
                if any(mat in material_lower for mat in materials):
                    return 1.0
                else:
                    return 0.6
        
        return 0.7  # Default for unknown industries
    
    def _assess_geographic_relevance(self, material_name: str, location: str) -> float:
        """Assess geographic relevance of material"""
        # Simplified - in production would use regional market data
        if not location:
            return 0.5
        
        # MENA region bonus for certain materials
        if 'middle east' in location.lower() or 'gulf' in location.lower():
            if any(mat in material_name.lower() for mat in ['oil', 'gas', 'petrochemical', 'polymer']):
                return 1.0
        
        return 0.7  # Default score
    
    def _assess_quantity_company_size_fit(self, quantity: float, employee_count: int) -> float:
        """Assess if quantity fits company size"""
        if employee_count == 0:
            return 0.5
        
        # Rough heuristic: larger companies handle larger quantities
        expected_ratio = employee_count / 100  # Tons per 100 employees (rough estimate)
        
        if expected_ratio == 0:
            return 0.5
        
        actual_ratio = quantity / expected_ratio
        
        if 0.1 <= actual_ratio <= 10:  # Within order of magnitude
            return 1.0
        elif 0.01 <= actual_ratio <= 100:  # Within two orders
            return 0.7
        else:
            return 0.4
    
    def _get_industry_specific_terms(self, industry: str) -> List[str]:
        """Get industry-specific terminology"""
        industry_terms = {
            'petrochemical': ['cracker', 'distillation', 'catalyst', 'feedstock', 'naptha'],
            'steel': ['blast furnace', 'rolling mill', 'galvanized', 'coking', 'sinter'],
            'pharmaceutical': ['API', 'GMP', 'formulation', 'excipient', 'bioavailability'],
            'automotive': ['OEM', 'tier', 'just-in-time', 'assembly', 'powertrain'],
            'electronics': ['PCB', 'semiconductor', 'SMT', 'wafer', 'chip']
        }
        
        industry_lower = industry.lower()
        for ind, terms in industry_terms.items():
            if ind in industry_lower:
                return terms
        
        return []
    
    def _extract_neural_features(self, listing: Dict[str, Any], company_context: Dict[str, Any]) -> torch.Tensor:
        """Extract features for neural quality prediction"""
        features = []
        
        # Text features
        description = listing.get('description', '')
        features.extend([
            len(description),
            len(description.split()),
            description.count('.'),
            description.count(','),
            len(set(description.split()))  # Unique words
        ])
        
        # Numeric features
        features.extend([
            listing.get('quantity', 0),
            listing.get('potential_value', 0),
            company_context.get('employee_count', 0),
            company_context.get('sustainability_score', 0),
            len(listing.get('metadata', {}))
        ])
        
        # Categorical features (one-hot encoded)
        material_types = ['raw', 'processed', 'waste', 'specialty']
        material_type = listing.get('material_type', '').lower()
        for mt in material_types:
            features.append(1.0 if mt in material_type else 0.0)
        
        # Industry features (simplified one-hot)
        industries = ['petrochemical', 'manufacturing', 'steel', 'food', 'electronics']
        industry = company_context.get('industry', '').lower()
        for ind in industries:
            features.append(1.0 if ind in industry else 0.0)
        
        # Pad or truncate to fixed size
        target_size = 50
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _predict_quality_neural(self, features: torch.Tensor) -> float:
        """Use neural network to predict quality score"""
        try:
            self.quality_predictor.eval()
            with torch.no_grad():
                prediction = self.quality_predictor(features)
                return prediction.item()
        except Exception as e:
            self.logger.error(f"Neural prediction error: {e}")
            return 0.5
    
    def _determine_quality_grade(self, score: float) -> str:
        """Determine quality grade based on score"""
        if score >= 0.85:
            return QualityGrade.PREMIUM.value
        elif score >= 0.70:
            return QualityGrade.HIGH.value
        elif score >= 0.55:
            return QualityGrade.STANDARD.value
        elif score >= 0.40:
            return QualityGrade.BASIC.value
        else:
            return QualityGrade.LOW.value
    
    def _generate_quality_feedback(self, metrics: QualityMetrics, listing: Dict[str, Any]) -> List[str]:
        """Generate specific feedback based on quality metrics"""
        feedback = []
        
        if metrics.completeness_score < 0.7:
            feedback.append("Consider adding more details about specifications, certifications, or availability.")
        
        if metrics.technical_accuracy < 0.6:
            feedback.append("Include more technical specifications and material properties.")
        
        if metrics.market_alignment < 0.6:
            feedback.append("Align pricing and description better with current market conditions.")
        
        if metrics.sustainability_score < 0.5:
            feedback.append("Highlight sustainability aspects and environmental certifications.")
        
        if metrics.linguistic_quality < 0.7:
            feedback.append("Improve description clarity and use more professional terminology.")
        
        if metrics.uniqueness_score < 0.6:
            feedback.append("Make description more specific to avoid generic template-like content.")
        
        return feedback
    
    def _generate_improvement_suggestions(self, metrics: QualityMetrics, listing: Dict[str, Any]) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        
        # Find weakest areas
        metric_scores = {
            'completeness': metrics.completeness_score,
            'technical': metrics.technical_accuracy,
            'market': metrics.market_alignment,
            'sustainability': metrics.sustainability_score,
            'contextual': metrics.contextual_relevance,
            'linguistic': metrics.linguistic_quality,
            'uniqueness': metrics.uniqueness_score
        }
        
        # Sort by score (lowest first)
        weakest_areas = sorted(metric_scores.items(), key=lambda x: x[1])[:3]
        
        for area, score in weakest_areas:
            if area == 'completeness' and score < 0.8:
                suggestions.append("Add minimum order quantity, lead time, and payment terms.")
            elif area == 'technical' and score < 0.8:
                suggestions.append("Include material grade, purity levels, and test certificates.")
            elif area == 'market' and score < 0.8:
                suggestions.append("Update pricing based on current market rates and add demand indicators.")
            elif area == 'sustainability' and score < 0.8:
                suggestions.append("Add carbon footprint data and circular economy potential.")
            elif area == 'linguistic' and score < 0.8:
                suggestions.append("Rewrite description with varied sentence structure and professional tone.")
            elif area == 'uniqueness' and score < 0.8:
                suggestions.append("Add company-specific details and avoid generic templates.")
        
        return suggestions
    
    def _get_default_assessment(self) -> Dict[str, Any]:
        """Return default assessment in case of errors"""
        return {
            'quality_grade': QualityGrade.STANDARD.value,
            'overall_score': 0.5,
            'metrics': {
                'completeness': 0.5,
                'technical_accuracy': 0.5,
                'market_alignment': 0.5,
                'sustainability': 0.5,
                'contextual_relevance': 0.5,
                'innovation': 0.5,
                'data_freshness': 0.5,
                'linguistic_quality': 0.5,
                'uniqueness': 0.5,
                'verification': 0.5
            },
            'neural_score': 0.5,
            'feedback': ["Unable to perform complete assessment. Using default values."],
            'improvement_suggestions': ["Ensure all required data is provided for accurate assessment."]
        } 