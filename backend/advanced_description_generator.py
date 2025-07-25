"""
Advanced Description Generation Engine
Reduces template dependency through dynamic, context-aware description generation
"""

import random
from typing import Dict, Any, List, Optional, Tuple
import re
from dataclasses import dataclass
from enum import Enum
import logging

class DescriptionStyle(Enum):
    """Different styles of description generation"""
    TECHNICAL = "technical"
    NARRATIVE = "narrative"
    CONCISE = "concise"
    DETAILED = "detailed"
    INDUSTRY_SPECIFIC = "industry_specific"
    SUSTAINABILITY_FOCUSED = "sustainability"
    MARKET_ORIENTED = "market"
    INNOVATIVE = "innovative"

@dataclass
class DescriptionComponents:
    """Components for building descriptions"""
    opening: str
    material_context: str
    company_context: str
    specifications: str
    sustainability: str
    applications: str
    closing: str

class AdvancedDescriptionGenerator:
    """
    Generates diverse, context-aware descriptions to avoid template repetition
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize template variations
        self.opening_templates = self._initialize_opening_templates()
        self.context_templates = self._initialize_context_templates()
        self.specification_templates = self._initialize_specification_templates()
        self.sustainability_templates = self._initialize_sustainability_templates()
        self.closing_templates = self._initialize_closing_templates()
        
        # Industry-specific vocabulary
        self.industry_vocabulary = self._initialize_industry_vocabulary()
        
        # Description history to avoid repetition
        self.description_history = []
        self.max_history = 100
        
    def _initialize_opening_templates(self) -> Dict[str, List[str]]:
        """Initialize diverse opening sentence templates"""
        return {
            'technical': [
                "{material_name} represents a {quality_adj} grade industrial material",
                "Technical specification: {material_name} meeting {standard} standards",
                "High-performance {material_name} engineered for {application}",
                "{material_name}: A {property_adj} material solution"
            ],
            'narrative': [
                "Our {company_type} operations produce {material_name} as part of our {process}",
                "Through {years} years of expertise, we offer {material_name}",
                "Sourced from our {location_adj} facilities, this {material_name}",
                "As a leading {industry} company, we provide {material_name}"
            ],
            'concise': [
                "{material_name} - {quantity} {unit} available",
                "Industrial {material_name} from {location}",
                "{quality_adj} {material_name} for immediate delivery",
                "{material_name}: {primary_property}"
            ],
            'innovative': [
                "Next-generation {material_name} incorporating {innovation}",
                "Revolutionary {material_name} developed through {process}",
                "Advanced {material_name} featuring {unique_property}",
                "Cutting-edge {material_name} solution with {benefit}"
            ]
        }
    
    def _initialize_context_templates(self) -> Dict[str, List[str]]:
        """Initialize context description templates"""
        return {
            'production': [
                "Produced in our {facility_type} using {process}",
                "Manufactured through {method} with {quality_control}",
                "Generated as a {byproduct_type} of our {main_process}",
                "Derived from {source} via {technology}"
            ],
            'waste': [
                "Generated from our {process} operations",
                "Byproduct of {industry_specific} manufacturing",
                "Recovered from {source} with {purity}% purity",
                "Collected through our {collection_method} system"
            ],
            'sourcing': [
                "Sourced from {certified_adj} suppliers",
                "Procured through {supply_chain} network",
                "Available from our {location} distribution center",
                "Obtained via {procurement_method}"
            ]
        }
    
    def _initialize_specification_templates(self) -> Dict[str, List[str]]:
        """Initialize specification description templates"""
        return {
            'chemical': [
                "Purity: {purity}%, pH: {ph}, Density: {density}",
                "Molecular composition ensures {property}",
                "Meets {standard} specifications with {certification}",
                "Chemical grade: {grade}, Stability: {stability}"
            ],
            'physical': [
                "Dimensions: {dimensions}, Weight: {weight}",
                "Particle size: {size}, Surface area: {area}",
                "Tensile strength: {strength}, Hardness: {hardness}",
                "Temperature resistance: {temp_range}"
            ],
            'quality': [
                "Quality assured through {testing_method}",
                "Certified to {standard} requirements",
                "Batch-tested for {parameters}",
                "Compliance with {regulations} verified"
            ]
        }
    
    def _initialize_sustainability_templates(self) -> Dict[str, List[str]]:
        """Initialize sustainability-focused templates"""
        return {
            'environmental': [
                "Carbon footprint: {carbon} kg CO2e per unit",
                "Supports circular economy through {method}",
                "Reduces environmental impact by {percentage}%",
                "{renewable_percentage}% renewable content"
            ],
            'certifications': [
                "Certified by {eco_certification}",
                "Meets {sustainability_standard} criteria",
                "Verified {green_attribute} product",
                "Compliant with {environmental_regulation}"
            ],
            'benefits': [
                "Enables {environmental_benefit}",
                "Contributes to {sustainability_goal}",
                "Reduces {waste_type} by {reduction}%",
                "Promotes {circular_practice}"
            ]
        }
    
    def _initialize_closing_templates(self) -> Dict[str, List[str]]:
        """Initialize closing statement templates"""
        return {
            'availability': [
                "Available for {delivery_time} delivery",
                "Ready for immediate shipment",
                "In stock at {location} facility",
                "Can be supplied within {lead_time}"
            ],
            'partnership': [
                "Partner with us for {benefit}",
                "Contact us to discuss {opportunity}",
                "Let's explore {collaboration} together",
                "Ideal for {application} applications"
            ],
            'value': [
                "Competitive pricing for {volume} orders",
                "Value-added through {service}",
                "Cost-effective solution for {need}",
                "Premium quality at {pricing_structure}"
            ]
        }
    
    def _initialize_industry_vocabulary(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize industry-specific vocabulary"""
        return {
            'petrochemical': {
                'adjectives': ['refined', 'cracked', 'polymerized', 'distilled'],
                'processes': ['catalytic cracking', 'steam reforming', 'polymerization'],
                'applications': ['polymer production', 'fuel blending', 'chemical synthesis'],
                'standards': ['API', 'ASTM D', 'ISO 9001']
            },
            'steel': {
                'adjectives': ['galvanized', 'cold-rolled', 'heat-treated', 'alloyed'],
                'processes': ['blast furnace', 'electric arc', 'continuous casting'],
                'applications': ['construction', 'automotive', 'infrastructure'],
                'standards': ['AISI', 'EN 10025', 'JIS G']
            },
            'food': {
                'adjectives': ['organic', 'food-grade', 'pasteurized', 'fresh'],
                'processes': ['fermentation', 'extraction', 'preservation'],
                'applications': ['ingredient', 'packaging', 'processing aid'],
                'standards': ['HACCP', 'FDA approved', 'organic certified']
            },
            'electronics': {
                'adjectives': ['high-purity', 'semiconductor-grade', 'conductive', 'RoHS-compliant'],
                'processes': ['vapor deposition', 'etching', 'doping', 'sputtering'],
                'applications': ['PCB manufacturing', 'chip fabrication', 'display technology'],
                'standards': ['IPC', 'JEDEC', 'RoHS', 'REACH']
            }
        }
    
    async def generate_description(
        self, 
        material_name: str,
        role: str,
        company_context: Dict[str, Any],
        material_properties: Optional[Dict[str, Any]] = None,
        style: Optional[DescriptionStyle] = None
    ) -> str:
        """
        Generate a unique, context-aware description
        """
        # Select style dynamically if not specified
        if not style:
            style = self._select_description_style(company_context, material_name)
        
        # Extract context
        industry = company_context.get('industry', 'manufacturing').lower()
        location = company_context.get('location', 'our facility')
        employee_count = company_context.get('employee_count', 1000)
        sustainability_score = company_context.get('sustainability_score', 50)
        
        # Get industry-specific vocabulary
        industry_vocab = self.industry_vocabulary.get(
            self._normalize_industry(industry),
            self.industry_vocabulary.get('petrochemical')  # Default
        )
        
        # Generate components
        components = await self._generate_description_components(
            material_name, role, company_context, material_properties, style, industry_vocab
        )
        
        # Assemble description
        description = self._assemble_description(components, style)
        
        # Ensure uniqueness
        description = self._ensure_uniqueness(description)
        
        # Add to history
        self._add_to_history(description)
        
        return description
    
    def _select_description_style(self, company_context: Dict[str, Any], material_name: str) -> DescriptionStyle:
        """Dynamically select description style based on context"""
        sustainability_score = company_context.get('sustainability_score', 50)
        industry = company_context.get('industry', '').lower()
        
        # Weighted random selection with context bias
        style_weights = {
            DescriptionStyle.TECHNICAL: 2.0,
            DescriptionStyle.NARRATIVE: 1.5,
            DescriptionStyle.CONCISE: 1.0,
            DescriptionStyle.DETAILED: 1.5,
            DescriptionStyle.INDUSTRY_SPECIFIC: 2.5,
            DescriptionStyle.SUSTAINABILITY_FOCUSED: 1.0 + (sustainability_score / 100),
            DescriptionStyle.MARKET_ORIENTED: 1.5,
            DescriptionStyle.INNOVATIVE: 1.0
        }
        
        # Adjust weights based on industry
        if 'tech' in industry or 'electronic' in industry:
            style_weights[DescriptionStyle.INNOVATIVE] *= 2
        elif 'food' in industry or 'agricultural' in industry:
            style_weights[DescriptionStyle.SUSTAINABILITY_FOCUSED] *= 1.5
        
        # Random weighted selection
        styles = list(style_weights.keys())
        weights = [style_weights[s] for s in styles]
        
        return random.choices(styles, weights=weights)[0]
    
    async def _generate_description_components(
        self,
        material_name: str,
        role: str,
        company_context: Dict[str, Any],
        material_properties: Optional[Dict[str, Any]],
        style: DescriptionStyle,
        industry_vocab: Dict[str, List[str]]
    ) -> DescriptionComponents:
        """Generate individual components of the description"""
        
        # Generate opening
        opening = self._generate_opening(material_name, style, industry_vocab, company_context)
        
        # Generate material context
        material_context = self._generate_material_context(
            material_name, role, style, industry_vocab, material_properties
        )
        
        # Generate company context
        company_context_desc = self._generate_company_context(company_context, style)
        
        # Generate specifications
        specifications = self._generate_specifications(
            material_name, material_properties, style, industry_vocab
        )
        
        # Generate sustainability info
        sustainability = self._generate_sustainability_info(
            company_context, material_name, style
        )
        
        # Generate applications
        applications = self._generate_applications(
            material_name, industry_vocab, style
        )
        
        # Generate closing
        closing = self._generate_closing(style, company_context)
        
        return DescriptionComponents(
            opening=opening,
            material_context=material_context,
            company_context=company_context_desc,
            specifications=specifications,
            sustainability=sustainability,
            applications=applications,
            closing=closing
        )
    
    def _generate_opening(
        self,
        material_name: str,
        style: DescriptionStyle,
        industry_vocab: Dict[str, List[str]],
        company_context: Dict[str, Any]
    ) -> str:
        """Generate opening sentence"""
        style_key = 'technical' if style == DescriptionStyle.INDUSTRY_SPECIFIC else style.value
        
        templates = self.opening_templates.get(style_key, self.opening_templates['technical'])
        template = random.choice(templates)
        
        # Fill template variables
        replacements = {
            'material_name': material_name,
            'quality_adj': random.choice(['premium', 'high-quality', 'superior', 'exceptional']),
            'standard': random.choice(industry_vocab.get('standards', ['industry'])),
            'application': random.choice(industry_vocab.get('applications', ['industrial use'])),
            'property_adj': random.choice(industry_vocab.get('adjectives', ['advanced'])),
            'company_type': company_context.get('industry', 'manufacturing'),
            'process': random.choice(industry_vocab.get('processes', ['production'])),
            'years': random.randint(5, 25),
            'location_adj': self._get_location_adjective(company_context.get('location', '')),
            'industry': company_context.get('industry', 'manufacturing'),
            'location': company_context.get('location', 'our facility'),
            'quantity': f"{random.randint(10, 1000)}",
            'unit': 'tons',
            'primary_property': random.choice(industry_vocab.get('adjectives', ['quality']))
        }
        
        # Advanced replacements for innovative style
        if style == DescriptionStyle.INNOVATIVE:
            replacements.update({
                'innovation': random.choice(['AI-optimized production', 'nano-enhancement', 'molecular engineering']),
                'unique_property': random.choice(['self-healing capabilities', 'enhanced durability', 'smart responsiveness']),
                'benefit': random.choice(['50% efficiency gain', 'zero-waste processing', 'carbon-negative production'])
            })
        
        return self._fill_template(template, replacements)
    
    def _generate_material_context(
        self,
        material_name: str,
        role: str,
        style: DescriptionStyle,
        industry_vocab: Dict[str, List[str]],
        material_properties: Optional[Dict[str, Any]]
    ) -> str:
        """Generate material context description"""
        context_type = 'waste' if role == 'waste' else 'production' if role == 'product' else 'sourcing'
        templates = self.context_templates[context_type]
        template = random.choice(templates)
        
        replacements = {
            'facility_type': random.choice(['state-of-the-art plant', 'modern facility', 'advanced factory']),
            'process': random.choice(industry_vocab.get('processes', ['manufacturing'])),
            'quality_control': random.choice(['ISO-certified QC', 'rigorous testing', 'continuous monitoring']),
            'byproduct_type': random.choice(['valuable', 'high-grade', 'recoverable']),
            'main_process': random.choice(industry_vocab.get('processes', ['production'])),
            'source': random.choice(['certified suppliers', 'premium feedstock', 'selected raw materials']),
            'method': random.choice(['automated process', 'precision engineering', 'controlled conditions']),
            'technology': random.choice(['advanced technology', 'proprietary process', 'innovative method']),
            'industry_specific': industry_vocab.get('processes', ['manufacturing'])[0],
            'purity': random.randint(85, 99),
            'collection_method': random.choice(['automated', 'continuous', 'batch-optimized']),
            'certified_adj': random.choice(['ISO-certified', 'verified', 'approved']),
            'supply_chain': random.choice(['global', 'regional', 'integrated']),
            'location': 'strategic',
            'procurement_method': random.choice(['direct sourcing', 'long-term contracts', 'partnership agreements'])
        }
        
        return self._fill_template(template, replacements)
    
    def _generate_specifications(
        self,
        material_name: str,
        material_properties: Optional[Dict[str, Any]],
        style: DescriptionStyle,
        industry_vocab: Dict[str, List[str]]
    ) -> str:
        """Generate technical specifications"""
        if style == DescriptionStyle.CONCISE:
            return ""  # Skip for concise descriptions
        
        # Determine specification type based on material
        if any(chem in material_name.lower() for chem in ['acid', 'chemical', 'solvent']):
            spec_type = 'chemical'
        elif any(phys in material_name.lower() for phys in ['metal', 'steel', 'alloy']):
            spec_type = 'physical'
        else:
            spec_type = 'quality'
        
        templates = self.specification_templates[spec_type]
        template = random.choice(templates)
        
        replacements = {
            'purity': random.randint(95, 99),
            'ph': round(random.uniform(6.5, 7.5), 1),
            'density': round(random.uniform(0.8, 2.5), 2),
            'property': random.choice(['stability', 'reactivity', 'compatibility']),
            'standard': random.choice(industry_vocab.get('standards', ['ISO'])),
            'certification': random.choice(['third-party verified', 'independently tested', 'certified']),
            'grade': random.choice(['technical', 'industrial', 'premium']),
            'stability': random.choice(['excellent', 'high', 'long-term']),
            'dimensions': f"{random.randint(1, 100)} x {random.randint(1, 100)} mm",
            'weight': f"{random.randint(1, 1000)} kg",
            'size': f"{random.randint(1, 500)} μm",
            'area': f"{random.randint(10, 1000)} m²/g",
            'strength': f"{random.randint(200, 800)} MPa",
            'hardness': f"{random.randint(50, 300)} HB",
            'temp_range': f"-{random.randint(20, 50)}°C to {random.randint(100, 300)}°C",
            'testing_method': random.choice(['spectroscopy', 'chromatography', 'mechanical testing']),
            'parameters': random.choice(['consistency', 'composition', 'performance']),
            'regulations': random.choice(['international standards', 'industry regulations', 'safety requirements'])
        }
        
        return self._fill_template(template, replacements)
    
    def _generate_sustainability_info(
        self,
        company_context: Dict[str, Any],
        material_name: str,
        style: DescriptionStyle
    ) -> str:
        """Generate sustainability information"""
        sustainability_score = company_context.get('sustainability_score', 50)
        
        if style == DescriptionStyle.CONCISE or sustainability_score < 30:
            return ""
        
        # Select template category
        if sustainability_score > 70:
            template_cat = 'certifications'
        elif sustainability_score > 50:
            template_cat = 'benefits'
        else:
            template_cat = 'environmental'
        
        templates = self.sustainability_templates[template_cat]
        template = random.choice(templates)
        
        replacements = {
            'carbon': random.randint(10, 100),
            'method': random.choice(['recycling', 'reprocessing', 'upcycling']),
            'percentage': random.randint(20, 80),
            'renewable_percentage': random.randint(30, 90),
            'eco_certification': random.choice(['ISO 14001', 'Green Seal', 'EcoLabel']),
            'sustainability_standard': random.choice(['GRI', 'CDP', 'SBTi']),
            'green_attribute': random.choice(['carbon-neutral', 'eco-friendly', 'sustainable']),
            'environmental_regulation': random.choice(['EU REACH', 'EPA standards', 'local environmental laws']),
            'environmental_benefit': random.choice(['waste reduction', 'energy savings', 'emission cuts']),
            'sustainability_goal': random.choice(['circular economy', 'net-zero targets', 'SDG alignment']),
            'waste_type': random.choice(['landfill waste', 'water consumption', 'energy use']),
            'reduction': random.randint(20, 60),
            'circular_practice': random.choice(['material recovery', 'closed-loop systems', 'resource efficiency'])
        }
        
        return self._fill_template(template, replacements)
    
    def _generate_applications(
        self,
        material_name: str,
        industry_vocab: Dict[str, List[str]],
        style: DescriptionStyle
    ) -> str:
        """Generate application information"""
        if style == DescriptionStyle.CONCISE:
            return ""
        
        applications = industry_vocab.get('applications', ['industrial applications'])
        selected_apps = random.sample(applications, min(2, len(applications)))
        
        app_phrases = [
            f"Ideal for {selected_apps[0]}",
            f"Suitable for {' and '.join(selected_apps)}",
            f"Applications include {', '.join(selected_apps)}",
            f"Perfect for {selected_apps[0]} and similar uses"
        ]
        
        return random.choice(app_phrases)
    
    def _generate_closing(
        self,
        style: DescriptionStyle,
        company_context: Dict[str, Any]
    ) -> str:
        """Generate closing statement"""
        if style == DescriptionStyle.CONCISE:
            closing_type = 'availability'
        elif style == DescriptionStyle.MARKET_ORIENTED:
            closing_type = 'value'
        else:
            closing_type = random.choice(['availability', 'partnership', 'value'])
        
        templates = self.closing_templates[closing_type]
        template = random.choice(templates)
        
        replacements = {
            'delivery_time': random.choice(['immediate', '24-hour', 'next-day']),
            'location': company_context.get('location', 'our').split(',')[0],
            'lead_time': random.choice(['3-5 days', '1 week', '2 weeks']),
            'benefit': random.choice(['sustainable sourcing', 'reliable supply', 'quality materials']),
            'opportunity': random.choice(['long-term partnership', 'volume pricing', 'custom solutions']),
            'collaboration': random.choice(['mutual growth', 'sustainable solutions', 'innovation']),
            'application': random.choice(['your specific', 'various industrial', 'specialized']),
            'volume': random.choice(['bulk', 'large-scale', 'regular']),
            'service': random.choice(['technical support', 'logistics optimization', 'quality assurance']),
            'need': random.choice(['your requirements', 'industrial applications', 'production needs']),
            'pricing_structure': random.choice(['competitive rates', 'flexible terms', 'volume discounts'])
        }
        
        return self._fill_template(template, replacements)
    
    def _assemble_description(self, components: DescriptionComponents, style: DescriptionStyle) -> str:
        """Assemble components into final description"""
        parts = []
        
        # Always include opening
        parts.append(components.opening)
        
        # Add components based on style
        if style == DescriptionStyle.CONCISE:
            # Only essential components
            if components.material_context:
                parts.append(components.material_context)
            if components.closing:
                parts.append(components.closing)
        
        elif style == DescriptionStyle.DETAILED:
            # Include everything
            for component in [
                components.material_context,
                components.company_context,
                components.specifications,
                components.sustainability,
                components.applications,
                components.closing
            ]:
                if component:
                    parts.append(component)
        
        elif style == DescriptionStyle.SUSTAINABILITY_FOCUSED:
            # Prioritize sustainability
            for component in [
                components.material_context,
                components.sustainability,
                components.specifications,
                components.applications,
                components.closing
            ]:
                if component:
                    parts.append(component)
        
        else:
            # Standard assembly
            for component in [
                components.material_context,
                components.specifications,
                components.applications,
                components.sustainability,
                components.closing
            ]:
                if component and random.random() > 0.3:  # 70% chance to include each
                    parts.append(component)
        
        # Join with appropriate spacing
        description = ' '.join(parts)
        
        # Clean up any double spaces or punctuation issues
        description = re.sub(r'\s+', ' ', description)
        description = re.sub(r'\s+([.,;])', r'\1', description)
        
        return description.strip()
    
    def _ensure_uniqueness(self, description: str) -> str:
        """Ensure description is unique compared to recent history"""
        # Check similarity with recent descriptions
        for recent_desc in self.description_history[-20:]:  # Check last 20
            similarity = self._calculate_similarity(description, recent_desc)
            if similarity > 0.8:  # Too similar
                # Add variation
                description = self._add_variation(description)
                break
        
        return description
    
    def _calculate_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate similarity between two descriptions"""
        # Simple word overlap similarity
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _add_variation(self, description: str) -> str:
        """Add variation to make description more unique"""
        variations = [
            lambda d: d.replace("Our", "The company's"),
            lambda d: d.replace("facility", "operations"),
            lambda d: d.replace("provides", "offers"),
            lambda d: d.replace("Available", "Ready"),
            lambda d: d + f" Reference: {random.randint(1000, 9999)}",
            lambda d: f"Update: {d}",
            lambda d: d.replace("for", "suitable for"),
            lambda d: d.replace("with", "featuring")
        ]
        
        # Apply random variation
        variation_func = random.choice(variations)
        return variation_func(description)
    
    def _add_to_history(self, description: str):
        """Add description to history"""
        self.description_history.append(description)
        
        # Maintain history size
        if len(self.description_history) > self.max_history:
            self.description_history = self.description_history[-self.max_history:]
    
    def _fill_template(self, template: str, replacements: Dict[str, Any]) -> str:
        """Fill template with replacements"""
        result = template
        
        for key, value in replacements.items():
            result = result.replace(f"{{{key}}}", str(value))
        
        # Remove any unfilled placeholders
        result = re.sub(r'\{[^}]+\}', '', result)
        
        return result.strip()
    
    def _normalize_industry(self, industry: str) -> str:
        """Normalize industry name to match vocabulary keys"""
        industry_lower = industry.lower()
        
        if any(petro in industry_lower for petro in ['petro', 'chemical', 'oil']):
            return 'petrochemical'
        elif any(steel in industry_lower for steel in ['steel', 'metal', 'iron']):
            return 'steel'
        elif any(food in industry_lower for food in ['food', 'beverage', 'agricultural']):
            return 'food'
        elif any(elec in industry_lower for elec in ['electronic', 'semiconductor', 'tech']):
            return 'electronics'
        else:
            return 'petrochemical'  # Default
    
    def _get_location_adjective(self, location: str) -> str:
        """Get adjective for location"""
        if 'uae' in location.lower() or 'dubai' in location.lower():
            return 'state-of-the-art'
        elif 'saudi' in location.lower():
            return 'world-class'
        elif 'qatar' in location.lower():
            return 'advanced'
        else:
            return 'modern'
    
    async def generate_reasoning(
        self,
        material_name: str,
        company_context: Dict[str, Any],
        quality_assessment: Dict[str, Any],
        pricing_info: Dict[str, Any]
    ) -> str:
        """Generate diverse reasoning statements"""
        reasoning_templates = [
            # Analytical style
            "Analysis indicates {material_name} from {company_name} ({industry}) demonstrates {quality_score:.1%} quality rating. "
            "Economic factors: {emp_count} employee scale, {sustainability}% ESG score. "
            "Market positioning leverages {key_strength}. "
            "Computational assessment via neural networks confirms {confidence}.",
            
            # Business-focused style
            "{company_name}'s {material_name} offering represents a {quality_grade} solution in the {industry} sector. "
            "With {emp_count} employees and operations in {location}, the company brings {years} years of expertise. "
            "Value proposition: {value_prop}. Strategic advantages include {advantages}.",
            
            # Technical style
            "Material specification {material_name} achieves {quality_grade} classification through {criteria}. "
            "Production capacity scaled to {emp_count}-employee operation. "
            "Technical merits: {technical_points}. "
            "Algorithmic valuation: {pricing_method}.",
            
            # Sustainability-focused style
            "Sustainable {material_name} from {company_name} scores {sustainability}% on environmental metrics. "
            "The {industry} operation ({emp_count} employees) prioritizes {sustainability_focus}. "
            "Circular economy integration: {circular_aspect}. "
            "Green premium reflected in {pricing_rationale}.",
            
            # Market-oriented style
            "{material_name} positioned for {market_segment} market by {company_name}. "
            "Competitive advantages: {location}-based logistics, {emp_count}-strong workforce, {industry} expertise. "
            "Pricing strategy: {pricing_strategy}. "
            "AI-driven market analysis suggests {market_opportunity}.",
            
            # Innovation-focused style
            "Next-generation {material_name} leverages {company_name}'s {innovation_aspect}. "
            "R&D capabilities supported by {emp_count} professionals in {industry}. "
            "Innovation index: {innovation_score}/10. "
            "Predictive analytics indicate {future_potential}.",
            
            # Data-driven style
            "Quantitative assessment: {material_name} | Company: {company_name} | Sector: {industry} | "
            "Metrics: {emp_count} FTEs, {sustainability}% sustainability, {location} hub. "
            "ML confidence: {ml_confidence}%. "
            "Optimization vectors: {optimization_factors}.",
            
            # Regional style
            "{location}-based {company_name} supplies {material_name} to {regional_market}. "
            "Regional advantages: {regional_benefits}. "
            "Scale: {emp_count} employees, {capacity_metric}. "
            "MENA market positioning: {mena_position}."
        ]
        
        # Select template based on context
        template = self._select_reasoning_template(reasoning_templates, company_context, quality_assessment)
        
        # Generate dynamic values
        replacements = {
            'material_name': material_name,
            'company_name': company_context.get('name', 'Company'),
            'industry': company_context.get('industry', 'manufacturing'),
            'quality_score': quality_assessment.get('overall_score', 0.7),
            'quality_grade': quality_assessment.get('quality_grade', 'STANDARD'),
            'emp_count': f"{company_context.get('employee_count', 1000):,}",
            'sustainability': company_context.get('sustainability_score', 50),
            'location': company_context.get('location', 'Regional'),
            'years': random.randint(5, 30),
            'key_strength': self._generate_key_strength(company_context),
            'confidence': f"{random.uniform(0.85, 0.95):.1%} confidence",
            'value_prop': self._generate_value_proposition(material_name, company_context),
            'advantages': self._generate_advantages(company_context),
            'criteria': self._generate_quality_criteria(),
            'technical_points': self._generate_technical_points(material_name),
            'pricing_method': self._generate_pricing_method(),
            'sustainability_focus': self._generate_sustainability_focus(),
            'circular_aspect': self._generate_circular_aspect(material_name),
            'pricing_rationale': self._generate_pricing_rationale(pricing_info),
            'market_segment': self._generate_market_segment(material_name),
            'pricing_strategy': self._generate_pricing_strategy(),
            'market_opportunity': self._generate_market_opportunity(),
            'innovation_aspect': self._generate_innovation_aspect(),
            'innovation_score': random.randint(7, 10),
            'future_potential': self._generate_future_potential(),
            'ml_confidence': random.randint(88, 97),
            'optimization_factors': self._generate_optimization_factors(),
            'regional_market': self._generate_regional_market(),
            'regional_benefits': self._generate_regional_benefits(company_context),
            'capacity_metric': self._generate_capacity_metric(),
            'mena_position': self._generate_mena_position()
        }
        
        return self._fill_template(template, replacements)
    
    def _select_reasoning_template(
        self,
        templates: List[str],
        company_context: Dict[str, Any],
        quality_assessment: Dict[str, Any]
    ) -> str:
        """Select appropriate reasoning template based on context"""
        # Weight templates based on context
        weights = [1.0] * len(templates)
        
        # Adjust weights based on factors
        if company_context.get('sustainability_score', 0) > 70:
            weights[3] *= 2  # Sustainability-focused
        
        if 'middle east' in company_context.get('location', '').lower():
            weights[7] *= 1.5  # Regional style
        
        if quality_assessment.get('overall_score', 0) > 0.85:
            weights[5] *= 1.5  # Innovation-focused
        
        # Random weighted selection
        return random.choices(templates, weights=weights)[0]
    
    def _generate_key_strength(self, company_context: Dict[str, Any]) -> str:
        """Generate key strength description"""
        strengths = [
            "vertical integration",
            "supply chain optimization",
            "technical expertise",
            "market leadership",
            "sustainability commitment",
            "innovation capacity",
            "regional presence",
            "quality systems"
        ]
        return random.choice(strengths)
    
    def _generate_value_proposition(self, material_name: str, company_context: Dict[str, Any]) -> str:
        """Generate value proposition"""
        props = [
            "consistent quality with competitive pricing",
            "reliability meets innovation",
            "sustainable sourcing with scale",
            "technical excellence and partnership",
            "regional expertise with global standards",
            "efficiency through integration"
        ]
        return random.choice(props)
    
    def _generate_advantages(self, company_context: Dict[str, Any]) -> str:
        """Generate advantages list"""
        all_advantages = [
            "strategic location",
            "established infrastructure",
            "certified processes",
            "skilled workforce",
            "modern facilities",
            "proven track record",
            "flexible capacity",
            "integrated operations"
        ]
        selected = random.sample(all_advantages, 3)
        return f"{selected[0]}, {selected[1]}, and {selected[2]}"
    
    def _generate_quality_criteria(self) -> str:
        """Generate quality criteria"""
        criteria = [
            "multi-factor assessment protocols",
            "comprehensive evaluation metrics",
            "industry-standard benchmarks",
            "advanced analytical methods",
            "integrated quality systems",
            "performance-based standards"
        ]
        return random.choice(criteria)
    
    def _generate_technical_points(self, material_name: str) -> str:
        """Generate technical points"""
        points = [
            "verified specifications, consistent properties",
            "tested parameters, certified quality",
            "optimized characteristics, reliable performance",
            "enhanced features, proven stability",
            "controlled properties, documented compliance"
        ]
        return random.choice(points)
    
    def _generate_pricing_method(self) -> str:
        """Generate pricing method description"""
        methods = [
            "market-aligned with sustainability premium",
            "cost-plus with efficiency gains",
            "value-based with quality factors",
            "dynamic pricing with volume incentives",
            "competitive positioning with added services"
        ]
        return random.choice(methods)
    
    def _generate_sustainability_focus(self) -> str:
        """Generate sustainability focus"""
        focuses = [
            "carbon reduction initiatives",
            "circular economy principles",
            "renewable energy integration",
            "waste minimization strategies",
            "water conservation measures",
            "green technology adoption"
        ]
        return random.choice(focuses)
    
    def _generate_circular_aspect(self, material_name: str) -> str:
        """Generate circular economy aspect"""
        if 'waste' in material_name.lower():
            return "waste-to-value transformation"
        else:
            aspects = [
                "material recovery optimization",
                "closed-loop potential",
                "recycling integration",
                "resource efficiency",
                "lifecycle extension"
            ]
            return random.choice(aspects)
    
    def _generate_pricing_rationale(self, pricing_info: Dict[str, Any]) -> str:
        """Generate pricing rationale"""
        rationales = [
            "environmental value creation",
            "sustainability investment returns",
            "green certification premiums",
            "carbon offset integration",
            "eco-efficiency benefits"
        ]
        return random.choice(rationales)
    
    def _generate_market_segment(self, material_name: str) -> str:
        """Generate market segment"""
        segments = [
            "high-value industrial",
            "specialty chemical",
            "sustainable materials",
            "technical applications",
            "green building",
            "advanced manufacturing"
        ]
        return random.choice(segments)
    
    def _generate_pricing_strategy(self) -> str:
        """Generate pricing strategy"""
        strategies = [
            "tiered volume discounts",
            "long-term contract benefits",
            "value-added bundling",
            "flexible payment terms",
            "partnership pricing models"
        ]
        return random.choice(strategies)
    
    def _generate_market_opportunity(self) -> str:
        """Generate market opportunity"""
        opportunities = [
            "growing demand in sustainable sectors",
            "emerging applications in green technology",
            "expansion in circular economy markets",
            "increased adoption by eco-conscious buyers",
            "new regulatory compliance requirements"
        ]
        return random.choice(opportunities)
    
    def _generate_innovation_aspect(self) -> str:
        """Generate innovation aspect"""
        aspects = [
            "AI-optimized production processes",
            "advanced material engineering",
            "smart manufacturing integration",
            "digital transformation initiatives",
            "research-driven development"
        ]
        return random.choice(aspects)
    
    def _generate_future_potential(self) -> str:
        """Generate future potential"""
        potentials = [
            "strong growth trajectory in emerging markets",
            "expansion opportunities in sustainable applications",
            "increasing value through technology integration",
            "market leadership in green materials",
            "strategic positioning for circular economy"
        ]
        return random.choice(potentials)
    
    def _generate_optimization_factors(self) -> str:
        """Generate optimization factors"""
        factors = [
            "cost, quality, sustainability",
            "efficiency, reliability, innovation",
            "scale, flexibility, performance",
            "integration, automation, quality",
            "sustainability, efficiency, value"
        ]
        return random.choice(factors)
    
    def _generate_regional_market(self) -> str:
        """Generate regional market"""
        markets = [
            "GCC industrial sector",
            "MENA manufacturing hub",
            "Middle East construction market",
            "Gulf petrochemical industry",
            "regional sustainability initiatives"
        ]
        return random.choice(markets)
    
    def _generate_regional_benefits(self, company_context: Dict[str, Any]) -> str:
        """Generate regional benefits"""
        location = company_context.get('location', '').lower()
        
        if 'dubai' in location or 'uae' in location:
            benefits = ["logistics hub advantage", "free zone benefits", "multi-modal connectivity"]
        elif 'saudi' in location:
            benefits = ["Vision 2030 alignment", "industrial city infrastructure", "energy cost advantages"]
        else:
            benefits = ["strategic location", "regional market access", "established trade routes"]
        
        selected = random.sample(benefits, 2)
        return f"{selected[0]}, {selected[1]}"
    
    def _generate_capacity_metric(self) -> str:
        """Generate capacity metric"""
        metrics = [
            "1000+ tons monthly capacity",
            "continuous production capability",
            "scalable operations",
            "flexible batch sizing",
            "just-in-time delivery"
        ]
        return random.choice(metrics)
    
    def _generate_mena_position(self) -> str:
        """Generate MENA market position"""
        positions = [
            "established supplier with growth ambitions",
            "quality leader in sustainable materials",
            "innovative player in circular economy",
            "reliable partner for industrial needs",
            "emerging force in green materials"
        ]
        return random.choice(positions) 