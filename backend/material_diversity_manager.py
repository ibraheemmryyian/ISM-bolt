"""
Material Diversity Manager
Ensures diverse material generation and prevents excessive repetition across companies
"""

import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, Counter
import random
import numpy as np
from datetime import datetime, timedelta
import hashlib
import json
import asyncio
from dataclasses import dataclass
from enum import Enum

class DiversityStrategy(Enum):
    """Different strategies for ensuring diversity"""
    STRICT = "strict"          # No duplicates allowed
    MODERATE = "moderate"      # Limited duplicates with variations
    RELAXED = "relaxed"        # Allow duplicates but encourage diversity
    ADAPTIVE = "adaptive"      # Adjust based on industry and context

@dataclass
class MaterialUsageRecord:
    """Record of material usage"""
    material_name: str
    company_id: str
    company_name: str
    industry: str
    timestamp: datetime
    variations_applied: List[str]
    quality_grade: str
    description_hash: str

class MaterialDiversityManager:
    """
    Manages material diversity across companies to prevent excessive repetition
    """
    
    def __init__(self, strategy: DiversityStrategy = DiversityStrategy.MODERATE):
        self.logger = logging.getLogger(__name__)
        self.strategy = strategy
        
        # Track material usage
        self.material_usage: Dict[str, List[MaterialUsageRecord]] = defaultdict(list)
        self.company_materials: Dict[str, Set[str]] = defaultdict(set)
        self.industry_materials: Dict[str, Counter] = defaultdict(Counter)
        
        # Diversity parameters
        self.max_global_usage = self._get_max_usage_by_strategy()
        self.max_industry_usage = self._get_max_industry_usage()
        self.similarity_threshold = 0.8
        
        # Material alternatives database
        self.material_alternatives = self._initialize_alternatives()
        
        # Track recent generations for pattern breaking
        self.recent_generations = []
        self.pattern_window = 10
        
    def _get_max_usage_by_strategy(self) -> Dict[str, int]:
        """Get maximum usage limits based on strategy"""
        if self.strategy == DiversityStrategy.STRICT:
            return {"global": 1, "per_industry": 1}
        elif self.strategy == DiversityStrategy.MODERATE:
            return {"global": 5, "per_industry": 3}
        elif self.strategy == DiversityStrategy.RELAXED:
            return {"global": 10, "per_industry": 7}
        else:  # ADAPTIVE
            return {"global": 7, "per_industry": 5}
    
    def _get_max_industry_usage(self) -> Dict[str, int]:
        """Get industry-specific usage limits"""
        return {
            "petrochemical": 8,    # Higher limit due to standardized materials
            "steel": 6,            # Moderate limit
            "manufacturing": 7,    # Higher variety expected
            "food": 5,             # Lower limit for specialty
            "electronics": 4,      # Highly specialized
            "construction": 8,     # Common materials
            "default": 5
        }
    
    def _initialize_alternatives(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize material alternatives database"""
        return {
            # Petrochemical alternatives
            "ethylene": [
                {"name": "propylene", "similarity": 0.8, "properties": ["polymer feedstock", "olefin"]},
                {"name": "butylene", "similarity": 0.7, "properties": ["chemical intermediate"]},
                {"name": "styrene", "similarity": 0.6, "properties": ["aromatic", "monomer"]}
            ],
            "polypropylene": [
                {"name": "polyethylene", "similarity": 0.85, "properties": ["thermoplastic", "polymer"]},
                {"name": "PVC", "similarity": 0.7, "properties": ["polymer", "versatile"]},
                {"name": "PET", "similarity": 0.65, "properties": ["recyclable", "polymer"]}
            ],
            
            # Steel alternatives
            "steel scrap": [
                {"name": "iron ore fines", "similarity": 0.6, "properties": ["raw material", "ferrous"]},
                {"name": "recycled steel", "similarity": 0.9, "properties": ["sustainable", "ferrous"]},
                {"name": "steel billets", "similarity": 0.7, "properties": ["semi-finished", "steel"]}
            ],
            "slag": [
                {"name": "fly ash", "similarity": 0.7, "properties": ["industrial byproduct", "cementitious"]},
                {"name": "bottom ash", "similarity": 0.65, "properties": ["waste material", "aggregate"]},
                {"name": "mill scale", "similarity": 0.6, "properties": ["steel byproduct", "iron oxide"]}
            ],
            
            # Generic waste alternatives
            "industrial waste": [
                {"name": "process residues", "similarity": 0.8, "properties": ["byproduct", "recoverable"]},
                {"name": "manufacturing scraps", "similarity": 0.75, "properties": ["recyclable", "waste"]},
                {"name": "production offcuts", "similarity": 0.7, "properties": ["reusable", "waste"]}
            ]
        }
    
    async def check_material_diversity(
        self, 
        material_name: str, 
        company_id: str,
        company_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if material selection maintains diversity
        Returns suggestions for alternatives if needed
        """
        industry = company_context.get('industry', 'default').lower()
        
        # Check global usage
        global_usage = len(self.material_usage.get(material_name.lower(), []))
        max_global = self.max_global_usage["global"]
        
        # Check industry usage
        industry_usage = self.industry_materials[industry][material_name.lower()]
        max_industry = self.max_industry_usage.get(industry, self.max_industry_usage["default"])
        
        # Check company usage
        company_has_material = material_name.lower() in self.company_materials[company_id]
        
        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(
            global_usage, max_global, industry_usage, max_industry, company_has_material
        )
        
        # Determine if we need alternatives
        needs_alternative = diversity_score < 0.4
        
        # Get suggestions if needed
        suggestions = []
        if needs_alternative or diversity_score < 0.6:
            suggestions = await self._generate_alternative_suggestions(
                material_name, industry, company_context
            )
        
        return {
            "material_name": material_name,
            "diversity_score": diversity_score,
            "needs_alternative": needs_alternative,
            "global_usage": global_usage,
            "industry_usage": industry_usage,
            "suggestions": suggestions,
            "recommendation": self._get_recommendation(diversity_score, suggestions)
        }
    
    def _calculate_diversity_score(
        self,
        global_usage: int,
        max_global: int,
        industry_usage: int,
        max_industry: int,
        company_has_material: bool
    ) -> float:
        """Calculate diversity score (0-1, higher is better)"""
        # Penalize based on usage ratios
        global_penalty = min(global_usage / max_global, 1.0)
        industry_penalty = min(industry_usage / max_industry, 1.0)
        company_penalty = 1.0 if company_has_material else 0.0
        
        # Weight the penalties
        weights = {
            "global": 0.4,
            "industry": 0.3,
            "company": 0.3
        }
        
        total_penalty = (
            weights["global"] * global_penalty +
            weights["industry"] * industry_penalty +
            weights["company"] * company_penalty
        )
        
        # Convert penalty to score
        diversity_score = 1.0 - total_penalty
        
        # Apply strategy modifiers
        if self.strategy == DiversityStrategy.STRICT:
            diversity_score *= 0.8  # More conservative
        elif self.strategy == DiversityStrategy.RELAXED:
            diversity_score *= 1.2  # More lenient
        
        return max(0.0, min(1.0, diversity_score))
    
    async def _generate_alternative_suggestions(
        self,
        material_name: str,
        industry: str,
        company_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate alternative material suggestions"""
        suggestions = []
        
        # Check predefined alternatives
        material_lower = material_name.lower()
        if material_lower in self.material_alternatives:
            alternatives = self.material_alternatives[material_lower]
            for alt in alternatives:
                # Check if alternative is less used
                alt_usage = len(self.material_usage.get(alt["name"].lower(), []))
                if alt_usage < len(self.material_usage.get(material_lower, [])):
                    suggestions.append({
                        "material": alt["name"],
                        "similarity": alt["similarity"],
                        "properties": alt["properties"],
                        "usage_count": alt_usage,
                        "reason": "Lower usage count, similar properties"
                    })
        
        # Generate variations
        variations = self._generate_material_variations(material_name, industry, company_context)
        suggestions.extend(variations)
        
        # Generate industry-specific alternatives
        industry_alts = self._generate_industry_alternatives(material_name, industry)
        suggestions.extend(industry_alts)
        
        # Sort by relevance and diversity
        suggestions = sorted(suggestions, key=lambda x: (x.get("similarity", 0.5), -x.get("usage_count", 0)), reverse=True)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _generate_material_variations(
        self,
        material_name: str,
        industry: str,
        company_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate variations of the material name"""
        variations = []
        
        # Grade variations
        grades = ["technical grade", "industrial grade", "premium grade", "certified"]
        for grade in grades:
            variation = f"{grade} {material_name}"
            if variation.lower() not in self.material_usage:
                variations.append({
                    "material": variation,
                    "similarity": 0.9,
                    "properties": ["grade variation"],
                    "usage_count": 0,
                    "reason": "Grade specification adds uniqueness"
                })
        
        # Source variations
        if company_context.get('location'):
            location = company_context['location'].split(',')[0]
            sourced_variation = f"{location}-sourced {material_name}"
            if sourced_variation.lower() not in self.material_usage:
                variations.append({
                    "material": sourced_variation,
                    "similarity": 0.85,
                    "properties": ["regional variant"],
                    "usage_count": 0,
                    "reason": "Regional sourcing differentiation"
                })
        
        # Process variations
        processes = {
            "recycled": ["sustainable", "circular economy"],
            "refined": ["high purity", "processed"],
            "raw": ["unprocessed", "natural state"],
            "treated": ["enhanced", "modified"]
        }
        
        for process, properties in processes.items():
            if process not in material_name.lower():
                variation = f"{process} {material_name}"
                if variation.lower() not in self.material_usage:
                    variations.append({
                        "material": variation,
                        "similarity": 0.8,
                        "properties": properties,
                        "usage_count": 0,
                        "reason": f"Process differentiation: {process}"
                    })
        
        return variations
    
    def _generate_industry_alternatives(self, material_name: str, industry: str) -> List[Dict[str, Any]]:
        """Generate industry-specific alternative materials"""
        alternatives = []
        
        industry_materials = {
            "petrochemical": {
                "polymer": ["resin", "plastic compound", "polymer blend", "copolymer"],
                "chemical": ["intermediate", "feedstock", "derivative", "specialty chemical"],
                "waste": ["off-gas", "residue", "byproduct stream", "process waste"]
            },
            "steel": {
                "metal": ["alloy", "ferrous material", "steel grade", "metal composite"],
                "waste": ["mill scale", "metal fines", "furnace dust", "processing residue"],
                "scrap": ["recovered metal", "end-of-life steel", "demolition scrap", "production scrap"]
            },
            "manufacturing": {
                "material": ["engineered material", "composite", "substrate", "component material"],
                "waste": ["production waste", "defective products", "trim waste", "packaging waste"],
                "chemical": ["process chemical", "treatment chemical", "auxiliary material"]
            },
            "food": {
                "organic": ["bio-waste", "organic residue", "food byproduct", "agricultural waste"],
                "packaging": ["food-grade packaging", "recyclable packaging", "biodegradable material"],
                "ingredient": ["food additive", "processing aid", "nutritional component"]
            }
        }
        
        # Get relevant category for the material
        material_category = self._categorize_material(material_name)
        
        if industry in industry_materials and material_category in industry_materials[industry]:
            for alt_name in industry_materials[industry][material_category]:
                if alt_name.lower() not in self.material_usage:
                    alternatives.append({
                        "material": alt_name,
                        "similarity": 0.7,
                        "properties": [industry, material_category],
                        "usage_count": 0,
                        "reason": f"Industry-specific alternative for {industry}"
                    })
        
        return alternatives
    
    def _categorize_material(self, material_name: str) -> str:
        """Categorize material into broad categories"""
        material_lower = material_name.lower()
        
        if any(keyword in material_lower for keyword in ["polymer", "plastic", "resin", "polyethylene", "polypropylene"]):
            return "polymer"
        elif any(keyword in material_lower for keyword in ["chemical", "acid", "solvent", "compound"]):
            return "chemical"
        elif any(keyword in material_lower for keyword in ["waste", "scrap", "residue", "byproduct"]):
            return "waste"
        elif any(keyword in material_lower for keyword in ["metal", "steel", "iron", "alloy"]):
            return "metal"
        elif any(keyword in material_lower for keyword in ["organic", "food", "bio"]):
            return "organic"
        elif any(keyword in material_lower for keyword in ["packaging", "container", "bottle"]):
            return "packaging"
        else:
            return "material"
    
    def _get_recommendation(self, diversity_score: float, suggestions: List[Dict[str, Any]]) -> str:
        """Get recommendation based on diversity score"""
        if diversity_score >= 0.8:
            return "Proceed with original material - good diversity"
        elif diversity_score >= 0.6:
            return "Consider using suggested variation for better diversity"
        elif diversity_score >= 0.4:
            return "Recommend using alternative material from suggestions"
        else:
            return "Strongly recommend alternative - material is overused"
    
    async def apply_diversity_constraints(
        self,
        materials: List[str],
        company_id: str,
        company_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply diversity constraints to a list of materials
        Returns modified list with alternatives where needed
        """
        modified_materials = []
        
        for material in materials:
            # Check diversity
            diversity_check = await self.check_material_diversity(material, company_id, company_context)
            
            if diversity_check["needs_alternative"] and diversity_check["suggestions"]:
                # Use the best suggestion
                best_suggestion = diversity_check["suggestions"][0]
                modified_materials.append({
                    "original": material,
                    "selected": best_suggestion["material"],
                    "reason": best_suggestion["reason"],
                    "diversity_score": diversity_check["diversity_score"]
                })
                
                # Record usage of the alternative
                await self.record_material_usage(
                    best_suggestion["material"],
                    company_id,
                    company_context
                )
            else:
                # Use original material
                modified_materials.append({
                    "original": material,
                    "selected": material,
                    "reason": "Maintains diversity",
                    "diversity_score": diversity_check["diversity_score"]
                })
                
                # Record usage
                await self.record_material_usage(material, company_id, company_context)
        
        return modified_materials
    
    async def record_material_usage(
        self,
        material_name: str,
        company_id: str,
        company_context: Dict[str, Any],
        description: str = "",
        quality_grade: str = "STANDARD"
    ) -> None:
        """Record material usage for tracking"""
        material_lower = material_name.lower()
        industry = company_context.get('industry', 'default').lower()
        
        # Create usage record
        record = MaterialUsageRecord(
            material_name=material_name,
            company_id=company_id,
            company_name=company_context.get('name', 'Unknown'),
            industry=industry,
            timestamp=datetime.now(),
            variations_applied=[],
            quality_grade=quality_grade,
            description_hash=hashlib.md5(description.encode()).hexdigest() if description else ""
        )
        
        # Update tracking structures
        self.material_usage[material_lower].append(record)
        self.company_materials[company_id].add(material_lower)
        self.industry_materials[industry][material_lower] += 1
        
        # Update recent generations for pattern detection
        self.recent_generations.append(material_lower)
        if len(self.recent_generations) > self.pattern_window:
            self.recent_generations.pop(0)
    
    def get_diversity_report(self) -> Dict[str, Any]:
        """Generate diversity report"""
        # Calculate statistics
        total_unique_materials = len(self.material_usage)
        total_usage_count = sum(len(records) for records in self.material_usage.values())
        
        # Find most used materials
        usage_counts = {material: len(records) for material, records in self.material_usage.items()}
        most_used = sorted(usage_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Industry distribution
        industry_stats = {}
        for industry, counter in self.industry_materials.items():
            industry_stats[industry] = {
                "unique_materials": len(counter),
                "total_usage": sum(counter.values()),
                "most_common": counter.most_common(5)
            }
        
        # Pattern detection
        patterns = self._detect_patterns()
        
        return {
            "total_unique_materials": total_unique_materials,
            "total_usage_count": total_usage_count,
            "average_usage_per_material": total_usage_count / max(total_unique_materials, 1),
            "most_used_materials": most_used,
            "industry_statistics": industry_stats,
            "detected_patterns": patterns,
            "diversity_health": self._calculate_diversity_health()
        }
    
    def _detect_patterns(self) -> List[str]:
        """Detect repetitive patterns in material generation"""
        patterns = []
        
        # Check for repeated sequences
        if len(self.recent_generations) >= 3:
            # Look for repeated pairs
            pairs = [(self.recent_generations[i], self.recent_generations[i+1]) 
                     for i in range(len(self.recent_generations)-1)]
            pair_counts = Counter(pairs)
            
            for pair, count in pair_counts.items():
                if count >= 2:
                    patterns.append(f"Repeated sequence: {pair[0]} -> {pair[1]} ({count} times)")
        
        # Check for material dominance
        recent_counter = Counter(self.recent_generations)
        for material, count in recent_counter.items():
            if count >= self.pattern_window * 0.3:  # 30% of recent window
                patterns.append(f"Material '{material}' dominates recent generations ({count}/{self.pattern_window})")
        
        return patterns
    
    def _calculate_diversity_health(self) -> str:
        """Calculate overall diversity health"""
        if not self.material_usage:
            return "No data"
        
        # Calculate metrics
        unique_ratio = len(self.material_usage) / max(sum(len(r) for r in self.material_usage.values()), 1)
        max_usage = max(len(records) for records in self.material_usage.values()) if self.material_usage else 0
        
        # Determine health
        if unique_ratio > 0.7 and max_usage < 10:
            return "Excellent"
        elif unique_ratio > 0.5 and max_usage < 20:
            return "Good"
        elif unique_ratio > 0.3 and max_usage < 30:
            return "Fair"
        else:
            return "Poor - Consider increasing diversity"
    
    async def suggest_underused_materials(self, industry: str, count: int = 5) -> List[str]:
        """Suggest underused materials for an industry"""
        # Get all materials used in the industry
        industry_materials = self.industry_materials.get(industry.lower(), Counter())
        
        # Find materials used in other industries but not this one
        all_materials = set(self.material_usage.keys())
        industry_used = set(industry_materials.keys())
        unused_in_industry = all_materials - industry_used
        
        # Also include lightly used materials
        lightly_used = [mat for mat, count in industry_materials.items() if count <= 2]
        
        suggestions = list(unused_in_industry) + lightly_used
        
        # Sort by global usage (prefer less used globally)
        suggestions.sort(key=lambda x: len(self.material_usage.get(x, [])))
        
        return suggestions[:count] 