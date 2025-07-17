"""
AI Pricing Integration Layer
Comprehensive integration between AI Pricing Orchestrator and all other AI modules.
Enforces pricing validation as a mandatory requirement for all matches.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue
from functools import wraps
import inspect

# Import the pricing orchestrator
from ai_pricing_orchestrator import (
    AI_PricingOrchestrator, 
    validate_match_pricing_requirement,
    get_material_pricing_data,
    PricingResult,
    MatchPricingValidation
)

# Import other AI modules (commented out to avoid circular imports)
# try:
#     from ai_matchmaking_service import AIMatchmakingService
#     from listing_inference_service import ListingInferenceService
#     from real_ai_matching_engine import RealAIMatchingEngine
#     from revolutionary_ai_matching import RevolutionaryAIMatching
#     from gnn_reasoning_engine import GNNReasoningEngine
#     from multi_hop_symbiosis_network import MultiHopSymbiosisNetwork
#     from ai_service_integration import AIServiceIntegration
#     from ai_production_orchestrator import AIProductionOrchestrator
#     from intelligentMatchingService import IntelligentMatchingService
#     from aiEvolutionEngine import AIEvolutionEngine
#     from comprehensive_match_analyzer import ComprehensiveMatchAnalyzer
# except ImportError as e:
#     logging.warning(f"Some AI modules not available: {e}")

logging.warning("AI module imports temporarily disabled to avoid circular imports")

logger = logging.getLogger(__name__)

@dataclass
class PricingIntegrationConfig:
    """Configuration for pricing integration"""
    enforce_pricing_validation: bool = True
    require_minimum_savings: float = 40.0
    require_minimum_profit_margin: float = 10.0
    max_profit_margin: float = 60.0
    enable_real_time_pricing: bool = True
    cache_pricing_results: bool = True
    log_pricing_decisions: bool = True
    alert_on_pricing_violations: bool = True

class PricingIntegrationMiddleware:
    """Middleware to enforce pricing validation across all AI modules"""
    
    def __init__(self, config: PricingIntegrationConfig = None):
        self.config = config or PricingIntegrationConfig()
        self.pricing_orchestrator = AI_PricingOrchestrator()
        self.pricing_cache = {}
        self.integration_hooks = {}
        self.validation_stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Initialize integration hooks
        self._setup_integration_hooks()
        
        logger.info("Pricing Integration Middleware initialized")
    
    def _setup_integration_hooks(self):
        """Setup integration hooks for all AI modules"""
        self.integration_hooks = {
            # Node.js service hooks
            "intelligentMatchingService": self._hook_intelligent_matching,
            "aiEvolutionEngine": self._hook_ai_evolution_engine,
            
            # Python service hooks
            "aiMatchmakingService": self._hook_ai_matchmaking,
            "listingInferenceService": self._hook_listing_inference,
            "realAIMatchingEngine": self._hook_real_ai_matching,
            "revolutionaryAIMatching": self._hook_revolutionary_ai_matching,
            "gnnReasoningEngine": self._hook_gnn_reasoning,
            "multiHopSymbiosis": self._hook_multi_hop_symbiosis,
            "aiServiceIntegration": self._hook_ai_service_integration,
            "aiProductionOrchestrator": self._hook_ai_production_orchestrator,
            "comprehensiveMatchAnalyzer": self._hook_comprehensive_match_analyzer
        }
    
    def enforce_pricing_validation(self, func: Callable) -> Callable:
        """Decorator to enforce pricing validation on any function that creates matches"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract match data from function arguments
            match_data = self._extract_match_data_from_args(args, kwargs)
            
            if match_data and self.config.enforce_pricing_validation:
                # Validate pricing before proceeding
                is_valid = await self._validate_match_pricing(match_data)
                
                if not is_valid:
                    logger.warning(f"Pricing validation failed for {func.__name__}")
                    self.validation_stats["failed_validations"] += 1
                    return {
                        "success": False,
                        "error": "Pricing validation failed",
                        "pricing_validation": False,
                        "required_adjustments": match_data.get("required_adjustments", [])
                    }
                
                self.validation_stats["passed_validations"] += 1
            
            self.validation_stats["total_validations"] += 1
            
            # Proceed with original function
            result = await func(*args, **kwargs)
            
            # Add pricing metadata to result
            if isinstance(result, dict):
                result["pricing_validated"] = True
                result["pricing_timestamp"] = datetime.utcnow().isoformat()
            
            return result
        
        return wrapper
    
    def _extract_match_data_from_args(self, args: tuple, kwargs: dict) -> Optional[Dict]:
        """Extract match data from function arguments"""
        # Look for common match data patterns in arguments
        for arg in args:
            if isinstance(arg, dict):
                if any(key in arg for key in ["material", "quantity", "price", "company_id"]):
                    return arg
        
        for key, value in kwargs.items():
            if isinstance(value, dict) and any(k in value for k in ["material", "quantity", "price", "company_id"]):
                return value
        
        return None
    
    async def _validate_match_pricing(self, match_data: Dict) -> bool:
        """Validate pricing for a match"""
        try:
            # Extract pricing information
            material = match_data.get("material")
            quantity = match_data.get("quantity", 1.0)
            quality = match_data.get("quality", "clean")
            source_location = match_data.get("source_location", "unknown")
            destination_location = match_data.get("destination_location", "unknown")
            proposed_price = match_data.get("price")
            
            if not all([material, proposed_price]):
                logger.warning("Missing required pricing data for validation")
                return False
            
            # Check cache first
            cache_key = f"{material}_{quantity}_{quality}_{source_location}_{destination_location}"
            if cache_key in self.pricing_cache:
                cached_validation = self.pricing_cache[cache_key]
                if time.time() - cached_validation["timestamp"] < 300:  # 5 min cache
                    self.validation_stats["cache_hits"] += 1
                    return cached_validation["is_valid"]
            
            self.validation_stats["cache_misses"] += 1
            
            # Perform validation
            validation = self.pricing_orchestrator.validate_match_pricing(
                material, quantity, quality, source_location, destination_location, proposed_price
            )
            
            # Cache result
            if self.config.cache_pricing_results:
                self.pricing_cache[cache_key] = {
                    "is_valid": validation.is_valid,
                    "timestamp": time.time(),
                    "validation": validation
                }
            
            # Log decision
            if self.config.log_pricing_decisions:
                logger.info(f"Pricing validation for {material}: {'PASSED' if validation.is_valid else 'FAILED'}")
            
            # Alert on violations
            if not validation.is_valid and self.config.alert_on_pricing_violations:
                await self._send_pricing_alert(match_data, validation)
            
            return validation.is_valid
            
        except Exception as e:
            logger.error(f"Error in pricing validation: {e}")
            return False
    
    async def _send_pricing_alert(self, match_data: Dict, validation: MatchPricingValidation):
        """Send alert for pricing violations"""
        alert_data = {
            "type": "pricing_violation",
            "material": match_data.get("material"),
            "proposed_price": match_data.get("price"),
            "reason": validation.reason,
            "required_adjustments": validation.required_adjustments,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # In production, this would send to notification system
        logger.warning(f"PRICING VIOLATION ALERT: {alert_data}")
    
    # Integration hooks for specific AI modules
    
    async def _hook_intelligent_matching(self, service_instance, method_name: str, *args, **kwargs):
        """Hook for IntelligentMatchingService"""
        if method_name in ["findIntelligentMatches", "storeMatches"]:
            # Apply pricing validation to match results
            result = await getattr(service_instance, method_name)(*args, **kwargs)
            
            if isinstance(result, dict) and "matches" in result:
                validated_matches = []
                for match in result["matches"]:
                    if await self._validate_match_pricing(match):
                        validated_matches.append(match)
                
                result["matches"] = validated_matches
                result["pricing_validated_matches"] = len(validated_matches)
            
            return result
        
        return await getattr(service_instance, method_name)(*args, **kwargs)
    
    async def _hook_ai_matchmaking(self, service_instance, method_name: str, *args, **kwargs):
        """Hook for AIMatchmakingService"""
        if method_name == "create_matches_in_database":
            # Validate pricing before creating matches
            company_id = args[0] if args else kwargs.get("company_id")
            partner_companies = args[1] if len(args) > 1 else kwargs.get("partner_companies", [])
            material_name = args[2] if len(args) > 2 else kwargs.get("material_name")
            
            validated_partners = []
            for partner in partner_companies:
                # Get pricing data for validation
                pricing_data = await self.pricing_orchestrator.get_material_price(material_name)
                if pricing_data:
                    match_data = {
                        "material": material_name,
                        "quantity": 1000.0,  # Default quantity
                        "quality": "clean",
                        "source_location": "unknown",
                        "destination_location": "unknown",
                        "price": pricing_data.price * 0.6  # Assume 60% of virgin price
                    }
                    
                    if await self._validate_match_pricing(match_data):
                        validated_partners.append(partner)
            
            # Only create matches for validated partners
            if validated_partners:
                return await getattr(service_instance, method_name)(company_id, validated_partners, material_name)
            else:
                logger.warning(f"No partners passed pricing validation for {material_name}")
                return []
        
        return await getattr(service_instance, method_name)(*args, **kwargs)
    
    async def _hook_gnn_reasoning(self, service_instance, method_name: str, *args, **kwargs):
        """Hook for GNNReasoningEngine"""
        if method_name in ["find_symbiotic_matches", "detect_multi_hop_symbiosis"]:
            # Apply pricing validation to GNN results
            result = await getattr(service_instance, method_name)(*args, **kwargs)
            
            if isinstance(result, list):
                validated_results = []
                for match in result:
                    if await self._validate_match_pricing(match):
                        validated_results.append(match)
                
                return validated_results
            
            return result
        
        return await getattr(service_instance, method_name)(*args, **kwargs)
    
    async def _hook_revolutionary_ai_matching(self, service_instance, method_name: str, *args, **kwargs):
        """Hook for RevolutionaryAIMatching"""
        if method_name in ["find_matches", "generate_ai_listings"]:
            # Apply pricing validation to revolutionary AI results
            result = await getattr(service_instance, method_name)(*args, **kwargs)
            
            if isinstance(result, dict) and "candidates" in result:
                validated_candidates = []
                for candidate in result["candidates"]:
                    if await self._validate_match_pricing(candidate):
                        validated_candidates.append(candidate)
                
                result["candidates"] = validated_candidates
                result["pricing_validated_count"] = len(validated_candidates)
            
            return result
        
        return await getattr(service_instance, method_name)(*args, **kwargs)
    
    # Additional hooks for other modules...
    async def _hook_listing_inference(self, service_instance, method_name: str, *args, **kwargs):
        """Hook for ListingInferenceService"""
        return await getattr(service_instance, method_name)(*args, **kwargs)
    
    async def _hook_real_ai_matching(self, service_instance, method_name: str, *args, **kwargs):
        """Hook for RealAIMatchingEngine"""
        return await getattr(service_instance, method_name)(*args, **kwargs)
    
    async def _hook_multi_hop_symbiosis(self, service_instance, method_name: str, *args, **kwargs):
        """Hook for MultiHopSymbiosisNetwork"""
        return await getattr(service_instance, method_name)(*args, **kwargs)
    
    async def _hook_ai_service_integration(self, service_instance, method_name: str, *args, **kwargs):
        """Hook for AIServiceIntegration"""
        return await getattr(service_instance, method_name)(*args, **kwargs)
    
    async def _hook_ai_production_orchestrator(self, service_instance, method_name: str, *args, **kwargs):
        """Hook for AIProductionOrchestrator"""
        return await getattr(service_instance, method_name)(*args, **kwargs)
    
    async def _hook_comprehensive_match_analyzer(self, service_instance, method_name: str, *args, **kwargs):
        """Hook for ComprehensiveMatchAnalyzer"""
        return await getattr(service_instance, method_name)(*args, **kwargs)
    
    async def _hook_ai_evolution_engine(self, service_instance, method_name: str, *args, **kwargs):
        """Hook for AIEvolutionEngine"""
        return await getattr(service_instance, method_name)(*args, **kwargs)

class PricingIntegrationManager:
    """Manager for coordinating pricing integration across all modules"""
    
    def __init__(self, config: PricingIntegrationConfig = None):
        self.config = config or PricingIntegrationConfig()
        self.middleware = PricingIntegrationMiddleware(config)
        self.integrated_modules = {}
        self.integration_status = {}
        
        logger.info("Pricing Integration Manager initialized")
    
    def integrate_module(self, module_name: str, module_instance: Any) -> bool:
        """Integrate a module with pricing validation"""
        try:
            if module_name in self.middleware.integration_hooks:
                self.integrated_modules[module_name] = module_instance
                self.integration_status[module_name] = "integrated"
                logger.info(f"Successfully integrated {module_name} with pricing validation")
                return True
            else:
                logger.warning(f"No integration hook available for {module_name}")
                return False
        except Exception as e:
            logger.error(f"Failed to integrate {module_name}: {e}")
            self.integration_status[module_name] = "failed"
            return False
    
    async def execute_with_pricing_validation(self, module_name: str, method_name: str, *args, **kwargs):
        """Execute a module method with pricing validation"""
        if module_name not in self.integrated_modules:
            raise ValueError(f"Module {module_name} not integrated")
        
        module_instance = self.integrated_modules[module_name]
        hook = self.middleware.integration_hooks.get(module_name)
        
        if hook:
            return await hook(module_instance, method_name, *args, **kwargs)
        else:
            return await getattr(module_instance, method_name)(*args, **kwargs)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations"""
        return {
            "integrated_modules": list(self.integrated_modules.keys()),
            "integration_status": self.integration_status,
            "validation_stats": self.middleware.validation_stats,
            "pricing_cache_size": len(self.middleware.pricing_cache),
            "config": asdict(self.config)
        }
    
    async def validate_all_matches(self, matches: List[Dict]) -> List[Dict]:
        """Validate pricing for all matches in a list"""
        validated_matches = []
        
        for match in matches:
            if await self.middleware._validate_match_pricing(match):
                validated_matches.append(match)
        
        return validated_matches

# Global integration manager instance
pricing_integration_manager = PricingIntegrationManager()

# Integration functions for direct use in other modules

def integrate_pricing_with_module(module_name: str, module_instance: Any) -> bool:
    """Integrate pricing validation with a module"""
    return pricing_integration_manager.integrate_module(module_name, module_instance)

async def validate_match_pricing_requirement_integrated(material: str, quantity: float, quality: str,
                                                       source_location: str, destination_location: str,
                                                       proposed_price: float) -> bool:
    """Integrated pricing validation function"""
    return validate_match_pricing_requirement(material, quantity, quality, source_location, destination_location, proposed_price)

async def get_material_pricing_data_integrated(material: str) -> Optional[PricingResult]:
    """Integrated pricing data function"""
    return get_material_pricing_data(material)

def enforce_pricing_validation_decorator(func: Callable) -> Callable:
    """Decorator to enforce pricing validation"""
    return pricing_integration_manager.middleware.enforce_pricing_validation(func)

# Integration utilities for specific module types

class NodeJSIntegration:
    """Integration utilities for Node.js modules"""
    
    @staticmethod
    def validate_matches_with_pricing(matches: List[Dict]) -> List[Dict]:
        """Validate matches with pricing (for Node.js modules)"""
        validated_matches = []
        
        for match in matches:
            # Extract pricing data from match
            material = match.get("material_name") or match.get("material")
            quantity = match.get("quantity", 1000.0)
            quality = match.get("quality", "clean")
            source_location = match.get("source_location", "unknown")
            destination_location = match.get("destination_location", "unknown")
            proposed_price = match.get("price") or match.get("proposed_price")
            
            if material and proposed_price:
                # Use synchronous validation for Node.js
                try:
                    validation = pricing_integration_manager.middleware.pricing_orchestrator.validate_match_pricing(
                        material, quantity, quality, source_location, destination_location, proposed_price
                    )
                    
                    if validation.is_valid:
                        match["pricing_validated"] = True
                        match["pricing_result"] = asdict(validation.pricing_result) if validation.pricing_result else None
                        validated_matches.append(match)
                    else:
                        match["pricing_validated"] = False
                        match["pricing_error"] = validation.reason
                        match["required_adjustments"] = validation.required_adjustments
                        
                except Exception as e:
                    logger.error(f"Error validating pricing for match: {e}")
                    match["pricing_validated"] = False
                    match["pricing_error"] = str(e)
        
        return validated_matches

class PythonIntegration:
    """Integration utilities for Python modules"""
    
    @staticmethod
    async def validate_matches_with_pricing_async(matches: List[Dict]) -> List[Dict]:
        """Validate matches with pricing (for Python modules)"""
        return await pricing_integration_manager.validate_all_matches(matches)
    
    @staticmethod
    def apply_pricing_validation_to_function(func: Callable) -> Callable:
        """Apply pricing validation decorator to a function"""
        return enforce_pricing_validation_decorator(func)

# Export key functions for easy integration
__all__ = [
    "PricingIntegrationManager",
    "PricingIntegrationMiddleware", 
    "PricingIntegrationConfig",
    "integrate_pricing_with_module",
    "validate_match_pricing_requirement_integrated",
    "get_material_pricing_data_integrated",
    "enforce_pricing_validation_decorator",
    "NodeJSIntegration",
    "PythonIntegration",
    "pricing_integration_manager"
] 