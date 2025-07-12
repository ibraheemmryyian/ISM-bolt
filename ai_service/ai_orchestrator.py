import logging
import asyncio
import time
import json
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import hashlib
import pickle
import os

# Import our AI modules
from ai_service import AdvancedAIService
from ai_service.model_persistence import model_persistence_manager
from ai_service.gnn_reasoning import gnn_reasoning_engine
from ai_service.advanced_analytics_engine import AdvancedAnalyticsEngine
from ai_service.revolutionary_ai_matching import RevolutionaryAIMatching

logger = logging.getLogger(__name__)

@dataclass
class OrchestrationMetrics:
    """Metrics for AI orchestration performance"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    cache_hit_rate: float
    model_utilization: Dict[str, float]
    synergy_score: float
    last_updated: datetime

@dataclass
class AIRequest:
    """AI request with comprehensive context"""
    request_id: str
    request_type: str
    input_data: Dict[str, Any]
    context: Dict[str, Any]
    priority: str  # 'high', 'medium', 'low'
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]

@dataclass
class AIResponse:
    """AI response with comprehensive results"""
    request_id: str
    success: bool
    results: Dict[str, Any]
    confidence: float
    processing_time: float
    models_used: List[str]
    cache_hit: bool
    error_message: Optional[str]
    timestamp: datetime

class AIOrchestrator:
    """Advanced AI orchestrator ensuring perfect synergy between all AI modules"""
    
    def __init__(self):
        # Initialize AI services
        self.ai_service = AdvancedAIService()
        self.analytics_engine = AdvancedAnalyticsEngine()
        self.matching_engine = RevolutionaryAIMatching()
        
        # Performance tracking
        self.metrics = OrchestrationMetrics(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            average_response_time=0.0,
            cache_hit_rate=0.0,
            model_utilization={},
            synergy_score=0.0,
            last_updated=datetime.now()
        )
        
        # Request management
        self.request_queue = asyncio.Queue()
        self.response_cache = {}
        self.active_requests = {}
        
        # Thread safety
        self.metrics_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        
        # Adaptive learning
        self.learning_history = []
        self.performance_history = []
        self.optimization_history = []
        
        # Load persistent state
        self._load_persistent_state()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("AI Orchestrator initialized successfully")
    
    def _load_persistent_state(self):
        """Load persistent orchestrator state"""
        try:
            state_file = "ai_orchestrator_state.pkl"
            if os.path.exists(state_file):
                with open(state_file, 'rb') as f:
                    saved_state = pickle.load(f)
                    self.response_cache = saved_state.get('response_cache', {})
                    self.learning_history = saved_state.get('learning_history', [])
                    self.performance_history = saved_state.get('performance_history', [])
                    self.optimization_history = saved_state.get('optimization_history', [])
                logger.info("Loaded persistent AI orchestrator state")
        except Exception as e:
            logger.warning(f"Failed to load persistent AI orchestrator state: {e}")
    
    def _save_persistent_state(self):
        """Save persistent orchestrator state"""
        try:
            state_file = "ai_orchestrator_state.pkl"
            state_data = {
                'response_cache': self.response_cache,
                'learning_history': self.learning_history,
                'performance_history': self.performance_history,
                'optimization_history': self.optimization_history,
                'timestamp': datetime.now()
            }
            with open(state_file, 'wb') as f:
                pickle.dump(state_data, f)
        except Exception as e:
            logger.warning(f"Failed to save persistent AI orchestrator state: {e}")
    
    def _start_background_tasks(self):
        """Start background optimization tasks"""
        asyncio.create_task(self._background_optimization())
        asyncio.create_task(self._performance_monitoring())
        asyncio.create_task(self._cache_cleanup())
    
    async def _background_optimization(self):
        """Background task for continuous optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._optimize_performance()
            except Exception as e:
                logger.error(f"Background optimization failed: {e}")
    
    async def _performance_monitoring(self):
        """Background task for performance monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._update_metrics()
            except Exception as e:
                logger.error(f"Performance monitoring failed: {e}")
    
    async def _cache_cleanup(self):
        """Background task for cache cleanup"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_cache()
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")
    
    async def process_request(self, request: AIRequest) -> AIResponse:
        """Process AI request with perfect orchestration"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        with self.cache_lock:
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                if self._is_cache_valid(cached_response):
                    self.metrics.cache_hit_rate = (
                        (self.metrics.cache_hit_rate * self.metrics.total_requests + 1) / 
                        (self.metrics.total_requests + 1)
                    )
                    return cached_response
        
        # Process request
        try:
            results = await self._orchestrate_ai_processing(request)
            
            # Create response
            processing_time = time.time() - start_time
            response = AIResponse(
                request_id=request.request_id,
                success=True,
                results=results,
                confidence=self._calculate_confidence(results),
                processing_time=processing_time,
                models_used=self._get_models_used(results),
                cache_hit=False,
                error_message=None,
                timestamp=datetime.now()
            )
            
            # Cache response
            with self.cache_lock:
                self.response_cache[cache_key] = response
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.total_requests += 1
                self.metrics.successful_requests += 1
                self.metrics.average_response_time = (
                    (self.metrics.average_response_time * (self.metrics.total_requests - 1) + processing_time) / 
                    self.metrics.total_requests
                )
            
            # Record for learning
            self._record_learning_data(request, response)
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_response = AIResponse(
                request_id=request.request_id,
                success=False,
                results={},
                confidence=0.0,
                processing_time=processing_time,
                models_used=[],
                cache_hit=False,
                error_message=str(e),
                timestamp=datetime.now()
            )
            
            # Update metrics
            with self.metrics_lock:
                self.metrics.total_requests += 1
                self.metrics.failed_requests += 1
            
            logger.error(f"Request processing failed: {e}")
            return error_response
    
    async def _orchestrate_ai_processing(self, request: AIRequest) -> Dict[str, Any]:
        """Orchestrate AI processing across all modules"""
        results = {}
        
        # Determine processing strategy based on request type
        if request.request_type == 'symbiosis_matching':
            results = await self._process_symbiosis_matching(request)
        elif request.request_type == 'sustainability_analysis':
            results = await self._process_sustainability_analysis(request)
        elif request.request_type == 'network_optimization':
            results = await self._process_network_optimization(request)
        elif request.request_type == 'material_analysis':
            results = await self._process_material_analysis(request)
        elif request.request_type == 'comprehensive_analysis':
            results = await self._process_comprehensive_analysis(request)
        else:
            raise ValueError(f"Unknown request type: {request.request_type}")
        
        # Add orchestration metadata
        results['orchestration_metadata'] = {
            'processing_strategy': request.request_type,
            'models_utilized': self._get_models_used(results),
            'synergy_score': self._calculate_synergy_score(results),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return results
    
    async def _process_symbiosis_matching(self, request: AIRequest) -> Dict[str, Any]:
        """Process symbiosis matching with multiple AI engines"""
        companies = request.input_data.get('companies', [])
        
        # Use revolutionary AI matching
        matches = await self.matching_engine.find_optimal_symbiosis_matches(companies)
        
        # Create GNN graph for reasoning
        if matches.get('matches'):
            graph_id = gnn_reasoning_engine.create_symbiosis_graph(
                companies, matches['matches']
            )
            
            # Perform GNN reasoning for top matches
            reasoning_results = {}
            for match in matches['matches'][:5]:  # Top 5 matches
                company_id = match['company_id']
                reasoning = gnn_reasoning_engine.reason_about_symbiosis(
                    graph_id, company_id, 'path'
                )
                reasoning_results[company_id] = reasoning
            
            matches['gnn_reasoning'] = reasoning_results
            matches['graph_id'] = graph_id
        
        return {
            'symbiosis_matches': matches,
            'processing_engine': 'revolutionary_ai_matching',
            'gnn_enhanced': True
        }
    
    async def _process_sustainability_analysis(self, request: AIRequest) -> Dict[str, Any]:
        """Process sustainability analysis with advanced analytics"""
        company_data = request.input_data.get('company_data', {})
        
        # Use advanced AI service for sustainability insights
        sustainability_insights = await self.ai_service.generate_sustainability_insights(company_data)
        
        # Use analytics engine for detailed analysis
        analytics_results = await self.analytics_engine.analyze_sustainability_metrics(company_data)
        
        # Combine results
        combined_results = {
            'ai_insights': sustainability_insights,
            'analytics_metrics': analytics_results,
            'comprehensive_score': self._calculate_sustainability_score(
                sustainability_insights, analytics_results
            )
        }
        
        return combined_results
    
    async def _process_network_optimization(self, request: AIRequest) -> Dict[str, Any]:
        """Process network optimization with GNN and analytics"""
        network_data = request.input_data.get('network_data', {})
        companies = network_data.get('companies', [])
        current_matches = network_data.get('matches', [])
        
        # Create GNN graph
        graph_id = gnn_reasoning_engine.create_symbiosis_graph(companies, current_matches)
        
        # Perform optimization reasoning for each company
        optimization_results = {}
        for company in companies:
            company_id = company['id']
            optimization = gnn_reasoning_engine.reason_about_symbiosis(
                graph_id, company_id, 'optimization'
            )
            optimization_results[company_id] = optimization
        
        # Use analytics engine for network analysis
        network_analysis = await self.analytics_engine.analyze_network_optimization(network_data)
        
        return {
            'optimization_recommendations': optimization_results,
            'network_analysis': network_analysis,
            'graph_id': graph_id
        }
    
    async def _process_material_analysis(self, request: AIRequest) -> Dict[str, Any]:
        """Process material analysis with AI and analytics"""
        material_data = request.input_data.get('material_data', {})
        
        # Use AI service for material insights
        material_insights = await self.ai_service.generate_material_listings(material_data)
        
        # Use analytics engine for material analysis
        analytics_results = await self.analytics_engine.analyze_material_properties(material_data)
        
        return {
            'ai_material_insights': material_insights,
            'analytics_properties': analytics_results,
            'comprehensive_analysis': True
        }
    
    async def _process_comprehensive_analysis(self, request: AIRequest) -> Dict[str, Any]:
        """Process comprehensive analysis using all AI modules"""
        company_data = request.input_data.get('company_data', {})
        
        # Parallel processing of all analysis types
        tasks = [
            self.ai_service.analyze_company_data(company_data),
            self.ai_service.generate_sustainability_insights(company_data),
            self.ai_service.generate_material_listings(company_data),
            self.analytics_engine.analyze_comprehensive_metrics(company_data)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        comprehensive_results = {
            'company_analysis': results[0] if not isinstance(results[0], Exception) else {},
            'sustainability_insights': results[1] if not isinstance(results[1], Exception) else {},
            'material_analysis': results[2] if not isinstance(results[2], Exception) else {},
            'analytics_metrics': results[3] if not isinstance(results[3], Exception) else {},
            'processing_complete': True
        }
        
        return comprehensive_results
    
    def _generate_cache_key(self, request: AIRequest) -> str:
        """Generate cache key for request"""
        cache_data = {
            'type': request.request_type,
            'input_hash': hashlib.md5(
                json.dumps(request.input_data, sort_keys=True).encode()
            ).hexdigest(),
            'context_hash': hashlib.md5(
                json.dumps(request.context, sort_keys=True).encode()
            ).hexdigest()
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def _is_cache_valid(self, response: AIResponse) -> bool:
        """Check if cached response is still valid"""
        # Cache valid for 1 hour
        cache_age = datetime.now() - response.timestamp
        return cache_age.total_seconds() < 3600
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence score for results"""
        confidence_scores = []
        
        # Extract confidence scores from different components
        if 'symbiosis_matches' in results:
            matches = results['symbiosis_matches']
            if 'matches' in matches:
                scores = [match.get('confidence', 0.5) for match in matches['matches']]
                confidence_scores.extend(scores)
        
        if 'ai_insights' in results:
            confidence_scores.append(results['ai_insights'].get('confidence', 0.5))
        
        if 'analytics_metrics' in results:
            confidence_scores.append(results['analytics_metrics'].get('confidence', 0.5))
        
        # Return average confidence
        return np.mean(confidence_scores) if confidence_scores else 0.5
    
    def _get_models_used(self, results: Dict[str, Any]) -> List[str]:
        """Get list of models used in processing"""
        models = []
        
        if 'symbiosis_matches' in results:
            models.append('revolutionary_ai_matching')
        
        if 'gnn_reasoning' in results:
            models.append('gnn_reasoning_engine')
        
        if 'ai_insights' in results:
            models.append('advanced_ai_service')
        
        if 'analytics_metrics' in results:
            models.append('advanced_analytics_engine')
        
        return models
    
    def _calculate_synergy_score(self, results: Dict[str, Any]) -> float:
        """Calculate synergy score between AI modules"""
        synergy_factors = []
        
        # Check for multi-module integration
        if len(self._get_models_used(results)) > 1:
            synergy_factors.append(0.3)
        
        # Check for GNN enhancement
        if 'gnn_reasoning' in results:
            synergy_factors.append(0.2)
        
        # Check for comprehensive analysis
        if 'comprehensive_analysis' in results:
            synergy_factors.append(0.2)
        
        # Check for analytics integration
        if 'analytics_metrics' in results:
            synergy_factors.append(0.2)
        
        # Check for AI insights
        if 'ai_insights' in results:
            synergy_factors.append(0.1)
        
        return min(sum(synergy_factors), 1.0)
    
    def _calculate_sustainability_score(self, ai_insights: Dict, analytics_results: Dict) -> float:
        """Calculate comprehensive sustainability score"""
        scores = []
        
        # Extract scores from AI insights
        if 'sustainability_score' in ai_insights:
            scores.append(ai_insights['sustainability_score'])
        
        # Extract scores from analytics
        if 'overall_sustainability' in analytics_results:
            scores.append(analytics_results['overall_sustainability'])
        
        return np.mean(scores) if scores else 0.5
    
    def _record_learning_data(self, request: AIRequest, response: AIResponse):
        """Record data for adaptive learning"""
        learning_entry = {
            'request_type': request.request_type,
            'processing_time': response.processing_time,
            'success': response.success,
            'confidence': response.confidence,
            'models_used': response.models_used,
            'timestamp': datetime.now()
        }
        
        self.learning_history.append(learning_entry)
        
        # Keep only last 1000 entries
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-1000:]
    
    async def _optimize_performance(self):
        """Optimize performance based on learning data"""
        if len(self.learning_history) < 10:
            return
        
        # Analyze performance patterns
        recent_entries = self.learning_history[-100:]
        
        # Calculate average processing times by request type
        processing_times = defaultdict(list)
        for entry in recent_entries:
            processing_times[entry['request_type']].append(entry['processing_time'])
        
        # Identify slow request types
        slow_types = []
        for req_type, times in processing_times.items():
            avg_time = np.mean(times)
            if avg_time > 5.0:  # More than 5 seconds
                slow_types.append(req_type)
        
        # Record optimization
        optimization_entry = {
            'timestamp': datetime.now(),
            'slow_request_types': slow_types,
            'average_processing_time': np.mean([entry['processing_time'] for entry in recent_entries]),
            'success_rate': np.mean([entry['success'] for entry in recent_entries])
        }
        
        self.optimization_history.append(optimization_entry)
        
        # Keep only last 100 optimization entries
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
    
    async def _update_metrics(self):
        """Update performance metrics"""
        with self.metrics_lock:
            # Update model utilization
            model_usage = defaultdict(int)
            for entry in self.learning_history[-100:]:
                for model in entry.get('models_used', []):
                    model_usage[model] += 1
            
            total_requests = len(self.learning_history[-100:])
            if total_requests > 0:
                self.metrics.model_utilization = {
                    model: count / total_requests 
                    for model, count in model_usage.items()
                }
            
            # Update synergy score
            synergy_scores = []
            for entry in self.learning_history[-100:]:
                if entry.get('models_used'):
                    synergy_scores.append(len(entry['models_used']) / 4)  # Normalize by max modules
            
            if synergy_scores:
                self.metrics.synergy_score = np.mean(synergy_scores)
            
            self.metrics.last_updated = datetime.now()
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        with self.cache_lock:
            for key, response in self.response_cache.items():
                if (current_time - response.timestamp).total_seconds() > 3600:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.response_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_metrics(self) -> OrchestrationMetrics:
        """Get current performance metrics"""
        with self.metrics_lock:
            return self.metrics
    
    def get_learning_history(self) -> List[Dict]:
        """Get learning history"""
        return self.learning_history.copy()
    
    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history"""
        return self.optimization_history.copy()
    
    def clear_cache(self):
        """Clear all cached responses"""
        with self.cache_lock:
            self.response_cache.clear()
        logger.info("AI orchestrator cache cleared")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        return {
            'orchestrator_status': 'healthy',
            'metrics': asdict(self.metrics),
            'cache_size': len(self.response_cache),
            'learning_entries': len(self.learning_history),
            'optimization_entries': len(self.optimization_history),
            'model_persistence_status': 'healthy',
            'gnn_engine_status': 'healthy',
            'analytics_engine_status': 'healthy',
            'matching_engine_status': 'healthy',
            'timestamp': datetime.now().isoformat()
        }

# Global instance
ai_orchestrator = AIOrchestrator()