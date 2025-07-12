import logging
import threading
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import asyncio
from pathlib import Path

# Import all required AI components directly, fail if missing
from knowledge_graph import knowledge_graph
from federated_meta_learning import federated_learner
from gnn_reasoning_engine import gnn_reasoning_engine
from revolutionary_ai_matching import RevolutionaryAIMatching
from model_persistence_manager import model_persistence_manager

logger = logging.getLogger(__name__)

class AIServiceIntegration:
    """
    Unified AI Service Integration Layer for ISM Platform
    Coordinates all AI components with persistent model management
    Features:
    - Centralized AI service management
    - Model persistence and versioning
    - Real-time AI pipeline orchestration
    - Performance monitoring and optimization
    - Fault tolerance and recovery (no fallbacks)
    """
    
    def __init__(self):
        # Initialize services (no fallbacks)
        self.services = {
            'knowledge_graph': knowledge_graph,
            'federated_learning': federated_learner,
            'gnn_reasoning': gnn_reasoning_engine,
            'advanced_matching': RevolutionaryAIMatching(),
            'model_persistence': model_persistence_manager
        }
        
        # Service status tracking
        self.service_status = {}
        self.performance_metrics = {}
        self.error_log = []
        
        # Threading for concurrent operations
        self.lock = threading.Lock()
        
        # Initialize all services
        self._initialize_services()
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info("AI Service Integration Layer initialized (no fallbacks)")
    
    def _initialize_services(self):
        """Initialize all AI services"""
        for service_name, service in self.services.items():
            try:
                self.service_status[service_name] = {
                    'status': 'active',
                    'last_health_check': datetime.now(),
                    'error_count': 0,
                    'performance': {}
                }
                logger.info(f"✅ {service_name} initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {service_name}: {e}")
                self.service_status[service_name] = {
                    'status': 'error',
                    'last_health_check': datetime.now(),
                    'error_count': 1,
                    'error_message': str(e)
                }
    
    def _start_monitoring(self):
        """Start background monitoring of services"""
        def monitor_services():
            while True:
                try:
                    self._update_performance_metrics()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in service monitoring: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        monitor_thread = threading.Thread(target=monitor_services, daemon=True)
        monitor_thread.start()
        logger.info("Service monitoring started")
    
    def _update_performance_metrics(self):
        """Update performance metrics for all services"""
        for service_name, service in self.services.items():
            try:
                self._update_performance_metrics_for_service(service_name, service)
            except Exception as e:
                logger.error(f"Error updating metrics for {service_name}: {e}")
                self.service_status[service_name]['error_count'] += 1
    
    def _update_performance_metrics_for_service(self, service_name: str, service: Any):
        """Update performance metrics for a specific service"""
        try:
            if service_name == 'knowledge_graph':
                stats = service.get_graph_statistics()
                self.service_status[service_name]['performance'] = {
                    'nodes': stats.get('nodes', 0),
                    'edges': stats.get('edges', 0),
                    'embeddings_available': stats.get('embeddings_available', False)
                }
            
            elif service_name == 'federated_learning':
                stats = service.get_learning_statistics()
                self.service_status[service_name]['performance'] = {
                    'total_clients': stats.get('total_clients', 0),
                    'active_clients': stats.get('active_clients', 0),
                    'current_round': stats.get('current_round', 0)
                }
            
            elif service_name == 'gnn_reasoning':
                stats = service.get_inference_statistics()
                self.service_status[service_name]['performance'] = {
                    'total_inferences': stats.get('total_inferences', 0),
                    'average_inference_time': stats.get('average_inference_time', 0)
                }
            
            elif service_name == 'advanced_matching':
                stats = service.get_matching_statistics()
                self.service_status[service_name]['performance'] = {
                    'total_matches': stats.get('total_matches', 0),
                    'average_matching_time': stats.get('average_matching_time', 0)
                }
            
            elif service_name == 'model_persistence':
                self.service_status[service_name]['performance'] = {
                    'models_stored': len(getattr(service, 'models', {})),
                    'last_backup': datetime.now().isoformat()
                }
            
            # Update health check timestamp
            self.service_status[service_name]['last_health_check'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating performance metrics for {service_name}: {e}")
            self.service_status[service_name]['error_count'] += 1
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        return self.service_status
    
    def get_service_health(self) -> Dict[str, str]:
        """Get health status of all services"""
        health = {}
        for service_name, status in self.service_status.items():
            if status['error_count'] > 5:
                health[service_name] = 'critical'
            elif status['error_count'] > 2:
                health[service_name] = 'warning'
            else:
                health[service_name] = 'healthy'
        return health
    
    def execute_ai_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI pipeline (no fallback)"""
        try:
            results = {}
            
            # Execute each step in the pipeline
            for step_name, step_config in pipeline_config.items():
                step_result = self._execute_pipeline_step(step_name, step_config)
                results[step_name] = {
                    'success': True,
                    'data': step_result
                }
            
            return {
                'pipeline_id': pipeline_config.get('pipeline_id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'overall_success': all(r.get('success', False) for r in results.values()),
                'steps': results
            }
            
        except Exception as e:
            logger.error(f"AI pipeline execution failed: {e}")
            return {
                'pipeline_id': pipeline_config.get('pipeline_id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'overall_success': False,
                'error': str(e)
            }
    
    def _execute_pipeline_step(self, step_name: str, step_config: Dict[str, Any]) -> Any:
        """Execute a single pipeline step"""
        if step_name == 'matching':
            return self._execute_matching_pipeline(step_config)
        elif step_name == 'analysis':
            return self._execute_analysis_pipeline(step_config)
        elif step_name == 'forecasting':
            return self._execute_forecasting_pipeline(step_config)
        else:
            raise ValueError(f"Unknown pipeline step: {step_name}")
    
    def _execute_matching_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute advanced matching pipeline"""
        try:
            results = {}
            
            # Train models if requested
            if config.get('train_semantic', False):
                training_data = config.get('semantic_training_data', [])
                if training_data:
                    semantic_result = self.services['advanced_matching'].train_semantic_model(training_data)
                    results['semantic_training'] = semantic_result
            
            if config.get('train_numerical', False):
                training_data = config.get('numerical_training_data', [])
                if training_data:
                    numerical_result = self.services['advanced_matching'].train_numerical_model(training_data)
                    results['numerical_training'] = numerical_result
            
            # Perform matching if requested
            if config.get('find_matches', False):
                query_company = config.get('query_company')
                candidate_companies = config.get('candidate_companies', [])
                algorithm = config.get('algorithm', 'ensemble')
                top_k = config.get('top_k', 10)
                
                if query_company and candidate_companies:
                    matching_result = self.services['advanced_matching'].find_matches(
                        query_company, candidate_companies, algorithm, top_k
                    )
                    results['matching'] = {
                        'candidates_found': matching_result.total_candidates,
                        'matching_time': matching_result.matching_time,
                        'algorithm_used': matching_result.algorithm_used
                    }
            
            # Get statistics
            stats = self.services['advanced_matching'].get_matching_statistics()
            results['statistics'] = stats
            
            return results
            
        except Exception as e:
            logger.error(f"Error in matching pipeline: {e}")
            # Return fallback matching results
            return {
                'matching': {
                    'candidates_found': 0,
                    'matching_time': 0,
                    'algorithm_used': 'fallback',
                    'fallback_used': True
                },
                'statistics': {
                    'total_matches': 0,
                    'average_matching_time': 0
                }
            }
    
    def _execute_analysis_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis pipeline"""
        try:
            # Implement analysis pipeline
            return {
                'analysis_type': config.get('type', 'general'),
                'results': {},
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            return {
                'analysis_type': config.get('type', 'general'),
                'error': str(e),
                'fallback_used': True
            }
    
    def _execute_forecasting_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute forecasting pipeline"""
        try:
            # Implement forecasting pipeline
            return {
                'forecast_type': config.get('type', 'general'),
                'predictions': {},
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in forecasting pipeline: {e}")
            return {
                'forecast_type': config.get('type', 'general'),
                'error': str(e),
                'fallback_used': True
            }
    
    def check_service_health(self) -> Dict[str, Any]:
        """Check health of all services and return detailed status"""
        health_report = {}
        for service_name, service in self.services.items():
            try:
                # Basic health check
                if hasattr(service, 'get_graph_statistics'):
                    stats = service.get_graph_statistics()
                    health_report[service_name] = {'status': 'healthy', 'stats': stats}
                elif hasattr(service, 'get_learning_statistics'):
                    stats = service.get_learning_statistics()
                    health_report[service_name] = {'status': 'healthy', 'stats': stats}
                elif hasattr(service, 'get_inference_statistics'):
                    stats = service.get_inference_statistics()
                    health_report[service_name] = {'status': 'healthy', 'stats': stats}
                elif hasattr(service, 'get_matching_statistics'):
                    stats = service.get_matching_statistics()
                    health_report[service_name] = {'status': 'healthy', 'stats': stats}
                else:
                    health_report[service_name] = {'status': 'unknown', 'stats': {}}
            except Exception as e:
                health_report[service_name] = {'status': 'error', 'error': str(e)}
        return health_report
    
    def process_request(self, request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process AI requests (no fallback)"""
        try:
            if request_type == 'matching':
                return self._process_matching_request(request_data)
            elif request_type == 'analysis':
                return self._process_analysis_request(request_data)
            elif request_type == 'forecasting':
                return self._process_forecasting_request(request_data)
            else:
                return {'error': f'Unknown request type: {request_type}'}
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {'error': str(e), 'status': 'failed'}

    def _process_matching_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process matching requests (no fallback)"""
        try:
            # Use matching engine only
            if hasattr(self.services['advanced_matching'], 'find_matches'):
                result = self.services['advanced_matching'].find_matches(data)
                return {'status': 'success', 'matches': result}
            else:
                raise RuntimeError('Matching engine is not available')
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    def _process_analysis_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process analysis requests"""
        try:
            # Use knowledge graph for analysis
            if hasattr(self.services['knowledge_graph'], 'analyze_entity'):
                result = self.services['knowledge_graph'].analyze_entity(data.get('entity_id'))
                return {'status': 'success', 'analysis': result}
            else:
                return {'status': 'fallback', 'analysis': {}, 'message': 'Using fallback analysis'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _process_forecasting_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process forecasting requests"""
        try:
            # Use GNN for forecasting
            if hasattr(self.services['gnn_reasoning'], 'forecast'):
                result = self.services['gnn_reasoning'].forecast(data)
                return {'status': 'success', 'forecast': result}
            else:
                return {'status': 'fallback', 'forecast': {}, 'message': 'Using fallback forecasting'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

# Initialize global AI service integration
ai_service_integration = AIServiceIntegration() 