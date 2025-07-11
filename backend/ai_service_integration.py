import logging
import threading
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import asyncio
from pathlib import Path

# Import our AI components
from knowledge_graph import knowledge_graph
from federated_meta_learning import federated_learner
from gnn_reasoning_engine import gnn_reasoning_engine
from revolutionary_ai_matching import advanced_matching_engine
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
    - Fault tolerance and recovery
    """
    
    def __init__(self):
        self.services = {
            'knowledge_graph': knowledge_graph,
            'federated_learning': federated_learner,
            'gnn_reasoning': gnn_reasoning_engine,
            'advanced_matching': advanced_matching_engine,
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
        
        logger.info("AI Service Integration Layer initialized")

    def _initialize_services(self):
        """Initialize all AI services"""
        try:
            for service_name, service in self.services.items():
                self.service_status[service_name] = {
                    'status': 'initializing',
                    'last_heartbeat': datetime.now(),
                    'error_count': 0,
                    'performance': {}
                }
                
                # Test service availability
                if self._test_service(service_name, service):
                    self.service_status[service_name]['status'] = 'active'
                    logger.info(f"Service {service_name} initialized successfully")
                else:
                    self.service_status[service_name]['status'] = 'error'
                    logger.error(f"Service {service_name} failed to initialize")
                    
        except Exception as e:
            logger.error(f"Error initializing services: {e}")

    def _test_service(self, service_name: str, service: Any) -> bool:
        """Test if a service is working properly"""
        try:
            if service_name == 'knowledge_graph':
                # Test knowledge graph
                stats = service.get_graph_statistics()
                return 'nodes' in stats
            
            elif service_name == 'federated_learning':
                # Test federated learning
                stats = service.get_learning_statistics()
                return 'total_clients' in stats
            
            elif service_name == 'gnn_reasoning':
                # Test GNN reasoning
                models = service.list_available_models()
                return isinstance(models, list)
            
            elif service_name == 'advanced_matching':
                # Test advanced matching
                stats = service.get_matching_statistics()
                return 'total_matches' in stats or 'error' in stats
            
            elif service_name == 'model_persistence':
                # Test model persistence
                models = service.list_models()
                return isinstance(models, list)
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing service {service_name}: {e}")
            return False

    def _start_monitoring(self):
        """Start service monitoring"""
        def monitor_services():
            while True:
                try:
                    self._update_service_status()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in service monitoring: {e}")
        
        monitor_thread = threading.Thread(target=monitor_services, daemon=True)
        monitor_thread.start()
        logger.info("Service monitoring started")

    def _update_service_status(self):
        """Update service status and performance metrics"""
        try:
            with self.lock:
                for service_name, service in self.services.items():
                    try:
                        # Test service health
                        is_healthy = self._test_service(service_name, service)
                        
                        if is_healthy:
                            self.service_status[service_name]['status'] = 'active'
                            self.service_status[service_name]['error_count'] = 0
                        else:
                            self.service_status[service_name]['status'] = 'error'
                            self.service_status[service_name]['error_count'] += 1
                        
                        self.service_status[service_name]['last_heartbeat'] = datetime.now()
                        
                        # Update performance metrics
                        self._update_performance_metrics(service_name, service)
                        
                    except Exception as e:
                        self.service_status[service_name]['status'] = 'error'
                        self.service_status[service_name]['error_count'] += 1
                        self.error_log.append({
                            'service': service_name,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
                        
        except Exception as e:
            logger.error(f"Error updating service status: {e}")

    def _update_performance_metrics(self, service_name: str, service: Any):
        """Update performance metrics for a service"""
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
                stats = service.get_performance_stats()
                self.service_status[service_name]['performance'] = {
                    'total_models': stats.get('total_models', 0),
                    'total_versions': stats.get('total_versions', 0)
                }
                
        except Exception as e:
            logger.error(f"Error updating performance metrics for {service_name}: {e}")

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all AI services"""
        return self.service_status.copy()

    def get_service_health(self) -> Dict[str, str]:
        """Get health status of all services"""
        return {
            service_name: info['status'] 
            for service_name, info in self.service_status.items()
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all services"""
        summary = {
            'total_services': len(self.services),
            'active_services': sum(1 for info in self.service_status.values() if info['status'] == 'active'),
            'error_services': sum(1 for info in self.service_status.values() if info['status'] == 'error'),
            'services': {}
        }
        
        for service_name, info in self.service_status.items():
            summary['services'][service_name] = {
                'status': info['status'],
                'performance': info['performance'],
                'last_heartbeat': info['last_heartbeat'].isoformat(),
                'error_count': info['error_count']
            }
        
        return summary

    def execute_ai_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete AI pipeline"""
        try:
            start_time = time.time()
            results = {}
            
            # Knowledge Graph Processing
            if 'knowledge_graph' in pipeline_config:
                kg_config = pipeline_config['knowledge_graph']
                if kg_config.get('enabled', True):
                    kg_result = self._execute_knowledge_graph_pipeline(kg_config)
                    results['knowledge_graph'] = kg_result
            
            # Federated Learning
            if 'federated_learning' in pipeline_config:
                fl_config = pipeline_config['federated_learning']
                if fl_config.get('enabled', True):
                    fl_result = self._execute_federated_learning_pipeline(fl_config)
                    results['federated_learning'] = fl_result
            
            # GNN Reasoning
            if 'gnn_reasoning' in pipeline_config:
                gnn_config = pipeline_config['gnn_reasoning']
                if gnn_config.get('enabled', True):
                    gnn_result = self._execute_gnn_pipeline(gnn_config)
                    results['gnn_reasoning'] = gnn_result
            
            # Advanced Matching
            if 'advanced_matching' in pipeline_config:
                matching_config = pipeline_config['advanced_matching']
                if matching_config.get('enabled', True):
                    matching_result = self._execute_matching_pipeline(matching_config)
                    results['advanced_matching'] = matching_result
            
            pipeline_time = time.time() - start_time
            results['pipeline_time'] = pipeline_time
            results['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"AI pipeline completed in {pipeline_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error executing AI pipeline: {e}")
            return {'error': str(e)}

    def _execute_knowledge_graph_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute knowledge graph pipeline"""
        try:
            results = {}
            
            # Add entities if provided
            if 'entities' in config:
                for entity in config['entities']:
                    knowledge_graph.add_entity(
                        entity['id'], 
                        entity['attributes'], 
                        entity.get('type', 'company')
                    )
                results['entities_added'] = len(config['entities'])
            
            # Add relationships if provided
            if 'relationships' in config:
                for rel in config['relationships']:
                    knowledge_graph.add_relationship(
                        rel['source'], 
                        rel['target'], 
                        rel['type'], 
                        rel.get('attributes', {})
                    )
                results['relationships_added'] = len(config['relationships'])
            
            # Run reasoning if requested
            if config.get('run_reasoning', False):
                reasoning_type = config.get('reasoning_type', 'opportunity_discovery')
                reasoning_result = knowledge_graph.run_gnn_reasoning(reasoning_type)
                results['reasoning'] = reasoning_result
            
            # Get statistics
            stats = knowledge_graph.get_graph_statistics()
            results['statistics'] = stats
            
            return results
            
        except Exception as e:
            logger.error(f"Error in knowledge graph pipeline: {e}")
            return {'error': str(e)}

    def _execute_federated_learning_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute federated learning pipeline"""
        try:
            results = {}
            
            # Register clients if provided
            if 'clients' in config:
                for client in config['clients']:
                    federated_learner.register_client(
                        client['id'],
                        client['model_params'],
                        client.get('metadata', {})
                    )
                results['clients_registered'] = len(config['clients'])
            
            # Aggregate global model if requested
            if config.get('aggregate', False):
                aggregation_result = federated_learner.aggregate_global_model()
                if aggregation_result:
                    results['aggregation'] = {
                        'round': aggregation_result.round_number,
                        'participants': len(aggregation_result.participating_clients),
                        'quality': aggregation_result.aggregation_metrics['aggregation_quality']
                    }
            
            # Run meta-learning if requested
            if config.get('meta_learn', False):
                meta_result = federated_learner.meta_learn()
                results['meta_learning'] = meta_result
            
            # Get statistics
            stats = federated_learner.get_learning_statistics()
            results['statistics'] = stats
            
            return results
            
        except Exception as e:
            logger.error(f"Error in federated learning pipeline: {e}")
            return {'error': str(e)}

    def _execute_gnn_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GNN reasoning pipeline"""
        try:
            results = {}
            
            # Train model if requested
            if config.get('train_model', False):
                graph_data = config.get('graph_data')
                model_name = config.get('model_name', 'default')
                model_type = config.get('model_type', 'GCN')
                
                if graph_data:
                    training_result = gnn_reasoning_engine.train_model(
                        graph_data, model_name, model_type
                    )
                    results['training'] = training_result
            
            # Perform inference if requested
            if config.get('inference', False):
                graph_data = config.get('graph_data')
                model_name = config.get('model_name', 'default')
                inference_type = config.get('inference_type', 'node_embeddings')
                
                if graph_data:
                    inference_result = gnn_reasoning_engine.infer(
                        graph_data, model_name, inference_type
                    )
                    results['inference'] = inference_result
            
            # Get statistics
            stats = gnn_reasoning_engine.get_inference_statistics()
            results['statistics'] = stats
            
            return results
            
        except Exception as e:
            logger.error(f"Error in GNN pipeline: {e}")
            return {'error': str(e)}

    def _execute_matching_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute advanced matching pipeline"""
        try:
            results = {}
            
            # Train models if requested
            if config.get('train_semantic', False):
                training_data = config.get('semantic_training_data', [])
                if training_data:
                    semantic_result = advanced_matching_engine.train_semantic_model(training_data)
                    results['semantic_training'] = semantic_result
            
            if config.get('train_numerical', False):
                training_data = config.get('numerical_training_data', [])
                if training_data:
                    numerical_result = advanced_matching_engine.train_numerical_model(training_data)
                    results['numerical_training'] = numerical_result
            
            # Perform matching if requested
            if config.get('find_matches', False):
                query_company = config.get('query_company')
                candidate_companies = config.get('candidate_companies', [])
                algorithm = config.get('algorithm', 'ensemble')
                top_k = config.get('top_k', 10)
                
                if query_company and candidate_companies:
                    matching_result = advanced_matching_engine.find_matches(
                        query_company, candidate_companies, algorithm, top_k
                    )
                    results['matching'] = {
                        'candidates_found': matching_result.total_candidates,
                        'matching_time': matching_result.matching_time,
                        'algorithm_used': matching_result.algorithm_used
                    }
            
            # Get statistics
            stats = advanced_matching_engine.get_matching_statistics()
            results['statistics'] = stats
            
            return results
            
        except Exception as e:
            logger.error(f"Error in matching pipeline: {e}")
            return {'error': str(e)}

    def save_all_models(self) -> Dict[str, bool]:
        """Save all models using the persistence manager"""
        try:
            results = {}
            
            # Save knowledge graph
            try:
                kg_data = {
                    'graph': knowledge_graph.graph,
                    'embeddings': knowledge_graph.node_embeddings,
                    'stats': knowledge_graph.stats
                }
                success = model_persistence_manager.save_model(
                    'knowledge_graph', kg_data, {'type': 'knowledge_graph'}
                )
                results['knowledge_graph'] = success
            except Exception as e:
                logger.error(f"Error saving knowledge graph: {e}")
                results['knowledge_graph'] = False
            
            # Save federated learning models
            try:
                fl_data = {
                    'global_model': federated_learner.global_model,
                    'client_registry': federated_learner.client_registry,
                    'learning_config': federated_learner.learning_config
                }
                success = model_persistence_manager.save_model(
                    'federated_learning', fl_data, {'type': 'federated_learning'}
                )
                results['federated_learning'] = success
            except Exception as e:
                logger.error(f"Error saving federated learning: {e}")
                results['federated_learning'] = False
            
            # Save GNN models
            try:
                if gnn_reasoning_engine.model_manager:
                    for model_name in gnn_reasoning_engine.model_manager.list_models():
                        model = gnn_reasoning_engine.model_manager.get_model(model_name)
                        if model:
                            success = model_persistence_manager.save_model(
                                f'gnn_{model_name}', model, {'type': 'gnn_model'}
                            )
                            results[f'gnn_{model_name}'] = success
            except Exception as e:
                logger.error(f"Error saving GNN models: {e}")
                results['gnn_models'] = False
            
            # Save matching models
            try:
                matching_data = {
                    'semantic_model': advanced_matching_engine.semantic_model,
                    'numerical_model': advanced_matching_engine.numerical_model,
                    'scaler': advanced_matching_engine.scaler,
                    'label_encoders': advanced_matching_engine.label_encoders
                }
                success = model_persistence_manager.save_model(
                    'advanced_matching', matching_data, {'type': 'advanced_matching'}
                )
                results['advanced_matching'] = success
            except Exception as e:
                logger.error(f"Error saving matching models: {e}")
                results['advanced_matching'] = False
            
            logger.info(f"Model save results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error saving all models: {e}")
            return {'error': str(e)}

    def load_all_models(self) -> Dict[str, bool]:
        """Load all models using the persistence manager"""
        try:
            results = {}
            
            # Load knowledge graph
            try:
                kg_data = model_persistence_manager.load_model('knowledge_graph')
                if kg_data:
                    knowledge_graph.graph = kg_data['graph']
                    knowledge_graph.node_embeddings = kg_data['embeddings']
                    knowledge_graph.stats = kg_data['stats']
                    results['knowledge_graph'] = True
                else:
                    results['knowledge_graph'] = False
            except Exception as e:
                logger.error(f"Error loading knowledge graph: {e}")
                results['knowledge_graph'] = False
            
            # Load federated learning models
            try:
                fl_data = model_persistence_manager.load_model('federated_learning')
                if fl_data:
                    federated_learner.global_model = fl_data['global_model']
                    federated_learner.client_registry = fl_data['client_registry']
                    federated_learner.learning_config = fl_data['learning_config']
                    results['federated_learning'] = True
                else:
                    results['federated_learning'] = False
            except Exception as e:
                logger.error(f"Error loading federated learning: {e}")
                results['federated_learning'] = False
            
            # Load GNN models
            try:
                gnn_models = model_persistence_manager.list_models()
                for model_name in gnn_models:
                    if model_name.startswith('gnn_'):
                        model = model_persistence_manager.load_model(model_name)
                        if model and gnn_reasoning_engine.model_manager:
                            gnn_reasoning_engine.model_manager.models[model_name[4:]] = model
                            results[f'gnn_{model_name[4:]}'] = True
                        else:
                            results[f'gnn_{model_name[4:]}'] = False
            except Exception as e:
                logger.error(f"Error loading GNN models: {e}")
                results['gnn_models'] = False
            
            # Load matching models
            try:
                matching_data = model_persistence_manager.load_model('advanced_matching')
                if matching_data:
                    advanced_matching_engine.semantic_model = matching_data['semantic_model']
                    advanced_matching_engine.numerical_model = matching_data['numerical_model']
                    advanced_matching_engine.scaler = matching_data['scaler']
                    advanced_matching_engine.label_encoders = matching_data['label_encoders']
                    results['advanced_matching'] = True
                else:
                    results['advanced_matching'] = False
            except Exception as e:
                logger.error(f"Error loading matching models: {e}")
                results['advanced_matching'] = False
            
            logger.info(f"Model load results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading all models: {e}")
            return {'error': str(e)}

    def get_error_log(self) -> List[Dict[str, Any]]:
        """Get error log"""
        return self.error_log.copy()

    def clear_error_log(self):
        """Clear error log"""
        self.error_log.clear()

# Initialize global AI service integration
ai_service_integration = AIServiceIntegration() 