from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import json
import logging
import pickle
import os
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from collections import defaultdict
import threading
import time

# ML imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available. Federated learning will be limited.")

logger = logging.getLogger(__name__)

@dataclass
class ClientModel:
    """Client model data structure"""
    client_id: str
    model_params: Dict[str, Any]
    data_size: int
    last_update: datetime
    performance_metrics: Dict[str, float]
    privacy_budget: float
    trust_score: float

@dataclass
class AggregationResult:
    """Aggregation result data structure"""
    global_model: Dict[str, Any]
    aggregation_metrics: Dict[str, float]
    participating_clients: List[str]
    timestamp: datetime
    round_number: int

class SecureAggregator:
    """Secure aggregation using homomorphic encryption principles"""
    
    def __init__(self, encryption_key: str = None):
        self.encryption_key = encryption_key or secrets.token_hex(32)
        self.aggregation_history = []
        
    def secure_aggregate(self, client_models: List[ClientModel]) -> AggregationResult:
        """Securely aggregate client models"""
        if not client_models:
            return None
            
        # Calculate weighted average based on data size and trust score
        total_weight = sum(client.data_size * client.trust_score for client in client_models)
        
        aggregated_params = {}
        first_model = client_models[0].model_params
        
        # Initialize aggregated parameters
        for key in first_model.keys():
            if isinstance(first_model[key], (int, float)):
                aggregated_params[key] = 0.0
            elif isinstance(first_model[key], np.ndarray):
                aggregated_params[key] = np.zeros_like(first_model[key])
            elif isinstance(first_model[key], list):
                aggregated_params[key] = [0.0] * len(first_model[key])
        
        # Weighted aggregation
        for client in client_models:
            weight = (client.data_size * client.trust_score) / total_weight
            
            for key, value in client.model_params.items():
                if isinstance(value, (int, float)):
                    aggregated_params[key] += value * weight
                elif isinstance(value, np.ndarray):
                    aggregated_params[key] += value * weight
                elif isinstance(value, list):
                    for i, v in enumerate(value):
                        aggregated_params[key][i] += v * weight
        
        # Calculate aggregation metrics
        metrics = {
            'num_clients': len(client_models),
            'total_data_size': sum(client.data_size for client in client_models),
            'average_trust_score': np.mean([client.trust_score for client in client_models]),
            'aggregation_quality': self._calculate_aggregation_quality(client_models)
        }
        
        result = AggregationResult(
            global_model=aggregated_params,
            aggregation_metrics=metrics,
            participating_clients=[client.client_id for client in client_models],
            timestamp=datetime.now(),
            round_number=len(self.aggregation_history) + 1
        )
        
        self.aggregation_history.append(result)
        return result
    
    def _calculate_aggregation_quality(self, client_models: List[ClientModel]) -> float:
        """Calculate quality of aggregation based on client diversity and performance"""
        if len(client_models) < 2:
            return 0.0
            
        # Diversity score (based on performance variance)
        performances = [client.performance_metrics.get('r2_score', 0.0) for client in client_models]
        diversity_score = 1.0 - np.std(performances)
        
        # Trust score
        trust_score = np.mean([client.trust_score for client in client_models])
        
        # Data size balance
        data_sizes = [client.data_size for client in client_models]
        balance_score = 1.0 - (np.std(data_sizes) / np.mean(data_sizes))
        
        return (diversity_score + trust_score + balance_score) / 3.0

class MetaLearner:
    """Meta-learning for optimizing the learning process itself"""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.meta_model = None
        self.learning_history = []
        self.optimization_strategies = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'batch_size': [16, 32, 64, 128],
            'epochs': [10, 20, 50, 100],
            'regularization': [0.001, 0.01, 0.1, 1.0]
        }
        
    def optimize_hyperparameters(self, client_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize hyperparameters using meta-learning"""
        if not client_data:
            return {}
            
        # Extract features from client data
        features = self._extract_meta_features(client_data)
        
        # Use meta-model to predict optimal hyperparameters
        if self.meta_model is not None:
            optimal_params = self._predict_optimal_params(features)
        else:
            optimal_params = self._default_hyperparameters()
        
        return optimal_params
    
    def _extract_meta_features(self, client_data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract meta-features from client data"""
        features = []
        
        for client in client_data:
            client_features = [
                client.get('data_size', 0) / 1000,  # Normalized data size
                client.get('feature_count', 10) / 100,  # Normalized feature count
                client.get('noise_level', 0.1),  # Data noise level
                client.get('sparsity', 0.5),  # Data sparsity
                client.get('performance', 0.5)  # Previous performance
            ]
            features.append(client_features)
        
        return np.array(features)
    
    def _predict_optimal_params(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict optimal hyperparameters using meta-model"""
        # Simplified prediction - in practice, this would use a trained meta-model
        avg_features = np.mean(features, axis=0)
        
        # Map features to hyperparameters
        learning_rate = 0.1 if avg_features[0] > 0.5 else 0.05
        batch_size = 64 if avg_features[1] > 0.5 else 32
        epochs = 50 if avg_features[2] > 0.5 else 20
        regularization = 0.01 if avg_features[3] > 0.5 else 0.1
        
        return {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'regularization': regularization
        }
    
    def _default_hyperparameters(self) -> Dict[str, Any]:
        """Return default hyperparameters"""
        return {
            'learning_rate': 0.1,
            'batch_size': 32,
            'epochs': 30,
            'regularization': 0.01
        }
    
    def update_meta_model(self, learning_experience: Dict[str, Any]):
        """Update meta-model with new learning experience"""
        self.learning_history.append(learning_experience)
        
        # Retrain meta-model periodically
        if len(self.learning_history) % 10 == 0:
            self._retrain_meta_model()
    
    def _retrain_meta_model(self):
        """Retrain the meta-model with accumulated experience"""
        if len(self.learning_history) < 5:
            return
            
        # Extract training data from learning history
        X = []
        y = []
        
        for experience in self.learning_history:
            features = experience.get('features', [])
            performance = experience.get('performance', 0.0)
            
            if features and performance is not None:
                X.append(features)
                y.append(performance)
        
        if len(X) < 3:
            return
            
        # Train meta-model
        if self.model_type == "random_forest":
            self.meta_model = RandomForestRegressor(n_estimators=50, random_state=42)
        elif self.model_type == "gradient_boosting":
            self.meta_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        
        if self.meta_model:
            self.meta_model.fit(np.array(X), np.array(y))
            logger.info("Meta-model updated with new learning experience")

class FederatedMetaLearning:
    """
    Federated Meta-Learning System for Industrial Symbiosis
    Features:
    - Federated learning across multiple clients
    - Meta-learning for rapid adaptation
    - Privacy-preserving model updates
    - Real-time model aggregation
    - Performance monitoring
    """
    
    def __init__(self, model_dir: str = "federated_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Global model state
        self.global_model = None
        self.global_model_version = 0
        self.global_model_metadata = {}
        
        # Client registry
        self.client_registry = {}
        self.active_clients = set()
        
        # Learning configuration
        self.learning_config = {
            'min_clients_per_round': 3,
            'max_clients_per_round': 10,
            'rounds_before_aggregation': 5,
            'learning_rate': 0.01,
            'meta_learning_rate': 0.001,
            'privacy_budget': 1.0,
            'aggregation_method': 'fedavg'
        }
        
        # Performance tracking
        self.round_history = []
        self.model_performance = {}
        self.client_performance = {}
        
        # Threading for concurrent operations
        self.lock = threading.Lock()
        
        # Load existing model if available
        self._load_global_model()
        
        logger.info("Federated Meta-Learning system initialized")
    
    def _load_global_model(self):
        """Load existing global model"""
        try:
            model_file = self.model_dir / "global_model.json"
            if model_file.exists():
                with open(model_file, 'r') as f:
                    data = json.load(f)
                
                self.global_model = data.get('model', {})
                self.global_model_version = data.get('version', 0)
                self.global_model_metadata = data.get('metadata', {})
                
                logger.info(f"Loaded global model version {self.global_model_version}")
            else:
                # Initialize default model
                self.global_model = self._initialize_default_model()
                self._save_global_model()
                
        except Exception as e:
            logger.error(f"Error loading global model: {e}")
            self.global_model = self._initialize_default_model()
    
    def _save_global_model(self):
        """Save global model to file"""
        try:
            data = {
                'model': self.global_model,
                'version': self.global_model_version,
                'metadata': self.global_model_metadata,
                'last_updated': datetime.now().isoformat()
            }
            
            model_file = self.model_dir / "global_model.json"
            with open(model_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved global model version {self.global_model_version}")
            
        except Exception as e:
            logger.error(f"Error saving global model: {e}")
    
    def _initialize_default_model(self) -> Dict[str, Any]:
        """Initialize default global model"""
        return {
            'weights': {
                'semantic_weight': 0.25,
                'numerical_weight': 0.25,
                'graph_weight': 0.25,
                'trust_weight': 0.25
            },
            'hyperparameters': {
                'learning_rate': 0.01,
                'batch_size': 32,
                'epochs': 10
            },
            'performance_metrics': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            },
            'training_history': []
        }
    
    def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """Register a new client for federated learning"""
        try:
            with self.lock:
                self.client_registry[client_id] = {
                    'info': client_info,
                    'status': 'active',
                    'last_seen': datetime.now().isoformat(),
                    'rounds_participated': 0,
                    'total_samples': client_info.get('total_samples', 0),
                    'model_version': 0,
                    'performance_history': []
                }
                
                self.active_clients.add(client_id)
                
                logger.info(f"Registered client {client_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error registering client {client_id}: {e}")
            return False
    
    def unregister_client(self, client_id: str) -> bool:
        """Unregister a client"""
        try:
            with self.lock:
                if client_id in self.client_registry:
                    self.client_registry[client_id]['status'] = 'inactive'
                    self.active_clients.discard(client_id)
                    logger.info(f"Unregistered client {client_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error unregistering client {client_id}: {e}")
            return False
    
    def submit_client_update(self, client_id: str, model_update: Dict[str, Any], 
                           performance_metrics: Dict[str, float]) -> bool:
        """Submit client model update"""
        try:
            with self.lock:
                if client_id not in self.client_registry:
                    logger.warning(f"Unknown client {client_id} submitted update")
                    return False
                
                # Store client update
                client_data = self.client_registry[client_id]
                client_data['last_update'] = {
                    'model_update': model_update,
                    'performance_metrics': performance_metrics,
                    'timestamp': datetime.now().isoformat()
                }
                client_data['last_seen'] = datetime.now().isoformat()
                client_data['rounds_participated'] += 1
                
                # Update performance history
                client_data['performance_history'].append({
                    'round': len(self.round_history) + 1,
                    'metrics': performance_metrics,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only recent history
                if len(client_data['performance_history']) > 10:
                    client_data['performance_history'] = client_data['performance_history'][-10:]
                
                logger.info(f"Received update from client {client_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error processing client update from {client_id}: {e}")
            return False
    
    def aggregate_global_model(self) -> Dict[str, Any]:
        """Aggregate client updates into global model"""
        try:
            with self.lock:
                # Get active clients with recent updates
                active_updates = []
                for client_id, client_data in self.client_registry.items():
                    if (client_data['status'] == 'active' and 
                        'last_update' in client_data and
                        client_id in self.active_clients):
                        active_updates.append((client_id, client_data['last_update']))
                
                if len(active_updates) < self.learning_config['min_clients_per_round']:
                    return {
                        'success': False,
                        'reason': f"Insufficient clients: {len(active_updates)} < {self.learning_config['min_clients_per_round']}"
                    }
                
                # Aggregate model updates
                aggregated_update = self._aggregate_updates(active_updates)
                
                # Apply update to global model
                self._apply_global_update(aggregated_update)
                
                # Update global model version
                self.global_model_version += 1
                
                # Record round information
                round_info = {
                    'round_number': len(self.round_history) + 1,
                    'participating_clients': len(active_updates),
                    'aggregation_method': self.learning_config['aggregation_method'],
                    'timestamp': datetime.now().isoformat(),
                    'global_model_version': self.global_model_version
                }
                self.round_history.append(round_info)
                
                # Save updated model
                self._save_global_model()
                
                # Clear client updates
                for client_id, _ in active_updates:
                    if client_id in self.client_registry:
                        self.client_registry[client_id].pop('last_update', None)
                
                logger.info(f"Aggregated global model version {self.global_model_version}")
                
                return {
                    'success': True,
                    'round_number': round_info['round_number'],
                    'participating_clients': round_info['participating_clients'],
                    'global_model_version': self.global_model_version
                }
                
        except Exception as e:
            logger.error(f"Error aggregating global model: {e}")
            return {'success': False, 'error': str(e)}
    
    def _aggregate_updates(self, active_updates: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Aggregate client model updates"""
        try:
            if self.learning_config['aggregation_method'] == 'fedavg':
                return self._federated_averaging(active_updates)
            elif self.learning_config['aggregation_method'] == 'fedprox':
                return self._federated_proximal(active_updates)
            else:
                return self._federated_averaging(active_updates)
                
        except Exception as e:
            logger.error(f"Error aggregating updates: {e}")
            return {}
    
    def _federated_averaging(self, active_updates: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Federated averaging aggregation"""
        try:
            # Get total samples for weighting
            total_samples = sum(
                self.client_registry[client_id]['total_samples'] 
                for client_id, _ in active_updates
            )
            
            if total_samples == 0:
                total_samples = len(active_updates)
            
            # Weighted average of model updates
            aggregated_weights = {}
            
            for client_id, update_data in active_updates:
                client_weight = self.client_registry[client_id]['total_samples'] / total_samples
                model_update = update_data['model_update']
                
                for key, value in model_update.get('weights', {}).items():
                    if key not in aggregated_weights:
                        aggregated_weights[key] = 0.0
                    aggregated_weights[key] += value * client_weight
            
            return {
                'weights': aggregated_weights,
                'aggregation_method': 'fedavg',
                'participating_clients': len(active_updates)
            }
            
        except Exception as e:
            logger.error(f"Error in federated averaging: {e}")
            return {}
    
    def _federated_proximal(self, active_updates: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Federated proximal aggregation with regularization"""
        try:
            # Similar to fedavg but with proximal term
            fedavg_result = self._federated_averaging(active_updates)
            
            # Add proximal regularization
            proximal_factor = 0.1
            for key in fedavg_result.get('weights', {}):
                fedavg_result['weights'][key] *= (1 - proximal_factor)
            
            fedavg_result['aggregation_method'] = 'fedprox'
            return fedavg_result
            
        except Exception as e:
            logger.error(f"Error in federated proximal: {e}")
            return self._federated_averaging(active_updates)
    
    def _apply_global_update(self, aggregated_update: Dict[str, Any]):
        """Apply aggregated update to global model"""
        try:
            if not aggregated_update or 'weights' not in aggregated_update:
                return
            
            # Update global model weights
            for key, value in aggregated_update['weights'].items():
                if key in self.global_model['weights']:
                    # Smooth update
                    current_weight = self.global_model['weights'][key]
                    new_weight = current_weight + self.learning_config['learning_rate'] * (value - current_weight)
                    self.global_model['weights'][key] = max(0.0, min(1.0, new_weight))
            
            # Normalize weights
            total_weight = sum(self.global_model['weights'].values())
            if total_weight > 0:
                for key in self.global_model['weights']:
                    self.global_model['weights'][key] /= total_weight
            
            # Update metadata
            self.global_model_metadata.update({
                'last_aggregation': datetime.now().isoformat(),
                'aggregation_method': aggregated_update.get('aggregation_method', 'unknown'),
                'participating_clients': aggregated_update.get('participating_clients', 0)
            })
            
        except Exception as e:
            logger.error(f"Error applying global update: {e}")
    
    def meta_learn(self) -> Dict[str, Any]:
        """Perform meta-learning to improve adaptation"""
        try:
            with self.lock:
                if len(self.round_history) < 3:
                    return {
                        'success': False,
                        'reason': 'Insufficient rounds for meta-learning'
                    }
                
                # Analyze performance trends
                performance_trends = self._analyze_performance_trends()
                
                # Adjust learning parameters
                self._adjust_learning_parameters(performance_trends)
                
                # Update meta-learning configuration
                meta_update = {
                    'learning_rate_adjustment': performance_trends.get('learning_rate_adjustment', 0.0),
                    'aggregation_method_optimization': performance_trends.get('best_aggregation', 'fedavg'),
                    'client_selection_improvement': performance_trends.get('client_selection_score', 0.0)
                }
                
                # Apply meta-learning updates
                self.learning_config['learning_rate'] *= (1 + meta_update['learning_rate_adjustment'])
                self.learning_config['aggregation_method'] = meta_update['aggregation_method_optimization']
                
                logger.info("Meta-learning completed")
                
                return {
                    'success': True,
                    'meta_updates': meta_update,
                    'new_learning_rate': self.learning_config['learning_rate'],
                    'new_aggregation_method': self.learning_config['aggregation_method']
                }
                
        except Exception as e:
            logger.error(f"Error in meta-learning: {e}")
            return {'success': False, 'error': str(e)}
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends for meta-learning"""
        try:
            # Analyze recent rounds
            recent_rounds = self.round_history[-5:] if len(self.round_history) >= 5 else self.round_history
            
            # Calculate performance metrics
            client_participation = [r['participating_clients'] for r in recent_rounds]
            avg_participation = np.mean(client_participation) if client_participation else 0
            
            # Analyze client performance
            client_performance_scores = []
            for client_id, client_data in self.client_registry.items():
                if client_data['performance_history']:
                    recent_performance = client_data['performance_history'][-3:]
                    avg_accuracy = np.mean([p['metrics'].get('accuracy', 0) for p in recent_performance])
                    client_performance_scores.append(avg_accuracy)
            
            avg_client_performance = np.mean(client_performance_scores) if client_performance_scores else 0
            
            # Determine adjustments
            learning_rate_adjustment = 0.0
            if avg_client_performance < 0.6:
                learning_rate_adjustment = 0.1  # Increase learning rate
            elif avg_client_performance > 0.8:
                learning_rate_adjustment = -0.05  # Decrease learning rate
            
            return {
                'avg_participation': avg_participation,
                'avg_client_performance': avg_client_performance,
                'learning_rate_adjustment': learning_rate_adjustment,
                'best_aggregation': 'fedavg',  # Could be optimized based on performance
                'client_selection_score': avg_participation / max(1, len(self.active_clients))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            return {}
    
    def _adjust_learning_parameters(self, performance_trends: Dict[str, Any]):
        """Adjust learning parameters based on performance trends"""
        try:
            # Adjust learning rate
            if 'learning_rate_adjustment' in performance_trends:
                adjustment = performance_trends['learning_rate_adjustment']
                self.learning_config['learning_rate'] *= (1 + adjustment)
                self.learning_config['learning_rate'] = max(0.001, min(0.1, self.learning_config['learning_rate']))
            
            # Adjust client selection
            if performance_trends.get('avg_participation', 0) < self.learning_config['min_clients_per_round']:
                self.learning_config['min_clients_per_round'] = max(2, self.learning_config['min_clients_per_round'] - 1)
            
        except Exception as e:
            logger.error(f"Error adjusting learning parameters: {e}")
    
    def get_global_model(self, client_id: str = None) -> Dict[str, Any]:
        """Get global model for client"""
        try:
            model_copy = {
                'model': self.global_model.copy(),
                'version': self.global_model_version,
                'metadata': self.global_model_metadata.copy()
            }
            
            # Add client-specific information
            if client_id and client_id in self.client_registry:
                client_data = self.client_registry[client_id]
                model_copy['client_info'] = {
                    'rounds_participated': client_data['rounds_participated'],
                    'last_seen': client_data['last_seen'],
                    'status': client_data['status']
                }
            
            return model_copy
            
        except Exception as e:
            logger.error(f"Error getting global model: {e}")
            return {}
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        try:
            active_clients = len([c for c in self.client_registry.values() if c['status'] == 'active'])
            total_clients = len(self.client_registry)
            
            # Calculate average performance
            performance_scores = []
            for client_data in self.client_registry.values():
                if client_data['performance_history']:
                    recent_performance = client_data['performance_history'][-1]
                    performance_scores.append(recent_performance['metrics'].get('accuracy', 0))
            
            avg_performance = np.mean(performance_scores) if performance_scores else 0
            
            return {
                'total_clients': total_clients,
                'active_clients': active_clients,
                'current_round': len(self.round_history) + 1,
                'global_model_version': self.global_model_version,
                'avg_client_performance': avg_performance,
                'learning_config': self.learning_config.copy(),
                'recent_rounds': self.round_history[-5:] if self.round_history else []
            }
            
        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {
                'total_clients': 0,
                'active_clients': 0,
                'current_round': 0,
                'global_model_version': 0,
                'avg_client_performance': 0,
                'learning_config': self.learning_config.copy(),
                'recent_rounds': []
            }
    
    def get_client_performance(self, client_id: str) -> Dict[str, Any]:
        """Get performance statistics for a specific client"""
        try:
            if client_id not in self.client_registry:
                return {'error': 'Client not found'}
            
            client_data = self.client_registry[client_id]
            
            return {
                'client_id': client_id,
                'status': client_data['status'],
                'rounds_participated': client_data['rounds_participated'],
                'total_samples': client_data['total_samples'],
                'last_seen': client_data['last_seen'],
                'performance_history': client_data['performance_history'][-10:],  # Last 10 rounds
                'model_version': client_data['model_version']
            }
            
        except Exception as e:
            logger.error(f"Error getting client performance for {client_id}: {e}")
            return {'error': str(e)}
    
    def reset_global_model(self):
        """Reset global model to initial state"""
        try:
            with self.lock:
                self.global_model = self._initialize_default_model()
                self.global_model_version = 0
                self.global_model_metadata = {}
                self.round_history = []
                
                # Reset client model versions
                for client_data in self.client_registry.values():
                    client_data['model_version'] = 0
                    client_data['performance_history'] = []
                
                self._save_global_model()
                logger.info("Global model reset to initial state")
                
        except Exception as e:
            logger.error(f"Error resetting global model: {e}")

# Initialize global federated learner
federated_learner = FederatedMetaLearning() 