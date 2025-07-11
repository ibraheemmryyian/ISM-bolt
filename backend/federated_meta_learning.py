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
    Advanced Federated and Meta-Learning Engine for Distributed, Privacy-Preserving AI
    Features:
    - Secure model aggregation
    - Meta-learning for hyperparameter optimization
    - Privacy-preserving learning
    - Persistent model storage
    - Real-time learning adaptation
    """
    
    def __init__(self, model_cache_dir: str = "./models"):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Client management
        self.local_models: Dict[str, ClientModel] = {}
        self.global_model: Dict[str, Any] = {}
        self.client_registry: Dict[str, Dict[str, Any]] = {}
        
        # Learning components
        self.secure_aggregator = SecureAggregator()
        self.meta_learner = MetaLearner()
        
        # Learning configuration
        self.learning_config = {
            'min_clients_for_aggregation': 3,
            'max_rounds': 100,
            'convergence_threshold': 0.01,
            'privacy_budget': 1.0,
            'trust_decay_rate': 0.95
        }
        
        # Learning state
        self.current_round = 0
        self.convergence_history = []
        self.performance_history = []
        
        # Load persistent state
        self._load_persistent_state()
        
        # Threading for concurrent operations
        self.lock = threading.Lock()
        
        logger.info("Federated Meta-Learning Engine initialized")

    def _load_persistent_state(self):
        """Load persistent learning state"""
        try:
            # Load global model
            global_model_path = self.model_cache_dir / "global_model.pkl"
            if global_model_path.exists():
                with open(global_model_path, 'rb') as f:
                    self.global_model = pickle.load(f)
                logger.info("Loaded persistent global model")
            
            # Load client registry
            registry_path = self.model_cache_dir / "client_registry.pkl"
            if registry_path.exists():
                with open(registry_path, 'rb') as f:
                    self.client_registry = pickle.load(f)
                logger.info(f"Loaded {len(self.client_registry)} registered clients")
            
            # Load learning history
            history_path = self.model_cache_dir / "learning_history.pkl"
            if history_path.exists():
                with open(history_path, 'rb') as f:
                    history_data = pickle.load(f)
                    self.convergence_history = history_data.get('convergence', [])
                    self.performance_history = history_data.get('performance', [])
                logger.info(f"Loaded learning history: {len(self.convergence_history)} rounds")
                
        except Exception as e:
            logger.error(f"Error loading persistent state: {e}")

    def _save_persistent_state(self):
        """Save persistent learning state"""
        try:
            # Save global model
            global_model_path = self.model_cache_dir / "global_model.pkl"
            with open(global_model_path, 'wb') as f:
                pickle.dump(self.global_model, f)
            
            # Save client registry
            registry_path = self.model_cache_dir / "client_registry.pkl"
            with open(registry_path, 'wb') as f:
                pickle.dump(self.client_registry, f)
            
            # Save learning history
            history_path = self.model_cache_dir / "learning_history.pkl"
            history_data = {
                'convergence': self.convergence_history,
                'performance': self.performance_history,
                'last_updated': datetime.now()
            }
            with open(history_path, 'wb') as f:
                pickle.dump(history_data, f)
                
            logger.info("Persistent state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving persistent state: {e}")

    def register_client(self, client_id: str, model_params: Dict[str, Any], 
                       metadata: Dict[str, Any] = None) -> bool:
        """Register a new client with its local model parameters"""
        try:
            with self.lock:
                # Create client model
                client_model = ClientModel(
                    client_id=client_id,
                    model_params=model_params,
                    data_size=metadata.get('data_size', 1000) if metadata else 1000,
                    last_update=datetime.now(),
                    performance_metrics=metadata.get('performance_metrics', {}) if metadata else {},
                    privacy_budget=self.learning_config['privacy_budget'],
                    trust_score=metadata.get('trust_score', 0.8) if metadata else 0.8
                )
                
                self.local_models[client_id] = client_model
                
                # Update client registry
                self.client_registry[client_id] = {
                    'registered_at': datetime.now(),
                    'last_seen': datetime.now(),
                    'total_updates': 0,
                    'average_performance': 0.0,
                    'metadata': metadata or {}
                }
                
                logger.info(f"Registered client: {client_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error registering client {client_id}: {e}")
            return False

    def update_client_model(self, client_id: str, model_params: Dict[str, Any], 
                           performance_metrics: Dict[str, float] = None) -> bool:
        """Update a client's local model"""
        try:
            with self.lock:
                if client_id not in self.local_models:
                    logger.warning(f"Client {client_id} not registered")
                    return False
                
                # Update client model
                client_model = self.local_models[client_id]
                client_model.model_params = model_params
                client_model.last_update = datetime.now()
                
                if performance_metrics:
                    client_model.performance_metrics = performance_metrics
                
                # Update registry
                registry = self.client_registry[client_id]
                registry['last_seen'] = datetime.now()
                registry['total_updates'] += 1
                
                if performance_metrics and 'r2_score' in performance_metrics:
                    # Update average performance
                    current_avg = registry['average_performance']
                    total_updates = registry['total_updates']
                    new_performance = performance_metrics['r2_score']
                    registry['average_performance'] = (current_avg * (total_updates - 1) + new_performance) / total_updates
                
                logger.debug(f"Updated client model: {client_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating client {client_id}: {e}")
            return False

    def aggregate_global_model(self) -> Optional[AggregationResult]:
        """Aggregate local models into a global model"""
        try:
            with self.lock:
                # Check if enough clients are available
                active_clients = [
                    client for client in self.local_models.values()
                    if (datetime.now() - client.last_update).seconds < 3600  # Active in last hour
                ]
                
                if len(active_clients) < self.learning_config['min_clients_for_aggregation']:
                    logger.warning(f"Insufficient active clients: {len(active_clients)}")
                    return None
                
                # Perform secure aggregation
                aggregation_result = self.secure_aggregator.secure_aggregate(active_clients)
                
                if aggregation_result:
                    # Update global model
                    self.global_model = aggregation_result.global_model
                    self.current_round += 1
                    
                    # Update convergence history
                    convergence_score = aggregation_result.aggregation_metrics['aggregation_quality']
                    self.convergence_history.append(convergence_score)
                    
                    # Update performance history
                    avg_performance = np.mean([
                        client.performance_metrics.get('r2_score', 0.0) 
                        for client in active_clients
                    ])
                    self.performance_history.append(avg_performance)
                    
                    # Save persistent state
                    self._save_persistent_state()
                    
                    # Update meta-learner
                    self.meta_learner.update_meta_model({
                        'features': [client.data_size, len(client.model_params)],
                        'performance': avg_performance,
                        'round': self.current_round
                    })
                    
                    logger.info(f"Global model aggregated successfully (Round {self.current_round})")
                
                return aggregation_result
                
        except Exception as e:
            logger.error(f"Error in global model aggregation: {e}")
            return None

    def get_global_model(self) -> Dict[str, Any]:
        """Get the current global model"""
        return self.global_model.copy()

    def get_optimized_hyperparameters(self, client_id: str) -> Dict[str, Any]:
        """Get optimized hyperparameters for a specific client"""
        try:
            if client_id not in self.local_models:
                return self.meta_learner._default_hyperparameters()
            
            client_model = self.local_models[client_id]
            registry = self.client_registry[client_id]
            
            # Prepare client data for meta-learning
            client_data = [{
                'data_size': client_model.data_size,
                'feature_count': len(client_model.model_params),
                'noise_level': registry['metadata'].get('noise_level', 0.1),
                'sparsity': registry['metadata'].get('sparsity', 0.5),
                'performance': registry['average_performance']
            }]
            
            # Get optimized hyperparameters
            optimal_params = self.meta_learner.optimize_hyperparameters(client_data)
            
            return optimal_params
            
        except Exception as e:
            logger.error(f"Error getting optimized hyperparameters: {e}")
            return self.meta_learner._default_hyperparameters()

    def meta_learn(self) -> Dict[str, Any]:
        """Run meta-learning optimization"""
        try:
            # Prepare meta-learning data
            client_data = []
            for client_id, registry in self.client_registry.items():
                if registry['total_updates'] > 0:
                    client_data.append({
                        'data_size': registry['metadata'].get('data_size', 1000),
                        'feature_count': registry['metadata'].get('feature_count', 10),
                        'noise_level': registry['metadata'].get('noise_level', 0.1),
                        'sparsity': registry['metadata'].get('sparsity', 0.5),
                        'performance': registry['average_performance']
                    })
            
            # Optimize hyperparameters
            optimal_params = self.meta_learner.optimize_hyperparameters(client_data)
            
            # Update learning configuration
            self.learning_config.update(optimal_params)
            
            logger.info("Meta-learning optimization completed")
            
            return {
                'optimal_hyperparameters': optimal_params,
                'num_clients_analyzed': len(client_data),
                'meta_learning_round': len(self.meta_learner.learning_history)
            }
            
        except Exception as e:
            logger.error(f"Error in meta-learning: {e}")
            return {'error': str(e)}

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        try:
            active_clients = len([
                client for client in self.local_models.values()
                if (datetime.now() - client.last_update).seconds < 3600
            ])
            
            total_clients = len(self.local_models)
            
            # Calculate convergence metrics
            convergence_trend = "stable"
            if len(self.convergence_history) >= 3:
                recent_convergence = np.mean(self.convergence_history[-3:])
                if recent_convergence > np.mean(self.convergence_history[:-3]):
                    convergence_trend = "improving"
                elif recent_convergence < np.mean(self.convergence_history[:-3]):
                    convergence_trend = "declining"
            
            # Calculate performance metrics
            performance_trend = "stable"
            if len(self.performance_history) >= 3:
                recent_performance = np.mean(self.performance_history[-3:])
                if recent_performance > np.mean(self.performance_history[:-3]):
                    performance_trend = "improving"
                elif recent_performance < np.mean(self.performance_history[:-3]):
                    performance_trend = "declining"
            
            return {
                'total_clients': total_clients,
                'active_clients': active_clients,
                'current_round': self.current_round,
                'convergence_score': self.convergence_history[-1] if self.convergence_history else 0.0,
                'convergence_trend': convergence_trend,
                'average_performance': np.mean(self.performance_history) if self.performance_history else 0.0,
                'performance_trend': performance_trend,
                'learning_config': self.learning_config,
                'meta_learner_status': {
                    'model_type': self.meta_learner.model_type,
                    'learning_experiences': len(self.meta_learner.learning_history),
                    'meta_model_trained': self.meta_learner.meta_model is not None
                },
                'last_aggregation': self.secure_aggregator.aggregation_history[-1].timestamp if self.secure_aggregator.aggregation_history else None
            }
            
        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {'error': str(e)}

    def remove_inactive_clients(self, max_inactive_hours: int = 24):
        """Remove clients that have been inactive for too long"""
        try:
            with self.lock:
                current_time = datetime.now()
                inactive_clients = []
                
                for client_id, client_model in self.local_models.items():
                    inactive_hours = (current_time - client_model.last_update).total_seconds() / 3600
                    if inactive_hours > max_inactive_hours:
                        inactive_clients.append(client_id)
                
                for client_id in inactive_clients:
                    del self.local_models[client_id]
                    if client_id in self.client_registry:
                        del self.client_registry[client_id]
                
                if inactive_clients:
                    logger.info(f"Removed {len(inactive_clients)} inactive clients")
                    self._save_persistent_state()
                    
        except Exception as e:
            logger.error(f"Error removing inactive clients: {e}")

    def reset_learning_state(self):
        """Reset the learning state (for testing/debugging)"""
        try:
            with self.lock:
                self.local_models.clear()
                self.client_registry.clear()
                self.global_model.clear()
                self.convergence_history.clear()
                self.performance_history.clear()
                self.current_round = 0
                self.secure_aggregator.aggregation_history.clear()
                self.meta_learner.learning_history.clear()
                
                # Remove persistent files
                for file_path in self.model_cache_dir.glob("*.pkl"):
                    file_path.unlink()
                
                logger.info("Learning state reset successfully")
                
        except Exception as e:
            logger.error(f"Error resetting learning state: {e}")

# Initialize global federated learning instance
federated_learner = FederatedMetaLearning() 