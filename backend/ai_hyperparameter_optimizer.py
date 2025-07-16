"""
Production-Grade AI Hyperparameter Optimization System
Automated hyperparameter tuning for all AI models using Optuna and advanced techniques
"""

import optuna
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import time
from pathlib import Path
import pickle
import hashlib
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# ML imports
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import joblib

# AI component imports
from backend.gnn_reasoning_engine import GNNReasoningEngine
from backend.federated_meta_learning import FederatedMetaLearning
from backend.knowledge_graph import KnowledgeGraph
from revolutionary_ai_matching import RevolutionaryAIMatching
from backend.model_persistence_manager import ModelPersistenceManager

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    model_name: str
    optimization_type: str  # 'bayesian', 'random', 'tpe', 'cmaes'
    n_trials: int
    timeout: int  # seconds
    metric: str  # 'accuracy', 'f1', 'mse', 'custom'
    direction: str  # 'maximize', 'minimize'
    constraints: Dict[str, Any]
    search_space: Dict[str, Any]

@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization"""
    optimization_id: str
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    duration: float
    n_trials: int
    status: str  # 'completed', 'failed', 'timeout'
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class ModelPerformance:
    """Model performance tracking for optimization"""
    model_name: str
    params: Dict[str, Any]
    score: float
    training_time: float
    inference_time: float
    memory_usage: float
    timestamp: datetime

class AIHyperparameterOptimizer:
    """
    Production-Grade AI Hyperparameter Optimization System
    """
    
    def __init__(self, storage_dir: str = "optimization_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.model_manager = ModelPersistenceManager()
        self.ai_components = self._initialize_ai_components()
        
        # Optimization state
        self.active_optimizations = {}
        self.optimization_history = {}
        self.performance_cache = {}
        
        # Threading
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Load existing optimizations
        self._load_optimization_history()
        
        logger.info("AI Hyperparameter Optimizer initialized")
    
    def _initialize_ai_components(self) -> Dict[str, Any]:
        """Initialize AI components for optimization"""
        components = {}
        
        try:
            components['gnn'] = GNNReasoningEngine()
            logger.info("✅ GNN engine initialized for optimization")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GNN engine: {e}")
        
        try:
            components['federated'] = FederatedMetaLearning()
            logger.info("✅ Federated learner initialized for optimization")
        except Exception as e:
            logger.error(f"❌ Failed to initialize federated learner: {e}")
        
        try:
            components['knowledge_graph'] = KnowledgeGraph()
            logger.info("✅ Knowledge graph initialized for optimization")
        except Exception as e:
            logger.error(f"❌ Failed to initialize knowledge graph: {e}")
        
        try:
            components['matching'] = RevolutionaryAIMatching()
            logger.info("✅ Matching engine initialized for optimization")
        except Exception as e:
            logger.error(f"❌ Failed to initialize matching engine: {e}")
        
        return components
    
    def _load_optimization_history(self):
        """Load existing optimization history"""
        try:
            history_file = self.storage_dir / "optimization_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                
                for opt_data in data.get('optimizations', []):
                    result = OptimizationResult(
                        optimization_id=opt_data['optimization_id'],
                        model_name=opt_data['model_name'],
                        best_params=opt_data['best_params'],
                        best_score=opt_data['best_score'],
                        optimization_history=opt_data['optimization_history'],
                        duration=opt_data['duration'],
                        n_trials=opt_data['n_trials'],
                        status=opt_data['status'],
                        metadata=opt_data['metadata'],
                        timestamp=datetime.fromisoformat(opt_data['timestamp'])
                    )
                    self.optimization_history[result.optimization_id] = result
                
                logger.info(f"Loaded {len(self.optimization_history)} optimization results")
                
        except Exception as e:
            logger.error(f"Error loading optimization history: {e}")
    
    def _save_optimization_history(self):
        """Save optimization history"""
        try:
            data = {
                'optimizations': [asdict(result) for result in self.optimization_history.values()],
                'last_updated': datetime.now().isoformat()
            }
            
            history_file = self.storage_dir / "optimization_history.json"
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Optimization history saved")
            
        except Exception as e:
            logger.error(f"Error saving optimization history: {e}")
    
    async def optimize_hyperparameters(self, config: OptimizationConfig, 
                                     training_data: Any = None) -> str:
        """
        Start hyperparameter optimization for a model
        """
        try:
            optimization_id = f"opt_{uuid.uuid4().hex[:8]}"
            
            # Validate configuration
            self._validate_optimization_config(config)
            
            # Create optimization task
            optimization_task = {
                'optimization_id': optimization_id,
                'config': config,
                'training_data': training_data,
                'status': 'running',
                'start_time': datetime.now(),
                'progress': 0.0
            }
            
            # Add to active optimizations
            with self.lock:
                self.active_optimizations[optimization_id] = optimization_task
            
            # Start optimization in background
            self.executor.submit(self._run_optimization, optimization_id)
            
            logger.info(f"Started hyperparameter optimization {optimization_id} for {config.model_name}")
            return optimization_id
            
        except Exception as e:
            logger.error(f"Error starting optimization: {e}")
            raise
    
    def _validate_optimization_config(self, config: OptimizationConfig):
        """Validate optimization configuration"""
        if config.model_name not in self.ai_components:
            raise ValueError(f"Model {config.model_name} not available for optimization")
        
        if config.n_trials < 1:
            raise ValueError("Number of trials must be at least 1")
        
        if config.timeout < 60:
            raise ValueError("Timeout must be at least 60 seconds")
        
        if config.direction not in ['maximize', 'minimize']:
            raise ValueError("Direction must be 'maximize' or 'minimize'")
    
    def _run_optimization(self, optimization_id: str):
        """Run hyperparameter optimization"""
        try:
            task = self.active_optimizations[optimization_id]
            config = task['config']
            training_data = task['training_data']
            
            logger.info(f"Running optimization {optimization_id} for {config.model_name}")
            
            # Create Optuna study
            study = optuna.create_study(
                direction=config.direction,
                sampler=self._get_sampler(config.optimization_type)
            )
            
            # Define objective function
            objective = self._create_objective_function(config, training_data)
            
            # Run optimization
            start_time = time.time()
            study.optimize(
                objective,
                n_trials=config.n_trials,
                timeout=config.timeout,
                show_progress_bar=False
            )
            duration = time.time() - start_time
            
            # Create optimization result
            result = OptimizationResult(
                optimization_id=optimization_id,
                model_name=config.model_name,
                best_params=study.best_params,
                best_score=study.best_value,
                optimization_history=self._extract_optimization_history(study),
                duration=duration,
                n_trials=len(study.trials),
                status='completed',
                metadata={
                    'optimization_type': config.optimization_type,
                    'metric': config.metric,
                    'constraints': config.constraints
                },
                timestamp=datetime.now()
            )
            
            # Store result
            with self.lock:
                self.optimization_history[optimization_id] = result
                del self.active_optimizations[optimization_id]
            
            # Save history
            self._save_optimization_history()
            
            # Apply best parameters if auto-apply is enabled
            if config.metadata.get('auto_apply', False):
                self._apply_optimized_parameters(config.model_name, study.best_params)
            
            logger.info(f"Completed optimization {optimization_id} with best score: {study.best_value}")
            
        except Exception as e:
            logger.error(f"Error in optimization {optimization_id}: {e}")
            
            # Mark as failed
            with self.lock:
                if optimization_id in self.active_optimizations:
                    del self.active_optimizations[optimization_id]
    
    def _get_sampler(self, optimization_type: str) -> optuna.samplers.BaseSampler:
        """Get Optuna sampler based on optimization type"""
        if optimization_type == 'bayesian':
            return optuna.samplers.TPESampler()
        elif optimization_type == 'random':
            return optuna.samplers.RandomSampler()
        elif optimization_type == 'cmaes':
            return optuna.samplers.CmaEsSampler()
        else:
            return optuna.samplers.TPESampler()  # Default
    
    def _create_objective_function(self, config: OptimizationConfig, 
                                 training_data: Any) -> Callable:
        """Create objective function for optimization"""
        def objective(trial):
            try:
                # Suggest hyperparameters
                params = self._suggest_hyperparameters(trial, config.search_space)
                
                # Train and evaluate model
                score = self._evaluate_model(config.model_name, params, training_data, config.metric)
                
                # Apply constraints
                if not self._check_constraints(params, config.constraints):
                    return float('inf') if config.direction == 'minimize' else float('-inf')
                
                return score
                
            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return float('inf') if config.direction == 'minimize' else float('-inf')
        
        return objective
    
    def _suggest_hyperparameters(self, trial: optuna.Trial, 
                               search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest hyperparameters using Optuna trial"""
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
            elif param_type == 'loguniform':
                params[param_name] = trial.suggest_loguniform(
                    param_name,
                    param_config['low'],
                    param_config['high']
                )
        
        return params
    
    def _evaluate_model(self, model_name: str, params: Dict[str, Any], 
                       training_data: Any, metric: str) -> float:
        """Evaluate model with given parameters"""
        try:
            start_time = time.time()
            
            # Get model component
            model_component = self.ai_components.get(model_name)
            if not model_component:
                raise ValueError(f"Model {model_name} not found")
            
            # Apply parameters to model
            self._apply_parameters_to_model(model_component, params)
            
            # Train model (if applicable)
            if hasattr(model_component, 'train') and training_data:
                model_component.train(training_data)
            
            # Evaluate model
            if hasattr(model_component, 'evaluate'):
                score = model_component.evaluate(training_data, metric=metric)
            else:
                # Default evaluation
                score = self._default_evaluation(model_component, training_data, metric)
            
            training_time = time.time() - start_time
            
            # Cache performance
            performance = ModelPerformance(
                model_name=model_name,
                params=params,
                score=score,
                training_time=training_time,
                inference_time=0.0,  # Would measure this separately
                memory_usage=0.0,    # Would measure this separately
                timestamp=datetime.now()
            )
            
            self.performance_cache[f"{model_name}_{hash(str(params))}"] = performance
            
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return float('inf') if metric in ['mse', 'mae'] else float('-inf')
    
    def _apply_parameters_to_model(self, model_component: Any, params: Dict[str, Any]):
        """Apply parameters to model component"""
        try:
            # Apply parameters based on model type
            if hasattr(model_component, 'set_hyperparameters'):
                model_component.set_hyperparameters(params)
            elif hasattr(model_component, 'config'):
                model_component.config.update(params)
            else:
                # Try to set parameters directly
                for param_name, param_value in params.items():
                    if hasattr(model_component, param_name):
                        setattr(model_component, param_name, param_value)
            
        except Exception as e:
            logger.error(f"Error applying parameters to model: {e}")
    
    def _default_evaluation(self, model_component: Any, training_data: Any, metric: str) -> float:
        """Default model evaluation"""
        try:
            # This is a placeholder - actual evaluation would depend on the model
            if hasattr(model_component, 'predict'):
                # Simple prediction-based evaluation
                predictions = model_component.predict(training_data)
                if metric == 'accuracy':
                    return accuracy_score(training_data['y_true'], predictions)
                elif metric == 'f1':
                    return f1_score(training_data['y_true'], predictions, average='weighted')
                elif metric == 'mse':
                    return mean_squared_error(training_data['y_true'], predictions)
                else:
                    return 0.5  # Default score
            
            return 0.5  # Default score
            
        except Exception as e:
            logger.error(f"Error in default evaluation: {e}")
            return 0.0
    
    def _check_constraints(self, params: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """Check if parameters satisfy constraints"""
        try:
            for constraint_name, constraint_config in constraints.items():
                if constraint_name == 'max_training_time':
                    if params.get('training_time', 0) > constraint_config:
                        return False
                elif constraint_name == 'max_memory':
                    if params.get('memory_usage', 0) > constraint_config:
                        return False
                elif constraint_name == 'min_score':
                    if params.get('score', 0) < constraint_config:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking constraints: {e}")
            return False
    
    def _extract_optimization_history(self, study: optuna.Study) -> List[Dict[str, Any]]:
        """Extract optimization history from Optuna study"""
        history = []
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial_number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    'duration': trial.duration.total_seconds() if trial.duration else 0
                })
        
        return history
    
    def _apply_optimized_parameters(self, model_name: str, best_params: Dict[str, Any]):
        """Apply optimized parameters to the model"""
        try:
            model_component = self.ai_components.get(model_name)
            if model_component:
                self._apply_parameters_to_model(model_component, best_params)
                
                # Save optimized model
                self.model_manager.save_model(
                    f"{model_name}_optimized",
                    model_component,
                    metadata={'optimization_params': best_params}
                )
                
                logger.info(f"Applied optimized parameters to {model_name}")
            
        except Exception as e:
            logger.error(f"Error applying optimized parameters: {e}")
    
    def get_optimization_status(self, optimization_id: str) -> Dict[str, Any]:
        """Get status of optimization"""
        try:
            if optimization_id in self.active_optimizations:
                task = self.active_optimizations[optimization_id]
                return {
                    'status': task['status'],
                    'progress': task['progress'],
                    'start_time': task['start_time'].isoformat(),
                    'duration': (datetime.now() - task['start_time']).total_seconds()
                }
            elif optimization_id in self.optimization_history:
                result = self.optimization_history[optimization_id]
                return {
                    'status': result.status,
                    'best_score': result.best_score,
                    'best_params': result.best_params,
                    'duration': result.duration,
                    'n_trials': result.n_trials,
                    'completed_at': result.timestamp.isoformat()
                }
            else:
                return {'status': 'not_found'}
                
        except Exception as e:
            logger.error(f"Error getting optimization status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_optimization_history(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get optimization history"""
        try:
            history = []
            
            for result in self.optimization_history.values():
                if model_name is None or result.model_name == model_name:
                    history.append({
                        'optimization_id': result.optimization_id,
                        'model_name': result.model_name,
                        'best_score': result.best_score,
                        'best_params': result.best_params,
                        'status': result.status,
                        'duration': result.duration,
                        'n_trials': result.n_trials,
                        'timestamp': result.timestamp.isoformat()
                    })
            
            # Sort by timestamp (newest first)
            history.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting optimization history: {e}")
            return []
    
    def get_best_parameters(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get best parameters for a model"""
        try:
            best_result = None
            best_score = float('-inf')
            
            for result in self.optimization_history.values():
                if (result.model_name == model_name and 
                    result.status == 'completed' and 
                    result.best_score > best_score):
                    best_result = result
                    best_score = result.best_score
            
            return best_result.best_params if best_result else None
            
        except Exception as e:
            logger.error(f"Error getting best parameters: {e}")
            return None
    
    def get_performance_cache(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get performance cache"""
        try:
            cache_data = []
            
            for key, performance in self.performance_cache.items():
                if model_name is None or performance.model_name == model_name:
                    cache_data.append({
                        'model_name': performance.model_name,
                        'params': performance.params,
                        'score': performance.score,
                        'training_time': performance.training_time,
                        'timestamp': performance.timestamp.isoformat()
                    })
            
            return cache_data
            
        except Exception as e:
            logger.error(f"Error getting performance cache: {e}")
            return []
    
    def clear_performance_cache(self):
        """Clear performance cache"""
        try:
            self.performance_cache.clear()
            logger.info("Performance cache cleared")
        except Exception as e:
            logger.error(f"Error clearing performance cache: {e}")
    
    def get_optimization_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined optimization configurations"""
        return {
            'gnn_optimization': {
                'model_name': 'gnn',
                'optimization_type': 'bayesian',
                'n_trials': 50,
                'timeout': 3600,
                'metric': 'accuracy',
                'direction': 'maximize',
                'constraints': {
                    'max_training_time': 300,
                    'min_score': 0.7
                },
                'search_space': {
                    'embedding_dim': {'type': 'int', 'low': 32, 'high': 256},
                    'num_layers': {'type': 'int', 'low': 2, 'high': 8},
                    'learning_rate': {'type': 'loguniform', 'low': 1e-4, 'high': 1e-2},
                    'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5}
                }
            },
            'matching_optimization': {
                'model_name': 'matching',
                'optimization_type': 'tpe',
                'n_trials': 30,
                'timeout': 1800,
                'metric': 'f1',
                'direction': 'maximize',
                'constraints': {
                    'max_training_time': 200,
                    'min_score': 0.6
                },
                'search_space': {
                    'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                    'max_depth': {'type': 'int', 'low': 3, 'high': 15}
                }
            },
            'federated_optimization': {
                'model_name': 'federated',
                'optimization_type': 'bayesian',
                'n_trials': 40,
                'timeout': 2400,
                'metric': 'accuracy',
                'direction': 'maximize',
                'constraints': {
                    'max_training_time': 400,
                    'min_score': 0.65
                },
                'search_space': {
                    'learning_rate': {'type': 'loguniform', 'low': 1e-4, 'high': 1e-2},
                    'local_epochs': {'type': 'int', 'low': 1, 'high': 10},
                    'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]}
                }
            }
        }

# Global optimizer instance
hyperparameter_optimizer = AIHyperparameterOptimizer() 