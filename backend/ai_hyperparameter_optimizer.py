"""
Production-Grade AI Hyperparameter Optimization System
Automated hyperparameter tuning for all AI models using Optuna and advanced techniques
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.visualization import plot_optimization_history, plot_param_importances
import scipy.optimize as scipy_opt
from scipy.stats import uniform, loguniform, randint
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
import mlflow
import wandb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import joblib
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ML Core imports
from ml_core.models import (
    ModelFactory,
    ModelArchitecture,
    ModelConfig
)
from ml_core.training import (
    ModelTrainer,
    TrainingConfig,
    TrainingMetrics
)
from ml_core.data_processing import (
    DataProcessor,
    DataValidator
)
from ml_core.optimization import (
    BaseOptimizer,
    OptimizationStrategy,
    SearchSpace,
    HyperparameterOptimizer
)
from ml_core.monitoring import (
    MLMetricsTracker,
    OptimizationMonitor
)
from ml_core.utils import (
    ModelRegistry,
    ExperimentTracker,
    ConfigManager
)

class HyperparameterSearchSpace:
    """Comprehensive search space definition for hyperparameter optimization"""
    def __init__(self, model_type: str = 'transformer'):
        self.model_type = model_type
        self.search_spaces = self._define_search_spaces()
    
    def _define_search_spaces(self) -> Dict:
        """Define fine comprehensive search spaces for different model types"""
        base_spaces = {
            # Learning rate and optimization
            'learning_rate': {
                'type': 'uniform',
                'min': 1e-6,
                'max': 1e-3,
                'default': 1e-4
            },
            'weight_decay': {
                'type': 'uniform',
                'min': 1e-6,
                'max': 1e-2,
                'default': 1e-4
            },
            'batch_size': {
                'type': 'categorical',
                'choices': [8, 16, 32, 64],
                'default': 32
            },
            
            # Model architecture
            'hidden_size': {
                'type': 'categorical',
                'choices': [256, 512, 768],
                'default': 768
            },
            'num_layers': {
                'type': 'int_uniform',
                'min': 2,
                'max': 4,
                'default': 6
            },
            'num_heads': {
                'type': 'categorical',
                'choices': [4, 8, 12, 16, 24],
                'default': 12
            },
            'dropout': {
                'type': 'uniform',
                'min': 0.0,
                'max': 0.5,
                'default': 0.1
            },
            
            # Training configuration
            'warmup_steps': {
                'type': 'int_uniform',
                'min': 0,
                'max': 10,
                'default': 10
            },
            'gradient_accumulation_steps': {
                'type': 'categorical',
                'choices': [1, 2, 4, 8],
                'default': 1
            },
            'max_grad_norm': {
                'type': 'uniform',
                'min': 1.0,
                'max': 10.0,
                'default': 10.0
            },
            
            # Regularization
            'label_smoothing': {
                'type': 'uniform',
                'min': 0.0,
                'max': 0.2,
                'default': 0.0
            },
            'mixup_alpha': {
                'type': 'uniform',
                'min': 0.0,
                'max': 0.0,
                'default': 0.0
            },
            
            # Advanced features
            'use_amp': {
                'type': 'categorical',
                'choices': [True, False],
                'default': True
            },
            'use_gradient_checkpointing': {
                'type': 'categorical',
                'choices': [True, False],
                'default': False
            }
        }
        
        # Model-specific spaces
        if self.model_type == 'transformer':
            base_spaces.update({
                'attention_dropout': {
                    'type': 'uniform',
                    'min': 0.0,
                    'max': 0.5,
                    'default': 0.1
                },
                'layer_norm_eps': {
                    'type': 'uniform',
                    'min': -5.0,
                    'max': 1e-3,
                    'default': 1e-5
                }
            })
        elif self.model_type == 'cnn':
            base_spaces.update({
                'kernel_size': {
                    'type': 'categorical',
                    'choices': [3, 5, 7, 9],
                    'default': 3
                },
                'num_filters': {
                    'type': 'categorical',
                    'choices': [32, 64, 128],
                    'default': 64
                }
            })
        
        return base_spaces
    
    def get_optuna_space(self) -> Dict:
        """Convert to Optuna search space"""
        space = {}
        
        for param_name, param_config in self.search_spaces.items():
            if param_config['type'] == 'uniform':
                space[param_name] = optuna.distributions.UniformDistribution(
                    param_config['min'], param_config['max']
                )
            elif param_config['type'] == 'log_uniform':
                space[param_name] = optuna.distributions.LogUniformDistribution(
                    param_config['min'], param_config['max']
                )
            elif param_config['type'] == 'int_uniform':
                space[param_name] = optuna.distributions.IntUniformDistribution(
                    param_config['min'], param_config['max']
                )
            elif param_config['type'] == 'categorical':
                space[param_name] = optuna.distributions.CategoricalDistribution(
                    param_config['choices']
                )
        
        return space
    
    def get_hyperopt_space(self) -> Dict:
        """Convert to Hyperopt search space"""
        space = {}
        
        for param_name, param_config in self.search_spaces.items():
            if param_config['type'] == 'uniform':
                space[param_name] = hp.uniform(param_name, param_config['min'], param_config['max'])
            elif param_config['type'] == 'log_uniform':
                space[param_name] = hp.loguniform(param_name, np.log(param_config['min']), np.log(param_config['max']))
            elif param_config['type'] == 'int_uniform':
                space[param_name] = hp.randint(param_name, param_config['min'], param_config['max'] + 1)
            elif param_config['type'] == 'categorical':
                space[param_name] = hp.choice(param_name, param_config['choices'])
        
        return space
    
    def get_ray_tune_space(self) -> Dict:
        """Convert to Ray Tune search space"""
        space = {}
        
        for param_name, param_config in self.search_spaces.items():
            if param_config['type'] == 'uniform':
                space[param_name] = tune.uniform(param_config['min'], param_config['max'])
            elif param_config['type'] == 'log_uniform':
                space[param_name] = tune.loguniform(param_config['min'], param_config['max'])
            elif param_config['type'] == 'int_uniform':
                space[param_name] = tune.randint(param_config['min'], param_config['max'] + 1)
            elif param_config['type'] == 'categorical':
                space[param_name] = tune.choice(param_config['choices'])
        
        return space

class MultiObjectiveOptimizer:
    """Multi-objective hyperparameter optimization using NSGA-II"""
    def __init__(self, objectives: List[str], weights: List[float] = None):
        self.objectives = objectives
        self.weights = weights if weights else [1.0] * len(objectives)
        
    def optimize(self, 
                objective_function: Callable,
                search_space: HyperparameterSearchSpace,
                n_trials: int = 100,
                population_size: int = 50) -> Dict:
        """Perform multi-objective optimization using NSGA-II"""
        def multi_objective_function(trial):
            # Sample hyperparameters
            params = {}
            optuna_space = search_space.get_optuna_space()
            
            for param_name, distribution in optuna_space.items():
                params[param_name] = trial.suggest(param_name, distribution)
            
            # Evaluate objectives
            results = objective_function(params)
            
            # Return multiple objectives
            return [results[obj] for obj in self.objectives]
        
        # Create study for multi-objective optimization
        study = optuna.create_study(
            directions=['maximize'] * len(self.objectives),
            sampler=optuna.samplers.NSGAIISampler(population_size=population_size)
        )
        
        study.optimize(multi_objective_function, n_trials=n_trials)
        
        # Get Pareto front
        pareto_front = study.best_trials
        
        return {
         'pareto_front': pareto_front,
          'study': study,
        'best_params': self._select_best_from_pareto(pareto_front)
        }
    
    def _select_best_from_pareto(self, pareto_front: List) -> Dict:
        """Select best solution from Pareto front using weighted sum"""
        best_score = float('-inf')
        best_params = None
        
        for trial in pareto_front:
            weighted_score = sum(
                trial.values[i] * self.weights[i] 
                for i in range(len(self.objectives))
            )
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_params = trial.params
        
        return best_params

class NeuralArchitectureSearch:
    """Neural Architecture Search using reinforcement learning"""
    def __init__(self, 
                 model_factory: ModelFactory,
                 search_space: HyperparameterSearchSpace):
        self.model_factory = model_factory
        self.search_space = search_space
        self.controller = self._build_controller()
        
    def _build_controller(self) -> nn.Module:
        """Build RNN controller for architecture search"""
        return nn.LSTM(
            input_size=len(self.search_space.search_spaces),
            hidden_size=100,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
    
    def search(self, 
              objective_function: Callable,
              n_episodes: int = 100,
              n_architectures_per_episode: int = 10) -> Dict:
        """Perform neural architecture search"""
        best_architecture = None
        best_score = float('-inf')
        architecture_history = []
        
        for episode in range(n_episodes):
            # Generate architectures
            architectures = self._generate_architectures(n_architectures_per_episode)
            
            # Evaluate architectures
            scores = []
            for arch in architectures:
                try:
                    score = objective_function(arch)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_architecture = arch
            
                except Exception as e:
                    scores.append(float('-inf'))
            
            # Update controller
            self._update_controller(architectures, scores)
            
            # Log progress
            architecture_history.append({
                'episode': episode,
                'avg_score': np.mean(scores),
                'best_score': max(scores),
                'best_architecture': best_architecture
            })
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Best Score = {best_score:.4f}")
        
        return {
            'best_architecture': best_architecture,
            'best_score': best_score,
            'history': architecture_history
        }
    
    def _generate_architectures(self, n_architectures: int) -> List[Dict]:
        """Generate architectures using controller"""
        architectures = []
        
        for _ in range(n_architectures):
            # Sample from controller
            controller_output = torch.randn(1, len(self.search_space.search_spaces))
            
            # Convert to architecture
            arch = {}
            for i, (param_name, param_config) in enumerate(self.search_space.search_spaces.items()):
                if param_config['type'] == 'categorical':
                    arch[param_name] = param_config['choices'][
                        int(controller_output[0, i] * len(param_config['choices']))
                    ]
                else:
                    # Map to parameter range
                    min_val = param_config['min']
                    max_val = param_config['max']
                    arch[param_name] = min_val + (max_val - min_val) * torch.sigmoid(controller_output[0, i]).item()
            
            architectures.append(arch)
        
        return architectures
    
    def _update_controller(self, architectures: List[Dict], scores: List[float]):
        """Update controller using REINFORCE"""
        # Convert architectures to sequences
        sequences = []
        for arch in architectures:
            seq = []
            for param_name in self.search_space.search_spaces.keys():
                param_value = arch[param_name]
                # Normalize parameter value
                if isinstance(param_value, (int, float)):
                    seq.append(param_value)
                else:
                    seq.append(hash(param_value) % 1000)  # Hash categorical values
            sequences.append(seq)
        
        # Convert to tensor
        sequences_tensor = torch.FloatTensor(sequences)
        
        # Forward pass
        controller_output, _ = self.controller(sequences_tensor)
        
        # Calculate loss (simplified REINFORCE)
        scores_tensor = torch.FloatTensor(scores)
        loss = -torch.mean(controller_output * scores_tensor.unsqueeze(1))
        
        # Backward pass
        loss.backward()
        
        # Update controller (simplified - would need proper optimizer)
        with torch.no_grad():
            for param in self.controller.parameters():
                param -= 0.01 * param.grad # Simplified update
                param.grad.zero_()

class AdvancedHyperparameterOptimizer:
    """Advanced hyperparameter optimizer with multiple optimization strategies"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model_factory = ModelFactory()
        self.model_registry = ModelRegistry()
        self.experiment_tracker = ExperimentTracker()
        self.metrics_tracker = MLMetricsTracker()
        self.optimization_monitor = OptimizationMonitor()
        self.config_manager = ConfigManager()
        
        # Optimization strategies
        self.optimization_strategies = {
            'bayesian': self._bayesian_optimization,
            'evolutionary': self._evolutionary_optimization,
            'multi_objective': self._multi_objective_optimization,
            'neural_architecture_search': self._neural_architecture_search,
            'ray_tune': self._ray_tune_optimization,
            'hyperopt': self._hyperopt_optimization
        }
        
        # Initialize experiment tracking
        self._setup_experiment_tracking()
        
        # Optimization configuration
        self.optimization_config = {
            'n_trials': 100,
            'timeout': 3600,  # 1hour
            'n_jobs': 4,
            'pruner': 'median',
            'sampler': 'tpe',
            'study_name': 'hyperparameter_optimization'
        }
    
    def _setup_experiment_tracking(self):
        """Experiment tracking"""
        try:
            # MLflow setup
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("hyperparameter_optimization")
            
            # WandB setup
            if os.getenv('WANDB_API_KEY'):
                wandb.init(project="hyperparameter-optimization")
            
        except Exception as e:
            self.logger.warning(f"Experiment tracking setup failed: {e}")
    
    async def optimize_hyperparameters(self,
                                     model_type: str,
                                     training_data: List[Dict],
                                     validation_data: List[Dict],
                                     optimization_strategy: str = 'bayesian',
                                     search_space: Dict = None,
                                     objectives: List[str] = None,
                                     constraints: Dict = None) -> Dict:
        """Optimize hyperparameters using advanced techniques"""
        try:
            self.logger.info(f"Starting hyperparameter optimization with strategy: {optimization_strategy}")
            
            # Start experiment tracking
            with mlflow.start_run(run_name=f"hpo_{model_type}_{optimization_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Define search space
                if search_space is None:
                    search_space = HyperparameterSearchSpace(model_type)
                
                # Define objectives
                if objectives is None:
                    objectives = ['accuracy', 'latency']
                
                # Prepare objective function
                objective_function = self._create_objective_function(
                    model_type, training_data, validation_data, objectives
                )
                
                # Run optimization
                if optimization_strategy in self.optimization_strategies:
                    results = await self.optimization_strategies[optimization_strategy](
                        objective_function, search_space, objectives, constraints
                    )
                else:
                    raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")
                
                # Log results
                mlflow.log_params(results['best_params'])
                mlflow.log_metrics(results['best_scores'])
                
                # Save optimization results
                self._save_optimization_results(results, model_type, optimization_strategy)
                
                return results
            
        except Exception as e:
            self.logger.error(f"Error during hyperparameter optimization: {e}")
            raise
    
    def _create_objective_function(self,
                                 model_type: str,
                                 training_data: List[Dict],
                                 validation_data: List[Dict],
                                 objectives: List[str]) -> Callable:
        """Objective function for optimization"""
        def objective_function(params: Dict) -> Union[float, Dict]:
            try:
                # Create model with given parameters
                model = self.model_factory.create_model(model_type, params)
                
                # Train model
                trainer = ModelTrainer()
                training_config = TrainingConfig(**params)
                
                training_results = trainer.train_model(
                    model=model,
                    training_data=training_data,
                    validation_data=validation_data,
                    config=training_config
                )
                
                # Evaluate model
                evaluation_results = trainer.evaluate_model(
                    model=model,
                    validation_data=validation_data
                )
                
                # Calculate objectives
                objective_values = {}
                for objective in objectives:
                    if objective == 'accuracy':
                        objective_values[objective] = evaluation_results.get('accuracy', 0.0)
                    elif objective == 'precision':
                        objective_values[objective] = evaluation_results.get('precision', 0.0)
                    elif objective == 'recall':
                        objective_values[objective] = evaluation_results.get('recall', 0.0)
                    elif objective == 'f1':
                        objective_values[objective] = evaluation_results.get('f1_score', 0.0)
                    elif objective == 'latency':
                        objective_values[objective] = self._measure_latency(model, validation_data)
                    elif objective == 'memory':
                        objective_values[objective] = self._measure_memory_usage(model)
                    else:
                        objective_values[objective] = evaluation_results.get(objective, 0.0)
                
                # Track metrics
                self.metrics_tracker.record_optimization_trial(params, objective_values)
                
                # Return single value for single-objective optimization
                if len(objectives) == 1:
                    return objective_values[objectives[0]]
                else:
                    return objective_values
                
            except Exception as e:
                self.logger.error(f"Error in objective function: {e}")
                # Return worst possible values
                if len(objectives) == 1:
                    return float('-inf')
                else:
                    return {obj: float('-inf') for obj in objectives}
        
        return objective_function
    
    async def _bayesian_optimization(self,
                                   objective_function: Callable,
                                   search_space: HyperparameterSearchSpace,
                                   objectives: List[str],
                                   constraints: Dict = None) -> Dict:
        """Bayesian optimization using Optuna"""
        def optuna_objective(trial):
            # Sample parameters
            params = {}
            optuna_space = search_space.get_optuna_space()
            for param_name, distribution in optuna_space.items():
                params[param_name] = trial.suggest(param_name, distribution)
            # Evaluate objective
            result = objective_function(params)
            # Handle multi-objective case
            if isinstance(result, dict):
                # Log all objectives
                for obj_name, obj_value in result.items():
                    trial.set_user_attr(obj_name, obj_value)
                # Return primary objective
                return result[objectives[0]]
            else:
                return result
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(),
            pruner=MedianPruner()
        )
        # Optimize
        study.optimize(
            optuna_objective,
            n_trials=self.optimization_config['n_trials'],
            timeout=self.optimization_config['timeout']
        )
        # Get best results
        best_params = study.best_params
        best_value = study.best_value
        # Extract all objectives from best trial
        best_trial = study.best_trial
        best_scores = {}
        for objective in objectives:
            best_scores[objective] = best_trial.user_attrs.get(objective, best_value)
        return {
        'best_params': best_params,
        'best_scores': best_scores,
          'study': study,
       'optimization_history': study.trials_dataframe()
        }
    
    async def _evolutionary_optimization(self,
                                       objective_function: Callable,
                                       search_space: HyperparameterSearchSpace,
                                       objectives: List[str],
                                       constraints: Dict = None) -> Dict:
        """Evolutionary optimization using CMA-ES"""
        def cma_objective(params_array):
            # Convert array to parameter dict
            params = {}
            param_names = list(search_space.search_spaces.keys())
            
            for i, param_name in enumerate(param_names):
                param_config = search_space.search_spaces[param_name]
                
                if param_config['type'] == 'categorical':
                    # Map continuous value to categorical choice
                    choice_idx = int(params_array[i] * len(param_config['choices']))
                    params[param_name] = param_config['choices'][choice_idx]
                else:
                    # Scale to parameter range
                    min_val = param_config['min']
                    max_val = param_config['max']
                    params[param_name] = min_val + (max_val - min_val) * params_array[i]
            
            # Evaluate objective
            result = objective_function(params)
            
            if isinstance(result, dict):
                return -result[objectives[0]]  # Minimize negative of primary objective
            else:
                return -result  # Minimize negative of objective
        
        # Initialize CMA-ES
        n_params = len(search_space.search_spaces)
        x0= np.array([0.5] * n_params)  # Initial guess
        
        # Run optimization
        result = scipy_opt.minimize(
            cma_objective,
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1)] * n_params,
            options={'maxiter': self.optimization_config['n_trials']}
        )
        
        # Convert back to parameters
        best_params = {}
        param_names = list(search_space.search_spaces.keys())
        
        for i, param_name in enumerate(param_names):
            param_config = search_space.search_spaces[param_name]
            
            if param_config['type'] == 'categorical':
                choice_idx = int(result.x[i] * len(param_config['choices']))
                best_params[param_name] = param_config['choices'][choice_idx]
            else:
                min_val = param_config['min']
                max_val = param_config['max']
                best_params[param_name] = min_val + (max_val - min_val) * result.x[i]
        
        # Evaluate best parameters
        best_result = objective_function(best_params)
        
        if isinstance(best_result, dict):
            best_scores = best_result
        else:
            best_scores = {objectives[0]: best_result}
        return {
            'best_params': best_params,
            'best_scores': best_scores,
            'optimization_result': result
        }
    
    async def _multi_objective_optimization(self,
                                          objective_function: Callable,
                                          search_space: HyperparameterSearchSpace,
                                          objectives: List[str],
                                          constraints: Dict = None) -> Dict:
        """Multi-objective optimization using NSGA-II"""
        optimizer = MultiObjectiveOptimizer(objectives)
        
        results = optimizer.optimize(
            objective_function=objective_function,
            search_space=search_space,
            n_trials=self.optimization_config['n_trials']
        )
        
        return results
    
    async def _neural_architecture_search(self,
                                        objective_function: Callable,
                                        search_space: HyperparameterSearchSpace,
                                        objectives: List[str],
                                        constraints: Dict = None) -> Dict:
        """Neural Architecture Search"""
        nas = NeuralArchitectureSearch(
            model_factory=self.model_factory,
            search_space=search_space
        )
        
        results = nas.search(
            objective_function=objective_function,
            n_episodes=self.optimization_config['n_trials'] // 10
        )
        
        return results
    
    async def _ray_tune_optimization(self,
                                   objective_function: Callable,
                                   search_space: HyperparameterSearchSpace,
                                   objectives: List[str],
                                   constraints: Dict = None) -> Dict:
        """Tune optimization"""
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
        
        def ray_objective(config):
            # Evaluate objective
            result = objective_function(config)
            
            if isinstance(result, dict):
                # Report all objectives
                for obj_name, obj_value in result.items():
                    tune.report(**{obj_name: obj_value})
                return result[objectives[0]]
            else:
                tune.report(objective=result)
                return result
        
        # Define search space
        tune_space = search_space.get_ray_tune_space()
        
        # Define scheduler
        scheduler = ASHAScheduler(
            metric=objectives[0],
            mode='max',
            max_t=100,
            grace_period=10
        )
        
        # Run optimization
        analysis = tune.run(
            ray_objective,
            config=tune_space,
            num_samples=self.optimization_config['n_trials'],
            scheduler=scheduler,
            resources_per_trial={'cpu': 1, 'gpu': 1 if torch.cuda.is_available() else 0}
        )
        
        # Get best results
        best_trial = analysis.get_best_trial(objectives[0], 'max')
        best_params = best_trial.config
        best_scores = best_trial.last_result
        return {
            'best_params': best_params,
            'best_scores': best_scores,
            'analysis': analysis
        }
    
    async def _hyperopt_optimization(self,
                                   objective_function: Callable,
                                   search_space: HyperparameterSearchSpace,
                                   objectives: List[str],
                                   constraints: Dict = None) -> Dict:
        """Hyperopt optimization"""
        def hyperopt_objective(params):
            # Evaluate objective
            result = objective_function(params)
            
            if isinstance(result, dict):
                # Return loss (negative of primary objective)
                return {'loss': -result[objectives[0]], 'status': STATUS_OK}
            else:
                return {'loss': -result, 'status': STATUS_OK}
        
        # Define search space
        hyperopt_space = search_space.get_hyperopt_space()
        
        # Run optimization
        trials = Trials()
        best = fmin(
            fn=hyperopt_objective,
            space=hyperopt_space,
            algo=tpe.suggest,
            max_evals=self.optimization_config['n_trials'],
            trials=trials
        )
        
        # Get best results
        best_params = best
        best_result = objective_function(best_params)
        
        if isinstance(best_result, dict):
            best_scores = best_result
        else:
            best_scores = {objectives[0]: best_result}
        return {
        'best_params': best_params,
        'best_scores': best_scores,
           'trials': trials
        }
    
    def _measure_latency(self, model: nn.Module, validation_data: List[Dict]) -> float:
        """Measure model inference latency"""
        try:
            model.eval()
            
            # Prepare sample data
            sample_data = validation_data[:10] # Use first 10 samples for warm-up
            
            # Warm up
            with torch.no_grad():
                for _ in range(5):
                    _ = model(torch.randn(1, 512).to(self.device))
            
            # Measure latency
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time and end_time:
                start_time.record()
                with torch.no_grad():
                    for sample in sample_data:
                        _ = model(torch.randn(1, 512).to(self.device))
                end_time.record()
                torch.cuda.synchronize()
                latency = start_time.elapsed_time(end_time) / len(sample_data)
            else:
                import time
                start_time = time.time()
                with torch.no_grad():
                    for sample in sample_data:
                        _ = model(torch.randn(1, 512).to(self.device))
                end_time = time.time()
                latency = (end_time - start_time) * 1000 / len(sample_data)  # Convert to ms
            
            return latency
            
        except Exception as e:
            self.logger.error(f"Error measuring latency: {e}")
            return float('inf')
    
    def _measure_memory_usage(self, model: nn.Module) -> float:
        """Ensure model memory usage"""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Estimate memory usage (rough approximation)
            memory_mb = total_params * 4 / (1024 * 1024) # Bytes per parameter
            
            return memory_mb
            
        except Exception as e:
            self.logger.error(f"Error measuring memory usage: {e}")
            return float('inf')
    
    def _save_optimization_results(self, results: Dict, model_type: str, strategy: str):
        """Save optimization results"""
        try:
            # Create results directory
            results_dir = f"optimization_results/{model_type}/{strategy}"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save parameters
            with open(f"{results_dir}/best_params_{timestamp}.json", 'w') as f:
                json.dump(results['best_params'], f, indent=2)
            
            # Save scores
            with open(f"{results_dir}/best_scores_{timestamp}.json", 'w') as f:
                json.dump(results['best_scores'], f, indent=2)
            
            # Save optimization history if available
            if 'optimization_history' in results:
                results['optimization_history'].to_csv(
                    f"{results_dir}/optimization_history_{timestamp}.csv",
                    index=False
                )
            
            # Create visualization
            self._create_optimization_visualization(results, f"{results_dir}/visualization_{timestamp}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimization results: {e}")
    
    def _create_optimization_visualization(self, results: Dict, output_path: str):
        """Create optimization visualization"""
        try:
            if 'optimization_history' in results:
                df = results['optimization_history']
                
                # Create subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Optimization history
                if 'value' in df.columns:
                    axes[0,0].plot(df['value'])
                    axes[0, 0].set_title('Optimization History')
                    axes[0, 0].set_xlabel('Trial')
                    axes[0, 0].set_ylabel('Objective Value')
                
                # Parameter importance (if available)
                if hasattr(results['study'], 'get_param_importances'):
                    importances = results['study'].get_param_importances()
                    param_names = list(importances.keys())
                    importance_values = list(importances.values())
                    
                    axes[0, 1].barh(param_names, importance_values)
                    axes[0, 1].set_title('Parameter Importance')
                    axes[0, 1].set_xlabel('Importance')
                
                # Parameter distributions
                if 'params' in df.columns:
                    # Plot distribution of a few key parameters
                    key_params = ['learning_rate', 'batch_size', 'hidden_size']
                    for i, param in enumerate(key_params):
                        if param in df.columns:
                            axes[1, i].hist(df[param], bins=20, alpha=0.7)
                            axes[1, i].set_title(f'{param} Distribution')
                            axes[1, i].set_xlabel(param)
                
                plt.tight_layout()
                plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
    
    async def get_system_health(self) -> Dict:
        """Get system health metrics"""
        try:
            health_metrics = {
                'status': 'healthy',
                'device': str(self.device),
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'optimization_monitor_status': self.optimization_monitor.get_status(),
                'metrics_tracker_status': self.metrics_tracker.get_status(),
                'experiment_tracking_status': 'active' if mlflow.active_run() else 'inactive',
                'ray_status': 'initialized' if ray.is_initialized() else 'not_initialized'
            }
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {'status': 'error', 'error': str(e)}

class AIHyperparameterOptimizer(HyperparameterOptimizer):
    pass

# Initialize service
advanced_hyperparameter_optimizer = AdvancedHyperparameterOptimizer() 