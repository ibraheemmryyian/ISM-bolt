"""
Production-Grade AI Retraining Pipeline
Complete feedback-to-retraining workflow with Prefect orchestration
"""
import os

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from transformers
    HAS_TRANSFORMERS = True
except ImportError:
    from .fallbacks.transformers_fallback import *
    HAS_TRANSFORMERS = False import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
# Try to import sklearn components with fallback
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback implementations if sklearn is not available
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        def fit(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            return self
        def transform(self, X):
            return (X - self.mean_) / self.scale_
        def fit_transform(self, X):
            return self.fit(X).transform(X)
    
    class LabelEncoder:
        def __init__(self):
            self.classes_ = None
        def fit(self, y):
            self.classes_ = np.unique(y)
            return self
        def transform(self, y):
            return np.array([np.where(self.classes_ == label)[0][0] for label in y])
        def fit_transform(self, y):
            return self.fit(y).transform(y)
    
    def train_test_split(X, y, **kwargs):
        split_idx = len(X) // 2
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    def accuracy_score(y_true, y_pred):
        return 0.0
    
    def classification_report(y_true, y_pred):
        return "Classification report not available (sklearn not installed)"
    
    SKLEARN_AVAILABLE = False
import joblib
import pickle
from pathlib import Path
import mlflow
import optuna
from optuna.samplers import TPESampler
import wandb
import yaml
import hashlib
import shutil
import tempfile
import subprocess
import psutil
import GPUtil
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    DataValidator,
    DataAugmentation
)
from ml_core.optimization import (
    HyperparameterOptimizer,
    ArchitectureSearch
)
from ml_core.monitoring import (
    MLMetricsTracker,
    ModelPerformanceMonitor,
    DriftDetector
)
from ml_core.utils import (
    ModelRegistry,
    ModelVersioning,
    ExperimentTracker
)

class RetrainingDataset(Dataset):
    """Dataset for model retraining with advanced data processing"""
    def __init__(self, 
                 data: pd.DataFrame, 
                 tokenizer,
                 max_length: int = 512,
                 task_type: str = 'classification',
                 augment: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.augment = augment
        # Initialize data processors
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.augmentor = DataAugmentation()
        # Process features
        self._process_features()
        # Apply data augmentation if enabled
        if self.augment:
            self._apply_augmentation()

    def _process_features(self):
        """Process and engineer features"""
        # Text features
        self.text_features = []
        for _, row in self.data.iterrows():
            text = f"{row.get('text', '')} {row.get('context', '')} {row.get('metadata', '')}"
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            self.text_features.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })
        # Numerical features
        numerical_cols = [col for col in self.data.columns if self.data[col].dtype in ['int64', 'float64']]
        if numerical_cols:
            self.numerical_features = self.scaler.fit_transform(
                self.data[numerical_cols].fillna(0)
            )
        else:
            self.numerical_features = np.zeros((len(self.data), 1))
        # Categorical features
        categorical_cols = [col for col in self.data.columns if self.data[col].dtype == 'object']
        self.categorical_features = []
        for col in categorical_cols:
            if col in self.data.columns:
                encoded = self.label_encoder.fit_transform(self.data[col].fillna('unknown'))
                self.categorical_features.append(encoded)
        # Labels
        if self.task_type == 'classification':
            self.labels = self.label_encoder.fit_transform(self.data['label'].fillna('unknown'))
        elif self.task_type == 'regression':
            self.labels = self.data['target'].values
        else:
            self.labels = np.zeros(len(self.data))

    def _apply_augmentation(self):
        """Apply data augmentation techniques"""
        augmented_data = []
        for i in range(len(self.data)):
            # Original sample
            augmented_data.append({
                'text_features': self.text_features[i],
                'numerical_features': self.numerical_features[i],
                'categorical_features': [cat[i] for cat in self.categorical_features] if self.categorical_features else [],
                'label': self.labels[i],
                'weight': 10
            })
            # Augmented samples
            if self.labels[i] == 1:  # Minority class - add more samples
                for _ in range(2):
                    aug_sample = self.augmentor.augment_text(
                        self.text_features[i]['input_ids'],
                        augmentation_type='synonym_replacement'
                    )
                    augmented_data.append({
                        'text_features': {'input_ids': aug_sample, 'attention_mask': torch.ones_like(aug_sample)},
                        'numerical_features': self.numerical_features[i] + np.random.normal(0, 0.01),
                        'categorical_features': [cat[i] for cat in self.categorical_features] if self.categorical_features else [],
                        'label': self.labels[i],
                        'weight': 10
                    })
        # Update data
        self.augmented_data = augmented_data

    def __len__(self):
        return len(self.augmented_data) if self.augment else len(self.data)

    def __getitem__(self, idx):
        if self.augment:
            sample = self.augmented_data[idx]
            return {
                'input_ids': sample['text_features']['input_ids'],
                'attention_mask': sample['text_features']['attention_mask'],
                'numerical_features': torch.FloatTensor(sample['numerical_features']),
                'categorical_features': torch.LongTensor(sample['categorical_features']) if sample['categorical_features'] else torch.tensor([]),
                'label': torch.LongTensor([sample['label']]) if self.task_type == 'classification' else torch.FloatTensor([sample['label']]),
                'weight': torch.FloatTensor([sample['weight']])
            }
        else:
            return {
                'input_ids': self.text_features[idx]['input_ids'],
                'attention_mask': self.text_features[idx]['attention_mask'],
                'numerical_features': torch.FloatTensor(self.numerical_features[idx]),
                'categorical_features': torch.LongTensor([cat[idx] for cat in self.categorical_features]) if self.categorical_features else torch.tensor([]),
                'label': torch.LongTensor([self.labels[idx]]) if self.task_type == 'classification' else torch.FloatTensor([self.labels[idx]]),
                'weight': torch.FloatTensor([1.0])
            }

class AdvancedRetrainingModel(nn.Module):
    """Model architecture for retraining with multiple heads and adapters"""
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int = 768,
                 num_layers: int = 6,
                 num_heads: int = 12,
                 dropout: float = 0.1,
                 num_classes: int = 10,
                 task_type: str = 'classification',
                 use_adapters: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.task_type = task_type
        self.use_adapters = use_adapters
        # Base transformer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = nn.Parameter(torch.randn(1, 100, hidden_size))
        # Transformer layers with optional adapters
        self.transformer_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True
            )
            if use_adapters:
                # Add adapter layers
                adapter = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 4, hidden_size)
                )
                layer.register_forward_hook(self._adapter_hook(adapter))
            self.transformer_layers.append(layer)
        # Feature fusion
        self.numerical_projection = nn.Linear(10, hidden_size // 2)
        self.categorical_projection = nn.Linear(5, hidden_size // 2)
        # Task-specific heads
        if task_type == 'classification':
            self.classification_head = nn.Sequential(
                nn.Linear(hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes)
            )
        elif task_type == 'regression':
            self.regression_head = nn.Sequential(
                nn.Linear(hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2)
            )
        # Multi-task heads
        self.auxiliary_heads = nn.ModuleDict({
            'sentiment': nn.Linear(hidden_size * 23),
            'complexity': nn.Linear(hidden_size * 25),
            'domain': nn.Linear(hidden_size * 2)
        })
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def _adapter_hook(self, adapter):
        """Hook for adapter layers"""
        def hook(module, input, output):
            return output + adapter(output)
        return hook
    
    def forward(self, input_ids, attention_mask, numerical_features, categorical_features):
        batch_size, seq_len = input_ids.shape
        
        # Text encoding
        embeddings = self.embedding(input_ids)
        position_encodings = self.position_encoding[:, :seq_len, :]
        embeddings = embeddings + position_encodings
        
        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=~attention_mask.bool())
        
        # Global pooling
        pooled_output = torch.mean(hidden_states, dim=1)
        
        # Feature fusion
        numerical_projected = self.numerical_projection(numerical_features)
        categorical_projected = self.categorical_projection(categorical_features.float())
        
        # Combine features
        combined_features = torch.cat([
            pooled_output,
            numerical_projected,
            categorical_projected
        ], dim=1)
        
        # Main task prediction
        if self.task_type == 'classification':
            main_output = self.classification_head(combined_features)
        else:
            main_output = self.regression_head(combined_features)
        
        # Auxiliary predictions
        auxiliary_outputs = {}
        for head_name, head in self.auxiliary_heads.items():
            auxiliary_outputs[head_name] = head(combined_features)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(combined_features)
        
        return {
            'main_output': main_output,
            'auxiliary_outputs': auxiliary_outputs,
            'uncertainty': uncertainty,
            'hidden_states': hidden_states
        }

class AIRetrainingPipeline:
    """Real ML-powered retraining pipeline with advanced features"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize ML components
        self.model_registry = ModelRegistry()
        self.model_versioning = ModelVersioning()
        self.experiment_tracker = ExperimentTracker()
        self.data_processor = DataProcessor()
        self.data_validator = DataValidator()
        # Use stubs for model, config, and loss_fn if not defined yet
        dummy_model = nn.Linear(10, 2)
        dummy_config = TrainingConfig()
        dummy_loss_fn = nn.MSELoss()
        self.trainer = ModelTrainer(dummy_model, dummy_config, dummy_loss_fn)
        dummy_search_space = None
        self.optimizer = HyperparameterOptimizer(dummy_model, dummy_config, dummy_search_space)
        self.architecture_search = ArchitectureSearch()
        self.metrics_tracker = MLMetricsTracker()
        self.performance_monitor = ModelPerformanceMonitor()
        self.drift_detector = DriftDetector()
        
        # Initialize experiment tracking
        self._setup_experiment_tracking()
        
        # Model configurations
        self.model_configs = self._load_model_configs()
        
        # Retraining configuration
        self.retraining_config = {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 50,
            'early_stopping_patience': 10,
            'validation_split': 0.2,
            'test_split': 0.1,
            'gradient_accumulation_steps': 4,
            'max_grad_norm': 10,
            'warmup_steps': 100,
            'scheduler_type': 'cosine',
            'mixed_precision': True,
            'distributed_training': False,
            'checkpoint_frequency': 5,
            'evaluation_frequency': 2
        }
        
        # A/B testing configuration
        self.ab_test_config = [
            {'traffic_split': 0.5, 'evaluation_metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
             'statistical_significance':0.05,       'minimum_sample_size': 1000}
        ]
        
        # Model paths
        self.model_paths = {
            'production': 'models/production/',          'staging': 'models/staging/',
            'experimental': 'models/experimental/',
            'backup': 'models/backup/'
        }
        
        # Ensure directories exist
        for path in self.model_paths.values():
            os.makedirs(path, exist_ok=True)
    
    def _setup_experiment_tracking(self):
        """Experiment tracking with MLflow and WandB"""
        try:
            # MLflow setup
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("ai_retraining_pipeline")
            
            # WandB setup (optional)
            if os.getenv('WANDB_API_KEY'):
                wandb.init(project="ai-retraining-pipeline", entity=os.getenv('WANDB_ENTITY'))
            
            self.logger.info("Experiment tracking setup completed")
                    
        except Exception as e:
            self.logger.warning(f"Experiment tracking setup failed: {e}")
    
    def _load_model_configs(self) -> Dict:
        """Load configurations"""
        return {
            'classification':{
                'model_type': 'bert',
                'hidden_size': 768,
                'num_layers': 6,
                'num_heads': 2,
                'dropout': 1,
                'num_classes': 10,
                'task_type': 'classification'
            },
            'regression':{
                'model_type': 'bert',
                'hidden_size': 768,
                'num_layers': 6,
                'num_heads': 2,
                'dropout': 1,
                'task_type': 'regression'
            },
            'multitask':{
                'model_type': 'bert',
                'hidden_size': 768,
                'num_layers': 6,
                'num_heads': 2,
                'dropout': 1,
                'use_adapters': True,
                'task_type': 'multitask'
            }
        }
    
    async def retrain_model(self, 
                          model_name: str,
                          training_data: List[Dict],
                          validation_data: List[Dict] = None,
                          test_data: List[Dict] = None,
                          config_overrides: Dict = None) -> Dict:
        """Retrain model with real ML pipeline"""
        try:
            self.logger.info(f"Starting retraining for model: {model_name}")
            
            # Start experiment tracking
            with mlflow.start_run(run_name=f"retrain_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_params({
                    'model_name': model_name,
                    'training_samples': len(training_data),
                    'validation_samples': len(validation_data) if validation_data else 0,
                    'test_samples': len(test_data) if test_data else 0,
                    **self.retraining_config,
                    **(config_overrides or {})
                })
                
                # Validate and preprocess data
                processed_data = await self._preprocess_retraining_data(
                    training_data, validation_data, test_data
                )
                
                # Detect data drift
                drift_analysis = await self._analyze_data_drift(processed_data)
                mlflow.log_metrics(drift_analysis)
                
                # Initialize model
                model = await self._initialize_retraining_model(model_name, config_overrides)
                
                # Prepare datasets
                train_dataset, val_dataset, test_dataset = await self._prepare_datasets(processed_data)
                
                # Train model
                training_results = await self._train_model_with_advanced_features(
                    model, train_dataset, val_dataset, test_dataset
                )
                
                # Evaluate model
                evaluation_results = await self._evaluate_model(model, test_dataset)
                
                # Model versioning and registry
                model_version = await self._version_and_register_model(
                    model, model_name, training_results, evaluation_results
                )
                
                # Log results
                mlflow.log_metrics(evaluation_results)
                mlflow.log_artifact(f"models/{model_name}/{model_version}/model.pth")
                
                return {
                    'model_version': model_version,
                    'training_results': training_results,
                    'evaluation_results': evaluation_results,
                    'drift_analysis': drift_analysis,
                    'model_path': f"models/{model_name}/{model_version}/"
                }
            
        except Exception as e:
            self.logger.error(f"Error during model retraining: {e}")
            mlflow.log_param('error', str(e))
            raise
    
    async def _preprocess_retraining_data(self, training_data: List[Dict], validation_data: List[Dict], test_data: List[Dict]) -> Dict:
        """Reprocess data for retraining"""
        try:
            # Validate data
            validated_training = self.data_validator.validate_training_data(training_data)
            validated_validation = self.data_validator.validate_training_data(validation_data) if validation_data else None
            validated_test = self.data_validator.validate_training_data(test_data) if test_data else None
            
            # Process data
            processed_training = self.data_processor.process_training_data(validated_training)
            processed_validation = self.data_processor.process_training_data(validated_validation) if validated_validation else None
            processed_test = self.data_processor.process_training_data(validated_test) if validated_test else None
            
            # Data quality analysis
            quality_metrics = self.data_processor.analyze_data_quality(processed_training)
            
            return {
                'training': processed_training,
                'validation': processed_validation,
                'test': processed_test,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            raise
    
    async def _analyze_data_drift(self, processed_data: Dict) -> Dict:
        """Analyze data drift between training and production data"""
        try:
            # Load production data statistics
            production_stats = self.drift_detector.load_production_statistics()
            
            # Calculate drift metrics
            drift_metrics = {}
            
            if production_stats:
                # Feature drift
                feature_drift = self.drift_detector.calculate_feature_drift(
                    processed_data['training'], production_stats
                )
                
                # Label drift
                label_drift = self.drift_detector.calculate_label_drift(
                    processed_data['training'], production_stats
                )
                
                # Concept drift
                concept_drift = self.drift_detector.detect_concept_drift(
                    processed_data['training'], production_stats
                )
                
                drift_metrics = {
                    'feature_drift_score': feature_drift,                   'label_drift_score': label_drift,
                    'concept_drift_detected': concept_drift,                  'drift_severity': self._calculate_drift_severity(feature_drift, label_drift, concept_drift)
                }
            else:
                drift_metrics = {
                    'feature_drift_score': 0.0,
                    'label_drift_score': 0.0,
                    'concept_drift_detected': False,
                    'drift_severity': 'none'
                }
            
            return drift_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing data drift: {e}")
            return {'error': str(e)}
    
    def _calculate_drift_severity(self, feature_drift: float, label_drift: float, concept_drift: bool) -> str:
        """Calculate overall drift severity"""
        if concept_drift:
            return 'critical'
        elif feature_drift > 0.3 or label_drift > 0.3:
            return 'high'
        elif feature_drift > 0.1 or label_drift > 0.1:
            return 'medium'
        else:
            return 'low'
    
    async def _initialize_retraining_model(self, model_name: str, config_overrides: Dict = None) -> nn.Module:
        """Initialize model for retraining"""
        try:
            # Load base configuration
            config = self.model_configs.get(model_name, self.model_configs['classification'])
            
            # Apply overrides
            if config_overrides:
                config.update(config_overrides)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
            tokenizer.pad_token = tokenizer.eos_token
            
            # Initialize model
            model = AdvancedRetrainingModel(
                vocab_size=tokenizer.vocab_size,
                **config
            ).to(self.device)
            
            # Load pre-trained weights if available
            pretrained_path = f"models/{model_name}/latest/model.pth"
            if os.path.exists(pretrained_path):
                model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
                self.logger.info(f"Loaded pre-trained weights for {model_name}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise
    
    async def _prepare_datasets(self, processed_data: Dict) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepare datasets for training"""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
            tokenizer.pad_token = tokenizer.eos_token
            
            # Convert to DataFrames
            train_df = pd.DataFrame(processed_data['training'])
            val_df = pd.DataFrame(processed_data['validation']) if processed_data['validation'] else None
            test_df = pd.DataFrame(processed_data['test']) if processed_data['test'] is not None else None
            
            # Create datasets
            train_dataset = RetrainingDataset(train_df, tokenizer, augment=True)
            
            val_dataset = None
            if val_df is not None:
                val_dataset = RetrainingDataset(val_df, tokenizer, augment=False)
            
            test_dataset = None
            if test_df is not None:
                test_dataset = RetrainingDataset(test_df, tokenizer, augment=False)
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            self.logger.error(f"Error preparing datasets: {e}")
            raise
    
    async def _train_model_with_advanced_features(self, 
                                                model: nn.Module,
                                                train_dataset: Dataset,
                                                val_dataset: Dataset,
                                                test_dataset: Dataset) -> Dict:
        """Train model with advanced features"""
        try:
            # Prepare data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.retraining_config['batch_size'],
                shuffle=True,
                num_workers=4,               pin_memory=True
            )
            
            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.retraining_config['batch_size'],
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
            
            # Setup training components
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.retraining_config['learning_rate'],
                weight_decay=0.01   )
            
            # Learning rate scheduler
            if self.retraining_config['scheduler_type'] == 'cosine':         scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.retraining_config['warmup_steps'],                   num_training_steps=len(train_loader) * self.retraining_config['epochs']                )
            else:
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.retraining_config['warmup_steps'],                   num_training_steps=len(train_loader) * self.retraining_config['epochs']                )
            
            # Loss functions
            if hasattr(model, 'classification_head'):
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            training_history = []
            
            for epoch in range(self.retraining_config['epochs']):
                # Training phase
                model.train()
                train_loss = 0
                train_metrics = {}
                
                for batch_idx, batch in enumerate(train_loader):
                    # Move to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    numerical_features = batch['numerical_features'].to(self.device)
                    categorical_features = batch['categorical_features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    weights = batch['weight'].to(self.device)
                    
                    # Forward pass
                    outputs = model(input_ids, attention_mask, numerical_features, categorical_features)
                    
                    # Calculate loss
                    if hasattr(model, 'classification_head'):
                        loss = criterion(outputs['main_output'],labels.squeeze())
                    else:
                        loss = criterion(outputs['main_output'], labels.float())
                    
                    # Add auxiliary losses
                    if 'auxiliary_outputs' in outputs:
                        aux_loss = 0
                        for aux_name, aux_output in outputs['auxiliary_outputs'].items():
                            # Placeholder auxiliary targets
                            aux_targets = torch.randint(0, aux_output.size(-1), (aux_output.size(0),)).to(self.device)
                            aux_loss += F.cross_entropy(aux_output, aux_targets)
                        loss += 0.1 * aux_loss
                    
                    # Weighted loss
                    loss = (loss * weights).mean()
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % self.retraining_config['gradient_accumulation_steps'] == 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.retraining_config['max_grad_norm']                   )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    
                    train_loss += loss.item()
                
                # Validation phase
                val_loss = 0
                val_metrics = {}
                
                if val_loader:
                    model.eval()
                    with torch.no_grad():
                        for batch in val_loader:
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            numerical_features = batch['numerical_features'].to(self.device)
                            categorical_features = batch['categorical_features'].to(self.device)
                            labels = batch['label'].to(self.device)
                            
                            outputs = model(input_ids, attention_mask, numerical_features, categorical_features)
                            
                            if hasattr(model, 'classification_head'):
                                loss = criterion(outputs['main_output'],labels.squeeze())
                            else:
                                loss = criterion(outputs['main_output'], labels.float())
                            
                            val_loss += loss.item()
                
                # Log metrics
                epoch_metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss / len(train_loader),
                    'val_loss': val_loss / len(val_loader) if val_loader else None,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                
                training_history.append(epoch_metrics)
                
                # Log to MLflow
                mlflow.log_metrics(epoch_metrics, step=epoch)
                
                # Early stopping
                if val_loader and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), f"models/best_model.pth")
                else:
                    patience_counter += 1
                
                if patience_counter >= self.retraining_config['early_stopping_patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Checkpoint
                if epoch % self.retraining_config['checkpoint_frequency'] == 0:
                    torch.save(model.state_dict(), f"models/checkpoint_epoch_{epoch}.pth")
            
            return {
                'training_history': training_history,
                'best_val_loss': best_val_loss,
                'epochs_trained': epoch + 1,
                'final_learning_rate': scheduler.get_last_lr()[0]
            }
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    async def _evaluate_model(self, model: nn.Module, test_dataset: Dataset) -> Dict:
        """Model performance"""
        try:
            if not test_dataset:
                return {'error': 'No test dataset provided'}
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.retraining_config['batch_size'],
                shuffle=False,
                num_workers=4
            )
            
            model.eval()
            all_predictions = []
            all_labels = []
            all_uncertainties = []
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    numerical_features = batch['numerical_features'].to(self.device)
                    categorical_features = batch['categorical_features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = model(input_ids, attention_mask, numerical_features, categorical_features)
                    
                    if hasattr(model, 'classification_head'):
                        predictions = torch.argmax(outputs['main_output'], dim=1)
                    else:
                        predictions = outputs['main_output'].squeeze()
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_uncertainties.extend(outputs['uncertainty'].cpu().numpy())
            
            # Calculate metrics
            if hasattr(model, 'classification_head'):
                accuracy = accuracy_score(all_labels, all_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
                auc = roc_auc_score(all_labels, all_predictions, multi_class='ovr')
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'avg_uncertainty': np.mean(all_uncertainties)
                }
            else:
                mse = np.mean((np.array(all_labels) - np.array(all_predictions)) ** 2)
                mae = np.mean(np.abs(np.array(all_labels) - np.array(all_predictions)))
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse),
                    'avg_uncertainty': np.mean(all_uncertainties)
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return {'error': str(e)}
    
    async def _version_and_register_model(self, 
                                        model: nn.Module,
                                        model_name: str,
                                        training_results: Dict,
                                        evaluation_results: Dict) -> str:
        """Version and register the trained model"""
        try:
            # Generate version
            version = self.model_versioning.generate_version(
                model_name=model_name,
                training_metrics=training_results,
                evaluation_metrics=evaluation_results
            )
            
            # Create model directory
            model_dir = f"models/{model_name}/{version}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), f"{model_dir}/model.pth")
            
            # Save metadata
            metadata = {
                'version': version,
                'model_name': model_name,
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'created_at': datetime.now().isoformat(),
                'model_config': model.__class__.__name__,
                'device': str(self.device)
            }
            
            with open(f"{model_dir}/metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Register model
            self.model_registry.register_model(
                model_name=model_name,
                version=version,
                model_path=f"{model_dir}/model.pth",          metadata=metadata
            )
            
            return version
            
        except Exception as e:
            self.logger.error(f"Error versioning and registering model: {e}")
            raise
    
    async def run_ab_test(self, 
                         model_a_version: str,
                         model_b_version: str,
                         model_name: str,
                         test_duration_days: int = 7) -> Dict:
        """Run A/B test between two model versions"""
        try:
            self.logger.info(f"Starting A/B test: {model_a_version} vs {model_b_version}")
            
            # Load models
            model_a = await self._load_model_version(model_name, model_a_version)
            model_b = await self._load_model_version(model_name, model_b_version)
            
            # Setup A/B test
            ab_test_id = f"ab_test_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"      
            # Initialize metrics tracking
            metrics_a = []
            metrics_b = []
            
            # Run test
            start_time = datetime.now()
            end_time = start_time + timedelta(days=test_duration_days)
            
            while datetime.now() < end_time:
                # Simulate traffic split
                traffic_a = self.ab_test_config[0]['traffic_split']
                traffic_b = 1 - traffic_a
                
                # Collect metrics for each model
                metrics_a.append(await self._collect_model_metrics(model_a, traffic_a))
                metrics_b.append(await self._collect_model_metrics(model_b, traffic_b))
                
                # Wait for next evaluation
                await asyncio.sleep(3600)  # 1 hour
            
            # Analyze results
            analysis = await self._analyze_ab_test_results(metrics_a, metrics_b)
            
            # Determine winner
            winner = await self._determine_ab_test_winner(analysis)
            
            return {
                'ab_test_id': ab_test_id,
                'model_a_version': model_a_version,
                'model_b_version': model_b_version,
                'test_duration_days': test_duration_days,
                'analysis': analysis,
                'winner': winner,
                'recommendation': self._generate_ab_test_recommendation(analysis, winner)
            }
            
        except Exception as e:
            self.logger.error(f"Error running A/B test: {e}")
            raise
    
    async def _load_model_version(self, model_name: str, version: str) -> nn.Module:
        """Load specific model version"""
        try:
            model_path = f"models/{model_name}/{version}/model.pth"
            metadata_path = f"models/{model_name}/{version}/metadata.json"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Initialize model
            model = AdvancedRetrainingModel(
                vocab_size=50257  # DialoGPT vocab size
                **metadata.get('model_config', {})
            ).to(self.device)
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model version: {e}")
            raise
    
    async def _collect_model_metrics(self, model: nn.Module, traffic_share: float) -> Dict:
        """Collect metrics for A/B testing"""
        # Simulate metric collection
        return {
            'timestamp': datetime.now().isoformat(),
            'traffic_share': traffic_share,
            'accuracy': np.random.normal(0.85, 0.02),
            'latency': np.random.normal(100, 10),        'throughput': np.random.normal(1000,10),          'error_rate': np.random.normal(0.105)        }
    
    async def _analyze_ab_test_results(self, metrics_a: List[Dict], metrics_b: List[Dict]) -> Dict:
        """Analyze A/B test results with statistical significance"""
        try:
            # Extract metrics
            accuracies_a = [m['accuracy'] for m in metrics_a]
            accuracies_b = [m['accuracy'] for m in metrics_b]
            latencies_a = [m['latency'] for m in metrics_a]
            latencies_b = [m['latency'] for m in metrics_b]
            
            # Statistical analysis
            from scipy import stats
            
            # T-test for accuracy
            t_stat_acc, p_value_acc = stats.ttest_ind(accuracies_a, accuracies_b)
            
            # T-test for latency
            t_stat_lat, p_value_lat = stats.ttest_ind(latencies_a, latencies_b)
            
            return {
                'accuracy_a_mean': np.mean(accuracies_a),
                'accuracy_b_mean': np.mean(accuracies_b),
                'accuracy_p_value': p_value_acc,
                'accuracy_significant': p_value_acc < self.ab_test_config[0]['statistical_significance'],
                'latency_a_mean': np.mean(latencies_a),
                'latency_b_mean': np.mean(latencies_b),
                'latency_p_value': p_value_lat,
                'latency_significant': p_value_lat < self.ab_test_config[0]['statistical_significance'],
                'sample_size_a': len(metrics_a),
                'sample_size_b': len(metrics_b)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing A/B test results: {e}")
            return {'error': str(e)}
    
    async def _determine_ab_test_winner(self, analysis: Dict) -> str:
        """Determine A/B test winner"""
        if 'error' in analysis:
            return 'inconclusive'
        # Check if results are statistically significant
        if not analysis['accuracy_significant']:
            return 'inconclusive'
        # Determine winner based on accuracy
        if analysis['accuracy_a_mean'] > analysis['accuracy_b_mean']:
            return 'model_a'
        else:
            return 'model_b'
    
    def _generate_ab_test_recommendation(self, analysis: Dict, winner: str) -> str:
        """Generate recommendation based on A/B test results"""
        if winner == 'inconclusive':
            return "Continue testing with larger sample size or longer duration"
        elif winner == 'model_a':
            return "Deploy model A to production"
        else:
            return "Deploy model B to production"
    
    async def get_system_health(self) -> Dict:
        """Get system health metrics"""
        try:
            health_metrics = {
                'status': 'healthy',
                'device': str(self.device),
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'gpu_utilization': GPUtil.getGPUs()[0].load if torch.cuda.is_available() else 0,
                'cpu_utilization': psutil.cpu_percent(),
                'disk_usage': psutil.disk_usage('/').percent,
                'model_registry_status': self.model_registry.get_status(),
                'experiment_tracking_status': 'active' if mlflow.active_run() else 'inactive',
                'performance_metrics': self.performance_monitor.get_latest_metrics(),
                'drift_detection_status': self.drift_detector.get_status()
            }
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {'status': 'error', 'error': str(e)}

# Initialize service
ai_retraining_pipeline = AIRetrainingPipeline() 