"""
Production-Grade AI Fusion Layer
Combines outputs from all AI engines for optimal decision making
"""

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
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    DataValidator,
    FeatureExtractor
)
from ml_core.optimization import (
    HyperparameterOptimizer,
    EnsembleOptimizer
)
from ml_core.monitoring import (
    MLMetricsTracker,
    FusionMonitor
)
from ml_core.utils import (
    ModelRegistry,
    EnsembleManager,
    ConfigManager
)

class MultiModalDataset(Dataset):
  dataset for multi-modal fusion with real data processing
    def __init__(self, 
                 text_data: List[str],
                 numerical_data: np.ndarray,
                 categorical_data: np.ndarray,
                 labels: np.ndarray,
                 tokenizer,
                 max_length: int = 512
        self.text_data = text_data
        self.numerical_data = numerical_data
        self.categorical_data = categorical_data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Process text data
        self.text_features = self._process_text_data()
        
        # Process numerical data
        self.scaler = StandardScaler()
        self.numerical_features = self.scaler.fit_transform(numerical_data)
        
        # Process categorical data
        self.label_encoders =  self.categorical_features = []
        
        for i in range(categorical_data.shape[1]):
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(categorical_data[:, i])
            self.label_encoders.append(encoder)
            self.categorical_features.append(encoded)
        
        self.categorical_features = np.column_stack(self.categorical_features)
    
    def _process_text_data(self) -> List[Dict]:
     Process text data with tokenization"""
        text_features = []
        
        for text in self.text_data:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length,
                max_length=self.max_length,
                return_tensors='pt'
            )
            text_features.append({
          input_ids: encoding[input_ids'].squeeze(),
               attention_mask': encodingattention_mask'].squeeze()
            })
        
        return text_features
    
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        return {
          text_features': self.text_features[idx],
        numerical_features': torch.FloatTensor(self.numerical_features[idx]),
          categorical_features': torch.LongTensor(self.categorical_features[idx]),
              label': torch.LongTensor([self.labels[idx]]) if len(self.labels.shape) == 1 torch.FloatTensor(self.labels[idx])
        }

class AttentionFusionModule(nn.Module):
    ntion-based fusion module for multi-modal data
    def __init__(self, 
                 text_dim: int,
                 numerical_dim: int,
                 categorical_dim: int,
                 fusion_dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 00.1        super().__init__()
        
        self.text_dim = text_dim
        self.numerical_dim = numerical_dim
        self.categorical_dim = categorical_dim
        self.fusion_dim = fusion_dim
        
        # Feature projections
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.numerical_projection = nn.Linear(numerical_dim, fusion_dim)
        self.categorical_projection = nn.Linear(categorical_dim, fusion_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim *3 fusion_dim * 2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm1n.LayerNorm(fusion_dim)
        self.layer_norm2n.LayerNorm(fusion_dim)
        
    def forward(self, text_features, numerical_features, categorical_features):
        # Project features to common space
        text_proj = self.text_projection(text_features)
        numerical_proj = self.numerical_projection(numerical_features)
        categorical_proj = self.categorical_projection(categorical_features)
        
        # Self-attention within each modality
        text_attended, _ = self.attention(text_proj, text_proj, text_proj)
        numerical_attended, _ = self.attention(numerical_proj, numerical_proj, numerical_proj)
        categorical_attended, _ = self.attention(categorical_proj, categorical_proj, categorical_proj)
        
        # Cross-modal attention
        text_cross, _ = self.cross_attention(text_attended, numerical_attended, categorical_attended)
        numerical_cross, _ = self.cross_attention(numerical_attended, text_attended, categorical_attended)
        categorical_cross, _ = self.cross_attention(categorical_attended, text_attended, numerical_attended)
        
        # Residual connections and layer normalization
        text_fused = self.layer_norm1(text_attended + text_cross)
        numerical_fused = self.layer_norm1(numerical_attended + numerical_cross)
        categorical_fused = self.layer_norm1(categorical_attended + categorical_cross)
        
        # Global pooling
        text_pooled = torch.mean(text_fused, dim=1)
        numerical_pooled = torch.mean(numerical_fused, dim=1)
        categorical_pooled = torch.mean(categorical_fused, dim=1)
        
        # Concatenate and fuse
        combined = torch.cat([text_pooled, numerical_pooled, categorical_pooled], dim=1)
        fused_features = self.fusion_layers(combined)
        
        return fused_features

class EnsembleFusionModel(nn.Module):
    emble fusion model with multiple base models and fusion strategies
    def __init__(self, 
                 base_models: List[nn.Module],
                 fusion_strategy: str = 'attention',
                 num_classes: int = 10,
                 fusion_dim: int = 512        super().__init__()
        
        self.base_models = nn.ModuleList(base_models)
        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes
        
        # Get output dimensions from base models
        self.base_output_dims = []
        for model in base_models:
            # Get the output dimension of the last layer
            if hasattr(model, 'classifier'):
                self.base_output_dims.append(model.classifier.out_features)
            else:
                self.base_output_dims.append(768t for transformers
        
        # Fusion strategies
        if fusion_strategy == 'attention':
            self.fusion_module = AttentionFusionModule(
                text_dim=self.base_output_dims[0],
                numerical_dim=self.base_output_dims[1] if len(self.base_output_dims) > 1 else 64       categorical_dim=self.base_output_dims[2] if len(self.base_output_dims) > 2 else 32            fusion_dim=fusion_dim
            )
        elif fusion_strategy == 'weighted':
            self.fusion_module = WeightedFusionModule(
                input_dims=self.base_output_dims,
                fusion_dim=fusion_dim
            )
        elif fusion_strategy == 'gated':
            self.fusion_module = GatedFusionModule(
                input_dims=self.base_output_dims,
                fusion_dim=fusion_dim
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2
            nn.ReLU(),
            nn.Dropout(0.1,
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4
            nn.ReLU(),
            nn.Linear(fusion_dim // 41,
            nn.Sigmoid()
        )
    
    def forward(self, *inputs):
        # Get outputs from base models
        base_outputs =      for i, model in enumerate(self.base_models):
            if i < len(inputs):
                output = model(inputs[i])
                if isinstance(output, tuple):
                    output = output[0]  # Take first element if tuple
                base_outputs.append(output)
        
        # Apply fusion
        if self.fusion_strategy == 'attention':
            fused_features = self.fusion_module(*base_outputs)
        else:
            fused_features = self.fusion_module(base_outputs)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Uncertainty
        uncertainty = self.uncertainty_head(fused_features)
        
        return logits, uncertainty

class WeightedFusionModule(nn.Module):
    hted fusion module with learnable weights
    def __init__(self, input_dims: List[int], fusion_dim: int):
        super().__init__()
        
        self.input_dims = input_dims
        self.fusion_dim = fusion_dim
        
        # Learnable weights for each input
        self.weights = nn.Parameter(torch.ones(len(input_dims)))
        
        # Projection layers
        self.projections = nn.ModuleList(         nn.Linear(dim, fusion_dim) for dim in input_dims
        ])
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1        )
    
    def forward(self, inputs: List[torch.Tensor]):
        # Project inputs to common space
        projected =]
        for i, (input_tensor, projection) in enumerate(zip(inputs, self.projections)):
            proj = projection(input_tensor)
            projected.append(proj)
        
        # Apply weights
        weights = F.softmax(self.weights, dim=0)
        weighted_sum = sum(w * p for w, p in zip(weights, projected))
        
        # Final fusion
        fused = self.fusion_layer(weighted_sum)
        
        return fused

class GatedFusionModule(nn.Module):
 ated fusion module with gating mechanisms
    def __init__(self, input_dims: List[int], fusion_dim: int):
        super().__init__()
        
        self.input_dims = input_dims
        self.fusion_dim = fusion_dim
        
        # Projection layers
        self.projections = nn.ModuleList(         nn.Linear(dim, fusion_dim) for dim in input_dims
        ])
        
        # Gating networks
        self.gates = nn.ModuleList(           nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.Sigmoid()
            ) for dim in input_dims
        ])
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1        )
    
    def forward(self, inputs: List[torch.Tensor]):
        # Project and gate inputs
        gated_outputs =]
        for i, (input_tensor, projection, gate) in enumerate(zip(inputs, self.projections, self.gates)):
            proj = projection(input_tensor)
            gate_value = gate(input_tensor)
            gated = proj * gate_value
            gated_outputs.append(gated)
        
        # Sum gated outputs
        fused = sum(gated_outputs)
        
        # Final fusion
        fused = self.fusion_layer(fused)
        
        return fused

class AdvancedEnsembleModel(nn.Module):
    ""Advanced ensemble model with diversity promotion
    def __init__(self, 
                 base_models: List[nn.Module],
                 ensemble_size: int = 5,
                 diversity_weight: float = 00.1        super().__init__()
        
        self.base_models = nn.ModuleList(base_models)
        self.ensemble_size = ensemble_size
        self.diversity_weight = diversity_weight
        
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(ensemble_size))
        
        # Diversity regularization
        self.diversity_regularizer = DiversityRegularizer(ensemble_size)
    
    def forward(self, *inputs):
        # Get predictions from all models
        predictions = []
        for model in self.base_models:
            pred = model(*inputs)
            if isinstance(pred, tuple):
                pred = pred[0]
            predictions.append(pred)
        
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=0)  # [ensemble_size, batch_size, num_classes]
        
        # Apply ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        weighted_preds = stacked_preds * weights.unsqueeze(1).unsqueeze(2)
        
        # Sum weighted predictions
        ensemble_pred = torch.sum(weighted_preds, dim=0)
        
        return ensemble_pred, stacked_preds
    
    def diversity_loss(self, predictions):
  alculate diversity loss to promote ensemble diversity       return self.diversity_regularizer(predictions)

class DiversityRegularizer(nn.Module):
   larizer to promote diversity in ensemble predictions
    def __init__(self, ensemble_size: int):
        super().__init__()
        self.ensemble_size = ensemble_size
    
    def forward(self, predictions: torch.Tensor):
  alculate diversity loss based on prediction correlations"""
        # predictions shape: [ensemble_size, batch_size, num_classes]
        
        # Convert to probabilities
        probs = F.softmax(predictions, dim=-1)
        
        # Calculate pairwise correlations
        diversity_loss = 0    for i in range(self.ensemble_size):
            for j in range(i + 1, self.ensemble_size):
                # Correlation between predictions of models i and j
                corr = torch.corrcoef(torch.stack([probs[i].flatten(), probsj].flatten()]))[0, 1         diversity_loss += corr ** 2   
        return diversity_loss / (self.ensemble_size * (self.ensemble_size - 12lass AIFusionLayer:
    Real ML-powered AI fusion layer with advanced ensemble and multi-modal fusion
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model_factory = ModelFactory()
        self.model_registry = ModelRegistry()
        self.ensemble_manager = EnsembleManager()
        self.metrics_tracker = MLMetricsTracker()
        self.fusion_monitor = FusionMonitor()
        self.config_manager = ConfigManager()
        
        # Load pre-trained models
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize fusion models
        self.fusion_models = {}
        self.ensemble_models = {}
        
        # Fusion configuration
        self.fusion_config = {
            batch_size:32     learning_rate':1e-4
        epochs:30          early_stopping_patience': 5,
           validation_split': 0.2        fusion_strategies: [ttention', weighted', gated
         ensemble_size': 5,
         diversity_weight': 0.1
        }
        
        # Model paths
        self.model_paths =[object Object]
         fusion:models/fusion/',
         ensemble':models/ensemble/,
         base: els/base/'
        }
        
        # Ensure directories exist
        for path in self.model_paths.values():
            os.makedirs(path, exist_ok=True)
    
    async def create_fusion_model(self,
                                model_configs: List[Dict],
                                fusion_strategy: str = 'attention',
                                task_type: str =classification) -> str:
     reate a new fusion model with specified base models"""
        try:
            self.logger.info(f"Creating fusion model with strategy: {fusion_strategy}")
            
            # Create base models
            base_models =         for config in model_configs:
                model = self.model_factory.create_model(
                    model_type=config['type'],
                    config=config
                ).to(self.device)
                base_models.append(model)
            
            # Create fusion model
            fusion_model = EnsembleFusionModel(
                base_models=base_models,
                fusion_strategy=fusion_strategy,
                num_classes=config.get('num_classes,10)
            ).to(self.device)
            
            # Generate model ID
            model_id = f"fusion_{fusion_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S)}"      
            # Save model
            torch.save(fusion_model.state_dict(), f{self.model_paths['fusion]}/{model_id}.pth")
            
            # Register model
            self.fusion_models[model_id] = fusion_model
            
            # Save metadata
            metadata =[object Object]             model_id': model_id,
                fusion_strategy: fusion_strategy,
                task_type': task_type,
                base_models: model_configs,
                created_at:datetime.now().isoformat()
            }
            
            with open(f{self.model_paths['fusion']}/{model_id}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Fusion model created: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error creating fusion model: {e}")
            raise
    
    async def train_fusion_model(self,
                               model_id: str,
                               training_data: Dict,
                               validation_data: Dict = None) -> Dict:
     Train fusion model with real ML pipeline"""
        try:
            self.logger.info(f"Training fusion model: {model_id}")
            
            # Load model
            if model_id not in self.fusion_models:
                fusion_model = await self._load_fusion_model(model_id)
            else:
                fusion_model = self.fusion_models[model_id]
            
            # Prepare datasets
            train_dataset = await self._prepare_fusion_dataset(training_data)
            val_dataset = await self._prepare_fusion_dataset(validation_data) if validation_data else None
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.fusion_config['batch_size'],
                shuffle=True,
                num_workers=4
            )
            
            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.fusion_config['batch_size'],
                    shuffle=False,
                    num_workers=4
                )
            
            # Setup training
            optimizer = torch.optim.AdamW(
                fusion_model.parameters(),
                lr=self.fusion_config['learning_rate']
            )
            
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.fusion_config['epochs']
            )
            
            # Training loop
            best_val_loss = float(inf          patience_counter = 0
            training_history = []
            
            for epoch in range(self.fusion_config['epochs']):
                # Training phase
                fusion_model.train()
                train_loss = 0.0
                
                for batch in train_loader:
                    # Move to device
                    text_features = batchtext_features                   numerical_features = batch['numerical_features'].to(self.device)
                    categorical_features = batch['categorical_features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Forward pass
                    logits, uncertainty = fusion_model(
                        text_features['input_ids'].to(self.device),
                        text_features[attention_mask'].to(self.device),
                        numerical_features,
                        categorical_features
                    )
                    
                    # Calculate loss
                    loss = criterion(logits, labels.squeeze())
                    
                    # Add uncertainty regularization
                    uncertainty_loss = torch.mean(uncertainty)
                    total_loss = loss +0.1* uncertainty_loss
                    
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                scheduler.step()
                
                # Validation phase
                val_loss = 00                if val_loader:
                    fusion_model.eval()
                    with torch.no_grad():
                        for batch in val_loader:
                            text_features = batchtext_features                   numerical_features = batch['numerical_features'].to(self.device)
                            categorical_features = batch['categorical_features'].to(self.device)
                            labels = batch['label'].to(self.device)
                            
                            logits, _ = fusion_model(
                                text_features['input_ids'].to(self.device),
                                text_features[attention_mask'].to(self.device),
                                numerical_features,
                                categorical_features
                            )
                            
                            loss = criterion(logits, labels.squeeze())
                            val_loss += loss.item()
                
                # Log metrics
                epoch_metrics = {
                  epoch': epoch,
                 train_loss': train_loss / len(train_loader),
                 val_loss': val_loss / len(val_loader) if val_loader else None,
                 learning_rate': scheduler.get_last_lr()[0                }
                
                training_history.append(epoch_metrics)
                
                # Early stopping
                if val_loader and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    torch.save(fusion_model.state_dict(), f{self.model_paths['fusion']}/{model_id}_best.pth)              else:
                    patience_counter += 1
                
                if patience_counter >= self.fusion_config['early_stopping_patience']:
                    self.logger.info(fEarly stopping at epoch {epoch}")
                    break
            
            # Track metrics
            self.metrics_tracker.record_fusion_training_metrics({
                model_id': model_id,
                final_train_loss': train_loss / len(train_loader),
                final_val_loss: best_val_loss,
                epochs_trained': epoch +1  })
            
            return[object Object]             model_id': model_id,
                training_history': training_history,
                best_val_loss': best_val_loss
            }
            
        except Exception as e:
            self.logger.error(f"Error training fusion model: {e}")
            raise
    
    async def create_ensemble_model(self,
                                  base_model_ids: List[str],
                                  ensemble_size: int = 5> str:
        ate ensemble model with diversity promotion"""
        try:
            self.logger.info(fCreating ensemble model with {ensemble_size} base models")
            
            # Load base models
            base_models =          for model_id in base_model_ids:
                model = await self._load_base_model(model_id)
                base_models.append(model)
            
            # Create ensemble model
            ensemble_model = AdvancedEnsembleModel(
                base_models=base_models,
                ensemble_size=ensemble_size,
                diversity_weight=self.fusion_config['diversity_weight']
            ).to(self.device)
            
            # Generate ensemble ID
            ensemble_id = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S)}"      
            # Save ensemble
            torch.save(ensemble_model.state_dict(), f{self.model_paths['ensemble]}/{ensemble_id}.pth")
            
            # Register ensemble
            self.ensemble_models[ensemble_id] = ensemble_model
            
            # Save metadata
            metadata =[object Object]          ensemble_id': ensemble_id,
                base_model_ids': base_model_ids,
                ensemble_size: ensemble_size,
                created_at:datetime.now().isoformat()
            }
            
            with open(f{self.model_paths['ensemble]}/{ensemble_id}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Ensemble model created: {ensemble_id}")
            return ensemble_id
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble model: {e}")
            raise
    
    async def fuse_predictions(self,
                             model_id: str,
                             input_data: Dict,
                             fusion_strategy: str =auto) -> Dict:
        "e predictions from multiple models using real ML fusion"""
        try:
            # Load model
            if model_id not in self.fusion_models:
                fusion_model = await self._load_fusion_model(model_id)
            else:
                fusion_model = self.fusion_models[model_id]
            
            # Prepare input data
            processed_data = await self._process_fusion_input(input_data)
            
            # Get predictions
            fusion_model.eval()
            with torch.no_grad():
                text_features = processed_data['text_features]         numerical_features = processed_data['numerical_features'].to(self.device)
                categorical_features = processed_data['categorical_features'].to(self.device)
                
                logits, uncertainty = fusion_model(
                    text_features['input_ids'].to(self.device),
                    text_features[attention_mask'].to(self.device),
                    numerical_features,
                    categorical_features
                )
                
                # Convert to probabilities
                probabilities = F.softmax(logits, dim=1       predictions = torch.argmax(probabilities, dim=1)
            
            # Track fusion metrics
            self.fusion_monitor.record_fusion_metrics({
                model_id': model_id,
                uncertainty: float(uncertainty.mean()),
                confidence': float(probabilities.max(dim=1)[0].mean())
            })
            
            return[object Object]       predictions:predictions.cpu().numpy().tolist(),
                probabilities: probabilities.cpu().numpy().tolist(),
                uncertainty:uncertainty.cpu().numpy().tolist(),
                fusion_strategy: fusion_strategy
            }
            
        except Exception as e:
            self.logger.error(f"Error fusing predictions: {e}")
            raise
    
    async def ensemble_predict(self,
                             ensemble_id: str,
                             input_data: Dict) -> Dict:
     semble predictions with diversity analysis"""
        try:
            # Load ensemble
            if ensemble_id not in self.ensemble_models:
                ensemble_model = await self._load_ensemble_model(ensemble_id)
            else:
                ensemble_model = self.ensemble_models[ensemble_id]
            
            # Prepare input data
            processed_data = await self._process_ensemble_input(input_data)
            
            # Get ensemble predictions
            ensemble_model.eval()
            with torch.no_grad():
                ensemble_pred, individual_preds = ensemble_model(*processed_data)
                
                # Convert to probabilities
                ensemble_probs = F.softmax(ensemble_pred, dim=1          ensemble_prediction = torch.argmax(ensemble_probs, dim=1)
                
                # Calculate diversity
                diversity_score = ensemble_model.diversity_loss(individual_preds)
            
            # Analyze ensemble behavior
            ensemble_analysis = self._analyze_ensemble_behavior(individual_preds, ensemble_probs)
            
            return[object Object]          ensemble_prediction': ensemble_prediction.cpu().numpy().tolist(),
                ensemble_probabilities': ensemble_probs.cpu().numpy().tolist(),
                individual_predictions: individual_preds.cpu().numpy().tolist(),
                diversity_score: float(diversity_score),
                ensemble_analysis': ensemble_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error getting ensemble predictions: {e}")
            raise
    
    def _analyze_ensemble_behavior(self, individual_preds: torch.Tensor, ensemble_probs: torch.Tensor) -> Dict:
 ensemble behavior and agreement"""
        # Calculate agreement among individual models
        individual_argmax = torch.argmax(individual_preds, dim=-1)  # [ensemble_size, batch_size]
        
        # Agreement ratio
        agreement_ratios = []
        for i in range(individual_argmax.size(1)):  # For each sample
            predictions = individual_argmax[:, i]
            most_common = torch.mode(predictions)[0]
            agreement = (predictions == most_common).float().mean()
            agreement_ratios.append(agreement.item())
        
        # Confidence analysis
        confidence_scores = ensemble_probs.max(dim=1)0.numpy()
        
        return [object Object]            avg_agreement_ratio': np.mean(agreement_ratios),
            confidence_distribution':[object Object]
              mean': float(np.mean(confidence_scores)),
             std': float(np.std(confidence_scores)),
             min': float(np.min(confidence_scores)),
             max': float(np.max(confidence_scores))
            },
            ensemble_stability: float(np.std(agreement_ratios))
        }
    
    async def _prepare_fusion_dataset(self, data: Dict) -> MultiModalDataset:
        repare dataset for fusion training"""
        try:
            # Extract data components
            text_data = data.get('text, [])         numerical_data = np.array(data.get(numerical, []))
            categorical_data = np.array(data.get('categorical', []))
            labels = np.array(data.get('labels',       
            # Create dataset
            dataset = MultiModalDataset(
                text_data=text_data,
                numerical_data=numerical_data,
                categorical_data=categorical_data,
                labels=labels,
                tokenizer=self.tokenizer
            )
            
            return dataset
                
        except Exception as e:
            self.logger.error(f"Error preparing fusion dataset: {e}")
            raise
    
    async def _process_fusion_input(self, input_data: Dict) -> Dict:
      Process input data for fusion inference"""
        try:
            # Process text data
            text = input_data.get('text, )
            text_encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length,
                max_length=512            return_tensors='pt'
            )
            
            # Process numerical data
            numerical_data = np.array(input_data.get(numerical', [0.0] * 10))
            numerical_features = torch.FloatTensor(numerical_data).unsqueeze(0)
            
            # Process categorical data
            categorical_data = np.array(input_data.get(categorical', [0] * 5))
            categorical_features = torch.LongTensor(categorical_data).unsqueeze(0)
            
            return[object Object]              text_features: text_encoding,
                numerical_features': numerical_features,
                categorical_features': categorical_features
            }
            
        except Exception as e:
            self.logger.error(f"Error processing fusion input: {e}")
            raise
    
    async def _process_ensemble_input(self, input_data: Dict) -> List[torch.Tensor]:
      Process input data for ensemble inference"""
        try:
            # Convert input to tensor format expected by ensemble
            processed_inputs = []
            
            # Text input
            if 'text' in input_data:
                text_encoding = self.tokenizer(
                    input_data['text'],
                    truncation=True,
                    padding='max_length,
                    max_length=512,
                    return_tensors='pt'
                )
                processed_inputs.append(text_encoding['input_ids'].to(self.device))
                processed_inputs.append(text_encoding[attention_mask'].to(self.device))
            
            # Numerical input
            if 'numerical' in input_data:
                numerical = torch.FloatTensor(input_data['numerical']).unsqueeze(0).to(self.device)
                processed_inputs.append(numerical)
            
            # Categorical input
            if 'categorical' in input_data:
                categorical = torch.LongTensor(input_data['categorical']).unsqueeze(0).to(self.device)
                processed_inputs.append(categorical)
            
            return processed_inputs
            
        except Exception as e:
            self.logger.error(f"Error processing ensemble input: {e}")
            raise
    
    async def _load_fusion_model(self, model_id: str) -> nn.Module:
    Load fusion model from disk"""
        try:
            model_path = f{self.model_paths['fusion]}/{model_id}.pth"
            metadata_path = f{self.model_paths['fusion']}/{model_id}_metadata.json"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Fusion model not found: {model_path}")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create base models
            base_models =         for config in metadata['base_models']:
                model = self.model_factory.create_model(
                    model_type=config['type'],
                    config=config
                ).to(self.device)
                base_models.append(model)
            
            # Create fusion model
            fusion_model = EnsembleFusionModel(
                base_models=base_models,
                fusion_strategy=metadata['fusion_strategy'],
                num_classes=metadata.get('num_classes,10)
            ).to(self.device)
            
            # Load weights
            fusion_model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            return fusion_model
            
        except Exception as e:
            self.logger.error(f"Error loading fusion model: {e}")
            raise
    
    async def _load_base_model(self, model_id: str) -> nn.Module:
        ad base model from disk"""
        try:
            model_path = f{self.model_paths[base]}/{model_id}.pth"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Base model not found: {model_path}")
            
            # Load model (simplified - would need proper model loading logic)
            model = nn.Linear(76810).to(self.device)  # Placeholder
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            return model
                
        except Exception as e:
            self.logger.error(f"Error loading base model: {e}")
            raise
    
    async def _load_ensemble_model(self, ensemble_id: str) -> nn.Module:
      oad ensemble model from disk"""
        try:
            model_path = f{self.model_paths['ensemble]}/{ensemble_id}.pth"
            metadata_path = f{self.model_paths['ensemble]}/{ensemble_id}_metadata.json"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Ensemble model not found: {model_path}")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load base models
            base_models =          for model_id in metadata['base_model_ids']:
                model = await self._load_base_model(model_id)
                base_models.append(model)
            
            # Create ensemble model
            ensemble_model = AdvancedEnsembleModel(
                base_models=base_models,
                ensemble_size=metadata['ensemble_size']
            ).to(self.device)
            
            # Load weights
            ensemble_model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            return ensemble_model
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble model: {e}")
            raise
    
    async def get_system_health(self) -> Dict:
   Get system health metrics"""
        try:
            health_metrics =[object Object]            status': 'healthy,
                device:str(self.device),
                memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0            fusion_models_loaded: len(self.fusion_models),
                ensemble_models_loaded': len(self.ensemble_models),
                fusion_monitor_status: self.fusion_monitor.get_status(),
                metrics_tracker_status': self.metrics_tracker.get_status(),
                performance_metrics': {
                    avg_fusion_time: self.fusion_monitor.get_avg_fusion_time(),
                    avg_ensemble_time: self.fusion_monitor.get_avg_ensemble_time(),
                    fusion_accuracy': self.metrics_tracker.get_fusion_accuracy()
                }
            }
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return[object Object]status:error', error': str(e)}

# Initialize service
ai_fusion_layer = AIFusionLayer() 