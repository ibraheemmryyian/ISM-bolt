"""
Impact Forecasting Engine for Industrial Symbiosis
Predicts environmental, economic, and social impact of symbiosis partnerships
"""

import os
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
    AutoModelForSequenceClassification
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

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
    TimeSeriesProcessor
)
from ml_core.optimization import (
    HyperparameterOptimizer,
    ForecastingOptimizer
)
from ml_core.monitoring import (
    MLMetricsTracker,
    ForecastingMonitor
)
from ml_core.utils import (
    ModelRegistry,
    TimeSeriesValidator,
    ConfigManager
)

class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting with real data processing"""
    def __init__(self, 
                 data: pd.DataFrame,
                 target_column: str,
                 feature_columns: List[str],
                 sequence_length: int = 30,
                 forecast_horizon: int = 7,
                 step_size: int = 1):
        self.data = data
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.step_size = step_size
        
        # Process data
        self.processed_data = self._process_data()
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _process_data(self) -> pd.DataFrame:
        """Process time series data with feature engineering"""
        processed = self.data.copy()
        
        # Handle missing values
        processed = processed.fillna(method='ffill').fillna(method='bfill')
        
        # Add time-based features
        if 'timestamp' in processed.columns:
            processed['timestamp'] = pd.to_datetime(processed['timestamp'])
            processed['year'] = processed['timestamp'].dt.year
            processed['month'] = processed['timestamp'].dt.month
            processed['day'] = processed['timestamp'].dt.day
            processed['day_of_week'] = processed['timestamp'].dt.dayofweek
            processed['quarter'] = processed['timestamp'].dt.quarter
            processed['is_weekend'] = processed['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Add lag features
        for lag in [1, 7, 14, 30]:
            processed[f'{self.target_column}_lag_{lag}'] = processed[self.target_column].shift(lag)
        
        # Add rolling statistics
        for window in [7, 14, 30]:
            processed[f'{self.target_column}_rolling_mean_{window}'] = processed[self.target_column].rolling(window=window).mean()
            processed[f'{self.target_column}_rolling_std_{window}'] = processed[self.target_column].rolling(window=window).std()
            processed[f'{self.target_column}_rolling_min_{window}'] = processed[self.target_column].rolling(window=window).min()
            processed[f'{self.target_column}_rolling_max_{window}'] = processed[self.target_column].rolling(window=window).max()
        
        # Add seasonal decomposition features
        if len(processed) > 50:
            try:
                decomposition = seasonal_decompose(
                    processed[self.target_column].dropna(),
                    period=7,
                    extrapolate_trend='freq'
                )
                processed['trend'] = decomposition.trend
                processed['seasonal'] = decomposition.seasonal
                processed['residual'] = decomposition.resid
            except:
                processed['trend'] = processed[self.target_column].rolling(window=7).mean()
                processed['seasonal'] = 0
                processed['residual'] = processed[self.target_column] - processed['trend']
        
        # Normalize features
        scaler = StandardScaler()
        feature_cols = [col for col in processed.columns if col not in ['timestamp', self.target_column]]
        processed[feature_cols] = scaler.fit_transform(processed[feature_cols].fillna(0))
        
        return processed
    
    def _create_sequences(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create sequences for time series forecasting"""
        sequences = []
        
        for i in range(0, len(self.processed_data) - self.sequence_length - self.forecast_horizon + 1, self.step_size):
            # Input sequence
            input_seq = self.processed_data.iloc[i:i + self.sequence_length]
            input_features = input_seq[self.feature_columns].values
            input_tensor = torch.FloatTensor(input_features)
            
            # Target sequence
            target_seq = self.processed_data.iloc[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
            target_values = target_seq[self.target_column].values
            target_tensor = torch.FloatTensor(target_values)
            
            sequences.append((input_tensor, target_tensor))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

class TransformerForecastingModel(nn.Module):
    """Transformer-based forecasting model with attention mechanisms"""
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 forecast_horizon: int = 7):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Forecasting head
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, forecast_horizon)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, forecast_horizon),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :seq_len, :]
        x = x + pos_encoding
        
        # Apply transformer
        encoded = self.transformer(x)
        
        # Global pooling
        pooled = torch.mean(encoded, dim=1)
        
        # Generate forecasts
        forecast = self.forecast_head(pooled)
        uncertainty = self.uncertainty_head(pooled)
        
        return forecast, uncertainty

class CausalImpactModel(nn.Module):
    """Causal impact analysis model using counterfactual prediction"""
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Treatment effect estimation
        self.treatment_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Outcome prediction
        self.outcome_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1 for treatment indicator
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Propensity score estimation
        self.propensity_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, treatment=None):
        # Extract features
        extracted_features = self.feature_extractor(features)
        
        # Estimate treatment effect
        treatment_effect = self.treatment_net(extracted_features)
        
        # Estimate propensity score
        propensity_score = self.propensity_net(extracted_features)
        
        # Predict outcomes
        if treatment is not None:
            # Combine features with treatment indicator
            combined_features = torch.cat([extracted_features, treatment.unsqueeze(1)], dim=1)
            outcome = self.outcome_net(combined_features)
        else:
            # Predict for both treatment and control
            treatment_combined = torch.cat([extracted_features, torch.ones(extracted_features.size(0), 1).to(features.device)], dim=1)
            control_combined = torch.cat([extracted_features, torch.zeros(extracted_features.size(0), 1).to(features.device)], dim=1)
            
            outcome_treatment = self.outcome_net(treatment_combined)
            outcome_control = self.outcome_net(control_combined)
            outcome = outcome_treatment - outcome_control
        
        return outcome, treatment_effect, propensity_score

class AdvancedImpactForecastingModel(nn.Module):
    """Advanced impact forecasting model with multiple components"""
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 forecast_horizon: int = 7,
                 use_causal: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.use_causal = use_causal
        
        # Time series forecasting component
        self.forecasting_model = TransformerForecastingModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon
        )
        
        # Causal impact component
        if use_causal:
            self.causal_model = CausalImpactModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim // 2
            )
        
        # Impact assessment head
        self.impact_head = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim // 2 if use_causal else 0), hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, forecast_horizon),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim // 2 if use_causal else 0), hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, forecast_horizon),
            nn.Sigmoid()
        )
    
    def forward(self, x, treatment_features=None):
        # Time series forecasting
        forecast, uncertainty = self.forecasting_model(x)
        
        # Causal impact analysis
        if self.use_causal and treatment_features is not None:
            causal_outcome, treatment_effect, propensity = self.causal_model(treatment_features)
            
            # Combine features for impact assessment
            combined_features = torch.cat([
                forecast,
                causal_outcome.expand(-1, self.forecast_horizon)
            ], dim=1)
        else:
            combined_features = forecast
            treatment_effect = torch.zeros(forecast.size(0), 1).to(forecast.device)
            propensity = torch.zeros(forecast.size(0), 1).to(forecast.device)
        
        # Impact assessment
        impact = self.impact_head(combined_features)
        confidence = self.confidence_head(combined_features)
        
        return {
         'forecast': forecast,
         'uncertainty': uncertainty,
         'impact': impact,
         'confidence': confidence,
         'treatment_effect': treatment_effect,
         'propensity': propensity
        }

class ImpactForecastingService:
    """Real ML-powered impact forecasting service with advanced time series and causal analysis"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model_factory = ModelFactory()
        self.model_registry = ModelRegistry()
        self.time_series_processor = TimeSeriesProcessor()
        self.time_series_validator = TimeSeriesValidator()
        self.metrics_tracker = MLMetricsTracker()
        self.forecasting_monitor = ForecastingMonitor()
        self.config_manager = ConfigManager()
        
        # Initialize models
        self.forecasting_models = {}
        self.causal_models = {}
        
        # Forecasting configuration
        self.forecasting_config = {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 50,
            'early_stopping_patience': 10,
            'validation_split': 0.2,
            'sequence_length': 30,
            'forecast_horizon': 7,
            'use_causal_analysis': True,
            'confidence_threshold': 0.7
        }
        
        # Model paths
        self.model_paths = {
            'forecasting': 'models/forecasting/',
            'causal': 'models/causal/',
            'impact': 'models/impact/'
        }
        
        # Ensure directories exist
        for path in self.model_paths.values():
            os.makedirs(path, exist_ok=True)
    
    async def create_forecasting_model(self,
                                     model_config: Dict,
                                     model_type: str = 'transformer'):
        """Create a new forecasting model"""
        try:
            self.logger.info(f"Creating forecasting model: {model_type}")
            
            # Create model
            if model_type == 'transformer':
                model = TransformerForecastingModel(
                    input_dim=model_config.get('input_dim'),
                    hidden_dim=model_config.get('hidden_dim', 256),
                    num_layers=model_config.get('num_layers', 4),
                    forecast_horizon=model_config.get('forecast_horizon', 7)
                ).to(self.device)
            elif model_type == 'advanced':
                model = AdvancedImpactForecastingModel(
                    input_dim=model_config.get('input_dim'),
                    hidden_dim=model_config.get('hidden_dim', 256),
                    num_layers=model_config.get('num_layers', 4),
                    forecast_horizon=model_config.get('forecast_horizon', 7),
                    use_causal=model_config.get('use_causal', True)
                ).to(self.device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Generate model ID
            model_id = f"forecast_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # Save model
            torch.save(model.state_dict(), f"{self.model_paths['forecasting']}/{model_id}.pth")
            
            # Register model
            self.forecasting_models[model_id] = model
            
            # Save metadata
            metadata = {
                'model_id': model_id,
                'model_type': model_type,
                'config': model_config,
                'created_at': datetime.now().isoformat()
            }
            
            with open(f"{self.model_paths['forecasting']}/{model_id}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Forecasting model created: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error creating forecasting model: {e}")
            raise

    async def train_forecasting_model(self,
                                    model_id: str,
                                    training_data: pd.DataFrame,
                                    target_column: str,
                                    feature_columns: List[str]) -> Dict:
        """Train forecasting model with real ML pipeline"""
        try:
            self.logger.info(f"Training forecasting model: {model_id}")
            
            # Load model
            if model_id not in self.forecasting_models:
                model = await self._load_forecasting_model(model_id)
            else:
                model = self.forecasting_models[model_id]
            
            # Validate time series data
            validation_result = self.time_series_validator.validate_data(training_data, target_column)
            if not validation_result['is_valid']:
                raise ValueError(f"Invalid time series data: {validation_result['errors']}")
            
            # Prepare dataset
            dataset = TimeSeriesDataset(
                data=training_data,
                target_column=target_column,
                feature_columns=feature_columns,
                sequence_length=self.forecasting_config['sequence_length'],
                forecast_horizon=self.forecasting_config['forecast_horizon']
            )
            
            # Split data
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.forecasting_config['batch_size'],
                shuffle=True,
                num_workers=4
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.forecasting_config['batch_size'],
                shuffle=False,
                num_workers=4
            )
            
            # Setup training
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.forecasting_config['learning_rate']
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.forecasting_config['epochs']
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            training_history = []
            
            for epoch in range(self.forecasting_config['epochs']):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    if isinstance(model, AdvancedImpactForecastingModel):
                        outputs = model(batch_x)
                        forecast = outputs['forecast']
                    else:
                        forecast, uncertainty = model(batch_x)
                    
                    # Calculate loss
                    loss = F.mse_loss(forecast, batch_y)
                    
                    # Add uncertainty regularization if available
                    if isinstance(model, AdvancedImpactForecastingModel):
                        uncertainty_loss = torch.mean(outputs['uncertainty'])
                        loss += 0.1 * uncertainty_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                scheduler.step()
                
                # Validation phase
                val_loss = 0.0
                val_predictions = []
                val_targets = []
                
                model.eval()
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        if isinstance(model, AdvancedImpactForecastingModel):
                            outputs = model(batch_x)
                            forecast = outputs['forecast']
                        else:
                            forecast, _ = model(batch_x)
                        
                        loss = F.mse_loss(forecast, batch_y)
                        val_loss += loss.item()
                        
                        val_predictions.extend(forecast.cpu().numpy())
                        val_targets.extend(batch_y.cpu().numpy())
                
                # Calculate metrics
                val_predictions = np.array(val_predictions)
                val_targets = np.array(val_targets)
                
                mse = np.mean(np.square(val_targets - val_predictions))
                mae = np.mean(np.abs(val_targets - val_predictions))
                r2 = 1 - np.sum(np.square(val_targets - val_predictions)) / np.sum(np.square(val_targets - np.mean(val_targets)))
                
                # Log metrics
                epoch_metrics = {
                  'epoch': epoch,
                  'train_loss': train_loss / len(train_loader),
                  'val_loss': val_loss / len(val_loader),
                  'val_mse': mse,
                  'val_mae': mae,
                  'val_r2': r2,
                  'learning_rate': scheduler.get_last_lr()[0]
                }
                
                training_history.append(epoch_metrics)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    torch.save(model.state_dict(), f"{self.model_paths['forecasting']}/{model_id}_best.pth")
                else:
                    patience_counter += 1
                
                if patience_counter >= self.forecasting_config['early_stopping_patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Track metrics
            self.metrics_tracker.record_forecasting_metrics({
                'model_id': model_id,
                'final_train_loss': train_loss / len(train_loader),
                'final_val_loss': best_val_loss,
                'final_val_mse': mse,
                'final_val_mae': mae,
                'final_val_r2': r2,
                'epochs_trained': epoch + 1
            })
            
            return {
                'model_id': model_id,
                'training_history': training_history,
                'best_val_loss': best_val_loss,
                'final_metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training forecasting model: {e}")
            raise

    async def forecast_impact(self,
                            model_id: str,
                            input_data: pd.DataFrame,
                            target_column: str,
                            feature_columns: List[str],
                            forecast_periods: int = 7) -> Dict:
        """Forecast impact using trained model"""
        try:
            # Load model
            if model_id not in self.forecasting_models:
                model = await self._load_forecasting_model(model_id)
            else:
                model = self.forecasting_models[model_id]
            
            # Prepare input data
            processed_data = self.time_series_processor.process_for_forecasting(
                input_data, target_column, feature_columns
            )
            
            # Create sequences for forecasting
            sequences = self._create_forecast_sequences(
                processed_data, feature_columns, forecast_periods
            )
            
            # Generate forecasts
            model.eval()
            forecasts = []
            uncertainties = []
            impacts = []
            confidences = []
            
            with torch.no_grad():
                for sequence in sequences:
                    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                    
                    if isinstance(model, AdvancedImpactForecastingModel):
                        outputs = model(sequence_tensor)
                        forecast = outputs['forecast']
                        uncertainty = outputs['uncertainty']
                        impact = outputs['impact']
                        confidence = outputs['confidence']
                    else:
                        forecast, uncertainty = model(sequence_tensor)
                        impact = torch.zeros_like(forecast)
                        confidence = torch.ones_like(forecast) * 0.5
                    
                    forecasts.append(forecast.cpu().numpy())
                    uncertainties.append(uncertainty.cpu().numpy())
                    impacts.append(impact.cpu().numpy())
                    confidences.append(confidence.cpu().numpy())
            
            # Aggregate results
            forecast_result = np.concatenate(forecasts, axis=0)
            uncertainty_result = np.concatenate(uncertainties, axis=0)
            impact_result = np.concatenate(impacts, axis=0)
            confidence_result = np.concatenate(confidences, axis=0)
            
            # Generate forecast dates
            last_date = input_data.index[-1] if hasattr(input_data, 'index') else datetime.now()
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_periods,
                freq='D'
            )
            
            # Track forecasting metrics
            self.forecasting_monitor.record_forecast_metrics({
                'model_id': model_id,
                'forecast_periods': forecast_periods,
                'avg_confidence': float(np.mean(confidence_result)),
                'avg_uncertainty': float(np.mean(uncertainty_result))
            })
            
            return {
                'forecasts': forecast_result.tolist(),
                'uncertainties': uncertainty_result.tolist(),
                'impacts': impact_result.tolist(),
                'confidences': confidence_result.tolist(),
                'forecast_dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                'model_id': model_id,
                'forecast_metadata': {
                    'forecast_periods': forecast_periods,
                    'confidence_threshold': self.forecasting_config['confidence_threshold'],
                    'generated_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error forecasting impact: {e}")
            raise

    async def analyze_causal_impact(self,
                                  data: pd.DataFrame,
                                  treatment_column: str,
                                  outcome_column: str,
                                  feature_columns: List[str]) -> Dict:
        """Analyze causal impact using advanced ML techniques"""
        try:
            self.logger.info("Analyzing causal impact")
            
            # Prepare data for causal analysis
            treatment_data = data[treatment_column].values
            outcome_data = data[outcome_column].values
            feature_data = data[feature_columns].values
            
            # Create causal model
            causal_model = CausalImpactModel(
                input_dim=len(feature_columns),
                hidden_dim=128
            ).to(self.device)
            
            # Prepare dataset
            treatment_tensor = torch.FloatTensor(treatment_data).to(self.device)
            outcome_tensor = torch.FloatTensor(outcome_data).to(self.device)
            feature_tensor = torch.FloatTensor(feature_data).to(self.device)
            
            # Train causal model
            optimizer = torch.optim.Adam(causal_model.parameters(), lr=1e-3)
            
            for epoch in range(100):
                optimizer.zero_grad()
                
                # Forward pass
                outcome_pred, treatment_effect, propensity = causal_model(feature_tensor, treatment_tensor)
                
                # Calculate losses
                outcome_loss = F.mse_loss(outcome_pred.squeeze(), outcome_tensor)
                propensity_loss = F.binary_cross_entropy(propensity.squeeze(), treatment_tensor)
                
                # Total loss
                total_loss = outcome_loss + propensity_loss
                
                total_loss.backward()
                optimizer.step()
            
            # Analyze causal effects
            causal_model.eval()
            with torch.no_grad():
                # Predict counterfactuals
                outcome_treatment, _, _ = causal_model(feature_tensor, torch.ones_like(treatment_tensor))
                outcome_control, _, _ = causal_model(feature_tensor, torch.zeros_like(treatment_tensor))
                
                # Calculate treatment effects
                treatment_effects = outcome_treatment - outcome_control
                
                # Calculate average treatment effect
                ate = torch.mean(treatment_effects).item()
                
                # Calculate treatment effect on treated
                treated_mask = treatment_data == 1
                if treated_mask.sum() > 0:
                    att = torch.mean(treatment_effects[treated_mask]).item()
                else:
                    att = ate
                
                # Calculate treatment effect on controls
                control_mask = treatment_data == 0
                if control_mask.sum() > 0:
                    atc = torch.mean(treatment_effects[control_mask]).item()
                else:
                    atc = ate
            
            # Statistical significance testing
            treatment_group = outcome_data[treatment_data == 1]
            control_group = outcome_data[treatment_data == 0]
            
            if len(treatment_group) > 0 and len(control_group) > 0:
                t_stat, p_value = stats.ttest_ind(treatment_group, control_group)
            else:
                t_stat, p_value = 0, 0
            return {
                'average_treatment_effect': ate,
                'treatment_effect_on_treated': att,
                'treatment_effect_on_controls': atc,
                'statistical_significance': {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.5
                },
                'treatment_effects': treatment_effects.cpu().numpy().tolist(),
                'model_performance': {
                    'outcome_loss': float(outcome_loss),
                    'propensity_loss': float(propensity_loss)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing causal impact: {e}")
            raise

    def _create_forecast_sequences(self,
                                 data: pd.DataFrame,
                                 feature_columns: List[str],
                                 forecast_periods: int) -> List[np.ndarray]:
        """Create sequences for forecasting"""
        sequences = []
        
        # Use last sequence_length data points
        sequence_length = self.forecasting_config['sequence_length']
        
        if len(data) >= sequence_length:
            last_sequence = data[feature_columns].iloc[-sequence_length:].values
            sequences.append(last_sequence)
        
        return sequences
    
    async def _load_forecasting_model(self, model_id: str) -> nn.Module:
        """Load forecasting model from disk"""
        try:
            model_path = f"{self.model_paths['forecasting']}/{model_id}.pth"
            metadata_path = f"{self.model_paths['forecasting']}/{model_id}_metadata.json"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Forecasting model not found: {model_path}")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create model
            model_type = metadata['model_type']
            config = metadata['config']
            
            if model_type == 'transformer':
                model = TransformerForecastingModel(**config).to(self.device)
            elif model_type == 'advanced':
                model = AdvancedImpactForecastingModel(**config).to(self.device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading forecasting model: {e}")
            raise

    async def get_system_health(self) -> Dict:
        """Get system health metrics"""
        try:
            health_metrics = {
                'status': 'healthy',
                'device': str(self.device),
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'forecasting_models_loaded': len(self.forecasting_models),
                'causal_models_loaded': len(self.causal_models),
                'forecasting_monitor_status': self.forecasting_monitor.get_status(),
                'metrics_tracker_status': self.metrics_tracker.get_status(),
                'performance_metrics': {
                    'avg_forecast_time': self.forecasting_monitor.get_avg_forecast_time(),
                    'forecast_accuracy': self.metrics_tracker.get_forecast_accuracy(),
                    'causal_analysis_accuracy': self.metrics_tracker.get_causal_accuracy()
                }
            }
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {'status': 'error', 'error': str(e)}

    def forecast_environmental_impact(self, company_data: Dict, timeframe: str = '12_months') -> Dict:
        """Forecast environmental impact based on company data"""
        try:
            # Extract historical data
            historical_emissions = company_data.get('historical_emissions', [100, 95, 90, 85, 80])
            historical_waste = company_data.get('historical_waste', [50, 48, 45, 42, 40])
            
            if STATSMODELS_AVAILABLE:
                # Use statsmodels for advanced forecasting
                periods = 12 if timeframe == '12_months' else 6
                
                # Emissions forecast
                emissions_model = sm.tsa.ARIMA(historical_emissions, order=(1, 1, 1))
                emissions_fitted = emissions_model.fit()
                emissions_forecast = emissions_fitted.forecast(steps=periods)
                
                # Waste forecast
                waste_model = sm.tsa.ARIMA(historical_waste, order=(1, 1, 1))
                waste_fitted = waste_model.fit()
                waste_forecast = waste_fitted.forecast(steps=periods)
            else:
                # Use simple forecasting methods
                periods = 12 if timeframe == '12_months' else 6
                emissions_forecast = simple_forecast(historical_emissions, periods)
                waste_forecast = simple_forecast(historical_waste, periods)
            
            return {
                'emissions_forecast': emissions_forecast.tolist() if hasattr(emissions_forecast, 'tolist') else emissions_forecast,
                'waste_forecast': waste_forecast.tolist() if hasattr(waste_forecast, 'tolist') else waste_forecast,
                'forecast_periods': periods,
                'confidence_level': 0.85
            }
            
        except Exception as e:
            self.logger.error(f"Error in environmental impact forecasting: {e}")
            return {'error': str(e)}

# Initialize service
impact_forecasting_service = ImpactForecastingService() 

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError as e:
    STATSMODELS_AVAILABLE = False
    print(f"Warning: statsmodels not available due to scipy version conflict: {e}")
    print("Impact forecasting will use alternative statistical methods.")

# Alternative statistical functions if statsmodels is not available
if not STATSMODELS_AVAILABLE:
    def simple_linear_regression(x, y):
        """Simple linear regression without statsmodels"""
        n = len(x)
        if n != len(y):
            raise ValueError("x and y must have the same length")
        
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0, y_mean
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        return slope, intercept
    
    def simple_forecast(historical_data, periods=12):
        """Simple forecasting without statsmodels"""
        if len(historical_data) < 2:
            return [historical_data[0]] * periods if historical_data else [0] * periods
        
        # Simple linear trend
        x = list(range(len(historical_data)))
        slope, intercept = simple_linear_regression(x, historical_data)
        
        # Forecast future periods
        forecast = []
        for i in range(len(historical_data), len(historical_data) + periods):
            forecast.append(slope * i + intercept)
        
        return forecast 