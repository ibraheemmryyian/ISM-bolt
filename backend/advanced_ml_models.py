import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE, MDS
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
import networkx as nx
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import pearsonr, spearmanr
import joblib
import pickle
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    target_columns: List[str]
    validation_split: float = 0.2
    random_state: int = 42
    n_jobs: int = -1

class SymbiosisDataset(Dataset):
    """Custom dataset for symbiosis data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, transform=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        targets = self.targets[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, targets

class AdvancedNeuralNetwork(nn.Module):
    """Advanced neural network for symbiosis prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout_rate: float = 0.3, activation: str = 'relu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class AttentionMechanism(nn.Module):
    """Attention mechanism for sequence modeling"""
    
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.attention_dim = attention_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Compute attention scores
        queries = self.query(x)  # (batch_size, seq_len, attention_dim)
        keys = self.key(x)       # (batch_size, seq_len, attention_dim)
        values = self.value(x)   # (batch_size, seq_len, attention_dim)
        
        # Compute attention weights
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(self.attention_dim)
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention
        attended_values = torch.bmm(attention_weights, values)
        
        return attended_values, attention_weights

class TransformerBlock(nn.Module):
    """Transformer block for advanced sequence modeling"""
    
    def __init__(self, input_dim: int, num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SymbiosisTransformer(nn.Module):
    """Transformer model for symbiosis prediction"""
    
    def __init__(self, input_dim: int, num_layers: int = 6, num_heads: int = 8, 
                 ff_dim: int = 2048, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, input_dim)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(input_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.output_projection(x)
        
        return x

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for network analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Activation and normalization
        self.activation = nn.ReLU()
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x, adj_matrix):
        # x: node features (batch_size, num_nodes, input_dim)
        # adj_matrix: adjacency matrix (batch_size, num_nodes, num_nodes)
        
        batch_size, num_nodes, _ = x.size()
        
        for i, conv_layer in enumerate(self.conv_layers):
            # Graph convolution
            x = torch.bmm(adj_matrix, x)  # Aggregate neighbor information
            x = conv_layer(x)
            x = self.batch_norms[i](x.transpose(1, 2)).transpose(1, 2)
            x = self.activation(x)
        
        # Global pooling
        x = torch.mean(x, dim=1)  # Average pooling across nodes
        x = self.output_layer(x)
        
        return x

class EnsembleModel:
    """Ensemble model combining multiple ML algorithms"""
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all models in the ensemble"""
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += weight * pred
        
        return weighted_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble probability prediction"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X).reshape(-1, 1)
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += weight * pred
        
        return weighted_pred

class AdvancedClusteringModel:
    """Advanced clustering model for company segmentation"""
    
    def __init__(self, method: str = 'dbscan', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """Fit clustering model"""
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'dbscan':
            self.model = DBSCAN(**self.kwargs)
        elif self.method == 'kmeans':
            self.model = KMeans(**self.kwargs)
        elif self.method == 'agglomerative':
            self.model = AgglomerativeClustering(**self.kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get cluster centers"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster centers")
        
        if hasattr(self.model, 'cluster_centers_'):
            return self.scaler.inverse_transform(self.model.cluster_centers_)
        else:
            return None

class AnomalyDetectionModel:
    """Anomaly detection model for identifying unusual patterns"""
    
    def __init__(self, method: str = 'isolation_forest', **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.model = None
        self.scaler = RobustScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """Fit anomaly detection model"""
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'isolation_forest':
            self.model = IsolationForest(**self.kwargs)
        else:
            raise ValueError(f"Unknown anomaly detection method: {self.method}")
        
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomalies, 1 for normal)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower values indicate more anomalous)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)

class FeatureEngineeringPipeline:
    """Advanced feature engineering pipeline"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_selector = None
        self.is_fitted = False
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                     use_pca: bool = True, use_feature_selection: bool = True) -> np.ndarray:
        """Fit and transform features"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA if requested
        if use_pca and X_scaled.shape[1] > 10:
            n_components = min(10, X_scaled.shape[1] // 2)
            self.pca = PCA(n_components=n_components)
            X_scaled = self.pca.fit_transform(X_scaled)
        
        # Apply feature selection if requested
        if use_feature_selection and y is not None and X_scaled.shape[1] > 5:
            n_features = min(5, X_scaled.shape[1] // 2)
            self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
            X_scaled = self.feature_selector.fit_transform(X_scaled, y)
        
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted pipeline"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transformation")
        
        X_scaled = self.scaler.transform(X)
        
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        return X_scaled

class ModelManager:
    """Manager for ML models with persistence and versioning"""
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        self.models = {}
        self.configs = {}
        self.metrics = {}
        
        # Create model directory if it doesn't exist
        import os
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, name: str, model: Any, config: ModelConfig, metrics: Dict[str, float]):
        """Save model with configuration and metrics"""
        model_path = f"{self.model_dir}/{name}.pkl"
        config_path = f"{self.model_dir}/{name}_config.json"
        metrics_path = f"{self.model_dir}/{name}_metrics.json"
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Save metrics
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Update internal state
        self.models[name] = model
        self.configs[name] = config
        self.metrics[name] = metrics
        
        logger.info(f"Model {name} saved successfully")
    
    def load_model(self, name: str) -> Tuple[Any, ModelConfig, Dict[str, float]]:
        """Load model with configuration and metrics"""
        model_path = f"{self.model_dir}/{name}.pkl"
        config_path = f"{self.model_dir}/{name}_config.json"
        metrics_path = f"{self.model_dir}/{name}_metrics.json"
        
        # Load model
        model = joblib.load(model_path)
        
        # Load configuration
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            config = ModelConfig(**config_dict)
        
        # Load metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Update internal state
        self.models[name] = model
        self.configs[name] = config
        self.metrics[name] = metrics
        
        logger.info(f"Model {name} loaded successfully")
        return model, config, metrics
    
    def list_models(self) -> List[str]:
        """List all available models"""
        import os
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
        return [f.replace('.pkl', '') for f in model_files]
    
    def get_best_model(self, metric: str = 'score') -> str:
        """Get the best model based on a specific metric"""
        if not self.metrics:
            raise ValueError("No models loaded")
        
        best_model = max(self.metrics.items(), key=lambda x: x[1].get(metric, 0))
        return best_model[0]

class SymbiosisPredictor:
    """Main predictor class for symbiosis matching"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.ensemble_model = None
        self.is_fitted = False
    
    def prepare_features(self, companies_data: List[Dict]) -> np.ndarray:
        """Prepare features from company data"""
        features = []
        
        for company in companies_data:
            company_features = self._extract_company_features(company)
            features.append(company_features)
        
        return np.array(features)
    
    def _extract_company_features(self, company: Dict) -> np.ndarray:
        """Extract features from a single company"""
        features = []
        
        # Basic company features
        features.extend([
            company.get('employee_count', 0),
            company.get('annual_revenue', 0),
            company.get('sustainability_score', 0),
            company.get('carbon_footprint', 0),
            company.get('water_usage', 0)
        ])
        
        # Industry encoding
        industry_encoder = LabelEncoder()
        industry_encoded = industry_encoder.fit_transform([company.get('industry', 'unknown')])[0]
        features.append(industry_encoded)
        
        # Location features (if available)
        location = company.get('location', {})
        features.extend([
            location.get('lat', 0),
            location.get('lng', 0)
        ])
        
        # Material features
        materials = company.get('materials_inventory', [])
        features.extend([
            len(materials),
            sum(m.get('market_value', 0) for m in materials),
            np.mean([m.get('sustainability_score', 0) for m in materials]) if materials else 0
        ])
        
        return np.array(features)
    
    def fit(self, companies_data: List[Dict], target_scores: np.ndarray):
        """Fit the symbiosis predictor"""
        # Prepare features
        X = self.prepare_features(companies_data)
        
        # Apply feature engineering
        X_processed = self.feature_pipeline.fit_transform(X, target_scores)
        
        # Create ensemble model
        models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            AdvancedNeuralNetwork(
                input_dim=X_processed.shape[1],
                hidden_dims=[128, 64, 32],
                output_dim=1
            )
        ]
        
        self.ensemble_model = EnsembleModel(models, weights=[0.4, 0.4, 0.2])
        
        # Convert to PyTorch tensors for neural network
        X_tensor = torch.FloatTensor(X_processed)
        y_tensor = torch.FloatTensor(target_scores)
        
        # Train neural network separately
        nn_model = models[2]
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = nn_model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
        
        # Train other models
        for i, model in enumerate(models[:2]):
            model.fit(X_processed, target_scores)
        
        self.is_fitted = True
        
        # Calculate metrics
        predictions = self.predict(companies_data)
        metrics = {
            'mse': np.mean((predictions - target_scores) ** 2),
            'mae': np.mean(np.abs(predictions - target_scores)),
            'r2': 1 - np.sum((target_scores - predictions) ** 2) / np.sum((target_scores - np.mean(target_scores)) ** 2)
        }
        
        # Save model
        config = ModelConfig(
            model_type='ensemble',
            hyperparameters={'num_models': len(models)},
            feature_columns=[f'feature_{i}' for i in range(X_processed.shape[1])],
            target_columns=['symbiosis_score']
        )
        
        self.model_manager.save_model('symbiosis_predictor', self.ensemble_model, config, metrics)
    
    def predict(self, companies_data: List[Dict]) -> np.ndarray:
        """Predict symbiosis scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        X = self.prepare_features(companies_data)
        
        # Apply feature engineering
        X_processed = self.feature_pipeline.transform(X)
        
        # Make predictions
        predictions = self.ensemble_model.predict(X_processed)
        
        return predictions
    
    def predict_proba(self, companies_data: List[Dict]) -> np.ndarray:
        """Predict symbiosis probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        X = self.prepare_features(companies_data)
        
        # Apply feature engineering
        X_processed = self.feature_pipeline.transform(X)
        
        # Make probability predictions
        probabilities = self.ensemble_model.predict_proba(X_processed)
        
        return probabilities

# Global instances
model_manager = ModelManager()
symbiosis_predictor = SymbiosisPredictor()