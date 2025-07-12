#!/usr/bin/env python3
"""
ML Model Factory for Perfect AI System
Provides real ML implementations for all AI functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, RGCNConv
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE, MDS
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.ensemble import VotingRegressor, VotingClassifier, StackingRegressor, StackingClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, pipeline
import networkx as nx
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh, svds

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner

import shap
import lime
import lime.lime_tabular
from lime.lime_text import LimeTextExplainer

import umap
import hdbscan
from sklearn.cluster import SpectralClustering

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
import joblib
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_type: str
    hyperparameters: Dict[str, Any]
    feature_engineering: Dict[str, Any]
    training_config: Dict[str, Any]
    evaluation_metrics: List[str]
    model_path: Optional[str] = None

class MLModelFactory:
    """Factory for creating and managing ML models"""
    
    def __init__(self, model_cache_dir: str = "./models/ml_factory"):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.models = {}
        self.model_configs = {}
        self.feature_encoders = {}
        self.scalers = {}
        
        # Initialize transformers
        self._initialize_transformers()
        
        logger.info(f"ML Model Factory initialized at {self.model_cache_dir}")

    def _initialize_transformers(self):
        """Initialize transformer models"""
        try:
            # Sentence transformers for semantic matching
            self.semantic_model = SentenceTransformer('all-mpnet-base-v2')
            self.industry_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.material_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Text classification pipeline
            self.text_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            
            # Named entity recognition
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english"
            )
            
            logger.info("Transformer models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize some transformer models: {e}")
            self.semantic_model = None
            self.industry_model = None
            self.material_model = None
            self.text_classifier = None
            self.ner_pipeline = None

    def create_gnn_model(self, model_type: str, input_dim: int, hidden_dim: int = 64, 
                        output_dim: int = 32, num_layers: int = 3) -> nn.Module:
        """Create GNN model with specified architecture"""
        
        if model_type == 'gcn':
            return GCNModel(input_dim, hidden_dim, output_dim, num_layers)
        elif model_type == 'sage':
            return GraphSAGEModel(input_dim, hidden_dim, output_dim, num_layers)
        elif model_type == 'gat':
            return GATModel(input_dim, hidden_dim, output_dim, num_layers)
        elif model_type == 'gin':
            return GINModel(input_dim, hidden_dim, output_dim, num_layers)
        elif model_type == 'rgcn':
            return RGCNModel(input_dim, hidden_dim, output_dim, num_layers)
        else:
            raise ValueError(f"Unknown GNN model type: {model_type}")

    def create_ensemble_model(self, model_types: List[str], task_type: str = 'regression') -> Any:
        """Create ensemble model with multiple base models"""
        
        base_models = []
        
        for model_type in model_types:
            if model_type == 'random_forest':
                if task_type == 'regression':
                    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            elif model_type == 'gradient_boosting':
                if task_type == 'regression':
                    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                else:
                    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
            elif model_type == 'xgboost':
                if task_type == 'regression':
                    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                else:
                    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
            elif model_type == 'lightgbm':
                if task_type == 'regression':
                    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                else:
                    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
            elif model_type == 'catboost':
                if task_type == 'regression':
                    model = CatBoostRegressor(iterations=100, learning_rate=0.1, random_state=42, verbose=False)
                else:
                    model = CatBoostClassifier(iterations=100, learning_rate=0.1, random_state=42, verbose=False)
            elif model_type == 'neural_network':
                if task_type == 'regression':
                    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
                else:
                    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            base_models.append((model_type, model))
        
        if task_type == 'regression':
            ensemble = VotingRegressor(estimators=base_models)
        else:
            ensemble = VotingClassifier(estimators=base_models, voting='soft')
        
        return ensemble

    def create_clustering_model(self, model_type: str, **kwargs) -> Any:
        """Create clustering model"""
        
        if model_type == 'kmeans':
            return KMeans(n_clusters=kwargs.get('n_clusters', 5), random_state=42)
        elif model_type == 'dbscan':
            return DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5))
        elif model_type == 'hdbscan':
            return hdbscan.HDBSCAN(min_cluster_size=kwargs.get('min_cluster_size', 5))
        elif model_type == 'spectral':
            return SpectralClustering(n_clusters=kwargs.get('n_clusters', 5), random_state=42)
        elif model_type == 'agglomerative':
            return AgglomerativeClustering(n_clusters=kwargs.get('n_clusters', 5))
        else:
            raise ValueError(f"Unknown clustering model type: {model_type}")

    def create_anomaly_detector(self, model_type: str, **kwargs) -> Any:
        """Create anomaly detection model"""
        
        if model_type == 'isolation_forest':
            return IsolationForest(contamination=kwargs.get('contamination', 0.1), random_state=42)
        elif model_type == 'one_class_svm':
            from sklearn.svm import OneClassSVM
            return OneClassSVM(kernel=kwargs.get('kernel', 'rbf'), nu=kwargs.get('nu', 0.1))
        elif model_type == 'local_outlier_factor':
            from sklearn.neighbors import LocalOutlierFactor
            return LocalOutlierFactor(contamination=kwargs.get('contamination', 0.1))
        else:
            raise ValueError(f"Unknown anomaly detection model type: {model_type}")

    def create_feature_engineering_pipeline(self, features: List[str]) -> Dict[str, Any]:
        """Create feature engineering pipeline"""
        
        pipeline = {
            'scalers': {},
            'encoders': {},
            'selectors': {},
            'reducers': {}
        }
        
        # Create scalers for numerical features
        numerical_features = [f for f in features if f in ['volume', 'price', 'distance', 'capacity']]
        for feature in numerical_features:
            pipeline['scalers'][feature] = StandardScaler()
        
        # Create encoders for categorical features
        categorical_features = [f for f in features if f in ['industry', 'location', 'material_type']]
        for feature in categorical_features:
            pipeline['encoders'][feature] = LabelEncoder()
        
        # Create feature selectors
        pipeline['selectors']['kbest'] = SelectKBest(score_func=f_regression, k=min(10, len(features)))
        
        # Create dimensionality reduction
        pipeline['reducers']['pca'] = PCA(n_components=min(5, len(features)))
        pipeline['reducers']['umap'] = umap.UMAP(n_components=min(3, len(features)), random_state=42)
        
        return pipeline

    def create_optimization_model(self, optimization_type: str, **kwargs) -> Any:
        """Create optimization model"""
        
        if optimization_type == 'linear_programming':
            import pulp
            return pulp.LpProblem("Optimization", pulp.LpMinimize)
        elif optimization_type == 'genetic_algorithm':
            from deap import base, creator, tools, algorithms
            # Initialize genetic algorithm components
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
            return {
                'creator': creator,
                'tools': tools,
                'algorithms': algorithms
            }
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")

    def create_time_series_model(self, model_type: str, **kwargs) -> Any:
        """Create time series forecasting model"""
        
        if model_type == 'prophet':
            from prophet import Prophet
            return Prophet(
                yearly_seasonality=kwargs.get('yearly_seasonality', True),
                weekly_seasonality=kwargs.get('weekly_seasonality', True),
                daily_seasonality=kwargs.get('daily_seasonality', False)
            )
        elif model_type == 'arima':
            from statsmodels.tsa.arima.model import ARIMA
            return lambda data: ARIMA(data, order=kwargs.get('order', (1, 1, 1)))
        elif model_type == 'lstm':
            return LSTMModel(
                input_size=kwargs.get('input_size', 1),
                hidden_size=kwargs.get('hidden_size', 50),
                num_layers=kwargs.get('num_layers', 2),
                output_size=kwargs.get('output_size', 1)
            )
        else:
            raise ValueError(f"Unknown time series model type: {model_type}")

    def create_explainer(self, model: Any, data: np.ndarray, feature_names: List[str]) -> Any:
        """Create model explainer for interpretability"""
        
        # Create SHAP explainer
        if hasattr(model, 'predict'):
            try:
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict, data)
                return explainer
            except:
                pass
        
        # Create LIME explainer
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                data,
                feature_names=feature_names,
                class_names=['class_0', 'class_1'],
                mode='regression'
            )
            return explainer
        except:
            pass
        
        return None

    def save_model(self, model: Any, model_name: str, config: ModelConfig) -> bool:
        """Save model with configuration"""
        try:
            model_dir = self.model_cache_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save configuration
            config_path = model_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            
            # Update registry
            self.models[model_name] = model
            self.model_configs[model_name] = config
            
            logger.info(f"Model {model_name} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            return False

    def load_model(self, model_name: str) -> Tuple[Any, ModelConfig]:
        """Load model with configuration"""
        try:
            model_dir = self.model_cache_dir / model_name
            
            # Load model
            model_path = model_dir / "model.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load configuration
            config_path = model_dir / "config.json"
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                config = ModelConfig(**config_dict)
            
            # Update registry
            self.models[model_name] = model
            self.model_configs[model_name] = config
            
            logger.info(f"Model {model_name} loaded successfully")
            return model, config
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None, None

    def get_model(self, model_name: str) -> Any:
        """Get model from registry"""
        return self.models.get(model_name)

    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.models.keys())

# GNN Model Classes
class GCNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(GCNModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_weight)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x, edge_index, edge_weight)
        return x

class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(GraphSAGEModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(SAGEConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(SAGEConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_weight)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x, edge_index, edge_weight)
        return x

class GATModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, heads: int = 4):
        super(GATModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GATConv(input_dim, hidden_dim, heads=heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        
        # Output layer
        self.layers.append(GATConv(hidden_dim * heads, output_dim, heads=1))
        
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_weight)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x, edge_index, edge_weight)
        return x

class GINModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(GINModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )))
        
        # Output layer
        self.layers.append(GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )))
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_weight=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_weight)
            x = self.dropout(x)
        
        x = self.layers[-1](x, edge_index, edge_weight)
        return x

class RGCNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, num_relations: int = 1):
        super(RGCNModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(RGCNConv(input_dim, hidden_dim, num_relations))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
        
        # Output layer
        self.layers.append(RGCNConv(hidden_dim, output_dim, num_relations))
        
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_type, edge_weight)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.layers[-1](x, edge_index, edge_type, edge_weight)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Initialize global ML model factory
ml_model_factory = MLModelFactory()