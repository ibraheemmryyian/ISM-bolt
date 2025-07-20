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
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
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
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import random

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
    RecommendationOptimizer
)
from ml_core.monitoring import (
    MLMetricsTracker,
    InferenceMonitor
)
from ml_core.utils import (
    ModelRegistry,
    RecommendationEngine,
    ConfigManager
)

class ListingDataset(Dataset):
    """Dataset for listing inference with real data processing"""
    def __init__(self, 
                 listings: List[Dict],
                 interactions: List[Dict],
                 tokenizer,
                 max_length: int = 512,
                 task_type: str = 'recommendation'):
        self.listings = listings
        self.interactions = interactions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        # Process listings
        self.processed_listings = self._process_listings()
        # Process interactions
        self.processed_interactions = self._process_interactions()
        # Create training pairs
        self.training_pairs = self._create_training_pairs()

    def _process_listings(self) -> Dict:
        """Process listings with feature engineering"""
        processed = {}
        # Text features
        text_features = []
        for listing in self.listings:
            text = f"{listing.get('title', '')} {listing.get('description', '')} {listing.get('category', '')}"
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            text_features.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })
        processed['text_features'] = text_features
        # Numerical features
        numerical_features = []
        for listing in self.listings:
            features = [
                listing.get('price', 0),
                listing.get('rating', 0),
                listing.get('review_count', 0),
                listing.get('view_count', 0),
                listing.get('favorite_count', 0),
                listing.get('days_since_created', 0),
                listing.get('seller_rating', 0),
                listing.get('seller_review_count', 0)
            ]
            numerical_features.append(features)
        # Normalize numerical features
        scaler = StandardScaler()
        processed['numerical_features'] = scaler.fit_transform(numerical_features)
        # Categorical features
        categorical_features = []
        for listing in self.listings:
            features = [
                listing.get('category', ''),
                listing.get('subcategory', ''),
                listing.get('condition', ''),
                listing.get('location', ''),
                listing.get('seller_type', '')
            ]
            categorical_features.append(features)
        # Encode categorical features
        label_encoders = {}
        encoded_categorical = []
        for i in range(len(categorical_features[0])):
            encoder = LabelEncoder()
            column = [cat[i] for cat in categorical_features]
            encoded = encoder.fit_transform(column)
            label_encoders[i] = encoder
            encoded_categorical.append(encoded)
        processed['categorical_features'] = np.column_stack(encoded_categorical)
        return processed
    
    def _process_interactions(self):
        """Process user interactions"""
        processed = {}

        # Extract user-item interactions
        user_item_pairs = []
        ratings = []

        for interaction in self.interactions:
            user_item_pairs.append((
                interaction['user_id'],
                interaction['listing_id']
            ))
            ratings.append(interaction.get('rating', 1))

        processed['user_item_pairs'] = user_item_pairs
        processed['ratings'] = ratings

        # Create user and item mappings
        unique_users = list(set(pair[0] for pair in user_item_pairs))
        unique_items = list(set(pair[1] for pair in user_item_pairs))

        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}

        processed['user_to_idx'] = user_to_idx
        processed['item_to_idx'] = item_to_idx
        processed['num_users'] = len(unique_users)
        processed['num_items'] = len(unique_items)

        return processed

    def _create_training_pairs(self) -> list:
        """Create training pairs for recommendation"""
        pairs = []

        for i, (user_id, item_id) in enumerate(self.processed_interactions['user_item_pairs']):
            rating = self.processed_interactions['ratings'][i]

            # Positive sample
            pairs.append((user_id, item_id, rating, 1))
            # Negative sample (random item not interacted with)
            all_items = list(self.processed_interactions['item_to_idx'].keys())
            negative_item = np.random.choice([item for item in all_items if item != item_id])
            pairs.append((user_id, negative_item, 0, 0))

        return pairs

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        user_id, item_id, rating, is_positive = self.training_pairs[idx]

        # Get item features
        item_idx = self.processed_interactions['item_to_idx'][item_id]

        return {
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'is_positive': is_positive,
            'text_features': self.processed_listings['text_features'][item_idx],
            'numerical_features': torch.FloatTensor(self.processed_listings['numerical_features'][item_idx]),
            'categorical_features': torch.LongTensor(self.processed_listings['categorical_features'][item_idx])
        }

class ContentBasedFilteringModel(nn.Module):
    """Content-based filtering model using deep learning"""
    def __init__(self, 
                 vocab_size: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        # Feature fusion
        self.numerical_projection = nn.Linear(8, hidden_dim // 2)
        self.categorical_projection = nn.Linear(5, hidden_dim // 2)
        # Content representation
        self.content_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Similarity projection
        self.similarity_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, text_features, numerical_features, categorical_features):
        # Text encoding
        text_embeddings = self.text_embedding(text_features['input_ids'])
        text_encoded = self.text_transformer(text_embeddings)
        text_pooled = torch.mean(text_encoded, dim=1)
        # Feature fusion
        numerical_projected = self.numerical_projection(numerical_features)
        categorical_projected = self.categorical_projection(categorical_features.float())
        # Combine features
        combined = torch.cat([text_pooled, numerical_projected, categorical_projected], dim=1)
        content_representation = self.content_encoder(combined)
        # Similarity score
        similarity = self.similarity_head(content_representation)
        return content_representation, similarity

class CollaborativeFilteringModel(nn.Module):
    """collaborative filtering model with neural networks"""
    def __init__(self, 
                 num_users: int,
                 num_items: int,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # User and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # Interaction network
        self.interaction_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
    
    def forward(self, user_indices, item_indices):
        # Get embeddings
        user_embeds = self.user_embeddings(user_indices)
        item_embeds = self.item_embeddings(item_indices)
        
        # Combine embeddings
        combined = torch.cat([user_embeds, item_embeds], dim=1)
        
        # Predict interaction
        interaction_score = self.interaction_net(combined)
        
        return interaction_score

class HybridRecommendationModel(nn.Module):
    """hybrid recommendation model combining content-based and collaborative filtering"""
    def __init__(self, 
                 content_model: ContentBasedFilteringModel,
                 collaborative_model: CollaborativeFilteringModel,
                 fusion_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(content_model.hidden_dim //2 + 1, fusion_dim),  # +1 for collaborative score
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, user_indices, item_indices, text_features, numerical_features, categorical_features):
        # Content-based features
        content_representation, content_similarity = self.content_model(
            text_features, numerical_features, categorical_features
        )
        
        # Collaborative filtering score
        collaborative_score = self.collaborative_model(user_indices, item_indices)
        
        # Fusion
        combined_features = torch.cat([content_representation, collaborative_score], dim=1)
        fused_representation = self.fusion_layer(combined_features)
        
        # Apply attention
        attended_features, _ = self.attention(
            fused_representation.unsqueeze(1),
            fused_representation.unsqueeze(1),
            fused_representation.unsqueeze(1)
        )
        
        final_score = attended_features.squeeze(1)
        
        return final_score, content_similarity, collaborative_score

class AdvancedListingInferenceModel(nn.Module):
    """advanced listing inference model with multiple components"""
    def __init__(self, 
                 vocab_size: int,
                 num_users: int,
                 num_items: int,
                 hidden_dim: int = 256,
                 embedding_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Content-based component
        self.content_model = ContentBasedFilteringModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Collaborative filtering component
        self.collaborative_model = CollaborativeFilteringModel(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Hybrid model
        self.hybrid_model = HybridRecommendationModel(
            content_model=self.content_model,
            collaborative_model=self.collaborative_model,
            fusion_dim=hidden_dim,
            dropout=dropout
        )
        
        # Multi-task heads
        self.rating_head = nn.Linear(hidden_dim, 5)  #1ar ratings
        self.category_head = nn.Linear(hidden_dim, 10)  # Category prediction
        self.price_head = nn.Linear(hidden_dim, 1)  # Price prediction
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_indices, item_indices, text_features, numerical_features, categorical_features):
        # Get hybrid recommendation
        recommendation_score, content_similarity, collaborative_score = self.hybrid_model(
            user_indices, item_indices, text_features, numerical_features, categorical_features
        )
        
        # Multi-task predictions
        rating_logits = self.rating_head(recommendation_score)
        category_logits = self.category_head(recommendation_score)
        price_prediction = self.price_head(recommendation_score)
        
        # Uncertainty
        uncertainty = self.uncertainty_head(recommendation_score)

        return {
         'recommendation_score': recommendation_score,
         'content_similarity': content_similarity,
         'collaborative_score': collaborative_score,
         'rating_logits': rating_logits,
         'category_logits': category_logits,
         'price_prediction': price_prediction,
         'uncertainty': uncertainty
        }

class ListingInferenceService:
    """Real ML-powered listing inference service with advanced recommendation systems"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model_factory = ModelFactory()
        self.model_registry = ModelRegistry()
        self.recommendation_engine = RecommendationEngine()
        self.metrics_tracker = MLMetricsTracker()
        self.inference_monitor = InferenceMonitor()
        self.config_manager = ConfigManager()
        
        # Load pre-trained models
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize models
        self.inference_models = {}
        self.recommendation_models = {}
        
        # Inference configuration
        self.inference_config = {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 30,
            'early_stopping_patience': 5,
            'validation_split': 0.2,
            'recommendation_strategies': ['content_based', 'collaborative', 'hybrid'],
            'top_k_recommendations': 10,
            'similarity_threshold': 0.7
        }
        
        # Model paths
        self.model_paths = {
            'inference': 'models/inference/',
            'recommendation': 'models/recommendation/',
            'embeddings': 'models/embeddings/'
        }
        
        # Ensure directories exist
        for path in self.model_paths.values():
            os.makedirs(path, exist_ok=True)
        
        # Initialize embeddings cache
        self.embeddings_cache = {}
    async def create_inference_model(self,
                                   model_config: Dict,
                                   model_type: str = 'advanced')-> str:
        """Create a new inference model"""
        try:
            self.logger.info(f"Creating inference model: {model_type}")
            
            # Create model
            if model_type == 'advanced':             model = AdvancedListingInferenceModel(
                    vocab_size=self.tokenizer.vocab_size,
                    num_users=model_config.get('num_users', 1000),
                    num_items=model_config.get('num_items', 1000),
                    hidden_dim=model_config.get('hidden_dim', 256),
                    embedding_dim=model_config.get('embedding_dim', 128),
                    num_layers=model_config.get('num_layers', 3)
            ).to(self.device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Generate model ID
            model_id = f"inference_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"      
            # Save model
            torch.save(model.state_dict(), f"{self.model_paths['inference']}/{model_id}.pth")
            
            # Register model
            self.inference_models[model_id] = model
            
            # Save metadata
            metadata = {
                'model_id': model_id,
                'model_type': model_type,
                'config': model_config,
                'created_at': datetime.now().isoformat()
            }
            
            with open(f"{self.model_paths['inference']}/{model_id}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Inference model created: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error creating inference model: {e}")
            raise
    
    async def train_inference_model(self,
                                  model_id: str,
                                  listings_data: List[Dict],
                                  interactions_data: List[Dict]) -> Dict:
        """Train inference model with real ML pipeline"""
        try:
            self.logger.info(f"Training inference model: {model_id}")
            
            # Load model
            if model_id not in self.inference_models:
                model = await self._load_inference_model(model_id)
            else:
                model = self.inference_models[model_id]
            
            # Prepare dataset
            dataset = ListingDataset(
                listings=listings_data,
                interactions=interactions_data,
                tokenizer=self.tokenizer
            )
            
            # Split data
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.inference_config['batch_size'],
                shuffle=True,
                num_workers=4
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.inference_config['batch_size'],
                shuffle=False,
                num_workers=4
            )
            
            # Setup training
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.inference_config['learning_rate']
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.inference_config['epochs']
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            training_history = []
            
            for epoch in range(self.inference_config['epochs']):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch in train_loader:
                    # Move to device
                    user_indices = torch.LongTensor([
                        dataset.processed_interactions['user_to_idx'][user_id] 
                        for user_id in batch['user_id']
                    ]).to(self.device)
                    
                    item_indices = torch.LongTensor([
                        dataset.processed_interactions['item_to_idx'][item_id] 
                        for item_id in batch['item_id']
                    ]).to(self.device)
                    
                    text_features = batch['text_features']
                    numerical_features = batch['numerical_features'].to(self.device)
                    categorical_features = batch['categorical_features'].to(self.device)
                    targets = batch['is_positive'].float().to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(user_indices, item_indices, text_features, numerical_features, categorical_features)
                    
                    # Calculate loss
                    recommendation_loss = F.binary_cross_entropy(outputs['recommendation_score'].squeeze(), targets)
                    
                    # Multi-task losses
                    rating_loss = F.cross_entropy(outputs['rating_logits'], batch['rating'].long().to(self.device))
                    category_loss = F.cross_entropy(outputs['category_logits'], categorical_features[:, 0])  # Use first categorical feature as category
                    
                    # Total loss
                    total_loss = recommendation_loss + 0.1 * rating_loss + 0.01 * category_loss
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += total_loss.item()
                
                scheduler.step()
                
                # Validation phase
                val_loss = 0.0
                val_predictions = []
                val_targets = []
                
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        user_indices = torch.LongTensor([
                            dataset.processed_interactions['user_to_idx'][user_id] 
                            for user_id in batch['user_id']
                        ]).to(self.device)
                        
                        item_indices = torch.LongTensor([
                            dataset.processed_interactions['item_to_idx'][item_id] 
                            for item_id in batch['item_id']
                        ]).to(self.device)
                        
                        text_features = batch['text_features']
                        numerical_features = batch['numerical_features'].to(self.device)
                        categorical_features = batch['categorical_features'].to(self.device)
                        targets = batch['is_positive'].float().to(self.device)
                        
                        outputs = model(user_indices, item_indices, text_features, numerical_features, categorical_features)
                        
                        loss = F.binary_cross_entropy(outputs['recommendation_score'].squeeze(), targets)
                        val_loss += loss.item()
                        
                        val_predictions.extend(outputs['recommendation_score'].cpu().numpy())
                        val_targets.extend(targets.cpu().numpy())
                
                # Calculate metrics
                val_predictions = np.array(val_predictions)
                val_targets = np.array(val_targets)
                
                precision = precision_score(val_targets, val_predictions > 0.5, average='weighted')
                recall = recall_score(val_targets, val_predictions > 0.5, average='weighted')
                f1 = f1_score(val_targets, val_predictions > 0.5, average='weighted')
                
                # Log metrics
                epoch_metrics = {
                  'epoch': epoch,
                 'train_loss': train_loss / len(train_loader),
                 'val_loss': val_loss / len(val_loader),
                 'val_precision': precision,
                 'val_recall': recall,
                 'val_f1': f1,
                 'learning_rate': scheduler.get_last_lr()[0]
                }
                
                training_history.append(epoch_metrics)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), f"{self.model_paths['inference']}/{model_id}_best.pth")
                else:
                    patience_counter += 1
                
                if patience_counter >= self.inference_config['early_stopping_patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Track metrics
            self.metrics_tracker.record_inference_metrics({
                'model_id': model_id,
                'final_train_loss': train_loss / len(train_loader),
                'final_val_loss': best_val_loss,
                'final_precision': precision,
                'final_recall': recall,
                'final_f1': f1,
                'epochs_trained': epoch + 1  })
            
            return {
                'model_id': model_id,
                'training_history': training_history,
                'best_val_loss': best_val_loss,
                'final_metrics': {
                 'precision': precision,
                 'recall': recall,
                 'f1': f1
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error training inference model: {e}")
            raise
    
    async def generate_recommendations(self,
                                     model_id: str,
                                     user_id: str,
                                     listings_data: List[Dict],
                                     strategy: str = 'hybrid',
                                     top_k: int = 10)-> List[Dict]:
        """Generate recommendations using trained model"""
        try:
            # Load model
            if model_id not in self.inference_models:
                model = await self._load_inference_model(model_id)
            else:
                model = self.inference_models[model_id]
            
            # Prepare listings data
            processed_listings = await self._process_listings_for_inference(listings_data)
            
            # Generate recommendations based on strategy
            if strategy == 'content_based':   recommendations = await self._content_based_recommendations(
                    model, user_id, processed_listings, top_k
                )
            elif strategy == 'collaborative':   recommendations = await self._collaborative_recommendations(
                    model, user_id, processed_listings, top_k
                )
            elif strategy == 'hybrid':   recommendations = await self._hybrid_recommendations(
                    model, user_id, processed_listings, top_k
                )
            else:
                raise ValueError(f"Unknown recommendation strategy: {strategy}")
            
            # Track recommendation metrics
            self.inference_monitor.record_recommendation_metrics({
                'model_id': model_id,
                'user_id': user_id,
                'strategy': strategy,
                'num_recommendations': len(recommendations),
                'avg_confidence': np.mean([rec['confidence'] for rec in recommendations])
            })
            
            return {
                'recommendations': recommendations,
                'strategy': strategy,
                'model_id': model_id,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            raise
    
    async def _content_based_recommendations(self,
                                           model: nn.Module,
                                           user_id: str,
                                           listings_data: List[Dict],
                                           top_k: int) -> List[Dict]:
        """Content-based recommendations"""
        recommendations = []
        
        # Get user profile (simplified - would need actual user data)
        user_profile = self._get_user_profile(user_id)
        
        model.eval()
        with torch.no_grad():
            for listing in listings_data:
                # Prepare listing features
                text_features = listing['text_features']
                numerical_features = listing['numerical_features'].unsqueeze(0).to(self.device)
                categorical_features = listing['categorical_features'].unsqueeze(0).to(self.device)
                
                # Get content similarity
                content_representation, similarity = model(
                    text_features, numerical_features, categorical_features
                )
                
                # Calculate recommendation score
                score = similarity.item()
                
                recommendations.append({
                    'listing_id': listing['listing_id'],
                    'title': listing['title'],
                    'score': score,
                    'confidence': score,
                    'reason': 'Content similarity'
                })
        
        # Sort by score and return top-k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]
    
    async def _collaborative_recommendations(self,
                                           model: nn.Module,
                                           user_id: str,
                                           listings_data: List[Dict],
                                           top_k: int) -> List[Dict]:
        """Collaborative filtering recommendations"""
        recommendations = []
        
        # Get user index
        user_idx = self._get_user_index(user_id)
        
        model.eval()
        with torch.no_grad():
            for listing in listings_data:
                # Get item index
                item_idx = self._get_item_index(listing['listing_id'])
                
                # Get collaborative score
                user_tensor = torch.LongTensor([user_idx]).to(self.device)
                item_tensor = torch.LongTensor([item_idx]).to(self.device)
                
                collaborative_score = model.collaborative_model(user_tensor, item_tensor)
                
                recommendations.append({
                    'listing_id': listing['listing_id'],
                    'title': listing['title'],
                    'score': collaborative_score.item(),
                    'confidence': collaborative_score.item(),
                    'reason': 'Collaborative filtering'
                })
        
        # Sort by score and return top-k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]
    
    async def _hybrid_recommendations(self,
                                    model: nn.Module,
                                    user_id: str,
                                    listings_data: List[Dict],
                                    top_k: int) -> List[Dict]:
        """Hybrid recommendations combining content and collaborative"""
        recommendations = []
        
        # Get user index
        user_idx = self._get_user_index(user_id)
        
        model.eval()
        with torch.no_grad():
            for listing in listings_data:
                # Prepare features
                text_features = listing['text_features']
                numerical_features = listing['numerical_features'].unsqueeze(0).to(self.device)
                categorical_features = listing['categorical_features'].unsqueeze(0).to(self.device)
                
                user_tensor = torch.LongTensor([user_idx]).to(self.device)
                item_idx = self._get_item_index(listing['listing_id'])
                item_tensor = torch.LongTensor([item_idx]).to(self.device)
                
                # Get hybrid recommendation
                outputs = model(user_tensor, item_tensor, text_features, numerical_features, categorical_features)
                
                recommendation_score = outputs['recommendation_score'].item()
                content_similarity = outputs['content_similarity'].item()
                collaborative_score = outputs['collaborative_score'].item()
                uncertainty = outputs['uncertainty'].item()
                
                recommendations.append({
                    'listing_id': listing['listing_id'],
                    'title': listing['title'],
                    'score': recommendation_score,
                    'confidence': 1,
                    'content_similarity': content_similarity,
                    'collaborative_score': collaborative_score,
                    'reason': 'Hybrid recommendation'
                })
        
        # Sort by score and return top-k
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]
    
    async def _process_listings_for_inference(self, listings_data: List[Dict]) -> List[Dict]:
        """Process listings for inference"""
        processed_listings = []
        
        for listing in listings_data:
            # Process text
            text = f"{listing.get('title', '')} {listing.get('description', '')} {listing.get('category', '')}"
            text_encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Process numerical features
            numerical_features = torch.FloatTensor([
                listing.get('price', 0),
                listing.get('rating', 0),
                listing.get('review_count', 0),
                listing.get('view_count', 0),
                listing.get('favorite_count', 0),
                listing.get('days_since_created', 0),
                listing.get('seller_rating', 0),
                listing.get('seller_review_count', 0)  ])
            
            # Process categorical features
            categorical_features = torch.LongTensor([
                hash(listing.get('category', '')) % 1000,
                hash(listing.get('subcategory', '')) % 1000,
                hash(listing.get('condition', '')) % 1000,
                hash(listing.get('location', '')) % 1000,
                hash(listing.get('seller_type', '')) % 100  ])
            
            processed_listings.append({
                'listing_id': listing.get('id'),
                'title': listing.get('title'),
                'text_features': text_encoding,
                'numerical_features': numerical_features,
                'categorical_features': categorical_features
            })
        
        return processed_listings
    
    def _get_user_profile(self, user_id: str) -> Dict:
        """Get user profile (simplified)       # In a real implementation, this would fetch user preferences from database"""
        return {
            'preferred_categories': ['electronics', 'books'],
            'preferred_price_range': [10, 100],
            'preferred_rating': 4.0
        }
    
    def _get_user_index(self, user_id: str) -> int:
        """Get user index (simplified)       # In a real implementation, this would use a proper user mapping"""
        return hash(user_id) % 10
    
    def _get_item_index(self, item_id: str) -> int:
        """Get item index (simplified)       # In a real implementation, this would use a proper item mapping"""
        return hash(item_id) % 10    
    async def _load_inference_model(self, model_id: str) -> nn.Module:
        """Load inference model from disk"""
        try:
            model_path = f"{self.model_paths['inference']}/{model_id}.pth"
            metadata_path = f"{self.model_paths['inference']}/{model_id}_metadata.json"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Inference model not found: {model_path}")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Create model
            model_type = metadata['model_type']
            config = metadata['config']
            
            if model_type == 'advanced':             model = AdvancedListingInferenceModel(**config).to(self.device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading inference model: {e}")
            raise
    
    async def generate_listings_from_profile(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate world-class material listings from company profile using advanced AI"""
        self.logger.info(f"🚀 Generating world-class material listings for: {company_profile.get('name', 'Unknown Company')}")
        
        company_name = company_profile.get('name', 'Unknown Company')
        industry = company_profile.get('industry', 'manufacturing')
        location = company_profile.get('location', 'Global')
        materials = company_profile.get('materials', [])
        waste_streams = company_profile.get('waste_streams', [])
        products = company_profile.get('products', [])
        
        # Advanced material analysis
        material_analysis = await self._analyze_company_materials(company_profile)
        
        # Generate primary material listings
        primary_listings = await self._generate_primary_materials(company_profile, material_analysis)
        
        # Generate waste material listings
        waste_listings = await self._generate_waste_materials(company_profile, material_analysis)
        
        # Generate specialty material listings
        specialty_listings = await self._generate_specialty_materials(company_profile, material_analysis)
        
        # Generate by-product listings
        byproduct_listings = await self._generate_byproduct_materials(company_profile, material_analysis)
        
        # Combine all listings
        all_listings = primary_listings + waste_listings + specialty_listings + byproduct_listings
        
        # Enhance listings with market intelligence
        enhanced_listings = await self._enhance_listings_with_market_intelligence(all_listings, company_profile)
        
        # Validate and filter listings
        validated_listings = await self._validate_listings(enhanced_listings, company_profile)
        
        # Generate market insights
        market_insights = await self._generate_market_insights(validated_listings, company_profile)
        
        result = {
            'company_name': company_name,
            'industry': industry,
            'location': location,
            'predicted_outputs': validated_listings,
            'material_analysis': material_analysis,
            'market_insights': market_insights,
            'generation_metadata': {
                'total_listings': len(validated_listings),
                'primary_materials': len(primary_listings),
                'waste_materials': len(waste_listings),
                'specialty_materials': len(specialty_listings),
                'byproduct_materials': len(byproduct_listings),
                'total_estimated_value': sum(listing.get('potential_value', 0) for listing in validated_listings),
                'generated_at': datetime.now().isoformat(),
                'ai_confidence_score': self._calculate_generation_confidence(validated_listings, company_profile)
            }
        }
        
        self.logger.info(f"✅ Generated {len(validated_listings)} world-class material listings for {company_name}")
        return result
    
    async def _analyze_company_materials(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced analysis of company materials and capabilities"""
        analysis = {
            'primary_materials': [],
            'waste_streams': [],
            'byproducts': [],
            'specialty_materials': [],
            'material_capabilities': {},
            'processing_capabilities': [],
            'quality_standards': [],
            'market_positioning': {},
            'sustainability_metrics': {}
        }
        
        company_name = company_profile.get('name', '')
        industry = company_profile.get('industry', '')
        materials = company_profile.get('materials', [])
        waste_streams = company_profile.get('waste_streams', [])
        products = company_profile.get('products', [])
        
        # Analyze primary materials
        for material in materials:
            material_info = await self._get_material_intelligence(material, industry)
            analysis['primary_materials'].append(material_info)
        
        # Analyze waste streams
        for waste in waste_streams:
            waste_info = await self._get_waste_intelligence(waste, industry)
            analysis['waste_streams'].append(waste_info)
        
        # Analyze byproducts from products
        for product in products:
            byproduct_info = await self._get_byproduct_intelligence(product, industry)
            if byproduct_info:
                analysis['byproducts'].append(byproduct_info)
        
        # Determine material capabilities
        analysis['material_capabilities'] = await self._determine_material_capabilities(materials, industry)
        
        # Determine processing capabilities
        analysis['processing_capabilities'] = await self._determine_processing_capabilities(industry, materials)
        
        # Determine quality standards
        analysis['quality_standards'] = await self._determine_quality_standards(industry, company_name)
        
        # Determine market positioning
        analysis['market_positioning'] = await self._determine_market_positioning(company_name, industry, materials)
        
        # Calculate sustainability metrics
        analysis['sustainability_metrics'] = await self._calculate_sustainability_metrics(materials, waste_streams, industry)
        
        return analysis
    
    async def _get_material_intelligence(self, material: str, industry: str) -> Dict[str, Any]:
        """Get comprehensive intelligence about a material"""
        material_lower = material.lower()
        
        # Material properties database
        material_properties = {
            'steel': {
                'type': 'metal',
                'density': 7.85,
                'melting_point': 1370,
                'tensile_strength': 400,
                'corrosion_resistance': 'low',
                'recyclability': 'high',
                'market_demand': 'high',
                'value_range': (2000, 8000),
                'processing_methods': ['melting', 'casting', 'rolling', 'heat_treatment'],
                'applications': ['construction', 'automotive', 'manufacturing', 'aerospace'],
                'quality_grades': ['A', 'B', 'C'],
                'units': ['tons', 'kg', 'pieces']
            },
            'aluminum': {
                'type': 'metal',
                'density': 2.7,
                'melting_point': 660,
                'tensile_strength': 310,
                'corrosion_resistance': 'high',
                'recyclability': 'very_high',
                'market_demand': 'high',
                'value_range': (3000, 12000),
                'processing_methods': ['smelting', 'casting', 'extrusion', 'anodizing'],
                'applications': ['automotive', 'aerospace', 'packaging', 'construction'],
                'quality_grades': ['A', 'B', 'C'],
                'units': ['tons', 'kg', 'pieces']
            },
            'copper': {
                'type': 'metal',
                'density': 8.96,
                'melting_point': 1085,
                'tensile_strength': 220,
                'corrosion_resistance': 'high',
                'recyclability': 'very_high',
                'market_demand': 'high',
                'value_range': (8000, 15000),
                'processing_methods': ['refining', 'casting', 'drawing', 'annealing'],
                'applications': ['electronics', 'electrical', 'construction', 'automotive'],
                'quality_grades': ['A', 'B', 'C'],
                'units': ['tons', 'kg', 'pieces']
            },
            'plastic': {
                'type': 'polymer',
                'density': 0.92,
                'melting_point': 130,
                'tensile_strength': 30,
                'corrosion_resistance': 'very_high',
                'recyclability': 'moderate',
                'market_demand': 'high',
                'value_range': (1500, 4000),
                'processing_methods': ['extrusion', 'injection_molding', 'blow_molding'],
                'applications': ['packaging', 'automotive', 'electronics', 'construction'],
                'quality_grades': ['A', 'B', 'C'],
                'units': ['tons', 'kg']
            },
            'chemical': {
                'type': 'chemical',
                'density': 1.0,
                'melting_point': 0,
                'tensile_strength': 0,
                'corrosion_resistance': 'variable',
                'recyclability': 'low',
                'market_demand': 'high',
                'value_range': (500, 2000),
                'processing_methods': ['synthesis', 'purification', 'formulation'],
                'applications': ['pharmaceutical', 'agriculture', 'manufacturing', 'energy'],
                'quality_grades': ['A', 'B', 'C'],
                'units': ['tons', 'liters', 'kg']
            }
        }
        
        # Find matching material
        for key, properties in material_properties.items():
            if key in material_lower or material_lower in key:
                return {
                    'name': material,
                    'properties': properties,
                    'industry_specific': await self._get_industry_specific_properties(material, industry),
                    'market_analysis': await self._get_market_analysis(material, industry),
                    'quality_assessment': await self._assess_material_quality(material, industry)
                }
        
        # Default material info
        return {
            'name': material,
            'properties': {
                'type': 'unknown',
                'density': 1.0,
                'market_demand': 'moderate',
                'value_range': (1000, 5000),
                'processing_methods': ['general_processing'],
                'applications': ['general'],
                'quality_grades': ['B', 'C'],
                'units': ['tons', 'kg']
            },
            'industry_specific': {},
            'market_analysis': {},
            'quality_assessment': {'grade': 'B', 'confidence': 0.7}
        }
    
    async def _get_waste_intelligence(self, waste: str, industry: str) -> Dict[str, Any]:
        """Get comprehensive intelligence about waste materials"""
        waste_lower = waste.lower()
        
        waste_properties = {
            'scrap': {
                'type': 'recyclable_waste',
                'recyclability': 'high',
                'processing_required': 'moderate',
                'value_potential': 'high',
                'market_demand': 'high',
                'value_range': (300, 1500),
                'processing_methods': ['sorting', 'cleaning', 'shredding', 'melting'],
                'applications': ['recycling', 'remelting', 'refining'],
                'quality_grades': ['B', 'C', 'D'],
                'units': ['tons', 'kg']
            },
            'waste': {
                'type': 'general_waste',
                'recyclability': 'moderate',
                'processing_required': 'high',
                'value_potential': 'moderate',
                'market_demand': 'moderate',
                'value_range': (100, 800),
                'processing_methods': ['sorting', 'cleaning', 'processing'],
                'applications': ['recycling', 'energy_recovery', 'landfill'],
                'quality_grades': ['C', 'D'],
                'units': ['tons', 'kg']
            },
            'sludge': {
                'type': 'liquid_waste',
                'recyclability': 'low',
                'processing_required': 'high',
                'value_potential': 'low',
                'market_demand': 'low',
                'value_range': (50, 300),
                'processing_methods': ['dewatering', 'treatment', 'disposal'],
                'applications': ['treatment', 'landfill', 'incineration'],
                'quality_grades': ['D'],
                'units': ['tons', 'liters']
            }
        }
        
        # Find matching waste type
        for key, properties in waste_properties.items():
            if key in waste_lower:
                return {
                    'name': waste,
                    'properties': properties,
                    'industry_specific': await self._get_industry_specific_waste_properties(waste, industry),
                    'market_analysis': await self._get_waste_market_analysis(waste, industry),
                    'quality_assessment': await self._assess_waste_quality(waste, industry)
                }
        
        # Default waste info
        return {
            'name': waste,
            'properties': {
                'type': 'general_waste',
                'recyclability': 'moderate',
                'processing_required': 'high',
                'value_potential': 'moderate',
                'market_demand': 'moderate',
                'value_range': (100, 500),
                'processing_methods': ['sorting', 'cleaning'],
                'applications': ['recycling', 'disposal'],
                'quality_grades': ['C', 'D'],
                'units': ['tons', 'kg']
            },
            'industry_specific': {},
            'market_analysis': {},
            'quality_assessment': {'grade': 'C', 'confidence': 0.6}
        }
    
    async def _get_byproduct_intelligence(self, product: str, industry: str) -> Optional[Dict[str, Any]]:
        """Get intelligence about potential byproducts from products"""
        product_lower = product.lower()
        
        # Byproduct mapping
        byproduct_mapping = {
            'steel': ['steel_scrap', 'iron_oxide', 'slag'],
            'aluminum': ['aluminum_scrap', 'alumina', 'red_mud'],
            'plastic': ['plastic_waste', 'polymer_scraps', 'recycled_pellets'],
            'chemical': ['chemical_waste', 'solvents', 'catalysts'],
            'oil': ['oil_sludge', 'waste_oil', 'refinery_waste'],
            'gas': ['gas_waste', 'condensate', 'sulfur']
        }
        
        for key, byproducts in byproduct_mapping.items():
            if key in product_lower:
                return {
                    'product': product,
                    'byproducts': byproducts,
                    'value_potential': 'moderate',
                    'market_demand': 'moderate'
                }
        
        return None
    
    async def _determine_material_capabilities(self, materials: List[str], industry: str) -> Dict[str, Any]:
        """Determine company's material processing capabilities"""
        capabilities = {
            'primary_processing': [],
            'secondary_processing': [],
            'quality_control': [],
            'testing_capabilities': [],
            'certification_standards': []
        }
        
        # Industry-specific capabilities
        if 'steel' in industry or 'metal' in industry:
            capabilities['primary_processing'].extend(['melting', 'casting', 'rolling', 'forging'])
            capabilities['secondary_processing'].extend(['heat_treatment', 'surface_treatment', 'machining'])
            capabilities['quality_control'].extend(['chemical_analysis', 'mechanical_testing', 'non_destructive_testing'])
            capabilities['certification_standards'].extend(['ISO_9001', 'ASTM', 'API'])
        
        elif 'chemical' in industry:
            capabilities['primary_processing'].extend(['synthesis', 'purification', 'formulation'])
            capabilities['secondary_processing'].extend(['packaging', 'quality_control', 'storage'])
            capabilities['quality_control'].extend(['chemical_analysis', 'purity_testing', 'stability_testing'])
            capabilities['certification_standards'].extend(['ISO_9001', 'GMP', 'FDA'])
        
        elif 'plastic' in industry or 'polymer' in industry:
            capabilities['primary_processing'].extend(['extrusion', 'injection_molding', 'blow_molding'])
            capabilities['secondary_processing'].extend(['thermoforming', 'assembly', 'packaging'])
            capabilities['quality_control'].extend(['dimensional_testing', 'material_testing', 'performance_testing'])
            capabilities['certification_standards'].extend(['ISO_9001', 'ASTM', 'UL'])
        
        return capabilities
    
    async def _determine_processing_capabilities(self, industry: str, materials: List[str]) -> List[str]:
        """Determine processing capabilities based on industry and materials"""
        capabilities = []
        
        # Material-based capabilities
        for material in materials:
            material_lower = material.lower()
            if 'steel' in material_lower or 'iron' in material_lower:
                capabilities.extend(['melting', 'casting', 'rolling', 'heat_treatment'])
            elif 'aluminum' in material_lower:
                capabilities.extend(['smelting', 'casting', 'extrusion', 'anodizing'])
            elif 'plastic' in material_lower or 'polymer' in material_lower:
                capabilities.extend(['extrusion', 'injection_molding', 'blow_molding'])
            elif 'chemical' in material_lower:
                capabilities.extend(['synthesis', 'purification', 'formulation'])
        
        # Industry-based capabilities
        if 'manufacturing' in industry:
            capabilities.extend(['quality_control', 'packaging', 'logistics'])
        elif 'mining' in industry:
            capabilities.extend(['extraction', 'processing', 'refinement'])
        elif 'energy' in industry:
            capabilities.extend(['generation', 'transmission', 'distribution'])
        
        return list(set(capabilities))  # Remove duplicates
    
    async def _determine_quality_standards(self, industry: str, company_name: str) -> List[str]:
        """Determine quality standards based on industry and company"""
        standards = ['ISO_9001']  # Base standard
        
        # Industry-specific standards
        if 'steel' in industry or 'metal' in industry:
            standards.extend(['ASTM', 'API', 'ASME'])
        elif 'chemical' in industry:
            standards.extend(['GMP', 'FDA', 'REACH'])
        elif 'automotive' in industry:
            standards.extend(['IATF_16949', 'VDA'])
        elif 'aerospace' in industry:
            standards.extend(['AS9100', 'NADCAP'])
        elif 'food' in industry:
            standards.extend(['HACCP', 'FSSC_22000'])
        
        # Company-specific standards (based on name patterns)
        if any(word in company_name.lower() for word in ['international', 'global', 'world']):
            standards.extend(['ISO_14001', 'OHSAS_18001'])
        
        return standards
    
    async def _determine_market_positioning(self, company_name: str, industry: str, materials: List[str]) -> Dict[str, Any]:
        """Determine market positioning and competitive advantages"""
        positioning = {
            'market_segment': 'general',
            'competitive_advantages': [],
            'target_markets': [],
            'value_proposition': '',
            'differentiation_factors': []
        }
        
        # Determine market segment
        if len(materials) > 10:
            positioning['market_segment'] = 'diversified'
        elif any('premium' in material.lower() for material in materials):
            positioning['market_segment'] = 'premium'
        elif any('waste' in material.lower() for material in materials):
            positioning['market_segment'] = 'waste_management'
        
        # Determine competitive advantages
        if 'international' in company_name.lower():
            positioning['competitive_advantages'].append('global_presence')
        if len(materials) > 5:
            positioning['competitive_advantages'].append('diversified_portfolio')
        if 'steel' in industry or 'metal' in industry:
            positioning['competitive_advantages'].append('heavy_industry_expertise')
        
        # Determine target markets
        if 'steel' in industry:
            positioning['target_markets'].extend(['construction', 'automotive', 'manufacturing'])
        elif 'chemical' in industry:
            positioning['target_markets'].extend(['pharmaceutical', 'agriculture', 'energy'])
        elif 'plastic' in industry:
            positioning['target_markets'].extend(['packaging', 'automotive', 'electronics'])
        
        return positioning
    
    async def _calculate_sustainability_metrics(self, materials: List[str], waste_streams: List[str], industry: str) -> Dict[str, Any]:
        """Calculate sustainability metrics"""
        metrics = {
            'recyclability_score': 0.0,
            'carbon_footprint': 0.0,
            'water_usage': 0.0,
            'energy_intensity': 0.0,
            'waste_reduction_potential': 0.0,
            'circular_economy_contribution': 0.0
        }
        
        # Calculate recyclability score
        recyclable_materials = sum(1 for material in materials if any(word in material.lower() for word in ['steel', 'aluminum', 'copper', 'plastic']))
        metrics['recyclability_score'] = recyclable_materials / len(materials) if materials else 0.0
        
        # Calculate waste reduction potential
        if waste_streams:
            metrics['waste_reduction_potential'] = 0.7  # Base potential
        
        # Industry-specific metrics
        if 'steel' in industry or 'metal' in industry:
            metrics['carbon_footprint'] = 0.8
            metrics['energy_intensity'] = 0.9
        elif 'chemical' in industry:
            metrics['carbon_footprint'] = 0.7
            metrics['water_usage'] = 0.8
        elif 'plastic' in industry:
            metrics['carbon_footprint'] = 0.6
            metrics['energy_intensity'] = 0.7
        
        return metrics

    async def _generate_primary_materials(self, company_profile: Dict[str, Any], material_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate primary material listings with advanced AI"""
        listings = []
        materials = company_profile.get('materials', [])
        
        for material in materials:
            material_info = next((m for m in material_analysis['primary_materials'] if m['name'] == material), None)
            if material_info:
                listing = await self._create_material_listing(material_info, company_profile, 'primary')
                listings.append(listing)
        
        return listings
    
    async def _generate_waste_materials(self, company_profile: Dict[str, Any], material_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate waste material listings with advanced AI"""
        listings = []
        waste_streams = company_profile.get('waste_streams', [])
        
        for waste in waste_streams:
            waste_info = next((w for w in material_analysis['waste_streams'] if w['name'] == waste), None)
            if waste_info:
                listing = await self._create_material_listing(waste_info, company_profile, 'waste')
                listings.append(listing)
        
        return listings
    
    async def _generate_specialty_materials(self, company_profile: Dict[str, Any], material_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specialty material listings with advanced AI"""
        listings = []
        industry = company_profile.get('industry', '')
        company_name = company_profile.get('name', '')
        
        # Generate specialty materials based on industry
        specialty_materials = await self._get_specialty_materials(industry, company_name)
        
        for material_name in specialty_materials:
            material_info = {
                'name': material_name,
                'properties': await self._get_material_intelligence(material_name, industry),
                'industry_specific': {},
                'market_analysis': {},
                'quality_assessment': {'grade': 'A', 'confidence': 0.8}
            }
            
            listing = await self._create_material_listing(material_info, company_profile, 'specialty')
            listings.append(listing)
        
        return listings
    
    async def _generate_byproduct_materials(self, company_profile: Dict[str, Any], material_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate byproduct material listings with advanced AI"""
        listings = []
        byproducts = material_analysis.get('byproducts', [])
        
        for byproduct_info in byproducts:
            for byproduct_name in byproduct_info.get('byproducts', []):
                material_info = {
                    'name': byproduct_name,
                    'properties': await self._get_material_intelligence(byproduct_name, company_profile.get('industry', '')),
                    'industry_specific': {},
                    'market_analysis': {},
                    'quality_assessment': {'grade': 'B', 'confidence': 0.7}
                }
                
                listing = await self._create_material_listing(material_info, company_profile, 'byproduct')
                listings.append(listing)
        
        return listings
    
    async def _create_material_listing(self, material_info: Dict[str, Any], company_profile: Dict[str, Any], material_type: str) -> Dict[str, Any]:
        """Create sophisticated material listing with advanced features"""
        material_name = material_info['name']
        properties = material_info['properties']
        quality_assessment = material_info.get('quality_assessment', {'grade': 'B', 'confidence': 0.7})
        
        # Determine quantity and unit
        quantity, unit = await self._determine_quantity_and_unit(properties, company_profile)
        
        # Calculate potential value
        potential_value = await self._calculate_material_value(material_info, company_profile, quantity)
        
        # Generate description
        description = await self._generate_material_description(material_info, company_profile)
        
        # Determine quality grade
        quality_grade = quality_assessment.get('grade', 'B')
        
        # Determine material type
        material_category = properties.get('type', 'unknown')
        
        listing = {
            'company_id': company_profile.get('id', ''),
            'company_name': company_profile.get('name', 'Unknown Company'),
            'material_name': material_name.replace('_', ' ').title(),
            'material_type': material_type,
            'category': material_category,
            'quantity': quantity,
            'unit': unit,
            'description': description,
            'quality_grade': quality_grade,
            'potential_value': potential_value,
            'ai_generated': True,
            'generated_at': datetime.now().isoformat(),
            'metadata': {
                'material_properties': properties,
                'quality_assessment': quality_assessment,
                'market_analysis': material_info.get('market_analysis', {}),
                'industry_specific': material_info.get('industry_specific', {}),
                'confidence_score': quality_assessment.get('confidence', 0.7)
            }
        }
        
        return listing
    
    async def _determine_quantity_and_unit(self, properties: Dict[str, Any], company_profile: Dict[str, Any]) -> Tuple[float, str]:
        """Determine quantity and unit based on material properties and company profile"""
        # Get available units
        units = properties.get('units', ['tons'])
        unit = random.choice(units)
        
        # Determine quantity based on material type and company size
        company_size = company_profile.get('employee_count', 1000)
        material_type = properties.get('type', 'unknown')
        
        if material_type == 'metal':
            if company_size > 10000:
                quantity = random.uniform(500, 2000)
            elif company_size > 5000:
                quantity = random.uniform(200, 800)
            else:
                quantity = random.uniform(50, 300)
        elif material_type == 'chemical':
            if company_size > 10000:
                quantity = random.uniform(100, 500)
            elif company_size > 5000:
                quantity = random.uniform(50, 200)
            else:
                quantity = random.uniform(10, 100)
        elif material_type == 'polymer':
            if company_size > 10000:
                quantity = random.uniform(300, 1000)
            elif company_size > 5000:
                quantity = random.uniform(100, 400)
            else:
                quantity = random.uniform(20, 150)
        else:
            quantity = random.uniform(50, 500)
        
        return quantity, unit
    
    async def _calculate_material_value(self, material_info: Dict[str, Any], company_profile: Dict[str, Any], quantity: float) -> float:
        """Calculate sophisticated material value"""
        properties = material_info['properties']
        value_range = properties.get('value_range', (1000, 5000))
        base_value_per_unit = random.uniform(value_range[0], value_range[1])
        
        # Apply market demand multiplier
        market_demand = properties.get('market_demand', 'moderate')
        demand_multipliers = {'high': 1.3, 'moderate': 1.0, 'low': 0.7}
        demand_multiplier = demand_multipliers.get(market_demand, 1.0)
        
        # Apply quality multiplier
        quality_assessment = material_info.get('quality_assessment', {'grade': 'B'})
        quality_grade = quality_assessment.get('grade', 'B')
        quality_multipliers = {'A': 1.4, 'B': 1.0, 'C': 0.8, 'D': 0.6}
        quality_multiplier = quality_multipliers.get(quality_grade, 1.0)
        
        # Apply company size multiplier
        company_size = company_profile.get('employee_count', 1000)
        if company_size > 10000:
            size_multiplier = 1.2
        elif company_size > 5000:
            size_multiplier = 1.1
        else:
            size_multiplier = 1.0
        
        # Apply industry multiplier
        industry = company_profile.get('industry', '')
        industry_multipliers = {
            'steel': 1.1,
            'chemical': 1.3,
            'automotive': 1.2,
            'aerospace': 1.4,
            'pharmaceutical': 1.5
        }
        industry_multiplier = 1.0
        for key, multiplier in industry_multipliers.items():
            if key in industry.lower():
                industry_multiplier = multiplier
                break
        
        # Calculate final value
        final_value = base_value_per_unit * quantity * demand_multiplier * quality_multiplier * size_multiplier * industry_multiplier
        
        return round(final_value, 2)
    
    async def _generate_material_description(self, material_info: Dict[str, Any], company_profile: Dict[str, Any]) -> str:
        """Generate sophisticated material description"""
        material_name = material_info['name']
        properties = material_info['properties']
        company_name = company_profile.get('name', 'Unknown Company')
        industry = company_profile.get('industry', '')
        
        description = f"{material_name.replace('_', ' ').title()} from {company_name} operations. "
        
        # Add material properties
        material_type = properties.get('type', 'unknown')
        if material_type == 'metal':
            description += "High-quality metallic material with excellent mechanical properties. "
        elif material_type == 'chemical':
            description += "Industrial-grade chemical material with precise specifications. "
        elif material_type == 'polymer':
            description += "Advanced polymer material with superior performance characteristics. "
        
        # Add applications
        applications = properties.get('applications', [])
        if applications:
            app_list = ', '.join(applications[:3])
            description += f"Suitable for {app_list} applications. "
        
        # Add quality information
        quality_assessment = material_info.get('quality_assessment', {'grade': 'B'})
        quality_grade = quality_assessment.get('grade', 'B')
        description += f"Produced under {quality_grade}-grade quality standards. "
        
        # Add industry-specific information
        if 'steel' in industry or 'metal' in industry:
            description += "Manufactured using advanced metallurgical processes. "
        elif 'chemical' in industry:
            description += "Synthesized using state-of-the-art chemical processes. "
        elif 'automotive' in industry:
            description += "Certified for automotive industry applications. "
        
        return description
    
    async def _get_specialty_materials(self, industry: str, company_name: str) -> List[str]:
        """Get specialty materials based on industry and company"""
        specialty_materials = []
        
        # Industry-specific specialty materials
        if 'steel' in industry or 'metal' in industry:
            specialty_materials.extend(['titanium_alloy', 'stainless_steel', 'high_strength_steel'])
        elif 'chemical' in industry:
            specialty_materials.extend(['catalysts', 'specialty_chemicals', 'pharmaceutical_intermediates'])
        elif 'automotive' in industry:
            specialty_materials.extend(['lightweight_alloys', 'composite_materials', 'advanced_polymers'])
        elif 'aerospace' in industry:
            specialty_materials.extend(['aerospace_alloys', 'composite_materials', 'specialty_metals'])
        elif 'pharmaceutical' in industry:
            specialty_materials.extend(['active_pharmaceutical_ingredients', 'excipients', 'intermediates'])
        
        # Company-specific specialty materials
        if 'international' in company_name.lower():
            specialty_materials.extend(['export_grade_materials', 'international_standards'])
        if 'advanced' in company_name.lower():
            specialty_materials.extend(['advanced_materials', 'innovative_solutions'])
        
        return specialty_materials[:3]  # Limit to 3 specialty materials
    
    async def _enhance_listings_with_market_intelligence(self, listings: List[Dict[str, Any]], company_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance listings with market intelligence and trends"""
        enhanced_listings = []
        
        for listing in listings:
            enhanced_listing = listing.copy()
            
            # Add market intelligence
            market_intelligence = await self._get_market_intelligence(listing, company_profile)
            enhanced_listing['market_intelligence'] = market_intelligence
            
            # Add pricing intelligence
            pricing_intelligence = await self._get_pricing_intelligence(listing, company_profile)
            enhanced_listing['pricing_intelligence'] = pricing_intelligence
            
            # Add demand forecasting
            demand_forecast = await self._get_demand_forecast(listing, company_profile)
            enhanced_listing['demand_forecast'] = demand_forecast
            
            enhanced_listings.append(enhanced_listing)
        
        return enhanced_listings
    
    async def _get_market_intelligence(self, listing: Dict[str, Any], company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get market intelligence for a material"""
        material_name = listing['material_name']
        industry = company_profile.get('industry', '')
        
        intelligence = {
            'market_trend': 'stable',
            'demand_outlook': 'moderate',
            'supply_conditions': 'balanced',
            'price_trend': 'stable',
            'competitive_landscape': 'moderate',
            'regulatory_environment': 'stable'
        }
        
        # Material-specific intelligence
        if 'steel' in material_name.lower():
            intelligence.update({
                'market_trend': 'growing',
                'demand_outlook': 'high',
                'price_trend': 'increasing'
            })
        elif 'aluminum' in material_name.lower():
            intelligence.update({
                'market_trend': 'growing',
                'demand_outlook': 'high',
                'price_trend': 'increasing'
            })
        elif 'chemical' in material_name.lower():
            intelligence.update({
                'market_trend': 'stable',
                'demand_outlook': 'moderate',
                'regulatory_environment': 'evolving'
            })
        
        return intelligence
    
    async def _get_pricing_intelligence(self, listing: Dict[str, Any], company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get pricing intelligence for a material"""
        current_value = listing.get('potential_value', 0)
        
        return {
            'current_price': current_value,
            'price_range': (current_value * 0.8, current_value * 1.2),
            'price_volatility': 'low',
            'seasonal_factors': 'minimal',
            'cost_drivers': ['raw_materials', 'energy', 'transportation'],
            'pricing_strategy': 'market_based'
        }
    
    async def _get_demand_forecast(self, listing: Dict[str, Any], company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get demand forecast for a material"""
        material_name = listing['material_name']
        
        return {
            'short_term_demand': 'stable',
            'medium_term_demand': 'growing',
            'long_term_demand': 'growing',
            'growth_rate': '3-5%',
            'key_drivers': ['industrial_growth', 'infrastructure_development', 'technological_advancement'],
            'risk_factors': ['economic_downturn', 'regulatory_changes', 'supply_disruptions']
        }
    
    async def _validate_listings(self, listings: List[Dict[str, Any]], company_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate and filter listings"""
        validated_listings = []
        
        for listing in listings:
            if await self._validate_listing(listing, company_profile):
                validated_listings.append(listing)
        
        return validated_listings
    
    async def _validate_listing(self, listing: Dict[str, Any], company_profile: Dict[str, Any]) -> bool:
        """Validate individual listing"""
        # Check required fields
        required_fields = ['material_name', 'company_name', 'quantity', 'potential_value', 'quality_grade']
        for field in required_fields:
            if not listing.get(field):
                return False
        
        # Check reasonable values
        if listing.get('quantity', 0) <= 0:
            return False
        if listing.get('potential_value', 0) <= 0:
            return False
        
        # Check quality grade
        valid_grades = ['A', 'B', 'C', 'D']
        if listing.get('quality_grade') not in valid_grades:
            return False
        
        return True
    
    async def _generate_market_insights(self, listings: List[Dict[str, Any]], company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market insights from listings"""
        insights = {
            'total_market_value': sum(listing.get('potential_value', 0) for listing in listings),
            'material_distribution': {},
            'quality_distribution': {},
            'value_distribution': {},
            'market_opportunities': [],
            'risk_assessment': {},
            'recommendations': []
        }
        
        # Material distribution
        for listing in listings:
            material_type = listing.get('material_type', 'unknown')
            insights['material_distribution'][material_type] = insights['material_distribution'].get(material_type, 0) + 1
        
        # Quality distribution
        for listing in listings:
            quality_grade = listing.get('quality_grade', 'B')
            insights['quality_distribution'][quality_grade] = insights['quality_distribution'].get(quality_grade, 0) + 1
        
        # Market opportunities
        if len(listings) > 0:
            insights['market_opportunities'].append('Expand material portfolio')
            insights['market_opportunities'].append('Improve quality standards')
            insights['market_opportunities'].append('Optimize pricing strategy')
        
        # Risk assessment
        insights['risk_assessment'] = {
            'market_volatility': 'low',
            'supply_chain_risk': 'moderate',
            'regulatory_risk': 'low',
            'competitive_risk': 'moderate'
        }
        
        # Recommendations
        insights['recommendations'] = [
            'Focus on high-value materials',
            'Improve quality control processes',
            'Expand market reach',
            'Invest in sustainability initiatives'
        ]
        
        return insights
    
    def _calculate_generation_confidence(self, listings: List[Dict[str, Any]], company_profile: Dict[str, Any]) -> float:
        """Calculate confidence score for the generation process"""
        if not listings:
            return 0.0
        
        # Base confidence
        confidence = 0.7
        
        # Data completeness bonus
        if company_profile.get('materials') and company_profile.get('waste_streams'):
            confidence += 0.1
        
        # Listing quality bonus
        high_quality_listings = sum(1 for listing in listings if listing.get('quality_grade') in ['A', 'B'])
        quality_ratio = high_quality_listings / len(listings)
        confidence += quality_ratio * 0.1
        
        # Value distribution bonus
        total_value = sum(listing.get('potential_value', 0) for listing in listings)
        if total_value > 100000:  # High value portfolio
            confidence += 0.1
        
        return min(1.0, confidence)

    async def get_system_health(self) -> Dict:
        """Get system health metrics"""
        try:
            health_metrics = {
                'status': 'healthy',
                'device': str(self.device),
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'inference_models_loaded': len(self.inference_models),
                'recommendation_models_loaded': len(self.recommendation_models),
                'inference_monitor_status': self.inference_monitor.get_status(),
                'metrics_tracker_status': self.metrics_tracker.get_status(),
                'performance_metrics': {
                    'avg_inference_time': self.inference_monitor.get_avg_inference_time(),
                    'recommendation_accuracy': self.metrics_tracker.get_recommendation_accuracy(),
                    'avg_recommendation_confidence': self.inference_monitor.get_avg_recommendation_confidence()
                }
            }
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {'status': 'error', 'error': str(e)}

# Initialize service
listing_inference_service = ListingInferenceService() 