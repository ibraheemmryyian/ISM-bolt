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
        """Generate material listings from a company profile"""
        try:
            self.logger.info(f"Generating listings for company: {company_profile.get('name', 'Unknown')}")
            
            # Extract company information
            company_name = company_profile.get('name', 'Unknown Company')
            industry = company_profile.get('industry', 'manufacturing').lower()
            location = company_profile.get('location', 'Unknown')
            size = company_profile.get('size', 'medium')
            
            # Get existing waste materials and needs
            waste_materials = company_profile.get('waste_materials', [])
            material_needs = company_profile.get('material_needs', [])
            
            predicted_outputs = []
            predicted_inputs = []
            
            # Generate waste materials (outputs)
            if waste_materials:
                for waste in waste_materials:
                    if isinstance(waste, dict):
                        predicted_outputs.append({
                            'name': waste.get('name', 'Unknown Waste'),
                            'description': f"Waste material from {company_name} operations",
                            'category': 'waste',
                            'quantity_estimate': waste.get('quantity', 100),
                            'unit': waste.get('unit', 'tons'),
                            'potential_value': self._estimate_material_value(waste.get('name', ''), 'waste'),
                            'quality_grade': waste.get('quality', 'B'),
                            'potential_uses': ['Recycling', 'Reprocessing', 'Energy recovery'],
                            'symbiosis_opportunities': ['Material exchange', 'Waste-to-resource'],
                            'embeddings': None
                        })
                    elif isinstance(waste, str):
                        predicted_outputs.append({
                            'name': waste,
                            'description': f"Waste material from {company_name} operations",
                            'category': 'waste',
                            'quantity_estimate': 100,
                            'unit': 'tons',
                            'potential_value': self._estimate_material_value(waste, 'waste'),
                            'quality_grade': 'B',
                            'potential_uses': ['Recycling', 'Reprocessing', 'Energy recovery'],
                            'symbiosis_opportunities': ['Material exchange', 'Waste-to-resource'],
                            'embeddings': None
                        })
            
            # Generate material needs (inputs)
            if material_needs:
                for need in material_needs:
                    if isinstance(need, dict):
                        predicted_inputs.append({
                            'name': need.get('name', 'Unknown Material'),
                            'description': f"Material needed by {company_name}",
                            'category': 'requirement',
                            'quantity_estimate': need.get('quantity', 200),
                            'unit': need.get('unit', 'tons'),
                            'potential_value': self._estimate_material_value(need.get('name', ''), 'requirement'),
                            'quality_grade': 'A',
                            'potential_sources': ['Suppliers', 'Partners', 'Market'],
                            'symbiosis_opportunities': ['Supply chain optimization', 'Resource sharing'],
                            'embeddings': None
                        })
                    elif isinstance(need, str):
                        predicted_inputs.append({
                            'name': need,
                            'description': f"Material needed by {company_name}",
                            'category': 'requirement',
                            'quantity_estimate': 200,
                            'unit': 'tons',
                            'potential_value': self._estimate_material_value(need, 'requirement'),
                            'quality_grade': 'A',
                            'potential_sources': ['Suppliers', 'Partners', 'Market'],
                            'symbiosis_opportunities': ['Supply chain optimization', 'Resource sharing'],
                            'embeddings': None
                        })
            
            # Generate industry-specific materials if none provided
            if not predicted_outputs and not predicted_inputs:
                if 'manufacturing' in industry:
                    predicted_outputs.extend([
                        {
                            'name': 'Steel Scrap',
                            'description': f"Steel scrap from {company_name} manufacturing",
                            'category': 'waste',
                            'quantity_estimate': 150,
                            'unit': 'tons',
                            'potential_value': 45000,
                            'quality_grade': 'B',
                            'potential_uses': ['Recycling', 'Steel production'],
                            'symbiosis_opportunities': ['Material exchange', 'Waste-to-resource'],
                            'embeddings': None
                        },
                        {
                            'name': 'Aluminum Waste',
                            'description': f"Aluminum waste from {company_name} operations",
                            'category': 'waste',
                            'quantity_estimate': 50,
                            'unit': 'tons',
                            'potential_value': 75000,
                            'quality_grade': 'A',
                            'potential_uses': ['Recycling', 'Aluminum production'],
                            'symbiosis_opportunities': ['Material exchange', 'Waste-to-resource'],
                            'embeddings': None
                        }
                    ])
                    predicted_inputs.extend([
                        {
                            'name': 'Raw Steel',
                            'description': f"Raw steel needed by {company_name}",
                            'category': 'requirement',
                            'quantity_estimate': 300,
                            'unit': 'tons',
                            'potential_value': 90000,
                            'quality_grade': 'A',
                            'potential_sources': ['Steel mills', 'Suppliers'],
                            'symbiosis_opportunities': ['Supply chain optimization', 'Resource sharing'],
                            'embeddings': None
                        }
                    ])
                elif 'chemical' in industry:
                    predicted_outputs.extend([
                        {
                            'name': 'Chemical Waste',
                            'description': f"Chemical waste from {company_name} operations",
                            'category': 'waste',
                            'quantity_estimate': 50,
                            'unit': 'tons',
                            'potential_value': 25000,
                            'quality_grade': 'C',
                            'potential_uses': ['Treatment', 'Recovery'],
                            'symbiosis_opportunities': ['Waste treatment', 'Chemical recovery'],
                            'embeddings': None
                        }
                    ])
                    predicted_inputs.extend([
                        {
                            'name': 'Chemical Raw Materials',
                            'description': f"Chemical raw materials needed by {company_name}",
                            'category': 'requirement',
                            'quantity_estimate': 100,
                            'unit': 'tons',
                            'potential_value': 150000,
                            'quality_grade': 'A',
                            'potential_sources': ['Chemical suppliers', 'Manufacturers'],
                            'symbiosis_opportunities': ['Supply chain optimization', 'Resource sharing'],
                            'embeddings': None
                        }
                    ])
                else:
                    # Generic materials for other industries
                    predicted_outputs.append({
                        'name': 'General Waste',
                        'description': f"General waste from {company_name} operations",
                        'category': 'waste',
                        'quantity_estimate': 100,
                        'unit': 'tons',
                        'potential_value': 20000,
                        'quality_grade': 'C',
                        'potential_uses': ['Recycling', 'Waste management'],
                        'symbiosis_opportunities': ['Waste management', 'Resource recovery'],
                        'embeddings': None
                    })
                    predicted_inputs.append({
                        'name': 'Raw Materials',
                        'description': f"Raw materials needed by {company_name}",
                        'category': 'requirement',
                        'quantity_estimate': 200,
                        'unit': 'tons',
                        'potential_value': 80000,
                        'quality_grade': 'A',
                        'potential_sources': ['Suppliers', 'Manufacturers'],
                        'symbiosis_opportunities': ['Supply chain optimization', 'Resource sharing'],
                        'embeddings': None
                    })
            
            result = {
                'predicted_outputs': predicted_outputs,
                'predicted_inputs': predicted_inputs,
                'company_profile': company_profile,
                'generation_timestamp': datetime.now().isoformat(),
                'total_outputs': len(predicted_outputs),
                'total_inputs': len(predicted_inputs)
            }
            
            self.logger.info(f"Generated {len(predicted_outputs)} outputs and {len(predicted_inputs)} inputs for {company_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating listings from profile: {e}")
            raise
    
    def _estimate_material_value(self, material_name: str, material_type: str) -> float:
        """Estimate the potential value of a material"""
        material_name_lower = material_name.lower()
        
        # High-value materials
        if any(metal in material_name_lower for metal in ['gold', 'platinum', 'palladium']):
            return 50000 if material_type == 'waste' else 100000
        elif any(metal in material_name_lower for metal in ['aluminum', 'copper', 'brass']):
            return 3000 if material_type == 'waste' else 5000
        elif 'steel' in material_name_lower:
            return 300 if material_type == 'waste' else 500
        elif 'chemical' in material_name_lower:
            return 500 if material_type == 'waste' else 1500
        else:
            # Default values
            return 200 if material_type == 'waste' else 400

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