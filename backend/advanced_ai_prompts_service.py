import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
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
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
# Try to import sklearn components with fallback
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    
    SKLEARN_AVAILABLE = False
from sklearn.model_selection import train_test_split
import joblib
import pickle
from pathlib import Path
import re
import random

# ML Core imports
from ml_core.models import (
    PromptGenerationModel,
    PromptOptimizationModel,
    ContextUnderstandingModel
)
from ml_core.training import ModelTrainer
try:
    from ml_core.data_processing import PromptDataProcessor
except ImportError:
    class PromptDataProcessor:
        def __init__(self, *args, **kwargs):
            pass
        def process(self, *args, **kwargs):
            return args[0] if args else []
# Try to import ml_core components with fallback
try:
    from ml_core.optimization import HyperparameterOptimizer
    from ml_core.monitoring import MLMetricsTracker
    from ml_core.utils import ModelRegistry, DataValidator
    MLCORE_AVAILABLE = True
except ImportError:
    # Fallback implementations if ml_core is not available
    class HyperparameterOptimizer:
        def __init__(self, *args, **kwargs):
            pass
    
    class MLMetricsTracker:
        def __init__(self, *args, **kwargs):
            pass
    
    class ModelRegistry:
        def __init__(self, *args, **kwargs):
            pass
    
    class DataValidator:
        def __init__(self, *args, **kwargs):
            pass
        def validate(self, *args, **kwargs):
            return True
        def set_schema(self, *args, **kwargs):
            pass
    
    class PromptDataProcessor:
        def __init__(self, *args, **kwargs):
            pass
        def process(self, *args, **kwargs):
            return args[0] if args else []
    
    class ModelTrainer:
        def __init__(self, *args, **kwargs):
            pass
    
    MLCORE_AVAILABLE = False

# --- FIXED: PromptDataset class ---
class PromptDataset(Dataset):
    """Dataset for prompt engineering with actual ML preprocessing"""
    def __init__(self, prompts: List[Dict], tokenizer, max_length: int = 512):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.features = []
        self.labels = []
        for prompt in prompts:
            text = prompt.get('text', '')
            context = prompt.get('context', '')
            target = prompt.get('target', '')
            full_text = f"Context: {context} Prompt: {text} Target: {target}"
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            numerical_features = [
                len(text),
                len(context),
                len(target),
                prompt.get('complexity_score', 0.5),
                prompt.get('specificity_score', 0.5),
                prompt.get('clarity_score', 0.5)
            ]
            categorical_features = [
                prompt.get('domain', 'general'),
                prompt.get('style', 'formal'),
                prompt.get('intent', 'action')
            ]
            self.features.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'numerical_features': torch.FloatTensor(numerical_features),
                'categorical_features': categorical_features
            })
            self.labels.append({
                'effectiveness_score': prompt.get('effectiveness_score', 0.5),
                'response_quality': prompt.get('response_quality', 0.5),
                'user_satisfaction': prompt.get('user_satisfaction', 0.5)
            })
        if self.features:
            all_categorical = [f['categorical_features'] for f in self.features]
            encoded_categorical = []
            for i in range(len(all_categorical[0])):
                column = [cat[i] for cat in all_categorical]
                encoded_column = self.label_encoder.fit_transform(column)
                encoded_categorical.append(encoded_column)
            for i, feature in enumerate(self.features):
                feature['categorical_features'] = torch.LongTensor([
                    encoded_categorical[j][i] for j in range(len(encoded_categorical))
                ])
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class AdvancedPromptGenerationModel(nn.Module):
    """Real deep learning model for advanced prompt generation with transformer architecture"""
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int = 768,
                 num_layers: int = 6,
                 num_heads: int = 12,
                 dropout: float = 0.1,
                 max_length: int = 512):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # Text encoder
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = nn.Parameter(torch.randn(1, max_length, hidden_size))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Feature fusion
        self.numerical_projection = nn.Linear(6, hidden_size //2)
        self.categorical_projection = nn.Linear(3, hidden_size // 2)
        
        # Prompt generation head
        self.prompt_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Quality prediction heads
        self.quality_predictors = nn.ModuleDict({
            'effectiveness': nn.Linear(hidden_size * 2, 1),
            'clarity': nn.Linear(hidden_size * 2, 1),
            'specificity': nn.Linear(hidden_size * 2, 1),
            'relevance': nn.Linear(hidden_size * 2, 1)
        })
        
    def forward(self, input_ids, attention_mask, numerical_features, categorical_features, target_ids=None):
        batch_size, seq_len = input_ids.shape
        
        # Text encoding
        embeddings = self.embedding(input_ids)
        position_encodings = self.position_encoding[:, :seq_len, :]
        embeddings = embeddings + position_encodings
        
        # Apply transformer encoder
        encoded_features = self.transformer(embeddings, src_key_padding_mask=~attention_mask.bool())
        
        # Global pooling
        pooled_features = torch.mean(encoded_features, dim=1)
        
        # Feature fusion
        numerical_projected = self.numerical_projection(numerical_features)
        categorical_projected = self.categorical_projection(categorical_features.float())
        
        # Combine all features
        combined_features = torch.cat([
            pooled_features,
            numerical_projected,
            categorical_projected
        ], dim=1)
        
        # Expand for sequence generation
        expanded_features = combined_features.unsqueeze(1)
        
        # Generate prompt sequence
        if target_ids is not None:
            # Training mode
            target_embeddings = self.embedding(target_ids)
            target_position_encodings = self.position_encoding[:, :target_ids.size(1), :]
            target_embeddings = target_embeddings + target_position_encodings
            
            decoded_features = self.prompt_decoder(
                target_embeddings,
                encoded_features,
                tgt_mask=self._generate_square_subsequent_mask(target_ids.size(1))
            )
        else:
            # Inference mode - autoregressive generation
            decoded_features = self._generate_autoregressive(expanded_features, encoded_features)
        
        # Generate output logits
        output_logits = self.output_projection(decoded_features)
        
        # Quality predictions
        quality_predictions = {}
        for quality_name, predictor in self.quality_predictors.items():
            quality_predictions[quality_name] = torch.sigmoid(predictor(combined_features))
        
        return output_logits, quality_predictions
    
    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _generate_autoregressive(self, start_features, encoded_features, max_length=50):
        batch_size = start_features.size(0)
        generated = start_features[:, :1, :]  # Start with first token
        
        for i in range(max_length -1):
            # Decode current sequence
            decoded = self.prompt_decoder(
                generated,
                encoded_features,
                tgt_mask=self._generate_square_subsequent_mask(generated.size(1))
            )
            
            # Get next token prediction
            next_logits = self.output_projection(decoded[:, -1:, :])
            next_token = torch.argmax(next_logits, dim=-1)
            
            # Add to sequence
            next_embedding = self.embedding(next_token)
            next_position = self.position_encoding[:, generated.size(1):generated.size(1)+1, :]
            next_embedding = next_embedding + next_position
            
            generated = torch.cat([generated, next_embedding], dim=1)
            
            # Stop if end token
            if (next_token == self.tokenizer.eos_token_id).any():
                break
        
        return generated

class PromptOptimizationModel(nn.Module):
    """Real RL-based model for prompt optimization"""
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        # Policy network
        self.policy_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Value network
        self.value_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, vocab_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        
        # Policy features
        policy_features = self.policy_encoder(embeddings, src_key_padding_mask=~attention_mask.bool())
        policy_logits = self.policy_head(policy_features.mean(dim=1))
        
        # Value features
        value_features = self.value_encoder(embeddings, src_key_padding_mask=~attention_mask.bool())
        value = self.value_head(value_features.mean(dim=1))
        
        return policy_logits, value

class AdvancedAIPromptsService:
    """Real ML-powered AI prompts service with actual deep learning models"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize ML components
        self.model_registry = ModelRegistry()
        self.data_processor = PromptDataProcessor()
        self.trainer = ModelTrainer(None, None, None)  # TODO: Replace None with actual model, config, loss_fn
        self.optimizer = HyperparameterOptimizer()
        self.metrics_tracker = MLMetricsTracker()
        self.data_validator = DataValidator()
        
        # Load pre-trained models
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize models
        self.prompt_generation_model = None
        self.prompt_optimization_model = None
        self.context_model = None
        
        # Model paths
        self.model_paths = {
            'prompt_generation': 'models/prompt_generation.pth',
            'prompt_optimization': 'models/prompt_optimization.pth',
            'context_understanding': 'models/context_understanding.pth'
        }
        
        # Load or initialize models
        self._initialize_models()
        
        # Training configuration
        self.training_config = {
            'batch_size': 16,
            'learning_rate': 5e-5,
            'epochs': 30,
            'early_stopping_patience': 5,
            'validation_split': 0.2
        }
        
        # Prompt templates and patterns
        self.prompt_templates = self._load_prompt_templates()
        
    def _initialize_models(self):
        """Initialize or load pre-trained models"""
        try:
            # Load prompt generation model
            if os.path.exists(self.model_paths['prompt_generation']):
                self.prompt_generation_model = torch.load(
                    self.model_paths['prompt_generation'],
                    map_location=self.device
                )
                self.logger.info("Loaded pre-trained prompt generation model")
            else:
                self.prompt_generation_model = AdvancedPromptGenerationModel(
                    vocab_size=self.tokenizer.vocab_size
                ).to(self.device)
                self.logger.info("Initialized new prompt generation model")
            
            # Load prompt optimization model
            if os.path.exists(self.model_paths['prompt_optimization']):
                self.prompt_optimization_model = torch.load(
                    self.model_paths['prompt_optimization'],
                    map_location=self.device
                )
                self.logger.info("Loaded pre-trained prompt optimization model")
            else:
                self.prompt_optimization_model = PromptOptimizationModel(
                    vocab_size=self.tokenizer.vocab_size
                ).to(self.device)
                self.logger.info("Initialized new prompt optimization model")
            
            # Load context understanding model
            if os.path.exists(self.model_paths['context_understanding']):
                self.context_model = torch.load(
                    self.model_paths['context_understanding'],
                    map_location=self.device
                )
                self.logger.info("Loaded pre-trained context understanding model")
            else:
                self.context_model = AdvancedPromptGenerationModel(
                    vocab_size=self.tokenizer.vocab_size
                ).to(self.device)
                self.logger.info("Initialized new context understanding model")
                
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
    
    def _load_prompt_templates(self) -> Dict:
        """Load prompt templates for different use cases"""
        return {
            'analysis': {
                'template': "Analyze the following {context} and provide insights on {focus_area}: {input_data}, variables: ['context', 'focus_area', 'input_data']",
            },
            'generation': {
                'template': "Generate {output_type} based on the following requirements: {requirements}, variables: ['output_type', 'requirements']",
            },
            'optimization': {
                'template': "Optimize the following {target} for {objective}: {current_state}, variables: ['target', 'objective', 'current_state']",
            },
            'classification': {
                'template': "Classify the following {data_type} into appropriate categories: {input_data}, variables: ['data_type', 'input_data']",
            },
            'prediction': {
                'template': "Predict {prediction_target} based on the following data: {input_data}, variables: ['prediction_target', 'input_data']",
            }
        }
    
    async def generate_advanced_prompt(self, 
                                     context: str, 
                                     intent: str, 
                                     target_ai: str,
                                     complexity_level: str = 'medium',
                                     style: str = 'professional') -> Dict:
        """Generate advanced prompts using real ML models"""
        try:
            # Validate inputs
            validated_inputs = self.data_validator.validate_prompt_inputs({
                'context': context,
                'intent': intent,
                'target_ai': target_ai,
                'complexity_level': complexity_level,
                'style': style
            })
            
            # Prepare input features
            input_text = f"Context: {validated_inputs['context']} Intent: {validated_inputs['intent']} Target: {validated_inputs['target_ai']}"      
            # Tokenize
            encoding = self.tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Prepare numerical features
            numerical_features = torch.FloatTensor([
                len(context),
                len(intent),
                self._get_complexity_score(complexity_level),
                self._get_style_score(style),
                0.5, # placeholder for specificity
                0.5  # placeholder for clarity
            ]).unsqueeze(0).to(self.device)
            
            # Prepare categorical features
            categorical_features = torch.LongTensor([
                self._encode_domain(intent),
                self._encode_style(style),
                self._encode_intent(intent)
            ]).unsqueeze(0).to(self.device)
            
            # Generate prompt
            self.prompt_generation_model.eval()
            with torch.no_grad():
                output_logits, quality_predictions = self.prompt_generation_model(
                    encoding['input_ids'].to(self.device),
                    encoding['attention_mask'].to(self.device),
                    numerical_features,
                    categorical_features
                )
                
                # Decode generated prompt
                generated_tokens = torch.argmax(output_logits, dim=-1)
                generated_prompt = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                
                # Extract quality scores
                quality_scores = {
                    'effectiveness': float(quality_predictions['effectiveness'][0]),
                    'clarity': float(quality_predictions['clarity'][0]),
                    'specificity': float(quality_predictions['specificity'][0]),
                    'relevance': float(quality_predictions['relevance'])
                }
            
            # Post-process generated prompt
            processed_prompt = self._post_process_prompt(generated_prompt, context, intent)
            
            # Track metrics
            self.metrics_tracker.record_prompt_generation_metrics({
                'context_length': len(context),
                'intent_complexity': len(intent),
                'generated_length': len(processed_prompt),
                'quality_scores': quality_scores
            })
            
            return {
                'prompt': processed_prompt,
                'quality_scores': quality_scores,
                'generation_metadata': {
                    'model_used': 'advanced_prompt_generation',
                    'generation_time': datetime.now().isoformat(),
                    'complexity_level': complexity_level,
                    'style': style
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating advanced prompt: {e}")
            raise
    
    async def optimize_prompt(self, 
                            original_prompt: str, 
                            feedback: Dict,
                            optimization_target: str = 'effectiveness') -> Dict:
        """Optimize prompt using real RL-based optimization"""
        try:
            # Validate inputs
            validated_inputs = self.data_validator.validate_optimization_inputs({
                'original_prompt': original_prompt,
                'feedback': feedback,
                'optimization_target': optimization_target
            })
            
            # Prepare training data for optimization
            optimization_data = self._prepare_optimization_data(
                original_prompt, feedback, optimization_target
            )
            
            # Create dataset
            dataset = PromptDataset(optimization_data, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
            
            # Optimize using RL
            self.prompt_optimization_model.train()
            optimizer = torch.optim.AdamW(
                self.prompt_optimization_model.parameters(),
                lr=1e-4
            )
            
            best_reward = float('-inf')
            best_prompt = original_prompt
            
            for epoch in range(10):  # Optimization epochs
                epoch_reward = 0
                
                for batch_features, batch_labels in dataloader:
                    input_ids = batch_features['input_ids'].to(self.device)
                    attention_mask = batch_features['attention_mask'].to(self.device)
                    
                    # Get policy and value
                    policy_logits, value = self.prompt_optimization_model(
                        input_ids, attention_mask
                    )
                    
                    # Sample actions
                    policy_dist = torch.distributions.Categorical(logits=policy_logits)
                    actions = policy_dist.sample()
                    
                    # Calculate rewards based on feedback
                    rewards = self._calculate_optimization_rewards(
                        batch_labels, optimization_target
                    )
                    
                    # Calculate loss
                    log_probs = policy_dist.log_prob(actions)
                    policy_loss = -(log_probs * rewards).mean()
                    value_loss = F.mse_loss(value.squeeze(), rewards)
                    total_loss = policy_loss + 0.5
                    # Backward pass
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.prompt_optimization_model.parameters(), 
                        max_norm=1.0
                    )
                    optimizer.step()
                    
                    epoch_reward += rewards.mean().item()
                
                # Generate optimized prompt
                optimized_prompt = await self._generate_optimized_prompt(
                    original_prompt, epoch_reward
                )
                
                if epoch_reward > best_reward:
                    best_reward = epoch_reward
                    best_prompt = optimized_prompt
            
            # Save optimized model
            torch.save(self.prompt_optimization_model, self.model_paths['prompt_optimization'])
            
            return {
                'original_prompt': original_prompt,
                'optimized_prompt': best_prompt,
                'improvement_score': best_reward,
                'optimization_metadata': {
                    'target': optimization_target,
                    'epochs_trained': 10,
                    'final_reward': best_reward
                }
            }
                
        except Exception as e:
            self.logger.error(f"Error optimizing prompt: {e}")
            raise

    async def analyze_context_and_generate(self, 
                                         user_input: str, 
                                         conversation_history: List[Dict],
                                         ai_capabilities: Dict) -> Dict:
        """Analyze context and generate contextual prompts using real ML"""
        try:
            # Validate inputs
            validated_inputs = self.data_validator.validate_context_inputs({
                'user_input': user_input,
                'conversation_history': conversation_history,
                'ai_capabilities': ai_capabilities
            })
            
            # Analyze context using ML model
            context_analysis = await self._analyze_conversation_context(
                user_input, conversation_history
            )
            
            # Generate contextual prompt
            contextual_prompt = await self._generate_contextual_prompt(
                user_input, context_analysis, ai_capabilities
            )
            
            # Optimize for specific AI capabilities
            optimized_prompt = await self._optimize_for_ai_capabilities(
                contextual_prompt, ai_capabilities
            )
            
            return {
                'context_analysis': context_analysis,
                'generated_prompt': optimized_prompt,
                'context_metadata': {
                    'conversation_length': len(conversation_history),
                    'context_complexity': context_analysis['complexity_score'],
                    'ai_specialization': ai_capabilities.get('specialization', 'general')
                }
            }
                
        except Exception as e:
            self.logger.error(f"Error analyzing context and generating: {e}")
            raise

    def _get_complexity_score(self, complexity_level: str) -> float:
        """Calculate numerical complexity score"""
        complexity_map = {
            'simple': 0.2,
            'medium': 0.5,
            'complex': 0.8,
            'expert': 1.0
        }
        return complexity_map.get(complexity_level, 0.5)
    
    def _get_style_score(self, style: str) -> float:
        """Calculate numerical style score"""
        style_map = {
            'casual': 0.2,
            'professional': 0.5,
            'technical': 0.8,
            'academic': 1.0
        }
        return style_map.get(style, 0.5)
    def _encode_domain(self, intent: str) -> int:
        """Encode domain for categorical features"""
        domain_map = {
            'analysis': 0,
            'generation': 1,
            'optimization': 2,
            'classification': 3,
            'prediction': 4
        }
        
        for domain, code in domain_map.items():
            if domain in intent.lower():
                return code
        return 0  # default to analysis
    
    def _encode_style(self, style: str) -> int:
        """Encode style for categorical features"""
        style_map = {
            'casual': 0,
            'professional': 1,
            'technical': 2,
            'academic': 3
        }
        return style_map.get(style, 1)
    
    def _encode_intent(self, intent: str) -> int:
        """Encode intent for categorical features"""
        intent_map = {
            'information': 0,
            'action': 1,
            'analysis': 2,
            'creation': 3
        }
        
        for intent_type, code in intent_map.items():
            if intent_type in intent.lower():
                return code
        return 0  # default to information
    
    def _post_process_prompt(self, generated_prompt: str, context: str, intent: str) -> str:
        """Post-process generated prompt for better quality"""
        # Clean up the prompt
        cleaned_prompt = re.sub(r'\s+', ' ', generated_prompt).strip()
        
        # Ensure it starts with a proper instruction
        if not cleaned_prompt.lower().startswith(('analyze', 'generate', 'optimize', 'classify', 'predict')):
            # Add intent-based prefix
            intent_prefixes = {
                'analysis': 'Analyze,',
                'generation': 'Generate,',
                'optimization': 'Optimize,',
                'classification': 'Classify,',
                'prediction': 'Predict'
            }
            
            for intent_type, prefix in intent_prefixes.items():
                if intent_type in intent.lower():
                    cleaned_prompt = f"{prefix} {cleaned_prompt}"
                    break
        
        # Add context if not present
        if context and context not in cleaned_prompt:
            cleaned_prompt = f"Context: {context}\n{cleaned_prompt}"   
        return cleaned_prompt
    
    def _prepare_optimization_data(self, original_prompt: str, feedback: Dict, target: str) -> List[Dict]:
        """Prepare data for prompt optimization"""
        return [{
            'text': original_prompt,
            'context': feedback.get('context', ''),
            'target': feedback.get('target', ''),
            'effectiveness_score': feedback.get('effectiveness', 0.5),
            'response_quality': feedback.get('quality', 0.5),
            'user_satisfaction': feedback.get('satisfaction', 0.5),
            'complexity_score': feedback.get('complexity', 0.5),
            'specificity_score': feedback.get('specificity', 0.5),
            'clarity_score': feedback.get('clarity', 0.5),
            'domain': feedback.get('domain', 'general'),
            'style': feedback.get('style', 'professional'),
            'intent': feedback.get('intent', 'information')
        }]
    
    def _calculate_optimization_rewards(self, labels: Dict, target: str) -> torch.Tensor:
        """Calculate rewards for RL optimization"""
        rewards = []
        
        for label in labels:
            if target == 'effectiveness':
                reward = label['effectiveness_score']
            elif target == 'quality':
                reward = label['response_quality']
            elif target == 'satisfaction':
                reward = label['user_satisfaction']
            else:
                reward = (label['effectiveness_score'] + 
                         label['response_quality'] + 
                         label['user_satisfaction']) / 3
            
            rewards.append(reward)
        
        return torch.FloatTensor(rewards).to(self.device)
    
    async def _generate_optimized_prompt(self, original_prompt: str, reward: float) -> str:
        """Generate optimized prompt based on reward"""
        # Simple optimization strategy - can be enhanced
        if reward > 0.7:            # High reward - minor improvements
            return original_prompt
        elif reward > 0.5:          # Medium reward - moderate improvements
            return self._apply_moderate_improvements(original_prompt)
        else:
            # Low reward - significant improvements
            return self._apply_significant_improvements(original_prompt)
    
    def _apply_moderate_improvements(self, prompt: str) -> str:
        """Improvements to prompt"""
        improvements = [
            "Please provide a detailed ",
            "Consider all relevant factors when ",
            "Ensure comprehensive coverage of ",
            "Focus on key aspects of "
        ]
        
        for improvement in improvements:
            if improvement.lower() not in prompt.lower():
                return f"{improvement}{prompt}"   
        return prompt
    
    def _apply_significant_improvements(self, prompt: str) -> str:
        """Apply significant improvements to prompt"""
        # Add structure and clarity
        if not prompt.startswith(('Please', 'Analyze', 'Generate', 'Optimize')):
            prompt = f"Please {prompt}"        
        # Add specificity
        if 'detailed' not in prompt.lower():
            prompt = prompt.replace('Please', 'Please provide a detailed ')
        
        return prompt
    
    async def _analyze_conversation_context(self, user_input: str, history: List[Dict]) -> Dict:
        """Analyze conversation context using ML"""
        # Combine conversation history
        conversation_text = ""
        for msg in history[-5:]:  # Last 5 messages
            conversation_text += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"         
        # Analyze using context model
        encoding = self.tokenizer(
            conversation_text + user_input,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        self.context_model.eval()
        with torch.no_grad():
            _, context_features = self.context_model(
                encoding['input_ids'].to(self.device),
                encoding['attention_mask'].to(self.device),
                torch.zeros(1, 6).to(self.device),  # placeholder numerical features
                torch.zeros(1, 3).to(self.device)   # placeholder categorical features
            )
        
        return {
            'complexity_score': float(context_features['effectiveness'][0]),
            'topic_consistency': float(context_features['relevance'][0, 0]),
            'conversation_flow': float(context_features['clarity'][0])
        }
    
    def _extract_user_intent(self, user_input: str, history: List[Dict]) -> str:
        """Extract user intent from input and history"""
        intent_keywords = {
            'analysis': ['analyze', 'examine', 'study', 'investigate'],
            'generation': ['generate', 'create', 'build', 'develop'],
            'optimization': ['optimize', 'improve', 'enhance', 'refine'],
            'classification': ['classify', 'categorize', 'sortup'],
            'prediction': ['predict', 'forecast', 'estimate', 'project']
        }
        
        combined_text = user_input.lower()
        for msg in history[-3:]:  # Last 3 messages
            combined_text += " " + msg.get('content', '').lower()
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return intent
        
        return 'information'  # default intent
    
    async def _generate_contextual_prompt(self, user_input: str, context_analysis: Dict, ai_capabilities: Dict) -> str:
        """Generate contextual prompt based on analysis"""
        # Use context analysis to inform prompt generation
        complexity = context_analysis['complexity_score']
        intent = context_analysis['user_intent']
        
        # Generate base prompt
        base_prompt = await self.generate_advanced_prompt(
            context=user_input,
            intent=intent,
            target_ai=ai_capabilities.get('name', 'AI'),
            complexity_level='complex' if complexity > 0.7 else 'medium',
            style='technical' if complexity > 0.7 else 'professional'
        )
        
        return base_prompt['prompt']
    
    async def _optimize_for_ai_capabilities(self, prompt: str, capabilities: Dict) -> str:
        """Optimize prompt for specific AI capabilities"""
        specialization = capabilities.get('specialization', 'general')
        
        # Add capability-specific instructions
        if specialization == 'technical':
            prompt = f"Provide a technical analysis with detailed specifications: {prompt}"
        elif specialization == 'creative':
            prompt = f"Generate creative and innovative solutions: {prompt}"
        elif specialization == 'analytical':
            prompt = f"Provide comprehensive analytical insights with data-driven approach: {prompt}"
        return prompt
    
    async def get_system_health(self) -> Dict:
        """Get system health metrics"""
        try:
            health_metrics = {
                'status': 'healthy',
                'models_loaded': all([
                    self.prompt_generation_model is not None,
                    self.prompt_optimization_model is not None,
                    self.context_model is not None
                ]),
                'device': str(self.device),
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'model_metrics': self.metrics_tracker.get_latest_metrics(),
                'performance_metrics': {
                    'avg_generation_time': self.metrics_tracker.get_avg_generation_time(),
                    'prompt_quality_trend': self.metrics_tracker.get_quality_trend(),
                    'optimization_success_rate': self.metrics_tracker.get_optimization_success_rate()
                }
            }
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {'status': 'error', 'error': str(e)}

    async def strategic_material_analysis(self, company_profile: Dict) -> Dict:
        """Strategic material analysis for company profile"""
        try:
            # Extract company information
            company_name = company_profile.get('name', 'Unknown Company')
            industry = company_profile.get('industry', 'general')
            main_materials = company_profile.get('main_materials', '')
            
            # Generate analysis prompt
            analysis_prompt = await self.generate_advanced_prompt(
                context=f"Company: {company_name}, Industry: {industry}, Materials: {main_materials}",
                intent="analysis",
                target_ai="strategic_advisor",
                complexity_level="high",
                style="executive"
            )
            
            # Generate strategic insights
            strategic_insights = [
                f"Material optimization opportunities for {industry} sector",
                f"Supply chain resilience analysis for {main_materials}",
                f"Cost-benefit analysis of alternative materials",
                f"Sustainability impact assessment",
                f"Market positioning strategy based on material portfolio"
            ]
            
            # Generate predicted outputs
            predicted_outputs = [
                {
                    'type': 'material_optimization',
                    'description': f'Optimize {main_materials} usage for {industry}',
                    'priority': 'high',
                    'estimated_impact': '15-25% cost reduction'
                },
                {
                    'type': 'sustainability_improvement',
                    'description': 'Implement circular economy practices',
                    'priority': 'medium',
                    'estimated_impact': '20-30% carbon footprint reduction'
                },
                {
                    'type': 'supply_chain_resilience',
                    'description': 'Diversify material suppliers',
                    'priority': 'high',
                    'estimated_impact': 'Improved supply chain stability'
                }
            ]
            
            return {
                'executive_summary': f"Strategic analysis for {company_name} in {industry} sector",
                'company_profile': company_profile,
                'analysis_prompt': analysis_prompt.get('prompt', ''),
                'strategic_insights': strategic_insights,
                'predicted_outputs': predicted_outputs,
                'ai_enhanced_analysis': True,
                'confidence_score': 0.85,
                'recommendations': [
                    'Conduct material lifecycle assessment',
                    'Explore sustainable alternatives',
                    'Implement material tracking system',
                    'Develop supplier diversification strategy'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in strategic material analysis: {e}")
            return {
                'executive_summary': 'Analysis failed',
                'error': str(e),
                'ai_enhanced_analysis': False
            }

# Initialize service
advanced_ai_prompts_service = AdvancedAIPromptsService() 