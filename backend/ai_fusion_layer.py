"""
Production-Grade AI Fusion Layer
Combines outputs from all AI engines for optimal decision making
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import time
from pathlib import Path
import pickle
import hashlib
import uuid

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# AI component imports
from backend.gnn_reasoning_engine import GNNReasoningEngine
from backend.federated_meta_learning import FederatedMetaLearning
from backend.knowledge_graph import KnowledgeGraph
from revolutionary_ai_matching import RevolutionaryAIMatching

logger = logging.getLogger(__name__)

@dataclass
class EngineOutput:
    """Structured output from an AI engine"""
    engine_name: str
    confidence_score: float
    prediction: Any
    features: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    processing_time: float

@dataclass
class FusionResult:
    """Result of multi-engine fusion"""
    final_score: float
    confidence: float
    engine_contributions: Dict[str, float]
    explanation: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class FusionModel:
    """Fusion model configuration"""
    model_id: str
    model_type: str  # 'weighted_sum', 'ml_model', 'ensemble'
    weights: Dict[str, float]
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    last_updated: datetime
    version: str

class AIFusionLayer:
    """
    Production-Grade AI Fusion Layer
    Combines outputs from multiple AI engines for optimal decision making
    """
    
    def __init__(self, model_dir: str = "fusion_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize AI engines
        self.engines = self._initialize_engines()
        
        # Fusion models
        self.fusion_models = {}
        self.active_model = None
        
        # Performance tracking
        self.performance_history = []
        self.fusion_stats = {
            'total_fusions': 0,
            'successful_fusions': 0,
            'average_confidence': 0.0,
            'last_updated': datetime.now()
        }
        
        # Threading for concurrent operations
        self.lock = threading.Lock()
        
        # Load existing fusion models
        self._load_fusion_models()
        
        # Initialize default fusion model
        self._initialize_default_model()
        
        logger.info("AI Fusion Layer initialized")
    
    def _initialize_engines(self) -> Dict[str, Any]:
        """Initialize AI engines"""
        engines = {}
        
        try:
            engines['gnn'] = GNNReasoningEngine()
            logger.info("✅ GNN engine initialized for fusion")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GNN engine: {e}")
        
        try:
            engines['federated'] = FederatedMetaLearning()
            logger.info("✅ Federated learner initialized for fusion")
        except Exception as e:
            logger.error(f"❌ Failed to initialize federated learner: {e}")
        
        try:
            engines['knowledge_graph'] = KnowledgeGraph()
            logger.info("✅ Knowledge graph initialized for fusion")
        except Exception as e:
            logger.error(f"❌ Failed to initialize knowledge graph: {e}")
        
        try:
            engines['matching'] = RevolutionaryAIMatching()
            logger.info("✅ Matching engine initialized for fusion")
        except Exception as e:
            logger.error(f"❌ Failed to initialize matching engine: {e}")
        
        return engines
    
    def _load_fusion_models(self):
        """Load existing fusion models"""
        try:
            models_file = self.model_dir / "fusion_models.json"
            if models_file.exists():
                with open(models_file, 'r') as f:
                    data = json.load(f)
                    
                for model_data in data.get('models', []):
                    model = FusionModel(
                        model_id=model_data['model_id'],
                        model_type=model_data['model_type'],
                        weights=model_data['weights'],
                        parameters=model_data['parameters'],
                        performance_metrics=model_data['performance_metrics'],
                        last_updated=datetime.fromisoformat(model_data['last_updated']),
                        version=model_data['version']
                    )
                    self.fusion_models[model.model_id] = model
                
                # Set active model
                active_model_id = data.get('active_model_id')
                if active_model_id and active_model_id in self.fusion_models:
                    self.active_model = self.fusion_models[active_model_id]
                
                logger.info(f"Loaded {len(self.fusion_models)} fusion models")
                
        except Exception as e:
            logger.error(f"Error loading fusion models: {e}")
    
    def _save_fusion_models(self):
        """Save fusion models to file"""
        try:
            data = {
                'models': [asdict(model) for model in self.fusion_models.values()],
                'active_model_id': self.active_model.model_id if self.active_model else None,
                'last_updated': datetime.now().isoformat()
            }
            
            models_file = self.model_dir / "fusion_models.json"
            with open(models_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Fusion models saved")
            
        except Exception as e:
            logger.error(f"Error saving fusion models: {e}")
    
    def _initialize_default_model(self):
        """Initialize default fusion model"""
        try:
            default_model = FusionModel(
                model_id="default_weighted_sum",
                model_type="weighted_sum",
                weights={
                    'gnn': 0.3,
                    'federated': 0.25,
                    'knowledge_graph': 0.25,
                    'matching': 0.2
                },
                parameters={
                    'normalize_scores': True,
                    'confidence_threshold': 0.5
                },
                performance_metrics={
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                },
                last_updated=datetime.now(),
                version="1.0"
            )
            
            self.fusion_models[default_model.model_id] = default_model
            self.active_model = default_model
            
            self._save_fusion_models()
            
            logger.info("Default fusion model initialized")
            
        except Exception as e:
            logger.error(f"Error initializing default model: {e}")
    
    async def fuse_engine_outputs(self, engine_outputs: List[EngineOutput], 
                                context: Dict[str, Any] = None) -> FusionResult:
        """
        Fuse outputs from multiple AI engines
        """
        try:
            start_time = time.time()
            
            if not engine_outputs:
                raise ValueError("No engine outputs provided")
            
            # Validate engine outputs
            validated_outputs = self._validate_engine_outputs(engine_outputs)
            
            # Apply fusion model
            if self.active_model:
                fusion_result = self._apply_fusion_model(validated_outputs, context)
            else:
                # Fallback to simple averaging
                fusion_result = self._simple_fusion(validated_outputs)
            
            # Add metadata
            fusion_result.metadata.update({
                'fusion_model_id': self.active_model.model_id if self.active_model else 'fallback',
                'processing_time': time.time() - start_time,
                'engine_count': len(validated_outputs)
            })
            
            # Update statistics
            self._update_fusion_stats(fusion_result)
            
            logger.info(f"Fused {len(validated_outputs)} engine outputs in {fusion_result.metadata['processing_time']:.3f}s")
            
            return fusion_result
            
        except Exception as e:
            logger.error(f"Error in engine fusion: {e}")
            raise
    
    def _validate_engine_outputs(self, outputs: List[EngineOutput]) -> List[EngineOutput]:
        """Validate and filter engine outputs"""
        validated = []
        
        for output in outputs:
            try:
                # Check if engine is available
                if output.engine_name not in self.engines:
                    logger.warning(f"Engine {output.engine_name} not available, skipping")
                    continue
                
                # Validate confidence score
                if not (0.0 <= output.confidence_score <= 1.0):
                    logger.warning(f"Invalid confidence score for {output.engine_name}: {output.confidence_score}")
                    continue
                
                # Validate prediction
                if output.prediction is None:
                    logger.warning(f"Null prediction for {output.engine_name}")
                    continue
                
                validated.append(output)
                
            except Exception as e:
                logger.error(f"Error validating output from {output.engine_name}: {e}")
        
        return validated
    
    def _apply_fusion_model(self, outputs: List[EngineOutput], 
                           context: Dict[str, Any] = None) -> FusionResult:
        """Apply the active fusion model"""
        try:
            if self.active_model.model_type == "weighted_sum":
                return self._weighted_sum_fusion(outputs, context)
            elif self.active_model.model_type == "ml_model":
                return self._ml_model_fusion(outputs, context)
            elif self.active_model.model_type == "ensemble":
                return self._ensemble_fusion(outputs, context)
            else:
                logger.warning(f"Unknown fusion model type: {self.active_model.model_type}, using simple fusion")
                return self._simple_fusion(outputs)
                
        except Exception as e:
            logger.error(f"Error applying fusion model: {e}")
            return self._simple_fusion(outputs)
    
    def _weighted_sum_fusion(self, outputs: List[EngineOutput], 
                            context: Dict[str, Any] = None) -> FusionResult:
        """Weighted sum fusion"""
        try:
            weights = self.active_model.weights
            total_score = 0.0
            total_weight = 0.0
            engine_contributions = {}
            
            for output in outputs:
                weight = weights.get(output.engine_name, 0.1)  # Default weight
                contribution = output.confidence_score * weight
                
                total_score += contribution
                total_weight += weight
                engine_contributions[output.engine_name] = contribution
            
            # Normalize final score
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 0.0
            
            # Calculate overall confidence
            confidence = np.mean([output.confidence_score for output in outputs])
            
            # Generate explanation
            explanation = {
                'method': 'weighted_sum',
                'weights_used': weights,
                'engine_scores': {output.engine_name: output.confidence_score for output in outputs},
                'normalized_score': final_score
            }
            
            return FusionResult(
                final_score=final_score,
                confidence=confidence,
                engine_contributions=engine_contributions,
                explanation=explanation,
                metadata={},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in weighted sum fusion: {e}")
            raise
    
    def _ml_model_fusion(self, outputs: List[EngineOutput], 
                        context: Dict[str, Any] = None) -> FusionResult:
        """Machine learning model fusion"""
        try:
            # Prepare features for ML model
            features = self._extract_ml_features(outputs, context)
            
            # Load ML model
            model_file = self.model_dir / f"{self.active_model.model_id}.joblib"
            if model_file.exists():
                ml_model = joblib.load(model_file)
                
                # Make prediction
                prediction = ml_model.predict([features])[0]
                
                # Calculate confidence (using model's predict_proba if available)
                if hasattr(ml_model, 'predict_proba'):
                    confidence = np.max(ml_model.predict_proba([features])[0])
                else:
                    confidence = np.mean([output.confidence_score for output in outputs])
                
            else:
                logger.warning(f"ML model file not found: {model_file}")
                return self._weighted_sum_fusion(outputs, context)
            
            # Calculate engine contributions
            engine_contributions = {
                output.engine_name: output.confidence_score 
                for output in outputs
            }
            
            # Generate explanation
            explanation = {
                'method': 'ml_model',
                'model_id': self.active_model.model_id,
                'features_used': list(features.keys()),
                'prediction': prediction
            }
            
            return FusionResult(
                final_score=prediction,
                confidence=confidence,
                engine_contributions=engine_contributions,
                explanation=explanation,
                metadata={},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in ML model fusion: {e}")
            return self._weighted_sum_fusion(outputs, context)
    
    def _ensemble_fusion(self, outputs: List[EngineOutput], 
                        context: Dict[str, Any] = None) -> FusionResult:
        """Ensemble fusion using multiple methods"""
        try:
            # Apply multiple fusion methods
            weighted_result = self._weighted_sum_fusion(outputs, context)
            ml_result = self._ml_model_fusion(outputs, context)
            
            # Combine results
            final_score = (weighted_result.final_score + ml_result.final_score) / 2
            confidence = (weighted_result.confidence + ml_result.confidence) / 2
            
            # Combine engine contributions
            engine_contributions = {}
            for engine_name in set(weighted_result.engine_contributions.keys()) | set(ml_result.engine_contributions.keys()):
                weighted_contrib = weighted_result.engine_contributions.get(engine_name, 0)
                ml_contrib = ml_result.engine_contributions.get(engine_name, 0)
                engine_contributions[engine_name] = (weighted_contrib + ml_contrib) / 2
            
            # Generate explanation
            explanation = {
                'method': 'ensemble',
                'weighted_sum_score': weighted_result.final_score,
                'ml_model_score': ml_result.final_score,
                'ensemble_score': final_score
            }
            
            return FusionResult(
                final_score=final_score,
                confidence=confidence,
                engine_contributions=engine_contributions,
                explanation=explanation,
                metadata={},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble fusion: {e}")
            return self._weighted_sum_fusion(outputs, context)
    
    def _simple_fusion(self, outputs: List[EngineOutput]) -> FusionResult:
        """Simple averaging fusion (fallback)"""
        try:
            scores = [output.confidence_score for output in outputs]
            final_score = np.mean(scores)
            confidence = np.std(scores)  # Lower std = higher confidence
            
            engine_contributions = {
                output.engine_name: output.confidence_score 
                for output in outputs
            }
            
            explanation = {
                'method': 'simple_average',
                'scores': scores,
                'mean_score': final_score
            }
            
            return FusionResult(
                final_score=final_score,
                confidence=confidence,
                engine_contributions=engine_contributions,
                explanation=explanation,
                metadata={},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in simple fusion: {e}")
            raise
    
    def _extract_ml_features(self, outputs: List[EngineOutput], 
                           context: Dict[str, Any] = None) -> Dict[str, float]:
        """Extract features for ML model"""
        features = {}
        
        try:
            # Engine-specific features
            for output in outputs:
                engine_name = output.engine_name
                features[f"{engine_name}_confidence"] = output.confidence_score
                features[f"{engine_name}_processing_time"] = output.processing_time
                
                # Extract additional features from output
                if hasattr(output, 'features') and output.features:
                    for key, value in output.features.items():
                        if isinstance(value, (int, float)):
                            features[f"{engine_name}_{key}"] = value
            
            # Context features
            if context:
                for key, value in context.items():
                    if isinstance(value, (int, float)):
                        features[f"context_{key}"] = value
            
            # Statistical features
            confidences = [output.confidence_score for output in outputs]
            features['mean_confidence'] = np.mean(confidences)
            features['std_confidence'] = np.std(confidences)
            features['max_confidence'] = np.max(confidences)
            features['min_confidence'] = np.min(confidences)
            features['engine_count'] = len(outputs)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting ML features: {e}")
            return {}
    
    def _update_fusion_stats(self, result: FusionResult):
        """Update fusion statistics"""
        try:
            self.fusion_stats['total_fusions'] += 1
            self.fusion_stats['successful_fusions'] += 1
            
            # Update average confidence
            current_avg = self.fusion_stats['average_confidence']
            total_fusions = self.fusion_stats['total_fusions']
            self.fusion_stats['average_confidence'] = (
                (current_avg * (total_fusions - 1) + result.confidence) / total_fusions
            )
            
            self.fusion_stats['last_updated'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating fusion stats: {e}")
    
    async def train_fusion_model(self, training_data: List[Tuple[List[EngineOutput], float]], 
                                model_type: str = "ml_model") -> str:
        """Train a new fusion model"""
        try:
            model_id = f"fusion_model_{uuid.uuid4().hex[:8]}"
            
            if model_type == "ml_model":
                success = await self._train_ml_fusion_model(model_id, training_data)
            else:
                success = await self._train_weighted_fusion_model(model_id, training_data)
            
            if success:
                logger.info(f"Trained new fusion model: {model_id}")
                return model_id
            else:
                raise Exception("Failed to train fusion model")
                
        except Exception as e:
            logger.error(f"Error training fusion model: {e}")
            raise
    
    async def _train_ml_fusion_model(self, model_id: str, 
                                   training_data: List[Tuple[List[EngineOutput], float]]) -> bool:
        """Train ML-based fusion model"""
        try:
            # Prepare training data
            X = []
            y = []
            
            for outputs, target in training_data:
                features = self._extract_ml_features(outputs)
                if features:
                    X.append(list(features.values()))
                    y.append(target)
            
            if len(X) < 10:
                logger.warning("Insufficient training data for ML model")
                return False
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Save model
            model_file = self.model_dir / f"{model_id}.joblib"
            joblib.dump(model, model_file)
            
            # Create fusion model record
            fusion_model = FusionModel(
                model_id=model_id,
                model_type="ml_model",
                weights={},
                parameters={
                    'feature_names': list(features.keys()),
                    'training_samples': len(X)
                },
                performance_metrics={
                    'r2_score': r2_score(y, model.predict(X)),
                    'mse': mean_squared_error(y, model.predict(X))
                },
                last_updated=datetime.now(),
                version="1.0"
            )
            
            self.fusion_models[model_id] = fusion_model
            self._save_fusion_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training ML fusion model: {e}")
            return False
    
    async def _train_weighted_fusion_model(self, model_id: str, 
                                         training_data: List[Tuple[List[EngineOutput], float]]) -> bool:
        """Train weighted fusion model"""
        try:
            # Calculate optimal weights using simple optimization
            engine_weights = {}
            
            for outputs, target in training_data:
                for output in outputs:
                    engine_name = output.engine_name
                    if engine_name not in engine_weights:
                        engine_weights[engine_name] = []
                    engine_weights[engine_name].append(output.confidence_score)
            
            # Calculate average weights
            final_weights = {}
            for engine_name, scores in engine_weights.items():
                final_weights[engine_name] = np.mean(scores)
            
            # Normalize weights
            total_weight = sum(final_weights.values())
            if total_weight > 0:
                final_weights = {k: v/total_weight for k, v in final_weights.items()}
            
            # Create fusion model record
            fusion_model = FusionModel(
                model_id=model_id,
                model_type="weighted_sum",
                weights=final_weights,
                parameters={
                    'normalize_scores': True,
                    'confidence_threshold': 0.5
                },
                performance_metrics={
                    'training_samples': len(training_data)
                },
                last_updated=datetime.now(),
                version="1.0"
            )
            
            self.fusion_models[model_id] = fusion_model
            self._save_fusion_models()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training weighted fusion model: {e}")
            return False
    
    def set_active_model(self, model_id: str) -> bool:
        """Set active fusion model"""
        try:
            if model_id in self.fusion_models:
                self.active_model = self.fusion_models[model_id]
                self._save_fusion_models()
                logger.info(f"Set active fusion model: {model_id}")
                return True
            else:
                logger.error(f"Fusion model {model_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error setting active model: {e}")
            return False
    
    def get_fusion_models(self) -> List[Dict[str, Any]]:
        """Get list of available fusion models"""
        try:
            models = []
            for model in self.fusion_models.values():
                model_data = asdict(model)
                model_data['is_active'] = (self.active_model and self.active_model.model_id == model.model_id)
                models.append(model_data)
            
            return models
            
        except Exception as e:
            logger.error(f"Error getting fusion models: {e}")
            return []
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion statistics"""
        try:
            stats = self.fusion_stats.copy()
            stats['active_model'] = self.active_model.model_id if self.active_model else None
            stats['total_models'] = len(self.fusion_models)
            return stats
            
        except Exception as e:
            logger.error(f"Error getting fusion stats: {e}")
            return {}

# Global fusion layer instance
fusion_layer = AIFusionLayer() 