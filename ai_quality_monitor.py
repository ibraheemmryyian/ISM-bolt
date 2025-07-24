import numpy as np
from typing import List, Dict, Any
import logging
from dataclasses import dataclass
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class QualityMetrics:
    """Quality metrics for AI output"""
    quality_score: float
    data_consistency: float
    ai_performance: float
    anomaly_score: float
    confidence_score: float
    model_drift: float
    timestamp: str

class AIQualityMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔍 Initializing AI Quality Monitor")
        
        # Initialize quality checking models
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/quality-classifier")
        self.quality_model = AutoModelForSequenceClassification.from_pretrained("deepseek-ai/quality-classifier")
        
        # Quality thresholds
        self.thresholds = {
            'quality_score': 0.7,
            'data_consistency': 0.8,
            'ai_performance': 0.75,
            'anomaly_threshold': 0.9
        }
        
    async def analyze_listings(self, listings: List[Any]) -> List[Dict[str, Any]]:
        """Analyze quality of AI-generated listings"""
        quality_reports = []
        
        for listing in listings:
            # Check data completeness
            completeness = self._check_data_completeness(listing)
            
            # Check data consistency
            consistency = self._check_data_consistency(listing)
            
            # Check AI performance
            ai_performance = self._evaluate_ai_performance(listing)
            
            # Check for anomalies
            anomaly_score = self._detect_anomalies(listing)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(listing)
            
            # Check for model drift
            drift = self._check_model_drift(listing)
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(
                completeness, consistency, ai_performance, anomaly_score, confidence
            )
            
            quality_reports.append({
                'listing_id': getattr(listing, 'id', 'unknown'),
                'material_name': getattr(listing, 'material_name', 'unknown'),
                'quality_score': quality_score,
                'data_consistency': consistency,
                'data_completeness': completeness,
                'ai_performance': ai_performance,
                'anomaly_score': anomaly_score,
                'confidence_score': confidence,
                'model_drift': drift,
                'timestamp': datetime.now().isoformat()
            })
            
        return quality_reports
    
    async def analyze_matches(self, matches: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze quality of AI-generated matches"""
        quality_reports = []
        
        for match in matches:
            # Check match quality
            match_quality = self._evaluate_match_quality(match)
            
            # Check prediction confidence
            confidence = self._evaluate_match_confidence(match)
            
            # Check for matching anomalies
            anomalies = self._detect_match_anomalies(match)
            
            quality_reports.append({
                'match_id': match.get('id', 'unknown'),
                'quality_score': match_quality,
                'confidence_score': confidence,
                'anomaly_score': anomalies,
                'timestamp': datetime.now().isoformat()
            })
            
        return quality_reports
    
    def _check_data_completeness(self, listing: Any) -> float:
        """Check if all required fields are present and valid"""
        required_fields = [
            'material_name', 'material_type', 'quantity', 'unit',
            'description', 'quality_grade', 'potential_value'
        ]
        
        present_fields = sum(1 for field in required_fields if hasattr(listing, field))
        return present_fields / len(required_fields)
    
    def _check_data_consistency(self, listing: Any) -> float:
        """Check if data is consistent across fields"""
        try:
            consistency_checks = [
                isinstance(listing.quantity, (int, float)) and listing.quantity > 0,
                isinstance(listing.potential_value, (int, float)) and listing.potential_value > 0,
                len(listing.description) > 10,
                listing.unit in ['kg', 'tons', 'units']
            ]
            return sum(consistency_checks) / len(consistency_checks)
        except:
            return 0.0
    
    def _evaluate_ai_performance(self, listing: Any) -> float:
        """Evaluate AI model performance"""
        try:
            # Check if AI features are present
            has_neural = hasattr(listing, 'neural_embedding') and listing.neural_embedding is not None
            has_quantum = hasattr(listing, 'quantum_vector') and listing.quantum_vector is not None
            has_knowledge = hasattr(listing, 'knowledge_graph_features') and listing.knowledge_graph_features is not None
            
            # Calculate feature quality scores
            scores = []
            
            if has_neural:
                scores.append(self._evaluate_neural_embedding(listing.neural_embedding))
            if has_quantum:
                scores.append(self._evaluate_quantum_vector(listing.quantum_vector))
            if has_knowledge:
                scores.append(self._evaluate_knowledge_features(listing.knowledge_graph_features))
                
            return np.mean(scores) if scores else 0.0
        except:
            return 0.0
    
    def _detect_anomalies(self, listing: Any) -> float:
        """Detect anomalies in the listing"""
        try:
            anomaly_scores = []
            
            # Check for price anomalies
            if hasattr(listing, 'potential_value') and listing.potential_value > 0:
                if listing.potential_value > 1000000:  # Suspicious high value
                    anomaly_scores.append(1.0)
                elif listing.potential_value < 1:  # Suspicious low value
                    anomaly_scores.append(1.0)
            
            # Check for quantity anomalies
            if hasattr(listing, 'quantity') and listing.quantity > 0:
                if listing.quantity > 10000:  # Suspicious high quantity
                    anomaly_scores.append(1.0)
                elif listing.quantity < 0.1:  # Suspicious low quantity
                    anomaly_scores.append(1.0)
            
            return max(anomaly_scores) if anomaly_scores else 0.0
        except:
            return 0.0
    
    def _calculate_confidence(self, listing: Any) -> float:
        """Calculate confidence score for the listing"""
        try:
            # Convert listing description to quality score using the model
            inputs = self.tokenizer(listing.description, return_tensors="pt", truncation=True)
            outputs = self.quality_model(**inputs)
            confidence = torch.softmax(outputs.logits, dim=1)[0][1].item()
            return confidence
        except:
            return 0.5
    
    def _check_model_drift(self, listing: Any) -> float:
        """Check for model drift"""
        try:
            # Compare current listing features with historical distributions
            drift_scores = []
            
            if hasattr(listing, 'neural_embedding'):
                drift_scores.append(self._calculate_feature_drift(listing.neural_embedding))
            
            if hasattr(listing, 'quantum_vector'):
                drift_scores.append(self._calculate_feature_drift(listing.quantum_vector))
                
            return np.mean(drift_scores) if drift_scores else 0.0
        except:
            return 0.0
    
    def _calculate_quality_score(self, completeness: float, consistency: float, 
                               ai_performance: float, anomaly_score: float, 
                               confidence: float) -> float:
        """Calculate overall quality score"""
        weights = {
            'completeness': 0.2,
            'consistency': 0.2,
            'ai_performance': 0.3,
            'anomaly': 0.15,
            'confidence': 0.15
        }
        
        score = (
            completeness * weights['completeness'] +
            consistency * weights['consistency'] +
            ai_performance * weights['ai_performance'] +
            (1 - anomaly_score) * weights['anomaly'] +
            confidence * weights['confidence']
        )
        
        return score
    
    def _evaluate_match_quality(self, match: Dict) -> float:
        """Evaluate quality of a match"""
        try:
            quality_factors = []
            
            # Check match score
            if 'match_score' in match:
                quality_factors.append(match['match_score'])
            
            # Check confidence level
            if 'confidence_level' in match:
                quality_factors.append(match['confidence_level'])
            
            # Check scoring breakdown
            if 'scoring_breakdown' in match:
                breakdown = match['scoring_breakdown']
                for score in breakdown.values():
                    if isinstance(score, dict) and 'value' in score:
                        quality_factors.append(score['value'])
            
            return np.mean(quality_factors) if quality_factors else 0.0
        except:
            return 0.0
    
    def _evaluate_match_confidence(self, match: Dict) -> float:
        """Evaluate confidence of a match"""
        try:
            if 'confidence_level' in match:
                return float(match['confidence_level'])
            return 0.5
        except:
            return 0.5
    
    def _detect_match_anomalies(self, match: Dict) -> float:
        """Detect anomalies in a match"""
        try:
            anomaly_scores = []
            
            # Check for suspiciously high match scores
            if match.get('match_score', 0) > 0.95:
                anomaly_scores.append(0.8)
            
            # Check for suspiciously low confidence
            if match.get('confidence_level', 1) < 0.2:
                anomaly_scores.append(0.9)
            
            return max(anomaly_scores) if anomaly_scores else 0.0
        except:
            return 0.0
    
    def _evaluate_neural_embedding(self, embedding: np.ndarray) -> float:
        """Evaluate quality of neural embedding"""
        try:
            # Check if embedding is normalized
            norm = np.linalg.norm(embedding)
            if abs(norm - 1.0) > 0.1:
                return 0.5
            
            # Check if embedding has reasonable values
            if np.any(np.abs(embedding) > 5):
                return 0.7
                
            return 0.9
        except:
            return 0.0
    
    def _evaluate_quantum_vector(self, vector: np.ndarray) -> float:
        """Evaluate quality of quantum vector"""
        try:
            # Check if vector has reasonable distribution
            if np.mean(np.abs(vector)) > 3:
                return 0.6
            
            return 0.9
        except:
            return 0.0
    
    def _evaluate_knowledge_features(self, features: Dict) -> float:
        """Evaluate quality of knowledge graph features"""
        try:
            # Check if essential features are present
            required_features = ['degree', 'centrality']
            present = sum(1 for f in required_features if f in features)
            return present / len(required_features)
        except:
            return 0.0
    
    def _calculate_feature_drift(self, features: np.ndarray) -> float:
        """Calculate feature drift from baseline"""
        try:
            # Compare with historical feature distributions
            # This is a placeholder - should be replaced with actual drift detection
            return np.random.uniform(0, 0.3)
        except:
            return 0.0
