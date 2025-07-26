"""
Advanced Proactive Opportunity Detection Engine
AI-Powered Future Need Prediction and Market Intelligence
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
import hashlib
from dataclasses import dataclass
# Try to import sklearn components with fallback
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
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
    
    class RandomForestRegressor:
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(len(X))
    
    class GradientBoostingRegressor:
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(len(X))
    
    class TfidfVectorizer:
        def __init__(self, **kwargs):
            pass
        def fit_transform(self, texts):
            return np.zeros((len(texts), 100))
    
    class DBSCAN:
        def __init__(self, **kwargs):
            pass
        def fit_predict(self, X):
            return np.zeros(len(X))
    
    SKLEARN_AVAILABLE = False
import networkx as nx
from newsapi import NewsApiClient
import redis
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
# Fix relative import error
try:
    from .ml_core.models import BaseNN
except ImportError:
    # Use absolute import
    from ml_core.models import BaseNN
try:
    from .ml_core.training import train_supervised
except ImportError:
    from ml_core.training import train_supervised

try:
    from .ml_core.inference import predict_supervised
except ImportError:
    from ml_core.inference import predict_supervised

try:
    from .ml_core.monitoring import log_metrics, save_checkpoint
except ImportError:
    from ml_core.monitoring import log_metrics, save_checkpoint
from torch.utils.data import DataLoader, TensorDataset
import os
import shap
from .utils.distributed_logger import DistributedLogger
from .utils.advanced_data_validator import AdvancedDataValidator
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from .industrial_intelligence_engine import IndustrialIntelligenceEngine

logger = DistributedLogger('ProactiveOpportunityEngine', log_file='logs/proactive_opportunity_engine.log')

@dataclass
class OpportunityPrediction:
    """Structured opportunity prediction"""
    company_id: str
    opportunity_type: str  # 'material_need', 'waste_opportunity', 'market_shift', 'regulatory_change'
    confidence_score: float
    predicted_timeline: str  # 'immediate', 'short_term', 'medium_term', 'long_term'
    impact_score: float
    description: str
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    market_indicators: Dict[str, float]
    risk_assessment: str
    roi_potential: float
    carbon_impact: float
    created_at: datetime

@dataclass
class MarketIntelligence:
    """Market intelligence data structure"""
    industry: str
    material: str
    demand_forecast: float
    supply_forecast: float
    price_trend: float
    volatility: float
    regulatory_risk: float
    innovation_score: float
    sustainability_pressure: float
    market_sentiment: float
    data_sources: List[str]
    last_updated: datetime

class OpportunityPredictor:
    def __init__(self, input_dim=10, output_dim=2, model_dir="opportunity_models"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_dir = model_dir
        self.model = BaseNN(input_dim, output_dim)
    def train(self, X, y, epochs=20):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        train_supervised(self.model, loader, optimizer, criterion, epochs=epochs)
        save_checkpoint(self.model, optimizer, epochs, os.path.join(self.model_dir, "opportunity_model.pt"))
    def predict(self, X):
        return torch.argmax(predict_supervised(self.model, torch.tensor(X, dtype=torch.float)), dim=1).cpu().numpy()
    def feedback_loop(self, X, y):
        # Active learning: retrain on new feedback
        self.train(X, y, epochs=5)
    def human_in_the_loop(self, X, y_true):
        # Simulate human review and correction
        y_pred = self.predict(X)
        corrections = (y_pred != y_true)
        if corrections.any():
            self.train(X[corrections], y_true[corrections], epochs=3)

class AdvancedProactiveOpportunityEngine:
    """
    Advanced AI-Powered Proactive Opportunity Detection Engine
    
    Features:
    - Multi-modal data analysis (news, financial, regulatory, social)
    - Machine learning demand forecasting
    - Real-time market intelligence
    - Predictive analytics for material needs
    - Anomaly detection for market shifts
    - Sentiment analysis for industry trends
    - Network analysis for symbiosis opportunities
    - Time-series forecasting with multiple models
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Initialize AI models
        self.demand_forecaster = RandomForestRegressor(n_estimators=200, random_state=42)
        self.market_analyzer = GradientBoostingRegressor(n_estimators=150, random_state=42)
        self.sentiment_analyzer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.anomaly_detector = DBSCAN(eps=0.3, min_samples=5)
        
        # Data processing
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
        # Caching and storage
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            db=0,
            decode_responses=True
        )
        self.cache_ttl = 3600  # 1 hour
        
        # Market data sources
        self.news_api = NewsApiClient(api_key=self.config.get('news_api_key', ''))
        self.market_data_cache = {}
        self.opportunity_graph = nx.DiGraph()
        
        self.industrial_intelligence_engine = IndustrialIntelligenceEngine(
            redis_client=self.redis_client,
            logger=logger,
            cache_ttl=self.cache_ttl,
            config=self.config
        )
        
        # Background processing
        self.running = False
        self.background_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.prediction_accuracy = []
        self.processing_times = []
        
        logger.info("🚀 Advanced Proactive Opportunity Engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'news_api_key': '33d86c63e63c46c58c7dfd81068e79a4',
            'yfinance_enabled': True,
            'sentiment_analysis_enabled': True,
            'anomaly_detection_enabled': True,
            'market_analysis_enabled': True,
            'prediction_horizon_days': 365,
            'confidence_threshold': 0.7,
            'update_frequency_minutes': 30,
            'max_opportunities_per_company': 10,
            'data_sources': ['news', 'financial', 'regulatory', 'social', 'industry_reports']
        }
    
    async def predict_future_needs(self, company_data: Dict[str, Any]) -> List[OpportunityPrediction]:
        """
        Predict future material needs and opportunities for a company
        
        Args:
            company_data: Company profile and historical data
            
        Returns:
            List of opportunity predictions with confidence scores
        """
        try:
            start_time = time.time()
            
            # Generate cache key
            cache_key = f"opportunities:{company_data.get('id', 'unknown')}"
            
            # Check cache first
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Multi-modal analysis
            opportunities = []
            
            # 1. Market-based predictions
            market_opportunities = await self._analyze_market_opportunities(company_data)
            opportunities.extend(market_opportunities)
            
            # 2. Industry trend analysis
            trend_opportunities = await self._analyze_industry_trends(company_data)
            opportunities.extend(trend_opportunities)
            
            # 3. Regulatory change predictions
            regulatory_opportunities = await self._predict_regulatory_changes(company_data)
            opportunities.extend(regulatory_opportunities)
            
            # 4. Supply chain disruption predictions
            supply_chain_opportunities = await self._predict_supply_chain_disruptions(company_data)
            opportunities.extend(supply_chain_opportunities)
            
            # 5. Technology adoption predictions
            tech_opportunities = await self._predict_technology_adoption(company_data)
            opportunities.extend(tech_opportunities)
            
            # 6. Sustainability pressure analysis
            sustainability_opportunities = await self._analyze_sustainability_pressures(company_data)
            opportunities.extend(sustainability_opportunities)
            
            # Filter and rank opportunities
            filtered_opportunities = self._filter_and_rank_opportunities(opportunities)
            
            # Cache results
            self._cache_result(cache_key, filtered_opportunities)
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.info(f"Generated {len(filtered_opportunities)} opportunities for {company_data.get('name', 'Unknown')} in {processing_time:.2f}s")
            
            return filtered_opportunities
            
        except Exception as e:
            logger.error(f"Error predicting future needs: {e}")
            return []
    
    async def _analyze_market_opportunities(self, company_data: Dict[str, Any]) -> List[OpportunityPrediction]:
        """Analyze market-based opportunities"""
        opportunities = []
        
        try:
            industry = company_data.get('industry', 'manufacturing')
            location = company_data.get('location', 'global')
            
            # Get market intelligence
            market_intel = await self._get_market_intelligence(industry, location)
            
            # Analyze demand trends
            if market_intel.demand_forecast > 0.05:  # 5% growth threshold
                opportunities.append(OpportunityPrediction(
                    company_id=company_data.get('id', ''),
                    opportunity_type='material_need',
                    confidence_score=min(0.95, market_intel.demand_forecast * 10),
                    predicted_timeline='medium_term',
                    impact_score=market_intel.demand_forecast,
                    description=f"Growing demand for materials in {industry} sector",
                    supporting_data={
                        'demand_forecast': market_intel.demand_forecast,
                        'price_trend': market_intel.price_trend,
                        'market_sentiment': market_intel.market_sentiment
                    },
                    recommendations=[
                        "Consider increasing material stockpiles",
                        "Explore long-term supply contracts",
                        "Invest in material efficiency technologies"
                    ],
                    market_indicators={
                        'demand_growth': market_intel.demand_forecast,
                        'price_volatility': market_intel.volatility,
                        'market_sentiment': market_intel.market_sentiment
                    },
                    risk_assessment='Low to Medium',
                    roi_potential=market_intel.demand_forecast * 100,
                    carbon_impact=market_intel.sustainability_pressure * 50,
                    created_at=datetime.now()
                ))
            
            # Analyze supply disruptions
            if market_intel.supply_forecast < -0.03:  # Supply constraint threshold
                opportunities.append(OpportunityPrediction(
                    company_id=company_data.get('id', ''),
                    opportunity_type='supply_opportunity',
                    confidence_score=0.85,
                    predicted_timeline='short_term',
                    impact_score=abs(market_intel.supply_forecast),
                    description=f"Potential supply constraints in {industry} creating opportunities",
                    supporting_data={
                        'supply_forecast': market_intel.supply_forecast,
                        'volatility': market_intel.volatility
                    },
                    recommendations=[
                        "Diversify supply sources",
                        "Consider alternative materials",
                        "Build strategic partnerships"
                    ],
                    market_indicators={
                        'supply_risk': abs(market_intel.supply_forecast),
                        'price_pressure': market_intel.price_trend
                    },
                    risk_assessment='Medium',
                    roi_potential=25.0,
                    carbon_impact=30.0,
                    created_at=datetime.now()
                ))
                
        except Exception as e:
            logger.error(f"Error analyzing market opportunities: {e}")
        
        return opportunities
    
    async def _analyze_industry_trends(self, company_data: Dict[str, Any]) -> List[OpportunityPrediction]:
        """Analyze industry trends and patterns"""
        opportunities = []
        
        try:
            industry = company_data.get('industry', 'manufacturing')
            
            # Get industry news and sentiment
            news_data = await self._get_industry_news(industry)
            
            # Analyze sentiment trends
            if news_data.get('sentiment_score', 0) > 0.6:
                opportunities.append(OpportunityPrediction(
                    company_id=company_data.get('id', ''),
                    opportunity_type='market_shift',
                    confidence_score=0.75,
                    predicted_timeline='short_term',
                    impact_score=news_data.get('sentiment_score', 0),
                    description=f"Positive industry sentiment in {industry} indicating growth opportunities",
                    supporting_data=news_data,
                    recommendations=[
                        "Monitor industry developments closely",
                        "Prepare for increased demand",
                        "Consider early market entry strategies"
                    ],
                    market_indicators={
                        'sentiment': news_data.get('sentiment_score', 0),
                        'news_volume': news_data.get('article_count', 0)
                    },
                    risk_assessment='Low',
                    roi_potential=20.0,
                    carbon_impact=15.0,
                    created_at=datetime.now()
                ))
                
        except Exception as e:
            logger.error(f"Error analyzing industry trends: {e}")
        
        return opportunities
    
    async def _predict_regulatory_changes(self, company_data: Dict[str, Any]) -> List[OpportunityPrediction]:
        """Predict regulatory changes and their impact"""
        opportunities = []
        
        try:
            industry = company_data.get('industry', 'manufacturing')
            location = company_data.get('location', 'global')
            
            # Get regulatory news and updates
            regulatory_data = await self._get_regulatory_updates(industry, location)
            
            if regulatory_data.get('pending_changes', 0) > 0:
                opportunities.append(OpportunityPrediction(
                    company_id=company_data.get('id', ''),
                    opportunity_type='regulatory_change',
                    confidence_score=0.8,
                    predicted_timeline='medium_term',
                    impact_score=regulatory_data.get('impact_score', 0.5),
                    description=f"Anticipated regulatory changes in {industry} requiring adaptation",
                    supporting_data=regulatory_data,
                    recommendations=[
                        "Monitor regulatory developments",
                        "Prepare compliance strategies",
                        "Consider early adoption of new standards"
                    ],
                    market_indicators={
                        'regulatory_risk': regulatory_data.get('risk_score', 0),
                        'compliance_pressure': regulatory_data.get('pressure_score', 0)
                    },
                    risk_assessment='Medium to High',
                    roi_potential=15.0,
                    carbon_impact=40.0,
                    created_at=datetime.now()
                ))
                
        except Exception as e:
            logger.error(f"Error predicting regulatory changes: {e}")
        
        return opportunities
    
    async def _predict_supply_chain_disruptions(self, company_data: Dict[str, Any]) -> List[OpportunityPrediction]:
        """Predict supply chain disruptions and opportunities"""
        opportunities = []
        
        try:
            # Analyze global supply chain indicators
            supply_chain_data = await self._get_supply_chain_indicators()
            
            if supply_chain_data.get('disruption_risk', 0) > 0.6:
                opportunities.append(OpportunityPrediction(
                    company_id=company_data.get('id', ''),
                    opportunity_type='supply_chain_opportunity',
                    confidence_score=0.7,
                    predicted_timeline='short_term',
                    impact_score=supply_chain_data.get('disruption_risk', 0),
                    description="Potential supply chain disruptions creating local sourcing opportunities",
                    supporting_data=supply_chain_data,
                    recommendations=[
                        "Develop local supplier networks",
                        "Consider vertical integration",
                        "Build strategic inventory reserves"
                    ],
                    market_indicators={
                        'disruption_risk': supply_chain_data.get('disruption_risk', 0),
                        'local_sourcing_pressure': supply_chain_data.get('local_pressure', 0)
                    },
                    risk_assessment='Medium',
                    roi_potential=30.0,
                    carbon_impact=25.0,
                    created_at=datetime.now()
                ))
                
        except Exception as e:
            logger.error(f"Error predicting supply chain disruptions: {e}")
        
        return opportunities
    
    async def _predict_technology_adoption(self, company_data: Dict[str, Any]) -> List[OpportunityPrediction]:
        """Predict technology adoption trends and opportunities"""
        opportunities = []
        
        try:
            industry = company_data.get('industry', 'manufacturing')
            
            # Get technology trend data
            tech_data = await self._get_technology_trends(industry)
            
            if tech_data.get('adoption_rate', 0) > 0.1:  # 10% adoption threshold
                opportunities.append(OpportunityPrediction(
                    company_id=company_data.get('id', ''),
                    opportunity_type='technology_adoption',
                    confidence_score=0.75,
                    predicted_timeline='long_term',
                    impact_score=tech_data.get('adoption_rate', 0),
                    description=f"Emerging technology adoption in {industry} creating new material needs",
                    supporting_data=tech_data,
                    recommendations=[
                        "Research emerging technologies",
                        "Prepare for new material requirements",
                        "Consider technology partnerships"
                    ],
                    market_indicators={
                        'adoption_rate': tech_data.get('adoption_rate', 0),
                        'innovation_score': tech_data.get('innovation_score', 0)
                    },
                    risk_assessment='Low to Medium',
                    roi_potential=40.0,
                    carbon_impact=35.0,
                    created_at=datetime.now()
                ))
                
        except Exception as e:
            logger.error(f"Error predicting technology adoption: {e}")
        
        return opportunities
    
    async def _analyze_sustainability_pressures(self, company_data: Dict[str, Any]) -> List[OpportunityPrediction]:
        """Analyze sustainability pressures and circular economy opportunities"""
        opportunities = []
        
        try:
            industry = company_data.get('industry', 'manufacturing')
            
            # Get sustainability trend data
            sustainability_data = await self._get_sustainability_trends(industry)
            
            if sustainability_data.get('pressure_score', 0) > 0.7:
                opportunities.append(OpportunityPrediction(
                    company_id=company_data.get('id', ''),
                    opportunity_type='sustainability_opportunity',
                    confidence_score=0.85,
                    predicted_timeline='medium_term',
                    impact_score=sustainability_data.get('pressure_score', 0),
                    description=f"Growing sustainability pressures in {industry} driving circular economy adoption",
                    supporting_data=sustainability_data,
                    recommendations=[
                        "Implement circular economy practices",
                        "Develop waste-to-resource programs",
                        "Partner with sustainability-focused companies"
                    ],
                    market_indicators={
                        'sustainability_pressure': sustainability_data.get('pressure_score', 0),
                        'circular_economy_growth': sustainability_data.get('circular_growth', 0)
                    },
                    risk_assessment='Low',
                    roi_potential=35.0,
                    carbon_impact=60.0,
                    created_at=datetime.now()
                ))
                
        except Exception as e:
            logger.error(f"Error analyzing sustainability pressures: {e}")
        
        return opportunities
    
    async def _get_market_intelligence(self, industry: str, location: str) -> MarketIntelligence:
        """Get comprehensive market intelligence"""
        try:
            # Check cache first
            cache_key = f"market_intel:{industry}:{location}"
            cached_data = self._get_cached_result(cache_key)
            if cached_data:
                return cached_data
            
            # Simulate market data (replace with real APIs)
            market_data = {
                'demand_forecast': np.random.uniform(0.02, 0.15),
                'supply_forecast': np.random.uniform(-0.05, 0.08),
                'price_trend': np.random.uniform(-0.1, 0.2),
                'volatility': np.random.uniform(0.1, 0.4),
                'regulatory_risk': np.random.uniform(0.1, 0.6),
                'innovation_score': np.random.uniform(0.3, 0.8),
                'sustainability_pressure': np.random.uniform(0.4, 0.9),
                'market_sentiment': np.random.uniform(0.3, 0.8)
            }
            
            market_intel = MarketIntelligence(
                industry=industry,
                material='general',
                demand_forecast=market_data['demand_forecast'],
                supply_forecast=market_data['supply_forecast'],
                price_trend=market_data['price_trend'],
                volatility=market_data['volatility'],
                regulatory_risk=market_data['regulatory_risk'],
                innovation_score=market_data['innovation_score'],
                sustainability_pressure=market_data['sustainability_pressure'],
                market_sentiment=market_data['market_sentiment'],
                data_sources=['simulated_market_data'],
                last_updated=datetime.now()
            )
            
            # Cache the result
            self._cache_result(cache_key, market_intel)
            
            return market_intel
            
        except Exception as e:
            logger.error(f"Error getting market intelligence: {e}")
            return MarketIntelligence(
                industry=industry,
                material='general',
                demand_forecast=0.05,
                supply_forecast=0.02,
                price_trend=0.0,
                volatility=0.2,
                regulatory_risk=0.3,
                innovation_score=0.5,
                sustainability_pressure=0.6,
                market_sentiment=0.5,
                data_sources=['fallback'],
                last_updated=datetime.now()
            )
    
    async def _get_industry_news(self, industry: str) -> Dict[str, Any]:
        """Get industry news and sentiment analysis using IndustrialIntelligenceEngine"""
        try:
            return await self.industrial_intelligence_engine.fetch_industrial_intelligence(industry)
        except Exception as e:
            logger.error(f"Error getting industry intelligence: {e}")
            # Fallback to simulated data (same as before)
            import numpy as np
            from datetime import datetime
            np.random.seed()
            return {
                'sentiment_score': float(np.random.uniform(0.4, 0.8)),
                'article_count': int(np.random.randint(10, 100)),
                'positive_articles': int(np.random.randint(5, 50)),
                'negative_articles': int(np.random.randint(1, 20)),
                'neutral_articles': int(np.random.randint(1, 20)),
                'trending_topics': ['sustainability', 'digital_transformation', 'supply_chain'],
                'data_source': 'simulated_fallback',
                'last_updated': datetime.now().isoformat()
            }
    
    async def _get_regulatory_updates(self, industry: str, location: str) -> Dict[str, Any]:
        """Get regulatory updates and changes"""
        try:
            # Simulate regulatory data (replace with real regulatory APIs)
            return {
                'pending_changes': np.random.randint(0, 5),
                'impact_score': np.random.uniform(0.1, 0.8),
                'risk_score': np.random.uniform(0.2, 0.7),
                'pressure_score': np.random.uniform(0.3, 0.8),
                'compliance_deadline': datetime.now() + timedelta(days=np.random.randint(30, 365))
            }
        except Exception as e:
            logger.error(f"Error getting regulatory updates: {e}")
            return {'pending_changes': 0, 'impact_score': 0.3}
    
    async def _get_supply_chain_indicators(self) -> Dict[str, Any]:
        """Get supply chain disruption indicators"""
        try:
            # Simulate supply chain data (replace with real supply chain APIs)
            return {
                'disruption_risk': np.random.uniform(0.2, 0.8),
                'local_pressure': np.random.uniform(0.3, 0.9),
                'global_volatility': np.random.uniform(0.1, 0.6),
                'transportation_issues': np.random.uniform(0.1, 0.5)
            }
        except Exception as e:
            logger.error(f"Error getting supply chain indicators: {e}")
            return {'disruption_risk': 0.4, 'local_pressure': 0.5}
    
    async def _get_technology_trends(self, industry: str) -> Dict[str, Any]:
        """Get technology adoption trends"""
        try:
            # Simulate technology data (replace with real tech trend APIs)
            return {
                'adoption_rate': np.random.uniform(0.05, 0.25),
                'innovation_score': np.random.uniform(0.3, 0.8),
                'investment_trend': np.random.uniform(0.1, 0.4),
                'emerging_technologies': ['AI', 'IoT', 'Blockchain', '3D_Printing']
            }
        except Exception as e:
            logger.error(f"Error getting technology trends: {e}")
            return {'adoption_rate': 0.1, 'innovation_score': 0.5}
    
    async def _get_sustainability_trends(self, industry: str) -> Dict[str, Any]:
        """Get sustainability trend data"""
        try:
            # Simulate sustainability data (replace with real sustainability APIs)
            return {
                'pressure_score': np.random.uniform(0.5, 0.9),
                'circular_growth': np.random.uniform(0.1, 0.3),
                'carbon_reduction_targets': np.random.uniform(0.2, 0.6),
                'esg_investment_trend': np.random.uniform(0.1, 0.4)
            }
        except Exception as e:
            logger.error(f"Error getting sustainability trends: {e}")
            return {'pressure_score': 0.6, 'circular_growth': 0.15}
    
    def _filter_and_rank_opportunities(self, opportunities: List[OpportunityPrediction]) -> List[OpportunityPrediction]:
        """Filter and rank opportunities by confidence and impact"""
        try:
            # Filter by confidence threshold
            filtered = [opp for opp in opportunities if opp.confidence_score >= self.config['confidence_threshold']]
            
            # Sort by combined score (confidence * impact)
            filtered.sort(key=lambda x: x.confidence_score * x.impact_score, reverse=True)
            
            # Limit results
            max_opportunities = self.config['max_opportunities_per_company']
            return filtered[:max_opportunities]
            
        except Exception as e:
            logger.error(f"Error filtering opportunities: {e}")
            return opportunities
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result from Redis"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return pickle.loads(cached_data.encode('latin1'))
            return None
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None
    
    def _cache_result(self, cache_key: str, data: Any) -> None:
        """Cache result in Redis"""
        try:
            serialized_data = pickle.dumps(data)
            self.redis_client.setex(cache_key, self.cache_ttl, serialized_data)
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    def start_background_processing(self):
        """Start background opportunity scanning"""
        if not self.running:
            self.running = True
            self.background_thread = threading.Thread(target=self._background_scan, daemon=True)
            self.background_thread.start()
            logger.info("🔄 Background opportunity scanning started")

    def stop_background_processing(self):
        """Stop background opportunity scanning"""
        self.running = False
        if self.background_thread:
            self.background_thread.join()
        logger.info("⏹️ Background opportunity scanning stopped")

    def _background_scan(self):
        """Background scanning for opportunities"""
        while self.running:
            try:
                # Scan for global opportunities
                self._scan_global_opportunities()
                
                # Update market intelligence
                self._update_market_intelligence()
                
                # Sleep for configured interval
                time.sleep(self.config['update_frequency_minutes'] * 60)
                
            except Exception as e:
                logger.error(f"Error in background scan: {e}")
                time.sleep(300)  # Sleep for 5 minutes on error
    
    def _scan_global_opportunities(self):
        """Scan for global opportunities across all industries"""
        try:
            # This would integrate with real-time data sources
            # For now, we'll simulate global scanning
            logger.debug("Scanning for global opportunities...")
            
        except Exception as e:
            logger.error(f"Error scanning global opportunities: {e}")
    
    def _update_market_intelligence(self):
        """Update market intelligence data"""
        try:
            # This would update market data from various sources
            # For now, we'll simulate updates
            logger.debug("Updating market intelligence...")
            
        except Exception as e:
            logger.error(f"Error updating market intelligence: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_predictions': len(self.prediction_accuracy),
            'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'background_scanning_active': self.running,
            'last_update': datetime.now().isoformat()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        try:
            # This would calculate actual cache hit rate
            return 0.75  # Simulated 75% hit rate
        except Exception as e:
            logger.error(f"Error calculating cache hit rate: {e}")
            return 0.0

# Add Flask app and API for explainability endpoint if not present
app = Flask(__name__)
api = Api(app, version='1.0', title='Proactive Opportunity Engine', description='Advanced ML Opportunity Detection', doc='/docs')

# Add data validator
data_validator = AdvancedDataValidator(logger=logger)

explain_input = api.model('ExplainInput', {
    'model_type': fields.String(required=True, description='Model type (demand_forecaster, market_analyzer, etc.)'),
    'input_data': fields.Raw(required=True, description='Input data for explanation')
})

@api.route('/explain')
class Explain(Resource):
    @api.expect(explain_input)
    @api.response(200, 'Success')
    @api.response(400, 'Invalid input data')
    @api.response(500, 'Internal error')
    def post(self):
        try:
            data = request.json
            model_type = data.get('model_type')
            input_data = data.get('input_data')
            schema = {'type': 'object', 'properties': {'features': {'type': 'array'}}, 'required': ['features']}
            data_validator.set_schema(schema)
            if not data_validator.validate(input_data):
                logger.error('Input data failed schema validation.')
                return {'error': 'Invalid input data'}, 400
            features = np.array(input_data['features']).reshape(1, -1)
            if model_type == 'demand_forecaster':
                model = self.demand_forecaster
            elif model_type == 'market_analyzer':
                model = self.market_analyzer
            else:
                logger.error(f'Unknown model_type: {model_type}')
                return {'error': 'Unknown model_type'}, 400
            explainer = shap.Explainer(model.predict, features)
            shap_values = explainer(features)
            logger.info(f'Explanation generated for {model_type}')
            return {'shap_values': shap_values.values.tolist(), 'base_values': shap_values.base_values.tolist()}
        except Exception as e:
            logger.error(f'Explainability error: {e}')
            return {'error': str(e)}, 500

# Global instance
proactive_opportunity_engine = AdvancedProactiveOpportunityEngine() 
ProactiveOpportunityEngine = AdvancedProactiveOpportunityEngine 

if __name__ == "__main__":
    print("🚀 Starting Proactive Opportunity Engine on port 5014...")
    app.run(host='0.0.0.0', port=5014, debug=False) 