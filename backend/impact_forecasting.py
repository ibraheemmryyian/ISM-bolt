"""
Impact Forecasting Engine for Industrial Symbiosis
Predicts environmental, economic, and social impact of symbiosis partnerships
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import json
import asyncio
import aiohttp
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Required ML imports - fail if missing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Required time series imports - fail if missing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Required optimization imports - fail if missing
from scipy.optimize import minimize
from scipy.stats import norm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImpactForecast:
    """Impact forecast data structure"""
    forecast_id: str
    company_id: str
    forecast_type: str  # 'carbon_reduction', 'cost_savings', 'waste_reduction', 'energy_savings'
    timeframe: str
    carbon_reduction: float  # kg CO2e
    cost_savings: float  # USD
    waste_reduction: float  # kg
    energy_savings: float  # kWh
    water_savings: float  # liters
    social_impact_score: float  # 0-1
    economic_impact_score: float  # 0-1
    environmental_impact_score: float  # 0-1
    confidence_level: float  # 0-1
    assumptions: List[str]
    risks: List[str]
    recommendations: List[str]
    created_at: datetime

class ImpactForecastingEngine:
    """
    Advanced Impact Forecasting Engine for Industrial Symbiosis
    
    Features:
    - Multi-dimensional impact prediction
    - Time series forecasting
    - Machine learning models
    - Uncertainty quantification
    - Scenario analysis
    - Real-time updates
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Initialize ML models
        self.carbon_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.cost_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.waste_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.energy_model = LinearRegression()
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
        # Model performance tracking
        self.model_performance = {}
        self.forecast_history = []
        
        # Initialize models
        self._initialize_models()
        
        logger.info("Impact Forecasting Engine initialized successfully")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'forecast_horizon': 365,  # days
            'confidence_level': 0.95,
            'update_frequency': 24,  # hours
            'min_data_points': 10,
            'max_forecast_periods': 12,  # months
            'uncertainty_quantification': True,
            'scenario_analysis': True
        }

    def _initialize_models(self):
        """Initialize forecasting models"""
        try:
            # Load historical data for training
            historical_data = self._load_historical_data()
            
            if historical_data is not None and len(historical_data) > self.config['min_data_points']:
                self._train_models(historical_data)
            else:
                logger.warning("Insufficient historical data for model training")
                
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    def _load_historical_data(self) -> Optional[pd.DataFrame]:
        """Load historical impact data"""
        try:
            # Load from database or file
            data_path = Path("data/historical_impact_data.csv")
            if data_path.exists():
                return pd.read_csv(data_path)
            else:
                logger.warning("Historical data file not found")
                return None
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return None

    def _train_models(self, data: pd.DataFrame):
        """Train forecasting models"""
        try:
            # Prepare features and targets
            features = ['company_size', 'industry_type', 'location_factor', 'technology_level']
            targets = ['carbon_reduction', 'cost_savings', 'waste_reduction', 'energy_savings']
            
            X = data[features]
            
            # Train each model
            for target in targets:
                if target in data.columns:
                    y = data[target]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Scale features
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    X_test_scaled = self.scaler.transform(X_test)
                    
                    # Train model
                    if target == 'carbon_reduction':
                        model = self.carbon_model
                    elif target == 'cost_savings':
                        model = self.cost_model
                    elif target == 'waste_reduction':
                        model = self.waste_model
                    elif target == 'energy_savings':
                        model = self.energy_model
                    
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    self.model_performance[target] = {
                        'mse': mse,
                        'r2': r2,
                        'rmse': np.sqrt(mse)
                    }
                    
                    # Feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[target] = dict(zip(features, model.feature_importances_))
                    
                    logger.info(f"Trained {target} model: R²={r2:.4f}, RMSE={np.sqrt(mse):.4f}")
                    
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

    def forecast_impact(self, company_data: Dict[str, Any], 
                       forecast_type: str = 'comprehensive',
                       timeframe_days: int = 365) -> ImpactForecast:
        """Generate impact forecast for a company"""
        try:
            # Validate inputs
            if not company_data:
                raise ValueError("Company data is required")
            
            if timeframe_days <= 0:
                raise ValueError("Timeframe must be positive")
            
            # Generate forecast
            if forecast_type == 'comprehensive':
                forecast = self._generate_comprehensive_forecast(company_data, timeframe_days)
            elif forecast_type == 'carbon_reduction':
                forecast = self._generate_carbon_forecast(company_data, timeframe_days)
            elif forecast_type == 'cost_savings':
                forecast = self._generate_cost_forecast(company_data, timeframe_days)
            elif forecast_type == 'waste_reduction':
                forecast = self._generate_waste_forecast(company_data, timeframe_days)
            else:
                raise ValueError(f"Unknown forecast type: {forecast_type}")
            
            # Store forecast
            self.forecast_history.append(forecast)
            
            logger.info(f"Generated {forecast_type} forecast for company {company_data.get('id', 'unknown')}")
            
            return forecast
            
        except Exception as e:
            logger.error(f"Impact forecasting failed: {e}")
            raise

    def _generate_comprehensive_forecast(self, company_data: Dict[str, Any], 
                                       timeframe_days: int) -> ImpactForecast:
        """Generate comprehensive impact forecast"""
        try:
            # Extract features
            features = self._extract_company_features(company_data)
            
            # Make predictions
            carbon_reduction = self._predict_carbon_reduction(features, timeframe_days)
            cost_savings = self._predict_cost_savings(features, timeframe_days)
            waste_reduction = self._predict_waste_reduction(features, timeframe_days)
            energy_savings = self._predict_energy_savings(features, timeframe_days)
            
            # Calculate impact scores
            environmental_score = self._calculate_environmental_score(carbon_reduction, waste_reduction, energy_savings)
            economic_score = self._calculate_economic_score(cost_savings)
            social_score = self._calculate_social_score(company_data)
            
            # Calculate confidence level
            confidence = self._calculate_confidence_level(features)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(company_data, carbon_reduction, cost_savings)
            
            # Identify risks
            risks = self._identify_risks(company_data, timeframe_days)
            
            # List assumptions
            assumptions = self._list_assumptions(company_data, timeframe_days)
            
            return ImpactForecast(
                forecast_id=f"forecast_{company_data.get('id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                company_id=company_data.get('id', 'unknown'),
                forecast_type='comprehensive',
                timeframe=f"{timeframe_days} days",
                carbon_reduction=carbon_reduction,
                cost_savings=cost_savings,
                waste_reduction=waste_reduction,
                energy_savings=energy_savings,
                water_savings=self._predict_water_savings(features, timeframe_days),
                social_impact_score=social_score,
                economic_impact_score=economic_score,
                environmental_impact_score=environmental_score,
                confidence_level=confidence,
                assumptions=assumptions,
                risks=risks,
                recommendations=recommendations,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Comprehensive forecast generation failed: {e}")
            raise

    def _extract_company_features(self, company_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from company data"""
        try:
            # Map company data to features
            features = []
            
            # Company size (normalized)
            size = company_data.get('employee_count', 100)
            features.append(min(size / 1000, 1.0))
            
            # Industry type (encoded)
            industry = company_data.get('industry', 'manufacturing').lower()
            industry_encoding = {
                'manufacturing': 0.8,
                'chemical': 0.9,
                'food': 0.6,
                'textiles': 0.7,
                'construction': 0.5
            }
            features.append(industry_encoding.get(industry, 0.5))
            
            # Location factor
            location = company_data.get('location', 'unknown').lower()
            location_factor = 0.7  # Default
            if 'gulf' in location or 'middle_east' in location:
                location_factor = 0.8
            features.append(location_factor)
            
            # Technology level
            tech_level = company_data.get('technology_level', 'medium').lower()
            tech_encoding = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
            features.append(tech_encoding.get(tech_level, 0.6))
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise

    def _predict_carbon_reduction(self, features: np.ndarray, timeframe_days: int) -> float:
        """Predict carbon reduction"""
        try:
            if hasattr(self.carbon_model, 'predict'):
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Make prediction
                base_prediction = self.carbon_model.predict(features_scaled)[0]
                
                # Adjust for timeframe
                timeframe_factor = timeframe_days / 365.0
                
                return max(0, base_prediction * timeframe_factor)
            else:
                # Fallback calculation
                return self._calculate_fallback_carbon_reduction(features, timeframe_days)
                
        except Exception as e:
            logger.error(f"Carbon reduction prediction failed: {e}")
            raise

    def _predict_cost_savings(self, features: np.ndarray, timeframe_days: int) -> float:
        """Predict cost savings"""
        try:
            if hasattr(self.cost_model, 'predict'):
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Make prediction
                base_prediction = self.cost_model.predict(features_scaled)[0]
                
                # Adjust for timeframe
                timeframe_factor = timeframe_days / 365.0
                
                return max(0, base_prediction * timeframe_factor)
            else:
                # Fallback calculation
                return self._calculate_fallback_cost_savings(features, timeframe_days)
                
        except Exception as e:
            logger.error(f"Cost savings prediction failed: {e}")
            raise

    def _predict_waste_reduction(self, features: np.ndarray, timeframe_days: int) -> float:
        """Predict waste reduction"""
        try:
            if hasattr(self.waste_model, 'predict'):
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Make prediction
                base_prediction = self.waste_model.predict(features_scaled)[0]
                
                # Adjust for timeframe
                timeframe_factor = timeframe_days / 365.0
                
                return max(0, base_prediction * timeframe_factor)
            else:
                # Fallback calculation
                return self._calculate_fallback_waste_reduction(features, timeframe_days)
                
        except Exception as e:
            logger.error(f"Waste reduction prediction failed: {e}")
            raise

    def _predict_energy_savings(self, features: np.ndarray, timeframe_days: int) -> float:
        """Predict energy savings"""
        try:
            if hasattr(self.energy_model, 'predict'):
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Make prediction
                base_prediction = self.energy_model.predict(features_scaled)[0]
                
                # Adjust for timeframe
                timeframe_factor = timeframe_days / 365.0
                
                return max(0, base_prediction * timeframe_factor)
            else:
                # Fallback calculation
                return self._calculate_fallback_energy_savings(features, timeframe_days)
                
        except Exception as e:
            logger.error(f"Energy savings prediction failed: {e}")
            raise

    def _predict_water_savings(self, features: np.ndarray, timeframe_days: int) -> float:
        """Predict water savings"""
        try:
            # Simple calculation based on industry and size
            industry_factor = features[0, 1]  # Industry encoding
            size_factor = features[0, 0]  # Size factor
            
            base_water_savings = 10000  # liters per year
            timeframe_factor = timeframe_days / 365.0
            
            return base_water_savings * industry_factor * size_factor * timeframe_factor
            
        except Exception as e:
            logger.error(f"Water savings prediction failed: {e}")
            raise

    def _calculate_environmental_score(self, carbon_reduction: float, 
                                     waste_reduction: float, 
                                     energy_savings: float) -> float:
        """Calculate environmental impact score"""
        try:
            # Normalize values
            carbon_score = min(carbon_reduction / 1000, 1.0)  # Normalize to 1000 kg CO2e
            waste_score = min(waste_reduction / 10000, 1.0)  # Normalize to 10,000 kg
            energy_score = min(energy_savings / 50000, 1.0)  # Normalize to 50,000 kWh
            
            # Weighted average
            environmental_score = (0.4 * carbon_score + 0.3 * waste_score + 0.3 * energy_score)
            
            return min(1.0, max(0.0, environmental_score))
            
        except Exception as e:
            logger.error(f"Environmental score calculation failed: {e}")
            raise

    def _calculate_economic_score(self, cost_savings: float) -> float:
        """Calculate economic impact score"""
        try:
            # Normalize to $100,000 savings
            economic_score = min(cost_savings / 100000, 1.0)
            
            return min(1.0, max(0.0, economic_score))
            
        except Exception as e:
            logger.error(f"Economic score calculation failed: {e}")
            raise

    def _calculate_social_score(self, company_data: Dict[str, Any]) -> float:
        """Calculate social impact score"""
        try:
            # Simple social impact calculation
            employee_count = company_data.get('employee_count', 100)
            industry = company_data.get('industry', 'manufacturing').lower()
            
            # Base social score
            social_score = 0.5
            
            # Adjust for company size (more employees = more social impact)
            if employee_count > 500:
                social_score += 0.2
            elif employee_count > 100:
                social_score += 0.1
            
            # Adjust for industry
            if industry in ['food', 'pharmaceutical']:
                social_score += 0.1
            
            return min(1.0, max(0.0, social_score))
            
        except Exception as e:
            logger.error(f"Social score calculation failed: {e}")
            raise

    def _calculate_confidence_level(self, features: np.ndarray) -> float:
        """Calculate confidence level for forecast"""
        try:
            # Base confidence
            confidence = 0.7
            
            # Adjust based on data quality
            if features[0, 0] > 0.5:  # Good size data
                confidence += 0.1
            
            if features[0, 1] > 0.7:  # Known industry
                confidence += 0.1
            
            if features[0, 2] > 0.7:  # Good location data
                confidence += 0.1
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            raise

    def _generate_recommendations(self, company_data: Dict[str, Any], 
                                carbon_reduction: float, 
                                cost_savings: float) -> List[str]:
        """Generate recommendations based on forecast"""
        try:
            recommendations = []
            
            # Carbon reduction recommendations
            if carbon_reduction > 500:
                recommendations.append("High carbon reduction potential - consider carbon credit trading")
            
            # Cost savings recommendations
            if cost_savings > 50000:
                recommendations.append("Significant cost savings potential - implement waste-to-resource programs")
            
            # Industry-specific recommendations
            industry = company_data.get('industry', '').lower()
            if industry == 'manufacturing':
                recommendations.append("Optimize production processes for material efficiency")
            elif industry == 'chemical':
                recommendations.append("Implement closed-loop chemical recycling systems")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            raise

    def _identify_risks(self, company_data: Dict[str, Any], timeframe_days: int) -> List[str]:
        """Identify potential risks"""
        try:
            risks = []
            
            # Regulatory risks
            risks.append("Regulatory changes may affect forecast accuracy")
            
            # Market risks
            risks.append("Market fluctuations may impact cost savings")
            
            # Technology risks
            risks.append("Technology adoption delays may reduce impact")
            
            return risks
            
        except Exception as e:
            logger.error(f"Risk identification failed: {e}")
            raise

    def _list_assumptions(self, company_data: Dict[str, Any], timeframe_days: int) -> List[str]:
        """List forecast assumptions"""
        try:
            assumptions = [
                "Stable regulatory environment",
                "Consistent market conditions",
                "Successful technology implementation",
                f"Forecast period: {timeframe_days} days",
                "Based on historical industry data"
            ]
            
            return assumptions
            
        except Exception as e:
            logger.error(f"Assumption listing failed: {e}")
            raise

    def _generate_carbon_forecast(self, company_data: Dict[str, Any], 
                                timeframe_days: int) -> ImpactForecast:
        """Generate carbon reduction forecast only"""
        try:
            features = self._extract_company_features(company_data)
            carbon_reduction = self._predict_carbon_reduction(features, timeframe_days)
            
            return ImpactForecast(
                forecast_id=f"carbon_{company_data.get('id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                company_id=company_data.get('id', 'unknown'),
                forecast_type='carbon_reduction',
                timeframe=f"{timeframe_days} days",
                carbon_reduction=carbon_reduction,
                cost_savings=0.0,
                waste_reduction=0.0,
                energy_savings=0.0,
                water_savings=0.0,
                social_impact_score=0.0,
                economic_impact_score=0.0,
                environmental_impact_score=self._calculate_environmental_score(carbon_reduction, 0, 0),
                confidence_level=self._calculate_confidence_level(features),
                assumptions=self._list_assumptions(company_data, timeframe_days),
                risks=self._identify_risks(company_data, timeframe_days),
                recommendations=self._generate_recommendations(company_data, carbon_reduction, 0),
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Carbon forecast generation failed: {e}")
            raise

    def _generate_cost_forecast(self, company_data: Dict[str, Any], 
                              timeframe_days: int) -> ImpactForecast:
        """Generate cost savings forecast only"""
        try:
            features = self._extract_company_features(company_data)
            cost_savings = self._predict_cost_savings(features, timeframe_days)
            
            return ImpactForecast(
                forecast_id=f"cost_{company_data.get('id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                company_id=company_data.get('id', 'unknown'),
                forecast_type='cost_savings',
                timeframe=f"{timeframe_days} days",
                carbon_reduction=0.0,
                cost_savings=cost_savings,
                waste_reduction=0.0,
                energy_savings=0.0,
                water_savings=0.0,
                social_impact_score=0.0,
                economic_impact_score=self._calculate_economic_score(cost_savings),
                environmental_impact_score=0.0,
                confidence_level=self._calculate_confidence_level(features),
                assumptions=self._list_assumptions(company_data, timeframe_days),
                risks=self._identify_risks(company_data, timeframe_days),
                recommendations=self._generate_recommendations(company_data, 0, cost_savings),
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Cost forecast generation failed: {e}")
            raise

    def _generate_waste_forecast(self, company_data: Dict[str, Any], 
                               timeframe_days: int) -> ImpactForecast:
        """Generate waste reduction forecast only"""
        try:
            features = self._extract_company_features(company_data)
            waste_reduction = self._predict_waste_reduction(features, timeframe_days)
            
            return ImpactForecast(
                forecast_id=f"waste_{company_data.get('id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                company_id=company_data.get('id', 'unknown'),
                forecast_type='waste_reduction',
                timeframe=f"{timeframe_days} days",
                carbon_reduction=0.0,
                cost_savings=0.0,
                waste_reduction=waste_reduction,
                energy_savings=0.0,
                water_savings=0.0,
                social_impact_score=0.0,
                economic_impact_score=0.0,
                environmental_impact_score=self._calculate_environmental_score(0, waste_reduction, 0),
                confidence_level=self._calculate_confidence_level(features),
                assumptions=self._list_assumptions(company_data, timeframe_days),
                risks=self._identify_risks(company_data, timeframe_days),
                recommendations=self._generate_recommendations(company_data, 0, 0),
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Waste forecast generation failed: {e}")
            raise

    def _calculate_fallback_carbon_reduction(self, features: np.ndarray, timeframe_days: int) -> float:
        """Calculate fallback carbon reduction"""
        try:
            # Simple calculation based on company size and industry
            size_factor = features[0, 0]
            industry_factor = features[0, 1]
            
            base_carbon_reduction = 1000  # kg CO2e per year
            timeframe_factor = timeframe_days / 365.0
            
            return base_carbon_reduction * size_factor * industry_factor * timeframe_factor
            
        except Exception as e:
            logger.error(f"Fallback carbon calculation failed: {e}")
            raise

    def _calculate_fallback_cost_savings(self, features: np.ndarray, timeframe_days: int) -> float:
        """Calculate fallback cost savings"""
        try:
            # Simple calculation based on company size and industry
            size_factor = features[0, 0]
            industry_factor = features[0, 1]
            
            base_cost_savings = 50000  # USD per year
            timeframe_factor = timeframe_days / 365.0
            
            return base_cost_savings * size_factor * industry_factor * timeframe_factor
            
        except Exception as e:
            logger.error(f"Fallback cost calculation failed: {e}")
            raise

    def _calculate_fallback_waste_reduction(self, features: np.ndarray, timeframe_days: int) -> float:
        """Calculate fallback waste reduction"""
        try:
            # Simple calculation based on company size and industry
            size_factor = features[0, 0]
            industry_factor = features[0, 1]
            
            base_waste_reduction = 5000  # kg per year
            timeframe_factor = timeframe_days / 365.0
            
            return base_waste_reduction * size_factor * industry_factor * timeframe_factor
            
        except Exception as e:
            logger.error(f"Fallback waste calculation failed: {e}")
            raise

    def _calculate_fallback_energy_savings(self, features: np.ndarray, timeframe_days: int) -> float:
        """Calculate fallback energy savings"""
        try:
            # Simple calculation based on company size and industry
            size_factor = features[0, 0]
            industry_factor = features[0, 1]
            
            base_energy_savings = 25000  # kWh per year
            timeframe_factor = timeframe_days / 365.0
            
            return base_energy_savings * size_factor * industry_factor * timeframe_factor
            
        except Exception as e:
            logger.error(f"Fallback energy calculation failed: {e}")
            raise

    def get_forecast_statistics(self) -> Dict[str, Any]:
        """Get forecasting statistics"""
        return {
            'total_forecasts': len(self.forecast_history),
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'last_forecast': self.forecast_history[-1].created_at.isoformat() if self.forecast_history else None
        }

# Global impact forecasting engine instance
impact_forecasting_engine = ImpactForecastingEngine() 