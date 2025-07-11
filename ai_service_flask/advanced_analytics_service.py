"""
Advanced Analytics Service for Industrial Symbiosis
Predictive Modeling, Impact Forecasting, and Real-Time Insights
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from flask import Flask, request, jsonify
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import redis
import pickle
from datetime import datetime, timedelta
import hashlib
import hmac
import secrets
import os
import threading
import queue
import time
from collections import defaultdict, deque
import heapq
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Advanced Analytics Configuration
@dataclass
class AnalyticsConfig:
    """Advanced Analytics Configuration"""
    prediction_horizon: int = 365  # days
    confidence_level: float = 0.95
    model_update_frequency: int = 24  # hours
    real_time_processing: bool = True
    cache_results: bool = True
    max_cache_size: int = 1000
    parallel_processing: bool = True
    num_workers: int = 4
    model_selection: str = "ensemble"  # single, ensemble, auto
    feature_engineering: bool = True
    anomaly_detection: bool = True
    trend_analysis: bool = True
    impact_forecasting: bool = True
    optimization_engine: bool = True

class AdvancedPredictiveModel:
    """Advanced Predictive Model with Ensemble Learning"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.is_trained = False
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different types of models"""
        if self.config.model_selection == "ensemble":
            self.models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=0.1)
            }
        else:
            # Single model approach
            self.models = {
                'main_model': RandomForestRegressor(n_estimators=100, random_state=42)
            }
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering"""
        if not self.config.feature_engineering:
            return data
        
        df = data.copy()
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['day'] = df['timestamp'].dt.day
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['quarter'] = df['timestamp'].dt.quarter
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['year', 'month', 'day', 'day_of_week', 'quarter', 'is_weekend']:
                df[f'{col}_lag_1'] = df[col].shift(1)
                df[f'{col}_lag_7'] = df[col].shift(7)
                df[f'{col}_lag_30'] = df[col].shift(30)
                
                # Rolling statistics
                df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7).mean()
                df[f'{col}_rolling_std_7'] = df[col].rolling(window=7).std()
                df[f'{col}_rolling_max_7'] = df[col].rolling(window=7).max()
                df[f'{col}_rolling_min_7'] = df[col].rolling(window=7).min()
        
        # Interaction features
        if len(numeric_columns) >= 2:
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i+1:]:
                    df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        
        # Polynomial features
        for col in numeric_columns[:3]:  # Limit to first 3 columns to avoid explosion
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_cubed'] = df[col] ** 3
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train all models"""
        # Feature engineering
        X_engineered = self.engineer_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, random_state=42
        )
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                # Scale features
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                self.model_performance[model_name] = {
                    'mse': mse,
                    'r2': r2,
                    'mae': mae
                }
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(zip(
                        X_engineered.columns, model.feature_importances_
                    ))
                
                logging.info(f"Trained {model_name}: R²={r2:.4f}, MSE={mse:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {model_name}: {e}")
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> Dict:
        """Make predictions using ensemble"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Feature engineering
        X_engineered = self.engineer_features(X)
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Scale features
                X_scaled = self.scalers[model_name].transform(X_engineered)
                
                # Make prediction
                pred = model.predict(X_scaled)
                predictions[model_name] = pred
                
            except Exception as e:
                logging.error(f"Error predicting with {model_name}: {e}")
        
        # Ensemble prediction
        if len(predictions) > 1:
            # Weighted average based on model performance
            weights = []
            pred_values = []
            
            for model_name, pred in predictions.items():
                if model_name in self.model_performance:
                    weight = self.model_performance[model_name]['r2']
                    weights.append(max(0, weight))  # Ensure non-negative weights
                    pred_values.append(pred)
            
            if weights and pred_values:
                weights = np.array(weights) / sum(weights)
                ensemble_pred = np.average(pred_values, axis=0, weights=weights)
                predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance for all models"""
        return self.feature_importance
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for all models"""
        return self.model_performance

class ImpactForecaster:
    """Advanced Impact Forecasting Engine"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.forecast_models = {}
        self.impact_metrics = {}
        self.forecast_cache = {}
        
    def forecast_environmental_impact(self, historical_data: pd.DataFrame, 
                                    forecast_periods: int = None) -> Dict:
        """Forecast environmental impact"""
        if forecast_periods is None:
            forecast_periods = self.config.prediction_horizon
        
        forecasts = {}
        
        # Carbon emissions forecast
        if 'carbon_emissions' in historical_data.columns:
            carbon_forecast = self._time_series_forecast(
                historical_data, 'carbon_emissions', forecast_periods
            )
            forecasts['carbon_emissions'] = carbon_forecast
        
        # Waste reduction forecast
        if 'waste_reduction' in historical_data.columns:
            waste_forecast = self._time_series_forecast(
                historical_data, 'waste_reduction', forecast_periods
            )
            forecasts['waste_reduction'] = waste_forecast
        
        # Energy savings forecast
        if 'energy_savings' in historical_data.columns:
            energy_forecast = self._time_series_forecast(
                historical_data, 'energy_savings', forecast_periods
            )
            forecasts['energy_savings'] = energy_forecast
        
        return forecasts
    
    def forecast_economic_impact(self, historical_data: pd.DataFrame,
                               forecast_periods: int = None) -> Dict:
        """Forecast economic impact"""
        if forecast_periods is None:
            forecast_periods = self.config.prediction_horizon
        
        forecasts = {}
        
        # Cost savings forecast
        if 'cost_savings' in historical_data.columns:
            cost_forecast = self._time_series_forecast(
                historical_data, 'cost_savings', forecast_periods
            )
            forecasts['cost_savings'] = cost_forecast
        
        # Revenue increase forecast
        if 'revenue_increase' in historical_data.columns:
            revenue_forecast = self._time_series_forecast(
                historical_data, 'revenue_increase', forecast_periods
            )
            forecasts['revenue_increase'] = revenue_forecast
        
        return forecasts
    
    def _time_series_forecast(self, data: pd.DataFrame, column: str, 
                            periods: int) -> Dict:
        """Time series forecasting using Prophet"""
        try:
            # Prepare data for Prophet
            df = data[['timestamp', column]].copy()
            df.columns = ['ds', 'y']
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) < 10:
                return {'error': 'Insufficient data for forecasting'}
            
            # Create and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=self.config.confidence_level
            )
            
            model.fit(df)
            
            # Make forecast
            future = model.make_future_dataframe(periods=periods, freq='D')
            forecast = model.predict(future)
            
            # Extract forecast results
            forecast_result = {
                'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict('records'),
                'trend': model.plot_components(forecast),
                'model_performance': {
                    'mae': mean_absolute_error(df['y'], forecast['yhat'][:len(df)]),
                    'rmse': np.sqrt(mean_squared_error(df['y'], forecast['yhat'][:len(df)]))
                }
            }
            
            return forecast_result
            
        except Exception as e:
            logging.error(f"Time series forecast error for {column}: {e}")
            return {'error': str(e)}
    
    def calculate_cumulative_impact(self, forecasts: Dict) -> Dict:
        """Calculate cumulative impact over forecast period"""
        cumulative_impact = {}
        
        for metric, forecast in forecasts.items():
            if 'error' not in forecast and 'forecast' in forecast:
                values = [point['yhat'] for point in forecast['forecast']]
                cumulative_impact[metric] = {
                    'total': sum(values),
                    'average_daily': np.mean(values),
                    'peak_value': max(values),
                    'growth_rate': (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
                }
        
        return cumulative_impact

class AnomalyDetector:
    """Advanced Anomaly Detection System"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.anomaly_models = {}
        self.thresholds = {}
        
    def detect_anomalies(self, data: pd.DataFrame) -> Dict:
        """Detect anomalies in data"""
        anomalies = {}
        
        # Statistical anomaly detection
        statistical_anomalies = self._statistical_anomaly_detection(data)
        anomalies['statistical'] = statistical_anomalies
        
        # Isolation Forest for complex patterns
        isolation_anomalies = self._isolation_forest_detection(data)
        anomalies['isolation_forest'] = isolation_anomalies
        
        # Time series anomaly detection
        if 'timestamp' in data.columns:
            time_series_anomalies = self._time_series_anomaly_detection(data)
            anomalies['time_series'] = time_series_anomalies
        
        return anomalies
    
    def _statistical_anomaly_detection(self, data: pd.DataFrame) -> Dict:
        """Statistical anomaly detection using Z-score and IQR"""
        anomalies = {}
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            values = data[column].dropna()
            
            if len(values) == 0:
                continue
            
            # Z-score method
            z_scores = np.abs(stats.zscore(values))
            z_anomalies = values[z_scores > 3]
            
            # IQR method
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            iqr_anomalies = values[(values < (Q1 - 1.5 * IQR)) | (values > (Q3 + 1.5 * IQR))]
            
            anomalies[column] = {
                'z_score_anomalies': z_anomalies.tolist(),
                'iqr_anomalies': iqr_anomalies.tolist(),
                'z_score_count': len(z_anomalies),
                'iqr_count': len(iqr_anomalies)
            }
        
        return anomalies
    
    def _isolation_forest_detection(self, data: pd.DataFrame) -> Dict:
        """Isolation Forest anomaly detection"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Select numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return {}
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(numeric_data)
            
            # Find anomalies (predictions == -1)
            anomaly_indices = np.where(predictions == -1)[0]
            
            return {
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_count': len(anomaly_indices),
                'anomaly_percentage': len(anomaly_indices) / len(predictions) * 100
            }
            
        except Exception as e:
            logging.error(f"Isolation Forest error: {e}")
            return {'error': str(e)}
    
    def _time_series_anomaly_detection(self, data: pd.DataFrame) -> Dict:
        """Time series anomaly detection"""
        anomalies = {}
        
        if 'timestamp' not in data.columns:
            return anomalies
        
        # Sort by timestamp
        data_sorted = data.sort_values('timestamp')
        
        numeric_columns = data_sorted.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column == 'timestamp':
                continue
            
            values = data_sorted[column].dropna()
            
            if len(values) < 10:
                continue
            
            # Calculate rolling statistics
            rolling_mean = values.rolling(window=7, min_periods=1).mean()
            rolling_std = values.rolling(window=7, min_periods=1).std()
            
            # Detect anomalies based on deviation from rolling mean
            threshold = 2  # Standard deviations
            upper_bound = rolling_mean + threshold * rolling_std
            lower_bound = rolling_mean - threshold * rolling_std
            
            anomalies_mask = (values > upper_bound) | (values < lower_bound)
            anomaly_values = values[anomalies_mask]
            
            anomalies[column] = {
                'anomaly_values': anomaly_values.tolist(),
                'anomaly_count': len(anomaly_values),
                'anomaly_percentage': len(anomaly_values) / len(values) * 100
            }
        
        return anomalies

class TrendAnalyzer:
    """Advanced Trend Analysis Engine"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        
    def analyze_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze trends in data"""
        trends = {}
        
        # Time series trends
        if 'timestamp' in data.columns:
            time_trends = self._analyze_time_trends(data)
            trends['time_series'] = time_trends
        
        # Seasonal patterns
        seasonal_patterns = self._analyze_seasonality(data)
        trends['seasonality'] = seasonal_patterns
        
        # Correlation analysis
        correlations = self._analyze_correlations(data)
        trends['correlations'] = correlations
        
        return trends
    
    def _analyze_time_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze time-based trends"""
        trends = {}
        
        if 'timestamp' not in data.columns:
            return trends
        
        data_sorted = data.sort_values('timestamp')
        numeric_columns = data_sorted.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column == 'timestamp':
                continue
            
            values = data_sorted[column].dropna()
            
            if len(values) < 10:
                continue
            
            # Linear trend
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Trend classification
            if p_value < 0.05:  # Statistically significant
                if slope > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            
            trends[column] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend_direction': trend_direction,
                'trend_strength': abs(r_value)
            }
        
        return trends
    
    def _analyze_seasonality(self, data: pd.DataFrame) -> Dict:
        """Analyze seasonal patterns"""
        seasonality = {}
        
        if 'timestamp' not in data.columns:
            return seasonality
        
        data_sorted = data.sort_values('timestamp')
        data_sorted['timestamp'] = pd.to_datetime(data_sorted['timestamp'])
        
        numeric_columns = data_sorted.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column == 'timestamp':
                continue
            
            # Monthly seasonality
            monthly_avg = data_sorted.groupby(data_sorted['timestamp'].dt.month)[column].mean()
            
            # Weekly seasonality
            weekly_avg = data_sorted.groupby(data_sorted['timestamp'].dt.dayofweek)[column].mean()
            
            seasonality[column] = {
                'monthly_pattern': monthly_avg.to_dict(),
                'weekly_pattern': weekly_avg.to_dict(),
                'seasonal_strength': self._calculate_seasonal_strength(data_sorted[column])
            }
        
        return seasonality
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict:
        """Analyze correlations between variables"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {}
        
        correlation_matrix = numeric_data.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    def _calculate_seasonal_strength(self, series: pd.Series) -> float:
        """Calculate seasonal strength"""
        if len(series) < 12:
            return 0.0
        
        # Calculate seasonal component strength
        seasonal_variance = series.rolling(window=12).var().mean()
        total_variance = series.var()
        
        if total_variance == 0:
            return 0.0
        
        return seasonal_variance / total_variance

class AdvancedAnalyticsService:
    """Comprehensive Advanced Analytics Service"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.predictive_model = AdvancedPredictiveModel(config)
        self.impact_forecaster = ImpactForecaster(config)
        self.anomaly_detector = AnomalyDetector(config)
        self.trend_analyzer = TrendAnalyzer(config)
        self.cache = {}
        self.cache_timestamps = {}
        
    def analyze_data(self, data: pd.DataFrame) -> Dict:
        """Comprehensive data analysis"""
        analysis_results = {}
        
        # Anomaly detection
        if self.config.anomaly_detection:
            anomalies = self.anomaly_detector.detect_anomalies(data)
            analysis_results['anomalies'] = anomalies
        
        # Trend analysis
        if self.config.trend_analysis:
            trends = self.trend_analyzer.analyze_trends(data)
            analysis_results['trends'] = trends
        
        # Impact forecasting
        if self.config.impact_forecasting:
            environmental_forecast = self.impact_forecaster.forecast_environmental_impact(data)
            economic_forecast = self.impact_forecaster.forecast_economic_impact(data)
            
            analysis_results['forecasts'] = {
                'environmental': environmental_forecast,
                'economic': economic_forecast
            }
            
            # Calculate cumulative impact
            all_forecasts = {**environmental_forecast, **economic_forecast}
            cumulative_impact = self.impact_forecaster.calculate_cumulative_impact(all_forecasts)
            analysis_results['cumulative_impact'] = cumulative_impact
        
        return analysis_results
    
    def train_predictive_models(self, X: pd.DataFrame, y: pd.Series):
        """Train predictive models"""
        self.predictive_model.train(X, y)
    
    def make_predictions(self, X: pd.DataFrame) -> Dict:
        """Make predictions using trained models"""
        return self.predictive_model.predict(X)
    
    def get_model_insights(self) -> Dict:
        """Get insights from trained models"""
        return {
            'feature_importance': self.predictive_model.get_feature_importance(),
            'model_performance': self.predictive_model.get_model_performance()
        }

# Flask Application for Advanced Analytics
analytics_app = Flask(__name__)

# Initialize analytics service
analytics_config = AnalyticsConfig()
analytics_service = AdvancedAnalyticsService(analytics_config)

@analytics_app.route('/health', methods=['GET'])
def analytics_health_check():
    """Health check for analytics service"""
    return jsonify({
        'status': 'healthy',
        'service': 'advanced_analytics',
        'models_trained': analytics_service.predictive_model.is_trained,
        'cache_size': len(analytics_service.cache)
    })

@analytics_app.route('/analyze', methods=['POST'])
def analyze_data():
    """Analyze data comprehensively"""
    try:
        data = request.get_json()
        df_data = data.get('data', [])
        
        if not df_data:
            return jsonify({'error': 'Data required'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(df_data)
        
        # Perform analysis
        analysis_results = analytics_service.analyze_data(df)
        
        return jsonify({
            'status': 'success',
            'analysis': analysis_results
        })
        
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@analytics_app.route('/train', methods=['POST'])
def train_models():
    """Train predictive models"""
    try:
        data = request.get_json()
        features = data.get('features', [])
        target = data.get('target', [])
        
        if not features or not target:
            return jsonify({'error': 'Features and target required'}), 400
        
        # Convert to DataFrame and Series
        X = pd.DataFrame(features)
        y = pd.Series(target)
        
        # Train models
        analytics_service.train_predictive_models(X, y)
        
        return jsonify({
            'status': 'success',
            'message': 'Models trained successfully'
        })
        
    except Exception as e:
        logging.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500

@analytics_app.route('/predict', methods=['POST'])
def make_predictions():
    """Make predictions"""
    try:
        data = request.get_json()
        features = data.get('features', [])
        
        if not features:
            return jsonify({'error': 'Features required'}), 400
        
        # Convert to DataFrame
        X = pd.DataFrame(features)
        
        # Make predictions
        predictions = analytics_service.make_predictions(X)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions
        })
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@analytics_app.route('/insights', methods=['GET'])
def get_insights():
    """Get model insights"""
    try:
        insights = analytics_service.get_model_insights()
        
        return jsonify({
            'status': 'success',
            'insights': insights
        })
        
    except Exception as e:
        logging.error(f"Insights error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    analytics_app.run(host='0.0.0.0', port=5004, debug=False) 