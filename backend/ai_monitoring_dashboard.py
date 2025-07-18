"""
Production-Grade AI Monitoring Dashboard
Real-time monitoring and insights for AI system performance
"""

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
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import mlflow
import wandb
import redis
import psutil
import GPUtil
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

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
    DataValidator
)
from ml_core.optimization import (
    HyperparameterOptimizer,
    MonitoringOptimizer
)
from ml_core.monitoring import (
    MLMetricsTracker,
    ProductionMonitor,
    AnomalyDetector
)
from ml_core.utils import (
    ModelRegistry,
    MonitoringManager,
    ConfigManager
)

from .utils.distributed_logger import DistributedLogger
from .utils.advanced_data_validator import AdvancedDataValidator
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import shap

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MetricDefinition:
    name: str
    type: MetricType
    description: str
    labels: List[str]
    buckets: Optional[List[float]] = None

class AnomalyDetectionModel(nn.Module):
    """Real ML model for anomaly detection in monitoring data"""
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Anomaly score head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Anomaly score
        anomaly_score = self.anomaly_head(encoded)
        
        return decoded, anomaly_score

class DriftDetectionModel(nn.Module):
    """Real ML model for concept drift detection"""
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Drift detector
        self.drift_detector = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Distribution comparator
        self.distribution_comparator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 10),  # 10 distribution bins
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Detect drift
        drift_score = self.drift_detector(features)
        
        # Compare distributions
        distribution = self.distribution_comparator(features)
        
        return drift_score, distribution

class RealTimeMetricsCollector:
    """Real-time metrics collector with advanced aggregation"""
    def __init__(self):
        self.metrics = {}
        self.metric_definitions = self._define_metrics()
        self.initialize_metrics()
        
        # Data storage
        self.metric_history = {}
        self.alert_queue = queue.Queue()
        
        # Start collection thread
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
    
    def _define_metrics(self) -> Dict[str, MetricDefinition]:
        """Define monitoring metrics"""
        return {
            # Model performance metrics
            'model_accuracy': MetricDefinition(
                name='model_accuracy',              type=MetricType.GAUGE,
                description='Model accuracy over time,            labels=['model_id', 'version]    ),
            'model_latency': MetricDefinition(
                name='model_latency',              type=MetricType.HISTOGRAM,
                description='Model inference latency,            labels=['model_id', 'version'],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]    ),
            'model_throughput': MetricDefinition(
                name='model_throughput',              type=MetricType.COUNTER,
                description='Model requests per second,            labels=['model_id', 'version]    ),
            'model_errors': MetricDefinition(
                name='model_errors',              type=MetricType.COUNTER,
                description='Model errors,            labels=['model_id', 'version', 'error_type]  ),
            
            # System metrics
            'cpu_usage': MetricDefinition(
                name='cpu_usage',              type=MetricType.GAUGE,
                description='CPU usage percentage,            labels=['node', 'pod]    ),
            'memory_usage': MetricDefinition(
                name='memory_usage',              type=MetricType.GAUGE,
                description='Memory usage percentage,            labels=['node', 'pod]    ),
            'gpu_usage': MetricDefinition(
                name='gpu_usage',              type=MetricType.GAUGE,
                description='GPU usage percentage,            labels=[node_id]  ),
            
            # Business metrics
            'user_satisfaction': MetricDefinition(
                name='user_satisfaction',              type=MetricType.GAUGE,
                description='User satisfaction score,            labels=['model_id', 'user_segment]    ),
            'revenue_impact': MetricDefinition(
                name='revenue_impact',              type=MetricType.COUNTER,
                description='Revenue impact of predictions,            labels=['model_id', 'prediction_type']
            )
        }
    
    def initialize_metrics(self):
        """Initialize Prometheus metrics"""
        for metric_name, definition in self.metric_definitions.items():
            if definition.type == MetricType.COUNTER:
                self.metrics[metric_name] = Counter(
                    definition.name,
                    definition.description,
                    definition.labels
                )
            elif definition.type == MetricType.GAUGE:
                self.metrics[metric_name] = Gauge(
                    definition.name,
                    definition.description,
                    definition.labels
                )
            elif definition.type == MetricType.HISTOGRAM:
                self.metrics[metric_name] = Histogram(
                    definition.name,
                    definition.description,
                    definition.labels,
                    buckets=definition.buckets
                )
            elif definition.type == MetricType.SUMMARY:
                self.metrics[metric_name] = Summary(
                    definition.name,
                    definition.description,
                    definition.labels
                )
    
    def record_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        try:
            if metric_name not in self.metrics:
                logger.warning(f"Unknown metric: {metric_name}")
                return
            
            metric = self.metrics[metric_name]
            label_values = [labels.get(label, '') for label in self.metric_definitions[metric_name].labels]
            
            if isinstance(metric, Counter):
                metric.labels(*label_values).inc(value)
            elif isinstance(metric, Gauge):
                metric.labels(*label_values).set(value)
            elif isinstance(metric, Histogram):
                metric.labels(*label_values).observe(value)
            elif isinstance(metric, Summary):
                metric.labels(*label_values).observe(value)
            
            # Store in history
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = deque(maxlen=100)
            self.metric_history[metric_name].append({
                'timestamp': datetime.now(),
                'value': value,
                'labels': labels or {}
            })
            
        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {e}")
    
    def _collection_loop(self):
        """Collection loop"""
        while True:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect model metrics
                self._collect_model_metrics()
                
                # Check for anomalies
                self._check_anomalies()
                
                # Sleep
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('cpu_usage', cpu_percent, {'node': 'main', 'pod': 'g'})
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric('memory_usage', memory.percent, {'node': 'main', 'pod': 'g'})
            
            # GPU usage
            if torch.cuda.is_available():
                gpu_usage = GPUtil.getGPUs()[0].load * 100
                self.record_metric('gpu_usage', gpu_usage, {'node_id': 'main', 'gpu_id': '0'})
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_model_metrics(self):
        """Collect model-specific metrics"""
        try:
            # This would integrate with actual model serving infrastructure
            # For now, simulate some metrics
            
            # Simulate model accuracy
            accuracy = np.random.normal(0.85, 0.05)
            self.record_metric('model_accuracy', accuracy, {'model_id': 't_model', 'version': '1'})
            # Simulate model latency
            latency = np.random.exponential(0.5)
            self.record_metric('model_latency', latency, {'model_id': 't_model', 'version': '1'})
            # Simulate throughput
            throughput = np.random.poisson(100)
            self.record_metric('model_throughput', throughput, {'model_id': 't_model', 'version': '1'})
            # Simulate errors
            error_type = np.random.choice(['inference', 'data', 'model'])
            self.record_metric('model_errors', 1, {'model_id': 't_model', 'version': '1', 'error_type': error_type})
            
        except Exception as e:
            logger.error(f"Error collecting model metrics: {e}")
    
    def _check_anomalies(self):
        """Check for anomalies in collected metrics"""
        try:
            for metric_name, history in self.metric_history.items():
                if len(history) < 10: # Need minimum data points
                    continue
                
                # Calculate baseline
                recent_values = [entry['value'] for entry in list(history)[-10:]]
                baseline_mean = np.mean(recent_values)
                baseline_std = np.std(recent_values)
                
                # Check for anomalies
                current_value = recent_values[-1]
                z_score = abs(current_value - baseline_mean) / baseline_std if baseline_std > 0 else 0
                
                if z_score > 3:  # 3-sigma rule
                    self.alert_queue.put({
                        'type': 'anomaly',
                        'metric': metric_name,
                        'value': current_value,
                        'baseline': baseline_mean,
                        'z_score': z_score,
                        'timestamp': datetime.now()
                    })
            
        except Exception as e:
            logger.error(f"Error checking anomalies: {e}")

class AdvancedAnomalyDetector:
    """Advanced anomaly detection using ML models"""
    def __init__(self):
        self.anomaly_model = None
        self.drift_model = None
        self.scaler = StandardScaler()
        
        # Initialize models
        self._initialize_models()
        
        # Anomaly thresholds
        self.anomaly_threshold = 0.8
        self.drift_threshold = 0.7        
        # Historical data
        self.historical_data = deque(maxlen=100)
        self.baseline_distribution = None
    
    def _initialize_models(self):
        """Initialize anomaly detection models"""
        try:
            # Anomaly detection model
            self.anomaly_model = AnomalyDetectionModel(
                input_dim=10,  # Number of features
                hidden_dim=64
            )
            
            # Drift detection model
            self.drift_model = DriftDetectionModel(
                input_dim=10,            hidden_dim=128   )
            
            # Load pre-trained weights if available
            anomaly_path = 'models/anomaly_detection.pth'
            drift_path = 'models/drift_detection.pth'
            
            if os.path.exists(anomaly_path):
                self.anomaly_model.load_state_dict(torch.load(anomaly_path))
            
            if os.path.exists(drift_path):
                self.drift_model.load_state_dict(torch.load(drift_path))
            
            self.anomaly_model.eval()
            self.drift_model.eval()
            
        except Exception as e:
            logger.error(f"Error initializing anomaly detection models: {e}")
    
    def detect_anomalies(self, metrics_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Detect anomalies in metrics data"""
        try:
            # Prepare features
            features = self._prepare_features(metrics_data)
            
            if features is None:
                return {'anomalies': [], 'anomaly_scores': [], 'drift_detected': False}
            
            # Detect anomalies
            with torch.no_grad():
                reconstructed, anomaly_scores = self.anomaly_model(features)
                
                # Calculate reconstruction error
                reconstruction_error = F.mse_loss(features, reconstructed, reduction='mean').item()
                
                # Combine scores
                combined_scores = (anomaly_scores.squeeze() + reconstruction_error) / 2
                
                # Identify anomalies
                anomalies = (combined_scores > self.anomaly_threshold).cpu().numpy()
                
                # Detect drift
                drift_scores, distributions = self.drift_model(features)
                drift_detected = (drift_scores > self.drift_threshold).any().item()
            
            return {
                'anomalies': anomalies.tolist(),
                'anomaly_scores': combined_scores.cpu().numpy().tolist(),
                'drift_detected': drift_detected,
                'drift_scores': drift_scores.cpu().numpy().tolist(),
                'reconstruction_error': reconstruction_error
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {'anomalies': [], 'anomaly_scores': [], 'drift_detected': False}
    
    def _prepare_features(self, metrics_data: Dict[str, List[float]]) -> Optional[torch.Tensor]:
        """Prepare features for anomaly detection"""
        try:
            # Extract numerical features
            feature_names = ['cpu_usage', 'memory_usage', 'gpu_usage', 'model_accuracy', 
                           'model_latency', 'model_throughput', 'model_errors',
                           'user_satisfaction', 'revenue_impact', 'response_time']
            
            features = []
            for name in feature_names:
                if name in metrics_data and metrics_data[name]:
                    features.append(metrics_data[name][-1])  # Latest value
                else:
                    features.append(0.0)  # Default value
            
            # Normalize features
            features_array = np.array(features).reshape(1, -1)
            features_normalized = self.scaler.fit_transform(features_array)
            
            return torch.FloatTensor(features_normalized)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

class AIMonitoringDashboard:
    """Real ML monitoring dashboard with advanced features"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.metrics_collector = RealTimeMetricsCollector()
        self.anomaly_detector = AdvancedAnomalyDetector()
        self.metrics_tracker = MLMetricsTracker()
        self.production_monitor = ProductionMonitor()
        self.monitoring_manager = MonitoringManager()
        self.config_manager = ConfigManager()
        
        # Initialize dashboard
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self._setup_dashboard()
        
        # Monitoring configuration
        self.monitoring_config = {
            'refresh_interval': 5000,
            'history_length': 10,
            'alert_thresholds': {
                'cpu_usage': 80.0,
                'memory_usage': 80.0,
                'gpu_usage': 80.0,
                'model_accuracy': 0.7,
                'model_latency': 2.0
            }
        }
        
        # Start monitoring
        self._start_monitoring()
    
    def _setup_dashboard(self):
        """Setup dashboard layout and callbacks"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("AI Monitoring Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # System Overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Overview"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H4(id="cpu-usage", className="text-center"),
                                    html.P("CPU Usage", className="text-center")
                                ]),
                                dbc.Col([
                                    html.H4(id="memory-usage", className="text-center"),
                                    html.P("Memory Usage", className="text-center")
                                ]),
                                dbc.Col([
                                    html.H4(id="gpu-usage", className="text-center"),
                                    html.P("GPU Usage", className="text-center")
                                ])
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Model Performance
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Model Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="model-performance-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Anomaly Detection"),
                        dbc.CardBody([
                            dcc.Graph(id="anomaly-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Metrics History
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Metrics History"),
                        dbc.CardBody([
                            dcc.Graph(id="metrics-history-chart")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Alerts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Alerts"),
                        dbc.CardBody([
                            html.Div(id="alerts-list")
                        ])
                    ])
                ])
            ]),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',          interval=self.monitoring_config['refresh_interval'],
                n_intervals=0
            )
        ], fluid=True)
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            [Output("cpu-usage", "children"),
             Output("memory-usage", "children"),
             Output("gpu-usage", "children")],            [Input("interval-component", "n_intervals")]
        )
        def update_system_metrics(n):
            try:
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                
                gpu_usage = 0
                if torch.cuda.is_available():
                    gpu_usage = GPUtil.getGPUs()[0].load * 100
                
                return f"{cpu_usage:0.1f}%, {memory_usage:.1f}%, {gpu_usage:.1f}%"
                
            except Exception as e:
                self.logger.error(f"Error updating system metrics: {e}")
                return "N/A", "N/A", "N/A"
        
        @self.app.callback(
            Output("model-performance-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_model_performance(n):
            try:
                # Get model performance data
                accuracy_data = self.metrics_collector.metric_history.get('model_accuracy', [])
                latency_data = self.metrics_collector.metric_history.get('model_latency', [])
                
                if not accuracy_data or not latency_data:
                    return go.Figure()
                
                # Prepare data
                timestamps = [entry['timestamp'] for entry in accuracy_data[-50:]]
                accuracy_values = [entry['value'] for entry in accuracy_data[-50:]]
                latency_values = [entry['value'] for entry in latency_data[-50:]]
                
                # Create figure
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Model Accuracy', 'Model Latency')
                )
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=accuracy_values, name='Accuracy', line=dict(color='blue')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=timestamps, y=latency_values, name='Latency', line=dict(color='red')),
                    row=2, col=1
                )
                
                fig.update_layout(height=400, showlegend=False)
                
                return fig
                
        except Exception as e:
                self.logger.error(f"Error updating model performance: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output("anomaly-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_anomaly_chart(n):
            try:
                # Get anomaly data
                anomaly_data = self._get_anomaly_data()
                
                if not anomaly_data:
                    return go.Figure()
                
                # Create anomaly chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=anomaly_data['timestamps'],
                    y=anomaly_data['scores'],
                    mode='lines+markers',
                    name='Anomaly Score',
                    line=dict(color='orange')
                ))
                
                # Add threshold line
                fig.add_hline(y=self.anomaly_detector.anomaly_threshold, 
                            line_dash='dash', line_color="red",
                            annotation_text="Threshold")
                
                fig.update_layout(
                    title="Anomaly Detection Scores",
                    xaxis_title="Time",
                    yaxis_title="Anomaly Score",
                    height=400
                )
                
                return fig
                
            except Exception as e:
                self.logger.error(f"Error updating anomaly chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output("metrics-history-chart", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_metrics_history(n):
            try:
                # Get all metrics data
                metrics_data = {}
                for metric_name, history in self.metrics_collector.metric_history.items():
                    if history:
                        metrics_data[metric_name] = {
                            'timestamps': [entry['timestamp'] for entry in history[-100:]],
                            'values': [entry['value'] for entry in history[-100:]]
                        }
                
                if not metrics_data:
                    return go.Figure()
                
                # Create figure
                fig = go.Figure()
                
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
                for i, (metric_name, data) in enumerate(metrics_data.items()):
                    fig.add_trace(go.Scatter(
                        x=data['timestamps'],
                        y=data['values'],
                        mode='lines',
                        name=metric_name,
                        line=dict(color=colors[i % len(colors)])
                    ))
                
                fig.update_layout(
                    title="Metrics History",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    height=400
                )
                
                return fig
            
        except Exception as e:
                self.logger.error(f"Error updating metrics history: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output("alerts-list", "children"),
            [Input("interval-component", "n_intervals")]
        )
        def update_alerts(n):
            try:
                # Get recent alerts
                alerts = []
                while not self.metrics_collector.alert_queue.empty():
                    try:
                        alert = self.metrics_collector.alert_queue.get_nowait()
                        alerts.append(alert)
                    except queue.Empty:
                        break
                
                if not alerts:
                    return html.P("No recent alerts, className='text-muted'")
                
                # Create alert list
                alert_items = []
                for alert in alerts[-10:]:  # Show last 10 alerts
                    alert_items.append(html.Div([
                        html.Strong(f"{alert['type'].title()}: {alert['metric']}"),
                        html.Br(),
                        html.Small(f"Value: {alert['value']:.2f}, Z-Score: {alert['z_score']:.2f}"),
                        html.Br(),
                        html.Small(alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')),
                        html.Hr()
                    ]))
                
                return alert_items
            
        except Exception as e:
                self.logger.error(f"Error updating alerts: {e}")
                return html.P("Error loading alerts, className='text-danger'")
    
    def _get_anomaly_data(self) -> Dict[str, List]:
        """Get anomaly detection data"""
        try:
            # Prepare metrics data for anomaly detection
            metrics_data = {}
            for metric_name, history in self.metrics_collector.metric_history.items():
                if history:
                    metrics_data[metric_name] = [entry['value'] for entry in history[-50:]]
            # Detect anomalies
            anomaly_result = self.anomaly_detector.detect_anomalies(metrics_data)
            
            if not anomaly_result['anomaly_scores']:
                return {}
            
            # Prepare timestamps
            timestamps = []
            for metric_name, history in self.metrics_collector.metric_history.items():
                if history:
                    timestamps = [entry['timestamp'] for entry in history[-len(anomaly_result['anomaly_scores']):]]
                    break
            
            return {
                'timestamps': timestamps,
                'scores': anomaly_result['anomaly_scores']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting anomaly data: {e}")
            return {}   
    def _start_monitoring(self):
        """Start monitoring services"""
        try:
            # Start metrics collection
            self.logger.info("Starting metrics collection...")
            
            # Start anomaly detection
            self.logger.info("Starting anomaly detection...")
            
            # Start dashboard
            self.logger.info("Starting monitoring dashboard...")
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {e}")
    
    def run_dashboard(self, host: str = '0.0.0.0', port: int = 8500, debug: bool = False):
        """Run the monitoring dashboard"""
        try:
            self.logger.info(f"Starting dashboard on {host}:{port}")
            self.app.run_server(host=host, port=port, debug=debug)
        except Exception as e:
            self.logger.error(f"Error running dashboard: {e}")
    
    async def get_system_health(self) -> Dict:
        """Get system health metrics"""
        try:
            health_metrics = {
                'status': 'healthy',
                'device': str(self.device),
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'metrics_collector_status': 'active',
                'anomaly_detector_status': 'active',
                'dashboard_status': 'running',
                'performance_metrics': {
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'gpu_usage': GPUtil.getGPUs()[0].load * 100 if torch.cuda.is_available() else 0,
                    'active_metrics': len(self.metrics_collector.metric_history)
                }
            }
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {'status': 'error', 'error': str(e)}

# Initialize service
ai_monitoring_dashboard = AIMonitoringDashboard() 

# Add Flask app and API for explainability endpoint if not present
app = Flask(__name__)
api = Api(app, version='1.0', title='AI Monitoring Dashboard', description='Advanced ML Monitoring and Explainability', doc='/docs')

# Add data validator
data_validator = AdvancedDataValidator(logger=logger)

explain_input = api.model('ExplainInput', {
    'model_type': fields.String(required=True, description='Model type (anomaly, drift)'),
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
            if model_type == 'anomaly':
                model = self.anomaly_model
            elif model_type == 'drift':
                model = self.drift_model
            else:
                logger.error(f'Unknown model_type: {model_type}')
                return {'error': 'Unknown model_type'}, 400
            explainer = shap.Explainer(lambda x: model(x)[1].detach().numpy(), features)
            shap_values = explainer(features)
            logger.info(f'Explanation generated for {model_type} model')
            return {'shap_values': shap_values.values.tolist(), 'base_values': shap_values.base_values.tolist()}
        except Exception as e:
            logger.error(f'Explainability error: {e}')
            return {'error': str(e)}, 500 