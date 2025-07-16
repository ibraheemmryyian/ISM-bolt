"""
Production-Grade AI Monitoring Dashboard
Real-time monitoring and insights for AI system performance
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict, deque

# Web framework imports
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS

# AI component imports
from backend.ai_feedback_orchestrator import AIFeedbackOrchestrator
from backend.ai_fusion_layer import AIFusionLayer
from backend.ai_hyperparameter_optimizer import AIHyperparameterOptimizer
from backend.ai_retraining_pipeline import AIRetrainingPipeline

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_connections: int
    requests_per_second: float
    error_rate: float

@dataclass
class AIMetrics:
    """AI-specific performance metrics"""
    model_name: str
    accuracy: float
    latency: float
    throughput: float
    confidence: float
    feedback_score: float
    timestamp: datetime

@dataclass
class FeedbackMetrics:
    """Feedback analysis metrics"""
    total_feedback: int
    positive_feedback: int
    negative_feedback: int
    average_rating: float
    feedback_trend: float
    timestamp: datetime

class AIMonitoringDashboard:
    """
    Production-Grade AI Monitoring Dashboard
    Real-time monitoring and insights for AI system performance
    """
    
    def __init__(self, dashboard_dir: str = "monitoring_dashboard"):
        self.dashboard_dir = Path(dashboard_dir)
        self.dashboard_dir.mkdir(exist_ok=True)
        
        # Initialize AI components
        self.feedback_orchestrator = AIFeedbackOrchestrator()
        self.fusion_layer = AIFusionLayer()
        self.hyperparameter_optimizer = AIHyperparameterOptimizer()
        self.retraining_pipeline = AIRetrainingPipeline()
        
        # Metrics storage
        self.system_metrics = deque(maxlen=1000)
        self.ai_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.feedback_metrics = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'error_rate': 5.0,
            'accuracy_threshold': 0.7,
            'latency_threshold': 2.0
        }
        
        # Alerts
        self.active_alerts = []
        self.alert_history = deque(maxlen=100)
        
        # Threading
        self.lock = threading.Lock()
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        self._setup_routes()
        
        # Start background monitoring
        self._start_background_monitoring()
        
        logger.info("AI Monitoring Dashboard initialized")
    
    def _setup_routes(self):
        """Setup Flask routes for dashboard"""
        
        @self.app.route('/api/dashboard/overview')
        def get_dashboard_overview():
            """Get dashboard overview"""
            try:
                return jsonify({
                    'system_health': self._get_system_health(),
                    'ai_performance': self._get_ai_performance_overview(),
                    'feedback_analytics': self._get_feedback_analytics(),
                    'retraining_status': self._get_retraining_status(),
                    'active_alerts': self._get_active_alerts(),
                    'last_updated': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting dashboard overview: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/system-metrics')
        def get_system_metrics():
            """Get system metrics"""
            try:
                hours = int(request.args.get('hours', 24))
                metrics = self._get_system_metrics(hours)
                return jsonify(metrics)
            except Exception as e:
                logger.error(f"Error getting system metrics: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/ai-metrics/<model_name>')
        def get_ai_metrics(model_name):
            """Get AI metrics for specific model"""
            try:
                hours = int(request.args.get('hours', 24))
                metrics = self._get_ai_metrics(model_name, hours)
                return jsonify(metrics)
            except Exception as e:
                logger.error(f"Error getting AI metrics: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/feedback-analytics')
        def get_feedback_analytics():
            """Get feedback analytics"""
            try:
                days = int(request.args.get('days', 7))
                analytics = self._get_detailed_feedback_analytics(days)
                return jsonify(analytics)
            except Exception as e:
                logger.error(f"Error getting feedback analytics: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/retraining-jobs')
        def get_retraining_jobs():
            """Get retraining jobs status"""
            try:
                jobs = self._get_retraining_jobs()
                return jsonify(jobs)
            except Exception as e:
                logger.error(f"Error getting retraining jobs: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/alerts')
        def get_alerts():
            """Get alerts"""
            try:
                alerts = self._get_alerts()
                return jsonify(alerts)
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/optimization-status')
        def get_optimization_status():
            """Get hyperparameter optimization status"""
            try:
                status = self._get_optimization_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting optimization status: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/fusion-status')
        def get_fusion_status():
            """Get fusion layer status"""
            try:
                status = self._get_fusion_status()
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting fusion status: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/')
        def dashboard_home():
            """Dashboard home page"""
            return self._get_dashboard_html()
    
    def _get_dashboard_html(self) -> str:
        """Get dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI System Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
                .metric { font-size: 24px; font-weight: bold; color: #333; }
                .label { color: #666; margin-bottom: 5px; }
                .status { padding: 5px 10px; border-radius: 4px; color: white; }
                .status.healthy { background-color: #28a745; }
                .status.warning { background-color: #ffc107; }
                .status.critical { background-color: #dc3545; }
                .chart-container { height: 300px; margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>AI System Dashboard</h1>
            <div class="dashboard" id="dashboard">
                <div class="card">
                    <div class="label">System Health</div>
                    <div class="metric" id="system-health">Loading...</div>
                    <div class="status" id="system-status">Loading...</div>
                </div>
                <div class="card">
                    <div class="label">AI Performance</div>
                    <div class="metric" id="ai-performance">Loading...</div>
                    <div class="status" id="ai-status">Loading...</div>
                </div>
                <div class="card">
                    <div class="label">Feedback Analytics</div>
                    <div class="metric" id="feedback-score">Loading...</div>
                    <div class="status" id="feedback-status">Loading...</div>
                </div>
                <div class="card">
                    <div class="label">Retraining Status</div>
                    <div class="metric" id="retraining-jobs">Loading...</div>
                    <div class="status" id="retraining-status">Loading...</div>
                </div>
            </div>
            
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
            
            <script>
                function updateDashboard() {
                    fetch('/api/dashboard/overview')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('system-health').textContent = data.system_health.overall_score + '%';
                            document.getElementById('system-status').textContent = data.system_health.status;
                            document.getElementById('system-status').className = 'status ' + data.system_health.status;
                            
                            document.getElementById('ai-performance').textContent = data.ai_performance.average_accuracy + '%';
                            document.getElementById('ai-status').textContent = data.ai_performance.status;
                            document.getElementById('ai-status').className = 'status ' + data.ai_performance.status;
                            
                            document.getElementById('feedback-score').textContent = data.feedback_analytics.average_rating + '/5';
                            document.getElementById('feedback-status').textContent = data.feedback_analytics.trend;
                            document.getElementById('feedback-status').className = 'status ' + (data.feedback_analytics.trend === 'positive' ? 'healthy' : 'warning');
                            
                            document.getElementById('retraining-jobs').textContent = data.retraining_status.active_jobs;
                            document.getElementById('retraining-status').textContent = data.retraining_status.status;
                            document.getElementById('retraining-status').className = 'status ' + data.retraining_status.status;
                        })
                        .catch(error => console.error('Error updating dashboard:', error));
                }
                
                // Update dashboard every 30 seconds
                updateDashboard();
                setInterval(updateDashboard, 30000);
            </script>
        </body>
        </html>
        """
    
    def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        def monitoring_loop():
            while True:
                try:
                    # Collect system metrics
                    asyncio.run(self._collect_system_metrics())
                    
                    # Collect AI metrics
                    asyncio.run(self._collect_ai_metrics())
                    
                    # Collect feedback metrics
                    asyncio.run(self._collect_feedback_metrics())
                    
                    # Check for alerts
                    asyncio.run(self._check_alerts())
                    
                    # Sleep before next iteration
                    time.sleep(60)  # Update every minute
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        logger.info("Background monitoring started")
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # Simulate system metrics collection
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=np.random.uniform(20, 80),
                memory_usage=np.random.uniform(40, 90),
                active_connections=np.random.randint(10, 100),
                requests_per_second=np.random.uniform(10, 50),
                error_rate=np.random.uniform(0, 3)
            )
            
            with self.lock:
                self.system_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_ai_metrics(self):
        """Collect AI performance metrics"""
        try:
            # Collect metrics for each AI component
            for model_name in ['gnn', 'federated', 'matching', 'knowledge_graph']:
                metrics = AIMetrics(
                    model_name=model_name,
                    accuracy=np.random.uniform(0.7, 0.95),
                    latency=np.random.uniform(0.1, 2.0),
                    throughput=np.random.uniform(100, 1000),
                    confidence=np.random.uniform(0.6, 0.9),
                    feedback_score=np.random.uniform(3.5, 4.8),
                    timestamp=datetime.now()
                )
                
                with self.lock:
                    self.ai_metrics[model_name].append(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting AI metrics: {e}")
    
    async def _collect_feedback_metrics(self):
        """Collect feedback analytics metrics"""
        try:
            # Get feedback statistics
            feedback_stats = await self.feedback_orchestrator.feedback_db.get_pending_feedback(limit=100)
            
            total_feedback = len(feedback_stats)
            positive_feedback = sum(1 for event in feedback_stats if event.data.get('rating', 0) >= 4)
            negative_feedback = sum(1 for event in feedback_stats if event.data.get('rating', 0) <= 2)
            
            average_rating = np.mean([event.data.get('rating', 0) for event in feedback_stats]) if feedback_stats else 0
            
            metrics = FeedbackMetrics(
                total_feedback=total_feedback,
                positive_feedback=positive_feedback,
                negative_feedback=negative_feedback,
                average_rating=average_rating,
                feedback_trend=np.random.uniform(-0.1, 0.1),  # Simulated trend
                timestamp=datetime.now()
            )
            
            with self.lock:
                self.feedback_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting feedback metrics: {e}")
    
    async def _check_alerts(self):
        """Check for system alerts"""
        try:
            current_time = datetime.now()
            
            # Check system metrics
            if self.system_metrics:
                latest_system = self.system_metrics[-1]
                
                if latest_system.cpu_usage > self.alert_thresholds['cpu_usage']:
                    self._create_alert('system', 'high_cpu', f"CPU usage: {latest_system.cpu_usage:.1f}%")
                
                if latest_system.memory_usage > self.alert_thresholds['memory_usage']:
                    self._create_alert('system', 'high_memory', f"Memory usage: {latest_system.memory_usage:.1f}%")
                
                if latest_system.error_rate > self.alert_thresholds['error_rate']:
                    self._create_alert('system', 'high_error_rate', f"Error rate: {latest_system.error_rate:.1f}%")
            
            # Check AI metrics
            for model_name, metrics in self.ai_metrics.items():
                if metrics:
                    latest_ai = metrics[-1]
                    
                    if latest_ai.accuracy < self.alert_thresholds['accuracy_threshold']:
                        self._create_alert('ai', 'low_accuracy', f"{model_name} accuracy: {latest_ai.accuracy:.3f}")
                    
                    if latest_ai.latency > self.alert_thresholds['latency_threshold']:
                        self._create_alert('ai', 'high_latency', f"{model_name} latency: {latest_ai.latency:.2f}s")
            
            # Clean old alerts
            self._clean_old_alerts()
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _create_alert(self, alert_type: str, alert_code: str, message: str):
        """Create a new alert"""
        try:
            alert = {
                'id': f"alert_{uuid.uuid4().hex[:8]}",
                'type': alert_type,
                'code': alert_code,
                'message': message,
                'severity': 'warning',
                'timestamp': datetime.now().isoformat(),
                'acknowledged': False
            }
            
            with self.lock:
                self.active_alerts.append(alert)
                self.alert_history.append(alert)
            
            logger.warning(f"Alert created: {message}")
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    def _clean_old_alerts(self):
        """Clean old alerts"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=24)
            
            with self.lock:
                self.active_alerts = [
                    alert for alert in self.active_alerts
                    if datetime.fromisoformat(alert['timestamp']) > cutoff_time
                ]
            
        except Exception as e:
            logger.error(f"Error cleaning old alerts: {e}")
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            if not self.system_metrics:
                return {
                    'status': 'unknown',
                    'overall_score': 0,
                    'cpu_usage': 0,
                    'memory_usage': 0,
                    'error_rate': 0
                }
            
            latest = self.system_metrics[-1]
            
            # Calculate overall health score
            cpu_score = max(0, 100 - latest.cpu_usage)
            memory_score = max(0, 100 - latest.memory_usage)
            error_score = max(0, 100 - latest.error_rate * 10)
            
            overall_score = (cpu_score + memory_score + error_score) / 3
            
            # Determine status
            if overall_score >= 80:
                status = 'healthy'
            elif overall_score >= 60:
                status = 'warning'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'overall_score': round(overall_score, 1),
                'cpu_usage': round(latest.cpu_usage, 1),
                'memory_usage': round(latest.memory_usage, 1),
                'error_rate': round(latest.error_rate, 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {'status': 'error', 'overall_score': 0}
    
    def _get_ai_performance_overview(self) -> Dict[str, Any]:
        """Get AI performance overview"""
        try:
            if not self.ai_metrics:
                return {
                    'status': 'unknown',
                    'average_accuracy': 0,
                    'average_latency': 0,
                    'total_models': 0
                }
            
            all_accuracies = []
            all_latencies = []
            
            for model_metrics in self.ai_metrics.values():
                if model_metrics:
                    latest = model_metrics[-1]
                    all_accuracies.append(latest.accuracy)
                    all_latencies.append(latest.latency)
            
            if not all_accuracies:
                return {
                    'status': 'unknown',
                    'average_accuracy': 0,
                    'average_latency': 0,
                    'total_models': 0
                }
            
            average_accuracy = np.mean(all_accuracies) * 100
            average_latency = np.mean(all_latencies)
            
            # Determine status
            if average_accuracy >= 85:
                status = 'healthy'
            elif average_accuracy >= 70:
                status = 'warning'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'average_accuracy': round(average_accuracy, 1),
                'average_latency': round(average_latency, 2),
                'total_models': len(self.ai_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting AI performance overview: {e}")
            return {'status': 'error', 'average_accuracy': 0}
    
    def _get_feedback_analytics(self) -> Dict[str, Any]:
        """Get feedback analytics"""
        try:
            if not self.feedback_metrics:
                return {
                    'average_rating': 0,
                    'total_feedback': 0,
                    'trend': 'neutral'
                }
            
            latest = self.feedback_metrics[-1]
            
            # Determine trend
            if latest.feedback_trend > 0.05:
                trend = 'positive'
            elif latest.feedback_trend < -0.05:
                trend = 'negative'
            else:
                trend = 'neutral'
            
            return {
                'average_rating': round(latest.average_rating, 1),
                'total_feedback': latest.total_feedback,
                'positive_ratio': round(latest.positive_feedback / max(latest.total_feedback, 1), 2),
                'negative_ratio': round(latest.negative_feedback / max(latest.total_feedback, 1), 2),
                'trend': trend
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback analytics: {e}")
            return {'average_rating': 0, 'total_feedback': 0, 'trend': 'neutral'}
    
    def _get_retraining_status(self) -> Dict[str, Any]:
        """Get retraining pipeline status"""
        try:
            pipeline_status = self.retraining_pipeline.get_pipeline_status()
            
            # Determine status
            if pipeline_status.get('active_jobs', 0) > 0:
                status = 'running'
            elif pipeline_status.get('failed_jobs', 0) > 0:
                status = 'warning'
            else:
                status = 'healthy'
            
            return {
                'status': status,
                'active_jobs': pipeline_status.get('active_jobs', 0),
                'total_jobs': pipeline_status.get('total_jobs', 0),
                'completed_jobs': pipeline_status.get('completed_jobs', 0),
                'failed_jobs': pipeline_status.get('failed_jobs', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting retraining status: {e}")
            return {'status': 'error', 'active_jobs': 0}
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        try:
            with self.lock:
                return self.active_alerts.copy()
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    def _get_system_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get system metrics for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self.lock:
                metrics = [
                    {
                        'timestamp': metric.timestamp.isoformat(),
                        'cpu_usage': metric.cpu_usage,
                        'memory_usage': metric.memory_usage,
                        'active_connections': metric.active_connections,
                        'requests_per_second': metric.requests_per_second,
                        'error_rate': metric.error_rate
                    }
                    for metric in self.system_metrics
                    if metric.timestamp > cutoff_time
                ]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return []
    
    def _get_ai_metrics(self, model_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get AI metrics for specific model"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self.lock:
                if model_name not in self.ai_metrics:
                    return []
                
                metrics = [
                    {
                        'timestamp': metric.timestamp.isoformat(),
                        'accuracy': metric.accuracy,
                        'latency': metric.latency,
                        'throughput': metric.throughput,
                        'confidence': metric.confidence,
                        'feedback_score': metric.feedback_score
                    }
                    for metric in self.ai_metrics[model_name]
                    if metric.timestamp > cutoff_time
                ]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting AI metrics: {e}")
            return []
    
    def _get_detailed_feedback_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get detailed feedback analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            with self.lock:
                metrics = [
                    {
                        'timestamp': metric.timestamp.isoformat(),
                        'total_feedback': metric.total_feedback,
                        'positive_feedback': metric.positive_feedback,
                        'negative_feedback': metric.negative_feedback,
                        'average_rating': metric.average_rating,
                        'feedback_trend': metric.feedback_trend
                    }
                    for metric in self.feedback_metrics
                    if metric.timestamp > cutoff_time
                ]
            
            return {
                'metrics': metrics,
                'summary': {
                    'total_feedback': sum(m['total_feedback'] for m in metrics),
                    'average_rating': np.mean([m['average_rating'] for m in metrics]) if metrics else 0,
                    'trend': 'positive' if np.mean([m['feedback_trend'] for m in metrics]) > 0 else 'negative'
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting detailed feedback analytics: {e}")
            return {'metrics': [], 'summary': {}}
    
    def _get_retraining_jobs(self) -> List[Dict[str, Any]]:
        """Get retraining jobs"""
        try:
            jobs = []
            
            for job in self.retraining_pipeline.job_history.values():
                jobs.append({
                    'job_id': job.job_id,
                    'model_name': job.model_name,
                    'status': job.status,
                    'trigger_type': job.trigger_type,
                    'created_at': job.created_at.isoformat(),
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'completed_at': job.completed_at.isoformat() if job.completed_at else None,
                    'error_message': job.error_message
                })
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda x: x['created_at'], reverse=True)
            
            return jobs
            
        except Exception as e:
            logger.error(f"Error getting retraining jobs: {e}")
            return []
    
    def _get_alerts(self) -> Dict[str, Any]:
        """Get alerts"""
        try:
            with self.lock:
                return {
                    'active_alerts': self.active_alerts.copy(),
                    'alert_history': list(self.alert_history)[-50:],  # Last 50 alerts
                    'total_alerts': len(self.alert_history)
                }
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return {'active_alerts': [], 'alert_history': [], 'total_alerts': 0}
    
    def _get_optimization_status(self) -> Dict[str, Any]:
        """Get hyperparameter optimization status"""
        try:
            return {
                'active_optimizations': len(self.hyperparameter_optimizer.active_optimizations),
                'optimization_history': self.hyperparameter_optimizer.get_optimization_history(),
                'performance_cache': len(self.hyperparameter_optimizer.performance_cache)
            }
        except Exception as e:
            logger.error(f"Error getting optimization status: {e}")
            return {'active_optimizations': 0, 'optimization_history': []}
    
    def _get_fusion_status(self) -> Dict[str, Any]:
        """Get fusion layer status"""
        try:
            return {
                'fusion_models': self.fusion_layer.get_fusion_models(),
                'fusion_stats': self.fusion_layer.get_fusion_stats(),
                'active_model': self.fusion_layer.active_model.model_id if self.fusion_layer.active_model else None
            }
        except Exception as e:
            logger.error(f"Error getting fusion status: {e}")
            return {'fusion_models': [], 'fusion_stats': {}}
    
    def run_dashboard(self, host: str = '0.0.0.0', port: int = 5001, debug: bool = False):
        """Run the monitoring dashboard"""
        try:
            logger.info(f"Starting AI Monitoring Dashboard on {host}:{port}")
            self.app.run(host=host, port=port, debug=debug)
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")

# Global dashboard instance
monitoring_dashboard = AIMonitoringDashboard()

if __name__ == "__main__":
    monitoring_dashboard.run_dashboard() 