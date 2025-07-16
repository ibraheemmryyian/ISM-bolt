#!/usr/bin/env python3
"""
🏥 SYSTEM HEALTH MONITOR
Comprehensive monitoring and alerting system for SymbioFlows
Features:
- Real-time health checks
- Performance monitoring
- Error tracking
- Circuit breaker monitoring
- Database health
- API endpoint monitoring
- Alert system
- Performance metrics
"""

import os
import asyncio
import aiohttp
import logging
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sqlite3
import redis
import psutil
import requests
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_health.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Patch StreamHandler to use UTF-8 encoding if possible
try:
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream.reconfigure(encoding='utf-8')
except Exception:
    pass
logger = logging.getLogger(__name__)

@dataclass
class HealthMetric:
    """Health metric data structure"""
    metric_id: str
    component: str
    metric_name: str
    value: float
    unit: str
    status: str  # healthy, warning, critical
    timestamp: datetime
    details: Dict[str, Any]

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    severity: str  # info, warning, critical
    component: str
    message: str
    timestamp: datetime
    resolved: bool
    resolution_time: Optional[datetime]

class SystemHealthMonitor:
    """
    Comprehensive system health monitor
    """
    
    def __init__(self):
        # Configuration
        self.check_interval = 30  # seconds
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0,
            'error_rate': 10.0,
            'database_connections': 80
        }
        
        # Database
        self.db_path = Path("system_health.db")
        self._init_database()
        
        # Redis
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=1, socket_timeout=5)
            self.redis_client.ping()
            self.redis_available = True
        except:
            self.redis_available = False
            logger.warning("Redis not available for health monitoring")
        
        # Monitoring state
        self.metrics = {}
        self.alerts = []
        self.health_status = 'healthy'
        self.last_check = datetime.now()
        
        # Threading
        self.monitoring_thread = None
        self.stop_monitoring = False
        self.lock = threading.Lock()
        
        # Email configuration
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'email': os.getenv('ALERT_EMAIL', 'alerts@symbioflows.com'),
            'password': os.getenv('ALERT_PASSWORD', ''),
            'recipients': os.getenv('ALERT_RECIPIENTS', 'admin@symbioflows.com').split(',')
        }
        
        logger.info("🏥 System Health Monitor initialized")
    
    def _init_database(self):
        """Initialize health monitoring database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_metrics (
                    metric_id TEXT PRIMARY KEY,
                    component TEXT,
                    metric_name TEXT,
                    value REAL,
                    unit TEXT,
                    status TEXT,
                    timestamp TEXT,
                    details TEXT
                )
            ''')
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    severity TEXT,
                    component TEXT,
                    message TEXT,
                    timestamp TEXT,
                    resolved INTEGER,
                    resolution_time TEXT
                )
            ''')
            
            # Create performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metric_id TEXT PRIMARY KEY,
                    metric_name TEXT,
                    value REAL,
                    timestamp TEXT,
                    context TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Health monitoring database initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("🚀 Health monitoring started")
    
    def stop_monitoring_service(self):
        """Stop health monitoring"""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("🛑 Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_monitoring:
            try:
                self._run_health_checks()
                self._check_alerts()
                self._update_health_status()
                self._save_metrics()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _run_health_checks(self):
        """Run all health checks"""
        try:
            # System metrics
            self._check_system_metrics()
            
            # Database health
            self._check_database_health()
            
            # API endpoints
            self._check_api_endpoints()
            
            # Service health
            self._check_service_health()
            
            # Performance metrics
            self._check_performance_metrics()
            
            self.last_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Error running health checks: {e}")
    
    def _check_system_metrics(self):
        """Check system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_metric('system', 'cpu_usage', cpu_percent, '%', 
                              'critical' if cpu_percent > self.alert_thresholds['cpu_usage'] else 'healthy')
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._record_metric('system', 'memory_usage', memory_percent, '%',
                              'critical' if memory_percent > self.alert_thresholds['memory_usage'] else 'healthy')
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._record_metric('system', 'disk_usage', disk_percent, '%',
                              'critical' if disk_percent > self.alert_thresholds['disk_usage'] else 'healthy')
            
            # Network I/O
            network = psutil.net_io_counters()
            self._record_metric('system', 'network_bytes_sent', network.bytes_sent, 'bytes', 'healthy')
            self._record_metric('system', 'network_bytes_recv', network.bytes_recv, 'bytes', 'healthy')
            
        except Exception as e:
            logger.error(f"Error checking system metrics: {e}")
    
    def _check_database_health(self):
        """Check database health"""
        try:
            # Test database connection
            start_time = time.time()
            conn = sqlite3.connect(self.db_path, timeout=10)
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            response_time = time.time() - start_time
            conn.close()
            
            self._record_metric('database', 'response_time', response_time, 'seconds',
                              'critical' if response_time > self.alert_thresholds['response_time'] else 'healthy')
            
            # Check database size
            db_size = self.db_path.stat().st_size / (1024 * 1024)  # MB
            self._record_metric('database', 'size', db_size, 'MB', 'healthy')
            
            # Check table counts
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            
            self._record_metric('database', 'table_count', len(tables), 'count', 'healthy')
            
        except Exception as e:
            logger.error(f"Error checking database health: {e}")
            self._record_metric('database', 'connection_status', 0, 'status', 'critical')
    
    def _check_api_endpoints(self):
        """Check API endpoint health"""
        health_checks = [
            {'name': 'backend_health', 'url': os.environ.get('BACKEND_HEALTH_URL', 'http://localhost:3000/health'), 'method': 'GET'},
            {'name': 'api_health', 'url': os.environ.get('API_HEALTH_URL', 'http://localhost:3000/api/health'), 'method': 'GET'},
            {'name': 'companies_api', 'url': os.environ.get('COMPANIES_API_URL', 'http://localhost:3000/api/companies'), 'method': 'GET'},
            {'name': 'materials_api', 'url': os.environ.get('MATERIALS_API_URL', 'http://localhost:3000/api/materials'), 'method': 'GET'},
            {'name': 'ai_listings_api', 'url': os.environ.get('AI_LISTINGS_API_URL', 'http://localhost:3000/api/ai/listings'), 'method': 'GET'},
        ]
        
        for endpoint in health_checks:
            try:
                start_time = time.time()
                response = requests.get(endpoint['url'], timeout=10)
                response_time = time.time() - start_time
                
                status = 'healthy' if response.status_code == 200 else 'warning'
                if response_time > self.alert_thresholds['response_time']:
                    status = 'critical'
                
                self._record_metric('api', f"{endpoint['name']}_response_time", response_time, 'seconds', status)
                self._record_metric('api', f"{endpoint['name']}_status_code", response.status_code, 'code', status)
                
            except Exception as e:
                logger.error(f"Error checking endpoint {endpoint['name']}: {e}")
                self._record_metric('api', f"{endpoint['name']}_status", 0, 'status', 'critical')
    
    def _check_service_health(self):
        """Check service health"""
        services = [
            {'name': 'onboarding_system', 'file': 'bulletproof_onboarding_system.py'},
            {'name': 'ai_listings_generator', 'file': 'ultra_ai_listings_generator.py'},
            {'name': 'matching_engine', 'file': 'revolutionary_ai_matching.py'},
            {'name': 'logistics_service', 'file': 'logistics_cost_engine.py'}
        ]
        
        for service in services:
            try:
                file_path = Path(f"backend/{service['file']}")
                if file_path.exists():
                    # Check if file is recent (modified in last hour)
                    modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    is_recent = (datetime.now() - modified_time).total_seconds() < 3600
                    
                    self._record_metric('service', f"{service['name']}_status", 1 if is_recent else 0, 'status', 'healthy')
                else:
                    self._record_metric('service', f"{service['name']}_status", 0, 'status', 'critical')
                    
            except Exception as e:
                logger.error(f"Error checking service {service['name']}: {e}")
                self._record_metric('service', f"{service['name']}_status", 0, 'status', 'critical')
    
    def _check_performance_metrics(self):
        """Check performance metrics"""
        try:
            # Check active processes
            active_processes = len(psutil.pids())
            self._record_metric('performance', 'active_processes', active_processes, 'count', 'healthy')
            
            # Check Python processes
            python_processes = len([p for p in psutil.process_iter(['name']) if 'python' in p.info['name'].lower()])
            self._record_metric('performance', 'python_processes', python_processes, 'count', 'healthy')
            
            # Check memory usage by Python processes
            python_memory = 0
            for proc in psutil.process_iter(['name', 'memory_info']):
                if 'python' in proc.info['name'].lower():
                    python_memory += proc.info['memory_info'].rss / (1024 * 1024)  # MB
            
            self._record_metric('performance', 'python_memory_usage', python_memory, 'MB', 'healthy')
            
        except Exception as e:
            logger.error(f"Error checking performance metrics: {e}")
    
    def _record_metric(self, component: str, metric_name: str, value: float, unit: str, status: str):
        """Record a health metric"""
        metric_id = f"{component}_{metric_name}_{int(time.time())}"
        metric = HealthMetric(
            metric_id=metric_id,
            component=component,
            metric_name=metric_name,
            value=value,
            unit=unit,
            status=status,
            timestamp=datetime.now(),
            details={}
        )
        
        with self.lock:
            self.metrics[metric_id] = metric
        
        # Check for alerts
        if status in ['warning', 'critical']:
            self._create_alert(component, metric_name, value, status)
    
    def _create_alert(self, component: str, metric_name: str, value: float, severity: str):
        """Create an alert"""
        alert_id = f"alert_{component}_{metric_name}_{int(time.time())}"
        message = f"{component.upper()} {metric_name}: {value} ({severity.upper()})"
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            component=component,
            message=message,
            timestamp=datetime.now(),
            resolved=False,
            resolution_time=None
        )
        
        with self.lock:
            self.alerts.append(alert)
        
        # Send email alert for critical issues
        if severity == 'critical':
            self._send_email_alert(alert)
        
        logger.warning(f"🚨 ALERT: {message}")
    
    def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            if not self.email_config['password']:
                logger.warning("Email alerts not configured")
                return
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['email']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = f"SYMBIOFLOWS ALERT: {alert.severity.upper()} - {alert.component}"
            
            body = f"""
            🚨 SYSTEM ALERT
            
            Component: {alert.component}
            Severity: {alert.severity.upper()}
            Message: {alert.message}
            Time: {alert.timestamp}
            
            Please check the system immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['email'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.component}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _check_alerts(self):
        """Check and resolve alerts"""
        with self.lock:
            current_alerts = [alert for alert in self.alerts if not alert.resolved]
            
            for alert in current_alerts:
                # Check if alert is resolved (metric is back to healthy)
                metric_key = f"{alert.component}_{alert.message.split(':')[0].split()[-1]}"
                
                # Simple resolution logic - if metric is healthy for 5 minutes, resolve alert
                if alert.timestamp < datetime.now() - timedelta(minutes=5):
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    logger.info(f"✅ Alert resolved: {alert.message}")
    
    def _update_health_status(self):
        """Update overall health status"""
        with self.lock:
            critical_metrics = [m for m in self.metrics.values() if m.status == 'critical']
            warning_metrics = [m for m in self.metrics.values() if m.status == 'warning']
            
            if critical_metrics:
                self.health_status = 'critical'
            elif warning_metrics:
                self.health_status = 'warning'
            else:
                self.health_status = 'healthy'
    
    def _save_metrics(self):
        """Save metrics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            with self.lock:
                for metric in self.metrics.values():
                    cursor.execute('''
                        INSERT OR REPLACE INTO health_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metric.metric_id,
                        metric.component,
                        metric.metric_name,
                        metric.value,
                        metric.unit,
                        metric.status,
                        metric.timestamp.isoformat(),
                        json.dumps(metric.details)
                    ))
                
                for alert in self.alerts:
                    cursor.execute('''
                        INSERT OR REPLACE INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        alert.alert_id,
                        alert.severity,
                        alert.component,
                        alert.message,
                        alert.timestamp.isoformat(),
                        1 if alert.resolved else 0,
                        alert.resolution_time.isoformat() if alert.resolution_time else None
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        with self.lock:
            return {
                'overall_status': self.health_status,
                'last_check': self.last_check.isoformat(),
                'metrics_count': len(self.metrics),
                'active_alerts': len([a for a in self.alerts if not a.resolved]),
                'critical_alerts': len([a for a in self.alerts if a.severity == 'critical' and not a.resolved]),
                'warning_alerts': len([a for a in self.alerts if a.severity == 'warning' and not a.resolved])
            }
    
    def get_metrics(self, component: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metrics"""
        with self.lock:
            metrics = list(self.metrics.values())
            
            if component:
                metrics = [m for m in metrics if m.component == component]
            
            # Sort by timestamp (newest first)
            metrics.sort(key=lambda x: x.timestamp, reverse=True)
            
            return [asdict(m) for m in metrics[:limit]]
    
    def get_alerts(self, severity: Optional[str] = None, resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get alerts"""
        with self.lock:
            alerts = list(self.alerts)
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            if resolved is not None:
                alerts = [a for a in alerts if a.resolved == resolved]
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            return [asdict(a) for a in alerts]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            # System performance
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process information
            processes = len(psutil.pids())
            python_processes = len([p for p in psutil.process_iter(['name']) if 'python' in p.info['name'].lower()])
            
            return {
                'system': {
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory.percent,
                    'disk_usage': (disk.used / disk.total) * 100,
                    'active_processes': processes,
                    'python_processes': python_processes
                },
                'health': self.get_health_status(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def clear_old_metrics(self, days: int = 7):
        """Clear old metrics"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM health_metrics WHERE timestamp < ?', (cutoff_date.isoformat(),))
            cursor.execute('DELETE FROM alerts WHERE timestamp < ? AND resolved = 1', (cutoff_date.isoformat(),))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleared metrics older than {days} days")
            
        except Exception as e:
            logger.error(f"Error clearing old metrics: {e}")

# Global health monitor instance
health_monitor = SystemHealthMonitor()

# Test function
def test_health_monitor():
    """Test the health monitor"""
    print("🏥 Testing System Health Monitor...")
    
    # Start monitoring
    health_monitor.start_monitoring()
    
    # Run for a few seconds
    time.sleep(10)
    
    # Get status
    status = health_monitor.get_health_status()
    print(f"Health Status: {status}")
    
    # Get performance summary
    summary = health_monitor.get_performance_summary()
    print(f"Performance Summary: {summary}")
    
    # Stop monitoring
    health_monitor.stop_monitoring_service()
    
    print("✅ Health monitor test completed")

if __name__ == "__main__":
    test_health_monitor() 