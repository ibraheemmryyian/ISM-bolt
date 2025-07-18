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

import asyncio
import logging
import json
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import psutil
import requests
import os

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
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_connections: int
    requests_per_second: float
    error_rate: float
    disk_usage: float
    active_processes: int
    python_processes: int

@dataclass
class HealthAlert:
    """Alert data structure"""
    alert_id: str
    alert_type: str
    severity: str  # info, warning, critical
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False

class SystemHealthMonitor:
    """
    Comprehensive system health monitor (Redis-free for production testing)
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
        
        # In-memory storage (replacing Redis)
        self.memory_cache = defaultdict(lambda: deque(maxlen=100)) # Store last 100 metrics for each component
        self.cache_lock = threading.Lock()
        
        # Monitoring state
        self.metrics = defaultdict(lambda: deque(maxlen=100)) # Store last 100 metrics for each component
        self.alerts = []
        self.health_status = 'healthy'
        self.last_check = datetime.now()
        
        # Threading
        self.monitoring_thread = None
        self._stop_monitoring_flag = False
        self.lock = threading.Lock()
        
        # Email configuration
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'email': os.getenv('ALERT_EMAIL', 'alerts@symbioflows.com'),
            'password': os.getenv('ALERT_PASSWORD', ''),
            'recipients': os.getenv('ALERT_RECIPIENTS', 'admin@symbioflows.com').split(',')
        }
        
        logger.info("🏥 System Health Monitor initialized (Redis-free)")
    
    def _init_database(self):
        """Initialize SQLite database for health monitoring"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS health_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS health_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT UNIQUE NOT NULL,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        threshold REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                    ON health_metrics(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                    ON health_alerts(timestamp)
                """)
                
            logger.info("✅ Health monitoring database initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize health database: {e}")
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100      
            # Process count
            active_processes = len(psutil.pids())
            python_processes = len([p for p in psutil.process_iter(['name']) 
                                  if 'python' in p.info['name'].lower()])
            
            # Network connections (simplified)
            active_connections = len(psutil.net_connections())
            
            # Calculate requests per second (simplified)
            requests_per_second = 0 # Would need to track actual requests
            
            # Error rate (simplified)
            error_rate = 0 # Would need to track actual errors
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                active_connections=active_connections,
                requests_per_second=requests_per_second,
                error_rate=error_rate,
                disk_usage=disk_usage,
                active_processes=active_processes,
                python_processes=python_processes
            )
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                active_connections=0,
                requests_per_second=0.0,
                error_rate=0.0,
                disk_usage=0.0,
                active_processes=0,
                python_processes=0
            )
    
    def _check_api_health(self) -> Dict[str, Any]:
        """Health of various API endpoints"""
        api_checks = {}
        
        # Check core services
        services = {
            'companies_api': 'http://localhost:3000/api/companies',
            'materials_api': 'http://localhost:3000/api/materials',
            'ai_listings_api': 'http://localhost:3000/api/ai/listings'
        }
        
        for service_name, url in services.items():
            try:
                start_time = time.time()
                response = requests.get(url, timeout=5)
                response_time = time.time() - start_time
                
                api_checks[f"{service_name}_response_time"] = response_time
                api_checks[f"{service_name}_status_code"] = response.status_code
                
                if response.status_code >= 400:
                    logger.warning(f"🚨 ALERT: API {service_name} returned {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"🚨 ALERT: API {service_name} failed: {e}")
                api_checks[f"{service_name}_response_time"] = 999.0
                api_checks[f"{service_name}_status_code"] = 0   
        return api_checks
    
    def _check_service_health(self) -> Dict[str, Any]:
        """Health of various services"""
        service_checks = {}
        
        # Check if services are running (simplified)
        services = {
            'onboarding_system': 519,
            'ai_listings_generator': 5011,
            'matching_engine': 512,
            'logistics_service': 5006
        }
        
        for service_name, port in services.items():
            try:
                response = requests.get(f'http://localhost:{port}/health', timeout=2)
                service_checks[f"{service_name}_status"] = 1 if response.status_code == 200 else 0
            except:
                service_checks[f"{service_name}_status"] = 0
                logger.warning(f"🚨 ALERT: SERVICE {service_name} not responding")
        
        return service_checks
    
    def _store_metrics(self, metrics: SystemMetrics, api_checks: Dict[str, Any], service_checks: Dict[str, Any]):
        """Store metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store system metrics
                conn.execute("""
                    INSERT INTO health_metrics (timestamp, metric_name, metric_value)
                    VALUES (?, ?, ?)
                """, (metrics.timestamp.isoformat(), 'cpu_usage', metrics.cpu_usage))
                
                conn.execute("""
                    INSERT INTO health_metrics (timestamp, metric_name, metric_value)
                    VALUES (?, ?, ?)
                """, (metrics.timestamp.isoformat(), 'memory_usage', metrics.memory_usage))
                
                conn.execute("""
                    INSERT INTO health_metrics (timestamp, metric_name, metric_value)
                    VALUES (?, ?, ?)
                """, (metrics.timestamp.isoformat(), 'disk_usage', metrics.disk_usage))
                
                # Store API checks
                for metric_name, value in api_checks.items():
                    conn.execute("""
                        INSERT INTO health_metrics (timestamp, metric_name, metric_value)
                        VALUES (?, ?, ?)
                    """, (metrics.timestamp.isoformat(), metric_name, value))
                
                # Store service checks
                for metric_name, value in service_checks.items():
                    conn.execute("""
                        INSERT INTO health_metrics (timestamp, metric_name, metric_value)
                        VALUES (?, ?, ?)
                    """, (metrics.timestamp.isoformat(), metric_name, value))
                    
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
    
    def _check_alerts(self, metrics: SystemMetrics, api_checks: Dict[str, Any], service_checks: Dict[str, Any]):
        """Check for alerts and store them"""
        alerts = []
        
        # System alerts
        if metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(HealthAlert(
                alert_id=f"cpu_{int(time.time())}",
                alert_type="SYSTEM",
                severity="WARNING" if metrics.cpu_usage < 90 else "CRITICAL",
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                metric_name="cpu_usage",
                metric_value=metrics.cpu_usage,
                threshold=self.alert_thresholds['cpu_usage'],
                timestamp=datetime.now()
            ))
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append(HealthAlert(
                alert_id=f"memory_{int(time.time())}",
                alert_type="SYSTEM",
                severity="WARNING" if metrics.memory_usage < 95 else "CRITICAL",
                message=f"High memory usage: {metrics.memory_usage:.1f}%",
                metric_name="memory_usage",
                metric_value=metrics.memory_usage,
                threshold=self.alert_thresholds['memory_usage'],
                timestamp=datetime.now()
            ))
        if metrics.disk_usage > self.alert_thresholds['disk_usage']:
            alerts.append(HealthAlert(
                alert_id=f"disk_{int(time.time())}",
                alert_type="SYSTEM",
                severity="CRITICAL",
                message=f"High disk usage: {metrics.disk_usage:.1f}%",
                metric_name="disk_usage",
                metric_value=metrics.disk_usage,
                threshold=self.alert_thresholds['disk_usage'],
                timestamp=datetime.now()
            ))
        
        # API alerts
        for metric_name, value in api_checks.items():
            if "response_time" in metric_name and value > self.alert_thresholds['response_time']:
                alerts.append(HealthAlert(
                    alert_id=f"{metric_name}_{int(time.time())}",
                    alert_type="API",
                    severity="WARNING",
                    message=f"Slow API response: {value:.3f}s",
                    metric_name=metric_name,
                    metric_value=value,
                    threshold=self.alert_thresholds['response_time'],
                    timestamp=datetime.now()
                ))
            
            if 'status_code' in metric_name and value >= 400:
                alerts.append(HealthAlert(
                    alert_id=f"{metric_name}_{int(time.time())}",
                    alert_type="API",
                    severity="WARNING" if value < 500 else "CRITICAL",
                    message=f"API error: {value}",
                    metric_name=metric_name,
                    metric_value=value,
                    threshold=200,
                    timestamp=datetime.now()
                ))
        
        # Service alerts
        for metric_name, value in service_checks.items():
            if value == 0:
                alerts.append(HealthAlert(
                    alert_id=f"{metric_name}_{int(time.time())}",
                    alert_type="SERVICE",
                    severity="CRITICAL",
                    message=f"Service not responding: {metric_name}",
                    metric_name=metric_name,
                    metric_value=value,
                    threshold=1,
                    timestamp=datetime.now()
                ))
        
        # Store alerts
        if alerts:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    for alert in alerts:
                        conn.execute("""
                            INSERT OR REPLACE INTO health_alerts 
                            (alert_id, alert_type, severity, message, metric_name, metric_value, threshold, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            alert.alert_id, alert.alert_type, alert.severity, alert.message,
                            alert.metric_name, alert.metric_value, alert.threshold, alert.timestamp.isoformat()
                        ))
            except Exception as e:
                logger.error(f"Error storing alerts: {e}")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"🚨 ALERT: {alert.severity} - {alert.message}")
            if self.email_config.get('password'):
                self._send_email_alert(alert)
    
    def _send_email_alert(self, alert: HealthAlert):
        """Send email alert (simplified)"""
        # This would implement actual email sending
        # For now, just log that we would send an email
        logger.warning(f"Email alerts not configured")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        with self.lock:
            return {
                'overall_status': self.health_status,
                'last_check': self.last_check.isoformat(),
                'metrics_count': len(self.metrics),
                'active_alerts': len([a for a in self.alerts if not a.resolved]),
                'critical_alerts': len([a for a in self.alerts if a.severity == 'CRITICAL' and not a.resolved]),
                'warning_alerts': len([a for a in self.alerts if a.severity == 'WARNING' and not a.resolved])
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        metrics = self._get_system_metrics()
        api_checks = self._check_api_health()
        service_checks = self._check_service_health()
        
        return {
            'system': {
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'disk_usage': metrics.disk_usage,
                'active_processes': metrics.active_processes,
                'python_processes': metrics.python_processes
            },
            'health': self.get_health_status(),
            'timestamp': datetime.now().isoformat()
        }
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self._stop_monitoring_flag = False
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("🚀 Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self._stop_monitoring_flag = True
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("🛑 Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._stop_monitoring_flag:
            try:
                # Get metrics
                metrics = self._get_system_metrics()
                api_checks = self._check_api_health()
                service_checks = self._check_service_health()
                
                # Store metrics
                self._store_metrics(metrics, api_checks, service_checks)
                
                # Check alerts
                self._check_alerts(metrics, api_checks, service_checks)
                
                # Update state
                with self.lock:
                    self.metrics = {
                        'system': asdict(metrics),
                        'api': api_checks,
                        'services': service_checks
                    }
                    self.last_check = datetime.now()
                
                # Wait for next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)

# Test function
async def test_health_monitor():
    """Test the health monitor"""
    monitor = SystemHealthMonitor()
    
    print("🏥 Testing System Health Monitor...")
    
    # Start monitoring
    await monitor.start_monitoring()
    
    # Let it run for a few seconds
    await asyncio.sleep(10)
    
    # Get status
    status = monitor.get_health_status()
    performance = monitor.get_performance_summary()
    
    print(f"Health Status: {status}")
    print(f"Performance Summary: {performance}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("✅ Health monitor test completed")

if __name__ == "__main__":
    import sys
    asyncio.run(test_health_monitor()) 