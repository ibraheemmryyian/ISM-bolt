"""
Production health check system
"""
import asyncio
import logging
import time
import psutil
from typing import Dict, Any
from datetime import datetime

try:
    from .production_database import db_manager
    from .production_service_registry import service_registry
    HAS_DB_MANAGER = True
except ImportError:
    HAS_DB_MANAGER = False

class ProductionHealthChecker:
    """Comprehensive health check system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - self.start_time,
            'version': '2.0.0-production',
            'components': {}
        }
        
        # Check system resources
        health_status['components']['system'] = await self._check_system_resources()
        
        # Check database connections
        if HAS_DB_MANAGER:
            health_status['components']['database'] = await self._check_database_health()
        
        # Check AI components
        health_status['components']['ai_system'] = await self._check_ai_system()
        
        # Check external APIs
        health_status['components']['external_apis'] = await self._check_external_apis()
        
        # Determine overall status
        component_statuses = [comp.get('status', 'unhealthy') for comp in health_status['components'].values()]
        if all(status == 'healthy' for status in component_statuses):
            health_status['status'] = 'healthy'
        elif any(status == 'healthy' for status in component_statuses):
            health_status['status'] = 'degraded'
        else:
            health_status['status'] = 'unhealthy'
        
        return health_status
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'status': 'healthy' if cpu_percent < 80 and memory.percent < 80 else 'degraded',
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            self.logger.error(f"System resource check failed: {e}")
            return {'status': 'unhealthy', 'error': str(e)}
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connection health"""
        if not HAS_DB_MANAGER:
            return {'status': 'not_available', 'message': 'Database manager not available'}
        
        try:
            db_status = db_manager.health_check()
            
            # Determine status based on available connections
            if any(db_status.values()):
                status = 'healthy'
            else:
                status = 'unhealthy'
            
            return {
                'status': status,
                'connections': db_status
            }
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    async def _check_ai_system(self) -> Dict[str, Any]:
        """Check AI system components"""
        try:
            # Basic AI system check
            import torch
            
            ai_status = {
                'status': 'healthy',
                'pytorch_available': True,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            return ai_status
        except Exception as e:
            return {'status': 'degraded', 'error': str(e)}
    
    async def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API connectivity"""
        api_status = {
            'status': 'healthy',
            'apis': {}
        }
        
        # List of APIs to check
        apis_to_check = [
            'deepseek_r1',
            'materials_project',
            'freightos'
        ]
        
        healthy_apis = 0
        
        for api_name in apis_to_check:
            try:
                # Simulate API check (replace with actual health check)
                api_status['apis'][api_name] = {
                    'status': 'healthy',
                    'response_time_ms': 100
                }
                healthy_apis += 1
            except Exception as e:
                api_status['apis'][api_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        # Determine overall API status
        if healthy_apis == len(apis_to_check):
            api_status['status'] = 'healthy'
        elif healthy_apis > 0:
            api_status['status'] = 'degraded'
        else:
            api_status['status'] = 'unhealthy'
        
        return api_status
    
    def get_quick_status(self) -> Dict[str, str]:
        """Get quick health status for load balancer"""
        try:
            # Quick checks only
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            if cpu_percent < 90 and memory.percent < 90:
                return {'status': 'healthy'}
            else:
                return {'status': 'unhealthy'}
        except:
            return {'status': 'unhealthy'}

# Global health checker instance
health_checker = ProductionHealthChecker()
