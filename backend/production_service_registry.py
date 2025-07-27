"""
Production-ready service registry for microservice communication
"""
import os
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    host: str
    port: int
    health_endpoint: str = "/health"
    timeout: int = 30
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

class ProductionServiceRegistry:
    """Production-ready service registry"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.services = self._load_services()
        self.session: Optional[aiohttp.ClientSession] = None
    
    def _load_services(self) -> Dict[str, ServiceEndpoint]:
        """Load service configurations from environment"""
        services = {}
        
        # Core services
        services['ai_matching'] = ServiceEndpoint(
            name='ai_matching',
            host=os.getenv('AI_MATCHING_HOST', 'localhost'),
            port=int(os.getenv('AI_MATCHING_PORT', 8001))
        )
        
        services['materials_analysis'] = ServiceEndpoint(
            name='materials_analysis',
            host=os.getenv('MATERIALS_ANALYSIS_HOST', 'localhost'),
            port=int(os.getenv('MATERIALS_ANALYSIS_PORT', 8002))
        )
        
        services['logistics_optimizer'] = ServiceEndpoint(
            name='logistics_optimizer',
            host=os.getenv('LOGISTICS_HOST', 'localhost'),
            port=int(os.getenv('LOGISTICS_PORT', 8003))
        )
        
        return services
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def call_service(self, service_name: str, endpoint: str, 
                          data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a microservice endpoint"""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        service = self.services[service_name]
        url = f"{service.base_url}{endpoint}"
        
        try:
            session = await self.get_session()
            
            if data:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"Service call failed: {response.status}")
            else:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"Service call failed: {response.status}")
        
        except Exception as e:
            self.logger.error(f"Failed to call {service_name}{endpoint}: {e}")
            raise
    
    async def health_check(self, service_name: str) -> bool:
        """Check if service is healthy"""
        try:
            result = await self.call_service(service_name, '/health')
            return result.get('status') == 'healthy'
        except:
            return False
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None

# Global service registry instance
service_registry = ProductionServiceRegistry()
