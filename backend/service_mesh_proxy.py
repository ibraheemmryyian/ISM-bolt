#!/usr/bin/env python3
"""
Service Mesh Proxy for SymbioFlows
Real inter-service communication with advanced routing and load balancing
"""

import asyncio
import aiohttp
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import redis
import jwt
import hashlib
import hmac
import base64
from functools import wraps
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class ServiceInstance:
    service_id: str
    service_name: str
    host: str
    port: int
    protocol: str
    health_endpoint: str
    status: ServiceStatus
    last_health_check: datetime
    response_time: float
    error_rate: float
    load_balancer_weight: float
    active_connections: int
    max_connections: int
    metadata: Dict[str, Any]

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 2
    expected_exception: type = Exception

class CircuitBreaker:
    """Advanced circuit breaker with state management"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.metrics = {
            'requests_total': Counter('circuit_breaker_requests_total', 'Total requests', ['state']),
            'failures_total': Counter('circuit_breaker_failures_total', 'Total failures'),
            'state_changes_total': Counter('circuit_breaker_state_changes_total', 'State changes', ['from_state', 'to_state'])
        }
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        self.metrics['requests_total'].labels(self.state.value).inc()
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        else:
            self.success_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0
        
        self.metrics['failures_total'].inc()
        
        if self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if not self.last_failure_time:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.config.recovery_timeout
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        if self.state != CircuitState.OPEN:
            self.metrics['state_changes_total'].labels(self.state.value, 'open').inc()
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        if self.state != CircuitState.HALF_OPEN:
            self.metrics['state_changes_total'].labels(self.state.value, 'half_open').inc()
            self.state = CircuitState.HALF_OPEN
            logger.info("Circuit breaker transitioning to HALF_OPEN")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        if self.state != CircuitState.CLOSED:
            self.metrics['state_changes_total'].labels(self.state.value, 'closed').inc()
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker closed")

class LoadBalancer:
    """Advanced load balancer with multiple strategies"""
    
    def __init__(self, strategy: str = "weighted_round_robin"):
        self.strategy = strategy
        self.current_index = 0
        self.metrics = {
            'load_balancer_requests': Counter('load_balancer_requests_total', 'Load balancer requests', ['strategy']),
            'load_balancer_selections': Counter('load_balancer_selections_total', 'Service selections', ['service_id'])
        }
    
    def select_service(self, services: List[ServiceInstance], request_context: Dict[str, Any] = None) -> ServiceInstance:
        """Select service using load balancing strategy"""
        
        if not services:
            raise ValueError("No services available")
        
        # Filter healthy services
        healthy_services = [s for s in services if s.status == ServiceStatus.HEALTHY]
        
        if not healthy_services:
            # Fallback to any available service
            healthy_services = services
        
        self.metrics['load_balancer_requests'].labels(self.strategy).inc()
        
        if self.strategy == "round_robin":
            selected = self._round_robin(healthy_services)
        elif self.strategy == "weighted_round_robin":
            selected = self._weighted_round_robin(healthy_services)
        elif self.strategy == "least_connections":
            selected = self._least_connections(healthy_services)
        elif self.strategy == "fastest_response":
            selected = self._fastest_response(healthy_services)
        else:
            selected = healthy_services[0]
        
        self.metrics['load_balancer_selections'].labels(selected.service_id).inc()
        return selected
    
    def _round_robin(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin selection"""
        service = services[self.current_index % len(services)]
        self.current_index += 1
        return service
    
    def _weighted_round_robin(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round-robin selection"""
        total_weight = sum(s.load_balancer_weight for s in services)
        
        if total_weight == 0:
            return services[0]
        
        # Use current index to maintain state
        current_weight = 0
        for service in services:
            current_weight += service.load_balancer_weight
            if self.current_index < current_weight:
                self.current_index += 1
                return service
        
        # Reset if we've gone through all services
        self.current_index = 0
        return services[0]
    
    def _least_connections(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection"""
        return min(services, key=lambda s: s.active_connections)
    
    def _fastest_response(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Fastest response time selection"""
        return min(services, key=lambda s: s.response_time)

class ServiceMeshProxy:
    """Service mesh proxy for inter-service communication"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.load_balancers: Dict[str, LoadBalancer] = {}
        self.session = None
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        self.metrics = {
            'proxy_requests': Counter('proxy_requests_total', 'Proxy requests', ['service', 'method', 'status']),
            'proxy_latency': Histogram('proxy_latency_seconds', 'Proxy request latency', ['service', 'method']),
            'proxy_errors': Counter('proxy_errors_total', 'Proxy errors', ['service', 'error_type'])
        }
        
        # Start health checking
        asyncio.create_task(self._health_check_loop())
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def register_service(self, service_id: str, service_instances: List[ServiceInstance]):
        """Register service instances"""
        self.services[service_id] = service_instances
        
        # Create circuit breaker for service
        if service_id not in self.circuit_breakers:
            self.circuit_breakers[service_id] = CircuitBreaker(CircuitBreakerConfig())
        
        # Create load balancer for service
        if service_id not in self.load_balancers:
            self.load_balancers[service_id] = LoadBalancer(strategy="weighted_round_robin")
    
    async def call_service(self, 
                          service_id: str, 
                          method: str, 
                          data: Dict[str, Any],
                          timeout: int = 30,
                          retry_attempts: int = 3) -> Dict[str, Any]:
        """Call service through proxy with advanced features"""
        
        if service_id not in self.services:
            raise ValueError(f"Service {service_id} not registered")
        
        service_instances = self.services[service_id]
        load_balancer = self.load_balancers[service_id]
        circuit_breaker = self.circuit_breakers[service_id]
        
        start_time = time.time()
        
        try:
            # Select service instance
            selected_service = load_balancer.select_service(service_instances)
            
            # Call through circuit breaker
            result = await circuit_breaker.call(
                lambda: self._make_request(selected_service, method, data, timeout)
            )
            
            latency = time.time() - start_time
            self.metrics['proxy_latency'].labels(service_id, method).observe(latency)
            self.metrics['proxy_requests'].labels(service_id, method, 'success').inc()
            
            return result
            
        except Exception as e:
            self.metrics['proxy_requests'].labels(service_id, method, 'error').inc()
            self.metrics['proxy_errors'].labels(service_id, type(e).__name__).inc()
            
            # Retry logic
            if retry_attempts > 0:
                logger.warning(f"Retrying {service_id}.{method} after error: {e}")
                await asyncio.sleep(1)  # Backoff
                return await self.call_service(service_id, method, data, timeout, retry_attempts - 1)
            
            raise
    
    async def _make_request(self, service: ServiceInstance, method: str, data: Dict[str, Any], timeout: int) -> Dict[str, Any]:
        """Make actual HTTP request to service"""
        
        url = f"http://{service.host}:{service.port}/{method.lstrip('/')}"
        
        headers = {
            'Content-Type': 'application/json',
            'X-Request-ID': str(uuid.uuid4()),
            'X-Service-Caller': 'service-mesh-proxy',
            'X-Circuit-Breaker-State': self.circuit_breakers[service.service_id].state.value
        }
        
        # Add authentication if needed
        if service.metadata.get('requires_auth'):
            headers['Authorization'] = self._generate_auth_header(service)
        
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        
        async with self.session.post(url, json=data, headers=headers, timeout=timeout_obj) as response:
            response.raise_for_status()
            return await response.json()
    
    def _generate_auth_header(self, service: ServiceInstance) -> str:
        """Generate authentication header for service"""
        # This would implement proper service-to-service authentication
        # For now, return a simple token
        payload = {
            'service_id': 'service-mesh-proxy',
            'target_service': service.service_id,
            'exp': datetime.utcnow() + timedelta(minutes=5)
        }
        
        secret = service.metadata.get('auth_secret', 'default-secret')
        token = jwt.encode(payload, secret, algorithm='HS256')
        
        return f"Bearer {token}"
    
    async def _health_check_loop(self):
        """Background health checking loop"""
        while True:
            try:
                await self._check_all_services()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_all_services(self):
        """Check health of all registered services"""
        for service_id, instances in self.services.items():
            for instance in instances:
                try:
                    await self._check_service_health(instance)
                except Exception as e:
                    logger.error(f"Health check failed for {service_id}: {e}")
    
    async def _check_service_health(self, service: ServiceInstance):
        """Check health of a single service instance"""
        try:
            url = f"http://{service.host}:{service.port}{service.health_endpoint}"
            
            timeout_obj = aiohttp.ClientTimeout(total=5)
            async with self.session.get(url, timeout=timeout_obj) as response:
                if response.status == 200:
                    health_data = await response.json()
                    service.status = ServiceStatus.HEALTHY
                    service.response_time = health_data.get('response_time', 0.1)
                    service.error_rate = health_data.get('error_rate', 0.0)
                else:
                    service.status = ServiceStatus.UNHEALTHY
                
                service.last_health_check = datetime.now()
                
        except Exception as e:
            service.status = ServiceStatus.OFFLINE
            service.last_health_check = datetime.now()
            logger.warning(f"Health check failed for {service.service_id}: {e}")

# Initialize service mesh proxy
service_mesh_proxy = ServiceMeshProxy()

# Register default services
default_services = {
    "ai_listings_generator": [
        ServiceInstance(
            service_id="ai_listings_generator_1",
            service_name="AI Listings Generator",
            host="localhost",
            port=5010,
            protocol="http",
            health_endpoint="/health",
            status=ServiceStatus.HEALTHY,
            last_health_check=datetime.now(),
            response_time=0.1,
            error_rate=0.0,
            load_balancer_weight=1.0,
            active_connections=0,
            max_connections=100,
            metadata={}
        )
    ],
    "ai_matchmaking_service": [
        ServiceInstance(
            service_id="ai_matchmaking_service_1",
            service_name="AI Matchmaking Service",
            host="localhost",
            port=8020,
            protocol="http",
            health_endpoint="/health",
            status=ServiceStatus.HEALTHY,
            last_health_check=datetime.now(),
            response_time=0.2,
            error_rate=0.0,
            load_balancer_weight=1.0,
            active_connections=0,
            max_connections=100,
            metadata={}
        )
    ],
    "ai_pricing_service": [
        ServiceInstance(
            service_id="ai_pricing_service_1",
            service_name="AI Pricing Service",
            host="localhost",
            port=5005,
            protocol="http",
            health_endpoint="/health",
            status=ServiceStatus.HEALTHY,
            last_health_check=datetime.now(),
            response_time=0.15,
            error_rate=0.0,
            load_balancer_weight=1.0,
            active_connections=0,
            max_connections=100,
            metadata={}
        )
    ]
}

for service_id, instances in default_services.items():
    service_mesh_proxy.register_service(service_id, instances)

# Flask app for API endpoints
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Service Mesh Proxy',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/proxy/<service_id>/<path:method>', methods=['POST'])
async def proxy_request(service_id, method):
    """Proxy request to service"""
    try:
        data = request.json or {}
        timeout = request.args.get('timeout', 30, type=int)
        retry_attempts = request.args.get('retry', 3, type=int)
        
        async with service_mesh_proxy:
            result = await service_mesh_proxy.call_service(
                service_id, method, data, timeout, retry_attempts
            )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Proxy request error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/services', methods=['GET'])
def list_services():
    """List all registered services"""
    services_info = {}
    
    for service_id, instances in service_mesh_proxy.services.items():
        services_info[service_id] = {
            'instances': [
                {
                    'service_id': instance.service_id,
                    'host': instance.host,
                    'port': instance.port,
                    'status': instance.status.value,
                    'response_time': instance.response_time,
                    'error_rate': instance.error_rate,
                    'active_connections': instance.active_connections
                }
                for instance in instances
            ],
            'circuit_breaker_state': service_mesh_proxy.circuit_breakers[service_id].state.value,
            'load_balancer_strategy': service_mesh_proxy.load_balancers[service_id].strategy
        }
    
    return jsonify(services_info)

@app.route('/services/<service_id>/health', methods=['GET'])
def service_health(service_id):
    """Get health status of specific service"""
    if service_id not in service_mesh_proxy.services:
        return jsonify({'error': 'Service not found'}), 404
    
    instances = service_mesh_proxy.services[service_id]
    circuit_breaker = service_mesh_proxy.circuit_breakers[service_id]
    
    return jsonify({
        'service_id': service_id,
        'instances': [
            {
                'service_id': instance.service_id,
                'status': instance.status.value,
                'last_health_check': instance.last_health_check.isoformat(),
                'response_time': instance.response_time,
                'error_rate': instance.error_rate
            }
            for instance in instances
        ],
        'circuit_breaker_state': circuit_breaker.state.value,
        'failure_count': circuit_breaker.failure_count,
        'success_count': circuit_breaker.success_count
    })

if __name__ == "__main__":
    print("🚀 Starting Service Mesh Proxy on port 5019...")
    app.run(host='0.0.0.0', port=5019, debug=False) 