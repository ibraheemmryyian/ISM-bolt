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
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import grpc
import grpc.aio
from grpc import RpcError
import kubernetes
from kubernetes import client, config
import docker
from docker.errors import DockerException
import consul
import redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import circuitbreaker
from circuitbreaker import circuit
import jwt
import hashlib
import hmac
import base64
from dataclasses import dataclass
from enum import Enum
import yaml
import subprocess
import signal
import threading
import queue
import time
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    ServiceOptimizer
)
from ml_core.monitoring import (
    MLMetricsTracker,
    ServiceMonitor
)
from ml_core.utils import (
    ModelRegistry,
    ServiceRegistry,
    ConfigManager
)

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

class ServiceType(Enum):
    ML_MODEL = "ml_model"
    DATA_PROCESSING = "data_processing"
    INFERENCE = "inference"
    TRAINING = "training"
    MONITORING = "monitoring"

@dataclass
class ServiceEndpoint:
    service_id: str
    service_type: ServiceType
    url: str
    port: int
    health_check_url: str
    status: ServiceStatus
    load_balancer_weight: float
    last_health_check: datetime
    response_time: float
    error_rate: float

class ServiceDiscovery:
    """Service discovery with health checking and load balancing"""
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.consul_host = consul_host
        self.consul_port = consul_port
        
        try:
            self.consul_client = consul.Consul(host=consul_host, port=consul_port)
        except Exception as e:
            logging.warning(f"Consul not available: {e}")
            self.consul_client = None
        
        # Service registry
        self.services = {}
        
        # Health check configuration
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 5    # seconds
        
        # Start health checking
        self._start_health_checking()
    
    def register_service(self, service_endpoint: ServiceEndpoint):
        """Register a service endpoint"""
        try:
            self.services[service_endpoint.service_id] = service_endpoint
            
            # Register with Consul if available
            if self.consul_client:
                self.consul_client.agent.service.register(
                    name=service_endpoint.service_id,
                    service_id=service_endpoint.service_id,
                    address=service_endpoint.url,
                    port=service_endpoint.port,
                    check=consul.Check.http(
                        url=f"http://{service_endpoint.url}:{service_endpoint.port}{service_endpoint.health_check_url}",
                        interval=f"{self.health_check_interval}s",
                        timeout=f"{self.health_check_timeout}s"
                    )
                )
            
            logging.info(f"Registered service: {service_endpoint.service_id}")
            
        except Exception as e:
            logging.error(f"Error registering service: {e}")
    
    def discover_service(self, service_type: ServiceType) -> List[ServiceEndpoint]:
        """Discover services of a specific type"""
        try:
            if self.consul_client:
                # Query Consul
                _, services = self.consul_client.health.service(service_type.value, passing=True)
                return [self._consul_service_to_endpoint(s) for s in services]
            else:
                # Use local registry
                return [service for service in self.services.values() 
                        if service.service_type == service_type and service.status == ServiceStatus.HEALTHY]
        except Exception as e:
            logging.error(f"Error discovering services: {e}")
            return []
    
    def _consul_service_to_endpoint(self, consul_service) -> ServiceEndpoint:
        """Convert Consul service to endpoint"""
        service = consul_service['Service']
        checks = consul_service['Checks']
        
        # Determine status from health checks
        status = ServiceStatus.HEALTHY
        for check in checks:
            if check['Status'] != 'passing':
                status = ServiceStatus.UNHEALTHY
                break
        
        return ServiceEndpoint(
            service_id=service['ID'],
            service_type=ServiceType(service['Service']),
            url=service['Address'],
            port=service['Port'],
            health_check_url='/health',
            status=status,
            load_balancer_weight=1.0,
            last_health_check=datetime.now(),
            response_time=0.0,
            error_rate=0.0
        )
    
    def _start_health_checking(self):
        """Start health checking thread"""
        def health_checker():
            while True:
                try:
                    for service_id, service in self.services.items():
                        self._check_service_health(service)
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logging.error(f"Health checking error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        thread = threading.Thread(target=health_checker, daemon=True)
        thread.start()
    
    def _check_service_health(self, service: ServiceEndpoint):
        """Check health of a service endpoint"""
        try:
            start_time = time.time()
            
            # Make health check request
            response = requests.get(
                f"http://{service.url}:{service.port}{service.health_check_url}",
                timeout=self.health_check_timeout
            )
            
            response_time = time.time() - start_time
            
            # Update service status
            service.last_health_check = datetime.now()
            service.response_time = response_time
            
            if response.status_code == 200:
                service.status = ServiceStatus.HEALTHY
                service.error_rate = max(0, service.error_rate - 0.1) # Reduce error rate
            else:
                service.status = ServiceStatus.UNHEALTHY
                service.error_rate = min(1.0, service.error_rate + 0.2) # Increase error rate
                
        except Exception as e:
            service.status = ServiceStatus.OFFLINE
            service.error_rate = min(1.0, service.error_rate + 0.3)
            logging.warning(f"Health check failed for {service.service_id}: {e}")

class LoadBalancer:
    """Intelligent load balancer with ML-based routing"""
    def __init__(self, strategy: str = "weighted_round_robin"):
        self.strategy = strategy
        self.service_weights = {}
        self.request_counts = {}
        
        # ML-based routing model
        self.routing_model = self._create_routing_model()
    
    def _create_routing_model(self) -> nn.Module:
        """Create ML routing model"""
        return nn.Sequential(
            nn.Linear(10, 64),  # 10 features: service load, response time, error rate, etc.
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def select_service(self, services: List[ServiceEndpoint], request_context: Dict = None) -> ServiceEndpoint:
        """Select service using load balancing strategy"""
        if not services:
            raise ValueError("No services available")
        
        if self.strategy == "weighted_round_robin":
            return self._weighted_round_robin(services)
        elif self.strategy == "least_connections":
            return self._least_connections(services)
        elif self.strategy == "ml_based":
            return self._ml_based_routing(services, request_context)
        else:
            return self._round_robin(services)
    
    def _weighted_round_robin(self, services: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted round-robin selection"""
        # Calculate total weight
        total_weight = sum(service.load_balancer_weight for service in services)
        
        # Select based on weights
        random_value = np.random.uniform(0, total_weight)
        current_weight = 0      
        for service in services:
            current_weight += service.load_balancer_weight
            if random_value <= current_weight:
                return service
        
        return services[0]  # Fallback
    
    def _least_connections(self, services: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least connections selection"""
        return min(services, key=lambda s: self.request_counts.get(s.service_id, 0))
    
    def _ml_based_routing(self, services: List[ServiceEndpoint], request_context: Dict) -> ServiceEndpoint:
        """ML-based routing selection"""
        try:
            # Prepare features for each service
            service_features = []
            for service in services:
                features = [
                    service.response_time,
                    service.error_rate,
                    self.request_counts.get(service.service_id, 0),
                    service.load_balancer_weight,
                    request_context.get('request_size', 1),
                    request_context.get('priority', 1),
                    request_context.get('user_tier', 1),
                    time.time() % 86400,  # Time of day (normalized)
                    request_context.get('complexity', 1),
                    request_context.get('urgency', 1)
                ]
                service_features.append(features)
            # Get ML predictions
            features_tensor = torch.FloatTensor(service_features)
            with torch.no_grad():
                scores = self.routing_model(features_tensor).squeeze()
            # Select service with highest score
            best_service_idx = torch.argmax(scores).item()
            return services[best_service_idx]
        except Exception as e:
            logging.error(f"ML routing error: {e}")
            return self._round_robin(services)
    
    def _round_robin(self, services: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Simple round-robin selection"""
        service_id = services[0].service_id
        self.request_counts[service_id] = self.request_counts.get(service_id, 0) + 1
        return services[0]

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # "closed", "open", "half-open"
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0    
        if self.state == "half-open":
            self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class APIGateway:
    """API Gateway with authentication, rate limiting, and routing"""
    def __init__(self):
        self.routes = {}
        self.middleware = []
        self.rate_limiters = {}
        self.authentication_tokens = {}
        
        # Metrics
        self.request_counter = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
        self.response_time_histogram = Histogram('api_response_time_seconds', 'API response time', ['endpoint'])
        self.error_counter = Counter('api_errors_total', 'Total API errors', ['endpoint', 'error_type'])
    
    def add_route(self, path: str, method: str, handler, middleware: List = None):
        """Add route to gateway"""
        route_key = f"{method}:{path}"
        self.routes[route_key] = {
            'handler': handler,
            'middleware': middleware or []
        }
    
    def add_middleware(self, middleware):
        """Add global middleware"""
        self.middleware.append(middleware)
    
    async def handle_request(self, path: str, method: str, headers: Dict, body: Dict) -> Dict:
        """Handle incoming request"""
        start_time = time.time()
        
        try:
            # Record request
            self.request_counter.labels(endpoint=path, method=method).inc()
            
            # Check rate limiting
            if not self._check_rate_limit(path, headers):
                raise Exception("Rate limit exceeded")
            
            # Authenticate request
            if not self._authenticate_request(headers):
                raise Exception("Authentication failed")
            
            # Find route
            route_key = f"{method}:{path}"
            if route_key not in self.routes:
                raise Exception("Route not found")
            
            route = self.routes[route_key]
            
            # Apply middleware
            processed_body = body
            for middleware in self.middleware + route['middleware']:
                processed_body = await middleware(processed_body, headers)
            
            # Call handler
            result = await route['handler'](processed_body, headers)
            
            # Record response time
            response_time = time.time() - start_time
            self.response_time_histogram.labels(endpoint=path).observe(response_time)
            
            return result
            
        except Exception as e:
            # Record error
            self.error_counter.labels(endpoint=path, error_type=type(e).__name__).inc()
            raise e
    
    def _check_rate_limit(self, path: str, headers: Dict) -> bool:
        """Check rate limiting"""
        # Simple rate limiting implementation
        client_id = headers.get('X-Client-ID', 'default')
        rate_key = f"{client_id}:{path}"
        if rate_key not in self.rate_limiters:
            self.rate_limiters[rate_key] = deque(maxlen=100)
        
        now = time.time()
        requests = self.rate_limiters[rate_key]
        
        # Remove old requests
        while requests and now - requests[0] > 60:  # 1 minute window
            requests.popleft()
        
        # Check limit
        if len(requests) >= 100: # 100 requests per minute
            return False
        
        requests.append(now)
        return True
    
    def _authenticate_request(self, headers: Dict) -> bool:
        """Authenticate request"""
        # Simple token-based authentication
        token = headers.get('Authorization', '').replace('Bearer ', '')
        
        if not token:
            return False
        
        # Check if token is valid
        return token in self.authentication_tokens

class ServiceMesh:
    """Service mesh for inter-service communication"""
    def __init__(self):
        self.services = {}
        self.policies = {}
        self.metrics = {}
        
        # Initialize metrics
        self.service_call_counter = Counter('service_calls_total', 'Total service calls', ['from_service', 'to_service'])
        self.service_latency_histogram = Histogram('service_latency_seconds', 'Service call latency', ['from_service', 'to_service'])
    
    def register_service(self, service_id: str, service_info: Dict):
        """Register service in mesh"""
        self.services[service_id] = service_info
    
    def add_policy(self, service_id: str, policy: Dict):
        """Add policy for service"""
        self.policies[service_id] = policy
    
    async def call_service(self, from_service: str, to_service: str, method: str, data: Dict) -> Dict:
        """Call service through mesh"""
        start_time = time.time()
        
        try:
            # Record call
            self.service_call_counter.labels(from_service=from_service, to_service=to_service).inc()
            
            # Apply policies
            if to_service in self.policies:
                data = await self._apply_policies(to_service, data)
            
            # Make actual service call
            result = await self._make_service_call(to_service, method, data)
            
            # Record latency
            latency = time.time() - start_time
            self.service_latency_histogram.labels(from_service=from_service, to_service=to_service).observe(latency)
            
            return result
            
        except Exception as e:
            logging.error(f"Service call failed: {e}")
            raise e
    
    async def _apply_policies(self, service_id: str, data: Dict) -> Dict:
        """Apply service policies"""
        policy = self.policies[service_id]
        
        # Apply transformations
        if 'transform' in policy:
            for transform in policy['transform']:
                if transform['type'] == 'add_header':
                    data['headers'] = data.get('headers', {})
                    data['headers'][transform['key']] = transform['value']
        
        return data
    
    async def _make_service_call(self, service_id: str, method: str, data: Dict) -> Dict:
        """Make actual service call"""
        # This would integrate with actual service communication
        # For now, simulate a service call
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return {
            'service_id': service_id,
            'method': method,
            'result': 'success',
            'data': data
        }

class AIServiceIntegration:
    """Real ML-powered service integration with advanced orchestration"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.service_discovery = ServiceDiscovery()
        self.load_balancer = LoadBalancer(strategy='ml_based')
        self.api_gateway = APIGateway()
        self.service_mesh = ServiceMesh()
        self.metrics_tracker = MLMetricsTracker()
        self.service_monitor = ServiceMonitor()
        self.config_manager = ConfigManager()
        
        # Service registry
        self.service_registry = ServiceRegistry()
        
        # Circuit breakers
        self.circuit_breakers = {}
        
        # Service integration configuration
        self.integration_config = {
           'timeout': 30,
           'retry_attempts': 3,
           'retry_delay': 1,
           'max_concurrent_requests': 100,
           'health_check_interval': 30
        }
        
        # Initialize services
        self._initialize_services()
        
        # Setup API gateway routes
        self._setup_gateway_routes()
    
    def _initialize_services(self):
        """Initialize core services"""
        try:
            # Register ML services
            ml_services = [
                ServiceEndpoint(
                    service_id='ml-inference',
                    service_type=ServiceType.INFERENCE,
                    url='localhost',
                    port=8001,
                    health_check_url='/health',
                    status=ServiceStatus.HEALTHY,
                    load_balancer_weight=1.0,
                    last_health_check=datetime.now(),
                    response_time=0.1,
                    error_rate=0.0
                ),
                ServiceEndpoint(
                    service_id='ml-training',
                    service_type=ServiceType.TRAINING,
                    url='localhost',
                    port=8002,
                    health_check_url='/health',
                    status=ServiceStatus.HEALTHY,
                    load_balancer_weight=1.0,
                    last_health_check=datetime.now(),
                    response_time=0.2,
                    error_rate=0.0
                ),
                ServiceEndpoint(
                    service_id='ml-monitoring',
                    service_type=ServiceType.MONITORING,
                    url='localhost',
                    port=8003,
                    health_check_url='/health',
                    status=ServiceStatus.HEALTHY,
                    load_balancer_weight=1.0,
                    last_health_check=datetime.now(),
                    response_time=0.05,
                    error_rate=0.0
                )
            ]
            
            for service in ml_services:
                self.service_discovery.register_service(service)
                self.service_mesh.register_service(service.service_id, {
                    'type': service.service_type.value,
                    'url': f"http://{service.url}:{service.port}",
                    'health_check_url': service.health_check_url
                })
                
                # Create circuit breaker
                self.circuit_breakers[service.service_id] = CircuitBreaker(
                    failure_threshold=5,
                    recovery_timeout=60
                )
            
            self.logger.info("Services initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing services: {e}")
    
    def _setup_gateway_routes(self):
        """Setup API gateway routes"""
        try:
            # ML Inference route
            self.api_gateway.add_route(
                path='/api/v1/inference',
                method='POST',
                handler=self._handle_inference_request
            )
            
            # ML Training route
            self.api_gateway.add_route(
                path='/api/v1/training',
                method='POST',
                handler=self._handle_training_request
            )
            
            # Health check route
            self.api_gateway.add_route(
                path='/health',
                method='GET',
                handler=self._handle_health_check
            )
            
            # Add middleware
            self.api_gateway.add_middleware(self._logging_middleware)
            self.api_gateway.add_middleware(self._metrics_middleware)
            
            self.logger.info("API gateway routes configured")
            
        except Exception as e:
            self.logger.error(f"Error setting up gateway routes: {e}")
    
    async def _handle_inference_request(self, body: Dict, headers: Dict) -> Dict:
        """Handle ML inference request"""
        try:
            # Discover inference services
            services = self.service_discovery.discover_service(ServiceType.INFERENCE)
            
            if not services:
                raise Exception("No inference services available")
            
            # Select service using load balancer
            selected_service = self.load_balancer.select_service(services, {
                'request_size': len(str(body)),
                'priority': headers.get('X-Priority', 1),
                'user_tier': headers.get('X-User-Tier', 1),
                'complexity': body.get('complexity', 1),
                'urgency': headers.get('X-Urgency', 1)
            })
            
            # Call service through mesh
            result = await self.service_mesh.call_service(
                from_service='api-gateway',
                to_service=selected_service.service_id,
                method='inference',
                data=body
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Inference request failed: {e}")
            raise e
    
    async def _handle_training_request(self, body: Dict, headers: Dict) -> Dict:
        """Handle ML training request"""
        try:
            # Discover training services
            services = self.service_discovery.discover_service(ServiceType.TRAINING)
            
            if not services:
                raise Exception("No training services available")
            
            # Select service
            selected_service = self.load_balancer.select_service(services, {
                'request_size': len(str(body)),
                'priority': headers.get('X-Priority', 1),
                'complexity': body.get('complexity', 1)
            })
            
            # Call service through mesh
            result = await self.service_mesh.call_service(
                from_service='api-gateway',
                to_service=selected_service.service_id,
                method='training',
                data=body
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Training request failed: {e}")
            raise e
    
    async def _handle_health_check(self, body: Dict, headers: Dict) -> Dict:
        """Handle health check request"""
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': len(self.service_discovery.services)
        }
    
    async def _logging_middleware(self, body: Dict, headers: Dict) -> Dict:
        """Logging middleware"""
        self.logger.info(f"Request: {headers.get('X-Request-ID', 'unknown')}")
        return body
    
    async def _metrics_middleware(self, body: Dict, headers: Dict) -> Dict:
        """Metrics middleware"""
        # Add request ID if not present
        if 'X-Request-ID' not in headers:
            headers['X-Request-ID'] = hashlib.md5(str(time.time()).encode()).hexdigest()
        return body
    
    async def call_ml_service(self, 
                            service_type: ServiceType,
                            method: str,
                            data: Dict,
                            request_context: Dict = None) -> Dict:
        """Call ML service with advanced orchestration"""
        try:
            # Discover services
            services = self.service_discovery.discover_service(service_type)
            
            if not services:
                raise Exception(f"No {service_type.value} services available")
            
            # Select service
            selected_service = self.load_balancer.select_service(services, request_context or {})
            
            # Get circuit breaker
            circuit_breaker = self.circuit_breakers.get(selected_service.service_id)
            
            if circuit_breaker:
                # Call with circuit breaker protection
                result = circuit_breaker.call(
                    lambda: asyncio.run(self._make_service_call(selected_service, method, data))
                )
            else:
                # Direct call
                result = await self._make_service_call(selected_service, method, data)
            
            # Track metrics
            self.metrics_tracker.record_service_call_metrics({
               'service_type': service_type.value,
               'service_id': selected_service.service_id,
               'method': method,
               'response_time': selected_service.response_time,
               'success': True
            })
            
            return result
            
        except Exception as e:
            # Track error metrics
            self.metrics_tracker.record_service_call_metrics({
               'service_type': service_type.value,
               'service_id': selected_service.service_id if 'selected_service' in locals() else 'unknown',
               'method': method,
               'success': False,
               'error': str(e)
            })
            
            self.logger.error(f"Service call failed: {e}")
            raise e
    
    async def _make_service_call(self, service: ServiceEndpoint, method: str, data: Dict) -> Dict:
        """Make actual service call"""
        try:
            # Prepare request
            url = f"http://{service.url}:{service.port}/api/v1/{method}"
            headers = {
                'Content-Type': 'application/json',
                'X-Service-ID': service.service_id
            }
            
            # Make request with retry logic
            for attempt in range(self.integration_config['retry_attempts']):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json=data, headers=headers, 
                                              timeout=self.integration_config['timeout']) as response:
                            if response.status == 200:
                                return await response.json()
                            else:
                                raise Exception(f"Service returned status {response.status}")
                                
                except Exception as e:
                    if attempt == self.integration_config['retry_attempts'] - 1:
                        raise e
                    await asyncio.sleep(self.integration_config['retry_delay'] * (2 ** attempt))
            
        except Exception as e:
            self.logger.error(f"Service call failed: {e}")
            raise e
    
    async def get_service_health(self) -> Dict:
        """Get health status of all services"""
        try:
            health_status = {
                'overall_status': 'healthy',
                'services': {}
            }
            
            for service_id, service in self.service_discovery.services.items():
                service_health = {
                    'status': service.status.value,
                    'last_health_check': service.last_health_check.isoformat(),
                    'response_time': service.response_time,
                    'error_rate': service.error_rate,
                    'circuit_breaker_state': self.circuit_breakers.get(service_id, {'state': 'unknown'})
                }
                
                health_status['services'][service_id] = service_health
                
                # Update overall status
                if service.status != ServiceStatus.HEALTHY:
                    health_status['overall_status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error getting service health: {e}")
            return {'overall_status': 'error', 'error': str(e)}
    
    async def get_system_health(self) -> Dict:
        """Get system health metrics"""
        try:
            health_metrics = {
                'status': 'healthy',
                'device': str(self.device),
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'registered_services': len(self.service_discovery.services),
                'active_circuit_breakers': len(self.circuit_breakers),
                'service_monitor_status': self.service_monitor.get_status(),
                'metrics_tracker_status': self.metrics_tracker.get_status(),
                'performance_metrics': {
                    'avg_response_time': self._calculate_avg_response_time(),
                    'service_availability': self._calculate_service_availability(),
                    'error_rate': self._calculate_error_rate()
                }
            }
            
            return health_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return {'status': 'error', 'error': str(e)}

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time across services"""
        if not self.service_discovery.services:
            return 0.0   
        response_times = [service.response_time for service in self.service_discovery.services.values()]
        return np.mean(response_times)
    
    def _calculate_service_availability(self) -> float:
        """Calculate service availability percentage"""
        if not self.service_discovery.services:
            return 0.0   
        healthy_services = sum(1 for service in self.service_discovery.services.values() 
                             if service.status == ServiceStatus.HEALTHY)
        return healthy_services / len(self.service_discovery.services)
    
    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate"""
        if not self.service_discovery.services:
            return 0.0   
        error_rates = [service.error_rate for service in self.service_discovery.services.values()]
        return np.mean(error_rates)

# Initialize service
ai_service_integration = AIServiceIntegration() 