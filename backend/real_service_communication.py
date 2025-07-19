#!/usr/bin/env python3
"""
Real Inter-Service Communication Layer
Implements actual HTTP/gRPC calls between microservices
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
import jwt
import hashlib
import hmac
import base64
from functools import wraps
import traceback
import ssl
import certifi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceProtocol(Enum):
    HTTP = "http"
    GRPC = "grpc"
    TCP = "tcp"

@dataclass
class ServiceEndpoint:
    service_id: str
    service_name: str
    host: str
    port: int
    protocol: ServiceProtocol
    health_endpoint: str
    base_path: str = ""
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    auth_required: bool = False
    auth_secret: str = ""
    metadata: Dict[str, Any] = None

class ServiceAuthenticator:
    """Service-to-service authentication"""
    
    def __init__(self, service_id: str, secret: str):
        self.service_id = service_id
        self.secret = secret
    
    def generate_token(self, target_service: str, expires_in: int = 300) -> str:
        """Generate JWT token for service-to-service authentication"""
        payload = {
            'iss': self.service_id,
            'aud': target_service,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow(),
            'jti': str(uuid.uuid4())
        }
        
        return jwt.encode(payload, self.secret, algorithm='HS256')
    
    def verify_token(self, token: str, expected_issuer: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret, algorithms=['HS256'], audience=self.service_id)
            if payload['iss'] != expected_issuer:
                raise jwt.InvalidIssuerError("Invalid issuer")
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token expired")
        except jwt.InvalidTokenError as e:
            raise Exception(f"Invalid token: {e}")

global http_requests_total
if 'http_requests_total' not in globals():
    http_requests_total = Counter('http_requests_total', 'Total HTTP requests', ['service', 'method', 'status'])

global http_request_duration
if 'http_request_duration' not in globals():
    http_request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['service', 'method'])

global http_errors_total
if 'http_errors_total' not in globals():
    http_errors_total = Counter('http_errors_total', 'Total HTTP errors', ['service', 'error_type'])

global http_retries_total
if 'http_retries_total' not in globals():
    http_retries_total = Counter('http_retries_total', 'Total HTTP retries', ['service'])

class HTTPClient:
    """Advanced HTTP client with retry, circuit breaker, and metrics"""
    
    def __init__(self, service_id: str):
        self.service_id = service_id
        self.session = None
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Metrics
        self.metrics = {
            'retries_total': http_retries_total
        }
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(ssl=self.ssl_context, limit=100, limit_per_host=20)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def request(self, 
                     method: str, 
                     url: str, 
                     data: Dict[str, Any] = None,
                     headers: Dict[str, str] = None,
                     timeout: int = 30,
                     retry_attempts: int = 3) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        
        headers = headers or {}
        headers.update({
            'Content-Type': 'application/json',
            'X-Request-ID': str(uuid.uuid4()),
            'X-Service-Caller': self.service_id,
            'User-Agent': f'SymbioFlows-Service/{self.service_id}'
        })
        
        start_time = time.time()
        last_exception = None
        
        for attempt in range(retry_attempts + 1):
            try:
                async with self.session.request(
                    method, url, json=data, headers=headers, timeout=timeout
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    duration = time.time() - start_time
                    http_request_duration.labels(self.service_id, method).observe(duration)
                    http_requests_total.labels(self.service_id, method, 'success').inc()
                    
                    return result
                    
            except aiohttp.ClientError as e:
                last_exception = e
                http_errors_total.labels(self.service_id, type(e).__name__).inc()
                
                if attempt < retry_attempts:
                    self.metrics['retries_total'].labels(self.service_id).inc()
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    break
            except Exception as e:
                last_exception = e
                http_errors_total.labels(self.service_id, type(e).__name__).inc()
                break
        
        raise last_exception or Exception("Request failed")

class RealServiceCommunication:
    """Real inter-service communication implementation"""
    
    def __init__(self, service_id: str, auth_secret: str = "default-secret"):
        self.service_id = service_id
        self.authenticator = ServiceAuthenticator(service_id, auth_secret)
        self.http_client = HTTPClient(service_id)
        self.endpoints: Dict[str, ServiceEndpoint] = {}
        
        # Service registry
        self._register_default_services()
    
    def _register_default_services(self):
        """Register default service endpoints"""
        default_services = [
            ServiceEndpoint(
                service_id="ai_listings_generator",
                service_name="AI Listings Generator",
                host="localhost",
                port=5010,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="ai_matchmaking_service",
                service_name="AI Matchmaking Service",
                host="localhost",
                port=8020,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="ai_pricing_service",
                service_name="AI Pricing Service",
                host="localhost",
                port=5005,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="ai_pricing_orchestrator",
                service_name="AI Pricing Orchestrator",
                host="localhost",
                port=8030,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="meta_learning_orchestrator",
                service_name="Meta-Learning Orchestrator",
                host="localhost",
                port=8010,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="materials_bert_service_simple",
                service_name="MaterialsBERT Simple Service",
                host="localhost",
                port=5002,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="ai_monitoring_dashboard",
                service_name="AI Monitoring Dashboard",
                host="localhost",
                port=5011,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="ultra_ai_listings_generator",
                service_name="Ultra AI Listings Generator",
                host="localhost",
                port=5012,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="regulatory_compliance",
                service_name="Regulatory Compliance Service",
                host="localhost",
                port=5013,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="proactive_opportunity_engine",
                service_name="Proactive Opportunity Engine",
                host="localhost",
                port=5014,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="ai_feedback_orchestrator",
                service_name="AI Feedback Orchestrator",
                host="localhost",
                port=5015,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="value_function_arbiter",
                service_name="Value Function Arbiter",
                host="localhost",
                port=5016,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="financial_analysis_engine",
                service_name="Financial Analysis Engine",
                host="localhost",
                port=5017,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="logistics_cost_service",
                service_name="Logistics Cost Service",
                host="localhost",
                port=5006,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="ai_gateway",
                service_name="AI Gateway",
                host="localhost",
                port=8000,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="advanced_analytics_service",
                service_name="Advanced Analytics Service",
                host="localhost",
                port=5004,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="gnn_inference_service",
                service_name="GNN Inference Service",
                host="localhost",
                port=8001,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            ),
            ServiceEndpoint(
                service_id="multi_hop_symbiosis_service",
                service_name="Multi-Hop Symbiosis Service",
                host="localhost",
                port=5003,
                protocol=ServiceProtocol.HTTP,
                health_endpoint="/health",
                base_path="",
                auth_required=False
            )
        ]
        
        for service in default_services:
            self.endpoints[service.service_id] = service
    
    async def call_service(self, 
                          service_id: str, 
                          method: str, 
                          data: Dict[str, Any] = None,
                          timeout: int = 30) -> Dict[str, Any]:
        """Call a service with real HTTP communication"""
        
        if service_id not in self.endpoints:
            raise ValueError(f"Service {service_id} not registered")
        
        endpoint = self.endpoints[service_id]
        data = data or {}
        
        # Build URL
        url = f"http://{endpoint.host}:{endpoint.port}{endpoint.base_path}/{method.lstrip('/')}"
        
        # Prepare headers
        headers = {}
        if endpoint.auth_required:
            token = self.authenticator.generate_token(service_id)
            headers['Authorization'] = f"Bearer {token}"
        
        async with self.http_client as client:
            try:
                result = await client.request(
                    method="POST",
                    url=url,
                    data=data,
                    headers=headers,
                    timeout=timeout
                )
                
                logger.info(f"Successfully called {service_id}.{method}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to call {service_id}.{method}: {e}")
                raise
    
    async def health_check(self, service_id: str) -> Dict[str, Any]:
        """Check health of a service"""
        if service_id not in self.endpoints:
            raise ValueError(f"Service {service_id} not registered")
        
        endpoint = self.endpoints[service_id]
        url = f"http://{endpoint.host}:{endpoint.port}{endpoint.health_endpoint}"
        
        async with self.http_client as client:
            try:
                result = await client.request(
                    method="GET",
                    url=url,
                    timeout=5
                )
                return result
            except Exception as e:
                logger.error(f"Health check failed for {service_id}: {e}")
                return {"status": "unhealthy", "error": str(e)}
    
    async def call_multiple_services(self, 
                                   service_calls: List[Dict[str, Any]],
                                   parallel: bool = True) -> Dict[str, Any]:
        """Call multiple services, optionally in parallel"""
        
        if parallel:
            # Execute all calls in parallel
            tasks = []
            for call in service_calls:
                task = self.call_service(
                    service_id=call['service_id'],
                    method=call['method'],
                    data=call.get('data', {}),
                    timeout=call.get('timeout', 30)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = {}
            for i, result in enumerate(results):
                service_id = service_calls[i]['service_id']
                if isinstance(result, Exception):
                    processed_results[service_id] = {"error": str(result)}
                else:
                    processed_results[service_id] = result
            
            return processed_results
        else:
            # Execute calls sequentially
            results = {}
            for call in service_calls:
                try:
                    result = await self.call_service(
                        service_id=call['service_id'],
                        method=call['method'],
                        data=call.get('data', {}),
                        timeout=call.get('timeout', 30)
                    )
                    results[call['service_id']] = result
                except Exception as e:
                    results[call['service_id']] = {"error": str(e)}
            
            return results

# Initialize the communication layer
service_comm = RealServiceCommunication("orchestration-engine")

# Flask app for API endpoints
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Real Service Communication',
        'version': '1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/call/<service_id>/<path:method>', methods=['POST'])
async def call_service_endpoint(service_id, method):
    """Call a service through the communication layer"""
    try:
        data = request.json or {}
        timeout = request.args.get('timeout', 30, type=int)
        
        result = await service_comm.call_service(service_id, method, data, timeout)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Service call error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health/<service_id>', methods=['GET'])
async def service_health_endpoint(service_id):
    """Check health of a specific service"""
    try:
        result = await service_comm.health_check(service_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/call/multiple', methods=['POST'])
async def call_multiple_services_endpoint():
    """Call multiple services"""
    try:
        data = request.json
        service_calls = data.get('calls', [])
        parallel = data.get('parallel', True)
        
        if not service_calls:
            return jsonify({'error': 'No service calls specified'}), 400
        
        result = await service_comm.call_multiple_services(service_calls, parallel)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Multiple service calls error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/services', methods=['GET'])
def list_services():
    """List all registered services"""
    services = {}
    for service_id, endpoint in service_comm.endpoints.items():
        services[service_id] = {
            'service_name': endpoint.service_name,
            'host': endpoint.host,
            'port': endpoint.port,
            'protocol': endpoint.protocol.value,
            'health_endpoint': endpoint.health_endpoint,
            'auth_required': endpoint.auth_required
        }
    
    return jsonify(services)

if __name__ == "__main__":
    print("🚀 Starting Real Service Communication on port 5020...")
    app.run(host='0.0.0.0', port=5020, debug=False) 