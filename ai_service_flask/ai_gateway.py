"""
Advanced AI Service Gateway
Orchestrates all AI services with intelligent routing, load balancing, and health monitoring
"""

import asyncio
import aiohttp
import json
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from flask import Flask, request, jsonify
import redis
import pickle
from datetime import datetime, timedelta
import hashlib
import hmac
import secrets
import os
import queue
import statistics
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from urllib.parse import urljoin
import ssl
import certifi

# AI Gateway Configuration
@dataclass
class AIGatewayConfig:
    """AI Gateway Configuration"""
    services: Dict[str, str] = None
    load_balancing: bool = True
    health_check_interval: int = 30  # seconds
    request_timeout: int = 30  # seconds
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60  # seconds
    cache_enabled: bool = True
    cache_ttl: int = 300  # seconds
    rate_limiting: bool = True
    max_requests_per_minute: int = 100
    authentication: bool = True
    api_key_required: bool = True
    monitoring: bool = True
    metrics_collection: bool = True
    parallel_processing: bool = True
    max_workers: int = 10

class ServiceHealthMonitor:
    """Service Health Monitoring System"""
    
    def __init__(self, config: AIGatewayConfig):
        self.config = config
        self.service_status = {}
        self.service_metrics = defaultdict(list)
        self.circuit_breakers = {}
        self.last_health_check = {}
        
    async def check_service_health(self, service_name: str, service_url: str) -> Dict:
        """Check health of a specific service"""
        try:
            async with aiohttp.ClientSession() as session:
                health_url = urljoin(service_url, '/health')
                start_time = time.time()
                
                async with session.get(health_url, timeout=self.config.request_timeout) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        health_data = await response.json()
                        
                        # Update service status
                        self.service_status[service_name] = {
                            'status': 'healthy',
                            'response_time': response_time,
                            'last_check': datetime.now(),
                            'data': health_data
                        }
                        
                        # Update metrics
                        self.service_metrics[service_name].append({
                            'timestamp': datetime.now(),
                            'response_time': response_time,
                            'status': 'healthy'
                        })
                        
                        # Keep only last 100 metrics
                        if len(self.service_metrics[service_name]) > 100:
                            self.service_metrics[service_name] = self.service_metrics[service_name][-100:]
                        
                        return {
                            'status': 'healthy',
                            'response_time': response_time,
                            'data': health_data
                        }
                    else:
                        self._update_service_status(service_name, 'unhealthy', response_time)
                        return {
                            'status': 'unhealthy',
                            'response_time': response_time,
                            'error': f'HTTP {response.status}'
                        }
                        
        except Exception as e:
            response_time = time.time() - start_time if 'start_time' in locals() else 0
            self._update_service_status(service_name, 'error', response_time, str(e))
            return {
                'status': 'error',
                'response_time': response_time,
                'error': str(e)
            }
    
    def _update_service_status(self, service_name: str, status: str, 
                             response_time: float, error: str = None):
        """Update service status"""
        self.service_status[service_name] = {
            'status': status,
            'response_time': response_time,
            'last_check': datetime.now(),
            'error': error
        }
        
        # Update circuit breaker
        if status != 'healthy':
            self._update_circuit_breaker(service_name, False)
        else:
            self._update_circuit_breaker(service_name, True)
    
    def _update_circuit_breaker(self, service_name: str, success: bool):
        """Update circuit breaker for service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = {
                'failures': 0,
                'last_failure': None,
                'state': 'closed'  # closed, open, half-open
            }
        
        cb = self.circuit_breakers[service_name]
        
        if success:
            if cb['state'] == 'half-open':
                cb['state'] = 'closed'
            cb['failures'] = 0
        else:
            cb['failures'] += 1
            cb['last_failure'] = datetime.now()
            
            if cb['failures'] >= self.config.circuit_breaker_threshold:
                cb['state'] = 'open'
    
    def is_service_available(self, service_name: str) -> bool:
        """Check if service is available (circuit breaker)"""
        if service_name not in self.circuit_breakers:
            return True
        
        cb = self.circuit_breakers[service_name]
        
        if cb['state'] == 'closed':
            return True
        elif cb['state'] == 'open':
            # Check if timeout has passed
            if cb['last_failure'] and (datetime.now() - cb['last_failure']).seconds > self.config.circuit_breaker_timeout:
                cb['state'] = 'half-open'
                return True
            return False
        else:  # half-open
            return True
    
    def get_service_metrics(self, service_name: str) -> Dict:
        """Get metrics for a service"""
        if service_name not in self.service_metrics:
            return {}
        
        metrics = self.service_metrics[service_name]
        
        if not metrics:
            return {}
        
        response_times = [m['response_time'] for m in metrics]
        
        return {
            'total_requests': len(metrics),
            'avg_response_time': statistics.mean(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'success_rate': len([m for m in metrics if m['status'] == 'healthy']) / len(metrics),
            'last_24h_requests': len([m for m in metrics if (datetime.now() - m['timestamp']).days < 1])
        }

class LoadBalancer:
    """Intelligent Load Balancer"""
    
    def __init__(self, config: AIGatewayConfig, health_monitor: ServiceHealthMonitor):
        self.config = config
        self.health_monitor = health_monitor
        self.service_weights = defaultdict(lambda: 1.0)
        self.request_counts = defaultdict(int)
        
    def select_service(self, service_type: str, available_services: List[str]) -> Optional[str]:
        """Select best service based on load balancing strategy"""
        if not available_services:
            return None
        
        # Filter available services
        healthy_services = [s for s in available_services if self.health_monitor.is_service_available(s)]
        
        if not healthy_services:
            return None
        
        if len(healthy_services) == 1:
            return healthy_services[0]
        
        # Weighted round-robin with health-based adjustments
        service_scores = {}
        
        for service in healthy_services:
            # Base weight
            base_weight = self.service_weights[service]
            
            # Health-based adjustment
            health_metrics = self.health_monitor.get_service_metrics(service)
            if health_metrics:
                response_time_factor = 1.0 / (1.0 + health_metrics['avg_response_time'])
                success_rate_factor = health_metrics['success_rate']
                health_score = response_time_factor * success_rate_factor
            else:
                health_score = 1.0
            
            # Load-based adjustment
            request_count = self.request_counts[service]
            load_factor = 1.0 / (1.0 + request_count * 0.1)
            
            # Final score
            service_scores[service] = base_weight * health_score * load_factor
        
        # Select service with highest score
        selected_service = max(service_scores.keys(), key=lambda s: service_scores[s])
        
        # Update request count
        self.request_counts[selected_service] += 1
        
        return selected_service
    
    def update_service_weight(self, service_name: str, weight: float):
        """Update service weight for load balancing"""
        self.service_weights[service_name] = max(0.1, weight)
    
    def reset_request_counts(self):
        """Reset request counts (called periodically)"""
        self.request_counts.clear()

class RateLimiter:
    """Rate Limiting System"""
    
    def __init__(self, config: AIGatewayConfig):
        self.config = config
        self.request_counts = defaultdict(list)
        
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        if not self.config.rate_limiting:
            return True
        
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=1)
        
        # Clean old requests
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if req_time > window_start
        ]
        
        # Check if limit exceeded
        if len(self.request_counts[client_id]) >= self.config.max_requests_per_minute:
            return False
        
        # Add current request
        self.request_counts[client_id].append(current_time)
        return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        if not self.config.rate_limiting:
            return float('inf')
        
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=1)
        
        # Clean old requests
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if req_time > window_start
        ]
        
        return max(0, self.config.max_requests_per_minute - len(self.request_counts[client_id]))

class CacheManager:
    """Intelligent Caching System"""
    
    def __init__(self, config: AIGatewayConfig):
        self.config = config
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.config.cache_enabled:
            return None
        
        if key in self.cache:
            # Check if cache is still valid
            if key in self.cache_timestamps:
                if (datetime.now() - self.cache_timestamps[key]).seconds < self.config.cache_ttl:
                    self.cache_hits += 1
                    return self.cache[key]
                else:
                    # Cache expired
                    del self.cache[key]
                    del self.cache_timestamps[key]
        
        self.cache_misses += 1
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        if not self.config.cache_enabled:
            return
        
        self.cache[key] = value
        self.cache_timestamps[key] = datetime.now()
        
        # Clean expired entries
        self._clean_expired()
    
    def _clean_expired(self):
        """Clean expired cache entries"""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, timestamp in self.cache_timestamps.items():
            if (current_time - timestamp).seconds > self.config.cache_ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
            del self.cache_timestamps[key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }

class AIServiceGateway:
    """Advanced AI Service Gateway"""
    
    def __init__(self, config: AIGatewayConfig):
        self.config = config
        self.health_monitor = ServiceHealthMonitor(config)
        self.load_balancer = LoadBalancer(config, self.health_monitor)
        self.rate_limiter = RateLimiter(config)
        self.cache_manager = CacheManager(config)
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Service registry
        self.services = config.services or {
            'gnn_reasoning': 'http://localhost:5001',
            'federated_learning': 'http://localhost:5002',
            'multi_hop_symbiosis': 'http://localhost:5003',
            'advanced_analytics': 'http://localhost:5004'
        }
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        def health_check_loop():
            while True:
                try:
                    asyncio.run(self._health_check_all_services())
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    logging.error(f"Health check error: {e}")
                    time.sleep(60)
        
        def cache_cleanup_loop():
            while True:
                try:
                    self.cache_manager._clean_expired()
                    time.sleep(60)
                except Exception as e:
                    logging.error(f"Cache cleanup error: {e}")
                    time.sleep(60)
        
        def metrics_reset_loop():
            while True:
                try:
                    self.load_balancer.reset_request_counts()
                    time.sleep(300)  # Reset every 5 minutes
                except Exception as e:
                    logging.error(f"Metrics reset error: {e}")
                    time.sleep(60)
        
        # Start background threads
        threading.Thread(target=health_check_loop, daemon=True).start()
        threading.Thread(target=cache_cleanup_loop, daemon=True).start()
        threading.Thread(target=metrics_reset_loop, daemon=True).start()
    
    async def _health_check_all_services(self):
        """Check health of all services"""
        tasks = []
        for service_name, service_url in self.services.items():
            task = self.health_monitor.check_service_health(service_name, service_url)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def route_request(self, service_type: str, endpoint: str, 
                          method: str = 'POST', data: Dict = None, 
                          client_id: str = None) -> Dict:
        """Route request to appropriate service"""
        # Rate limiting
        if client_id and not self.rate_limiter.is_allowed(client_id):
            return {
                'error': 'Rate limit exceeded',
                'remaining_requests': 0
            }
        
        # Check cache for GET requests
        if method == 'GET':
            cache_key = f"{service_type}:{endpoint}:{hash(str(data))}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                return cached_result
        
        # Select service
        available_services = list(self.services.keys())
        selected_service = self.load_balancer.select_service(service_type, available_services)
        
        if not selected_service:
            return {
                'error': f'No available services for {service_type}',
                'service_status': self.health_monitor.service_status
            }
        
        # Make request
        service_url = self.services[selected_service]
        full_url = urljoin(service_url, endpoint)
        
        try:
            async with aiohttp.ClientSession() as session:
                if method == 'GET':
                    async with session.get(full_url, timeout=self.config.request_timeout) as response:
                        result = await response.json()
                else:
                    async with session.post(full_url, json=data, timeout=self.config.request_timeout) as response:
                        result = await response.json()
                
                # Cache result for GET requests
                if method == 'GET':
                    cache_key = f"{service_type}:{endpoint}:{hash(str(data))}"
                    self.cache_manager.set(cache_key, result)
                
                return {
                    'service': selected_service,
                    'result': result,
                    'response_time': time.time()
                }
                
        except Exception as e:
            logging.error(f"Request error to {selected_service}: {e}")
            return {
                'error': str(e),
                'service': selected_service
            }
    
    def get_gateway_status(self) -> Dict:
        """Get comprehensive gateway status"""
        return {
            'services': {
                name: {
                    'url': url,
                    'status': self.health_monitor.service_status.get(name, {}),
                    'available': self.health_monitor.is_service_available(name),
                    'metrics': self.health_monitor.get_service_metrics(name)
                }
                for name, url in self.services.items()
            },
            'cache_stats': self.cache_manager.get_stats(),
            'rate_limiting': {
                'enabled': self.config.rate_limiting,
                'max_requests_per_minute': self.config.max_requests_per_minute
            },
            'load_balancing': {
                'enabled': self.config.load_balancing,
                'service_weights': dict(self.load_balancer.service_weights)
            }
        }

# Flask Application for AI Gateway
gateway_app = Flask(__name__)

# Initialize gateway
gateway_config = AIGatewayConfig()
gateway = AIServiceGateway(gateway_config)

@gateway_app.route('/health', methods=['GET'])
def gateway_health():
    """Gateway health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'ai_gateway',
        'timestamp': datetime.now().isoformat()
    })

@gateway_app.route('/status', methods=['GET'])
def gateway_status():
    """Get comprehensive gateway status"""
    try:
        status = gateway.get_gateway_status()
        return jsonify({
            'status': 'success',
            'gateway_status': status
        })
    except Exception as e:
        logging.error(f"Status error: {e}")
        return jsonify({'error': str(e)}), 500

@gateway_app.route('/gnn/<endpoint>', methods=['POST'])
async def gnn_service(endpoint):
    """Route to GNN reasoning service"""
    try:
        data = request.get_json()
        client_id = request.headers.get('X-Client-ID', 'anonymous')
        
        result = await gateway.route_request('gnn_reasoning', f'/{endpoint}', 'POST', data, client_id)
        
        return jsonify(result)
    except Exception as e:
        logging.error(f"GNN service error: {e}")
        return jsonify({'error': str(e)}), 500

@gateway_app.route('/federated/<endpoint>', methods=['POST'])
async def federated_service(endpoint):
    """Route to federated learning service"""
    try:
        data = request.get_json()
        client_id = request.headers.get('X-Client-ID', 'anonymous')
        
        result = await gateway.route_request('federated_learning', f'/{endpoint}', 'POST', data, client_id)
        
        return jsonify(result)
    except Exception as e:
        logging.error(f"Federated service error: {e}")
        return jsonify({'error': str(e)}), 500

@gateway_app.route('/symbiosis/<endpoint>', methods=['POST'])
async def symbiosis_service(endpoint):
    """Route to multi-hop symbiosis service"""
    try:
        data = request.get_json()
        client_id = request.headers.get('X-Client-ID', 'anonymous')
        
        result = await gateway.route_request('multi_hop_symbiosis', f'/{endpoint}', 'POST', data, client_id)
        
        return jsonify(result)
    except Exception as e:
        logging.error(f"Symbiosis service error: {e}")
        return jsonify({'error': str(e)}), 500

@gateway_app.route('/analytics/<endpoint>', methods=['POST'])
async def analytics_service(endpoint):
    """Route to advanced analytics service"""
    try:
        data = request.get_json()
        client_id = request.headers.get('X-Client-ID', 'anonymous')
        
        result = await gateway.route_request('advanced_analytics', f'/{endpoint}', 'POST', data, client_id)
        
        return jsonify(result)
    except Exception as e:
        logging.error(f"Analytics service error: {e}")
        return jsonify({'error': str(e)}), 500

@gateway_app.route('/orchestrate', methods=['POST'])
async def orchestrate_services():
    """Orchestrate multiple AI services for complex workflows"""
    try:
        data = request.get_json()
        workflow = data.get('workflow', [])
        client_id = request.headers.get('X-Client-ID', 'anonymous')
        
        if not workflow:
            return jsonify({'error': 'Workflow required'}), 400
        
        results = []
        
        for step in workflow:
            service_type = step.get('service')
            endpoint = step.get('endpoint')
            step_data = step.get('data', {})
            
            if not service_type or not endpoint:
                continue
            
            result = await gateway.route_request(service_type, endpoint, 'POST', step_data, client_id)
            results.append({
                'step': step,
                'result': result
            })
        
        return jsonify({
            'status': 'success',
            'workflow_results': results
        })
        
    except Exception as e:
        logging.error(f"Orchestration error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    gateway_app.run(host='0.0.0.0', port=5000, debug=False) 